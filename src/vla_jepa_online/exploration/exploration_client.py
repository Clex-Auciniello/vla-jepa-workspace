"""Client ROS per pretraining online VLA-JEPA con upload asincrono verso Flask.

Il client muove il robot, registra in RAM e chiede approvazione manuale.
Accettazione operatore: aggiorna IMMEDIATAMENTE ID e contatori del tracker.
POST /train_episode viene effettuato in un thread separato; la registrazione
successiva NON attende il training. Lo stato training non modifica le visite.

Niente video persistenti su disco, niente training qui, niente retry automatico
di POST ambiguo (un timeout non prova che il server non abbia ricevuto il video).
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from vla_jepa_online.exploration.coverage_tracker import CoverageTracker, DEFAULT_STATE_PATH
from vla_jepa_online.exploration.motion_sampler import MotionSampler
from vla_jepa_online.exploration.random_walker import RandomWalker, load_exploration_seed
from vla_jepa_online.exploration.robot_executor import (
    ROSMoveItBackend, RobotExecutionError, RobotExecutor, RobotExecutorConfig,
)
from vla_jepa_online.exploration.trajectory_recorder import (
    RecordedEpisode, TrajectoryRecorder, TrajectoryRecorderConfig,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "exploration.yaml"
DEFAULT_GRID = ROOT / "configs" / "workspace_grid.json"


@dataclass(frozen=True)
class ExplorationClientConfig:
    server_url: str
    request_timeout_s: float
    max_queued_uploads: int
    min_frames: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExplorationClientConfig":
        import yaml
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream)["exploration_client"]
        cfg = cls(**{name: values[name] for name in cls.__dataclass_fields__})
        if (not cfg.server_url.startswith(("http://", "https://"))
                or cfg.request_timeout_s <= 0
                or type(cfg.max_queued_uploads) is not int or cfg.max_queued_uploads < 1
                or type(cfg.min_frames) is not int or cfg.min_frames < 2):
            raise ValueError("Parametri exploration_client non validi")
        return cfg


class TrainingSubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class UploadTask:
    episode_id: str
    table_mode: str
    waypoint_ids: tuple[int, ...]
    episode: RecordedEpisode


@dataclass(frozen=True)
class TrainingEvent:
    episode_id: str
    status: str
    message: str = ""


class FlaskTrainingTransport:
    """HTTP client di un server Flask. Flask NON e' una dipendenza del client.

    POST /train_episode -> {episode_id,status:'queued'|'training'|'trained'}.
    Flask risponde dopo aver ACCODATO il job, senza aspettare il training.
    Il server dovrebbe riconoscere ID duplicati per evitare doppio training.
    """

    def __init__(self, config: ExplorationClientConfig):
        self.cfg = config
        self.base_url = config.server_url.rstrip("/")

    @staticmethod
    def encode(task: UploadTask) -> bytes:
        buffer = io.BytesIO()
        manifest_frames = []
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for frame in task.episode.frames:
                front = f"frames/front/{frame.index:06d}.jpg"
                wrist = f"frames/gripper/{frame.index:06d}.jpg"
                archive.writestr(front, frame.front_jpeg)
                archive.writestr(wrist, frame.gripper_jpeg)
                manifest_frames.append({
                    "index": frame.index,
                    "front_stamp_ns": frame.front_stamp_ns,
                    "gripper_stamp_ns": frame.gripper_stamp_ns,
                    "front_path": front,
                    "gripper_path": wrist,
                })
            archive.writestr("manifest.json", json.dumps({
                "schema_version": 1,
                "episode_id": task.episode_id,
                "table_mode": task.table_mode,
                "waypoint_ids": list(task.waypoint_ids),
                "recording_fps_limit": task.episode.fps,
                "frames": manifest_frames,
            }, separators=(",", ":")))
        return buffer.getvalue()

    def _read_json(self, request: Request, *, timeout: float | None = None) -> dict:
        with urlopen(request, timeout=timeout or self.cfg.request_timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise TrainingSubmissionError("Risposta Flask non e' un oggetto JSON")
        return data

    @staticmethod
    def _validate(reply: dict, episode_id: str) -> str:
        status = reply.get("status")
        if reply.get("episode_id") != episode_id or status not in (
            "queued", "training", "trained", "failed",
        ):
            raise TrainingSubmissionError(f"Risposta non valida per {episode_id}: {reply!r}")
        if status == "failed":
            raise TrainingSubmissionError(f"Server Flask ha rifiutato o fallito {episode_id}: {reply!r}")
        return str(status)

    def upload(self, task: UploadTask) -> str:
        request = Request(
            self.base_url + "/train_episode", data=self.encode(task),
            method="POST", headers={
                "Content-Type": "application/zip", "Accept": "application/json",
                "X-Episode-ID": task.episode_id,
            },
        )
        return self._validate(self._read_json(request), task.episode_id)



class AsyncTrainingSender:
    """Invia in background i video a Flask, senza attendere il training.

    Trasmette eventuali esiti/errori del POST al thread principale solo per
    visualizzarli: NON modifica ID o conteggi degli episodi accettati.
    """

    def __init__(self, transport: FlaskTrainingTransport, *, max_queued_uploads: int):
        self.transport = transport
        self.uploads: queue.Queue[UploadTask] = queue.Queue(maxsize=max_queued_uploads)
        self.events: queue.SimpleQueue[TrainingEvent] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="vla_jepa_flask_sender")

    def start(self) -> None:
        self._worker.start()

    def can_enqueue(self) -> bool:
        return not self.uploads.full()

    def enqueue(self, task: UploadTask) -> None:
        self.uploads.put_nowait(task)

    def drain_events(self) -> list[TrainingEvent]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        while not (self._stop.is_set() and self.uploads.empty()):
            try:
                task = self.uploads.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                status = self.transport.upload(task)
                self.events.put(TrainingEvent(task.episode_id, status))
            except Exception as exc:
                self.events.put(TrainingEvent(task.episode_id, "upload_error", str(exc)))
            finally:
                self.uploads.task_done()
                del task  # rilascia i JPEG di quell'episodio

    def shutdown(self) -> None:
        """Attende solo che terminino i POST in coda, non il training Flask."""
        self._stop.set()
        self._worker.join()


class ExplorationClient:
    def __init__(
        self, *, grid: dict, tracker: CoverageTracker, walker: RandomWalker,
        sampler: MotionSampler, robot: RobotExecutor, recorder: TrajectoryRecorder,
        sender: AsyncTrainingSender, min_frames: int,
        ask: Callable[[str], str] = input,
        report: Callable[[str], None] = print,
    ) -> None:
        self.grid, self.tracker, self.walker, self.sampler = grid, tracker, walker, sampler
        self.robot, self.recorder = robot, recorder
        self.sender, self.min_frames = sender, min_frames
        self.ask, self.report = ask, report

    def _drain_training_events(self) -> None:
        for event in self.sender.drain_events():
            if event.status == "upload_error":
                self.report(f"[SERVER] {event.episode_id}: ERRORE upload: {event.message}. "
                            "Conteggi gia' aggiornati all'accettazione; nessun retry automatico.")
            else:
                self.report(f"[SERVER] {event.episode_id}: POST accettato ({event.status}).")

    def _mode(self, previous: str | None) -> str | None:
        while True:
            extra = f" [INVIO={previous}]" if previous else ""
            answer = self.ask(f"Tavolo [e=EMPTY / o=OCCUPIED / q=esci]{extra}: ").strip().lower()
            if answer == "q":
                return None
            if answer in ("e", "empty"):
                return "empty"
            if answer in ("o", "occupied"):
                return "occupied"
            if answer == "" and previous:
                return previous
            self.report("Seleziona e oppure o.")

    def _enter(self, prompt: str) -> bool:
        while True:
            answer = self.ask(prompt + " [INVIO / q=esci]: ").strip().lower()
            if answer == "":
                return True
            if answer == "q":
                return False

    def _decision(self) -> str:
        while True:
            answer = self.ask("Registrazione: [a=accetta / s=scarta / q=esci]: ").strip().lower()
            if answer in ("a", "s", "q"):
                return answer

    def _recover(self, error: Exception) -> bool:
        self.report(f"ERRORE ROBOT: {error}. Episodio scartato.")
        self.report("Nessun recupero automatico. Ripristina il robot dal tablet.")
        return self._enter("Dopo il ripristino, riparti con un NUOVO tentativo")

    def run(self) -> None:
        mode: str | None = None
        self.sender.start()
        self.report("Client VLA-JEPA: upload/training asincroni via Flask, RGB solo in RAM.")
        try:
            while True:
                self._drain_training_events()
                mode = self._mode(mode)
                if mode is None:
                    break
                episode_id = self.tracker.next_episode_id
                self.report(f"\n=== {episode_id} | {mode.upper()} ===")
                if not self._enter("GoHome -> posa iniziale -> gripper OPEN"):
                    break
                try:
                    start_pose = self.robot.prepare()
                except RobotExecutionError as exc:
                    self.tracker.discard_episode(episode_id)
                    if not self._recover(exc):
                        break
                    continue

                walk = self.walker.sample(
                    self.grid, self.tracker.accepted_visits, table_mode=mode, rng=self.tracker.rng,
                )
                motion = self.sampler.sample(
                    self.grid, walk.waypoint_ids, table_mode=mode,
                    rng=self.tracker.rng, initially_closed=False,
                )
                plan = self.robot.build_commands(motion, start_pose, grid=self.grid)
                self.report("Waypoint nominali: " + ", ".join(map(str, plan.waypoint_ids)))
                if not self._enter(f"Prepara tavolo {mode.upper()}; avvia registrazione ed esecuzione"):
                    self.tracker.discard_episode(episode_id)
                    break

                self.recorder.start()
                robot_error: Exception | None = None
                try:
                    self.robot.execute(plan)
                except RobotExecutionError as exc:
                    robot_error = exc
                finally:
                    episode = self.recorder.stop()
                    self.recorder.clear()  # tuple in 'episode' still owns compressed frames

                self.report(f"Acquisiti {len(episode.frames)} frame; "
                            f"durata {episode.duration_s:.2f}s")
                self._drain_training_events()
                if robot_error is not None:
                    self.tracker.discard_episode(episode_id)
                    if not self._recover(robot_error):
                        break
                    continue
                if len(episode.frames) < self.min_frames:
                    self.report(f"Scartato: {len(episode.frames)} coppie, minimo {self.min_frames}.")
                    self.tracker.discard_episode(episode_id)
                    continue

                choice = self._decision()
                if choice != "a":
                    self.tracker.discard_episode(episode_id)
                    if choice == "q":
                        break
                    continue

                # Exceptional backpressure only: do not retain unbounded JPEGs.
                while not self.sender.can_enqueue():
                    self.report("Coda upload RAM piena: attendi che finisca un INVIO HTTP "
                                "(non il training) prima di accettare questo episodio.")
                    time.sleep(0.5)
                    self._drain_training_events()
                self.tracker.accept_episode(episode_id, plan.waypoint_ids)
                self.sender.enqueue(UploadTask(episode_id, mode, plan.waypoint_ids, episode))
                self.report(f"{episode_id} ACCETTATO: visite aggiornate e POST in coda. "
                            f"Prossimo ID: {self.tracker.next_episode_id}. "
                            "Puoi iniziare subito una nuova registrazione.")
        finally:
            self.report("Uscita: completo gli eventuali POST ancora in RAM; "
                        "NON attendo il training Flask.")
            self.sender.shutdown()
            self._drain_training_events()


def _arguments(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Client esplorazione UR5e con server Flask")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--validate-only", action="store_true", help="Valida config senza ROS")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    opts = _arguments(argv)
    grid = json.loads(opts.grid.read_text(encoding="utf-8"))
    seed = load_exploration_seed(opts.config)
    walker = RandomWalker.from_yaml(opts.config)
    sampler = MotionSampler.from_yaml(opts.config)
    robot_cfg = RobotExecutorConfig.from_yaml(opts.config)
    recorder_cfg = TrajectoryRecorderConfig.from_yaml(opts.config)
    cfg = ExplorationClientConfig.from_yaml(opts.config)
    if opts.validate_only:
        print(f"OK: {len(grid['waypoints'])} waypoint, seed {seed}; "
              f"{recorder_cfg.fps} FPS; Flask {cfg.server_url}")
        return 0

    tracker = CoverageTracker(grid, opts.state, seed=seed)
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError as exc:
        raise RuntimeError("Avviare nel container ROS Jazzy") from exc

    rclpy.init()
    node = None
    ros_executor = None
    spinner = None
    try:
        node = rclpy.create_node("vla_jepa_exploration_client")
        ros_executor = MultiThreadedExecutor(num_threads=3)
        ros_executor.add_node(node)
        robot = RobotExecutor(ROSMoveItBackend(node, robot_cfg), robot_cfg)
        recorder = TrajectoryRecorder(node, recorder_cfg)
        spinner = threading.Thread(target=ros_executor.spin, daemon=True, name="vla_jepa_ros")
        spinner.start()
        sender = AsyncTrainingSender(
            FlaskTrainingTransport(cfg), max_queued_uploads=cfg.max_queued_uploads,
        )
        ExplorationClient(grid=grid, tracker=tracker, walker=walker, sampler=sampler,
                          robot=robot, recorder=recorder, sender=sender,
                          min_frames=cfg.min_frames).run()
    finally:
        if ros_executor is not None:
            ros_executor.shutdown(timeout_sec=5.0)
        if spinner is not None:
            spinner.join(timeout=5.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (TrainingSubmissionError, ValueError, OSError) as exc:
        print(f"Client interrotto: {exc}", file=sys.stderr)
        sys.exit(1)
