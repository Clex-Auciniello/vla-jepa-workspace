"""Plays Client UR5e: task e colore da terminale, sempre DUE click front-ZED.

Default: record (muove e salva SOLO i Plays accettati, nessun POST di default).
--plan-only: solo stampa piano, nessun movimento, acquisizione o PKL.
--test: esegue piano, nessun recorder, PKL o POST.
--record (o --execute, alias compatibile): come default.
--upload --ack-updated-batch-builder: SOLO in record e DOPO patch server.

Riusa moduli ORIGINALI Exploration senza modificarli: RobotExecutor/backend,
TrajectoryRecorder, RobotStateReader, AsyncTrainingSender e lettura PKL.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from vla_jepa_online.exploration.robot_executor import (
    ROSMoveItBackend, RobotExecutor, RobotExecutorConfig, PoseTarget,
)
from vla_jepa_online.exploration.robot_state_reader import RobotStateReader
from vla_jepa_online.exploration.trajectory_recorder import (
    TrajectoryRecorder, TrajectoryRecorderConfig,
)
from vla_jepa_online.exploration.exploration_client import AsyncTrainingSender
from vla_jepa_online.plays.plays import (
    TASKS, COLORS, EXPERIMENTAL, PlayRequest, build_play_plan, semantic_prompt,
)
from vla_jepa_online.plays.primitives import PrimitiveConfig
from vla_jepa_online.plays.plays_trajectory_saver import PlaysTrajectorySaver
from vla_jepa_online.plays.plays_transport import (
    PlaysTrainingTransport, PlaysTransportConfig, PlaysUploadTask,
)
from vla_jepa_online.plays.vision import (
    ClickCollector, robust_depth_at, deproject_pixel, camera_point_to_aruco,
    aruco_point_to_table0, load_camera_calibration,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAYS = ROOT / "configs" / "plays.yaml"
DEFAULT_EXPLORATION = ROOT / "configs" / "exploration.yaml"


@dataclass(frozen=True)
class PlaysClientConfig:
    click_camera_name: str
    camera_node_name: str
    camera_calibration_path: str
    table_frame: str
    depth_window: int
    preview_sync_slop_s: float
    preview_timeout_s: float
    pick_y_offset_m: float
    move_to_initial_pose: bool
    pose_before_first_inference: tuple[float, ...]
    final_settle_s: float
    min_frames: int
    save_path: str
    allow_experimental_execution: bool

    @classmethod
    def from_yaml(cls, path: Path):
        values = yaml.safe_load(path.read_text(encoding="utf-8"))["plays"]
        values["pose_before_first_inference"] = tuple(values["pose_before_first_inference"])
        cfg = cls(**values)
        if (len(cfg.pose_before_first_inference) != 7 or cfg.min_frames < 8 or
            cfg.preview_sync_slop_s <= 0 or cfg.preview_timeout_s <= 0 or
            cfg.depth_window < 1 or cfg.final_settle_s < 0):
            raise ValueError("Configurazione Plays non valida")
        return cfg


class FrontDepthSelector:
    """RGB, depth e CameraInfo della STESSA camera; coppia RGB-D sincronizzata.

    L'anteprima usa le coordinate pixel ORIGINALI, non il crop usato da VLA.
    Il backend mantiene già il TF buffer in ascolto con un ROS executor esterno.
    """
    def __init__(self, node, backend, cfg: PlaysClientConfig):
        import message_filters
        from sensor_msgs.msg import Image, CameraInfo
        from cv_bridge import CvBridge
        from rclpy.qos import qos_profile_sensor_data
        self.node, self.backend, self.cfg = node, backend, cfg
        self.bridge = CvBridge()
        self.calibration = load_camera_calibration(cfg.camera_calibration_path)
        if cfg.click_camera_name not in self.calibration:
            raise ValueError(f"Calibrazione mancante per {cfg.click_camera_name}")
        self._cv = threading.Condition()
        self._last = None
        self._K = None
        prefix = f"/{cfg.click_camera_name}/{cfg.camera_node_name}"
        rgb_sub = message_filters.Subscriber(node, Image,
            f"{prefix}/rgb/color/rect/image", qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(node, Image,
            f"{prefix}/depth/depth_registered", qos_profile=qos_profile_sensor_data)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=15, slop=cfg.preview_sync_slop_s,
        )
        self._rgb_sub, self._depth_sub = rgb_sub, depth_sub  # live references
        self._sync.registerCallback(self._on_pair)
        self._info_sub = node.create_subscription(
            CameraInfo, f"{prefix}/rgb/color/rect/camera_info", self._on_info,
            qos_profile_sensor_data,
        )

    def _on_info(self, msg):
        K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        with self._cv:
            self._K = K
            self._cv.notify_all()

    def _on_pair(self, rgb_msg, depth_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth = np.asarray(depth)
            if depth.ndim != 2:
                return
            if depth_msg.encoding.lower() in ("16uc1", "mono16"):
                depth = depth.astype(np.float32) / 1000.0
            elif depth.dtype.kind != "f":
                return  # unità profondità ignote: non inventarle
            if rgb.shape[:2] != depth.shape:
                return
            stamp_ns = (int(rgb_msg.header.stamp.sec)*1_000_000_000 +
                        int(rgb_msg.header.stamp.nanosec))
            with self._cv:
                self._last = (rgb.copy(), depth.copy(), stamp_ns)
                self._cv.notify_all()
        except Exception as exc:
            self.node.get_logger().warning(f"Coppia preview scartata: {exc}")

    def snapshot(self):
        deadline = time.monotonic() + self.cfg.preview_timeout_s
        with self._cv:
            while self._last is None or self._K is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("RGB-D/CameraInfo non disponibile per il click")
                self._cv.wait(timeout=remaining)
            rgb, depth, stamp = self._last
            current_ns = self.node.get_clock().now().nanoseconds
            if abs(current_ns-stamp) > 2_000_000_000:
                raise RuntimeError("Anteprima RGB-D più vecchia di 2 secondi")
            return rgb.copy(), depth.copy(), self._K.copy()

    def click_two(self, first_label: str, second_label: str):
        rgb, depth, K = self.snapshot()
        clicks = ClickCollector()
        try:
            uv1 = clicks.collect(rgb, f"FIRST: {first_label}\nSPACE confirm, ESC abort")
            uv2 = clicks.collect(rgb, f"SECOND: {second_label}\nSPACE confirm, ESC abort")
        finally:
            clicks.close()
        h, w = depth.shape
        for u, v in (uv1, uv2):
            if not (0 <= u < w and 0 <= v < h):
                raise RuntimeError("Click oltre dimensione originale immagine")
        return uv1, uv2, self._to_base(uv1, depth, K), self._to_base(uv2, depth, K)

    def _to_base(self, uv, depth, K):
        from rclpy.time import Time
        from rclpy.duration import Duration
        u, v = uv
        d = robust_depth_at(depth, u, v, window=self.cfg.depth_window)
        if d is None or not (0.05 <= d <= 5.0):
            raise RuntimeError(f"Depth assente/non plausibile al click {uv}: {d}")
        camera = deproject_pixel(u, v, d, K)
        aruco = camera_point_to_aruco(camera, self.calibration[self.cfg.click_camera_name])
        table = aruco_point_to_table0(aruco)
        tf = self.backend.tf_buffer.lookup_transform(
            "base_link", self.cfg.table_frame, Time(), timeout=Duration(seconds=5.0),
        )
        q = tf.transform.rotation
        x, y, z, w = (q.x, q.y, q.z, q.w)
        # Identica convenzione quaternion xyzw del backend ROS.
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], dtype=np.float64)
        t = tf.transform.translation
        point = R @ table + np.array([t.x, t.y, t.z])
        if not np.isfinite(point).all():
            raise RuntimeError("Deproiezione non finita")
        return tuple(float(n) for n in point)


class PlaysClient:
    def __init__(self, *, cfg, primitive_cfg, robot, selector, recorder=None, saver=None,
                 mode: str = "record", allow_experimental: bool = False, sender=None):
        if mode not in ("plan", "test", "record"):
            raise ValueError(f"Modalità Plays non valida: {mode}")
        if mode == "record" and (recorder is None or saver is None):
            raise ValueError("La modalità record richiede recorder e saver")
        if mode != "record" and sender is not None:
            raise ValueError("Upload consentito soltanto in modalità record")
        self.cfg, self.primitive_cfg = cfg, primitive_cfg
        self.robot, self.selector, self.recorder = robot, selector, recorder
        self.saver, self.sender = saver, sender
        self.mode, self.allow_experimental = mode, allow_experimental

    def _select_request(self):
        print("\n=== TASKS PLAYS ===")
        for tid, title in TASKS.items():
            print(f"{tid}: {title}")
        task = input("Task [02..09; q=esci]: ").strip().lower()
        if task == "q":
            return None
        if task not in TASKS:
            print("Task non valido")
            return False
        color = input("Colore cubo da manipolare [red/green/blue/yellow]: ").strip().lower()
        if color not in COLORS:
            print("Colore non valido")
            return False
        target_color = None
        if task == "05":
            target_color = input("Colore cubo di supporto: ").strip().lower()
            if target_color not in COLORS:
                print("Secondo colore non valido")
                return False
        rotation_deg = None
        if task == "03":
            # Campionamento per episodio, segno casuale (CW/CCW). L'angolo
            # viene salvato nel PKL, NON inserito nel prompt semantico.
            rotation_deg = random.uniform(-30.0, 30.0)
            while abs(rotation_deg) < 0.1:
                rotation_deg = random.uniform(-30.0, 30.0)
            print(f"Yaw cubo campionato: {rotation_deg:+.2f} gradi (base_link Z)")
        elif task == "09":
            try:
                rotation_deg = float(input(
                    "Yaw gripper prima del grasp [-30, +30] gradi: "
                ).strip())
            except ValueError:
                print("Angolo non numerico")
                return False
            if not math.isfinite(rotation_deg) or not -30.0 <= rotation_deg <= 30.0:
                print("Angolo non valido: serve un valore tra -30 e +30")
                return False
        labels = {
            "02": "cube", "03": "cube", "04": "cube to push",
            "05": "cube to stack", "06": "top cube", "07": "cube inside bin",
            "08": "cube near bin edge", "09": "cube (rotate gripper before pick)",
        }
        destinations = {
            "02": "new place point", "03": "place after rotating",
            "04": "push destination", "05": "support cube", "06": "free point on table",
            "07": "free point outside bin", "08": "point inside destination bin",
            "09": "place after rotated grasp",
        }
        try:
            uv1, uv2, pos1, pos2 = self.selector.click_two(labels[task], destinations[task])
            # Mantiene lo storico offset Y=-0.06 SOLO per i task con grasp.
            # Per la nuova spinta la geometria è distinta e non eredita l'offset.
            offset = 0.0 if task in ("04", "08") else self.cfg.pick_y_offset_m
            req = PlayRequest(task, color, target_color, uv1, uv2, pos1, pos2, offset,
                              rotation_deg=rotation_deg)
        except RuntimeError as exc:
            print(f"Selezione annullata/non valida: {exc}")
            return False
        return req

    def _prepare_robot(self):
        self.robot.backend.check_interfaces()
        self.robot.prepare_home()  # solo prima della registrazione
        if self.cfg.move_to_initial_pose:
            values = self.cfg.pose_before_first_inference
            self.robot.backend.move_to_pose(PoseTarget(tuple(values[:3]), tuple(values[3:])))

    def _preview_plan(self, plan):
        from vla_jepa_online.utils.geometry_utils import build_linear_waypoints
        cursor = plan.start_pose
        total = 0
        print(f"\nPiano: {len(plan.commands)} comandi")
        for index, cmd in enumerate(plan.commands, 1):
            if cmd.kind == "pose":
                wps = build_linear_waypoints(
                    cursor.xyz, cursor.quaternion_xyzw,
                    cmd.target.xyz, cmd.target.quaternion_xyzw,
                    self.primitive_cfg.min_step_m,
                )
                total += len(wps)
                print(f"  {index:02d} POSE {cmd.description} -> {cmd.target.xyz} "
                      f"({len(wps)} micro-step)")
                cursor = cmd.target
            else:
                total += 1
                print(f"  {index:02d} GRIPPER {'CLOSE' if cmd.gripper_closed else 'OPEN'}")
        if self.mode == "record":
            print(f"Catture attese: START + {total} comandi/micro-step + FINAL = {total+2}")
        else:
            print(f"Movimenti/azioni previsti: {total} (nessuna acquisizione in questa modalità)")
        print("NOTA: waypoint cartesiani e verifica Z NON garantiscono assenza collisioni; "
              "MoveIt deve validare i singoli target. Controllare percorso e tavolo.")

    def _drain_events(self):
        if self.sender is not None:
            for ev in self.sender.drain_events():
                print(f"[POST {ev.episode_id}] {ev.status}: {ev.message}")

    def _record_play(self, plan):
        """Acquisisci in RAM; ritorna l'episodio soltanto se l'esecuzione riesce.

        Nessun salvataggio è previsto per errori, insuccessi o scarti.
        """
        self.recorder.start()
        try:
            self.recorder.capture_step(
                "start", motion_label="Start the selected play",
                target_pose=plan.start_pose, gripper_closed=False,
            )
            self.robot.execute(plan, on_step=self.recorder.capture_step)
            time.sleep(self.cfg.final_settle_s)
            last_pose = next(
                (cmd.target for cmd in reversed(plan.commands) if cmd.kind == "pose"),
                plan.start_pose,
            )
            self.recorder.capture_step(
                "final_observation", motion_label="Observe the final result",
                target_pose=last_pose, gripper_closed=False,
            )
        finally:
            episode = self.recorder.stop()
            self.recorder.clear()
        return episode

    def _accept_or_discard(self, episode, request):
        print(f"Catturate {len(episode.frames)} coppie RGB; "
              f"durata={episode.duration_s:.2f}s")
        if len(episode.frames) < self.cfg.min_frames:
            print(f"Episodio scartato: {len(episode.frames)} frame, "
                  f"minimo richiesto {self.cfg.min_frames}. Nessun PKL.")
            return
        outcome = input("Esito: [a=accetta / s=scarta]: ").strip().lower()
        if outcome != "a":
            print("Episodio scartato: nessun PKL e nessun POST.")
            return
        episode_id = self.saver.next_episode_id()
        path = self.saver.save(
            episode_id, episode, request,
            operator_success=True, operator_approved=True,
        )
        print(f"PKL Plays salvato: {path}")
        if self.sender is not None:
            while not self.sender.can_enqueue():
                print("Coda locale POST piena; attendo solo l'invio HTTP...")
                self._drain_events()
                time.sleep(0.5)
            self.sender.enqueue(PlaysUploadTask(episode_id, path))
            print(f"{episode_id} in coda di invio; 202 significa QUEUED, non TRAINED")

    def run(self):
        if self.sender is not None:
            self.sender.start()
        robot_prepared = False
        try:
            while True:
                self._drain_events()
                if self.mode != "plan" and not robot_prepared:
                    print("Preparare tavolo sgombro da traiettorie non previste.")
                    if input("HOME + OPEN + posa iniziale? [yes/N]: ").strip() != "yes":
                        break
                    self._prepare_robot()
                    robot_prepared = True

                request = self._select_request()
                if request is None:
                    break
                if request is False:
                    continue
                print(f"\nPrompt: {semantic_prompt(request)}")
                try:
                    start_pose = self.robot.backend.get_tcp_pose()
                    plan = build_play_plan(start_pose, request, self.primitive_cfg)
                except Exception as exc:
                    print(f"Pianificazione rifiutata: {exc}")
                    continue

                self._preview_plan(plan)
                if self.mode == "plan":
                    print("[PLAN ONLY] Nessun movimento, nessun PKL, nessun POST.")
                    continue

                if request.task_id in EXPERIMENTAL and not (
                    self.allow_experimental and self.cfg.allow_experimental_execution
                ):
                    print("Task sperimentale: esecuzione DISABILITATA. "
                          "Abilitare esplicitamente --allow-experimental e YAML solo dopo test.")
                    continue

                # Non richiedere una seconda approvazione dopo la stampa del piano:
                # l'operatore ha confermato all'inizio la preparazione del robot.
                try:
                    if self.mode == "test":
                        print("[TEST] Eseguo il piano SENZA recorder, PKL o POST.")
                        self.robot.execute(plan)
                        input("[TEST] Verifica il risultato; INVIO per tornare a HOME: ")
                    else:
                        print("[RECORD] Eseguo il piano e registro i micro-step.")
                        episode = self._record_play(plan)
                except Exception as exc:
                    robot_prepared = False
                    print(f"[ERRORE ESECUZIONE] {exc}. Episodio non salvato. "
                          "Nessun recovery automatico: controllare robot e gripper "
                          "e ripristinare manualmente prima di un nuovo HOME.")
                    continue

                # Un errore di persistenza/POST non è un errore di movimento.
                # Il file potrebbe già essere stato scritto: non dichiararlo scartato.
                if self.mode == "record":
                    try:
                        self._accept_or_discard(episode, request)
                    except Exception as exc:
                        print(f"[ERRORE SALVATAGGIO/UPLOAD] {exc}. Verificare su disco "
                              "l'eventuale PKL prima di ritentare l'upload.")

                # Dopo un test/play completato: HOME fuori dalla registrazione.
                try:
                    self.robot.return_home()
                    robot_prepared = False
                except Exception as exc:
                    print(f"GoHome fallito: {exc}; recuperare manualmente")
                    robot_prepared = False
        finally:
            if self.sender is not None:
                print("Attendo i POST locali rimanenti, non il training remoto...")
                self.sender.shutdown()
                self._drain_events()


def _arguments(argv=None):
    p = argparse.ArgumentParser(description="VLA-JEPA Plays Client UR5e")
    p.add_argument("--plays-config", type=Path, default=DEFAULT_PLAYS)
    p.add_argument("--exploration-config", type=Path, default=DEFAULT_EXPLORATION)
    p.add_argument("--validate-only", action="store_true", help="Controlla gli YAML senza ROS")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--plan-only", action="store_true", help="Solo piano, nessun movimento/PKL")
    group.add_argument("--test", action="store_true", help="Muove il robot, NON registra né salva")
    group.add_argument("--record", action="store_true", help="Muove e registra; MODALITÀ DEFAULT")
    group.add_argument("--execute", action="store_true", help="Alias compatibile di --record")
    p.add_argument("--allow-experimental", action="store_true",
                   help="Richiede ANCHE plays.allow_experimental_execution=true")
    p.add_argument("--upload", action="store_true", help="Invio asincrono allo stesso server Flask")
    p.add_argument("--ack-updated-batch-builder", action="store_true",
                   help="Conferma esplicita che il SERVER supporta prompt Plays e padding")
    return p.parse_args(argv)


def main(argv=None):
    opts = _arguments(argv)
    mode = "plan" if opts.plan_only else "test" if opts.test else "record"
    pcfg = PlaysClientConfig.from_yaml(opts.plays_config)
    params = yaml.safe_load(opts.plays_config.read_text("utf-8"))
    primitive_values = dict(params["primitives"])
    primitive_values["grasp_quaternion_xyzw"] = tuple(primitive_values["grasp_quaternion_xyzw"])
    prim = PrimitiveConfig(**primitive_values)
    backend_cfg = RobotExecutorConfig.from_yaml(opts.exploration_config)
    recorder_cfg = TrajectoryRecorderConfig.from_yaml(opts.exploration_config)
    upload_cfg = PlaysTransportConfig(**params["upload"])
    if opts.upload and (mode != "record" or not opts.ack_updated_batch_builder):
        raise ValueError("--upload richiede modalità record e --ack-updated-batch-builder: "
                         "il server ATTUALE non supporta ancora Plays")
    if opts.validate_only:
        print("OK: configurazioni Plays/Exploration, primitive, recorder e backend valide")
        print("ATTENZIONE: deproiezione, calibrazione TF, collisioni e grasp NON testati")
        return 0

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init()
    node = None
    spin = None
    executor = None
    try:
        node = rclpy.create_node("vla_jepa_plays_client")
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True, name="plays_ros_spin")
        spin.start()
        backend = ROSMoveItBackend(node, backend_cfg)
        robot = RobotExecutor(backend, backend_cfg)
        selector = FrontDepthSelector(node, backend, pcfg)

        # In plan e test non creiamo né il recorder né il saver né il sender.
        recorder = None
        saver = None
        sender = None
        if mode == "record":
            state_reader = RobotStateReader(
                node, backend.get_tcp_pose,
                joint_states_topic=recorder_cfg.joint_states_topic,
                joint_max_age_s=recorder_cfg.joint_max_age_s,
            )
            recorder = TrajectoryRecorder(node, recorder_cfg, state_reader=state_reader.read)
            saver = PlaysTrajectorySaver(pcfg.save_path)
            if opts.upload:
                sender = AsyncTrainingSender(
                    PlaysTrainingTransport(upload_cfg),
                    max_queued_uploads=upload_cfg.max_queued_uploads,
                )
        PlaysClient(
            cfg=pcfg, primitive_cfg=prim, robot=robot,
            selector=selector, recorder=recorder,
            saver=saver, sender=sender, mode=mode,
            allow_experimental=opts.allow_experimental,
        ).run()
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=5.0)
        if spin is not None:
            spin.join(timeout=5.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Plays Client: {exc}", file=sys.stderr)
        sys.exit(1)
