
"""Server Flask minimale per il pretraining online VLA-JEPA.

Un singolo processo Flask e un singolo worker GPU condividono una coda FIFO in
RAM. Non esistono SQLite, spool su disco o replay automatici degli episodi.

Il client continua a fare POST /train_episode con un archivio ZIP di PNG e
manifest.json. La risposta HTTP 202 significa "queued", NON "trained".

Gli UNICI dati persistenti sono due checkpoint alternati, contenenti i pesi
completi, AdamW, contatori e RNG necessari al resume. Arrestare il server solo
quando /status mostra safe_to_stop=true: coda vuota e ultimo blocco da 10
video consolidato in un checkpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import queue
import re
import tempfile
import threading
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vla_jepa_online.training_server.online_batch_builder import (
    OnlineBatchBuilder,
    OnlineBatchBuilderConfig,
)

LOGGER = logging.getLogger("vla_jepa_online.server")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trainer_monitor_stats(trainer: Any) -> dict[str, int]:
    """Telemetria E/P con sola lettura dei contatori del trainer.

    Compatibile con trainer vecchi o fittizi: gli episodi senza classificazione
    vengono esposti come legacy, non assegnati artificialmente a E o P.
    """
    video = getattr(trainer, "source_video_counts", {})
    windows = getattr(trainer, "source_window_counts", {})
    return {
        "global_step": int(trainer.global_step),
        "processed_videos": int(trainer.processed_videos),
        "pending_windows": int(trainer.pending_windows),
        "videos_exploration": int(video.get("E", 0)),
        "videos_plays": int(video.get("P", 0)),
        "windows_exploration": int(windows.get("E", 0)),
        "windows_plays": int(windows.get("P", 0)),
        "videos_other": int(video.get("other", 0)),
        "videos_legacy": int(getattr(trainer, "legacy_unclassified_videos", 0)
                             if hasattr(trainer, "source_video_counts")
                             else trainer.processed_videos),
    }


def _checkpoint_latest(root: Path) -> Path | None:
    """Restituisce lo slot completo piu' recente, basandosi sui metadati.

    Si controlla che i file essenziali esistano. Se i bytes dei pesi/optimizer
    sono corrotti, il caricamento del trainer fallira': nessun fallback
    silenzioso ai vecchi pesi, per non nascondere un problema di recovery.
    """
    candidates: list[tuple[int, Path]] = []

    for name in ("checkpoint_A", "checkpoint_B"):
        folder = root / name

        if not folder.exists():
            continue

        try:
            metadata = json.loads(
                (folder / "checkpoint_meta.json").read_text("utf-8")
            )

            if metadata["format_version"] != 3:
                raise ValueError("format_version non supportata")

            for filename in (
                "config.json",
                "model.safetensors",
                "online_trainer_state.pt",
            ):
                if not (folder / filename).is_file():
                    raise FileNotFoundError(f"{filename} mancante")

            candidates.append(
                (int(metadata["checkpoint_sequence"]), folder)
            )

        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOGGER.warning(
                "Checkpoint %s non valido: %s", folder, exc
            )

    return (
        max(candidates, key=lambda item: item[0])[1]
        if candidates
        else None
    )


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    checkpoint_root: str | Path = "checkpoints/exploration"

    # Episodi in attesa, escluso quello attualmente in training.
    max_queued_episodes: int = 3

    # E1 is the original online Exploration server. E2 uses offline PKLs.
    mode: str = "exploration"
    offline_exploration_dir: str | None = None


@dataclass(frozen=True)
class QueuedEpisode:
    """L'archivio ZIP resta in RAM solo mentre e' in coda/in elaborazione."""

    episode_id: str
    payload: bytes


class TrainingService:
    """Stato di una SINGOLA sessione: coda RAM + worker GPU + telemetria RAM.

    trainer_factory(resume_path) viene chiamata nel worker una sola volta.
    La classe si puo' testare senza istanziare il vero Qwen sulla GPU.

    Un thread Flask puo' leggere /status o accodare mentre il worker lavora.

    Il lock protegge soltanto la coda e i piccoli dizionari di stato, NON
    tutto il forward/backward; il monitoraggio resta quindi reattivo.
    """

    def __init__(
        self,
        cfg: ServerConfig,
        trainer_factory: Callable[[Path | None], Any],
        builder: OnlineBatchBuilder | None = None,
    ) -> None:

        if cfg.max_queued_episodes < 1:
            raise ValueError(
                "Il limite della coda deve essere almeno 1"
            )

        if cfg.mode not in ("exploration", "mixed"):
            raise ValueError(f"Unknown experiment mode: {cfg.mode}")
        if cfg.mode == "mixed" and not cfg.offline_exploration_dir:
            raise ValueError("E2 requires offline_exploration_dir")
        self.cfg = cfg
        self.trainer_factory = trainer_factory

        self.builder = builder or OnlineBatchBuilder(
            OnlineBatchBuilderConfig()
        )

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._monitor_subscribers: list[Callable[[str], None]] = []

        # Coda FIFO dei video ricevuti: nessun file su disco.
        self._pending: deque[QueuedEpisode] = deque()

        # Storico SOLO in RAM per monitoraggio e deduplicazione
        # degli episodi durante questa sessione.
        self._episodes: dict[str, dict[str, Any]] = {}

        self._stop = False
        self._phase = "initializing"
        self._error: str | None = None
        self._current: str | None = None

        self._trainer: Any = None
        self._stats: dict[str, int] = {}

        self._last_checkpoint: str | None = None

        # Numero cumulativo di video consolidati su disco.
        self._checkpointed_videos = 0

        # Only E2 uses these fields. They are initialized before starting the worker.
        self._exploration_catalog = None
        self._exploration_offset = 0
        self._exploration_initial_offset = 0
        self._mixed_last_checkpointed_play_number = 0
        self._mixed_last_accepted_play_number = 0

        # Un solo worker GPU per tutto il server.
        self._worker = threading.Thread(
            target=self._run,
            name="vla-jepa-gpu-worker",
            daemon=True,
        )

        self._worker.start()

    def subscribe_monitor(self, listener: Callable[[str], None]) -> None:
        """Consegna una fotografia iniziale e ogni transizione successiva.

        Nessun timer/polling. Il callback deve essere non bloccante:
        ServerStatusReporter inserisce semplicemente una stringa in coda.
        """
        with self._lock:
            self._monitor_subscribers.append(listener)
            listener(format_server_monitor(self, "STATO INIZIALE"))

    def _notify_monitor_unlocked(self, event: str) -> None:
        """Chiamata mentre il lock di stato e' gia' acquisito."""
        if not self._monitor_subscribers:
            return
        line = format_server_monitor(self, event)
        for listener in tuple(self._monitor_subscribers):
            try:
                listener(line)
            except Exception:
                LOGGER.exception("Callback monitor fallita")

    def close(self, timeout: float = 5.0) -> None:
        """Arresto cooperativo: usare solo dopo safe_to_stop=true.

        Non si forza l'interruzione di un backward CUDA.

        daemon=True evita blocchi alla chiusura del processo, ma
        l'operatore deve arrestare il server solo a fine checkpoint.
        """

        with self._cv:

            if not self._safe_to_stop_unlocked():
                LOGGER.warning(
                    "Arresto richiesto senza safe_to_stop: "
                    "lavoro non consolidato"
                )

            self._stop = True
            self._notify_monitor_unlocked("ARRESTO RICHIESTO")
            self._cv.notify_all()

        self._worker.join(timeout=timeout)

    def _safe_to_stop_unlocked(self) -> bool:
        """True solo se i pesi di TUTTI i video elaborati sono su disco."""

        return (
            self._phase == "ready"
            and not self._stop
            and self._current is None
            and not self._pending
            and self._last_checkpoint is not None
            and self._trainer is not None
            and int(
                self._stats.get("processed_videos", -1)
            ) == self._checkpointed_videos
            and int(
                self._stats.get("pending_windows", -1)
            ) == 0
            and self._checkpointed_videos > 0
        )

    def status(self) -> dict[str, Any]:
        """Ritorna lo stato generale del server."""

        with self._lock:
            queued = len(self._pending)

            return {
                "server": self._phase,

                # Disponibilita' a RICEVERE nuovi video.
                # Non significa che la GPU sia inattiva.
                "available": (
                    self._phase == "ready"
                    and not self._stop
                    and queued < self.cfg.max_queued_episodes
                ),

                "trainer_loaded": self._trainer is not None,
                "current_episode_id": self._current,

                "queued": queued,
                "queue_capacity": self.cfg.max_queued_episodes,

                "checkpoint_path": self._last_checkpoint,
                "checkpointed_videos": self._checkpointed_videos,

                "trainer": dict(self._stats),

                "safe_to_stop": self._safe_to_stop_unlocked(),
                "error": self._error,
            }

    def queue(self) -> dict[str, Any]:
        """Restituisce la coda FIFO e la posizione di ciascun video."""

        with self._lock:

            return {
                "processing": self._current,

                "queued": [
                    {
                        "episode_id": pending.episode_id,
                        "position": index + 1,
                        "received_at": self._episodes[
                            pending.episode_id
                        ]["received_at"],
                        "frames": self._episodes[
                            pending.episode_id
                        ]["frames"],
                        "windows_total": self._episodes[
                            pending.episode_id
                        ]["windows_total"],
                    }
                    for index, pending in enumerate(self._pending)
                ],

                "depth": len(self._pending),
                "capacity": self.cfg.max_queued_episodes,
            }

    def episode(self, episode_id: str) -> dict[str, Any] | None:
        """Restituisce l'avanzamento di un singolo episodio."""

        with self._lock:
            info = self._episodes.get(episode_id)

            # I byte del video NON sono inclusi nella risposta.
            return dict(info) if info is not None else None

    def _initialize_mixed(self, trainer: Any, resume: Path | None) -> None:
        """Check E2 identity, recover the offline cursor and last checkpointed Play."""
        catalog = self.builder.catalog_exploration_pkls(self.cfg.offline_exploration_dir)
        counts = trainer.source_video_counts
        windows = trainer.source_window_counts
        p_count = int(counts.get("P", 0))
        e_count = int(counts.get("E", 0))
        p_windows = int(windows.get("P", 0))
        e_windows = int(windows.get("E", 0))
        if (p_count != e_count or p_windows != e_windows
                or trainer.processed_videos != 2 * p_count
                or int(counts.get("other", 0)) != 0
                or int(getattr(trainer, "legacy_unclassified_videos", 0)) != 0
                or e_windows > catalog.total_windows):
            raise RuntimeError("E2 requires a balanced E2 checkpoint or fresh DROID weights")

        last_number = 0
        if resume is not None:
            marker_path = resume / "mixed_replay_state.json"
            if not marker_path.is_file():
                raise RuntimeError(
                    f"Missing E2 resume marker {marker_path}. "
                    "Refusing to recover an ambiguous/incomplete checkpoint."
                )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            checkpoint_meta = json.loads(
                (resume / "checkpoint_meta.json").read_text(encoding="utf-8")
            )
            last_id = marker.get("last_play_id")
            if (marker.get("version") != 1 or marker.get("mode") != "mixed"
                    or marker.get("dataset_fingerprint") != catalog.fingerprint
                    or marker.get("checkpoint_sequence") != checkpoint_meta.get("checkpoint_sequence")
                    or marker.get("exploration_offset") != e_windows
                    or marker.get("play_windows") != p_windows
                    or marker.get("play_count") != p_count
                    or not isinstance(last_id, str)
                    or not re.fullmatch(r"P\d{5,}", last_id)):
                raise RuntimeError("E2 resume marker is inconsistent with checkpoint or offline dataset")
            last_number = int(last_id[1:])
        self._exploration_catalog = catalog
        self._exploration_offset = e_windows
        self._exploration_initial_offset = e_windows
        self._mixed_last_checkpointed_play_number = last_number
        self._mixed_last_accepted_play_number = last_number
        LOGGER.info(
            "E2 offline dataset: %s, %d PKL, %d windows, cursor=%d, last checkpointed Play=%d",
            self.cfg.offline_exploration_dir, len(catalog.entries),
            catalog.total_windows, e_windows, last_number,
        )

    def _write_mixed_checkpoint_marker(self, checkpoint: str, last_play_id: str) -> None:
        """Commit resume metadata alongside trainer's completed A/B checkpoint.

        A crash between the trainer's atomic checkpoint and this marker fails
        closed on restart instead of silently replaying/omitting windows.
        """
        path = Path(checkpoint) / "mixed_replay_state.json"
        meta = json.loads(
            (Path(checkpoint) / "checkpoint_meta.json").read_text(encoding="utf-8")
        )
        record = {
            "version": 1,
            "mode": "mixed",
            "checkpoint_sequence": meta["checkpoint_sequence"],
            "dataset_fingerprint": self._exploration_catalog.fingerprint,
            "exploration_offset": self._exploration_offset,
            "play_windows": int(self._trainer.source_window_counts["P"]),
            "play_count": int(self._trainer.source_video_counts["P"]),
            "last_play_id": last_play_id,
        }
        fd, temporary = tempfile.mkstemp(prefix=".mixed_marker_", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(record, output, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._mixed_last_checkpointed_play_number = int(last_play_id[1:])

    def submit(
        self,
        request: Any,
        supplied_id: str | None,
    ) -> tuple[dict[str, Any], int]:
        """Riceve lo ZIP, verifica il manifest e inserisce il video in coda.

        La richiesta HTTP non esegue il training.

        Lo ZIP viene mantenuto in RAM. Non viene applicato alcun limite
        applicativo sulla dimensione della richiesta.

        Il controllo di disponibilita' viene ripetuto dopo la lettura
        per gestire richieste HTTP contemporanee.
        """

        media_type = (
            request.content_type
            .split(";", 1)[0]
            .strip()
            .lower()
        )

        if media_type not in (
            "application/zip",
            "application/octet-stream",
        ):
            return {
                "error": "Content-Type atteso: application/zip"
            }, 415

        if supplied_id is not None:
            if not supplied_id.strip() or len(supplied_id) > 256:
                return {
                    "error": "X-Episode-ID non valido"
                }, 400

        # Prima verifica: duplicati e spazio nella coda.
        with self._lock:

            if (
                supplied_id is not None
                and supplied_id in self._episodes
            ):
                state = self._episodes[supplied_id]

                return {
                    "episode_id": supplied_id,
                    "status": state["status"],
                    "checkpointed": state["checkpointed"],
                }, 200

            if self._phase != "ready" or self._stop:
                return {
                    "error": "Server non pronto",
                    "server": self._phase,
                }, 503

            if len(self._pending) >= self.cfg.max_queued_episodes:
                return {
                    "error": "Coda piena",
                    "queued": len(self._pending),
                }, 503

        # Lettura completa dello ZIP in RAM, senza soglia max_upload_mb.
        buffer = io.BytesIO()

        while chunk := request.stream.read(1024 * 1024):
            buffer.write(chunk)

        payload = buffer.getvalue()

        # Verifica che lo ZIP e il manifest siano compatibili
        # con il nostro online_batch_builder.
        try:
            summary = self.builder.inspect_episode(payload)

        except Exception as exc:
            LOGGER.warning(
                "Archivio rifiutato: %s", exc
            )

            return {
                "error": f"Archivio non valido: {exc}"
            }, 400

        episode_id = str(summary["episode_id"])

        if supplied_id is not None and supplied_id != episode_id:
            return {
                "error": (
                    "X-Episode-ID diverso da manifest.episode_id"
                )
            }, 400

        if self.cfg.mode == "mixed":
            # ZIPs remain the online Plays wire format; E comes from local PKLs.
            try:
                with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                if (manifest.get("source") != "plays"
                        or re.fullmatch(r"P\d{5,}", episode_id) is None):
                    raise ValueError("E2 accepts only online Plays P*.zip")
            except Exception as exc:
                return {"error": f"E2 source not valid: {exc}"}, 400

        # Seconda verifica sotto lock:
        # un'altra richiesta potrebbe aver occupato la coda.
        with self._cv:

            existing = self._episodes.get(episode_id)

            if existing is not None:
                return {
                    "episode_id": episode_id,
                    "status": existing["status"],
                    "checkpointed": existing["checkpointed"],
                }, 200

            if self._phase != "ready" or self._stop:
                return {
                    "error": "Server non pronto",
                    "server": self._phase,
                }, 503

            if len(self._pending) >= self.cfg.max_queued_episodes:
                return {
                    "error": "Coda piena",
                    "queued": len(self._pending),
                }, 503

            if self.cfg.mode == "mixed":
                play_number = int(episode_id[1:])
                if play_number <= self._mixed_last_checkpointed_play_number:
                    return {"error": "Play already consolidated in the E2 checkpoint"}, 409
                if play_number <= self._mixed_last_accepted_play_number:
                    return {"error": "Play ID must increase monotonically in E2"}, 409
                # Count every accepted Play since the checkpoint, including
                # already trained ones; this avoids a race with replay commit.
                reserved = sum(int(record["play_windows"])
                               for record in self._episodes.values())
                required = self._exploration_initial_offset + reserved + int(summary["windows"])
                if required > self._exploration_catalog.total_windows:
                    return {
                        "error": "Insufficient unused offline Exploration windows",
                        "exploration_remaining_unreserved": (
                            self._exploration_catalog.total_windows
                            - self._exploration_initial_offset - reserved
                        ),
                    }, 409
                self._mixed_last_accepted_play_number = play_number

            # Registrazione in memoria del video accettato.
            self._episodes[episode_id] = {
                "episode_id": episode_id,
                "status": "queued",
                "checkpointed": False,

                "received_at": _now(),
                "started_at": None,
                "finished_at": None,

                "frames": int(summary["frames"]),
                "windows_total": (2 * int(summary["windows"])
                                  if self.cfg.mode == "mixed" else int(summary["windows"])),
                "play_windows": int(summary["windows"]) if self.cfg.mode == "mixed" else None,
                "windows_done": 0,
                "latest_prompt": None,
                "latest_source": None,
                "latest_original_episode_id": None,
                "latest_window_index": None,

                "latest_wm_loss": None,
                "error": None,
                "metrics": None,
            }

            self._pending.append(
                QueuedEpisode(episode_id, payload)
            )
            self._notify_monitor_unlocked(f"RICEVUTO {episode_id}")

            # Sveglia il worker GPU, se sta aspettando.
            self._cv.notify_all()

        LOGGER.info(
            "ACCEPT %s: frames=%d windows=%d bytes=%d",
            episode_id,
            summary["frames"],
            summary["windows"],
            len(payload),
        )

        # 202 Accepted: episodio accodato, NON addestrato.
        return {
            "episode_id": episode_id,
            "status": "queued",
            "frames": summary["frames"],
            "windows": summary["windows"],
        }, 202

    def _window_progress(
        self,
        episode_id: str,
        done: int,
        wm_loss: float,
        *,
        source: str | None = None,
        original_episode_id: str | None = None,
        window_index: int | None = None,
        prompt: str | None = None,
    ) -> None:
        """Aggiorna il progresso visibile durante il training."""

        try:
            with self._lock:
                self._episodes[episode_id]["windows_done"] = done
                self._episodes[episode_id]["latest_wm_loss"] = wm_loss
                self._episodes[episode_id]["latest_source"] = source
                self._episodes[episode_id]["latest_original_episode_id"] = original_episode_id
                self._episodes[episode_id]["latest_window_index"] = window_index
                self._episodes[episode_id]["latest_prompt"] = prompt
                self._notify_monitor_unlocked(f"FINESTRA {episode_id} {done}/{self._episodes[episode_id]['windows_total']}")

        except Exception:
            LOGGER.exception(
                "Impossibile aggiornare telemetria per %s",
                episode_id,
            )

    def _run(self) -> None:
        """Worker GPU unico.

        1. Carica DROID oppure riprende da un checkpoint.
        2. Attende gli episodi nella FIFO.
        3. Costruisce le finestre tramite il batch builder.
        4. Esegue il training dell'intero episodio.
        5. Aggiorna gli stati e i contatori.
        """

        try:
            root = Path(self.cfg.checkpoint_root)

            # Identifica l'ultimo checkpoint A/B disponibile.
            resume = _checkpoint_latest(root)

            LOGGER.info(
                "Caricamento VLA-JEPA da %s",
                resume or "DROID base",
            )

            # Il modello viene creato una sola volta.
            trainer = self.trainer_factory(resume)
            if self.cfg.mode == "mixed":
                self._initialize_mixed(trainer, resume)

            with self._cv:
                self._trainer = trainer

                self._stats = _trainer_monitor_stats(trainer)

                self._checkpointed_videos = (
                    int(trainer.processed_videos)
                    if resume
                    else 0
                )

                self._last_checkpoint = (
                    str(resume)
                    if resume
                    else None
                )

                self._phase = "ready"
                self._notify_monitor_unlocked("MODELLO PRONTO")
                self._cv.notify_all()

            LOGGER.info(
                "READY: GPU worker FIFO, resume=%s",
                resume,
            )

        except Exception as exc:
            LOGGER.exception(
                "Caricamento VLA-JEPA fallito"
            )

            with self._cv:
                self._phase = "error"
                self._error = (
                    f"Inizializzazione fallita: {exc}"
                )
                self._notify_monitor_unlocked("ERRORE INIZIALIZZAZIONE")

            return

        # Ciclo continuo della sessione.
        while True:

            with self._cv:

                # Attendi un episodio disponibile.
                while not self._pending and not self._stop:
                    self._cv.wait()

                if self._stop:
                    self._phase = "stopped"
                    self._notify_monitor_unlocked("STOP")
                    return

                # Preleva il primo episodio della FIFO.
                item = self._pending.popleft()
                episode_id = item.episode_id
                payload = item.payload

                self._current = episode_id

                info = self._episodes[episode_id]
                info["status"] = "training"
                info["started_at"] = _now()
                info["windows_done"] = 0
                self._notify_monitor_unlocked(f"TRAIN START {episode_id}")

            LOGGER.info(
                "TRAIN %s: %d finestre",
                episode_id,
                info["windows_total"],
            )

            try:
                # Same ZIP preprocessing in E1 and for the Plays half of E2.
                # The last yielded window metadata is available synchronously
                # to the callback after its backward/optimizer step.
                last_window: dict[str, Any] = {}
                windows = self.builder.iter_episode_windows(payload)

                def play_batches():
                    for window in windows:
                        last_window.update(
                            source="P" if episode_id.startswith("P") else "E",
                            original_episode_id=window.episode_id,
                            window_index=window.window_index,
                            prompt=window.batch["task"][0],
                        )
                        yield window.batch

                if self.cfg.mode == "exploration":
                    # E1: original single train_episode call, with prompt logging.
                    result = self._trainer.train_episode(
                        episode_id,
                        play_batches(),
                        on_window_complete=lambda n, loss: self._window_progress(
                            episode_id, n, loss, **last_window
                        ),
                    )
                else:
                    n_play = int(info["play_windows"])
                    spans, next_offset = self.builder.plan_exploration_replay(
                        self._exploration_catalog, self._exploration_offset, n_play
                    )
                    # Catalog startup already validated every PKL and window.
                    # The planner checked file size/mtime before training this Play.

                    play_result = self._trainer.train_episode(
                        episode_id,
                        play_batches(),
                        on_window_complete=lambda n, loss: self._window_progress(
                            episode_id, n, loss, **last_window
                        ),
                    )
                    if int(play_result["windows"]) != n_play:
                        raise RuntimeError("Number of trained Play windows disagrees with ZIP")

                    # Source counters in the unchanged trainer recognize E* IDs.
                    replay_episode_id = f"E_REPLAY_{episode_id}"
                    replay_window: dict[str, Any] = {}

                    def exploration_batches():
                        for span in spans:
                            for window in self.builder.iter_exploration_pkl_windows(
                                span.entry.path, start_window=span.first_window,
                                stop_window=span.stop_window
                            ):
                                replay_window.update(
                                    source="E", original_episode_id=window.episode_id,
                                    window_index=window.window_index,
                                    prompt=window.batch["task"][0],
                                )
                                yield window.batch

                    exploration_result = self._trainer.train_episode(
                        replay_episode_id,
                        exploration_batches(),
                        on_window_complete=lambda n, loss: self._window_progress(
                            episode_id, n_play + n, loss, **replay_window
                        ),
                    )
                    if int(exploration_result["windows"]) != n_play:
                        raise RuntimeError("Replay window count mismatch")

                    self._exploration_offset = next_offset
                    if exploration_result.get("checkpoint_path"):
                        # The original trainer saves after ten E/P units:
                        # five complete Play+Exploration pairs.
                        self._write_mixed_checkpoint_marker(
                            exploration_result["checkpoint_path"], episode_id
                        )
                    if (self._trainer.source_window_counts["E"]
                            != self._trainer.source_window_counts["P"]):
                        raise RuntimeError("E2 source window counters are unbalanced")
                    with self._lock:
                        self._stats = _trainer_monitor_stats(self._trainer)

                    result = {
                        **exploration_result,
                        "episode_id": episode_id,
                        "windows": 2 * n_play,
                        "play_windows": n_play,
                        "exploration_windows": n_play,
                        "mean_wm_loss": (play_result["mean_wm_loss"]
                                         + exploration_result["mean_wm_loss"]) / 2,
                        "play_mean_wm_loss": play_result["mean_wm_loss"],
                        "exploration_mean_wm_loss": exploration_result["mean_wm_loss"],
                        "optimizer_steps_for_video": (
                            play_result["optimizer_steps_for_video"]
                            + exploration_result["optimizer_steps_for_video"]
                        ),
                    }
                    LOGGER.info(
                        "E2 PAIR %s: P=%d E=%d offline_cursor=%d/%d",
                        episode_id, n_play, n_play, self._exploration_offset,
                        self._exploration_catalog.total_windows,
                    )

                with self._cv:
                    info = self._episodes[episode_id]

                    info.update({
                        "status": "trained",
                        "finished_at": _now(),
                        "windows_done": info["windows_total"],
                        "metrics": result,
                        "error": None,
                    })

                    self._stats = _trainer_monitor_stats(self._trainer)

                    if result.get("checkpoint_path"):

                        # Il trainer torna dopo il salvataggio
                        # riuscito: tutti i video precedenti
                        # sono ora consolidati nel checkpoint.
                        self._last_checkpoint = str(
                            result["checkpoint_path"]
                        )

                        self._checkpointed_videos = int(
                            self._trainer.processed_videos
                        )

                        for record in self._episodes.values():
                            if record["status"] == "trained":
                                record["checkpointed"] = True

                    self._current = None
                    self._notify_monitor_unlocked(f"TRAINED {episode_id}")

                LOGGER.info(
                    "TRAINED %s: video=%s step=%s checkpoint=%s",
                    episode_id,
                    result["processed_videos"],
                    result["global_step"],
                    result.get("checkpoint_path"),
                )

            except Exception as exc:
                LOGGER.exception(
                    "Errore training %s",
                    episode_id,
                )

                with self._cv:
                    self._episodes[episode_id]["status"] = "failed"
                    self._episodes[episode_id]["error"] = str(exc)[:2000]
                    self._episodes[episode_id]["finished_at"] = _now()

                    # Un backward potrebbe essere rimasto incompleto.
                    # Blocchiamo il server per evitare di continuare
                    # con uno stato dell'ottimizzatore incerto.
                    self._phase = "error"
                    self._error = (
                        f"Fallito {episode_id}: {exc}. "
                        "Riavviare dall'ultimo checkpoint."
                    )
                    self._current = None
                    self._notify_monitor_unlocked(f"ERRORE TRAIN {episode_id}")

                return

            finally:
                # Rilascia lo ZIP dell'episodio terminato.
                payload = None
                item = None
                windows = None

                with self._lock:
                    self._current = None


def format_server_monitor(service: TrainingService, event: str = "STATUS") -> str:
    """Un evento = una riga stampata: FIFO server, GPU e safe_to_stop."""
    status = service.status()
    fifo = service.queue()
    current = status["current_episode_id"]
    ongoing = service.episode(current) if current else None
    ids = [str(item["episode_id"]) for item in fifo["queued"]]
    stats = status.get("trainer") or {}
    progress = ""
    if ongoing is not None:
        done = int(ongoing["windows_done"])
        total = int(ongoing["windows_total"])
        filled = int(20 * done / max(total, 1))
        bar = "#" * filled + "-" * (20 - filled)
        loss = ongoing.get("latest_wm_loss")
        loss_text = f" loss={loss:.5f}" if isinstance(loss, (int, float)) else ""
        progress = f" [{bar}] {done}/{total} finestre{loss_text}"
    counts = (
        f" E={stats.get('videos_exploration', 0)}v/{stats.get('windows_exploration', 0)}w"
        f" P={stats.get('videos_plays', 0)}v/{stats.get('windows_plays', 0)}w"
    )
    if stats.get("videos_legacy", 0):
        counts += f" legacy={stats['videos_legacy']}v"
    if stats.get("videos_other", 0):
        counts += f" other={stats['videos_other']}v"
    prompt_text = ""
    if event.startswith("FINESTRA") and ongoing is not None:
        # Only describe the just-completed window, not a stale earlier event.
        label = ongoing.get("latest_prompt")
        if label is not None:
            prompt_text = (
                f" | source={ongoing.get('latest_source') or '?'}"
                f" original={ongoing.get('latest_original_episode_id') or '?'}"
                f" window_index={ongoing.get('latest_window_index')}"
                f" | prompt={label}"
            )
    return (
        f"[{event}] server={status['server']} | FIFO={status['queued']}/{status['queue_capacity']}"
        f" [{', '.join(ids) or '-'}] | GPU={current or '-'}{progress}"
        f" | video={stats.get('processed_videos', '?')}" + counts
        + f" step={stats.get('global_step', '?')}"
        f" pending_windows={stats.get('pending_windows', '?')}"
        f" checkpointed={status['checkpointed_videos']}"
        f" | safe_to_stop={'TRUE' if status['safe_to_stop'] else 'FALSE'}"
        + (f" | ERRORE={status['error']}" if status.get("error") else "")
        + prompt_text
    )


class ServerStatusReporter:
    """Stampa SOLO su eventi reali, compreso ogni backward completato.

    La callback deposita una fotografia dello stato in SimpleQueue;
    il thread di log non blocca il worker CUDA e non usa polling.
    """

    def __init__(self, service: TrainingService):
        self.service = service
        self._messages: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vla-jepa-terminal-monitor"
        )

    def start(self) -> None:
        self._thread.start()
        self.service.subscribe_monitor(self._messages.put)

    def close(self) -> None:
        self._messages.put(None)
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            item = self._messages.get()
            if item is None:
                return
            LOGGER.info("%s", item)


def create_app(service: TrainingService):
    """Crea l'app REST Flask.

    Il modello NON viene caricato nelle route HTTP.
    Tutto il training avviene nel worker GPU dedicato.
    """

    from flask import Flask, jsonify, request

    app = Flask(__name__)

    # Non e' configurata autenticazione tramite token.
    # Non e' configurato MAX_CONTENT_LENGTH.
    # Rimane soltanto il limite sul numero di episodi in coda.

    @app.get("/")
    def index():
        return jsonify({
            "service": "VLA-JEPA online training",
            "endpoints": [
                "GET /health",
                "GET /status",
                "GET /queue",
                "GET /episodes/<id>",
                "POST /train_episode",
            ],
        })

    @app.get("/health")
    def health():
        data = service.status()

        return jsonify(data), (
            200
            if data["server"] == "ready"
            else 503
        )

    @app.get("/status")
    def status():
        return jsonify(service.status())

    @app.get("/queue")
    def queue():
        return jsonify(service.queue())

    @app.get("/episodes/<path:episode_id>")
    def episode_status(episode_id: str):
        result = service.episode(episode_id)

        if result is not None:
            return jsonify(result), 200

        return jsonify({
            "error": (
                "Episodio sconosciuto nella sessione corrente"
            )
        }), 404

    @app.post("/train_episode")
    def train_episode():
        result, code = service.submit(
            request,
            request.headers.get("X-Episode-ID"),
        )

        return jsonify(result), code

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flask VLA-JEPA online: "
            "coda RAM, checkpoint A/B"
        )
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/exploration",
    )

    parser.add_argument(
        "--pretrained-path",
        default="checkpoints/base",
    )

    parser.add_argument(
        "--max-queue",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--qwen-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--predictor-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Seed PyTorch, NumPy e Python per il training.",
    )

    parser.add_argument(
        "--mode", choices=("exploration", "mixed"), default="exploration",
        help="exploration=original E1; mixed=online Plays + offline PKL Exploration",
    )
    parser.add_argument(
        "--offline-exploration-dir", default=None,
        help="Trusted E*.pkl directory, only required for mode=mixed",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = ServerConfig(
        host=args.host,
        port=args.port,
        checkpoint_root=args.checkpoint_root,
        max_queued_episodes=args.max_queue,
        mode=args.mode,
        offline_exploration_dir=args.offline_exploration_dir,
    )

    def factory(resume_path: Path | None):
        """Costruisce il trainer nel worker GPU.

        Se resume_path e' disponibile, riparte da quello.
        Altrimenti carica il pretraining DROID da checkpoints/base.
        """

        from vla_jepa_online.training_server.online_trainer import (
            OnlineTrainerConfig,
            OnlineVideoTrainer,
        )

        return OnlineVideoTrainer(
            OnlineTrainerConfig(
                pretrained_path=args.pretrained_path,
                resume_from=resume_path,
                checkpoint_root=cfg.checkpoint_root,
                device="cuda",
                seed=args.seed,
                qwen_learning_rate=args.qwen_lr,
                predictor_learning_rate=args.predictor_lr,
            )
        )

    service = TrainingService(
        cfg,
        trainer_factory=factory,
    )

    app = create_app(service)
    monitor = ServerStatusReporter(service)
    monitor.start()

    LOGGER.info(
        "PANNELLO SERVER: GET /status, GET /queue, GET /episodes/<id>. "
        "Monitor automatico; Ctrl+C SOLO se safe_to_stop=TRUE."
    )

    LOGGER.warning(
        "Avviare un solo processo Flask; "
        "niente reloader o worker multipli"
    )

    try:
        app.run(
            host=cfg.host,
            port=cfg.port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    finally:
        # Mostra sempre l'ultimo valore PRIMA dell'arresto cooperativo.
        monitor.close()
        LOGGER.warning("ARRESTO RICHIESTO: %s", format_server_monitor(service))
        service.close()


if __name__ == "__main__":
    main()
