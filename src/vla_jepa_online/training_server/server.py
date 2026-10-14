
"""Server Flask minimale per il pretraining online VLA-JEPA.

Un singolo processo Flask e un singolo worker GPU condividono una coda FIFO in
RAM. Non esistono SQLite, spool su disco o replay automatici degli episodi.

Il client continua a fare POST /train_episode con un archivio ZIP di JPEG e
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
import threading
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

        self.cfg = cfg
        self.trainer_factory = trainer_factory

        self.builder = builder or OnlineBatchBuilder(
            OnlineBatchBuilderConfig()
        )

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

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

        # Un solo worker GPU per tutto il server.
        self._worker = threading.Thread(
            target=self._run,
            name="vla-jepa-gpu-worker",
            daemon=True,
        )

        self._worker.start()

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

            # Registrazione in memoria del video accettato.
            self._episodes[episode_id] = {
                "episode_id": episode_id,
                "status": "queued",
                "checkpointed": False,

                "received_at": _now(),
                "started_at": None,
                "finished_at": None,

                "frames": int(summary["frames"]),
                "windows_total": int(summary["windows"]),
                "windows_done": 0,

                "latest_wm_loss": None,
                "error": None,
                "metrics": None,
            }

            self._pending.append(
                QueuedEpisode(episode_id, payload)
            )

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
    ) -> None:
        """Aggiorna il progresso visibile durante il training."""

        try:
            with self._lock:
                self._episodes[episode_id]["windows_done"] = done
                self._episodes[episode_id]["latest_wm_loss"] = wm_loss

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

            with self._cv:
                self._trainer = trainer

                self._stats = {
                    "global_step": int(trainer.global_step),
                    "processed_videos": int(
                        trainer.processed_videos
                    ),
                    "pending_windows": int(
                        trainer.pending_windows
                    ),
                }

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

            return

        # Ciclo continuo della sessione.
        while True:

            with self._cv:

                # Attendi un episodio disponibile.
                while not self._pending and not self._stop:
                    self._cv.wait()

                if self._stop:
                    self._phase = "stopped"
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

            LOGGER.info(
                "TRAIN %s: %d finestre",
                episode_id,
                info["windows_total"],
            )

            try:
                # Estrazione lazy delle finestre.
                # Le immagini vengono preparate progressivamente.
                windows = self.builder.iter_episode_windows(payload)

                result = self._trainer.train_episode(
                    episode_id,
                    (window.batch for window in windows),
                    on_window_complete=lambda n, loss: (
                        self._window_progress(
                            episode_id,
                            n,
                            loss,
                        )
                    ),
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

                    self._stats = {
                        "global_step": int(
                            self._trainer.global_step
                        ),
                        "processed_videos": int(
                            self._trainer.processed_videos
                        ),
                        "pending_windows": int(
                            self._trainer.pending_windows
                        ),
                    }

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

                return

            finally:
                # Rilascia lo ZIP dell'episodio terminato.
                payload = None
                item = None
                windows = None

                with self._lock:
                    self._current = None


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
                qwen_learning_rate=args.qwen_lr,
                predictor_learning_rate=args.predictor_lr,
            )
        )

    service = TrainingService(
        cfg,
        trainer_factory=factory,
    )

    app = create_app(service)

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
        service.close()


if __name__ == "__main__":
    main()
