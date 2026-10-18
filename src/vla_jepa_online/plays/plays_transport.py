"""Trasporto specifico Plays: PKL -> manifest+PNG ZIP RAM -> Flask.

AsyncTrainingSender di Exploration viene riusato senza modificarlo.
ATTENZIONE: il server corrente non interpreta ancora episode_prompt/padding;
non abilitarne l'upload finché il batch builder non viene aggiornato.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from vla_jepa_online.plays.plays_trajectory_saver import PlaysTrajectorySaver


@dataclass(frozen=True)
class PlaysUploadTask:
    episode_id: str
    archive_path: Path


@dataclass(frozen=True)
class PlaysTransportConfig:
    server_url: str
    request_timeout_s: float
    max_queued_uploads: int


class PlaysTrainingTransport:
    def __init__(self, cfg: PlaysTransportConfig):
        if not cfg.server_url.startswith(("http://", "https://")) or cfg.request_timeout_s <= 0:
            raise ValueError("Server Plays non valido")
        self.cfg = cfg
        self.base_url = cfg.server_url.rstrip("/")

    @staticmethod
    def encode(meta: dict, episode) -> bytes:
        if meta.get("source") != "plays" or meta.get("prompt_mode") != "episode_prompt":
            raise ValueError("PKL non Plays")
        manifest = dict(meta)
        manifest["frames"] = []
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
            for frame in episode.frames:
                front = f"frames/front/{frame.index:06d}.png"
                wrist = f"frames/gripper/{frame.index:06d}.png"
                zf.writestr(front, frame.front_png)
                zf.writestr(wrist, frame.gripper_png)
                record = {
                    "index": frame.index,
                    "front_path": front, "gripper_path": wrist,
                    "front_stamp_ns": frame.front_stamp_ns,
                    "gripper_stamp_ns": frame.gripper_stamp_ns,
                    "event": frame.event, "segment_id": frame.segment_id,
                    "motion_label": frame.motion_label,
                    "robot_state": frame.robot_state,
                    "commanded_action": list(frame.commanded_action),
                }
                manifest["frames"].append(record)
            zf.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        return buf.getvalue()

    def upload(self, task: PlaysUploadTask) -> str:
        meta, episode = PlaysTrajectorySaver.load_episode(task.archive_path)
        if meta["episode_id"] != task.episode_id or not meta.get("operator_approved") \
                or not meta.get("operator_success"):
            raise ValueError("Episodio non approvato o ID non corrispondente")
        payload = self.encode(meta, episode)
        request = Request(
            self.base_url + "/train_episode", data=payload, method="POST",
            headers={
                "Content-Type": "application/zip", "Accept": "application/json",
                "X-Episode-ID": task.episode_id,
            },
        )
        # Come Exploration: nessun retry su timeout ambiguo e HTTP 202 = queued.
        with urlopen(request, timeout=self.cfg.request_timeout_s) as response:
            reply = json.loads(response.read().decode("utf-8"))
        if reply.get("episode_id") != task.episode_id or reply.get("status") not in (
            "queued", "training", "trained",
        ):
            raise RuntimeError(f"Risposta Flask anomala: {reply!r}")
        return str(reply["status"])
