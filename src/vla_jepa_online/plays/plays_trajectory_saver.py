"""PKL Plays identico nel layout di osservazioni a TrajectorySaver Exploration.

Invece di fabbricare waypoint, scrive metadati propri; per leggere riutilizza
TrajectorySaver.load_episode() esistente (e quindi savers.Trajectory originale).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from vla_jepa_online.exploration.trajectory_recorder import RecordedEpisode
from vla_jepa_online.exploration.trajectory_saver import (
    TrajectorySaver, _trajectory_type, _decode_png_bgr,
)
from vla_jepa_online.plays.plays import PlayRequest, semantic_prompt


class PlaysTrajectorySaver:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def next_episode_id(self) -> str:
        existing = [int(p.stem[1:]) for p in self.root.glob("P*.pkl")
                    if re.fullmatch(r"P\d{5,}", p.stem)]
        return f"P{max(existing, default=0) + 1:05d}"

    def save(self, episode_id: str, episode: RecordedEpisode, req: PlayRequest,
             *, operator_success: bool, operator_approved: bool) -> Path:
        if not re.fullmatch(r"P\d{5,}", episode_id):
            raise ValueError("ID Plays deve essere P00001, P00002, ...")
        if not episode.frames:
            raise ValueError("Episodio vuoto")
        if not operator_approved or not operator_success:
            raise ValueError(
                "Un episodio scartato o fallito non deve essere salvato"
            )
        target = self.root / f"{episode_id}.pkl"
        if target.exists():
            raise FileExistsError(target)
        meta = {
            "schema_version": 1,
            "episode_id": episode_id,
            "source": "plays",
            "sampling_mode": "motion_step",
            "prompt_mode": "episode_prompt",
            "terminal_padding": "repeat_last",
            "episode_prompt": semantic_prompt(req),
            "play_type": req.task_id,
            "object_color": req.color,
            "target_color": req.target_color,
            "pick_uv": list(req.pick_uv),
            "place_uv": list(req.place_uv),
            "pick_xyz": list(req.pick_xyz),
            "place_xyz": list(req.place_xyz),
            "pick_y_offset_m": req.pick_y_offset_m,
            "rotation_deg": req.rotation_deg,
            "operator_success": bool(operator_success),
            "operator_approved": bool(operator_approved),
            "recording_fps_limit": episode.fps,
            "ai_controller_target": "vla_jepa_plays_client",
        }
        Trajectory = _trajectory_type()
        traj = Trajectory(config_str=json.dumps(meta, separators=(",", ":")))
        for index, frame in enumerate(episode.frames, start=1):
            if frame.index != index or frame.robot_state is None or frame.commanded_action is None:
                raise ValueError(f"Frame {index}: sequenza o stato robotico non validi")
            action = np.asarray(frame.commanded_action, dtype=np.float64)
            if action.shape != (8,) or not np.isfinite(action).all():
                raise ValueError(f"Frame {index}: commanded_action non 8D finita")
            state = frame.robot_state
            obs = {
                "front_camera_image": _decode_png_bgr(frame.front_png, "front"),
                "gripper_camera_image": _decode_png_bgr(frame.gripper_png, "gripper"),
                "joint_pos": np.asarray(state["joint_pos"], dtype=np.float64),
                "joint_vel": np.asarray(state["joint_vel"], dtype=np.float64),
                "gripper_qpos": np.asarray(state["gripper_qpos"], dtype=np.float64),
                "gripper_qvel": np.asarray(state["gripper_qvel"], dtype=np.float64),
                "eef_pos": np.asarray(state["eef_pos"], dtype=np.float64),
                "eef_quat": np.asarray(state["eef_quat"], dtype=np.float64),
            }
            info = {
                "index": index,
                "front_stamp_ns": frame.front_stamp_ns,
                "gripper_stamp_ns": frame.gripper_stamp_ns,
                "event": frame.event,
                "segment_id": frame.segment_id,
                "motion_label": frame.motion_label,
                "robot_state": state,
                "commanded_action": list(frame.commanded_action),
            }
            traj.append(obs=obs, action=action, info=info,
                        reward=None, done=(index == len(episode.frames)))
        fd, tmp = tempfile.mkstemp(prefix=f".{episode_id}.", suffix=".part", dir=self.root)
        os.close(fd)
        try:
            traj.save(tmp, **meta)
            with open(tmp, "rb") as handle:
                os.fsync(handle.fileno())
            os.link(tmp, target)  # nessun overwrite di file preesistente
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target

    @staticmethod
    def load_episode(path: str | Path):
        return TrajectorySaver.load_episode(path)
