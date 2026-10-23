"""Offline dataset manifest and adapters (trusted local PKL or paired RGB folders).

The two experiments use exactly the same reader. Only the JSONL index differs.
Each JSONL row contains episode_id, task and either pkl_path, or front_dir
and gripper_dir. Optional task_id is used to exclude held-out tasks.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
FRONT_KEYS = ("front_camera_image", "camera_front_image", "front_rgb")
WRIST_KEYS = ("gripper_camera_image", "camera_gripper_image", "wrist_rgb")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    task: str
    task_id: int | None
    pkl_path: Path | None
    front_dir: Path | None
    gripper_dir: Path | None


def _resolve(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty path, got {value!r}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_index(path: str | Path) -> tuple[list[EpisodeSpec], str]:
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    entries: list[EpisodeSpec] = []
    seen: set[str] = set()
    for lineno, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            ident, task = d["episode_id"], d["task"]
            if not isinstance(ident, str) or not ident.strip() or "/" in ident:
                raise ValueError("episode_id must be nonempty without slash")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("task prompt missing")
            if ident in seen:
                raise ValueError(f"Duplicate episode_id {ident}")
            tid = d.get("task_id")
            if tid is not None and (type(tid) is not int or tid < 0):
                raise ValueError("task_id must be nonnegative integer")
            is_pkl = "pkl_path" in d
            is_dirs = "front_dir" in d and "gripper_dir" in d
            if is_pkl == is_dirs:
                raise ValueError("Provide exactly one: pkl_path OR both RGB directories")
            if is_pkl:
                pkl = _resolve(path.parent, d["pkl_path"])
                if not pkl.is_file() or pkl.suffix.lower() != ".pkl":
                    raise FileNotFoundError(pkl)
                fd = gd = None
            else:
                pkl = None
                fd = _resolve(path.parent, d["front_dir"])
                gd = _resolve(path.parent, d["gripper_dir"])
                if not fd.is_dir() or not gd.is_dir():
                    raise FileNotFoundError(f"Missing RGB dir: {fd} or {gd}")
            entries.append(EpisodeSpec(ident, task, tid, pkl, fd, gd))
            seen.add(ident)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid manifest {path}:{lineno}: {exc}") from exc
    if not entries:
        raise ValueError(f"No episodes listed in {path}")
    return entries, fingerprint


def _file_list(directory: Path) -> tuple[Path, ...]:
    paths = tuple(sorted((p for p in directory.iterdir()
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                         key=lambda p: p.name))
    if not paths:
        raise ValueError(f"No frames in {directory}")
    return paths


def _extract_array(obs: dict, candidates: tuple[str, ...], camera: str) -> np.ndarray:
    for key in candidates:
        if key in obs:
            arr = np.asarray(obs[key])
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"{camera} {key}: expected uint8 BGR [H,W,3], got {arr.shape}/{arr.dtype}")
            return arr
    raise ValueError(f"Missing {camera} RGB in observation; tried {candidates}; found {tuple(obs)}")


def _load_traj(path: Path) -> Any:
    # A legacy savers.Trajectory object is reconstructed by Python pickle.
    # The original savers module MUST be on PYTHONPATH. Never unpickle untrusted data.
    importlib.import_module("savers")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "traj" not in payload:
        raise ValueError(f"Expected PKL dict with key 'traj': {path}")
    traj = payload["traj"]
    if not hasattr(traj, "__len__") or not callable(getattr(traj, "get", None)):
        raise ValueError(f"Unsupported trajectory API in {path}")
    return traj


def _traj_step(traj: Any, index: int) -> dict:
    # Dataset collector typically accepts decompress=False; some legacy
    # Multi-Task-LFD Trajectory implementations only accept get(index).
    try:
        item = traj.get(index, decompress=False)
    except TypeError:
        item = traj.get(index)
    if not isinstance(item, dict) or not isinstance(item.get("obs"), dict):
        raise ValueError(f"Trajectory step {index} does not contain obs dict")
    return item["obs"]


class OfflineEpisode:
    """Context-managed, per-episode reader; no cross-episode windows."""
    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        if spec.pkl_path is not None:
            self.traj = _load_traj(spec.pkl_path)
            self.front_files = self.wrist_files = ()
            self.count = len(self.traj)
        else:
            self.traj = None
            self.front_files = _file_list(spec.front_dir)
            self.wrist_files = _file_list(spec.gripper_dir)
            if len(self.front_files) != len(self.wrist_files):
                raise ValueError(f"{spec.episode_id}: RGB view frame counts differ")
            self.count = len(self.front_files)
        if self.count < 8:
            raise ValueError(f"{spec.episode_id}: at least 8 frames required; got {self.count}")

    def rgb(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Two BGR uint8 arrays (matching online_batch_builder convention)."""
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self.traj is not None:
            obs = _traj_step(self.traj, index)
            return (_extract_array(obs, FRONT_KEYS, "front"),
                    _extract_array(obs, WRIST_KEYS, "gripper"))
        import cv2
        a = cv2.imread(str(self.front_files[index]), cv2.IMREAD_COLOR)
        b = cv2.imread(str(self.wrist_files[index]), cv2.IMREAD_COLOR)
        if a is None or b is None:
            raise ValueError(f"Cannot read RGB frame {index} in {self.spec.episode_id}")
        return a, b

    def validate(self, crop_front, crop_gripper, *, max_frames: int = 2000) -> int:
        if self.count > max_frames:
            raise ValueError(f"{self.spec.episode_id}: too many frames {self.count}")
        expected: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        for idx in range(self.count):
            front, wrist = self.rgb(idx)
            shapes = (front.shape, wrist.shape)
            if expected is None:
                expected = shapes
            if shapes != expected:
                raise ValueError(f"{self.spec.episode_id}: changing RGB dimensions at frame {idx}")
            for label, img, crop in (("front", front, crop_front), ("gripper", wrist, crop_gripper)):
                if crop is not None:
                    if (min(crop.left, crop.top, crop.right, crop.bottom) < 0
                            or img.shape[1] <= crop.left + crop.right
                            or img.shape[0] <= crop.top + crop.bottom):
                        raise ValueError(f"{self.spec.episode_id}: invalid {label} crop for {img.shape}")
        return self.count
