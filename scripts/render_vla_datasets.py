#!/usr/bin/env python3
"""Render trusted preprocessed VLA-JEPA Plays/Human/Robot PKLs as MP4.

Video content: FRONT | GRIPPER, with the dataset's actual language prompt below.
All preprocessed ndarray images are RGB; OpenCV decodes compressed JPEG as BGR.
Human E3 trajectories contain only the front view; duplicate front in the right pane,
just like the two-view input adaptation for E3. This is NOT a real gripper view.

The `Trajectory` class contained in the pickles must be importable before loading:
set PYTHONPATH to include the project's `training` and `multi_task_il/datasets`.
Never unpickle untrusted files.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

FRONT_KEYS = ("camera_front_image", "front_camera_image")
GRIPPER_KEYS = ("camera_gripper_image", "gripper_camera_image")
PROMPT_KEYS = ("episode_prompt", "language_instruction", "instruction", "task")


def get_observation(traj, index: int) -> dict:
    if hasattr(traj, "get"):
        try:
            step = traj.get(index, decompress=False)
        except TypeError:
            step = traj.get(index)
    else:
        step = traj[index]
    if isinstance(step, dict):
        obs = step.get("obs", step.get("observation"))
    elif isinstance(step, (tuple, list)) and len(step) >= 1:
        obs = step[0]
    else:
        raise TypeError(f"Unsupported trajectory step {index}: {type(step)}")
    if not isinstance(obs, dict):
        raise TypeError(f"Step {index}: expected observation dict; got {type(obs)}")
    return obs


def image_from_obs(obs: dict, keys: tuple[str, ...], path: Path, index: int):
    for key in keys:
        if key in obs:
            return obs[key]
    raise KeyError(f"{path.name} frame {index}: none of {keys}; keys={list(obs)}")


def to_bgr(value, size: int) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        bgr = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if arr.dtype != np.uint8:
                raise ValueError(f"Expected uint8 RGB image; received {arr.dtype}")
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.dtype == np.uint8 and (
            arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape)
        ):
            bgr = cv2.imdecode(arr.reshape(-1), cv2.IMREAD_COLOR)
        else:
            raise ValueError(f"Unsupported image shape={arr.shape}, dtype={arr.dtype}")
    if bgr is None:
        raise ValueError("Could not decode compressed image")
    if bgr.shape[:2] != (size, size):
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(bgr)


def find_prompt(payload: dict, traj, path: Path) -> str:
    sources = [payload] if isinstance(payload, dict) else []
    first_obs = get_observation(traj, 0)
    sources.append(first_obs)
    # Also check per-frame metadata, if present.
    if hasattr(traj, "get"):
        try:
            first_step = traj.get(0, decompress=False)
        except TypeError:
            first_step = traj.get(0)
        if isinstance(first_step, dict) and isinstance(first_step.get("info"), dict):
            sources.append(first_step["info"])
    for source in sources:
        for key in PROMPT_KEYS:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.strip().split())
    raise ValueError(
        f"{path}: no textual prompt in fields {PROMPT_KEYS}. "
        "The MP4 was not generated rather than inventing a prompt."
    )


def wrap_prompt(text: str, width: int, font: int, scale: float, thick: int) -> list[str]:
    words = text.split()
    if not words:
        raise ValueError("Empty prompt")
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if cv2.getTextSize(candidate, font, scale, thick)[0][0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
                current = word
            else:
                # If a single long word does not fit, break it into pieces.
                while cv2.getTextSize(word, font, scale, thick)[0][0] > width:
                    limit = 1
                    while limit < len(word) and cv2.getTextSize(word[:limit+1], font, scale, thick)[0][0] <= width:
                        limit += 1
                    lines.append(word[:limit])
                    word = word[limit:]
                current = word
    if current:
        lines.append(current)
    return lines


def render(path: Path, output_dir: Path, fps: float, size: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)  # Trusted local dataset files only!
    traj = payload["traj"] if isinstance(payload, dict) and "traj" in payload else payload
    count = int(traj.T) if hasattr(traj, "T") else len(traj)
    if count < 1:
        raise ValueError(f"{path}: no frames")
    prompt = find_prompt(payload, traj, path)

    first_obs = get_observation(traj, 0)
    has_gripper = any(k in first_obs for k in GRIPPER_KEYS)
    if not has_gripper and not any(k in str(path) for k in ("human_rgb_pick_place", "human_traj_")):
        raise KeyError(f"{path.name} has no gripper view and is not a human E3 PKL")

    width = size * 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.53, 1
    lines = wrap_prompt(prompt, width - 26, font, scale, thick)
    line_height = 25
    prompt_height = max(58, 20 + line_height * len(lines))
    height = size + prompt_height
    # mp4v expects even dimensions on some FFmpeg builds.
    height += height % 2

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{path.stem}_front_gripper.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {out}")

    # Same prompt panel for every video frame; don't draw other annotations.
    prompt_panel = np.full((height - size, width, 3), (32, 32, 32), dtype=np.uint8)
    top = (prompt_panel.shape[0] - len(lines) * line_height) // 2 + 18
    for line_idx, line in enumerate(lines):
        text_width = cv2.getTextSize(line, font, scale, thick)[0][0]
        cv2.putText(
            prompt_panel, line, ((width - text_width) // 2, top + line_idx * line_height),
            font, scale, (255, 255, 255), thick, cv2.LINE_AA,
        )
    try:
        for index in range(count):
            obs = get_observation(traj, index)
            front = to_bgr(image_from_obs(obs, FRONT_KEYS, path, index), size)
            if has_gripper:
                gripper = to_bgr(image_from_obs(obs, GRIPPER_KEYS, path, index), size)
            else:
                gripper = front  # E3 human: no gripper camera; front view duplicated.
            frame = np.vstack((np.hstack((front, gripper)), prompt_panel))
            writer.write(frame)
    finally:
        writer.release()

    suffix = " (human: duplicated front, no wrist camera)" if not has_gripper else ""
    print(f"OK {out} | {count} frames | {width}x{height} | {fps:g} fps{suffix}")
    print(f"   Prompt: {prompt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render VLA-JEPA preprocessed PKLs as FRONT | GRIPPER with prompt")
    parser.add_argument("--pkl", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    if args.fps <= 0 or args.size <= 0:
        parser.error("--fps and --size must be positive")
    for path in args.pkl:
        render(path, args.output_dir, args.fps, args.size)


if __name__ == "__main__":
    main()
