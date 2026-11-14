#!/usr/bin/env python3
"""Export one or more trusted Exploration PKLs to side-by-side FRONT | GRIPPER MP4s.

Preprocessed exploration frames are expected to be uint8 RGB HxWx3.
The original, non-preprocessed PKLs can instead contain BGR and are supported
via --raw-color-order bgr. JPEG-encoded observations are decoded as BGR.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

FRONT_KEYS = ("camera_front_image", "front_camera_image")
GRIPPER_KEYS = ("camera_gripper_image", "gripper_camera_image")


def select_image(obs: dict, names: tuple[str, ...], pkl: Path, frame: int):
    for name in names:
        if name in obs:
            return obs[name]
    raise KeyError(f"{pkl.name}: frame {frame}: missing {names}; available keys={list(obs)}")


def decode_as_bgr(value, *, raw_color_order: str, size: int):
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {arr.dtype}")
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if raw_color_order == "rgb" else arr
    elif arr.dtype == np.uint8 and (arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape)):
        # OpenCV's imdecode returns BGR regardless of the original JPEG source.
        bgr = cv2.imdecode(arr.reshape(-1), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode compressed image")
    elif isinstance(value, (bytes, bytearray)):
        bgr = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode compressed image")
    else:
        raise ValueError(f"Unsupported image shape={arr.shape}, dtype={arr.dtype}")
    if bgr.shape[:2] != (size, size):
        bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(bgr)


def read_step(traj, index: int):
    try:
        step = traj.get(index, decompress=False)
    except TypeError:
        step = traj.get(index)
    if not isinstance(step, dict) or "obs" not in step:
        raise TypeError(f"Unexpected trajectory step type at {index}: {type(step)}")
    return step["obs"]


def render_file(path: Path, output_dir: Path, fps: float, size: int, raw_color_order: str):
    # Only load trusted PKLs: unpickling can execute Python code.
    with path.open("rb") as f:
        payload = pickle.load(f)
    traj = payload["traj"] if isinstance(payload, dict) and "traj" in payload else payload
    count = int(traj.T) if hasattr(traj, "T") else len(traj)
    if count < 1:
        raise ValueError(f"{path} contains no frames")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{path.stem}_front_gripper.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size * 2, size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not initialize MP4 writer for {out}")

    try:
        for i in range(count):
            obs = read_step(traj, i)
            front = decode_as_bgr(select_image(obs, FRONT_KEYS, path, i), raw_color_order=raw_color_order, size=size)
            gripper = decode_as_bgr(select_image(obs, GRIPPER_KEYS, path, i), raw_color_order=raw_color_order, size=size)
            combined = np.concatenate((front, gripper), axis=1)
            # Labels with a black outline remain readable on bright and dark images.
            # for x, label in ((8, "FRONT"), (size + 8, "GRIPPER")):
            #     cv2.putText(combined, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            #     cv2.putText(combined, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(combined)
    finally:
        writer.release()
    print(f"OK: {out} | {count} frames | {size * 2}x{size} | {fps:g} fps")


def main():
    p = argparse.ArgumentParser(description="Render Exploration PKLs as FRONT | GRIPPER MP4")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--pkl", type=Path, nargs="+", help="One or more explicit PKL paths")
    source.add_argument("--input-dir", type=Path, help="Directory containing Exploration PKLs")
    p.add_argument("--pattern", default="E*.pkl", help="Used with --input-dir (default: E*.pkl)")
    p.add_argument("--limit", type=int, default=0, help="Limit the number of input PKLs (0 = all)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=10.0, help="MP4 playback FPS (default: 10)")
    p.add_argument("--size", type=int, default=224, help="Square size of each view (default: 224)")
    p.add_argument("--raw-color-order", choices=("rgb", "bgr"), default="rgb", help="Raw ndarray color order (default: rgb for preprocessed PKLs)")
    args = p.parse_args()
    if args.fps <= 0 or args.size <= 0 or args.limit < 0:
        p.error("fps and size must be positive; limit must be nonnegative")
    paths = sorted(args.pkl if args.pkl is not None else args.input_dir.glob(args.pattern))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        p.error("No PKL files found")
    for path in paths:
        render_file(path, args.output_dir, args.fps, args.size, args.raw_color_order)


if __name__ == "__main__":
    main()
