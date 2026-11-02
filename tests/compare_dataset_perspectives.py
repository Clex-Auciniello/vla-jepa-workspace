#!/usr/bin/env python3
"""Compare front-camera perspective across ROBOT (TFRecord-derived), SCRIPTED and HUMAN.

Run from the VLA-JEPA workspace with the original savers modules on PYTHONPATH.
All sources are *read only*. Output is a PNG with (top) raw images + crop
rectangles and (bottom) resized RGB 224x224 images.

IMPORTANT: The robot reference from E4/tfrecord_traj_*.pkl is ALREADY 224x224;
leave --robot-crop at 0 0 0 0. Cropping this file cannot restore its lost FOV.

Crop convention for each source: TOP BOTTOM LEFT RIGHT, in ORIGINAL pixels.
Channel-order defaults reproduce the existing E3/E4 preprocessing:
robot RGB, scripted BGR, human RGB after OpenCV decode (empirically verified).
Use --human-order bgr to test the alternative color interpretation.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Works without a graphical desktop on HPC login nodes.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Needed to unpickle both savers.Trajectory variants. The PYTHONPATH on the
# cluster must make these packages importable.
import savers  # noqa: F401
import multi_task_il.datasets.savers  # noqa: F401

DEFAULT_ROOT = Path("/mnt/beegfs/a.cuciniello9/datasets")
DEFAULT_ROBOT = DEFAULT_ROOT / "robotic/vla_jepa_offline_merged"
DEFAULT_SCRIPTED = DEFAULT_ROOT / "robotic/scripted_trajectories"
DEFAULT_HUMAN = DEFAULT_ROOT / "human/human_rgb_pick_place"
SIZE = (224, 224)


def first_episode(source: Path, kind: str) -> Path:
    """Choose a deterministic first episode, not an arbitrary first PKL."""
    if source.is_file():
        if source.suffix.lower() != ".pkl":
            raise ValueError(f"{kind}: expected .pkl: {source}")
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"{kind}: directory/file not found: {source}")

    patterns = {
        "robot": ("task_01/tfrecord_traj_000.pkl", "task_01/tfrecord_traj_*.pkl", "**/tfrecord_traj_*.pkl"),
        "scripted": ("task_01/traj_000.pkl", "task_01/traj*.pkl", "**/traj*.pkl"),
        "human": ("task_01/traj000.pkl", "task_01/traj*.pkl", "**/traj*.pkl"),
    }
    for pattern in patterns[kind]:
        matches = sorted(source.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"{kind}: no matching PKL found under {source}")


def load_front_rgb(path: Path, frame: int, kind: str, channel_order: str) -> np.ndarray:
    with path.open("rb") as f:
        payload = pickle.load(f)  # Only use PKLs from trusted data sources.
    if not isinstance(payload, dict) or "traj" not in payload:
        raise ValueError(f"{path}: expected dict with 'traj'")
    raw_steps = getattr(payload["traj"], "_data", None)
    if not isinstance(raw_steps, list):
        raise ValueError(f"{path}: Trajectory._data must be list")
    if frame < 0 or frame >= len(raw_steps):
        raise IndexError(f"{path}: frame {frame} out of bounds (T={len(raw_steps)})")
    step = raw_steps[frame]
    if not isinstance(step, (tuple, list)) or len(step) != 5:
        raise ValueError(f"{path}: frame {frame} not a native 5-tuple")
    obs = step[0]
    if not isinstance(obs, dict):
        raise ValueError(f"{path}: frame {frame} observation not dict")
    keys = {
        "robot": ("camera_front_image", "front_camera_image"),
        "scripted": ("camera_front_image", "front_camera_image"),
        "human": ("camera_front_image", "front_camera_image"),
    }[kind]
    key = next((k for k in keys if k in obs), None)
    if key is None:
        raise KeyError(f"{path}: missing front camera key; found {sorted(obs)}")
    value = obs[key]
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = np.frombuffer(value, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"{path}: cannot decode {key} bytes")
        pixels = decoded
    else:
        array = np.asarray(value)
        if array.dtype != np.uint8:
            raise ValueError(f"{path}: expected uint8 image, got {array.dtype}")
        if array.ndim == 3 and array.shape[2] == 3:
            pixels = array
        elif array.ndim == 1 or (array.ndim == 2 and 1 in array.shape):
            pixels = cv2.imdecode(np.ascontiguousarray(array.reshape(-1)), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f"{path}: cv2.imdecode failed on {key}")
        else:
            raise ValueError(f"{path}: unsupported shape {array.shape}")
    rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB) if channel_order == "bgr" else pixels.copy()
    rgb = np.ascontiguousarray(rgb)
    print(f"{kind.upper():8s} | {path} | frame={frame} | image={rgb.shape} | input_order={channel_order}")
    return rgb


def crop_and_resize(rgb: np.ndarray, margins: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = margins
    h, w = rgb.shape[:2]
    if any(v < 0 for v in margins):
        raise ValueError(f"Negative crop margins: {margins}")
    if top + bottom >= h or left + right >= w:
        raise ValueError(f"Invalid crop {margins} on image {w}x{h}")
    roi = rgb[top:h - bottom, left:w - right]
    return cv2.resize(roi, SIZE, interpolation=cv2.INTER_LINEAR)


def draw_comparison(images: dict[str, np.ndarray], crops: dict[str, tuple[int, ...]],
                    frames: dict[str, int], paths: dict[str, Path], out: Path, grid: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=140)
    titles = {"robot": "ROBOT — TFRecord E4 (riferimento)",
              "scripted": "SCRIPTED — originale", "human": "UMANO — originale"}
    for col, kind in enumerate(("robot", "scripted", "human")):
        image = images[kind]
        crop = crops[kind]
        top, bottom, left, right = crop
        h, w = image.shape[:2]
        processed = crop_and_resize(image, crop)

        ax_raw = axes[0, col]
        ax_raw.imshow(image)
        ax_raw.add_patch(Rectangle((left - 0.5, top - 0.5), w - left - right,
                                   h - top - bottom, linewidth=2.5, edgecolor="yellow",
                                   facecolor="none"))
        ax_raw.set_title(f"{titles[kind]}\nOriginale {w}x{h} · frame {frames[kind]}", fontsize=12)
        ax_raw.axis("off")

        ax_out = axes[1, col]
        ax_out.imshow(processed, interpolation="nearest")
        if grid:
            for p in (0.25, 0.5, 0.75):
                v = p * 224 - 0.5
                ax_out.axvline(v, color="white", alpha=0.4, linewidth=0.8, linestyle="--")
                ax_out.axhline(v, color="white", alpha=0.4, linewidth=0.8, linestyle="--")
        ax_out.set_title(f"Risultato RGB 224x224\nCrop T,B,L,R = {crop}", fontsize=12)
        ax_out.axis("off")

    fig.suptitle("CONFRONTO PROSPETTIVE — frame iniziale, crop indipendenti, INTER_LINEAR",
                 fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nConfronto salvato: {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--robot-path", type=Path, default=DEFAULT_ROBOT,
                    help="TFRecord-derived robot PKL or root containing task_01/tfrecord_traj_000.pkl")
    ap.add_argument("--scripted-path", type=Path, default=DEFAULT_SCRIPTED,
                    help="Raw scripted PKL or root containing task_01/traj_000.pkl")
    ap.add_argument("--human-path", type=Path, default=DEFAULT_HUMAN,
                    help="Raw human PKL or root containing task_01/trajNNN.pkl")
    for kind in ("robot", "scripted", "human"):
        ap.add_argument(f"--{kind}-frame", type=int, default=0)
        ap.add_argument(f"--{kind}-crop", type=int, nargs=4,
                        default=(0, 0, 0, 0) if kind == "robot" else (0, 10, 140, 90),
                        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
        ap.add_argument(f"--{kind}-order", choices=("rgb", "bgr"),
                        default="bgr" if kind == "scripted" else "rgb")
    ap.add_argument("--output", type=Path, default=Path("tests/outputs/perspective_comparison.png"))
    ap.add_argument("--grid", action="store_true", help="Overlay grid on resized images")
    ap.add_argument("--interactive", action="store_true",
                    help="Change crop margins in a terminal loop and save numbered PNGs")
    args = ap.parse_args()

    images, crops, frames, paths = {}, {}, {}, {}
    for kind in ("robot", "scripted", "human"):
        source = getattr(args, f"{kind}_path")
        paths[kind] = first_episode(source, kind)
        frames[kind] = getattr(args, f"{kind}_frame")
        images[kind] = load_front_rgb(paths[kind], frames[kind], kind,
                                      getattr(args, f"{kind}_order"))
        crops[kind] = tuple(getattr(args, f"{kind}_crop"))

    if not args.interactive:
        draw_comparison(images, crops, frames, paths, args.output, args.grid)
        return

    print("\nModalità interattiva. Comandi: 's 0 10 140 90', 'h 0 10 120 100',")
    print("'r 0 0 0 0' (robot), 'save' (nuova figura), 'q' (esci).")
    print("L'ordine dei margini è sempre TOP BOTTOM LEFT RIGHT.")
    revision = 0
    while True:
        out = args.output.with_name(f"{args.output.stem}_{revision:03d}{args.output.suffix}")
        try:
            draw_comparison(images, crops, frames, paths, out, args.grid)
        except ValueError as exc:
            print(f"Crop non valido: {exc}")
        command = input("crop> ").strip().lower()
        if command in ("q", "quit", "exit"):
            break
        tokens = command.split()
        if command in ("save", ""):
            revision += 1
            continue
        if len(tokens) == 5 and tokens[0] in ("r", "s", "h"):
            try:
                margins = tuple(int(x) for x in tokens[1:])
            except ValueError:
                print("Inserire quattro interi, es.: s 0 10 140 90")
                continue
            kind = {"r": "robot", "s": "scripted", "h": "human"}[tokens[0]]
            try:
                crop_and_resize(images[kind], margins)
            except ValueError as exc:
                print(exc)
                continue
            crops[kind] = margins
            revision += 1
            continue
        print("Comando non valido. Esempio: h 0 10 120 100, oppure q")


if __name__ == "__main__":
    main()
