#!/usr/bin/env python3
"""Prepara i PKL di esplorazione per il pretraining offline E1/E2.

Ogni episodio viene salvato una sola volta con immagini RGB 224x224 e
prompt precomputati per ciascuna finestra di 8 frame (stride 5).
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import tempfile
from pathlib import Path

from vla_jepa_online.training_server.online_batch_builder import (
    OnlineBatchBuilderConfig,
    _open_exploration_pkl,
    _pkl_bgr_to_tensor,
    _prompt_for_window,
    _window_starts,
)

DEFAULT_INPUT = Path("/mnt/beegfs/a.cuciniello9/datasets/robotic/exploration")
DEFAULT_OUTPUT = Path("/mnt/beegfs/a.cuciniello9/datasets/robotic/exploration_preprocessed")


def preprocess_image(image, crop, image_size):
    """Riutilizza la trasformazione online e conserva i pixel come RGB uint8."""
    tensor = _pkl_bgr_to_tensor(image, crop, image_size)
    return tensor.mul(255).round_().byte().permute(1, 2, 0).contiguous().numpy()


def preprocess_episode(path: Path, align_last_window: bool, cfg: OnlineBatchBuilderConfig) -> dict:
    """Converte i frame una sola volta e prepara i prompt delle finestre."""
    original = _open_exploration_pkl(path)
    frame_count = len(original)
    if not cfg.window_size <= frame_count <= cfg.max_frames:
        raise ValueError(f"{path}: numero di frame non valido ({frame_count})")

    # Manteniamo lo stesso contenitore Trajectory, senza action e stati inutilizzati dal WM.
    converted = original.__class__(config_str=original.config_str)
    labels = []

    for index in range(frame_count):
        step = original.get(index, decompress=False)
        info = step["info"]
        if not isinstance(info, dict) or info.get("index") != index + 1:
            raise ValueError(f"{path}: metadati del frame {index + 1} non validi")
        labels.append({"motion_label": info.get("motion_label")})

        obs = step["obs"]
        front = preprocess_image(obs["front_camera_image"], cfg.front_crop, cfg.image_size)
        gripper = preprocess_image(obs["gripper_camera_image"], cfg.gripper_crop, cfg.image_size)
        converted._data.append((
            {"camera_front_image": front, "camera_gripper_image": gripper},
            None, index == frame_count - 1, None, None,
        ))
        converted._raw_state.append(None)

    # Usiamo gli stessi inizi e gli stessi prompt del batch builder online.
    starts = _window_starts(frame_count, window_size=cfg.window_size,
                            stride=cfg.stride, repeat_last=align_last_window)
    windows = [
        {"start": start, "prompt": _prompt_for_window(labels[start:start + cfg.window_size])}
        for start in starts
    ]
    crop = cfg.front_crop
    return {
        "episode_id": path.stem,
        "source": "exploration",
        "source_file": str(path.resolve()),
        "traj": converted,
        "window_prompts": windows,
        "preprocessing": {
            "front_crop_margins_top_bottom_left_right": (
                [crop.top, crop.bottom, crop.left, crop.right] if crop else None
            ),
            "gripper_crop": None,
            "image_size": list(cfg.image_size),
            "color_order": "RGB",
            "resize_interpolation": "cv2.INTER_LINEAR",
            "window_size": cfg.window_size,
            "stride": cfg.stride,
            "align_last_window": align_last_window,
        },
    }


def save_episode(path: Path, payload: dict, overwrite: bool) -> None:
    """Scrive il PKL in modo atomico, senza lasciare file parziali."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"File già presente: {path} (usa --overwrite)")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp",
                                         prefix=f".{path.stem}_", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--align-last-window", action="store_true",
                        help="Allinea l'ultima finestra agli ultimi 8 frame, come nel builder online")
    parser.add_argument("--limit", type=int, default=0, help="Numero di episodi da provare (0=tutti)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit non può essere negativo")
    if args.input_root.resolve() == args.output_root.resolve():
        parser.error("Input e output devono essere directory diverse")

    paths = sorted(
        (p for p in args.input_root.glob("E*.pkl") if re.fullmatch(r"E\d{5,}", p.stem)),
        key=lambda p: int(p.stem[1:]),
    )
    if not paths:
        parser.error(f"Nessun PKL di esplorazione trovato in {args.input_root}")
    if args.limit:
        paths = paths[:args.limit]

    cfg = OnlineBatchBuilderConfig()
    total_windows = 0
    for index, source in enumerate(paths, start=1):
        destination = args.output_root / source.name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"File già presente: {destination} (usa --overwrite)")
        payload = preprocess_episode(source, args.align_last_window, cfg)
        save_episode(destination, payload, args.overwrite)
        total_windows += len(payload["window_prompts"])
        print(f"[{index}/{len(paths)}] {source.name}: "
              f"{len(payload['traj'])} frame, {len(payload['window_prompts'])} finestre", flush=True)

    print(f"Completato: {len(paths)} episodi, {total_windows} finestre; "
          f"allineamento finale={args.align_last_window}", flush=True)


if __name__ == "__main__":
    main()
