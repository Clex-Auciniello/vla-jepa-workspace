
#!/usr/bin/env python3
"""Preprocessing offline delle traiettorie Plays per VLA-JEPA E2.

Converte le immagini BGR originali in RGB uint8 224x224,
utilizzando la stessa trasformazione di Exploration E1.

Un solo episode_prompt per traiettoria.
Finestre di 8 frame, stride 5, ultima finestra allineata.
Nessun padding temporale e nessuna modifica agli originali.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import tempfile
from pathlib import Path

from vla_jepa_offline.dataset import _open_episode
from vla_jepa_online.training_server.online_batch_builder import (
    OnlineBatchBuilderConfig,
    _pkl_bgr_to_tensor,
)

DEFAULT_INPUT = Path(
    "/mnt/beegfs/a.cuciniello9/datasets/robotic/plays_trajectories"
)
DEFAULT_OUTPUT = Path(
    "/mnt/beegfs/a.cuciniello9/datasets/robotic/plays_preprocessed"
)

ALIGN_LAST_WINDOW = True


def preprocess_image(image, crop, image_size):
    """Identica conversione BGR -> RGB uint8 usata per E1."""
    tensor = _pkl_bgr_to_tensor(image, crop, image_size)
    return (
        tensor.mul(255)
        .round_()
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def aligned_window_starts(frame_count, window_size, stride):
    """Sposta l'ultima finestra sugli ultimi frame disponibili.

    Mantiene il numero di finestre e non duplica frame.
    """
    starts = list(range(
        0,
        frame_count - window_size + 1,
        stride,
    ))

    if not starts:
        raise ValueError(
            f"Episodio troppo corto: {frame_count} frame"
        )

    if starts[-1] + window_size < frame_count:
        starts[-1] = frame_count - window_size

    return starts


def preprocess_episode(path: Path, cfg: OnlineBatchBuilderConfig) -> dict:
    """Converte un episodio Plays nel formato RGB offline."""

    data = _open_episode(str(path))

    if not isinstance(data, dict):
        raise TypeError(f"{path}: atteso dizionario")

    if data.get("source") != "plays":
        raise ValueError(f"{path}: source non è 'plays'")

    if data.get("episode_id") != path.stem:
        raise ValueError(f"{path}: episode_id non coerente")

    prompt = data.get("episode_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path}: episode_prompt mancante")

    original = data["traj"]
    frame_count = len(original)

    if frame_count < cfg.window_size:
        raise ValueError(
            f"{path}: {frame_count} frame, "
            f"minimo richiesto {cfg.window_size}"
        )

    # Conserva la classe Trajectory utilizzata nel dataset.
    config_str = getattr(original, "config_str", None)
    if config_str is None:
        config_str = data.get("config_str")

    if config_str is None:
        raise ValueError(f"{path}: config_str mancante")

    converted = original.__class__(config_str=config_str)

    for index in range(frame_count):
        step = original.get(index, decompress=False)
        obs = step["obs"]

        front = preprocess_image(
            obs["front_camera_image"],
            cfg.front_crop,
            cfg.image_size,
        )

        gripper = preprocess_image(
            obs["gripper_camera_image"],
            cfg.gripper_crop,
            cfg.image_size,
        )

        # Stesso schema minimale utilizzato da E1.
        converted._data.append((
            {
                "camera_front_image": front,
                "camera_gripper_image": gripper,
            },
            None,
            index == frame_count - 1,
            None,
            None,
        ))
        converted._raw_state.append(None)

    starts = aligned_window_starts(
        frame_count,
        cfg.window_size,
        cfg.stride,
    )

    crop = cfg.front_crop

    # Conserviamo i metadati originali di Plays.
    payload = dict(data)
    payload["traj"] = converted
    payload["source_file"] = str(path.resolve())

    # Il prompt rimane unico per episodio.
    payload["episode_prompt"] = prompt

    # Indici precomputati: il loader E2 potrà leggerli
    # senza dover ricostruire l'allineamento finale.
    payload["window_starts"] = starts

    payload.pop("window_prompts", None)

    payload["preprocessing"] = {
        "front_crop_margins_top_bottom_left_right": (
            [crop.top, crop.bottom, crop.left, crop.right]
            if crop else None
        ),
        "gripper_crop": None,
        "image_size": list(cfg.image_size),
        "color_order": "RGB",
        "resize_interpolation": "cv2.INTER_LINEAR",
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "align_last_window": ALIGN_LAST_WINDOW,
    }

    return payload


def save_episode(path: Path, payload: dict, overwrite: bool) -> None:
    """Salvataggio atomico, come nello script di E1."""

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File già presente: {path} (usa --overwrite)"
        )

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            suffix=".tmp",
            prefix=f".{path.stem}_",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(
                payload,
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        os.replace(temporary, path)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Numero episodi da elaborare (0=tutti)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit non può essere negativo")

    if args.input_root.resolve() == args.output_root.resolve():
        parser.error("Input e output devono essere differenti")

    paths = sorted(
        (
            p for p in args.input_root.glob("P*.pkl")
            if re.fullmatch(r"P\d{5,}", p.stem)
        ),
        key=lambda p: int(p.stem[1:]),
    )

    if not paths:
        parser.error(
            f"Nessun PKL Plays in {args.input_root}"
        )

    if args.limit:
        paths = paths[:args.limit]

    cfg = OnlineBatchBuilderConfig()

    # Verifichiamo di utilizzare la stessa ricetta di E1.
    if cfg.window_size != 8 or cfg.stride != 5:
        raise ValueError(
            "Attesi window_size=8 e stride=5"
        )

    total_windows = 0

    for index, source in enumerate(paths, start=1):
        destination = args.output_root / source.name

        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"File già presente: {destination}"
            )

        payload = preprocess_episode(source, cfg)
        save_episode(destination, payload, args.overwrite)

        starts = payload["window_starts"]
        frame_count = len(payload["traj"])
        total_windows += len(starts)

        print(
            f"[{index}/{len(paths)}] {source.name}: "
            f"{frame_count} frame, "
            f"{len(starts)} finestre, "
            f"ultima={starts[-1]}-{starts[-1] + 7}",
            flush=True,
        )

    print(
        f"\nCompletato: {len(paths)} episodi, "
        f"{total_windows} finestre; "
        f"align_last_window={ALIGN_LAST_WINDOW}",
        flush=True,
    )


if __name__ == "__main__":
    main()
