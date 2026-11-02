#!/usr/bin/env python3
"""Verifica un PKL E1 preprocessato contro l'originale e il batch builder online."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from vla_jepa_online.training_server.online_batch_builder import (
    OnlineBatchBuilder,
    OnlineBatchBuilderConfig,
    _open_exploration_pkl,
    _pkl_bgr_to_tensor,
    _prompt_for_window,
    _window_starts,
)


def expected_rgb(image, crop, size):
    """Ottiene gli stessi byte RGB della pipeline online."""
    tensor = _pkl_bgr_to_tensor(image, crop, size)
    return tensor.mul(255).round().byte().permute(1, 2, 0).contiguous().numpy()


def make_preview(original, processed, starts, destination, cfg):
    """Salva un confronto RGB di tre finestre rappresentative."""
    # Le colonne mostrano: frontale originale con crop, frontale finale, wrist finale.
    picks = sorted({0, len(starts) // 2, len(starts) - 1})
    tile, header, row_height = 224, 32, 266
    sheet = Image.new("RGB", (3 * tile, header + len(picks) * row_height), "#f0f0f0")
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(("Front originale + crop", "Front RGB 224", "Wrist RGB 224")):
        draw.text((col * tile + 8, 10), title, fill="black")

    for row, window_idx in enumerate(picks):
        start = starts[window_idx]
        frame_idx = start + cfg.window_size // 2
        old = original.get(frame_idx, decompress=False)["obs"]["front_camera_image"]
        raw_rgb = cv2.cvtColor(np.asarray(old), cv2.COLOR_BGR2RGB)
        raw_img = Image.fromarray(raw_rgb)
        crop = cfg.front_crop
        box = (crop.left, crop.top, raw_img.width - crop.right, raw_img.height - crop.bottom)
        ImageDraw.Draw(raw_img).rectangle(box, outline="yellow", width=4)
        raw_img.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        background = Image.new("RGB", (tile, tile), "#777777")
        background.paste(raw_img, ((tile - raw_img.width) // 2, (tile - raw_img.height) // 2))

        step = processed.get(frame_idx, decompress=False)["obs"]
        images = (background, Image.fromarray(step["camera_front_image"]),
                  Image.fromarray(step["camera_gripper_image"]))
        y = header + row * row_height
        for col, img in enumerate(images):
            sheet.paste(img, (col * tile, y))
        draw.text((8, y + tile + 6), f"Finestra {window_idx}, frame {frame_idx + 1}", fill="black")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=Path("tests/outputs/e1_preview.png"))
    args = parser.parse_args()

    cfg = OnlineBatchBuilderConfig()
    original = _open_exploration_pkl(args.original)
    with args.processed.open("rb") as stream:
        data = pickle.load(stream)  # Solo PKL locali fidati.
    processed = data["traj"]
    meta = data["preprocessing"]
    frames = len(original)

    assert data["episode_id"] == args.original.stem == args.processed.stem
    assert data["source"] == "exploration"
    assert len(processed) == frames
    assert meta["front_crop_margins_top_bottom_left_right"] == [0, 10, 140, 90]
    assert meta["gripper_crop"] is None
    assert meta["image_size"] == [224, 224]
    assert meta["color_order"] == "RGB"
    assert meta["resize_interpolation"] == "cv2.INTER_LINEAR"
    assert meta["window_size"] == cfg.window_size and meta["stride"] == cfg.stride

    # Controllo di TUTTI i frame, senza limitarci a un'anteprima.
    labels = []
    for index in range(frames):
        raw = original.get(index, decompress=False)
        saved = processed.get(index, decompress=False)
        labels.append({"motion_label": raw["info"].get("motion_label")})
        assert set(saved["obs"]) == {"camera_front_image", "camera_gripper_image"}
        # La lettura pubblica Trajectory.get() può omettere campi con valore None.
        # Verifichiamo i cinque campi effettivamente salvati nella tupla nativa.
        obs_native, reward, done, info, action = processed._data[index]
        assert set(obs_native) == {"camera_front_image", "camera_gripper_image"}
        assert reward is None and info is None and action is None
        assert bool(done) == (index == frames - 1)
        for source_key, output_key, crop in (
            ("front_camera_image", "camera_front_image", cfg.front_crop),
            ("gripper_camera_image", "camera_gripper_image", cfg.gripper_crop),
        ):
            actual = saved["obs"][output_key]
            assert isinstance(actual, np.ndarray) and actual.dtype == np.uint8
            assert actual.shape == (224, 224, 3)
            expected = expected_rgb(raw["obs"][source_key], crop, cfg.image_size)
            if not np.array_equal(actual, expected):
                raise AssertionError(f"Pixel diversi: frame {index + 1}, camera {output_key}")

    # Gli indici e i prompt devono corrispondere alle funzioni originali online.
    starts = _window_starts(frames, window_size=cfg.window_size,
                            stride=cfg.stride, repeat_last=meta["align_last_window"])
    windows = data["window_prompts"]
    assert len(starts) == len(windows)
    for i, start in enumerate(starts):
        assert windows[i]["start"] == start, f"Start errato nella finestra {i}"
        expected_prompt = _prompt_for_window(labels[start:start + cfg.window_size])
        assert windows[i]["prompt"] == expected_prompt, f"Prompt errato nella finestra {i}"

    # Ulteriore confronto con i batch emessi dal builder online (modalità E1 standard).
    if not meta["align_last_window"]:
        online = OnlineBatchBuilder(cfg)
        for item in online.iter_exploration_pkl_windows(args.original):
            start = starts[item.window_index]
            assert item.start_index == start
            assert item.batch["task"] == [windows[item.window_index]["prompt"]]
            for feature, key in (
                (cfg.front_feature, "camera_front_image"),
                (cfg.gripper_feature, "camera_gripper_image"),
            ):
                images = np.stack([
                    processed.get(i, decompress=False)["obs"][key]
                    for i in range(start, start + cfg.window_size)
                ])
                offline_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
                assert torch.equal(offline_tensor, item.batch[feature][0]), (
                    f"Differenza rispetto al batch online: finestra {item.window_index}, {key}"
                )

    make_preview(original, processed, starts, args.preview, cfg)
    print(f"OK: {args.original.name}: {frames} frame e {len(windows)} finestre")
    print("OK: metadati, 2 camere RGB uint8 224x224, pixel identici, prompt identici")
    print(f"Allineamento ultima finestra: {meta['align_last_window']}")
    print(f"Inizi delle finestre: {starts}")
    for i in sorted({0, len(windows) // 2, len(windows) - 1}):
        print(f"Prompt finestra {i} (start={starts[i]}): {windows[i]['prompt']}")
    print(f"Anteprima RGB: {args.preview}")


if __name__ == "__main__":
    main()
