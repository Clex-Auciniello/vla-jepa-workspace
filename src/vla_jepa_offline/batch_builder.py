"""Exact OnlineBatchBuilder RGB conversion reused for E3 and E4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from vla_jepa_online.training_server.online_batch_builder import (
    CropMargins, _pkl_bgr_to_tensor, _window_starts,
)
from .dataset import OfflineEpisode


@dataclass(frozen=True)
class OfflineBatchConfig:
    front_feature: str = "observation.images.exterior_1_left"
    gripper_feature: str = "observation.images.exterior_2_left"
    window_size: int = 8
    stride: int = 5
    front_crop: CropMargins | None = None
    gripper_crop: CropMargins | None = None
    align_last: bool = True


def parse_crop(margins) -> CropMargins | None:
    if margins is None:
        return None
    if not isinstance(margins, (list, tuple)) or len(margins) != 4:
        raise ValueError("Crop must be null or [left,top,right,bottom]")
    ints = [int(v) for v in margins]
    if any(a != b for a, b in zip(ints, margins)) or min(ints) < 0:
        raise ValueError(f"Invalid crop: {margins}")
    return CropMargins(*ints)


def iter_batches(episode: OfflineEpisode, cfg: OfflineBatchConfig) -> Iterator[dict]:
    starts = _window_starts(episode.count, window_size=cfg.window_size,
                            stride=cfg.stride, repeat_last=cfg.align_last)
    for start in starts:
        raw = [episode.rgb(idx) for idx in range(start, start + cfg.window_size)]
        front = torch.stack([
            _pkl_bgr_to_tensor(pair[0], cfg.front_crop, (224, 224))
            for pair in raw
        ]).unsqueeze(0)
        wrist = torch.stack([
            _pkl_bgr_to_tensor(pair[1], cfg.gripper_crop, (224, 224))
            for pair in raw
        ]).unsqueeze(0)
        yield {
            cfg.front_feature: front,
            cfg.gripper_feature: wrist,
            "task": [episode.spec.task],
        }
