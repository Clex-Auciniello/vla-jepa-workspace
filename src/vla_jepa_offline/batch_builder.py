"""Converte le finestre RGB già pronte nel batch usato da VLA-JEPA."""
from __future__ import annotations

import torch

from .dataset import OfflineWindowDataset

FRONT_FEATURE = "observation.images.exterior_1_left"
GRIPPER_FEATURE = "observation.images.exterior_2_left"


def _rgb_to_tensor(images: torch.Tensor) -> torch.Tensor:
    # [B, T, H, W, C] uint8 RGB -> [B, T, C, H, W] float32 [0, 1].
    if images.dtype != torch.uint8 or tuple(images.shape[-3:]) != (224, 224, 3):
        raise ValueError(f"Attese immagini RGB uint8 224x224: {images.shape}, {images.dtype}")
    return images.permute(0, 1, 4, 2, 3).contiguous().float().div_(255.0)


def make_batch(dataset: OfflineWindowDataset, indices: list[int], device: torch.device) -> dict:
    samples = [dataset[i] for i in indices]
    front = _rgb_to_tensor(torch.stack([sample["front"] for sample in samples])).to(device)
    if dataset.camera_count == 2:
        wrist = _rgb_to_tensor(torch.stack([sample["gripper"] for sample in samples])).to(device)
    else:
        # E3: la seconda vista è la stessa frontale. Nessuna copia nel PKL.
        wrist = front
    return {
        FRONT_FEATURE: front,
        GRIPPER_FEATURE: wrist,
        "task": [sample["task"] for sample in samples],
    }
