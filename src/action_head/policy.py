"""Caricamento VLA-JEPA E1–E4: Qwen congelato e DiT addestrabile."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


def hash_action_head(checkpoint):
    """Impronta della DiT attiva, senza caricare Qwen in RAM o su GPU."""
    path = Path(checkpoint) / "model.safetensors"
    digest = hashlib.sha256()
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = sorted(k for k in f.keys() if k.startswith("model.action_model.")
                      and not k.startswith("model.action_model.state_encoder."))
        if not keys:
            raise ValueError(f"Action head non trovata in {path}")
        for key in keys:
            value = f.get_tensor(key).contiguous()
            digest.update(key.encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verifica_heads(paths):
    """Da eseguire prima dei quattro esperimenti: i pesi iniziali devono coincidere."""
    hashes = {name: hash_action_head(path) for name, path in paths.items()}
    for name, digest in hashes.items():
        print(f"{name}: {digest}")
    if len(set(hashes.values())) != 1:
        raise ValueError("Le action head iniziali E1–E4 sono differenti: serve esaminare i checkpoint")
    print("Action head iniziali identiche.")


def inizializza_policy(cfg, pretrained_path, local_rank):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import NormalizationMode
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
    from .dataset import RENAMES

    c = PreTrainedConfig.from_pretrained(pretrained_path)
    c.device = f"cuda:{local_rank}"
    c.freeze_qwen = True
    c.enable_world_model = False
    c.reinit_modules = None  # I checkpoint E1–E4 devono già contenere la DiT 7D.
    c.use_relative_actions = False
    c.pre_snap_gripper_action = False
    c.binarize_gripper_action = False
    c.clip_normalized_actions = True
    c.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,  # Nessuno stato fornito alla policy.
        "ACTION": NormalizationMode.MIN_MAX,
    }
    if c.action_dim != 7 or c.chunk_size != 7 or set(c.input_features) != set(RENAMES.values()):
        raise ValueError("Checkpoint atteso: due camere 224x224, action 7D, senza stato input")

    policy = VLAJEPAPolicy.from_pretrained(pretrained_path, config=c, strict=False)
    policy.to(c.device)
    head = policy.model.action_model
    policy.requires_grad_(False)
    head.requires_grad_(True)
    if head.state_encoder is not None:
        head.state_encoder.requires_grad_(False)
    policy.model.qwen.eval()
    return policy, hash_action_head(pretrained_path)


class FlowMatchingLoss(nn.Module):
    """Usa la loss originale LeRobot con reduction='none', una media per campione."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def train(self, mode=True):
        super().train(mode)
        self.policy.model.qwen.eval()
        return self

    def forward(self, batch):
        losses, _ = self.policy(batch, reduction="none")
        return losses  # [B]; ogni osservazione pesa una volta, anche alla fine.
