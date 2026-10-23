"""Thin adapter around the already-tested OnlineVideoTrainer (no second WM loss)."""
from __future__ import annotations

import json
from pathlib import Path

from vla_jepa_online.training_server.online_trainer import OnlineTrainerConfig, OnlineVideoTrainer


def latest_checkpoint(root: str | Path) -> Path | None:
    root = Path(root)
    found: list[tuple[int, Path]] = []
    for name in ("checkpoint_A", "checkpoint_B"):
        folder = root / name
        if not folder.exists():
            continue
        meta_path = folder / "checkpoint_meta.json"
        if not meta_path.is_file():
            raise RuntimeError(f"Checkpoint folder without metadata: {folder}")
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("format_version") != 3:
            raise RuntimeError(f"Checkpoint format mismatch: {folder}")
        for filename in ("config.json", "model.safetensors", "online_trainer_state.pt"):
            if not (folder / filename).is_file():
                raise RuntimeError(f"Incomplete checkpoint: {folder / filename}")
        found.append((int(meta["checkpoint_sequence"]), folder))
    return max(found, key=lambda x: x[0])[1] if found else None


def create_trainer(*, pretrained_path: str, checkpoint_root: str, seed: int,
                   qwen_lr: float, predictor_lr: float) -> OnlineVideoTrainer:
    checkpoint = latest_checkpoint(checkpoint_root)
    cfg = OnlineTrainerConfig(
        pretrained_path=str(Path(pretrained_path).resolve()),
        resume_from=checkpoint,
        checkpoint_root=str(Path(checkpoint_root).resolve()),
        seed=seed,
        qwen_learning_rate=qwen_lr,
        predictor_learning_rate=predictor_lr,
    )
    trainer = OnlineVideoTrainer(cfg)
    model = trainer.policy.model
    for label in ("video_encoder", "action_model"):
        module = getattr(model, label)
        if any(param.requires_grad for param in module.parameters()):
            raise RuntimeError(f"{label} is not frozen: refusing offline pretraining")
    for label in ("qwen", "video_predictor"):
        module = getattr(model, label)
        if not any(param.requires_grad for param in module.parameters()):
            raise RuntimeError(f"{label} has no trainable parameters")
    return trainer
