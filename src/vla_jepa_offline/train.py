"""Entrypoint comune per il continued pretraining offline E3 ed E4.

Esecuzione su quattro A100: torchrun --standalone --nproc_per_node=4
    -m vla_jepa_offline.train --config configs/offline/real_robot.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from .dataset import OfflineWindowDataset, build_catalog
from .trainer import OfflineTrainer


def init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def dataset_identity(catalog, cfg: dict) -> str:
    """Impronta semplice di file e impostazioni per un resume coerente."""
    digest = hashlib.sha256()
    for episode in catalog:
        stat = Path(episode.path).stat()
        digest.update(f"{episode.path}|{stat.st_size}|{stat.st_mtime_ns}|"
                      f"{episode.task_id}|{episode.frames}|{episode.task}\n".encode())
    settings = {key: cfg[key] for key in (
        "pretrained_path", "camera_count", "window_size", "stride", "seed",
        "batch_size", "gradient_accumulation_steps", "qwen_learning_rate",
        "predictor_learning_rate", "grad_clip_norm")}
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Controlla i PKL senza caricare il modello")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rank, local_rank, world_size = init_distributed()

    try:
        if int(cfg["batch_size"]) != 4 or int(cfg["gradient_accumulation_steps"]) != 2:
            raise ValueError("Ricetta fissata: batch_size=4 per GPU e accumulation=2")
        if int(cfg["camera_count"]) not in (1, 2):
            raise ValueError("camera_count deve essere 1 per E3 oppure 2 per E4")
        if not args.preflight_only and world_size != 4:
            raise ValueError("Questo esperimento richiede torchrun con quattro processi GPU")
        if int(cfg["window_size"]) != 8 or int(cfg["stride"]) != 5:
            raise ValueError("Ricetta fissata: finestre di 8 frame con stride 5")

        # Il rank 0 apre i PKL una sola volta per ottenere lunghezze e prompt.
        catalog = build_catalog(cfg["dataset_root"]) if rank == 0 else None
        if world_size > 1:
            result = [catalog]
            dist.broadcast_object_list(result, src=0)
            catalog = result[0]
        dataset = OfflineWindowDataset(
            catalog, cfg["window_size"], cfg["stride"], cfg["camera_count"])
        identity = dataset_identity(catalog, cfg)

        if rank == 0:
            print(f"Dataset: {len(catalog)} episodi, {len(dataset)} finestre ",
                  f"({cfg['camera_count']} camera/e)", flush=True)
            first = dataset[0]
            print(f"Prima finestra: RGB uint8 {tuple(first['front'].shape)}, "
                  f"prompt={first['task']!r}", flush=True)
        if args.preflight_only:
            return

        # Seed iniziale distinto per GPU; l'ordine delle finestre è comune.
        random.seed(cfg["seed"] + rank)
        np.random.seed(cfg["seed"] + rank)
        torch.manual_seed(cfg["seed"] + rank)
        torch.cuda.manual_seed(cfg["seed"] + rank)
        trainer = OfflineTrainer(cfg, rank=rank, local_rank=local_rank,
                                 world_size=world_size, dataset_identity=identity)

        total_batch = cfg["batch_size"] * world_size * cfg["gradient_accumulation_steps"] # 4 x 4 x 2 = 32
        if rank == 0:
            print(f"DDP: {world_size} GPU, batch/GPU={cfg['batch_size']}, "
                  f"accum={cfg['gradient_accumulation_steps']}, globale={total_batch}; "
                  f"resume da epoca {trainer.next_epoch}", flush=True)

        for epoch in range(trainer.next_epoch, int(cfg["epochs"])):
            order = dataset.epoch_indices(int(cfg["seed"]), epoch)
            loss_sum = 0.0
            for start in range(0, len(order), total_batch):
                # Nell'ultimo gruppo possono esserci meno di 32 finestre.
                group = order[start:start + total_batch]
                loss_sum += trainer.train_group(dataset, group)
                if rank == 0 and trainer.global_step % 20 == 0:
                    print(f"Epoca {epoch + 1}/{cfg['epochs']} | "
                          f"optimizer step {trainer.global_step}", flush=True)

            values = torch.tensor([loss_sum, len(order)], dtype=torch.float64,
                                  device=f"cuda:{local_rank}")
            # Solo la loss è una somma locale; len(order) è già globale.
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            epoch_loss = float(values[0] / len(order))
            saved = trainer.save_checkpoint(next_epoch=epoch + 1)
            if rank == 0:
                report = {"epoch": epoch + 1, "mean_wm_loss": epoch_loss,
                          "global_step": trainer.global_step, "windows": len(order),
                          "checkpoint": str(saved)}
                with (trainer.root / "offline_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
                print(json.dumps(report), flush=True)

        if rank == 0:
            print("TRAINING COMPLETATO", flush=True)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
