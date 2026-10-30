"""Entrypoint: torchrun ... -m action_head.train --config ... --experiment E1."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml

from vla_jepa_offline.train import init_distributed
from .dataset import ActionDataset, collate, prepara_dati
from .policy import inizializza_policy, verifica_heads
from .processors import crea_processors
from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", choices=("E1", "E2", "E3", "E4"))
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--check-heads", action="store_true",
                        help="Controlla su CPU che le teste iniziali E1-E4 coincidano")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Questi due comandi non richiedono GPU.
    if args.stats_only:
        train, val, stats = prepara_dati(cfg["dataset_root"], cfg["split_manifest"], cfg["seed"],
                                         cfg["val_tfrecord"], cfg["val_scripted"])
        ds = ActionDataset(train)
        conteggi = ds.conteggi_task()
        print(json.dumps({"train_episodes": len(train), "val_episodes": len(val),
                          "frame_originali_per_task": conteggi,
                          "frame_bilanciati_per_epoca": 12 * max(conteggi.values()),
                          "optimizer_step_per_epoca": (max(conteggi.values()) + cfg["grad_accum"] - 1)
                                                        // cfg["grad_accum"],
                          "action_statistics": stats}, indent=2))
        return
    if args.check_heads:
        verifica_heads(cfg["pretrained_paths"])
        return
    if not args.experiment:
        parser.error("Il training richiede --experiment E1/E2/E3/E4")

    rank, local_rank, world_size = init_distributed()
    try:
        if world_size != 4:
            raise ValueError("Il training richiede 4 GPU tramite torchrun")
        # Scelta sperimentale: seed identico per tutti i rank.
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed(cfg["seed"])

        dati = (prepara_dati(cfg["dataset_root"], cfg["split_manifest"], cfg["seed"],
                             cfg["val_tfrecord"], cfg["val_scripted"])
                if rank == 0 else None)
        payload = [dati]
        dist.broadcast_object_list(payload, src=0)
        train, val, stats = payload[0]

        pretrained_path = cfg["pretrained_paths"][args.experiment]
        policy, head_hash = inizializza_policy(cfg, pretrained_path, local_rank)
        policy.config.repeated_diffusion_steps = int(cfg["repeated_diffusion_steps"])
        pre, post = crea_processors(policy.config, stats)

        # Verifica semplice: MIN_MAX e MIN_MAX inversa devono tornare ai valori iniziali.
        campione = collate([ActionDataset(train)[0]])
        recuperato = post(pre(campione)["action"][:, 0, :]).cpu()
        if not torch.allclose(recuperato, campione["action"][:, 0, :], atol=2e-4, rtol=2e-4):
            raise ValueError("MIN_MAX: round-trip pre/postprocessor non corretto")

        trainer = Trainer(cfg, args.experiment, policy, pre, post, train, val, stats,
                          head_hash, rank, local_rank, world_size, smoke=args.smoke)
        if rank == 0:
            print(f"Esperimento {args.experiment} | action head iniziale: {head_hash}", flush=True)
        if args.smoke:
            if trainer.step:
                raise ValueError("La cartella smoke contiene già un checkpoint")
            train_loss = trainer.epoca(0, solo_un_passo=True)
            val_loss = trainer.validazione()
            trainer.salva(next_epoch=0)
            if rank == 0:
                print(f"Smoke OK: train_loss={train_loss:.6f}, "
                      f"val_macro_task_loss={val_loss['macro_task_loss']:.6f}")
            return

        for epoch in range(trainer.epoch_start, int(cfg["epochs"])):
            train_loss = trainer.epoca(epoch)
            val_loss = trainer.validazione()
            trainer.salva(next_epoch=epoch + 1)
            if rank == 0:
                report = {"experiment": args.experiment, "epoch": epoch + 1,
                          "step": trainer.step, "train_loss": train_loss,
                          "val_macro_task_loss": val_loss["macro_task_loss"],
                          "val_frame_weighted_loss": val_loss["frame_weighted_loss"],
                          "val_by_task": val_loss["by_task"]}
                with (trainer.root / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(report) + "\n")
                print(report, flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
