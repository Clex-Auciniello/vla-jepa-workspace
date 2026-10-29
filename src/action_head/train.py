"""Entrypoint: `torchrun --standalone --nproc_per_node=4 -m action_head.train --config ...`."""
import argparse
import json
import random

import numpy as np
import torch
import torch.distributed as dist
import yaml

from vla_jepa_offline.train import init_distributed
from .dataset import ActionDataset, collate, prepara_dati
from .policy import inizializza_policy
from .processors import crea_processors
from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stats-only", action="store_true", help="Calcola split e MIN_MAX senza caricare Qwen")
    parser.add_argument("--smoke", action="store_true", help="Un solo optimizer step su GPU")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    # Per --stats-only non servono GPU né NCCL.
    if args.stats_only:
        train, val, stats = prepara_dati(cfg["dataset_root"], cfg["split_manifest"], cfg["seed"])
        print(json.dumps({"train_episodes": len(train), "val_episodes": len(val), "statistics": stats}, indent=2))
        return

    rank, local_rank, world_size = init_distributed()  # Riutilizziamo NCCL di E3/E4.
    if world_size != 4:
        raise ValueError("La configurazione cluster richiede 4 GPU (torchrun nproc_per_node=4)")
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)
    torch.cuda.manual_seed(cfg["seed"] + rank)
    try:
        if rank == 0:
            dati = prepara_dati(cfg["dataset_root"], cfg["split_manifest"], cfg["seed"])
        else:
            dati = None
        result = [dati]
        dist.broadcast_object_list(result, src=0)
        train, val, stats = result[0]
        if rank == 0:
            print(f"Episodi training={len(train)} validazione={len(val)}; batch effettivo={4*cfg['batch_size']*cfg['grad_accum']}", flush=True)

        policy, head_hash = inizializza_policy(cfg, local_rank)
        policy.config.repeated_diffusion_steps = int(cfg["repeated_diffusion_steps"])
        pre, post = crea_processors(policy.config, stats)
        # Test minimo delle statistiche e del contratto con il controller.
        campione = collate([ActionDataset(train)[0]])
        normalizzato = pre(campione)["action"][:, 0, :]
        recuperato = post(normalizzato).cpu()
        if not torch.allclose(recuperato, campione["action"][:, 0, :], atol=2e-4, rtol=2e-4):
            raise RuntimeError("MIN_MAX: normalizzazione e denormalizzazione non inverse")
        if rank == 0:
            print("SHA256 action head iniziale:", head_hash, flush=True)

        trainer = Trainer(cfg, policy, pre, post, train, val, stats, head_hash,
                          rank, local_rank, world_size)
        if args.smoke:
            if trainer.step:
                raise ValueError("Usa un output_root vuoto per lo smoke test")
            loss = trainer.epoca(0, solo_un_passo=True)
            if rank == 0:
                print("Smoke forward/backward/optimizer:", loss, flush=True)
            trainer.salva(next_epoch=0)  # Il checkpoint smoke non è destinato al training reale.
            return

        for epoch in range(trainer.epoch_start, cfg["epochs"]):
            loss_train = trainer.epoca(epoch)
            loss_val = trainer.validazione()
            trainer.salva(next_epoch=epoch+1)
            if rank == 0:
                report = {"epoch": epoch+1, "step": trainer.step,
                          "train_loss": loss_train, "val_loss": loss_val}
                with (trainer.root / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(report) + "\n")
                print(report, flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
