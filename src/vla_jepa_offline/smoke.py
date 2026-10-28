"""Smoke test reale del trainer DDP offline su poche finestre RGB.

Usa gli stessi dataset, batch builder, OfflineTrainer e checkpoint del training.
Le due modalità vengono eseguite in processi torchrun distinti:
- train: un solo optimizer step e salvataggio immediato;
- verify: ricaricamento completo del checkpoint e dello stato AdamW.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from .dataset import EpisodeSpec, OfflineWindowDataset, TRAIN_TASKS, _open_episode
from .train import dataset_identity, init_distributed
from .trainer import OfflineTrainer, latest_checkpoint


def small_catalog(dataset_root: str, *, windows: int, window_size: int, stride: int):
    """Legge solo i primi episodi necessari a ottenere il numero di finestre richiesto."""
    root = Path(dataset_root).expanduser().resolve()
    catalog: list[EpisodeSpec] = []
    count = 0

    for task_id in TRAIN_TASKS:
        for path in sorted((root / f"task_{task_id:02d}").glob("*.pkl")):
            data = _open_episode(str(path))
            n_frames = len(data["traj"])
            n_windows = len(range(0, n_frames - window_size + 1, stride))
            if n_windows == 0:
                continue
            if int(data["task_id"]) != task_id:
                raise ValueError(f"task_id non corrispondente alla cartella: {path}")
            prompt = data["language_instruction"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Istruzione mancante nel PKL: {path}")
            catalog.append(EpisodeSpec(str(path), task_id, prompt, n_frames))
            count += n_windows
            if count >= windows:
                return catalog

    raise ValueError(f"Finestre insufficienti: richieste {windows}, trovate {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--mode", choices=("train", "verify"), required=True)
    args = parser.parse_args()

    if not (1 <= args.windows <= 32):
        parser.error("--windows deve essere compreso fra 1 e 32")

    cfg = yaml.safe_load(args.config.read_text())
    # Sovrascrive solo la destinazione: nessun rischio per i checkpoint E3/E4 ufficiali.
    cfg["checkpoint_root"] = str(args.checkpoint_root.expanduser().resolve())
    rank, local_rank, world_size = init_distributed()

    try:
        if world_size != 4:
            raise RuntimeError(f"Smoke test previsto per 4 rank, ottenuti {world_size}")
        if int(cfg["batch_size"]) != 4 or int(cfg["gradient_accumulation_steps"]) != 2:
            raise RuntimeError("Configurazione attesa: batch_size=4, gradient_accumulation_steps=2")

        if rank == 0:
            source = Path(cfg["pretrained_path"]).expanduser().resolve()
            if not source.is_dir() or not (source / "config.json").is_file():
                raise FileNotFoundError(f"Checkpoint base non trovato: {source}")
            catalog = small_catalog(cfg["dataset_root"], windows=args.windows,
                                    window_size=int(cfg["window_size"]), stride=int(cfg["stride"]))
            print(f"[SMOKE] Esperimento: {args.config.name}, modalità: {args.mode}", flush=True)
            print(f"[SMOKE] Checkpoint base: {source}", flush=True)
            print(f"[SMOKE] PKL selezionati: {len(catalog)}; finestre richieste: {args.windows}", flush=True)
            print(f"[SMOKE] Cartella checkpoint test: {cfg['checkpoint_root']}", flush=True)
        else:
            catalog = None

        payload = [catalog]
        dist.broadcast_object_list(payload, src=0)
        catalog = payload[0]

        dataset = OfflineWindowDataset(catalog, cfg["window_size"], cfg["stride"], cfg["camera_count"])
        identity = dataset_identity(catalog, cfg)
        order = dataset.epoch_indices(int(cfg["seed"]), 0)[:args.windows]

        if rank == 0:
            first = dataset[order[0]]
            cameras = "front+wrist" if "gripper" in first else "front duplicata"
            print(f"[SMOKE] Prima finestra {tuple(first['front'].shape)}, viste: {cameras}", flush=True)

        torch.manual_seed(int(cfg["seed"]) + rank)
        torch.cuda.manual_seed(int(cfg["seed"]) + rank)
        trainer = OfflineTrainer(cfg, rank=rank, local_rank=local_rank,
                                 world_size=world_size, dataset_identity=identity)

        if args.mode == "train":
            if trainer.next_epoch != 0 or trainer.global_step != 0:
                raise RuntimeError("Cartella test già usata: scegliere un checkpoint_root nuovo")

            # Testa il percorso reale (Qwen -> predictor -> backward DDP -> AdamW).
            local_sum = trainer.train_group(dataset, order)
            totals = torch.tensor([local_sum], dtype=torch.float64, device=trainer.device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            mean_loss = float(totals.item() / len(order))
            if not math.isfinite(mean_loss) or trainer.global_step != 1:
                raise RuntimeError(f"Loss o optimizer step non validi: {mean_loss}, {trainer.global_step}")

            if rank == 0:
                print(f"[SMOKE] Optimizer step completato: {trainer.global_step}", flush=True)
                print(f"[SMOKE] World Model loss media: {mean_loss:.8f}", flush=True)

            checkpoint = trainer.save_checkpoint(next_epoch=1)
            if rank == 0:
                meta = json.loads((checkpoint / "checkpoint_meta.json").read_text())
                if meta["global_step"] != 1 or meta["next_epoch"] != 1:
                    raise RuntimeError(f"Metadata checkpoint errati: {meta}")
                if not (checkpoint / "offline_trainer_state.pt").is_file():
                    raise FileNotFoundError("Checkpoint privo dello stato optimizer")
                print(f"[SMOKE] Checkpoint creato: {checkpoint}", flush=True)
                print(f"[SMOKE] Metadata: {meta}", flush=True)
                print("[SMOKE] Memoria allocata GPU0 GiB:",
                      round(torch.cuda.max_memory_allocated() / 1024**3, 2), flush=True)
        else:
            # In questa invocazione OfflineTrainer ha già ricaricato pesi + optimizer + RNG.
            checkpoint = latest_checkpoint(cfg["checkpoint_root"])
            if checkpoint is None or trainer.next_epoch != 1 or trainer.global_step != 1:
                raise RuntimeError("Il resume del checkpoint non ha ripristinato epoch=1 e step=1")
            if rank == 0:
                print(f"[SMOKE] Resume riuscito da: {checkpoint}", flush=True)
                print(f"[SMOKE] next_epoch={trainer.next_epoch}; global_step={trainer.global_step}", flush=True)
                print("[SMOKE] Verifica conclusa: OK", flush=True)

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
