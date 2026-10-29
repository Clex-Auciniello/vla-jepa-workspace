"""Training DDP e checkpoint A/B, sulla falsariga del trainer offline E3/E4."""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from vla_jepa_offline.trainer import _rng_state, _restore_rng
from .dataset import ActionDataset, collate
from .policy import FlowMatchingLoss
from .processors import esporta


def ultimo_checkpoint(root):
    disponibili = []
    for nome in ("checkpoint_A", "checkpoint_B"):
        folder = Path(root) / nome
        meta = folder / "metadata.json"
        if meta.exists():
            disponibili.append((json.loads(meta.read_text())["sequence"], folder))
    return max(disponibili)[1] if disponibili else None


class Trainer:
    def __init__(self, cfg, policy, pre, post, train, val, stats, init_hash, rank, local_rank, world_size):
        self.cfg, self.rank, self.world_size = cfg, rank, world_size
        self.device = torch.device("cuda", local_rank)
        self.policy, self.pre, self.post = policy, pre, post
        self.stats, self.init_hash = stats, init_hash
        self.root = Path(cfg["output_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch_start = self.step = self.sequence = 0

        bsz = int(cfg["batch_size"])
        workers = int(cfg["num_workers"])
        train_set, val_set = ActionDataset(train), ActionDataset(val)
        self.sampler = DistributedSampler(train_set, world_size, rank, shuffle=True,
                                          seed=int(cfg["seed"]), drop_last=True)
        self.loader = DataLoader(train_set, batch_size=bsz, sampler=self.sampler,
                                 drop_last=True, num_workers=workers, pin_memory=True,
                                 persistent_workers=workers > 0, collate_fn=collate)
        # Evita di duplicare campioni di validazione quando non sono divisibili per 4 GPU.
        val_indices = range(rank, len(val_set), world_size)
        self.validation_loader = DataLoader(Subset(val_set, list(val_indices)), batch_size=bsz,
                                            num_workers=0, collate_fn=collate)
        self.accum = int(cfg["grad_accum"])
        self.steps_per_epoch = len(self.loader) // self.accum
        if self.steps_per_epoch == 0:
            raise ValueError("Il dataset non consente neppure un optimizer step")

        self.params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=float(cfg["learning_rate"]),
            betas=(0.9, 0.95), eps=1e-8, weight_decay=float(cfg["weight_decay"]))
        totale = self.steps_per_epoch * int(cfg["epochs"])
        warm = int(cfg["warmup_steps"])
        low = float(cfg["min_learning_rate"]) / float(cfg["learning_rate"])
        def lr_lambda(s):
            if s < warm:
                return (s + 1) / max(warm, 1)
            p = min(1., (s-warm) / max(1, totale-warm))
            return low + (1-low) * (1+math.cos(math.pi*p)) / 2
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.ddp = DDP(FlowMatchingLoss(policy), device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False, find_unused_parameters=False)
        self.optimizer.zero_grad(set_to_none=True)
        path = ultimo_checkpoint(self.root)
        if path is not None:
            self.resume(path)

    def _batch(self, raw):
        # Normalizzazione ACTION MIN_MAX ed estrazione delle feature tramite LeRobot.
        return self.pre(raw)

    def _optimizer_step(self, iterator):
        somma = 0.
        for micro in range(self.accum):
            batch = self._batch(next(iterator))
            ctx = nullcontext() if micro == self.accum-1 else self.ddp.no_sync()
            with ctx:
                loss = self.ddp(batch)
                (loss / self.accum).backward()
            somma += float(loss.detach()) / self.accum
        torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg["grad_clip_norm"]), error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return somma

    def epoca(self, epoch, solo_un_passo=False):
        self.sampler.set_epoch(epoch)
        self.ddp.train()  # FlowMatchingLoss rimette Qwen in eval().
        it = iter(self.loader)
        losses = []
        for _ in range(1 if solo_un_passo else self.steps_per_epoch):
            losses.append(self._optimizer_step(it))
            if self.rank == 0 and self.step % int(self.cfg["log_every"]) == 0:
                print(f"step={self.step} loss={losses[-1]:.5f} lr={self.optimizer.param_groups[0]['lr']:.3g}", flush=True)
        result = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=self.device)
        dist.all_reduce(result)
        return float(result[0] / result[1])

    @torch.no_grad()
    def validazione(self):
        self.ddp.eval()
        somma, n = 0., 0
        for raw in self.validation_loader:
            loss = self.ddp.module(self._batch(raw))
            b = len(raw["task"])
            somma += float(loss) * b
            n += b
        valori = torch.tensor([somma, n], dtype=torch.float64, device=self.device)
        dist.all_reduce(valori)
        return float(valori[0] / valori[1])

    def salva(self, next_epoch):
        # Come E3/E4, salviamo gli RNG di TUTTI i rank.
        rng = [None] * self.world_size
        dist.all_gather_object(rng, _rng_state())
        dist.barrier()
        if self.rank == 0:
            sequence = self.sequence + 1
            destinazione = self.root / ("checkpoint_A" if sequence % 2 else "checkpoint_B")
            tmp = Path(tempfile.mkdtemp(dir=self.root, prefix=".incomplete_"))
            try:
                esporta(self.ddp.module.policy, self.pre, self.post, tmp / "pretrained_model")
                torch.save({"optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(), "rng": rng},
                           tmp / "trainer_state.pt")
                meta = {"sequence": sequence, "next_epoch": next_epoch, "step": self.step,
                        "fingerprint": self.stats["fingerprint"], "head_hash": self.init_hash,
                        "config": self.cfg, "world_size": self.world_size}
                (tmp / "metadata.json").write_text(json.dumps(meta, indent=2))
                if destinazione.exists():
                    shutil.rmtree(destinazione)
                os.replace(tmp, destinazione)
                self.sequence = sequence
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)
        dist.barrier()

    def resume(self, checkpoint):
        meta = json.loads((checkpoint / "metadata.json").read_text())
        if (meta["fingerprint"] != self.stats["fingerprint"] or meta["head_hash"] != self.init_hash
            or meta["config"] != self.cfg or meta["world_size"] != self.world_size):
            raise ValueError("Resume incompatibile con dati, configurazione o inizializzazione")
        state = load_file(checkpoint / "pretrained_model/model.safetensors", device="cpu")
        self.ddp.module.policy.load_state_dict(state, strict=True)
        del state
        training = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(training["optimizer"])
        self.scheduler.load_state_dict(training["scheduler"])
        _restore_rng(training["rng"][self.rank])
        self.epoch_start, self.step, self.sequence = meta["next_epoch"], meta["step"], meta["sequence"]
        if self.rank == 0:
            print(f"Resume: epoca {self.epoch_start}, optimizer step {self.step}", flush=True)
