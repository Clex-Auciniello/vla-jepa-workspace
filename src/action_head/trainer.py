"""Training della DiT su quattro GPU, con un batch globale bilanciato per task."""
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

from vla_jepa_offline.dataset import TRAIN_TASKS
from vla_jepa_offline.trainer import _rng_state, _restore_rng
from .dataset import ActionDataset, collate
from .policy import FlowMatchingLoss
from .processors import esporta


def ultimo_checkpoint(root):
    disponibili = []
    for nome in ("checkpoint_A", "checkpoint_B"):
        cartella = Path(root) / nome
        if (cartella / "metadata.json").is_file():
            meta = json.loads((cartella / "metadata.json").read_text())
            disponibili.append((meta["sequence"], cartella))
    return max(disponibili, default=(0, None))[1]


class Trainer:
    def __init__(self, cfg, experiment, policy, pre, post, train, val, stats, head_hash,
                 rank, local_rank, world_size, smoke=False):
        self.cfg, self.rank, self.world_size = cfg, rank, world_size
        self.device = torch.device("cuda", local_rank)
        self.pre, self.post, self.stats, self.init_hash = pre, post, stats, head_hash
        self.experiment = experiment
        self.root = Path(cfg["output_root"]) / ("smoke" if smoke else "") / experiment
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch_start = self.step = self.sequence = 0

        self.train_set = ActionDataset(train)
        self.val_set = ActionDataset(val)
        self.batch_size = int(cfg["batch_size"])
        self.accum = int(cfg["grad_accum"])
        if world_size * self.batch_size != len(TRAIN_TASKS):
            raise ValueError("Per 12 task e 4 GPU occorre batch_size=3 per GPU")
        if self.accum < 1:
            raise ValueError("grad_accum deve essere positivo")

        # Ogni riga globale contiene esattamente 12 task; ogni step usa 2 righe.
        self.max_frame_task = max(self.train_set.conteggi_task().values())
        self.steps_per_epoch = math.ceil(self.max_frame_task / self.accum)
        self.global_batch = len(TRAIN_TASKS) * self.accum

        self.params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.params, lr=float(cfg["learning_rate"]), betas=(0.9, 0.95),
            eps=1e-8, weight_decay=float(cfg["weight_decay"]),
        )
        totale = self.steps_per_epoch * int(cfg["epochs"])
        warm = int(cfg["warmup_steps"])
        minimo = float(cfg["min_learning_rate"]) / float(cfg["learning_rate"])

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / max(warm, 1)
            progresso = min(1.0, (step - warm) / max(1, totale - warm))
            return minimo + (1 - minimo) * (1 + math.cos(math.pi * progresso)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.ddp = DDP(FlowMatchingLoss(policy), device_ids=[local_rank],
                       output_device=local_rank, broadcast_buffers=False,
                       find_unused_parameters=False)
        self.optimizer.zero_grad(set_to_none=True)
        precedente = ultimo_checkpoint(self.root)
        if precedente is not None:
            self.resume(precedente)

        if rank == 0:
            n = sum(p.numel() for p in self.params)
            print(f"{experiment}: parametri addestrabili={n:,} | "
                  f"frame originali={len(self.train_set)} | "
                  f"frame bilanciati/epoca={12 * self.max_frame_task} | "
                  f"step/epoca={self.steps_per_epoch}", flush=True)
            print(f"Frame per task prima dell'oversampling: "
                  f"{self.train_set.conteggi_task()}", flush=True)

    def _batch(self, indices, dataset):
        return self.pre(collate([dataset[i] for i in indices]))

    def _optimizer_step(self, gruppo):
        """Da 12 a 24 campioni: ciascuna GPU riceve sempre 3 per microbatch."""
        micro_globali = len(TRAIN_TASKS)
        num_micro = len(gruppo) // micro_globali
        somma_locale = 0.0
        for micro in range(num_micro):
            inizio = micro * micro_globali + self.rank * self.batch_size
            indici = gruppo[inizio:inizio + self.batch_size]
            contesto = nullcontext() if micro == num_micro - 1 else self.ddp.no_sync()
            with contesto:
                loss_per_campione = self.ddp(self._batch(indici, self.train_set))
                somma = loss_per_campione.sum()
                # DDP media sui rank; la loss finale è la media dei campioni globali.
                (somma * (self.world_size / len(gruppo))).backward()
            somma_locale += float(somma.detach())

        torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg["grad_clip_norm"]),
                                       error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return somma_locale

    def epoca(self, epoch, solo_un_passo=False):
        self.ddp.train()
        # Stesso ordinamento globale su tutti i rank, ma sottobatch diversi.
        ordine = self.train_set.epoch_indices(int(self.cfg["seed"]), epoch)
        totale_locale = 0.0
        for inizio in range(0, len(ordine), self.global_batch):
            gruppo = ordine[inizio:inizio + self.global_batch]
            totale_locale += self._optimizer_step(gruppo)
            if self.rank == 0 and self.step % int(self.cfg["log_every"]) == 0:
                print(f"epoca={epoch+1} step={self.step} "
                      f"lr={self.optimizer.param_groups[0]['lr']:.3g}", flush=True)
            if solo_un_passo:
                break
        n = min(self.global_batch, len(ordine)) if solo_un_passo else len(ordine)
        totale = torch.tensor([totale_locale], dtype=torch.float64, device=self.device)
        dist.all_reduce(totale)
        return float(totale.item() / n)

    @torch.no_grad()
    def validazione(self):
        """Nessun oversampling; media globale e macro-media dei 12 task."""
        self.ddp.eval()
        somme = torch.zeros(len(TRAIN_TASKS), dtype=torch.float64, device=self.device)
        conteggi = torch.zeros(len(TRAIN_TASKS), dtype=torch.float64, device=self.device)
        task_to_pos = {task: i for i, task in enumerate(TRAIN_TASKS)}

        # Il rumore della validazione è riproducibile e non cambia l'RNG del train.
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(int(self.cfg["seed"]))
            torch.cuda.manual_seed(int(self.cfg["seed"]))
            for base in range(self.rank * self.batch_size,
                              len(self.val_set), self.world_size * self.batch_size):
                indici = list(range(base, min(base + self.batch_size, len(self.val_set))))
                if not indici:
                    continue
                loss = self.ddp.module(self._batch(indici, self.val_set))
                for pos, indice in enumerate(indici):
                    episodio, _ = self.val_set.indici[indice]
                    task = self.val_set.episodes[episodio].task_id
                    j = task_to_pos[task]
                    somme[j] += loss[pos].double()
                    conteggi[j] += 1

        dist.all_reduce(somme)
        dist.all_reduce(conteggi)
        medie = somme / conteggi.clamp_min(1)
        by_task = {f"{task:02d}": float(medie[i]) for i, task in enumerate(TRAIN_TASKS)}
        return {
            "macro_task_loss": float(medie.mean()),
            "frame_weighted_loss": float(somme.sum() / conteggi.sum()),
            "by_task": by_task,
        }

    def salva(self, next_epoch):
        rng = [None] * self.world_size
        dist.all_gather_object(rng, _rng_state())
        dist.barrier()
        if self.rank == 0:
            sequence = self.sequence + 1
            destinazione = self.root / ("checkpoint_A" if sequence % 2 else "checkpoint_B")
            temporanea = Path(tempfile.mkdtemp(dir=self.root, prefix=".incomplete_"))
            try:
                esporta(self.ddp.module.policy, self.pre, self.post,
                        temporanea / "pretrained_model")
                torch.save({"optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(), "rng": rng},
                           temporanea / "trainer_state.pt")
                meta = {"sequence": sequence, "next_epoch": next_epoch, "step": self.step,
                        "fingerprint": self.stats["fingerprint"], "head_hash": self.init_hash,
                        "experiment": self.experiment, "config": self.cfg,
                        "world_size": self.world_size}
                (temporanea / "metadata.json").write_text(json.dumps(meta, indent=2))
                if destinazione.exists():
                    shutil.rmtree(destinazione)
                os.replace(temporanea, destinazione)
                self.sequence = sequence
            finally:
                if temporanea.exists():
                    shutil.rmtree(temporanea)
        dist.barrier()

    def resume(self, checkpoint):
        meta = json.loads((checkpoint / "metadata.json").read_text())
        if (meta["fingerprint"] != self.stats["fingerprint"] or
            meta["head_hash"] != self.init_hash or meta["config"] != self.cfg or
            meta["experiment"] != self.experiment or meta["world_size"] != self.world_size):
            raise ValueError("Resume non coerente con dati o configurazione")
        pesi = load_file(checkpoint / "pretrained_model/model.safetensors", device="cpu")
        self.ddp.module.policy.load_state_dict(pesi, strict=True)
        training = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(training["optimizer"])
        self.scheduler.load_state_dict(training["scheduler"])
        _restore_rng(training["rng"][self.rank])
        self.epoch_start, self.step, self.sequence = meta["next_epoch"], meta["step"], meta["sequence"]
        if self.rank == 0:
            print(f"Resume {self.experiment}: epoca {self.epoch_start}, "
                  f"step {self.step}", flush=True)
