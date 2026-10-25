"""Trainer offline DDP: stesso forward VLA-JEPA, checkpoint a fine epoca.

Aggiorniamo Qwen e il World Model Predictor; V-JEPA2 e Action Head
rimangono congelati. Ogni optimizer step usa fino a 32 finestre reali
(4 GPU x batch 4 x accumulo 2); l'ultimo gruppo può essere più piccolo.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .batch_builder import FRONT_FEATURE, GRIPPER_FEATURE, make_batch

CHECKPOINT_FORMAT = 4


def latest_checkpoint(root: str | Path) -> Path | None:
    """Restituisce lo slot A/B con la sequenza di salvataggio più recente."""
    found = []
    for name in ("checkpoint_A", "checkpoint_B"):
        path = Path(root) / name
        if not path.exists():
            continue
        meta = json.loads((path / "checkpoint_meta.json").read_text())
        if meta["format_version"] != CHECKPOINT_FORMAT:
            raise ValueError(f"Checkpoint non compatibile: {path}")
        found.append((meta["sequence"], path))
    return max(found, key=lambda item: item[0])[1] if found else None


def _rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])


class OfflineTrainer:
    def __init__(self, cfg: dict, *, rank: int, local_rank: int, world_size: int,
                 dataset_identity: str):
        # Si inizializza una replica su ciascuna GPU, assegnata da torchrun.
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.dataset_identity = dataset_identity
        self.root = Path(cfg["checkpoint_root"]).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.next_epoch = 0 # numero prossima epoca da eseguire
        self.global_step = 0 # numero di chiamate a optimizer.step(). Ne avviene uno ogni 4 (GPU) x 4 (batch size) x 2 (grad accum) = 32 finestre
        self.sequence = 0 # numero progressivo dei checkpoint salvati

        checkpoint = latest_checkpoint(self.root)
        source = checkpoint or Path(cfg["pretrained_path"]).expanduser().resolve()
        model_cfg = PreTrainedConfig.from_pretrained(str(source))
        if not isinstance(model_cfg, VLAJEPAConfig):
            raise TypeError("Il checkpoint non contiene una policy VLA-JEPA")
        if not model_cfg.enable_world_model or model_cfg.freeze_qwen or model_cfg.use_peft:
            raise ValueError("Richiesto World Model attivo, Qwen addestrabile e nessun PEFT")
        if model_cfg.num_video_frames != cfg["window_size"]:
            raise ValueError("window_size diversa dalla configurazione del checkpoint")
        if model_cfg.num_world_model_views != 2:
            raise ValueError("Questo pretraining conserva le due viste del checkpoint originale")
        model_cfg.causal_world_model_context = True
        model_cfg.device = str(self.device)

        policy = VLAJEPAPolicy.from_pretrained(str(source), config=model_cfg, strict=False)
        policy.to(self.device)
        model = policy.model
        model.video_encoder.requires_grad_(False)
        model.action_model.requires_grad_(False)
        model.qwen.requires_grad_(True)
        model.video_predictor.requires_grad_(True)

        qwen = model.qwen.model
        # La modalità non-reentrant è preferibile con DDP e gradient checkpointing.
        qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        qwen.config.use_cache = False

        image_keys = tuple(policy.config.image_features)
        if set(image_keys) != {FRONT_FEATURE, GRIPPER_FEATURE}:
            raise ValueError(f"Feature visive inattese: {image_keys}")

        self.params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW([
            {"params": list(model.qwen.parameters()),
             "lr": float(cfg["qwen_learning_rate"])},
            {"params": list(model.video_predictor.parameters()),
             "lr": float(cfg["predictor_learning_rate"])},
        ], betas=model_cfg.optimizer_betas, eps=model_cfg.optimizer_eps,
           weight_decay=model_cfg.optimizer_weight_decay)

        # Tutti i rank hanno la stessa replica. Alcuni pesi Qwen (lm_head)
        # non entrano nel forward del World Model: gestiamo i parametri inutilizzati.
        self.ddp = DDP(policy, device_ids=[local_rank], output_device=local_rank,
                       broadcast_buffers=False, find_unused_parameters=True)
        self.optimizer.zero_grad(set_to_none=True)
        self._set_training_modes()

        if checkpoint is not None:
            self._resume(checkpoint)

    def _set_training_modes(self) -> None:
        self.ddp.train()
        self.ddp.module.model.video_encoder.eval()
        self.ddp.module.model.action_model.eval()

    def train_group(self, dataset, group: list[int]) -> float:
        """Un optimizer step: normalizzazione sul numero REALE di finestre.

        DDP media i gradienti sui rank: moltiplichiamo la somma locale
        per world_size / len(group) prima del backward. I rank senza
        campioni usano un batch fittizio di peso zero, solo per DDP.
        """
        self._set_training_modes()
        local_batch = int(self.cfg["batch_size"]) # 4 batch elaborati dalla singola GPU
        global_batch = local_batch * self.world_size # 16 batch elaborati in totale dalle 4 GPU
        micro_count = (len(group) + global_batch - 1) // global_batch
        local_loss_total = 0.0

        for micro in range(micro_count):
            offset = micro * global_batch + self.rank * local_batch
            indices = group[offset:offset + local_batch]
            # Esempio il gruppo contiene 32 finestre, da finestra 0 a finestra 31
            # Rank	Primo microbatch	Secondo microbatch
            # 0	           0–3	            16–19
            # 1	           4–7	            20–23
            # 2	           8–11	            24–27
            # 3	           12–15	        28–31
            # Ogni GPU elabora 8 finestre in totale

            actual_count = len(indices)
            # Il rank deve partecipare al backward anche se ha 0 campioni reali.
            if not indices:
                indices = [group[0]]
            batch = make_batch(dataset, indices, self.device)

            # Sincronizziamo solo durante l'ultimo backward del gruppo.
            context = nullcontext() if micro == micro_count - 1 else self.ddp.no_sync() # se non siamo all'ultimo microbatch, usiamo no_sync(). Con 32 finestre, primo microbatch di 16 finestre backward locale senza sincronizzazione tra GPU; secondo microbatch avviene backward con sincornizzazione DDP dei gradienti accumulati.
            # in questo modo evitiamo comunicazioni distribuite tra lle GPU non necessarie ad ogni microbatch, ma solo dopo che si è processato il microbatch che ha portato al raggiungimento del grad accum impostato.
            with context:
                losses, _ = self.ddp(batch, reduction="none") # ddp fa da wrapper per il calcolo parallelo, la chiamata arriva a VLAJEPAPolicy.forward()
                if losses.ndim != 1:
                    raise ValueError("Richiesta loss per finestra: reduction='none'")
                real_loss = losses[:actual_count].sum() if actual_count else losses.sum() * 0.0
                local_loss_total += float(real_loss.detach())
                (real_loss * (self.world_size / len(group))).backward()

        torch.nn.utils.clip_grad_norm_(self.params, float(self.cfg["grad_clip_norm"]),
                                       error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return local_loss_total

    def save_checkpoint(self, *, next_epoch: int) -> Path | None:
        """Salva solo dal rank 0, dopo l'ultima sincronizzazione dell'epoca."""
        states = [None] * self.world_size
        dist.all_gather_object(states, _rng_state()) # ogni GPU ha stato generatore proprio, si raggruppano per essere salvati tutti per un resume coerente.
        dist.barrier() # ferma allo stesso punto tutti i 4 processi di addestramento.
        saved_path = None

        if self.rank == 0: # Solo GPU 0 scrive sul disco il modello comune a tutti.
            root = self.root
            slots = [root / "checkpoint_A", root / "checkpoint_B"]
            available = [path for path in slots if not path.exists()]
            if available:
                target = available[0]
            else:
                target = min(slots, key=lambda path: json.loads(
                    (path / "checkpoint_meta.json").read_text())["sequence"])
                shutil.rmtree(target)

            sequence = self.sequence + 1
            with tempfile.TemporaryDirectory(prefix=".checkpoint_tmp_", dir=root) as temp_dir:
                temporary = Path(temp_dir)
                self.ddp.module.save_pretrained(temporary)
                torch.save({
                    "format_version": CHECKPOINT_FORMAT,
                    "optimizer": self.optimizer.state_dict(),
                    "next_epoch": next_epoch,
                    "global_step": self.global_step,
                    "sequence": sequence,
                    "dataset_identity": self.dataset_identity,
                    "batch_size": self.cfg["batch_size"],
                    "gradient_accumulation_steps": self.cfg["gradient_accumulation_steps"],
                    "world_size": self.world_size,
                    "rng_states": states,
                }, temporary / "offline_trainer_state.pt")
                (temporary / "checkpoint_meta.json").write_text(json.dumps({
                    "format_version": CHECKPOINT_FORMAT,
                    "sequence": sequence,
                    "next_epoch": next_epoch,
                    "global_step": self.global_step,
                }, indent=2) + "\n")
                os.replace(temporary, target)
            self.sequence = sequence
            saved_path = target
        dist.barrier()
        return saved_path

    def _resume(self, checkpoint: Path) -> None:
        state = torch.load(checkpoint / "offline_trainer_state.pt", map_location="cpu",
                           weights_only=True)
        if state["format_version"] != CHECKPOINT_FORMAT:
            raise ValueError("Formato checkpoint DDP non compatibile")
        if state["dataset_identity"] != self.dataset_identity:
            raise ValueError("Dataset/configurazione diversi dal checkpoint: usare una nuova cartella")
        for key, expected in (("world_size", self.world_size),
                              ("batch_size", self.cfg["batch_size"]),
                              ("gradient_accumulation_steps", self.cfg["gradient_accumulation_steps"])):
            if state[key] != expected:
                raise ValueError(f"Resume non compatibile: {key}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.next_epoch = int(state["next_epoch"])
        self.global_step = int(state["global_step"])
        self.sequence = int(state["sequence"])
        _restore_rng(state["rng_states"][self.rank])
