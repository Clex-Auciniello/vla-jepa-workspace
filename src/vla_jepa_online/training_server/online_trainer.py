"""Pretraining ONLINE video-only di VLA-JEPA (versione LeRobot dell'utente).

Strategia A:
- aggiornare TUTTO Qwen3-VL e TUTTO il World Model Predictor;
- congelare V-JEPA2 (encoder target) e DiT Action Head;
- ricevere mini-batch di UNA finestra video da 8 frame per vista;
- accumulare gradienti di DIECI finestre anche ATTRAVERSO episodi diversi;
- contare l'episodio quando TUTTE le sue finestre hanno fatto backward;
- prima del checkpoint ogni DIECI episodi, applicare i gradienti residui;
- usare solo DUE slot di checkpoint completi, sovrascrivendo il piu' vecchio.

Il trainer non legge ZIP, non fa crop/JPEG e non parla con ROS o Flask.
Queste responsabilita' spettano a online_batch_builder e al server FIFO.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

def seed_everything(seed: int) -> None:
    """Imposta i generatori casuali del processo di training."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Evita la selezione automatica di algoritmi cuDNN
    # sulla base di benchmark che potrebbero variare.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class OnlineTrainerConfig:
    # Pretraining aggiuntivo dal modello DROID: NON dai pesi UR5e gia' fine-tunati.
    pretrained_path: str = "checkpoints/base"
    revision: str | None = None
    resume_from: str | Path | None = None
    device: str = "cuda"
    seed: int = 1000

    # Full training, no PEFT/LoRA. LR separati per facilitare la regolazione.
    qwen_learning_rate: float = 1e-5
    predictor_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0
    qwen_gradient_checkpointing: bool = True

    # B=1 e accumulo=10: batch effettivo fino a 10 finestre.
    batch_size: int = 1
    gradient_accumulation_steps: int = 10

    # I salvataggi dipendono dai VIDEO completati, non dagli optimizer step.
    checkpoint_every_videos: int = 10
    checkpoint_root: str | Path = "checkpoints/exploration"
    # Due soli slot sul disco: checkpoint_A e checkpoint_B.
    keep_last_checkpoints: int = 2


class OnlineVideoTrainer:
    """Un modello residente su singola GPU, un AdamW persistente e un worker FIFO.

    Il metodo pubblico `train_episode()` contiene tutto il ciclo di training di
    un video: solo alla fine aumenta `processed_videos`. L'accumulo di gradienti
    NON e' azzerato tra episodi; ogni 10 finestre viene effettuato uno step.
    Il lock protegge aggiornamenti e checkpoint dall'accesso concorrente.
    """

    def __init__(self, cfg: OnlineTrainerConfig) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
        from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

        self.cfg = cfg
        self._lock = threading.Lock()
        self.global_step = 0            # numero di optimizer.step() effettuati
        self.processed_videos = 0       # episodi completati con successo
        self.completed_episode_ids: set[str] = set()  # deduplica solo nella sessione corrente (non su disco)
        # Telemetria persistente per numero video E/P (ID), non dati/prompt.
        # "other" raccoglie eventuali ID non standard; "legacy" i video dei
        # checkpoint precedenti che non memorizzavano provenienza.
        self.source_video_counts = {"E": 0, "P": 0, "other": 0}
        self.source_window_counts = {"E": 0, "P": 0, "other": 0}
        self.legacy_unclassified_videos = 0
        self.pending_windows = 0       # backward eseguiti, ancora senza optimizer.step()
        self._checkpoint_sequence = 0  # numero progressivo dei salvataggi completati
        self._requires_recovery = False  # errore in un episodio parzialmente addestrato
        self.device = torch.device(cfg.device)
        seed_everything(cfg.seed)

        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Il trainer VLA-JEPA richiede CUDA e BF16")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("La GPU non supporta BF16")
        if cfg.resume_from is not None and cfg.revision is not None:
            raise ValueError("revision non si applica ai checkpoint locali in resume_from")
        if cfg.batch_size != 1 or cfg.gradient_accumulation_steps != 10:
            raise ValueError("Configurazione fissata: batch_size=1, gradient_accumulation_steps=10")
        if cfg.checkpoint_every_videos != 10:
            raise ValueError("Configurazione fissata: checkpoint ogni 10 video completati")
        if cfg.keep_last_checkpoints != 2:
            raise ValueError("Questa pipeline conserva esattamente due slot di checkpoint")
        for key, value in (
            ("qwen_learning_rate", cfg.qwen_learning_rate),
            ("predictor_learning_rate", cfg.predictor_learning_rate),
            ("grad_clip_norm", cfg.grad_clip_norm),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} deve essere positivo e finito")

        # Caricamento della configurazione tramite la classe base LeRobot.
        # Il campo "type" nel config.json seleziona la sottoclasse registrata.
        # Non carichiamo direttamente con VLAJEPAConfig.from_pretrained(),
        # per evitare l'errore di deserializzazione Draccus.

        load_path = str(cfg.resume_from or cfg.pretrained_path)
        revision = None if cfg.resume_from is not None else cfg.revision

        model_cfg = PreTrainedConfig.from_pretrained(
            load_path,
            revision=revision,
        )

        # Verifica che LeRobot abbia riconosciuto la policy corretta.
        if not isinstance(model_cfg, VLAJEPAConfig):
            raise TypeError(
                f"Configurazione inattesa: {type(model_cfg).__name__}"
            )

        if model_cfg.type != "vla_jepa":
            raise ValueError(f"Checkpoint non VLA-JEPA: {model_cfg.type}")
        
        if not model_cfg.enable_world_model or model_cfg.freeze_qwen:
            raise ValueError("Servono enable_world_model=True e freeze_qwen=False")
        if model_cfg.use_peft:
            raise ValueError("Strategia A: usare un checkpoint completo senza PEFT")
        if model_cfg.num_video_frames != 8 or model_cfg.jepa_tubelet_size != 2:
            raise ValueError("Questa pipeline richiede 8 frame e 2 viste")

        if not hasattr(model_cfg, "causal_world_model_context"):
            raise RuntimeError(
                "La versione importata di LeRobot non supporta "
                "causal_world_model_context"
            )

        model_cfg.causal_world_model_context = True
        model_cfg.device = str(self.device)

        self.policy = VLAJEPAPolicy.from_pretrained(
            load_path, config=model_cfg, revision=revision, strict=False
        )
        self.policy.to(self.device)
        model = self.policy.model
        if model.video_encoder is None or model.video_predictor is None:
            raise RuntimeError("Checkpoint senza World Model")

        # Separazione dei gradienti: Qwen e Predictor sono gli unici moduli
        # addestrabili. Il video encoder e' un teacher di embedding stabile.
        model.video_encoder.requires_grad_(False)
        model.action_model.requires_grad_(False)
        model.qwen.requires_grad_(True)
        model.video_predictor.requires_grad_(True)

        if cfg.qwen_gradient_checkpointing:
            qwen = model.qwen.model
            if not hasattr(qwen, "gradient_checkpointing_enable"):
                raise RuntimeError("Qwen non espone gradient_checkpointing_enable")
            qwen.gradient_checkpointing_enable()
            # KV cache non serve nel forward di training ed e' incompatibile
            # con il gradient checkpointing di molte versioni Transformers.
            qwen.config.use_cache = False

        self.image_keys = tuple(self.policy.config.image_features.keys())
        if len(self.image_keys) != 2:
            raise ValueError(f"Servono 2 input visivi; trovati {self.image_keys}")
        self.frame_count = int(self.policy.config.num_video_frames)
        self.image_shape = tuple(self.policy.config.image_features[self.image_keys[0]].shape)
        if self.image_shape != (3, 224, 224):
            raise ValueError(f"Shape immagine inattesa: {self.image_shape}")
        for key in self.image_keys[1:]:
            if tuple(self.policy.config.image_features[key].shape) != self.image_shape:
                raise ValueError("Le due viste devono avere la stessa shape")

        qwen_params = [p for p in model.qwen.parameters() if p.requires_grad]
        predictor_params = [p for p in model.video_predictor.parameters() if p.requires_grad]
        if not qwen_params or not predictor_params:
            raise RuntimeError("Nessun parametro addestrabile in Qwen o World Model")
        self._trainable_params = qwen_params + predictor_params
        self.num_trainable_parameters = sum(p.numel() for p in self._trainable_params)

        # Due param groups permettono di controllare indipendentemente le due LR.
        self.optimizer = torch.optim.AdamW(
            [
                {"params": qwen_params, "lr": cfg.qwen_learning_rate, "name": "qwen"},
                {"params": predictor_params, "lr": cfg.predictor_learning_rate, "name": "predictor"},
            ],
            betas=model_cfg.optimizer_betas,
            eps=model_cfg.optimizer_eps,
            weight_decay=model_cfg.optimizer_weight_decay,
        )
        # Nessuno scheduler implicito: con uno stream online illimitato uno
        # scheduler offline da 16k/30k step consumerebbe il LR arbitrariamente.

        if cfg.resume_from is not None:
            self._load_optimizer_state(Path(cfg.resume_from))
        # Lo azzeriamo SOLO all'inizio o dopo uno step, non fra gli episodi.
        self.optimizer.zero_grad(set_to_none=True)
        self._set_training_modes()

    def _set_training_modes(self) -> None:
        # policy.train() metterebbe anche gli encoder congelati in train mode;
        # li riportiamo subito in eval (dropout/BatchNorm non stocastici).
        self.policy.train()
        self.policy.model.video_encoder.eval()
        self.policy.model.action_model.eval()

    def _validate_batch(self, batch: Mapping[str, Any]) -> None:
        # Il forward nativo senza `action` restituisce solo la L1 WM non pesata.
        if batch.get("action") is not None or batch.get("observation.state") is not None:
            raise ValueError("Video-only: non fornire action oppure observation.state")
        texts = batch.get("task")
        if not isinstance(texts, (list, tuple)) or len(texts) != self.cfg.batch_size:
            raise ValueError("batch_size=1: 'task' deve contenere esattamente un prompt")
        if not isinstance(texts[0], str) or not texts[0].strip():
            raise ValueError("Prompt del batch vuoto o non valido")
        for key in self.image_keys:
            pixels = batch.get(key)
            if not isinstance(pixels, torch.Tensor):
                raise ValueError(f"Manca il tensore {key}")
            expected = (1, self.frame_count, *self.image_shape)
            if tuple(pixels.shape) != expected:
                raise ValueError(f"{key}: attesa {expected}, ottenuta {tuple(pixels.shape)}")
            if pixels.dtype not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"{key}: dtype non supportato {pixels.dtype}")
            if pixels.is_floating_point() and not torch.isfinite(pixels).all():
                raise ValueError(f"{key}: immagine contenente NaN o inf")

    def _optimizer_update(self, *, accumulated_windows: int) -> float:
        """Applica la MEDIA dei gradienti gia' accumulati.

        Ogni finestra esegue backward(loss / 10): per i gruppi completi
        abbiamo gia' la media corretta. SOLO quando dobbiamo salvare e ci
        sono 1..9 finestre residue moltiplichiamo i gradienti per 10/N.
        Il clipping si applica dopo questa correzione, prima di AdamW.step().
        """
        if not (1 <= accumulated_windows <= self.cfg.gradient_accumulation_steps):
            raise ValueError("Numero di finestre accumulato non valido")
        if accumulated_windows != self.pending_windows:
            raise RuntimeError("Contatore dei gradienti incoerente")
        if accumulated_windows < self.cfg.gradient_accumulation_steps:
            factor = self.cfg.gradient_accumulation_steps / accumulated_windows
            for param in self._trainable_params:
                if param.grad is not None:
                    param.grad.mul_(factor)

        norm = torch.nn.utils.clip_grad_norm_(
            self._trainable_params,
            max_norm=self.cfg.grad_clip_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending_windows = 0
        self.global_step += 1
        return float(norm.detach().item())

    def _flush_pending_unlocked(self) -> float | None:
        """Completa gli eventuali gradienti residui prima del salvataggio.

        In questo modo uno snapshot rappresenta *tutti* i video conteggiati e
        non dobbiamo memorizzare enormi gradienti Qwen nel checkpoint.
        """
        if self.pending_windows == 0:
            return None
        return self._optimizer_update(accumulated_windows=self.pending_windows)

    def train_episode(
        self,
        episode_id: str,
        batches: Iterable[Mapping[str, Any]],
        on_window_complete: Callable[[int, float], None] | None = None,
    ) -> dict[str, Any]:
        """Elabora tutte le finestre del video, MA non forza uno step al confine.

        Esempio: video A=6 finestre, video B=7. Uno step avviene dopo
        A(6)+B(4); le altre 3 finestre B rimangono in accumulo per il video C.
        L'episodio diventa 'completato' dopo l'ultimo backward, mentre lo
        step che applica i suoi gradienti puo' avvenire nel video seguente.
        """
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id deve essere una stringa non vuota")

        with self._lock:
            if self._requires_recovery:
                raise RuntimeError(
                    "Trainer in stato non recuperabile dopo un episodio parziale: "
                    "ripartire dall'ultimo checkpoint e riprodurre gli input mancanti"
                )
            if episode_id in self.completed_episode_ids:
                raise ValueError(f"Episodio gia' completato: {episode_id}")
            self._set_training_modes()

            windows = 0
            optimizer_steps = 0
            sum_wm_loss = 0.0
            last_grad_norm: float | None = None
            started_forward = False
            try:
                for batch in batches:
                    self._validate_batch(batch)
                    # Non resettiamo i gradienti: potrebbero gia' contenere le
                    # finestre degli episodi precedenti ancora sotto quota 10.
                    started_forward = True
                    loss, logs = self.policy(dict(batch))
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(f"World Model loss non finita: {logs!r}")
                    # B fisico 1: liberiamo subito il grafo dopo ogni backward.
                    # Il contributo e' 1/10 indipendentemente dall'episode_id.
                    (loss / self.cfg.gradient_accumulation_steps).backward()
                    windows += 1
                    self.pending_windows += 1
                    sum_wm_loss += float(logs["wm_loss"])
                    started_forward = False

                    if self.pending_windows == self.cfg.gradient_accumulation_steps:
                        last_grad_norm = self._optimizer_update(
                            accumulated_windows=self.pending_windows
                        )
                        optimizer_steps += 1

                    # Callback facoltativa verso il server. Viene invocata SOLO
                    # dopo backward e, dove previsto, optimizer.step(): i numeri
                    # di avanzamento non precedono il lavoro GPU effettivo.
                    # Il server deve gestire eventuali errori di telemetria,
                    # senza sollevare eccezioni da questa callback.
                    if on_window_complete is not None:
                        on_window_complete(windows, float(logs["wm_loss"]))

                if windows == 0:
                    raise ValueError("Episodio senza finestre: non puo' essere completato")
            except Exception:
                # Se abbiamo elaborato anche una sola finestra, potrebbero
                # esserci modifiche ai pesi o gradienti misti con altri video.
                # NON e' possibile annullare esattamente un episodio fallito:
                # blocchiamo per sicurezza altri step/snapshot fino a recovery.
                if windows > 0 or started_forward:
                    self._requires_recovery = True
                    self.optimizer.zero_grad(set_to_none=True)
                    self.pending_windows = 0
                raise

            # Un video conta SOLO quando tutte le sue finestre hanno eseguito
            # backward; non serve che i gradienti siano gia' stati applicati.
            self.processed_videos += 1
            self.completed_episode_ids.add(episode_id)
            source = "E" if episode_id.startswith("E") else (
                "P" if episode_id.startswith("P") else "other"
            )
            self.source_video_counts[source] += 1
            self.source_window_counts[source] += windows

            checkpoint: Path | None = None
            if self.processed_videos % self.cfg.checkpoint_every_videos == 0:
                # Eccezione al batch effettivo 10: forziamo uno step sulla
                # media delle finestre residue solo al confine dei checkpoint.
                if self.pending_windows:
                    last_grad_norm = self._flush_pending_unlocked()
                    optimizer_steps += 1
                checkpoint = self._save_checkpoint_unlocked()

            return {
                "episode_id": episode_id,
                "processed_videos": self.processed_videos,
                "windows": windows,
                "optimizer_steps_for_video": optimizer_steps,
                "global_step": self.global_step,
                "pending_windows": self.pending_windows,
                "mean_wm_loss": sum_wm_loss / windows,
                "last_grad_norm": last_grad_norm,
                "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
            }

    def save_checkpoint(self) -> Path:
        """Snapshot manuale: prima applica i gradienti ancora pendenti.

        Anche i checkpoint manuali ruotano negli stessi DUE slot; nessun
        checkpoint extra viene lasciato nel disco condiviso della Spark.
        Chiamarlo fra episodi, non mentre si sta processando un video.
        """
        with self._lock:
            if self._requires_recovery:
                raise RuntimeError("Trainer incoerente: checkpoint bloccato fino a recovery")
            self._flush_pending_unlocked()
            return self._save_checkpoint_unlocked()

    def _select_checkpoint_slot(self, root: Path) -> Path:
        """Sceglie lo slot libero oppure quello con salvataggio piu' vecchio.

        Leggiamo soltanto un piccolo file JSON di metadati, NON lo stato AdamW
        che puo' occupare molti gigabyte nella versione full-Qwen.
        """
        slots = [root / "checkpoint_A", root / "checkpoint_B"]

        def sequence(slot: Path) -> int:
            meta_path = slot / "checkpoint_meta.json"
            if not meta_path.is_file():
                raise RuntimeError(f"Metadati checkpoint mancanti: {meta_path}")
            with meta_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("format_version") != 3:
                raise RuntimeError(f"Metadati checkpoint incompatibili: {slot}")
            return int(data["checkpoint_sequence"])

        existing = [(slot, sequence(slot)) for slot in slots if slot.exists()]
        # Un root che contiene checkpoint piu' recenti dello stato attuale
        # indica un resume dal checkpoint sbagliato o un nuovo run sullo stesso
        # output. Non sovrascriviamo silenziosamente il progresso successivo.
        if existing and max(seq for _, seq in existing) > self._checkpoint_sequence:
            raise RuntimeError(
                "Il checkpoint_root contiene uno snapshot piu' recente di quello "
                "caricato: riprendere dal piu' recente o scegliere un nuovo root"
            )
        for slot in slots:
            if not slot.exists():
                return slot
        return min(existing, key=lambda item: item[1])[0]

    def _save_checkpoint_unlocked(self) -> Path:
        """Salva i pesi COMPLETI e AdamW usando al massimo due slot.

        Procedura: identifichiamo lo slot piu' vecchio, lo eliminiamo, creiamo
        un nuovo checkpoint in una directory temporanea, infine lo rinominiamo
        nello slot libero. Il checkpoint piu' recente RIMANE intatto anche se
        la scrittura del nuovo fallisce. Il picco di spazio occupato e' al
        massimo due snapshot, piu' gli eventuali file temporanei parziali.
        """
        if self.pending_windows:
            raise RuntimeError("Non salvare con gradienti pendenti: chiamare prima flush")
        if self._requires_recovery:
            raise RuntimeError("Non salvare dopo un episodio parziale fallito")

        root = Path(self.cfg.checkpoint_root)
        root.mkdir(parents=True, exist_ok=True)
        target = self._select_checkpoint_slot(root)
        # Incrementiamo la sequenza solo DOPO il commit su filesystem.
        next_sequence = self._checkpoint_sequence + 1

        # L'eventuale vecchio checkpoint da sovrascrivere viene eliminato PRIMA
        # di salvare: non occupiamo lo spazio di TRE copie complete su disco.
        # L'altro slot resta disponibile per il resume dopo un crash.
        if target.exists():
            shutil.rmtree(target)

        with tempfile.TemporaryDirectory(prefix=".checkpoint_tmp_", dir=root) as tmp:
            temporary = Path(tmp)
            self.policy.save_pretrained(temporary)
            torch.save(
                {
                    "format_version": 3,
                    "global_step": self.global_step,
                    "processed_videos": self.processed_videos,
                    "source_video_counts": dict(self.source_video_counts),
                    "source_window_counts": dict(self.source_window_counts),
                    "legacy_unclassified_videos": self.legacy_unclassified_videos,
                    "optimizer": self.optimizer.state_dict(),
                    "base_pretrained_path": self.cfg.pretrained_path,
                    "batch_size": self.cfg.batch_size,
                    "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
                    "checkpoint_every_videos": self.cfg.checkpoint_every_videos,
                    "pending_windows": 0,  # nessun gradiente lasciato fuori dallo snapshot
                    "checkpoint_sequence": next_sequence,
                    # RNG dei diversi generatori per ripresa quanto piu' fedele
                    # possibile del training; non contiene dati degli episodi.
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": self._numpy_rng_snapshot(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "seed": self.cfg.seed,
                    },
                },
                temporary / "online_trainer_state.pt",
            )
            (temporary / "checkpoint_meta.json").write_text(
                json.dumps({
                    "format_version": 3,
                    "checkpoint_sequence": next_sequence,
                    "processed_videos": self.processed_videos,
                    "global_step": self.global_step,
                    "source_video_counts": dict(self.source_video_counts),
                    "source_window_counts": dict(self.source_window_counts),
                    "legacy_unclassified_videos": self.legacy_unclassified_videos,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            # tmp e target sono sullo stesso filesystem: rename della directory
            # completa (non rende il checkpoint immune a perdite di corrente).
            os.replace(temporary, target)

        self._checkpoint_sequence = next_sequence
        return target

    def _load_optimizer_state(self, checkpoint_dir: Path) -> None:
        """Riprende da un checkpoint COMPLETO di questa versione del trainer."""
        state_path = checkpoint_dir / "online_trainer_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Stato optimizer mancante: {state_path}")
        # Usare solo file locali fidati; weights_only=True limita pickle.
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 3:
            raise ValueError("Formato checkpoint incompatibile (richiesta versione 3)")
        saved_seed = payload.get("rng_state", {}).get("seed")
        if saved_seed is not None and saved_seed != self.cfg.seed:
            raise ValueError(
                f"Seed incompatibile nel resume: "
                f"checkpoint={saved_seed}, configurazione={self.cfg.seed}"
            )
        for key in ("batch_size", "gradient_accumulation_steps", "checkpoint_every_videos"):
            if payload.get(key) != getattr(self.cfg, key):
                raise ValueError(f"Config resume incompatibile: {key}")
        if payload.get("pending_windows") != 0:
            raise ValueError("Checkpoint contiene gradienti pendenti non ripristinabili")

        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload["global_step"])
        self.processed_videos = int(payload["processed_videos"])
        has_counters = "source_video_counts" in payload
        # Checkpoint v3 piu' vecchi: NON attribuire arbitrariamente i
        # video storici a Exploration, perche' la sorgente non era persistita.
        self.legacy_unclassified_videos = int(payload.get(
            "legacy_unclassified_videos", 0 if has_counters else self.processed_videos
        ))
        self.source_video_counts = {
            k: int(payload.get("source_video_counts", {}).get(k, 0))
            for k in ("E", "P", "other")
        }
        self.source_window_counts = {
            k: int(payload.get("source_window_counts", {}).get(k, 0))
            for k in ("E", "P", "other")
        }
        if (self.legacy_unclassified_videos < 0 or
            any(x < 0 for x in self.source_video_counts.values()) or
            any(x < 0 for x in self.source_window_counts.values()) or
            self.legacy_unclassified_videos + sum(self.source_video_counts.values()) != self.processed_videos):
            raise ValueError("Contatori E/P e storico checkpoint incoerenti")
        # Gli ID precedenti non servono al resume dei pesi: deduplica soltanto
        # dentro la sessione Flask corrente. Leggiamo anche snapshot v3 vecchi
        # che contenevano completed_episode_ids, senza richiederli.
        self.completed_episode_ids = set()
        self._checkpoint_sequence = int(payload["checkpoint_sequence"])
        self.pending_windows = 0
        if (self.global_step < 0 or self.processed_videos < 0 or
                self._checkpoint_sequence < 1 or
                self.pending_windows != 0):
            raise ValueError("Contatori/ID non coerenti nel checkpoint")
        self._restore_rng_states(payload.get("rng_state", {}))

    @staticmethod
    def _numpy_rng_snapshot() -> tuple[str, list[int], int, int, float]:
        """Conversione a tipi semplici compatibili con torch.load(weights_only=True)."""
        name, state, pos, has_gauss, cached = np.random.get_state()
        return (name, state.tolist(), int(pos), int(has_gauss), float(cached))

    @staticmethod
    def _restore_rng_states(states: dict[str, Any]) -> None:
        """Ripristina gli RNG dopo il caricamento dei pesi e dell'optimizer."""
        if not states:
            return  # Compatibilità con i checkpoint v3 precedenti.
        random.setstate(states["python"])
        name, values, pos, has_gauss, cached = states["numpy"]
        np.random.set_state((name, np.array(values, dtype=np.uint32), pos, has_gauss, cached))
        torch.set_rng_state(states["torch_cpu"])
        if torch.cuda.is_available() and states["torch_cuda"]:
            if len(states["torch_cuda"]) != torch.cuda.device_count():
                raise ValueError("Checkpoint RNG CUDA incompatibile con il numero di GPU")
            torch.cuda.set_rng_state_all(states["torch_cuda"])
