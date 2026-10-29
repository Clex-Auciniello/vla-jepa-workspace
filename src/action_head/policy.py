"""Policy VLA-JEPA congelata, eccetto i parametri effettivamente usati dalla DiT."""
import hashlib

import torch
from torch import nn
from pathlib import Path
from safetensors import safe_open


def inizializza_policy(cfg, local_rank):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import NormalizationMode
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    c = PreTrainedConfig.from_pretrained(cfg["pretrained_path"])
    c.device = f"cuda:{local_rank}"
    c.freeze_qwen = True
    c.enable_world_model = False
    c.reinit_modules = None
    c.use_relative_actions = False
    c.pre_snap_gripper_action = False
    c.binarize_gripper_action = False
    c.clip_normalized_actions = True
    c.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,  # Non esiste observation.state nel batch.
        "ACTION": NormalizationMode.MIN_MAX,
    }
    from .dataset import RENAMES
    if (c.action_dim != 7 or c.chunk_size != 7 or
        set(c.input_features) != set(RENAMES.values())):
        raise ValueError("Il checkpoint deve essere UR5e 7D, con due immagini e senza stato")

    policy = VLAJEPAPolicy.from_pretrained(cfg["pretrained_path"], config=c, strict=False)
    policy.to(c.device)
    head = policy.model.action_model

    # Partenza identica per E1-E4. Supporta anche un riferimento DROID con action_dim=8:
    # copia i pesi compatibili e reinizializza solo le proiezioni che cambiano dimensione.
    filename = str(Path(cfg["head_init_path"]) / "model.safetensors")
    stato = head.state_dict()
    copied, mismatched = {}, []
    with safe_open(filename, framework="pt", device="cpu") as f:
        for key in stato:
            if key.startswith("state_encoder."):
                continue  # Non usato, quindi non entra nell'esperimento.
            full_key = "model.action_model." + key
            if full_key not in f.keys():
                raise ValueError(f"Peso della DiT mancante nel checkpoint iniziale: {full_key}")
            value = f.get_tensor(full_key)
            if value.shape == stato[key].shape:
                copied[key] = value
            else:
                mismatched.append(key)

    # Tutti i rank inizializzano i layer incompatibili con lo stesso seed.
    # I soli mismatch previsti con DROID 8D sono input/output projection 7D.
    previsti = {"action_encoder.layer1.weight", "action_decoder.layer2.weight", "action_decoder.layer2.bias"}
    if set(mismatched) - previsti:
        raise ValueError(f"Dimensioni incompatibili non previste: {mismatched}")
    with torch.random.fork_rng(devices=[local_rank]):
        torch.manual_seed(int(cfg["head_init_seed"]))
        torch.cuda.manual_seed(int(cfg["head_init_seed"]))
        if "action_encoder.layer1.weight" in mismatched:
            head.action_encoder.layer1.reset_parameters()
        if any(k.startswith("action_decoder.layer2.") for k in mismatched):
            head.action_decoder.layer2.reset_parameters()
    head.load_state_dict(copied, strict=False)

    policy.model.qwen.requires_grad_(False)
    policy.model.qwen.eval()
    head.requires_grad_(True)
    if head.state_encoder is not None:
        head.state_encoder.requires_grad_(False)

    # Un'impronta della sola DiT attiva per confrontare E1, E2, E3 ed E4.
    digest = hashlib.sha256()
    for key, val in sorted(head.state_dict().items()):
        if not key.startswith("state_encoder."):
            digest.update(key.encode() + val.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return policy, digest.hexdigest()


class FlowMatchingLoss(nn.Module):
    """Adapter DDP: la loss rimane quella originale di VLA-JEPA in LeRobot."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def train(self, mode=True):
        super().train(mode)
        self.policy.model.qwen.eval()
        return self

    def forward(self, batch):
        # Forward originale LeRobot: World Model spento, quindi solo action loss.
        loss, _ = self.policy(batch)
        return loss
