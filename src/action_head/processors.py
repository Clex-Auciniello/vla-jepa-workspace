"""Processor LeRobot: unica normalizzazione attiva, ACTION=MIN_MAX."""
from pathlib import Path
import shutil

import torch

from .dataset import RENAMES


def crea_processors(config, stats):
    from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors
    from lerobot.processor import RenameObservationsProcessorStep

    # LeRobot riceve statistiche del training split, non quelle del checkpoint di origine.
    valori = {k: torch.tensor(stats[k], dtype=torch.float32)
              for k in ("min", "max", "mean", "std")}
    dataset_stats = {"action": valori, "action.world": {k: v.clone() for k, v in valori.items()}}
    pre, post = make_vla_jepa_pre_post_processors(config, dataset_stats)
    for step in pre.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            step.rename_map = RENAMES.copy()
    return pre, post


def esporta(policy, pre, post, directory):
    """Checkpoint LeRobot completo, compatibile col nome richiesto dal controller."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
    pre.save_pretrained(directory)
    post.save_pretrained(directory)
    shutil.copyfile(directory / "policy_postprocessor.json", directory / "policy_postprocessor_ur5e.json")
