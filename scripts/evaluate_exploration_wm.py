#!/usr/bin/env python3
"""Valuta la World Model loss di VLA-JEPA sul validation set E1.

Tutte le policy ricevono esattamente le stesse finestre RGB e gli stessi prompt.
Non vengono eseguiti backward, optimizer step o salvataggi dei pesi.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from vla_jepa_offline.batch_builder import FRONT_FEATURE, GRIPPER_FEATURE, make_batch
from vla_jepa_offline.dataset import OfflineWindowDataset, build_exploration_catalog

DEFAULT_VALIDATION = Path("/mnt/beegfs/a.cuciniello9/datasets/robotic/exploration_validation")
EXPECTED_IDS = (
    *range(141, 148), *range(198, 202), *range(243, 247),
    *range(267, 274), *range(374, 380), 440,
)


def resolve_checkpoint(path: Path) -> Path:
    """Accetta sia una cartella del modello sia una root con checkpoint_A/B."""
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path

    candidates = []
    for name in ("checkpoint_A", "checkpoint_B"):
        slot = path / name
        meta_file = slot / "checkpoint_meta.json"
        if not meta_file.is_file() or not (slot / "config.json").is_file():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        sequence = meta.get("sequence", meta.get("checkpoint_sequence"))
        if sequence is None:
            continue
        candidates.append((int(sequence), slot))
    if not candidates:
        raise FileNotFoundError(f"Checkpoint VLA-JEPA non trovato in {path}")
    return max(candidates, key=lambda item: item[0])[1]


def load_policy(path: Path, device: torch.device):
    """Carica solo i pesi del modello: nessuno stato di training/optimizer."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    model_cfg = PreTrainedConfig.from_pretrained(str(path))
    if not isinstance(model_cfg, VLAJEPAConfig):
        raise TypeError(f"Il checkpoint non è VLA-JEPA: {path}")
    if not model_cfg.enable_world_model or model_cfg.num_video_frames != 8:
        raise ValueError(f"Configurazione World Model incompatibile: {path}")
    if model_cfg.num_world_model_views != 2:
        raise ValueError(f"Attese due viste video: {path}")
    if not hasattr(model_cfg, "causal_world_model_context"):
        raise ValueError(f"Manca causal_world_model_context nel checkpoint: {path}")

    # Stessa modalità causale utilizzata nei trainer online e offline.
    model_cfg.causal_world_model_context = True
    model_cfg.device = str(device)
    policy = VLAJEPAPolicy.from_pretrained(str(path), config=model_cfg, strict=False)
    if set(policy.config.image_features) != {FRONT_FEATURE, GRIPPER_FEATURE}:
        raise ValueError(f"Feature visive incompatibili: {path}")
    policy.to(device)
    policy.eval()
    return policy


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Salva risultati completi; non crea file CSV parziali."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_checkpoint(name: str, path: Path, dataset: OfflineWindowDataset,
                        device: torch.device, batch_size: int, seed: int,
                        output_dir: Path) -> dict:
    print(f"\n[{name}] Carico {path}", flush=True)
    policy = load_policy(path, device)

    window_rows = []
    episode_losses = defaultdict(list)
    total_windows = len(dataset)

    # Seme fisso per rendere confrontabili eventuali scelte stocastiche nel forward.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        for offset in range(0, total_windows, batch_size):
            indices = list(range(offset, min(offset + batch_size, total_windows)))
            batch = make_batch(dataset, indices, device)
            losses, _ = policy(batch, reduction="none")
            if losses.ndim != 1 or losses.numel() != len(indices):
                raise ValueError(f"[{name}] Loss per finestra non valida: {tuple(losses.shape)}")
            if not torch.isfinite(losses).all():
                raise FloatingPointError(f"[{name}] World Model loss non finita")
            values = losses.detach().float().cpu().tolist()
            for idx, loss in zip(indices, values):
                episode_idx, start, prompt = dataset.windows[idx]
                episode_id = Path(dataset.catalog[episode_idx].path).stem
                episode_losses[episode_id].append(loss)
                window_rows.append({
                    "model": name, "episode_id": episode_id,
                    "window_index": len(episode_losses[episode_id]) - 1,
                    "start_frame_0_based": start, "wm_loss": loss, "prompt": prompt,
                })
            if offset == 0 or (offset // batch_size + 1) % 20 == 0 or offset + batch_size >= total_windows:
                print(f"[{name}] {min(offset + batch_size, total_windows)}/{total_windows} finestre", flush=True)
            del batch, losses

    episode_rows = []
    for episode_id, losses in episode_losses.items():
        episode_rows.append({
            "model": name, "episode_id": episode_id,
            "windows": len(losses), "mean_wm_loss": sum(losses) / len(losses),
        })

    # Due metriche: media delle finestre e media delle medie per episodio.
    all_losses = [r["wm_loss"] for r in window_rows]
    summary = {
        "model": name,
        "checkpoint": str(path),
        "episodes": len(episode_rows),
        "windows": len(all_losses),
        "mean_wm_loss_per_window": sum(all_losses) / len(all_losses),
        "mean_wm_loss_per_episode": (
            sum(r["mean_wm_loss"] for r in episode_rows) / len(episode_rows)
        ),
    }

    write_csv(output_dir / f"windows_{name}.csv", window_rows,
              ["model", "episode_id", "window_index", "start_frame_0_based", "wm_loss", "prompt"])
    write_csv(output_dir / f"episodes_{name}.csv", episode_rows,
              ["model", "episode_id", "windows", "mean_wm_loss"])
    (output_dir / f"summary_{name}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{name}] WM loss media: {summary['mean_wm_loss_per_window']:.8f}", flush=True)

    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e1_validation_wm"))
    parser.add_argument("--checkpoint", action="append", required=False, default=[],
                        metavar="NAME=PATH", help="Ripetere per ogni modello da confrontare")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size deve essere positivo")
    if not args.preflight_only and not args.checkpoint:
        parser.error("Specificare almeno un --checkpoint NAME=PATH")

    checkpoints = []
    used_names = set()
    for entry in args.checkpoint:
        if "=" not in entry:
            parser.error(f"Checkpoint non valido: {entry}; formato NAME=PATH")
        name, raw_path = entry.split("=", 1)
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not raw_path:
            parser.error(f"Nome/path checkpoint non valido: {entry}")
        if name in used_names:
            parser.error(f"Nome checkpoint duplicato: {name}")
        used_names.add(name)
        checkpoints.append((name, resolve_checkpoint(Path(raw_path))))

    catalog = build_exploration_catalog(args.dataset_root, window_size=8, stride=5)
    actual_ids = sorted(int(Path(spec.path).stem[1:]) for spec in catalog)
    if actual_ids != sorted(EXPECTED_IDS):
        raise ValueError(f"ID di validation diversi dai 29 attesi: {actual_ids}")
    dataset = OfflineWindowDataset(catalog, window_size=8, stride=5, camera_count=2)
    print(f"Validation: {len(catalog)} episodi, {len(dataset)} finestre", flush=True)
    sample = dataset[0]
    print(f"Prima finestra: front={tuple(sample['front'].shape)}, "
          f"wrist={tuple(sample['gripper'].shape)}, task={sample['task']!r}", flush=True)
    if args.preflight_only:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("La valutazione del World Model richiede una GPU CUDA")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, path in checkpoints:
        result = evaluate_checkpoint(name, path, dataset, device,
                                     args.batch_size, args.seed, output_dir)
        results.append(result)
        write_csv(output_dir / "comparison.csv", results,
                  ["model", "checkpoint", "episodes", "windows",
                   "mean_wm_loss_per_window", "mean_wm_loss_per_episode"])
    print(f"\nConfronto salvato: {output_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
