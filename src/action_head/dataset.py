"""Dataset action-head: riusa il catalogo e la lettura dei PKL di E3/E4."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vla_jepa_offline.batch_builder import _rgb_to_tensor
from vla_jepa_offline.dataset import TRAIN_TASKS, _open_episode, build_catalog, OfflineWindowDataset


FRONT = "observation.images.front"
GRIPPER = "observation.images.gripper"
RENAMES = {
    FRONT: "observation.images.exterior_1_left",
    GRIPPER: "observation.images.exterior_2_left",
}


def prepara_dati(root, manifest_path, seed=1000, val_tfrecord=4, val_scripted=2):
    """Split per traiettoria e statistiche condivise fra tutti gli esperimenti."""
    catalogo = build_catalog(root)  # Funzione già utilizzata per E3/E4.
    manifest_path = Path(manifest_path)
    gruppi = {}
    for ep in catalogo:
        origine = "scripted" if Path(ep.path).name.startswith("scripted_") else "tfrecord"
        gruppi.setdefault((ep.task_id, origine), []).append(ep.path)

    if manifest_path.exists():
        val_paths = set(json.loads(manifest_path.read_text())["validation"])
        if not val_paths.issubset({ep.path for ep in catalogo}):
            raise ValueError("Split non coerente con il dataset corrente")
    else:
        val_paths = set()
        for task in TRAIN_TASKS:
            for origine, n in (("tfrecord", val_tfrecord), ("scripted", val_scripted)):
                files = sorted(gruppi.get((task, origine), []))
                if len(files) <= n:
                    raise ValueError(f"Traiettorie insufficienti: task {task}, {origine}")
                val_paths.update(random.Random(seed + 100 * task + (origine == "scripted")).sample(files, n))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"validation": sorted(val_paths)}, indent=2))

    train = [ep for ep in catalogo if ep.path not in val_paths]
    validation = [ep for ep in catalogo if ep.path in val_paths]
    if not validation:
        raise ValueError("Validation split vuoto")

    # Statistiche SOLO del training split; nessuna action viene contata più volte.
    stats_path = manifest_path.with_suffix(".action_stats.json")
    impronta = hashlib.sha256()
    for ep in train:
        f = Path(ep.path)
        s = f.stat()
        impronta.update(f"{f}|{s.st_size}|{s.st_mtime_ns}|{ep.frames}\n".encode())
    fingerprint = impronta.hexdigest()
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if stats["fingerprint"] != fingerprint:
            raise ValueError("Statistiche riferite a un dataset diverso")
    else:
        minimo = np.full(7, np.inf, dtype=np.float64)
        massimo = np.full(7, -np.inf, dtype=np.float64)
        somma = np.zeros(7, np.float64)
        somma_quadrati = np.zeros(7, np.float64)
        n = 0
        for ep in train:
            traj = _open_episode(ep.path)["traj"]
            actions = np.stack([traj.get(t, decompress=False)["action"] for t in range(ep.frames)]).astype(np.float64)
            if not np.allclose(actions[-1], 0):
                raise ValueError(f"Ultima action non nulla: {ep.path}")
            minimo = np.minimum(minimo, actions.min(0))
            massimo = np.maximum(massimo, actions.max(0))
            somma += actions.sum(0)
            somma_quadrati += np.square(actions).sum(0)
            n += len(actions)
        if not np.allclose([minimo[6], massimo[6]], [0, 20]):
            raise ValueError("Gripper action atteso nel range 0..20")
        media = somma / n
        std = np.sqrt(np.maximum(somma_quadrati / n - media**2, 0))
        stats = {"min": minimo.tolist(), "max": massimo.tolist(),
                 "mean": media.tolist(), "std": std.tolist(), "count": n,
                 "fingerprint": fingerprint}
        stats_path.write_text(json.dumps(stats, indent=2))
    return train, validation, stats


class ActionDataset(Dataset):
    """Un frame per camera, istruzione testuale e sette azioni consecutive."""

    def __init__(self, episodes, chunk_size=7):
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.indici = [(i, t) for i, ep in enumerate(episodes) for t in range(ep.frames)]

    def __len__(self):
        return len(self.indici)

    def __getitem__(self, index):
        ei, t = self.indici[index]
        ep = self.episodes[ei]
        traj = OfflineWindowDataset._cached_episode(ep.path)  # Cache PKL già collaudata in E3/E4.
        obs = traj.get(t, decompress=False)["obs"]
        action = np.zeros((self.chunk_size, 7), np.float32)
        pad = np.ones(self.chunk_size, dtype=np.bool_)
        for k in range(min(self.chunk_size, ep.frames-t)):
            action[k] = traj.get(t+k, decompress=False)["action"]
            pad[k] = False  # Anche l'ultima azione nulla è un target valido.
        return {
            "front": torch.from_numpy(np.ascontiguousarray(obs["camera_front_image"])),
            "gripper": torch.from_numpy(np.ascontiguousarray(obs["camera_gripper_image"])),
            "task": ep.task,
            "action": torch.from_numpy(action),
            "action_is_pad": torch.from_numpy(pad),
        }


def collate(samples):
    """Riusa la conversione RGB uint8 -> float [0,1] di E3/E4, senza resize."""
    front = torch.stack([s["front"] for s in samples]).unsqueeze(1)
    gripper = torch.stack([s["gripper"] for s in samples]).unsqueeze(1)
    return {
        FRONT: _rgb_to_tensor(front).squeeze(1),
        GRIPPER: _rgb_to_tensor(gripper).squeeze(1),
        "task": [s["task"] for s in samples],
        "action": torch.stack([s["action"] for s in samples]),
        "action_is_pad": torch.stack([s["action_is_pad"] for s in samples]),
    }
