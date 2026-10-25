"""Lettura dei PKL RGB già preprocessati per E3 ed E4.

Le immagini sono già RGB uint8 224x224; qui non facciamo crop o resize.
Le finestre restano interamente all'interno della singola traiettoria.
"""
from __future__ import annotations

import importlib
import pickle
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

TRAIN_TASKS = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)


@dataclass(frozen=True)
class EpisodeSpec:
    path: str
    task_id: int
    task: str
    frames: int


def _prepare_pickle_modules() -> None:
    # E3 ed E4 sono stati serializzati con due percorsi Python diversi.
    importlib.import_module("savers")
    importlib.import_module("multi_task_il.datasets.savers")


def _open_episode(path: str) -> dict:
    _prepare_pickle_modules()
    with open(path, "rb") as stream:
        return pickle.load(stream)


def build_catalog(dataset_root: str | Path) -> list[EpisodeSpec]:
    """Legge una volta i metadati; in DDP il rank 0 li passa agli altri rank."""
    root = Path(dataset_root).expanduser().resolve()
    catalog = []
    for task_id in TRAIN_TASKS:
        directory = root / f"task_{task_id:02d}"
        paths = sorted(directory.glob("*.pkl"))
        if not paths:
            raise FileNotFoundError(f"Nessun PKL in {directory}")
        for path in paths:
            data = _open_episode(str(path))
            if int(data["task_id"]) != task_id:
                raise ValueError(f"task_id errato: {path}")
            task = data["language_instruction"]
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"Istruzione mancante: {path}")
            catalog.append(EpisodeSpec(str(path), task_id, task, len(data["traj"])))
    return catalog


class OfflineWindowDataset(torch.utils.data.Dataset):
    """Indice delle finestre di 8 frame, con lettura differita dei PKL."""

    def __init__(self, catalog: list[EpisodeSpec], window_size: int = 8, stride: int = 5,
                 camera_count: int = 2):
        self.catalog = catalog
        self.window_size = window_size
        self.camera_count = camera_count
        self.by_episode: list[list[int]] = []
        self.windows: list[tuple[int, int]] = []

        for episode_idx, spec in enumerate(catalog):
            indices = []
            # Solo finestre complete: nessun padding o allineamento artificiale.
            for start in range(0, spec.frames - window_size + 1, stride):
                indices.append(len(self.windows)) #indici globali delle finestre su tutti gli episodi
                self.windows.append((episode_idx, start)) # ogni finestra contiene indice dell'espisodio + frame di start su quell'episodio. episodio = traiettoria
            self.by_episode.append(indices) # self.by_episode contiene gli indici delle finestre raggruppati per episodio
            # self.by_episode = [
            #     [0, 1, 2],  per la traiettoria 1
            #     [3]         per la traiettoria 2
            # ] esempio di cosa contiene.

        if not self.windows:
            raise ValueError("Il dataset non contiene finestre complete")

    @staticmethod
    @lru_cache(maxsize=4)
    def _cached_episode(path: str):
        return _open_episode(path)["traj"]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        episode_idx, start = self.windows[index]
        spec = self.catalog[episode_idx]
        traj = self._cached_episode(spec.path)
        obs = [traj.get(t, decompress=False)["obs"]
               for t in range(start, start + self.window_size)]

        front = np.stack([step["camera_front_image"] for step in obs])
        sample = {"front": torch.from_numpy(front), "task": spec.task}
        if self.camera_count == 2:
            wrist = np.stack([step["camera_gripper_image"] for step in obs])
            sample["gripper"] = torch.from_numpy(wrist)
        return sample

    def epoch_indices(self, seed: int, epoch: int) -> list[int]:
        """Shuffle riproducibile: episodi mescolati, finestre vicine sul disco."""
        rng = random.Random(seed + epoch)
        order = list(range(len(self.catalog)))
        rng.shuffle(order)
        result = []
        for episode_idx in order:
            indices = self.by_episode[episode_idx].copy()
            rng.shuffle(indices)
            result.extend(indices)
        return result
