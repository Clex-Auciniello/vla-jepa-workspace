"""Lettura delle finestre RGB preprocessate per E1, E3 ed E4.

E3/E4: un prompt per episodio; finestre complete di 8 frame, stride 5.
E1: prompt e indici iniziali delle finestre già presenti nei PKL.
Le immagini sono RGB uint8 224x224: nessun crop o resize durante il training.
"""
from __future__ import annotations

import importlib
import pickle
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

TRAIN_TASKS = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)


@dataclass(frozen=True)
class EpisodeSpec:
    path: str
    task_id: int | None
    task: str | None
    frames: int
    # Solo E1: coppie (frame iniziale, prompt) nell'ordine salvato dal preprocessore.
    window_prompts: tuple[tuple[int, str], ...] | None = None


def _prepare_pickle_modules() -> None:
    # I PKL sono stati serializzati con percorsi Python diversi.
    importlib.import_module("savers")
    importlib.import_module("multi_task_il.datasets.savers")


def _open_episode(path: str) -> dict:
    _prepare_pickle_modules()
    with open(path, "rb") as stream:
        return pickle.load(stream)


def build_catalog(dataset_root: str | Path) -> list[EpisodeSpec]:
    """Catalogo E3/E4: una sola istruzione per ciascuna traiettoria."""
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


def build_exploration_catalog(
    dataset_root: str | Path,
    window_size: int = 8,
    stride: int = 5,
    max_windows: int | None = None,
) -> list[EpisodeSpec]:
    """Catalogo E1: legge i prompt e gli start già calcolati offline.

    max_windows è usato soltanto dallo smoke test per fermarsi presto.
    """
    root = Path(dataset_root).expanduser().resolve()
    paths = sorted(
        (p for p in root.glob("E*.pkl") if re.fullmatch(r"E\d{5,}", p.stem)),
        key=lambda p: int(p.stem[1:]),
    )
    if not paths:
        raise FileNotFoundError(f"Nessun PKL di esplorazione in {root}")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows deve essere positivo")

    catalog = []
    total_windows = 0
    for path in paths:
        data = _open_episode(str(path))
        if data.get("source") != "exploration" or data.get("episode_id") != path.stem:
            raise ValueError(f"Metadati episodio E1 non validi: {path}")
        preprocessing = data.get("preprocessing", {})
        if (preprocessing.get("window_size") != window_size
                or preprocessing.get("stride") != stride
                or preprocessing.get("color_order") != "RGB"
                or preprocessing.get("image_size") != [224, 224]
                or preprocessing.get("front_crop_margins_top_bottom_left_right") != [0, 10, 140, 90]
                or preprocessing.get("gripper_crop") is not None):
            raise ValueError(f"Preprocessing E1 incompatibile: {path}")
        align_last = preprocessing.get("align_last_window")
        if not isinstance(align_last, bool):
            raise ValueError(f"Flag align_last_window mancante: {path}")

        frames = len(data["traj"])
        starts = list(range(0, frames - window_size + 1, stride))
        if not starts:
            raise ValueError(f"Episodio senza finestre complete: {path}")
        if align_last and starts[-1] + window_size < frames:
            # Solo la finestra finale viene spostata, senza aggiungerne altre.
            starts[-1] = frames - window_size

        saved_windows = data.get("window_prompts")
        if not isinstance(saved_windows, list) or len(saved_windows) != len(starts):
            raise ValueError(f"Numero di prompt non coerente con le finestre: {path}")
        pairs = []
        for expected_start, item in zip(starts, saved_windows):
            if not isinstance(item, dict):
                raise ValueError(f"Finestra non valida: {path}")
            start, prompt = item.get("start"), item.get("prompt")
            if (type(start) is not int or start != expected_start
                    or not isinstance(prompt, str) or not prompt.strip()):
                raise ValueError(f"Start o prompt non valido in {path}, start={start}")
            pairs.append((start, prompt))

        catalog.append(EpisodeSpec(str(path), None, None, frames, tuple(pairs)))
        total_windows += len(pairs)
        if max_windows is not None and total_windows >= max_windows:
            break
    return catalog


class OfflineWindowDataset(torch.utils.data.Dataset):
    """Indice delle finestre con caricamento differito delle traiettorie."""

    def __init__(self, catalog: list[EpisodeSpec], window_size: int = 8, stride: int = 5,
                 camera_count: int = 2):
        self.catalog = catalog
        self.window_size = window_size
        self.camera_count = camera_count
        self.by_episode: list[list[int]] = []
        self.windows: list[tuple[int, int, str]] = []

        for episode_idx, spec in enumerate(catalog):
            indices = []
            if spec.window_prompts is None:
                # E3/E4: stesso prompt per tutte le finestre di una traiettoria.
                windows = ((start, spec.task) for start in range(
                    0, spec.frames - window_size + 1, stride))
            else:
                # E1: indici e prompt già salvati, incluso l'eventuale allineamento finale.
                windows = iter(spec.window_prompts)

            for start, prompt in windows:
                indices.append(len(self.windows))
                self.windows.append((episode_idx, start, prompt))
            self.by_episode.append(indices)

        if not self.windows:
            raise ValueError("Il dataset non contiene finestre complete")

    @staticmethod
    @lru_cache(maxsize=4)
    def _cached_episode(path: str):
        return _open_episode(path)["traj"]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        episode_idx, start, prompt = self.windows[index]
        spec = self.catalog[episode_idx]
        traj = self._cached_episode(spec.path)
        obs = [traj.get(t, decompress=False)["obs"]
               for t in range(start, start + self.window_size)]

        front = np.stack([step["camera_front_image"] for step in obs])
        sample = {"front": torch.from_numpy(front), "task": prompt}
        if self.camera_count == 2:
            wrist = np.stack([step["camera_gripper_image"] for step in obs])
            sample["gripper"] = torch.from_numpy(wrist)
        return sample

    def epoch_indices(self, seed: int, epoch: int) -> list[int]:
        """Shuffle riproducibile per episodio e per finestra."""
        rng = random.Random(seed + epoch)
        order = list(range(len(self.catalog)))
        rng.shuffle(order)
        result = []
        for episode_idx in order:
            indices = self.by_episode[episode_idx].copy()
            rng.shuffle(indices)
            result.extend(indices)
        return result
