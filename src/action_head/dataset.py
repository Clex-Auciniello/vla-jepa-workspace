"""PKL robotici: due immagini correnti, task e sette azioni future."""
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


def prepara_dati(root, manifest_path, seed=1000, val_tfrecord=1, val_scripted=1):
    """Uno split per traiettoria (1 TFRecord + 1 scripted per task) e MIN_MAX sul train."""
    catalogo = build_catalog(root)  # Lettura PKL già usata in E3/E4.
    manifest_path = Path(manifest_path)
    gruppi = {}
    for ep in catalogo:
        origine = "scripted" if Path(ep.path).name.startswith("scripted_") else "tfrecord"
        gruppi.setdefault((ep.task_id, origine), []).append(ep.path)

    if manifest_path.exists():
        val_paths = set(json.loads(manifest_path.read_text())["validation"])
    else:
        val_paths = set()
        for task in TRAIN_TASKS:
            for origine, quanti in (("tfrecord", val_tfrecord), ("scripted", val_scripted)):
                files = sorted(gruppi[(task, origine)])
                val_paths.update(random.Random(seed + task * 100 + (origine == "scripted")).sample(files, quanti))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"validation": sorted(val_paths)}, indent=2))

    train = [ep for ep in catalogo if ep.path not in val_paths]
    val = [ep for ep in catalogo if ep.path in val_paths]
    for task in TRAIN_TASKS:
        for origine, quanti in (("tfrecord", val_tfrecord), ("scripted", val_scripted)):
            contati = sum(ep.path in val_paths for ep in catalogo
                          if ep.task_id == task and
                          ("scripted" if Path(ep.path).name.startswith("scripted_") else "tfrecord") == origine)
            if contati != quanti:
                raise ValueError(f"Split errato: task {task}, {origine}: {contati} invece di {quanti}")

    impronta = hashlib.sha256()
    for ep in train:
        p = Path(ep.path)
        st = p.stat()
        impronta.update(f"{p}|{st.st_size}|{st.st_mtime_ns}|{ep.frames}\n".encode())
    fingerprint = impronta.hexdigest()
    stats_path = manifest_path.with_suffix(".action_stats.json")
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if stats["fingerprint"] != fingerprint:
            raise ValueError("Statistiche non coerenti con i PKL correnti")
    else:
        minimo = np.full(7, np.inf, dtype=np.float64)
        massimo = np.full(7, -np.inf, dtype=np.float64)
        somma = np.zeros(7, dtype=np.float64)
        somma_q = np.zeros(7, dtype=np.float64)
        count = 0
        for ep in train:
            traj = _open_episode(ep.path)["traj"]
            action = np.stack([traj.get(t, decompress=False)["action"] for t in range(ep.frames)])
            action = action.astype(np.float64)
            if not np.allclose(action[-1], 0):
                raise ValueError(f"Ultima azione non nulla: {ep.path}")
            minimo = np.minimum(minimo, action.min(axis=0))
            massimo = np.maximum(massimo, action.max(axis=0))
            somma += action.sum(axis=0)
            somma_q += (action ** 2).sum(axis=0)
            count += len(action)
        if not np.allclose([minimo[6], massimo[6]], [0, 20]):
            raise ValueError("Il gripper deve avere valori nell'intervallo [0,20]")
        media = somma / count
        std = np.sqrt(np.maximum(0, somma_q / count - media ** 2))
        stats = {"min": minimo.tolist(), "max": massimo.tolist(),
                 "mean": media.tolist(), "std": std.tolist(), "count": count,
                 "fingerprint": fingerprint}
        stats_path.write_text(json.dumps(stats, indent=2))
    return train, val, stats


class ActionDataset(Dataset):
    """Un elemento per timestep. L'ultimo zero reale NON è padding."""

    def __init__(self, episodes, chunk_size=7):
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.indici = [(i, t) for i, ep in enumerate(episodes) for t in range(ep.frames)]
        self.per_episodio = []
        inizio = 0
        for ep in episodes:
            self.per_episodio.append(list(range(inizio, inizio + ep.frames)))
            inizio += ep.frames

    def conteggi_task(self):
        """Numero reale di frame disponibili per ciascuno dei 12 task."""
        return {
            task: sum(ep.frames for ep in self.episodes if ep.task_id == task)
            for task in TRAIN_TASKS
        }

    def epoch_indices(self, seed, epoch):
        """Batch GLOBALI da 12: un frame per task, senza perdere frame originali.

        Ogni traiettoria contribuisce con tutti i suoi timestep una volta.
        I task più corti ripetono frame casuali fino alla lunghezza del task
        più lungo. Le ripetizioni cambiano a ogni epoca.
        """
        rng = random.Random(seed + epoch)
        conteggi = self.conteggi_task()
        if not all(conteggi.values()):
            raise ValueError("Uno dei task di training non contiene frame")
        massimo = max(conteggi.values())
        per_task = {}

        for task in TRAIN_TASKS:
            # Round-robin casuale sulle traiettorie del task.
            code = []
            for ei, ep in enumerate(self.episodes):
                if ep.task_id == task:
                    indici = self.per_episodio[ei].copy()
                    rng.shuffle(indici)
                    code.append(indici)
            rng.shuffle(code)
            ordine_originale = []
            while code:
                rng.shuffle(code)
                prossime = []
                for indici in code:
                    ordine_originale.append(indici.pop())
                    if indici:
                        prossime.append(indici)
                code = prossime

            # Distribuiamo i duplicati in posizioni casuali dell'epoca.
            extra = massimo - len(ordine_originale)
            posti_extra = set(rng.sample(range(massimo), extra))
            aggiuntivi = [rng.choice(ordine_originale) for _ in range(extra)]
            it_originali = iter(ordine_originale)
            it_extra = iter(aggiuntivi)
            per_task[task] = [next(it_extra) if i in posti_extra else next(it_originali)
                              for i in range(massimo)]

        ordine = []
        for i in range(massimo):
            riga = [per_task[task][i] for task in TRAIN_TASKS]
            rng.shuffle(riga)  # Evita GPU specializzate sempre sugli stessi task.
            ordine.extend(riga)
        return ordine

    def __len__(self):
        return len(self.indici)

    def __getitem__(self, index):
        ei, t = self.indici[index]
        ep = self.episodes[ei]
        traj = OfflineWindowDataset._cached_episode(ep.path)
        obs = traj.get(t, decompress=False)["obs"]
        action = np.zeros((self.chunk_size, 7), dtype=np.float32)
        pad = np.ones(self.chunk_size, dtype=np.bool_)
        for k in range(min(self.chunk_size, ep.frames - t)):
            action[k] = traj.get(t + k, decompress=False)["action"]
            pad[k] = False
        return {"front": torch.from_numpy(np.ascontiguousarray(obs["camera_front_image"])),
                "gripper": torch.from_numpy(np.ascontiguousarray(obs["camera_gripper_image"])),
                "task": ep.task,
                "action": torch.from_numpy(action),
                "action_is_pad": torch.from_numpy(pad)}


def collate(samples):
    """Conversione condivisa con E3/E4: RGB uint8 -> CHW float [0,1]."""
    front = torch.stack([s["front"] for s in samples]).unsqueeze(1)
    wrist = torch.stack([s["gripper"] for s in samples]).unsqueeze(1)
    return {FRONT: _rgb_to_tensor(front).squeeze(1),
            GRIPPER: _rgb_to_tensor(wrist).squeeze(1),
            "task": [s["task"] for s in samples],
            "action": torch.stack([s["action"] for s in samples]),
            "action_is_pad": torch.stack([s["action_is_pad"] for s in samples])}
