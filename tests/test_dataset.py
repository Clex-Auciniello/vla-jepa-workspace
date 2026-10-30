"""Test CPU dei PKL, del padding e del bilanciamento sui dodici task."""
import collections
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# Simuliamo solo la classe Trajectory: i moduli di training reali sono gli stessi E3/E4.
savers = types.ModuleType("savers")


class Trajectory:
    def __init__(self, actions):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 0] = 128
        self.steps = [({"camera_front_image": img, "camera_gripper_image": img},
                       0., False, {}, a) for a in actions]

    def __len__(self):
        return len(self.steps)

    def get(self, t, decompress=False):
        obs, reward, done, info, action = self.steps[t]
        return {"obs": obs, "action": action, "reward": reward, "done": done, "info": info}


Trajectory.__module__ = "savers"
savers.Trajectory = Trajectory
sys.modules["savers"] = savers
multi = types.ModuleType("multi_task_il")
multi.__path__ = []
datasets = types.ModuleType("multi_task_il.datasets")
datasets.__path__ = []
sys.modules["multi_task_il"] = multi
sys.modules["multi_task_il.datasets"] = datasets
sys.modules["multi_task_il.datasets.savers"] = savers

from vla_jepa_offline.dataset import TRAIN_TASKS
from action_head.dataset import ActionDataset, collate, prepara_dati


class TestDataset(unittest.TestCase):
    def test_pkl_and_balanced_epochs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for task in TRAIN_TASKS:
                folder = root / f"task_{task:02d}"
                folder.mkdir()
                for source in ("tfrecord", "scripted"):
                    for i in (0, 1):
                        # Dimensioni diverse fra task: forziamo l'oversampling.
                        n = 3 + task % 5
                        actions = [np.array([2, 0, 0, 0, 0, 0, 20], np.float32)
                                   for _ in range(n - 1)]
                        actions.append(np.zeros(7, np.float32))
                        with (folder / f"{source}_traj_{i:03d}.pkl").open("wb") as f:
                            pickle.dump({"traj": Trajectory(actions), "task_id": f"{task:02d}",
                                         "language_instruction": "Pick a cube"}, f)

            train, val, stats = prepara_dati(root, root / "split.json",
                                             val_tfrecord=1, val_scripted=1)
            self.assertEqual((len(train), len(val)), (24, 24))
            self.assertEqual(stats["min"][6], 0)
            self.assertEqual(stats["max"][6], 20)
            ds = ActionDataset(train)
            ultimo = ds[ds.per_episodio[0][-1]]
            self.assertFalse(bool(ultimo["action_is_pad"][0]))
            self.assertTrue(bool(ultimo["action_is_pad"][1]))
            self.assertTrue(np.array_equal(ultimo["action"][0].numpy(), np.zeros(7)))

            nmax = max(ds.conteggi_task().values())
            for epoch in (0, 1):
                ordine = ds.epoch_indices(seed=1000, epoch=epoch)
                self.assertEqual(len(ordine), 12 * nmax)
                self.assertEqual(set(ordine), set(range(len(ds))))
                frequenze = collections.Counter(ordine)
                for task in TRAIN_TASKS:
                    task_indices = [j for j, (ei, _) in enumerate(ds.indici)
                                    if ds.episodes[ei].task_id == task]
                    self.assertEqual(sum(frequenze[j] for j in task_indices), nmax)
                    if len(task_indices) == nmax:
                        self.assertTrue(all(frequenze[j] == 1 for j in task_indices))
                for base in range(0, len(ordine), 12):
                    riga = ordine[base:base + 12]
                    task_ids = [ds.episodes[ds.indici[j][0]].task_id for j in riga]
                    self.assertEqual(set(task_ids), set(TRAIN_TASKS))
                for base in range(0, len(ordine), 24):
                    gruppo = ordine[base:base + 24]
                    self.assertIn(len(gruppo), (12, 24))
                    for task in TRAIN_TASKS:
                        self.assertEqual(sum(ds.episodes[ds.indici[j][0]].task_id == task
                                             for j in gruppo), len(gruppo) // 12)

            batch = collate([ds[0], ds[1]])
            self.assertEqual(tuple(batch["action"].shape), (2, 7, 7))
            self.assertEqual(tuple(batch["observation.images.front"].shape),
                             (2, 3, 224, 224))
            self.assertAlmostEqual(float(batch["observation.images.front"][0, 0, 0, 0]),
                                   128 / 255, places=6)
            self.assertNotIn("observation.state", batch)


if __name__ == "__main__":
    unittest.main()
