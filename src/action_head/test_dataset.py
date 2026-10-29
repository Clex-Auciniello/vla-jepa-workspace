"""Test CPU: batch RGB, chunk future, padding e split/statistiche MIN_MAX."""
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# I test usano una Trajectory minimale con la stessa API di savers.Trajectory.
savers = types.ModuleType("savers")
class Trajectory:
    def __init__(self, actions):
        image = np.zeros((224,224,3), dtype=np.uint8)
        image[:,:,0] = 128
        self.steps = [({"camera_front_image": image, "camera_gripper_image": image}, 0., False, {}, a)
                      for a in actions]
    def __len__(self):
        return len(self.steps)
    def get(self, t, decompress=False):
        obs, reward, done, info, action = self.steps[t]
        return {"obs": obs, "action": action, "reward": reward, "done": done, "info": info}
Trajectory.__module__ = "savers"
savers.Trajectory = Trajectory
sys.modules["savers"] = savers

# _open_episode() di E3/E4 importa anche questo vecchio nome.
multi = types.ModuleType("multi_task_il")
multi.__path__ = []
datasets = types.ModuleType("multi_task_il.datasets")
datasets.__path__ = []
sys.modules["multi_task_il"] = multi
sys.modules["multi_task_il.datasets"] = datasets
sys.modules["multi_task_il.datasets.savers"] = savers

from vla_jepa_offline.dataset import EpisodeSpec
from action_head.dataset import ActionDataset, collate, prepara_dati


class TestDataset(unittest.TestCase):
    def test_chunk_pad_and_minmax(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Due episodi per sorgente e task per permettere lo split a 1 in validazione.
            for task in (1,2,3,4,6,7,8,9,11,12,13,14):
                folder = root / f"task_{task:02d}"
                folder.mkdir()
                actions = [np.array([2,0,0,0,0,0,20], np.float32), np.zeros(7, np.float32)]
                for source in ("tfrecord", "scripted"):
                    for i in (0,1):
                        with (folder / f"{source}_traj_{i:03d}.pkl").open("wb") as f:
                            pickle.dump({"traj": Trajectory(actions), "task_id": f"{task:02d}",
                                         "language_instruction": "Pick a cube"}, f)
            train, val, stats = prepara_dati(root, root/"split.json", val_tfrecord=1, val_scripted=1)
            self.assertEqual((len(train), len(val)), (24,24))
            self.assertEqual(stats["count"], 48)
            self.assertEqual(stats["max"][6], 20)
            self.assertEqual(stats["min"][6], 0)
            # L'ultimo zero è un target reale, quello dopo è padding.
            ds = ActionDataset(train)
            last = ds[1]
            self.assertFalse(bool(last["action_is_pad"][0]))
            self.assertTrue(bool(last["action_is_pad"][1]))
            batch = collate([ds[0], ds[1]])
            self.assertEqual(tuple(batch["action"].shape), (2,7,7))
            self.assertEqual(tuple(batch["observation.images.front"].shape), (2,3,224,224))
            self.assertAlmostEqual(float(batch["observation.images.front"][0,0,0,0]), 128/255, places=6)
            self.assertNotIn("observation.state", batch)

if __name__ == "__main__":
    unittest.main()
