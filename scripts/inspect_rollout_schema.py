#!/usr/bin/env python3
"""Read-only structural inspection of PKL/JSON robot rollouts under task_* folders.

Works with trusted savers.Trajectory PKLs. It NEVER opens debug/context PKLs
outside task folders and never changes the source datasets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
import re
from pathlib import Path

import numpy as np
import savers  # noqa: F401 -- needed to unpickle savers.Trajectory

DEFAULT_SOURCES = [
    "/mnt/beegfs/a.cuciniello9/datasets/mixture/cod_epoch_100_no_wrist",
    "/mnt/beegfs/a.cuciniello9/datasets/mixture/cod_epoch_100_wrist",
    "/mnt/beegfs/a.cuciniello9/datasets/mixture/cod_rollout_finetuning/no_wrist",
]
SUCCESS_KEYS = ("object_reached", "object_picked", "object_placed")


def describe(value):
    """Shape/dtype and, for small numeric arrays, values; never dumps images."""
    if value is None:
        return "None"
    if isinstance(value, dict):
        return f"dict keys={sorted(map(str, value.keys()))}"
    try:
        a = np.asarray(value)
    except Exception as exc:
        return f"{type(value).__name__}: {type(exc).__name__}: {exc}"
    result = f"type={type(value).__name__} shape={a.shape} dtype={a.dtype}"
    if a.size <= 10 and a.dtype.kind in "biuf":
        result += f" values={a.reshape(-1).tolist()}"
    return result


def step_at(traj, index):
    if hasattr(traj, "get"):
        try:
            value = traj.get(index, decompress=False)
        except TypeError:
            value = traj.get(index)
    else:
        value = traj[index]
    if isinstance(value, dict):
        return value
    if isinstance(value, tuple) and len(value) == 5:
        obs, reward, done, info, action = value
        return {"obs": obs, "reward": reward, "done": done,
                "info": info, "action": action}
    raise TypeError(f"Unsupported step type: {type(value).__name__}")


def inspect_one(path, root):
    print(f"\n  ----- {path.relative_to(root)} -----")
    with path.open("rb") as f:
        payload = pickle.load(f)  # Only use on trusted PKLs.
    if isinstance(payload, dict):
        print("  Payload keys:", sorted(map(str, payload.keys())))
        traj = payload.get("traj")
    else:
        print("  Payload type:", type(payload).__name__)
        traj = payload
    if traj is None:
        raise ValueError("No 'traj' in payload")
    print("  Trajectory type:", type(traj).__name__, "frames:", len(traj))
    if len(traj) == 0:
        return

    for t in sorted({0, len(traj) // 2, len(traj) - 1}):
        s = step_at(traj, t)
        print(f"  Step {t}: fields={sorted(map(str, s.keys()))}")
        if t == 0:
            obs = s.get("obs")
            if not isinstance(obs, dict):
                print("    obs:", describe(obs))
            else:
                print("    Observation keys and properties:")
                for key in sorted(obs):
                    print(f"      {key}: {describe(obs[key])}")
            print("    info:", describe(s.get("info")))
        print("    action:", describe(s.get("action")))
        obs = s.get("obs")
        if isinstance(obs, dict):
            for key in ("eef_pos", "eef_quat", "EEF_state",
                        "gripper_qpos", "gripper_state", "joint_pos",
                        "joint_positions"):
                if key in obs:
                    print(f"    {key}: {describe(obs[key])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path,
                        help="Source directory (repeatable); defaults to all 3 COD sources")
    parser.add_argument("--samples-per-source", type=int, default=2,
                        help="Number of PKLs to open per source, default 2")
    args = parser.parse_args()
    if args.samples_per_source < 1:
        parser.error("--samples-per-source must be >= 1")
    sources = args.source or [Path(p) for p in DEFAULT_SOURCES]

    for root in sources:
        root = root.expanduser().resolve()
        print("\n" + "=" * 76)
        print("SOURCE:", root)
        print("=" * 76)
        if not root.is_dir():
            print("ERROR: source directory not found")
            continue

        # Only immediate task_XX/traj_NNN.pkl; ignore debug subtrees/context.pkl.
        paths = sorted(
            p for p in root.glob("task_*/*.pkl")
            if p.parent.name.startswith("task_")
            and re.fullmatch(r"traj_\d+", p.stem) is not None
        )
        print("PKLs in task_*/traj_*.pkl:", len(paths))
        print("Task counts:", dict(sorted(Counter(p.parent.name for p in paths).items())))

        json_status = Counter()
        json_keys = None
        for path in paths:
            jpath = path.with_suffix(".json")
            if not jpath.is_file():
                json_status["missing_json"] += 1
                continue
            try:
                with jpath.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                if not isinstance(meta, dict):
                    json_status["non_dict_json"] += 1
                    continue
                if json_keys is None:
                    json_keys = sorted(meta)
                if not all(k in meta for k in SUCCESS_KEYS):
                    json_status["missing_success_fields"] += 1
                elif all(meta[k] == 1 for k in SUCCESS_KEYS):
                    json_status["success_1_1_1"] += 1
                else:
                    json_status["not_success"] += 1
            except Exception as exc:
                json_status[f"invalid_json_{type(exc).__name__}"] += 1
        print("JSON result:", dict(json_status))
        print("Example JSON keys:", json_keys)

        if not paths:
            continue
        if args.samples_per_source == 1:
            picks = [paths[0]]
        else:
            # Spread inspection across trajectory paths rather than adjacent files.
            indices = np.linspace(0, len(paths) - 1,
                                  num=min(args.samples_per_source, len(paths)))
            picks = [paths[int(i)] for i in dict.fromkeys(indices.round().astype(int))]
        for path in picks:
            try:
                inspect_one(path, root)
            except Exception as exc:
                print(f"  ERROR inspecting {path.relative_to(root)}: "
                      f"{type(exc).__name__}: {exc}")

    print("\nFinished. Read-only; no files changed.")
    print("NOTE: shape/dtype cannot prove whether image channels are BGR or RGB.")


if __name__ == "__main__":
    main()
