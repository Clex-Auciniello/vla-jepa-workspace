#!/usr/bin/env python3
"""Held-out offline PKL evaluation of VLA-JEPA's video-only WM loss.

Subcommands:
  prepare   freeze stratified trajectories in a JSON manifest (CPU only)
  evaluate  run ONE checkpoint on the frozen set (no optimizer, no backward)
  compare   pair losses from separately evaluated checkpoints

Input files must be trusted PKLs containing savers.Trajectory. RGB from robot
cameras was captured as OpenCV BGR. This evaluator deliberately uses the same
crop, cv2.resize, [0,1] scaling, two input features, 8-frame / stride-5
windowing and policy(batch) forward as OnlineVideoTrainer / OnlineBatchBuilder.

Caveat: the offline robot PKLs do not contain Exploration motion_label. They
are evaluated with a CONSTANT pick-and-place task prompt. This is a valid
paired OOD test, not a measurement of training-prompt-matched Exploration loss.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import random
import statistics
import sys
import zlib
from collections import defaultdict
from pathlib import Path

IMAGE_SHAPE = (224, 224)
WINDOW_SIZE = 8
STRIDE = 5
FRONT_FEATURE = "observation.images.exterior_1_left"
GRIPPER_FEATURE = "observation.images.exterior_2_left"
DEFAULT_PROMPT = "Pick up the object and place it in the target bin."
FRONT_KEYS = ("front_camera_image", "camera_front_image", "front_rgb")
GRIPPER_KEYS = ("gripper_camera_image", "camera_gripper_image", "gripper_rgb")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_manifest(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("episodes"), list):
        raise ValueError(f"Manifest non valido: {path}")
    if (data.get("window_size"), data.get("stride")) != (WINDOW_SIZE, STRIDE):
        raise ValueError("Finestratura manifest diversa dal training")
    if not data.get("episodes"):
        raise ValueError("Nessun episodio nel manifest")
    return data


def prepare(args):
    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest = args.validation_manifest.resolve()
    if manifest.exists() and not args.overwrite_manifest:
        raise FileExistsError(
            f"Manifest gia' presente: {manifest}. Non lo sostituisco senza --overwrite-manifest."
        )
    chosen = []
    rng = random.Random(args.seed)
    tasks = sorted(root.glob("task_[0-9][0-9]"))
    if len(tasks) != 16:
        raise ValueError(f"Attesi 16 task_00..task_15, trovati {len(tasks)} in {root}")
    for idx in range(16):
        task_dir = root / f"task_{idx:02d}"
        episodes = sorted(task_dir.glob("traj*.pkl"))
        if len(episodes) < args.episodes_per_task:
            raise ValueError(f"{task_dir}: solo {len(episodes)} PKL")
        selected = sorted(rng.sample(episodes, args.episodes_per_task))
        chosen.extend({"task_id": idx, "path": str(path)} for path in selected)
    data = {
        "schema_version": 1,
        "dataset_root": str(root),
        "description": "Offline robot pick-and-place, not used in either WM checkpoint",
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_windows_per_episode": args.max_windows_per_episode,
        "prompt": args.prompt,
        "image_encoding": "OpenCV BGR uint8 stored in savers.Trajectory",
        "preprocessing": {
            "front_crop_margins_ltrb": [140, 0, 90, 10],
            "gripper_crop": None,
            "resize_wh": list(IMAGE_SHAPE),
            "pixel_range": "float32 [0,1]",
        },
        "episodes": chosen,
    }
    _write_json(manifest, data)
    print(f"Manifest: {manifest} | task={len(tasks)} | PKL={len(chosen)}", flush=True)
    for e in chosen:
        print(f"  task_{e['task_id']:02d}: {e['path']}")


def _load_traj(path: Path):
    # CRITICAL: importing 'savers' makes pickle's Trajectory class resolvable.
    # Only load trusted university-produced PKL data.
    import savers  # noqa: F401
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "traj" not in payload:
        raise ValueError(f"{path}: expected {{'traj': savers.Trajectory, ...}}")
    traj = payload["traj"]
    if not hasattr(traj, "get") or not hasattr(traj, "__len__"):
        raise ValueError(f"{path}: traj not compatible with savers.Trajectory")
    return traj


def _pick_image(obs, candidates, name, path):
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"{path}: no {name} image. Available observation keys: {list(obs)}")


def _obs_at(traj, index, path):
    # 'decompress' is supported by the dataset_collector savers module, but
    # some historical variants only support Trajectory.get(index).
    try:
        item = traj.get(index, decompress=False)
    except TypeError:
        item = traj.get(index)
    if "obs" not in item:
        raise KeyError(f"{path} frame={index}: missing 'obs'")
    return item["obs"]


def _preprocess_image(img, *, front, path):
    import cv2
    import numpy as np
    import torch
    array = np.asarray(img)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 HWC BGR; got {array.shape} {array.dtype}")
    # These robot PKLs were saved via cv_bridge desired_encoding='bgr8'.
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    if front:
        h, w = rgb.shape[:2]
        if w <= 140 + 90 or h <= 10:
            raise ValueError(f"{path}: front crop does not fit {w}x{h}")
        rgb = rgb[0:h - 10, 140:w - 90]
    rgb = cv2.resize(rgb, IMAGE_SHAPE, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous().float().div_(255.0)


def _window_starts(num_frames, max_windows):
    if num_frames < WINDOW_SIZE:
        raise ValueError(f"Episodio troppo corto: {num_frames} frames")
    all_starts = list(range(0, num_frames - WINDOW_SIZE + 1, STRIDE))
    if max_windows <= 0 or len(all_starts) <= max_windows:
        return all_starts
    if max_windows == 1:
        return [all_starts[len(all_starts) // 2]]
    # Spread deterministically across the whole episode; no random evaluation
    # subsampling, and no dependence on the checkpoint being evaluated.
    indices = [(i * (len(all_starts) - 1)) // (max_windows - 1) for i in range(max_windows)]
    return [all_starts[i] for i in indices]


def _make_batch(traj, start, episode_path, prompt):
    import torch
    fronts, grippers = [], []
    for i in range(start, start + WINDOW_SIZE):
        obs = _obs_at(traj, i, episode_path)
        fronts.append(_preprocess_image(
            _pick_image(obs, FRONT_KEYS, "front", episode_path), front=True, path=episode_path
        ))
        grippers.append(_preprocess_image(
            _pick_image(obs, GRIPPER_KEYS, "gripper", episode_path), front=False, path=episode_path
        ))
    return {
        FRONT_FEATURE: torch.stack(fronts).unsqueeze(0),
        GRIPPER_FEATURE: torch.stack(grippers).unsqueeze(0),
        "task": [prompt],
    }


def _load_policy(checkpoint):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if not isinstance(config, VLAJEPAConfig):
        raise TypeError(f"Checkpoint is not VLA-JEPA: {type(config)}")
    if not config.enable_world_model or config.num_video_frames != WINDOW_SIZE:
        raise ValueError("Checkpoint without matching World Model / video frame count")
    if config.jepa_tubelet_size != 2 or len(config.image_features) != 2:
        raise ValueError("Checkpoint encoder or camera count does not match online trainer")
    for key in (FRONT_FEATURE, GRIPPER_FEATURE):
        if key not in config.image_features or tuple(config.image_features[key].shape) != (3, 224, 224):
            raise ValueError(f"Checkpoint missing correct image feature {key}")
    if not hasattr(config, "causal_world_model_context"):
        raise RuntimeError("Current LeRobot lacks 'causal_world_model_context'")
    # The online trainer forcibly sets this BEFORE loading either policy.
    config.causal_world_model_context = True
    config.device = "cuda"
    policy = VLAJEPAPolicy.from_pretrained(str(checkpoint), config=config, strict=False)
    policy.to(torch.device("cuda:0"))
    policy.eval()
    # Only video-only WM forward; action head remains unused.
    return policy


def _seed_window(seed, path, start):
    import torch
    # If WM internals still sample randomness in eval mode, use identical
    # deterministic RNG per window for both checkpoints.
    deterministic = (seed + zlib.crc32(f"{path}:{start}".encode("utf-8"))) % (2**32)
    torch.manual_seed(deterministic)
    torch.cuda.manual_seed_all(deterministic)


def evaluate(args):
    import numpy as np
    import torch

    manifest = _read_manifest(args.validation_manifest)
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(ckpt)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expected an NVIDIA GPU supporting BF16 (e.g. A100)")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"windows_{args.name}.csv"
    if results_path.exists() and not args.resume:
        raise FileExistsError(f"Existing output {results_path}; use --resume to continue")
    completed = _read_results(results_path) if results_path.exists() else {}
    print(f"Loading policy {args.name} from {ckpt}", flush=True)
    policy = _load_policy(ckpt)
    print(f"Loaded. device=cuda, eval=True, windows={len(manifest['episodes'])} episodes", flush=True)

    n_written = 0
    with results_path.open("a" if results_path.exists() else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "episode", "start_index", "wm_loss"])
        if stream.tell() == 0:
            writer.writeheader()
            stream.flush()
        for eidx, episode in enumerate(manifest["episodes"], start=1):
            path = Path(episode["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            traj = _load_traj(path)
            starts = _window_starts(len(traj), manifest["max_windows_per_episode"])
            if not starts:
                raise RuntimeError(f"No evaluation windows: {path}")
            last_value = None
            for start in starts:
                key = (int(episode["task_id"]), str(path), start)
                if key in completed:
                    last_value = completed[key]
                    continue
                batch = _make_batch(traj, start, str(path), manifest["prompt"])
                _seed_window(args.seed, path, start)
                # Identical policy(batch) used in OnlineVideoTrainer.train_episode;
                # evaluation mode disables dropout; no backward or optimizer.
                with torch.no_grad():
                    loss, logs = policy(batch)
                value = float(logs["wm_loss"])
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite WM loss {args.name} {path} @ {start}: {value}")
                writer.writerow({"task_id": key[0], "episode": key[1],
                                 "start_index": start, "wm_loss": value})
                # Persist each window, so interrupted SLURM runs can resume.
                stream.flush()
                completed[key] = value
                n_written += 1
                last_value = value
                del batch, loss, logs
            print(f"[{args.name} {eidx}/{len(manifest['episodes'])}] {path.name} "
                  f"windows={len(starts)} last_loss={last_value:.6f}", flush=True)
            del traj
    print(f"Saved {len(completed)} window losses ({n_written} new): {results_path}", flush=True)
    del policy
    gc.collect()
    torch.cuda.empty_cache()


def _read_results(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No values in {path}")
    out = {}
    for row in rows:
        key = (int(row["task_id"]), row["episode"], int(row["start_index"]))
        if key in out:
            raise ValueError(f"Duplicate window {key}")
        out[key] = float(row["wm_loss"])
    return out


def _mean_sd(values):
    return (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


def compare(args):
    out = args.output_dir.resolve()
    a = _read_results(out / "windows_droid.csv")
    b = _read_results(out / "windows_exploration160.csv")
    if a.keys() != b.keys():
        raise ValueError(f"Paired sets differ: only_droid={len(a.keys()-b.keys())}, only_exploration={len(b.keys()-a.keys())}")
    # Fail closed: do not summarize an interrupted SLURM run as if complete.
    manifest = _read_manifest(args.validation_manifest)
    expected = set()
    for item in manifest["episodes"]:
        path = Path(item["path"])
        traj = _load_traj(path)
        for start in _window_starts(len(traj), manifest["max_windows_per_episode"]):
            expected.add((int(item["task_id"]), str(path), start))
        del traj
    if expected != set(a):
        raise ValueError(
            f"Evaluation incomplete/mismatched: expected={len(expected)}, got={len(a)}, "
            f"missing={len(expected-set(a))}, unexpected={len(set(a)-expected)}"
        )
    per_episode = defaultdict(list)
    per_task = defaultdict(list)
    windows = []
    for key in sorted(a):
        task_id, episode, start = key
        droid, explor = a[key], b[key]
        delta = droid - explor
        windows.append({"task_id": task_id, "episode": episode,
                        "start_index": start, "droid_wm_loss": droid,
                        "exploration160_wm_loss": explor, "improvement_droid_minus_exploration": delta})
        per_episode[(task_id, episode)].append((droid, explor))
    episodes = []
    for (task_id, path), window_pairs in sorted(per_episode.items()):
        avg_a = statistics.mean(x for x, _ in window_pairs)
        avg_b = statistics.mean(y for _, y in window_pairs)
        row = {"task_id": task_id, "episode": path, "n_windows": len(window_pairs),
               "droid_mean_wm_loss": avg_a, "exploration160_mean_wm_loss": avg_b,
               "improvement_droid_minus_exploration": avg_a - avg_b}
        episodes.append(row)
        per_task[task_id].append(row)
    tasks = []
    for task_id, ep_rows in sorted(per_task.items()):
        am, asd = _mean_sd([x["droid_mean_wm_loss"] for x in ep_rows])
        bm, bsd = _mean_sd([x["exploration160_mean_wm_loss"] for x in ep_rows])
        dm, dsd = _mean_sd([x["improvement_droid_minus_exploration"] for x in ep_rows])
        tasks.append({"task_id": task_id, "n_episodes": len(ep_rows),
                      "droid_mean": am, "droid_std": asd, "exploration160_mean": bm,
                      "exploration160_std": bsd, "delta_mean": dm, "delta_std": dsd})
    for file, records in (("paired_windows.csv", windows), ("paired_episodes.csv", episodes),
                          ("paired_tasks.csv", tasks)):
        with (out / file).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    # Mean across EPISODES: do not count longer videos more strongly.
    av, av_sd = _mean_sd([e["droid_mean_wm_loss"] for e in episodes])
    bv, bv_sd = _mean_sd([e["exploration160_mean_wm_loss"] for e in episodes])
    dv, dv_sd = _mean_sd([e["improvement_droid_minus_exploration"] for e in episodes])
    summary = {"validation": "offline_pick_place", "checkpoint_comparison": "DROID vs Exploration-160",
               "episodes": len(episodes), "windows": len(windows),
               "droid_episode_mean": av, "droid_episode_std": av_sd,
               "exploration160_episode_mean": bv, "exploration160_episode_std": bv_sd,
               "delta_episode_mean": dv, "delta_episode_std": dv_sd,
               "delta_definition": "DROID loss - Exploration-160 loss; positive = better Exploration-160",
               "prompt": _read_manifest(args.validation_manifest)["prompt"],
               "warning": "OOD offline pick-and-place prompt, not matched to Exploration motion labels."
               }
    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Create a frozen stratified manifest; no GPU needed")
    prep.add_argument("--dataset-root", type=Path, required=True)
    prep.add_argument("--validation-manifest", type=Path, required=True)
    prep.add_argument("--episodes-per-task", type=int, default=2)
    prep.add_argument("--max-windows-per-episode", type=int, default=4)
    prep.add_argument("--prompt", default=DEFAULT_PROMPT)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--overwrite-manifest", action="store_true")
    prep.set_defaults(fn=prepare)

    ev = sub.add_parser("evaluate", help="Evaluate a single checkpoint on the GPU")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--name", choices=("droid", "exploration160"), required=True)
    ev.add_argument("--validation-manifest", type=Path, required=True)
    ev.add_argument("--output-dir", type=Path, required=True)
    ev.add_argument("--seed", type=int, default=1000)
    ev.add_argument("--resume", action="store_true", help="Append only missing window evaluations")
    ev.set_defaults(fn=evaluate)

    cmp_ = sub.add_parser("compare", help="Compare paired CSV files; no GPU needed")
    cmp_.add_argument("--validation-manifest", type=Path, required=True)
    cmp_.add_argument("--output-dir", type=Path, required=True)
    cmp_.set_defaults(fn=compare)
    args = p.parse_args()
    if getattr(args, "episodes_per_task", 1) < 1 or getattr(args, "max_windows_per_episode", 1) < 1:
        p.error("episodes-per-task and max-windows-per-episode must be positive")
    args.fn(args)


if __name__ == "__main__":
    main()
