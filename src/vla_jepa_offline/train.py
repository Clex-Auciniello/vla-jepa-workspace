"""One shared offline CLI for E3/E4; no Flask or ROS.

Example:
  python -m vla_jepa_offline.train --config configs/offline/e3.yaml --preflight-only
  python -m vla_jepa_offline.train --config configs/offline/e3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import yaml

from .batch_builder import OfflineBatchConfig, iter_batches, parse_crop
from .dataset import OfflineEpisode, load_index
from .trainer import create_trainer, latest_checkpoint

stop_requested = False


def _request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print(f"Received signal {signum}: finish current episode, checkpoint, then stop", flush=True)


def _absolute(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _atomic_json(path: Path, data: dict):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _dataset_files(entries):
    from .dataset import _file_list
    for e in entries:
        if e.pkl_path is not None:
            yield e.pkl_path
        else:
            yield from _file_list(e.front_dir)
            yield from _file_list(e.gripper_dir)


def _identity(entries, index_digest, cfg: dict, batch_cfg: OfflineBatchConfig):
    # Manifest AND the file identities prevent silent resume after an edit.
    h = hashlib.sha256()
    for p in _dataset_files(entries):
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
    p = Path(cfg["pretrained_path"]).resolve()
    source = {}
    for fn in ("model.safetensors", "config.json"):
        f = p / fn
        if not f.is_file():
            raise FileNotFoundError(f"Missing pretrained {f}")
        st = f.stat()
        source[fn] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return {
        "format_version": 1,
        "dataset_index_sha256": index_digest,
        "dataset_files_stat_sha256": h.hexdigest(),
        "episode_ids": [e.episode_id for e in entries],
        "pretrained_path": _absolute(cfg["pretrained_path"]),
        "pretrained_stat": source,
        "qwen_lr": float(cfg.get("qwen_learning_rate", 1e-5)),
        "predictor_lr": float(cfg.get("predictor_learning_rate", 1e-5)),
        "seed": int(cfg.get("seed", 1000)),
        "batch": {
            "front_feature": batch_cfg.front_feature,
            "gripper_feature": batch_cfg.gripper_feature,
            "window_size": batch_cfg.window_size,
            "stride": batch_cfg.stride,
            "align_last": batch_cfg.align_last,
            "front_crop": cfg.get("front_crop"),
            "gripper_crop": cfg.get("gripper_crop"),
        },
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument("--preflight-only", action="store_true")
    cli.add_argument("--max-episodes", type=int, default=0,
                     help="For smoke tests; choose a SEPARATE checkpoint_root")
    args = cli.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML dictionary")
    for key in ("dataset_index", "pretrained_path", "checkpoint_root"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Missing YAML setting {key}")
    if args.max_episodes < 0:
        raise ValueError("max-episodes must be >=0")
    entries, index_digest = load_index(cfg["dataset_index"])
    forbidden = set(cfg.get("forbidden_task_ids", []))
    for e in entries:
        if forbidden and e.task_id is None:
            raise ValueError(f"Task ID required to exclude held-out tasks: {e.episode_id}")
        if e.task_id in forbidden:
            raise ValueError(f"Held-out task {e.task_id} found in WM data: {e.episode_id}")
    if args.max_episodes:
        entries = entries[:args.max_episodes]
    batch = OfflineBatchConfig(
        front_feature=cfg.get("front_feature", "observation.images.exterior_1_left"),
        gripper_feature=cfg.get("gripper_feature", "observation.images.exterior_2_left"),
        front_crop=parse_crop(cfg.get("front_crop")),
        gripper_crop=parse_crop(cfg.get("gripper_crop")),
        align_last=bool(cfg.get("align_last", True)),
    )
    if batch.front_feature == batch.gripper_feature:
        raise ValueError("Two input feature names must be distinct")
    processed_frames, processed_windows = 0, 0
    print(f"Index: {cfg['dataset_index']}; episodes={len(entries)}", flush=True)
    if args.preflight_only:
        from vla_jepa_online.training_server.online_batch_builder import _window_starts
        for j, spec in enumerate(entries, 1):
            reader = OfflineEpisode(spec)
            processed_frames += reader.validate(batch.front_crop, batch.gripper_crop)
            processed_windows += len(_window_starts(reader.count, window_size=8, stride=5,
                                                     repeat_last=batch.align_last))
            if j % 20 == 0 or j == len(entries):
                print(f"PREFLIGHT {j}/{len(entries)} videos; {processed_frames} frames; "
                      f"{processed_windows} windows", flush=True)
        print("PREFLIGHT_OK (without any model/optimizer updates)", flush=True)
        return

    root = Path(cfg["checkpoint_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "offline_dataset_identity.json"
    identity = _identity(entries, index_digest, cfg, batch)
    if identity_path.is_file():
        old = json.loads(identity_path.read_text("utf-8"))
        if old != identity:
            raise RuntimeError("Dataset or training configuration changed since checkpoint; use a new output dir")
    else:
        if latest_checkpoint(root) is not None:
            raise RuntimeError("Checkpoint exists but no offline dataset identity: refuse unsafe resume")
        _atomic_json(identity_path, identity)

    trainer = create_trainer(
        pretrained_path=cfg["pretrained_path"],
        checkpoint_root=str(root),
        seed=int(cfg.get("seed", 1000)),
        qwen_lr=float(cfg.get("qwen_learning_rate", 1e-5)),
        predictor_lr=float(cfg.get("predictor_learning_rate", 1e-5)),
    )
    if trainer.processed_videos > len(entries):
        raise RuntimeError("Checkpoint contains more videos than offline dataset")
    if set(trainer.image_keys) != {batch.front_feature, batch.gripper_feature}:
        raise RuntimeError(f"Features disagree with checkpoint: {trainer.image_keys}")
    if trainer.processed_videos and latest_checkpoint(root) is None:
        raise RuntimeError("Invalid resume state")
    print(f"READY resumed_videos={trainer.processed_videos}; global_step={trainer.global_step}; "
          f"trainable_parameters={trainer.num_trainable_parameters}", flush=True)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    metrics = root / "offline_metrics.jsonl"
    # A metrics line is written AFTER the episode has finished. The checkpoint
    # is authoritative: on crash it is safe to replay episodes not checkpointed.
    for index in range(trainer.processed_videos, len(entries)):
        if stop_requested:
            break
        spec = entries[index]
        reader = OfflineEpisode(spec)
        reader.validate(batch.front_crop, batch.gripper_crop)
        started = time.monotonic()
        stats = trainer.train_episode(spec.episode_id, iter_batches(reader, batch))
        row = {
            "event": "episode_complete", "episode_id": spec.episode_id,
            "task_id": spec.task_id, "duration_sec": round(time.monotonic() - started, 3),
            **stats,
        }
        with metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        del reader

    if trainer.processed_videos > 0 and trainer.processed_videos % trainer.cfg.checkpoint_every_videos != 0:
        final_ckpt = trainer.save_checkpoint()
        print(f"SAFE_CHECKPOINT={final_ckpt}", flush=True)
    elif trainer.processed_videos > 0:
        print(f"SAFE_CHECKPOINT={latest_checkpoint(root)}", flush=True)
    print(f"EXIT videos={trainer.processed_videos}/{len(entries)} "
          f"steps={trainer.global_step} interrupted={stop_requested}", flush=True)


if __name__ == "__main__":
    main()
