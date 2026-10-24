#!/usr/bin/env python3
"""Read-only PKL + RGB preflight for the existing vla_jepa_offline package.

Designed to run on the cluster from the repository root after setting PYTHONPATH
for src/ and the original trusted savers.py module. Never trains or saves weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import cv2
import numpy as np

from vla_jepa_offline.dataset import EpisodeSpec, OfflineEpisode, load_index, _traj_step
from vla_jepa_offline.batch_builder import OfflineBatchConfig, iter_batches, parse_crop


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Invalid RGB preview shape/dtype: {rgb.shape}, {rgb.dtype}")
    ok = cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not write {path}")


def save_raw_interpretations(out: Path, label: str, raw: np.ndarray) -> None:
    # The viewer can compare both conventions. No heuristic can establish the
    # correct channel convention in every image without semantic evidence.
    bgr_assumption = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    rgb_assumption = raw
    save_rgb(out / f"{label}_raw_assume_BGR.png", bgr_assumption)
    save_rgb(out / f"{label}_raw_assume_RGB.png", rgb_assumption)


def summary_pkl(path: Path) -> dict:
    # PKL files MUST be trusted local data. Import savers before unpickling.
    import importlib
    importlib.import_module('savers')
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'PKL top-level type is {type(payload).__name__}, expected dict')
    keys = sorted(map(str, payload.keys()))
    if 'traj' not in payload:
        raise ValueError(f"MISSING 'traj' in {path}; keys={keys}")
    traj = payload['traj']
    obs = _traj_step(traj, 0)
    image_meta = {
        k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
        for k, value in obs.items()
        if isinstance(value, np.ndarray) and (v := value).ndim == 3 and v.shape[-1] == 3
    }
    return {'pkl': str(path), 'top_level_keys': keys,
            'trajectory_type': f'{type(traj).__module__}.{type(traj).__name__}',
            'num_frames': len(traj), 'first_obs_keys': sorted(map(str, obs.keys())),
            'image_arrays': image_meta}


def tensor_rgb(t) -> np.ndarray:
    a = t.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.round(a * 255.0).astype(np.uint8)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument('--pkl', type=Path)
    source.add_argument('--index', type=Path)
    cli.add_argument('--config', type=Path, help='Existing configs/offline/e3.yaml or e4.yaml')
    cli.add_argument('--episode-number', type=int, default=0, help='0-based index row to inspect')
    cli.add_argument('--frame-number', type=int, default=0, help='0-based original episode frame')
    cli.add_argument('--window-number', type=int, default=0, help='0-based training window to preview')
    cli.add_argument('--output-dir', type=Path, required=True)
    args = cli.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(args.config.read_text('utf-8'))
    batch_cfg = OfflineBatchConfig(
        front_feature=cfg.get('front_feature', OfflineBatchConfig.front_feature),
        gripper_feature=cfg.get('gripper_feature', OfflineBatchConfig.gripper_feature),
        front_crop=parse_crop(cfg.get('front_crop')),
        gripper_crop=parse_crop(cfg.get('gripper_crop')),
        align_last=bool(cfg.get('align_last', True)),
    )
    if args.pkl:
        pkl = args.pkl.expanduser().resolve()
        spec = EpisodeSpec(pkl.stem, '(visual check only)', None, pkl, None, None)
    else:
        entries, _ = load_index(args.index)
        spec = entries[args.episode_number]
    result = {'episode_id': spec.episode_id, 'task': spec.task,
              'config': str(args.config) if args.config else None}
    if spec.pkl_path:
        result['pkl_audit'] = summary_pkl(spec.pkl_path)
    episode = OfflineEpisode(spec)
    result['num_frames'] = episode.count
    assert 0 <= args.frame_number < episode.count
    front, wrist = episode.rgb(args.frame_number)
    for label, image in [('front', front), ('gripper', wrist)]:
        save_raw_interpretations(out, label, image)
    iterator = iter_batches(episode, batch_cfg)
    for _ in range(args.window_number + 1):
        batch = next(iterator)
    result['batch_task'] = batch['task']
    result['batch_shapes'] = {}
    result['batch_ranges'] = {}
    for label, key in [('front', batch_cfg.front_feature), ('gripper', batch_cfg.gripper_feature)]:
        tensor = batch[key]
        result['batch_shapes'][label] = list(tensor.shape)
        result['batch_ranges'][label] = [round(float(tensor.min()), 6), round(float(tensor.max()), 6)]
        save_rgb(out / f'{label}_MODEL_BATCH_t0.png', tensor_rgb(tensor[0, 0]))
        save_rgb(out / f'{label}_MODEL_BATCH_tlast.png', tensor_rgb(tensor[0, -1]))
    (out / 'inspection.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', 'utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'PREVIEW_FILES={out}')


if __name__ == '__main__':
    main()
