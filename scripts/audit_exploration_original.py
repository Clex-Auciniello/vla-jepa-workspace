#!/usr/bin/env python3
"""Audit non distruttivo dei PKL di exploration ORIGINALI.

Usa lo stesso loader del vecchio preprocessore WM. Non modifica né ricrea
alcuna Trajectory. Scrive schema e statistiche in una directory separata.

Esempio:
 python audit_exploration_original.py --limit 10
 python audit_exploration_original.py --limit 0

I pickle vanno caricati solo da sorgenti fidate.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/exploration')
DEFAULT_OUTPUT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/exploration_original_audit')
POS_NAMES = ('eef_pos', 'eef_position', 'ee_pos', 'tcp_pos')
QUAT_NAMES = ('eef_quat', 'eef_quaternion', 'ee_quat', 'tcp_quat')
GRIP_NAMES = ('gripper_qpos',)
IMAGE_NAMES = ('front_camera_image', 'gripper_camera_image',
               'camera_front_image', 'camera_gripper_image')


def summarize(value, depth=0):
    if isinstance(value, np.ndarray):
        return {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        if depth >= 2:
            return {'type': 'dict', 'keys': sorted(map(str, value))[:60]}
        return {str(k): summarize(v, depth + 1) for k, v in list(value.items())[:55]}
    if isinstance(value, (tuple, list)):
        if depth >= 2:
            return {'type': type(value).__name__, 'length': len(value)}
        return {'type': type(value).__name__, 'length': len(value),
                'first': summarize(value[0], depth + 1) if value else None}
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value if len(str(value)) <= 160 else str(value)[:157] + '...'
    return {'type': f'{type(value).__module__}.{type(value).__name__}',
            'attrs': sorted(vars(value).keys())[:45] if hasattr(value, '__dict__') else []}


def get_step(traj, i):
    try:
        return traj.get(i, decompress=False)
    except TypeError:
        return traj.get(i)


def dict_field(containers, names):
    for container_name, item in containers:
        if not isinstance(item, dict):
            continue
        for name in names:
            if name in item:
                return item[name], f'{container_name}.{name}'
        for nested in ('robot_state', 'state', 'proprio', 'robot', 'observation'):
            child = item.get(nested)
            if isinstance(child, dict):
                for name in names:
                    if name in child:
                        return child[name], f'{container_name}.{nested}.{name}'
    return None, ''


def vector(value, dim):
    if value is None:
        return None
    try:
        a = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.shape != (dim,) or not np.isfinite(a).all():
        return None
    return a


def examine(path, loader, examples_left):
    from scipy.spatial.transform import Rotation
    traj = loader(path)
    length = len(traj)
    if length < 1:
        raise ValueError('empty Trajectory')
    fields = Counter()
    images = Counter()
    all_pos, all_quat, all_grip = [], [], []
    qpos_values = []
    motion_labels = Counter()
    step_examples = []
    grip_paths = Counter()
    pos_paths = Counter()
    quat_paths = Counter()
    for t in range(length):
        step = get_step(traj, t)
        if not isinstance(step, dict) or not isinstance(step.get('obs'), dict):
            raise TypeError(f'Frame {t}: expected step dict with obs dict, got {summarize(step)}')
        obs = step['obs']
        info = step.get('info') or {}
        raw_state = getattr(traj, '_raw_state', None)
        if isinstance(raw_state, (tuple, list)) and t < len(raw_state):
            raw = raw_state[t]
        else:
            raw = None
        containers = [('obs', obs), ('_raw_state', raw)]
        p, p_key = dict_field(containers, POS_NAMES)
        q, q_key = dict_field(containers, QUAT_NAMES)
        g, g_key = dict_field(containers, GRIP_NAMES)
        p = vector(p, 3)
        q = vector(q, 4)
        g = vector(g, 1)
        all_pos.append(p)
        all_quat.append(q)
        all_grip.append(g)
        if g is not None:
            qpos_values.append(float(g[0]))
        pos_paths[p_key or '<missing>'] += 1
        quat_paths[q_key or '<missing>'] += 1
        grip_paths[g_key or '<missing>'] += 1
        fields.update(obs.keys())
        for key in IMAGE_NAMES:
            if key in obs:
                im = obs[key]
                if isinstance(im, np.ndarray):
                    images[f'{key}:{im.shape}:{im.dtype}'] += 1
                else:
                    images[f'{key}:{type(im).__name__}'] += 1
        label = info.get('motion_label') if isinstance(info, dict) else None
        motion_labels[str(label)] += 1
        if examples_left and t in {0, length // 2, length - 1}:
            sample = {'index': t, 'obs': summarize(obs),
                      'info': summarize(info),
                      '_raw_state': summarize(raw),
                      'action': summarize(step.get('action')),
                      'recognized_fields': {'pos': p_key, 'quat': q_key, 'grip': g_key},
                      'values': {'pos': p.tolist() if p is not None else None,
                                 'quat': q.tolist() if q is not None else None,
                                 'grip': float(g[0]) if g is not None else None}}
            step_examples.append(sample)
    pos_all = all(x is not None for x in all_pos)
    quat_all = all(x is not None for x in all_quat)
    grip_all = all(x is not None for x in all_grip)
    xyz_p95_mm = None
    xyz_max_mm = None
    rot_p95_deg = None
    rot_max_deg = None
    if pos_all and length >= 2:
        p = np.stack(all_pos)
        jumps = np.linalg.norm(np.diff(p, axis=0), axis=1) * 1000.0
        xyz_p95_mm = float(np.percentile(jumps, 95))
        xyz_max_mm = float(np.max(jumps))
    if quat_all and length >= 2:
        quat = np.stack(all_quat)
        norms = np.linalg.norm(quat, axis=1)
        if np.all(np.abs(norms - 1) < .01):
            r = Rotation.from_quat(quat)
            jumps = np.degrees((r[1:] * r[:-1].inv()).magnitude())
            rot_p95_deg = float(np.percentile(jumps, 95))
            rot_max_deg = float(np.max(jumps))
    grip_min = float(min(qpos_values)) if qpos_values else None
    grip_max = float(max(qpos_values)) if qpos_values else None
    grip_unique = sorted({round(v, 6) for v in qpos_values})
    result = {
        'file': path.name, 'frames': length, 'pos_valid': sum(v is not None for v in all_pos),
        'quat_valid': sum(v is not None for v in all_quat),
        'grip_valid': sum(v is not None for v in all_grip),
        'pos_paths': dict(pos_paths), 'quat_paths': dict(quat_paths),
        'grip_paths': dict(grip_paths), 'obs_keys': dict(fields),
        'image_shapes': dict(images), 'motion_labels': dict(motion_labels),
        'xyz_p95_mm': xyz_p95_mm, 'xyz_max_mm': xyz_max_mm,
        'rot_p95_deg': rot_p95_deg, 'rot_max_deg': rot_max_deg,
        'gripper_min': grip_min, 'gripper_max': grip_max,
        'gripper_unique': grip_unique,
        'gripper_binary_counts_at_0_4': ({'open': int(sum(v < .4 for v in qpos_values)),
                                          'closed': int(sum(v >= .4 for v in qpos_values))}
                                         if qpos_values else None),
    }
    return result, step_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int, default=10, help='0 = tutti gli episodi')
    parser.add_argument('--examples', type=int, default=3, help='Episodi dei quali salvare 3 step dettagliati')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Input inesistente: {root}')
    if output == root or root in output.parents:
        parser.error('La directory di output non puo essere dentro la sorgente')
    if args.limit < 0 or args.examples < 0:
        parser.error('--limit e --examples devono essere >= 0')
    paths = sorted((p for p in root.glob('E*.pkl') if re.fullmatch(r'E\d{5,}', p.stem)),
                   key=lambda p: int(p.stem[1:]))
    if not paths:
        parser.error('Nessun episodio E00000.pkl trovato')
    selected = paths[:args.limit] if args.limit else paths
    try:
        from vla_jepa_online.training_server.online_batch_builder import _open_exploration_pkl
    except ImportError as exc:
        parser.error('Impossibile importare il loader originale. Lancia nell\'ambiente/repository '
                     f'del precedente preprocessore WM: {exc}')
    output.mkdir(parents=True, exist_ok=True)
    records, samples = [], {}
    for n, path in enumerate(selected):
        try:
            record, examples = examine(path, _open_exploration_pkl, n < args.examples)
            record['status'] = 'OK'
            if examples:
                samples[path.name] = examples
            print(f'[{n+1}/{len(selected)}] {path.name}: {record["frames"]} frame | '
                  f'pose={record["pos_valid"]}/{record["quat_valid"]} '
                  f'gripper={record["grip_valid"]} | XYZ p95={record["xyz_p95_mm"]}', flush=True)
        except Exception as exc:
            record = {'file': path.name, 'status': 'ERROR',
                      'error': f'{type(exc).__name__}: {exc}',
                      'traceback': traceback.format_exc(limit=3)}
            print(f'[{n+1}/{len(selected)}] {path.name}: ERROR {record["error"]}', flush=True)
        records.append(record)
    summary = {
        'source': str(root), 'total_discovered': len(paths), 'analyzed': len(selected),
        'ok': sum(r['status'] == 'OK' for r in records),
        'errors': sum(r['status'] == 'ERROR' for r in records),
        'complete_pose': sum(r.get('pos_valid', -1) == r.get('frames') and
                             r.get('quat_valid', -1) == r.get('frames')
                             for r in records if r['status'] == 'OK'),
        'complete_pose_and_gripper': sum(r.get('pos_valid', -1) == r.get('frames') and
                                         r.get('quat_valid', -1) == r.get('frames') and
                                         r.get('grip_valid', -1) == r.get('frames')
                                         for r in records if r['status'] == 'OK'),
        'obs_key_frequency': dict(sum((Counter(r.get('obs_keys', {})) for r in records), Counter())),
        'image_shapes_frequency': dict(sum((Counter(r.get('image_shapes', {})) for r in records), Counter())),
        'motion_labels_frequency': dict(sum((Counter(r.get('motion_labels', {})) for r in records), Counter())),
        'pos_paths_frequency': dict(sum((Counter(r.get('pos_paths', {})) for r in records), Counter())),
        'quat_paths_frequency': dict(sum((Counter(r.get('quat_paths', {})) for r in records), Counter())),
        'grip_paths_frequency': dict(sum((Counter(r.get('grip_paths', {})) for r in records), Counter())),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output / 'schema_samples.json').write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    (output / 'episodes.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
    cols = ('file', 'status', 'frames', 'pos_valid', 'quat_valid', 'grip_valid',
            'xyz_p95_mm', 'xyz_max_mm', 'rot_p95_deg', 'rot_max_deg',
            'gripper_min', 'gripper_max', 'error')
    with (output / 'episodes.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec.get(key, '') for key in cols})
    print('\n=== RIEPILOGO ===')
    print(f'Individuati {len(paths)} episodi; analizzati {len(selected)}; '
          f'OK {summary["ok"]}; ERROR {summary["errors"]}')
    print(f'Pose complete: {summary["complete_pose"]}/{summary["ok"]}')
    print(f'Pose + gripper completi: {summary["complete_pose_and_gripper"]}/{summary["ok"]}')
    print('Campi posizione:', summary['pos_paths_frequency'])
    print('Campi quaternione:', summary['quat_paths_frequency'])
    print('Campi gripper:', summary['grip_paths_frequency'])
    print('Report:', output)


if __name__ == '__main__':
    main()
