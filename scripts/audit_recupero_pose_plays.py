#!/usr/bin/env python3
"""Diagnosi NON DISTRUTTIVA del recupero pose in Plays.

Confronta le pose EEF osservate con le pose 8D COMANDATE e con joint_pos.
Le pose comandate sono TARGET, NON misure dell'EEF. Non modifica i PKL.

Uso sul cluster:
  export PYTHONPATH="/mnt/beegfs/a.cuciniello9/vla-jepa/vla-jepa-workspace/src:${PYTHONPATH:-}"
  python audit_recupero_pose_plays.py --limit 10
  python audit_recupero_pose_plays.py --limit 0

Output: summary.json, episodes.csv, jumps.csv, errors.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/plays_trajectories')
DEFAULT_OUTPUT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/plays_pose_recovery_audit')


def percentiles(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {'n': 0, 'median': None, 'p95': None, 'max': None}
    return {
        'n': int(a.size), 'median': float(np.median(a)),
        'p95': float(np.percentile(a, 95)), 'max': float(np.max(a)),
    }


def norm_quats(q, label):
    q = np.asarray(q, dtype=np.float64)
    assert q.ndim == 2 and q.shape[1] == 4, (label, q.shape)
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or np.any(norms < 0.95) or np.any(norms > 1.05):
        raise ValueError(f'{label}: quaternioni invalidi o lontani da norma unitaria')
    return q / norms[:, None]


def validate_arr(x, dims, filename, idx, label):
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.shape != (dims,) or not np.isfinite(y).all():
        raise ValueError(f'{filename} frame {idx}: {label} shape={y.shape} / non finite')
    return y


def summarize_episode(path, obj):
    if not isinstance(obj, dict) or obj.get('source') != 'plays':
        raise ValueError(f'{path.name}: formato Plays non valido')
    traj = obj['traj']
    T = len(traj)
    if T < 2:
        raise ValueError(f'{path.name}: servono almeno due frame')
    pos, quat, joints, cmd_pos, cmd_quat, events, stamps = [], [], [], [], [], [], []
    for i in range(T):
        try:
            step = traj.get(i, decompress=False)
        except TypeError:
            step = traj.get(i)
        obs, info = step['obs'], step['info']
        action8 = validate_arr(step['action'], 8, path.name, i, 'action originale 8D')
        pos.append(validate_arr(obs['eef_pos'], 3, path.name, i, 'eef_pos'))
        quat.append(validate_arr(obs['eef_quat'], 4, path.name, i, 'eef_quat'))
        joints.append(validate_arr(obs['joint_pos'], 6, path.name, i, 'joint_pos'))
        cmd_pos.append(action8[:3])
        cmd_quat.append(action8[3:7])
        events.append(str(info.get('event') or ''))
        stamps.append(info.get('front_stamp_ns'))

    pos = np.stack(pos)
    quat = norm_quats(np.stack(quat), path.name+' eef_quat')
    joints = np.stack(joints)
    cmd_pos = np.stack(cmd_pos)
    cmd_quat = norm_quats(np.stack(cmd_quat), path.name+' command_quat')
    ro = Rotation.from_quat(quat)
    rc = Rotation.from_quat(cmd_quat)

    observed_xyz_mm = 1000*np.linalg.norm(np.diff(pos, axis=0), axis=1)
    commanded_xyz_mm = 1000*np.linalg.norm(np.diff(cmd_pos, axis=0), axis=1)
    joint_delta_rad = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    obs_delta_rot = ro[1:]*ro[:-1].inv()
    cmd_delta_rot = rc[1:]*rc[:-1].inv()
    observed_rot_deg = np.degrees(obs_delta_rot.magnitude())
    commanded_rot_deg = np.degrees(cmd_delta_rot.magnitude())
    static = bool((observed_xyz_mm < 1e-8).all() and (observed_rot_deg < 1e-8).all())
    jump = bool(np.any(observed_xyz_mm > 100))
    status = 'eef_static' if static else ('jump_gt_100mm' if jump else 'healthy')

    pos_error_mm = 1000*np.linalg.norm(pos-cmd_pos, axis=1)
    rot_error_deg = np.degrees((rc*ro.inv()).magnitude())
    xyz_delta_error_mm = 1000*np.linalg.norm(np.diff(pos, axis=0)-np.diff(cmd_pos, axis=0), axis=1)
    rot_delta_error_deg = np.degrees((cmd_delta_rot*obs_delta_rot.inv()).magnitude())
    offset = pos-cmd_pos  # offsets per tool/base model, for diagnostic only

    ix = int(np.argmax(observed_xyz_mm))
    row = {
        'file': path.name, 'play_type': str(obj.get('play_type')), 'frames': T,
        'status': status, 'eef_static': static, 'eef_jump_gt_100mm': jump,
        'joint_step_max_rad': float(joint_delta_rad.max()),
        'joint_step_median_rad': float(np.median(joint_delta_rad)),
        'observed_xyz_step_max_mm': float(observed_xyz_mm.max()),
        'commanded_xyz_step_max_mm': float(commanded_xyz_mm.max()),
        'commanded_xyz_step_p95_mm': float(np.percentile(commanded_xyz_mm, 95)),
        'commanded_rot_step_max_deg': float(commanded_rot_deg.max()),
        'commanded_translation_total_mm': float(commanded_xyz_mm.sum()),
        'eef_command_pos_error_median_mm': float(np.median(pos_error_mm)),
        'eef_command_pos_error_p95_mm': float(np.percentile(pos_error_mm, 95)),
        'eef_command_rot_error_median_deg': float(np.median(rot_error_deg)),
        'eef_command_rot_error_p95_deg': float(np.percentile(rot_error_deg, 95)),
        'delta_xyz_obs_vs_command_median_mm': float(np.median(xyz_delta_error_mm)),
        'delta_xyz_obs_vs_command_p95_mm': float(np.percentile(xyz_delta_error_mm, 95)),
        'delta_rot_obs_vs_command_p95_deg': float(np.percentile(rot_delta_error_deg, 95)),
        'offset_obs_minus_command_x_m': float(np.median(offset[:, 0])),
        'offset_obs_minus_command_y_m': float(np.median(offset[:, 1])),
        'offset_obs_minus_command_z_m': float(np.median(offset[:, 2])),
    }
    detail = None
    if jump:
        ix = int(np.argmax(observed_xyz_mm))
        dt = None
        try:
            dt = (int(stamps[ix+1])-int(stamps[ix]))/1e9
        except (TypeError, ValueError):
            pass
        detail = {
            'file': path.name, 'play_type': str(obj.get('play_type')),
            'from_frame': ix, 'to_frame': ix+1,
            'from_event': events[ix], 'to_event': events[ix+1], 'gap_seconds': dt,
            'observed_xyz_delta_mm': float(observed_xyz_mm[ix]),
            'commanded_xyz_delta_mm': float(commanded_xyz_mm[ix]),
            'joint_delta_norm_rad': float(joint_delta_rad[ix]),
            'observed_rot_delta_deg': float(observed_rot_deg[ix]),
            'commanded_rot_delta_deg': float(commanded_rot_deg[ix]),
            'from_target_error_mm': float(pos_error_mm[ix]),
            'to_target_error_mm': float(pos_error_mm[ix+1]),
        }
    vectors = {
        'pos_error_mm': pos_error_mm,
        'rot_error_deg': rot_error_deg,
        'xyz_delta_error_mm': xyz_delta_error_mm,
        'rot_delta_error_deg': rot_delta_error_deg,
    }
    return row, detail, vectors


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.limit < 0:
        ap.error('--limit deve essere >= 0')
    if args.root.resolve() == args.output.resolve():
        ap.error('--root e --output devono essere diversi')
    try:
        from vla_jepa_offline.dataset import _open_episode
    except Exception as e:
        ap.error(f'Import vla_jepa_offline.dataset._open_episode fallito: {e}. '
                 'Esporta PYTHONPATH=/mnt/beegfs/a.cuciniello9/vla-jepa/vla-jepa-workspace/src:${PYTHONPATH:-}')
    files = sorted(
        [p for p in args.root.glob('P*.pkl') if re.fullmatch(r'P\d{5,}', p.stem)],
        key=lambda p: int(p.stem[1:]),
    )
    if not files:
        ap.error(f'Nessun episodio P*.pkl: {args.root}')
    if args.limit:
        files = files[:args.limit]
    rows, jumps, errors = [], [], []
    vectors_by_group = defaultdict(lambda: defaultdict(list))
    for n, path in enumerate(files, 1):
        try:
            row, jump, vectors = summarize_episode(path, _open_episode(str(path)))
            rows.append(row)
            if jump:
                jumps.append(jump)
            for key, values in vectors.items():
                vectors_by_group[row['status']][key].append(values)
            print(f'[{n}/{len(files)}] {path.name} {row["status"]}: '
                  f'joint_max={row["joint_step_max_rad"]:.4f} rad '
                  f'command_motion={row["commanded_translation_total_mm"]:.1f} mm '
                  f'target-vs-EEF median={row["eef_command_pos_error_median_mm"]:.1f} mm', flush=True)
        except Exception as exc:
            errors.append({'file': path.name, 'error': f'{type(exc).__name__}: {exc}'})
            print(f'ERROR {path.name}: {exc}', file=sys.stderr)

    result = {'root': str(args.root), 'total': len(files), 'ok': len(rows),
              'errors': errors, 'categories': dict(Counter(x['status'] for x in rows)),
              'validation': {}}
    for status, measurements in vectors_by_group.items():
        result['validation'][status] = {
            name: percentiles(np.concatenate(arrays))
            for name, arrays in measurements.items()
        }
    result['static_with_joint_motion'] = sum(
        r['status'] == 'eef_static' and r['joint_step_max_rad'] > 1e-5 for r in rows)
    result['static_with_command_motion'] = sum(
        r['status'] == 'eef_static' and r['commanded_translation_total_mm'] > 5 for r in rows)
    result['static_with_nonzero_command_rotation'] = sum(
        r['status'] == 'eef_static' and r['commanded_rot_step_max_deg'] > 0.1 for r in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'episodes.csv', rows)
    write_csv(args.output/'jumps.csv', jumps)
    write_csv(args.output/'errors.csv', errors)
    (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nSUMMARY:\n'+json.dumps(result, ensure_ascii=False, indent=2))
    print(f'\nFile di audit: {args.output}. PKL originali non modificati.')
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
