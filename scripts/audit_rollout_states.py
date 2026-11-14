#!/usr/bin/env python3
"""Read-only audit of successful robot rollouts: EEF/joint coherence and jumps.

By default audits the three COD sources. Use --source NAME=PATH (repeatable)
for other rollout families. Only task_*/traj_*.pkl is scanned: debug
context.pkl and unrelated folders are ignored.

A success is a paired JSON with all object_reached/object_picked/
object_placed equal to 1. No PKL or JSON is changed/copied.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import savers  # noqa: F401, required by pickle for savers.Trajectory

BASE = Path('/mnt/beegfs/a.cuciniello9/datasets')
DEFAULT_SOURCES = {
    'cod_epoch_100_no_wrist': BASE / 'mixture/cod_epoch_100_no_wrist',
    'cod_epoch_100_wrist': BASE / 'mixture/cod_epoch_100_wrist',
    'cod_finetuning_no_wrist': BASE / 'mixture/cod_rollout_finetuning/no_wrist',
}
DEFAULT_OUTPUT = BASE / 'cod_state_audit'
SUCCESS_KEYS = ('object_reached', 'object_picked', 'object_placed')
POS_EPS_M = 1e-6
ROT_EPS_RAD = 1e-5
JOINT_EPS_RAD = 1e-3
MIN_STALE_RUN = 5
JUMP_WARN_M = 0.06  # diagnostic, not an automatic rejection


def get_step(traj, t):
    if hasattr(traj, 'get'):
        try:
            return traj.get(t, decompress=False)
        except TypeError:
            return traj.get(t)
    return traj[t]


def vector(value, size, name, path, t):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size or not np.isfinite(arr).all():
        raise ValueError(f'{path}, step {t}: invalid {name}: {arr.shape}')
    return arr


def rotation_distance(q0, q1):
    """Shortest SO(3) angular distance; quaternion order XYZW; q and -q equivalent."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1) / (
        np.linalg.norm(q0, axis=-1) * np.linalg.norm(q1, axis=-1)
    )
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def longest_run(flags):
    best_start, best_len, run_start, run_len = None, 0, None, 0
    for i, yes in enumerate(flags):
        if yes:
            if not run_len:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_start, run_len = None, 0
    return best_start, best_len


def json_success(path):
    path = path.with_suffix('.json')
    if not path.is_file():
        return False, 'missing_json', None
    try:
        with path.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return False, 'invalid_json', None
    except (OSError, ValueError, TypeError) as exc:
        return False, f'invalid_json:{type(exc).__name__}', None
    if all(obj.get(k) == 1 for k in SUCCESS_KEYS):
        return True, 'success_1_1_1', obj
    return False, 'not_success', obj


def inspect(path):
    with path.open('rb') as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise TypeError('expected payload dict with traj')
    traj = payload['traj']
    T = len(traj)
    if T < 2:
        raise ValueError(f'only {T} frames')

    pos, quat, joint, grip, commands = [], [], [], [], []
    wrist_flags, front_flags = [], []
    for t in range(T):
        step = get_step(traj, t)
        obs = step['obs']
        pos.append(vector(obs['eef_pos'], 3, 'eef_pos', path, t))
        quat.append(vector(obs['eef_quat'], 4, 'eef_quat', path, t))
        joint.append(vector(obs['joint_pos'], 6, 'joint_pos', path, t))
        grip.append(float(vector(obs['gripper_qpos'], 1, 'gripper_qpos', path, t)[0]))
        act = vector(step['action'], 8, 'action', path, t)
        commands.append(float(act[-1]))
        front_flags.append('camera_front_image' in obs)
        wrist_flags.append('camera_gripper_image' in obs)

    pos = np.stack(pos)
    quat = np.stack(quat)
    joint = np.stack(joint)
    grip = np.asarray(grip)
    commands = np.asarray(commands)
    norm = np.linalg.norm(quat, axis=1)
    if not np.allclose(norm, 1.0, atol=1e-3, rtol=0):
        raise ValueError(f'quaternions not unit length; min/max={norm.min():.5f}/{norm.max():.5f}')

    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    drot = rotation_distance(quat[:-1], quat[1:])
    djoint = np.linalg.norm(np.diff(joint, axis=0), axis=1)

    pose_static = (dpos <= POS_EPS_M) & (drot <= ROT_EPS_RAD)
    joint_moving = djoint > JOINT_EPS_RAD
    stale = pose_static & joint_moving
    stale_start, stale_length = longest_run(stale)

    pos_span = float(np.linalg.norm(pos - pos[0], axis=1).max())
    rot_span = float(rotation_distance(quat[:1], quat).max())
    full_frozen = (
        pos_span <= POS_EPS_M and rot_span <= ROT_EPS_RAD
        and int(joint_moving.sum()) >= 3
    )
    if full_frozen:
        status = 'FROZEN_FULL'
    elif stale_length >= MIN_STALE_RUN:
        status = 'PARTIAL_STALE'
    else:
        status = 'OK'

    jump_indices = np.flatnonzero(dpos > JUMP_WARN_M)
    return {
        'status': status,
        'frames': T,
        'transitions': T-1,
        'joint_moving_transitions': int(joint_moving.sum()),
        'eef_static_transitions': int(pose_static.sum()),
        'stale_transitions': int(stale.sum()),
        'stale_max_run': stale_length,
        'stale_run_start': stale_start if stale_start is not None else '',
        'eef_pos_span_m': pos_span,
        'eef_rot_span_rad': rot_span,
        'joint_span_rad': float(np.linalg.norm(joint - joint[0], axis=1).max()),
        'xyz_step_p50_mm': float(np.percentile(dpos, 50)*1000),
        'xyz_step_p95_mm': float(np.percentile(dpos, 95)*1000),
        'xyz_step_p99_mm': float(np.percentile(dpos, 99)*1000),
        'xyz_step_max_mm': float(dpos.max()*1000),
        'rotation_step_p95_deg': float(np.degrees(np.percentile(drot, 95))),
        'xyz_jumps_gt_60mm': len(jump_indices),
        'jump_step_indices': ','.join(map(str, jump_indices)),
        'front_all_frames': all(front_flags),
        'wrist_all_frames': all(wrist_flags),
        'gripper_initial_qpos': float(grip[0]),
        'gripper_final_qpos': float(grip[-1]),
        'gripper_final_command': float(commands[-1]),
        'original_gripper_commands': ','.join(str(x) for x in np.unique(commands)),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', action='append', metavar='NAME=PATH',
                   help='Repeat to replace default COD source selection')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    sources = {}
    if args.source:
        for raw in args.source:
            if '=' not in raw:
                p.error('--source expects NAME=PATH')
            name, source = raw.split('=', 1)
            if not name or not source or name in sources:
                p.error(f'invalid/duplicate --source: {raw}')
            sources[name] = Path(source).expanduser().resolve()
    else:
        sources = DEFAULT_SOURCES

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    eligible_paths, ok_paths, suspect_paths = [], [], []

    for name, root in sources.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths = sorted(root.glob('task_*/traj_*.pkl'))
        if not paths:
            raise RuntimeError(f'no task_*/traj_*.pkl in {root}')
        selection = Counter()
        quality = Counter()
        n_wrist = n_jumps = 0
        print(f'\n[{name}] total task PKLs={len(paths)}', flush=True)
        for idx, path in enumerate(paths, 1):
            success, reason, jmeta = json_success(path)
            selection[reason] += 1
            if not success:
                continue
            rel = path.relative_to(root)
            eligible_paths.append(str(path))
            row = {'source': name, 'relative_path': rel.as_posix(), 'source_pkl': str(path),
                   'aborted_json': jmeta.get('aborted'),
                   'trajectory_complete_json': jmeta.get('trajectory_complete'),
                   'error': ''}
            try:
                data = inspect(path)
                row.update(data)
                status = data['status']
                n_wrist += int(data['wrist_all_frames'])
                n_jumps += int(data['xyz_jumps_gt_60mm'] > 0)
                if status == 'OK':
                    ok_paths.append(str(path))
                else:
                    suspect_paths.append(f'{status}\t{path}')
            except Exception as exc:
                status = 'ERROR'
                row.update({'status': status, 'error': f'{type(exc).__name__}: {exc}'})
                suspect_paths.append(f'ERROR\t{path}\t{exc}')
            quality[status] += 1
            rows.append(row)
            if idx == 1 or idx % 20 == 0 or status != 'OK':
                print(f'  [{idx}/{len(paths)}] {rel}: {status}', flush=True)
        summary[name] = {
            'directory': str(root),
            'total_task_pkls': len(paths),
            'json_selection': dict(selection),
            'success_audited': sum(quality.values()),
            'quality': dict(quality),
            'successful_with_wrist_all_frames': n_wrist,
            'success_with_xyz_jump_gt_60mm': n_jumps,
        }
        print(f'  JSON: {dict(selection)}')
        print(f'  TF:   {dict(quality)}')
        print(f'  Wrist present every frame in {n_wrist}/{sum(quality.values())} successes')
        print(f'  XYZ jumps >60mm in {n_jumps} successful trajectories (diagnostic only)')

    # Ensure stable columns despite errors having fewer properties.
    base_cols = ['source', 'relative_path', 'source_pkl',
                 'aborted_json', 'trajectory_complete_json', 'status', 'error']
    extra_cols = sorted(set().union(*(set(r) for r in rows)) - set(base_cols))
    all_cols = base_cols + extra_cols
    with (out / 'per_trajectory.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps({'parameters': {
        'position_epsilon_m': POS_EPS_M,
        'rotation_epsilon_rad': ROT_EPS_RAD,
        'joint_epsilon_rad': JOINT_EPS_RAD,
        'minimum_stale_run': MIN_STALE_RUN,
        'jump_warning_m': JUMP_WARN_M,
    }, 'sources': summary}, indent=2) + '\n', encoding='utf-8')
    for filename, values in [
        ('json_success_paths.txt', eligible_paths),
        ('tf_ok_paths.txt', ok_paths),
        ('anomalous_paths.txt', suspect_paths),
    ]:
        (out / filename).write_text('\n'.join(values) + ('\n' if values else ''),
                                    encoding='utf-8')
    print('\nReports:', out)
    print(' - summary.json, per_trajectory.csv')
    print(' - json_success_paths.txt, tf_ok_paths.txt, anomalous_paths.txt')
    print('NOTE: nothing was modified or filtered; >60mm is only reported.')


if __name__ == '__main__':
    main()
