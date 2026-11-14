#!/usr/bin/env python3
"""VLA-JEPA *_clean -> canonical per-task UR5e / VLA-JEPA trajectory PKLs.

Inputs: the 187 manually curated and (where needed) frame-0-trimmed PKLs in
       vla_jepa_rollouts_clean/{E1,E1_onepass,E2,E2_onepass,E3,E4}/task_*/.
Outputs: separate vla_jepa_preprocessed/task_XX/traj_NNN.pkl + JSON, manifest.csv.

Observed action, t < T-1:
  a[:3]  = (eef_pos[t+1] - eef_pos[t]) / 0.05
  a[3:6] = Euler_xyz(R[t+1] @ R[t]^-1) / 0.05
  a[6]   = float(gripper_qpos[t+1] >= 0.4) / 0.05
Terminal action at T-1: np.zeros(7, float32), stationary EEF and OPEN gripper.
The last observed gripper state is NEVER changed or invented.

RGB: raw BGR (as stored by AIControllerNode) -> RGB uint8 HWC, cv2.INTER_LINEAR;
front crop margins T,B,L,R=(0,10,140,90) then resize 224x224; gripper resize
only. Other image/crop/depth keys are discarded. Original PKLs stay untouched.

Pilot recommended: --tasks 0 1 --limit-per-task 1 --destination <pilot> --save-preview
Only unpickle files from trusted origins; pickle.load is unsafe on untrusted data.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import savers  # noqa: F401; needed to deserialize savers.Trajectory

BASE = Path('/mnt/beegfs/a.cuciniello9/datasets/mixture')
DEFAULT_SOURCE = BASE / 'vla_jepa_rollouts_clean'
DEFAULT_DEST = BASE / 'vla_jepa_preprocessed'
GROUPS = ('E1', 'E1_onepass', 'E2', 'E2_onepass', 'E3', 'E4')
EXPECTED = {'E1': 26, 'E1_onepass': 29, 'E2': 34,
            'E2_onepass': 31, 'E3': 25, 'E4': 42}
SUCCESS_KEYS = ('object_reached', 'object_picked', 'object_placed')
FRONT = 'camera_front_image'
GRIPPER_IMAGE = 'camera_gripper_image'
MARGINS = (0, 10, 140, 90)  # top, bottom, left, right
SCALE = 0.05
GRIPPER_THRESHOLD = 0.4
MAX_XYZ_M = 0.06
MAX_ROT_DEG = 30.
MAX_ROUNDTRIP_XYZ_MM = 0.01
MAX_ROUNDTRIP_ROT_DEG = 0.001


def step_at(traj, t):
    try:
        return traj.get(t, decompress=False)
    except TypeError:
        return traj.get(t)


def vec(value, size, key, src, t):
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size != size or not np.isfinite(out).all():
        raise ValueError(f'{src} t={t}: {key} expected finite {size}D, got {out.shape}')
    return out


def native_steps(traj, src):
    raw = getattr(traj, '_data', None)
    if not isinstance(raw, list) or len(raw) != len(traj) or len(raw) < 2:
        raise ValueError(f'{src}: invalid Trajectory._data or fewer than 2 frames')
    steps = []
    for t, entry in enumerate(raw):
        if not isinstance(entry, tuple) or len(entry) != 5 or not isinstance(entry[0], dict):
            raise TypeError(f'{src} t={t}: expected (obs,reward,done,info,action) tuple')
        steps.append(list(entry))
    return steps


def sha_array(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def check_json(src):
    path = src.with_suffix('.json')
    with path.open(encoding='utf-8') as f:
        meta = json.load(f)
    if not isinstance(meta, dict) or not all(meta.get(k) == 1 for k in SUCCESS_KEYS):
        raise ValueError(f'{path}: JSON not successful 1/1/1')
    return path, meta


def read_bgr(obs, key, src, t):
    if key not in obs:
        raise KeyError(f'{src} t={t}: missing camera {key}')
    a = np.asarray(obs[key])
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f'{src} t={t}: {key}: expected HWC BGR uint8; got {a.shape} {a.dtype}')
    return a


def rgb_224(bgr, front, src, t):
    if front:
        top, bottom, left, right = MARGINS
        h, w = bgr.shape[:2]
        if top + bottom >= h or left + right >= w:
            raise ValueError(f'{src} t={t}: invalid front crop {MARGINS} for {(h,w)}')
        bgr = bgr[top:h-bottom, left:w-right]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise AssertionError('Unexpected output image shape/dtype')
    return np.ascontiguousarray(rgb)


def gripper_audit(qpos, raw_command, src):
    binary = (qpos >= GRIPPER_THRESHOLD).astype(np.uint8)
    transitions = np.flatnonzero(np.diff(binary) != 0) + 1
    if (binary[0] != 0 or binary[-1] != 1 or len(transitions) != 1
            or binary[transitions[0]] != 1):
        raise ValueError(f'{src}: expected observed 0->1 after frame-0 clean; '
                         f'initial/final={binary[0]}/{binary[-1]} '
                         f'transitions={transitions.tolist()}')
    is_open = np.isclose(raw_command, 0., atol=0.5, rtol=0)
    is_close = np.isclose(raw_command, 255., atol=0.5, rtol=0)
    if not np.all(is_open | is_close):
        raise ValueError(f'{src}: unexpected original gripper commands')
    cmd = is_close.astype(np.uint8)
    close_times = np.flatnonzero(np.diff(cmd) == 1) + 1
    open_times = np.flatnonzero(np.diff(cmd) == -1) + 1
    if (cmd[0] != 0 or len(close_times) != 1 or len(open_times) > 1
            or (len(open_times) == 1 and open_times[0] <= close_times[0])):
        raise ValueError(f'{src}: unexpected original command transitions: '
                         f'CLOSE={close_times.tolist()}, OPEN={open_times.tolist()}')
    lag = int(transitions[0] - close_times[0])
    if lag < 0:
        raise ValueError(f'{src}: gripper observed closed before CLOSE command')
    terminal_original = 'CLOSE' if is_close[-1] else 'OPEN'
    return binary, {
        'observed_close_t': int(transitions[0]),
        'original_close_command_t': int(close_times[0]),
        'close_feedback_lag_frames': lag,
        'original_terminal_gripper_command': terminal_original,
        'original_missing_terminal_open': terminal_original == 'CLOSE',
    }


def transform(payload, src, source_root, out_rel, clean_trimmed):
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise ValueError(f'{src}: expected dict with traj')
    traj = payload['traj']
    steps = native_steps(traj, src)
    T = len(steps)
    pos = np.stack([vec(step[0]['eef_pos'], 3, 'eef_pos', src, t)
                    for t, step in enumerate(steps)])
    quat = np.stack([vec(step[0]['eef_quat'], 4, 'eef_quat', src, t)
                     for t, step in enumerate(steps)])
    qpos = np.array([vec(step[0]['gripper_qpos'], 1, 'gripper_qpos', src, t)[0]
                     for t, step in enumerate(steps)])
    original_actions = np.stack([vec(step[4], 8, 'original absolute target 8D', src, t)
                                 for t, step in enumerate(steps)])
    if not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-3, rtol=0):
        raise ValueError(f'{src}: nonunit quaternions')
    closed, grip_meta = gripper_audit(qpos, original_actions[:, 7], src)
    rotations = Rotation.from_quat(quat)  # XYZW
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Gimbal lock detected')
        euler = rotations.as_euler('xyz')
        rel_rot = rotations[1:] * rotations[:-1].inv()  # R_next @ R_current^-1
        delta_rpy = rel_rot.as_euler('xyz')
    delta_xyz = np.diff(pos, axis=0)
    xyz_step_m = np.linalg.norm(delta_xyz, axis=1)
    rot_step_deg = np.degrees(rel_rot.magnitude())
    if np.any(xyz_step_m > MAX_XYZ_M + 1e-10):
        bad = int(np.flatnonzero(xyz_step_m > MAX_XYZ_M + 1e-10)[0])
        raise ValueError(f'{src}: XYZ jump t={bad}, {xyz_step_m[bad]*1000:.3f}mm')
    if np.any(rot_step_deg > MAX_ROT_DEG + 1e-10):
        bad = int(np.flatnonzero(rot_step_deg > MAX_ROT_DEG + 1e-10)[0])
        raise ValueError(f'{src}: rotation jump t={bad}, {rot_step_deg[bad]:.3f}deg')

    actions = np.zeros((T, 7), dtype=np.float32)
    actions[:-1, :3] = delta_xyz / SCALE
    actions[:-1, 3:6] = delta_rpy / SCALE
    actions[:-1, 6] = closed[1:].astype(np.float32) / SCALE
    actions[-1, :] = 0.0  # synthetic terminal stationary+OPEN label

    rebuild_pos = pos[:-1] + actions[:-1, :3].astype(np.float64) * SCALE
    rebuild_rot = (Rotation.from_euler('xyz', actions[:-1, 3:6].astype(np.float64) * SCALE)
                   * rotations[:-1])
    xyz_error_mm = float(np.max(np.linalg.norm(rebuild_pos - pos[1:], axis=1)) * 1000.)
    rot_error_deg = float(np.max(np.degrees((rebuild_rot.inv() * rotations[1:]).magnitude())))
    if xyz_error_mm > MAX_ROUNDTRIP_XYZ_MM or rot_error_deg > MAX_ROUNDTRIP_ROT_DEG:
        raise ValueError(f'{src}: float32 roundtrip error XYZ={xyz_error_mm:.8f} mm, '
                         f'rotation={rot_error_deg:.8f} deg')

    signatures = {}
    preview = []
    sample_steps = {0, T // 2, T - 1}
    for t, step in enumerate(steps):
        obs = step[0]
        front = rgb_224(read_bgr(obs, FRONT, src, t), True, src, t)
        wrist = rgb_224(read_bgr(obs, GRIPPER_IMAGE, src, t), False, src, t)
        eef_state = np.r_[pos[t], euler[t]].astype(np.float32)
        joint_pos = vec(obs['joint_pos'], 6, 'joint_pos', src, t).astype(np.float32)
        joint_vel = vec(obs['joint_vel'], 6, 'joint_vel', src, t).astype(np.float32)
        for key in list(obs):
            if 'image' in key.lower() or 'depth' in key.lower():
                del obs[key]
        obs[FRONT] = front
        obs[GRIPPER_IMAGE] = wrist
        obs['EEF_state'] = eef_state
        obs['gripper_state'] = np.asarray([qpos[t], float(closed[t])], dtype=np.float32)
        obs['joint_positions'] = joint_pos
        obs['joint_velocities'] = joint_vel
        obs['action_world'] = np.r_[eef_state, float(closed[t])].astype(np.float32)
        step[4] = actions[t].copy()
        if t in sample_steps:
            signatures[t, FRONT] = sha_array(front)
            signatures[t, GRIPPER_IMAGE] = sha_array(wrist)
            preview.append((t, front.copy(), wrist.copy()))
    traj._data[:] = [tuple(step) for step in steps]
    group = src.relative_to(source_root).parts[0]
    payload['source'] = f'vla_jepa_rollouts_clean/{group}'
    payload['source_file'] = src.relative_to(source_root).as_posix()
    payload['source_traj_number'] = payload.get('traj_number')
    payload['task_id'] = int(out_rel.parent.name.split('_')[1])
    payload['traj_number'] = int(out_rel.stem.split('_')[1])
    payload['merge_preprocessing'] = {
        'version': 'vla_jepa_rollouts_v1',
        'original_group': group,
        'source_clean_relative_path': src.relative_to(source_root).as_posix(),
        'clean_removed_initial_frame': bool(clean_trimmed),
        'action_scale_factor': SCALE,
        'action_alignment': 'pose and gripper at t+1, for t < T-1',
        'terminal_action': 'synthetic 7D zeros, stationary EEF and OPEN gripper',
        'terminal_label_not_observed': True,
        'original_terminal_gripper_command': grip_meta['original_terminal_gripper_command'],
        'original_missing_terminal_open': grip_meta['original_missing_terminal_open'],
        'rotation': 'R_next @ R_current^-1, scipy Euler xyz',
        'original_action': '8D absolute MoveIt target, not used for reconstructed delta labels',
        'source_raw_image_encoding': 'BGR uint8 HWC (AIControllerNode)',
        'output_image_encoding': 'RGB uint8 HWC (224,224,3)',
        'front_crop_TBLR': list(MARGINS),
        'interpolation': 'cv2.INTER_LINEAR',
        'gripper_threshold': GRIPPER_THRESHOLD,
        'gripper_state_format': '[physical qpos, binary closed]'
    }
    return ({'frames': T, 'transitions': T-1,
             'xyz_max_mm': float(xyz_step_m.max() * 1000.),
             'rotation_max_deg': float(rot_step_deg.max()),
             'roundtrip_xyz_max_mm': xyz_error_mm,
             'roundtrip_rotation_max_deg': rot_error_deg,
             **grip_meta}, signatures, preview)


def verify_saved(path, n_frames, signatures):
    with path.open('rb') as f:
        payload = pickle.load(f)
    steps = native_steps(payload['traj'], path)
    if len(steps) != n_frames:
        raise AssertionError(f'{path}: saved trajectory length mismatch')
    states = []
    for t, (obs, reward, done, info, action) in enumerate(steps):
        action = np.asarray(action)
        if action.shape != (7,) or action.dtype != np.float32 or not np.isfinite(action).all():
            raise AssertionError(f'{path} t={t}: action not float32[7]')
        if t < n_frames - 1 and action[6] not in (0., 20.):
            raise AssertionError(f'{path} t={t}: gripper label not 0 or 20')
        image_keys = sorted(k for k in obs if 'image' in k.lower() or 'depth' in k.lower())
        if image_keys != sorted([FRONT, GRIPPER_IMAGE]):
            raise AssertionError(f'{path} t={t}: unexpected image keys {image_keys}')
        for key in (FRONT, GRIPPER_IMAGE):
            arr = np.asarray(obs[key])
            if arr.shape != (224, 224, 3) or arr.dtype != np.uint8:
                raise AssertionError(f'{path} t={t}: invalid {key} shape/dtype')
            if (t, key) in signatures and sha_array(arr) != signatures[t, key]:
                raise AssertionError(f'{path} t={t}: {key} changed after serialization')
        for key, length in (('EEF_state', 6), ('gripper_state', 2),
                            ('joint_positions', 6), ('joint_velocities', 6),
                            ('action_world', 7)):
            arr = np.asarray(obs[key])
            if arr.shape != (length,) or arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise AssertionError(f'{path} t={t}: invalid {key}')
        states.append(int(obs['gripper_state'][1]))
        if t in (0, n_frames // 2, n_frames - 1):
            from_get = step_at(payload['traj'], t)
            if not np.array_equal(np.asarray(from_get['action']), action):
                raise AssertionError(f'{path} t={t}: Trajectory.get action inconsistent')
    assert np.array_equal(np.asarray(steps[-1][4]), np.zeros(7, dtype=np.float32))
    observed_next = np.asarray(states[1:], dtype=np.float32) / SCALE
    label_gripper = np.asarray([step[4][6] for step in steps[:-1]], dtype=np.float32)
    if not np.array_equal(observed_next, label_gripper):
        raise AssertionError(f'{path}: wrong t -> t+1 gripper alignment')
    del payload
    gc.collect()


def atomic_pickle(payload, path, frames, signatures):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent,
                                         prefix=f'.{path.stem}.', suffix='.tmp',
                                         delete=False) as f:
            tmp = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        verify_saved(tmp, frames, signatures)
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def atomic_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=dst.parent,
                                         prefix=f'.{dst.name}.', suffix='.tmp',
                                         delete=False) as f:
            tmp = Path(f.name)
            with src.open('rb') as original:
                shutil.copyfileobj(original, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def atomic_csv(path, rows):
    if not rows:
        raise ValueError('No rows for manifest')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8',
                                         dir=path.parent, prefix='.manifest.',
                                         suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def read_clean_manifest(root):
    path = root / 'manifest.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Expected clean selection manifest: {path}')
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        rel = row['destination_relative_path']
        if rel in result:
            raise ValueError(f'Duplicate clean manifest path {rel}')
        trimmed = row['removed_initial_frame']
        if trimmed not in ('0', '1'):
            raise ValueError(f'Unexpected removed_initial_frame={trimmed} in {rel}')
        result[rel] = bool(int(trimmed))
    return result


def gather(root, groups, tasks, limit, manifest, full):
    files = []
    counts = Counter()
    for group in groups:
        base = root / group
        if not base.is_dir():
            raise FileNotFoundError(base)
        pkls = sorted(base.glob('task_*/traj_*.pkl'))
        counts[group] = len(pkls)
        files.extend((group, src) for src in pkls
                     if int(src.parent.name.split('_')[1]) in tasks)
    if full:
        if dict(counts) != EXPECTED:
            raise ValueError(f'Clean PKL counts changed: {dict(counts)} != {EXPECTED}')
        if len(manifest) != 187 or sum(manifest.values()) != 14:
            raise ValueError('Unexpected clean selection manifest (expect 187, 14 trims)')
        all_rel = {p.relative_to(root).as_posix()
                   for group in groups for p in (root / group).glob('task_*/traj_*.pkl')}
        if all_rel != set(manifest):
            raise ValueError('Clean PKL set and manifest disagree')
    for _, src in files:
        if src.relative_to(root).as_posix() not in manifest:
            raise ValueError(f'{src}: not present in clean manifest')
    files.sort(key=lambda x: (x[1].parent.name, GROUPS.index(x[0]), x[1].name))
    if limit is not None:
        seen = defaultdict(int)
        chosen = []
        for group, path in files:
            key = (group, path.parent.name)
            if seen[key] >= limit:
                continue
            seen[key] += 1
            chosen.append((group, path))
        files = chosen
    if not files:
        raise ValueError('No trajectories selected')
    assigned = Counter()
    jobs = []
    for group, src in files:
        task = src.parent.name
        idx = assigned[task]
        assigned[task] += 1
        rel = Path(task) / f'traj_{idx:03d}.pkl'
        jobs.append((group, src, rel, bool(manifest[src.relative_to(root).as_posix()])))
    if len({x[2] for x in jobs}) != len(jobs):
        raise AssertionError('Output path collision')
    return jobs


def previews_to_png(folder, rel, previews):
    from PIL import Image
    sub = folder / '_previews' / rel.parent.name / rel.stem
    sub.mkdir(parents=True, exist_ok=True)
    for t, front, wrist in previews:
        Image.fromarray(front, mode='RGB').save(sub / f'frame_{t:04d}_front.png')
        Image.fromarray(wrist, mode='RGB').save(sub / f'frame_{t:04d}_gripper.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--destination', type=Path, default=DEFAULT_DEST)
    ap.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    ap.add_argument('--tasks', nargs='+', type=int, default=list(range(16)))
    ap.add_argument('--limit-per-task', type=int, default=None,
                    help='Pilot: up to N PKLs per task per experimental group')
    ap.add_argument('--dry-run', action='store_true', help='Transform and validate; write nothing')
    ap.add_argument('--save-preview', action='store_true',
                    help='Save RGB PNGs for first selected PKL in each group (not with --dry-run)')
    args = ap.parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        ap.error('Source and destination overlap')
    groups = tuple(dict.fromkeys(args.groups))
    tasks = set(args.tasks)
    if not tasks or any(task not in range(16) for task in tasks):
        ap.error('Tasks must be within 00..15')
    if args.limit_per_task is not None and args.limit_per_task < 1:
        ap.error('--limit-per-task must be >=1')
    if args.save_preview and args.dry_run:
        ap.error('--save-preview cannot be used with --dry-run')
    full = (set(groups) == set(GROUPS) and tasks == set(range(16))
            and args.limit_per_task is None)
    manifest_clean = read_clean_manifest(source)
    jobs = gather(source, groups, tasks, args.limit_per_task, manifest_clean, full)
    if full and len(jobs) != 187:
        raise ValueError(f'Expected 187 trajectories, found {len(jobs)}')

    if not args.dry_run:
        existing = [str(destination / rel) for _, _, rel, _ in jobs
                    if (destination / rel).exists() or (destination / rel.with_suffix('.json')).exists()]
        if (destination / 'manifest.csv').exists():
            existing.append(str(destination / 'manifest.csv'))
        if existing:
            raise FileExistsError('Output exists; choose a fresh destination: ' + str(existing[:5]))

    print(f'Selected {len(jobs)} clean VLA-JEPA trajectories; '
          f'output={destination}; dry_run={args.dry_run}', flush=True)
    rows = []
    summary = Counter()
    shown_preview_groups = set()
    total_frames = 0
    for number, (group, src, dest_rel, trimmed) in enumerate(jobs, 1):
        src_json, original_meta = check_json(src)
        with src.open('rb') as f:
            payload = pickle.load(f)
        result, signatures, images_preview = transform(payload, src, source, dest_rel, trimmed)
        if not args.dry_run:
            atomic_pickle(payload, destination / dest_rel, result['frames'], signatures)
            atomic_copy(src_json, (destination / dest_rel).with_suffix('.json'))
            if args.save_preview and group not in shown_preview_groups:
                previews_to_png(destination, dest_rel, images_preview)
                shown_preview_groups.add(group)
        total_frames += result['frames']
        summary[group] += 1
        rows.append({
            'output_pkl': dest_rel.as_posix(),
            'source_group': group,
            'source_clean_pkl': src.relative_to(source).as_posix(),
            'task_id': int(dest_rel.parent.name.split('_')[1]),
            'frames': result['frames'],
            'transitions': result['transitions'],
            'frame0_removed_during_clean': int(trimmed),
            'first_close_command_t': result['original_close_command_t'],
            'first_closed_observed_t': result['observed_close_t'],
            'close_feedback_lag_frames': result['close_feedback_lag_frames'],
            'original_terminal_command': result['original_terminal_gripper_command'],
            'original_missing_terminal_open': int(result['original_missing_terminal_open']),
            'terminal_action_zero_synthetic': 1,
            'xyz_max_mm': f'{result["xyz_max_mm"]:.6f}',
            'rotation_max_deg': f'{result["rotation_max_deg"]:.6f}',
            'roundtrip_xyz_max_mm': f'{result["roundtrip_xyz_max_mm"]:.9f}',
            'roundtrip_rotation_max_deg': f'{result["roundtrip_rotation_max_deg"]:.9f}',
        })
        prefix = 'CHECK' if args.dry_run else 'OK'
        print(f'[{number:03d}/{len(jobs):03d}] {prefix} {group}/{src.parent.name}/{src.name}'
              f' -> {dest_rel}; T={result["frames"]}; '
              f'grip_lag={result["close_feedback_lag_frames"]}; '
              f'terminal_original={result["original_terminal_gripper_command"]}',
              flush=True)
        del payload, images_preview
        gc.collect()
    if full and not args.dry_run and dict(summary) != EXPECTED:
        raise AssertionError(f'Full run missing group(s): {dict(summary)}')
    if not args.dry_run:
        atomic_csv(destination / 'manifest.csv', rows)
    print('\n=== VLA-JEPA PREPROCESSING COMPLETE ===')
    print('Counts by group:', dict(summary))
    print('Total:', len(jobs), 'PKLs and JSONs; frames:', total_frames)
    print('Output:', destination)
    print('Manifest:', '(dry-run, not written)' if args.dry_run else destination / 'manifest.csv')
    print('No input clean files modified.')


if __name__ == '__main__':
    main()
