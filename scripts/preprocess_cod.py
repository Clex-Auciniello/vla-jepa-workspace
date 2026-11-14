#!/usr/bin/env python3
"""Convert selected COD *_clean robot rollouts to canonical 7D VLA-JEPA PKLs.

- Source PKLs/JSONs are NEVER modified. Output names are unique per task across
  both sources and include source provenance in each PKL and manifest.
- Next-observation EEF pose -> scaled world-frame XYZ + relative SO(3) Euler XYZ.
- Next-observation gripper_qpos >= 0.4 -> scaled 0/20 gripper label.
- Final action [0]*7 is a SYNTHETIC stationary+OPEN label, even if the original
  rollout ended with CLOSE. No final image or state is fabricated.
- Raw images BGR uint8 -> RGB uint8 224x224. Front crop margins T,B,L,R =
  0,10,140,90; wrist resize only. Other images and depth are removed.
- Preserve raw non-image proprioceptive fields; add canonical observation fields.

Safety: Load only trusted pickle files. Run --dry-run and a small --limit-per-task
pilot before the full run. Complete runs require 28 + 38 cleaned trajectories.
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
import savers  # noqa: F401: necessary to deserialize savers.Trajectory

BASE = Path('/mnt/beegfs/a.cuciniello9/datasets/mixture')
DEFAULT_SOURCES = (
    ('cod_epoch_100_wrist_clean', BASE / 'cod_epoch_100_wrist_clean'),
    ('cod_finetuning_no_wrist_clean', BASE / 'cod_rollout_finetuning/no_wrist_clean'),
)
DEFAULT_DEST = BASE / 'cod_preprocessed'
EXPECTED_COUNTS = {'cod_epoch_100_wrist_clean': 28,
                   'cod_finetuning_no_wrist_clean': 38}
SUCCESS_KEYS = ('object_reached', 'object_picked', 'object_placed')
FRONT_KEY = 'camera_front_image'
WRIST_KEY = 'camera_gripper_image'
FRONT_MARGINS = (0, 10, 140, 90)  # top, bottom, left, right
TARGET_SIZE_WH = (224, 224)
SCALE = 0.05
GRIPPER_THRESHOLD = 0.40
MAX_XYZ_STEP_M = 0.06
MAX_ROT_STEP_DEG = 30.
MAX_POS_REBUILD_ERROR_MM = 0.01
MAX_ROT_REBUILD_ERROR_DEG = 0.001


def get_step(traj, index):
    try:
        return traj.get(index, decompress=False)
    except TypeError:
        return traj.get(index)


def vector(value, size, label, path, t):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.size != size or not np.isfinite(a).all():
        raise ValueError(f'{path}, step={t}: {label} expected finite {size}D; got {a.shape}')
    return a


def native_steps(traj, path):
    """Keep original 5-tuple Trajectory._data representation intact."""
    raw = getattr(traj, '_data', None)
    if not isinstance(raw, list) or len(raw) != len(traj):
        raise TypeError(f'{path}: unexpected savers.Trajectory._data')
    if len(raw) < 2:
        raise ValueError(f'{path}: need at least two steps; got {len(raw)}')
    steps = []
    for t, entry in enumerate(raw):
        if not isinstance(entry, tuple) or len(entry) != 5:
            raise TypeError(f'{path}, step={t}: expected (obs,reward,done,info,action) tuple')
        obs, reward, done, info, action = entry
        if not isinstance(obs, dict):
            raise TypeError(f'{path}, step={t}: obs must be a dict')
        steps.append([obs, reward, done, info, action])
    return steps


def sha_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def validate_success_json(pkl):
    jpath = pkl.with_suffix('.json')
    if not jpath.is_file():
        raise FileNotFoundError(jpath)
    with jpath.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    if not isinstance(meta, dict) or not all(meta.get(k) == 1 for k in SUCCESS_KEYS):
        raise ValueError(f'{jpath}: JSON success flags are not 1/1/1')
    return jpath, meta


def read_bgr(obs, key, path, t):
    if key not in obs:
        raise KeyError(f'{path}, step={t}: missing {key}')
    a = np.asarray(obs[key])
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f'{path}, step={t}: {key}: expected HWC uint8 BGR, '
                         f'got {a.shape} {a.dtype}')
    if not np.isfinite(a).all():
        raise ValueError(f'{path}, step={t}: {key}: nonfinite pixels')
    return a


def to_rgb_224(bgr, is_front, path, t):
    if is_front:
        top, bottom, left, right = FRONT_MARGINS
        h, w = bgr.shape[:2]
        if top + bottom >= h or left + right >= w:
            raise ValueError(f'{path}, step={t}: bad crop {FRONT_MARGINS} for {(h,w)}')
        bgr = bgr[top:h-bottom, left:w-right]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, TARGET_SIZE_WH, interpolation=cv2.INTER_LINEAR)
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise AssertionError('Unexpected RGB size or dtype')
    return np.ascontiguousarray(rgb)


def derive_gripper(qpos, original_commands, path):
    binary = (qpos >= GRIPPER_THRESHOLD).astype(np.uint8)
    changes = np.flatnonzero(np.diff(binary) != 0) + 1
    if (binary[0] != 0 or binary[-1] != 1 or len(changes) != 1
            or binary[changes[0]] != 1):
        raise ValueError(f'{path}: invalid gripper 0->1 observed pattern; '
                         f'changes={changes.tolist()} start={binary[0]} end={binary[-1]}')
    close = np.isclose(original_commands, 255, atol=0.5, rtol=0)
    opened = np.isclose(original_commands, 0, atol=0.5, rtol=0)
    if not np.all(close | opened):
        raise ValueError(f'{path}: unknown original gripper commands')
    cbin = close.astype(np.uint8)
    closes = np.flatnonzero(np.diff(cbin) == 1) + 1
    opens = np.flatnonzero(np.diff(cbin) == -1) + 1
    # Accepted observed pattern includes missing final OPEN command.
    if cbin[0] or len(closes) != 1 or len(opens) > 1 or (
            len(opens) == 1 and opens[0] <= closes[0]):
        raise ValueError(f'{path}: invalid original command sequence: '
                         f'closes={closes.tolist()}, opens={opens.tolist()}')
    close_t = int(closes[0])
    closed_observed_t = int(changes[0])
    lag = closed_observed_t - close_t
    if lag < 0:
        raise ValueError(f'{path}: observed closure before CLOSE command')
    # Lag >3 is diagnostic only (known COD sample takes 4 frames).
    original_terminal = 'OPEN' if bool(opened[-1]) else 'CLOSE'
    return binary, {
        'original_terminal_gripper_command': original_terminal,
        'terminal_open_label_synthetic': original_terminal == 'CLOSE',
        'first_close_command_step': close_t,
        'first_closed_observed_step': closed_observed_t,
        'close_feedback_lag_frames': lag,
        'slow_close_feedback_lag_gt_3': lag > 3,
    }


def transform(payload, src, source_name, source_root, target_rel, json_meta):
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise TypeError(f'{src}: payload not a dict with traj')
    traj = payload['traj']
    steps = native_steps(traj, src)
    T = len(steps)

    xyz = np.stack([vector(s[0]['eef_pos'], 3, 'eef_pos', src, i)
                    for i, s in enumerate(steps)])
    quat = np.stack([vector(s[0]['eef_quat'], 4, 'eef_quat', src, i)
                     for i, s in enumerate(steps)])
    qpos = np.asarray([vector(s[0]['gripper_qpos'], 1, 'gripper_qpos', src, i)[0]
                       for i, s in enumerate(steps)], dtype=np.float64)
    originals = np.stack([vector(s[4], 8, 'original COD absolute action', src, i)
                          for i, s in enumerate(steps)])
    if not np.allclose(np.linalg.norm(quat, axis=1), 1., atol=1e-3, rtol=0):
        raise ValueError(f'{src}: invalid quaternion norm')
    grip_binary, grip_info = derive_gripper(qpos, originals[:, 7], src)

    rot = Rotation.from_quat(quat)
    absolute_euler = rot.as_euler('xyz', degrees=False)
    delta_xyz_m = np.diff(xyz, axis=0)
    rel_rot = rot[1:] * rot[:-1].inv()
    step_xyz_m = np.linalg.norm(delta_xyz_m, axis=1)
    step_rot_deg = np.degrees(rel_rot.magnitude())
    if np.any(step_xyz_m > MAX_XYZ_STEP_M + 1e-10):
        t = int(np.flatnonzero(step_xyz_m > MAX_XYZ_STEP_M + 1e-10)[0])
        raise ValueError(f'{src}: XYZ jump t={t}: {step_xyz_m[t]*1000:.3f}mm')
    if np.any(step_rot_deg > MAX_ROT_STEP_DEG + 1e-10):
        t = int(np.flatnonzero(step_rot_deg > MAX_ROT_STEP_DEG + 1e-10)[0])
        raise ValueError(f'{src}: rotation jump t={t}: {step_rot_deg[t]:.3f}deg')
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Gimbal lock detected')
        delta_euler = rel_rot.as_euler('xyz', degrees=False)

    actions = np.zeros((T, 7), dtype=np.float32)
    actions[:-1, :3] = delta_xyz_m / SCALE
    actions[:-1, 3:6] = delta_euler / SCALE
    actions[:-1, 6] = grip_binary[1:].astype(np.float32) / SCALE
    # One supervised terminal OPEN, never a fabricated observed OPEN state.
    actions[-1, :] = 0.

    p_rebuild = xyz[:-1] + actions[:-1, :3].astype(np.float64) * SCALE
    r_rebuild = (Rotation.from_euler('xyz', actions[:-1, 3:6].astype(np.float64) * SCALE)
                 * rot[:-1])
    max_pos_err = float(np.max(np.linalg.norm(p_rebuild - xyz[1:], axis=1) * 1000.))
    max_rot_err = float(np.max(np.degrees((r_rebuild.inv() * rot[1:]).magnitude())))
    if max_pos_err > MAX_POS_REBUILD_ERROR_MM or max_rot_err > MAX_ROT_REBUILD_ERROR_DEG:
        raise ValueError(f'{src}: float32 roundtrip error pos={max_pos_err:.8f}mm, '
                         f'rot={max_rot_err:.8f}deg')

    signature = {}
    samples = {0, T // 2, T - 1}
    for t, step in enumerate(steps):
        obs = step[0]
        front = to_rgb_224(read_bgr(obs, FRONT_KEY, src, t), True, src, t)
        wrist = to_rgb_224(read_bgr(obs, WRIST_KEY, src, t), False, src, t)
        eef = np.r_[xyz[t], absolute_euler[t]].astype(np.float32)
        grip_state = np.array([qpos[t], grip_binary[t]], dtype=np.float32)
        joint_pos = vector(obs['joint_pos'], 6, 'joint_pos', src, t).astype(np.float32)
        joint_vel = vector(obs['joint_vel'], 6, 'joint_vel', src, t).astype(np.float32)

        # All raw image/depth keys removed, including COD cropped_image.
        for k in list(obs):
            if 'image' in k.lower() or 'depth' in k.lower():
                del obs[k]
        obs[FRONT_KEY] = front
        obs[WRIST_KEY] = wrist
        obs['EEF_state'] = eef
        obs['gripper_state'] = grip_state
        obs['joint_positions'] = joint_pos
        obs['joint_velocities'] = joint_vel
        obs['action_world'] = np.r_[eef, float(grip_binary[t])].astype(np.float32)
        step[4] = actions[t].copy()
        if t in samples:
            signature[t, FRONT_KEY] = sha_array(front)
            signature[t, WRIST_KEY] = sha_array(wrist)

    traj._data[:] = [tuple(s) for s in steps]
    source_rel = src.relative_to(source_root)
    original_traj_number = payload.get('traj_number')
    payload['source'] = source_name
    payload['source_file'] = f'{source_name}/{source_rel.as_posix()}'
    payload['source_traj_number'] = original_traj_number
    payload['traj_number'] = int(target_rel.stem.split('_')[1])
    payload['task_id'] = int(target_rel.parent.name.split('_')[1])
    payload['merge_preprocessing'] = {
        'version': 'cod_v1',
        'scale_factor': SCALE,
        'gripper_qpos_threshold': GRIPPER_THRESHOLD,
        'action_alignment': 't -> t+1 observed EEF and gripper for t<T-1',
        'terminal_action': 'synthetic all-zero 7D: stationary EEF, OPEN gripper',
        'original_terminal_gripper_command': grip_info['original_terminal_gripper_command'],
        'terminal_open_label_synthetic': grip_info['terminal_open_label_synthetic'],
        'rotation': 'R_next * R_current.inv(), Euler xyz',
        'source_action_8d': 'absolute Cartesian target, ignored for pose delta',
        'source_image_encoding_assumption': 'BGR uint8 HWC',
        'output_image_encoding': 'RGB uint8 HWC 224x224',
        'front_crop_margins_TBLR': list(FRONT_MARGINS),
        'resize_interpolation': 'cv2.INTER_LINEAR',
        'gripper_state': '[raw qpos, binary closed]',
        'original_json_aborted': json_meta.get('aborted'),
        'original_json_trajectory_complete': json_meta.get('trajectory_complete'),
        'selected_from_curated_clean_source': True,
    }
    return ({'frames': T,
             'transitions': T - 1,
             'xyz_max_mm': float(np.max(step_xyz_m) * 1000),
             'rot_max_deg': float(np.max(step_rot_deg)),
             'pos_roundtrip_err_max_mm': max_pos_err,
             'rot_roundtrip_err_max_deg': max_rot_err,
             **grip_info}, signature)


def verify_saved(path, n_frames, signatures):
    with path.open('rb') as f:
        payload = pickle.load(f)
    steps = native_steps(payload['traj'], path)
    if len(steps) != n_frames:
        raise AssertionError(f'{path}: wrong length after serialization')
    closed = []
    for t, (obs, reward, done, info, action) in enumerate(steps):
        a = np.asarray(action)
        if a.shape != (7,) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise AssertionError(f'{path}, t={t}: action invalid')
        if t < n_frames - 1 and float(a[6]) not in (0., 20.):
            raise AssertionError(f'{path}, t={t}: gripper action not in 0/20')
        image_keys = sorted(k for k in obs if 'image' in k.lower() or 'depth' in k.lower())
        if image_keys != sorted([FRONT_KEY, WRIST_KEY]):
            raise AssertionError(f'{path}, t={t}: unexpected image keys: {image_keys}')
        for key in (FRONT_KEY, WRIST_KEY):
            image = np.asarray(obs[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise AssertionError(f'{path}, t={t}: bad {key} shape/dtype')
            if (t, key) in signatures and sha_array(image) != signatures[t, key]:
                raise AssertionError(f'{path}, t={t}: image changed across pickle roundtrip')
        for key, length in (('EEF_state', 6), ('gripper_state', 2),
                            ('joint_positions', 6), ('joint_velocities', 6),
                            ('action_world', 7)):
            x = np.asarray(obs[key])
            if x.shape != (length,) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise AssertionError(f'{path}, t={t}: bad {key}')
        closed.append(int(obs['gripper_state'][1]))
        if t in {0, n_frames//2, n_frames-1}:
            view = get_step(payload['traj'], t)
            if not np.array_equal(np.asarray(view['action']), a):
                raise AssertionError(f'{path}, t={t}: Trajectory.get action mismatch')
    if not np.array_equal(np.asarray(steps[-1][4]), np.zeros(7, dtype=np.float32)):
        raise AssertionError(f'{path}: terminal action must be all-zero')
    observed = np.asarray(closed[1:], dtype=np.float32) / SCALE
    action_grip = np.asarray([s[4][6] for s in steps[:-1]], dtype=np.float32)
    if not np.array_equal(observed, action_grip):
        raise AssertionError(f'{path}: t action gripper inconsistent with t+1 observed state')
    del payload
    gc.collect()


def atomic_dump_verified(payload, dest, n_frames, signatures):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=dest.parent,
                prefix=f'.{dest.stem}.', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        verify_saved(tmp, n_frames, signatures)
        os.replace(tmp, dest)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def atomic_copy_json(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=dest.parent,
                prefix=f'.{dest.stem}.', suffix='.json.tmp', delete=False) as f:
            tmp = Path(f.name)
            with src.open('rb') as original:
                shutil.copyfileobj(original, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def atomic_write_manifest(dest, rows):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='',
                dir=dest.parent, prefix='.manifest.', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def gather(sources, selected_tasks, limit_per_task, strict_count):
    found = []
    for name, root in sources:
        if not root.is_dir():
            raise FileNotFoundError(root)
        all_pkls = sorted(root.glob('task_*/traj_*.pkl'))
        if strict_count and len(all_pkls) != EXPECTED_COUNTS[name]:
            raise ValueError(f'{name}: expected {EXPECTED_COUNTS[name]} PKLs, found {len(all_pkls)}')
        candidates = [p for p in all_pkls if int(p.parent.name.split('_')[1]) in selected_tasks]
        if limit_per_task is not None:
            used = defaultdict(int)
            filtered = []
            for p in candidates:
                task = p.parent.name
                if used[task] >= limit_per_task:
                    continue
                used[task] += 1
                filtered.append(p)
            candidates = filtered
        if not candidates:
            raise RuntimeError(f'{name}: no selected task PKLs')
        print(f'[{name}] selected {len(candidates)} PKLs', flush=True)
        for p in candidates:
            validate_success_json(p)
            found.append((name, root, p))
    used_per_task = Counter()
    jobs = []
    for name, root, p in sorted(found,
                                key=lambda x: (x[2].parent.name, x[0], x[2].name)):
        task = p.parent.name
        i = used_per_task[task]
        used_per_task[task] += 1
        rel = Path(task) / f'traj_{i:03d}.pkl'
        jobs.append((name, root, p, rel))
    if len(set(x[3] for x in jobs)) != len(jobs):
        raise AssertionError('Output filename collision')
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', action='append', metavar='NAME=PATH',
                   help='Repeat to replace default input paths')
    p.add_argument('--destination', type=Path, default=DEFAULT_DEST)
    p.add_argument('--dry-run', action='store_true', help='Compute all, write nothing')
    p.add_argument('--tasks', type=int, nargs='+', default=list(range(16)))
    p.add_argument('--limit-per-task', type=int, default=None,
                   help='Pilot: up to N trajectories per task PER SOURCE')
    args = p.parse_args()
    if args.limit_per_task is not None and args.limit_per_task < 1:
        p.error('--limit-per-task must be >= 1')
    if any(task not in range(16) for task in args.tasks):
        p.error('--tasks must be IDs 0...15')

    sources = DEFAULT_SOURCES
    if args.source:
        parsed = []
        seen = set()
        for raw in args.source:
            if '=' not in raw:
                p.error('--source expects NAME=PATH')
            name, root = raw.split('=', 1)
            if name not in EXPECTED_COUNTS or name in seen or not root:
                p.error('source name must be one of: ' + ', '.join(EXPECTED_COUNTS))
            parsed.append((name, Path(root).expanduser().resolve()))
            seen.add(name)
        sources = tuple(parsed)
    dest = args.destination.expanduser().resolve()
    for name, root in sources:
        root = root.expanduser().resolve()
        if dest == root or root in dest.parents or dest in root.parents:
            p.error(f'destination overlaps source: {dest} vs {root}')

    full = (args.limit_per_task is None and set(args.tasks) == set(range(16))
            and {n for n, _ in sources} == set(EXPECTED_COUNTS))
    jobs = gather(sources, set(args.tasks), args.limit_per_task, strict_count=full)
    if full and len(jobs) != 66:
        raise RuntimeError(f'Expected 66 clean COD inputs, found {len(jobs)}')
    print(f'\nSelected {len(jobs)} trajectories; output={dest}; dry_run={args.dry_run}', flush=True)

    # Reject existing destinations: do not silently overwrite a prior pilot/full run.
    if not args.dry_run:
        preexisting = [str(dest / rel) for _, _, _, rel in jobs
                       if (dest / rel).exists() or (dest / rel.with_suffix('.json')).exists()]
        if (dest / 'manifest.csv').exists():
            preexisting.append(str(dest / 'manifest.csv'))
        if preexisting:
            raise FileExistsError('Output already exists; choose a fresh --destination: '
                                  + str(preexisting[:5]))

    manifest = []
    seen_sources = Counter()
    total_frames = 0
    for n, (source_name, source_root, src, target_rel) in enumerate(jobs, 1):
        src_json, json_meta = validate_success_json(src)
        with src.open('rb') as f:
            payload = pickle.load(f)
        result, signatures = transform(payload, src, source_name, source_root,
                                       target_rel, json_meta)
        if not args.dry_run:
            atomic_dump_verified(payload, dest / target_rel, result['frames'], signatures)
            atomic_copy_json(src_json, (dest / target_rel).with_suffix('.json'))
        total_frames += result['frames']
        seen_sources[source_name] += 1
        manifest.append({
            'output_pkl': target_rel.as_posix(),
            'output_json': target_rel.with_suffix('.json').as_posix(),
            'source_dataset': source_name,
            'source_pkl': f'{source_name}/{src.relative_to(source_root).as_posix()}',
            'source_json': f'{source_name}/{src_json.relative_to(source_root).as_posix()}',
            'task_id': int(target_rel.parent.name.split('_')[1]),
            'frames': result['frames'],
            'transitions': result['transitions'],
            'original_json_aborted': json_meta.get('aborted'),
            'original_json_trajectory_complete': json_meta.get('trajectory_complete'),
            'first_close_command_step': result['first_close_command_step'],
            'first_closed_observed_step': result['first_closed_observed_step'],
            'close_feedback_lag_frames': result['close_feedback_lag_frames'],
            'slow_close_feedback_lag_gt_3': result['slow_close_feedback_lag_gt_3'],
            'original_terminal_gripper_command': result['original_terminal_gripper_command'],
            'terminal_open_label_synthetic': result['terminal_open_label_synthetic'],
            'xyz_max_mm': f'{result["xyz_max_mm"]:.6f}',
            'rot_max_deg': f'{result["rot_max_deg"]:.6f}',
            'pos_roundtrip_max_mm': f'{result["pos_roundtrip_err_max_mm"]:.9f}',
            'rot_roundtrip_max_deg': f'{result["rot_roundtrip_err_max_deg"]:.9f}',
        })
        status = 'CHECK' if args.dry_run else 'OK'
        print(f'[{n:03d}/{len(jobs):03d}] {status} '
              f'{source_name}/{src.parent.name}/{src.name} -> {target_rel} '
              f'T={result["frames"]} lag={result["close_feedback_lag_frames"]} '
              f'terminal_original={result["original_terminal_gripper_command"]}', flush=True)
        del payload
        gc.collect()

    if not args.dry_run:
        atomic_write_manifest(dest / 'manifest.csv', manifest)
    print('\n=== DONE ===', flush=True)
    print('Count by source:', dict(seen_sources))
    print('Total trajectories:', len(jobs), 'total frames:', total_frames)
    print('Destination:', dest)
    print('Manifest:', '(not written: dry run)' if args.dry_run else dest / 'manifest.csv')
    print('Original PKLs and JSONs were not modified.')


if __name__ == '__main__':
    main()
