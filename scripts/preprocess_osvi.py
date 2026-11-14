#!/usr/bin/env python3
"""OSVI-AWDA -> VLA-JEPA canonical robot trajectories (non-destructive).

Inputs (only audited *_clean PKLs + their 1/1/1 JSON):
  .../datasets/mixture/osvi_awda_rollout_clean
  .../datasets/mixture/result_osvi_awda_human_clean

Output:
  .../datasets/mixture/osvi_preprocessed/task_XX/traj_NNN.pkl + .json
  .../datasets/mixture/osvi_preprocessed/manifest.csv

OBSERVED STATE at t:
  EEF_state = [xyz, Euler xyz] in base_link, gripper_state[1] = {0,1}.

ACTION at t, for t < T-1:
  a[:3]  = (p[t+1] - p[t]) / 0.05
  a[3:6] = Euler_xyz(R[t+1] @ R[t].inv()) / 0.05
  a[6]   = binary_gripper_qpos[t+1] / 0.05   (0 or 20)
  action[-1] = [0]*7; observed last qpos is closed but original
  last command is OPEN=0, so terminal action opens the gripper.

IMAGES: source raw BGR HWC uint8 -> RGB HWC uint8 224x224.
  front crop *margins* (top, bottom, left, right)=(0,10,140,90),
  then cv2.INTER_LINEAR resize; gripper resize only.

Only front+gripper RGB are retained. Existing non-image proprio is
preserved; additional canonical keys are added. All original data is untouched.

Run only on trusted PKLs; pickle.load executes arbitrary Python objects.
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
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import savers  # noqa: F401 - necessary to unpickle savers.Trajectory


BASE = Path('/mnt/beegfs/a.cuciniello9/datasets/mixture')
SOURCES = (
    ('osvi_awda_rollout_clean', BASE / 'osvi_awda_rollout_clean'),
    ('result_osvi_awda_human_clean', BASE / 'result_osvi_awda_human_clean'),
)
DEFAULT_DEST = BASE / 'osvi_preprocessed'
EXPECTED_COUNTS = {'osvi_awda_rollout_clean': 108,
                   'result_osvi_awda_human_clean': 83}
SCALE = 0.05
GRIPPER_THRESHOLD = 0.2
FRONT_CROP = (0, 10, 140, 90)  # (top, bottom, left, right) MARGINS
IMAGE_SIZE = (224, 224)  # cv2 (width, height)
MAX_DELTA_XYZ_M = 0.06
SUCCESS_KEYS = ('object_reached', 'object_picked', 'object_placed')
FRONT_NAMES = ('camera_front_image', 'front_camera_image')
GRIPPER_NAMES = ('camera_gripper_image', 'gripper_camera_image')
CANONICAL_IMAGE_NAMES = ('camera_front_image', 'camera_gripper_image')


def vec(value, n, label, path, t):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.size != n or not np.isfinite(a).all():
        raise ValueError(f'{path}, frame {t}: {label} must be finite {n}D; got {a.shape}')
    return a


def native_steps(traj, path):
    """Read savers.Trajectory's native (obs,reward,done,info,action) tuples."""
    raw = getattr(traj, '_data', None)
    if not isinstance(raw, list) or len(raw) != len(traj):
        raise TypeError(f'{path}: unexpected Trajectory._data format')
    steps = []
    for t, entry in enumerate(raw):
        if not isinstance(entry, tuple) or len(entry) != 5:
            raise TypeError(f'{path}, frame {t}: not a native 5-tuple')
        obs, reward, done, info, action = entry
        if not isinstance(obs, dict):
            raise TypeError(f'{path}, frame {t}: observation not dict')
        steps.append([obs, reward, done, info, action])
    if len(steps) < 2:
        raise ValueError(f'{path}: need at least two frames')
    return steps


def select_image(obs, candidates, path, t):
    keys = [key for key in candidates if key in obs]
    if len(keys) != 1:
        raise ValueError(f'{path}, frame {t}: expected exactly one image from {candidates}; got {keys}')
    a = np.asarray(obs[keys[0]])
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f'{path}, frame {t}: {keys[0]} is not raw HWC uint8 BGR; '
                         f'got shape={a.shape}, dtype={a.dtype}')
    return a


def preprocess_image(raw_bgr, front, path, t):
    if front:
        top, bottom, left, right = FRONT_CROP
        h, w = raw_bgr.shape[:2]
        if top + bottom >= h or left + right >= w:
            raise ValueError(f'{path}, frame {t}: crop {FRONT_CROP} incompatible '
                             f'with front size {w}x{h}')
        raw_bgr = raw_bgr[top:h-bottom, left:w-right]
    rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
    return np.ascontiguousarray(rgb)


def sha_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def check_json(path):
    jpath = path.with_suffix('.json')
    if not jpath.is_file():
        raise FileNotFoundError(f'Missing JSON for {path}')
    with jpath.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(data.get(k) == 1 for k in SUCCESS_KEYS):
        raise ValueError(f'Invalid success flags in {jpath}')
    return jpath


def gripper_validation(states, commands, path):
    """Strict observation pattern 0->1 plus terminal command open.

    For OSVI the observation is recorded *before* executing the command, so
    there is typically no observed 1->0 final transition.
    """
    qbin = (states >= GRIPPER_THRESHOLD).astype(np.uint8)
    changes = np.flatnonzero(np.diff(qbin) != 0) + 1
    if (qbin[0] != 0 or qbin[-1] != 1 or len(changes) != 1
            or qbin[changes[0]] != 1):
        raise ValueError(f'{path}: invalid observed gripper pattern at threshold '
                         f'{GRIPPER_THRESHOLD}; transitions={changes.tolist()}, '
                         f'initial={qbin[0]}, final={qbin[-1]}')

    if not np.isin(commands, [0, 255]).all():
        raise ValueError(f'{path}: expected original gripper commands 0/255, '
                         f'got {np.unique(commands)}')
    cbin = (commands == 255).astype(np.uint8)
    cmd_changes = np.flatnonzero(np.diff(cbin) != 0) + 1
    if (cbin[0] != 0 or cbin[-1] != 0 or len(cmd_changes) != 2
            or cbin[cmd_changes[0]] != 1 or cbin[cmd_changes[1]] != 0):
        raise ValueError(f'{path}: commands do not match open-close-open; '
                         f'changes={cmd_changes.tolist()}')
    # Feedback should begin to show closure shortly after the close command.
    close_observed = int(changes[0])
    close_command = int(cmd_changes[0])
    if not 0 <= close_observed - close_command <= 3:
        raise ValueError(f'{path}: close feedback step={close_observed} not aligned '
                         f'with close command step={close_command}')
    return qbin, close_observed, close_command


def transform_payload(payload, src, origin, target_rel):
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise TypeError(f'{src}: expected dict containing traj')
    traj = payload['traj']
    steps = native_steps(traj, src)
    T = len(steps)

    xyz = np.stack([vec(s[0]['eef_pos'], 3, 'eef_pos', src, t)
                    for t, s in enumerate(steps)])
    quat = np.stack([vec(s[0]['eef_quat'], 4, 'eef_quat', src, t)
                     for t, s in enumerate(steps)])
    qpos = np.array([vec(s[0]['gripper_qpos'], 1, 'gripper_qpos', src, t)[0]
                     for t, s in enumerate(steps)])
    old_actions = np.stack([vec(s[4], 8, 'original action', src, t)
                            for t, s in enumerate(steps)])
    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, rtol=0, atol=1e-3):
        raise ValueError(f'{src}: non-unit quaternions')

    # Original 8D OSVI action is a commanded ABSOLUTE target, not the
    # observed pose. We never compare action[:7] to the observed EEF pose.
    gripper_bin, close_obs, close_cmd = gripper_validation(qpos, old_actions[:, 7], src)
    R = Rotation.from_quat(quat)  # order XYZW
    rpy = R.as_euler('xyz', degrees=False)
    d_xyz_m = np.diff(xyz, axis=0)
    step_norm_m = np.linalg.norm(d_xyz_m, axis=1)
    if (step_norm_m > MAX_DELTA_XYZ_M + 1e-10).any():
        bad_t = np.flatnonzero(step_norm_m > MAX_DELTA_XYZ_M + 1e-10)[0]
        raise ValueError(f'{src}: large EEF jump t={bad_t}, '
                         f'{1000*step_norm_m[bad_t]:.3f}mm > 60mm; not clean')
    d_rpy = (R[1:] * R[:-1].inv()).as_euler('xyz', degrees=False)

    new_actions = np.zeros((T, 7), dtype=np.float32)
    new_actions[:-1, :3] = d_xyz_m / SCALE
    new_actions[:-1, 3:6] = d_rpy / SCALE
    new_actions[:-1, 6] = gripper_bin[1:].astype(np.float32) / SCALE
    # Terminal: all-zero action also commands OPEN gripper (gripper target=0).
    assert np.array_equal(new_actions[-1], np.zeros(7, dtype=np.float32))

    # Numerical round-trip checks after float32 quantization.
    rebuilt_p = xyz[:-1] + new_actions[:-1, :3].astype(np.float64) * SCALE
    pos_error_mm = np.linalg.norm(rebuilt_p - xyz[1:], axis=1) * 1000
    rebuilt_R = Rotation.from_euler('xyz',
                  new_actions[:-1, 3:6].astype(np.float64) * SCALE) * R[:-1]
    rot_error_deg = (rebuilt_R.inv() * R[1:]).magnitude() * 180 / np.pi
    if np.max(pos_error_mm) > 0.01 or np.max(rot_error_deg) > 0.001:
        raise ValueError(f'{src}: delta round-trip failed: max pos='
                         f'{pos_error_mm.max():g}mm, rot={rot_error_deg.max():g}deg')

    signatures = {}
    sample_indices = sorted({0, T // 2, T - 1})
    for t, step in enumerate(steps):
        obs = step[0]
        front_bgr = select_image(obs, FRONT_NAMES, src, t)
        wrist_bgr = select_image(obs, GRIPPER_NAMES, src, t)
        front_rgb = preprocess_image(front_bgr, True, src, t)
        wrist_rgb = preprocess_image(wrist_bgr, False, src, t)

        eef = np.r_[xyz[t], rpy[t]].astype(np.float32)
        grip = np.array([qpos[t], gripper_bin[t]], dtype=np.float32)
        jp = vec(obs['joint_pos'], 6, 'joint_pos', src, t).astype(np.float32)
        jv = vec(obs['joint_vel'], 6, 'joint_vel', src, t).astype(np.float32)

        # Keep original non-image fields, remove ALL other cameras and depth.
        # No incorrectly colored redundant images remain in the output.
        for key in list(obs):
            if ('image' in key.lower() or 'depth' in key.lower()):
                del obs[key]
        obs['camera_front_image'] = front_rgb
        obs['camera_gripper_image'] = wrist_rgb
        obs['EEF_state'] = eef
        obs['gripper_state'] = grip   # [raw gripper_qpos, closed_binary]
        obs['joint_positions'] = jp
        obs['joint_velocities'] = jv
        obs['action_world'] = np.r_[eef, float(gripper_bin[t])].astype(np.float32)
        step[4] = new_actions[t].copy()

        if t in sample_indices:
            for name in CANONICAL_IMAGE_NAMES:
                signatures[(t, name)] = sha_array(obs[name])

    # Native savers.Trajectory format is tuple-backed: DO NOT store dict steps.
    traj._data[:] = [tuple(s) for s in steps]

    # Preserve the original metadata and record exactly where this copy came from.
    old_traj_number = payload.get('traj_number')
    source_rel = src.relative_to(dict(SOURCES)[origin])
    payload['source'] = origin
    payload['source_file'] = f'{origin}/{source_rel.as_posix()}'
    payload['source_traj_number'] = old_traj_number
    payload['traj_number'] = int(target_rel.stem.split('_')[1])
    payload['task_id'] = int(target_rel.parent.name.split('_')[1])
    payload['merge_preprocessing'] = {
        'version': 'osvi_v1',
        'scale_factor': SCALE,
        'gripper_qpos_threshold': GRIPPER_THRESHOLD,
        'action_alignment': 'next observed EEF state and gripper for t<T-1',
        'terminal_action': 'all-zero 7D, matches original last OPEN command',
        'rotation': 'R_delta=R_next @ R_current.inv(); xyz Euler',
        'source_action_8d': 'absolute commanded Cartesian target; ignored for pose delta',
        'image_source_encoding': 'BGR ndarray uint8',
        'image_target_encoding': 'RGB ndarray uint8 224x224',
        'front_crop_margins_TBLR': list(FRONT_CROP),
        'resize': 'cv2.INTER_LINEAR',
        'gripper_state': '[raw_qpos, 0_or_1_closed]',
        'canonical_observation_fields': [
            'camera_front_image', 'camera_gripper_image', 'EEF_state',
            'gripper_state', 'joint_positions', 'joint_velocities', 'action_world'
        ],
    }
    result = {
        'frames': T,
        'transitions': T - 1,
        'gripper_close_observed_step': close_obs,
        'gripper_close_command_step': close_cmd,
        'max_step_mm': float(step_norm_m.max() * 1000),
        'mean_step_mm': float(step_norm_m.mean() * 1000),
        'pos_roundtrip_max_mm': float(pos_error_mm.max()),
        'rot_roundtrip_max_deg': float(rot_error_deg.max()),
    }
    return result, signatures


def verify_saved(path, expected_frames, signatures):
    with path.open('rb') as f:
        payload = pickle.load(f)
    steps = native_steps(payload['traj'], path)
    if len(steps) != expected_frames:
        raise ValueError(f'{path}: number of frames changed')
    last = np.asarray(steps[-1][4])
    if last.shape != (7,) or last.dtype != np.float32 or np.any(last != 0):
        raise ValueError(f'{path}: incorrect terminal action')
    observed_bin = []
    for t, (obs, reward, done, info, action) in enumerate(steps):
        a = np.asarray(action)
        if a.shape != (7,) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'{path}, frame {t}: invalid 7D action')
        if t < len(steps)-1 and a[6] not in (0.0, 20.0):
            raise ValueError(f'{path}, frame {t}: gripper action must be 0 or 20')
        if sorted(k for k in obs if 'image' in k.lower() or 'depth' in k.lower()) != sorted(CANONICAL_IMAGE_NAMES):
            raise ValueError(f'{path}, frame {t}: unexpected image/depth fields')
        for name in CANONICAL_IMAGE_NAMES:
            im = np.asarray(obs[name])
            if im.shape != (224, 224, 3) or im.dtype != np.uint8:
                raise ValueError(f'{path}, frame {t}: invalid {name} format')
            if (t, name) in signatures and sha_array(im) != signatures[(t, name)]:
                raise ValueError(f'{path}, frame {t}: image changed in save/reload: {name}')
        if np.asarray(obs['EEF_state']).shape != (6,):
            raise ValueError(f'{path}, frame {t}: invalid EEF_state')
        if np.asarray(obs['gripper_state']).shape != (2,):
            raise ValueError(f'{path}, frame {t}: invalid gripper_state')
        if np.asarray(obs['action_world']).shape != (7,):
            raise ValueError(f'{path}, frame {t}: invalid action_world')
        observed_bin.append(int(np.asarray(obs['gripper_state'])[1]))
    expected_gripper = np.asarray(observed_bin[1:], dtype=np.float32) / SCALE
    if not np.array_equal(np.stack([s[4] for s in steps])[:-1, 6], expected_gripper):
        raise ValueError(f'{path}: saved action gripper != next observed gripper')
    # Test public savers.Trajectory accessor, as in scripted template.
    for t in sorted({0, len(steps)//2, len(steps)-1}):
        try:
            visible = payload['traj'].get(t, decompress=False)
        except TypeError:
            visible = payload['traj'].get(t)
        if not np.array_equal(np.asarray(visible['action']), np.asarray(steps[t][4])):
            raise ValueError(f'{path}: public Trajectory.get action differs at {t}')
    del payload
    gc.collect()


def dump_atomic(payload, target, signatures, expected_frames):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=target.parent,
              prefix=f'.{target.stem}.', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        verify_saved(tmp, expected_frames, signatures)
        os.replace(tmp, target)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def copy_json_atomic(src, target):
    tmp = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('wb', dir=target.parent,
              prefix=f'.{target.stem}.', suffix='.json.tmp', delete=False) as f:
            tmp = Path(f.name)
            with src.open('rb') as original:
                shutil.copyfileobj(original, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def gather(limit_per_task, tasks, strict_counts):
    found = []
    for origin, root in SOURCES:
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates = sorted(p for p in root.glob('task_*/*.pkl')
                            if int(p.parent.name.split('_')[1]) in tasks)
        count_before_limit = len(candidates)
        if limit_per_task is not None:
            per_task = defaultdict(int)
            limited = []
            for p in candidates:
                if per_task[p.parent.name] >= limit_per_task:
                    continue
                limited.append(p)
                per_task[p.parent.name] += 1
            candidates = limited
        if strict_counts and count_before_limit != EXPECTED_COUNTS[origin]:
            raise ValueError(f'{origin}: expected {EXPECTED_COUNTS[origin]} clean PKLs, '
                             f'found {count_before_limit}; review source selection')
        if not candidates:
            raise RuntimeError(f'No source PKLs selected from {root}')
        print(f'{origin}: selected {len(candidates)} PKLs', flush=True)
        for p in candidates:
            check_json(p)
            found.append((origin, p))
    # First source then second, sorted per task and original filename.
    per_task_num = Counter()
    jobs = []
    for origin, p in sorted(found, key=lambda x: (x[1].parent.name, x[0], x[1].name)):
        task = p.parent.name
        index = per_task_num[task]
        per_task_num[task] += 1
        rel = Path(task) / f'traj_{index:03d}.pkl'
        jobs.append((origin, p, rel))
    if len(set(job[2] for job in jobs)) != len(jobs):
        raise AssertionError('Output filename collision')
    return jobs


def write_manifest_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='',
              dir=path.parent, prefix='.manifest.', suffix='.tmp', delete=False) as f:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--tasks', type=int, nargs='+', default=list(range(16)))
    parser.add_argument('--limit-per-task', type=int, default=None,
                        help='Pilot: process up to N trajectories per task PER SOURCE')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate and compute transformations but write nothing')
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly replace output files; original sources untouched')
    args = parser.parse_args()
    if args.limit_per_task is not None and args.limit_per_task <= 0:
        parser.error('--limit-per-task must be positive')
    if any(t not in range(16) for t in args.tasks):
        parser.error('tasks must be integers from 0 to 15')
    dest = args.destination.expanduser().resolve()
    for origin, root in SOURCES:
        root = root.resolve()
        if root == dest or root in dest.parents or dest in root.parents:
            parser.error(f'Destination and source must be separate: {root}')

    strict_counts = args.limit_per_task is None and set(args.tasks) == set(range(16))
    jobs = gather(args.limit_per_task, set(args.tasks), strict_counts)
    print(f'\nTotal selected: {len(jobs)}; destination={dest}; dry-run={args.dry_run}', flush=True)

    # Pre-flight all output paths before any changes.
    if not args.dry_run and not args.overwrite:
        preexisting = [str(dest / rel) for _, _, rel in jobs if (dest / rel).exists()
                       or (dest / rel.with_suffix('.json')).exists()]
        if (dest / 'manifest.csv').exists():
            preexisting.append(str(dest / 'manifest.csv'))
        if preexisting:
            raise FileExistsError('Output already exists (use a new destination or '
                                  '--overwrite only if intentional): ' + str(preexisting[:5]))

    manifest_rows = []
    source_counter = Counter()
    all_frames = 0
    for n, (origin, src, rel) in enumerate(jobs, 1):
        target = dest / rel
        with src.open('rb') as f:
            payload = pickle.load(f)
        result, signatures = transform_payload(payload, src, origin, rel)
        all_frames += result['frames']
        source_counter[origin] += 1
        if not args.dry_run:
            dump_atomic(payload, target, signatures, result['frames'])
            copy_json_atomic(src.with_suffix('.json'), target.with_suffix('.json'))
        row = {
            'output_pkl': rel.as_posix(),
            'output_json': rel.with_suffix('.json').as_posix(),
            'source_dataset': origin,
            'source_pkl': f'{origin}/{src.relative_to(dict(SOURCES)[origin]).as_posix()}',
            'source_json': f'{origin}/{src.relative_to(dict(SOURCES)[origin]).with_suffix(".json").as_posix()}',
            'task_id': int(rel.parent.name.split('_')[1]),
            'frames': result['frames'],
            'transitions': result['transitions'],
            'close_command_step': result['gripper_close_command_step'],
            'close_observed_step': result['gripper_close_observed_step'],
            'mean_delta_mm': f'{result["mean_step_mm"]:.6f}',
            'max_delta_mm': f'{result["max_step_mm"]:.6f}',
            'max_position_reconstruction_error_mm': f'{result["pos_roundtrip_max_mm"]:.9f}',
            'max_rotation_reconstruction_error_deg': f'{result["rot_roundtrip_max_deg"]:.9f}',
        }
        manifest_rows.append(row)
        print(f'[{n:03d}/{len(jobs):03d}] {"CHECK" if args.dry_run else "OK"} '
              f'{origin}/{src.parent.name}/{src.name} -> {rel} '
              f'T={result["frames"]} maxXYZ={result["max_step_mm"]:.2f}mm', flush=True)
        del payload
        gc.collect()

    if not args.dry_run:
        write_manifest_atomic(dest / 'manifest.csv', manifest_rows)
    print('\n=== DONE ===')
    print('Source counts:', dict(source_counter))
    print('Trajectories:', len(jobs), 'total frames:', all_frames)
    print('Destination:', dest)
    print('Manifest:', 'dry-run (not written)' if args.dry_run else dest / 'manifest.csv')


if __name__ == '__main__':
    main()
