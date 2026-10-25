#!/usr/bin/env python3
"""Convert scripted UR5e 8D absolute-pose PKLs into TFRecord-compatible 7D delta actions.

Preserves savers.Trajectory, observations (RGB/depth/proprio), flags, and metadata.
Only mutates action and renames the gripper camera RGB/depth keys in a NEW PKL.

Conventions verified against the robot TFRecord dataset:
  a[t, :3]   = (p[t+1] - p[t]) / 0.05, in base_link
  R_delta[t] = R[t+1] @ R[t].T, in base_link
  a[t, 3:6]  = scipy Rotation.from_matrix(R_delta).as_euler('xyz') / 0.05
  a[t, 6]    = old_absolute_gripper_command[t] / 0.05 (0 or 20)
  a[-1]      = 0 in all 7 dimensions
  quaternion convention: [x,y,z,w] (SciPy)

Use only on trusted PKL files. pickle.load executes Python objects.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import savers  # noqa: F401 -- required for deserializing savers.Trajectory


TRAIN_TASKS = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)
DEFAULT_SOURCE = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/scripted_trajectories')
DEFAULT_DEST = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/scripted_trajectories_delta')
SCALE = 0.05
IMAGE_NAMES = (
    'camera_front_image', 'camera_front_depth',
    'left_camera_image', 'left_camera_depth',
    'right_camera_image', 'right_camera_depth',
    'gripper_camera_image', 'gripper_camera_depth',
)
KEY_RENAMES = {
    'gripper_camera_image': 'camera_gripper_image',
    'gripper_camera_depth': 'camera_gripper_depth',
}


def as_shape(value, n: int, name: str, source: Path, t: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != n or not np.isfinite(array).all():
        raise ValueError(f'{source} step {t}: {name} must be finite ({n},), got {array.shape}')
    return array


def digest(value) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    sha = hashlib.sha256()
    sha.update(str(arr.shape).encode('ascii'))
    sha.update(str(arr.dtype).encode('ascii'))
    sha.update(arr.tobytes())
    return sha.hexdigest()


def visual_signatures(steps, names: tuple[str, ...]) -> dict:
    indices = sorted({0, len(steps) // 2, len(steps) - 1})
    signatures = {}
    for i in indices:
        obs = steps[i]['obs']
        for k in names:
            if k not in obs:
                raise KeyError(f'Frame {i}: camera/depth field absent: {k}')
            signatures[(i, KEY_RENAMES.get(k, k))] = digest(obs[k])
    return signatures


def mutable_steps(traj, src: Path):
    """Present tuple-backed Trajectory steps as mutable dictionaries.

    savers.Trajectory._data is a list of 5-tuples:
        (obs, reward, done, info, action)
    Trajectory.get(...) *creates* a dictionary view; its return value is not
    the stored representation. After converting actions, we MUST repack the
    dictionaries into 5-tuples, or future Trajectory.get() calls will fail.
    """
    raw = getattr(traj, '_data', None)
    if not isinstance(raw, list) or len(raw) != len(traj):
        raise TypeError(
            f'{src}: unsupported Trajectory._data storage {type(raw).__name__}, '
            f'len(_data)={len(raw) if hasattr(raw, "__len__") else "N/A"}. '
            'Inspect type(traj._data) and type(traj._data[0]) first.'
        )
    steps = []
    for i, item in enumerate(raw):
        if not isinstance(item, tuple) or len(item) != 5:
            raise TypeError(
                f'{src}: unexpected raw Trajectory step at index {i}: '
                f'{type(item).__name__} (expected 5-tuple)'
            )
        obs, reward, done, info, action = item
        if not isinstance(obs, dict) or action is None:
            raise TypeError(f'{src}: invalid observation/action at index {i}')
        steps.append({
            'obs': obs,
            'reward': reward,
            'done': done,
            'info': info,
            'action': action,
        })
    return steps


def repack_steps(traj, steps) -> None:
    """Write converted actions back in savers.Trajectory native format."""
    traj._data[:] = [
        (step['obs'], step['reward'], step['done'], step['info'], step['action'])
        for step in steps
    ]


def prepare_conversion(payload: dict, src: Path):
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise TypeError(f'{src}: expected dict with traj')
    traj = payload['traj']
    steps = mutable_steps(traj, src)
    T = len(steps)
    if T < 2:
        raise ValueError(f'{src}: only {T} frames; require at least 2')

    # Require the exact source camera schema and preserve all RGB/depth arrays bytewise.
    for t, step in enumerate(steps):
        obs = step['obs']
        for k in IMAGE_NAMES:
            if k not in obs:
                raise KeyError(f'{src} step {t}: missing expected RGB/depth field {k}')
        for old, new in KEY_RENAMES.items():
            if new in obs and old != new:
                raise ValueError(f'{src} step {t}: both {old} and {new} exist')

    visual_before = visual_signatures(steps, IMAGE_NAMES)

    xyz = np.stack([as_shape(s['obs']['eef_pos'], 3, 'eef_pos', src, t)
                    for t, s in enumerate(steps)])
    quat = np.stack([as_shape(s['obs']['eef_quat'], 4, 'eef_quat', src, t)
                     for t, s in enumerate(steps)])
    old_actions = np.stack([as_shape(s['action'], 8, 'absolute action', src, t)
                            for t, s in enumerate(steps)])

    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-3):
        raise ValueError(f'{src}: input quaternions do not have unit norm')

    # Dataset inspection established old 8D action == observed current 7D pose + gripper.
    if not np.allclose(old_actions[:, :3], xyz, rtol=0, atol=1e-6):
        raise ValueError(f'{src}: old action XYZ differs from observed eef_pos')
    if not np.allclose(old_actions[:, 3:7], quat, rtol=0, atol=1e-5):
        raise ValueError(f'{src}: old action quaternion differs from observed eef_quat')

    old_gripper = old_actions[:, 7]
    if not np.isclose(old_gripper, np.round(old_gripper), atol=1e-8).all() or \
            not np.isin(np.round(old_gripper).astype(np.int64), [0, 1]).all():
        raise ValueError(f'{src}: unexpected gripper command(s): {np.unique(old_gripper)}')

    # World-frame (base_link) action; SciPy as_quat expects XYZW, verified from data.
    R = Rotation.from_quat(quat)
    R_delta = R[1:] * R[:-1].inv()
    d_xyz = (xyz[1:] - xyz[:-1]) / SCALE
    d_euler = R_delta.as_euler('xyz', degrees=False) / SCALE

    delta_action = np.zeros((T, 7), dtype=np.float32)
    delta_action[:-1, 0:3] = d_xyz
    delta_action[:-1, 3:6] = d_euler
    delta_action[:-1, 6] = old_gripper[:-1] / SCALE
    # delta_action[-1, :] stays all zero, including the gripper.

    # Verify the EXACT actions we will store, including float32 rounding.
    rebuilt_xyz = xyz[:-1] + delta_action[:-1, :3].astype(np.float64) * SCALE
    pos_err_m = np.linalg.norm(rebuilt_xyz - xyz[1:], axis=1)
    R_rebuilt = Rotation.from_euler('xyz',
                                    delta_action[:-1, 3:6].astype(np.float64) * SCALE) * R[:-1]
    rot_err_deg = np.rad2deg((R_rebuilt.inv() * R[1:]).magnitude())
    max_pos_mm = float(np.max(pos_err_m) * 1000)
    max_rot_deg = float(np.max(rot_err_deg))
    if max_pos_mm > 0.01 or max_rot_deg > 0.001:
        raise ValueError(f'{src}: reconstruction mismatch pos={max_pos_mm:.8g} mm '
                         f'rot={max_rot_deg:.8g} deg')

    # Only replace action and rename gripper camera; preserve everything else.
    for t, step in enumerate(steps):
        step['action'] = delta_action[t].copy()
        for old_key, new_key in KEY_RENAMES.items():
            step['obs'][new_key] = step['obs'].pop(old_key)

    # Critical: Trajectory.get() unpacks native 5-tuples, not dictionaries.
    repack_steps(traj, steps)

    payload['action_conversion'] = {
        'source_format': 'scripted_absolute_xyz_xyzw_gripper_8d',
        'target_format': 'base_link_delta_xyz_euler_xyz_gripper_7d',
        'scale_factor': SCALE,
        'quaternion_order': 'xyzw',
        'rotation_composition': 'R_next = R_delta @ R_current',
        'action_alignment': 'action[t] encodes pose[t] to pose[t+1]',
        'gripper_command': '0=open,20=close (old 0/1 scaled); last action zero',
        'camera_renames': dict(KEY_RENAMES),
    }
    stats = {
        'frames': T,
        'max_pos_reconstruction_mm': max_pos_mm,
        'max_rot_reconstruction_deg': max_rot_deg,
        'gripper_closed_actions': int(np.count_nonzero(delta_action[:, 6] == 20)),
        'mean_step_cm': float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).mean() * 100),
    }
    return visual_before, stats


def verify_saved(path: Path, signatures: dict, expected_len: int) -> None:
    with path.open('rb') as f:
        payload = pickle.load(f)
    traj = payload['traj']
    steps = mutable_steps(traj, path)
    if len(steps) != expected_len:
        raise ValueError(f'{path}: frame count changed: {len(steps)} != {expected_len}')
    if not np.array_equal(np.asarray(steps[-1]['action']), np.zeros(7, dtype=np.float32)):
        raise ValueError(f'{path}: last action not zero')
    for i, step in enumerate(steps):
        action = np.asarray(step['action'])
        if action.shape != (7,) or action.dtype != np.float32 or not np.isfinite(action).all():
            raise ValueError(f'{path}, frame {i}: invalid saved action')
        obs = step['obs']
        if 'gripper_camera_image' in obs or 'gripper_camera_depth' in obs:
            raise ValueError(f'{path}, frame {i}: old camera key still present')
        if 'camera_gripper_image' not in obs or 'camera_gripper_depth' not in obs:
            raise ValueError(f'{path}, frame {i}: new gripper camera key missing')
    # Verify the class public accessor, not just the private storage we mutated.
    for i in sorted({0, len(steps) // 2, len(steps) - 1}):
        try:
            visible = traj.get(i, decompress=False)
        except TypeError:
            visible = traj.get(i)
        if not np.array_equal(np.asarray(visible['action']), np.asarray(steps[i]['action'])):
            raise ValueError(f'{path}, frame {i}: Trajectory.get() returns a different action')
        if 'camera_gripper_image' not in visible['obs']:
            raise ValueError(f'{path}, frame {i}: new image key missing from Trajectory.get()')
    for (i, key), expected in signatures.items():
        if digest(steps[i]['obs'][key]) != expected:
            raise ValueError(f'{path}, frame {i}: image/depth changed: {key}')


def convert_file(source: Path, target: Path, dry_run: bool, overwrite: bool, copy_json: bool):
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists() and not overwrite and not dry_run:
        return 'SKIP', None
    if source.resolve() == target.resolve():
        raise RuntimeError('Input and output are the same file; refusing conversion')

    with source.open('rb') as f:
        payload = pickle.load(f)
    signatures, stats = prepare_conversion(payload, source)

    if dry_run:
        return 'DRY-RUN', stats

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile('wb', prefix=f'.{target.stem}.', suffix='.tmp',
                                         dir=target.parent, delete=False) as f:
            tmp_path = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

        del payload
        gc.collect()
        verify_saved(tmp_path, signatures, stats['frames'])
        os.replace(tmp_path, target)  # atomic commit, only after successful verification
        tmp_path = None

        if copy_json:
            json_src = source.with_suffix('.json')
            if json_src.exists():
                shutil.copy2(json_src, target.with_suffix('.json'))
        return 'OK', stats
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--destination', type=Path, default=DEFAULT_DEST)
    p.add_argument('--tasks', nargs='+', type=int, default=list(TRAIN_TASKS),
                   help='Task IDs (default: the 12 training tasks; held-out 00,05,10,15 excluded)')
    p.add_argument('--limit', type=int, default=None, help='Max PKLs per task (for pilot)')
    p.add_argument('--dry-run', action='store_true', help='Validate and report; write nothing')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing converted PKLs')
    p.add_argument('--no-copy-json', dest='copy_json', action='store_false',
                   help='Do not copy original matching JSON metadata')
    args = p.parse_args()

    src_root = args.source.expanduser().resolve()
    dst_root = args.destination.expanduser().resolve()
    if src_root == dst_root or src_root in dst_root.parents or dst_root in src_root.parents:
        raise SystemExit('Source and destination must be separate, non-nested directories')
    if not src_root.is_dir():
        raise SystemExit(f'Input directory does not exist: {src_root}')
    if args.limit is not None and args.limit <= 0:
        raise SystemExit('--limit must be positive')

    processed = skipped = 0
    for task in args.tasks:
        folder = src_root / f'task_{task:02d}'
        files = sorted(folder.glob('*.pkl'))
        if not files:
            raise SystemExit(f'No PKLs in {folder}')
        if args.limit is not None:
            files = files[:args.limit]
        print(f'\nTASK {task:02d}: {len(files)} source PKLs', flush=True)
        for source in files:
            target = dst_root / folder.name / source.name
            status, stats = convert_file(source, target, args.dry_run, args.overwrite,
                                         args.copy_json)
            if status == 'SKIP':
                skipped += 1
                print(f'  SKIP {source.name}: already exists', flush=True)
                continue
            processed += 1
            print(f'  {status} {source.name}: T={stats["frames"]} '
                  f'mean_step={stats["mean_step_cm"]:.3f}cm '
                  f'pos_err={stats["max_pos_reconstruction_mm"]:.8f}mm '
                  f'rot_err={stats["max_rot_reconstruction_deg"]:.8f}deg '
                  f'closed_steps={stats["gripper_closed_actions"]}', flush=True)
    print(f'\nDONE: converted/checked={processed}, skipped={skipped}, dry_run={args.dry_run}',
          flush=True)


if __name__ == '__main__':
    main()
