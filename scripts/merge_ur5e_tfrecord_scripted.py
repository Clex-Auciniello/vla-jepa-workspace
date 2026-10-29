#!/usr/bin/env python3
"""Merge UR5e TFRecord episodes and converted scripted PKLs into canonical RGB PKLs.

OUTPUT (per task):
  task_XX/tfrecord_traj_000.pkl ... tfrecord_traj_039.pkl
  task_XX/scripted_traj_000.pkl ... scripted_traj_019.pkl
Each output is a dict containing a savers.Trajectory with _data 5-tuples.
Observation schema is identical for both sources:
  EEF_state (6,) float32: xyz + Euler XYZ (base_link)
  action_world (7,) float32: EEF_state + observed binary gripper state
  gripper_state (2,) float32
  joint_positions, joint_velocities (6,) float32
  camera_front_image, camera_gripper_image (224,224,3) uint8 in RGB order
Actions are (7,) float32, XYZ and Euler delta scaled by 1/0.05, gripper=0/20.

TFRecord correction exactly follows the supplied Interleave preprocessing:
  repair pose action[0] using EEF_state[0] -> EEF_state[1]
  repair pose action[-2] using EEF_state[-2] -> EEF_state[-1]
  leave their gripper components unchanged
  keep last action unchanged and require all seven values to be zero
  do not alter interior actions

Scripted source is assumed already converted to scaled delta actions by
convert_scripted_to_delta_fixed.py, and RGB camera data is BGR-ordered.
We decode any JPEG via OpenCV (BGR), crop/resize front, resize wrist, then
convert BGR->RGB. This is configurable via --scripted-color-order.

No changes to input files. Only load trusted pickle files.
Images are stored as raw uint8 RGB arrays: bypass Trajectory.append() because
savers.Trajectory.append() JPEG-encodes via OpenCV, which would alter colors.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import savers  # Needed for unpickling and creating savers.Trajectory

TRAIN_TASKS = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)
SCALE = 0.05
SCALE = 0.05
GRIPPER_THRESHOLD = 0.38
FRONT_CROP_MARGINS = (0, 10, 140, 90)  # top, bottom, left, right
IMG_SIZE = (224, 224)
OB_KEYS = (
    'EEF_state', 'action_world', 'gripper_state',
    'joint_positions', 'joint_velocities',
    'camera_front_image', 'camera_gripper_image',
)

DEFAULT_ROOT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic')


def as_float(value, shape: tuple, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite shape {shape}, got {a.shape}')
    return a


def decode_tfrecord_image(encoded: bytes) -> np.ndarray:
    """TFRecord source image is already a 224x224 RGB JPEG."""
    with Image.open(io.BytesIO(encoded)) as im:
        a = np.asarray(im.convert('RGB'), dtype=np.uint8).copy()
    if a.shape != (224, 224, 3):
        raise ValueError(f'Expected TFRecord image (224,224,3), got {a.shape}')
    return a


def scripted_image(value, *, crop_front: bool, input_order: str) -> np.ndarray:
    """Read either raw or JPEG scripted image; output uint8 RGB 224x224."""
    a = np.asarray(value)
    if a.ndim == 3 and a.shape[2] == 3:
        bgr_or_rgb = a
    elif a.dtype == np.uint8 and (a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1)):
        # JPEG/PNG are decoded by OpenCV into BGR, irrespective of input_order.
        bgr_or_rgb = cv2.imdecode(a.reshape(-1), cv2.IMREAD_COLOR)
        if bgr_or_rgb is None:
            raise ValueError('cv2.imdecode failed on encoded scripted image')
        input_order = 'bgr'
    else:
        raise ValueError(f'Unsupported scripted RGB image: {a.shape}, {a.dtype}')
    if bgr_or_rgb.dtype != np.uint8:
        raise ValueError(f'Expected uint8 scripted RGB, got {bgr_or_rgb.dtype}')
    if crop_front:
        if bgr_or_rgb.shape[:2] != (376, 672):
            raise ValueError(
                f'Scripted front resolution changed: expected (376,672), got {bgr_or_rgb.shape[:2]}'
            )
        top, bottom, left, right = FRONT_CROP_MARGINS
        h, w = bgr_or_rgb.shape[:2]
        bgr_or_rgb = bgr_or_rgb[top:h-bottom, left:w-right]
        if bgr_or_rgb.size == 0:
            raise ValueError('Empty front crop')
    resized = cv2.resize(bgr_or_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if input_order == 'bgr' else resized
    )


def pose_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[XYZ, Euler XYZ] -> 6D scaled base_link delta; exactly source code's formula."""
    p0 = np.asarray(a, dtype=np.float64)
    p1 = np.asarray(b, dtype=np.float64)
    r0 = Rotation.from_euler('xyz', p0[3:6]).as_matrix()
    r1 = Rotation.from_euler('xyz', p1[3:6]).as_matrix()
    dr = Rotation.from_matrix(r1 @ r0.T).as_euler('xyz')
    return np.concatenate([(p1[:3] - p0[:3]) / SCALE, dr / SCALE]).astype(np.float32)


def empty_trajectory():
    """Avoid Trajectory.append, which JPEG encodes images via cv2.imencode."""
    return savers.Trajectory(config_str=None)


def add_step(traj, obs, action, *, reward, done, info):
    if set(obs) != set(OB_KEYS):
        raise ValueError(f'Noncanonical observation keys: {sorted(set(obs)^set(OB_KEYS))}')
    for k in ('camera_front_image', 'camera_gripper_image'):
        if obs[k].shape != (224,224,3) or obs[k].dtype != np.uint8:
            raise ValueError(f'Invalid {k}: {obs[k].shape} {obs[k].dtype}')
    a = as_float(action, (7,), 'action')
    # savers.Trajectory native storage: obs,reward,done,info,action.
    traj._data.append((obs, float(reward), bool(done), info, a.copy()))
    traj._raw_state.append(None)


def encode_value(feature, *, name):
    if feature.WhichOneof('kind') != 'bytes_list' or len(feature.bytes_list.value) != 1:
        raise ValueError(f'{name}: expected one bytes_list element')
    return feature.bytes_list.value[0].decode('utf-8')


def feature_numeric(features, key: str, T: int, components: int, *, kind='float_list'):
    if key not in features:
        raise KeyError(f'Missing TFRecord feature {key}')
    f = features[key]
    actual_kind = f.WhichOneof('kind')
    if actual_kind != kind:
        raise ValueError(f'Feature {key}: expected {kind}, got {actual_kind}')
    values = (f.float_list.value if kind == 'float_list' else f.int64_list.value)
    arr = np.asarray(values, dtype=np.float32 if kind == 'float_list' else np.int64)
    if arr.size != T * components:
        raise ValueError(f'Feature {key}: expected {T * components}, got {arr.size}')
    result = arr.reshape(T, components)
    if not np.isfinite(result).all():
        raise ValueError(f'Feature {key} contains nonfinite numbers')
    return result


def tfrecord_episode(example, *, task, episode_index, input_file, instruction):
    f = example.features.feature
    actual_instruction = encode_value(f['language_instruction'], name='language_instruction')
    if actual_instruction != instruction:
        raise ValueError(f'Task {task} contains inconsistent instruction {actual_instruction!r}')
    if 'task_name' in f and encode_value(f['task_name'], name='task_name') != 'pick_place':
        raise ValueError(f'Unexpected task_name in {input_file}')

    T = len(f['steps/observation/camera_front_image'].bytes_list.value)
    if T < 2:
        raise ValueError(f'Episode with {T} frames')
    front_blobs = f['steps/observation/camera_front_image'].bytes_list.value
    gripper_blobs = f['steps/observation/camera_gripper_image'].bytes_list.value
    if len(gripper_blobs) != T:
        raise ValueError('Front/gripper frame count mismatch')

    state = feature_numeric(f, 'steps/observation/EEF_state', T, 6)
    grip = feature_numeric(f, 'steps/observation/gripper_state', T, 2)
    world = feature_numeric(f, 'steps/observation/action_world', T, 7)
    joints = feature_numeric(f, 'steps/observation/joint_positions', T, 6)
    vel = feature_numeric(f, 'steps/observation/joint_velocities', T, 6)
    action = feature_numeric(f, 'steps/action', T, 7).copy()
    reward = feature_numeric(f, 'steps/reward/reward', T, 1).reshape(-1)
    disc = feature_numeric(f, 'steps/discount/discount', T, 1).reshape(-1)
    first = feature_numeric(f, 'steps/is_first', T, 1, kind='int64_list').reshape(-1)
    last = feature_numeric(f, 'steps/is_last', T, 1, kind='int64_list').reshape(-1)
    terminal = feature_numeric(f, 'steps/is_terminal', T, 1, kind='int64_list').reshape(-1)

    if not (first[0] == 1 and last[-1] == 1):
        raise ValueError('Invalid first/last flags')
    if not np.allclose(action[-1], 0, atol=1e-7):
        raise ValueError('Source TFRecord last action is not zero; refusing to silently alter it')
    if not np.isin(np.round(action[:, 6]).astype(int), [0, 20]).all() or not np.allclose(
        action[:,6], np.round(action[:,6]), atol=1e-6
    ):
        raise ValueError('Unexpected TFRecord gripper action values')

    # ONLY TWO TRANSITIONS REPAIRED. Gripper value action[t,6] unchanged.
    diagnostics = {}
    for idx, name in ((0, 'first'), (T - 2, 'penultimate')):
        before = action[idx].copy()
        corrected = pose_delta(state[idx], state[idx + 1])
        action[idx, :6] = corrected
        if action[idx, 6] != before[6]:
            raise AssertionError('Correction unexpectedly changed gripper')
        err_mm = np.linalg.norm(state[idx, :3].astype(np.float64) + before[:3] * SCALE - state[idx+1,:3]) * 1000
        diag = {
            'index': int(idx), 'error_before_position_mm': float(err_mm),
            'delta_action_pose_change_l2': float(np.linalg.norm(before[:6] - corrected)),
        }
        diagnostics[name] = diag
        if not np.allclose(
            state[idx,:3] + action[idx,:3] * SCALE, state[idx+1,:3], atol=2e-6
        ):
            raise AssertionError(f'Pose delta translation repair failed at index {idx}')
    if not np.array_equal(action[-1], np.zeros(7, dtype=np.float32)):
        raise AssertionError('Last action changed')

    traj = empty_trajectory()
    for t in range(T):
        obs = {
            'EEF_state': state[t].copy(),
            'action_world': world[t].copy(),
            'gripper_state': grip[t].copy(),
            'joint_positions': joints[t].copy(),
            'joint_velocities': vel[t].copy(),
            'camera_front_image': decode_tfrecord_image(front_blobs[t]),
            'camera_gripper_image': decode_tfrecord_image(gripper_blobs[t]),
        }
        add_step(traj, obs, action[t], reward=reward[t], done=bool(terminal[t]), info={
            'discount': float(disc[t]), 'is_first': bool(first[t]),
            'is_last': bool(last[t]), 'is_terminal': bool(terminal[t]),
        })

    return {
        'traj': traj,
        'task_id': f'{task:02d}',
        'traj_number': episode_index,
        'source': 'tfrecord',
        'source_file': str(input_file),
        'task_name': 'pick_place',
        'language_instruction': instruction,
        'variation_id': int(f['variation_id'].int64_list.value[0]) if 'variation_id' in f else None,
        'merge_preprocessing': {
            'image_color_order': 'RGB', 'image_shape': [224,224,3],
            'action_scale_factor': SCALE,
            'tfrecord_pose_repaired_indices': [0, T-2],
            'tfrecord_gripper_unchanged': True,
            'last_action_kept_zero': True,
        },
    }, diagnostics


def scripted_episode(source: Path, *, task, instruction, input_order):
    with source.open('rb') as fh:
        payload = pickle.load(fh)
    traj_in = payload['traj']
    raw = getattr(traj_in, '_data', None)
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f'{source}: Trajectory._data must be a nonempty list of 5-tuples')
    T = len(raw)

    gripper_qpos = np.asarray([
        float(np.asarray(item[0]['gripper_qpos']).reshape(-1)[0])
        for item in raw
    ], dtype=np.float32)

    if not np.isfinite(gripper_qpos).all():
        raise ValueError(f'{source}: invalid gripper_qpos')

    observed_gripper = (gripper_qpos >= GRIPPER_THRESHOLD).astype(np.float32)

    traj_out = empty_trajectory()
    for t, item in enumerate(raw):
        if not isinstance(item, tuple) or len(item) != 5:
            raise ValueError(f'{source}: Trajectory._data[{t}] is not a 5-tuple')
        obs, reward, done, info, action = item
        a = as_float(action, (7,), f'{source}: action frame {t}')
        if not np.isclose(a[6], 0.0, atol=1e-5) and not np.isclose(a[6], 20.0, atol=1e-5):
            raise ValueError(f'{source}: invalid gripper action {a[6]}')
        if t == T-1 and not np.allclose(a, 0, atol=1e-6):
            raise ValueError(f'{source}: last action is not all zero')
        if t < T - 1:
            expected_gripper_action = observed_gripper[t + 1] / SCALE

            if not np.isclose(a[6], expected_gripper_action, atol=1e-5):
                raise ValueError(
                    f'{source}: action[{t}, 6] does not match gripper state[{t+1}]'
                )
        pos = as_float(obs['eef_pos'], (3,), 'eef_pos')
        quat = as_float(obs['eef_quat'], (4,), 'eef_quat')
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
            raise ValueError(f'{source}: invalid quaternion at step {t}')
        angles = Rotation.from_quat(quat.astype(np.float64)).as_euler('xyz').astype(np.float32)
        eef = np.concatenate([pos, angles]).astype(np.float32)
        # Stato corrente misurato del gripper.
        binary_grip = observed_gripper[t]

        gstate = np.asarray([0.0, binary_grip], dtype=np.float32)
        world = np.concatenate([eef, gstate[1:2]]).astype(np.float32)
        f_obs = {
            'EEF_state': eef,
            'action_world': world,
            'gripper_state': gstate,
            'joint_positions': as_float(obs['joint_pos'], (6,), 'joint_pos'),
            'joint_velocities': as_float(obs['joint_vel'], (6,), 'joint_vel'),
            'camera_front_image': scripted_image(obs['camera_front_image'], crop_front=True, input_order=input_order),
            'camera_gripper_image': scripted_image(obs['camera_gripper_image'], crop_front=False, input_order=input_order),
        }
        # Preserve available original reward and flags; make a canonical info dict.
        stepinfo = {
            'discount': 1.0,
            'is_first': t == 0,
            'is_last': t == T - 1,
            'is_terminal': bool(done) if done is not None else t == T - 1,
        }
        if t == T - 1 and not stepinfo['is_terminal']:
            # The task's last trajectory step is terminal in our imitation dataset.
            stepinfo['is_terminal'] = True
        add_step(traj_out, f_obs, a, reward=0.0 if reward is None else reward,
                 done=stepinfo['is_terminal'], info=stepinfo)

    stem_match = re.search(r'(\d+)$', source.stem)
    if not stem_match:
        raise ValueError(f'Cannot infer episode number from {source.name}')
    return {
        'traj': traj_out,
        'task_id': f'{task:02d}',
        'traj_number': int(stem_match.group(1)),
        'source': 'scripted',
        'source_file': str(source),
        'task_name': 'pick_place',
        'language_instruction': instruction,
        'variation_id': None,
        'merge_preprocessing': {
            'image_color_order': 'RGB', 'source_color_order': input_order.upper(),
            'front_crop_margins_top_bottom_left_right': list(FRONT_CROP_MARGINS),
            'camera_gripper_crop': False, 'image_shape': [224,224,3],
            'action_scale_factor': SCALE, 'scripted_action_already_delta': True,
            'scripted_gripper_state_proxy_source': 'observation.gripper_qpos',
            'scripted_gripper_threshold': GRIPPER_THRESHOLD,
            'scripted_gripper_action_target': 'next_frame_gripper_state',
        },
    }


def verify_payload(payload):
    steps = payload['traj']._data
    if len(steps) < 2:
        raise ValueError('Empty trajectory')
    for t, item in enumerate(steps):
        obs, reward, done, info, action = item
        if tuple(obs) != tuple(OB_KEYS):
            raise ValueError(f'{payload["source"]} frame {t}: incompatible observation schema')
        if any(obs[k].shape != (224,224,3) or obs[k].dtype != np.uint8 for k in ('camera_front_image','camera_gripper_image')):
            raise ValueError(f'Invalid image at frame {t}')
        if action.shape != (7,) or action.dtype != np.float32:
            raise ValueError(f'Invalid action at frame {t}')
    if not np.array_equal(np.asarray(steps[-1][4]), np.zeros(7, dtype=np.float32)):
        raise ValueError('Final action must be 7 zeros')
    visible = payload['traj'].get(0, decompress=False)
    if not isinstance(visible['obs']['camera_front_image'], np.ndarray):
        raise ValueError('Failed public get() raw RGB check')


def atomic_pickle(destination: Path, payload, *, overwrite):
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} already exists (use --overwrite)')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=destination.parent,
                                         prefix=f'.{destination.stem}_', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Verify the actual written PKL before replacing any destination.
        with tmp.open('rb') as f:
            restored = pickle.load(f)
        verify_payload(restored)
        os.replace(tmp, destination)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def preview(destination: Path, payload):
    """Preview RGB pixels without using cv2.imwrite (which assumes BGR)."""
    obs = payload['traj'].get(0, decompress=False)['obs']
    destination.mkdir(parents=True, exist_ok=True)
    label = f"task_{payload['task_id']}_{payload['source']}_{payload['traj_number']:03d}"
    for key, short in [('camera_front_image','front'), ('camera_gripper_image','gripper')]:
        Image.fromarray(obs[key], mode='RGB').save(destination / f'{label}_{short}.png')


def load_instructions(shards):
    import tensorflow as tf
    instructions = {}
    for task, shard in shards.items():
        first = next(iter(tf.data.TFRecordDataset(str(shard))), None)
        if first is None:
            raise ValueError(f'Empty shard {shard}')
        ex = tf.train.Example.FromString(first.numpy())
        instructions[task] = encode_value(ex.features.feature['language_instruction'], name='language_instruction')
    return instructions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tfrecord-root', type=Path, default=DEFAULT_ROOT/'pick_place')
    p.add_argument('--scripted-root', type=Path, default=DEFAULT_ROOT/'scripted_trajectories_delta')
    p.add_argument('--output-root', type=Path, default=DEFAULT_ROOT/'vla_jepa_offline_merged')
    p.add_argument('--scripted-color-order', choices=('bgr','rgb'), default='bgr',
                   help='Original decoded scripted image channel order. Default BGR.')
    p.add_argument('--tasks', type=int, nargs='+', default=list(TRAIN_TASKS),
                   help='Task IDs to process; default all 12 training tasks')
    p.add_argument('--limit-per-task', type=int, default=0,
                   help='For smoke test; 0 means full dataset')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Parse and validate but do not write files')
    p.add_argument('--skip-existing', action='store_true', help='Skip already-existing output PKLs')
    p.add_argument('--preview-dir', type=Path, default=None,
                   help='Save PNG RGB previews of first processed episode per source/task')
    args = p.parse_args()
    if args.skip_existing and args.overwrite:
        p.error('--overwrite and --skip-existing are mutually exclusive')
    if args.limit_per_task < 0:
        p.error('--limit-per-task cannot be negative')
    if not args.tasks or any(task not in TRAIN_TASKS for task in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        p.error('--tasks must be unique IDs among 1,2,3,4,6,7,8,9,11,12,13,14')

    import tensorflow as tf
    shards_list = sorted(args.tfrecord_root.glob('*-train.tfrecord-*'))
    if len(shards_list) != len(TRAIN_TASKS):
        raise RuntimeError(f'Expected 12 TFRecord shards, found {len(shards_list)}')
    shards = dict(zip(TRAIN_TASKS, shards_list))
    instructions = load_instructions(shards)
    print('Language instructions by task:')
    for task, instruction in instructions.items():
        print(f'  task_{task:02d}: {instruction}', flush=True)

    all_counts = defaultdict(lambda: defaultdict(int))
    repair_log = []
    for task in args.tasks:
        shard = shards[task]
        tf_count = 0
        print(f'\nTASK {task:02d}: {shard.name}', flush=True)
        for ei, record in enumerate(tf.data.TFRecordDataset(str(shard))):
            tf_count += 1
            if args.limit_per_task and tf_count > args.limit_per_task:
                break
            target = args.output_root / f'task_{task:02d}' / f'tfrecord_traj_{ei:03d}.pkl'
            if target.exists() and args.skip_existing:
                all_counts[task]['tfrecord_skipped'] += 1
                continue
            if target.exists() and not args.overwrite and not args.dry_run:
                raise FileExistsError(f'{target} exists. Use --overwrite, --skip-existing or new output-root.')
            ex = tf.train.Example.FromString(record.numpy())
            payload, diagnostic = tfrecord_episode(ex, task=task, episode_index=ei,
                                                  input_file=shard, instruction=instructions[task])
            verify_payload(payload)
            if not args.dry_run:
                atomic_pickle(target, payload, overwrite=args.overwrite)
                if args.preview_dir is not None and ei == 0:
                    preview(args.preview_dir, payload)
            all_counts[task]['tfrecord'] += 1
            repair_log.append({'task_id':f'{task:02d}','episode':ei, 'source_file':str(shard),
                               **{f'{where}_{key}':value for where, vals in diagnostic.items() for key,value in vals.items()}})
            if ei == 0 or (ei+1) % 10 == 0:
                print(f'  TFRecord converted {ei+1}', flush=True)
        if tf_count != 40 and args.limit_per_task == 0:
            raise RuntimeError(f'{shard}: expected 40 records, got {tf_count}')

        scripted_dir = args.scripted_root / f'task_{task:02d}'
        pkls = sorted(scripted_dir.glob('*.pkl'))
        if len(pkls) != 20:
            raise RuntimeError(f'{scripted_dir}: expected 20 PKLs, found {len(pkls)}')
        if args.limit_per_task:
            pkls = pkls[:args.limit_per_task]
        for i, source in enumerate(pkls):
            out = args.output_root / f'task_{task:02d}' / f'scripted_{source.name}'
            if out.exists() and args.skip_existing:
                all_counts[task]['scripted_skipped'] += 1
                continue
            if out.exists() and not args.overwrite and not args.dry_run:
                raise FileExistsError(f'{out} exists. Use --overwrite, --skip-existing or new output-root.')
            payload = scripted_episode(source, task=task, instruction=instructions[task],
                                       input_order=args.scripted_color_order)
            verify_payload(payload)
            if not args.dry_run:
                atomic_pickle(out, payload, overwrite=args.overwrite)
                if args.preview_dir is not None and i == 0:
                    preview(args.preview_dir, payload)
            all_counts[task]['scripted'] += 1
            if i == 0 or (i+1) % 10 == 0:
                print(f'  Scripted converted {i+1}', flush=True)

    print('\nSUMMARY', flush=True)
    for task in args.tasks:
        print(f"task_{task:02d}: {dict(all_counts[task])}", flush=True)
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root/'language_instructions_by_task.json').open('w') as f:
            json.dump({f'task_{k:02d}':v for k,v in instructions.items()}, f, indent=2)
        if repair_log:
            keys = list(repair_log[0])
            with (args.output_root/'tfrecord_boundary_repairs.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(repair_log)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
