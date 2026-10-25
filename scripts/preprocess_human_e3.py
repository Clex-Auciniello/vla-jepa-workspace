#!/usr/bin/env python3
"""Create E3 human-video PKLs in the same *visual* convention as E4 robot PKLs.

Inputs:
  human_root/task_XX/trajNNN.pkl                 - trusted savers.Trajectory
  robot_root/task_XX/tfrecord_traj_000.pkl       - task-specific instruction
Outputs:
  output_root/task_XX/human_traj_NNN.pkl         - trusted savers.Trajectory

Training task IDs: 01 02 03 04 06 07 08 09 11 12 13 14 only.
Preserves ALL episodes and ALL frames (regardless of video length).
Each output observation contains ONLY camera_front_image, RGB uint8 (224,224,3).

Input image: JPEG encoded by savers, decoded with cv2.imdecode. The RGB-preview
verification on this particular human dataset showed that the decoded pixel
array, when displayed as RGB *without swapping channels*, has correct colors.
Therefore, cv2.imdecode output is treated as RGB by default; do NOT blindly
apply cv2.COLOR_BGR2RGB here. --decoded-color-order bgr is an opt-in override.

Front crop: (top,bottom,left,right) = (0,10,140,90), applied before resize.
OpenCV resize: INTER_AREA, size=(224,224), exactly like E4 merge pipeline.
No fake proprioception, actions, gripper state, padding, truncation, or windows.

RGB arrays are stored directly in Trajectory._data native 5-tuples, because
savers.Trajectory.append() would JPEG-encode with cv2 and change colors.

WARNING: Only open PKLs from trusted local data; pickle executes Python code.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Both pickle module identities are needed: old human PKLs were stored with
# multi_task_il.datasets.savers.Trajectory, while E4 merged robot PKLs use
# savers.Trajectory. Configure PYTHONPATH as shown in the accompanying commands.
import savers  # noqa: F401
import multi_task_il.datasets.savers  # noqa: F401

TRAIN_TASKS = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)
CROP = (0, 10, 140, 90)
TARGET_SIZE = (224, 224)
EXPECTED_INPUT_HW = (376, 672)
DEFAULT_HUMAN = Path('/mnt/beegfs/a.cuciniello9/datasets/human/human_rgb_pick_place')
DEFAULT_ROBOT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/vla_jepa_offline_merged')
DEFAULT_OUTPUT = Path('/mnt/beegfs/a.cuciniello9/datasets/human/human_rgb_pick_place_preprocessed')


def load_pickle(path: Path):
    with path.open('rb') as fh:
        return pickle.load(fh)


def get_raw_steps(payload: dict, path: Path):
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise ValueError(f'{path}: expected dict with traj')
    traj = payload['traj']
    raw = getattr(traj, '_data', None)
    if not isinstance(raw, list) or len(raw) != len(traj) or len(raw) == 0:
        raise ValueError(f'{path}: unsupported/empty Trajectory._data')
    for t, item in enumerate(raw):
        if not isinstance(item, tuple) or len(item) != 5:
            raise ValueError(f'{path}: _data[{t}] is not a native 5-tuple')
    return traj, raw


def read_instruction(robot_root: Path, task: int) -> str:
    task_dir = robot_root / f'task_{task:02d}'
    path = task_dir / 'tfrecord_traj_000.pkl'
    if not path.is_file():
        candidates = sorted(task_dir.glob('tfrecord_traj_*.pkl'))
        if not candidates:
            raise FileNotFoundError(f'No TFRecord-derived PKL to source task instruction: {task_dir}')
        path = candidates[0]
    payload = load_pickle(path)
    if str(payload.get('task_id', '')).zfill(2) != f'{task:02d}':
        raise ValueError(f'{path}: task_id does not match task_{task:02d}')
    instruction = payload.get('language_instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f'{path}: missing nonempty language_instruction')
    return instruction


def decode_and_resize_front(value, *, path: Path, t: int, decoded_color_order: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        raise ValueError(f'{path} step {t}: expected uint8 camera image; got {arr.dtype}')
    if arr.ndim == 3 and arr.shape[2] == 3:
        image = arr
    elif arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        image = cv2.imdecode(np.ascontiguousarray(arr.reshape(-1)), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'{path} step {t}: JPEG decode failed')
    else:
        raise ValueError(f'{path} step {t}: unsupported image shape {arr.shape}')
    if tuple(image.shape) != (*EXPECTED_INPUT_HW, 3):
        raise ValueError(f'{path} step {t}: expected (376,672,3), got {image.shape}')
    top, bottom, left, right = CROP
    height, width = image.shape[:2]
    cropped = image[top:height-bottom, left:width-right]
    if cropped.shape != (366, 442, 3):
        raise AssertionError(f'{path} step {t}: unexpected crop shape {cropped.shape}')
    resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    if decoded_color_order == 'bgr':
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def new_trajectory_of_same_class(original):
    cls = original.__class__
    cfg = getattr(original, 'config_str', None)
    converted = cls(config_str=cfg)
    if not isinstance(getattr(converted, '_data', None), list):
        raise TypeError(f'Unexpected trajectory type {cls}')
    return converted


def convert_one(src: Path, *, task: int, instruction: str, decoded_color_order: str):
    original = load_pickle(src)
    traj, steps = get_raw_steps(original, src)
    stored_task = str(original.get('task_id', '')).zfill(2)
    if stored_task != f'{task:02d}':
        raise ValueError(f'{src}: task_id {stored_task!r}, expected {task:02d}')
    if 'len' in original and int(original['len']) != len(steps):
        raise ValueError(f'{src}: metadata len differs from trajectory length')
    m = re.search(r'(\d+)$', src.stem)
    if m is None:
        raise ValueError(f'{src}: expected a name ending in trajectory number')
    episode_num = int(m.group(1))

    result_traj = new_trajectory_of_same_class(traj)
    raw_state = getattr(traj, '_raw_state', None)
    if isinstance(raw_state, list) and len(raw_state) != len(steps):
        raise ValueError(f'{src}: unexpected _raw_state length')
    if raw_state is not None and not isinstance(raw_state, list):
        raise ValueError(f'{src}: _raw_state is not a list')

    done_flags = []
    for t, item in enumerate(steps):
        obs, reward, done, info, action = item
        if not isinstance(obs, dict) or set(obs) != {'camera_front_image'}:
            raise ValueError(f'{src} step {t}: unexpected obs keys {list(obs) if isinstance(obs, dict) else type(obs)}')
        if action is not None:
            raise ValueError(f'{src} step {t}: unexpected human action; refusing to discard labels')
        if done is None:
            raise ValueError(f'{src} step {t}: missing done flag')
        done_flag = bool(done)
        done_flags.append(done_flag)
        image_rgb = decode_and_resize_front(
            obs['camera_front_image'], path=src, t=t,
            decoded_color_order=decoded_color_order)
        # IMPORTANT: store decoded RGB directly (Trajectory.append would JPEG-encode).
        result_traj._data.append((
            {'camera_front_image': image_rgb},
            copy.deepcopy(reward),
            done_flag,
            copy.deepcopy(info),
            None,
        ))
        result_traj._raw_state.append(
            copy.deepcopy(raw_state[t]) if raw_state is not None else None)

    if not done_flags[-1] or any(done_flags[:-1]):
        raise ValueError(f'{src}: expected done only on last frame')
    if len(result_traj._data) != len(steps):
        raise AssertionError('Frame count changed during preprocessing')

    # Copy the original top-level metadata; use the original human Trajectory
    # type to retain pickle compatibility with the source codebase.
    converted_payload = {k: copy.deepcopy(v) for k, v in original.items() if k != 'traj'}
    converted_payload.update({
        'traj': result_traj,
        'len': len(steps),
        'task_id': f'{task:02d}',
        'traj_number': episode_num,
        'source': 'human',
        'source_file': str(src),
        'task_name': 'pick_place',
        'language_instruction': instruction,
        'preprocessing': {
            'camera_front_image': 'RGB uint8 224x224x3 (uncompressed)',
            'decoded_pixel_order_assumed': decoded_color_order.upper(),
            'front_crop_margins_top_bottom_left_right': list(CROP),
            'resize_interpolation': 'cv2.INTER_AREA',
            'all_original_frames_preserved': True,
            'action_labels_added': False,
        },
    })
    return converted_payload, episode_num


def verify_output(payload, *, expected_task: int, expected_instruction: str, expected_frames: int):
    if payload['task_id'] != f'{expected_task:02d}':
        raise ValueError('Saved PKL: task_id mismatch')
    if payload.get('language_instruction') != expected_instruction:
        raise ValueError('Saved PKL: language_instruction mismatch')
    traj, steps = get_raw_steps(payload, Path('[saved PKL]'))
    if len(steps) != expected_frames or int(payload['len']) != expected_frames:
        raise ValueError('Saved PKL: frame count mismatch')
    for i, item in enumerate(steps):
        obs, reward, done, info, action = item
        if set(obs) != {'camera_front_image'} or action is not None:
            raise ValueError(f'Saved PKL: unexpected action/obs keys at {i}')
        a = obs['camera_front_image']
        if not isinstance(a, np.ndarray) or a.shape != (224,224,3) or a.dtype != np.uint8:
            raise ValueError(f'Saved PKL: invalid RGB image at {i}')
        if bool(done) != (i == len(steps)-1):
            raise ValueError(f'Saved PKL: wrong done at {i}')
    # Check public reader compatibility, without asking it to decompress an RGB array.
    public = traj.get(0, decompress=False)
    if public['obs']['camera_front_image'].shape != (224,224,3):
        raise ValueError('Saved PKL: public Trajectory.get invalid')


def save_atomically(destination: Path, payload, *, overwrite: bool):
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination}: output exists (use --overwrite or --skip-existing)')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=destination.parent, prefix='.human_', suffix='.tmp', delete=False) as out:
            temp_name = Path(out.name)
            pickle.dump(payload, out, protocol=pickle.HIGHEST_PROTOCOL)
        restored = load_pickle(temp_name)
        verify_output(restored, expected_task=int(payload['task_id']),
                      expected_instruction=payload['language_instruction'],
                      expected_frames=len(payload['traj']))
        os.replace(temp_name, destination)
    finally:
        if temp_name is not None:
            temp_name.unlink(missing_ok=True)


def previews(preview_dir: Path, payload):
    t = payload['traj']
    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = f'task_{payload["task_id"]}_human_{payload["traj_number"]:03d}'
    for frame in sorted({0, len(t)//2, len(t)-1}):
        rgb = t.get(frame, decompress=False)['obs']['camera_front_image']
        Image.fromarray(rgb, mode='RGB').save(preview_dir / f'{stem}_frame_{frame:03d}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human-root', type=Path, default=DEFAULT_HUMAN)
    parser.add_argument('--robot-root', type=Path, default=DEFAULT_ROBOT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--tasks', type=int, nargs='+', default=list(TRAIN_TASKS))
    parser.add_argument('--limit-per-task', type=int, default=0,
                        help='For testing, 0 means ALL trajectories')
    parser.add_argument('--expected-per-task', type=int, default=40)
    parser.add_argument('--decoded-color-order', choices=('rgb','bgr'), default='rgb',
                        help='How to interpret cv2.imdecode output based on visual preview. '
                             'For these particular human PKLs, correct preview was RGB as-is.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preview-dir', type=Path, default=None)
    parser.add_argument('--summary-csv', type=Path, default=None)
    args = parser.parse_args()
    if args.overwrite and args.skip_existing:
        parser.error('--overwrite and --skip-existing cannot be combined')
    if args.limit_per_task < 0:
        parser.error('--limit-per-task must be nonnegative')
    if len(set(args.tasks)) != len(args.tasks) or any(task not in TRAIN_TASKS for task in args.tasks):
        parser.error('Only train task IDs 01 02 03 04 06 07 08 09 11 12 13 14 are allowed')

    # Resolve task instructions only from E4 robot PKLs; no invented prompts.
    instruction_by_task = {task: read_instruction(args.robot_root, task) for task in args.tasks}
    print('Instructions loaded from E4 robot dataset:', flush=True)
    for task in args.tasks:
        print(f'  task_{task:02d}: {instruction_by_task[task]}', flush=True)

    rows = []
    total_episodes = 0
    total_frames = 0
    for task in args.tasks:
        task_dir = args.human_root / f'task_{task:02d}'
        inputs = sorted(task_dir.glob('*.pkl'))
        if len(inputs) != args.expected_per_task:
            raise ValueError(f'{task_dir}: expected {args.expected_per_task} PKLs, found {len(inputs)}')
        if args.limit_per_task:
            inputs = inputs[:args.limit_per_task]
        per_task_frames = []
        converted_count = 0
        skipped_count = 0
        filenames_seen = set()
        print(f'\nTASK {task:02d}: {len(inputs)} human PKLs', flush=True)
        for episode_i, src in enumerate(inputs):
            number_match = re.search(r'(\d+)$', src.stem)
            if number_match is None:
                raise ValueError(f'Expected trajectory index in {src.name}')
            number = int(number_match.group(1))
            destination = args.output_root / f'task_{task:02d}' / f'human_traj_{number:03d}.pkl'
            if destination in filenames_seen:
                raise ValueError(f'Duplicate trajectory index in {task_dir}: {number}')
            filenames_seen.add(destination)
            if destination.exists() and args.skip_existing:
                saved = load_pickle(destination)
                verify_output(saved, expected_task=task,
                              expected_instruction=instruction_by_task[task],
                              expected_frames=len(saved['traj']))
                T = len(saved['traj'])
                skipped_count += 1
                print(f'  SKIP {destination.name}: verified T={T}', flush=True)
            else:
                if destination.exists() and not args.overwrite and not args.dry_run:
                    raise FileExistsError(f'{destination}: already exists')
                payload, number = convert_one(
                    src, task=task, instruction=instruction_by_task[task],
                    decoded_color_order=args.decoded_color_order)
                verify_output(payload, expected_task=task,
                              expected_instruction=instruction_by_task[task],
                              expected_frames=len(payload['traj']))
                T = len(payload['traj'])
                if not args.dry_run:
                    save_atomically(destination, payload, overwrite=args.overwrite)
                    if args.preview_dir is not None and episode_i == 0:
                        previews(args.preview_dir, payload)
                converted_count += 1
                print(f'  OK   {destination.name}: T={T}, RGB=(224,224,3)', flush=True)
            total_episodes += 1
            total_frames += T
            per_task_frames.append(T)
        print(f'  SUMMARY task_{task:02d}: episodes={len(inputs)} converted={converted_count} '
              f'skipped={skipped_count} frames={sum(per_task_frames)} '
              f'min={min(per_task_frames)} max={max(per_task_frames)}', flush=True)
        rows.append(dict(task_id=f'{task:02d}', episodes=len(inputs),
                         converted=converted_count, skipped=skipped_count,
                         frames=sum(per_task_frames), min_frames=min(per_task_frames),
                         max_frames=max(per_task_frames)))
    print(f'\nDONE episodes={total_episodes} frames={total_frames} dry_run={args.dry_run}', flush=True)
    if args.summary_csv is not None and not args.dry_run:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Wrote summary: {args.summary_csv}', flush=True)


if __name__ == '__main__':
    main()
