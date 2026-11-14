#!/usr/bin/env python3
"""Select VLA-JEPA rollout JSON 1/1/1 candidates after audit; trim frame 0 only when verified.

Reads original PKLs/JSON and audit CSV. Does NOT alter originals.
Default is dry-run; use --execute to write separate *_clean directory.
Only unpickle files from trusted sources.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import savers  # noqa: F401; required for savers.Trajectory deserialization

BASE = Path('/mnt/beegfs/a.cuciniello9/datasets')
EXPECTED_BY_GROUP = {
    'E1': (26, 1),
    'E1_onepass': (29, 3),
    'E2': (34, 4),
    'E2_onepass': (31, 0),
    'E3': (25, 0),
    'E4': (42, 6),
}
EXCLUDED = {
    'E2_onepass/task_00/traj_003.pkl',  # only 2 frames, no observed grasp
    'E2_onepass/task_04/traj_003.pkl',  # zero frames
}
SUCCESS_KEYS = ('object_reached', 'object_picked', 'object_placed')


def pkl_step(traj, t):
    try:
        return traj.get(t, decompress=False)
    except TypeError:
        return traj.get(t)


def obs_gripper_qpos(traj, t):
    step = pkl_step(traj, t)
    value = np.asarray(step['obs']['gripper_qpos'], dtype=np.float64).reshape(-1)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f'invalid gripper qpos at step {t}: {value}')
    return float(value[0])


def load_payload(p):
    with p.open('rb') as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or 'traj' not in d:
        raise ValueError(f'Invalid payload: {p}')
    return d


def validate_frame0_correction(src, threshold=0.4):
    payload = load_payload(src)
    traj = payload['traj']
    if len(traj) < 3 or not isinstance(getattr(traj, '_data', None), list):
        raise ValueError(f'Unexpected Trajectory: {src}')
    if len(traj._data) != len(traj):
        raise ValueError(f'_data mismatch: {src}')
    states = np.array([obs_gripper_qpos(traj, t) >= threshold for t in range(len(traj))], dtype=np.int8)
    if not (states[0] == 1 and states[1] == 0):
        raise ValueError(f'Expected only initial CLOSED->OPEN at frame 1: {src}; first binary={states[:5].tolist()}')
    after = states[1:]
    if not (after[0] == 0 and after[-1] == 1 and
            np.count_nonzero(np.diff(after) == 1) == 1 and
            np.count_nonzero(np.diff(after) == -1) == 0):
        raise ValueError(f'Observed gripper is not OPEN->CLOSED after removing frame 0: {src}')
    return len(traj), np.flatnonzero(np.diff(after) == 1)[0] + 1, obs_gripper_qpos(traj, 0), obs_gripper_qpos(traj, 1)


def write_trimmed(src, dest):
    payload = load_payload(src)
    traj = payload['traj']
    old_length = len(traj)
    traj._data = traj._data[1:]
    if len(traj) != old_length - 1:
        raise ValueError(f'Trajectory length not updated after frame removal: {src}')
    if obs_gripper_qpos(traj, 0) >= 0.4:
        raise ValueError(f'Trimmed first observation still CLOSED: {src}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=dest.parent,
                                         prefix=f'.{dest.name}.', suffix='.tmp',
                                         delete=False) as f:
            tmp = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        verification = load_payload(tmp)['traj']
        if len(verification) != old_length - 1 or obs_gripper_qpos(verification, 0) >= 0.4:
            raise ValueError(f'Trimmed file re-read check failed: {src}')
        os.replace(tmp, dest)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=BASE/'mixture/vla_jepa_rollouts')
    ap.add_argument('--audit', type=Path, default=BASE/'vla_jepa_rollouts_audit/successful_111_audit.csv')
    ap.add_argument('--destination', type=Path, default=BASE/'mixture/vla_jepa_rollouts_clean')
    ap.add_argument('--execute', action='store_true', help='Actually copy and trim; without this, dry-run')
    args = ap.parse_args()
    root = args.root.resolve()
    destination = args.destination.resolve()
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError('Source and destination must be independent directories')
    if not args.audit.is_file():
        raise FileNotFoundError(args.audit)

    with args.audit.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 189:
        raise ValueError(f'Expected 189 JSON-success audit rows, found {len(rows)}')

    selected = []
    omitted = set()
    counts = Counter()
    trimmed_count = Counter()
    seen = set()
    for row in rows:
        rel = Path(row['relative_path'])
        rel_str = rel.as_posix()
        if rel_str in seen:
            raise ValueError(f'Duplicate path in audit: {rel_str}')
        seen.add(rel_str)
        if len(rel.parts) != 3 or rel.parts[0] not in EXPECTED_BY_GROUP or not rel.parts[1].startswith('task_') or not rel.name.startswith('traj_') or rel.suffix != '.pkl':
            raise ValueError(f'Unexpected relative PKL path: {rel}')
        if row['json_success'] != '1' or row['json_status'] != 'OK':
            raise ValueError(f'Audit entry is not a valid success: {rel_str}')
        source = root / rel
        if not source.is_file():
            raise FileNotFoundError(source)
        source_json = source.with_suffix('.json')
        if not source_json.is_file():
            raise FileNotFoundError(source_json)
        with source_json.open(encoding='utf-8') as f:
            metadata = json.load(f)
        if not all(metadata.get(k) == 1 for k in SUCCESS_KEYS):
            raise ValueError(f'JSON no longer indicates 1/1/1: {source_json}')
        status = row['audit_status']
        flags = row['flags']
        if rel_str in EXCLUDED:
            if status not in ('ERROR', 'REVIEW'):
                raise ValueError(f'Expected excluded problematic file, got status={status} at {rel_str}')
            omitted.add(rel_str)
            continue
        if status == 'OK' and not flags:
            trim = False
        elif status == 'REVIEW' and flags == 'NON_01_OBSERVED_GRIPPER':
            trim = True
        else:
            raise ValueError(f'Unexpected status or flags for {rel_str}: {status}, {flags}')
        if trim:
            length, close_frame, q0, q1 = validate_frame0_correction(source)
            trimmed_count[rel.parts[0]] += 1
            print(f'[TRIM] {rel_str}: {length}->{length - 1} frames; qpos0={q0:.6f}, qpos1={q1:.6f}; close_after_trim@{int(close_frame)}')
        selected.append((rel, source, source_json, trim))
        counts[rel.parts[0]] += 1
    if omitted != EXCLUDED:
        raise ValueError(f'Unexpected excluded set: {omitted}')
    for group, (expected, expected_trimmed) in EXPECTED_BY_GROUP.items():
        if counts[group] != expected or trimmed_count[group] != expected_trimmed:
            raise ValueError(f'Wrong counts in {group}: selected={counts[group]} trimmed={trimmed_count[group]} expected={(expected, expected_trimmed)}')
    if len(selected) != 187 or sum(trimmed_count.values()) != 14:
        raise ValueError(f'Wrong final counts: selected={len(selected)} trimmed={sum(trimmed_count.values())}')

    print('\nSelection verified; no originals modified:')
    for group in EXPECTED_BY_GROUP:
        print(f'  {group}: {counts[group]} trajectories, {trimmed_count[group]} frame-0 corrections')
    print('  EXCLUDED:', ', '.join(sorted(omitted)))
    if not args.execute:
        print('\nDRY-RUN COMPLETE. Add --execute to create the separate clean dataset.')
        return

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Destination already nonempty (refusing to overwrite): {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (rel, src, src_json, trim) in enumerate(selected, 1):
        dest = destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if trim:
            write_trimmed(src, dest)
        else:
            shutil.copy2(src, dest)
        shutil.copy2(src_json, dest.with_suffix('.json'))
        manifest.append({'source_relative_path': rel.as_posix(),
                         'destination_relative_path': rel.as_posix(),
                         'removed_initial_frame': int(trim),
                         'status': 'OK'})
        if i % 20 == 0 or i == len(selected):
            print(f'  copied {i}/{len(selected)}', flush=True)
    manifest_path = destination/'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    actual_pkls = list(destination.glob('*/task_*/traj_*.pkl'))
    actual_jsons = list(destination.glob('*/task_*/traj_*.json'))
    if len(actual_pkls) != 187 or len(actual_jsons) != 187:
        raise RuntimeError(f'Post-copy counts mismatch: PKL={len(actual_pkls)}, JSON={len(actual_jsons)}')
    print(f'\nCOMPLETE: {len(actual_pkls)} PKL and {len(actual_jsons)} JSON, 14 frame-0 corrections')
    print('Destination:', destination)
    print('Manifest:', manifest_path)
    print('Original source files unchanged.')


if __name__ == '__main__':
    main()
