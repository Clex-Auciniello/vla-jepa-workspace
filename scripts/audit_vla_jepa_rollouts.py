#!/usr/bin/env python3
"""Audit SOLO LETTURA dei rollout VLA-JEPA (E1/E1_onepass/E2/E2_onepass/E3/E4).

Ordine logico:
  1) inventario PKL/JSON, schema e successi JSON 1/1/1;
  2) sugli 1/1/1, audit di EEF vs joint, delta XYZ/SO(3), gripper;
  3) statistiche delle delta del riferimento canonico VLA-JEPA.

ATTENZIONE: i rollout del controller VLA-JEPA hanno un'osservazione per
chiamata di INFERENZA, non necessariamente per ogni target di un action chunk.
Le delta fra osservazioni di tali rollout NON sono direttamente confrontabili
con quelle di un dataset campionato a frequenza piu' alta. Il superamento di
soglie su grandezza delle delta e ritardo del gripper e' solo INFORMATIVO.

Nessuna copia, modifica, rimozione o riscrittura di PKL/JSON.
Aprire soltanto PKL provenienti da fonti fidate.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import savers  # noqa: F401 - class definition for unpickling savers.Trajectory

DATA_ROOT = Path('/mnt/beegfs/a.cuciniello9/datasets')
SOURCE_DEFAULT = DATA_ROOT / 'mixture/vla_jepa_rollouts'
REFERENCE_DEFAULT = DATA_ROOT / 'robotic/vla_jepa_offline_merged'
OUTPUT_DEFAULT = DATA_ROOT / 'vla_jepa_rollouts_audit'
GROUPS = ('E1', 'E1_onepass', 'E2', 'E2_onepass', 'E3', 'E4')
SUCCESS_KEYS = ('object_reached', 'object_picked', 'object_placed')


def step_get(traj, i):
    if hasattr(traj, 'get') and not isinstance(traj, (list, tuple)):
        try:
            s = traj.get(i, decompress=False)
        except TypeError:
            s = traj.get(i)
    else:
        s = traj[i]
    if not isinstance(s, dict):
        raise TypeError(f'step {i}: expected dict, found {type(s).__name__}')
    return s


def finite_vec(value, length, key, i):
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    if v.size != length or not np.isfinite(v).all():
        raise ValueError(f'step {i} {key}: expected finite {length}-vector, got {v.shape}')
    return v


def obj_info(x):
    if x is None:
        return {'type': 'None'}
    if isinstance(x, dict):
        return {'type': 'dict', 'keys': sorted(map(str, x.keys()))}
    try:
        a = np.asarray(x)
        return {'type': type(x).__name__, 'dtype': str(a.dtype), 'shape': list(a.shape)}
    except Exception:
        return {'type': type(x).__name__}


def schema_sample(payload, traj, metadata):
    r = {
        'payload': obj_info(payload),
        'metadata': obj_info(metadata),
        'n_frames': len(traj),
    }
    if len(traj):
        s = step_get(traj, 0)
        r['step'] = obj_info(s)
        o = s.get('obs')
        r['observation'] = {str(k): obj_info(v) for k, v in sorted(o.items())} if isinstance(o, dict) else obj_info(o)
        r['raw_action'] = obj_info(s.get('action'))
    return r


def read_pickle(path):
    with path.open('rb') as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise TypeError('PKL does not contain dict["traj"]')
    return payload, payload['traj']


def parse_json(path):
    with path.open(encoding='utf-8') as f:
        j = json.load(f)
    if not isinstance(j, dict):
        raise ValueError('JSON root is not a dictionary')
    return j


def successful(j):
    return all(j.get(k) == 1 for k in SUCCESS_KEYS)


def valid_quaternions(q):
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
        raise ValueError(f'Invalid XYZW quaternions, norms={norms.min():.6f}..{norms.max():.6f}')
    return Rotation.from_quat(q)


def delta_metrics(pos, rotations, scale):
    if len(pos) < 2:
        raise ValueError('Need at least two observed frames for delta reconstruction')
    dp = np.diff(pos, axis=0)
    relative = rotations[1:] * rotations[:-1].inv()
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Gimbal lock detected')
        drpy = relative.as_euler('xyz')
    action6 = np.concatenate([dp / scale, drpy / scale], axis=1).astype(np.float32)
    rebuilt_pos = pos[:-1] + action6[:, :3].astype(np.float64) * scale
    rebuilt_rotation = (Rotation.from_euler('xyz', action6[:, 3:].astype(np.float64) * scale)
                        * rotations[:-1])
    return {
        'actions': action6,
        'xyz_mm': np.linalg.norm(dp, axis=1) * 1000.,
        'rot_deg': np.degrees(relative.magnitude()),
        'delta_xyz_norm': np.linalg.norm(action6[:, :3].astype(np.float64), axis=1),
        'delta_rot_norm': np.linalg.norm(action6[:, 3:].astype(np.float64), axis=1),
        'roundtrip_xyz_mm': np.linalg.norm(rebuilt_pos - pos[1:], axis=1) * 1000.,
        'roundtrip_rot_deg': np.degrees((rebuilt_rotation.inv() * rotations[1:]).magnitude()),
    }


def max_true_run(binary):
    best = length = 0
    for b in binary:
        length = length + 1 if b else 0
        best = max(best, length)
    return best


def quantiles(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {'n': 0}
    return {
        'n': int(a.size), 'min': float(np.min(a)), 'mean': float(np.mean(a)),
        'p50': float(np.percentile(a, 50)),
        'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)), 'max': float(np.max(a)),
    }


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def extract_rollout(traj):
    positions, quats, joints, qposes, raw_commands = [], [], [], [], []
    schema_missing = Counter()
    action_shapes = Counter()
    seen_obs_keys = Counter()
    for i in range(len(traj)):
        s = step_get(traj, i)
        o = s.get('obs')
        if not isinstance(o, dict):
            raise ValueError(f'step {i}: obs absent or non-dict')
        seen_obs_keys.update(o.keys())
        for k in ('camera_front_image', 'camera_gripper_image', 'joint_pos', 'joint_vel'):
            if k not in o or o[k] is None:
                schema_missing[k] += 1
        positions.append(finite_vec(o['eef_pos'], 3, 'eef_pos', i))
        quats.append(finite_vec(o['eef_quat'], 4, 'eef_quat', i))
        if 'joint_pos' in o:
            joints.append(finite_vec(o['joint_pos'], 6, 'joint_pos', i))
        else:
            joints.append(None)
        if 'gripper_qpos' in o and o['gripper_qpos'] is not None:
            qposes.append(finite_vec(o['gripper_qpos'], 1, 'gripper_qpos', i)[0])
        else:
            qposes.append(float('nan'))
            schema_missing['gripper_qpos'] += 1
        if 'action' in s and s['action'] is not None:
            act = np.asarray(s['action'], dtype=np.float64).reshape(-1)
            action_shapes[act.size] += 1
            if act.size == 8 and np.isfinite(act[7]):
                raw_commands.append(float(act[7]))
            else:
                raw_commands.append(float('nan'))
        else:
            action_shapes['missing'] += 1
            raw_commands.append(float('nan'))
    joint_array = np.stack(joints) if all(j is not None for j in joints) else None
    return {
        'pos': np.stack(positions), 'quat': np.stack(quats),
        'joint': joint_array, 'qpos': np.array(qposes), 'commands': np.array(raw_commands),
        'missing': dict(schema_missing), 'action_shapes': dict(action_shapes),
        'observed_keys_counts': dict(seen_obs_keys),
    }


def evaluate_gripper(qpos, commands, threshold):
    result = {
        'first_close_command_t': '', 'first_obs_closed_t': '', 'close_lag_frames': '',
        'obs_close_transitions': '', 'obs_open_transitions': '',
        'initial_closed': '', 'terminal_closed': '', 'max_stuck_during_close': '',
        'terminal_command': 'UNKNOWN', 'flags': [],
    }
    if not np.isfinite(qpos).all():
        result['flags'].append('MISSING_GRIPPER_QPOS')
        return result
    obs_closed = (qpos >= threshold).astype(int)
    rises = np.flatnonzero(np.diff(obs_closed) == 1) + 1
    falls = np.flatnonzero(np.diff(obs_closed) == -1) + 1
    result.update({
        'obs_close_transitions': len(rises), 'obs_open_transitions': len(falls),
        'initial_closed': int(obs_closed[0]), 'terminal_closed': int(obs_closed[-1]),
        'first_obs_closed_t': int(rises[0]) if len(rises) else '',
    })
    if len(rises) != 1 or len(falls) > 0 or obs_closed[0] != 0:
        result['flags'].append('NON_01_OBSERVED_GRIPPER')
    if not np.isfinite(commands).all() or not np.all(
        np.isclose(commands, 0, atol=.5) | np.isclose(commands, 255, atol=.5)
    ):
        result['flags'].append('RAW_GRIPPER_COMMAND_UNAVAILABLE_OR_UNEXPECTED')
        return result
    cmd_close = np.isclose(commands, 255, atol=.5).astype(int)
    cmd_rises = np.flatnonzero(np.diff(cmd_close) == 1) + 1
    result['terminal_command'] = 'CLOSE' if cmd_close[-1] else 'OPEN'
    if len(cmd_rises) == 1:
        first_cmd = int(cmd_rises[0])
        result['first_close_command_t'] = first_cmd
        if len(rises):
            result['close_lag_frames'] = int(rises[0]) - first_cmd
    else:
        result['flags'].append('RAW_CMD_CLOSE_TRANSITIONS_NOT_1')
    if np.any(cmd_close[:-1] == 1):
        mask = ((np.abs(np.diff(qpos)) <= 1e-8)
                & (qpos[:-1] < threshold) & (qpos[1:] < threshold)
                & (cmd_close[:-1] == 1))
        result['max_stuck_during_close'] = max_true_run(mask)
    else:
        result['max_stuck_during_close'] = 0
    return result


def freeze_ef_joint(pos, rot, joints):
    if joints is None:
        return {'available': False, 'max_stale_run': '', 'stale_transitions': '',
                'all_pose_static_joint_moving': False}
    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dang = (rot[1:] * rot[:-1].inv()).magnitude()
    djoint = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    static_eef = (dpos <= 1e-6) & (dang <= 1e-5)
    joint_moving = djoint > 1e-3
    suspicious = static_eef & joint_moving
    return {
        'available': True,
        'max_stale_run': max_true_run(suspicious),
        'stale_transitions': int(np.count_nonzero(suspicious)),
        'all_pose_static_joint_moving': bool(np.all(static_eef) and np.count_nonzero(joint_moving) >= 3),
        'max_joint_step_rad_l2': float(np.max(djoint)),
    }


def ref_audit(root, scale, report_limit=0):
    paths = sorted(p for p in root.glob('task_*/*traj*.pkl') if p.is_file())
    if report_limit:
        paths = paths[:report_limit]
    if not paths:
        raise RuntimeError(f'No reference task_*/*traj*.pkl under {root}')
    print(f'\n[reference] {len(paths)} canonical files', flush=True)
    arrays = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    errors = []
    for idx, path in enumerate(paths, 1):
        subtype = 'scripted' if 'scripted' in path.stem.lower() else (
            'tfrecord' if 'tfrecord' in path.stem.lower() else 'other')
        try:
            _, traj = read_pickle(path)
            if len(traj) < 2:
                raise ValueError('Less than two frames')
            eefs, a6 = [], []
            for t in range(len(traj)):
                s = step_get(traj, t)
                eefs.append(finite_vec(s['obs']['EEF_state'], 6, 'EEF_state', t))
                a6.append(finite_vec(s['action'], 7, 'action', t)[:6])
            eefs = np.stack(eefs)
            pos, rot = eefs[:, :3], Rotation.from_euler('xyz', eefs[:, 3:])
            d = delta_metrics(pos, rot, scale)
            for k in ('xyz_mm', 'rot_deg', 'delta_xyz_norm', 'delta_rot_norm'):
                arrays[subtype][k].extend(d[k].tolist())
            errors_a6 = np.asarray(a6)[:-1] - d['actions'].astype(np.float64)
            arrays[subtype]['saved_vs_rebuilt_xyz_l2'].extend(np.linalg.norm(errors_a6[:, :3], axis=1))
            arrays[subtype]['saved_vs_rebuilt_rot_l2'].extend(np.linalg.norm(errors_a6[:, 3:], axis=1))
            counts[subtype] += 1
        except Exception as exc:
            counts[subtype + '_ERROR'] += 1
            errors.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
        if idx == 1 or idx % 100 == 0 or idx == len(paths):
            print(f'  reference {idx}/{len(paths)}', flush=True)
    result = {
        'root': str(root), 'sampled': report_limit > 0, 'files_discovered': len(paths),
        'counts': dict(counts),
        'groups': {sub: {key: quantiles(val) for key, val in values.items()}
                   for sub, values in arrays.items()},
        'errors': errors,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=SOURCE_DEFAULT)
    parser.add_argument('--reference', type=Path, default=REFERENCE_DEFAULT)
    parser.add_argument('--output', type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument('--skip-reference', action='store_true')
    parser.add_argument('--max-reference-files', type=int, default=0,
                        help='0: all; >0: deterministic prefix (diagnostic only)')
    parser.add_argument('--scale', type=float, default=.05)
    parser.add_argument('--gripper-threshold', type=float, default=.4)
    parser.add_argument('--large-xyz-mm', type=float, default=60.)
    parser.add_argument('--large-rot-deg', type=float, default=30.)
    args = parser.parse_args()
    if args.scale <= 0 or args.max_reference_files < 0:
        parser.error('Invalid --scale or --max-reference-files')
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    ref = args.reference.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not args.skip_reference and not ref.is_dir():
        raise FileNotFoundError(ref)
    for data_dir in (root, *( [] if args.skip_reference else [ref] )):
        if output == data_dir or output.is_relative_to(data_dir):
            raise ValueError('Output cannot be under input/reference data')
    output.mkdir(parents=True, exist_ok=True)

    all_rows, good_rows, flagged_rows, transitions, grippers = [], [], [], [], []
    selected_paths = []
    schema = {}
    count = defaultdict(Counter)
    distributions = defaultdict(lambda: defaultdict(list))
    for group in GROUPS:
        subdir = root / group
        if not subdir.is_dir():
            print(f'\n[{group}] DIRECTORY MISSING: {subdir}', flush=True)
            count[group]['DIRECTORY_MISSING'] = 1
            continue
        # Only immediate children of task_*; skip context/debug PKLs.
        paths = sorted(p for p in subdir.rglob('traj_*.pkl')
                       if p.is_file() and p.parent.name.startswith('task_'))
        print(f'\n[{group}] PKL files: {len(paths)}', flush=True)
        schema[group] = {'sample': None, 'sample_path': None}
        count[group]['pkls'] = len(paths)
        for index, path in enumerate(paths, 1):
            rel = path.relative_to(root).as_posix()
            row = {
                'group': group, 'path': str(path), 'relative_path': rel,
                'json_status': 'ERROR', 'json_success': 0,
                'n_frames': '', 'schema_status': '', 'audit_status': 'NOT_SELECTED',
                'flags': '', 'notes': '', 'error': '',
            }
            j = None
            try:
                j = parse_json(path.with_suffix('.json'))
                row.update({k: j.get(k, '') for k in SUCCESS_KEYS})
                row['json_status'] = 'OK'
                row['json_success'] = int(successful(j))
            except Exception as exc:
                row['json_status'] = 'ERROR'
                row['error'] = 'JSON: ' + f'{type(exc).__name__}: {exc}'
            if j is None:
                count[group]['json_missing_or_invalid'] += 1
            elif row['json_success']:
                count[group]['success_111'] += 1
            else:
                count[group]['not_111'] += 1
            # Inspect at least one representative per group even if it is not a success.
            if schema[group]['sample'] is None:
                try:
                    payload, traj = read_pickle(path)
                    if len(traj):
                        schema[group]['sample'] = schema_sample(payload, traj, j)
                        schema[group]['sample_path'] = rel
                except Exception as exc:
                    schema[group]['first_sample_error'] = f'{type(exc).__name__}: {exc}'
            if not row['json_success']:
                all_rows.append(row)
                continue
            selected_paths.append(str(path))
            row['audit_status'] = 'ERROR'
            try:
                _, traj = read_pickle(path)
                row['n_frames'] = len(traj)
                if len(traj) < 2:
                    raise ValueError('Trajectory has less than 2 frames')
                data = extract_rollout(traj)
                pos, quat, joints = data['pos'], data['quat'], data['joint']
                rot = valid_quaternions(quat)
                dl = delta_metrics(pos, rot, args.scale)
                freeze = freeze_ef_joint(pos, rot, joints)
                gr = evaluate_gripper(data['qpos'], data['commands'], args.gripper_threshold)
                flags = list(gr['flags'])
                if joints is None:
                    flags.append('JOINT_STATES_MISSING_CANNOT_AUDIT_FREEZE')
                if data['missing'].get('joint_vel', 0):
                    flags.append('JOINT_VELOCITIES_MISSING_FRAMES')
                if (isinstance(gr['max_stuck_during_close'], int)
                        and gr['max_stuck_during_close'] >= 5):
                    flags.append('POSSIBLE_GRIPPER_QPOS_STUCK_WHILE_CLOSING')
                if freeze['all_pose_static_joint_moving']:
                    flags.append('EEF_FROZEN_ENTIRE_TRAJECTORY')
                if freeze['available'] and freeze['max_stale_run'] >= 5:
                    flags.append('EEF_STALE_WHILE_JOINTS_MOVE_5_TRANSITIONS')
                if data['missing'].get('camera_front_image', 0):
                    flags.append('FRONT_CAMERA_MISSING_FRAMES')
                if data['missing'].get('camera_gripper_image', 0):
                    flags.append('GRIPPER_CAMERA_MISSING_FRAMES')
                if dl['roundtrip_xyz_mm'].max() > .01 or dl['roundtrip_rot_deg'].max() > .001:
                    flags.append('FLOAT32_DELTA_ROUNDTRIP_ERROR')
                # WARN, not a quality rejection: sparse inference observations.
                notes = []
                if np.any(dl['xyz_mm'] > args.large_xyz_mm):
                    notes.append('LARGE_XYZ_STEP_SPARSE_SAMPLING_POSSIBLE')
                if np.any(dl['rot_deg'] > args.large_rot_deg):
                    notes.append('LARGE_ROT_STEP_SPARSE_SAMPLING_POSSIBLE')
                if isinstance(gr['close_lag_frames'], int) and gr['close_lag_frames'] > 3:
                    notes.append('SLOW_CLOSE_OBSERVED_SPARSE_SAMPLING')
                if gr['terminal_command'] == 'CLOSE':
                    notes.append('NO_TERMINAL_OPEN_COMMAND')
                if data['action_shapes'].get(8, 0) != len(traj):
                    notes.append('RAW_ACTION_NOT_8D_IN_SOME_FRAMES')
                row.update({
                    'schema_status': 'OK',
                    'audit_status': 'REVIEW' if flags else 'OK',
                    'flags': ';'.join(flags), 'notes': ';'.join(notes),
                    'n_transitions': len(traj)-1,
                    'joint_available': int(joints is not None),
                    'max_stale_eef_joint_run': freeze['max_stale_run'],
                    'stale_eef_joint_transitions': freeze['stale_transitions'],
                    'xyz_p50_mm': float(np.median(dl['xyz_mm'])),
                    'xyz_p95_mm': float(np.percentile(dl['xyz_mm'], 95)),
                    'xyz_max_mm': float(dl['xyz_mm'].max()),
                    'rot_p50_deg': float(np.median(dl['rot_deg'])),
                    'rot_p95_deg': float(np.percentile(dl['rot_deg'], 95)),
                    'rot_max_deg': float(dl['rot_deg'].max()),
                    'max_roundtrip_xyz_mm': float(dl['roundtrip_xyz_mm'].max()),
                    'max_roundtrip_rot_deg': float(dl['roundtrip_rot_deg'].max()),
                    'missing_front_frames': data['missing'].get('camera_front_image', 0),
                    'missing_gripper_frames': data['missing'].get('camera_gripper_image', 0),
                    **{k: gr[k] for k in ('first_close_command_t', 'first_obs_closed_t',
                                           'close_lag_frames', 'obs_close_transitions',
                                           'obs_open_transitions', 'terminal_command')},
                })
                for metric in ('xyz_mm', 'rot_deg', 'delta_xyz_norm', 'delta_rot_norm'):
                    distributions[group][metric].extend(dl[metric].tolist())
                grippers.append({
                    'group': group, 'relative_path': rel,
                    **{k: v for k, v in gr.items() if k != 'flags'},
                    'flags': ';'.join(gr['flags']),
                })
                for i in range(len(traj)-1):
                    action6 = dl['actions'][i]
                    transitions.append({
                        'group': group, 'relative_path': rel, 'step': i,
                        'dx': float(action6[0]), 'dy': float(action6[1]),
                        'dz': float(action6[2]),
                        'droll': float(action6[3]), 'dpitch': float(action6[4]),
                        'dyaw': float(action6[5]),
                        'gripper_label': (float(data['qpos'][i+1] >= args.gripper_threshold) / args.scale
                                          if np.isfinite(data['qpos'][i+1]) else ''),
                        'xyz_mm': float(dl['xyz_mm'][i]),
                        'rot_deg': float(dl['rot_deg'][i]),
                        'roundtrip_xyz_mm': float(dl['roundtrip_xyz_mm'][i]),
                        'roundtrip_rot_deg': float(dl['roundtrip_rot_deg'][i]),
                    })
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
                row['audit_status'] = 'ERROR'
            all_rows.append(row)
            good_rows.append(row)
            count[group]['audited_' + row['audit_status']] += 1
            if row['audit_status'] != 'OK':
                flagged_rows.append(row)
            if index == 1 or index % 20 == 0 or row['audit_status'] != 'OK':
                msg = row['flags'] or row['error'] or row['notes']
                print(f'  [{index}/{len(paths)}] {rel}: {row["audit_status"]} {msg}', flush=True)
        print('  counts:', dict(count[group]), flush=True)

    reference = None
    if not args.skip_reference:
        reference = ref_audit(ref, args.scale, args.max_reference_files)
    summary = {
        'input_root': str(root), 'reference_root': None if args.skip_reference else str(ref),
        'parameters': {
            'scale': args.scale, 'gripper_qpos_threshold': args.gripper_threshold,
            'xyz_diagnostic_warning_mm': args.large_xyz_mm,
            'rotation_diagnostic_warning_deg': args.large_rot_deg,
            'eef_static_xyz_eps_m': 1e-6, 'eef_static_angle_eps_rad': 1e-5,
            'joint_movement_eps_l2_rad': 1e-3, 'eef_stale_run_transitions': 5,
        },
        'scope': ('All PKLs enumerated for 1/1/1; geometry+gripper audits only on 1/1/1. '
                  'Rollout steps are per inference, not every target in action chunk. '
                  'Dataset/reference step size differences are descriptive, not exclusion criteria. '
                  'No input files modified.'),
        'groups': {
            name: {
                'counts': dict(count[name]),
                'delta_statistics': {key: quantiles(vals)
                                     for key, vals in distributions[name].items()},
            }
            for name in GROUPS
        },
        'reference': reference,
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    (output/'schema_samples.json').write_text(json.dumps(schema, indent=2)+'\n', encoding='utf-8')
    (output/'successful_111_paths.txt').write_text('\n'.join(selected_paths)+'\n', encoding='utf-8')
    traj_fields = [
        'group', 'relative_path', 'path', 'json_status', 'json_success',
        'object_reached', 'object_picked', 'object_placed',
        'n_frames', 'schema_status', 'audit_status', 'flags', 'notes', 'error',
        'n_transitions', 'joint_available', 'max_stale_eef_joint_run',
        'stale_eef_joint_transitions', 'xyz_p50_mm', 'xyz_p95_mm', 'xyz_max_mm',
        'rot_p50_deg', 'rot_p95_deg', 'rot_max_deg',
        'max_roundtrip_xyz_mm', 'max_roundtrip_rot_deg',
        'missing_front_frames', 'missing_gripper_frames',
        'first_close_command_t', 'first_obs_closed_t', 'close_lag_frames',
        'obs_close_transitions', 'obs_open_transitions', 'terminal_command',
    ]
    trans_fields = ['group', 'relative_path', 'step', 'dx', 'dy', 'dz',
                    'droll', 'dpitch', 'dyaw', 'gripper_label', 'xyz_mm', 'rot_deg',
                    'roundtrip_xyz_mm', 'roundtrip_rot_deg']
    gripper_fields = ['group', 'relative_path', 'first_close_command_t',
                      'first_obs_closed_t', 'close_lag_frames',
                      'obs_close_transitions', 'obs_open_transitions',
                      'initial_closed', 'terminal_closed', 'max_stuck_during_close',
                      'terminal_command', 'flags']
    write_csv(output/'all_rollouts.csv', all_rows, traj_fields)
    write_csv(output/'successful_111_audit.csv', good_rows, traj_fields)
    write_csv(output/'flagged_successful_111.csv', flagged_rows, traj_fields)
    write_csv(output/'delta_transitions.csv', transitions, trans_fields)
    write_csv(output/'gripper_events.csv', grippers, gripper_fields)
    print('\n=== VLA-JEPA ROLLOUT SUMMARY ===')
    for group in GROUPS:
        c = count[group]
        d = summary['groups'][group]['delta_statistics']
        print(f'{group}: PKL={c["pkls"]} JSON_1/1/1={c["success_111"]} '
              f'OK={c["audited_OK"]} REVIEW={c["audited_REVIEW"]} ERROR={c["audited_ERROR"]}')
        if d:
            print(f'  observed XYZ p95={d["xyz_mm"]["p95"]:.2f} mm, '
                  f'rot p95={d["rot_deg"]["p95"]:.2f} deg')
    if reference:
        for group, x in reference['groups'].items():
            print(f'reference {group}: n={reference["counts"].get(group, 0)} '
                  f'XYZ p95={x["xyz_mm"]["p95"]:.2f} mm, '
                  f'rot p95={x["rot_deg"]["p95"]:.2f} deg')
    print('Reports:', output)
    print('  summary.json, schema_samples.json, all_rollouts.csv,')
    print('  successful_111_audit.csv, flagged_successful_111.csv,')
    print('  successful_111_paths.txt, delta_transitions.csv, gripper_events.csv')
    print('NO INPUT PKL/JSON WAS MODIFIED.')


if __name__ == '__main__':
    main()
