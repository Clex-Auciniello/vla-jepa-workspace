#!/usr/bin/env python3
"""Audit di delta EEF e gripper dei rollout COD *_clean (SOLO LETTURA).

- Le delta 7D sono RICOSTRUITE dagli stati osservati consecutivi, non da
  action COD originale 8D (che e' un target assoluto).
- Rota: R_rel = R[t+1] * R[t].inv(), Euler XYZ / scale.
- Gripper: label t = 1[qpos[t+1] >= threshold] / scale.
- Terminale: label azione 7D zero (OPEN per il gripper), proposta per il
  preprocess; nessun file di origine e' modificato.
- Ref VLA-JEPA: confronto DESCRITTIVO delle distribuzioni per-step, non e' un
  criterio di esclusione (frequenze/strategie di raccolta possono differire).

Legge SOLO task_*/traj_*.pkl per COD e task_*/*traj*.pkl per riferimento.
Il formato pickle va caricato soltanto da sorgenti attendibili.
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
import savers  # noqa: F401, necessario per deserializzare savers.Trajectory

BASE = Path('/mnt/beegfs/a.cuciniello9/datasets')
SOURCES_DEFAULT = {
    'cod_epoch_100_wrist_clean': BASE / 'mixture/cod_epoch_100_wrist_clean',
    'cod_finetuning_no_wrist_clean': BASE / 'mixture/cod_rollout_finetuning/no_wrist_clean',
}
REFERENCE_DEFAULT = BASE / 'robotic/vla_jepa_offline_merged'
OUTPUT_DEFAULT = BASE / 'cod_delta_gripper_audit'
SUCCESS_KEYS = ('object_reached', 'object_picked', 'object_placed')


def step_dict(traj, t):
    if hasattr(traj, 'get') and not isinstance(traj, (list, tuple)):
        try:
            x = traj.get(t, decompress=False)
        except TypeError:
            x = traj.get(t)
    else:
        x = traj[t]
    if not isinstance(x, dict):
        raise TypeError(f'step {t}: get() did not return a dict')
    return x


def vec(x, n, desc):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != n or not np.isfinite(a).all():
        raise ValueError(f'{desc}: expected finite {n}D vector; received {a.shape}')
    return a


def load_traj(path):
    with path.open('rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or 'traj' not in obj:
        raise TypeError('PKL must contain dict["traj"]')
    traj = obj['traj']
    if len(traj) < 2:
        raise ValueError(f'trajectory has only {len(traj)} frames')
    return traj


def extract_cod(traj):
    poses, quats, qpos, cmd = [], [], [], []
    for t in range(len(traj)):
        x = step_dict(traj, t)
        o = x['obs']
        poses.append(vec(o['eef_pos'], 3, f'eef_pos at t={t}'))
        quats.append(vec(o['eef_quat'], 4, f'eef_quat at t={t}'))
        qpos.append(vec(o['gripper_qpos'], 1, f'gripper_qpos at t={t}')[0])
        act = vec(x['action'], 8, f'original COD action at t={t}')
        cmd.append(act[7])
    return np.stack(poses), np.stack(quats), np.asarray(qpos), np.asarray(cmd)


def extract_reference(traj):
    poses, quats, actions = [], [], []
    for t in range(len(traj)):
        x = step_dict(traj, t)
        o = x['obs']
        eef = vec(o['EEF_state'], 6, f'reference EEF_state at t={t}')
        poses.append(eef[:3])
        quats.append(Rotation.from_euler('xyz', eef[3:]).as_quat())
        actions.append(vec(x['action'], 7, f'reference action at t={t}'))
    return np.stack(poses), np.stack(quats), np.stack(actions)


def delta_from_observations(pos, quat, scale):
    norms = np.linalg.norm(quat, axis=1)
    if not np.all(np.isfinite(norms)) or not np.allclose(norms, 1., atol=1e-3, rtol=0):
        raise ValueError(f'non-unit quaternions: min={norms.min()}, max={norms.max()}')

    R = Rotation.from_quat(quat)
    dp = np.diff(pos, axis=0)
    rel = R[1:] * R[:-1].inv()  # World-frame relative orientation
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Gimbal lock detected')
        drpy = rel.as_euler('xyz', degrees=False)
    delta = np.concatenate([dp / scale, drpy / scale], axis=1).astype(np.float32)

    # Ricalcolo da label quantizzata float32, come sara' salvata nel dataset.
    pos_rebuilt = pos[:-1] + delta[:, :3].astype(np.float64) * scale
    R_rebuilt = Rotation.from_euler('xyz', delta[:, 3:6].astype(np.float64) * scale) * R[:-1]
    pos_err_mm = np.linalg.norm(pos_rebuilt - pos[1:], axis=1) * 1000.
    rot_err_deg = np.degrees((R_rebuilt.inv() * R[1:]).magnitude())
    return {
        'delta': delta,
        'dp_mm': np.linalg.norm(dp, axis=1) * 1000.,
        'rot_deg': np.degrees(rel.magnitude()),
        'delta_xyz_norm': np.linalg.norm(delta[:, :3].astype(np.float64), axis=1),
        'delta_rot_norm': np.linalg.norm(delta[:, 3:6].astype(np.float64), axis=1),
        'pos_err_mm': pos_err_mm,
        'rot_err_deg': rot_err_deg,
    }


def quantiles(a):
    a = np.asarray(a, dtype=np.float64)
    if not a.size:
        return {'n': 0}
    return {k: float(v) for k, v in zip(
        ('min', 'p50', 'p95', 'p99', 'max'),
        (np.min(a), np.percentile(a, 50), np.percentile(a, 95),
         np.percentile(a, 99), np.max(a)))
    } | {'n': int(a.size), 'mean': float(a.mean())}


def cmd_binary(command):
    if np.any(~np.isfinite(command)):
        raise ValueError('gripper commands not finite')
    is_open = np.isclose(command, 0.0, atol=0.5, rtol=0)
    is_close = np.isclose(command, 255.0, atol=0.5, rtol=0)
    if not (is_open | is_close).all():
        raise ValueError(f'unknown gripper commands: {np.unique(command[~(is_open | is_close)]).tolist()}')
    return is_close.astype(np.int8)


def max_static_below_threshold(qpos, cbin, threshold, eps=1e-8):
    """Max run di TRANSIZIONI a qpos quasi invariato, sotto soglia,
    quando CLOSE e' comandato. Un plateau dopo CLOSED e' normale."""
    same = (np.abs(np.diff(qpos)) <= eps)
    below = (qpos[:-1] < threshold) & (qpos[1:] < threshold)
    close = (cbin[:-1] == 1)
    flags = same & below & close
    run = best = 0
    for x in flags:
        run = run + 1 if x else 0
        best = max(run, best)
    return best


def gripper_events(qpos, commands, threshold, max_close_lag, freeze_run):
    obs_bin = (qpos >= threshold).astype(np.int8)
    cmd_bin = cmd_binary(commands)
    obs_changes = (np.flatnonzero(np.diff(obs_bin) != 0) + 1).tolist()
    obs_rises = (np.flatnonzero(np.diff(obs_bin) == 1) + 1).tolist()
    obs_falls = (np.flatnonzero(np.diff(obs_bin) == -1) + 1).tolist()
    cmd_rises = (np.flatnonzero(np.diff(cmd_bin) == 1) + 1).tolist()
    cmd_falls = (np.flatnonzero(np.diff(cmd_bin) == -1) + 1).tolist()

    flags = []
    if obs_bin[0] != 0:
        flags.append('START_OBS_CLOSED')
    if len(obs_rises) != 1 or len(obs_falls) != 0 or obs_bin[-1] != 1:
        flags.append('NON_01_OBS_PATTERN')
    if cmd_bin[0] != 0:
        flags.append('START_CMD_CLOSE')
    if len(cmd_rises) != 1:
        flags.append('CMD_CLOSE_COUNT_NOT_1')
    if len(cmd_falls) > 1:
        flags.append('CMD_MULTI_OPEN')

    # Primo comando CLOSE e prima chiusura fisica dopo il comando.
    close_cmd_t = cmd_rises[0] if cmd_rises else None
    observed_closed_t = obs_rises[0] if obs_rises else None
    lag = (observed_closed_t - close_cmd_t) if (
        close_cmd_t is not None and observed_closed_t is not None) else None
    if close_cmd_t is not None and observed_closed_t is None:
        flags.append('NO_OBS_CLOSED_AFTER_CLOSE_CMD')
    if lag is not None and lag < 0:
        flags.append('OBS_CLOSE_BEFORE_CMD')
    if lag is not None and lag > max_close_lag:
        flags.append('SLOW_CLOSE_FEEDBACK')

    static_run = max_static_below_threshold(qpos, cmd_bin, threshold)
    if static_run >= freeze_run:
        flags.append('POSSIBLE_QPOS_STUCK_WHILE_CLOSING')

    # In queste registrazioni il comando OPEN puo' arrivare all'ultimo frame;
    # non esiste il frame t+1 per misurare l'effettiva apertura.
    terminal_open = bool(cmd_bin[-1] == 0)
    if not terminal_open:
        flags.append('MISSING_TERMINAL_OPEN_COMMAND_INFO')

    return {
        'qpos_start': float(qpos[0]),
        'qpos_end': float(qpos[-1]),
        'first_close_command_t': close_cmd_t,
        'first_obs_closed_t': observed_closed_t,
        'close_feedback_lag_frames': lag,
        'obs_close_count': len(obs_rises),
        'obs_open_count': len(obs_falls),
        'obs_change_steps': ','.join(map(str, obs_changes)),
        'command_close_count': len(cmd_rises),
        'command_open_count': len(cmd_falls),
        'command_change_close_steps': ','.join(map(str, cmd_rises)),
        'command_change_open_steps': ','.join(map(str, cmd_falls)),
        'max_stuck_qpos_below_thr_close_transitions': static_run,
        'original_terminal_command': 'OPEN' if terminal_open else 'CLOSE',
        'terminal_open_label_will_be_inserted': True,
        'flags': flags,
        'obs_bin': obs_bin,
    }


def write_csv(path, rows, cols):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def format_summary(traces):
    if not traces:
        return {}
    return {metric: quantiles(np.concatenate([t[metric] for t in traces]))
            for metric in ('dp_mm', 'rot_deg', 'delta_xyz_norm', 'delta_rot_norm')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', metavar='NAME=PATH', action='append',
                   help='Repeat to replace the default clean COD sources')
    p.add_argument('--reference', type=Path, default=REFERENCE_DEFAULT,
                   help='Path to canonical VLA-JEPA reference; use --skip-reference to omit')
    p.add_argument('--skip-reference', action='store_true')
    p.add_argument('--output', type=Path, default=OUTPUT_DEFAULT)
    p.add_argument('--scale', type=float, default=0.05)
    p.add_argument('--gripper-threshold', type=float, default=0.40)
    p.add_argument('--max-close-lag', type=int, default=3)
    p.add_argument('--freeze-run', type=int, default=5)
    p.add_argument('--jump-mm', type=float, default=60.)
    p.add_argument('--rotation-jump-deg', type=float, default=30.)
    p.add_argument('--max-ref-files', type=int, default=0,
                   help='0 = all reference trajectories; >0 = sorted prefix, descriptive only')
    args = p.parse_args()
    if args.scale <= 0 or not np.isfinite(args.scale):
        p.error('--scale must be positive finite')
    if not np.isfinite(args.gripper_threshold):
        p.error('--gripper-threshold must be finite')
    if args.max_close_lag < 0 or args.freeze_run < 1 or args.max_ref_files < 0:
        p.error('invalid lag/freeze/ref-file count')

    sources = {}
    if args.source:
        for item in args.source:
            if '=' not in item:
                p.error('expected --source NAME=PATH')
            name, path = item.split('=', 1)
            if not name.strip() or not path.strip() or name in sources:
                p.error('invalid or duplicate --source NAME=PATH')
            sources[name] = Path(path).expanduser().resolve()
    else:
        sources = SOURCES_DEFAULT

    out = args.output.expanduser().resolve()
    roots = list(sources.values())
    if not args.skip_reference:
        roots.append(args.reference.expanduser().resolve())
    for r in roots:
        if not r.is_dir():
            raise FileNotFoundError(r)
        if out == r or out.is_relative_to(r):
            raise ValueError(f'output must be outside source/reference: {out}')
    out.mkdir(parents=True, exist_ok=True)

    traj_rows, trans_rows, event_rows, metrics = [], [], [], defaultdict(list)
    counters = defaultdict(Counter)
    for name, root in sources.items():
        paths = sorted(root.glob('task_*/traj_*.pkl'))
        if not paths:
            raise RuntimeError(f'no COD task_*/traj_*.pkl in {root}')
        print(f'\n[{name}] COD trajectories: {len(paths)}', flush=True)
        for idx, path in enumerate(paths, 1):
            rel = path.relative_to(root).as_posix()
            row = {'source': name, 'relative_path': rel, 'path': str(path),
                   'status': 'ERROR', 'flags': '', 'error': ''}
            try:
                with path.with_suffix('.json').open('r', encoding='utf-8') as f:
                    j = json.load(f)
                if not isinstance(j, dict) or not all(j.get(k) == 1 for k in SUCCESS_KEYS):
                    raise ValueError('paired JSON not successful 1/1/1')
                traj = load_traj(path)
                pos, quat, qpos, command = extract_cod(traj)
                dl = delta_from_observations(pos, quat, args.scale)
                gr = gripper_events(qpos, command, args.gripper_threshold,
                                    args.max_close_lag, args.freeze_run)
                metrics[name].append(dl)
                row.update({
                    'n_frames': len(traj), 'n_transitions': len(traj)-1,
                    'xyz_p50_mm': float(np.median(dl['dp_mm'])),
                    'xyz_p95_mm': float(np.percentile(dl['dp_mm'], 95)),
                    'xyz_max_mm': float(np.max(dl['dp_mm'])),
                    'rot_p95_deg': float(np.percentile(dl['rot_deg'], 95)),
                    'rot_max_deg': float(np.max(dl['rot_deg'])),
                    'pos_rebuild_max_error_mm': float(np.max(dl['pos_err_mm'])),
                    'rot_rebuild_max_error_deg': float(np.max(dl['rot_err_deg'])),
                })
                # Non classifichiamo le traiettorie basandoci sul valore delle
                # delta rispetto al reference: il logging puo' essere diverso.
                flags = list(gr['flags'])
                if np.any(dl['dp_mm'] > args.jump_mm):
                    flags.append('XYZ_STEP_JUMP')
                if np.any(dl['rot_deg'] > args.rotation_jump_deg):
                    flags.append('ROTATION_STEP_JUMP')
                if np.max(dl['pos_err_mm']) > 0.01 or np.max(dl['rot_err_deg']) > 0.001:
                    flags.append('FLOAT32_DELTA_ROUNDTRIP_ERROR')
                # MISSING_TERMINAL_OPEN_COMMAND_INFO non e' un rifiuto: la
                # label OPEN finale sara' aggiunta nel preprocessing.
                nonblocking = {'MISSING_TERMINAL_OPEN_COMMAND_INFO'}
                review_flags = [flag for flag in flags if flag not in nonblocking]
                row['status'] = 'REVIEW' if review_flags else 'OK'
                row['flags'] = ';'.join(flags)
                row.update({k: v for k, v in gr.items() if k not in ('flags', 'obs_bin')})
                event_rows.append({
                    'source': name, 'relative_path': rel, 'status': row['status'],
                    'flags': row['flags'],
                    **{k: v for k, v in gr.items() if k not in ('flags', 'obs_bin')},
                })
                for t in range(len(traj)-1):
                    trans_rows.append({
                        'source': name, 'relative_path': rel, 'step': t,
                        'dx_label': float(dl['delta'][t, 0]),
                        'dy_label': float(dl['delta'][t, 1]),
                        'dz_label': float(dl['delta'][t, 2]),
                        'droll_label': float(dl['delta'][t, 3]),
                        'dpitch_label': float(dl['delta'][t, 4]),
                        'dyaw_label': float(dl['delta'][t, 5]),
                        'gripper_label': float(gr['obs_bin'][t+1]) / args.scale,
                        'qpos_current': float(qpos[t]),
                        'qpos_next': float(qpos[t+1]),
                        'original_command_current': float(command[t]),
                        'delta_xyz_mm': float(dl['dp_mm'][t]),
                        'delta_rotation_deg': float(dl['rot_deg'][t]),
                        'roundtrip_pos_err_mm': float(dl['pos_err_mm'][t]),
                        'roundtrip_rot_err_deg': float(dl['rot_err_deg'][t]),
                    })
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            counters[name][row['status']] += 1
            traj_rows.append(row)
            if idx == 1 or idx % 15 == 0 or row['status'] != 'OK':
                print(f'  [{idx}/{len(paths)}] {rel}: {row["status"]} {row["flags"] or row["error"]}', flush=True)
        print(f'  summary: {dict(counters[name])}', flush=True)

    ref_summary, ref_counts = {}, Counter()
    if not args.skip_reference:
        root = args.reference.expanduser().resolve()
        paths = sorted(p for p in root.glob('task_*/*traj*.pkl') if p.is_file())
        if args.max_ref_files:
            paths = paths[:args.max_ref_files]
        if not paths:
            raise RuntimeError(f'no reference task_*/*traj*.pkl in {root}')
        print(f'\n[reference] {len(paths)} canonical VLA-JEPA files', flush=True)
        ref_metrics = defaultdict(list)
        # Confronto quantitativo opzionale delle ACTION 7D gia' presenti nel
        # riferimento e delle delta ricalcolate dagli stati.
        label_pos_err, label_rot_err = defaultdict(list), defaultdict(list)
        for i, path in enumerate(paths, 1):
            subgroup = ('tfrecord' if 'tfrecord' in path.stem.lower() else
                        'scripted' if 'scripted' in path.stem.lower() else 'other')
            try:
                traj = load_traj(path)
                pos, quat, reference_actions = extract_reference(traj)
                dl = delta_from_observations(pos, quat, args.scale)
                ref_metrics[subgroup].append(dl)
                diff = reference_actions[:-1, :6].astype(np.float64) - dl['delta'].astype(np.float64)
                label_pos_err[subgroup].extend(np.linalg.norm(diff[:, :3], axis=1))
                label_rot_err[subgroup].extend(np.linalg.norm(diff[:, 3:6], axis=1))
                ref_counts[f'{subgroup}_OK'] += 1
            except Exception as exc:
                ref_counts[f'{subgroup}_ERROR'] += 1
                if ref_counts[f'{subgroup}_ERROR'] <= 10:
                    print(f'  [REFERENCE ERROR] {path}: {type(exc).__name__}: {exc}', flush=True)
            if i == 1 or i % 100 == 0:
                print(f'  reference processed {i}/{len(paths)}', flush=True)
        for subgroup, group in ref_metrics.items():
            ref_summary[subgroup] = {
                'statistics': format_summary(group),
                'reference_label_vs_state_l2_scaled_position': quantiles(label_pos_err[subgroup]),
                'reference_label_vs_state_l2_scaled_rotation': quantiles(label_rot_err[subgroup]),
                'count_trajectories': len(group),
            }

    summary = {
        'parameters': {
            'scale_factor': args.scale,
            'gripper_closed_threshold': args.gripper_threshold,
            'slow_close_feedback_if_lag_gt_frames': args.max_close_lag,
            'possible_qpos_freeze_if_transitions_at_least': args.freeze_run,
            'xyz_jump_warning_mm': args.jump_mm,
            'rotation_jump_warning_deg': args.rotation_jump_deg,
            'terminal_gripper_action_label': 0.0,
            'notes': 'Only measured states, COD original 8D action used for gripper command timing. No file modification. Reference descriptive, not same-distribution requirement.',
        },
        'cod_sources': {n: {'root': str(root), 'quality_counts': dict(counters[n]),
                            'n_pkls': sum(counters[n].values()),
                            'statistics': format_summary(metrics[n])}
                        for n, root in sources.items()},
        'reference': {'root': (None if args.skip_reference else str(args.reference)),
                      'counts': dict(ref_counts), 'subgroups': ref_summary},
    }
    traj_cols = [
        'source', 'relative_path', 'path', 'status', 'flags', 'error',
        'n_frames', 'n_transitions', 'xyz_p50_mm', 'xyz_p95_mm', 'xyz_max_mm',
        'rot_p95_deg', 'rot_max_deg', 'pos_rebuild_max_error_mm',
        'rot_rebuild_max_error_deg', 'qpos_start', 'qpos_end',
        'first_close_command_t', 'first_obs_closed_t', 'close_feedback_lag_frames',
        'obs_close_count', 'obs_open_count', 'obs_change_steps',
        'command_close_count', 'command_open_count',
        'command_change_close_steps', 'command_change_open_steps',
        'max_stuck_qpos_below_thr_close_transitions',
        'original_terminal_command', 'terminal_open_label_will_be_inserted'
    ]
    transition_cols = [
        'source', 'relative_path', 'step', 'dx_label', 'dy_label', 'dz_label',
        'droll_label', 'dpitch_label', 'dyaw_label', 'gripper_label',
        'qpos_current', 'qpos_next', 'original_command_current',
        'delta_xyz_mm', 'delta_rotation_deg',
        'roundtrip_pos_err_mm', 'roundtrip_rot_err_deg'
    ]
    event_cols = ['source', 'relative_path', 'status', 'flags'] + [
        'qpos_start', 'qpos_end', 'first_close_command_t', 'first_obs_closed_t',
        'close_feedback_lag_frames', 'obs_close_count', 'obs_open_count',
        'obs_change_steps', 'command_close_count', 'command_open_count',
        'command_change_close_steps', 'command_change_open_steps',
        'max_stuck_qpos_below_thr_close_transitions',
        'original_terminal_command', 'terminal_open_label_will_be_inserted'
    ]
    write_csv(out / 'per_trajectory.csv', traj_rows, traj_cols)
    write_csv(out / 'per_transition.csv', trans_rows, transition_cols)
    write_csv(out / 'gripper_events.csv', event_rows, event_cols)
    write_csv(out / 'flagged_trajectories.csv',
              [r for r in traj_rows if r['status'] != 'OK'], traj_cols)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    print('\n=== FINAL COD SUMMARY ===')
    for n in sources:
        print(f'{n}: {dict(counters[n])}')
        st = summary['cod_sources'][n]['statistics']
        if st:
            print(f'  xyz mm   p50={st["dp_mm"]["p50"]:.3f} '
                  f'p95={st["dp_mm"]["p95"]:.3f} max={st["dp_mm"]["max"]:.3f}')
            print(f'  rot deg  p50={st["rot_deg"]["p50"]:.3f} '
                  f'p95={st["rot_deg"]["p95"]:.3f} max={st["rot_deg"]["max"]:.3f}')
    if ref_summary:
        print('REFERENCE:')
        for sg, st in ref_summary.items():
            print(f'  {sg}: n={st["count_trajectories"]} '
                  f'xyz_p95_mm={st["statistics"]["dp_mm"]["p95"]:.3f} '
                  f'rot_p95_deg={st["statistics"]["rot_deg"]["p95"]:.3f}')
    if any(k.endswith('_ERROR') and v for k, v in ref_counts.items()):
        print('WARNING: reference errors are reported in summary.json')
    print(f'Reports saved in: {out}')
    print('  summary.json, per_trajectory.csv, per_transition.csv,')
    print('  gripper_events.csv, flagged_trajectories.csv')
    print('NO INPUT PKL/JSON WAS MODIFIED.')


if __name__ == '__main__':
    main()
