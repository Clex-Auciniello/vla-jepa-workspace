#!/usr/bin/env python3
"""Audit non distruttivo dei PKL ORIGINALI Plays per la ricostruzione delle action 7D.

Analizza pose EEF, gripper, action originali 8D, immagini, prompt unico,
metadati, eventi grasp/release, salti tra frame e stato terminale.
Non esclude né modifica alcun file. Usa il loader Plays originale E2.

Prerequisito cluster:
  export PYTHONPATH="/mnt/beegfs/a.cuciniello9/vla-jepa/vla-jepa-workspace/src:${PYTHONPATH:-}"

Esempi:
  python audit_plays_original.py --limit 10
  python audit_plays_original.py --limit 0 | tee audit_plays_original.log
  python audit_plays_original.py --limit 0 --compare-legacy
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_ROOT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/plays_trajectories')
DEFAULT_LEGACY = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/plays_preprocessed')
DEFAULT_OUT = Path('/mnt/beegfs/a.cuciniello9/datasets/robotic/plays_original_audit')
THRESHOLDS = (0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45)
IMAGE_KEYS = ('front_camera_image', 'gripper_camera_image', 'left_camera_image', 'right_camera_image')


def backend():
    try:
        from vla_jepa_offline.dataset import _open_episode
    except Exception as exc:
        raise RuntimeError(
            'Import impossibile: vla_jepa_offline.dataset._open_episode. '
            'Esporta PYTHONPATH=/mnt/beegfs/a.cuciniello9/vla-jepa/vla-jepa-workspace/src:${PYTHONPATH:-}. '
            f'Errore: {type(exc).__name__}: {exc}'
        ) from exc
    return _open_episode


def get_step(traj, idx):
    try:
        return traj.get(idx, decompress=False)
    except TypeError:
        return traj.get(idx)


def vec(x, size, name, path, i):
    if x is None:
        raise ValueError(f'{path.name} frame {i}: missing {name}')
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{path.name} frame {i}: {name} invalid {a.shape}')
    return a


def as_int_stamp(value):
    if value is None:
        return None
    try:
        x = int(value)
        return x if x > 0 else None
    except (TypeError, ValueError):
        return None


def event_type(info):
    event = str(info.get('event') or '').lower().strip()
    label = str(info.get('motion_label') or '').lower().strip()
    if 'close' in label and 'gripper' in label or 'gripper_close' in event:
        return 'close'
    if 'release' in label or ('open' in label and 'gripper' in label) or 'gripper_open' in event:
        return 'open'
    return None


def windows(T, size=8, stride=5):
    # Replica letterale della funzione aligned_window_starts del vecchio preprocessore Plays.
    starts = list(range(0, T-size+1, stride))
    if not starts:
        raise ValueError(f'No windows for {T} frames')
    if starts[-1]+size < T:
        starts[-1] = T-size
    return starts


def audit_one(path, loader, examples, legacy_path=None):
    data = loader(str(path))
    if not isinstance(data, dict):
        raise TypeError(f'{path.name}: loader returned {type(data).__name__}, expected dict')
    if data.get('source') != 'plays' or data.get('episode_id') != path.stem:
        raise ValueError(f'{path.name}: invalid source/episode_id')
    prompt = data.get('episode_prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f'{path.name}: missing episode_prompt')
    traj = data['traj']
    T = len(traj)
    if T < 8:
        raise ValueError(f'{path.name}: {T} frames (<8)')
    state_p, state_q, state_g, joint, angles, info_list = [], [], [], [], [], []
    image_shapes = Counter()
    obs_keys = Counter()
    event_labels = Counter()
    samples = []
    abs_actions = []
    stamps = []
    for i in range(T):
        step = get_step(traj, i)
        obs = step.get('obs')
        info = step.get('info')
        if not isinstance(obs, dict) or not isinstance(info, dict):
            raise ValueError(f'{path.name} frame {i}: missing obs/info')
        if info.get('index') != i+1:
            raise ValueError(f'{path.name} frame {i}: info.index={info.get("index")!r} expected {i+1}')
        obs_keys.update(obs.keys())
        for key in IMAGE_KEYS:
            im = obs.get(key)
            if im is None or not isinstance(im, np.ndarray) or im.dtype != np.uint8 or im.ndim != 3 or im.shape[-1] != 3:
                raise ValueError(f'{path.name} frame {i}: invalid BGR image {key}')
            image_shapes[f'{key}:{tuple(im.shape)}:{im.dtype}'] += 1
        p = vec(obs.get('eef_pos'), 3, 'eef_pos', path, i)
        q = vec(obs.get('eef_quat'), 4, 'eef_quat', path, i)
        g = vec(obs.get('gripper_qpos'), 1, 'gripper_qpos', path, i)[0]
        jp = vec(obs.get('joint_pos'), 6, 'joint_pos', path, i)
        jv = vec(obs.get('joint_vel'), 6, 'joint_vel', path, i)
        aq = vec(step.get('action'), 8, 'original action 8D', path, i)
        if abs(np.linalg.norm(q) - 1.0) > 1e-3:
            raise ValueError(f'{path.name} frame {i}: non-unit quaternion')
        state_p.append(p); state_q.append(q); state_g.append(g); joint.append(jp); abs_actions.append(aq)
        stamp = as_int_stamp(info.get('front_stamp_ns'))
        stamps.append(stamp)
        info_list.append(info)
        if info.get('motion_label') is not None:
            event_labels[str(info['motion_label'])] += 1
        if len(samples) < examples and (i in {0, T//2, T-1}):
            samples.append({'frame': i, 'eef_pos': p.tolist(), 'eef_quat': q.tolist(),
                            'gripper_qpos': float(g), 'motion_label': info.get('motion_label'),
                            'event': info.get('event'), 'action8': aq.tolist(),
                            'front_stamp_ns': stamp})
    p = np.stack(state_p)
    q = np.stack(state_q)
    gr = np.asarray(state_g)
    joint = np.stack(joint)
    actions = np.stack(abs_actions)
    rot = Rotation.from_quat(q)
    dr = rot[1:] * rot[:-1].inv()
    xyz_steps_mm = np.linalg.norm(np.diff(p, axis=0), axis=1)*1000
    rot_steps_deg = np.degrees(dr.magnitude())
    joint_steps = np.linalg.norm(np.diff(joint, axis=0), axis=1)
    xyz_idx = int(np.argmax(xyz_steps_mm))
    rot_idx = int(np.argmax(rot_steps_deg))
    joint_max = float(np.max(joint_steps))
    final_info = info_list[-1]
    has_final = str(final_info.get('event')) == 'final_observation'
    evs = []
    for i, info in enumerate(info_list):
        kind = event_type(info)
        if kind is None:
            continue
        row = {'file':path.name,'play_type':str(data.get('play_type')),'frame':i,
               'kind':kind,'event':info.get('event'),'label':info.get('motion_label'),
               'previous_qpos':round(float(gr[i-1]),6) if i else None,
               'qpos_at_event':round(float(gr[i]),6),
               'next_qpos':round(float(gr[i+1]),6) if i+1<T else None,
               'last_frame':i==T-1}
        for th in (0.20, 0.40):
            closed = gr >= th
            matched = closed[i:] if kind=='close' else ~closed[i:]
            hits = np.flatnonzero(matched)
            row[f'frames_to_{kind}_{th:.2f}'] = int(hits[0]) if len(hits) else None
        evs.append(row)
    has_close = any(e['kind']=='close' for e in evs)
    has_open = any(e['kind']=='open' for e in evs)
    first_stamps = [int(s) for s in stamps if s is not None]
    valid_all_stamps = len(first_stamps)==T
    gaps = np.diff(first_stamps)/1e9 if valid_all_stamps else np.array([])
    row = {
        'file':path.name,'frames':T,'play_type':str(data.get('play_type')),
        'operator_success':data.get('operator_success'),
        'operator_approved':data.get('operator_approved'),
        'prompt_mode':data.get('prompt_mode'), 'episode_prompt':prompt,
        'window_count':len(windows(T)), 'last_window_start':windows(T)[-1],
        'source_action_dim':8,
        'pos_valid':T,'quat_valid':T,'grip_valid':T,
        'xyz_p95_mm':round(float(np.percentile(xyz_steps_mm,95)),6),
        'xyz_max_mm':round(float(xyz_steps_mm.max()),6),
        'xyz_jump_index':xyz_idx,
        'xyz_jump_from_event':info_list[xyz_idx].get('event'),
        'xyz_jump_to_event':info_list[xyz_idx+1].get('event'),
        'rot_p95_deg':round(float(np.percentile(rot_steps_deg,95)),6),
        'rot_max_deg':round(float(rot_steps_deg.max()),6),
        'rot_jump_index':rot_idx,
        'joint_max_delta_norm_rad':round(joint_max,8),
        'eef_static':bool(np.all(xyz_steps_mm <= 1e-8) and np.all(rot_steps_deg <= 1e-8)),
        'xyz_jump_gt_100mm':bool(np.any(xyz_steps_mm >100)),
        'gripper_min':round(float(gr.min()),6),
        'gripper_max':round(float(gr.max()),6),
        'has_close_label':int(has_close), 'has_open_label':int(has_open),
        'close_events':sum(e['kind']=='close' for e in evs),
        'open_events':sum(e['kind']=='open' for e in evs),
        'terminal_event':final_info.get('event'),
        'terminal_motion_label':final_info.get('motion_label'),
        'terminal_qpos':round(float(gr[-1]),6),
        'terminal_original_action_gripper':float(actions[-1,7]),
        'terminal_recorded_open_at_0.20':bool(gr[-1]<0.20),
        'has_final_observation':has_final,
        'all_front_timestamps':valid_all_stamps,
        'max_front_frame_gap_s':round(float(gaps.max()),6) if gaps.size else None,
        'nonincreasing_front_timestamp_count':int(np.sum(gaps<=0)) if gaps.size else None,
        'legacy_check':'not_requested', 'status':'OK'
    }
    for th in THRESHOLDS:
        closed = (gr >= th).astype(np.int8)
        edges = np.diff(closed)
        row[f'closed_at_{th:.2f}'] = int(np.any(closed))
        row[f'close_edges_at_{th:.2f}'] = int(np.sum(edges==1))
        row[f'open_edges_at_{th:.2f}'] = int(np.sum(edges==-1))
        row[f'closed_frames_at_{th:.2f}'] = int(closed.sum())
        row[f'terminal_open_at_{th:.2f}'] = int(not closed[-1])
    if legacy_path is not None:
        if not legacy_path.is_file():
            row['legacy_check']='MISSING'
        else:
            with legacy_path.open('rb') as f:
                legacy = pickle.load(f)
            if (legacy.get('episode_prompt') != prompt
                or legacy.get('window_starts') != windows(T)
                or len(legacy['traj']) != T):
                raise ValueError(f'{path.name}: legacy prompt/windows/frame count mismatch')
            row['legacy_check']='prompt_windows_frames_MATCH'
    freq = Counter(f'{x:.6f}' for x in gr)
    return row, {'file':path.name,'episode_prompt':prompt,'play_type':str(data.get('play_type')),
                 'samples':samples,'obs_key_counts':dict(obs_keys),
                 'image_shape_counts':dict(image_shapes)}, evs, freq, event_labels


def save_csv(path, records):
    if not records:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        fields = list(records[0])
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)


def json_write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--limit',type=int,default=0,help='0=tutti')
    ap.add_argument('--examples',type=int,default=3,help='Quanti PKL campionare nello schema JSON')
    ap.add_argument('--compare-legacy',action='store_true',help='Controlla prompt/finestre/T nei vecchi Plays preprocessati')
    ap.add_argument('--legacy-root',type=Path,default=DEFAULT_LEGACY)
    args=ap.parse_args()
    if args.limit<0 or args.examples<0:
        ap.error('--limit e --examples devono essere >=0')
    if args.root.resolve()==args.output.resolve():
        ap.error('Non usare output uguale alla sorgente')
    paths = sorted((p for p in args.root.glob('P*.pkl') if re.fullmatch(r'P\d{5,}\.pkl',p.name)),
                   key=lambda p:int(p.stem[1:]))
    if not paths:
        ap.error(f'Nessun PKL Plays originale trovato: {args.root}')
    discovered=len(paths)
    if args.limit: paths=paths[:args.limit]
    loader=backend()
    rows=[]; samples=[]; events=[]; freq=Counter(); labels=Counter(); obs_keys=Counter(); image_shapes=Counter()
    failures=[]
    for num, path in enumerate(paths,1):
        try:
            row, sample, evs, fq, lbl = audit_one(
                path,loader,3, args.legacy_root/path.name if args.compare_legacy else None)
            rows.append(row)
            if len(samples)<args.examples: samples.append(sample)
            events.extend(evs); freq.update(fq); labels.update(lbl)
            obs_keys.update(sample['obs_key_counts']); image_shapes.update(sample['image_shape_counts'])
            print(f'[{num}/{len(paths)}] {path.name} OK | T={row["frames"]} | '
                  f'type={row["play_type"]} | ΔXYZ max={row["xyz_max_mm"]:.2f}mm | '
                  f'terminal qpos={row["terminal_qpos"]:.4f} | '
                  f'final={row["has_final_observation"]}',flush=True)
        except Exception as exc:
            failures.append({'file':path.name,'status':'ERROR',
                             'error':f'{type(exc).__name__}: {exc}'})
            print(f'[{num}/{len(paths)}] {path.name} ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    save_csv(args.output/'episodes.csv',rows)
    save_csv(args.output/'errors.csv',failures)
    save_csv(args.output/'gripper_events.csv',events)
    save_csv(args.output/'gripper_distribution.csv',
             [{'qpos':key,'count':n} for key,n in sorted(freq.items(),key=lambda kv:float(kv[0]))])
    save_csv(args.output/'movement_outliers.csv',
             [r for r in rows if r['eef_static'] or r['xyz_jump_gt_100mm']])
    thresholds=[]
    for th in THRESHOLDS:
        k=f'{th:.2f}'
        tp=sum(int(r['has_close_label'] and r[f'closed_at_{k}']) for r in rows)
        fn=sum(int(r['has_close_label'] and not r[f'closed_at_{k}']) for r in rows)
        fp=sum(int(not r['has_close_label'] and r[f'closed_at_{k}']) for r in rows)
        tn=sum(int(not r['has_close_label'] and not r[f'closed_at_{k}']) for r in rows)
        thresholds.append({'threshold':k,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                           'close_transitions':sum(r[f'close_edges_at_{k}'] for r in rows),
                           'open_transitions':sum(r[f'open_edges_at_{k}'] for r in rows),
                           'closed_frames':sum(r[f'closed_frames_at_{k}'] for r in rows),
                           'terminal_observed_open':sum(r[f'terminal_open_at_{k}'] for r in rows)})
    save_csv(args.output/'gripper_thresholds.csv',thresholds)
    summary={
        'source':str(args.root),'legacy_root':str(args.legacy_root) if args.compare_legacy else None,
        'discovered':discovered,'analyzed':len(paths),'ok':len(rows),'errors':len(failures),
        'total_frames':sum(r['frames'] for r in rows),
        'per_play_type':dict(sorted(Counter(r['play_type'] for r in rows).items())),
        'operator_success_false':sum(r['operator_success'] is False for r in rows),
        'operator_approved_false':sum(r['operator_approved'] is False for r in rows),
        'missing_final_observation':sum(not r['has_final_observation'] for r in rows),
        'static_eef':sum(r['eef_static'] for r in rows),
        'xyz_jump_gt_100mm':sum(r['xyz_jump_gt_100mm'] for r in rows),
        'nonincreasing_timestamp_episodes':sum((r['nonincreasing_front_timestamp_count'] or 0)>0 for r in rows),
        'motion_labels_frequency':dict(labels.most_common()),
        'obs_key_frequency':dict(obs_keys),
        'image_shapes_frequency':dict(image_shapes),
        'gripper_event_counts':{'close':sum(x['kind']=='close' for x in events),
                                'open':sum(x['kind']=='open' for x in events)},
        'gripper_thresholds':thresholds,
        'legacy_checks':dict(Counter(r['legacy_check'] for r in rows))
    }
    json_write(args.output/'summary.json',summary)
    json_write(args.output/'episodes.json',rows)
    json_write(args.output/'schema_samples.json',samples)
    print('\n== RIEPILOGO ==')
    print(json.dumps({k:v for k,v in summary.items() if k not in (
        'motion_labels_frequency','obs_key_frequency','image_shapes_frequency','gripper_thresholds')},indent=2,ensure_ascii=False))
    print('\nGRIPPER (test statistico su episodi):')
    for r in thresholds:
        print(f"  threshold={r['threshold']}: TP={r['tp']} FP={r['fp']} "
              f"FN={r['fn']} TN={r['tn']} | open_edges={r['open_transitions']} "
              f"| terminal_open={r['terminal_observed_open']}")
    print(f'\nReport in {args.output}; originale mai modificato.')
    if failures:
        sys.exit(1)


if __name__=='__main__':
    main()
