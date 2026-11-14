#!/usr/bin/env python3
"""Preprocess selected VLA-JEPA exploration episodes for multimodal action training.

- Reads ORIGINAL exploration/E*.pkl (not the old video-only preprocessed PKLs).
- Uses the complete 440-episode audit to exclude exactly 33 static-EEF
  episodes and 7 episodes with a single observed XYZ step >100 mm.
- Reuses the ORIGINAL world-model transforms, prompt construction and windows
  from vla_jepa_online.training_server.online_batch_builder, unchanged.
- Validates the new RGB images and 8-frame prompts against the existing
  robotic/exploration_preprocessed PKLs, byte-for-byte (unless disabled).
- Adds 7-D actions from consecutive OBSERVED poses and gripper qpos, following
  the canonical COD / OSVI / VLA-JEPA convention (scale 0.05).
- Final action [0]*7 is a SYNTHETIC stationary+OPEN label. The last OBSERVED
  state/image is unmodified and no physical opening is claimed.
- Preserves source metadata and motion_label for future video WM training.

MODES:
  gripper-audit: read selected 400, analyze thresholds/events, write diagnostics.
  dry-run:       calculate/verify outputs and compare with old WM PKLs, write nothing.
  execute:       atomically save new PKLs, manifest.csv, excluded.csv.

Run only on trusted pickle files. Source and legacy datasets are NEVER modified.
Required import: export PYTHONPATH=/.../vla-jepa-workspace/src:$PYTHONPATH
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import re
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/mnt/beegfs/a.cuciniello9/datasets')
ORIGINAL = ROOT / 'robotic/exploration'
LEGACY = ROOT / 'robotic/exploration_preprocessed'
AUDIT = ROOT / 'robotic/exploration_original_audit/episodes.json'
DEST = ROOT / 'mixture/exploration_action_preprocessed'
AUDIT_OUT = ROOT / 'robotic/exploration_action_audit'
SCALE = 0.05
STATIC_EXPECTED = 33
JUMP_EXPECTED = 7
TOTAL_EXPECTED = 440
KEEP_EXPECTED = 400
JUMP_LIMIT_MM = 100.0
THRESHOLDS = (0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45)


def load_backend():
    try:
        from vla_jepa_online.training_server.online_batch_builder import (
            OnlineBatchBuilderConfig, _open_exploration_pkl,
            _pkl_bgr_to_tensor, _prompt_for_window, _window_starts,
        )
    except Exception as exc:
        raise RuntimeError(
            'Import del batch builder originale fallito. Esporta ' 
            'PYTHONPATH=/mnt/beegfs/a.cuciniello9/vla-jepa/vla-jepa-workspace/src:${PYTHONPATH:-} '
            'e usa l\'ambiente che ha eseguito il precedente audit. '
            f'Errore: {type(exc).__name__}: {exc}'
        ) from exc
    return (OnlineBatchBuilderConfig, _open_exploration_pkl,
            _pkl_bgr_to_tensor, _prompt_for_window, _window_starts)


def get_step(traj, t):
    try:
        return traj.get(t, decompress=False)
    except TypeError:
        return traj.get(t)


def vector(value, size, path, t, name):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(f'{path} frame={t}: invalid {name} shape/values {arr.shape}')
    return arr


def require_native(traj, path):
    raw = getattr(traj, '_data', None)
    if not isinstance(raw, list) or len(raw) != len(traj) or len(raw) < 8:
        raise ValueError(f'{path}: invalid native Trajectory / <8 frames')
    for i, row in enumerate(raw):
        if not isinstance(row, tuple) or len(row) != 5 or not isinstance(row[0], dict):
            raise TypeError(f'{path} frame={i}: expected (obs,reward,done,info,action)')
    return raw


def selection(original_root: Path, audit_json: Path):
    with audit_json.open(encoding='utf-8') as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) != TOTAL_EXPECTED:
        raise ValueError(f'{audit_json}: expected {TOTAL_EXPECTED} records')
    by_name = {}
    static, jumps, kept, exclusions = [], [], [], []
    for row in rows:
        name = row.get('file')
        if not isinstance(name, str) or not re.fullmatch(r'E\d{5,}\.pkl', name):
            raise ValueError(f'Bad audit filename: {name!r}')
        if name in by_name:
            raise ValueError(f'Duplicate audit entry: {name}')
        if row.get('status') != 'OK' or row.get('frames', 0) < 8:
            raise ValueError(f'Audit row not OK or too short: {name}')
        if any(row.get(k) != row['frames'] for k in ('pos_valid', 'quat_valid', 'grip_valid')):
            raise ValueError(f'Incomplete pose/gripper in {name}')
        if row.get('xyz_max_mm') is None or row.get('rot_max_deg') is None:
            raise ValueError(f'Missing movement statistics for {name}')
        by_name[name] = row
        # Same predicates used to identify the 33+7 in the completed audit.
        is_static = row['xyz_max_mm'] == 0.0 and row['rot_max_deg'] == 0.0
        is_jump = row['xyz_max_mm'] > JUMP_LIMIT_MM
        if is_static and is_jump:
            raise AssertionError('Static and >100 mm jump overlap')
        if is_static:
            static.append(name)
            exclusions.append({'file': name, 'reason': 'static_eef',
                               'frames': row['frames'], 'xyz_max_mm': row['xyz_max_mm']})
        elif is_jump:
            jumps.append(name)
            exclusions.append({'file': name, 'reason': 'xyz_step_gt_100mm',
                               'frames': row['frames'], 'xyz_max_mm': row['xyz_max_mm']})
        else:
            kept.append(name)
    found = {p.name for p in original_root.glob('E*.pkl')
             if re.fullmatch(r'E\d{5,}\.pkl', p.name)}
    if found != set(by_name):
        raise ValueError(f'Original set differs from audit: missing={sorted(set(by_name)-found)[:5]} '
                         f'extra={sorted(found-set(by_name))[:5]}')
    if (len(static), len(jumps), len(kept)) != (STATIC_EXPECTED, JUMP_EXPECTED, KEEP_EXPECTED):
        raise ValueError(f'Wrong exclusion counts static/jump/keep: '
                         f'{len(static)}/{len(jumps)}/{len(kept)}')
    print(f'Original={len(by_name)} | static excluded={len(static)} '
          f'| jumps excluded={len(jumps)} | selected={len(kept)}', flush=True)
    return sorted(kept, key=lambda x: int(x[1:-4])), exclusions, by_name


def extract_states(traj, path):
    native = require_native(traj, path)
    obs_list = [row[0] for row in native]
    p = np.stack([vector(o['eef_pos'], 3, path, t, 'eef_pos')
                  for t, o in enumerate(obs_list)])
    q = np.stack([vector(o['eef_quat'], 4, path, t, 'eef_quat')
                  for t, o in enumerate(obs_list)])
    g = np.array([vector(o['gripper_qpos'], 1, path, t, 'gripper_qpos')[0]
                  for t, o in enumerate(obs_list)], dtype=np.float64)
    if np.any(abs(np.linalg.norm(q, axis=1)-1) > 1e-3):
        raise ValueError(f'{path}: quaternion nonunit')
    for t, row in enumerate(native):
        info = row[3]
        if not isinstance(info, dict) or info.get('index') != t+1:
            raise ValueError(f'{path} frame={t}: invalid source info.index')
        if vector(row[4], 8, path, t, 'source absolute 8D').shape != (8,):
            raise AssertionError('Unexpected action')
    rot = Rotation.from_quat(q)
    dif = np.diff(p, axis=0)
    rel = rot[1:] * rot[:-1].inv()
    step_mm = np.linalg.norm(dif, axis=1)*1000
    rot_deg = np.degrees(rel.magnitude())
    if np.max(step_mm) > JUMP_LIMIT_MM + 1e-6:
        raise ValueError(f'{path}: unexpected >100mm step after filtering')
    if np.all(step_mm == 0) and np.all(rot_deg == 0):
        raise ValueError(f'{path}: unexpected static EEF after filtering')
    return native, p, q, g, rot, dif, rel, step_mm, rot_deg


def gripper_audit(names, original_root, loader, output, audit_rows, subset=None):
    output.mkdir(parents=True, exist_ok=True)
    overview, events, freq = [], [], Counter()
    episode_stats = {th: {'tp':0, 'fp':0, 'fn':0, 'tn':0,
                          'close_transitions':0, 'open_transitions':0,
                          'state_frames_closed':0} for th in THRESHOLDS}
    for idx, name in enumerate(names, 1):
        path = original_root / name
        traj = loader(path)
        native, p, q, gr, rot, dif, rel, step_mm, rot_deg = extract_states(traj, path)
        infos = [row[3] for row in native]
        has_close = any((inf.get('motion_label') or '').strip().lower() == 'close the gripper'
                        for inf in infos)
        for v in gr:
            freq[f'{v:.6f}'] += 1
        summary = {'file':name, 'frames':len(gr),
                   'has_close_label':int(has_close),
                   'close_label_count':sum(inf.get('motion_label') == 'Close the gripper' for inf in infos),
                   'open_label_count':sum(inf.get('motion_label') == 'Open the gripper' for inf in infos),
                   'min_qpos':f'{np.min(gr):.6f}', 'max_qpos':f'{np.max(gr):.6f}'}
        for th in THRESHOLDS:
            state = (gr >= th).astype(np.uint8)
            d = np.diff(state.astype(np.int8))
            has_state = bool(state.any())
            st = episode_stats[th]
            key = 'tp' if (has_close and has_state) else ('fn' if has_close else ('fp' if has_state else 'tn'))
            st[key] += 1
            st['close_transitions'] += int(np.sum(d == 1))
            st['open_transitions'] += int(np.sum(d == -1))
            st['state_frames_closed'] += int(state.sum())
            summary[f'closed_at_{th:.2f}'] = int(has_state)
            summary[f'close_edges_at_{th:.2f}'] = int(np.sum(d==1))
            summary[f'open_edges_at_{th:.2f}'] = int(np.sum(d==-1))
        overview.append(summary)
        for t, info in enumerate(infos):
            label = info.get('motion_label')
            if label not in ('Close the gripper', 'Open the gripper'):
                continue
            ev = {'file':name, 'frame':t, 'event':info.get('event'), 'label':label,
                  'prev_qpos': f'{gr[t-1]:.6f}' if t>0 else '',
                  'qpos_at_event':f'{gr[t]:.6f}'}
            for plus in (1,2,3,5):
                ev[f'qpos_plus_{plus}'] = f'{gr[t+plus]:.6f}' if t+plus<len(gr) else ''
            for th in (0.20,0.40):
                state = (gr >= th)
                ev[f'is_closed_at_{th:.2f}'] = int(state[t])
                following = np.flatnonzero(state[t:] if label=='Close the gripper' else ~state[t:])
                ev[f'frames_to_target_state_{th:.2f}'] = (int(following[0]) if len(following) else '')
            events.append(ev)
        if idx % 50 == 0 or idx == len(names):
            print(f'[GRIPPER AUDIT] {idx}/{len(names)}', flush=True)
        del traj
        gc.collect()
    thresholds = []
    for th in THRESHOLDS:
        s=episode_stats[th]
        thresholds.append({'threshold':f'{th:.2f}', **s, 'close_labeled_episodes':s['tp']+s['fn'],
                           'closed_observed_episodes':s['tp']+s['fp']})
    dump_csv(output/'gripper_thresholds.csv', thresholds)
    dump_csv(output/'gripper_episodes.csv', overview)
    dump_csv(output/'gripper_events.csv', events)
    dump_csv(output/'gripper_distribution.csv', [dict(qpos=k, count=v) for k,v in sorted(freq.items(), key=lambda it:float(it[0]))])
    (output/'gripper_summary.json').write_text(json.dumps({'episodes':len(names),
        'thresholds':thresholds,
        'close_events':sum(e['label']=='Close the gripper' for e in events),
        'open_events':sum(e['label']=='Open the gripper' for e in events)},indent=2))
    print('\n=== GRIPPER THRESHOLDS (episode-level) ===')
    print('threshold  TP  FP  FN  TN  close_edges  open_edges')
    for r in thresholds:
        print(f"{r['threshold']:>9}  {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
              f"{r['tn']:>3}  {r['close_transitions']:>11} {r['open_transitions']:>11}")
    print(f'Diagnostics in: {output}')


def dump_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def wm_preprocess_image(image, crop, image_size, tensorizer):
    # This block is IDENTICAL to the old WM preprocessor.
    tensor = tensorizer(image, crop, image_size)
    return tensor.mul(255).round_().byte().permute(1, 2, 0).contiguous().numpy()


def load_legacy(path, expected_length, cfg):
    with path.open('rb') as f:
        p = pickle.load(f)
    if not isinstance(p,dict) or 'traj' not in p:
        raise ValueError(f'{path}: invalid legacy payload')
    legacy_traj = p['traj']
    if len(legacy_traj) != expected_length:
        raise ValueError(f'{path}: legacy frames mismatch: {len(legacy_traj)} != {expected_length}')
    meta = p.get('preprocessing',{})
    if (meta.get('window_size') != cfg.window_size or meta.get('stride') != cfg.stride
            or list(meta.get('image_size', [])) != list(cfg.image_size)
            or meta.get('color_order') != 'RGB'
            or meta.get('resize_interpolation') != 'cv2.INTER_LINEAR'
            or meta.get('gripper_crop') is not None):
        raise ValueError(f'{path}: old WM preprocessing configuration differs from current builder')
    crop = cfg.front_crop
    expected_crop = [crop.top, crop.bottom, crop.left, crop.right] if crop else None
    if meta.get('front_crop_margins_top_bottom_left_right') != expected_crop:
        raise ValueError(f'{path}: front crop differs from current builder')
    if not isinstance(meta.get('align_last_window'), bool):
        raise ValueError(f'{path}: missing align_last_window: cannot reproduce windows')
    return p, bool(meta['align_last_window'])


def build_episode(src, loader, cfg, tensorizer, prompt_for_window, window_starts,
                  threshold, legacy_file, compare_legacy=True):
    traj = loader(src)
    native, pos, quat, gr, rot, dif, rel, step_mm, rot_deg = extract_states(traj, src)
    T = len(native)
    if not (cfg.window_size <= T <= cfg.max_frames):
        raise ValueError(f'{src}: frame count {T} outside old window builder rules')
    if cfg.window_size != 8 or cfg.stride != 5 or list(cfg.image_size) != [224, 224]:
        raise ValueError(f'{src}: old WM window/image settings changed: '
                         f'window={cfg.window_size}, stride={cfg.stride}, size={cfg.image_size}')
    if cfg.gripper_crop is not None:
        raise ValueError(f'{src}: expected no gripper crop in legacy WM')
    if compare_legacy:
        legacy, align_last_window = load_legacy(legacy_file, T, cfg)
    else:
        legacy, align_last_window = None, False

    binary = (gr >= threshold).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Gimbal lock detected')
        euler_abs = rot.as_euler('xyz')
        euler_rel = rel.as_euler('xyz')
    actions = np.zeros((T,7),dtype=np.float32)
    actions[:-1,:3] = dif/SCALE
    actions[:-1,3:6] = euler_rel/SCALE
    actions[:-1,6] = binary[1:].astype(np.float32)/SCALE
    # User-approved terminal convention; never claim final OPEN is observed.
    actions[-1] = 0.0

    pos2 = pos[:-1] + actions[:-1,:3].astype(np.float64)*SCALE
    rot2 = Rotation.from_euler('xyz',actions[:-1,3:6].astype(np.float64)*SCALE) * rot[:-1]
    max_err_mm = float(np.max(np.linalg.norm(pos2-pos[1:],axis=1))*1000)
    max_err_deg = float(np.max(np.degrees((rot2.inv()*rot[1:]).magnitude())))
    if max_err_mm > 0.01 or max_err_deg > 0.001:
        raise ValueError(f'{src}: reconstructed action round-trip error '
                         f'xyz={max_err_mm}mm, rot={max_err_deg}deg')
    converted = traj.__class__(config_str=traj.config_str)
    labels=[]
    for t, old in enumerate(native):
        raw_obs, reward, done, info, command8 = old
        labels.append({'motion_label':info.get('motion_label')})
        front = wm_preprocess_image(raw_obs['front_camera_image'],cfg.front_crop,cfg.image_size,tensorizer)
        gripper = wm_preprocess_image(raw_obs['gripper_camera_image'],cfg.gripper_crop,cfg.image_size,tensorizer)
        for nm, im in [('camera_front_image',front),('camera_gripper_image',gripper)]:
            if im.shape != (224,224,3) or im.dtype != np.uint8:
                raise ValueError(f'{src} t={t}: malformed {nm} {im.shape} {im.dtype}')
        if legacy is not None:
            prior = get_step(legacy['traj'],t)['obs']
            for nm, im in [('camera_front_image',front),('camera_gripper_image',gripper)]:
                if not np.array_equal(prior[nm],im):
                    raise ValueError(f'{src} t={t} {nm}: pixels DO NOT match old WM preprocessing')
        eef = np.r_[pos[t],euler_abs[t]].astype(np.float32)
        joint_pos = vector(raw_obs['joint_pos'],6,src,t,'joint_pos').astype(np.float32)
        joint_vel = vector(raw_obs['joint_vel'],6,src,t,'joint_vel').astype(np.float32)
        # Keep non-image sensor readings and add standardized aliases.
        obs = {key:val for key,val in raw_obs.items()
               if 'image' not in key.lower() and 'depth' not in key.lower()}
        obs['camera_front_image'] = front
        obs['camera_gripper_image'] = gripper
        obs['EEF_state'] = eef
        obs['gripper_state'] = np.asarray([gr[t],float(binary[t])],dtype=np.float32)
        obs['joint_positions'] = joint_pos
        obs['joint_velocities'] = joint_vel
        obs['action_world'] = np.r_[eef,float(binary[t])].astype(np.float32)
        converted._data.append((obs, reward, t==T-1, dict(info), actions[t].copy()))
        converted._raw_state.append(None)

    starts = window_starts(T, window_size=cfg.window_size, stride=cfg.stride,
                           repeat_last=align_last_window)
    windows = [{'start':start,'prompt':prompt_for_window(labels[start:start+cfg.window_size])}
               for start in starts]
    if legacy is not None and windows != legacy['window_prompts']:
        raise ValueError(f'{src}: old WM window prompts differ from reconstructed prompts')
    if len(converted) != T:
        raise AssertionError(f'{src}: wrong output frames')
    meta = {
        'front_crop_margins_top_bottom_left_right': (
            [cfg.front_crop.top,cfg.front_crop.bottom,cfg.front_crop.left,cfg.front_crop.right]
            if cfg.front_crop else None),
        'gripper_crop': None,
        'image_size':list(cfg.image_size),
        'color_order':'RGB',
        'resize_interpolation':'cv2.INTER_LINEAR',
        'window_size':cfg.window_size,'stride':cfg.stride,
        'align_last_window':align_last_window,
    }
    payload={'episode_id':src.stem, 'source':'exploration',
             'source_file':str(src.resolve()), 'traj':converted,
             'window_prompts':windows, 'preprocessing':meta,
             'action_preprocessing':{
                 'version':'exploration_actions_v1',
                 'action_shape':7, 'action_dtype':'float32',
                 'action_scale_factor':SCALE,
                 'action_alignment':'state[t+1] relative to state[t] for t<T-1',
                 'rotation':'Rotation[t+1] * Rotation[t].inv(); Euler xyz',
                 'gripper_qpos_threshold':float(threshold),
                 'gripper_action_values':{'open':0.0,'closed':20.0},
                 'terminal_action':'synthetic 7D zeros: stationary EEF plus OPEN',
                 'terminal_gripper_open_is_observed':False,
                 'original_source_action':'8D absolute target; preserved in info.commanded_action',
                 'source_selection':'33 static EEF + 7 XYZ step>100mm excluded',
                 'legacy_rgb_and_windows_verified':bool(legacy is not None),
             }}
    summary={'file':src.name,'output_pkl':src.name,'frames':T,'transitions':T-1,
             'window_prompts':len(windows),
             'align_last_window':int(align_last_window),
             'gripper_threshold':threshold,
             'has_close_label':int(any(i.get('motion_label')=='Close the gripper' for i in (r[3] for r in native))),
             'observed_closed':int(binary.any()),
             'closed_frames':int(binary.sum()),
             'xyz_max_mm':round(float(step_mm.max()),6),
             'rot_max_deg':round(float(rot_deg.max()),6),
             'rebuild_xyz_err_mm':round(max_err_mm,9),
             'rebuild_rot_err_deg':round(max_err_deg,9),
             'verified_equal_to_old_wm':int(legacy is not None)}
    # Caller holds only one episode at a time, then releases memory.
    return payload, summary


def verify_saved(path, length, expected_windows):
    with path.open('rb') as h:
        result = pickle.load(h)
    if len(result['traj']) != length or len(result['window_prompts']) != expected_windows:
        raise ValueError(f'{path}: saved episode length/prompts mismatch')
    for t in range(length):
        row = get_step(result['traj'],t)
        action = np.asarray(row['action'])
        if action.dtype != np.float32 or action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f'{path} frame={t}: malformed 7D action')
        if t == length-1:
            if not np.array_equal(action,np.zeros(7,dtype=np.float32)):
                raise ValueError(f'{path}: terminal action should be all-zero')
        else:
            if action[6] not in (0.0,20.0):
                raise ValueError(f'{path} frame={t}: invalid gripper action')
            next_obs = get_step(result['traj'], t+1)['obs']
            if action[6] != np.float32(next_obs['gripper_state'][1]/SCALE):
                raise ValueError(f'{path} frame={t}: gripper t+1 label mismatch')
        obs = row['obs']
        image_keys = {key for key in obs if 'image' in key.lower() or 'depth' in key.lower()}
        if image_keys != {'camera_front_image','camera_gripper_image'}:
            raise ValueError(f'{path} frame={t}: unexpected image keys {image_keys}')
        for key in image_keys:
            a=np.asarray(obs[key])
            if a.shape != (224,224,3) or a.dtype != np.uint8:
                raise ValueError(f'{path} frame={t}: malformed {key}')
        for key,n in [('EEF_state',6),('gripper_state',2),('action_world',7),
                       ('joint_positions',6),('joint_velocities',6)]:
            a=np.asarray(obs[key]);
            if a.shape != (n,) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f'{path} frame={t}: malformed {key}')


def atomic_pickle(payload, destination, nframes, nwindows):
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=destination.parent,
                                         prefix='.'+destination.stem+'_',suffix='.tmp',delete=False) as out:
            temp=Path(out.name)
            pickle.dump(payload,out,protocol=pickle.HIGHEST_PROTOCOL)
            out.flush();os.fsync(out.fileno())
        verify_saved(temp,nframes,nwindows)
        os.replace(temp,destination)
        temp=None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--mode',choices=('gripper-audit','dry-run','execute'),required=True)
    a.add_argument('--source',type=Path,default=ORIGINAL)
    a.add_argument('--legacy',type=Path,default=LEGACY)
    a.add_argument('--audit-json',type=Path,default=AUDIT)
    a.add_argument('--destination',type=Path,default=DEST)
    a.add_argument('--audit-output',type=Path,default=AUDIT_OUT)
    a.add_argument('--threshold',type=float,default=0.20,
                   help='Provisional 0.20; use gripper-audit first to confirm')
    a.add_argument('--limit',type=int,default=0,help='Optional pilot subset (0=all 400)')
    a.add_argument('--no-verify-legacy',action='store_true',
                   help='Skip pixel/prompt comparison with old video-only WM output (not recommended)')
    args=a.parse_args()
    if args.limit < 0 or not (0.1 < args.threshold < 0.8):
        a.error('Invalid limit or threshold')
    orig=args.source.expanduser().resolve(); leg=args.legacy.expanduser().resolve()
    dest=args.destination.expanduser().resolve()
    if not orig.is_dir():a.error(f'Original not found: {orig}')
    if not args.audit_json.is_file():a.error(f'Audit episodes.json not found: {args.audit_json}')
    if args.mode != 'gripper-audit':
        if orig==dest or orig in dest.parents or dest in orig.parents:
            a.error('Source and destination overlap')
        if leg==dest or leg in dest.parents or dest in leg.parents:
            a.error('Legacy and destination overlap')
        if not args.no_verify_legacy and not leg.is_dir():
            a.error(f'Legacy WM dataset not found: {leg}')
    names,exclusions,audit_rows=selection(orig,args.audit_json)
    if args.limit:
        names=names[:args.limit]
    backend=load_backend()
    cfg=backend[0]()
    loader,tensorizer,prompter,windower=backend[1:]
    if args.mode=='gripper-audit':
        gripper_audit(names,orig,loader,args.audit_output.expanduser().resolve(),audit_rows)
        return
    if args.mode=='execute':
        if dest.exists() and any(dest.iterdir()):
            a.error(f'Output directory is not empty: {dest} (use a fresh destination)')
        if args.limit and dest==DEST.resolve():
            a.error('Pilot must specify a distinct --destination')
    all_rows=[]
    print(f'Mode={args.mode}, selected={len(names)}, output={dest}, '
          f'gripper threshold={args.threshold}, '
          f'verify old WM={not args.no_verify_legacy}',flush=True)
    for index,name in enumerate(names,1):
        src=orig/name
        legacy_path=leg/name
        if not args.no_verify_legacy and not legacy_path.is_file():
            raise FileNotFoundError(f'Missing old WM preprocessed comparison: {legacy_path}')
        payload,row=build_episode(src,loader,cfg,tensorizer,prompter,windower,
                                  args.threshold,legacy_path,
                                  compare_legacy=not args.no_verify_legacy)
        if args.mode=='execute':
            atomic_pickle(payload,dest/name,row['frames'],row['window_prompts'])
        all_rows.append(row)
        print(f'[{index:3d}/{len(names)}] {name}: {row["frames"]} frames, '
              f'{row["window_prompts"]} windows, gripper closed frames={row["closed_frames"]} '
              f'| max XYZ={row["xyz_max_mm"]}mm | old WM match={bool(row["verified_equal_to_old_wm"])}',
              flush=True)
        del payload
        gc.collect()
    if args.mode=='execute':
        dump_csv(dest/'manifest.csv',all_rows)
        dump_csv(dest/'excluded.csv',exclusions)
        (dest/'summary.json').write_text(json.dumps({
            'source':str(orig),'legacy_wm':str(leg), 'output':str(dest),
            'selected_episodes':len(names), 'excluded_episodes':len(exclusions),
            'total_frames':sum(x['frames'] for x in all_rows),
            'observed_transitions':sum(x['transitions'] for x in all_rows),
            'synthetic_terminal_actions':len(names),
            'gripper_threshold':args.threshold,
            'legacy_verified':not args.no_verify_legacy,
            'originals_modified':False,
        },indent=2))
    print(f'DONE: {len(names)} episodes; {sum(x["frames"] for x in all_rows)} frames; '
          f'{sum(x["transitions"] for x in all_rows)} observed transitions; '
          f'{len(names)} synthetic terminal OPEN labels. Writes={args.mode=="execute"}',flush=True)


if __name__=='__main__':
    main()
