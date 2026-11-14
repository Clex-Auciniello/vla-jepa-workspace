#!/usr/bin/env python3
"""Preprocessing definitivo Plays per mixture VLA-JEPA: world-model E2 + action 7D.

INPUT:  /mnt/beegfs/a.cuciniello9/datasets/robotic/plays_trajectories/P*.pkl
LEGACY: /mnt/beegfs/a.cuciniello9/datasets/robotic/plays_preprocessed/P*.pkl
AUDIT:  /mnt/beegfs/a.cuciniello9/datasets/robotic/plays_original_audit/episodes.json
OUTPUT (nuovo, sicuro): /mnt/beegfs/a.cuciniello9/datasets/mixture/plays_preprocessed_actions/

Selezione dall'audit precedente: 190 - 48 static EEF - 4 jump >100mm = 138;
5027 frame, 4889 delta reali, 138 action terminali sintetiche, 866 finestre.

- Copia la ricetta E2 di preprocess_image ESATTAMENTE, usando _pkl_bgr_to_tensor
  da vla_jepa_online.training_server.online_batch_builder; RGB 224x224.
- Mantiene un episode_prompt unico, window_starts di 8 frame / stride 5,
  ultima finestra allineata, zero padding e frame finali osservati invariati.
- Confronta immagini PIXEL per PIXEL e metadati video/prompt contro legacy E2.
- Aggiunge EEF_state, gripper_state, joint_positions, joint_velocities,
  action_world e action 7D [dxyz, deuler_xyz, binary_next/0.05].
- Action: Rdelta = R[t+1] * R[t].inv(), Euler xyz, scale 0.05.
  Gripper closed quando qpos[t+1]>=0.20 (action 20), altrimenti 0.
- Ultima action 7D=0: etichetta sintetica; preserva il qpos misurato.
  Per i push type=04 il gripper puo' restare fisicamente chiuso, senza
  inventare una apertura nell'osservazione originale.
- Preserva i metadati del source e l'info di ogni step (motion_label ecc).
- NON altera input ne' legacy; nessun overwrite di file esistenti.

Esempi:
  python preprocess_plays_actions.py --mode dry-run --limit 2
  python preprocess_plays_actions.py --mode execute --limit 2 \
      --destination /mnt/beegfs/a.cuciniello9/datasets/mixture/plays_pilot
  python preprocess_plays_actions.py --mode dry-run
  python preprocess_plays_actions.py --mode execute

Richiede PYTHONPATH=/mnt/beegfs/a.cuciniello9/vla-jepa/vla-jepa-workspace/src:$PYTHONPATH
Leggere esclusivamente pickle fidati.
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

DATASETS = Path('/mnt/beegfs/a.cuciniello9/datasets')
DEFAULT_SOURCE = DATASETS / 'robotic/plays_trajectories'
DEFAULT_LEGACY = DATASETS / 'robotic/plays_preprocessed'
DEFAULT_AUDIT = DATASETS / 'robotic/plays_original_audit/episodes.json'
# Nuova directory: evitiamo di sovrascrivere una possibile vecchia
# mixture/plays_preprocessed gia' utilizzata in esperimenti precedenti.
DEFAULT_DEST = DATASETS / 'mixture/plays_preprocessed_actions'
EXPECTED = {'total': 190, 'static': 48, 'jump': 4, 'kept': 138,
            'frames_kept': 5027, 'windows_kept': 866}
SCALE = 0.05
THRESHOLD = 0.20
MAX_STEP_MM = 100.0


def load_backend():
    try:
        from vla_jepa_offline.dataset import _open_episode
        from vla_jepa_online.training_server.online_batch_builder import (
            OnlineBatchBuilderConfig, _pkl_bgr_to_tensor,
        )
    except Exception as exc:
        raise RuntimeError(
            'Import backend E2 fallito. Esporta PYTHONPATH del workspace VLA-JEPA. '
            f'Errore: {type(exc).__name__}: {exc}'
        ) from exc
    return _open_episode, OnlineBatchBuilderConfig, _pkl_bgr_to_tensor


def step_at(traj, t):
    try:
        return traj.get(t, decompress=False)
    except TypeError:
        return traj.get(t)


def vector(x, n, name, path, t):
    value = np.asarray(x, dtype=np.float64).reshape(-1)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError(f'{path.name} frame={t}: invalid {name} shape={value.shape}')
    return value


def native_steps(traj, path):
    data = getattr(traj, '_data', None)
    if not isinstance(data, list) or len(data) != len(traj) or len(data) < 8:
        raise ValueError(f'{path}: Trajectory._data invalid or fewer than 8 frames')
    for t, row in enumerate(data):
        if not isinstance(row, tuple) or len(row) != 5 or not isinstance(row[0], dict):
            raise TypeError(f'{path} frame={t}: expected tuple (obs,reward,done,info,action)')
    return data


def starts_aligned(T, window_size, stride):
    # Stesso codice del precedente preprocessing Plays E2.
    starts = list(range(0, T - window_size + 1, stride))
    if not starts:
        raise ValueError(f'Frame insufficienti: T={T}, window={window_size}')
    if starts[-1] + window_size < T:
        starts[-1] = T - window_size
    return starts


def safe_selection(source_root, audit_json):
    audit = json.loads(audit_json.read_text(encoding='utf-8'))
    if not isinstance(audit, list) or len(audit) != EXPECTED['total']:
        raise ValueError(f'{audit_json}: audit atteso su 190 episodi')
    found = {p.name for p in source_root.glob('P*.pkl') if re.fullmatch(r'P\d{5,}\.pkl', p.name)}
    rows = {}
    kept, exclusions = [], []
    reason_counts = Counter()
    for a in audit:
        name = a.get('file')
        if not isinstance(name, str) or not re.fullmatch(r'P\d{5,}\.pkl', name):
            raise ValueError(f'Audit file name non valido: {name!r}')
        if name in rows:
            raise ValueError(f'Duplicato audit: {name}')
        if a.get('status') != 'OK' or a.get('frames', 0) < 8:
            raise ValueError(f'Audit incompleto/non OK: {name}')
        if any(a.get(k) != a['frames'] for k in ('pos_valid','quat_valid','grip_valid')):
            raise ValueError(f'Campi posa incompleti: {name}')
        # I valori booleani sono salvati dall'audit di Plays originale.
        is_static = a.get('eef_static') is True
        is_jump = a.get('xyz_jump_gt_100mm') is True
        if (is_static != (float(a['xyz_max_mm']) == 0.0 and float(a['rot_max_deg']) == 0.0)
                or is_jump != (float(a['xyz_max_mm']) > MAX_STEP_MM)):
            raise ValueError(f'{name}: flag di audit incoerenti con statistiche originali')
        if is_static and is_jump:
            raise ValueError(f'{name}: sovrapposizione dei due filtri')
        rows[name] = a
        if is_static or is_jump:
            reason = 'static_eef' if is_static else 'xyz_step_gt_100mm'
            reason_counts[reason] += 1
            exclusions.append({'file': name, 'play_type': a.get('play_type'),
                               'frames': a['frames'], 'reason': reason,
                               'xyz_max_mm': a['xyz_max_mm']})
        else:
            kept.append(name)
    if set(rows) != found:
        raise ValueError(f'Originali diversi da audit: missing={sorted(set(rows)-found)[:5]}, '
                         f'new={sorted(found-set(rows))[:5]}')
    if (len(kept) != EXPECTED['kept'] or
        reason_counts['static_eef'] != EXPECTED['static'] or
        reason_counts['xyz_step_gt_100mm'] != EXPECTED['jump']):
        raise ValueError(f'Selezione inattesa: kept={len(kept)}, removed={reason_counts}')
    nframes = sum(rows[n]['frames'] for n in kept)
    nwindows = sum(rows[n]['window_count'] for n in kept)
    if (nframes, nwindows) != (EXPECTED['frames_kept'], EXPECTED['windows_kept']):
        raise ValueError(f'Frame/finestre selezionati inattesi: {nframes}/{nwindows}')
    kept.sort(key=lambda n: int(n[1:-4]))
    exclusions.sort(key=lambda row: int(row['file'][1:-4]))
    return kept, exclusions, rows


def preprocess_image(bgr, crop, size, tensorizer):
    # Mantiene la ricetta E2 senza conversioni addizionali, interpolazioni
    # alternative o normalizzazioni diverse.
    tensor = tensorizer(bgr, crop, size)
    return tensor.mul(255).round_().byte().permute(1, 2, 0).contiguous().numpy()


def expected_preprocessing(cfg):
    crop = cfg.front_crop
    return {
        'front_crop_margins_top_bottom_left_right': [crop.top,crop.bottom,crop.left,crop.right] if crop else None,
        'gripper_crop': None,
        'image_size': list(cfg.image_size),
        'color_order': 'RGB',
        'resize_interpolation': 'cv2.INTER_LINEAR',
        'window_size': cfg.window_size,
        'stride': cfg.stride,
        'align_last_window': True,
    }


def validate_config(cfg):
    if cfg.window_size != 8 or cfg.stride != 5 or list(cfg.image_size) != [224,224]:
        raise ValueError(f'Config E2 non corrispondente a window=8 stride=5 RGB 224: {cfg}')
    if cfg.gripper_crop is not None:
        raise ValueError('Config E2 cambiata: gripper_crop doveva essere None')


def load_legacy(path, T, expected_prompt, starts, preprocessing):
    if not path.is_file():
        raise FileNotFoundError(f'Legacy E2 mancante: {path}')
    with path.open('rb') as handle:
        old = pickle.load(handle)
    if not isinstance(old, dict) or 'traj' not in old:
        raise ValueError(f'{path}: legacy malformed')
    if old.get('episode_prompt') != expected_prompt:
        raise ValueError(f'{path}: episode_prompt non corrisponde')
    if old.get('window_starts') != starts:
        raise ValueError(f'{path}: window_starts non corrispondono')
    if len(old['traj']) != T:
        raise ValueError(f'{path}: lunghezza episodio E2 non corrisponde')
    old_meta = old.get('preprocessing')
    if not isinstance(old_meta, dict):
        raise ValueError(f'{path}: preprocessing metadata legacy mancanti')
    for key, val in preprocessing.items():
        if old_meta.get(key) != val:
            raise ValueError(f'{path}: ricetta E2 diversa per {key}: {old_meta.get(key)!r} != {val!r}')
    return old


def episode_transform(src, loader, cfg, tensorizer, threshold, legacy_root, verify_legacy=True):
    data = loader(str(src))
    if not isinstance(data, dict) or data.get('source') != 'plays' or data.get('episode_id') != src.stem:
        raise ValueError(f'{src}: non e un PKL Plays originale con ID valido')
    if data.get('prompt_mode') != 'episode_prompt':
        raise ValueError(f'{src}: prompt_mode non valido')
    prompt = data.get('episode_prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f'{src}: episode_prompt vuoto')
    if data.get('operator_success') is not True or data.get('operator_approved') is not True:
        raise ValueError(f'{src}: episodio non approvato')
    traj = data['traj']
    native = native_steps(traj, src)
    T = len(native)
    starts = starts_aligned(T, cfg.window_size, cfg.stride)
    wm_metadata = expected_preprocessing(cfg)
    legacy = load_legacy(legacy_root/src.name, T, prompt, starts, wm_metadata) if verify_legacy else None

    positions, quats, grips = [], [], []
    for t, row in enumerate(native):
        obs, _, _, info, original_action = row
        if not isinstance(info, dict) or info.get('index') != t+1:
            raise ValueError(f'{src} t={t}: info.index non valido')
        positions.append(vector(obs.get('eef_pos'),3,'eef_pos',src,t))
        quats.append(vector(obs.get('eef_quat'),4,'eef_quat',src,t))
        grips.append(vector(obs.get('gripper_qpos'),1,'gripper_qpos',src,t)[0])
        vector(obs.get('joint_pos'),6,'joint_pos',src,t)
        vector(obs.get('joint_vel'),6,'joint_vel',src,t)
        vector(original_action,8,'source commanded action8',src,t)
        for name in ('front_camera_image','gripper_camera_image'):
            im = np.asarray(obs.get(name))
            if im.dtype != np.uint8 or im.ndim != 3 or im.shape != (376,672,3):
                raise ValueError(f'{src} t={t}: malformed BGR {name}: {im.shape}/{im.dtype}')
    if native[-1][3].get('event') != 'final_observation':
        raise ValueError(f'{src}: ultima osservazione non e final_observation')
    pos = np.stack(positions)
    quat = np.stack(quats)
    grip = np.asarray(grips, dtype=np.float64)
    if np.any(np.abs(np.linalg.norm(quat,axis=1)-1) > 1e-3):
        raise ValueError(f'{src}: quaternion non unitari')
    rotations = Rotation.from_quat(quat)  # ordine xyzw, come scripts mixture precedenti
    rel = rotations[1:] * rotations[:-1].inv()
    diff_pos = np.diff(pos,axis=0)
    xyz_steps_mm = np.linalg.norm(diff_pos,axis=1)*1000.0
    rot_steps_deg = np.rad2deg(rel.magnitude())
    if np.all(xyz_steps_mm == 0) and np.all(rot_steps_deg == 0):
        raise ValueError(f'{src}: EEF statico, non dovrebbe essere incluso')
    if np.max(xyz_steps_mm) > MAX_STEP_MM + 1e-7:
        raise ValueError(f'{src}: jump > 100mm, non dovrebbe essere incluso')
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Gimbal lock detected')
        euler_abs = rotations.as_euler('xyz')
        euler_rel = rel.as_euler('xyz')

    closed = (grip >= threshold).astype(np.uint8)
    actions = np.zeros((T,7),dtype=np.float32)
    actions[:-1,:3] = diff_pos/SCALE
    actions[:-1,3:6] = euler_rel/SCALE
    actions[:-1,6] = closed[1:].astype(np.float32)/SCALE
    actions[-1] = 0.0  # terminale aperto *sintetico*; NON cambio stato reale

    reconstructed_pos = pos[:-1] + actions[:-1,:3].astype(np.float64)*SCALE
    reconstructed_rot = Rotation.from_euler('xyz', actions[:-1,3:6].astype(np.float64)*SCALE) * rotations[:-1]
    xyz_error_mm = float(np.max(np.linalg.norm(reconstructed_pos-pos[1:], axis=1))*1000)
    rot_error_deg = float(np.max(np.rad2deg((reconstructed_rot.inv()*rotations[1:]).magnitude())))
    if xyz_error_mm > 0.01 or rot_error_deg > 0.001:
        raise ValueError(f'{src}: round-trip action err={xyz_error_mm:.6f}mm,{rot_error_deg:.8f}deg')

    config_str = getattr(traj,'config_str',None)
    if config_str is None:
        config_str = data.get('config_str')
    if config_str is None:
        raise ValueError(f'{src}: config_str non disponibile')
    converted = traj.__class__(config_str=config_str)
    for t, (raw_obs,reward,done,info,original_action) in enumerate(native):
        front = preprocess_image(raw_obs['front_camera_image'], cfg.front_crop, cfg.image_size, tensorizer)
        wrist = preprocess_image(raw_obs['gripper_camera_image'], cfg.gripper_crop, cfg.image_size, tensorizer)
        if front.shape != (224,224,3) or front.dtype != np.uint8 or wrist.shape != (224,224,3) or wrist.dtype != np.uint8:
            raise ValueError(f'{src} t={t}: formato immagine finale non valido')
        if legacy is not None:
            legacy_obs = step_at(legacy['traj'],t)['obs']
            for key, image in [('camera_front_image',front),('camera_gripper_image',wrist)]:
                if not np.array_equal(legacy_obs.get(key), image):
                    raise ValueError(f'{src} t={t}: {key} diverso dal precedente E2 (pixel mismatch)')
        # Stesso schema canonico delle altre sorgenti. I sensori originali
        # non immagine vengono preservati: non alteriamo la qpos misurata.
        obs = {key:value for key,value in raw_obs.items()
               if 'image' not in key.lower() and 'depth' not in key.lower()}
        obs['camera_front_image'] = front
        obs['camera_gripper_image'] = wrist
        eef_state = np.r_[pos[t],euler_abs[t]].astype(np.float32)
        obs['EEF_state'] = eef_state
        obs['gripper_state'] = np.array([grip[t],float(closed[t])],dtype=np.float32)
        obs['joint_positions'] = vector(raw_obs['joint_pos'],6,'joint_pos',src,t).astype(np.float32)
        obs['joint_velocities'] = vector(raw_obs['joint_vel'],6,'joint_vel',src,t).astype(np.float32)
        obs['action_world'] = np.r_[eef_state,float(closed[t])].astype(np.float32)
        converted._data.append((obs, reward, t == T-1, dict(info), actions[t].copy()))
        converted._raw_state.append(None)
    if len(converted) != T:
        raise AssertionError(f'{src}: frame count converted non coerente')
    # Non rimuove ne' ripete frame e preserva i metadati dei task Plays.
    payload = dict(data)
    payload['traj'] = converted
    payload['source_file'] = str(src.resolve())
    payload['episode_prompt'] = prompt
    payload['window_starts'] = starts
    payload.pop('window_prompts',None)
    payload['preprocessing'] = wm_metadata
    payload['action_preprocessing'] = {
        'version':'plays_actions_v1',
        'action_shape':7,
        'action_dtype':'float32',
        'action_scale_factor':SCALE,
        'action_alignment':'t -> t+1 observed state, for t<T-1',
        'position':'(eef_pos[t+1] - eef_pos[t])/0.05 in base_link',
        'rotation':'Euler_xyz(R[t+1] * R[t].inv()) / 0.05',
        'quaternion_order':'xyzw',
        'gripper_qpos_threshold':float(threshold),
        'gripper_action_values':{'open':0.0,'closed':20.0},
        'gripper_state_format':'[raw_qpos, binary_closed]',
        'last_action':'synthetic 7D zeros; stationary pose+OPEN command',
        'last_observed_gripper_state_is_unmodified':True,
        'original_action':'8D commanded target preserved per-step in info.commanded_action',
        'source_filter':'48 static EEF and 4 XYZ jumps >100 mm excluded',
        'original_wm_images_episode_prompt_and_windows_verified':bool(legacy is not None),
    }
    last_closed = bool(closed[-1]); is_push = str(data.get('play_type')) == '04'
    if (last_closed and not is_push) or (is_push and not last_closed):
        raise ValueError(f'{src}: terminal gripper unexpected for play_type={data.get("play_type")}')
    record = {
        'file':src.name, 'play_type':str(data.get('play_type')),
        'frames':T, 'transitions':T-1, 'windows':len(starts),
        'terminal_closed':int(last_closed), 'push_type_04':int(is_push),
        'gripper_closed_frames':int(closed.sum()),
        'xyz_max_mm':round(float(xyz_steps_mm.max()),6),
        'rot_max_deg':round(float(rot_steps_deg.max()),6),
        'action_roundtrip_max_xyz_mm':round(xyz_error_mm,9),
        'action_roundtrip_max_rot_deg':round(rot_error_deg,9),
        'legacy_pixel_and_prompt_match':int(legacy is not None),
    }
    return payload, record


def verify_saved(path, frames, windows, expected_prompt=None, expected_starts=None, expected_threshold=None):
    with path.open('rb') as f:
        payload = pickle.load(f)
    if len(payload['traj']) != frames or len(payload['window_starts']) != windows:
        raise ValueError(f'{path}: roundtrip pickle frames/windows mismatch')
    if expected_prompt is not None and payload.get('episode_prompt') != expected_prompt:
        raise ValueError(f'{path}: il prompt esistente non coincide con quello della sorgente')
    if expected_starts is not None and payload.get('window_starts') != expected_starts:
        raise ValueError(f'{path}: indici delle finestre differenti')
    if expected_threshold is not None:
        meta = payload.get('action_preprocessing', {})
        if not np.isclose(meta.get('gripper_qpos_threshold',-1),expected_threshold,atol=1e-10):
            raise ValueError(f'{path}: soglia del gripper salvata diversa')
    if payload.get('window_prompts') is not None:
        raise ValueError(f'{path}: a Plays non devono essere associati window_prompts')
    for t in range(frames):
        step = step_at(payload['traj'],t)
        a = np.asarray(step['action'])
        if a.shape != (7,) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'{path}: action7 invalid t={t}')
        if t == frames-1:
            if not np.array_equal(a,np.zeros(7,dtype=np.float32)):
                raise ValueError(f'{path}: last action must be zeros')
        else:
            next_obs = step_at(payload['traj'],t+1)['obs']
            expected = float(next_obs['gripper_state'][1]) / SCALE
            if not np.isclose(float(a[6]), expected,atol=1e-5):
                raise ValueError(f'{path}: gripper not aligned with t+1, t={t}')
        obs = step['obs']
        keys = {key for key in obs if 'image' in key.lower() or 'depth' in key.lower()}
        if keys != {'camera_front_image','camera_gripper_image'}:
            raise ValueError(f'{path}: unexpected images {keys}')
        for key in ('camera_front_image','camera_gripper_image'):
            im = np.asarray(obs[key])
            if im.shape != (224,224,3) or im.dtype != np.uint8:
                raise ValueError(f'{path}: invalid RGB {key} at t={t}')
        for key,n in [('EEF_state',6),('gripper_state',2),('action_world',7),
                      ('joint_positions',6),('joint_velocities',6)]:
            v=np.asarray(obs[key])
            if v.shape != (n,) or v.dtype != np.float32 or not np.isfinite(v).all():
                raise ValueError(f'{path}: obs.{key} malformed at t={t}')


def atomic_save(payload, dest, frames, windows):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Already exists: {dest}')
    temp = None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=dest.parent,
                                         prefix='.'+dest.stem+'_',suffix='.tmp',delete=False) as f:
            temp=Path(f.name)
            pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
            f.flush();os.fsync(f.fileno())
        verify_saved(temp,frames,windows,payload['episode_prompt'],payload['window_starts'],
                     payload['action_preprocessing']['gripper_qpos_threshold'])
        # Destinazione protetta contro race: link non sovrascrive il file finale.
        os.link(temp,dest)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def dump_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('dry-run','execute'))
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--legacy',type=Path,default=DEFAULT_LEGACY)
    p.add_argument('--audit-json',type=Path,default=DEFAULT_AUDIT)
    p.add_argument('--destination',type=Path,default=DEFAULT_DEST)
    p.add_argument('--threshold',type=float,default=THRESHOLD)
    p.add_argument('--limit',type=int,default=0,help='0=tutti i 138')
    p.add_argument('--no-verify-legacy',action='store_true',help='Disabilita confronto pixel/prompt E2 (sconsigliato)')
    p.add_argument('--resume',action='store_true',help='Riprendi esecuzione interrotta, verificando i PKL gia salvati')
    args=p.parse_args()
    if args.limit<0 or not (0.1 < args.threshold < 0.8):
        p.error('limit o threshold non validi')
    source=args.source.expanduser().resolve()
    legacy=args.legacy.expanduser().resolve()
    audit=args.audit_json.expanduser().resolve()
    dest=args.destination.expanduser().resolve()
    if not source.is_dir() or not audit.is_file():
        p.error(f'Source o audit assenti: {source} ; {audit}')
    if not args.no_verify_legacy and not legacy.is_dir():
        p.error(f'Legacy directory assente: {legacy}')
    if (dest==source or source in dest.parents or dest in source.parents
            or dest==legacy or legacy in dest.parents or dest in legacy.parents):
        p.error('Output deve essere una directory distinta e non annidata in source/legacy')
    if args.mode=='execute':
        if dest.exists() and any(dest.iterdir()) and not args.resume:
            p.error(f'Directory output non vuota: {dest}; usa directory nuova o --resume')
        if args.limit and dest==DEFAULT_DEST.resolve():
            p.error('Per pilot specifica una destinazione diversa da quella definitiva')
    names, excluded, audit_rows=safe_selection(source,audit)
    if args.limit:
        names=names[:args.limit]
    loader, cfg_cls, tensorizer=load_backend()
    cfg=cfg_cls();validate_config(cfg)
    print(f'Piano: originali=190, esclusi static=48, jump=4, selezionati={len(names)} '
          f'| mode={args.mode}, threshold={args.threshold:.2f}, '
          f'legacy_pixels={not args.no_verify_legacy}, output={dest}',flush=True)
    manifest=[]
    for idx,name in enumerate(names,1):
        src=source/name
        payload,row=episode_transform(src,loader,cfg,tensorizer,args.threshold,legacy,
                                     verify_legacy=not args.no_verify_legacy)
        if row['frames']!=audit_rows[name]['frames'] or row['windows']!=audit_rows[name]['window_count']:
            raise ValueError(f'{name}: raw episode differs from audit counts')
        if args.mode=='execute':
            output=dest/name
            if output.exists():
                if not args.resume:
                    raise FileExistsError(output)
                verify_saved(output,row['frames'],row['windows'],payload['episode_prompt'],
                             payload['window_starts'],args.threshold)
            else:
                atomic_save(payload,output,row['frames'],row['windows'])
        manifest.append(row)
        print(f'[{idx:3d}/{len(names)}] {name}: task={row["play_type"]} '
              f'frame={row["frames"]}, windows={row["windows"]}, '
              f'EEF max delta={row["xyz_max_mm"]}mm '
              f'gripper final closed={row["terminal_closed"]} '
              f'legacy match={bool(row["legacy_pixel_and_prompt_match"])}',flush=True)
        del payload
        gc.collect()
    if args.mode=='execute':
        dump_csv(dest/'manifest.csv',manifest)
        dump_csv(dest/'excluded.csv',excluded)
        totals={
            'source':str(source), 'legacy_wm':str(legacy), 'output':str(dest),
            'total_original':190, 'excluded_static':48,'excluded_jump_gt_100mm':4,
            'selected_episodes':len(manifest),'total_frames':sum(r['frames'] for r in manifest),
            'real_transitions':sum(r['transitions'] for r in manifest),
            'synthetic_terminal_actions':len(manifest),
            'total_windows':sum(r['windows'] for r in manifest),
            'terminal_closed_push_episodes':sum(r['terminal_closed'] for r in manifest),
            'gripper_threshold':float(args.threshold),
            'legacy_images_and_prompt_verified':not args.no_verify_legacy,
            'originals_modified':False,
        }
        (dest/'summary.json').write_text(json.dumps(totals,indent=2),encoding='utf-8')
    print(f'DONE: episodes={len(manifest)}, frame={sum(r["frames"] for r in manifest)}, '
          f'transizioni={sum(r["transitions"] for r in manifest)}, '
          f'finestre={sum(r["windows"] for r in manifest)}, writes={args.mode=="execute"}',flush=True)


if __name__=='__main__':
    main()
