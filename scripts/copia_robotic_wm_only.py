#!/usr/bin/env python3
"""Copia, senza modificarli, i 33 Exploration e 48 Plays con EEF congelato.

Sorgenti: precedenti PKL *gia' preprocessati* per video/latent world model.
Esclusi: tutti i record che hanno solo un salto XYZ >100 mm (7 E, 4 P).

Default: controllo non distruttivo (dry-run).
--execute: copia atomica dei file selezionati, con SHA256 e manifest CSV/JSON.
--verify-only: controlla i file gia' copiati e i relativi SHA256.
--validate-pkl: opzionale; apre i trusted PKL con pickle per verificare
                lunghezza e schema dei prompt (richiede ambiente originale).

Non inserisce o riscrive action; il mask per la loss action va implementato
nel dataloader di training (questi dati devono avere loss action disabilitata).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/mnt/beegfs/a.cuciniello9/datasets')
EXPLORATION = BASE / 'robotic/exploration_preprocessed'
PLAYS = BASE / 'robotic/plays_preprocessed'
DESTINATION = BASE / 'mixture/robotic_wm_only'

# Lista ESATTA dell'utente: solo reason=static_eef. ID e frame dall'audit.
EXPLORATION_STATIC = '''
E00014 100
E00015 56
E00016 88
E00017 64
E00018 56
E00019 72
E00020 57
E00021 56
E00022 68
E00023 60
E00024 46
E00025 80
E00071 59
E00113 73
E00126 54
E00127 90
E00175 64
E00176 62
E00177 65
E00178 75
E00193 38
E00194 46
E00217 61
E00218 48
E00263 87
E00264 42
E00295 79
E00333 64
E00334 42
E00350 93
E00410 64
E00411 41
E00426 72
'''
PLAYS_STATIC = '''
P00005 35
P00027 36
P00028 50
P00029 43
P00030 43
P00041 35
P00042 38
P00043 49
P00044 34
P00045 34
P00046 41
P00047 35
P00048 54
P00049 38
P00050 50
P00085 43
P00086 46
P00087 31
P00088 38
P00089 38
P00090 44
P00091 31
P00092 30
P00093 30
P00094 45
P00095 44
P00096 35
P00097 41
P00106 36
P00107 51
P00108 40
P00109 49
P00110 36
P00127 48
P00132 34
P00133 44
P00134 49
P00135 37
P00136 41
P00161 37
P00162 35
P00163 32
P00164 35
P00165 38
P00166 34
P00180 34
P00181 34
P00182 34
'''

# Blocca esplicitamente i nomi esclusi per SALTI (non devono entrare).
EXCLUDED_JUMPS = {
    'E00044.pkl', 'E00069.pkl', 'E00161.pkl', 'E00240.pkl',
    'E00241.pkl', 'E00309.pkl', 'E00397.pkl',
    'P00040.pkl', 'P00077.pkl', 'P00126.pkl', 'P00145.pkl',
}


def selected_episodes(exploration_root: Path, plays_root: Path):
    rows = []
    for letter, text, root, dataset, prompt_schema in (
        ('E', EXPLORATION_STATIC, exploration_root, 'exploration', 'window_prompts'),
        ('P', PLAYS_STATIC, plays_root, 'plays', 'episode_prompt+window_starts'),
    ):
        for line in text.strip().splitlines():
            episode, count = line.split()
            name = episode + '.pkl'
            if not re.fullmatch(letter + r'\d{5,}', episode):
                raise AssertionError('ID episodio inatteso: ' + episode)
            rows.append({
                'file': name, 'episode_id': episode, 'source_dataset': dataset,
                'source': root / name, 'expected_frames': int(count),
                'prompt_schema': prompt_schema, 'training_role': 'world_model_only',
                'action_loss_enabled': False,
            })
    names = [r['file'] for r in rows]
    if len(names) != len(set(names)) or len(rows) != 81:
        raise AssertionError('Selezione errata o duplicati: ' + str(len(rows)))
    if len([r for r in rows if r['source_dataset'] == 'exploration']) != 33:
        raise AssertionError('Devono esserci 33 E statici')
    if len([r for r in rows if r['source_dataset'] == 'plays']) != 48:
        raise AssertionError('Devono esserci 48 P statici')
    if set(names) & EXCLUDED_JUMPS:
        raise AssertionError('La selezione include episodi con SALTI XYZ')
    if sum(r['expected_frames'] for r in rows) != 4011:
        raise AssertionError('Il totale dei frame deve essere 4011')
    return rows


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_pkl(path: Path, row: dict):
    """Solo per PKL fidati prodotti dai propri script di preprocessing."""
    with path.open('rb') as f:
        p = pickle.load(f)
    if not isinstance(p, dict) or 'traj' not in p:
        raise ValueError(f'{path}: PKL non dizionario con traj')
    count = len(p['traj'])
    if count != row['expected_frames']:
        raise ValueError(f'{path}: attesi {row["expected_frames"]} frame, trovati {count}')
    if row['source_dataset'] == 'exploration':
        w = p.get('window_prompts')
        if not isinstance(w, list) or not w:
            raise ValueError(f'{path}: window_prompts assenti/vuoti')
    else:
        if not isinstance(p.get('episode_prompt'), str) or not p['episode_prompt'].strip():
            raise ValueError(f'{path}: episode_prompt assente/vuoto')
        w = p.get('window_starts')
        if not isinstance(w, list) or not w:
            raise ValueError(f'{path}: window_starts assente/vuoto')
    return len(w)


def atomic_write_bytes(path: Path, contents: bytes):
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix='.' + path.name + '.',
                                     suffix='.tmp', delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def copy_atomically(src: Path, dest: Path, expected_sha: str):
    with tempfile.NamedTemporaryFile('wb', dir=dest.parent,
                                     prefix='.' + dest.name + '.', suffix='.tmp',
                                     delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        shutil.copy2(src, temp)
        if sha256(temp) != expected_sha:
            raise IOError(f'{src}: SHA256 differente dopo la copia')
        if dest.exists():
            # Altra esecuzione concomitante: non sovrascrivere.
            if sha256(dest) != expected_sha:
                raise FileExistsError(f'{dest}: esiste ma differisce dalla sorgente')
            return 'already_present'
        os.replace(temp, dest)
        if sha256(dest) != expected_sha:
            raise IOError(f'{dest}: SHA256 differente dopo il rename')
        return 'copied'
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true', help='Copia gli 81 PKL')
    group.add_argument('--verify-only', action='store_true', help='Controlla gli 81 PKL copiati')
    parser.add_argument('--exploration-root', type=Path, default=EXPLORATION)
    parser.add_argument('--plays-root', type=Path, default=PLAYS)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--validate-pkl', action='store_true',
                        help='Controlla schema e frame via pickle (solo file fidati)')
    args = parser.parse_args()

    rows = selected_episodes(args.exploration_root, args.plays_root)
    dest = args.destination
    if dest.resolve() in {args.exploration_root.resolve(), args.plays_root.resolve()}:
        parser.error('La destinazione non puo coincidere con una sorgente')

    print(f'Modo: {"VERIFY" if args.verify_only else ("EXECUTE" if args.execute else "DRY-RUN")}')
    print(f'Sorgente E: {args.exploration_root}')
    print(f'Sorgente P: {args.plays_root}')
    print(f'Destinazione: {dest}')
    print('Selezionati 33 E + 48 P = 81 PKL; 4011 frame. Salti XYZ esclusi: 11.')

    # Preflight completo: se manca un file, non copiamo niente.
    missing = [str(r['source']) for r in rows if not r['source'].is_file()]
    if missing:
        raise FileNotFoundError(f'Mancano {len(missing)} PKL sorgenti:\n' + '\n'.join(missing))
    existing_dest = [p.name for p in dest.glob('*.pkl')] if dest.is_dir() else []
    unexpected = sorted(set(existing_dest) - {r['file'] for r in rows})
    if unexpected:
        raise ValueError('PKL inattesi nella destinazione (non sovrascritti): ' + ', '.join(unexpected))
    jump_dest = sorted(set(existing_dest) & EXCLUDED_JUMPS)
    if jump_dest:
        raise ValueError('Presenti PKL con SALTI nella destinazione: ' + ', '.join(jump_dest))
    for r in rows:
        r['sha256'] = sha256(r['source'])
        r['bytes'] = r['source'].stat().st_size
        if args.validate_pkl:
            r['num_windows'] = validate_pkl(r['source'], r)
        d = dest / r['file']
        if d.exists() and sha256(d) != r['sha256']:
            raise FileExistsError(f'{d}: contenuto diverso, non lo sovrascrivo')
        if args.verify_only and not d.is_file():
            raise FileNotFoundError(f'{d}: copia mancante')
    print('Preflight OK: tutti i PKL sorgenti esistono, hash e selezione validi.')

    if args.execute:
        dest.mkdir(parents=True, exist_ok=True)
    count = Counter()
    for index, r in enumerate(rows, 1):
        dst = dest / r['file']
        if args.execute:
            outcome = copy_atomically(r['source'], dst, r['sha256'])
        elif args.verify_only:
            outcome = 'verified'
        else:
            outcome = 'already_present' if dst.is_file() else 'to_copy'
        r['copy_status'] = outcome
        count[outcome] += 1
        print(f'[{index:02d}/81] {r["file"]} {r["source_dataset"]:11s} '
              f'{r["expected_frames"]:3d} frames | {outcome}', flush=True)

    if args.execute:
        fields = ['file', 'episode_id', 'source_dataset', 'source_path', 'relative_path',
                  'expected_frames', 'prompt_schema', 'training_role', 'action_loss_enabled',
                  'bytes', 'sha256', 'copy_status']
        manifest_rows = []
        for r in rows:
            manifest_rows.append({
                'file': r['file'], 'episode_id': r['episode_id'],
                'source_dataset': r['source_dataset'], 'source_path': str(r['source']),
                'relative_path': r['file'], 'expected_frames': r['expected_frames'],
                'prompt_schema': r['prompt_schema'],
                'training_role': r['training_role'],
                'action_loss_enabled': r['action_loss_enabled'],
                'bytes': r['bytes'], 'sha256': r['sha256'],
                'copy_status': r['copy_status'],
            })
        import io
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
        atomic_write_bytes(dest / 'manifest.csv', buffer.getvalue().encode('utf-8'))
        summary = {
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'description': '81 original WM-only RGB preprocessed PKLs with frozen EEF poses',
            'destination': str(dest),
            'exploration_count': 33, 'exploration_frames': 2122,
            'plays_count': 48, 'plays_frames': 1889,
            'total_count': 81, 'total_frames': 4011,
            'excluded_jump_count': 11,
            'training_role': 'world_model_only',
            'action_loss_enabled': False,
            'loss_mask_note': 'Il mask NON e applicato ai PKL: va gestito nel dataloader.',
            'file_copy_integrity': 'SHA256 identico alle sorgenti per tutti i file',
            'counts': dict(count),
        }
        atomic_write_bytes(dest / 'summary.json',
                           (json.dumps(summary, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
        print('Manifest scritto:', dest / 'manifest.csv')
        print('Summary scritto:', dest / 'summary.json')
    print('Completato:', dict(count))


if __name__ == '__main__':
    main()
