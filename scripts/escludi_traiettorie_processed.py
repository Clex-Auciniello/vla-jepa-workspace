#!/usr/bin/env python3
"""Exclude 48 visually rejected COD/OSVI/VLA-JEPA trajectories, safely.

Default: read-only dry-run. With --execute, MOVE matched *.pkl and *.json
into a quarantine directory and atomically rewrite the three manifest.csv
files. No PKL is loaded/unpickled. Files remaining keep original names/IDs.

Requires the original 444-row valutazioni_ultimi_frame.csv (48 'exclude').
The 396 'unreviewed' entries are retained, not silently treated as 'ok'.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT_NAMES = {
    'COD': 'cod_preprocessed',
    'OSVI-AWDA': 'osvi_preprocessed',
    'VLA-JEPA': 'vla_jepa_preprocessed',
}
EXPECTED_TOTAL = {'COD': 66, 'OSVI-AWDA': 191, 'VLA-JEPA': 187}
EXPECTED_EXCLUDED = {'COD': 19, 'OSVI-AWDA': 12, 'VLA-JEPA': 17}
DEFAULT_BASE = Path('/mnt/beegfs/a.cuciniello9/datasets/mixture')
DEFAULT_QUARANTINE = 'quarantena_48_ultimo_frame_20261114'


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict]):
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
            wr = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def safe_relative(text: str):
    rel = Path(text)
    if (rel.is_absolute() or len(rel.parts) != 2 or '..' in rel.parts
            or rel.parent.name not in {f'task_{i:02d}' for i in range(16)}
            or not rel.name.startswith('traj_') or rel.suffix.lower() != '.pkl'):
        raise ValueError(f'Invalid canonical relative path: {text!r}')
    return rel


def ensure_no_symlink(path: Path):
    if path.is_symlink():
        raise ValueError(f'Symlink not allowed: {path}')


def preflight(base: Path, audit_csv: Path, quarantine: Path):
    if not audit_csv.is_file():
        raise FileNotFoundError(f'CSV not found: {audit_csv}')
    ensure_no_symlink(audit_csv)
    if quarantine.exists():
        raise FileExistsError(f'Quarantine destination already exists: {quarantine}')
    if quarantine == base or quarantine in base.parents:
        raise ValueError('Quarantine cannot be the base or one of its parents')
    for folder in ROOT_NAMES.values():
        if quarantine == base / folder or base / folder in quarantine.parents:
            raise ValueError('Quarantine cannot be inside a preprocessed dataset')

    audit_fields, audit_rows = read_csv(audit_csv)
    for col in ('dataset', 'task', 'pkl', 'source', 'valutazione'):
        if col not in audit_fields:
            raise ValueError(f'CSV missing required column {col!r}')
    if len(audit_rows) != sum(EXPECTED_TOTAL.values()):
        raise ValueError(f'Expected 444 audit rows, got {len(audit_rows)}')

    audit_index = {}
    rejects = {}
    for row in audit_rows:
        dataset = row['dataset'].strip()
        if dataset not in ROOT_NAMES:
            raise ValueError(f'Unknown dataset {dataset!r}')
        rel = safe_relative(row['pkl'].strip())
        if rel.parent.name != row['task'].strip():
            raise ValueError(f'Task mismatch: {dataset}/{rel} vs {row["task"]!r}')
        status = row['valutazione'].strip().lower()
        if status not in ('exclude', 'ok', 'review', 'unreviewed'):
            raise ValueError(f'Invalid status {status!r} for {dataset}/{rel}')
        key = (dataset, rel.as_posix())
        if key in audit_index:
            raise ValueError(f'Duplicate audit entry {key}')
        audit_index[key] = row
        if status == 'exclude':
            rejects[key] = row

    if Counter(k[0] for k in audit_index) != Counter(EXPECTED_TOTAL):
        raise ValueError('Audit dataset counts differ from 66/191/187')
    if Counter(k[0] for k in rejects) != Counter(EXPECTED_EXCLUDED):
        raise ValueError(f'Unexpected excluded counts: {dict(Counter(k[0] for k in rejects))}')

    source_manifests = {}
    planned_moves = []
    removal_rows = []
    for dataset, folder in ROOT_NAMES.items():
        root = base / folder
        ensure_no_symlink(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        manifest = root / 'manifest.csv'
        ensure_no_symlink(manifest)
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        fields, rows = read_csv(manifest)
        if 'output_pkl' not in fields:
            raise ValueError(f'{manifest}: missing output_pkl')
        if len(rows) != EXPECTED_TOTAL[dataset]:
            raise ValueError(f'{manifest}: expected {EXPECTED_TOTAL[dataset]} rows; found {len(rows)}')
        observed = {}
        for row in rows:
            rel = safe_relative(row['output_pkl'])
            key = (dataset, rel.as_posix())
            if key in observed:
                raise ValueError(f'Duplicate manifest key {key}')
            observed[key] = row
            src = root / rel
            ensure_no_symlink(src)
            if not src.is_file():
                raise FileNotFoundError(f'Missing PKL from manifest: {src}')
            # Check provenance if it exists in manifest.
            provenance = (row.get('source_dataset') or row.get('source_group') or '').strip()
            audit_provenance = audit_index.get(key, {}).get('source', '').strip()
            if provenance and audit_provenance not in (provenance, '—'):
                raise ValueError(f'Provenance mismatch {key}: manifest={provenance!r}, audit={audit_provenance!r}')

        audit_keys = {key for key in audit_index if key[0] == dataset}
        if set(observed) != audit_keys:
            extra = sorted(set(observed) - audit_keys)
            absent = sorted(audit_keys - set(observed))
            raise ValueError(f'{dataset}: manifest/audit mismatch, extra={extra[:5]}, missing={absent[:5]}')
        real_pkls = {p.relative_to(root).as_posix() for p in root.glob('task_*/traj_*.pkl')}
        manifest_pkls = {key[1] for key in observed}
        if real_pkls != manifest_pkls:
            raise ValueError(f'{dataset}: manifest and actual PKL set mismatch')

        kept = []
        removed = []
        for row in rows:
            rel = safe_relative(row['output_pkl'])
            key = (dataset, rel.as_posix())
            if key not in rejects:
                kept.append(row)
                continue
            removed.append(row)
            src_pkl = root / rel
            src_json = src_pkl.with_suffix('.json')
            ensure_no_symlink(src_json)
            if not src_json.is_file():
                raise FileNotFoundError(f'Paired JSON missing: {src_json}')
            for src in (src_pkl, src_json):
                dst = quarantine / folder / src.relative_to(root)
                if dst.exists():
                    raise FileExistsError(f'Destination already exists: {dst}')
                planned_moves.append((src, dst))
            removal_rows.append(dict(rejects[key]))
        source_manifests[dataset] = {
            'root': root, 'path': manifest, 'fields': fields,
            'original': rows, 'kept': kept, 'removed': removed,
        }
    if len(removal_rows) != 48 or len(planned_moves) != 96:
        raise AssertionError('Expected 48 PKL+JSON pairs')
    return source_manifests, planned_moves, removal_rows, audit_fields


def verify_post(base: Path, manifests, rejects):
    for dataset, data in manifests.items():
        fields, rows = read_csv(data['path'])
        if len(rows) != len(data['kept']) or fields != data['fields']:
            raise RuntimeError(f'{dataset}: manifest post-write count/header mismatch')
        expected = {r['output_pkl'] for r in data['kept']}
        actual = {r['output_pkl'] for r in rows}
        real = {p.relative_to(data['root']).as_posix()
                for p in data['root'].glob('task_*/traj_*.pkl')}
        if actual != expected or real != expected:
            raise RuntimeError(f'{dataset}: PKLs and final manifest disagree')
    for row in rejects:
        root = base / ROOT_NAMES[row['dataset']]
        rel = safe_relative(row['pkl'])
        if (root / rel).exists() or (root / rel.with_suffix('.json')).exists():
            raise RuntimeError(f'Excluded files still present: {root / rel}')


def execute(base, audit_csv, quarantine, manifests, moves, rejects, audit_fields):
    quarantine.mkdir(parents=True, exist_ok=False)
    moved = []
    updated_manifests = []
    try:
        shutil.copy2(audit_csv, quarantine / 'valutazioni_ultimi_frame_original.csv')
        with (quarantine / 'excluded.csv').open('w', encoding='utf-8', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=audit_fields, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(rejects)
        save_json(quarantine / 'summary.json', {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_audit_csv': str(audit_csv),
            'base': str(base),
            'excluded': EXPECTED_EXCLUDED,
            'kept': {k: EXPECTED_TOTAL[k] - EXPECTED_EXCLUDED[k] for k in ROOT_NAMES},
            'action': 'moved PKL and paired JSON to quarantine; updated manifests in place',
            'remaining_status_note': 'Unreviewed entries have not been automatically marked OK',
        })
        for dataset, data in manifests.items():
            backup = quarantine / ROOT_NAMES[dataset] / 'manifest_before.csv'
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data['path'], backup)
            write_csv_atomic(backup.parent / 'manifest_removed.csv', data['fields'], data['removed'])
            write_csv_atomic(backup.parent / 'manifest_remaining.csv', data['fields'], data['kept'])

        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.is_file() or dst.exists():
                raise RuntimeError(f'Source missing or destination already exists: {src}, {dst}')
            os.replace(src, dst)
            moved.append((src, dst))
        for dataset, data in manifests.items():
            write_csv_atomic(data['path'], data['fields'], data['kept'])
            updated_manifests.append(dataset)
        verify_post(base, manifests, rejects)
    except BaseException as error:
        failures = []
        for dataset in reversed(updated_manifests):
            data = manifests[dataset]
            try:
                write_csv_atomic(data['path'], data['fields'], data['original'])
            except BaseException as exc:
                failures.append(f'Restore manifest {dataset}: {exc}')
        for src, dst in reversed(moved):
            try:
                if src.exists():
                    raise FileExistsError(f'Original path already exists: {src}')
                src.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dst, src)
            except BaseException as exc:
                failures.append(f'Restore file {src}: {exc}')
        if failures:
            print('ERROR: rollback incomplete; inspect quarantine and manifest_before.csv', file=sys.stderr)
            for failure in failures:
                print('  ' + failure, file=sys.stderr)
        else:
            print('ERROR: operation failed; original PKLs and manifests restored.', file=sys.stderr)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, default=DEFAULT_BASE)
    ap.add_argument('--csv', type=Path, required=True, help='444-row valutazioni_ultimi_frame.csv')
    ap.add_argument('--quarantine', type=Path, default=None)
    ap.add_argument('--execute', action='store_true', help='Perform the move after strict preflight; default is dry-run')
    args = ap.parse_args()
    base = args.base.expanduser().resolve()
    audit_csv = args.csv.expanduser().resolve()
    quarantine = ((args.quarantine or (base / DEFAULT_QUARANTINE))
                  .expanduser().resolve())
    manifests, moves, rejects, fields = preflight(base, audit_csv, quarantine)

    print('\n=== VISUAL EXCLUSIONS: VERIFIED ===')
    print('CSV:', audit_csv)
    print('Quarantine:', quarantine)
    for dataset, data in manifests.items():
        print(f'{dataset:10s}: {len(data["original"]):3d} before | '
              f'{len(data["removed"]):2d} exclude | {len(data["kept"]):3d} remain')
    print('\nEXACT FILES TO EXCLUDE:')
    for row in sorted(rejects, key=lambda r: (r['dataset'], r['pkl'])):
        print(f"  {row['dataset']:<10s} {row['pkl']} ")
    print('\nTotal: 48 PKL + 48 paired JSON; 3 manifests to update.')
    if not args.execute:
        print('\nDRY-RUN: no files changed. Re-run with --execute to apply.')
        return

    execute(base, audit_csv, quarantine, manifests, moves, rejects, fields)
    print('\nSUCCESS: 396 PKL remain in preprocessed directories.')
    print('Excluded files and manifests backed up in:', quarantine)
    print('Original clean and raw datasets untouched.')
    print('NOTE: The already-generated HTML gallery is a snapshot; regenerate if needed.')


if __name__ == '__main__':
    main()
