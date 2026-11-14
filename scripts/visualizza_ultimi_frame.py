#!/usr/bin/env python3
"""Galleria HTML dell'ULTIMA osservazione di ogni PKL preprocessato COD/OSVI/VLA-JEPA.

Input (read-only):
  /mnt/beegfs/a.cuciniello9/datasets/mixture/{cod,osvi,vla_jepa}_preprocessed
Output (new directory):
  .../datasets/mixture/debug_ultimi_frame/{index.html,images/,index.csv,errors.csv}

Ogni traiettoria contribuisce UNA SOLA osservazione (t = len(traj)-1).
Entrambe le camere di tale osservazione sono gia' RGB 224x224: le immagini
vengono esportate come PNG senza resize, crop o scambio di canali.

Solo PKL fidati: pickle.load puo' eseguire codice arbitrario.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path('/mnt/beegfs/a.cuciniello9/datasets/mixture')
SOURCES = {
    'COD': 'cod_preprocessed',
    'OSVI-AWDA': 'osvi_preprocessed',
    'VLA-JEPA': 'vla_jepa_preprocessed',
}
EXPECTED = {'COD': 66, 'OSVI-AWDA': 191, 'VLA-JEPA': 187}
IMAGES = {
    'front': 'camera_front_image',
    'gripper': 'camera_gripper_image',
}


def load_savers(savers_dir: Path | None):
    if savers_dir is not None:
        if not savers_dir.is_dir():
            raise FileNotFoundError(f'--savers-dir non esiste: {savers_dir}')
        sys.path.insert(0, str(savers_dir))
    # Necessario se le Trajectory furono salvate come savers.Trajectory.
    # Il corretto PYTHONPATH e' gia' disponibile nell'ambiente usato per i preprocess.
    try:
        import savers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Impossibile importare savers. Usa lo stesso env/PYTHONPATH dei preprocessing '
            'oppure aggiungi --savers-dir /path/to/dataset_collector_pkg/scripts'
        ) from exc


def manifest_jobs(root: Path):
    manifest = root / 'manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'Manca il manifest dei PKL preprocessati: {manifest}')
    with manifest.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Manifest vuoto: {manifest}')
    jobs = []
    visited = set()
    for row in rows:
        rel = Path(row['output_pkl'])
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts) != 2:
            raise ValueError(f'Path manifest non valido: {rel}')
        if not rel.parent.name.startswith('task_') or rel.suffix.lower() != '.pkl':
            raise ValueError(f'PKL non canonico: {rel}')
        if rel.as_posix() in visited:
            raise ValueError(f'Duplicato nel manifest: {rel}')
        visited.add(rel.as_posix())
        jobs.append((root / rel, rel, row))
    jobs.sort(key=lambda x: (int(x[1].parent.name.split('_')[1]), x[1].name))
    return jobs


def last_obs(path: Path):
    with path.open('rb') as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or 'traj' not in payload:
        raise ValueError('Manca payload["traj"]')
    traj = payload['traj']
    n = len(traj)
    if n < 1:
        raise ValueError('Traiettoria senza frame')
    raw_data = getattr(traj, '_data', None)
    if isinstance(raw_data, list) and len(raw_data) == n:
        last = raw_data[-1]
        if isinstance(last, tuple) and len(last) == 5 and isinstance(last[0], dict):
            obs = last[0]
        else:
            raise ValueError('Formato interno della Trajectory inatteso')
    else:
        try:
            obs = traj.get(n - 1, decompress=False)['obs']
        except TypeError:
            obs = traj.get(n - 1)['obs']
    if not isinstance(obs, dict):
        raise ValueError('Ultima osservazione non e\' un dict')
    return obs, n


def validate_image(obs: dict, key: str, path: Path):
    if key not in obs:
        raise KeyError(f'Chiave mancante: {key}')
    im = np.asarray(obs[key])
    if im.dtype != np.uint8 or im.shape != (224, 224, 3):
        raise ValueError(f'{key}: atteso RGB uint8 (224,224,3), trovato {im.shape} {im.dtype}')
    return np.ascontiguousarray(im)


def save_png(rgb: np.ndarray, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, 'RGB').save(dst, format='PNG', optimize=False)


def write_csv(path: Path, records: list[dict]):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="it"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Audit visivo — ultimi frame COD / OSVI / VLA-JEPA</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color-scheme:light dark}
*{box-sizing:border-box}body{margin:0;background:#f6f7f9;color:#162233}
header{position:sticky;top:0;z-index:8;background:#fff;border-bottom:1px solid #dee3ea;padding:15px max(18px,calc((100vw - 1550px)/2));box-shadow:0 2px 5px #0c223008}
h1{font-size:20px;margin:0 0 5px}.sub{font-size:12px;color:#5b6575;line-height:1.5}
.filters{display:flex;gap:8px;align-items:end;flex-wrap:wrap;margin-top:13px}
.filters label{display:flex;gap:3px;flex-direction:column;color:#536073;font-size:12px;font-weight:600}
select,input,button,textarea{font:inherit;border:1px solid #cfd7e2;border-radius:7px;padding:8px;background:white;color:#172237}
input[type=search]{min-width:190px}button{cursor:pointer}.primary{background:#1f4265;color:white;border-color:#1f4265}
.container{max-width:1550px;margin:0 auto;padding:18px}.stats{margin-bottom:12px;color:#526073;font-size:13px}
.notice{margin-bottom:13px;border-left:3px solid #c3893c;background:#fff9ed;padding:10px 12px;font-size:13px;color:#534831}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
.card{background:white;border:1px solid #dfe5eb;border-radius:10px;overflow:hidden;box-shadow:0 2px 5px #152b3709;min-width:0}
.cardhead{display:flex;justify-content:space-between;gap:8px;padding:11px 12px 7px;align-items:start}
.name{font-weight:750;font-size:14px}.meta{color:#566273;font-size:11px;overflow-wrap:anywhere}
.chip{font-size:10px;padding:4px 6px;background:#edf2f7;color:#3d506b;border-radius:5px;white-space:nowrap}
.photos{display:grid;grid-template-columns:1fr;gap:5px;padding:0 12px 9px}
.photos.both{grid-template-columns:1fr 1fr;gap:7px}
.picture{min-width:0}.picture figcaption{text-align:center;color:#536073;font-size:10px;padding-top:3px}
.picture{margin:0}.picture img{width:100%;display:block;aspect-ratio:1;object-fit:contain;background:#e7ebef;border-radius:5px;image-rendering:auto}
.detail{font-size:11px;color:#566273;margin:0 12px 10px;min-height:28px}
.audit{padding:8px 12px 12px;border-top:1px solid #eef1f4;display:grid;gap:7px}
.audit select{width:100%;padding:6px;font-size:12px}.audit textarea{width:100%;min-height:40px;resize:vertical;font-size:11px}
.noresults{text-align:center;padding:55px;color:#5d6775}
@media(max-width:650px){header{position:static;padding:13px}.container{padding:12px}.grid{grid-template-columns:repeat(auto-fill,minmax(210px,1fr))}}
@media(prefers-color-scheme:dark){body{background:#101821;color:#e8edf4}header,.card{background:#17212c;border-color:#35404c}select,input,button,textarea{background:#1e2a37;border-color:#4b5b69;color:#edf1f6}.primary{background:#32649b}.sub,.stats,.meta,.detail,.picture figcaption,.filters label{color:#b4bfcd}.chip{background:#283949;color:#cbd8e9}.notice{background:#29281f;color:#f1dbb1}.audit{border-color:#34424e}.picture img{background:#2a3540}}
</style></head><body>
<header>
<h1>Audit visivo — ultimo frame di ogni traiettoria</h1>
<div class="sub">COD · OSVI-AWDA · VLA-JEPA | immagini già preprocessate RGB 224×224 | ogni scheda mostra soltanto t = T−1</div>
<div class="filters">
<label>Dataset<select id="dataset"><option value="">Tutti</option></select></label>
<label>Task<select id="task"><option value="">Tutti</option></select></label>
<label>Provenienza<select id="source"><option value="">Tutte</option></select></label>
<label>Vista<select id="view"><option value="front">Solo front</option><option value="gripper">Solo gripper</option><option value="both">Entrambe</option></select></label>
<label>Valutazione<select id="status"><option value="">Tutte</option><option value="unreviewed">Da controllare</option><option value="ok">OK</option><option value="review">Da rivedere</option><option value="exclude">Da escludere</option></select></label>
<label>Ordine<select id="sort"><option value="task">Task → dataset</option><option value="dataset">Dataset → task</option></select></label>
<label>Cerca<input id="search" type="search" placeholder="traj, task, sorgente..."></label>
<button class="primary" id="export">Esporta valutazioni CSV</button>
</div></header>
<main class="container">
<div class="notice">L'ultimo frame è un'osservazione prima dell'ultima action: non è una prova del rilascio fisico del cubo. Per ingrandire, clicca sull'immagine. Le valutazioni manuali restano nel browser quando disponibile; esporta il CSV periodicamente.</div>
<div class="stats" id="stats"></div>
<div class="grid" id="cards"></div><div class="noresults" id="empty" hidden>Nessuna traiettoria corrisponde ai filtri.</div>
</main>
<script>
const records = __DATA_JSON__;
const fields = ['dataset','task','source','pkl','frames','last_t','gripper_qpos','gripper_binary','terminal_original','valutazione','note'];
const stateKey = 'last-frame-audit-v1-' + records.map(r=>r.dataset+':'+r.pkl).join('|').length + '-' + records.length;
let annotations = {};
try { annotations = JSON.parse(localStorage.getItem(stateKey) || '{}') || {}; } catch(e){}
const el = id => document.getElementById(id);
const esc = text => String(text ?? '').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const sid = r => r.dataset + '/' + r.pkl;
function save(){try{localStorage.setItem(stateKey,JSON.stringify(annotations));}catch(e){}}
function fill(id, values){const sel=el(id); [...new Set(values)].sort((a,b)=>String(a).localeCompare(String(b),'it',{numeric:true})).forEach(v=>{const opt=document.createElement('option');opt.value=v;opt.textContent=v;sel.append(opt);});}
fill('dataset', records.map(r=>r.dataset));fill('task',records.map(r=>r.task));fill('source',records.map(r=>r.source));
function getStatus(r){return annotations[sid(r)]?.status || 'unreviewed'}
function compare(a,b){const c=(x,y)=>String(x).localeCompare(String(y),'it',{numeric:true}); return el('sort').value==='task' ? c(a.task,b.task)||c(a.dataset,b.dataset)||c(a.pkl,b.pkl) : c(a.dataset,b.dataset)||c(a.task,b.task)||c(a.pkl,b.pkl)}
function render(){
const filtered=records.filter(r=>(!el('dataset').value||r.dataset===el('dataset').value)&&(!el('task').value||r.task===el('task').value)&&(!el('source').value||r.source===el('source').value)&&(!el('status').value||getStatus(r)===el('status').value)&&((r.dataset+' '+r.task+' '+r.source+' '+r.pkl).toLowerCase().includes(el('search').value.toLowerCase()))).sort(compare);
const counts={ok:0,review:0,exclude:0,unreviewed:0}; records.forEach(r=>counts[getStatus(r)]++);
el('stats').textContent = `Visualizzate ${filtered.length} / ${records.length} traiettorie · Valutate: ${counts.ok} OK, ${counts.review} da rivedere, ${counts.exclude} da escludere · Non valutate: ${counts.unreviewed}`;
el('empty').hidden=filtered.length>0;
const view=el('view').value;
const makeImage=(r,cam)=>`<figure class="picture"><a href="${esc(r[cam])}" target="_blank" rel="noopener" title="Apri PNG a risoluzione originale"><img loading="lazy" src="${esc(r[cam])}" alt="${esc(r.dataset+' '+r.pkl+' '+cam)}"/></a><figcaption>${cam === 'front' ? 'Camera frontale' : 'Camera gripper'}</figcaption></figure>`;
el('cards').innerHTML=filtered.map(r=>{
 const a=annotations[sid(r)] || {};
 const images=(view==='both' ? makeImage(r,'front')+makeImage(r,'gripper') : makeImage(r,view));
 return `<article class="card"><div class="cardhead"><div><div class="name">${esc(r.dataset)} · ${esc(r.task)}</div><div class="meta">${esc(r.pkl)}<br>${esc(r.source)}</div></div><div class="chip">t=${r.last_t} / T=${r.frames}</div></div><div class="photos ${view==='both'?'both':''}">${images}</div><div class="detail">Gripper ultimo frame: qpos=${esc(r.gripper_qpos)} · ${esc(r.gripper_binary)}<br>Comando originale finale: ${esc(r.terminal_original||'non disponibile')}</div><div class="audit"><select class="assessment" data-id="${esc(sid(r))}" aria-label="Esito manuale"><option value="unreviewed" ${!a.status||a.status==='unreviewed'?'selected':''}>Non valutato</option><option value="ok" ${a.status==='ok'?'selected':''}>OK</option><option value="review" ${a.status==='review'?'selected':''}>Da rivedere</option><option value="exclude" ${a.status==='exclude'?'selected':''}>Da escludere</option></select><textarea class="note" data-id="${esc(sid(r))}" placeholder="Note sul posizionamento finale...">${esc(a.note||'')}</textarea></div></article>`;
}).join('');
}
['dataset','task','source','view','status','sort'].forEach(id=>el(id).addEventListener('change',render));el('search').addEventListener('input',render);
el('cards').addEventListener('change',e=>{if(e.target.matches('.assessment')){const id=e.target.dataset.id;annotations[id]={...(annotations[id]||{}),status:e.target.value};save();render();}});
el('cards').addEventListener('input',e=>{if(e.target.matches('.note')){const id=e.target.dataset.id;annotations[id]={...(annotations[id]||{}),note:e.target.value};save();}});
function csvCell(v){let s=String(v??'');return '"'+s.replace(/"/g,'""')+'"'}
el('export').addEventListener('click',()=>{
const rows=records.map(r=>({...r,valutazione:getStatus(r),note:annotations[sid(r)]?.note||''}));
const csv='\ufeff'+fields.join(',')+'\r\n'+rows.map(r=>fields.map(k=>csvCell(r[k])).join(',')).join('\r\n');
const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='valutazioni_ultimi_frame.csv';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE, help='Directory datasets/mixture')
    parser.add_argument('--output', type=Path, default=None, help='Directory per la galleria HTML')
    parser.add_argument('--savers-dir', type=Path, default=None,
                        help='Directory contenente savers.py (solo se non importabile)')
    parser.add_argument('--datasets', nargs='+', choices=list(SOURCES), default=list(SOURCES),
                        help='Sottoinsieme dei dataset, per debug')
    parser.add_argument('--limit', type=int, default=None, help='Pilot: massimo N traiettorie per dataset')
    parser.add_argument('--overwrite', action='store_true', help='Consenti rigenerazione galleria preesistente')
    args = parser.parse_args()
    base = args.base.expanduser().resolve()
    dest = (args.output or base / 'debug_ultimi_frame').expanduser().resolve()
    if dest == base or dest in base.parents:
        parser.error('Output non puo\' essere uguale o padre di datasets/mixture')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit deve essere > 0')
    if (dest / 'index.html').exists() and not args.overwrite:
        parser.error(f'La galleria esiste gia\': {dest}/index.html. Usa --overwrite per rigenerarla.')
    load_savers(args.savers_dir)
    jobs = []
    for label in args.datasets:
        root = base / SOURCES[label]
        if not root.is_dir():
            raise FileNotFoundError(f'Dataset preprocessato mancante: {root}')
        selected = manifest_jobs(root)
        if args.limit is None and len(selected) != EXPECTED[label]:
            print(f'[ATTENZIONE] {label}: trovati {len(selected)} PKL nel manifest, attesi {EXPECTED[label]}', flush=True)
        if args.limit is not None:
            selected = selected[:args.limit]
        jobs.extend((label, path, rel, meta) for path, rel, meta in selected)
        print(f'{label}: {len(selected)} traiettorie selezionate dal manifest', flush=True)
    if not jobs:
        raise RuntimeError('Nessun PKL selezionato')
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    errors = []
    for index, (label, path, rel, meta) in enumerate(jobs, start=1):
        try:
            obs, frames = last_obs(path)
            images = {name: validate_image(obs, key, path) for name, key in IMAGES.items()}
            task = rel.parent.name
            assets = {}
            for name, rgb in images.items():
                target_rel = Path('images') / SOURCES[label] / task / f'{rel.stem}_{name}.png'
                save_png(rgb, dest / target_rel)
                assets[name] = target_rel.as_posix()
            gripper_state = np.asarray(obs.get('gripper_state', []), dtype=np.float64).reshape(-1)
            if gripper_state.size != 2:
                raise ValueError(f'gripper_state deve essere 2D, trovato {gripper_state.shape}')
            provenance = str(meta.get('source_dataset') or meta.get('source_group') or '—')
            terminal_original = str(meta.get('original_terminal_gripper_command') or meta.get('original_terminal_command') or '')
            report.append({
                'dataset': label,
                'task': task,
                'source': provenance,
                'pkl': rel.as_posix(),
                'frames': frames,
                'last_t': frames - 1,
                'front': assets['front'],
                'gripper': assets['gripper'],
                'gripper_qpos': f'{float(gripper_state[0]):.6f}',
                'gripper_binary': 'CLOSED' if gripper_state[1] >= .5 else 'OPEN',
                'terminal_original': terminal_original,
            })
            if index % 25 == 0 or index == len(jobs):
                print(f'[{index:03d}/{len(jobs):03d}] OK  {label}/{rel}', flush=True)
        except Exception as exc:
            msg = f'{type(exc).__name__}: {exc}'
            errors.append({'dataset': label, 'pkl': rel.as_posix(), 'error': msg})
            print(f'[{index:03d}/{len(jobs):03d}] ERROR {label}/{rel}: {msg}', flush=True)
    if report:
        safe_json = json.dumps(report, ensure_ascii=False).replace('</', '<\\/')
        (dest / 'index.html').write_text(HTML_TEMPLATE.replace('__DATA_JSON__', safe_json), encoding='utf-8')
        write_csv(dest / 'index.csv', report)
    write_csv(dest / 'errors.csv', errors)
    print('\n=== AUDIT ULTIMI FRAME ===', flush=True)
    print(f'Elaborate: {len(report)} / {len(jobs)}; errori: {len(errors)}', flush=True)
    print(f'Per dataset: {dict(Counter(r["dataset"] for r in report))}', flush=True)
    print(f'Apri: {dest / "index.html"}', flush=True)
    print(f'Inventario: {dest / "index.csv"}', flush=True)
    if errors:
        print(f'Errori: {dest / "errors.csv"}', flush=True)
    print('Nessun input preprocessato modificato.', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
