#!/usr/bin/env python3
"""Anteprima RAW dei batch del training server VLA-JEPA da un PKL Exploration/Plays.

Eseguire nel container ROS con il package vla_jepa_online nel PYTHONPATH.
Ricostruisce gli indici e i prompt eseguendo le funzioni PURE del batch builder
installato, senza importarne Torch e senza avviare il modello.
Non applica crop, resize né normalizzazioni alle immagini esportate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping


DEFAULT_BUILDER = Path(
    '/opt/vla_jepa_train/src/vla_jepa_online/training_server/online_batch_builder.py'
)


def load_exact_batch_logic(builder_path: Path):
    """Carica le due funzioni effettive del batch builder, senza importare torch."""
    if not builder_path.is_file():
        raise FileNotFoundError(f'Batch builder non trovato: {builder_path}')
    tree = ast.parse(builder_path.read_text(encoding='utf-8'), filename=str(builder_path))
    needed = {'_window_starts', '_prompt_for_window'}
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in needed]
    found = {node.name for node in selected}
    if found != needed:
        raise RuntimeError(f'Funzioni mancanti nel batch builder: {needed - found}')
    namespace: dict[str, Any] = {'Mapping': Mapping, 'Any': Any}
    # Le due funzioni presenti nella versione corrente sono pure: usano solo builtins.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(builder_path), 'exec'), namespace)
    return namespace['_window_starts'], namespace['_prompt_for_window']


def generate_preview(pkl: Path, output: Path, batch_builder: Path) -> dict[str, Any]:
    # Usa il lettore originale della pipeline, che convalida struttura PKL,
    # frame 1-based, azioni 8D e metadati prima di produrre RecordedEpisode.
    from vla_jepa_online.exploration.trajectory_saver import TrajectorySaver

    window_starts, prompt_for_window = load_exact_batch_logic(batch_builder)
    meta, episode = TrajectorySaver.load_episode(pkl)
    frames = list(episode.frames)
    n = len(frames)
    if meta.get('sampling_mode') != 'motion_step':
        raise ValueError(f"sampling_mode inatteso: {meta.get('sampling_mode')!r}")
    if n < 8:
        raise ValueError(f'Il server richiede almeno 8 frame: il PKL ne contiene {n}')

    plays = meta.get('source') == 'plays'
    if plays and (meta.get('prompt_mode') != 'episode_prompt' or
                  not isinstance(meta.get('episode_prompt'), str) or
                  not meta['episode_prompt'].strip()):
        raise ValueError('Manifest Plays: episode_prompt mancante/invalido')

    # Per Exploration: finestre 1-8, 6-13, 11-18, ... (stride=5).
    # Per Plays: stessa funzione server, con sola ultima finestra riallineata.
    starts = window_starts(n, window_size=8, stride=5, repeat_last=plays)
    by_frame = [
        {'index': f.index,
         'front_stamp_ns': f.front_stamp_ns,
         'gripper_stamp_ns': f.gripper_stamp_ns,
         'event': f.event,
         'segment_id': f.segment_id,
         'motion_label': f.motion_label,
         'front': f'frames/front/{f.index:06d}.png',
         'gripper': f'frames/gripper/{f.index:06d}.png'}
        for f in frames
    ]

    windows = []
    for w_idx, start in enumerate(starts):
        selected = by_frame[start:start + 8]
        prompt = meta['episode_prompt'] if plays else prompt_for_window(selected)
        windows.append({
            'window_index': w_idx,
            'start_index_0based': start,
            'frame_indices_1based': [f['index'] for f in selected],
            'task': prompt,
            'frames': selected,
        })

    # Prima valida TUTTI i prompt, poi scrivi i file.
    output.mkdir(parents=True, exist_ok=True)
    (output / 'frames/front').mkdir(parents=True, exist_ok=True)
    (output / 'frames/gripper').mkdir(parents=True, exist_ok=True)
    for f in frames:
        # PNG decompressi/rigenerati dal TrajectorySaver: stessi pixel della
        # registrazione, senza crop/resize, nelle due viste originali.
        (output / 'frames/front' / f'{f.index:06d}.png').write_bytes(f.front_png)
        (output / 'frames/gripper' / f'{f.index:06d}.png').write_bytes(f.gripper_png)

    info = {'episode_id': meta.get('episode_id', pkl.stem),
            'source': 'plays' if plays else 'exploration',
            'num_frames': n, 'num_windows': len(windows),
            'window_size': 8, 'stride': 5,
            'front_training_feature': 'observation.images.exterior_1_left',
            'gripper_training_feature': 'observation.images.exterior_2_left',
            'model_input': 'due tensori RGB (post-crop/resize) + task testuale',
            'note': 'Le immagini estratte qui sono RAW; action e robot_state NON sono dati al WM.',
            'windows': windows}
    (output / 'windows.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')

    with (output / 'windows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['window_0based', 'frame_1based', 'event', 'segment_id',
                         'motion_label', 'task_sent_to_wm', 'front_png', 'gripper_png'])
        for window in windows:
            for f in window['frames']:
                writer.writerow([window['window_index'], f['index'], f['event'],
                                 f['segment_id'], f['motion_label'], window['task'],
                                 f['front'], f['gripper']])

    # Evita una dipendenza da un server HTTP: HTML statico, aperto via file://.
    payload = json.dumps(info, ensure_ascii=False).replace('<', r'\u003c')
    html_text = r'''<!doctype html>
<html lang="it"><head><meta charset="utf-8"><title>VLA-JEPA — Training PKL preview</title>
<style>
:root{font-family:system-ui,Arial,sans-serif;color-scheme:light}
body{margin:16px;max-width:1500px;background:#f8fafc;color:#17212f}
header{position:sticky;top:0;background:#fff;padding:14px 16px;border:1px solid #dde3ec;border-radius:10px;z-index:2}
label{font-weight:600}select,button{font-size:16px;padding:8px;margin:4px}
#prompt{white-space:pre-wrap;background:#edf4ff;padding:12px;border-radius:8px;line-height:1.5}
.note{font-size:13px;color:#525e6a} .frame{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px;margin:16px 0}
.images{display:grid;grid-template-columns:1fr 1fr;gap:10px}.images img{max-width:100%;height:auto;display:block}
.images a{display:block;overflow:auto} .name{font-weight:700;font-size:14px;padding:5px 0}
.meta{font-family:ui-monospace,monospace;font-size:13px;word-break:break-word}
@media(max-width:750px){.images{grid-template-columns:1fr}}
</style></head><body>
<header><h2 style="margin:0 0 6px">Anteprima dei batch di World Model training</h2>
<div id="stats"></div><div class="note">Immagini PNG originali (nessun crop/resize). Il browser le adatta alla larghezza dello schermo per visualizzarle: clicca su una vista per aprirla alla sua risoluzione nativa.</div>
<button id="prev">← Finestra precedente</button><label for="window">Finestra: </label><select id="window"></select><button id="next">Finestra successiva →</button>
<div id="indexes" class="meta"></div><p id="prompt"></p>
<div class="note">La stringa sopra è il campo <b>task[0]</b> passato al WM. <b>motion_label</b> viene usata per costruire il prompt Exploration, ma non è inviata come campo aggiuntivo. Azioni, stato robot e timestamp non sono usati nell'input video-only.</div>
</header><section id="images"></section>
<script>const DATA = __DATA__;
const sel=document.querySelector('#window'),area=document.querySelector('#images');
const esc=s=>String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
DATA.windows.forEach((w,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${i+1}/${DATA.num_windows} · frame ${w.frame_indices_1based[0]}–${w.frame_indices_1based.at(-1)}`;sel.append(o)});
document.querySelector('#stats').textContent=`${DATA.episode_id} · ${DATA.source} · ${DATA.num_frames} frame · ${DATA.num_windows} finestre · 8 frame/finestra · stride 5`;
function show(i){if(i<0||i>=DATA.num_windows)return;sel.value=i;const w=DATA.windows[i];
document.querySelector('#indexes').textContent=`Finestra ${i+1}/${DATA.num_windows} — start_index 0-based=${w.start_index_0based} — frame 1-based: ${w.frame_indices_1based.join(', ')}`;
document.querySelector('#prompt').textContent=`task[0] = ${JSON.stringify(w.task)}`;
area.innerHTML=w.frames.map((f,j)=>`<div class="frame"><div class="meta"><b>Frame ${j+1}/8 = ${f.index}</b> · event=${esc(f.event)} · segment=${esc(f.segment_id)} · motion_label=${esc(f.motion_label)}</div><div class="images"><div><div class="name">FRONT (RGB originale)</div><a href="${f.front}" target="_blank"><img loading="lazy" src="${f.front}"></a></div><div><div class="name">GRIPPER (RGB originale)</div><a href="${f.gripper}" target="_blank"><img loading="lazy" src="${f.gripper}"></a></div></div></div>`).join('');
window.scrollTo({top:0,behavior:'instant'});}
sel.addEventListener('change',()=>show(Number(sel.value)));
document.querySelector('#prev').onclick=()=>show(Number(sel.value)-1);
document.querySelector('#next').onclick=()=>show(Number(sel.value)+1);
show(0);
</script></body></html>'''.replace('__DATA__', payload)
    (output / 'index.html').write_text(html_text, encoding='utf-8')

    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pkl', type=Path, help='E00001.pkl o P00001.pkl')
    ap.add_argument('--out', type=Path, default=None, help='Directory output (default: accanto al PKL)')
    ap.add_argument('--batch-builder', type=Path, default=DEFAULT_BUILDER,
                    help='Sorgente del batch builder server: usato per prompt e finestre ESATTI')
    args = ap.parse_args()
    src = args.pkl.expanduser().resolve()
    out = args.out.expanduser().resolve() if args.out else src.parent / f'{src.stem}_training_preview'
    info = generate_preview(src, out, args.batch_builder)
    print(f"Episodio: {info['episode_id']} ({info['source']})")
    print(f"Frame: {info['num_frames']} | finestre: {info['num_windows']} | 8 frame, stride 5")
    for w in info['windows']:
        ids = w['frame_indices_1based']
        print(f"[{w['window_index']:03d}] frame {ids[0]:04d}–{ids[-1]:04d} | task = {w['task']}")
    print(f'\nApri nel browser: {out / "index.html"}')
    print(f'Tabella: {out / "windows.csv"}')
    print('Nota: immagini native; il vero batch del server viene poi croppato e ridimensionato a 224x224.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERRORE: {exc}', file=sys.stderr)
        raise SystemExit(1)
