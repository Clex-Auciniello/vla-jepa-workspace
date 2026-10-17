#!/usr/bin/env python3
"""Genera da ZERO il reticolo cartesiano UR5e dai SOLI 8 vertici misurati/raffinati.

Regole della configurazione approvata:
- coordinate base_link, reticolo cartesiano globale (non trilineare);
- passo esatto 0.085 m in X/Y/Z sulle righe regolari;
- origine fissata alla griglia concordata: (0.245, 0.287, -0.077) m;
- conserva solo i punti all'interno dell'inviluppo convesso degli 8 vertici;
- aggiunge su Z0 e Z1 una riga anteriore: Y del limite della parete a Z1,
  arretrata di 1 mm, stessa Y per entrambi i livelli;
- le ultime due righe distano ~4.55 cm in Y (eccezione intenzionale);
- salva il JSON dei waypoint con posizione cartesiana assoluta.

Non legge JSON di waypoint preesistenti. Richiede: numpy, scipy.

Esempio:
    python genera_reticolo_workspace.py \
        --vertices workspace_vertices_refined_input.json \
        --output workspace_grid_143.json

Il reticolo indica solo un insieme di target TCP geometricamente interni:
NON certifica collisioni, reachability o sicurezza delle traiettorie.
Il MotionSampler ROS attuale deve essere aggiornato per le perturbazioni.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

VERTEX_IDS = ('V000','V100','V010','V110','V001','V101','V011','V111')
DEFAULT_STEP = 0.085
DEFAULT_ORIGIN = (0.245, 0.287, -0.077)
DEFAULT_WALL_INSET_MM = 1.0


def read_vertices(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    raw = data.get('vertices')
    if not isinstance(raw, list) or len(raw) != 8:
        raise ValueError('Il JSON deve contenere esattamente 8 vertici')
    if data.get('home') is None:
        raise ValueError('Il JSON deve contenere HOME')
    if {v.get('vertex_id') for v in raw} != set(VERTEX_IDS):
        raise ValueError('ID vertici inattesi o duplicati')
    vertices = {
        v['vertex_id']: np.array([float(v['x_m']), float(v['y_m']), float(v['z_m'])])
        for v in raw
    }
    if not all(np.isfinite(v).all() for v in vertices.values()):
        raise ValueError('Coordinate dei vertici non finite')
    return data, vertices


def inside(point: np.ndarray, hull: ConvexHull, tolerance: float = 1e-9) -> bool:
    return bool(np.all(hull.equations[:,:3] @ point + hull.equations[:,3] <= tolerance))


def max_front_y_at_xz(x: float, z: float, hull: ConvexHull) -> float:
    """Intersezione con la parete a Y massimo dei semispazi del convex hull."""
    upper, lower = [], []
    for a,b,c,d in hull.equations:
        rhs = -a*x - c*z - d
        if b > 1e-10:
            upper.append(rhs / b)
        elif b < -1e-10:
            lower.append(rhs / b)
        elif rhs < -1e-9:
            raise ValueError(f'La retta a X={x:.4f}, Z={z:.4f} non incontra il volume')
    if not upper:
        raise ValueError('La geometria non definisce una parete anteriore Y+')
    ymax = float(min(upper))
    if lower and ymax < max(lower)-1e-9:
        raise ValueError('Nessuna intersezione valida lungo Y')
    return ymax


def build(data: dict, vertices: dict, step: float, origin: np.ndarray,
          wall_inset_mm: float, expected_count: int | None):
    if step <= 0 or not math.isfinite(step):
        raise ValueError('Passo deve essere positivo e finito')
    if wall_inset_mm < 0 or not math.isfinite(wall_inset_mm):
        raise ValueError('wall_inset_mm deve essere >= 0 e finito')
    if not np.isfinite(origin).all():
        raise ValueError('Origine non finita')

    points = np.vstack([vertices[k] for k in VERTEX_IDS])
    hull = ConvexHull(points)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    if not np.all((origin > mins) & (origin < maxs)):
        raise ValueError('Origine fuori dal bounding box del poliedro')

    # Convenzione i cresce verso -X; j cresce verso +Y; k verso +Z.
    nx = math.ceil((origin[0]-mins[0])/step)+1
    ny = math.ceil((maxs[1]-origin[1])/step)+1
    nz = math.ceil((maxs[2]-origin[2])/step)+1

    waypoints = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                xyz = origin + np.array((-i*step, j*step, k*step))
                if inside(xyz,hull):
                    waypoints.append({
                        'i': i, 'j': j, 'k': k,
                        'position': [round(float(v),6) for v in xyz],
                        'level': 'PICK' if k == 0 else 'TRANSFER',
                        'extra_front': False,
                    })
    nominal_count = len(waypoints)
    if nz < 2:
        raise ValueError('Servono almeno due livelli Z')

    # Ultima riga esistente con una o più colonne presenti su ENTRAMBI Z0/Z1.
    rows0 = {w['j'] for w in waypoints if w['k']==0}
    rows1 = {w['j'] for w in waypoints if w['k']==1}
    common_rows = rows0 & rows1
    if not common_rows:
        raise ValueError('Nessuna riga comune a Z0 e Z1')
    j_last = max(common_rows)
    cols0 = {w['i'] for w in waypoints if w['j']==j_last and w['k']==0}
    cols1 = {w['i'] for w in waypoints if w['j']==j_last and w['k']==1}
    cols = sorted(cols0 & cols1)
    if not cols:
        raise ValueError('Nessuna colonna condivisa sulle ultime righe Z0/Z1')

    y_previous = float(origin[1] + j_last*step)
    z0, z1 = float(origin[2]), float(origin[2]+step)
    wall_limits = {i: max_front_y_at_xz(float(origin[0]-i*step), z1, hull) for i in cols}
    y_wall = min(wall_limits.values())
    y_new = y_wall - wall_inset_mm/1000.0
    if y_new <= y_previous+1e-9:
        raise ValueError('Nessuno spazio per riga place supplementare')

    # Indice j distinto; il valore Y è volutamente NON origin_y + j*step.
    j_new = max(w['j'] for w in waypoints) + 1
    for k, z in ((0,z0),(1,z1)):
        for i in cols:
            xyz = np.array((float(origin[0]-i*step), y_new,z))
            if not inside(xyz,hull):
                raise ValueError(f'Riga frontale esterna per i={i}, k={k}: {xyz}')
            waypoints.append({
                'i':i,'j':j_new,'k':k,
                'position':[round(float(v),6) for v in xyz],
                'level':'PICK' if k==0 else 'TRANSFER',
                'extra_front':True,
            })

    waypoints.sort(key=lambda w: (w['k'],w['j'],w['i']))
    for identifier, wp in enumerate(waypoints):
        wp['id']=identifier

    # Controllo: distanze dei vicini ordinari 8.5 cm;
    # una e una sola differenza in Y sulle righe finali supplementari.
    idx = {(w['i'],w['j'],w['k']): np.array(w['position']) for w in waypoints}
    for w in waypoints:
        i,j,k = w['i'],w['j'],w['k']
        p = idx[(i,j,k)]
        for ax, delta in enumerate(((1,0,0),(0,1,0),(0,0,1))):
            key = (i+delta[0],j+delta[1],k+delta[2])
            q = idx.get(key)
            if q is None:
                continue
            d = float(np.linalg.norm(q-p))
            if ax==1 and ((w['extra_front'] or
                           any(v['extra_front'] for v in waypoints if (v['i'],v['j'],v['k'])==key))):
                # Questo caso è gestito sotto per evitare confusione: la nuova
                # riga rappresenta la sola eccezione al passo cartesiano fisso.
                continue
            if ax == 1 and key[1]==j_new:
                continue
            if abs(d-step)>2e-6:
                raise RuntimeError(f'Distanza non conforme {d:.6f} m: {w["id"]} -> {key}')
    if expected_count is not None and len(waypoints) != expected_count:
        raise RuntimeError(f'Attesi {expected_count} waypoint, ottenuti {len(waypoints)}. '
                           'Verifica vertici, origine e parametri.')

    levels = {str(k):sum(w['k']==k for w in waypoints) for k in range(nz)}
    return {
        'base_frame':data.get('base_frame','base_link'),
        'tcp_frame':data.get('tcp_frame','tcp_link'),
        'home':data['home'],
        'vertices':{k:vertices[k].tolist() for k in VERTEX_IDS},
        'preview_only':True,
        'warning':('Reticolo geometrico. Non utilizzare per comandare il robot prima '
                   'di aggiornare MotionSampler e validare collisioni/traiettorie.'),
        'sampling':{
            'step_m':step,
            'lattice_origin_m':origin.tolist(),
            'index_direction':{'i':'-X','j':'+Y','k':'+Z'},
            'wall_inset_mm':wall_inset_mm,
            'last_regular_y_m':y_previous,
            'front_boundary_y_at_z1_m':y_wall,
            'extra_row_y_m':y_new,
            'distance_previous_to_extra_cm':(y_new-y_previous)*100,
            'distance_previous_to_wall_cm':(y_wall-y_previous)*100,
            'extra_row_j':j_new,
            'extra_row_i':cols,
            'n_nominal':nominal_count,
            'n_added':len(waypoints)-nominal_count,
            'n_total':len(waypoints),
            'per_level_count':levels,
        },
        'grid_shape_bounds':{'nx':nx,'ny':ny,'nz':nz},
        'num_waypoints':len(waypoints),
        'waypoints':waypoints,
    }


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vertices',type=Path,default=root/'workspace_vertices_refined_input.json')
    parser.add_argument('--output',type=Path,default=root/'workspace_grid_143.json')
    parser.add_argument('--step-m',type=float,default=DEFAULT_STEP)
    parser.add_argument('--origin',nargs=3,type=float,default=DEFAULT_ORIGIN,metavar=('X','Y','Z'))
    parser.add_argument('--wall-inset-mm',type=float,default=DEFAULT_WALL_INSET_MM)
    parser.add_argument('--expected-count',type=int,default=143,
                        help='Controllo del numero totale; -1 per disattivare')
    args=parser.parse_args()
    data,vertices=read_vertices(args.vertices)
    result=build(data,vertices,args.step_m,np.array(args.origin,dtype=float),
                 args.wall_inset_mm, None if args.expected_count == -1 else args.expected_count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    meta=result['sampling']
    print(f"Creato: {args.output.resolve()}")
    print(f"Waypoint: {result['num_waypoints']} = {meta['n_nominal']} regolari + {meta['n_added']} supplementari")
    print('Per quota: '+', '.join(f"Z{k}: {n}" for k,n in meta['per_level_count'].items()))
    print(f"Ultima riga Y = {meta['last_regular_y_m']:.6f} m; nuova Y = {meta['extra_row_y_m']:.6f} m")
    print(f"Intervallo Y finale = {meta['distance_previous_to_extra_cm']:.3f} cm")
    print('AVVISO: reticolo geometrico, non ancora pronto per esecuzione ROS.')

if __name__=='__main__':
    main()
