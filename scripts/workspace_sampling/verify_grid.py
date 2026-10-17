#!/usr/bin/env python3
"""Legge un JSON di waypoint già generato e produce 2 PNG 3D + HTML interattivo.

Non rigenera e non modifica i waypoint. Colori = quote Z0, Z1, ...:
  - reticolo_vista_obliqua.png  (camera tridimensionale angolata)
  - reticolo_vista_laterale.png (camera tridimensionale quasi lungo X)
  - reticolo_interattivo.html   (self-contained/offline, zoom e rotazione)

Esempio:
    python visualizza_reticolo_workspace.py \
      --grid workspace_grid_143.json \
      --output-dir output_visualizzazione

Dipendenze: numpy, scipy, matplotlib, plotly.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

IDS=('V000','V100','V010','V110','V001','V101','V011','V111')
EDGES=(
  ('V000','V100'),('V010','V110'),('V001','V101'),('V011','V111'),
  ('V000','V010'),('V100','V110'),('V001','V011'),('V101','V111'),
  ('V000','V001'),('V100','V101'),('V010','V011'),('V110','V111'),
)
COLORS=("#09008d","#bdca01","#F60000","#5400a3","#f562ba")


def load_grid(path: Path):
    grid=json.loads(path.read_text(encoding='utf-8'))
    if not grid.get('waypoints') or not grid.get('vertices'):
        raise ValueError('JSON senza waypoints oppure vertices')
    vertices={key:np.array(grid['vertices'][key],dtype=float) for key in IDS}
    waypoints=grid['waypoints']
    positions={int(w['id']):np.array(w['position'],dtype=float) for w in waypoints}
    if len(positions)!=len(waypoints):
        raise ValueError('ID waypoint duplicati')
    if any(not np.isfinite(x).all() for x in [*vertices.values(),*positions.values()]):
        raise ValueError('Coordinate non finite')
    return grid,vertices,waypoints,positions


def connections(waypoints, positions):
    """Solo segmenti fra indici consecutivi; non influisce sulla navigazione ROS."""
    lookup={(int(w['i']),int(w['j']),int(w['k'])):int(w['id']) for w in waypoints}
    segments=[]
    for w in waypoints:
        i,j,k=int(w['i']),int(w['j']),int(w['k'])
        for di,dj,dk in ((1,0,0),(0,1,0),(0,0,1)):
            other=lookup.get((i+di,j+dj,k+dk))
            if other is not None:
                segments.append((positions[int(w['id'])],positions[other]))
    return segments


def common_ranges(vertices):
    values=np.vstack(list(vertices.values()))
    mins=values.min(axis=0)-np.array((0.045,0.045,0.04))
    maxs=values.max(axis=0)+np.array((0.045,0.045,0.04))
    return mins,maxs


def plot_matplotlib(path, vertices, waypoints, positions, line_segments, *, elev, azim, title, dpi):
    fig=plt.figure(figsize=(13,8.6), facecolor='white')
    ax=fig.add_subplot(111,projection='3d',proj_type='persp')

    pts=np.vstack([vertices[k] for k in IDS])
    hull=ConvexHull(pts)
    triangles=[pts[s] for s in hull.simplices]
    surface=Poly3DCollection(triangles,facecolor='#9cb4b9',edgecolor='none',alpha=0.085)
    ax.add_collection3d(surface)

    for a,b in EDGES:
        x,y=vertices[a],vertices[b]
        ax.plot([x[0],y[0]],[x[1],y[1]],[x[2],y[2]],
                color='#64748b',alpha=.78,linewidth=1.6,zorder=1)
    if line_segments:
        ax.add_collection3d(Line3DCollection(line_segments,colors='#94a3b8',
                                           linewidths=0.70,alpha=0.27))

    groups={}
    for w in waypoints:
        groups.setdefault(int(w['k']),[]).append(w)
    handles=[]
    for k,ww in sorted(groups.items()):
        color=COLORS[k % len(COLORS)]
        xyz=np.vstack([positions[int(w['id'])] for w in ww])
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=39, color=color,
                   alpha=0.98,edgecolor='white',linewidths=.25,depthshade=False)
        handles.append(Line2D([0],[0],marker='o', linestyle='None',markersize=9,
                              markerfacecolor=color,markeredgecolor='white',
                              label=f'Z{k}  ·  z={xyz[0,2]:+.3f} m  ({len(ww)} punti)'))

    # Vertex labels to aid geometric reading, without coloring additions separately.
    for name,p in vertices.items():
        ax.scatter([p[0]],[p[1]],[p[2]],color='#374151',s=23,depthshade=False)
        ax.text(p[0]+.006,p[1],p[2]+.009,name,fontsize=8,color='#334155')

    mins,maxs=common_ranges(vertices)
    ax.set_xlim(mins[0],maxs[0]); ax.set_ylim(mins[1],maxs[1]); ax.set_zlim(mins[2],maxs[2])
    ax.set_box_aspect(maxs-mins,zoom=0.90)
    ax.view_init(elev=elev,azim=azim)
    ax.set(xlabel='X base_link (m)', ylabel='Y base_link (m)', zlabel='Z base_link (m)')
    ax.tick_params(labelsize=9)
    ax.xaxis.set_pane_color((.97,.98,.99,1))
    ax.yaxis.set_pane_color((.97,.98,.99,1))
    ax.zaxis.set_pane_color((.97,.98,.99,1))
    fig.suptitle(title,fontsize=17,y=.965,fontweight='bold')
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.995,.91),
               fontsize=10,framealpha=.96,title='Livelli Z',title_fontsize=11)
    fig.text(.5,.045,'Reticolo cartesiano 8,5 cm · ultima riga frontale Z0/Z1: passo Y ridotto',
             ha='center',color='#475569',fontsize=10)
    fig.savefig(path,dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot_html(path, vertices, waypoints, positions, line_segments):
    fig=go.Figure()
    pts=np.vstack([vertices[k] for k in IDS])
    hull=ConvexHull(pts)
    fig.add_trace(go.Mesh3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],
                            i=hull.simplices[:,0],j=hull.simplices[:,1],k=hull.simplices[:,2],
                            opacity=.10,color='#94a3b8',name='Volume dei vertici',
                            hoverinfo='skip',showlegend=True))
    ex,ey,ez=[],[],[]
    for a,b in EDGES:
        p,q=vertices[a],vertices[b]
        ex += [p[0],q[0],None];ey += [p[1],q[1],None];ez += [p[2],q[2],None]
    fig.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',
                               line=dict(color='#64748b',width=5),name='Spigoli',hoverinfo='skip'))
    if line_segments:
        gx,gy,gz=[],[],[]
        for p,q in line_segments:
            gx.extend((p[0],q[0],None));gy.extend((p[1],q[1],None));gz.extend((p[2],q[2],None))
        fig.add_trace(go.Scatter3d(x=gx,y=gy,z=gz,mode='lines',
                                   line=dict(color='#b0beca',width=2),opacity=.48,
                                   name='Collegamenti geometrici',hoverinfo='skip'))
    keys=list(IDS)
    fig.add_trace(go.Scatter3d(
        x=[vertices[k][0] for k in keys],
        y=[vertices[k][1] for k in keys],
        z=[vertices[k][2] for k in keys],
        mode='markers+text',text=keys,textposition='top center',
        marker=dict(size=3,color='#334155'),name='Vertici',
        hovertext=[f'{k}<br>XYZ: {vertices[k].round(6).tolist()}' for k in keys],hoverinfo='text'))

    level_trace_indices=[]
    for k in sorted(set(int(w['k']) for w in waypoints)):
        ww=[w for w in waypoints if int(w['k'])==k]
        xyz=np.vstack([positions[int(w['id'])] for w in ww])
        label=f'Z{k} · z={xyz[0,2]:+.3f} m · {len(ww)} waypoint'
        fig.add_trace(go.Scatter3d(
            x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',
            marker=dict(size=5,color=COLORS[k % len(COLORS)],
                        line=dict(color='white',width=.5)),name=label,
            hovertext=[f"ID {w['id']} · Z{k}<br>i={w['i']} j={w['j']} k={w['k']}"
                       f"<br>X={p[0]:.6f} m<br>Y={p[1]:.6f} m<br>Z={p[2]:.6f} m"
                       for w,p in zip(ww,xyz)],hoverinfo='text'))
        level_trace_indices.append((k,len(fig.data)-1))

    mins,maxs=common_ranges(vertices)
    scene=dict(xaxis=dict(title='X base_link (m)',range=[mins[0],maxs[0]]),
               yaxis=dict(title='Y base_link (m)',range=[mins[1],maxs[1]]),
               zaxis=dict(title='Z base_link (m)',range=[mins[2],maxs[2]]),
               aspectmode='data',camera=dict(eye=dict(x=1.6,y=1.5,z=.95)))
    # One common 3D viewer, dropdown hides only level markers, not faces or edges.
    masks=[]
    total=len(fig.data)
    masks.append(dict(label='Tutte le quote',method='update',
                      args=[{'visible':[True]*total}]))
    for k,ix in level_trace_indices:
        vis=[True]*total
        for _,jx in level_trace_indices:
            vis[jx]=(jx==ix)
        masks.append(dict(label=f'Solo Z{k}',method='update',args=[{'visible':vis}]))

    fig.update_layout(
        title=f'Workspace raffinato · reticolo cartesiano · {len(waypoints)} waypoint',
        scene=scene,
        width=1280,height=850,
        updatemenus=[dict(type='dropdown',x=0.03,y=1.06,xanchor='left',yanchor='top',
                          direction='down',buttons=masks,showactive=True)],
        margin=dict(l=0,r=0,t=100,b=0),
        legend=dict(x=.78,y=.96,bgcolor='rgba(255,255,255,0.82)'),
    )
    fig.write_html(str(path),include_plotlyjs=True,full_html=True,
                   config={'responsive':True,'displaylogo':False})


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=Path,default=root/'workspace_grid_143.json')
    parser.add_argument('--output-dir',type=Path,default=root/'reticolo_visualizzazioni')
    parser.add_argument('--dpi',type=int,default=180)
    parser.add_argument('--hide-connections',action='store_true',
                        help='Non disegnare i collegamenti fra indici consecutivi')
    args=parser.parse_args()
    if args.dpi < 50:
        parser.error('--dpi deve essere almeno 50')
    grid,verts,wp,positions=load_grid(args.grid)
    edges=[] if args.hide_connections else connections(wp,positions)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    oblique=args.output_dir/'reticolo_vista_obliqua.png'
    lateral=args.output_dir/'reticolo_vista_laterale.png'
    interactive=args.output_dir/'reticolo_interattivo.html'
    plot_matplotlib(oblique,verts,wp,positions,edges,elev=5,azim=-65,
                    title=f'Workspace raffinato · vista obliqua 3D · {len(wp)} waypoint',dpi=args.dpi)
    plot_matplotlib(lateral,verts,wp,positions,edges,elev=5,azim=0,
                    title=f'Workspace raffinato · vista laterale 3D · {len(wp)} waypoint',dpi=args.dpi)
    plot_html(interactive,verts,wp,positions,edges)
    print('Letti',len(wp),'waypoint dal file',args.grid)
    for k,ct in sorted(Counter(int(w['k']) for w in wp).items()):
        print(f' Z{k}: {ct} waypoint')
    for item in (oblique,lateral,interactive):
        print('Creato:',item.resolve())

if __name__=='__main__':
    main()
