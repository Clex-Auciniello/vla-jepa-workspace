#!/usr/bin/env python3
"""Genera la griglia 3D a partire dai vertici misurati con workspace_vertex_collector.py.

- Spaziatura nominale 8.5 cm su X, Y e Z (modificabile da CLI).
- Aggiunge l'estremo finale soltanto se il residuo supera 5 cm.
- Interpola gli 8 vertici misurati: funziona anche con il volume ruotato
  rispetto a base_link e con piccole imprecisioni nella misura.
- Esporta JSON + CSV. 
"""

import argparse
import csv
import json
import math
from pathlib import Path


# Valori iniziali concordati (metri).
XY_STEP_M = 0.085
Z_STEP_M = 0.085
END_GAP_THRESHOLD_M = 0.05

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERTICES = REPO_ROOT / "configs" / "workspace_vertices.json"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "workspace_grid.json"

VERTEX_IDS = ("V000", "V100", "V010", "V110", "V001", "V101", "V011", "V111")
EDGES = {
    "x": (("V000", "V100"), ("V010", "V110"), ("V001", "V101"), ("V011", "V111")),
    "y": (("V000", "V010"), ("V100", "V110"), ("V001", "V011"), ("V101", "V111")),
    "z": (("V000", "V001"), ("V100", "V101"), ("V010", "V011"), ("V110", "V111")),
}


def read_vertices(path):
    """Restituisce metadata, HOME e coordinate dei vertici in base_link."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("vertices", [])
    by_id = {row["vertex_id"]: row for row in rows}
    if len(rows) != 8 or set(by_id) != set(VERTEX_IDS):
        raise ValueError("Il JSON deve contenere esattamente V000, V100, V010, V110, V001, V101, V011, V111")
    if data.get("home") is None:
        raise ValueError("Manca HOME nel JSON del collector")

    corners = {}
    for vertex_id in VERTEX_IDS:
        row = by_id[vertex_id]
        point = tuple(float(row[key]) for key in ("x_m", "y_m", "z_m"))
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"Coordinate non valide per {vertex_id}")
        corners[vertex_id] = point
    return data, corners


def axis_length(corners, axis):
    """Lunghezza media dei quattro spigoli paralleli all'asse locale."""
    lengths = [math.dist(corners[a], corners[b]) for a, b in EDGES[axis]]
    return sum(lengths) / len(lengths)


def sample_axis(length, step, threshold):
    """0, step, 2*step, ...; aggiunge length solo se il residuo > threshold."""
    if length <= 0 or step <= 0 or threshold < 0:
        raise ValueError("Lunghezze e passi devono essere positivi; soglia >= 0")
    n = math.floor((length + 1e-9) / step)
    samples = [min(i * step, length) for i in range(n + 1)]
    if length - samples[-1] > threshold + 1e-9:
        samples.append(length)
    return samples


def lerp(a, b, t):
    return tuple((1.0 - t) * x + t * y for x, y in zip(a, b))


def interpolate(corners, fx, fy, fz):
    """Interpolazione trilineare: usa tutti gli 8 vertici misurati."""
    bottom_back = lerp(corners["V000"], corners["V100"], fx)  # un punto tra i 2 vertici del bordo inferiore dietro, se fx = 0.5 si trova a metà
    bottom_front = lerp(corners["V010"], corners["V110"], fx) # stesso punto sul bordo inferiore avanti
    top_back = lerp(corners["V001"], corners["V101"], fx)     # stesso punto sul bordo superiore dietro
    top_front = lerp(corners["V011"], corners["V111"], fx)    # stesso punto sul bordo superiore avanti
    bottom = lerp(bottom_back, bottom_front, fy)              # punto sul fondo tra i due punti intermedi trovati
    top = lerp(top_back, top_front, fy)                       # stesso punto ma sul soffitto
    return lerp(bottom, top, fz)                              # stesso punto ma tra il soffitto e il fondo


def build_grid(data, corners, xy_step, z_step, threshold):
    lengths = {axis: axis_length(corners, axis) for axis in "xyz"}
    xs = sample_axis(lengths["x"], xy_step, threshold)
    ys = sample_axis(lengths["y"], xy_step, threshold)
    zs = sample_axis(lengths["z"], z_step, threshold)

    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        raise ValueError("Workspace troppo piccolo rispetto a passo e soglia: uno degli assi ha meno di 2 nodi")
    if sum(corners[key][2] for key in ("V001", "V101", "V011", "V111")) <= sum(
        corners[key][2] for key in ("V000", "V100", "V010", "V110")
    ):
        raise ValueError("I vertici ALTI non risultano sopra quelli BASSI in base_link")

    waypoints = []
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                px, py, pz = interpolate(
                    corners, x / lengths["x"], y / lengths["y"], z / lengths["z"]
                )
                waypoints.append({
                    "id": len(waypoints),
                    "i": i,
                    "j": j,
                    "k": k,
                    "position": [round(px, 6), round(py, 6), round(pz, 6)],
                    "level": "PICK" if k == 0 else "TRANSFER",
                })

    return {
        "base_frame": data.get("base_frame", "base_link"),
        "tcp_frame": data.get("tcp_frame", "tcp_link"),
        "home": data["home"],
        "vertices": {name: list(corners[name]) for name in VERTEX_IDS},
        "dimensions_m": {name: round(value, 6) for name, value in lengths.items()},
        "sampling": {
            "xy_step_m": xy_step,
            "z_step_m": z_step,
            "end_gap_threshold_m": threshold,
            "axis_offsets_m": {
                "x": [round(x, 6) for x in xs],
                "y": [round(y, 6) for y in ys],
                "z": [round(z, 6) for z in zs],
            },
        },
        "grid_shape": {"nx": len(xs), "ny": len(ys), "nz": len(zs)},
        "num_waypoints": len(waypoints),
        "waypoints": waypoints,
    }


def save_grid(grid, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(grid, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("waypoint_id", "i", "j", "k", "level", "x_m", "y_m", "z_m"))
        for wp in grid["waypoints"]:
            writer.writerow((wp["id"], wp["i"], wp["j"], wp["k"], wp["level"], *wp["position"]))
    return csv_path


def save_plot(grid, path):
    """Visualizzazione offline opzionale: python workspace_grid.py --plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Per --plot installa matplotlib: pip install matplotlib") from exc

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    waypoints = grid["waypoints"]
    for level, label in (("PICK", "Z0 - PICK"), ("TRANSFER", "Z superiori")):
        coords = [wp["position"] for wp in waypoints if wp["level"] == level]
        ax.scatter([p[0] for p in coords], [p[1] for p in coords],
                   [p[2] for p in coords], s=8, alpha=0.7, label=label)

    corners = grid["vertices"]
    for pairs in EDGES.values():
        for a, b in pairs:
            ax.plot([corners[a][0], corners[b][0]],
                    [corners[a][1], corners[b][1]],
                    [corners[a][2], corners[b][2]], linewidth=1, alpha=0.6)
    ax.set(xlabel="X base_link (m)", ylabel="Y base_link (m)",
           zlabel="Z base_link (m)", title=f"Workspace: {len(waypoints)} waypoint")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=Path, default=DEFAULT_VERTICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--xy-step", type=float, default=XY_STEP_M, help="Passo XY in metri")
    parser.add_argument("--z-step", type=float, default=Z_STEP_M, help="Passo Z in metri")
    parser.add_argument("--min-end-gap", type=float, default=END_GAP_THRESHOLD_M,
                        help="Includi estremo finale se residuo > soglia (metri)")
    parser.add_argument("--plot", action="store_true", help="Salva anche il grafico 3D in outputs/")
    args = parser.parse_args()

    data, corners = read_vertices(args.vertices)
    grid = build_grid(data, corners, args.xy_step, args.z_step, args.min_end_gap)
    csv_path = save_grid(grid, args.output)
    shape = grid["grid_shape"]

    print("\n=== Workspace grid ===")
    print(f"Vertici: {args.vertices}")
    print(f"Dimensioni stimate (m): {grid['dimensions_m']}")
    print(f"Passo XY: {args.xy_step:.3f} m | Z: {args.z_step:.3f} m")
    print(f"Estremo aggiunto solo se residuo > {args.min_end_gap:.3f} m")
    print(f"Griglia: {shape['nx']} x {shape['ny']} x {shape['nz']} = {grid['num_waypoints']} punti")
    print(f"PICK: {shape['nx'] * shape['ny']} | TRANSFER: {grid['num_waypoints'] - shape['nx'] * shape['ny']}")
    print(f"JSON: {args.output.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")

    if args.plot:
        plot_path = REPO_ROOT / "outputs" / "workspace_grid.png"
        save_plot(grid, plot_path)
        print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
