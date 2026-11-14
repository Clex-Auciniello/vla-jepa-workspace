
#!/usr/bin/env python3
"""
Confronta le delta-action ricostruite da:

1. Dataset canonico VLA-JEPA (TFRecord e scripted)
2. osvi_awda_rollout_clean
3. result_osvi_awda_human_clean

Usa la convenzione del dataset canonico:
  delta_xyz = (p[t+1] - p[t]) / 0.05
  delta_rpy = Euler_xyz(R[t+1] @ R[t].inv()) / 0.05

Per OSVI ignora le action originali 8D.
Per reference verifica la corrispondenza con le label 7D.

Non modifica alcun PKL.
"""

import csv
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation
import savers


BASE = Path("/mnt/beegfs/a.cuciniello9/datasets")

REFERENCE = BASE / "robotic/vla_jepa_offline_merged"

OSVI = BASE / "mixture/osvi_awda_rollout_clean"

OSVI_HUMAN = BASE / "mixture/result_osvi_awda_human_clean"

OUTPUT = BASE / "osvi_delta_scale_audit"

SCALE = 0.05

SUCCESS_KEYS = (
    "object_reached",
    "object_picked",
    "object_placed",
)


def get_step(traj, i):
    if hasattr(traj, "get"):
        try:
            return traj.get(i, decompress=False)
        except TypeError:
            return traj.get(i)

    return traj[i]


def load_trajectory(path):
    with path.open("rb") as f:
        payload = pickle.load(f)

    return payload["traj"]


def validate_success(path):
    json_path = path.with_suffix(".json")

    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not all(
        metadata.get(k) == 1
        for k in SUCCESS_KEYS
    ):
        raise ValueError(
            f"JSON senza terna 1/1/1: {json_path}"
        )


def read_poses(traj, is_reference):
    positions = []
    rotations = []
    actions = []

    for t in range(len(traj)):
        step = get_step(traj, t)
        obs = step["obs"]

        if is_reference:
            # Canonical dataset: XYZ + Euler XYZ
            state = np.asarray(
                obs["EEF_state"],
                dtype=np.float64
            ).reshape(-1)

            if state.size != 6:
                raise ValueError(
                    f"EEF_state non 6D allo step {t}"
                )

            positions.append(state[:3])

            rotations.append(
                Rotation.from_euler("xyz", state[3:6])
            )

            action = np.asarray(
                step["action"],
                dtype=np.float64
            ).reshape(-1)

            if action.size != 7:
                raise ValueError(
                    f"Action reference non 7D: {action.size}"
                )

            actions.append(action)

        else:
            # OSVI: absolute XYZ + quaternion XYZW
            pos = np.asarray(
                obs["eef_pos"],
                dtype=np.float64
            ).reshape(-1)

            quat = np.asarray(
                obs["eef_quat"],
                dtype=np.float64
            ).reshape(-1)

            if pos.size != 3 or quat.size != 4:
                raise ValueError(
                    f"Pose OSVI non valida allo step {t}"
                )

            positions.append(pos)

            rotations.append(
                Rotation.from_quat(quat)
            )

    positions = np.stack(positions)

    if not np.isfinite(positions).all():
        raise ValueError("Posizioni non finite")

    rotations = Rotation.concatenate(rotations)

    return positions, rotations, actions


def reconstruct_actions(positions, rotations):
    # Differenze tra osservazioni consecutive.
    delta_xyz = positions[1:] - positions[:-1]

    # Rotazione relativa in base_link.
    relative_rotation = (
        rotations[1:] * rotations[:-1].inv()
    )

    delta_euler = relative_rotation.as_euler(
        "xyz", degrees=False
    )

    # Formato canonico 6D, senza gripper.
    actions_6d = np.concatenate(
        [delta_xyz, delta_euler],
        axis=1
    ) / SCALE

    return actions_6d, delta_xyz, delta_euler


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]

    if not a.size:
        return {"n": 0}

    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def analyze_source(source_name, directory, reference=False):
    paths = sorted(directory.rglob("*.pkl"))

    if not paths:
        raise RuntimeError(
            f"Nessun PKL in {directory}"
        )

    print(f"\nSOURCE: {source_name}")
    print(f"PKL trovati: {len(paths)}")

    per_traj = []
    actions_all = defaultdict(list)
    label_errors = []

    for index, path in enumerate(paths, 1):
        if index == 1 or index % 20 == 0:
            print(
                f"  [{index}/{len(paths)}] {path.name}",
                flush=True
            )

        if not reference:
            validate_success(path)

        traj = load_trajectory(path)

        if len(traj) < 2:
            raise ValueError(
                f"Traiettoria troppo corta: {path}"
            )

        positions, rotations, original_actions = (
            read_poses(traj, reference)
        )

        action6, dxyz, drot = reconstruct_actions(
            positions, rotations
        )

        xyz_norm_m = np.linalg.norm(dxyz, axis=1)
        rot_norm_rad = np.linalg.norm(drot, axis=1)

        xyz_norm_scaled = np.linalg.norm(
            action6[:, :3], axis=1
        )

        rot_norm_scaled = np.linalg.norm(
            action6[:, 3:6], axis=1
        )

        if reference:
            original = np.stack(original_actions)

            # La label[t] deve descrivere il movimento
            # fra l'osservazione t e quella t+1.
            errors = np.linalg.norm(
                original[:-1, :6] - action6,
                axis=1
            )

            label_errors.extend(errors.tolist())

        group = source_name

        if reference:
            filename = path.name.lower()

            if "tfrecord" in filename:
                group = "reference_tfrecord"
            elif "scripted" in filename:
                group = "reference_scripted"
            else:
                group = "reference_other"

        for name, values in {
            "xyz_m": xyz_norm_m,
            "rot_rad": rot_norm_rad,
            "xyz_scaled": xyz_norm_scaled,
            "rot_scaled": rot_norm_scaled,
        }.items():
            actions_all[group, name].extend(
                values.tolist()
            )

        for j, component in enumerate(
            ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
        ):
            actions_all[group, component].extend(
                np.abs(action6[:, j]).tolist()
            )

        per_traj.append({
            "source": group,
            "path": str(path),
            "steps": len(traj),
            "transitions": len(action6),
            "xyz_median_mm": float(
                np.median(xyz_norm_m) * 1000
            ),
            "xyz_p95_mm": float(
                np.percentile(xyz_norm_m, 95) * 1000
            ),
            "xyz_max_mm": float(
                xyz_norm_m.max() * 1000
            ),
            "rot_median_deg": float(
                np.degrees(np.median(rot_norm_rad))
            ),
            "rot_p95_deg": float(
                np.degrees(np.percentile(rot_norm_rad, 95))
            ),
            "scaled_xyz_median": float(
                np.median(xyz_norm_scaled)
            ),
            "scaled_xyz_p95": float(
                np.percentile(xyz_norm_scaled, 95)
            ),
            "scaled_rot_median": float(
                np.median(rot_norm_scaled)
            ),
            "scaled_rot_p95": float(
                np.percentile(rot_norm_scaled, 95)
            ),
            "zero_xyz_fraction": float(
                np.mean(xyz_norm_m < 1e-6)
            ),
            "transitions_over_5cm": int(
                np.sum(xyz_norm_m > 0.05)
            ),
        })

    print(f"PKL analizzati: {len(per_traj)}")

    result = {}

    groups = sorted({
        g for g, _ in actions_all.keys()
    })

    for group in groups:
        result[group] = {
            name: stats(values)
            for (g, name), values in actions_all.items()
            if g == group
        }

        result[group]["trajectories"] = sum(
            r["source"] == group
            for r in per_traj
        )

        result[group]["transitions"] = len(
            actions_all[group, "xyz_m"]
        )

    if reference:
        result["reference_label_validation"] = (
            stats(label_errors)
        )

    return result, per_traj


def print_summary(name, data):
    print(f"\n{'=' * 65}")
    print(name)
    print(f"{'=' * 65}")

    print("Traiettorie:", data["trajectories"])
    print("Transizioni:", data["transitions"])

    metrics = (
        ("xyz_m", "Traslazione [mm]", 1000),
        ("rot_rad", "Rotazione [deg]", 180 / np.pi),
        ("xyz_scaled", "Traslazione /0.05", 1),
        ("rot_scaled", "Rotazione /0.05", 1),
    )

    for key, label, factor in metrics:
        v = data[key]

        print(
            f"{label:23s} "
            f"p50={v['p50'] * factor:8.4f} "
            f"p95={v['p95'] * factor:8.4f} "
            f"p99={v['p99'] * factor:8.4f} "
            f"max={v['max'] * factor:8.4f}"
        )

    print("\nComponenti assolute scalate:")

    for k in (
        "dx", "dy", "dz",
        "droll", "dpitch", "dyaw"
    ):
        v = stats(np.abs(data[k]["raw_values"])) \
            if "raw_values" in data[k] \
            else data[k]

        print(
            f"  {k:7s} "
            f"p50={v['p50']:.4f} "
            f"p95={v['p95']:.4f} "
            f"p99={v['p99']:.4f}"
        )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    configs = [
        ("reference", REFERENCE, True),
        ("osvi_awda_clean", OSVI, False),
        ("osvi_human_clean", OSVI_HUMAN, False),
    ]

    summary = {
        "scale_factor": SCALE,
        "sources": {},
    }

    all_rows = []

    for name, directory, is_reference in configs:
        source_result, rows = analyze_source(
            name, directory, is_reference
        )

        summary["sources"].update(source_result)
        all_rows.extend(rows)

    # Statistiche in valore assoluto per ogni componente
    # delle delta ricostruite.
    # I valori signed sono conservati nel riepilogo sopra.

    with (OUTPUT / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    csv_path = OUTPUT / "per_trajectory.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    for group, values in summary["sources"].items():
        if group == "reference_label_validation":
            continue

        print_summary(group, values)

    ref = summary["sources"][
        "reference_label_validation"
    ]

    print("\nVERIFICA REFERENCE ACTION")
    print("Errore L2 tra label e delta ricostruita:")
    print(f"  p50: {ref['p50']:.8f}")
    print(f"  p95: {ref['p95']:.8f}")
    print(f"  max: {ref['max']:.8f}")

    print("\nOUTPUT:")
    print(OUTPUT / "summary.json")
    print(OUTPUT / "per_trajectory.csv")


if __name__ == "__main__":
    main()
