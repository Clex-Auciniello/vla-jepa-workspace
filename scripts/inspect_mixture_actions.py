
#!/usr/bin/env python3
"""Analisi NON DISTRUTTIVA delle action label di reference e rollout OSVI-AWDA.

Confronta lo stato end-effector misurato in PKL consecutivi:
  Delta posizione = p[t+1] - p[t]
  Delta rotazione = Euler_xyz(R[t+1] @ R[t].inv())
  Azione canonica per le prime 6 componenti = delta / 0.05.

Legge le action registrate e riconosce 7D (potenziali delta) / 8D
(potenziali target assoluti), senza confonderle con le delta misurate.

Per OSVI applica SOLO il filtro sui JSON associati (3 success flags = 1).
Non scrive mai nelle sorgenti. Non apre né decodifica immagini.

Dipendenze: numpy, scipy, librerie per deserializzare i PKL custom.
Se il PKL contiene 'savers.Trajectory' impostare il PYTHONPATH corretto.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pickle
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

SUCCESS_KEYS = ("object_reached", "object_picked", "object_placed")
SCALE_FACTOR = 0.05

POS_KEYS = (
    "eef_pos", "ee_pos", "tcp_pos",
    "end_effector_pos", "eef_position", "position"
)
QUAT_KEYS = (
    "eef_quat", "ee_quat", "tcp_quat",
    "eef_quaternion", "orientation"
)
GRIP_KEYS = (
    "gripper_qpos", "gripper_state",
    "gripper_position", "gripper_pos"
)
STATE_KEYS = ("eef_state", "EEF_state")
TIMESTAMP_KEYS = (
    "timestamp", "time_sec", "time", "stamp",
    "frame_timestamp", "timestamp_sec"
)


def mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def get_field(obj, fields):
    d = mapping(obj)
    for field in fields:
        if field in d:
            return d[field], field
    return None, None


def compact_value(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype)
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value)
        }
    if isinstance(value, (str, int, bool, float)) or value is None:
        return value if len(str(value)) < 120 else str(value)[:120]
    return {"type": type(value).__name__}


def to_vector(value, n=None):
    if value is None:
        return None
    try:
        a = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if n is not None and a.size != n:
        return None
    if not np.isfinite(a).all():
        return None
    return a


def extract_payload(path):
    with path.open("rb") as stream:
        data = pickle.load(stream)

    root = mapping(data)
    trajectory = root.get(
        "traj",
        root.get("trajectory", root.get("steps", data))
    )
    return data, trajectory


def length_of_trajectory(traj):
    if hasattr(traj, "T"):
        value = getattr(traj, "T")
        if callable(value):
            value = value()
        try:
            n = int(value)
            if n >= 0:
                return n
        except (TypeError, ValueError):
            pass

    try:
        return len(traj)
    except (TypeError, ValueError):
        pass

    d = mapping(traj)
    for name in ("_data", "data", "steps", "_steps", "trajectory"):
        if isinstance(d.get(name), (list, tuple)):
            return len(d[name])

    raise TypeError(
        f"Impossibile trovare il numero di step per {type(traj)}"
    )


def get_step(traj, index):
    # Struttura standard multi_task_il.datasets.savers.Trajectory.
    if isinstance(traj, (list, tuple)):
        return traj[index]

    if isinstance(traj, dict):
        steps = traj.get(
            "steps",
            traj.get("traj", traj.get("data"))
        )
        if isinstance(steps, (list, tuple)):
            return steps[index]
        return traj[index] if index in traj else traj[str(index)]

    if hasattr(traj, "get"):
        try:
            return traj.get(index, decompress=False)
        except TypeError:
            return traj.get(index)

    if hasattr(traj, "__getitem__"):
        return traj[index]

    raise TypeError(
        f"Nessuna interfaccia di step disponibile: {type(traj)}"
    )


def extract_obs_and_info(step):
    d = mapping(step)
    obs = d.get(
        "obs",
        d.get("observation", d.get("observations", d))
    )
    info = d.get("info", {})
    return mapping(obs), mapping(info), d


def quaternion_xyzw(value):
    q = to_vector(value, 4)
    if q is None:
        return None

    norm = np.linalg.norm(q)
    if norm < 1e-8 or norm > 1e8:
        return None

    return q / norm


def extract_pose(obs):
    # Preferiamo pose assolute misurate,
    # non i target del campo action.
    pos_value, pos_key = get_field(obs, POS_KEYS)
    quat_value, quat_key = get_field(obs, QUAT_KEYS)

    pos = to_vector(pos_value, 3)
    quat = quaternion_xyzw(quat_value)

    # Fall-back per EEF_state 6D = xyz + euler xyz,
    # oppure 7D xyz+xyzw.
    st_val, st_key = get_field(obs, STATE_KEYS)
    st = to_vector(st_val)

    if st is not None:
        if pos is None and st.size >= 3:
            pos, pos_key = st[:3], f"{st_key}[:3]"

        if quat is None and st.size == 6:
            quat = Rotation.from_euler(
                "xyz", st[3:6]
            ).as_quat()
            quat_key = f"{st_key}[3:6]_euler_xyz"

        elif quat is None and st.size == 7:
            quat = quaternion_xyzw(st[3:7])
            quat_key = f"{st_key}[3:7]_xyzw_assumed"

    return pos, quat, pos_key, quat_key


def extract_timestamp(step_dict, obs, info):
    for source in (info, step_dict, obs):
        for key in TIMESTAMP_KEYS:
            if key not in source:
                continue

            value = source[key]
            if (
                isinstance(value, (int, float, np.number))
                and np.isfinite(value)
            ):
                return float(value)

    return None


def get_gripper(obs):
    value, key = get_field(obs, GRIP_KEYS)
    if value is None:
        return None, None

    try:
        arr = np.asarray(
            value, dtype=np.float64
        ).reshape(-1)

        # Non assumiamo che tutti i dataset
        # usino la stessa componente.
        if arr.size == 1 and np.isfinite(arr[0]):
            return float(arr[0]), key

        if (
            arr.size >= 2
            and key == "gripper_state"
            and np.isfinite(arr[1])
        ):
            return float(arr[1]), key + "[1]"

    except (ValueError, TypeError):
        pass

    return None, key


def extract_one_step(step):
    obs, info, d = extract_obs_and_info(step)
    p, q, pk, qk = extract_pose(obs)
    g, gk = get_gripper(obs)
    a = to_vector(d.get("action"))

    return {
        "pos": p,
        "quat": q,
        "gripper": g,
        "action": a,
        "timestamp": extract_timestamp(d, obs, info),
        "keys": {
            "step": sorted(map(str, d.keys())),
            "obs": sorted(map(str, obs.keys())),
            "info": sorted(map(str, info.keys())),
            "pos_key": pk,
            "quat_key": qk,
            "gripper_key": gk,
            "action_shape": (
                list(a.shape) if a is not None else None
            ),
        },
    }


def quantiles(
    values,
    stats=(0.0, .50, .90, .95, .99, 1.0)
):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if not arr.size:
        return {"n": 0}

    result = {
        "n": int(arr.size),
        "mean": float(arr.mean())
    }

    for pct, q in zip(
        ("min", "p50", "p90", "p95", "p99", "max"),
        stats
    ):
        result[pct] = float(np.quantile(arr, q))

    return result


def summarise_values(traces):
    keys = (
        "delta_pos_m",
        "delta_rot_rad",
        "scaled_delta_pos",
        "scaled_delta_rot",
        "dx_m",
        "dy_m",
        "dz_m",
        "droll_rad",
        "dpitch_rad",
        "dyaw_rad",
        "label7_pos_err_scaled",
        "label7_pos_err_raw",
        "label7_rot_err_scaled",
        "target8_pos_dist_current_m",
        "target8_pos_dist_next_m",
        "time_gap",
    )

    return {
        key: quantiles([
            r[key]
            for r in traces
            if r.get(key) is not None
        ])
        for key in keys
    }


def per_trajectory(path, root, traj, source):
    subgroup = (
        "tfrecord"
        if "tfrecord" in path.name.lower()
        else "scripted"
        if "scripted" in path.name.lower()
        else "other"
    )

    n = length_of_trajectory(traj)

    if n < 2:
        raise ValueError(
            "Traiettoria con meno di 2 step"
        )

    parsed = [
        extract_one_step(get_step(traj, t))
        for t in range(n)
    ]

    samples = [s["keys"] for s in parsed[:2]]

    record = {
        "source": source,
        "subgroup": subgroup,
        "path": str(path),
        "length": n,
        "action_shapes": dict(
            collections.Counter(
                str(s["action"].size)
                if s["action"] is not None
                else "none"
                for s in parsed
            )
        ),
        "num_pose": sum(
            s["pos"] is not None for s in parsed
        ),
        "num_quat": sum(
            s["quat"] is not None for s in parsed
        ),
        "num_timestamps": sum(
            s["timestamp"] is not None for s in parsed
        ),
        "gripper_observed": [
            s["gripper"]
            for s in parsed
            if s["gripper"] is not None
        ],
        "observed_actions7_gripper": [
            s["action"][6]
            for s in parsed
            if (
                s["action"] is not None
                and s["action"].size == 7
            )
        ],
        "schema": {
            "root_type": str(type(root)),
            "root_keys": sorted(
                map(str, mapping(root).keys())
            ),
            "steps": samples,
        },
    }

    traces = []

    for t in range(n - 1):
        now, nxt = parsed[t], parsed[t + 1]

        if (
            now["pos"] is None
            or nxt["pos"] is None
        ):
            continue

        dp = nxt["pos"] - now["pos"]

        entry = {
            "source": source,
            "subgroup": subgroup,
            "pkl": str(path),
            "step": t,
            "delta_pos_m": float(np.linalg.norm(dp)),
            "dx_m": float(dp[0]),
            "dy_m": float(dp[1]),
            "dz_m": float(dp[2]),
            "scaled_delta_pos": float(
                np.linalg.norm(dp / SCALE_FACTOR)
            ),
            "delta_rot_rad": None,
            "scaled_delta_rot": None,
            "droll_rad": None,
            "dpitch_rad": None,
            "dyaw_rad": None,
            "label7_pos_err_scaled": None,
            "label7_pos_err_raw": None,
            "label7_rot_err_scaled": None,
            "target8_pos_dist_current_m": None,
            "target8_pos_dist_next_m": None,
            "time_gap": None,
        }

        deuler = None

        if (
            now["quat"] is not None
            and nxt["quat"] is not None
        ):
            try:
                delta_rotation = (
                    Rotation.from_quat(nxt["quat"])
                    * Rotation.from_quat(
                        now["quat"]
                    ).inv()
                )

                deuler = delta_rotation.as_euler("xyz")

                entry.update(
                    delta_rot_rad=float(
                        np.linalg.norm(deuler)
                    ),
                    scaled_delta_rot=float(
                        np.linalg.norm(
                            deuler / SCALE_FACTOR
                        )
                    ),
                    droll_rad=float(deuler[0]),
                    dpitch_rad=float(deuler[1]),
                    dyaw_rad=float(deuler[2]),
                )

            except (ValueError, FloatingPointError):
                pass

        action = now["action"]

        if action is not None:
            if action.size == 7:
                entry["label7_pos_err_scaled"] = float(
                    np.linalg.norm(
                        action[:3]
                        - dp / SCALE_FACTOR
                    )
                )

                entry["label7_pos_err_raw"] = float(
                    np.linalg.norm(
                        action[:3] - dp
                    )
                )

                if deuler is not None:
                    entry["label7_rot_err_scaled"] = float(
                        np.linalg.norm(
                            action[3:6]
                            - deuler / SCALE_FACTOR
                        )
                    )

            elif action.size == 8:
                entry["target8_pos_dist_current_m"] = float(
                    np.linalg.norm(
                        action[:3] - now["pos"]
                    )
                )

                entry["target8_pos_dist_next_m"] = float(
                    np.linalg.norm(
                        action[:3] - nxt["pos"]
                    )
                )

        t0 = now["timestamp"]
        t1 = nxt["timestamp"]

        if (
            t0 is not None
            and t1 is not None
            and 0.0 < t1 - t0 < 3600.0
        ):
            entry["time_gap"] = float(t1 - t0)

        traces.append(entry)

    if not traces:
        raise ValueError(
            "Nessuna coppia consecutiva con eef_pos valido"
        )

    record.update({
        "transitions": len(traces),
        "delta_pos_m_p50": quantiles([
            r["delta_pos_m"]
            for r in traces
        ]).get("p50"),
        "delta_pos_m_p95": quantiles([
            r["delta_pos_m"]
            for r in traces
        ]).get("p95"),
        "delta_rot_rad_p50": quantiles([
            r["delta_rot_rad"]
            for r in traces
            if r["delta_rot_rad"] is not None
        ]).get("p50"),
        "label7_pos_err_scaled_p50": quantiles([
            r["label7_pos_err_scaled"]
            for r in traces
            if r["label7_pos_err_scaled"] is not None
        ]).get("p50"),
        "label7_pos_err_raw_p50": quantiles([
            r["label7_pos_err_raw"]
            for r in traces
            if r["label7_pos_err_raw"] is not None
        ]).get("p50"),
        "target8_pos_dist_next_p50": quantiles([
            r["target8_pos_dist_next_m"]
            for r in traces
            if r["target8_pos_dist_next_m"] is not None
        ]).get("p50"),
    })

    return record, traces


def passes_success(path):
    json_file = path.with_suffix(".json")

    if not json_file.is_file():
        return False, "missing_json"

    try:
        with json_file.open(
            "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "bad_json"

    if not isinstance(data, dict):
        return False, "json_not_object"

    missing = [
        k for k in SUCCESS_KEYS if k not in data
    ]

    if missing:
        return False, "missing_success_key"

    if all(
        data[k] == 1
        for k in SUCCESS_KEYS
    ):
        return True, "success"

    return False, "not_success"


def collect_paths(
    directory,
    is_osvi,
    rng,
    max_per_source
):
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Directory non trovata: {directory}"
        )

    all_pkl = sorted(directory.rglob("*.pkl"))
    reasons = collections.Counter()
    eligible = []

    for path in all_pkl:
        rel = path.relative_to(directory)

        if is_osvi:
            if (
                not rel.parts
                or not rel.parts[0].startswith("task_")
            ):
                reasons["outside_task_dir"] += 1
                continue

            yes, reason = passes_success(path)
            reasons[reason] += 1

            if not yes:
                continue

        eligible.append(path)

    rng.shuffle(eligible)

    chosen = (
        eligible[:max_per_source]
        if max_per_source > 0
        else eligible
    )

    return chosen, {
        "total_pkl": len(all_pkl),
        "eligible_pkl": len(eligible),
        "chosen_pkl": len(chosen),
        "filter_status": dict(reasons),
    }


def write_csv(path, rows, fields):
    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()

        for r in rows:
            writer.writerow(r)


def json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    raise TypeError(type(obj).__name__)


def fmt(
    summary,
    key,
    format_str,
    multiplier=1
):
    info = summary.get(key, {})

    if not info.get("n"):
        return "n/d"

    return format_str.format(
        info["p50"] * multiplier,
        info["p95"] * multiplier
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument(
        "--reference",
        required=True,
        type=Path
    )

    ap.add_argument(
        "--mixture",
        required=True,
        type=Path
    )

    ap.add_argument(
        "--output",
        required=True,
        type=Path
    )

    ap.add_argument(
        "--osvi-sources",
        nargs="+",
        default=[
            "osvi_awda_rollout",
            "result_osvi_awda_human"
        ]
    )

    ap.add_argument(
        "--other-sources",
        nargs="*",
        default=[
            "exploration_preprocessed",
            "plays_preprocessed"
        ],
        help=(
            "Altre sorgenti da esaminare SENZA filtro JSON "
            "(default: exploration e plays)"
        )
    )

    ap.add_argument(
        "--max-per-source",
        type=int,
        default=40,
        help=(
            "PKL analizzati per sorgente, casualmente "
            "dopo filtro successo; 0 = tutti"
        )
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = ap.parse_args()

    if args.max_per_source < 0:
        ap.error(
            "--max-per-source deve essere >= 0"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    rng = random.Random(args.seed)

    sources = [
        (
            "reference_robot",
            args.reference,
            False
        )
    ] + [
        (
            name,
            args.mixture / name,
            False
        )
        for name in args.other_sources
    ] + [
        (
            name,
            args.mixture / name,
            True
        )
        for name in args.osvi_sources
    ]

    results = {}
    all_traces = []
    all_pertraj = []
    schemas = {}

    for name, directory, is_osvi in sources:
        print(
            f"\n[{name}] {directory}",
            flush=True
        )

        if not directory.is_dir():
            print(
                "  Directory non trovata: saltata.",
                flush=True
            )
            results[name] = {
                "status": "missing_directory",
                "directory": str(directory)
            }
            continue

        paths, population = collect_paths(
            directory,
            is_osvi,
            rng,
            args.max_per_source
        )

        print(
            f"  PKL totali={population['total_pkl']} "
            f"eleggibili={population['eligible_pkl']} "
            f"selezionati={len(paths)}",
            flush=True
        )

        trace_list = []
        traj_list = []
        errors = []

        all_gripper_qpos = []
        all_gripper_action7 = []
        all_gripper_action8 = []

        for i, path in enumerate(paths, start=1):
            if i == 1 or i % 10 == 0:
                print(
                    f"  [{i}/{len(paths)}] "
                    f"{path.relative_to(directory)}",
                    flush=True
                )

            try:
                root, traj = extract_payload(path)

                record, traces = per_trajectory(
                    path,
                    root,
                    traj,
                    name
                )

                trace_list.extend(traces)

                grippers = record.pop(
                    "gripper_observed"
                )
                acts_gripper = record.pop(
                    "observed_actions7_gripper"
                )

                all_gripper_qpos.extend(grippers)
                all_gripper_action7.extend(
                    acts_gripper
                )

                if name not in schemas:
                    schemas[name] = {
                        "pkl": str(path),
                        **record.pop("schema")
                    }
                else:
                    record.pop("schema")

                record[
                    "gripper_observed_samples"
                ] = grippers[:10]

                record[
                    "gripper_action7_samples"
                ] = acts_gripper[:10]

                traj_list.append(record)

            except Exception as e:
                errors.append({
                    "pkl": str(path),
                    "error": (
                        f"{type(e).__name__}: {e}"
                    )
                })

        sumstat = summarise_values(trace_list)

        subgroups = {}

        if name == "reference_robot":
            for subgroup in sorted({
                x["subgroup"]
                for x in trace_list
            }):
                subset = [
                    x for x in trace_list
                    if x["subgroup"] == subgroup
                ]

                subgroups[subgroup] = {
                    "transitions": len(subset),
                    "statistics": summarise_values(
                        subset
                    )
                }

        result = {
            "status": (
                "ok"
                if traj_list
                else "no_valid_trajectories"
            ),
            "subgroups": subgroups,
            "directory": str(directory),
            **population,
            "processed_pkl": len(traj_list),
            "failed_pkl": len(errors),
            "total_transitions": len(trace_list),
            "gripper_qpos": quantiles(
                all_gripper_qpos
            ),
            "gripper_action7": quantiles(
                all_gripper_action7
            ),
            "gripper_action7_unique_rounded": sorted({
                round(float(g), 3)
                for g in all_gripper_action7
            })[:30],
            "action_shapes": dict(
                collections.Counter(
                    k
                    for r in traj_list
                    for k, count in r[
                        "action_shapes"
                    ].items()
                    for _ in range(count)
                )
            ),
            "trajectory_steps": quantiles([
                r["length"]
                for r in traj_list
            ]),
            "statistics": sumstat,
            "errors": errors[:30],
        }

        results[name] = result

        all_traces.extend(trace_list)
        all_pertraj.extend(traj_list)

        print(
            f"  PKL letti={len(traj_list)}, "
            f"transizioni={len(trace_list)}, "
            f"errori={len(errors)}",
            flush=True
        )

        print(
            "  Delta posizione (p50 / p95):",
            fmt(
                sumstat,
                "delta_pos_m",
                "{:.2f} / {:.2f} mm",
                1000
            ),
            flush=True
        )

        print(
            "  Delta rotazione (p50 / p95):",
            fmt(
                sumstat,
                "delta_rot_rad",
                "{:.3f} / {:.3f} gradi",
                180 / math.pi
            ),
            flush=True
        )

        print(
            "  Delta posizione /0.05 (p50 / p95):",
            fmt(
                sumstat,
                "scaled_delta_pos",
                "{:.4f} / {:.4f}"
            ),
            flush=True
        )

        print(
            "  Confronto label7 - delta scalata "
            "(p50 / p95):",
            fmt(
                sumstat,
                "label7_pos_err_scaled",
                "{:.4f} / {:.4f}"
            ),
            flush=True
        )

        print(
            "  Gap temporale (p50 / p95):",
            fmt(
                sumstat,
                "time_gap",
                "{:.3f} / {:.3f} s"
            ),
            flush=True
        )

        if errors:
            print(
                "  Primo errore:",
                errors[0]["error"],
                flush=True
            )

    write_csv(
        args.output / "per_trajectory.csv",
        all_pertraj,
        [
            "source",
            "subgroup",
            "path",
            "length",
            "transitions",
            "num_pose",
            "num_quat",
            "num_timestamps",
            "action_shapes",
            "delta_pos_m_p50",
            "delta_pos_m_p95",
            "delta_rot_rad_p50",
            "label7_pos_err_scaled_p50",
            "label7_pos_err_raw_p50",
            "target8_pos_dist_next_p50"
        ]
    )

    write_csv(
        args.output / "transitions.csv",
        all_traces,
        [
            "source",
            "subgroup",
            "pkl",
            "step",
            "delta_pos_m",
            "delta_rot_rad",
            "scaled_delta_pos",
            "scaled_delta_rot",
            "dx_m",
            "dy_m",
            "dz_m",
            "droll_rad",
            "dpitch_rad",
            "dyaw_rad",
            "label7_pos_err_scaled",
            "label7_pos_err_raw",
            "label7_rot_err_scaled",
            "target8_pos_dist_current_m",
            "target8_pos_dist_next_m",
            "time_gap"
        ]
    )

    with (
        args.output / "summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scale_factor": SCALE_FACTOR,
                "sampling_limit_per_source": (
                    args.max_per_source
                ),
                "seed": args.seed,
                "sources": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=json_serializable
        )

    with (
        args.output / "schema_examples.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            schemas,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_serializable
        )

    with (
        args.output / "README.txt"
    ).open("w", encoding="utf-8") as f:
        f.write(
            "ANALISI NON DISTRUTTIVA DELTA ACTION\n"
        )
        f.write(
            "- I file sorgente non vengono modificati.\n"
        )
        f.write(
            "- Il filtro dei rollout OSVI considera "
            "solo JSON associati con i tre success "
            "flag = 1.\n"
        )
        f.write(
            "- delta_pos_m / delta_rot_rad misurano "
            "il salto fra OSSERVAZIONI consecutive.\n"
        )
        f.write(
            "- scaled_delta_* = delta_* / 0.05, "
            "stessa convenzione candidata del "
            "reference.\n"
        )
        f.write(
            "- L'azione 7D del PKL viene confrontata "
            "con il delta ricostruito; 8D è "
            "interpretata SOLO come candidato "
            "target assoluto.\n"
        )
        f.write(
            "- I gap temporali sono disponibili solo "
            "se ci sono timestamp affidabili negli "
            "step.\n"
        )
        f.write(
            "- Differenze ampie nei rollout possono "
            "dipendere dalla frequenza di "
            "registrazione (step=inferenza), non "
            "dalla scala.\n"
        )
        f.write(
            "- Gripper non ricostruito "
            "automaticamente: unità e soglie "
            "da ispezionare per sorgente.\n"
        )
        f.write(
            "- Nel riferimento, i file sono "
            "raggruppati per nome TFRecord/scripted "
            "se il nome lo consente.\n"
        )
        f.write(
            "- Quaternioni di osservazione "
            "interpretati in ordine XYZW.\n"
        )
        f.write(
            "- Eventuali angle wrapping e step di "
            "inizio/fine possono influenzare "
            "l'errore label7.\n"
        )
        f.write(
            "- Per statistics globali su tutti i PKL "
            "usare --max-per-source 0.\n"
        )

    print(
        f"\nREPORT: {args.output}\n"
        "  summary.json\n"
        "  schema_examples.json\n"
        "  per_trajectory.csv\n"
        "  transitions.csv\n"
        "  README.txt",
        flush=True
    )


if __name__ == "__main__":
    main()
