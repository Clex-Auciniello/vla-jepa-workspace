import csv
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import savers  # Necessario per leggere savers.Trajectory


TF_ROOT = Path(os.environ["TF_ROOT"])
SCRIPTED_ROOT = Path(os.environ["SCRIPTED_ROOT"])
OUT = Path(os.environ["REPORT_DIR"])

OUT.mkdir(parents=True, exist_ok=True)

# Corrispondenza shard -> task:
# 00000 -> 01, 00001 -> 02, ... saltando 05, 10, 15.
# Il task 00 è già escluso dai TFRecord.
TRAIN_TASKS = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]

episodes = []
all_steps = []


def decode_feature(example, key, dims):
    features = example.features.feature

    if key not in features:
        candidates = [
            name for name in features
            if "eef" in name.lower()
        ]
        raise KeyError(
            f"Feature {key} assente. Candidati: {candidates}"
        )

    feature = features[key]
    kind = feature.WhichOneof("kind")

    if kind == "float_list":
        values = np.asarray(
            feature.float_list.value,
            dtype=np.float64,
        )

    elif kind == "int64_list":
        values = np.asarray(
            feature.int64_list.value,
            dtype=np.float64,
        )

    elif kind == "bytes_list":
        arrays = []

        for blob in feature.bytes_list.value:
            if len(blob) == dims * 4:
                arr = np.frombuffer(blob, dtype="<f4")

            elif len(blob) == dims * 8:
                arr = np.frombuffer(blob, dtype="<f8")

            else:
                arr = None

                for dtype in (tf.float32, tf.float64):
                    try:
                        parsed = tf.io.parse_tensor(
                            blob, out_type=dtype
                        ).numpy()

                        if parsed.size % dims == 0:
                            arr = parsed.reshape(-1)
                            break
                    except (tf.errors.InvalidArgumentError, ValueError):
                        pass

                if arr is None and len(blob) % (dims * 4) == 0:
                    arr = np.frombuffer(blob, dtype="<f4")

                if arr is None and len(blob) % (dims * 8) == 0:
                    arr = np.frombuffer(blob, dtype="<f8")

                if arr is None:
                    raise ValueError(
                        f"Codifica non riconosciuta: {key}, "
                        f"blob size={len(blob)}"
                    )

            arrays.append(np.asarray(arr, dtype=np.float64))

        if not arrays:
            raise ValueError(f"Feature vuota: {key}")

        values = np.concatenate(arrays)

    else:
        raise ValueError(
            f"Tipo feature non supportato: {key}, kind={kind}"
        )

    if values.size % dims != 0:
        raise ValueError(
            f"Dimensione non valida: {key}, size={values.size}"
        )

    return values.reshape(-1, dims)


def tfrecord_positions(record):
    example = tf.train.Example.FromString(record.numpy())

    # EEF_state = [x, y, z, roll, pitch, yaw]
    state = decode_feature(
        example,
        "steps/observation/EEF_state",
        6,
    )

    return state[:, :3]


def scripted_positions(path):
    with path.open("rb") as f:
        data = pickle.load(f)

    traj = data["traj"]

    positions = []

    for index in range(len(traj)):
        try:
            step = traj.get(index, decompress=False)
        except TypeError:
            step = traj.get(index)

        pos = np.asarray(
            step["obs"]["eef_pos"],
            dtype=np.float64,
        ).reshape(-1)

        positions.append(pos[:3])

    return np.asarray(positions)


def register_episode(source, task, episode, positions):
    positions = np.asarray(positions, dtype=np.float64)

    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or len(positions) < 2
        or not np.isfinite(positions).all()
    ):
        raise ValueError(
            f"Posizioni non valide: shape={positions.shape}"
        )

    # Distanza euclidea fra frame consecutivi.
    steps_m = np.linalg.norm(
        np.diff(positions, axis=0),
        axis=1,
    )

    episodes.append({
        "source": source,
        "task": task,
        "episode": episode,
        "frames": len(positions),
        "path_cm": float(steps_m.sum() * 100),
        "median_step_cm": float(np.median(steps_m) * 100),
    })

    for t, distance in enumerate(steps_m):
        all_steps.append({
            "source": source,
            "task": task,
            "episode": episode,
            "transition": t,
            "step_cm": float(distance * 100),
        })


print("\n========== TFRECORD ==========", flush=True)

shards = sorted(
    TF_ROOT.glob("ur5e_pick_place-train.tfrecord-*")
)

if len(shards) != 12:
    raise RuntimeError(
        f"Attesi 12 TFRecord, trovati {len(shards)}"
    )

for shard_index, (path, task) in enumerate(
    zip(shards, TRAIN_TASKS)
):
    count = 0

    dataset = tf.data.TFRecordDataset(str(path))

    for episode_index, record in enumerate(dataset):
        try:
            positions = tfrecord_positions(record)

            register_episode(
                "tfrecord",
                task,
                f"{shard_index:05d}_{episode_index:03d}",
                positions,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Errore {path.name}, episodio {episode_index}: {exc}"
            ) from exc

        count += 1

    print(
        f"{path.name} -> task_{task:02d}: "
        f"{count} episodi",
        flush=True,
    )

    if count != 40:
        print(f"ATTENZIONE: attesi 40 episodi, trovati {count}")


print("\n========== SCRIPTED ==========", flush=True)

for task in TRAIN_TASKS:
    task_dir = SCRIPTED_ROOT / f"task_{task:02d}"

    files = sorted(task_dir.glob("*.pkl"))

    if len(files) != 20:
        print(
            f"ATTENZIONE task_{task:02d}: "
            f"attesi 20 PKL, trovati {len(files)}"
        )

    for path in files:
        try:
            positions = scripted_positions(path)

            register_episode(
                "scripted",
                task,
                path.stem,
                positions,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Errore {path}: {exc}"
            ) from exc

    print(
        f"task_{task:02d}: {len(files)} episodi",
        flush=True,
    )


def save_csv(path, rows, columns):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


save_csv(
    OUT / "motion_per_transition.csv",
    all_steps,
    ["source", "task", "episode", "transition", "step_cm"],
)

save_csv(
    OUT / "motion_per_episode.csv",
    episodes,
    [
        "source", "task", "episode",
        "frames", "path_cm", "median_step_cm",
    ],
)


summary = []

print("\n========== CONFRONTO PER TASK ==========")
print(
    "SOURCE    TASK   N   FRAMES  "
    "MEAN(cm)  MEDIAN(cm)  P95(cm)  "
    "MAX(cm)  STATIC(%)  PATH(cm)"
)

for source in ("tfrecord", "scripted"):
    for task in TRAIN_TASKS:
        ep = [
            row for row in episodes
            if row["source"] == source and row["task"] == task
        ]

        distances = np.asarray([
            row["step_cm"]
            for row in all_steps
            if row["source"] == source and row["task"] == task
        ])

        if not ep or not len(distances):
            continue

        result = {
            "source": source,
            "task": task,
            "n_episodes": len(ep),
            "frames_mean": float(np.mean([
                row["frames"] for row in ep
            ])),
            "step_mean_cm": float(np.mean(distances)),
            "step_median_cm": float(np.median(distances)),
            "step_p95_cm": float(np.percentile(distances, 95)),
            "step_max_cm": float(np.max(distances)),
            "static_pct": float(np.mean(distances < 0.1) * 100),
            "path_mean_cm": float(np.mean([
                row["path_cm"] for row in ep
            ])),
        }

        summary.append(result)

        print(
            f"{source:9} {task:02d} "
            f"{result['n_episodes']:4d} "
            f"{result['frames_mean']:7.1f} "
            f"{result['step_mean_cm']:9.3f} "
            f"{result['step_median_cm']:11.3f} "
            f"{result['step_p95_cm']:8.3f} "
            f"{result['step_max_cm']:8.3f} "
            f"{result['static_pct']:9.2f} "
            f"{result['path_mean_cm']:9.2f}"
        )


save_csv(
    OUT / "motion_per_task.csv",
    summary,
    [
        "source", "task", "n_episodes",
        "frames_mean", "step_mean_cm", "step_median_cm",
        "step_p95_cm", "step_max_cm",
        "static_pct", "path_mean_cm",
    ],
)

print("\nReport salvati in:", OUT)
