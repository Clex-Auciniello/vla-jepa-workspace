
#!/usr/bin/env python3
"""Audit non distruttivo delle soglie gripper nei dataset OSVI clean."""

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import savers

BASE = Path("/mnt/beegfs/a.cuciniello9/datasets")

DATASETS = {
    "cod_epoch_100_wrist_clean":
        BASE / "mixture/cod_epoch_100_wrist_clean",

    "cod_finetuning_no_wrist_clean":
        BASE / "mixture/cod_rollout_finetuning/no_wrist_clean",
}

OUTPUT = BASE / "cod_gripper_threshold_audit"

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.35, 0.38,
    0.40, 0.41, 0.42, 0.425,
    0.43, 0.45, 0.50, 0.60, 0.70,
]

SHOW_STATE_THRESHOLDS = [0.30, 0.38, 0.42, 0.425]

# True: genera anche un grafico PNG per ogni traiettoria.
MAKE_PLOTS = False

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


def read_trajectory(path):
    json_path = path.with_suffix(".json")

    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if (
        not isinstance(metadata, dict)
        or not all(
            metadata.get(k) == 1
            for k in SUCCESS_KEYS
        )
    ):
        raise ValueError(
            f"JSON non 1/1/1: {path}"
        )

    with path.open("rb") as f:
        payload = pickle.load(f)

    traj = payload["traj"]

    if len(traj) < 2:
        raise ValueError(
            f"Traiettoria troppo corta: {path}"
        )

    qpos_values = []
    command_values = []

    for t in range(len(traj)):
        step = get_step(traj, t)
        obs = step["obs"]

        qpos = np.asarray(
            obs["gripper_qpos"],
            dtype=np.float64
        ).reshape(-1)

        action = np.asarray(
            step["action"],
            dtype=np.float64
        ).reshape(-1)

        if (
            qpos.size != 1
            or not np.isfinite(qpos[0])
        ):
            raise ValueError(
                f"gripper_qpos invalido: {path}, step {t}"
            )

        if (
            action.size != 8
            or not np.isfinite(action[7])
        ):
            raise ValueError(
                f"Action OSVI non 8D: {path}, step {t}"
            )

        qpos_values.append(float(qpos[0]))
        command_values.append(float(action[7]))

    return (
        np.asarray(qpos_values),
        np.asarray(command_values),
    )


def command_name(value):
    if abs(value) < 0.5:
        return "OPEN"

    if abs(value - 255.0) < 0.5:
        return "CLOSE"

    return "OTHER"


def classify(binary, final_command):
    changes = (
        np.flatnonzero(np.diff(binary) != 0) + 1
    )

    rises = int(np.count_nonzero(
        (binary[:-1] == 0)
        & (binary[1:] == 1)
    ))

    falls = int(np.count_nonzero(
        (binary[:-1] == 1)
        & (binary[1:] == 0)
    ))

    if binary[0] == 1:
        status = "START_CLOSED"

    elif (
        rises == 1
        and falls == 1
        and len(changes) == 2
        and binary[-1] == 0
    ):
        status = "OBS_010"

    elif (
        rises == 1
        and falls == 0
        and len(changes) == 1
    ):
        if final_command == "OPEN":
            status = "OBS_01_TERMINAL_OPEN_CMD"
        else:
            status = "OBS_01_NO_TERMINAL_OPEN_CMD"

    elif len(changes) == 0:
        status = "ALWAYS_OPEN"

    else:
        status = "IRREGULAR"

    return (
        status,
        changes.tolist(),
        rises,
        falls,
    )


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectory(source, rel, qpos, commands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(qpos))

    fig, (ax0, ax1) = plt.subplots(
        2, 1,
        figsize=(11, 5.5),
        sharex=True,
    )

    ax0.plot(
        t, qpos,
        "k.-",
        markersize=2,
        linewidth=1,
        label="gripper_qpos osservato",
    )

    for threshold in SHOW_STATE_THRESHOLDS:
        ax0.axhline(
            threshold,
            linestyle="--",
            linewidth=1,
            label=f"soglia {threshold:g}",
        )

    ax0.set_ylabel("gripper_qpos")
    ax0.legend(fontsize=8, ncol=3)

    for n, threshold in enumerate(
        SHOW_STATE_THRESHOLDS
    ):
        binary = (
            qpos >= threshold
        ).astype(int)

        ax1.step(
            t,
            binary + 1.25 * n,
            where="post",
            linewidth=1,
            label=f"{threshold:g}",
        )

    ax1.set_yticks(
        [
            1.25 * n + 0.5
            for n in range(len(SHOW_STATE_THRESHOLDS))
        ],
        [
            f"{threshold:g}"
            for threshold in SHOW_STATE_THRESHOLDS
        ],
    )

    ax1.set_ylabel("Soglia (binario 0/1)")
    ax1.set_xlabel("Step")
    ax1.grid(alpha=0.2)

    fig.suptitle(
        f"{source} / {rel.as_posix()}"
        f" | comando finale={command_name(commands[-1])}"
    )

    fig.tight_layout()

    target = (
        OUTPUT / "plots" / source
        / rel.with_suffix(".png")
    )

    target.parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(target, dpi=120)
    plt.close(fig)


def main():
    OUTPUT.mkdir(
        parents=True, exist_ok=True
    )

    details = []
    step_rows = []
    frequencies = []
    summaries = []

    for source, root in DATASETS.items():

        if not root.is_dir():
            raise FileNotFoundError(root)

        paths = sorted(root.glob("task_*/traj_*.pkl"))

        if not paths:
            raise RuntimeError(
                f"Nessun PKL: {root}"
            )

        counters = defaultdict(Counter)
        observed_values = Counter()

        print(
            f"\n[{source}] {len(paths)} PKL",
            flush=True,
        )

        for index, path in enumerate(paths, 1):

            if index == 1 or index % 20 == 0:
                print(
                    f"  {index}/{len(paths)} "
                    f"{path.relative_to(root)}",
                    flush=True,
                )

            qpos, commands = read_trajectory(path)

            rel = path.relative_to(root)

            final_command = command_name(
                commands[-1]
            )

            command_labels = [
                command_name(x)
                for x in commands
            ]

            # Distribuzione dei valori grezzi.
            for value in qpos:
                observed_values[
                    round(float(value), 6)
                ] += 1

            # CSV per-step: dati grezzi e alcune soglie.
            for t in range(len(qpos)):
                row = {
                    "source": source,
                    "pkl": str(rel),
                    "step": t,
                    "gripper_qpos": float(qpos[t]),
                    "action_gripper": float(commands[t]),
                    "command": command_labels[t],
                }

                for threshold in SHOW_STATE_THRESHOLDS:
                    row[
                        f"closed_th_{threshold:g}"
                    ] = int(qpos[t] >= threshold)

                step_rows.append(row)

            # Valutazione di tutte le soglie.
            for threshold in THRESHOLDS:

                binary = (
                    qpos >= threshold
                ).astype(np.int8)

                (
                    status,
                    change_steps,
                    rises,
                    falls,
                ) = classify(
                    binary, final_command
                )

                counters[threshold][status] += 1

                details.append({
                    "source": source,
                    "pkl": str(rel),
                    "threshold": threshold,
                    "n_steps": len(qpos),
                    "qpos_min": float(qpos.min()),
                    "qpos_initial": float(qpos[0]),
                    "qpos_final": float(qpos[-1]),
                    "qpos_max": float(qpos.max()),
                    "initial_binary": int(binary[0]),
                    "final_binary": int(binary[-1]),
                    "num_close_transitions": rises,
                    "num_open_transitions": falls,
                    "change_steps": ",".join(
                        map(str, change_steps)
                    ),
                    "status": status,
                    "final_command": final_command,
                })

            if MAKE_PLOTS:
                plot_trajectory(
                    source, rel, qpos, commands
                )

        # Frequenze di tutti i valori osservati.
        for value, count in sorted(
            observed_values.items()
        ):
            frequencies.append({
                "source": source,
                "qpos_rounded_6dp": value,
                "count": count,
            })

        # Riepilogo aggregato per soglia.
        for threshold in THRESHOLDS:

            c = counters[threshold]

            valid = (
                c["OBS_010"]
                + c["OBS_01_TERMINAL_OPEN_CMD"]
            )

            summaries.append({
                "source": source,
                "threshold": threshold,
                "trajectories": len(paths),
                "OBS_010": c["OBS_010"],
                "OBS_01_TERMINAL_OPEN_CMD":
                    c["OBS_01_TERMINAL_OPEN_CMD"],
                "valid_total": valid,
                "START_CLOSED": c["START_CLOSED"],
                "ALWAYS_OPEN": c["ALWAYS_OPEN"],
                "OBS_01_NO_TERMINAL_OPEN_CMD":
                    c["OBS_01_NO_TERMINAL_OPEN_CMD"],
                "IRREGULAR": c["IRREGULAR"],
            })

        print(
            "\nSoglia | OBS_010 | OBS_01 + open cmd "
            "| validi | altri"
        )

        for row in summaries:
            if row["source"] != source:
                continue

            print(
                f"{row['threshold']:6.3f} | "
                f"{row['OBS_010']:7d} | "
                f"{row['OBS_01_TERMINAL_OPEN_CMD']:17d} | "
                f"{row['valid_total']:6d} | "
                f"{len(paths) - row['valid_total']:5d}"
            )

    write_csv(
        OUTPUT / "per_trajectory_threshold.csv",
        details,
    )

    write_csv(
        OUTPUT / "summary_thresholds.csv",
        summaries,
    )

    write_csv(
        OUTPUT / "per_step_gripper.csv",
        step_rows,
    )

    write_csv(
        OUTPUT / "qpos_value_frequencies.csv",
        frequencies,
    )

    print("\nOutput:", OUTPUT)
    print(" - summary_thresholds.csv")
    print(" - per_trajectory_threshold.csv")
    print(" - per_step_gripper.csv")
    print(" - qpos_value_frequencies.csv")

    if MAKE_PLOTS:
        print(" - plots/<dataset>/task_XX/traj_YYY.png")


if __name__ == "__main__":
    main()
