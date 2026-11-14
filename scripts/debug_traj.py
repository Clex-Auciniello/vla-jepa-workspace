#!/usr/bin/env python3
import csv
import pickle
from pathlib import Path
import savers
import cv2
import numpy as np

PKL_PATH = Path(
    "/mnt/beegfs/a.cuciniello9/datasets/mixture/"
    "cod_rollout_finetuning/no_wrist/task_05/traj_004.pkl"
)

OUT_DIR = Path(
    "/mnt/beegfs/a.cuciniello9/datasets/"
    "cod_debug/cod_rollout_finetuning/task_05/traj_004.pkl"
)


def to_dict(x):
    if isinstance(x, dict):
        return x
    if hasattr(x, "__dict__"):
        return vars(x)
    return {}


def get_traj(payload):
    if isinstance(payload, dict) and "traj" in payload:
        return payload["traj"]
    return payload


def get_step(traj, i):
    if hasattr(traj, "get"):
        try:
            return traj.get(i, decompress=False)
        except TypeError:
            return traj.get(i)
    return traj[i]


def decode_image(img):
    if img is None:
        return None

    arr = np.asarray(img)

    # Caso immagine compressa 1D
    if arr.ndim == 1:
        dec = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return dec

    # Caso immagine HWC già decodificata
    if arr.ndim == 3:
        return arr.copy()

    return None


def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)


def save_rgb_from_bgr(bgr, path):
    cv2.imwrite(str(path), bgr)


def image_mean_abs_diff(img1_bgr, img2_bgr):
    if img1_bgr is None or img2_bgr is None:
        return None
    if img1_bgr.shape != img2_bgr.shape:
        return None
    diff = np.abs(img2_bgr.astype(np.float32) - img1_bgr.astype(np.float32))
    return float(diff.mean())


def main():
    ensure_dir(OUT_DIR)
    front_dir = OUT_DIR / "front_rgb"
    gripper_dir = OUT_DIR / "gripper_rgb"
    ensure_dir(front_dir)
    ensure_dir(gripper_dir)

    with PKL_PATH.open("rb") as f:
        payload = pickle.load(f)

    traj = get_traj(payload)
    T = len(traj)

    rows = []

    prev_front = None
    prev_gripper = None

    all_delta_pos = []
    all_front_diff = []
    all_gripper_diff = []

    for i in range(T):
        step = get_step(traj, i)
        step_d = to_dict(step)
        obs = to_dict(step_d.get("obs", {}))
        action = step_d.get("action", None)

        eef_pos = obs.get("eef_pos", None)
        eef_quat = obs.get("eef_quat", None)
        gripper_qpos = obs.get("gripper_qpos", None)

        front_bgr = decode_image(obs.get("camera_front_image", None))
        gripper_bgr = decode_image(obs.get("camera_gripper_image", None))

        if front_bgr is not None:
            save_rgb_from_bgr(front_bgr, front_dir / f"frame_{i:03d}.png")

        if gripper_bgr is not None:
            save_rgb_from_bgr(gripper_bgr, gripper_dir / f"frame_{i:03d}.png")

        front_diff = image_mean_abs_diff(prev_front, front_bgr)
        gripper_diff = image_mean_abs_diff(prev_gripper, gripper_bgr)

        if front_diff is not None:
            all_front_diff.append(front_diff)
        if gripper_diff is not None:
            all_gripper_diff.append(gripper_diff)

        prev_front = front_bgr
        prev_gripper = gripper_bgr

        delta_pos_next = None
        dx = dy = dz = None
        delta_quat_next = None

        if i < T - 1:
            next_step = get_step(traj, i + 1)
            next_obs = to_dict(to_dict(next_step).get("obs", {}))

            next_eef_pos = next_obs.get("eef_pos", None)
            next_eef_quat = next_obs.get("eef_quat", None)

            if eef_pos is not None and next_eef_pos is not None:
                p0 = np.asarray(eef_pos, dtype=np.float64).reshape(-1)
                p1 = np.asarray(next_eef_pos, dtype=np.float64).reshape(-1)
                if p0.size >= 3 and p1.size >= 3:
                    d = p1[:3] - p0[:3]
                    dx, dy, dz = map(float, d)
                    delta_pos_next = float(np.linalg.norm(d))
                    all_delta_pos.append(delta_pos_next)

            if eef_quat is not None and next_eef_quat is not None:
                q0 = np.asarray(eef_quat, dtype=np.float64).reshape(-1)
                q1 = np.asarray(next_eef_quat, dtype=np.float64).reshape(-1)
                if q0.size == 4 and q1.size == 4:
                    delta_quat_next = float(np.linalg.norm(q1 - q0))

        action_arr = None
        if action is not None:
            try:
                action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
            except Exception:
                action_arr = None

        row = {
            "step": i,
            "eef_x": None,
            "eef_y": None,
            "eef_z": None,
            "quat_x": None,
            "quat_y": None,
            "quat_z": None,
            "quat_w": None,
            "gripper_qpos": None,
            "action_dim": None,
            "action_x": None,
            "action_y": None,
            "action_z": None,
            "action_qx": None,
            "action_qy": None,
            "action_qz": None,
            "action_qw": None,
            "action_gripper": None,
            "delta_pos_to_next_m": delta_pos_next,
            "dx_to_next_m": dx,
            "dy_to_next_m": dy,
            "dz_to_next_m": dz,
            "delta_quat_l2_to_next": delta_quat_next,
            "front_mean_absdiff_from_prev": front_diff,
            "gripper_mean_absdiff_from_prev": gripper_diff,
        }

        if eef_pos is not None:
            p = np.asarray(eef_pos, dtype=np.float64).reshape(-1)
            if p.size >= 3:
                row["eef_x"], row["eef_y"], row["eef_z"] = map(float, p[:3])

        if eef_quat is not None:
            q = np.asarray(eef_quat, dtype=np.float64).reshape(-1)
            if q.size == 4:
                row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"] = map(float, q)

        if gripper_qpos is not None:
            g = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
            if g.size >= 1:
                row["gripper_qpos"] = float(g[0])

        if action_arr is not None:
            row["action_dim"] = int(action_arr.size)
            if action_arr.size >= 8:
                row["action_x"] = float(action_arr[0])
                row["action_y"] = float(action_arr[1])
                row["action_z"] = float(action_arr[2])
                row["action_qx"] = float(action_arr[3])
                row["action_qy"] = float(action_arr[4])
                row["action_qz"] = float(action_arr[5])
                row["action_qw"] = float(action_arr[6])
                row["action_gripper"] = float(action_arr[7])

        rows.append(row)

    csv_path = OUT_DIR / "debug_steps.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = []
    summary_lines.append(f"PKL: {PKL_PATH}")
    summary_lines.append(f"Num steps: {T}")

    if all_delta_pos:
        arr = np.asarray(all_delta_pos)
        summary_lines.append(f"delta_pos median [m]: {np.median(arr):.6f}")
        summary_lines.append(f"delta_pos p95 [m]: {np.quantile(arr, 0.95):.6f}")
        summary_lines.append(f"delta_pos max [m]: {arr.max():.6f}")
        summary_lines.append(f"num zero-like delta_pos (<1e-6): {(arr < 1e-6).sum()} / {arr.size}")

    if all_front_diff:
        arr = np.asarray(all_front_diff)
        summary_lines.append(f"front diff median: {np.median(arr):.6f}")
        summary_lines.append(f"front diff p95: {np.quantile(arr, 0.95):.6f}")

    if all_gripper_diff:
        arr = np.asarray(all_gripper_diff)
        summary_lines.append(f"gripper diff median: {np.median(arr):.6f}")
        summary_lines.append(f"gripper diff p95: {np.quantile(arr, 0.95):.6f}")

    summary_path = OUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved debug folder to: {OUT_DIR}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()