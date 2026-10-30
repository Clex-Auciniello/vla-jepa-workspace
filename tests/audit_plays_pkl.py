#!/usr/bin/env python3
"""Audit non distruttivo di un PKL Plays + anteprima ESATTA del batch builder.

Eseguire nell'ambiente in cui sono importabili:
  - savers (dataset_collector_pkg/scripts), vla_jepa_online, torch, numpy, cv2, PIL.

Il codice non comunica con il robot, non fa POST, non carica il World Model
ed esegue solo le funzioni di preprocessing del batch builder installato.
ATTENZIONE: aprire soltanto PKL fidati: pickle esegue codice durante la lettura.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CAMERAS = {
    "front": "front_camera_image",
    "left": "left_camera_image",
    "right": "right_camera_image",
    "gripper": "gripper_camera_image",
}
STATE_SHAPES = {
    "joint_pos": (6,), "joint_vel": (6,),
    "gripper_qpos": (1,), "gripper_qvel": (1,),
    "eef_pos": (3,), "eef_quat": (4,),
}
REQUIRED_META = (
    "schema_version", "episode_id", "source", "sampling_mode",
    "prompt_mode", "terminal_padding", "episode_prompt", "play_type",
    "object_color", "operator_success", "operator_approved",
)


def as_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (tuple, list)):
        return [as_list(v) for v in x]
    if isinstance(x, dict):
        return {str(k): as_list(v) for k, v in x.items()}
    return x


def text_msg(msg: Any) -> str:
    return "" if msg is None else str(msg)


def add(results: dict, level: str, message: str) -> None:
    results[level].append(message)
    print(f"[{level.upper():5s}] {message}")


def validate_meta(meta: dict, payload: dict, pkl: Path, results: dict) -> None:
    for key in REQUIRED_META:
        if key not in meta:
            add(results, "errors", f"Metadato mancante: {key}")
    expected = {
        "schema_version": 1,
        "source": "plays",
        "sampling_mode": "motion_step",
        "prompt_mode": "episode_prompt",
        "terminal_padding": "repeat_last",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            add(results, "errors", f"{key}: trovato {meta.get(key)!r}, atteso {value!r}")
    if meta.get("episode_id") != pkl.stem or not re.fullmatch(r"P\d{5,}", pkl.stem):
        add(results, "errors", f"ID incoerente: nome={pkl.stem!r}, metadato={meta.get('episode_id')!r}")
    if not isinstance(meta.get("episode_prompt"), str) or not meta["episode_prompt"].strip():
        add(results, "errors", "episode_prompt mancante, vuoto o non-stringa")
    if meta.get("operator_success") is not True or meta.get("operator_approved") is not True:
        add(results, "errors", "Play non approvato/con esito non positivo: non va addestrato")
    for key, value in meta.items():
        if key not in payload or payload[key] != value:
            add(results, "errors", f"Metadato {key!r} incoerente fra config_str e top-level")
    for key in ("pick_uv", "place_uv"):
        if key in meta and (not isinstance(meta[key], list) or len(meta[key]) != 2):
            add(results, "errors", f"{key} non è una coppia di pixel")
    for key in ("pick_xyz", "place_xyz"):
        if key in meta:
            try:
                xyz = np.asarray(meta[key], dtype=np.float64)
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError()
            except (TypeError, ValueError):
                add(results, "errors", f"{key} non è un vettore XYZ valido")


def validate_frame(step: dict, index: int, last_index: int, results: dict, stamps_last: dict,
                   sizes: dict, frame_table: list[dict],
                   contact_images: dict[int, dict[str, np.ndarray]], sample_ids: set[int]):
    err = lambda message: add(results, "errors", f"Frame {index:03d}: {message}")
    warn = lambda message: add(results, "warnings", f"Frame {index:03d}: {message}")
    row: dict[str, Any] = {"frame": index}
    obs, info = step.get("obs"), step.get("info")
    if not isinstance(obs, dict) or not isinstance(info, dict):
        err("obs/info non sono dizionari")
        frame_table.append(row)
        return
    if info.get("index") != index:
        err(f"info.index={info.get('index')!r}, atteso {index}")
    row.update({"event": info.get("event"), "segment_id": info.get("segment_id"),
                "motion_label": info.get("motion_label")})
    if not isinstance(info.get("event"), str) or not info["event"].strip():
        warn("event assente o vuoto")
    if info.get("motion_label") is not None and (not isinstance(info["motion_label"], str)
                                                  or not info["motion_label"].strip()):
        warn("motion_label non valido (per Plays non influenza episode_prompt)")
    if step.get("done") is not None and bool(step["done"]) != (index == last_index):
        err(f"done={step['done']!r}: atteso True soltanto sul frame finale")

    for cam, key in CAMERAS.items():
        val = obs.get(key)
        if val is None:
            err(f"camera assente: {key}")
            continue
        img = np.asarray(val)
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            err(f"{cam}: shape={img.shape}, dtype={img.dtype}; atteso [H,W,3] uint8 BGR")
            continue
        h, w = img.shape[:2]
        row[f"{cam}_shape"] = f"{w}x{h}"
        row[f"{cam}_min"] = int(img.min())
        row[f"{cam}_max"] = int(img.max())
        row[f"{cam}_std"] = round(float(img.std()), 2)
        row[f"{cam}_bgr_mean"] = "/".join(f"{x:.1f}" for x in img.mean(axis=(0, 1)))
        sizes.setdefault(cam, set()).add((w, h))
        if w < 240 or h < 240:
            warn(f"{cam}: risoluzione sospetta {w}x{h}")
        if cam == "front" and (w <= 230 or h <= 10):
            err(f"front {w}x{h}: crop builder (left=140,right=90,bottom=10) impossibile")
        if img.max() == img.min():
            warn(f"{cam}: immagine uniforme (min=max={int(img.min())})")
        elif float(img.std()) < 4:
            warn(f"{cam}: immagine quasi uniforme (std={float(img.std()):.2f})")
        if index in sample_ids:
            contact_images.setdefault(index, {})[cam] = img.copy()

    for key in ("front_stamp_ns", "gripper_stamp_ns", "left_stamp_ns", "right_stamp_ns"):
        stamp = info.get(key)
        if not isinstance(stamp, (int, np.integer)) or int(stamp) <= 0:
            err(f"{key} assente/non valido: {stamp!r}")
            continue
        stamp = int(stamp)
        row[key] = stamp
        if key in stamps_last and stamp <= stamps_last[key]:
            warn(f"{key} non crescente: {stamp} <= {stamps_last[key]}")
        stamps_last[key] = stamp
    if all(key in row for key in ("front_stamp_ns", "gripper_stamp_ns")):
        sync_ms = abs(row["front_stamp_ns"] - row["gripper_stamp_ns"]) / 1e6
        row["front_gripper_sync_ms"] = round(sync_ms, 2)
        if sync_ms > 100:
            warn(f"front/gripper non bene sincronizzati ({sync_ms:.1f} ms > 100 ms)")

    try:
        action = np.asarray(step.get("action"), dtype=np.float64)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError(f"shape={action.shape}, finite={np.isfinite(action).all()}")
        for name, value in zip(("a_x", "a_y", "a_z", "a_qx", "a_qy", "a_qz", "a_qw", "a_gripper"), action):
            row[name] = round(float(value), 7)
        if abs(float(np.linalg.norm(action[3:7])) - 1.0) > 0.03:
            warn(f"quaternion action non unitario: norm={np.linalg.norm(action[3:7]):.3f}")
        if not np.isclose(action[7], 0) and not np.isclose(action[7], 1):
            err(f"gripper action non binaria: {action[7]}")
        commanded = np.asarray(info.get("commanded_action"), dtype=np.float64)
        if commanded.shape != (8,) or not np.array_equal(action, commanded):
            err("step.action differisce da info.commanded_action")
    except (ValueError, TypeError) as exc:
        err(f"action non valida: {exc}")

    state = info.get("robot_state")
    if not isinstance(state, dict):
        err("info.robot_state assente/non-dizionario")
        state = {}
    for name, shape in STATE_SHAPES.items():
        try:
            a = np.asarray(obs.get(name), dtype=np.float64)
            b = np.asarray(state.get(name), dtype=np.float64)
            if a.shape != shape or b.shape != shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                err(f"{name}: shape/valori non validi obs={a.shape}, info={b.shape}, atteso={shape}")
                continue
            if not np.array_equal(a, b):
                err(f"{name}: valore in obs e robot_state discordante")
            if name == "eef_quat" and abs(float(np.linalg.norm(a)) - 1.0) > 0.03:
                warn(f"quaternion EEF non unitario: norm={np.linalg.norm(a):.3f}")
            row[f"{name}"] = ",".join(f"{v:.5f}" for v in a)
        except (ValueError, TypeError) as exc:
            err(f"{name}: lettura fallita ({exc})")
    frame_table.append(row)


def fit_tile(bgr: np.ndarray, width=340, height=190) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(width / w, height / h)
    img = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    tile = np.full((height, width, 3), 245, np.uint8)
    y, x = (height - img.shape[0]) // 2, (width - img.shape[1]) // 2
    tile[y:y + img.shape[0], x:x + img.shape[1]] = img
    return tile


def make_raw_contact(images: dict[int, dict[str, np.ndarray]], output: Path,
                     front_crop: Any) -> None:
    if not images:
        return
    strip = []
    for frame_id, cameras in sorted(images.items()):
        tiles = []
        for cam in CAMERAS:
            if cam not in cameras:
                tile = np.full((225, 340, 3), 220, np.uint8)
                cv2.putText(tile, f"{cam}: MISSING", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 200), 2)
            else:
                img = cameras[cam].copy()
                if cam == "front" and front_crop is not None:
                    h, w = img.shape[:2]
                    cv2.rectangle(img, (front_crop.left, front_crop.top),
                                  (w - front_crop.right - 1, h - front_crop.bottom - 1),
                                  (20, 220, 20), max(2, w // 500))
                tile = np.full((225, 340, 3), 245, np.uint8)
                tile[30:] = fit_tile(img, 340, 195)
                cv2.putText(tile, f"frame={frame_id} {cam} BGR raw", (7, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, .53, (35, 35, 35), 1, cv2.LINE_AA)
            tiles.append(tile)
        strip.append(cv2.hconcat(tiles))
    cv2.imwrite(str(output), cv2.vconcat(strip), [cv2.IMWRITE_JPEG_QUALITY, 88])


def make_window_image(front, wrist, frame_indices, dest: Path) -> None:
    """From REAL tensors: [1,8,3,H,W], RGB normalized [0,1]."""
    rows = []
    for label, tensor in (("FRONT", front), ("GRIPPER", wrist)):
        tiles = []
        for j, frame_idx in enumerate(frame_indices):
            rgb = np.clip(tensor[0, j].detach().cpu().permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            tile = np.full((h + 29, w, 3), 245, dtype=np.uint8)
            tile[29:] = bgr
            cv2.putText(tile, f"{label} #{frame_idx}", (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (35, 35, 35), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(cv2.hconcat(tiles))
    cv2.imwrite(str(dest), cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_html(output: Path, result: dict, raw_filename: str | None, window_rows: list[dict]) -> None:
    esc = html.escape
    title = esc(result["episode_id"])
    prompt = esc(text_msg(result["metadata"].get("episode_prompt")))
    status = "OK" if not result["errors"] else f"ERRORI: {len(result['errors'])}"
    warnings = "".join(f"<li>{esc(w)}</li>" for w in result["warnings"][:300])
    errors = "".join(f"<li>{esc(w)}</li>" for w in result["errors"][:300])
    raw = f'<img src="{raw_filename}" style="width:100%" alt="4 cam originali">' if raw_filename else "Immagini raw non esportate"
    windows = "\n".join(
        f'<details {"open" if i == 0 else ""}><summary>Finestra {r["window_index"]} — '
        f'frame {esc(str(r["first_frame"]))}–{esc(str(r["last_frame"]))}</summary>'
        f'<p><code>task[0] = {esc(json.dumps(r["prompt"], ensure_ascii=False))}</code></p>'
        f'<img src="{esc(r["preview"])}" style="width:100%" alt="tensori preprocessati"></details>'
        for i, r in enumerate(window_rows))
    doc = f'''<!doctype html><html lang="it"><head><meta charset="utf-8"><title>Audit {title}</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:20px auto;max-width:1400px;background:#f8fafc;color:#182434}}
section,details{{border:1px solid #dce1e8;background:white;border-radius:10px;padding:14px;margin:12px 0}}
summary{{font-weight:650;cursor:pointer}} code{{white-space:pre-wrap;overflow-wrap:anywhere}} img{{border-radius:6px}} 
.pill{{background:#eef3fa;padding:6px 9px;border-radius:5px}} li{{margin-bottom:6px}}</style></head><body>
<h1>Audit PKL Plays: {title}</h1><p class="pill"><b>{esc(status)}</b> · frame: {result['frame_count']} · finestre: {len(window_rows)} · warnings: {len(result['warnings'])}</p>
<section><h2>Prompt Plays unico (usato in ogni finestra)</h2><p><code>{prompt}</code></p></section>
<section><h2>Quattro camere nel PKL (BGR originali)</h2><p>La cornice verde sulla camera frontale indica l'area che sopravvive al crop del batch builder.</p>{raw}</section>
<section><h2>Finestre reali del batch builder (RGB, crop+resize 224)</h2>
<p>Ogni immagine sotto proviene direttamente dai tensori elaborati da <code>OnlineBatchBuilder.iter_episode_windows</code>. I primi 8 frame sono frontale, i secondi 8 gripper. Action e robot_state non entrano nel batch WM.</p>{windows}</section>
<section><h2>Errori</h2><ul>{errors or '<li>Nessuno</li>'}</ul><h2>Avvertimenti</h2><ul>{warnings or '<li>Nessuno</li>'}</ul></section>
<section><h2>Output completi</h2><p><code>audit.json</code>, <code>frames.csv</code>, <code>windows.csv</code>, <code>metadata.json</code>.</p></section>
</body></html>'''
    (output / "index.html").write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True, help="Percorso P00001.pkl (fidato)")
    parser.add_argument("--out", type=Path, default=None, help="Directory audit; default: accanto al PKL")
    parser.add_argument("--dataset-scripts", type=Path, default=None,
                        help="Directory contenente savers.py, se non già nel PYTHONPATH")
    parser.add_argument("--repo-src", type=Path, default=None,
                        help="Directory src contenente vla_jepa_online, se necessario")
    parser.add_argument("--preview-frames", type=int, default=6,
                        help="Numero di frame da mostrare nel montaggio RAW (default 6)")
    args = parser.parse_args()
    pkl = args.pkl.expanduser().resolve()
    if not pkl.is_file():
        parser.error(f"PKL non trovato: {pkl}")
    for directory in (args.dataset_scripts, args.repo_src):
        if directory:
            if not directory.is_dir():
                parser.error(f"Directory non trovata: {directory}")
            sys.path.insert(0, str(directory.resolve()))
    out = (args.out or pkl.parent / f"audit_{pkl.stem}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "episode_id": pkl.stem, "pkl_path": str(pkl),
        "errors": [], "warnings": [], "metadata": {}, "frame_count": 0,
        "batch_builder": {},
    }
    rows: list[dict] = []
    windows: list[dict] = []
    raw_images: dict[int, dict[str, np.ndarray]] = {}
    front_crop = None
    try:
        from vla_jepa_online.exploration.trajectory_saver import _trajectory_type, TrajectorySaver
        from vla_jepa_online.plays.plays_transport import PlaysTrainingTransport
        from vla_jepa_online.training_server.online_batch_builder import OnlineBatchBuilder

        print(f"\nPKL: {pkl}\nDimensione: {pkl.stat().st_size / (1024*1024):.2f} MiB")
        Trajectory = _trajectory_type()  # Imposta path di savers come nella pipeline originale.
        with pkl.open("rb") as f:
            payload = pickle.load(f)  # SOLO file fidati!
        if not isinstance(payload, dict) or not isinstance(payload.get("traj"), Trajectory):
            raise ValueError("Struttura errata: atteso dict con 'traj' di tipo savers.Trajectory")
        traj = payload["traj"]
        meta = json.loads(traj.config_str)
        if not isinstance(meta, dict):
            raise ValueError("traj.config_str deve essere JSON dict")
        results["metadata"] = as_list(meta)
        validate_meta(meta, payload, pkl, results)
        # Riproduce il prompt a partire dal catalogo Plays corrente.
        # Se il catalogo è cambiato dopo la registrazione, segnala la differenza
        # senza alterare il prompt realmente salvato nel PKL e inviato al modello.
        try:
            from vla_jepa_online.plays.plays import PlayRequest, semantic_prompt
            req = PlayRequest(
                task_id=str(meta["play_type"]), color=str(meta["object_color"]),
                target_color=meta.get("target_color"),
                pick_uv=tuple(meta["pick_uv"]), place_uv=tuple(meta["place_uv"]),
                pick_xyz=tuple(meta["pick_xyz"]), place_xyz=tuple(meta["place_xyz"]),
                pick_y_offset_m=float(meta["pick_y_offset_m"]),
                pick_x_offset_m=float(meta.get("pick_x_offset_m", 0.0)),
                rotation_deg=meta.get("rotation_deg"),
            )
            reconstructed_prompt = semantic_prompt(req)
            results["reconstructed_semantic_prompt"] = reconstructed_prompt
            if meta.get("episode_prompt") != reconstructed_prompt:
                add(results, "warnings", "Il prompt ricostruito con plays.py corrente differisce "
                    "da episode_prompt del PKL; quello salvato nel PKL è comunque il prompt passato al WM")
            else:
                print("[OK   ] Prompt semantico coerente con metadati + plays.py corrente")
        except Exception as exc:
            add(results, "warnings", f"Non posso rigenerare il prompt con plays.py: {type(exc).__name__}: {exc}")
        n = len(traj)
        results["frame_count"] = n
        if n < 8:
            add(results, "errors", f"Episodio troppo corto: {n} frame (minimo 8)")
        max_images = max(1, args.preview_frames)
        selected = sorted(set(np.linspace(1, max(n, 1), min(max_images, n), dtype=int).tolist())) if n else []
        selected_set = set(selected)
        sizes: dict[str, set] = {}
        stamps_last: dict[str, int] = {}
        for idx in range(n):
            step = traj.get(idx, decompress=False)
            if not isinstance(step, dict):
                add(results, "errors", f"Frame {idx + 1}: step non è un dict")
                continue
            validate_frame(step, idx + 1, n, results, stamps_last, sizes, rows,
                           raw_images, selected_set)
        for cam, dims in sizes.items():
            if len(dims) > 1:
                add(results, "errors", f"Camera {cam}: dimensioni non costanti tra frame: {sorted(dims)}")
        results["raw_camera_sizes"] = {k: sorted(list(v)) for k, v in sizes.items()}
        for cam in CAMERAS:
            if cam not in sizes:
                add(results, "errors", f"Nessuna immagine valida per camera {cam}")
        (out / "metadata.json").write_text(json.dumps(results["metadata"], indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(out / "frames.csv", rows)
        if results["errors"]:
            raise RuntimeError("Audit strutturale fallito; preprocessing non eseguito")

        # È la stessa catena di import, serializzazione e preprocessing usata nell'upload:
        # PKL -> TrajectorySaver.load_episode -> PlaysTrainingTransport.encode -> OnlineBatchBuilder.
        loaded_meta, episode = TrajectorySaver.load_episode(pkl)
        if loaded_meta != meta or len(episode.frames) != n:
            raise ValueError("Rilettura via TrajectorySaver differente dai dati originali")
        zip_payload = PlaysTrainingTransport.encode(loaded_meta, episode)
        print(f"\nZIP in RAM ricostruito con PlaysTrainingTransport.encode: {len(zip_payload)/(1024*1024):.2f} MiB")
        builder = OnlineBatchBuilder()
        cfg = builder.cfg
        front_crop = cfg.front_crop
        builder_file = inspect.getfile(OnlineBatchBuilder)
        builder_sha = hashlib.sha256(Path(builder_file).read_bytes()).hexdigest()
        results["batch_builder"] = {
            "source_file": builder_file,
            "source_sha256": builder_sha,
            "window_size": cfg.window_size,
            "stride": cfg.stride,
            "image_size": list(cfg.image_size),
            "front_crop": as_list(vars(cfg.front_crop)) if cfg.front_crop is not None else None,
            "gripper_crop": as_list(vars(cfg.gripper_crop)) if cfg.gripper_crop is not None else None,
            "front_feature": cfg.front_feature,
            "gripper_feature": cfg.gripper_feature,
            "algorithm": "plays: episode_prompt identico per tutte le finestre; ultimo start riallineato al frame finale, SENZA duplicare frame",
        }
        summary = builder.inspect_episode(zip_payload)  # Controlla davvero manifest e tutti i PNG.
        print(f"Preflight server: {summary}")
        windows_dir = out / "windows"
        windows_dir.mkdir(exist_ok=True)
        for win in builder.iter_episode_windows(zip_payload):
            text = win.batch["task"][0]
            front = win.batch[cfg.front_feature]
            gripper = win.batch[cfg.gripper_feature]
            expected = (1, cfg.window_size, 3, cfg.image_size[1], cfg.image_size[0])
            for cam, tensor in (("front", front), ("gripper", gripper)):
                if tuple(tensor.shape) != expected or tensor.dtype != __import__("torch").float32:
                    add(results, "errors", f"Finestra {win.window_index}: {cam} shape/dtype non validi: {tuple(tensor.shape)}/{tensor.dtype}")
                if not __import__("torch").isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                    add(results, "errors", f"Finestra {win.window_index}: {cam} pixel non finiti/fuori [0,1]")
            if text != meta["episode_prompt"]:
                add(results, "errors", f"Finestra {win.window_index}: task diverso da episode_prompt")
            indices = list(range(win.start_index + 1, win.start_index + 1 + cfg.window_size))
            image_file = f"window_{win.window_index:03d}.jpg"
            make_window_image(front, gripper, indices, windows_dir / image_file)
            windows.append({
                "window_index": win.window_index,
                "start_0based": win.start_index,
                "first_frame": indices[0], "last_frame": indices[-1],
                "frame_indices_1based": indices,
                "prompt": text,
                "front_shape": str(tuple(front.shape)),
                "gripper_shape": str(tuple(gripper.shape)),
                "front_range": f"[{float(front.min()):.3f}, {float(front.max()):.3f}]",
                "gripper_range": f"[{float(gripper.min()):.3f}, {float(gripper.max()):.3f}]",
                "preview": f"windows/{image_file}",
            })
        if len(windows) != summary["windows"]:
            add(results, "errors", f"Finestre generate {len(windows)}, attese {summary['windows']}")
        results["windows_count"] = len(windows)
        results["server_inspection"] = as_list(summary)
        write_csv(out / "windows.csv", windows)
        raw_file = "raw_4_cameras.jpg"
        make_raw_contact(raw_images, out / raw_file, front_crop)
        build_html(out, results, raw_file if (out / raw_file).is_file() else None, windows)
    except Exception as exc:
        add(results, "errors", f"Eccezione audit: {type(exc).__name__}: {exc}")
        if results.get("metadata"):
            build_html(out, results, None, windows)
    finally:
        results["status"] = "PASS" if not results["errors"] else "FAIL"
        (out / "audit.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n" + "=" * 60)
        print(f"AUDIT: {results['status']} | errori={len(results['errors'])} | warning={len(results['warnings'])} | frame={results['frame_count']} | windows={len(windows)}")
        if results["metadata"].get("episode_prompt"):
            print(f"PROMPT: {results['metadata']['episode_prompt']}")
        print(f"REPORT: {out / 'index.html'}")
        print(f"JSON:   {out / 'audit.json'}")
        print("=" * 60)
    return 0 if not results["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
