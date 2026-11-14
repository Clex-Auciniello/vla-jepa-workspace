#!/usr/bin/env python3
"""Estrae campioni delle immagini front/wrist da PKL eterogenei, SENZA modificare le sorgenti.

Uso:
 python extract_mixture_samples.py \
   --reference /mnt/beegfs/a.cuciniello9/datasets/robotic/vla_jepa_offline_merged \
   --mixture /mnt/beegfs/a.cuciniello9/datasets/mixture \
   --output /mnt/beegfs/a.cuciniello9/datasets/mixture_inspection_samples

Richiede numpy e Pillow. I PKL che serializzano classi custom richiedono che
il modulo Python di quelle classi sia importabile (es. impostando PYTHONPATH).
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

FRONT = re.compile(r"(^|[./_\-])(front|camera.front|image.0|rgb.0|observation.image.0|primary)([./_\-]|$)", re.I)
WRIST = re.compile(r"(^|[./_\-])(gripper|wrist|eye.in.hand|image.1|rgb.1)([./_\-]|$)", re.I)
BAD = re.compile(r"depth|mask|segmentation|point.cloud|heatmap|action|gripper.state|state", re.I)
IMAGE_HINT = re.compile(r"image|rgb|camera|frame|observation|video", re.I)


def view_for(path: str) -> str | None:
    p = path.lower()
    if BAD.search(p) and not (FRONT.search(p) or WRIST.search(p)):
        return None
    if re.search(r"depth|mask|segmentation|point.cloud|heatmap", p):
        return None
    if WRIST.search(p):
        return "gripper"
    if FRONT.search(p):
        return "front"
    if IMAGE_HINT.search(p):
        return "front_candidate"
    return None


def decode_image(value):
    """Restituisce (immagine RGB PIL, encoding), senza effettuare crop/resize."""
    if isinstance(value, Image.Image):
        return value.convert("RGB"), "PIL"
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return Image.open(io.BytesIO(bytes(value))).convert("RGB"), "compressed"
        except Exception:
            return None
    if isinstance(value, np.ndarray):
        a = value
        if a.dtype == object and a.size == 1:
            return decode_image(a.item())
        if a.ndim == 4 and a.shape[-1] in (3, 4) and a.shape[0] > 0:
            a = a[0]
        if a.ndim == 3 and a.shape[2] in (3, 4) and a.shape[0] >= 32 and a.shape[1] >= 32:
            if a.dtype != np.uint8:
                if np.issubdtype(a.dtype, np.floating) and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1:
                    a = (a * 255).clip(0, 255).astype(np.uint8)
                else:
                    return None
            return Image.fromarray(a[:, :, :3]).convert("RGB"), "raw_array_unverified_RGB_or_BGR"
        if a.dtype == np.uint8 and (a.ndim == 1 or (a.ndim == 2 and 1 in a.shape)):
            try:
                return Image.open(io.BytesIO(a.tobytes())).convert("RGB"), "compressed_array"
            except Exception:
                return None
    return None


def children(obj, rng: random.Random):
    if isinstance(obj, dict):
        pairs = list(obj.items())
        # Prima le strutture di osservazione e le immagini rispetto ad altri metadati.
        pairs.sort(key=lambda kv: (0 if IMAGE_HINT.search(str(kv[0])) else 1, str(kv[0])))
        for k, v in pairs[:90]:
            yield str(k), v
    elif isinstance(obj, (list, tuple)):
        n = len(obj)
        if n:
            positions = list(dict.fromkeys([rng.randrange(n), n // 2, 0, n - 1]))
            for i in positions:
                yield f"[{i}]", obj[i]
    elif isinstance(obj, np.ndarray) and obj.ndim > 3 and obj.shape[0] > 0:
        i = rng.randrange(obj.shape[0]); yield f"[{i}]", obj[i]
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if not k.startswith("__"):
                yield k, v


def find_views(obj, rng: random.Random):
    found = {}
    seen = set()
    nodes = [("root", obj, 0)]
    examined = 0
    while nodes and examined < 10000:
        path, value, depth = nodes.pop(0)
        examined += 1
        if depth > 12 or id(value) in seen:
            continue
        seen.add(id(value))
        label = view_for(path)
        if label and label not in found:
            result = decode_image(value)
            if result is not None:
                found[label] = (result[0], result[1], path)
                if "front" in found and "gripper" in found:
                    break
                continue
        if isinstance(value, (dict, list, tuple, np.ndarray)) or hasattr(value, "__dict__"):
            for k, child in children(value, rng):
                nodes.append((f"{path}.{k}", child, depth + 1))
    if "front" not in found and "front_candidate" in found:
        found["front"] = found.pop("front_candidate")
    return found, examined


def write_images(folder: Path, found):
    details = {}
    folder.mkdir(parents=True, exist_ok=True)
    for view, (im, encoding, keypath) in found.items():
        if view == "front_candidate":
            continue
        rawpath = folder / f"{view}_raw.png"
        im.save(rawpath)
        # Per array raw, la codifica RGB/BGR non è nota: salva due interpretazioni.
        if encoding == "raw_array_unverified_RGB_or_BGR":
            a = np.asarray(im)
            Image.fromarray(a[:, :, ::-1]).save(folder / f"{view}_swap_rb.png")
        details[view] = {"key": keypath, "size_wh": list(im.size), "encoding": encoding,
                         "files": [x.name for x in folder.glob(f"{view}_*.png")]}
    return details


def sample_source(name: str, directory: Path, output: Path, rng: random.Random, tries: int):
    record = {"source": name, "directory": str(directory), "status": "not_found"}
    if not directory.exists():
        return record
    candidates = list(directory.rglob("*.pkl"))
    rng.shuffle(candidates)
    errors = []
    for path in candidates[:tries]:
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
            found, examined = find_views(obj, rng)
            if not found:
                errors.append(f"{path.name}: nessuna immagine RGB riconosciuta (nodi {examined})")
                continue
            details = write_images(output / name, found)
            record.update(status="ok", example_pkl=str(path), root_type=str(type(obj)), views=details)
            return record
        except Exception as e:
            errors.append(f"{path.name}: {type(e).__name__}: {str(e)[:200]}")
    # Se non è accessibile alcun PKL, usa solo come fallback un PNG/JPG esistente.
    images = [p for p in directory.rglob("*.png") if p.is_file()]
    if not images:
        images = [p for p in directory.rglob("*.jpg") if p.is_file()]
    if images:
        path = rng.choice(images)
        dest = output / name
        dest.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as im:
            im.convert("RGB").save(dest / "existing_image_sample.png")
            sz = im.size
        record.update(status="image_file_fallback", example_image=str(path), size_wh=sz)
    else:
        record["status"] = "no_decodable_sample"
    record["attempted_pkl"] = min(len(candidates), tries)
    record["errors"] = errors[:8]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tries", type=int, default=12)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    folders = {"reference_robot": args.reference}
    for name in ("exploration_preprocessed", "plays_preprocessed", "osvi_awda_rollout",
                 "result_osvi_awda_human", "cod_rollout_finetuning",
                 "cod_epoch_100_no_wrist", "cod_epoch_100_wrist",
                 "vla_jepa_rollout", "vla_jepa_rollouts"):
        candidate = args.mixture / name
        if candidate.exists():
            folders[name] = candidate
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, path in folders.items():
        print(f"[{name}] {path}", flush=True)
        result = sample_source(name, path, args.output, rng, args.tries)
        summary.append(result)
        print(f"  {result['status']}: {result.get('example_pkl', result.get('example_image', ''))}", flush=True)
        if result.get("errors"):
            print("  Esempio errore:", result["errors"][0], flush=True)
    with (args.output / "inspection.json").open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nFile campione e metadati: {args.output}")

if __name__ == "__main__":
    main()
