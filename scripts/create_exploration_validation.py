#!/usr/bin/env python3
"""Copia nel validation set i 29 PKL di esplorazione selezionati."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path("/mnt/beegfs/a.cuciniello9/datasets/robotic/exploration_preprocessed")
DEFAULT_DEST = Path("/mnt/beegfs/a.cuciniello9/datasets/robotic/exploration_validation")

# ID delle traiettorie non viste dal checkpoint online di una epoca.
VALIDATION_IDS = (
    *range(141, 148),
    *range(198, 202),
    *range(243, 247),
    *range(267, 274),
    *range(374, 380),
    440,
)


def sha256_file(path: Path) -> str:
    """Calcola SHA256 leggendo il file a blocchi."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error("Sorgente e destinazione devono essere directory separate")

    names = [f"E{number:05d}.pkl" for number in VALIDATION_IDS]
    if len(names) != 29 or len(set(names)) != 29:
        raise AssertionError("Gli ID selezionati non sono 29 e univoci")
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"PKL mancanti nella sorgente: {', '.join(missing)}")

    destination.mkdir(parents=True, exist_ok=True)
    # Evita di utilizzare per errore altri PKL già presenti nella cartella.
    extras = sorted(p.name for p in destination.glob("E*.pkl") if p.name not in names)
    if extras:
        raise ValueError(f"La cartella di validation contiene PKL estranei: {extras}")

    records = []
    for i, name in enumerate(names, start=1):
        src, dst = source / name, destination / name
        # Se il file esiste, non lo sovrascriviamo: controlliamo che sia uguale.
        if not dst.exists():
            shutil.copy2(src, dst)
        source_hash = sha256_file(src)
        dest_hash = sha256_file(dst)
        if source_hash != dest_hash:
            raise RuntimeError(f"SHA256 diverso tra sorgente e destinazione: {name}")
        records.append({"episode_id": Path(name).stem, "filename": name,
                        "size_bytes": dst.stat().st_size, "sha256": dest_hash})
        print(f"[{i:02d}/29] {name} OK", flush=True)

    # Il manifest non viene letto dal loader, ma documenta esattamente la selezione.
    manifest = {"validation_ids": list(VALIDATION_IDS), "episodes": records}
    (destination / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validation set completo: {len(records)} PKL in {destination}", flush=True)


if __name__ == "__main__":
    main()
