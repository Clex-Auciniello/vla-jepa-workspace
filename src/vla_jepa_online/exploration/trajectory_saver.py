"""Archivio persistente delle traiettorie accettate da VLA-JEPA.

Ogni ZIP e' esattamente il payload che verra' inviato al server Flask:
PNG front/gripper e manifest.json con lo stato per ogni frame.
Salvataggio atomico e senza overwrite di episodi esistenti.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


class TrajectorySaver:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"Percorso di salvataggio non e' una directory: {self.root}")

    def save(self, episode_id: str, payload: bytes) -> Path:
        if not re.fullmatch(r"E\d{5,}", episode_id):
            raise ValueError(f"ID episodio inatteso: {episode_id!r}")
        if not payload:
            raise ValueError("ZIP della traiettoria vuoto")
        target = self.root / f"{episode_id}.zip"
        if target.exists():
            raise FileExistsError(f"Traiettoria gia' salvata: {target}")

        fd, tmp = tempfile.mkstemp(prefix=f".{episode_id}.", suffix=".part", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # link+unlink equivale a pubblicare atomicamente senza sovrascrivere
            # un episodio salvato in precedenza (stesso filesystem).
            os.link(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target
