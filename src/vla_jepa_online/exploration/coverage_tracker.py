"""Persistenza minimale della copertura per il pretraining online VLA-JEPA.

Conserva esclusivamente:
- numero del PROSSIMO episodio da accettare per il training;
- conteggi di visite negli episodi accettati dall'operatore;
- generatore random condiviso (seed + stato RNG);
- impronta della geometria della griglia.

Non salva episodi attivi, modalita' del tavolo o video.
Scrive lo stato solo alla prima inizializzazione e alla FINE di ogni
traiettoria (accettata oppure scartata), mai durante il movimento.

Un solo exploration client deve usare lo stesso file di stato alla volta.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


STATE_VERSION = 4
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATE_PATH = WORKSPACE_ROOT / "outputs" / "exploration_state.json"


def _rng_state_to_json(state: tuple) -> list:
    """Converte lo stato di random.Random in tipi serializzabili JSON."""
    return [state[0], list(state[1]), state[2]]


def _rng_state_from_json(data: list) -> tuple:
    """Ricostruisce la struttura attesa da random.Random.setstate()."""
    if not isinstance(data, list) or len(data) != 3 or not isinstance(data[1], list):
        raise ValueError("rng_state persistito non valido")
    return data[0], tuple(data[1]), data[2]


class CoverageTracker:
    def __init__(
        self,
        grid: Mapping,
        state_path: str | Path = DEFAULT_STATE_PATH,
        *,
        seed: int,
    ) -> None:
        """Crea uno stato nuovo o recupera contatori, numero episodio e RNG.

        La geometria della griglia e il seed devono coincidere con quelli
        dello stato esistente. In caso contrario non sovrascrive il file.
        """
        if type(seed) is not int:
            raise ValueError("seed deve essere un intero")
        if not isinstance(grid.get("waypoints"), list) or not grid["waypoints"]:
            raise ValueError("La griglia non contiene waypoint")

        waypoints = grid["waypoints"]
        ids = [wp["id"] for wp in waypoints]
        if any(type(wid) is not int for wid in ids) or len(ids) != len(set(ids)):
            raise ValueError("Gli ID dei waypoint devono essere interi univoci")
        self._ids = frozenset(ids)

        # L'impronta include la geometria necessaria al motion sampler.
        # HOME e i metadati non geometrici non influenzano la copertura.
        geometry = {
            "base_frame": grid.get("base_frame", "base_link"),
            "tcp_frame": grid.get("tcp_frame", "tcp_link"),
            "vertices": grid.get("vertices"),
            "dimensions_m": grid.get("dimensions_m"),
            "axis_offsets_m": grid.get("sampling", {}).get("axis_offsets_m"),
            "waypoints": [
                {
                    "id": wp["id"],
                    "i": wp["i"],
                    "j": wp["j"],
                    "k": wp["k"],
                    "level": wp["level"],
                    "position": wp["position"],
                }
                for wp in sorted(waypoints, key=lambda wp: wp["id"])
            ],
        }
        self._grid_hash = hashlib.sha256(
            json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        self.state_path = Path(state_path)
        self.seed = seed
        self.rng = random.Random(seed)

        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            old_version = self._state.get("version")
            if old_version == 3:
                self._state["version"] = STATE_VERSION
                self._state["start_visits"] = {str(wid): 0 for wid in sorted(self._ids)}
            self._validate_loaded()
            self.rng.setstate(_rng_state_from_json(self._state["rng_state"]))
            if old_version == 3:
                self._save()
        else:
            self._state = {
                "version": STATE_VERSION,
                "seed": seed,
                "grid_hash": self._grid_hash,
                "next_episode_number": 1,
                "accepted_visits": {str(wid): 0 for wid in sorted(self._ids)},
                "start_visits": {str(wid): 0 for wid in sorted(self._ids)},
                "rng_state": _rng_state_to_json(self.rng.getstate()),
            }
            self._save()

    def _validate_loaded(self) -> None:
        state = self._state
        if state.get("version") != STATE_VERSION:
            raise ValueError(
                f"Versione exploration_state.json non compatibile: "
                f"trovata {state.get('version')}, richiesta {STATE_VERSION}. "
                "Il formato precedente deve essere migrato prima di riutilizzarlo."
            )
        if state.get("grid_hash") != self._grid_hash:
            raise ValueError(
                "La geometria della griglia e' cambiata: impossibile riutilizzare "
                "automaticamente i vecchi conteggi."
            )
        if type(state.get("seed")) is not int or state["seed"] != self.seed:
            raise ValueError("Il seed in exploration.yaml non coincide con quello dello stato salvato")
        number = state.get("next_episode_number")
        if type(number) is not int or number < 1:
            raise ValueError("next_episode_number non valido")
        counts = state.get("accepted_visits")
        if not isinstance(counts, dict) or set(counts) != {str(wid) for wid in self._ids}:
            raise ValueError("accepted_visits non compatibile con la griglia")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise ValueError("accepted_visits contiene contatori non validi")
        starts = state.get("start_visits")
        if not isinstance(starts, dict) or set(starts) != {str(wid) for wid in self._ids}:
            raise ValueError("start_visits non compatibile con la griglia")
        if any(type(count) is not int or count < 0 for count in starts.values()):
            raise ValueError("start_visits contiene contatori non validi")
        try:
            random.Random().setstate(_rng_state_from_json(state.get("rng_state")))
        except (TypeError, ValueError) as exc:
            raise ValueError("rng_state persistito non valido") from exc

    @property
    def accepted_visits(self) -> dict[int, int]:
        """Restituisce una copia dei conteggi con chiavi intere per RandomWalker."""
        return {int(wid): count for wid, count in self._state["accepted_visits"].items()}

    @property
    def start_visits(self) -> dict[int, int]:
        """Conteggi separati dei waypoint nominali usati come START."""
        return {int(wid): count for wid, count in self._state["start_visits"].items()}

    @property
    def next_episode_number(self) -> int:
        return self._state["next_episode_number"]

    @property
    def next_episode_id(self) -> str:
        """ID che il client mostra prima della registrazione, es. E00018."""
        return f"E{self.next_episode_number:05d}"

    def accept_episode(self, episode_id: str, waypoint_ids: Sequence[int], start_waypoint_id: int) -> None:
        """Aggiorna le visite subito dopo l'accettazione dell'operatore.

        Non attende il training: i 5 ID sono quelli nominali dei waypoint
        esplorativi, mai i transiti ausiliari introdotti dall'executor.
        """
        self._check_episode_id(episode_id)
        selected = list(waypoint_ids)
        if (not selected or any(type(wid) is not int for wid in selected)
                or len(selected) != len(set(selected)) or set(selected) - self._ids):
            raise ValueError("waypoint_ids deve contenere ID nominali validi e distinti")

        if type(start_waypoint_id) is not int or start_waypoint_id not in self._ids:
            raise ValueError("start_waypoint_id non valido")
        if start_waypoint_id in selected:
            raise ValueError("Lo START non deve comparire tra i waypoint esplorativi")
        for wid in selected:
            self._state["accepted_visits"][str(wid)] += 1
        self._state["start_visits"][str(start_waypoint_id)] += 1
        self._state["next_episode_number"] += 1
        self._save()

    def discard_episode(self, episode_id: str) -> None:
        """Scarta la traiettoria; salva soltanto l'RNG avanzato.

        L'ID rimane lo stesso; accepted_visits e start_visits non cambiano.
        """
        self._check_episode_id(episode_id)
        self._save()

    def _check_episode_id(self, episode_id: str) -> None:
        if episode_id != self.next_episode_id:
            raise ValueError(
                f"ID episodio non valido: atteso {self.next_episode_id}, ricevuto {episode_id}"
            )

    def _save(self) -> None:
        """Scrittura atomica del JSON sul disco (con stato RNG corrente)."""
        self._state["rng_state"] = _rng_state_to_json(self.rng.getstate())
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.", suffix=".tmp", dir=self.state_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._state, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.state_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
