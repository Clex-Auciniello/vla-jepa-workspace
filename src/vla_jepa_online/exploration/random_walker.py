"""Seleziona waypoint nominali di esplorazione con campionamento pesato.

Responsabilita' del modulo:
- scegliere N waypoint DISTINTI dalla griglia, con N configurabile;
- privilegiare waypoint meno presenti negli episodi gia' utilizzati nel training;
- restituire gli ID nell'ordine di visita.

Non genera posizioni perturbate, rotazioni, comandi gripper o traiettorie
MoveIt. In modalita' OCCUPIED, le regole di salita, trasferimento in quota
e discesa finale verticale dovranno essere applicate dal robot executor.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


VALID_TABLE_MODES = ("empty", "occupied")


def load_exploration_seed(path: str | Path) -> int:
    """Legge il seed globale da exploration.yaml (da usare una volta per sessione).

    Il client deve creare un solo random.Random(seed) e condividerlo tra
    RandomWalker.sample() e MotionSampler.sample(). Non reinizializzarlo per
    ogni episodio, anche se gli altri parametri YAML vengono ricaricati.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Serve PyYAML: pip install pyyaml") from exc

    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or type(data.get("seed")) is not int:
        raise ValueError("exploration.yaml deve contenere 'seed' come intero")
    return data["seed"]


@dataclass(frozen=True)
class RandomWalkerConfig:
    num_waypoints: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RandomWalkerConfig":
        """Legge random_walker.num_waypoints da configs/exploration.yaml."""
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Serve PyYAML: pip install pyyaml") from exc

        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or not isinstance(data.get("random_walker"), dict):
            raise ValueError("Manca la sezione 'random_walker' in exploration.yaml")
        section = data["random_walker"]
        if "num_waypoints" not in section:
            raise ValueError("Manca 'random_walker.num_waypoints' in exploration.yaml")
        return cls(num_waypoints=section["num_waypoints"])


@dataclass(frozen=True)
class WalkPlan:
    table_mode: str
    waypoint_ids: tuple[int, ...]


class RandomWalker:
    def __init__(self, config: RandomWalkerConfig):
        self.cfg = config
        if type(self.cfg.num_waypoints) is not int or self.cfg.num_waypoints < 1:
            raise ValueError("num_waypoints deve essere un intero positivo")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RandomWalker":
        return cls(RandomWalkerConfig.from_yaml(path))

    def sample(
        self,
        grid: Mapping,
        accepted_visits: Mapping[int, int],
        *,
        table_mode: str,
        rng: random.Random,
    ) -> WalkPlan:
        """Campiona N waypoint senza reinserimento.

        Per ogni waypoint ancora disponibile:
            peso(id) = 1 / (1 + numero_visite_valide_training[id])
        Quindi ricalcola le probabilita' dopo ogni estrazione, eliminando
        il waypoint scelto. Non modifica accepted_visits ne' lo stato su disco.

        """
        if table_mode not in VALID_TABLE_MODES:
            raise ValueError("table_mode deve essere 'empty' o 'occupied'")
        if not isinstance(grid.get("waypoints"), list) or not grid["waypoints"]:
            raise ValueError("La griglia non contiene waypoint")

        ids = []
        for wp in grid["waypoints"]:
            if type(wp.get("id")) is not int:
                raise ValueError("Ogni waypoint deve avere un ID intero")
            if wp.get("level") not in ("PICK", "TRANSFER"):
                raise ValueError(f"Livello non valido per waypoint {wp['id']}")
            ids.append(wp["id"])
        if len(ids) != len(set(ids)):
            raise ValueError("La griglia contiene ID waypoint duplicati")
        if self.cfg.num_waypoints > len(ids):
            raise ValueError(
                f"Richiesti {self.cfg.num_waypoints} waypoint, "
                f"ma la griglia ne contiene soltanto {len(ids)}"
            )

        valid_ids = set(ids)
        for wid, count in accepted_visits.items():
            if wid not in valid_ids:
                raise ValueError(f"Conteggio riferito a waypoint inesistente: {wid}")
            if type(count) is not int or count < 0:
                raise ValueError(f"Conteggio non valido per waypoint {wid}: {count}")

        remaining = ids.copy()
        selected = []
        for _ in range(self.cfg.num_waypoints):
            weights = [1.0 / (1.0 + accepted_visits.get(wid, 0)) for wid in remaining]
            if not all(math.isfinite(w) and w > 0.0 for w in weights):
                raise ValueError("Pesi di campionamento non validi")
            picked = rng.choices(remaining, weights=weights, k=1)[0]
            selected.append(picked)
            remaining.remove(picked)

        return WalkPlan(table_mode=table_mode, waypoint_ids=tuple(selected))
