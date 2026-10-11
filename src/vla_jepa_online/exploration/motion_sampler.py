"""Campiona i target e gli eventi di una traiettoria esplorativa UR5e.

Il random walker seleziona gli ID nominali dei waypoint. Questo modulo:
- applica una piccola perturbazione XYZ negli assi locali del volume misurato;
- estrae eventi di rotazione e gripper per waypoint, in base a quota e tavolo;
- impone i limiti globali di sicurezza concordati per ogni traiettoria.

Non importa ROS, non pianifica movimenti e non comanda il robot.
Gli eventi di rotazione/gripper andranno eseguiti DOPO aver raggiunto
il waypoint, con il TCP fermo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


MAX_ROTATIONS_PER_TRAJECTORY = 2


@dataclass(frozen=True)
class MotionSamplerConfig:
    """Parametri obbligatori: l'unica fonte dei valori e' exploration.yaml."""

    rotation_prob: Mapping[str, Mapping[str, float]]
    gripper_close_prob: Mapping[str, Mapping[str, float]]
    gripper_reopen_prob: Mapping[str, Mapping[str, float]]

    # Perturbazioni lungo gli assi locali del workspace, in metri.
    xy_jitter_m: float
    z_jitter_pick_up_m: float
    z_jitter_transfer_m: float
    z_jitter_top_down_m: float

    # Yaw rispetto all'orientazione iniziale, attorno a Z di base_link.
    max_abs_yaw_deg: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> MotionSamplerConfig:
        """Legge le probabilita' e i parametri da configs/exploration.yaml.

        Richiede PyYAML (pip install pyyaml). Non carica il modello VLA-JEPA.
        """
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Serve PyYAML per leggere exploration.yaml: pip install pyyaml") from exc

        with Path(path).open("r", encoding="utf-8") as stream:
            content = yaml.safe_load(stream)
        if not isinstance(content, dict) or not isinstance(content.get("motion_sampler"), dict):
            raise ValueError("Il file YAML deve contenere la sezione 'motion_sampler'")

        params = content["motion_sampler"]
        fields = (
            "rotation_prob", "gripper_close_prob", "gripper_reopen_prob",
            "xy_jitter_m", "z_jitter_pick_up_m", "z_jitter_transfer_m",
            "z_jitter_top_down_m", "max_abs_yaw_deg",
        )
        missing = [key for key in fields if key not in params]
        if missing:
            raise ValueError(f"Parametri mancanti in motion_sampler: {missing}")
        return cls(**{key: params[key] for key in fields})


@dataclass(frozen=True)
class MotionStep:
    waypoint_id: int
    i: int
    j: int
    k: int
    level: str
    nominal_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    rotate_here: bool
    # Offset assoluto di yaw rispetto all'orientazione iniziale (non delta).
    yaw_offset_after_deg: float
    toggle_gripper_here: bool
    gripper_closed_after: bool


@dataclass(frozen=True)
class MotionPlan:
    table_mode: str
    steps: tuple[MotionStep, ...]



def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> tuple[float, float, float]:
    """Interpolazione lineare tra due punti 3D."""
    return tuple((1.0 - t) * x + t * y for x, y in zip(a, b))


def _interpolate(
    corners: Mapping[str, Sequence[float]], fx: float, fy: float, fz: float
) -> tuple[float, float, float]:
    """Coordinate locali normalizzate -> XYZ in base_link usando gli 8 vertici."""
    bottom_back = _lerp(corners["V000"], corners["V100"], fx)
    bottom_front = _lerp(corners["V010"], corners["V110"], fx)
    top_back = _lerp(corners["V001"], corners["V101"], fx)
    top_front = _lerp(corners["V011"], corners["V111"], fx)
    bottom = _lerp(bottom_back, bottom_front, fy)
    top = _lerp(top_back, top_front, fy)
    return _lerp(bottom, top, fz)


class MotionSampler:
    def __init__(self, config: MotionSamplerConfig):
        self.cfg = config

        for name, probs in (
            ("rotation_prob", self.cfg.rotation_prob),
            ("gripper_close_prob", self.cfg.gripper_close_prob),
            ("gripper_reopen_prob", self.cfg.gripper_reopen_prob),
        ):
            for mode in ("empty", "occupied"):
                for level in ("PICK", "TRANSFER"):
                    p = float(probs[mode][level])
                    if not 0.0 <= p <= 1.0:
                        raise ValueError(f"{name}.{mode}.{level} deve essere compreso tra 0 e 1")

        if min(
            self.cfg.xy_jitter_m,
            self.cfg.z_jitter_pick_up_m,
            self.cfg.z_jitter_transfer_m,
            self.cfg.z_jitter_top_down_m,
        ) < 0:
            raise ValueError("Le perturbazioni devono essere non negative")
        if not 0.0 <= self.cfg.max_abs_yaw_deg <= 30.0:
            raise ValueError("La rotazione massima deve essere compresa tra 0 e 30 gradi")

    @classmethod
    def from_yaml(cls, path: str | Path) -> MotionSampler:
        return cls(MotionSamplerConfig.from_yaml(path))

    def _perturb(self, grid: dict, wp: dict, rng: random.Random) -> tuple[float, float, float]:
        """Jitter negli assi del workspace, poi conversione in base_link.

        Gli assi locali sono definiti dagli otto vertici in grid["vertices"],
        non si assume che siano paralleli a quelli di base_link.

        Nessun clamp e nessuna correzione/scarto del jitter XY per piccoli
        dislivelli tra vertici: il margine fisico ai bordi e' a carico della
        misura del workspace e della successiva verifica di esecuzione.
        """
        dims = grid["dimensions_m"]
        offsets = grid["sampling"]["axis_offsets_m"]
        i, j, k = int(wp["i"]), int(wp["j"]), int(wp["k"])
        last_k = len(offsets["z"]) - 1

        # Coordinate nominali in metri lungo gli assi LOCALI della griglia.
        x = offsets["x"][i] + rng.uniform(-self.cfg.xy_jitter_m, self.cfg.xy_jitter_m)
        y = offsets["y"][j] + rng.uniform(-self.cfg.xy_jitter_m, self.cfg.xy_jitter_m)
        z = offsets["z"][k]

        if k == 0:
            z += rng.uniform(0.0, self.cfg.z_jitter_pick_up_m)
        elif k == last_k:
            z += rng.uniform(-self.cfg.z_jitter_top_down_m, 0.0)
        else:
            z += rng.uniform(-self.cfg.z_jitter_transfer_m, self.cfg.z_jitter_transfer_m)

        # Le frazioni (fx, fy, fz) individuano il punto nel volume misurato.
        # L'orientamento rispetto a base_link e' implicito nelle coordinate
        # XYZ reali degli otto vertici: non serve leggere un angolo separato.
        target = _interpolate(
            grid["vertices"],
            x / dims["x"],
            y / dims["y"],
            z / dims["z"],
        )
        return tuple(round(float(value), 6) for value in target)

    def sample(
        self,
        grid: dict,
        waypoint_ids: Sequence[int],
        *,
        table_mode: str,
        rng: random.Random,
        initially_closed: bool = False,
    ) -> MotionPlan:
        """Associa perturbazioni, rotazioni e gripper ai waypoint del walker.

        La posa iniziale deve avere gripper APERTO. La sequenza puo' essere:
          OPEN; OPEN->CLOSED; oppure OPEN->CLOSED->OPEN.
        Le rotazioni sono 0, 1 o 2; la seconda annulla esattamente la prima.
        """
        if table_mode not in ("empty", "occupied"):
            raise ValueError("table_mode deve essere 'empty' o 'occupied'")
        if initially_closed:
            raise ValueError("Il gripper deve essere aperto a inizio traiettoria")
        if not waypoint_ids or len(waypoint_ids) != len(set(waypoint_ids)):
            raise ValueError("La traiettoria deve avere waypoint distinti e non essere vuota")

        by_id = {int(wp["id"]): wp for wp in grid["waypoints"]}
        selected = [by_id[int(wid)] for wid in waypoint_ids]
        steps = []
        rotation_count = 0
        first_yaw_deg = 0.0
        current_yaw_deg = 0.0
        # 0: mai chiuso; 1: chiuso una volta; 2: riaperto (nessun altro cambio).
        gripper_phase = 0

        for wp in selected:
            level = str(wp["level"])

            rotate_here = False
            if rotation_count < MAX_ROTATIONS_PER_TRAJECTORY:
                rotate_here = rng.random() < float(self.cfg.rotation_prob[table_mode][level])
                if rotate_here:
                    if rotation_count == 0:
                        first_yaw_deg = rng.uniform(-self.cfg.max_abs_yaw_deg, self.cfg.max_abs_yaw_deg)
                        current_yaw_deg = first_yaw_deg
                    else:
                        # Seconda rotazione: delta opposto alla prima,
                        # quindi ritorno all'orientazione di partenza.
                        current_yaw_deg = 0.0
                    rotation_count += 1

            toggle_gripper_here = False
            if gripper_phase == 0:
                toggle_gripper_here = rng.random() < float(self.cfg.gripper_close_prob[table_mode][level])
                if toggle_gripper_here:
                    gripper_phase = 1
            elif gripper_phase == 1:
                toggle_gripper_here = rng.random() < float(self.cfg.gripper_reopen_prob[table_mode][level])
                if toggle_gripper_here:
                    gripper_phase = 2

            steps.append(MotionStep(
                waypoint_id=int(wp["id"]),
                i=int(wp["i"]), j=int(wp["j"]), k=int(wp["k"]),
                level=level,
                nominal_position=tuple(float(v) for v in wp["position"]),
                target_position=self._perturb(grid, wp, rng),
                rotate_here=rotate_here,
                yaw_offset_after_deg=round(current_yaw_deg, 3),
                toggle_gripper_here=toggle_gripper_here,
                gripper_closed_after=(gripper_phase == 1),
            ))

        return MotionPlan(table_mode=table_mode, steps=tuple(steps))
