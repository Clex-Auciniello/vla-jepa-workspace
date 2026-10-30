"""Catalogo Plays: task, prompt semantico unico ed espansione in primitive.

Due click sempre richiesti. Direzioni riferite a base_link:
+X = right, -X = left, +Y = forward, -Y = backward.
Il task 03 riceve un angolo generato dal client, il task 09 uno scelto dall'operatore.
L'angolo viene conservato nei metadati, ma non compare nel prompt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from vla_jepa_online.exploration.robot_executor import (
    PoseTarget, ExecutionPlan, _yaw_in_base,
)
from vla_jepa_online.plays.primitives import MotionPrimitives, PrimitiveConfig

COLORS = ("red", "green", "blue", "yellow")
TASKS: Mapping[str, str] = {
    # Non rinumerare: manteniamo stabili gli ID dal vecchio catalogo.
    "02": "Pick and relocate",
    "03": "Pick, rotate and relocate",
    "04": "Push cube across the table [EXPERIMENTAL]",
    "05": "Stack one cube on another [EXPERIMENTAL]",
    "06": "Unstack and relocate [EXPERIMENTAL]",
    "07": "Extract cube from bin [EXPERIMENTAL]",
    "08": "Pick and place cube into bin [EXPERIMENTAL]",
    "09": "Rotate gripper, pick and relocate [EXPERIMENTAL]",
}
EXPERIMENTAL = frozenset({"04", "05", "06", "07", "08", "09"})
ROTATION_TASKS = frozenset({"03", "09"})
RELOCATION_TASKS = frozenset({"02", "03", "09"})
IN_PLACE_PIXEL_THRESHOLD = 12


@dataclass(frozen=True)
class PlayRequest:
    task_id: str
    color: str
    target_color: str | None
    pick_uv: tuple[int, int]
    place_uv: tuple[int, int]
    pick_xyz: tuple[float, float, float]
    place_xyz: tuple[float, float, float]
    pick_y_offset_m: float
    pick_x_offset_m: float
    rotation_deg: float | None = None

    def __post_init__(self):
        if self.task_id not in TASKS:
            raise ValueError("Task Plays non riconosciuto")
        if self.color not in COLORS:
            raise ValueError("Colore cubo non riconosciuto")
        if self.task_id == "05":
            if self.target_color not in COLORS:
                raise ValueError("Il task Stack richiede un secondo colore valido")
        elif self.target_color is not None:
            raise ValueError("Il secondo colore serve solo nello Stack")
        if len(self.pick_uv) != 2 or len(self.place_uv) != 2:
            raise ValueError("Servono esattamente due click 2D")
        if (len(self.pick_xyz) != 3 or len(self.place_xyz) != 3 or
                not all(math.isfinite(float(v)) for v in (*self.pick_xyz, *self.place_xyz))):
            raise ValueError("Punti 3D non validi")
        if not math.isfinite(self.pick_y_offset_m):
            raise ValueError("Offset Y non valido")
        if self.task_id in ROTATION_TASKS:
            if (self.rotation_deg is None or
                    not math.isfinite(self.rotation_deg) or
                    not -30.0 <= self.rotation_deg <= 30.0):
                raise ValueError("Task rotazione: angolo obbligatorio tra -30 e +30 gradi")
        elif self.rotation_deg is not None:
            raise ValueError("L'angolo è consentito solo per i task 03 e 09")


def _near_clicks(req: PlayRequest) -> bool:
    """Una destinazione quasi coincidente equivale semanticamente a in place.

    Soglia in pixel intenzionale: l'operatore ha accettato questa convenzione.
    """
    return (abs(req.place_uv[0] - req.pick_uv[0]) < IN_PLACE_PIXEL_THRESHOLD and
            abs(req.place_uv[1] - req.pick_uv[1]) < IN_PLACE_PIXEL_THRESHOLD)


def _object_xyz(req: PlayRequest) -> tuple[float, float, float]:
    # L'offset Y del vecchio ScriptController si applica solo al target grasp.
    x, y, z = req.pick_xyz
    return (
        x + req.pick_x_offset_m,
        y + req.pick_y_offset_m,
        z,
    )

def _dest_xyz(req: PlayRequest) -> tuple[float, float, float]:
    # FIX in X per il place.
    x, y, z = req.place_xyz
    return (
        x + req.pick_x_offset_m,
        y + req.pick_y_offset_m,
        z,
    )


def _in_place(req: PlayRequest) -> bool:
    if req.task_id not in RELOCATION_TASKS:
        return False
    if _near_clicks(req):
        return True
    ox, oy, _ = _object_xyz(req)
    dx, dy = req.place_xyz[0] - ox, req.place_xyz[1] - oy
    return math.hypot(dx, dy) < 1e-8

def _direction(req: PlayRequest) -> str:
    """Determina la direzione del movimento in base_link.

    Direzioni cardinali:
        dx < 0 -> right
        dx > 0 -> left
        dy > 0 -> forward
        dy < 0 -> backward

    Direzioni diagonali:
        forward-left
        forward-right
        backward-left
        backward-right

    La classificazione utilizza 8 settori angolari da 45 gradi.
    """

    if _in_place(req):
        return "in place"

    ox, oy, _ = _object_xyz(req)

    dx = req.place_xyz[0] - ox
    dy = req.place_xyz[1] - oy

    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return "in place"

    # Soglia angolare per distinguere direzioni cardinali e diagonali.
    diagonal_ratio = math.tan(math.pi / 8)  # circa 0.4142

    # Movimento diagonale:
    # entrambe le componenti devono essere sufficientemente rilevanti.
    if min(abs(dx), abs(dy)) >= diagonal_ratio * max(abs(dx), abs(dy)):

        horizontal = "to the right" if dx < 0 else "to the left"
        vertical = "forward" if dy > 0 else "backward"

        return f"{vertical} and {horizontal}"

    # Movimento prevalentemente orizzontale.
    if abs(dx) >= abs(dy):
        return "to the right" if dx < 0 else "to the left"

    # Movimento prevalentemente verticale.
    return "forward" if dy > 0 else "backward"

def _relocation_instruction(req: PlayRequest) -> str:
    if _in_place(req):
        return "put it back in place"
    return f"move it {_direction(req)}"


def semantic_prompt(req: PlayRequest) -> str:
    c = f"the {req.color} cube"
    if req.task_id == "02":
        return f"Pick up {c} and {_relocation_instruction(req)}."
    if req.task_id == "03":
        return f"Pick up {c}, rotate it, and {_relocation_instruction(req)}."
    if req.task_id == "04":
        return f"Push {c} {_direction(req)}."
    if req.task_id == "05":
        return f"Pick up {c} and stack it on top of the {req.target_color} cube."
    if req.task_id == "06":
        return f"Take {c} off the stack and place it on the table."
    if req.task_id == "07":
        return f"Take {c} out of the bin and place it on the table."
    if req.task_id == "08":
        return (
            f"Correct the position of the {c} "
            f"by moving it inside the box {_direction(req)}. "
        )
    if req.task_id == "09":
        verb = "Rotate the gripper to grasp" if req.rotation_deg != 0 else "Align the gripper to grasp"
        return f"{verb} {c}, then {_relocation_instruction(req)}."
    raise AssertionError(req.task_id)


def build_play_plan(start_pose: PoseTarget, req: PlayRequest,
                    cfg: PrimitiveConfig) -> ExecutionPlan:
    """Costruisce un piano deterministico: la casualità è risolta nel client.

    Il task 09 ruota il gripper AL REACH (prima della discesa). Non cambia
    l'orientazione top-down di riferimento: applica solo yaw su Z base_link.
    Il task 03 ruota il cubo DOPO il lift.
    Il task 09 ripristina yaw(0) alla quota di trasporto, prima del place.
    """
    obj = PoseTarget(_object_xyz(req), cfg.grasp_quaternion_xyzw)
    # Se i due click quasi coincidono, niente traslazione laterale apparente:
    # evita che il solo offset storico Y faccia muovere il cubo di 6 cm.
    # destination_xyz = obj.xyz if _in_place(req) else req.place_xyz
    dst = PoseTarget(_dest_xyz(req), cfg.grasp_quaternion_xyzw)
    p = MotionPrimitives(start_pose, cfg)
    if req.task_id == "04":
        ax, ay, top_z = obj.xyz
        bx, by, _ = dst.xyz
        delta = math.hypot(bx - ax, by - ay)
        # if not (0.02 <= delta <= self.cfg.max_push_distance_m):
        #     raise ValueError("Distanza di push fuori intervallo; scegli una destinazione più vicina")
        dx, dy = (bx-ax)/delta, (by-ay)/delta
        offset = cfg.push_standoff_m
        start_x = ax - offset*dx
        start_y = ay - offset*dy - 0.07
        end_x = bx - offset*dx  
        end_y = by - offset*dy
        contact_z = top_z - cfg.cube_height_m + cfg.push_clearance_above_table_m
        # Gripper aperto, con orientazione top-down originale.
        q = cfg.grasp_quaternion_xyzw
        approach_push = PoseTarget((start_x, start_y, contact_z), q)
        start_push = PoseTarget((start_x, start_y, contact_z), q)
        end_push = PoseTarget((end_x, end_y, contact_z), q)
        p.reach(approach_push)
        p.pick()
        p.push(start_push, end_push)
        return p.plan()

    p.reach(obj)
    if req.task_id == "09" and req.rotation_deg != 0:
        p.yaw(req.rotation_deg)  # orienta il gripper IN HOVER, prima del grasp
        angled_object = PoseTarget(
            obj.xyz, _yaw_in_base(cfg.grasp_quaternion_xyzw, req.rotation_deg)
        )
        p.approaching(angled_object)  # la discesa NON ripristina lo yaw iniziale
    else:
        p.approaching(obj)

    p.pick()
    p.lift_up()
    if req.task_id == "03" and req.rotation_deg != 0:
        p.yaw(req.rotation_deg)  # ruota il cubo SOLLEVATO
    p.moving(dst)
    if req.task_id == "09":
        p.yaw(0.0)  # ripristina orientazione TCP alla quota di trasporto, PRIMA del place
    if req.task_id == "05":
        p.place_on_stack(dst)
    else:
        p.placing(dst)
    p.retract()
    return p.plan()
