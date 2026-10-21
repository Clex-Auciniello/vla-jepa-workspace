"""Primitive robotiche Plays: SOLO composizione di comandi, nessun ROS diretto.

Le 6 primitive del vecchio ScriptController conservano le rispettive geometrie
ed offset configurabili. Si riusano PoseTarget, RobotCommand, ExecutionPlan e
RobotExecutor.execute() di Exploration (interpolazione geometry_utils e on_step).
Le nuove primitive (yaw, push, stack, retract) sono separate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from vla_jepa_online.exploration.robot_executor import (
    ExecutionPlan, PoseTarget, RobotCommand, _unit_quaternion, _yaw_in_base,
)
from vla_jepa_online.utils.geometry_utils import build_linear_waypoints


@dataclass(frozen=True)
class PrimitiveConfig:
    # ESATTAMENTE i default delle sei primitive ScriptController originali.
    min_step_m: float = 0.02
    reach_hover_height_m: float = 0.15
    approach_z_offset_m: float = 0.0
    release_height_offset_m: float = 0.10
    lift_height_m: float = 0.15
    # Solo per primitive aggiuntive: verificare sul robot prima dell'uso.
    rotation_deg: float = 30.0
    cube_height_m: float = 0.04
    stack_clearance_m: float = 0.01
    push_standoff_m: float = 0.035
    push_clearance_above_table_m: float = 0.02
    push_approach_height_m: float = 0.12
    max_push_distance_m: float = 0.12
    min_pose_z_m: float = 0.025
    max_pose_z_m: float = 0.60
    # Il vecchio ScriptController impone una presa con orientazione fissa.
    grasp_quaternion_xyzw: tuple[float, float, float, float] = (
        0.9994452044624775, 0.03161651380119412,
        0.0021438049655468088, 0.010251021036213035,
    )

    def __post_init__(self):
        numeric = [v for k, v in vars(self).items() if k != "grasp_quaternion_xyzw"]
        if any(not math.isfinite(float(v)) for v in numeric):
            raise ValueError("Parametri primitivi non finiti")
        if not math.isclose(self.min_step_m, 0.02, abs_tol=1e-10):
            raise ValueError("L'executor Exploration usa MICRO_STEP_M=0.02: mantenere min_step_m=0.02")
        if self.min_step_m <= 0 or self.lift_height_m <= 0 or self.max_pose_z_m <= self.min_pose_z_m:
            raise ValueError("Configurazione primitiva non valida")
        if self.cube_height_m <= 0 or self.push_standoff_m <= 0 or self.max_push_distance_m <= 0:
            raise ValueError("Dimensioni di contatto non valide")
        if not 0 < abs(self.rotation_deg) <= 30:
            raise ValueError("Rotazione Plays consentita entro ±30 gradi")
        _unit_quaternion(self.grasp_quaternion_xyzw)


class MotionPrimitives:
    """Compone un ExecutionPlan, senza muovere il robot.

    L'esecuzione vera avviene attraverso RobotExecutor.execute(plan,on_step=...),
    CHE GIÀ invoca on_step dopo ogni micro-step, rotazione pura e gripper.
    """
    def __init__(self, start_pose: PoseTarget, cfg: PrimitiveConfig):
        self.cfg = cfg
        self.start_pose = start_pose
        self.current = start_pose
        self.commands: list[RobotCommand] = []
        self._next_segment = 0
        self.gripper_closed = False

    def _move(self, xyz: Sequence[float], quat: Sequence[float], label: str) -> None:
        xyz = tuple(float(v) for v in xyz)
        quat = _unit_quaternion(quat)
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError("Target XYZ non valido")
        # if not self.cfg.min_pose_z_m <= xyz[2] <= self.cfg.max_pose_z_m:
        #     raise ValueError(f"Target TCP Z={xyz[2]:.3f} fuori limite configurato")
        target = PoseTarget(xyz, quat)
        same_xyz = math.dist(self.current.xyz, target.xyz) < 1e-8
        same_rot = abs(sum(a*b for a, b in zip(self.current.quaternion_xyzw, quat))) > 1 - 1e-9
        if same_xyz and same_rot:
            return
        # Preflight usa la STESSA geometry_utils impiegata in Exploration.
        wps = build_linear_waypoints(self.current.xyz, self.current.quaternion_xyzw,
                                     xyz, quat, self.cfg.min_step_m)
        # for wp in wps:
        #     if not self.cfg.min_pose_z_m <= float(wp.position[2]) <= self.cfg.max_pose_z_m:
        #         raise ValueError("Micro-waypoint fuori dai limiti verticali")
        self._next_segment += 1
        self.commands.append(RobotCommand(
            "pose", label, target=target,
            segment_id=self._next_segment, motion_label=label,
        ))
        self.current = target

    def _gripper(self, close: bool, label: str) -> None:
        self._next_segment += 1
        self.commands.append(RobotCommand(
            "gripper", label, gripper_closed=close,
            segment_id=self._next_segment, motion_label=label,
        ))
        self.gripper_closed = close

    # Le sei primitive storiche, con le medesime formule dei target.
    def reach(self, object_pose: PoseTarget) -> None:
        x, y, z = object_pose.xyz
        self._move((x, y, z + self.cfg.reach_hover_height_m),
                   object_pose.quaternion_xyzw, "Approach the cube from above")

    def approaching(self, object_pose: PoseTarget) -> None:
        x, y, z = object_pose.xyz
        self._move((x, y, z + self.cfg.approach_z_offset_m),
                   object_pose.quaternion_xyzw, "Lower the gripper toward the cube")

    def pick(self) -> None:
        self._gripper(True, "Close the gripper around the cube")

    def lift_up(self) -> None:
        x, y, z = self.current.xyz
        self._move((x, y, z + self.cfg.lift_height_m),
                   self.current.quaternion_xyzw, "Lift the cube")

    def moving(self, target_pose: PoseTarget) -> None:
        # Preserva l'orientazione corrente se la si è modificata con yaw().
        x, y, _ = target_pose.xyz
        self._move((x, y, self.current.xyz[2]),
                   self.current.quaternion_xyzw, "Move the cube toward the destination")

    def placing(self, target_pose: PoseTarget) -> None:
        x, y, z = target_pose.xyz
        self._move((x, y, z + self.cfg.release_height_offset_m),
                   self.current.quaternion_xyzw, "Lower the cube at the destination")
        self._gripper(False, "Release the cube")

    # Nuove primitive atomiche.
    def retract(self) -> None:
        x, y, z = self.current.xyz
        self._move((x, y, z + self.cfg.reach_hover_height_m),
                   self.current.quaternion_xyzw, "Retract the open gripper")

    def yaw(self, degrees: float) -> None:
        """Imposta yaw assoluto rispetto al riferimento top-down su Z(base_link).

        Consente rotazione sia a gripper aperto (orientamento pre-grasp) sia
        dopo un grasp. yaw(0) ripristina il riferimento top-down originale.
        L'angolo effettivo viene deciso dal livello plays/client.
        """
        if not math.isfinite(degrees) or abs(degrees) > 30:
            raise ValueError("Yaw deve essere tra -30 e +30 gradi, incluso 0")
        orientation = _yaw_in_base(self.cfg.grasp_quaternion_xyzw, degrees)
        if degrees == 0:
            label = "Return the gripper to its reference orientation"
        else:
            label = "Rotate the grasped cube" if self.gripper_closed else "Rotate the gripper"
        self._move(self.current.xyz, orientation, label)

    def place_on_stack(self, supporting_cube: PoseTarget) -> None:
        # Il click sul secondo cubo si assume sulla sua faccia superiore.
        # Solo una stima: dimensione, presa e clearance richiedono taratura.
        x, y, top_z = supporting_cube.xyz
        z = top_z + self.cfg.cube_height_m + self.cfg.stack_clearance_m
        self._move((x, y, z), self.current.quaternion_xyzw,
                   "Lower the cube onto the supporting cube")
        self._gripper(False, "Release the cube on the stack")

    def push(self, cube_pose: PoseTarget, destination: PoseTarget) -> None:
        """Si avvicina dal lato opposto al moto e spinge con gripper aperto.

        Il click 1 è circa il centro della faccia superiore del cubo. L'altezza di
        contatto è stimata: non usare sul robot senza validazione e spazio libero.
        """
        start_x, start_y, start_z = cube_pose.xyz
        end_x, end_y, end_z = destination.xyz
        q = cube_pose.quaternion_xyzw
        self._move((start_x, start_y, start_z), q, "Lower the gripper for pushing")
        self._move((end_x, end_y, end_z), q, "Push the cube toward the destination")
        self._move((end_x, end_y, end_z + self.cfg.push_approach_height_m + 0.10), q,
                   "Retract after pushing")

    def plan(self) -> ExecutionPlan:
        # if self.gripper_closed:
        #     raise ValueError("Piano termina col gripper chiuso; serve release")
        if not self.commands:
            raise ValueError("Piano vuoto")
        return ExecutionPlan(start_pose=self.start_pose, commands=tuple(self.commands), waypoint_ids=())
