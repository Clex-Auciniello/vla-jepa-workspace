"""Campionamento stato robot per ogni coppia RGB catturata.

Nessun comando al robot: legge /joint_states e TF dal ROSMoveItBackend.
I valori sono osservati DOPO la cattura RGB, quindi non sono garantiti
sincroni ai timestamp camera. I timestamp sono conservati per l'analisi.
"""

from __future__ import annotations

import math
import threading
from typing import Callable

from vla_jepa_online.exploration.trajectory_recorder import FrameCaptureError, RecordedFrame

# Ordine identico allo script_controller_node originale.
DEFAULT_ARM_JOINTS = (
    "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
DEFAULT_GRIPPER_JOINTS = ("robotiq_85_left_knuckle_joint",)


class RobotStateReader:
    def __init__(
        self,
        node,
        pose_getter: Callable[[], object],
        *,
        joint_states_topic: str = "/joint_states",
        joint_max_age_s: float = 1.0,
        arm_joints: tuple[str, ...] = DEFAULT_ARM_JOINTS,
        gripper_joints: tuple[str, ...] = DEFAULT_GRIPPER_JOINTS,
    ) -> None:
        from sensor_msgs.msg import JointState

        if joint_max_age_s <= 0:
            raise ValueError("joint_max_age_s deve essere positivo")
        self.node = node
        self.pose_getter = pose_getter
        self.joint_max_age_ns = int(joint_max_age_s * 1e9)
        self.arm_joints = arm_joints
        self.gripper_joints = gripper_joints
        self._lock = threading.Lock()
        self._latest = None
        # Anche questa subscription riceve callback mentre il ROS executor gira.
        self._subscription = node.create_subscription(
            JointState, joint_states_topic, self._on_joint_state, 10,
        )

    def _on_joint_state(self, msg) -> None:
        with self._lock:
            self._latest = msg

    @staticmethod
    def _stamp_ns(msg) -> int:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def read(self, frame: RecordedFrame) -> dict:
        with self._lock:
            msg = self._latest
        if msg is None:
            raise FrameCaptureError("Nessun /joint_states ricevuto alla cattura")

        joint_stamp_ns = self._stamp_ns(msg)
        now_ns = int(self.node.get_clock().now().nanoseconds)
        if joint_stamp_ns <= 0:
            raise FrameCaptureError("/joint_states senza timestamp ROS valido")
        age_ns = now_ns - joint_stamp_ns
        if abs(age_ns) > self.joint_max_age_ns:
            raise FrameCaptureError(
                f"/joint_states non recente: offset={age_ns / 1e6:.1f} ms"
            )
        positions = dict(zip(msg.name, msg.position))
        velocities = dict(zip(msg.name, msg.velocity))

        def values(names: tuple[str, ...], table: dict, field: str) -> list[float]:
            missing = [name for name in names if name not in table]
            if missing:
                raise FrameCaptureError(f"{field}: giunti mancanti da /joint_states: {missing}")
            result = [float(table[name]) for name in names]
            if not all(math.isfinite(v) for v in result):
                raise FrameCaptureError(f"{field}: valori non finiti")
            return result

        try:
            pose = self.pose_getter()
            xyz = [float(v) for v in pose.xyz]
            xyzw = [float(v) for v in pose.quaternion_xyzw]
        except Exception as exc:
            raise FrameCaptureError(f"TF TCP non disponibile durante registrazione: {exc}") from exc
        if (len(xyz) != 3 or len(xyzw) != 4
                or not all(math.isfinite(v) for v in xyz + xyzw)):
            raise FrameCaptureError("Posa TCP non valida")

        return {
            "joint_names": list(self.arm_joints),
            "joint_pos": values(self.arm_joints, positions, "joint_pos"),
            "joint_vel": values(self.arm_joints, velocities, "joint_vel"),
            "gripper_joint_names": list(self.gripper_joints),
            "gripper_qpos": values(self.gripper_joints, positions, "gripper_qpos"),
            "gripper_qvel": values(self.gripper_joints, velocities, "gripper_qvel"),
            "eef_pos": xyz,
            "eef_quat": xyzw,
            "base_frame": "base_link",
            "tcp_frame": "tcp_link",
            "joint_stamp_ns": joint_stamp_ns,
            "state_sample_stamp_ns": now_ns,
            "front_stamp_ns": frame.front_stamp_ns,
            "gripper_stamp_ns": frame.gripper_stamp_ns,
        }
