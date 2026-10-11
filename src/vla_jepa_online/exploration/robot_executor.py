"""UR5e exploration executor: MotionPlan -> GoToPose / GoHome / Robotiq.

Z1 auxiliary stops in OCCUPIED come from the nominal grid (no perturbation).
No video, sampling, episode IDs or coverage counts belong to this module.
The ROS node passed to ROSMoveItBackend must be spun by another thread.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from vla_jepa_online.exploration.motion_sampler import MotionPlan

# [x, y, z, qx, qy, qz, qw] in base_link, for tcp_link.
# This is the GoToPose after GoHome, NOT the GoHome joint configuration.
INITIAL_POSE = (
    -0.15552094619366708, 0.34869994018501943, 0.1532803451753288,
    0.9994452044624775, 0.03161651380119412,
    0.0021438049655468088, 0.010251021036213035,
)


class RobotExecutionError(RuntimeError):
    """Any execution failure invalidates the whole trajectory."""


@dataclass(frozen=True)
class PoseTarget:
    xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotCommand:
    kind: Literal["pose", "gripper"]
    description: str
    target: PoseTarget | None = None
    gripper_closed: bool | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    start_pose: PoseTarget
    commands: tuple[RobotCommand, ...]
    waypoint_ids: tuple[int, ...]  # only the 5 walker IDs, never auxiliaries


class RobotBackend(Protocol):
    def get_tcp_pose(self) -> PoseTarget: ...
    def move_home(self) -> None: ...
    def move_to_pose(self, pose: PoseTarget) -> None: ...
    def set_gripper(self, closed: bool) -> None: ...


def _unit_quaternion(q: Sequence[float]) -> tuple[float, float, float, float]:
    if len(q) != 4 or not all(math.isfinite(float(v)) for v in q):
        raise ValueError("Quaternion non valido")
    norm = math.sqrt(sum(float(v)**2 for v in q))
    if norm < 1e-10:
        raise ValueError("Quaternion nullo")
    return tuple(float(v) / norm for v in q)


def _yaw_in_base(q: Sequence[float], degrees: float) -> tuple[float, float, float, float]:
    """Absolute yaw offset about Z(base_link): q_target = q_z * q_initial."""
    x, y, z, w = _unit_quaternion(q)
    angle = math.radians(degrees) / 2
    s, c = math.sin(angle), math.cos(angle)
    return _unit_quaternion((c*x-s*y, s*x+c*y, c*z+s*w, c*w-s*z))


@dataclass(frozen=True)
class RobotExecutorConfig:
    ros_timeout_s: float
    gripper_action_name: str
    gripper_open_position: float
    gripper_closed_position: float
    gripper_max_effort: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotExecutorConfig":
        import yaml
        with Path(path).open("r", encoding="utf-8") as f:
            values = yaml.safe_load(f)["robot_executor"]
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})


class RobotExecutor:
    """Builds a deterministic command list, then runs it through a backend."""

    def __init__(self, backend: RobotBackend, config: RobotExecutorConfig):
        self.backend, self.cfg = backend, config

    def prepare(self) -> PoseTarget:
        """Called on first operator ENTER, before recording starts."""
        if hasattr(self.backend, "check_interfaces"):
            self.backend.check_interfaces()
        self.backend.move_home()
        initial = PoseTarget(tuple(INITIAL_POSE[:3]), _unit_quaternion(INITIAL_POSE[3:]))
        self.backend.move_to_pose(initial)
        self.backend.set_gripper(closed=False)
        return self.backend.get_tcp_pose()

    def build_commands(self, motion_plan: MotionPlan, start_pose: PoseTarget, *, grid: dict) -> ExecutionPlan:
        """Pure geometry. OCCUPIED detours: Zx->Z1->Z0 or Z0->Z1->Z1->Z0.

        Nominal XY indices (i,j), not perturbed Cartesian XY, determine alignment.
        All movements, including the auxiliaries, are ordinary GoToPose goals;
        the TCP path between goals is not constrained to be a straight line.
        """
        if motion_plan.table_mode not in ("empty", "occupied") or not motion_plan.steps:
            raise ValueError("MotionPlan vuoto o table_mode non valido")
        if grid.get("base_frame") != "base_link" or grid.get("tcp_frame") != "tcp_link":
            raise ValueError("Frame della griglia non compatibili")
        by_id = {int(w["id"]): w for w in grid["waypoints"]}
        by_ijk = {(int(w["i"]), int(w["j"]), int(w["k"])): w for w in grid["waypoints"]}
        if len(by_id) != len(grid["waypoints"]) or len(by_ijk) != len(grid["waypoints"]):
            raise ValueError("Griglia con waypoint duplicati")

        steps = motion_plan.steps
        waypoint_ids = tuple(int(s.waypoint_id) for s in steps)
        if len(set(waypoint_ids)) != len(waypoint_ids):
            raise ValueError("I waypoint esplorativi non devono ripetersi")
        for step in steps:
            w = by_id.get(int(step.waypoint_id))
            if w is None or (step.i, step.j, step.k) != (w["i"], w["j"], w["k"]):
                raise ValueError(f"Waypoint {step.waypoint_id} non corrisponde alla griglia")
            if (step.k == 0) != (step.level == "PICK"):
                raise ValueError(f"Livello del waypoint {step.waypoint_id} incoerente")

        def z1(step) -> tuple[float, float, float]:
            point = by_ijk.get((step.i, step.j, 1))
            if point is None:
                raise ValueError(f"Manca Z1 sopra waypoint {step.waypoint_id}")
            return tuple(float(v) for v in point["position"])

        occupied = motion_plan.table_mode == "occupied"
        initial_q = _unit_quaternion(INITIAL_POSE[3:])
        commands: list[RobotCommand] = []
        current_pose = start_pose
        yaw = 0.0
        previous = None

        def move(xyz: Sequence[float], description: str, *, yaw_deg: float | None = None):
            nonlocal current_pose
            orientation = _yaw_in_base(initial_q, yaw if yaw_deg is None else yaw_deg)
            target = PoseTarget(tuple(float(v) for v in xyz), orientation)
            # Dot product treats q and -q as the same rotation.
            same_orientation = abs(sum(a*b for a, b in zip(
                _unit_quaternion(current_pose.quaternion_xyzw), orientation
            ))) >= 1.0 - 1e-10
            if math.dist(current_pose.xyz, target.xyz) > 1e-6 or not same_orientation:
                commands.append(RobotCommand("pose", description, target=target))
            current_pose = target

        for step in steps:
            target_xyz = step.target_position  # perturbed point
            if occupied and step.k == 0:
                if previous is not None and previous.k == 0:
                    move(z1(previous), "Z1 nominale sopra la partenza")
                    move(z1(step), "Z1 nominale sopra la destinazione")
                elif previous is None or (previous.i, previous.j) != (step.i, step.j):
                    move(z1(step), "Z1 nominale sopra la destinazione")
                move(target_xyz, f"PICK {step.waypoint_id}")
            else:
                move(target_xyz, f"Waypoint {step.waypoint_id}")

            if step.rotate_here:
                yaw = float(step.yaw_offset_after_deg)
                move(target_xyz, f"Yaw al waypoint {step.waypoint_id}")
            elif abs(float(step.yaw_offset_after_deg) - yaw) > 0.0011:
                raise ValueError("Yaw cambia senza evento rotate_here")

            if step.toggle_gripper_here:
                commands.append(RobotCommand(
                    "gripper", f"{'CLOSE' if step.gripper_closed_after else 'OPEN'} al waypoint {step.waypoint_id}",
                    gripper_closed=bool(step.gripper_closed_after),
                ))
            previous = step

        return ExecutionPlan(start_pose, tuple(commands), waypoint_ids)

    def execute(self, plan: ExecutionPlan) -> tuple[int, ...]:
        """Called after recorder starts. Success means ALL commands succeeded.

        On any error, stop issuing commands, invalidate episode; no retry.
        Protective stop recovery is left entirely to the operator/tablet.
        """
        try:
            for command in plan.commands:
                if command.kind == "pose":
                    assert command.target is not None
                    self.backend.move_to_pose(command.target)
                else:
                    assert command.gripper_closed is not None
                    self.backend.set_gripper(command.gripper_closed)
        except Exception as exc:
            raise RobotExecutionError(f"Esplorazione interrotta: {exc}") from exc
        return plan.waypoint_ids


class ROSMoveItBackend:
    """ROS Jazzy adapter. Node must be spinning in a separate thread.

    Only existing set_robot_to_home, set_robot_to_pose services and Robotiq
    GripperCommand action are used. MoveItControllerNode remains unchanged.
    """

    def __init__(self, node, config: RobotExecutorConfig,
                 *, base_frame: str = "base_link", tcp_frame: str = "tcp_link"):
        try:
            from moveit_controller_srvs.srv import GoHome, GoToPose
            from control_msgs.action import GripperCommand
            from rclpy.action import ActionClient
            from tf2_ros import Buffer, TransformListener
        except ImportError as exc:
            raise RuntimeError("Necessario container ROS 2 con driver UR e MoveIt") from exc

        self.node, self.cfg = node, config
        self.base_frame, self.tcp_frame = base_frame, tcp_frame
        self.GoHome, self.GoToPose, self.GripperCommand = GoHome, GoToPose, GripperCommand
        self.home_client = node.create_client(GoHome, "set_robot_to_home")
        self.pose_client = node.create_client(GoToPose, "set_robot_to_pose")
        self.gripper_client = ActionClient(node, GripperCommand, config.gripper_action_name)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node, spin_thread=False)

    def check_interfaces(self) -> None:
        for client, name in ((self.home_client, "GoHome"), (self.pose_client, "GoToPose")):
            if not client.wait_for_service(timeout_sec=5.0):
                raise RobotExecutionError(f"Servizio non disponibile: {name}")
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            raise RobotExecutionError("Action Robotiq non disponibile")

    def _await(self, future, label: str):
        deadline = time.monotonic() + self.cfg.ros_timeout_s
        while not future.done():
            if time.monotonic() >= deadline:
                raise RobotExecutionError(
                    f"Timeout {label}: movimento potenzialmente ancora attivo, attendere recovery manuale"
                )
            time.sleep(0.05)
        if future.exception() is not None:
            raise RobotExecutionError(f"{label}: {future.exception()}")
        result = future.result()
        if result is None:
            raise RobotExecutionError(f"{label}: nessuna risposta")
        return result

    def _service(self, client, request, label):
        return self._await(client.call_async(request), label)

    def get_tcp_pose(self) -> PoseTarget:
        """Get latest available base_link -> tcp_link TF, allowing brief TF delays.

        Used only to read the start pose after the confirmed initial GoToPose.
        No arbitrary 2-second TF-age check: a stationary robot may legitimately
        have temporarily delayed TF updates. Missing TF still fails after 5 s.
        """
        from rclpy.duration import Duration
        from rclpy.time import Time
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                Time(),
                timeout=Duration(seconds=5.0),
            )
        except Exception as exc:
            raise RobotExecutionError(
                f"TF {self.base_frame} -> {self.tcp_frame} non disponibile dopo 5 s: {exc}"
            ) from exc
        p, q = transform.transform.translation, transform.transform.rotation
        return PoseTarget((p.x, p.y, p.z), _unit_quaternion((q.x, q.y, q.z, q.w)))

    def move_home(self) -> None:
        response = self._service(self.home_client, self.GoHome.Request(), "GoHome")
        if not response.success:
            raise RobotExecutionError(f"GoHome fallito: {response.message}")

    def move_to_pose(self, pose: PoseTarget) -> None:
        request = self.GoToPose.Request()
        request.pose.header.frame_id = self.base_frame
        request.pose.header.stamp = self.node.get_clock().now().to_msg()
        p = request.pose.pose.position
        p.x, p.y, p.z = pose.xyz
        q = request.pose.pose.orientation
        q.x, q.y, q.z, q.w = pose.quaternion_xyzw
        response = self._service(self.pose_client, request, "GoToPose")
        if not response.success:
            raise RobotExecutionError(f"GoToPose fallito: {response.message}")

    def set_gripper(self, closed: bool) -> None:
        from action_msgs.msg import GoalStatus
        goal = self.GripperCommand.Goal()
        goal.command.position = self.cfg.gripper_closed_position if closed else self.cfg.gripper_open_position
        goal.command.max_effort = self.cfg.gripper_max_effort
        handle = self._await(self.gripper_client.send_goal_async(goal), "Goal Robotiq")
        if not handle.accepted:
            raise RobotExecutionError("Goal Robotiq rifiutato")
        response = self._await(handle.get_result_async(), "Risultato Robotiq")
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            raise RobotExecutionError("Robotiq action fallita")
