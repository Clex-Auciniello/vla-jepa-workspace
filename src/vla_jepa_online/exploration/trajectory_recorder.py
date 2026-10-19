"""Event-triggered, synchronized front + wrist RGB captures, PNG in RAM.

Camera callbacks always run, but an RGB pair is saved ONLY for an explicit
capture_step() request after robot movement/gripper completion. Each request
waits for a fresh synchronized pair with ROS stamps after the trigger.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Any


@dataclass(frozen=True)
class TrajectoryRecorderConfig:
    fps: float
    front_topic: str
    left_topic: str
    right_topic: str
    gripper_topic: str
    sync_queue_size: int
    sync_slop_s: float
    png_compression: int
    joint_states_topic: str = "/joint_states"
    joint_max_age_s: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrajectoryRecorderConfig":
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)["trajectory_recorder"]
        cfg = cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})
        if cfg.fps <= 0 or cfg.sync_queue_size < 1 or cfg.sync_slop_s <= 0:
            raise ValueError("fps, sync_queue_size e sync_slop_s devono essere positivi")
        if not 0 <= cfg.png_compression <= 9:
            raise ValueError("png_compression deve essere tra 0 e 9")
        if cfg.joint_max_age_s <= 0 or not cfg.joint_states_topic:
            raise ValueError("joint_states_topic o joint_max_age_s non validi")
        topics = (
            cfg.front_topic,
            cfg.left_topic,
            cfg.right_topic,
            cfg.gripper_topic,
        )

        if any(not topic or not topic.strip() for topic in topics):
            raise ValueError("I quattro topic RGB devono essere non vuoti")

        if len(set(topics)) != 4:
            raise ValueError("I quattro topic RGB devono essere distinti")
        return cfg


@dataclass(frozen=True)
class RecordedFrame:
    """One synchronized pair, with per-step telemetry and executed command."""

    index: int  # 1-based, consecutive, independent of the ROS stamp
    front_png: bytes
    gripper_png: bytes
    front_stamp_ns: int
    gripper_stamp_ns: int
    event: str = "move"
    segment_id: int | None = None
    motion_label: str | None = None
    robot_state: dict[str, Any] | None = None
    commanded_action: tuple[float, ...] | None = None  # xyz, xyzw, gripper (0/1)
    left_png: bytes | None = None
    right_png: bytes | None = None
    left_stamp_ns: int | None = None
    right_stamp_ns: int | None = None

    @property
    def sync_error_ms(self) -> float:
        stamps = [self.front_stamp_ns, self.gripper_stamp_ns]

        if self.left_stamp_ns is not None:
            stamps.append(self.left_stamp_ns)

        if self.right_stamp_ns is not None:
            stamps.append(self.right_stamp_ns)

        return (max(stamps) - min(stamps)) / 1e6


@dataclass(frozen=True)
class RecordedEpisode:
    fps: float | None  # None in event-triggered mode: no temporal FPS limit
    frames: tuple[RecordedFrame, ...]

    @property
    def duration_s(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        first = (self.frames[0].front_stamp_ns + self.frames[0].gripper_stamp_ns) / 2
        last = (self.frames[-1].front_stamp_ns + self.frames[-1].gripper_stamp_ns) / 2
        return (last - first) / 1e9


def decode_rgb_pair(frame: RecordedFrame):
    """Return two RGB uint8 NumPy arrays; useful later in online_batch_builder."""
    import cv2
    import numpy as np

    def decode(data: bytes):
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("PNG frame non decodificabile")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return decode(frame.front_png), decode(frame.gripper_png)


class FrameCaptureError(RuntimeError):
    """No valid post-trigger synchronized camera pair was captured in time."""


class TrajectoryRecorder:
    """Event-triggered RAM recorder with synchronized ROS Image subscriptions.

    The ROS executor must spin in another thread. The caller invokes
    capture_step() only after GoToPose / gripper completion; start/stop delimit
    the episode, but do not enable any automatic frame buffering.
    """

    def __init__(self, node, config: TrajectoryRecorderConfig,
                 state_reader: Callable[[RecordedFrame], dict[str, Any]] | None = None):
        import cv2
        import message_filters
        from cv_bridge import CvBridge
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.node = node
        self.cfg = config
        self.state_reader = state_reader
        self.bridge = CvBridge()
        self._cv2 = cv2
        self._condition = threading.Condition()
        self._recording = False
        self._frames: list[RecordedFrame] = []
        self._last_stamp_ns = -1
        self._capture_token = 0
        self._completed_token = 0
        self._pending: tuple[int, str, int, int | None, str | None] | None = None
        self._encoding_token: int | None = None
        self._decode_errors = 0

        self._front_sub = message_filters.Subscriber(
            node, Image, config.front_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self._left_sub = message_filters.Subscriber(
            node, Image, config.left_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self._right_sub = message_filters.Subscriber(
            node, Image, config.right_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self._gripper_sub = message_filters.Subscriber(
            node, Image, config.gripper_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._front_sub,
                self._left_sub,
                self._right_sub,
                self._gripper_sub,
            ],
            queue_size=config.sync_queue_size,
            slop=config.sync_slop_s,
        )

        self._sync.registerCallback(self._on_group)

    def start(self) -> None:
        """Start an episode with an empty frame list; no automatic captures."""
        with self._condition:
            if self._recording:
                raise RuntimeError("La registrazione e' gia' attiva")
            self._frames = []
            self._last_stamp_ns = -1
            self._decode_errors = 0
            self._pending = None
            self._encoding_token = None
            self._recording = True

    def capture_step(self, event: str = "move", *, segment_id: int | None = None,
                     motion_label: str | None = None, timeout_s: float = 5.0,
                     target_pose: object | None = None,
                     gripper_closed: bool | None = None) -> RecordedFrame:
        """Block until a NEW synchronized pair recorded after this request.

        Both camera header timestamps must be newer than the local ROS clock at
        the trigger, as well as all previously saved image timestamps. This
        avoids including a stale callback from before GoToPose completed.
        The executor supplies segment_id and motion_label; the recorder never
        infers directions or constructs a complete language instruction.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s deve essere positivo")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            if not self._recording:
                raise FrameCaptureError("Recorder non attivo")
            if self._pending is not None:
                raise FrameCaptureError("Un'altra cattura e' gia' in attesa")
            self._capture_token += 1
            token = self._capture_token
            after_ns = max(self.node.get_clock().now().nanoseconds, self._last_stamp_ns)
            self._pending = (token, event, after_ns, segment_id, motion_label)
            while self._completed_token != token:
                if not self._recording:
                    raise FrameCaptureError("Recorder fermato durante la cattura")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._pending is not None and self._pending[0] == token:
                        self._pending = None
                    raise FrameCaptureError(
                        f"Nessuna coppia RGB front/gripper nuova entro {timeout_s:.1f}s "
                        f"dopo l'evento {event!r}"
                    )
                self._condition.wait(timeout=remaining)
            recorded = self._frames[-1]

        # Lo stato e' campionato dopo la coppia di immagini, fuori dal lock ROS.
        # Separatamente conserviamo il target eseguito dall'executor (non il
        # TCP misurato): stessa convenzione xyz,xyzw,gripper normalizzato.
        try:
            commanded = None
            if target_pose is not None:
                if gripper_closed is None:
                    raise ValueError("target_pose richiede gripper_closed")
                commanded = (
                    *(float(v) for v in target_pose.xyz),
                    *(float(v) for v in target_pose.quaternion_xyzw),
                    float(bool(gripper_closed)),
                )
                if len(commanded) != 8:
                    raise ValueError("Comando non ha dimensione 8")
            state = self.state_reader(recorded) if self.state_reader is not None else None
            if self.state_reader is not None and not isinstance(state, dict):
                raise ValueError("state_reader non ha restituito un dizionario")
        except Exception as exc:
            raise FrameCaptureError(f"Cattura stato/comando robot fallita: {exc}") from exc
        with self._condition:
            updated = replace(recorded, robot_state=state, commanded_action=commanded)
            if not self._frames or self._frames[-1].index != recorded.index:
                raise FrameCaptureError("Ordine catture non coerente")
            self._frames[-1] = updated
            return updated

    def stop(self) -> RecordedEpisode:
        """Stop captures; return immutable frame references in RAM."""
        with self._condition:
            self._recording = False
            self._pending = None
            self._condition.notify_all()
            return RecordedEpisode(None, tuple(self._frames))

    def clear(self) -> None:
        """Release recorder frame references after stop()."""
        with self._condition:
            if self._recording:
                raise RuntimeError("Fermare la registrazione prima di clear()")
            self._frames = []
            self._last_stamp_ns = -1
            self._pending = None

    @property
    def frame_count(self) -> int:
        with self._condition:
            return len(self._frames)

    @property
    def decode_errors(self) -> int:
        with self._condition:
            return self._decode_errors

    @staticmethod
    def _stamp_ns(msg) -> int:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _on_group(self, front_msg, left_msg, right_msg, gripper_msg) -> None:
        """Registra un gruppo RGB sincronizzato dopo un trigger esplicito."""

        messages = {
            "front": front_msg,
            "left": left_msg,
            "right": right_msg,
            "gripper": gripper_msg,
        }

        stamps = {
            name: self._stamp_ns(msg)
            for name, msg in messages.items()
        }

        if min(stamps.values()) <= 0:
            return

        with self._condition:
            if (
                not self._recording
                or self._pending is None
                or self._encoding_token is not None
            ):
                return

            token, event, after_ns, segment_id, motion_label = self._pending

            # Tutte le immagini devono essere successive al trigger.
            if min(stamps.values()) <= after_ns:
                return

            self._encoding_token = token

        try:
            compression = [
                int(self._cv2.IMWRITE_PNG_COMPRESSION),
                self.cfg.png_compression,
            ]

            pngs = {}

            for name, msg in messages.items():
                rgb = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="rgb8"
                )

                bgr = self._cv2.cvtColor(
                    rgb, self._cv2.COLOR_RGB2BGR
                )

                ok, buffer = self._cv2.imencode(
                    ".png", bgr, compression
                )

                if not ok:
                    raise ValueError(f"Codifica PNG {name} fallita")

                pngs[name] = buffer.tobytes()

        except Exception as exc:
            with self._condition:
                self._decode_errors += 1

                if self._encoding_token == token:
                    self._encoding_token = None

            self.node.get_logger().warning(
                f"Recorder: gruppo RGB scartato: {exc}"
            )
            return

        with self._condition:
            if self._encoding_token == token:
                self._encoding_token = None

            if (
                not self._recording
                or self._pending is None
                or self._pending[0] != token
            ):
                return

            self._frames.append(
                RecordedFrame(
                    index=len(self._frames) + 1,
                    front_png=pngs["front"],
                    left_png=pngs["left"],
                    right_png=pngs["right"],
                    gripper_png=pngs["gripper"],
                    front_stamp_ns=stamps["front"],
                    left_stamp_ns=stamps["left"],
                    right_stamp_ns=stamps["right"],
                    gripper_stamp_ns=stamps["gripper"],
                    event=event,
                    segment_id=segment_id,
                    motion_label=motion_label,
                )
            )

            self._last_stamp_ns = max(stamps.values())
            self._completed_token = token
            self._pending = None
            self._condition.notify_all()
