"""Event-triggered, synchronized front + wrist RGB captures, JPEG in RAM.

Camera callbacks always run, but an RGB pair is saved ONLY for an explicit
capture_step() request after robot movement/gripper completion. Each request
waits for a fresh synchronized pair with ROS stamps after the trigger.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrajectoryRecorderConfig:
    fps: float
    front_topic: str
    gripper_topic: str
    sync_queue_size: int
    sync_slop_s: float
    jpeg_quality: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrajectoryRecorderConfig":
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)["trajectory_recorder"]
        cfg = cls(**{name: values[name] for name in cls.__dataclass_fields__})
        if cfg.fps <= 0 or cfg.sync_queue_size < 1 or cfg.sync_slop_s <= 0:
            raise ValueError("fps, sync_queue_size e sync_slop_s devono essere positivi")
        if not 1 <= cfg.jpeg_quality <= 100:
            raise ValueError("jpeg_quality deve essere compreso tra 1 e 100")
        if not cfg.front_topic or not cfg.gripper_topic or cfg.front_topic == cfg.gripper_topic:
            raise ValueError("I topic front e gripper devono essere distinti e non vuoti")
        return cfg


@dataclass(frozen=True)
class RecordedFrame:
    """One synchronized pair, with optional per-transition prompt metadata."""

    index: int  # 1-based, consecutive, independent of the ROS stamp
    front_jpeg: bytes
    gripper_jpeg: bytes
    front_stamp_ns: int
    gripper_stamp_ns: int
    event: str = "move"
    segment_id: int | None = None
    motion_label: str | None = None

    @property
    def sync_error_ms(self) -> float:
        return abs(self.front_stamp_ns - self.gripper_stamp_ns) / 1e6


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
            raise ValueError("JPEG frame non decodificabile")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return decode(frame.front_jpeg), decode(frame.gripper_jpeg)


class FrameCaptureError(RuntimeError):
    """No valid post-trigger synchronized camera pair was captured in time."""


class TrajectoryRecorder:
    """Event-triggered RAM recorder with synchronized ROS Image subscriptions.

    The ROS executor must spin in another thread. The caller invokes
    capture_step() only after GoToPose / gripper completion; start/stop delimit
    the episode, but do not enable any automatic frame buffering.
    """

    def __init__(self, node, config: TrajectoryRecorderConfig):
        import cv2
        import message_filters
        from cv_bridge import CvBridge
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.node = node
        self.cfg = config
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
            node, Image, config.front_topic, qos_profile=qos_profile_sensor_data,
        )
        self._gripper_sub = message_filters.Subscriber(
            node, Image, config.gripper_topic, qos_profile=qos_profile_sensor_data,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._front_sub, self._gripper_sub],
            queue_size=config.sync_queue_size, slop=config.sync_slop_s,
        )
        self._sync.registerCallback(self._on_pair)

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
                     motion_label: str | None = None, timeout_s: float = 5.0) -> RecordedFrame:
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
            return self._frames[-1]

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

    def _on_pair(self, front_msg, gripper_msg) -> None:
        """ROS callback: satisfy exactly ONE pending post-trigger capture."""
        front_stamp = self._stamp_ns(front_msg)
        gripper_stamp = self._stamp_ns(gripper_msg)
        if front_stamp <= 0 or gripper_stamp <= 0:
            return

        with self._condition:
            if not self._recording or self._pending is None or self._encoding_token is not None:
                return
            token, event, after_ns, segment_id, motion_label = self._pending
            if min(front_stamp, gripper_stamp) <= after_ns:
                return
            self._encoding_token = token

        try:
            front_rgb = self.bridge.imgmsg_to_cv2(front_msg, desired_encoding="rgb8")
            gripper_rgb = self.bridge.imgmsg_to_cv2(gripper_msg, desired_encoding="rgb8")
            quality = [int(self._cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality]
            front_ok, front_buf = self._cv2.imencode(
                ".jpg", self._cv2.cvtColor(front_rgb, self._cv2.COLOR_RGB2BGR), quality,
            )
            gripper_ok, gripper_buf = self._cv2.imencode(
                ".jpg", self._cv2.cvtColor(gripper_rgb, self._cv2.COLOR_RGB2BGR), quality,
            )
            if not front_ok or not gripper_ok:
                raise ValueError("Codifica JPEG non riuscita")
        except Exception as exc:
            with self._condition:
                self._decode_errors += 1
                if self._encoding_token == token:
                    self._encoding_token = None
            self.node.get_logger().warning(f"Recorder: coppia RGB scartata: {exc}")
            return

        with self._condition:
            if self._encoding_token == token:
                self._encoding_token = None
            if not self._recording or self._pending is None or self._pending[0] != token:
                return  # timed out/stopped while JPEG was being encoded
            self._frames.append(RecordedFrame(
                index=len(self._frames) + 1,
                front_jpeg=front_buf.tobytes(),
                gripper_jpeg=gripper_buf.tobytes(),
                front_stamp_ns=front_stamp,
                gripper_stamp_ns=gripper_stamp,
                event=event,
                segment_id=segment_id,
                motion_label=motion_label,
            ))
            self._last_stamp_ns = max(front_stamp, gripper_stamp)
            self._completed_token = token
            self._pending = None
            self._condition.notify_all()
