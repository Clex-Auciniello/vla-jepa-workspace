"""Record synchronized front + wrist RGB camera frames for an exploration episode.

No robot commands, frame-window selection, disk writes, or training logic.
The client must spin the supplied rclpy Node in a separate executor thread.
Frames are numbered 1..N; ROS camera stamps are used to synchronize and rate-limit
recording, not as the user-facing frame numbers. Original stamps are preserved.
"""

from __future__ import annotations

import threading
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
    """One synchronized pair; JPEG bytes can be decoded to RGB on the server."""

    index: int  # 1-based, consecutive, independent of the ROS stamp
    front_jpeg: bytes
    gripper_jpeg: bytes
    front_stamp_ns: int
    gripper_stamp_ns: int

    @property
    def sync_error_ms(self) -> float:
        return abs(self.front_stamp_ns - self.gripper_stamp_ns) / 1e6


@dataclass(frozen=True)
class RecordedEpisode:
    fps: float  # requested maximum recording rate, not guaranteed observed FPS
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


class TrajectoryRecorder:
    """RAM-only recorder using two synchronized ROS Image subscriptions.

    The ROS executor must keep spinning while the robot executes GoToPose.
    start()/stop()/clear() are safe to call from the client thread.
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
        self._lock = threading.Lock()
        self._recording = False
        self._frames: list[RecordedFrame] = []
        self._first_stamp_ns: int | None = None
        self._last_slot = -1
        self._decode_errors = 0

        # Keep subscriptions and synchronizer alive for the entire client session.
        self._front_sub = message_filters.Subscriber(
            node, Image, config.front_topic, qos_profile=qos_profile_sensor_data,
        )
        self._gripper_sub = message_filters.Subscriber(
            node, Image, config.gripper_topic, qos_profile=qos_profile_sensor_data,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._front_sub, self._gripper_sub],
            queue_size=config.sync_queue_size,
            slop=config.sync_slop_s,
        )
        self._sync.registerCallback(self._on_pair)

    def start(self) -> None:
        """Start a fresh episode; previously buffered data is discarded."""
        with self._lock:
            if self._recording:
                raise RuntimeError("La registrazione e' gia' attiva")
            self._frames = []
            self._first_stamp_ns = None
            self._last_slot = -1
            self._decode_errors = 0
            self._recording = True

    def stop(self) -> RecordedEpisode:
        """Freeze recording and return references to the compressed RAM frames."""
        with self._lock:
            self._recording = False
            return RecordedEpisode(self.cfg.fps, tuple(self._frames))

    def clear(self) -> None:
        """Release the recorder's frame references after sending or discarding."""
        with self._lock:
            if self._recording:
                raise RuntimeError("Fermare la registrazione prima di clear()")
            self._frames = []
            self._first_stamp_ns = None
            self._last_slot = -1

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def decode_errors(self) -> int:
        with self._lock:
            return self._decode_errors

    @staticmethod
    def _stamp_ns(msg) -> int:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _on_pair(self, front_msg, gripper_msg) -> None:
        """Called by ApproximateTimeSynchronizer with matching ROS stamps."""
        front_stamp = self._stamp_ns(front_msg)
        gripper_stamp = self._stamp_ns(gripper_msg)
        if front_stamp <= 0 or gripper_stamp <= 0:
            # A valid ROS camera source should populate header.stamp.
            return
        pair_stamp = (front_stamp + gripper_stamp) // 2

        with self._lock:
            if not self._recording:
                return
            start_ns = self._first_stamp_ns if self._first_stamp_ns is not None else pair_stamp
            if pair_stamp < start_ns:
                return
            # Time slots cap recording to ~fps without assuming camera FPS is exact.
            # Missing frames create gaps in slots, NOT invented duplicate frames.
            slot = int((pair_stamp - start_ns) * self.cfg.fps / 1e9 + 0.5)
            if slot <= self._last_slot:
                return

        try:
            # CvBridge yields RGB; OpenCV imencode expects BGR.
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
            with self._lock:
                self._decode_errors += 1
            self.node.get_logger().warning(f"Recorder: coppia RGB scartata: {exc}")
            return

        with self._lock:
            # Stop may have been called during JPEG encoding.
            if not self._recording:
                return
            if self._first_stamp_ns is None:
                self._first_stamp_ns = pair_stamp
                slot = 0
            else:
                slot = int((pair_stamp - self._first_stamp_ns) * self.cfg.fps / 1e9 + 0.5)
            if slot <= self._last_slot:
                return
            self._last_slot = slot
            self._frames.append(RecordedFrame(
                index=len(self._frames) + 1,
                front_jpeg=front_buf.tobytes(),
                gripper_jpeg=gripper_buf.tobytes(),
                front_stamp_ns=front_stamp,
                gripper_stamp_ns=gripper_stamp,
            ))
