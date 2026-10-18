"""Pytest offline: non servono ROS, robot, Flask né GPU.

Eseguire nell'ambiente in cui il package vla_jepa_online è importabile.
"""
import io
import json
import pickle
import sys
import types
import zipfile

import cv2
import numpy as np
import pytest

from vla_jepa_online.exploration.robot_executor import PoseTarget, RobotExecutor, RobotExecutorConfig
from vla_jepa_online.exploration.trajectory_recorder import RecordedEpisode, RecordedFrame
from vla_jepa_online.plays.plays import TASKS, PlayRequest, build_play_plan, semantic_prompt
from vla_jepa_online.plays.primitives import PrimitiveConfig
from vla_jepa_online.plays.plays_trajectory_saver import PlaysTrajectorySaver
from vla_jepa_online.plays.plays_transport import PlaysTrainingTransport

Q = PrimitiveConfig().grasp_quaternion_xyzw
START = PoseTarget((-0.18, 0.3, 0.28), Q)


def req(task):
    return PlayRequest(task, "red", "blue" if task == "05" else None,
                       (100, 100), (140, 100), (-0.18, 0.3, 0.13),
                       (-0.11, 0.32, 0.13), 0.0 if task in ("04", "08") else -0.06)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_all_tasks_build_and_prompt(task):
    request = req(task)
    plan = build_play_plan(START, request, PrimitiveConfig())
    assert plan.commands
    assert plan.start_pose == START
    assert "red cube" in semantic_prompt(request)
    assert plan.waypoint_ids == ()
    assert "blue cube" in semantic_prompt(request) if task == "05" else True


class MockBackend:
    def __init__(self):
        self.pose = START
        self.calls = []

    def move_to_pose(self, target):
        self.pose = target
        self.calls.append(("move", target))

    def set_gripper(self, close):
        self.calls.append(("gripper", close))


def test_record_every_motion_rotation_gripper():
    backend = MockBackend()
    executor = RobotExecutor(backend, RobotExecutorConfig(
        ros_timeout_s=5, gripper_action_name="dummy", gripper_open_position=0.1,
        gripper_closed_position=255, gripper_max_effort=50,
    ))
    plan = build_play_plan(START, req("03"), PrimitiveConfig())
    captures = []
    executor.execute(plan, on_step=lambda event, **fields: captures.append((event, fields)))
    assert len(captures) == len(backend.calls)
    assert any(event == "rotate" for event, _ in captures)
    assert any(event == "gripper_close" for event, _ in captures)
    assert any(event == "gripper_open" for event, _ in captures)
    assert all(fields["target_pose"] is not None for _, fields in captures)


def test_invalid_push_distance_rejected():
    bad = PlayRequest("04", "red", None, (10, 10), (12, 10),
                      (0.1, 0.2, 0.13), (0.101, 0.201, 0.13), 0.0)
    with pytest.raises(ValueError, match="Distanza di push"):
        build_play_plan(START, bad, PrimitiveConfig())


# Fake minimale del savers.Trajectory usato per verificare l'intero contratto PKL.
class FakeTrajectory:
    def __init__(self, config_str=""):
        self.config_str = config_str
        self.steps = []

    def append(self, **kwargs):
        self.steps.append(kwargs)

    def __len__(self):
        return len(self.steps)

    def get(self, i, decompress=False):
        return self.steps[i]

    def save(self, target, **metadata):
        with open(target, "wb") as f:
            pickle.dump({"traj": self, **metadata}, f)


def test_pkl_roundtrip_and_plays_manifest(monkeypatch, tmp_path):
    fake = types.ModuleType("savers")
    fake.Trajectory = FakeTrajectory
    fake.FakeTrajectory = FakeTrajectory
    FakeTrajectory.__module__ = "savers"
    monkeypatch.setitem(sys.modules, "savers", fake)
    blank_bgr = np.zeros((24, 32, 3), dtype=np.uint8)
    blank_bgr[:, :, 2] = 255
    valid, compressed = cv2.imencode(".png", blank_bgr)
    assert valid
    png = compressed.tobytes()
    state = {
        "joint_pos": [0.0]*6, "joint_vel": [0.0]*6,
        "gripper_qpos": [0.0], "gripper_qvel": [0.0],
        "eef_pos": list(START.xyz), "eef_quat": list(Q),
    }
    action = (*START.xyz, *Q, 0.0)
    frames = tuple(RecordedFrame(i, png, png, i*100000000, i*100000000,
                                 event="move", segment_id=1, motion_label="Lift the cube",
                                 robot_state=state, commanded_action=action)
                   for i in range(1, 9))
    saver = PlaysTrajectorySaver(tmp_path)
    path = saver.save("P00001", RecordedEpisode(None, frames), req("01"),
                      operator_success=True, operator_approved=True)
    meta, reconstructed = saver.load_episode(path)
    assert meta["source"] == "plays"
    assert meta["episode_prompt"] == semantic_prompt(req("01"))
    assert len(reconstructed.frames) == 8
    payload = PlaysTrainingTransport.encode(meta, reconstructed)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["prompt_mode"] == "episode_prompt"
        assert manifest["terminal_padding"] == "repeat_last"
        assert len(manifest["frames"]) == 8
        assert z.read("frames/front/000001.png").startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        saver.save("P00001", RecordedEpisode(None, frames), req("01"),
                   operator_success=True, operator_approved=True)
