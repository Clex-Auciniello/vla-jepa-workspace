"""Linear Cartesian micro-waypoints, following ScriptController geometry_utils.

The target endpoint is always included; quaternion orientation is SLERP-interpolated.
As in ScriptController, min_step is the *nominal minimum* separation when the
whole segment is longer than min_step, not a maximum displacement guarantee.
"""

from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

GripperPose = namedtuple("GripperPose", ["position", "orientation"])


def slerp(q0, q1, t):
    """Spherical linear interpolation between xyzw quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def compute_num_steps(distance, min_step):
    """ScriptController's subdivision rule: floor(distance / min_step), min 1."""
    if not math.isfinite(float(min_step)) or min_step <= 0:
        raise ValueError("min_step deve essere positivo e finito")
    if distance <= min_step:
        return 1
    return max(1, int(math.floor(distance / min_step)))


def build_linear_waypoints(start_pos, start_quat, end_pos, end_quat, min_step):
    """Return waypoints excluding start, ALWAYS including the exact end pose."""
    start_pos = np.asarray(start_pos, dtype=np.float64)
    end_pos = np.asarray(end_pos, dtype=np.float64)
    distance = float(np.linalg.norm(end_pos - start_pos))
    n_steps = compute_num_steps(distance, min_step)
    waypoints = []
    for i in range(1, n_steps + 1):
        t = i / n_steps
        pos = start_pos + (end_pos - start_pos) * t
        quat = slerp(start_quat, end_quat, t)
        waypoints.append(GripperPose(pos, quat))
    return waypoints
