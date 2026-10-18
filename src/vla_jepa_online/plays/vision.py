"""Vision helpers for script_controller_node: loading the ZED camera
extrinsics calibration (see zed_camera/zed_camera_calibration/estimated_camera_positions.yaml),
pixel+depth -> 3D deprojection, and the interactive click UI used to pick the
target object / target placing pixel on the frontal camera frame.
"""
import cv2
import numpy as np
import yaml


def load_camera_calibration(calibration_path):
    """Load estimated_camera_positions.yaml: {camera_name: {position, orientation_matrix}}.

    Positions/orientations are expressed with respect to the raw ArUco marker
    origin/axes as returned by cv2.aruco's pose estimation (placed at the table
    center), per zed_camera/zed_camera_calibration/scripts/interactive_aruco_calibration.py.
    This is NOT necessarily the same frame as the ``table_0`` TF frame - see
    ARUCO_TO_TABLE0_ROTATION below for the fixed offset between the two.
    """
    with open(calibration_path, 'r') as f:
        raw = yaml.safe_load(f)

    calibration = {}
    for camera_name, entry in raw.items():
        calibration[camera_name] = {
            'position': np.array(entry['position'], dtype=np.float64),
            'orientation_matrix': np.array(entry['orientation_matrix'], dtype=np.float64),
        }
    return calibration


def get_camera_intrinsics(node, camera_info_topic, time_to_wait=2.0, retries=5):
    """Fetch a CameraInfo message once and return its 3x3 intrinsics matrix K."""
    import rclpy.wait_for_message
    from sensor_msgs.msg import CameraInfo
    flag = False
    msg = None
    attempts = retries
    while not flag and attempts > 0:
        flag, msg = rclpy.wait_for_message.wait_for_message(
            topic=camera_info_topic,
            msg_type=CameraInfo,
            node=node,
            time_to_wait=time_to_wait,
        )
        attempts -= 1

    if not flag:
        raise RuntimeError(f'Failed to receive CameraInfo on topic: {camera_info_topic}')

    return np.array(msg.k, dtype=np.float64).reshape((3, 3))


def robust_depth_at(depth_image, u, v, window=5):
    """Median depth (meters) in a small window around (u, v), ignoring NaN/inf/<=0."""
    h, w = depth_image.shape[:2]
    half = window // 2
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    patch = np.asarray(depth_image[v0:v1, u0:u1], dtype=np.float64).flatten()
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject_pixel(u, v, depth, camera_matrix):
    """Pinhole deprojection: pixel (u, v) + depth (m) -> 3D point in the camera
    optical frame (X right, Y down, Z forward), using intrinsics K."""
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.array([x, y, depth], dtype=np.float64)


def camera_point_to_aruco(point_cam, camera_calib_entry):
    """Apply the camera->ArUco extrinsic transform loaded from the calibration
    yaml (camera position/orientation expressed in the raw ArUco marker frame)."""
    R_cm = camera_calib_entry['orientation_matrix']
    t_cm = camera_calib_entry['position']
    return R_cm @ point_cam + t_cm


# Fixed extrinsic offset between the raw ArUco marker origin/axes (as returned by
# cv2.aruco's pose estimation, i.e. the frame estimated_camera_positions.yaml is
# expressed in) and the table_0 TF frame: zero translation, quaternion (x, y, z, w)
# = (0, 0, 1, 0), i.e. a 180 degree rotation about Z (X and Y flip, Z unchanged).
ARUCO_TO_TABLE0_TRANSLATION = np.zeros(3)
ARUCO_TO_TABLE0_ROTATION = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
])


def aruco_point_to_table0(point_aruco):
    """Apply the fixed ArUco-origin -> table_0 transform (see
    ARUCO_TO_TABLE0_ROTATION above)."""
    return ARUCO_TO_TABLE0_ROTATION @ point_aruco + ARUCO_TO_TABLE0_TRANSLATION


class ClickCollector:
    """Interactive OpenCV UI: show an RGB frame and let the user left-click a
    pixel, confirming with SPACE (re-click to correct, ESC to abort)."""

    WINDOW_NAME = 'script_controller - click target'

    def __init__(self):
        self._clicked_xy = None
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicked_xy = (x, y)

    def collect(self, rgb_image_bgr, instructions):
        """Block until the user clicks a pixel and confirms with SPACE.

        Returns (u, v) pixel coordinates. Raises RuntimeError on ESC (abort).
        """
        self._clicked_xy = None
        while True:
            frame = rgb_image_bgr.copy()
            for i, line in enumerate(instructions.split('\n')):
                cv2.putText(frame, line, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)
            if self._clicked_xy is not None:
                cv2.drawMarker(frame, self._clicked_xy, (0, 0, 255),
                                markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

            cv2.imshow(self.WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                raise RuntimeError('Click collection aborted by user (ESC).')
            if key == 32 and self._clicked_xy is not None:  # SPACE confirms
                return self._clicked_xy

    def close(self):
        cv2.destroyWindow(self.WINDOW_NAME)
