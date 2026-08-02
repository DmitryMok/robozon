"""`pose/camera_pose.py` — конвенции OpenCV (R, t, K) для CameraIntrinsics/CameraPose, без Qt
и без ArUco (чистая геометрия)."""

from __future__ import annotations

import numpy as np

from roboson_tools.pose.camera_pose import CameraIntrinsics, CameraPose


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=900.0, fy=900.0, cx=320.0, cy=240.0, resolution_px=(480, 640))


def test_intrinsics_matrix():
    K = _intrinsics().matrix
    assert K.shape == (3, 3)
    np.testing.assert_allclose(
        K, [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]]
    )


def test_camera_pos_world_identity_rotation_zero_translation_is_origin():
    pose = CameraPose(rotation=np.eye(3), translation=np.zeros(3), intrinsics=_intrinsics())
    np.testing.assert_allclose(pose.camera_pos_world, [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pose.forward_world, [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(pose.right_world, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pose.down_world, [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pose.up_world, [0.0, -1.0, 0.0], atol=1e-9)


def test_camera_pos_world_pure_translation():
    # x_cam = R @ x_world + t, R = I, t = [0, 0, -1000] => камера смотрит вдоль +Z и стоит в
    # мировой точке (0, 0, 1000): x_cam=0 <=> x_world = -t.
    pose = CameraPose(
        rotation=np.eye(3), translation=np.array([0.0, 0.0, -1000.0]), intrinsics=_intrinsics()
    )
    np.testing.assert_allclose(pose.camera_pos_world, [0.0, 0.0, 1000.0], atol=1e-9)


def test_forward_world_flips_under_180_degree_yaw():
    # Поворот на 180° вокруг мировой оси Y: камера, "развёрнутая спиной", смотрит в
    # противоположную от исходной сторону.
    rotation_180_y = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    pose = CameraPose(rotation=rotation_180_y, translation=np.zeros(3), intrinsics=_intrinsics())
    np.testing.assert_allclose(pose.forward_world, [0.0, 0.0, -1.0], atol=1e-9)
    np.testing.assert_allclose(pose.right_world, [-1.0, 0.0, 0.0], atol=1e-9)
