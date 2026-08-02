"""`pose/aruco_estimator.py` — детекция ArUco-меток + solvePnP, сквозной тест на синтетическом
фото (не мок solvePnP): рисуем метки известного размера в известных мировых координатах,
проецируем их известной камерой в пиксели, вклеиваем в холст через `cv2.warpPerspective`,
затем прогоняем `estimate_camera_pose` целиком — включая настоящую `cv2.aruco.ArucoDetector`
детекцию — и сверяем восстановленную позу с исходной.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from roboson_tools.pose.aruco_estimator import estimate_camera_pose
from roboson_tools.pose.camera_pose import CameraIntrinsics
from roboson_tools.pose.marker_layout import MarkerLayout

_RESOLUTION_PX = (480, 640)  # (rows, cols)
_INTRINSICS = CameraIntrinsics(fx=900.0, fy=900.0, cx=320.0, cy=240.0, resolution_px=_RESOLUTION_PX)
_DIST_ZERO = np.zeros(5)

# Камера в мировом начале координат, смотрит вдоль +Z (identity rotation) — см.
# tests/test_camera_pose.py про эту же конвенцию.
_CAMERA_ROTATION = np.eye(3)
_CAMERA_TRANSLATION = np.zeros(3)

# Две метки, некомпланарные (разная глубина) и не перекрывающиеся в кадре — условия, для
# которых planar pose ambiguity одиночной метки не должна возникать (см. докстринг
# aruco_estimator.estimate_camera_pose).
_MARKER_A_ID = 3
_MARKER_A_CORNERS = np.array(
    [[-260.0, -160.0, 1000.0], [-140.0, -160.0, 1000.0], [-140.0, -40.0, 1000.0], [-260.0, -40.0, 1000.0]]
)
_MARKER_B_ID = 7
_MARKER_B_CORNERS = np.array(
    [[90.0, -160.0, 1200.0], [210.0, -160.0, 1200.0], [210.0, -40.0, 1200.0], [90.0, -40.0, 1200.0]]
)


def _render_marker(image: np.ndarray, marker_id: int, corners_world: np.ndarray, marker_px: int = 240) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_gray = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)

    rvec, _ = cv2.Rodrigues(_CAMERA_ROTATION)
    tvec = _CAMERA_TRANSLATION.reshape(3, 1)
    dst_pts, _ = cv2.projectPoints(corners_world, rvec, tvec, _INTRINSICS.matrix, _DIST_ZERO)
    dst_pts = dst_pts.reshape(4, 2).astype(np.float32)
    # src_pts — углы сгенерированного битмапа в его СОБСТВЕННОМ каноническом порядке
    # (TL,TR,BR,BL), тот же порядок, в котором лежат corners_world — см. "ВАЖНО" в
    # marker_layout.py про то, почему порядок важен.
    src_pts = np.array(
        [[0, 0], [marker_px - 1, 0], [marker_px - 1, marker_px - 1], [0, marker_px - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    n_rows, n_cols = image.shape[:2]
    warped = cv2.warpPerspective(marker_bgr, homography, (n_cols, n_rows), borderValue=(255, 255, 255))
    coverage = cv2.warpPerspective(
        np.full((marker_px, marker_px), 255, dtype=np.uint8), homography, (n_cols, n_rows)
    )
    image[coverage > 0] = warped[coverage > 0]


def _blank_photo() -> np.ndarray:
    n_rows, n_cols = _RESOLUTION_PX
    return np.full((n_rows, n_cols, 3), 255, dtype=np.uint8)


def _layout(markers: dict[int, np.ndarray]) -> MarkerLayout:
    return MarkerLayout(dictionary_name="DICT_4X4_50", corners_world_mm=markers)


def _rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Угол между двумя поворотами (°) — устойчивая метрика близости ориентации."""
    relative = r_a @ r_b.T
    cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def test_two_markers_recovers_known_pose():
    photo = _blank_photo()
    _render_marker(photo, _MARKER_A_ID, _MARKER_A_CORNERS)
    _render_marker(photo, _MARKER_B_ID, _MARKER_B_CORNERS)
    layout = _layout({_MARKER_A_ID: _MARKER_A_CORNERS, _MARKER_B_ID: _MARKER_B_CORNERS})

    result = estimate_camera_pose(photo, layout, _INTRINSICS, min_markers=2)

    assert result.status == "ok"
    assert result.marker_ids_used == [_MARKER_A_ID, _MARKER_B_ID]
    assert result.pose is not None
    np.testing.assert_allclose(result.pose.camera_pos_world, [0.0, 0.0, 0.0], atol=5.0)
    assert _rotation_angle_deg(result.pose.rotation, _CAMERA_ROTATION) < 2.0
    assert result.reprojection_error_px is not None
    assert result.reprojection_error_px < 2.0


def test_single_visible_marker_is_insufficient():
    photo = _blank_photo()
    _render_marker(photo, _MARKER_A_ID, _MARKER_A_CORNERS)
    # Раскладка знает про обе метки, но на фото нарисована только одна — ровно ситуация
    # "меньше min_markers меток попало в кадр".
    layout = _layout({_MARKER_A_ID: _MARKER_A_CORNERS, _MARKER_B_ID: _MARKER_B_CORNERS})

    result = estimate_camera_pose(photo, layout, _INTRINSICS, min_markers=2)

    assert result.status == "insufficient_markers"
    assert result.pose is None
    assert result.marker_ids_used == [_MARKER_A_ID]


def test_no_markers_visible_is_insufficient():
    photo = _blank_photo()
    layout = _layout({_MARKER_A_ID: _MARKER_A_CORNERS, _MARKER_B_ID: _MARKER_B_CORNERS})

    result = estimate_camera_pose(photo, layout, _INTRINSICS, min_markers=2)

    assert result.status == "insufficient_markers"
    assert result.pose is None
    assert result.marker_ids_used == []


def test_invalid_world_layout_fails_gracefully():
    """Метки на фото детектируются нормально, но раскладка (пиксели <-> мировые координаты)
    заведомо испорчена (NaN в координатах одной метки, например повреждённый/неверно
    заполненный aruco_markers.yaml) — cv2.solvePnP кидает исключение (проверено отдельно:
    для плоско-вырожденных, но конечных, точек OpenCV молча возвращает "успешный" мусорный
    результат, а не падает — NaN необходим, чтобы гарантированно воспроизвести реальный отказ
    solvePnP). `estimate_camera_pose` обязан поймать её и вернуть статус, а не уронить
    вызывающий код."""
    photo = _blank_photo()
    _render_marker(photo, _MARKER_A_ID, _MARKER_A_CORNERS)
    _render_marker(photo, _MARKER_B_ID, _MARKER_B_CORNERS)

    broken = _MARKER_A_CORNERS.copy()
    broken[0, 0] = np.nan
    layout = _layout({_MARKER_A_ID: broken, _MARKER_B_ID: _MARKER_B_CORNERS})

    result = estimate_camera_pose(photo, layout, _INTRINSICS, min_markers=2)

    assert result.status == "solve_failed"
    assert result.pose is None
