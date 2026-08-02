"""`pose/calibration.py::calibrate_camera` — калибровка интринсиков по фото известной плоской
ArUco-раскладки. Сквозной тест на синтетике (не мок `cv2.calibrateCamera`): рендерим один и тот
же 2-маркерный лист под РАЗНЫМИ позами камеры (см. докстринг модуля — фронтальных снимков с
одного места недостаточно для устойчивой калибровки), известной истинной камерой (fx/fy/cx/cy,
без дисторсии), затем проверяем, что калибровка восстанавливает эти интринсики близко к истине.
"""

from __future__ import annotations

import cv2
import numpy as np

from roboson_tools.pose.calibration import calibrate_camera
from roboson_tools.pose.camera_pose import CameraIntrinsics
from roboson_tools.pose.marker_layout import MarkerLayout

_RESOLUTION_PX = (480, 640)  # (rows, cols)
_TRUE_INTRINSICS = CameraIntrinsics(fx=900.0, fy=880.0, cx=320.0, cy=240.0, resolution_px=_RESOLUTION_PX)
_DIST_ZERO = np.zeros(5)
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Лист из 2 меток в его собственной локальной плоскости Z=0 (тот же паттерн, что
# assets/aruco_test_sheet/generate_sheet.py) — для калибровки не важно, что это "лист", а не
# чекерборд, важна только известная плоская геометрия.
_MARKER_SIDE = 45.0
_LAYOUT = MarkerLayout(
    dictionary_name="DICT_4X4_50",
    corners_world_mm={
        20: np.array(
            [[30.0, 120.0, 0.0], [75.0, 120.0, 0.0], [75.0, 165.0, 0.0], [30.0, 165.0, 0.0]]
        ),
        21: np.array(
            [[135.0, 120.0, 0.0], [180.0, 120.0, 0.0], [180.0, 165.0, 0.0], [135.0, 165.0, 0.0]]
        ),
    },
)


def _render_board(rotation: np.ndarray, translation: np.ndarray, marker_px: int = 240) -> np.ndarray:
    n_rows, n_cols = _RESOLUTION_PX
    image = np.full((n_rows, n_cols, 3), 255, dtype=np.uint8)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = translation.reshape(3, 1)
    for marker_id, corners_world in _LAYOUT.corners_world_mm.items():
        marker_gray = cv2.aruco.generateImageMarker(_DICT, marker_id, marker_px)
        marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
        dst_pts, _ = cv2.projectPoints(
            corners_world, rvec, tvec, _TRUE_INTRINSICS.matrix, _DIST_ZERO
        )
        dst_pts = dst_pts.reshape(4, 2).astype(np.float32)
        src_pts = np.array(
            [[0, 0], [marker_px - 1, 0], [marker_px - 1, marker_px - 1], [0, marker_px - 1]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(marker_bgr, homography, (n_cols, n_rows), borderValue=(255, 255, 255))
        coverage = cv2.warpPerspective(
            np.full((marker_px, marker_px), 255, dtype=np.uint8), homography, (n_cols, n_rows)
        )
        image[coverage > 0] = warped[coverage > 0]
    return image


def _varied_pose_photos() -> list[np.ndarray]:
    """Board-фиксированная камера "смотрит" на лист под разными углами/дистанциями/боковыми
    сдвигами — рендерим, двигая ЛИСТ перед фиксированной камерой (эквивалентно движению камеры
    вокруг листа). Важны ВСЕ три вида разнообразия (см. докстринг модуля): без разброса по
    наклону/дистанции fx/fy плохо отделяются от дистанции; без бокового разброса (лист всегда по
    центру кадра) плохо обусловлена дисторсия/главная точка — маркеры ни разу не проецируются к
    краям кадра (подобрано экспериментально: с одним только наклоном+дистанцией fx получался
    с ошибкой ~7%, с добавлением бокового сдвига — ~1.5%)."""
    photos = []
    for tilt_deg in (-25.0, -12.0, 0.0, 12.0, 25.0):
        for distance in (700.0, 1000.0, 1400.0):
            for offset_x, offset_y in ((0.0, 0.0), (150.0, 90.0), (-150.0, 90.0), (150.0, -90.0), (-150.0, -90.0)):
                pitch = np.radians(tilt_deg)
                yaw = np.radians(tilt_deg * 0.7)
                rot_x = np.array(
                    [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
                )
                rot_y = np.array(
                    [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
                )
                rotation = rot_y @ rot_x
                # Лист центрирован по X~105/Y~142.5 (центр между двумя метками), лежит на Z=0 в
                # своей плоскости; сдвигаем вдоль камерной оси Z на distance + вбок на
                # offset_x/offset_y, чтобы разные фото покрывали разные области кадра.
                center_local = np.array([105.0, 142.5, 0.0])
                translation = -rotation @ center_local + np.array([offset_x, offset_y, distance])
                photos.append(_render_board(rotation, translation))
    return photos


def test_calibration_recovers_true_intrinsics():
    photos = _varied_pose_photos()
    result = calibrate_camera(photos, _LAYOUT)

    assert result.intrinsics is not None, result.message
    assert len(result.used_image_indices) == len(photos)
    assert result.rms_reprojection_error_px is not None
    assert result.rms_reprojection_error_px < 1.0

    recovered = result.intrinsics
    assert abs(recovered.fx - _TRUE_INTRINSICS.fx) < _TRUE_INTRINSICS.fx * 0.05
    assert abs(recovered.fy - _TRUE_INTRINSICS.fy) < _TRUE_INTRINSICS.fy * 0.05
    assert abs(recovered.cx - _TRUE_INTRINSICS.cx) < 20.0
    assert abs(recovered.cy - _TRUE_INTRINSICS.cy) < 20.0
    assert recovered.resolution_px == _RESOLUTION_PX


def test_empty_photo_list_returns_none():
    result = calibrate_camera([], _LAYOUT)
    assert result.intrinsics is None
    assert result.message


def test_too_few_valid_photos_returns_none():
    # Только 2 валидных фото — ниже _MIN_VALID_IMAGES=3, даже если сама детекция отработала.
    photos = _varied_pose_photos()[:2]
    result = calibrate_camera(photos, _LAYOUT)
    assert result.intrinsics is None
    assert len(result.used_image_indices) == 2


def test_blank_photos_are_all_skipped():
    blanks = [np.full((*_RESOLUTION_PX, 3), 255, dtype=np.uint8) for _ in range(5)]
    result = calibrate_camera(blanks, _LAYOUT)
    assert result.intrinsics is None
    assert result.skipped_image_indices == [0, 1, 2, 3, 4]
    assert result.used_image_indices == []


def test_mismatched_resolution_returns_none():
    photos = _varied_pose_photos()[:4]
    photos[1] = np.full((480, 800, 3), 255, dtype=np.uint8)  # другое разрешение
    result = calibrate_camera(photos, _LAYOUT)
    assert result.intrinsics is None
    assert "разрешение" in result.message
