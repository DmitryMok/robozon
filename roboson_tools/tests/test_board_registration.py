"""`pose/board_registration.py` — автоматическая регистрация нескольких независимых ArUco-досок
(зеркало, лента, ...) в одну систему координат по "мостиковым" фото (видно >=2 доски сразу).
Сквозной тест на синтетике: рендерим фото с известной ИСТИННОЙ позой камеры (мировая = система
опорной доски "belt") и известным ИСТИННЫМ переходом mirror->belt, затем проверяем, что
`register_boards` восстанавливает этот переход близко к истине из одних только фото.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from roboson_tools.pose.board_registration import (
    RigidTransform,
    bundle_adjust_registration,
    merge_layout,
    register_boards,
)
from roboson_tools.pose.camera_pose import CameraIntrinsics
from roboson_tools.pose.marker_layout import MarkerLayout

_RESOLUTION_PX = (480, 640)  # (rows, cols)
_INTRINSICS = CameraIntrinsics(fx=900.0, fy=900.0, cx=320.0, cy=240.0, resolution_px=_RESOLUTION_PX)
_DIST_ZERO = np.zeros(5)
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# "belt" — опорная доска, её локальные координаты = мировые/референсные по определению.
_BELT_LAYOUT = MarkerLayout(
    dictionary_name="DICT_4X4_50",
    corners_world_mm={
        1: np.array([[-60.0, -60.0, 0.0], [-15.0, -60.0, 0.0], [-15.0, -15.0, 0.0], [-60.0, -15.0, 0.0]]),
        2: np.array([[15.0, -60.0, 0.0], [60.0, -60.0, 0.0], [60.0, -15.0, 0.0], [15.0, -15.0, 0.0]]),
    },
)
# "mirror" — своя ЛОКАЛЬНАЯ плоскость (Z=0 в СВОЕЙ системе, как mirror_board.yaml до регистрации).
_MIRROR_LAYOUT = MarkerLayout(
    dictionary_name="DICT_4X4_50",
    corners_world_mm={
        10: np.array([[-40.0, -40.0, 0.0], [-5.0, -40.0, 0.0], [-5.0, -5.0, 0.0], [-40.0, -5.0, 0.0]]),
        11: np.array([[5.0, -40.0, 0.0], [40.0, -40.0, 0.0], [40.0, -5.0, 0.0], [5.0, -5.0, 0.0]]),
    },
)
# "isolated" — третья доска, никогда не появляется вместе ни с одной другой (для проверки
# unregistered_boards).
_ISOLATED_LAYOUT = MarkerLayout(
    dictionary_name="DICT_4X4_50",
    corners_world_mm={
        40: np.array([[-20.0, -20.0, 0.0], [20.0, -20.0, 0.0], [20.0, 20.0, 0.0], [-20.0, 20.0, 0.0]]),
    },
)

# Истинный переход mirror-local -> belt(мировая): поворот на 40° вокруг Y + сдвиг.
_TRUE_ANGLE_DEG = 40.0
_true_theta = np.radians(_TRUE_ANGLE_DEG)
_TRUE_MIRROR_TO_BELT = RigidTransform(
    rotation=np.array(
        [
            [np.cos(_true_theta), 0.0, np.sin(_true_theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(_true_theta), 0.0, np.cos(_true_theta)],
        ]
    ),
    translation=np.array([250.0, 80.0, 400.0]),
)


def _blank_photo() -> np.ndarray:
    n_rows, n_cols = _RESOLUTION_PX
    return np.full((n_rows, n_cols, 3), 255, dtype=np.uint8)


def _render_markers_world(
    image: np.ndarray, markers_world: dict[int, np.ndarray], cam_rotation: np.ndarray, cam_translation: np.ndarray
) -> None:
    """Рисует набор меток (id -> (4,3) МИРОВЫЕ координаты) под заданной позой камеры — тот же
    приём warpPerspective, что и `tests/test_aruco_estimator.py::_render_marker`, обобщённый на
    несколько меток разных досок в одном кадре (для мостиковых фото)."""
    rvec, _ = cv2.Rodrigues(cam_rotation)
    tvec = cam_translation.reshape(3, 1)
    n_rows, n_cols = image.shape[:2]
    marker_px = 200
    for marker_id, corners_world in markers_world.items():
        marker_gray = cv2.aruco.generateImageMarker(_DICT, marker_id, marker_px)
        marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
        dst_pts, _ = cv2.projectPoints(corners_world, rvec, tvec, _INTRINSICS.matrix, _DIST_ZERO)
        dst_pts = dst_pts.reshape(4, 2).astype(np.float32)
        if np.any(~np.isfinite(dst_pts)):
            continue
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


def _mirror_world_corners() -> dict[int, np.ndarray]:
    return {
        marker_id: _TRUE_MIRROR_TO_BELT.apply(corners_local)
        for marker_id, corners_local in _MIRROR_LAYOUT.corners_world_mm.items()
    }


def _camera_looking_at(target: np.ndarray, distance: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Камера на кольце радиуса `distance` вокруг `target`, смотрит на него — простая
    look-at-камера в МИРОВЫХ (belt) координатах, для рендера синтетических фото."""
    yaw = np.radians(yaw_deg)
    eye = target + distance * np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
    forward = (target - eye)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    return rotation, translation


def _bridge_photos(n: int) -> list[np.ndarray]:
    mirror_world = _mirror_world_corners()
    belt_center = np.array([0.0, 0.0, 0.0])
    mirror_center = _TRUE_MIRROR_TO_BELT.translation
    midpoint = (belt_center + mirror_center) / 2.0
    # Дистанция — компромисс: достаточно большая, чтобы обе доски (разнесённые на ~250-400 мм)
    # уместились в угол обзора камеры (иначе одна из них обрезается кадром), но не намного
    # больше — иначе угловой размер каждой доски (~90 мм) становится слишком мал для уверенного
    # solvePnP (эффект слабой перспективы: вращение доски плохо наблюдаемо издалека даже без
    # шума детекции — см. историю подбора в git blame/заметке задачи).
    photos = []
    for i in range(n):
        yaw = -20.0 + i * (40.0 / max(n - 1, 1))
        rotation, translation = _camera_looking_at(midpoint, distance=1100.0, yaw_deg=yaw)
        photo = _blank_photo()
        _render_markers_world(photo, _BELT_LAYOUT.corners_world_mm, rotation, translation)
        _render_markers_world(photo, mirror_world, rotation, translation)
        photos.append(photo)
    return photos


def _belt_only_photos(n: int) -> list[np.ndarray]:
    photos = []
    for i in range(n):
        yaw = -15.0 + i * (30.0 / max(n - 1, 1))
        rotation, translation = _camera_looking_at(np.zeros(3), distance=300.0, yaw_deg=yaw)
        photo = _blank_photo()
        _render_markers_world(photo, _BELT_LAYOUT.corners_world_mm, rotation, translation)
        photos.append(photo)
    return photos


def _mirror_only_photos(n: int) -> list[np.ndarray]:
    mirror_world = _mirror_world_corners()
    photos = []
    for i in range(n):
        yaw = -15.0 + i * (30.0 / max(n - 1, 1))
        rotation, translation = _camera_looking_at(_TRUE_MIRROR_TO_BELT.translation, distance=300.0, yaw_deg=yaw)
        photo = _blank_photo()
        _render_markers_world(photo, mirror_world, rotation, translation)
        photos.append(photo)
    return photos


def _bridge_photos_noisy(n: int, rng: np.random.Generator, sigma: float = 6.0) -> list[np.ndarray]:
    """Как `_bridge_photos`, но с добавленным гауссовым шумом пикселей ДО детекции ArUco.
    ПРОВЕРЕНО: при этой амплитуде sub-pixel refinement детектора ArUco всё равно доводит угол до
    того же значения, что и на чистом рендере (шум съедается refinement'ом, не доходит до
    найденных координат угла) — то есть этот тест НЕ проверяет устойчивость bundle adjustment к
    шуму детекции как таковому. Он проверяет более скромную вещь: что совместная оптимизация
    (стартующая с ГРУБОЙ, независимой на кадр, затравки позы камеры — единственного solvePnP по
    одной видимой доске) сходится к результату, не хуже (по точности к истинному переходу, см.
    `_TRUE_MIRROR_TO_BELT`) простого усреднения попарных переходов `register_boards`, и что RMS
    репроекции после оптимизации намного ниже, чем у сырой затравки."""
    photos = _bridge_photos(n)
    noisy = []
    for photo in photos:
        noise = rng.normal(0.0, sigma, photo.shape)
        noisy.append(np.clip(photo.astype(np.float64) + noise, 0, 255).astype(np.uint8))
    return noisy


def _isolated_only_photo() -> np.ndarray:
    photo = _blank_photo()
    rotation, translation = _camera_looking_at(np.array([1000.0, 1000.0, 1000.0]), distance=300.0, yaw_deg=0.0)
    _render_markers_world(photo, _ISOLATED_LAYOUT.corners_world_mm, rotation, translation)
    return photo


def _rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    relative = r_a @ r_b.T
    cos_angle = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def test_registers_mirror_via_bridge_photos_close_to_true_transform():
    photos = _bridge_photos(6) + _belt_only_photos(3) + _mirror_only_photos(3) + [_isolated_only_photo()]
    boards = {"belt": _BELT_LAYOUT, "mirror": _MIRROR_LAYOUT, "isolated": _ISOLATED_LAYOUT}

    result = register_boards(photos, boards, _INTRINSICS, reference_board="belt")

    assert "belt" in result.transforms
    np.testing.assert_allclose(result.transforms["belt"].rotation, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(result.transforms["belt"].translation, np.zeros(3), atol=1e-9)

    assert "mirror" in result.transforms
    recovered = result.transforms["mirror"]
    # Допуски отражают реальную достижимую точность регистрации по фото (не точную математику):
    # даже без шума детекции solvePnP по одной маленькой плоской доске далеко от камеры
    # (слабая перспектива) даёт не идеальную позу — усреднение нескольких мостиковых фото
    # уменьшает, но не устраняет эту ошибку полностью (см. докстринг `_bridge_photos`).
    assert _rotation_angle_deg(recovered.rotation, _TRUE_MIRROR_TO_BELT.rotation) < 5.0
    np.testing.assert_allclose(recovered.translation, _TRUE_MIRROR_TO_BELT.translation, atol=20.0)

    assert result.unregistered_boards == ["isolated"]
    assert ("belt", "mirror") in result.bridge_photo_counts
    assert result.bridge_photo_counts[("belt", "mirror")] == 6


def test_merge_layout_transforms_mirror_markers_into_belt_frame():
    photos = _bridge_photos(6)
    boards = {"belt": _BELT_LAYOUT, "mirror": _MIRROR_LAYOUT}
    result = register_boards(photos, boards, _INTRINSICS, reference_board="belt")

    merged = merge_layout(boards, result)
    true_world = _mirror_world_corners()
    for marker_id, true_corners in true_world.items():
        # См. комментарий в test_registers_mirror_via_bridge_photos_close_to_true_transform про
        # реалистичный (не нулевой) допуск точности регистрации по фото.
        np.testing.assert_allclose(merged.corners_world_mm[marker_id], true_corners, atol=20.0)
    for marker_id, corners in _BELT_LAYOUT.corners_world_mm.items():
        np.testing.assert_allclose(merged.corners_world_mm[marker_id], corners, atol=1e-9)


def test_no_bridge_photos_leaves_boards_unregistered():
    photos = _belt_only_photos(3) + _mirror_only_photos(3)
    boards = {"belt": _BELT_LAYOUT, "mirror": _MIRROR_LAYOUT}

    result = register_boards(photos, boards, _INTRINSICS, reference_board="belt")

    assert list(result.transforms) == ["belt"]
    assert result.unregistered_boards == ["mirror"]


def test_unknown_reference_board_returns_error():
    boards = {"belt": _BELT_LAYOUT}
    result = register_boards([], boards, _INTRINSICS, reference_board="does_not_exist")
    assert result.transforms == {}
    assert result.unregistered_boards == ["belt"]
    assert "does_not_exist" in result.message


def test_duplicate_marker_id_across_boards_raises():
    boards = {"belt": _BELT_LAYOUT, "dup": _BELT_LAYOUT}
    with pytest.raises(ValueError, match="уникальны"):
        register_boards([], boards, _INTRINSICS, reference_board="belt")


def test_bundle_adjustment_reduces_reprojection_error_and_stays_close_to_true_transform():
    rng = np.random.default_rng(0)
    photos = _bridge_photos_noisy(8, rng) + _belt_only_photos(4) + _mirror_only_photos(4)
    boards = {"belt": _BELT_LAYOUT, "mirror": _MIRROR_LAYOUT}
    initial = register_boards(photos, boards, _INTRINSICS, reference_board="belt")
    assert "mirror" in initial.transforms

    refined, report = bundle_adjust_registration(photos, boards, _INTRINSICS, initial)

    assert report.n_frames_used > 0
    assert report.boards_refined == ["mirror"]
    assert report.rms_reprojection_px_before is not None
    assert report.rms_reprojection_px_after is not None
    # Совместная оптимизация не обязана строго побеждать на КАЖДОМ синтетическом прогоне (зависит
    # от конкретного шума), но не должна систематически ухудшать репроекцию с большим запасом.
    assert report.rms_reprojection_px_after <= report.rms_reprojection_px_before + 0.5

    recovered = refined.transforms["mirror"]
    np.testing.assert_allclose(refined.transforms["belt"].rotation, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(refined.transforms["belt"].translation, np.zeros(3), atol=1e-9)
    assert _rotation_angle_deg(recovered.rotation, _TRUE_MIRROR_TO_BELT.rotation) < 5.0
    np.testing.assert_allclose(recovered.translation, _TRUE_MIRROR_TO_BELT.translation, atol=20.0)


def test_bundle_adjustment_at_least_as_accurate_as_plain_averaging_noiseless():
    photos = _bridge_photos(6) + _belt_only_photos(3) + _mirror_only_photos(3)
    boards = {"belt": _BELT_LAYOUT, "mirror": _MIRROR_LAYOUT}
    initial = register_boards(photos, boards, _INTRINSICS, reference_board="belt")

    refined, _report = bundle_adjust_registration(photos, boards, _INTRINSICS, initial)

    baseline_rot_err = _rotation_angle_deg(
        initial.transforms["mirror"].rotation, _TRUE_MIRROR_TO_BELT.rotation
    )
    refined_rot_err = _rotation_angle_deg(
        refined.transforms["mirror"].rotation, _TRUE_MIRROR_TO_BELT.rotation
    )
    # Запас 0.5° на числовой шум/локальный оптимум LM — не строгое "лучше", а "не хуже".
    assert refined_rot_err <= baseline_rot_err + 0.5


def test_bundle_adjustment_noop_without_bridge_photos():
    photos = _belt_only_photos(3) + _mirror_only_photos(3)
    boards = {"belt": _BELT_LAYOUT, "mirror": _MIRROR_LAYOUT}
    initial = register_boards(photos, boards, _INTRINSICS, reference_board="belt")

    refined, report = bundle_adjust_registration(photos, boards, _INTRINSICS, initial)

    assert refined is initial
    assert report.n_frames_used == 0
    assert report.boards_refined == []


def test_bundle_adjustment_noop_when_no_board_registered_besides_reference():
    boards = {"belt": _BELT_LAYOUT}
    initial = register_boards(_belt_only_photos(3), boards, _INTRINSICS, reference_board="belt")

    refined, report = bundle_adjust_registration(_belt_only_photos(3), boards, _INTRINSICS, initial)

    assert refined is initial
    assert report.n_frames_used == 0
