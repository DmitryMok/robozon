"""Оценка позы камеры по ArUco-меткам: детекция на фото + solvePnP по объединённому набору
видимых меток. Чистая функция (изображение + раскладка + интринсики -> поза), без Qt и без
привязки к формату хранения фото — вызывается из GUI-воркера
(`gui/dialogs/photo_capture_dialog.py`), но одинаково пригодна для вызова из скрипта/теста.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .camera_pose import CameraIntrinsics, CameraPose
from .marker_layout import MarkerLayout, cv2_dictionary

# "read_failed" не производится внутри этого модуля (он ничего не читает с диска) — им
# пользуется вызывающий код (`photo_capture_dialog._PoseEstimationWorker`), которому нужен
# статус для фото, не открывшегося как изображение; здесь он просто часть общего словаря
# статусов, чтобы UI показывал их единообразно.
Status = Literal["ok", "insufficient_markers", "solve_failed", "read_failed"]


@dataclass
class PoseEstimate:
    pose: CameraPose | None
    marker_ids_used: list[int] = field(default_factory=list)
    reprojection_error_px: float | None = None
    # То же в % от короткой стороны кадра — px не сравним между фото разного разрешения (1px на
    # 480-строчном кадре и 1px на 960-строчном — разная относительная точность), см.
    # `pose/calibration.py` (тот же приём для CalibrationResult).
    reprojection_error_pct: float | None = None
    status: Status = "insufficient_markers"
    message: str = ""


def detect_markers(image_bgr: np.ndarray, layout: MarkerLayout) -> dict[int, np.ndarray]:
    """Детектирует ArUco-метки на фото, оставляет только те, что есть в `layout` (по id).
    Возвращает {id: (4,2) пиксельные координаты углов} в том же порядке, что и
    `layout.corners_world_mm[id]` (см. докстринг `marker_layout.py`)."""
    dictionary = cv2_dictionary(layout)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(image_bgr)
    if ids is None:
        return {}
    detected: dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in layout.corners_world_mm:
            detected[marker_id] = marker_corners.reshape(4, 2).astype(np.float64)
    return detected


def estimate_camera_pose(
    image_bgr: np.ndarray,
    layout: MarkerLayout,
    intrinsics: CameraIntrinsics,
    min_markers: int = 2,
) -> PoseEstimate:
    """Все видимые метки (>= `min_markers`) объединяются в ОДИН solvePnP по всем их углам
    сразу — не отдельный PnP на метку с выбором/усреднением результата. Одна плоская метка
    (4 компланарные точки) даёт классическую planar pose ambiguity (два решения со схожей
    ошибкой репроекции); объединение нескольких, как правило некомпланарных, меток (доска на
    зеркале + доска по бортам ленты видны под разными углами к камере) даёт хорошо
    обусловленную (>= 8 точек) задачу без этой неоднозначности."""
    detected = detect_markers(image_bgr, layout)
    if len(detected) < min_markers:
        return PoseEstimate(
            pose=None,
            marker_ids_used=sorted(detected.keys()),
            status="insufficient_markers",
            message=f"видно {len(detected)} меток из раскладки, нужно минимум {min_markers}",
        )

    marker_ids = sorted(detected.keys())
    object_points = np.concatenate([layout.corners_world_mm[i] for i in marker_ids], axis=0)
    image_points = np.concatenate([detected[i] for i in marker_ids], axis=0)

    dist_coeffs = intrinsics.dist_coeffs
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsics.matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        return PoseEstimate(
            pose=None,
            marker_ids_used=marker_ids,
            status="solve_failed",
            message=f"cv2.solvePnP упала: {exc}",
        )
    if not success:
        return PoseEstimate(
            pose=None,
            marker_ids_used=marker_ids,
            status="solve_failed",
            message="cv2.solvePnP не нашла решение",
        )

    rotation, _ = cv2.Rodrigues(rvec)
    pose = CameraPose(rotation=rotation, translation=tvec.reshape(3), intrinsics=intrinsics)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsics.matrix, dist_coeffs)
    reprojection_error = float(
        np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean()
    )
    narrow_side = float(min(image_bgr.shape[:2]))
    reprojection_error_pct = reprojection_error / narrow_side * 100.0

    return PoseEstimate(
        pose=pose,
        marker_ids_used=marker_ids,
        reprojection_error_px=reprojection_error,
        reprojection_error_pct=reprojection_error_pct,
        status="ok",
        message=(
            f"поза по {len(marker_ids)} меткам, ошибка репроекции "
            f"{reprojection_error_pct:.2f}% от короткой стороны кадра ({reprojection_error:.2f}px)"
        ),
    )
