"""Калибровка интринсиков камеры (fx/fy/cx/cy/dist_coeffs) по набору фото известной ArUco-
раскладки — БЕЗ отдельного чекерборда/ChArUco-доски. Переиспользует уже существующую
инфраструктуру: `aruco_estimator.detect_markers()` даёт 2D-углы меток на фото,
`marker_layout.MarkerLayout.corners_world_mm` уже хранит их известные 3D-координаты (то же, чем
пользуется `estimate_camera_pose` для solvePnP) — эти же точки становятся `objectPoints`/
`imagePoints` для `cv2.calibrateCamera`. Годится любая известная раскладка — как ПЛОСКАЯ (один
2-маркерный лист/доска), так и НЕПЛОСКАЯ (слитая многодосочная раскладка после
`board_registration.merge_layout()` — например все 4 доски рига зеркало+лента сразу, они не
лежат в одной плоскости).

Требование к набору фото (как для любой калибровки по плоской мишени, включая чекерборд):
снимать под РАЗНЫМИ углами/наклонами/дистанциями — иначе fx/fy плохо отделяются от дистанции до
мишени, калибровка получится плохо обусловленной. Фронтальных снимков с одного места
недостаточно.

`cv2.calibrateCamera` для НЕПЛОСКОЙ раскладки требует явное начальное приближение интринсиков
(`cameraMatrix=None` работает только для плоских целей — для неплоских бросает `cv2.error`, см.
`tests/test_calibration.py::test_non_planar_layout_needs_and_uses_initial_guess`) — поэтому
всегда передаём приближение (либо переданное явно, либо грубое из размера кадра) с флагом
`CALIB_USE_INTRINSIC_GUESS`; для плоских целей это тоже работает корректно, просто как разумная
стартовая точка оптимизации вместо автоматической (по гомографии)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .aruco_estimator import detect_markers
from .camera_pose import CameraIntrinsics
from .marker_layout import MarkerLayout

_MIN_VALID_IMAGES = 3  # ниже этого calibrateCamera либо упадёт, либо даст мусор


@dataclass
class CalibrationResult:
    intrinsics: CameraIntrinsics | None
    rms_reprojection_error_px: float | None = None
    # Тот же RMS в % от короткой стороны кадра — сравнимая метрика между фото/сессиями с разным
    # разрешением (1px на 480-строчном кадре и 1px на 960-строчном — РАЗНАЯ относительная
    # точность). См. докстринг модуля.
    rms_reprojection_error_pct: float | None = None
    per_image_error_px: list[float] = field(default_factory=list)  # по одному на used_image_indices
    per_image_error_pct: list[float] = field(default_factory=list)  # то же, в % короткой стороны
    used_image_indices: list[int] = field(default_factory=list)
    skipped_image_indices: list[int] = field(default_factory=list)  # мало меток на этом фото
    message: str = ""


def calibrate_camera(
    images_bgr: list[np.ndarray],
    layout: MarkerLayout,
    min_markers_per_image: int = 2,
    initial_intrinsics: CameraIntrinsics | None = None,
) -> CalibrationResult:
    """Калибрует камеру по набору фото одной и той же ИЗВЕСТНОЙ раскладки `layout` (плоской или
    неплоской — см. докстринг модуля). Фото с < `min_markers_per_image` видимых меток
    пропускаются (недостаточно точек для надёжного вклада в калибровку), не прерывая расчёт по
    остальным. `initial_intrinsics` — необязательное начальное приближение (например текущие
    грубые интринсики из конфига); если не передано, строится грубое приближение из размера
    кадра. `intrinsics=None` — калибровка не удалась: пустой список фото, разное разрешение между
    фото (`cv2.calibrateCamera` требует единый `imageSize`), меньше `_MIN_VALID_IMAGES` валидных
    фото, либо сама `cv2.calibrateCamera` упала (типично — вырожденный набор, все фото почти под
    одним углом)."""
    if not images_bgr:
        return CalibrationResult(intrinsics=None, message="нет фото")

    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    used_idx: list[int] = []
    skipped_idx: list[int] = []
    image_size: tuple[int, int] | None = None  # (width, height) — конвенция cv2.calibrateCamera
    for idx, image in enumerate(images_bgr):
        rows, cols = image.shape[:2]
        if image_size is None:
            image_size = (cols, rows)
        elif (cols, rows) != image_size:
            return CalibrationResult(
                intrinsics=None,
                message=(
                    f"фото {idx} имеет разрешение {(cols, rows)}, отличное от {image_size} — "
                    "все фото калибровки должны быть одного разрешения"
                ),
            )
        detected = detect_markers(image, layout)
        if len(detected) < min_markers_per_image:
            skipped_idx.append(idx)
            continue
        marker_ids = sorted(detected.keys())
        object_points_list.append(
            np.concatenate([layout.corners_world_mm[i] for i in marker_ids], axis=0).astype(
                np.float32
            )
        )
        image_points_list.append(
            np.concatenate([detected[i] for i in marker_ids], axis=0).astype(np.float32)
        )
        used_idx.append(idx)

    if len(used_idx) < _MIN_VALID_IMAGES:
        return CalibrationResult(
            intrinsics=None,
            used_image_indices=used_idx,
            skipped_image_indices=skipped_idx,
            message=(
                f"недостаточно валидных фото ({len(used_idx)}, нужно минимум {_MIN_VALID_IMAGES}"
                " — на практике для устойчивой калибровки нужно 10-20 фото под разными углами"
                "/дистанциями)"
            ),
        )

    # cv2.calibrateCamera для НЕПЛОСКОЙ раскладки требует явное начальное приближение (падает с
    # cv2.error "the initial intrinsic matrix must be specified", если cameraMatrix=None) — см.
    # докстринг модуля. Передаём приближение всегда (для плоских целей это тоже корректно).
    if initial_intrinsics is not None:
        guess_matrix = initial_intrinsics.matrix
    else:
        max_dim = float(max(image_size))
        guess_matrix = np.array(
            [[max_dim, 0.0, image_size[0] / 2.0], [0.0, max_dim, image_size[1] / 2.0], [0.0, 0.0, 1.0]]
        )

    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list, image_points_list, image_size, guess_matrix, None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        )
    except cv2.error as exc:
        return CalibrationResult(
            intrinsics=None,
            used_image_indices=used_idx,
            skipped_image_indices=skipped_idx,
            message=f"cv2.calibrateCamera упала: {exc}",
        )

    narrow_side = float(min(image_size))
    per_image_errors_px = []
    per_image_errors_pct = []
    for obj_pts, img_pts, rvec, tvec in zip(object_points_list, image_points_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        err_px = float(np.linalg.norm(projected.reshape(-1, 2) - img_pts, axis=1).mean())
        per_image_errors_px.append(err_px)
        per_image_errors_pct.append(err_px / narrow_side * 100.0)
    rms_pct = float(rms) / narrow_side * 100.0

    intrinsics = CameraIntrinsics(
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        resolution_px=(image_size[1], image_size[0]),  # (rows, cols) — конвенция CameraIntrinsics
        dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
    )
    return CalibrationResult(
        intrinsics=intrinsics,
        rms_reprojection_error_px=float(rms),
        rms_reprojection_error_pct=rms_pct,
        per_image_error_px=per_image_errors_px,
        per_image_error_pct=per_image_errors_pct,
        used_image_indices=used_idx,
        skipped_image_indices=skipped_idx,
        message=(
            f"калибровка по {len(used_idx)} фото (пропущено {len(skipped_idx)}), "
            f"RMS ошибка репроекции {rms_pct:.2f}% от короткой стороны кадра ({rms:.2f}px)"
        ),
    )
