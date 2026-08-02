"""Представление позы и интринсики камеры в конвенции OpenCV (R, t, K).

Общий "формат обмена" для позы камеры, не завязанный на источник: ArUco+solvePnP
(`pose/aruco_estimator.py`) либо, в будущем, ground truth из Webots-симуляции — обе стороны
просто конструируют `CameraPose`, дальше геометрия (проекция, visual hull) сможет работать
одинаково независимо от источника позы. Кольцевая модель камеры `core/camera_rig.py`/
`silhouette/camera.py` (angle_deg + distance_mm) этот тип пока не использует — переход
запланирован отдельным этапом.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..core.config import CONFIG_DIR

CAMERA_INTRINSICS_PATH = CONFIG_DIR / "camera_intrinsics.yaml"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Идеальная pinhole-модель. Этап 1: `dist_coeffs` всегда `None` (без дисторсии) —
    поле зарезервировано на будущее, когда линза перестанет считаться идеальной."""

    fx: float
    fy: float
    cx: float
    cy: float
    resolution_px: tuple[int, int]  # (n_rows, n_cols)
    dist_coeffs: np.ndarray | None = None

    @property
    def matrix(self) -> np.ndarray:
        """3x3 матрица K для cv2.solvePnP/cv2.projectPoints."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )


@dataclass(frozen=True)
class CameraPose:
    """Поза камеры в мировых координатах, конвенция OpenCV:
    `x_cam = rotation @ x_world + translation`. `rotation` — (3,3) ортонормированная матрица
    мир->камера (не rvec)."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    intrinsics: CameraIntrinsics

    @property
    def camera_pos_world(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    @property
    def right_world(self) -> np.ndarray:
        return self.rotation.T @ np.array([1.0, 0.0, 0.0])

    @property
    def down_world(self) -> np.ndarray:
        return self.rotation.T @ np.array([0.0, 1.0, 0.0])

    @property
    def forward_world(self) -> np.ndarray:
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])

    @property
    def up_world(self) -> np.ndarray:
        return -self.down_world


def load_camera_intrinsics(path: Path | str = CAMERA_INTRINSICS_PATH) -> CameraIntrinsics:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    res_raw = data["resolution_px"]
    dist_raw = data.get("dist_coeffs")
    return CameraIntrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        resolution_px=(int(res_raw[0]), int(res_raw[1])),
        dist_coeffs=None if dist_raw is None else np.array(dist_raw, dtype=np.float64),
    )


def save_camera_intrinsics(
    intrinsics: CameraIntrinsics, path: Path | str = CAMERA_INTRINSICS_PATH
) -> None:
    """Записывает интринсики в тот же YAML-формат, что читает `load_camera_intrinsics` — для
    сохранения результата калибровки (`pose/calibration.py::calibrate_camera`). Перезаписывает
    файл целиком (ручные комментарии в текущем `camera_intrinsics.yaml` будут потеряны — вызывающая
    сторона должна предупредить пользователя перед сохранением, если это важно)."""
    data = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "resolution_px": [int(intrinsics.resolution_px[0]), int(intrinsics.resolution_px[1])],
        "dist_coeffs": None if intrinsics.dist_coeffs is None else intrinsics.dist_coeffs.tolist(),
    }
    Path(path).write_text(
        "# Сохранено calibrate_camera() (pose/calibration.py) — cv2.calibrateCamera по фото\n"
        "# известной ArUco-раскладки.\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
