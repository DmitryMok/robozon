"""Кэш результата регистрации досок + калибровки камеры — сохраняется РЯДОМ С ФОТО (не в
`config/`), по требованию пользователя: одна папка с фото — это один сеанс съёмки одного и того
же физического рига, калибровка для неё не должна пересчитываться при каждой загрузке диалога.
При следующей загрузке фото из той же папки кэш подхватывается автоматически; явный пересчёт
(кнопка «Пересчитать калибровку» в UI) всегда возможен и перезаписывает файл.

Хранит СЛИТУЮ (уже в одной системе координат, после `board_registration.merge_layout`) раскладку
меток — в отличие от `assets/mirror_board/`, `assets/belt_board/` (там доски ещё не
зарегистрированы друг относительно друга)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .camera_pose import CameraIntrinsics
from .marker_layout import MarkerLayout

CALIBRATION_CACHE_FILENAME = "calibration_result.yaml"


@dataclass
class CalibrationCache:
    layout: MarkerLayout  # слитая раскладка всех досок в системе reference_board
    intrinsics: CameraIntrinsics
    reference_board: str
    rms_reprojection_error_pct: float | None = None
    source_photos: list[str] = field(default_factory=list)  # имена файлов, по которым считали


def calibration_cache_path(photo_dir: Path | str) -> Path:
    return Path(photo_dir) / CALIBRATION_CACHE_FILENAME


def load_calibration_cache(path: Path | str) -> CalibrationCache | None:
    """`None` — файла нет (обычная ситуация для ещё не калиброванной папки фото, не ошибка).
    Поднимает исключение, если файл ЕСТЬ, но повреждён/не парсится — не скрывать порчу кэша."""
    path = Path(path)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    intr = data["intrinsics"]
    dist_raw = intr.get("dist_coeffs")
    intrinsics = CameraIntrinsics(
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        cx=float(intr["cx"]),
        cy=float(intr["cy"]),
        resolution_px=(int(intr["resolution_px"][0]), int(intr["resolution_px"][1])),
        dist_coeffs=None if dist_raw is None else np.array(dist_raw, dtype=np.float64),
    )
    corners = {int(m["id"]): np.array(m["corners_mm"], dtype=np.float64) for m in data["markers"]}
    layout = MarkerLayout(dictionary_name=str(data["dictionary"]), corners_world_mm=corners)
    return CalibrationCache(
        layout=layout,
        intrinsics=intrinsics,
        reference_board=str(data.get("reference_board", "")),
        rms_reprojection_error_pct=data.get("rms_reprojection_error_pct"),
        source_photos=list(data.get("source_photos", [])),
    )


def save_calibration_cache(path: Path | str, cache: CalibrationCache) -> None:
    data = {
        "dictionary": cache.layout.dictionary_name,
        "reference_board": cache.reference_board,
        "rms_reprojection_error_pct": cache.rms_reprojection_error_pct,
        "source_photos": cache.source_photos,
        "intrinsics": {
            "fx": cache.intrinsics.fx,
            "fy": cache.intrinsics.fy,
            "cx": cache.intrinsics.cx,
            "cy": cache.intrinsics.cy,
            "resolution_px": [
                int(cache.intrinsics.resolution_px[0]),
                int(cache.intrinsics.resolution_px[1]),
            ],
            "dist_coeffs": (
                None
                if cache.intrinsics.dist_coeffs is None
                else cache.intrinsics.dist_coeffs.tolist()
            ),
        },
        "markers": [
            {"id": marker_id, "corners_mm": corners.tolist()}
            for marker_id, corners in cache.layout.corners_world_mm.items()
        ],
    }
    Path(path).write_text(
        "# Результат board_registration.register_boards+calibrate_camera для этой папки фото —\n"
        "# см. roboson_tools/pose/calibration_cache.py. Пересчитывается ТОЛЬКО по явной команде\n"
        "# пользователя (кнопка «Пересчитать калибровку»), не при каждой загрузке.\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
