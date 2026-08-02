"""Модель одного загруженного фото в диалоге `gui/dialogs/photo_capture_dialog.py` — путь к
файлу + результат оценки позы. Только в памяти сессии диалога, без персистентности (как список
STL-объектов в `MainWindow` сейчас)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aruco_estimator import PoseEstimate
from ..segmentation.base import SegmentationResult


@dataclass
class PhotoItem:
    path: Path
    image_bgr: np.ndarray | None = None
    pose_estimate: PoseEstimate | None = None
    enabled: bool = True  # задел под будущий этап: включение/исключение фото из реконструкции
    segmentation: SegmentationResult | None = None  # силуэт объекта НА ЛЕНТЕ (прямой вид)
    # Силуэт ТОГО ЖЕ объекта, но его ОТРАЖЕНИЯ в зеркале — отдельная сегментация по отдельно
    # замаскированной (только область зеркала) копии кадра, см.
    # gui/dialogs/photo_capture_dialog.py::_on_segment_clicked. Раньше прямой вид и отражение
    # конкурировали за ЕДИНСТВЕННУЮ лучшую детекцию в одном запросе SAM3 — прямой вид почти
    # всегда увереннее (крупнее, лучше освещён), поэтому отражение НИКОГДА не побеждало, даже
    # когда оно есть на каждом фото (см. заметку задачи — реальный кейс пользователя). Оба
    # силуэта участвуют в 3D-реконструкции как независимые наблюдения (см. pose_carving.Observation).
    segmentation_mirror: SegmentationResult | None = None
