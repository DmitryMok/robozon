"""Panel 2 — сетка силуэтов по всем view_angles одновременно (Analytical Mode).

Масштаб всех силуэтов единый: маски строятся с общим extent (см. experiment.build_silhouette_set),
поэтому одинаковые миниатюры соответствуют одинаковым мировым размерам. По умолчанию силуэт
рисуется контуром; чек-бокс "Заливка (сегментация)" переключает на закрашенную область —
как маска сегментации объекта.

Camera Mode (`set_camera_silhouettes`) показывает вместо ортографической сетки такую же по
раскладке сетку — но из ПЕРСПЕКТИВНЫХ силуэтов, по одному на каждый view_angle (то же число
ракурсов, что и в Analytical Mode) — см. заметку задачи "Добавить Camera Mode...". Это отдельный,
параллельный режим отображения (не влияет на реконструкцию/круглую проекцию).

`set_photo_observations` — третий, ещё более приоритетный режим: сетка масок, РЕАЛЬНО
использованных в последней 3D-реконструкции по фото (`PhotoCaptureDialog._on_reconstruct_clicked`)
— по одной плитке на наблюдение (прямое или через отражение), подписанной именем фото и
областью, а не углом обзора синтетической камеры. Приоритет между тремя режимами (`_rebuild`):
photo > camera > analytical — фото-реконструкция актуальна, пока диалог фото открыт и не
переключились на другой STL/ориентацию; `set_photo_observations(None)` возвращает обычную сетку.

Интерактивный зум/пан (см. обсуждение с пользователем): колесо мыши приближает/отдаляет,
перетаскивание ЛКМ двигает вид, двойной клик сбрасывает к исходному масштабу — ОДНО общее
состояние вида (`_zoom`/`_center`) на ВСЮ панель, применяется сразу ко всем плиткам сетки
(нужно для сравнения точности контура одного и того же участка формы на разных ракурсах), а не
к каждой плитке по отдельности."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...silhouette.base import Silhouette

THUMBNAIL_PX = 180
FILL_COLOR = (255, 163, 77)  # BGR — заливка сегментации
CONTOUR_COLOR = (255, 255, 255)

# Ориентировочный размер одной плитки грида (превью + подпись + отступы vbox
# setContentsMargins(2,2,2,2)) — используется только чтобы дать области прокрутки минимальный
# размер, при котором помещается сетка 3x3 без прокрутки (по просьбе пользователя: с реальными
# фото реконструкция по phone2/phone1 даёт 14-18 плиток — без прокрутки грид растягивал Panel 2
# и вытеснял Panel 1 с 3D-моделью за пределы окна).
_TILE_MARGIN_PX = 4
_TITLE_HEIGHT_PX = 24
_MIN_VISIBLE_TILES = 3

ZOOM_MIN = 1.0  # 1.0 = вся картинка целиком (исходное поведение, без зума)
ZOOM_MAX = 20.0
ZOOM_STEP = 1.15  # множитель масштаба на одно деление колеса


def _render_mask(mask: np.ndarray, filled: bool) -> np.ndarray:
    """RGB-картинка силуэта: контур либо закрашенная область сегментации."""
    mask_u8 = mask.astype(np.uint8) * 255
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if filled:
        img[mask] = FILL_COLOR[::-1]  # cv2 BGR -> RGB
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, CONTOUR_COLOR, 2)
    return img


def _to_pixmap(img: np.ndarray, size: int = THUMBNAIL_PX) -> QPixmap:
    img = np.ascontiguousarray(img)
    height, width, _ = img.shape
    qimage = QImage(img.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage).scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class _ZoomableImageLabel(QLabel):
    """Одна плитка сетки. Сама НЕ хранит состояние вида — только транслирует события мыши
    наверх, в SilhouettePanel, у которой вид один общий на все плитки (см. докстринг модуля)."""

    wheelZoomed = pyqtSignal(int)  # +1 (к себе/приблизить) или -1 (от себя/отдалить)
    doubleClicked = pyqtSignal()
    panned = pyqtSignal(int, int)  # dx, dy в экранных пикселях с прошлого события мыши

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drag_last: QPoint | None = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.wheelZoomed.emit(1 if event.angleDelta().y() > 0 else -1)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_last is not None:
            pos = event.position().toPoint()
            delta = pos - self._drag_last
            self._drag_last = pos
            if delta.x() or delta.y():
                self.panned.emit(delta.x(), delta.y())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = None
            self.unsetCursor()
        event.accept()


class SilhouettePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._silhouettes: dict[float, Silhouette] = {}
        # Camera Mode: не None -> сетка показывает ЭТИ (перспективные) силуэты вместо
        # ортографических, с тем же числом плиток (по одной на view_angle).
        self._camera_silhouettes: dict[float, Silhouette] | None = None
        self._camera_caption_suffix = ""
        # 3D-реконструкция по фото (см. докстринг класса): не None -> сетка показывает ЭТИ
        # маски (по одной на использованное наблюдение), приоритет выше camera_silhouettes.
        self._photo_observations: list[tuple[str, np.ndarray]] | None = None
        # Явная 3x3-сетка camera_hull_use_ends (main_window._SIM_RIG_ANGLES, задача "3D-
        # реконструкция объекта по кропам сетки ракурсов", пункт 4 запроса пользователя
        # 2026-07-25): не None -> ОРДЕР тайлов задаётся ВЫЗЫВАЮЩЕЙ стороной явно (не сортировкой
        # по углу, как camera_silhouettes) — под конкретную раскладку start/center/end top+side +
        # center остальных, совпадающую с robozon `sim/cv_grid.py::_COMPOSITE_LAYOUT`. Приоритет
        # выше camera_silhouettes, ниже photo_observations.
        self._camera_grid_tiles: list[tuple[str, np.ndarray]] | None = None
        self._camera_grid_caption_suffix = ""

        # Общий вид зума/пана: `_zoom` — во сколько раз приближено (1.0 — целиком), `_center` —
        # нормализованный [0, 1] центр видимой области, ОДИН на все плитки сетки. Переживает
        # `_rebuild()` (не сбрасывается при пересчёте/смене режима) — только явный двойной клик
        # или объяснимая смена набора ракурсов (см. _rebuild) возвращают его к дефолту.
        self._zoom = ZOOM_MIN
        self._center = (0.5, 0.5)
        # Полные (некадрированные) RGB-рендеры и виджеты плиток по углу — чтобы зум/пан
        # пересчитывали только пиксмапы (_refresh_images), не пересоздавая всю сетку виджетов.
        self._rendered: dict[float | str, np.ndarray] = {}
        self._image_labels: dict[float | str, _ZoomableImageLabel] = {}

        self.fill_checkbox = QCheckBox("Заливка (сегментация)")
        self.fill_checkbox.toggled.connect(lambda _checked: self._rebuild())

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        # Минимум — сетка 3x3 без прокрутки; больше плиток (реконструкция по многим фото)
        # прокручивается ВНУТРИ этой области, не растягивая саму панель/окно (см. константы
        # выше и заметку задачи "добавить в Panel 2 скролл").
        cell_w = THUMBNAIL_PX + _TILE_MARGIN_PX
        cell_h = THUMBNAIL_PX + _TITLE_HEIGHT_PX + _TILE_MARGIN_PX
        self._grid_host.setMinimumSize(
            cell_w * _MIN_VISIBLE_TILES, cell_h * _MIN_VISIBLE_TILES
        )

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll_area.setWidget(self._grid_host)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.fill_checkbox)
        layout.addWidget(self._scroll_area, stretch=1)

    def set_silhouettes(self, silhouettes: dict[float, Silhouette]) -> None:
        self._silhouettes = silhouettes
        self._rebuild()

    def set_camera_silhouettes(
        self, silhouettes: dict[float, Silhouette] | None, caption_suffix: str = ""
    ) -> None:
        """Camera Mode: показать сетку перспективных силуэтов (один на каждый view_angle,
        та же раскладка, что и у ортографической сетки) вместо неё. `caption_suffix`
        дописывается к подписи угла каждой плитки (например, "80° FOV, центр").
        `silhouettes=None` — вернуться к ортографической сетке (режим выключен)."""
        self._camera_silhouettes = silhouettes
        self._camera_caption_suffix = caption_suffix
        self._rebuild()

    def set_camera_silhouette_grid(
        self, tiles: list[tuple[str, np.ndarray]] | None, caption_suffix: str = ""
    ) -> None:
        """Явная 3x3-сетка (см. докстринг `__init__`) — `tiles` уже готовый, ОРДЕР важен
        (row-major, как `_COMPOSITE_LAYOUT` в robozon), в отличие от `set_camera_silhouettes`
        (сортировка по углу). `tiles=None` — вернуться к обычному режиму (`set_camera_
        silhouettes`/аналитический)."""
        self._camera_grid_tiles = tiles
        self._camera_grid_caption_suffix = caption_suffix
        self._rebuild()

    def set_photo_observations(self, observations: list[tuple[str, np.ndarray]] | None) -> None:
        """3D-реконструкция по фото: `observations` — (подпись, маска) для КАЖДОГО наблюдения,
        реально использованного в последней реконструкции (`PhotoCaptureDialog.
        _on_reconstruct_clicked`) — прямые и через отражение вперемешку, подпись уже включает
        имя фото + область ("photo_2.jpg — лента" / "photo_2.jpg — отражение"). Наивысший
        приоритет отображения (см. докстринг класса). `None` — вернуться к Camera/Analytical Mode."""
        self._photo_observations = observations
        self._rebuild()

    def _rebuild(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._image_labels = {}
        self._rendered = {}

        filled = self.fill_checkbox.isChecked()
        if self._photo_observations is not None:
            tiles: list[tuple[str, np.ndarray]] = list(self._photo_observations)
        elif self._camera_grid_tiles is not None:
            suffix = self._camera_grid_caption_suffix
            tiles = [
                (label + (f" — {suffix}" if suffix else ""), mask)
                for label, mask in self._camera_grid_tiles
            ]
        elif self._camera_silhouettes is not None:
            suffix = self._camera_caption_suffix
            tiles = [
                (f"{angle:g}°" + (f" — {suffix}" if suffix else ""), self._camera_silhouettes[angle].mask)
                for angle in sorted(self._camera_silhouettes)
            ]
        else:
            tiles = [(f"{angle:g}°", self._silhouettes[angle].mask) for angle in sorted(self._silhouettes)]

        cols = max(1, math.ceil(math.sqrt(len(tiles))))

        for i, (label, mask) in enumerate(tiles):
            row, col = divmod(i, cols)
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(2, 2, 2, 2)

            title = QLabel(label)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            image_label = _ZoomableImageLabel()
            image_label.wheelZoomed.connect(self._on_wheel_zoom)
            image_label.doubleClicked.connect(self._reset_view)
            image_label.panned.connect(self._on_pan)

            vbox.addWidget(title)
            vbox.addWidget(image_label)
            self._grid.addWidget(container, row, col)

            self._rendered[label] = _render_mask(mask, filled)
            self._image_labels[label] = image_label

        self._refresh_images()

    def _on_wheel_zoom(self, direction: int) -> None:
        factor = ZOOM_STEP if direction > 0 else 1.0 / ZOOM_STEP
        self._zoom = float(np.clip(self._zoom * factor, ZOOM_MIN, ZOOM_MAX))
        self._clamp_center()
        self._refresh_images()

    def _on_pan(self, dx_px: int, dy_px: int) -> None:
        if self._zoom <= ZOOM_MIN:
            return  # целиком помещается — двигать нечего
        half = 0.5 / self._zoom
        # Перетаскивание "как фото под курсором": контент следует за курсором, значит видимое
        # окно смещается в противоположную сторону движения мыши.
        norm_dx = (dx_px / THUMBNAIL_PX) * (2.0 * half)
        norm_dy = (dy_px / THUMBNAIL_PX) * (2.0 * half)
        cx, cy = self._center
        self._center = (cx - norm_dx, cy - norm_dy)
        self._clamp_center()
        self._refresh_images()

    def _reset_view(self) -> None:
        if self._zoom == ZOOM_MIN and self._center == (0.5, 0.5):
            return
        self._zoom = ZOOM_MIN
        self._center = (0.5, 0.5)
        self._refresh_images()

    def _clamp_center(self) -> None:
        half = 0.5 / self._zoom
        cx, cy = self._center
        cx = float(np.clip(cx, half, 1.0 - half))
        cy = float(np.clip(cy, half, 1.0 - half))
        self._center = (cx, cy)

    def _refresh_images(self) -> None:
        half = 0.5 / self._zoom
        cx, cy = self._center
        for angle, label in self._image_labels.items():
            img = self._rendered[angle]
            h, w, _ = img.shape
            x0 = int(round((cx - half) * w))
            x1 = int(round((cx + half) * w))
            y0 = int(round((cy - half) * h))
            y1 = int(round((cy + half) * h))
            x0, x1 = max(0, x0), min(w, max(x0 + 1, x1))
            y0, y1 = max(0, y0), min(h, max(y0 + 1, y1))
            label.setPixmap(_to_pixmap(img[y0:y1, x0:x1]))
