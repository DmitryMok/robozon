"""Panel 3 — восстановленное сечение (Visual Hull), вписанная/описанная окружность,
слайдер позиции среза вдоль продольной оси X и слайдер поворота сечения вокруг вертикальной
оси (Yaw — тот же угол, что и в общей панели ориентации, см. main_window._on_section_yaw_changed:
поворот вокруг X не меняет форму сечения, т.к. срез и так строится перпендикулярно X, а вот
Pitch/Yaw задают, КАКОЕ именно сечение объекта сейчас видно)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from ...core.experiment import SectionCheckHit
from ...metrics.roundness import RoundnessResult

SLIDER_STEPS = 1000
YAW_SLIDER_SCALE = 10  # слайдер оперирует целыми, шаг 0.1°


class VisualHullPanel(QWidget):
    axisPosChanged = pyqtSignal(float)
    yawChanged = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._figure = Figure(figsize=(4, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)

        self._axis_min = 0.0
        self._axis_max = 1.0
        self._last_polygon_coords: list[tuple[float, float]] | None = None
        self._last_roundness: RoundnessResult | None = None
        self._checked_hits: list[SectionCheckHit] = []
        self._projection_coords: list[tuple[float, float]] | None = None
        self._projection_roundness: RoundnessResult | None = None
        # 3D-реконструкция по фото (см. set_photo_reconstruction) — наивысший приоритет
        # отображения, пока не None (та же идея приоритета, что у SilhouettePanel).
        self._photo_polytope_coords: list[tuple[float, float]] | None = None
        self._photo_exact_coords: list[tuple[float, float]] | None = None
        self._photo_title: str | None = None

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setValue(SLIDER_STEPS // 2)
        self.value_label = QLabel("Slice X: 0.0")

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Slice X"))
        slider_row.addWidget(self.slider, stretch=1)
        slider_row.addWidget(self.value_label)

        self.yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_slider.setRange(0, 360 * YAW_SLIDER_SCALE)
        self.yaw_value_label = QLabel("Поворот (Yaw): 0.0°")

        yaw_row = QHBoxLayout()
        yaw_row.addWidget(QLabel("Поворот сечения"))
        yaw_row.addWidget(self.yaw_slider, stretch=1)
        yaw_row.addWidget(self.yaw_value_label)

        self.show_checked_checkbox = QCheckBox("Показать проверенные сечения")

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(slider_row)
        layout.addLayout(yaw_row)
        layout.addWidget(self.show_checked_checkbox)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.yaw_slider.valueChanged.connect(self._on_yaw_slider_changed)
        self.show_checked_checkbox.toggled.connect(lambda _checked: self._redraw())

    def set_axis_range(
        self, axis_min: float, axis_max: float, current: float | None = None
    ) -> None:
        self._axis_min = axis_min
        self._axis_max = max(axis_max, axis_min + 1e-6)
        if current is None:
            current = (axis_min + axis_max) / 2
        ratio = (current - self._axis_min) / (self._axis_max - self._axis_min)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(ratio * SLIDER_STEPS)))
        self.slider.blockSignals(False)
        self.value_label.setText(f"Slice X: {current:.1f}")

    def axis_pos(self) -> float:
        ratio = self.slider.value() / SLIDER_STEPS
        return self._axis_min + ratio * (self._axis_max - self._axis_min)

    def _on_slider_changed(self, _value: int) -> None:
        pos = self.axis_pos()
        self.value_label.setText(f"Slice X: {pos:.1f}")
        self.axisPosChanged.emit(pos)

    def set_yaw(self, yaw_deg: float) -> None:
        """Синхронизировать слайдер с текущим Yaw (обычно вызывается из MainWindow после
        пересчёта — например, если Yaw был изменён из общей панели ориентации, а не отсюда)."""
        yaw_deg = yaw_deg % 360.0
        slider_value = int(round(yaw_deg * YAW_SLIDER_SCALE))
        if self.yaw_slider.value() != slider_value:
            self.yaw_slider.blockSignals(True)
            self.yaw_slider.setValue(slider_value)
            self.yaw_slider.blockSignals(False)
        self.yaw_value_label.setText(f"Поворот (Yaw): {yaw_deg:.1f}°")

    def _on_yaw_slider_changed(self, value: int) -> None:
        yaw_deg = value / YAW_SLIDER_SCALE
        self.yaw_value_label.setText(f"Поворот (Yaw): {yaw_deg:.1f}°")
        self.yawChanged.emit(yaw_deg)

    def show_round_axis_marker(self, azim_deg: float, x_pos: float) -> None:
        """Отображение (НЕ интерактивное, без эмита сигналов) слайдеров Slice X / Yaw в
        положении, соответствующем найденной оси переката — вызывается из MainWindow._recompute
        поверх обычной синхронизации, пока активен показ круглой проекции, чтобы слайдеры не
        оставались в положении случайного «последнего обычного среза». `x_pos` — обычно центр
        объекта (0: см. MainWindow — начало координат инвариантно относительно поворота),
        `azim_deg` — азимут оси в текущей ориентации. Подъём (elevation) оси слайдером не
        отображается — Yaw даёт только поворот вокруг вертикали, второй степени свободы у
        текущих слайдеров нет."""
        ratio = (x_pos - self._axis_min) / (self._axis_max - self._axis_min)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(ratio * SLIDER_STEPS)))
        self.slider.blockSignals(False)
        self.value_label.setText(f"Slice X: {x_pos:.1f} (ось переката)")

        az = azim_deg % 360.0
        slider_value = int(round(az * YAW_SLIDER_SCALE))
        self.yaw_slider.blockSignals(True)
        self.yaw_slider.setValue(slider_value)
        self.yaw_slider.blockSignals(False)
        self.yaw_value_label.setText(f"Поворот (Yaw): {az:.1f}° (азимут оси переката)")

    def set_checked_sections(self, hits: list[SectionCheckHit]) -> None:
        """Контуры сечений реконструкции из последнего запуска "Проверить модель" (см.
        MainWindow._on_check_model_clicked) — сечения строятся в той же ориентации,
        что сейчас отображается."""
        self._checked_hits = list(hits)
        self._redraw()

    def set_projection(
        self,
        coords: list[tuple[float, float]] | None,
        roundness: RoundnessResult | None,
    ) -> None:
        """Показ круглой проекции (вдоль найденной оси переката) вместо сечения.
        `coords=None` — вернуться к обычному отображению сечения."""
        self._projection_coords = coords
        self._projection_roundness = roundness
        self._redraw()

    def set_result(
        self,
        polygon_coords: list[tuple[float, float]] | None,
        roundness: RoundnessResult | None,
    ) -> None:
        self._last_polygon_coords = polygon_coords
        self._last_roundness = roundness
        self._redraw()

    def set_photo_reconstruction(
        self,
        polytope_coords: list[tuple[float, float]] | None,
        exact_coords: list[tuple[float, float]] | None,
        title: str | None,
    ) -> None:
        """3D-реконструкция по фото (`PhotoCaptureDialog._on_reconstruct_clicked`) — контур
        "вид сверху" вдоль нормали ленты (см. `visual_hull.pose_carving.top_view_outline`,
        `segmentation.region_filter.fit_belt_up_normal`), не срез: у мировых координат рига нет
        заранее известной продольной оси, вдоль которой строится обычный срез Slice X. Оба
        контура (`polytope`/`exact`) не обязаны быть одновременно не-None — метод реконструкции
        мог не сойтись для одного из них (см. `pose_carving.carve_polytope`/`carve_exact`).
        Наивысший приоритет отображения — перекрывает обычное сечение/круглую проекцию, пока
        `title` не None; `set_photo_reconstruction(None, None, None)` возвращает обычный режим."""
        self._photo_polytope_coords = polytope_coords
        self._photo_exact_coords = exact_coords
        self._photo_title = title
        self._redraw()

    def _redraw(self) -> None:
        self._ax.clear()

        if self._photo_title is not None:
            if self._photo_polytope_coords is not None:
                coords = np.array(self._photo_polytope_coords)
                self._ax.plot(coords[:, 0], coords[:, 1], "-", color="#0891b2")
                self._ax.fill(coords[:, 0], coords[:, 1], color="#67e8f9", alpha=0.35)
            if self._photo_exact_coords is not None:
                coords = np.array(self._photo_exact_coords)
                self._ax.plot(coords[:, 0], coords[:, 1], "-", color="#c2410c")
                self._ax.fill(coords[:, 0], coords[:, 1], color="#fdba74", alpha=0.35)
            self._ax.set_title(self._photo_title, fontsize=8, wrap=True)
            self._ax.set_axis_on()
            self._ax.set_aspect("equal", adjustable="datalim")
            self._canvas.draw_idle()
            return

        # Режим "круглая проекция": вместо сечения показывается проекция формы вдоль
        # найденной оси переката (управляется чек-боксом в Panel 4)
        if self._projection_coords is not None:
            coords = np.array(self._projection_coords)
            self._ax.plot(coords[:, 0], coords[:, 1], "-", color="#333333")
            self._ax.fill(coords[:, 0], coords[:, 1], color="#ffe0b2", alpha=0.6)
            self._draw_circles(self._projection_roundness)
            self._ax.set_title("Круглая проекция (вдоль оси переката)", fontsize=9)
            self._ax.set_aspect("equal", adjustable="datalim")
            self._canvas.draw_idle()
            return

        if self.show_checked_checkbox.isChecked() and self._checked_hits:
            for hit in self._checked_hits:
                coords = np.array(hit.polygon_coords)
                color = "#2e7d32" if hit.passed else "#9e9e9e"
                self._ax.plot(
                    coords[:, 0], coords[:, 1], "-", color=color, linewidth=0.7, alpha=0.6
                )

        if self._last_polygon_coords is None:
            self._ax.text(0.5, 0.5, "Сечение вырождено", ha="center", va="center")
            self._ax.set_axis_off()
            self._canvas.draw_idle()
            return

        self._ax.set_axis_on()
        coords = np.array(self._last_polygon_coords)
        self._ax.plot(coords[:, 0], coords[:, 1], "-", color="#333333")
        self._ax.fill(coords[:, 0], coords[:, 1], color="#cfe0f0", alpha=0.5)

        self._draw_circles(self._last_roundness)

        self._ax.set_aspect("equal", adjustable="datalim")
        self._canvas.draw_idle()

    def _draw_circles(self, roundness: RoundnessResult | None) -> None:
        if roundness is None:
            return
        self._ax.add_patch(
            Circle(
                roundness.center_out,
                roundness.r_out,
                fill=False,
                color="tab:red",
                linewidth=1.5,
            )
        )
        self._ax.add_patch(
            Circle(
                roundness.center_in,
                roundness.r_in,
                fill=False,
                color="tab:green",
                linewidth=1.5,
            )
        )
