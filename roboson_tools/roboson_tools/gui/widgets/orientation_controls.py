"""Слайдер + числовой ввод для Roll/Pitch/Yaw, диапазон 0..360°."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QSlider,
    QVBoxLayout,
    QWidget,
)

SLIDER_SCALE = 10  # слайдер оперирует целыми, шаг 0.1°


class _AxisRow(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 360 * SLIDER_SCALE)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(0.0, 360.0)
        self.spinbox.setDecimals(1)
        self.spinbox.setSingleStep(0.5)
        self.spinbox.setSuffix("°")

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.slider, stretch=1)
        row_layout.addWidget(self.spinbox)
        layout.addRow(label, row)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, value: int) -> None:
        deg = value / SLIDER_SCALE
        if abs(self.spinbox.value() - deg) > 1e-9:
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(deg)
            self.spinbox.blockSignals(False)
        self.valueChanged.emit(deg)

    def _on_spinbox_changed(self, value: float) -> None:
        slider_value = round(value * SLIDER_SCALE)
        if self.slider.value() != slider_value:
            self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.spinbox.value()

    def set_value(self, deg: float) -> None:
        self.spinbox.setValue(deg)


class OrientationControls(QWidget):
    """Roll/Pitch/Yaw — три синхронизированные пары слайдер+спинбокс."""

    orientationChanged = pyqtSignal(float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.roll_row = _AxisRow("Roll")
        self.pitch_row = _AxisRow("Pitch")
        self.yaw_row = _AxisRow("Yaw")
        # Тестовые STL при исходной (0,0,0) ориентации в основном лежат «на боку» относительно
        # ленты — roll=90° даёт визуально правильное начальное положение для всего набора.
        self.roll_row.set_value(90.0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.roll_row)
        layout.addWidget(self.pitch_row)
        layout.addWidget(self.yaw_row)

        for row in (self.roll_row, self.pitch_row, self.yaw_row):
            row.valueChanged.connect(self._emit_changed)

    def _emit_changed(self, _value: float) -> None:
        self.orientationChanged.emit(*self.orientation())

    def orientation(self) -> tuple[float, float, float]:
        return self.roll_row.value(), self.pitch_row.value(), self.yaw_row.value()

    def set_orientation(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
        self.roll_row.set_value(roll_deg)
        self.pitch_row.set_value(pitch_deg)
        self.yaw_row.set_value(yaw_deg)
