"""Ввод произвольного набора view_angles + пресеты из ТЗ."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QWidget


def format_angles(angles: list[float]) -> str:
    return ", ".join(f"{a:g}" for a in angles)


def parse_angles(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


class AngleSetInput(QWidget):
    anglesChanged = pyqtSignal(list)

    def __init__(self, presets: list[list[float]], parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.combo = QComboBox()
        self.combo.addItem("Пресет...")
        for preset in presets:
            self.combo.addItem(format_angles(preset))
        self._presets = presets

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("0, 45, 90, 135")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo)
        layout.addWidget(self.line_edit, stretch=1)

        self.combo.currentIndexChanged.connect(self._on_preset_selected)
        self.line_edit.editingFinished.connect(self._on_text_changed)

    def _on_preset_selected(self, index: int) -> None:
        if index <= 0:
            return
        preset = self._presets[index - 1]
        self.set_angles(preset)
        self.anglesChanged.emit(preset)

    def _on_text_changed(self) -> None:
        try:
            angles = parse_angles(self.line_edit.text())
        except ValueError:
            return
        if angles:
            self.anglesChanged.emit(angles)

    def angles(self) -> list[float]:
        try:
            return parse_angles(self.line_edit.text())
        except ValueError:
            return []

    def set_angles(self, angles: list[float]) -> None:
        self.line_edit.setText(format_angles(angles))
