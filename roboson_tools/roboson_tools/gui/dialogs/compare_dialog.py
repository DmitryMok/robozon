"""Диалог сравнения конфигураций view_angles для текущей ориентации объекта."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.experiment import Orientation, compare_configurations
from ...geometry.mesh_io import Mesh


class CompareDialog(QDialog):
    def __init__(
        self,
        mesh: Mesh,
        orientation: Orientation,
        axis_pos: float | None,
        presets: list[list[float]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Сравнение конфигураций")
        self.resize(500, 400)

        self._mesh = mesh
        self._orientation = orientation
        self._axis_pos = axis_pos

        preset_text = "\n".join(", ".join(f"{a:g}" for a in p) for p in presets)
        self.angle_sets_edit = QPlainTextEdit(preset_text)

        self.run_button = QPushButton("Сравнить")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["view_angles", "Rin/Rout", "PASS/FAIL"])

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Наборы углов (по одному на строку):"))
        layout.addWidget(self.angle_sets_edit)
        layout.addWidget(self.run_button)
        layout.addWidget(self.table, stretch=1)

        self.run_button.clicked.connect(self._run_compare)

    def _run_compare(self) -> None:
        angle_sets: list[list[float]] = []
        for line in self.angle_sets_edit.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                angle_sets.append([float(x.strip()) for x in line.split(",") if x.strip()])
            except ValueError:
                continue

        entries = compare_configurations(
            self._mesh, self._orientation, angle_sets, axis_pos=self._axis_pos
        )

        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            angles_str = ", ".join(f"{a:g}" for a in entry.view_angles_deg)
            self.table.setItem(row, 0, QTableWidgetItem(angles_str))
            if entry.roundness is None:
                self.table.setItem(row, 1, QTableWidgetItem("—"))
                self.table.setItem(row, 2, QTableWidgetItem("вырождено"))
            else:
                self.table.setItem(row, 1, QTableWidgetItem(f"{entry.roundness.k:.4f}"))
                verdict = "PASS" if entry.roundness.passed else "FAIL"
                self.table.setItem(row, 2, QTableWidgetItem(verdict))
