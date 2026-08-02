"""Диалог экспорта текущего эксперимента в директорию."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.experiment import EvaluationResult, Orientation
from ...export.exporter import export_experiment


class ExportDialog(QDialog):
    def __init__(
        self,
        stl_path: Path,
        orientation: Orientation,
        result: EvaluationResult,
        default_out_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Экспорт эксперимента")

        self._stl_path = stl_path
        self._orientation = orientation
        self._result = result

        self.path_edit = QLineEdit(str(default_out_dir))
        browse_button = QPushButton("Обзор...")
        browse_button.clicked.connect(self._browse)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse_button)

        self.export_button = QPushButton("Экспортировать")
        self.status_label = QLabel("")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Директория для сохранения:"))
        layout.addLayout(path_row)
        layout.addWidget(self.export_button)
        layout.addWidget(self.status_label)

        self.export_button.clicked.connect(self._do_export)

    def _browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Выбрать директорию", self.path_edit.text()
        )
        if directory:
            self.path_edit.setText(directory)

    def _do_export(self) -> None:
        out_dir = Path(self.path_edit.text())
        try:
            export_experiment(self._stl_path, self._orientation, self._result, out_dir)
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Ошибка: {exc}")
            return
        self.status_label.setText(f"Сохранено в {out_dir}")
