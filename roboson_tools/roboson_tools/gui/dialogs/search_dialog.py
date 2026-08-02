"""Диалог поиска контрпримеров: перебор Roll/Pitch/Yaw с заданным шагом (в фоновом потоке)."""

from __future__ import annotations

import threading

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...geometry.mesh_io import Mesh
from ...search.counterexample import CounterexampleHit, search_counterexamples


class _SearchWorker(QThread):
    finishedWithHits = pyqtSignal(list)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        mesh: Mesh,
        view_angles: list[float],
        step_deg: float,
        threshold: float,
        resolution_px: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mesh = mesh
        self._view_angles = view_angles
        self._step_deg = step_deg
        self._threshold = threshold
        self._resolution_px = resolution_px
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Просит перебор остановиться и корректно погасить пул процессов.

        Не завершает поток немедленно — `run()` возвращается только после того, как
        `search_counterexamples` дождётся штатной остановки ProcessPoolExecutor.
        """
        self._cancel_event.set()

    def run(self) -> None:
        try:
            hits = search_counterexamples(
                self._mesh,
                self._view_angles,
                step_deg=self._step_deg,
                roundness_threshold=self._threshold,
                resolution_px=self._resolution_px,
                cancel_event=self._cancel_event,
            )
        except Exception as exc:  # noqa: BLE001 — показать пользователю, не уронить GUI
            self.failed.emit(str(exc))
            return
        if self._cancel_event.is_set():
            self.cancelled.emit()
            return
        self.finishedWithHits.emit(hits)


class SearchDialog(QDialog):
    orientationPicked = pyqtSignal(float, float, float)

    def __init__(
        self,
        mesh: Mesh,
        view_angles: list[float],
        is_round_reference: bool | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Поиск контрпримеров")
        self.resize(500, 400)

        self._mesh = mesh
        self._view_angles = view_angles
        self._worker: _SearchWorker | None = None
        self._hits: list[CounterexampleHit] = []

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 90.0)
        self.step_spin.setValue(5.0)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.8)

        self.resolution_edit = QLineEdit("512")

        form = QFormLayout()
        form.addRow("Шаг, °", self.step_spin)
        form.addRow("Порог k", self.threshold_spin)
        form.addRow("Разрешение, px", self.resolution_edit)

        self.run_button = QPushButton("Запустить")
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.setEnabled(False)
        self.status_label = QLabel("")

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Roll", "Pitch", "Yaw", "Ratio"])
        self.table.cellDoubleClicked.connect(self._on_row_activated)

        layout = QVBoxLayout(self)
        layout.addLayout(form)

        if is_round_reference:
            warning = QLabel(
                "Внимание: is_round_reference=true для этого объекта — найденные случаи "
                "НЕ являются настоящими контрпримерами (объект действительно круглый)."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #c62828;")
            layout.addWidget(warning)

        layout.addWidget(self.run_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(QLabel("Двойной клик по строке — применить ориентацию в главном окне"))

        self.run_button.clicked.connect(self._run_search)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)

    def _run_search(self) -> None:
        try:
            resolution = int(self.resolution_edit.text())
        except ValueError:
            resolution = 512

        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Идёт перебор...")
        self.table.setRowCount(0)

        self._worker = _SearchWorker(
            self._mesh,
            self._view_angles,
            self.step_spin.value(),
            self.threshold_spin.value(),
            resolution,
            self,
        )
        self._worker.finishedWithHits.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_cancel_clicked(self) -> None:
        if self._worker is None:
            return
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Отмена — ждём остановки воркеров...")
        self._worker.cancel()

    def _on_finished(self, hits: list[CounterexampleHit]) -> None:
        self._hits = hits
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText(f"Готово: {len(hits)} случаев с k >= порога")
        self.table.setRowCount(len(hits))
        for row, hit in enumerate(hits):
            self.table.setItem(row, 0, QTableWidgetItem(f"{hit.roll_deg:g}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{hit.pitch_deg:g}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{hit.yaw_deg:g}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{hit.k:.4f}"))

    def _on_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText(f"Ошибка: {message}")

    def _on_cancelled(self) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Отменено")

    def closeEvent(self, event) -> None:  # noqa: N802 — переопределение метода Qt
        """Не даёт окну закрыться, пока не остановлен фоновый ProcessPoolExecutor.

        Без этого закрытие диалога/приложения во время перебора обрывает QThread
        вместе с процессом, и воркеры ProcessPoolExecutor остаются висеть
        осиротевшими (см. утечку 2026-07-08 — 12 процессов python.exe от этого
        диалога провисели в системе неделю).
        """
        if self._worker is not None and self._worker.isRunning():
            self.status_label.setText("Закрытие — ждём остановки воркеров...")
            self._worker.cancel()
            self._worker.wait()
        super().closeEvent(event)

    def _on_row_activated(self, row: int, _col: int) -> None:
        hit = self._hits[row]
        self.orientationPicked.emit(hit.roll_deg, hit.pitch_deg, hit.yaw_deg)
