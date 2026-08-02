"""Диалог настройки рига камер: список (угол, дистанция) камер + именованные пресеты.

Заменяет прежний блок "список углов (AngleSetInput) + один общий camera_distance спинбокс" —
реальный риг имеет разную дистанцию на разных азимутах (прямые/зеркальные ракурсы, см.
core/camera_rig.py). fov/разрешение сюда не входят — общие настройки на весь риг (см.
main_window.py, отдельные спинбоксы camera_fov/camera_resolution).
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import pyqtSignal

from ...core.camera_rig import (
    CameraRigPreset,
    CameraSpec,
    apply_distance_to_all,
    delete_preset,
    load_camera_rig_presets,
    save_camera_rig_presets,
    upsert_preset,
)

_COL_ANGLE = 0
_COL_DISTANCE = 1


class CameraRigDialog(QDialog):
    rigChanged = pyqtSignal(list)  # list[CameraSpec]

    def __init__(self, initial_cameras: list[CameraSpec], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройка рига камер")
        self.resize(480, 420)

        self._presets = load_camera_rig_presets()

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Пресет...")
        for preset in self._presets:
            self.preset_combo.addItem(preset.name)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Угол, °", "Дистанция, мм"])

        self.add_row_button = QPushButton("+ камера")
        self.remove_row_button = QPushButton("− камера")
        self.apply_to_all_button = QPushButton("Дистанцию выбранной строки — на все")

        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Имя пресета")
        self.save_preset_button = QPushButton("Сохранить как пресет")
        self.delete_preset_button = QPushButton("Удалить пресет")

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #c62828;")

        self.apply_button = QPushButton("Применить")
        self.cancel_button = QPushButton("Отмена")

        layout = QVBoxLayout(self)
        layout.addWidget(self.preset_combo)
        layout.addWidget(self.table, stretch=1)

        row_buttons = QHBoxLayout()
        row_buttons.addWidget(self.add_row_button)
        row_buttons.addWidget(self.remove_row_button)
        row_buttons.addWidget(self.apply_to_all_button)
        layout.addLayout(row_buttons)

        preset_row = QHBoxLayout()
        preset_row.addWidget(self.preset_name_edit, stretch=1)
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.delete_preset_button)
        layout.addLayout(preset_row)

        layout.addWidget(self.status_label)

        action_row = QHBoxLayout()
        action_row.addWidget(self.apply_button)
        action_row.addWidget(self.cancel_button)
        layout.addLayout(action_row)

        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self.add_row_button.clicked.connect(self._on_add_row)
        self.remove_row_button.clicked.connect(self._on_remove_row)
        self.apply_to_all_button.clicked.connect(self._on_apply_to_all)
        self.save_preset_button.clicked.connect(self._on_save_preset)
        self.delete_preset_button.clicked.connect(self._on_delete_preset)
        self.apply_button.clicked.connect(self._on_apply)
        self.cancel_button.clicked.connect(self.reject)

        self._set_table_cameras(initial_cameras)

    def _set_table_cameras(self, cameras: list[CameraSpec]) -> None:
        self.table.setRowCount(len(cameras))
        for row, cam in enumerate(cameras):
            self.table.setItem(row, _COL_ANGLE, QTableWidgetItem(f"{cam.angle_deg:g}"))
            self.table.setItem(row, _COL_DISTANCE, QTableWidgetItem(f"{cam.distance_mm:g}"))

    def _read_table(self) -> list[CameraSpec] | None:
        """Читает текущее содержимое таблицы -> список камер, либо None (и сообщение в
        status_label) при нечисловом вводе — не даёт закрыть диалог/сохранить пресет с
        "мусором" в ячейках."""
        cameras: list[CameraSpec] = []
        for row in range(self.table.rowCount()):
            angle_item = self.table.item(row, _COL_ANGLE)
            distance_item = self.table.item(row, _COL_DISTANCE)
            try:
                angle = float(angle_item.text()) if angle_item else 0.0
                distance = float(distance_item.text()) if distance_item else 0.0
            except ValueError:
                self.status_label.setText(f"Строка {row + 1}: угол/дистанция должны быть числами")
                return None
            cameras.append(CameraSpec(angle_deg=angle, distance_mm=distance))
        if not cameras:
            self.status_label.setText("Риг должен содержать хотя бы одну камеру")
            return None
        angles = [c.angle_deg for c in cameras]
        if len(angles) != len(set(angles)):
            self.status_label.setText("Углы камер не должны повторяться")
            return None
        self.status_label.setText("")
        return cameras

    def _on_preset_selected(self, index: int) -> None:
        if index <= 0:
            return
        preset = self._presets[index - 1]
        self._set_table_cameras(preset.cameras)
        self.preset_name_edit.setText(preset.name if not preset.readonly else "")

    def _on_add_row(self) -> None:
        row = self.table.rowCount()
        self.table.setRowCount(row + 1)
        self.table.setItem(row, _COL_ANGLE, QTableWidgetItem("0"))
        self.table.setItem(row, _COL_DISTANCE, QTableWidgetItem("2000"))

    def _on_remove_row(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            row = self.table.rowCount() - 1
        if row >= 0:
            self.table.removeRow(row)

    def _on_apply_to_all(self) -> None:
        cameras = self._read_table()
        if cameras is None:
            return
        row = self.table.currentRow()
        if row < 0:
            row = 0
        try:
            updated = apply_distance_to_all(cameras, source_angle_deg=cameras[row].angle_deg)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._set_table_cameras(updated)

    def _on_save_preset(self) -> None:
        cameras = self._read_table()
        if cameras is None:
            return
        name = self.preset_name_edit.text().strip()
        if not name:
            self.status_label.setText("Укажите имя пресета для сохранения")
            return
        try:
            updated = upsert_preset(self._presets, CameraRigPreset(name=name, cameras=cameras))
            save_camera_rig_presets(updated)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._presets = updated
        self._reload_preset_combo(select_name=name)
        self.status_label.setText(f"Пресет {name!r} сохранён")

    def _on_delete_preset(self) -> None:
        name = self.preset_name_edit.text().strip()
        if not name:
            self.status_label.setText("Укажите имя пресета для удаления")
            return
        try:
            updated = delete_preset(self._presets, name)
            save_camera_rig_presets(updated)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._presets = updated
        self._reload_preset_combo()
        self.status_label.setText(f"Пресет {name!r} удалён")

    def _reload_preset_combo(self, select_name: str | None = None) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Пресет...")
        for preset in self._presets:
            self.preset_combo.addItem(preset.name)
        if select_name is not None:
            idx = self.preset_combo.findText(select_name)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def _on_apply(self) -> None:
        cameras = self._read_table()
        if cameras is None:
            return
        self.rigChanged.emit(cameras)
        self.accept()
