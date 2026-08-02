"""Точка входа GUI-приложения."""

from __future__ import annotations

import sys

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow

# Задаёт неявный namespace для QSettings() без аргументов (главное окно сохраняет туда параметры
# Camera Mode между запусками, см. MainWindow._restore_camera_settings/_save_camera_settings) —
# должно быть установлено ДО первого использования QSettings.
QCoreApplication.setOrganizationName("roboson_tools")
QCoreApplication.setApplicationName("roboson_tools_gui")


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
