"""Проверяет реальное фото листа `aruco_test_sheet.pdf` — печатает статус детекции/solvePnP и
восстановленную позицию камеры. Интринсики берутся из `config/camera_intrinsics.yaml` (тот же
файл использует GUI, `roboson_tools/pose/camera_pose.py::load_camera_intrinsics`) — ПЕРЕД
использованием поправьте там `resolution_px` под реальное разрешение фото (обязательно) и,
по возможности, `fx`/`fy`/`cx`/`cy` под реальную калибровку камеры (иначе позиция будет грубой
прикидкой, а не точным результатом — детекция и сам расчёт при этом всё равно отработают).

Запуск:
    python assets/aruco_test_sheet/check_photo.py путь/к/фото.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roboson_tools.pose.aruco_estimator import estimate_camera_pose  # noqa: E402
from roboson_tools.pose.camera_pose import load_camera_intrinsics  # noqa: E402
from roboson_tools.pose.marker_layout import load_marker_layout  # noqa: E402

LAYOUT_PATH = Path(__file__).resolve().parent / "aruco_test_sheet.yaml"


def main() -> int:
    if len(sys.argv) != 2:
        print("Использование: check_photo.py путь/к/фото.jpg")
        return 1

    photo_path = Path(sys.argv[1])
    image = cv2.imread(str(photo_path))
    if image is None:
        print(f"Не удалось прочитать {photo_path}")
        return 1

    intrinsics = load_camera_intrinsics()
    if intrinsics.resolution_px != image.shape[:2]:
        print(
            f"ВНИМАНИЕ: resolution_px в config/camera_intrinsics.yaml = "
            f"{intrinsics.resolution_px}, а у фото {image.shape[:2]} (rows, cols) — "
            f"поправьте конфиг под реальное фото перед тем, как доверять результату."
        )

    layout = load_marker_layout(LAYOUT_PATH)
    result = estimate_camera_pose(image, layout, intrinsics)

    print("Статус:", result.status, "—", result.message)
    print("Метки использованы:", result.marker_ids_used)
    if result.pose is not None:
        pos = result.pose.camera_pos_world
        print(f"Позиция камеры, мм (x,y,z листа): {pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}")
        print(f"Ошибка репроекции, px: {result.reprojection_error_px:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
