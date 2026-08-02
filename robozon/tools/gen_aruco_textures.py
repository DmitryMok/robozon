#!/usr/bin/env python3
"""Генерация PNG-текстур ArUco-меток для чисто визуального оформления сцены
demo_mechanism.wbt (шаг 6/6, задача "Новая сцена демо исполнительного
механизма (3 шибера + зона накопления)").

Декоративная задача: метки видны в кадре CV-камер рига, но НЕ читаются
никаким кодом (нет solvePnP/детектора в этой сцене) — реального требования
к точному размеру/расположению нет, в отличие от отложенной задачи
[[Добавить визуальные ArUco-метки в сцену Webots]] (Backlog, для будущей
реальной регистрации позы камер). ID и раскладка (4 угла зеркала + 2 доски
по 2 метки на бортах ленты) СОВПАДАЮТ с уже существующей раскладкой
`roboson_tools/assets/{mirror,belt}_board/*.yaml` (словарь DICT_4X4_50,
mirror_edge_left: 22/23, mirror_edge_right: 24/25, belt_left: 26/27,
belt_right: 28/29) — только чтобы не плодить случайные ID, мировые
координаты в этой сцене никак не связаны с теми yaml (эта сцена не
измеряется).

Белый фон обязателен вокруг чёрного маркера — иначе метка сольётся с тёмным
полотном зеркала/лентой на общем рендере (см. докстринг Backlog-задачи).

Запуск (одноразовый, PNG коммитятся в assets/aruco/):
    C:\\Projects\\.venvs\\venv-cv-pytorch\\Scripts\\python.exe tools/gen_aruco_textures.py
"""
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "aruco"

DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_PX = 400        # маркер (чёрно-белый паттерн) без полей
MARGIN_PX = 60         # белое поле вокруг маркера с каждой стороны
CANVAS_PX = MARKER_PX + 2 * MARGIN_PX

# id -> имя файла (см. докстринг: совпадают с roboson_tools mirror/belt_board.yaml,
# оба маркера каждой доски — на зеркале 4 угла, на каждом борту ленты 2 метки)
MARKER_IDS = {
    22: "mirror_left_top",
    23: "mirror_left_bottom",
    24: "mirror_right_top",
    25: "mirror_right_bottom",
    26: "belt_left_top",
    27: "belt_left_bottom",
    28: "belt_right_top",
    29: "belt_right_bottom",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for marker_id, label in MARKER_IDS.items():
        marker_img = aruco.generateImageMarker(DICTIONARY, marker_id, MARKER_PX)
        canvas = np.full((CANVAS_PX, CANVAS_PX), 255, dtype=np.uint8)
        canvas[MARGIN_PX:MARGIN_PX + MARKER_PX, MARGIN_PX:MARGIN_PX + MARKER_PX] = marker_img
        out_path = OUT_DIR / f"marker_{marker_id}_{label}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
