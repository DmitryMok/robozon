#!/usr/bin/env python3
"""Генерация PNG-текстур с буквами C/D для подписи боковой стенки ролл-кейджей
в demo_mechanism.wbt (по просьбе пользователя — чтобы визуально было видно,
какой кейдж соответствует какой зоне продакшн-сортировщика: cage1 (round) ->
D, cage2 (oversize) -> C, см. `sim/router.py::CATEGORY_TO_ZONE`). Чисто
декоративная текстура, как и `tools/gen_aruco_textures.py` — не читается
никаким кодом.

Запуск (одноразовый, PNG коммитятся в assets/cage_labels/):
    C:\\Projects\\.venvs\\venv-cv-pytorch\\Scripts\\python.exe tools/gen_cage_labels.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets" / "cage_labels"

CANVAS_PX = 400
MARGIN_PX = 30
FONT = cv2.FONT_HERSHEY_DUPLEX
THICKNESS = 14
LETTERS = ["C", "D"]


def _fit_font_scale(letter: str, target_px: int) -> float:
    """Подбор font_scale так, чтобы высота буквы влезла в target_px (без
    готовой формулы у cv2 под целевой размер в пикселях — перебор)."""
    scale = 1.0
    while True:
        (w, h), _ = cv2.getTextSize(letter, FONT, scale, THICKNESS)
        if max(w, h) >= target_px:
            return scale
        scale += 0.5


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = CANVAS_PX - 2 * MARGIN_PX
    for letter in LETTERS:
        canvas = np.full((CANVAS_PX, CANVAS_PX, 3), 255, dtype=np.uint8)
        scale = _fit_font_scale(letter, target)
        (w, h), baseline = cv2.getTextSize(letter, FONT, scale, THICKNESS)
        x = (CANVAS_PX - w) // 2
        y = (CANVAS_PX + h) // 2
        cv2.putText(canvas, letter, (x, y), FONT, scale, (20, 20, 20), THICKNESS, cv2.LINE_AA)
        out_path = OUT_DIR / f"label_{letter}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
