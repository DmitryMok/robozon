"""Генерирует печатный лист A4 с двумя ArUco-метками известного размера и известного взаимного
положения — для быстрого теста `roboson_tools/pose/` (детекция + solvePnP) на реальном фото,
до сборки физического рига (доска на зеркале + доска по бортам ленты).

Печатать строго БЕЗ масштабирования ("Actual size" / 100%, НЕ "Fit to page") — иначе реальный
размер метки на бумаге разойдётся с corners_mm в aruco_test_sheet.yaml и поза будет посчитана
неверно. После печати желательно проверить линейкой контрольную метку 50 мм на листе.

Запуск: python assets/aruco_test_sheet/generate_sheet.py
Результат: aruco_test_sheet.pdf и aruco_test_sheet.yaml рядом с этим скриптом.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml

OUT_DIR = Path(__file__).resolve().parent

# Мировая система координат теста: плоский лист лежит в плоскости Z=0, X — вправо по листу
# (мм от левого края), Y — ВНИЗ по листу (мм от верхнего края, как читается страница) — тот же
# порядок TL,TR,BR,BL и тот же знак Y, что уже проверен end-to-end на синтетике (см.
# tests/test_aruco_estimator.py) и на реальном примере в config/aruco_markers.yaml.
DICTIONARY_NAME = "DICT_4X4_50"
MARKER_SIDE_MM = 45.0
MARKERS = {
    20: (30.0, 120.0),   # id -> (x0, y0) = мировые координаты ВЕРХНЕГО ЛЕВОГО угла метки, мм
    21: (135.0, 120.0),
}
PAGE_W_MM, PAGE_H_MM = 210.0, 297.0  # A4
MM_PER_INCH = 25.4


def marker_corners_mm(x0: float, y0: float, side: float) -> list[list[float]]:
    return [
        [x0, y0, 0.0],
        [x0 + side, y0, 0.0],
        [x0 + side, y0 + side, 0.0],
        [x0, y0 + side, 0.0],
    ]


def build_pdf(path: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))

    fig = plt.figure(figsize=(PAGE_W_MM / MM_PER_INCH, PAGE_H_MM / MM_PER_INCH))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, PAGE_W_MM)
    ax.set_ylim(PAGE_H_MM, 0.0)  # Y растёт вниз по листу — см. докстринг про world-конвенцию
    ax.set_axis_off()

    for marker_id, (x0, y0) in MARKERS.items():
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)
        # origin="upper" (по умолчанию) кладёт img[0,0] (СОБСТВЕННЫЙ верхний левый угол
        # битмапа метки) в (left, top)extent-бокса — этим и обеспечивается совпадение с
        # TL=(x0,y0) из marker_corners_mm.
        ax.imshow(
            marker_img, cmap="gray", extent=(x0, x0 + MARKER_SIDE_MM, y0 + MARKER_SIDE_MM, y0)
        )
        ax.text(
            x0, y0 - 4, f"id={marker_id}", fontsize=9, family="monospace", va="bottom"
        )

    # Контрольный отрезок 50 мм — проверить линейкой после печати, что страница не была
    # промасштабирована принтером/PDF-вьювером.
    ruler_y = 220.0
    ax.plot([30.0, 80.0], [ruler_y, ruler_y], color="black", linewidth=1.0)
    ax.plot([30.0, 30.0], [ruler_y - 2, ruler_y + 2], color="black", linewidth=1.0)
    ax.plot([80.0, 80.0], [ruler_y - 2, ruler_y + 2], color="black", linewidth=1.0)
    ax.text(30.0, ruler_y + 8, "50 мм — проверить линейкой после печати", fontsize=8)

    ax.text(
        30.0,
        40.0,
        (
            f"roboson_tools — тестовый лист ArUco ({DICTIONARY_NAME})\n"
            f"Метки id={list(MARKERS)}, сторона {MARKER_SIDE_MM:g} мм\n"
            "Печать: 100% / Actual size, БЕЗ 'Fit to page'."
        ),
        fontsize=10,
    )

    fig.savefig(path, format="pdf")
    plt.close(fig)


def build_layout_yaml(path: Path) -> None:
    data = {
        "dictionary": DICTIONARY_NAME,
        "boards": [
            {
                "name": "test_sheet",
                "markers": [
                    {"id": marker_id, "corners_mm": marker_corners_mm(x0, y0, MARKER_SIDE_MM)}
                    for marker_id, (x0, y0) in MARKERS.items()
                ],
            }
        ],
    }
    path.write_text(
        "# Сгенерировано assets/aruco_test_sheet/generate_sheet.py — координаты соответствуют\n"
        "# распечатанному aruco_test_sheet.pdf (лист лежит плоско, Z=0).\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_pdf(OUT_DIR / "aruco_test_sheet.pdf")
    build_layout_yaml(OUT_DIR / "aruco_test_sheet.yaml")
    print("Готово:", OUT_DIR / "aruco_test_sheet.pdf", OUT_DIR / "aruco_test_sheet.yaml")
