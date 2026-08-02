"""Генерирует печатный лист A4 с 4 ArUco-метками для монтажа на края зеркала теста реальной
камеры (см. заметку задачи "3D-реконструкция объекта по реальным фото...", раздел про мостик
зеркала): по 2 метки на левый и правый край зеркала (вдоль его стороны 300 мм).

Раскладка: ОДИН лист A4 (портрет) делится вертикальной линией разреза пополам на две узкие
колонки (105 мм каждая) — «разрежу вдоль лист A4» (разрез вдоль длинной стороны листа). Левая
колонка -> левый край зеркала, правая колонка -> правый край зеркала.

ВЫРАВНИВАНИЕ — по ГОРИЗОНТАЛИ, от ВНЕШНЕГО (не разрезаемого) края колонки: у левой колонки это
левый край исходного листа (x=0), у правой — правый край исходного листа (x=210 мм). Именно этот
край (заводской, ровный — не линия разреза) кладётся вплотную к краю зеркала; отступ метки от
него — 20 мм (2 см) — и есть то смещение метки внутрь зеркала от его физического края, на
которое ориентируется расчёт позы. Разрезанный (внутренний) край колонки ни на что не
выравнивается. По вертикали 2 метки в колонке разнесены к верхнему и нижнему короткому краю
листа (максимальная база вдоль стороны 300 мм) — это разнесение НЕ является привязкой к
физическому краю зеркала, просто раскладка на листе.

ВАЖНО (как и в `assets/aruco_test_sheet/`): печатать строго БЕЗ масштабирования ("Actual size" /
100%, НЕ "Fit to page") — иначе реальный размер метки разойдётся с corners_mm в
mirror_board.yaml. После печати проверить линейкой контрольную метку 50 мм.

`mirror_board.yaml` даёт координаты меток в ЛОКАЛЬНОЙ (плоскость листа, Z=0, до разрезания и
наклейки) системе — НЕ мировые координаты рига. Реальные мировые координаты (после того, как
колонки наклеены на зеркало, а зеркало установлено в риге) нужно обмерить отдельно — см. задачу
"Заменить тестовый лист ArUco на раскладку реального рига + откалибровать камеру".

Запуск: python assets/mirror_board/generate_sheet.py
Результат: mirror_board.pdf и mirror_board.yaml рядом с этим скриптом.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml

OUT_DIR = Path(__file__).resolve().parent

DICTIONARY_NAME = "DICT_4X4_50"
MARKER_SIDE_MM = 45.0  # как в assets/aruco_test_sheet/ — тот же типоразмер, уже проверен печатью
EDGE_MARGIN_MM = 20.0  # 2 см — отступ метки от ВНЕШНЕГО (не разрезаемого) края колонки, см. докстринг
ROW_MARGIN_MM = 20.0  # отступ верхней/нижней метки от короткого края листа (просто раскладка)
PAGE_W_MM, PAGE_H_MM = 210.0, 297.0  # A4
CUT_X_MM = PAGE_W_MM / 2.0  # линия разреза — вертикально посередине листа
MM_PER_INCH = 25.4

# Внешний (выравнивающий) край каждой колонки — противоположные стороны исходного листа.
_OUTER_EDGE_X = {"left": 0.0, "right": PAGE_W_MM}
# ID проверяются при загрузке (marker_layout.load_boards) на зеркальную неоднозначность — см.
# roboson_tools/pose/reflection_safety.py (для DICT_4X4_50 избегать id 8, 9, 17, 47: их
# отражение в зеркале рига неотличимо от прямого вида по геометрии).
MARKERS = {
    22: ("left", "top"),
    23: ("left", "bottom"),
    24: ("right", "top"),
    25: ("right", "bottom"),
}


def _marker_origin(column: str, row: str) -> tuple[float, float]:
    # x0 — от ВНЕШНЕГО края колонки, см. докстринг: для левой (внешний край x=0) метка начинается
    # сразу за отступом; для правой (внешний край x=210) метка кончается за отступом ДО края.
    if column == "left":
        x0 = EDGE_MARGIN_MM
    else:
        x0 = PAGE_W_MM - EDGE_MARGIN_MM - MARKER_SIDE_MM
    y0 = ROW_MARGIN_MM if row == "top" else PAGE_H_MM - ROW_MARGIN_MM - MARKER_SIDE_MM
    return x0, y0


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
    ax.set_ylim(PAGE_H_MM, 0.0)  # Y растёт вниз по листу, как в aruco_test_sheet/generate_sheet.py
    ax.set_axis_off()

    for marker_id, (column, row) in MARKERS.items():
        x0, y0 = _marker_origin(column, row)
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)
        ax.imshow(
            marker_img, cmap="gray", extent=(x0, x0 + MARKER_SIDE_MM, y0 + MARKER_SIDE_MM, y0)
        )
        ax.text(x0, y0 - 4, f"id={marker_id}", fontsize=8, family="monospace", va="bottom")

    # Линия разреза — вертикально посередине листа (штрихпунктир, чтобы не спутать с рамкой).
    ax.plot(
        [CUT_X_MM, CUT_X_MM], [5.0, PAGE_H_MM - 5.0],
        color="black", linewidth=0.8, linestyle=(0, (6, 4)),
    )
    ax.text(
        CUT_X_MM + 3.0, PAGE_H_MM / 2.0, "линия разреза\n(вдоль листа)",
        fontsize=7, rotation=90, va="center",
    )

    # Отметки отступа 2 см от ВНЕШНЕГО (выравнивающего) края каждой колонки — визуальная
    # проверка после печати, что метка стоит в 20 мм именно от него, а не от линии разреза.
    for column in ("left", "right"):
        outer_x = _OUTER_EDGE_X[column]
        direction = 1.0 if column == "left" else -1.0
        mark_x = outer_x + direction * EDGE_MARGIN_MM
        ax.plot(
            [mark_x, mark_x], [PAGE_H_MM / 2.0 - 10.0, PAGE_H_MM / 2.0 + 10.0],
            color="gray", linewidth=0.6,
        )
    ax.text(
        EDGE_MARGIN_MM, PAGE_H_MM / 2.0 + 14.0, "^ 2 см от\nвнешнего края",
        fontsize=6, ha="center", color="gray",
    )
    ax.text(
        PAGE_W_MM - EDGE_MARGIN_MM, PAGE_H_MM / 2.0 + 14.0, "2 см от\nвнешнего края ^",
        fontsize=6, ha="center", color="gray",
    )

    ruler_y = PAGE_H_MM / 2.0 - 40.0
    ax.plot([CUT_X_MM - 25.0, CUT_X_MM + 25.0], [ruler_y, ruler_y], color="black", linewidth=1.0)
    ax.plot([CUT_X_MM - 25.0, CUT_X_MM - 25.0], [ruler_y - 2, ruler_y + 2], color="black", linewidth=1.0)
    ax.plot([CUT_X_MM + 25.0, CUT_X_MM + 25.0], [ruler_y - 2, ruler_y + 2], color="black", linewidth=1.0)
    ax.text(CUT_X_MM - 25.0, ruler_y + 8, "50 мм — проверить линейкой после печати", fontsize=6, ha="left")

    ax.text(
        CUT_X_MM,
        PAGE_H_MM / 2.0 - 60.0,
        (
            f"roboson_tools — метки для краёв зеркала ({DICTIONARY_NAME})\n"
            f"Метки id={sorted(MARKERS)}, сторона {MARKER_SIDE_MM:g} мм\n"
            f"Выравнивание — от ВНЕШНЕГО края колонки (не от линии разреза), отступ {EDGE_MARGIN_MM:g} мм (2 см)\n"
            "Печать: 100% / Actual size, БЕЗ 'Fit to page'.\n"
            "Разрезать по вертикальной линии -> левая колонка на левый край зеркала,\n"
            "правая колонка на правый край (сторона 300 мм), внешним краем листа к краю зеркала."
        ),
        fontsize=7,
        ha="center",
    )

    fig.savefig(path, format="pdf")
    plt.close(fig)


def build_layout_yaml(path: Path) -> None:
    data = {
        "dictionary": DICTIONARY_NAME,
        "boards": [
            {
                "name": f"mirror_edge_{column}",
                "markers": [
                    {
                        "id": marker_id,
                        "corners_mm": marker_corners_mm(*_marker_origin(col, row), MARKER_SIDE_MM),
                    }
                    for marker_id, (col, row) in MARKERS.items()
                    if col == column
                ],
            }
            for column in ("left", "right")
        ],
    }
    path.write_text(
        "# Сгенерировано assets/mirror_board/generate_sheet.py — координаты в ЛОКАЛЬНОЙ\n"
        "# системе листа (Z=0, до разрезания/наклейки), НЕ мировые координаты рига. После\n"
        "# наклейки колонок на края зеркала и установки зеркала в риг — обмерить реальные\n"
        "# мировые координаты (см. задачу 'Заменить тестовый лист ArUco...').\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_pdf(OUT_DIR / "mirror_board.pdf")
    build_layout_yaml(OUT_DIR / "mirror_board.yaml")
    print("Готово:", OUT_DIR / "mirror_board.pdf", OUT_DIR / "mirror_board.yaml")
