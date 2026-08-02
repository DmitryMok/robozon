"""Тесты sim/cv_grid_v2.py — top-локализация + проекция (Фаза 3.2) и
разделение объектов в многообъектном кадре (Фаза 4).

Запускать в venv с cv2 (Windows-копия или `.venv-cv-pytorch`,
см. 01 Sources/03 Conventions.md)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.cv_grid import CELL_CAMERA_MOMENT, CELL_NAMES                   # noqa: E402
from sim.cv_grid_v2 import assemble_grid_v2                              # noqa: E402


def _rig_cfg() -> dict:
    return {
        "x": 2.55,
        "belt_z": 0.70,
        "fov_deg": 68.0,
        "resolution": [2592, 1944],
        "cameras": {
            "top": {"angle": 90, "distance": 2.0, "mirror": False},
            "side": {"angle": 5, "distance": 1.1, "mirror": False},
            "diag": {"angle": 70, "distance": 1.9, "mirror": False},
            "top_mirror": {"angle": 157.26, "mirror": True, "reflects": "top"},
            "diag_mirror": {"angle": 171.95, "mirror": True, "reflects": "diag"},
        },
        "mirror_distance_by_angle": {157.26: 2.5706, 171.95: 2.6943},
        "mirror_normal_deg": 113.0,
        "background_subtraction": {
            "h_threshold": 10.0, "s_threshold": 10.0, "min_area_px": 50.0,
            "morph_kernel": 1, "v_threshold": 30.0,
        },
        "moment_detection": {
            "margin_factor": 0.92,
            "worst_case_extent_x": 0.225,
            "worst_case_transverse_radius": 0.225,
            "mask_edge_tolerance_px": 6,
        },
        "grid": {"cell_px": 900, "crop_margin_factor": 0.15, "sam3_max_dim": 1008},
    }


def _synthetic_frame(shape: tuple[int, int],
                     rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Серый фон + яркие прямоугольники (row0, col0, h, w) — объект/объекты на
    ленте. Фон (пустая лента) = uniform-серый, объект — ярче по V, отлично
    отличается по HSV."""
    frame = np.full((*shape, 3), 120, dtype=np.uint8)
    for row0, col0, h, w in rects:
        frame[row0:row0 + h, col0:col0 + w] = (220, 220, 220)
    return frame


def test_assemble_grid_v2_multi_object_chooses_target_with_hint():
    """Многообъектный кадр (Фаза 4): в кадре ДВА объекта — мелкий ЦЕЛЕВОЙ в
    центре и КРУПНЫЙ ЧУЖОЙ сбоку. Без target_hint argmax по площади выбрал бы
    крупного соседа (как в замерах triple_pen_boxlarge_boxsmall — box_large
    вместо pen на ВСЕХ 9 ячейках). С target_hint (ожидаемый центр целевого
    из top-локализации) — выбирается целевой: центры кропов близки к hint, а
    не к центру крупного соседа.

    Синтетика: target — маленький прямоугольник в центре кадра каждой ячейки
    (ряд ~n_rows/2, столбец ~n_cols/2, как при моменте center), сосед —
    крупный прямоугольник, заметно смещённый по столбцу (имитация объекта на
    соседней полосе ленты). target_hint = (row_center, col_center) для каждой
    ячейки — ожидаемый центр целевого."""
    rig = _rig_cfg()
    shape = (2592, 1944)
    object_frames, prev_frames, background_frames = {}, {}, {}
    # Целевой — маленький (30×30) в центре кадра; сосед — крупный (200×200),
    # смещён на 600px по столбцу (вправо) — имитация соседа на соседней полосе.
    target_row0 = shape[0] // 2 - 15
    target_col0 = shape[1] // 2 - 15
    neighbor_row0 = shape[0] // 2 - 100
    neighbor_col0 = shape[1] // 2 + 300
    for cell_name in CELL_NAMES:
        object_frames[cell_name] = _synthetic_frame(
            shape, [(target_row0, target_col0, 30, 30),
                    (neighbor_row0, neighbor_col0, 200, 200)])
    for camera in {cam for cam, _m in CELL_CAMERA_MOMENT.values()}:
        background_frames[camera] = _synthetic_frame(shape, [])

    # target_hint: ожидаемый центр ЦЕЛЕВОГО в кадре каждой ячейки = центр кадра
    # (объект в cv_rig.x при моменте center; для старта/конца — тоже центр,
    # т.к. в синтетике объект не сдвинут по X между ячейками для простоты).
    target_hint = {name: (float(shape[0] / 2), float(shape[1] / 2))
                   for name in CELL_NAMES}

    result = assemble_grid_v2(object_frames, prev_frames, background_frames, rig,
                              cell_px=900, sam3_max_dim=1008,
                              margin_factor=0.15, target_hint=target_hint)

    # Все 9 ячеек собраны, единый scale/side (инвариант сетки).
    assert set(result.cells) == set(CELL_NAMES)
    assert len({c.scale for c in result.cells.values()}) == 1
    assert len({c.crop_side for c in result.cells.values()}) == 1

    # Ключевая проверка: центры кропов близки к ЦЕЛЕВОМУ (центр кадра), а не к
    # крупному соседу. Сосед на col ~shape[1]/2 + 400 (= 1372); целевой на
    # col ~972. crop_x (верхний левый угол окна) должен быть близок к
    # 972 - native_side/2, а не к 1372 - native_side/2. Допуск: native_side
    # (~30×1.3×4px на ds8 ≈ ~500px) — проверяем, что crop_x ближе к целевому,
    # чем к соседу на величину больше пол-объекта (15px native по центру).
    for name in CELL_NAMES:
        c = result.cells[name]
        # cx кропа (центр окна) = crop_x + crop_side/2. Должен быть близко к
        # центру кадра (972), а не к соседу (1372).
        cx_crop = c.crop_x + c.crop_side / 2.0
        cy_crop = c.crop_y + c.crop_side / 2.0
        # Центр окна по столбцу — ближе к целевому (972), отклонение < 200px
        # (сосед на +400px; без hint выбрало бы соседа → отклонение ~+400).
        assert abs(cx_crop - shape[1] / 2.0) < 250, (
            f"{name}: cx_crop={cx_crop:.0f} слишком далеко от целевого "
            f"({shape[1] / 2.0:.0f}) — выбран сосед?")
        # Центр окна по строке — близко к центру (целевой в центре по строке).
        assert abs(cy_crop - shape[0] / 2.0) < 250, (
            f"{name}: cy_crop={cy_crop:.0f} слишком далеко от целевого по строке")


def test_assemble_grid_v2_single_object_no_hint_unchanged():
    """Однообъектный кадр без target_hint — прежнее поведение Фазы 3.2:
    один объект в полосе, argmax площади выбирает его, кроп центрирован на
    нём. Регрессия: сигнал что сигнатура target_hint (опциональный) не сломала
    однообъектный путь."""
    rig = _rig_cfg()
    shape = (2592, 1944)
    object_frames, prev_frames, background_frames = {}, {}, {}
    for cell_name in CELL_NAMES:
        object_frames[cell_name] = _synthetic_frame(
            shape, [(shape[0] // 2 - 40, shape[1] // 2 - 40, 80, 80)])
    for camera in {cam for cam, _m in CELL_CAMERA_MOMENT.values()}:
        background_frames[camera] = _synthetic_frame(shape, [])

    result = assemble_grid_v2(object_frames, prev_frames, background_frames, rig,
                              cell_px=900, sam3_max_dim=1008,
                              margin_factor=0.15, target_hint=None)

    assert set(result.cells) == set(CELL_NAMES)
    assert len({c.scale for c in result.cells.values()}) == 1
    # Объект в центре кадра — кроп центрирован на нём.
    for name in CELL_NAMES:
        c = result.cells[name]
        cx_crop = c.crop_x + c.crop_side / 2.0
        cy_crop = c.crop_y + c.crop_side / 2.0
        assert abs(cx_crop - shape[1] / 2.0) < 300
        assert abs(cy_crop - shape[0] / 2.0) < 300