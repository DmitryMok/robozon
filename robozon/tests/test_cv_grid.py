"""Тесты сборки сетки 9 ракурсов (Фаза 3) — sim/cv_grid.py. В отличие от
tests/test_cv_moments.py, этот модуль тянет cv2 (crop/resize/compute_mask) —
запускать в venv с cv2 (Windows-копия или `.venv-cv`, см.
`01 Sources/03 Conventions.md`), не в "чистом" venv Фазы 2.2."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.cv_grid import (                                            # noqa: E402
    CELL_CAMERA_MOMENT, CELL_NAMES, assemble_grid, bbox_center_and_size,
    crop_square_padded, expected_pixel_row, fallback_center_and_size,
)
from sim.cv_localization import (                                    # noqa: E402
    build_background_model, compute_mask, select_target_contour,
)
from sim.cv_moments import estimate_x_from_mask                       # noqa: E402


def test_expected_pixel_row_is_inverse_of_estimate_x_from_mask():
    """Круговой тест: row -> estimate_x_from_mask -> expected_pixel_row
    должен вернуть исходную row (та же геометрия, оба знака согласованы)."""
    n_rows = 2592
    n_cols = 1944
    row = 700.0
    mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
    mask[int(row) - 10:int(row) + 10, 900:1044] = 255
    est_x = estimate_x_from_mask(mask, camera_distance_m=2.0, fov_deg=68.0, cv_rig_x=2.55)
    assert est_x is not None
    back_row = expected_pixel_row(est_x, cv_rig_x=2.55, camera_distance_m_=2.0,
                                   fov_deg=68.0, n_rows=n_rows, n_cols=n_cols)
    assert abs(back_row - row) < 1.0


def test_bbox_center_and_size_adds_margin_to_larger_dimension():
    bbox = (10, 5, 40, 60)   # x, y, w, h — h больше w
    cx, cy, natural = bbox_center_and_size(bbox, margin_factor=0.5)
    assert cx == 10 + 40 / 2.0 and cy == 5 + 60 / 2.0
    assert natural == 60 * (1.0 + 2 * 0.5)   # запас считается от БОЛЬШЕЙ стороны (h)


def test_fallback_center_and_size_scales_with_distance():
    """Дальше камера (больше distance) -> тот же реальный габарит занимает
    МЕНЬШЕ пикселей -> меньше и natural_size."""
    frame_shape = (2592, 1944)
    _cx, _cy, near = fallback_center_and_size(
        1296.0, 972.0, camera_distance_m_=1.0, fov_deg=68.0, frame_shape=frame_shape,
        worst_case_extent_x=0.225, worst_case_transverse_radius=0.225)
    _cx, _cy, far = fallback_center_and_size(
        1296.0, 972.0, camera_distance_m_=2.0, fov_deg=68.0, frame_shape=frame_shape,
        worst_case_extent_x=0.225, worst_case_transverse_radius=0.225)
    assert far < near


def test_crop_square_padded_same_size_regardless_of_frame_edge():
    """Окно у самого края кадра (и даже частично за ним) всё равно даёт
    side x side — недостающее паддится, а не обрезается неравномерно (иначе
    кропы старта/конца отличались бы по размеру от кропов центра)."""
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    center_crop = crop_square_padded(frame, cx=40, cy=50, side=30)
    edge_crop = crop_square_padded(frame, cx=5, cy=5, side=30)   # центр у самого угла
    assert center_crop.shape == (30, 30, 3)
    assert edge_crop.shape == (30, 30, 3)   # тот же размер, несмотря на выход за кадр
    # Часть, ушедшая за кадр, западдена (не осталась нулевой/произвольной) — проверяем
    # что хотя бы часть edge_crop равна паддингу 128, а не кадра.
    assert (edge_crop == 128).any()


def _synthetic_frame(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Серый фон + один или несколько ярких прямоугольников (row0, col0, h, w)
    — заменяет реальный рендер камеры для юнит-тестов compute_mask/select_target_contour."""
    frame = np.full((*shape, 3), 120, dtype=np.uint8)
    for row0, col0, h, w in rects:
        frame[row0:row0 + h, col0:col0 + w] = (220, 220, 220)
    return frame


def test_select_target_contour_prefers_nearest_row_over_largest_area():
    """Два объекта в одном кадре (реалистичный сценарий при частом спавне,
    см. заметку задачи Фазы 2.2 "Не тронуто") — compute_mask сам по себе взял
    бы БОЛЬШИЙ (чужой) контур; select_target_contour должен выбрать тот, что
    ближе к ожидаемой строке нашего объекта."""
    shape = (200, 160)
    background = build_background_model(_synthetic_frame(shape, []))
    # Маленький объект (наш, expected_row=50) + больший чужой (expected_row далеко, row~150).
    frame = _synthetic_frame(shape, [(40, 60, 20, 20), (130, 60, 40, 40)])
    _mask, _bbox, contours = compute_mask(
        frame, background, h_threshold=10, s_threshold=10, min_area_px=50,
        morph_kernel=1, v_threshold=30, return_all_contours=True)
    assert len(contours) == 2
    bbox = select_target_contour(contours, expected_row=50.0)
    assert bbox is not None
    x, y, w, h = bbox
    assert abs((y + h / 2.0) - 50.0) < 5.0   # выбран маленький (наш), не больший


def _rig_cfg() -> dict:
    return {
        "x": 2.55,
        "fov_deg": 68.0,
        "cameras": {
            "top": {"angle": 90, "distance": 2.0, "mirror": False},
            "side": {"angle": 5, "distance": 1.1, "mirror": False},
            "diag": {"angle": 70, "distance": 1.9, "mirror": False},
            "top_mirror": {"angle": 157.26, "mirror": True, "reflects": "top"},
            "diag_mirror": {"angle": 171.95, "mirror": True, "reflects": "diag"},
        },
        "mirror_distance_by_angle": {157.26: 2.5, 171.95: 2.6},
        "background_subtraction": {
            "h_threshold": 10.0, "s_threshold": 10.0, "min_area_px": 50.0,
            "morph_kernel": 1, "v_threshold": 30.0,
        },
        "moment_detection": {
            "worst_case_extent_x": 0.225, "worst_case_transverse_radius": 0.225,
        },
    }


def test_assemble_grid_produces_uniform_square_cells_with_one_shared_scale():
    """Решение пользователя (уточнено по итогам визуальной проверки первого
    прогона): у КАЖДОЙ ячейки свой ЦЕНТР кропа, но РАЗМЕР окна (native,
    квадратный) — ОДИН на всю сетку, подобранный по самому крупному объекту.
    Все итоговые object_crop — РОВНО cell_px x cell_px (не просто "не больше")
    — иначе следующим стадиям (SAM3-батч, torchhull) пришлось бы иметь дело
    с разными размерами тайлов."""
    shape = (300, 260)
    rig = _rig_cfg()
    object_frames, prev_frames, background_frames, positions_x = {}, {}, {}, {}
    for i, cell_name in enumerate(CELL_NAMES):
        camera, _moment = CELL_CAMERA_MOMENT[cell_name]
        # Разный размер "объекта" по ячейкам — крупнейший (60x60) в одной,
        # мельче в остальных, чтобы проверить выбор общего масштаба/окна.
        size = 60 if i == 0 else 20 + i
        row0, col0 = 150 - size // 2, 130 - size // 2
        object_frames[cell_name] = _synthetic_frame(shape, [(row0, col0, size, size)])
        background_frames.setdefault(camera, _synthetic_frame(shape, []))
        positions_x[cell_name] = rig["x"]   # объект в точке наблюдения на каждом снимке

    cell_px = 100
    result = assemble_grid(object_frames, prev_frames, background_frames, positions_x,
                            rig, cell_px=cell_px, sam3_max_dim=64, margin_factor=0.1)

    assert set(result.cells) == set(CELL_NAMES)
    scales = {c.scale for c in result.cells.values()}
    assert len(scales) == 1   # один и тот же scale на ВСЮ сетку
    sides = {c.crop_side for c in result.cells.values()}
    assert len(sides) == 1    # один и тот же РАЗМЕР native-окна на ВСЮ сетку
    for cell in result.cells.values():
        assert cell.object_crop.shape == (cell_px, cell_px, 3)   # РОВНО, не "не больше"
        assert cell.background_crop is not None                 # фон задан для всех камер в тесте
        assert cell.background_crop.shape == (cell_px, cell_px, 3)
    # Самая крупная (60x60 + margin=0.1) ячейка задаёт масштаб сетки. Точное
    # значение `native_side` зависит от алгоритма локализации: full-res
    # compute_mask давал ~72 (bbox 60 + margin), ds8-локализация (Фаза 3.1)
    # даёт ~91 — INTER_AREA расширяет границу объекта на ~1px в ds8 (×8 в
    # native) перед HSV-порогом. Проверяем ИНВАРИАНТ (объект+запас помещается,
    # масштаб единый), а не конкретное число: native_side покрывает объект с
    # margin и не превышает его больше чем на bleed ds8 (2px × factor × margin).
    native_side = next(iter(sides))
    assert native_side >= 60 * 1.2
    assert native_side <= 60 * 1.2 + 2 * 8 * 1.2 + 4   # ~2px bleed на ds8 + допуск округления
