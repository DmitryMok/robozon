"""Тесты sim/cv_top_tracking.py — top-локализация нескольких объектов +
трекинг по uid (часть 1 задачи "CV-триггер момента по top-локализации").
Запускать в venv с cv2 (Windows-копия или venv-cv-pytorch, см.
01 Sources/03 Conventions.md)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.cv_grid_v2 import build_camera_geom                          # noqa: E402
from sim.cv_top_tracking import TopTracker, localize_top_components   # noqa: E402


def _rig_cfg() -> dict:
    return {
        "x": 2.55,
        "fov_deg": 68.0,
        "resolution": [2592, 1944],
        "cameras": {
            "top": {"angle": 90, "distance": 2.0, "mirror": False},
        },
        "mirror_distance_by_angle": {},
        "mirror_normal_deg": 113.0,
    }


def _bs_cfg() -> dict:
    return {"h_threshold": 10.0, "s_threshold": 10.0, "min_area_px": 50.0}


def _synthetic_frame(shape: tuple[int, int],
                      rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Серый фон + ЦВЕТНЫЕ прямоугольники (row0, col0, h, w) — объекты на
    ленте. В отличие от аналогичного helper'а в test_cv_grid.py/
    test_cv_grid_v2.py (там объект отличается от фона только по V — ловится
    их bs-конфигом с `v_threshold`), здесь `_downscale_components`
    (используемая `localize_top_components`) сознательно не проверяет V
    (см. её docstring) — объект должен отличаться по H/S, иначе HSV
    нейтрального серого (S=0) от нейтрального серого фона неотличим."""
    frame = np.full((*shape, 3), 120, dtype=np.uint8)
    for row0, col0, h, w in rects:
        frame[row0:row0 + h, col0:col0 + w] = (60, 60, 180)   # BGR, насыщенный
    return frame


def test_localize_top_components_single_object_at_center_gives_cv_rig_x():
    """Объект точно в центре кадра top -> world_x == cv_rig.x (та же
    калибровочная точка, что test_estimate_x_from_mask_centered_object..."""
    rig = _rig_cfg()
    top = build_camera_geom("top", rig)
    shape = (2592, 1944)
    frame = _synthetic_frame(shape, [(shape[0] // 2 - 20, shape[1] // 2 - 20, 40, 40)])
    bg = _synthetic_frame(shape, [])
    components = localize_top_components(frame, bg, _bs_cfg(), top, cv_rig_x=rig["x"])
    assert len(components) == 1
    world_x, _belt_y = components[0]
    assert abs(world_x - rig["x"]) < 0.01


def test_localize_top_components_two_objects_gives_two_distinct_x():
    """Многообъектный кадр: два объекта на разных строках -> два разных
    world_x, упорядоченных по убыванию площади (крупный первым)."""
    rig = _rig_cfg()
    top = build_camera_geom("top", rig)
    shape = (2592, 1944)
    # Крупный объект левее по кадру (большая строка -> меньшая world_x),
    # мелкий — правее (меньшая строка -> большая world_x).
    frame = _synthetic_frame(shape, [
        (1800, 900, 100, 100),   # крупный, строка > центра -> X < cv_rig_x
        (400, 900, 30, 30),      # мелкий, строка < центра -> X > cv_rig_x
    ])
    bg = _synthetic_frame(shape, [])
    components = localize_top_components(frame, bg, _bs_cfg(), top, cv_rig_x=rig["x"])
    assert len(components) == 2
    (x_large, _), (x_small, _) = components   # отсортировано по убыванию площади
    assert x_large < rig["x"] < x_small


def test_localize_top_components_empty_frame_returns_empty_list():
    rig = _rig_cfg()
    top = build_camera_geom("top", rig)
    shape = (2592, 1944)
    frame = _synthetic_frame(shape, [])
    bg = _synthetic_frame(shape, [])
    assert localize_top_components(frame, bg, _bs_cfg(), top, cv_rig_x=rig["x"]) == []


def test_tracker_matches_known_uid_to_nearest_component_within_tolerance():
    tracker = TopTracker(max_match_distance_m=0.1)
    tracker._last_x[1] = 2.0
    matched = tracker.update([(2.03, 0.0), (3.5, 0.0)])   # второй компонент — далеко
    assert matched == {1: 2.03}
    assert tracker.last_x(1) == 2.03


def test_tracker_does_not_match_beyond_max_distance():
    tracker = TopTracker(max_match_distance_m=0.05)
    tracker._last_x[1] = 2.0
    matched = tracker.update([(2.2, 0.0)])   # 0.2м — за пределом допуска
    assert matched == {}
    assert tracker.last_x(1) == 2.0   # состояние не тронуто (нет наблюдения)


def test_tracker_assigns_pending_uid_to_leftover_component_fifo():
    """Новый uid (ещё не сопоставлялся) появляется в кадре -> берётся из
    очереди `_pending` (FIFO, в порядке добавления = порядке появления в
    зоне CV на однонаправленной ленте)."""
    tracker = TopTracker(max_match_distance_m=0.1)
    tracker.add_pending(1)
    tracker.add_pending(2)
    # uid=1 появился первым (спереди, дальше по ленте -> БОЛЬШАЯ X);
    # компоненты по убыванию X: первая — uid 1, вторая — uid 2.
    matched = tracker.update([(3.0, 0.0), (2.5, 0.0)])
    assert matched == {1: 3.0, 2: 2.5}
    assert tracker.last_x(1) == 3.0
    assert tracker.last_x(2) == 2.5


def test_tracker_known_uid_matched_before_pending_leftover_goes_to_pending():
    """Один уже отслеживаемый uid + одна лишняя компонента (новый объект,
    ещё не сопоставленный) — известный забирает ближайшую, лишняя достаётся
    pending."""
    tracker = TopTracker(max_match_distance_m=0.1)
    tracker._last_x[1] = 2.0
    tracker.add_pending(2)
    matched = tracker.update([(2.02, 0.0), (3.0, 0.0)])
    assert matched == {1: 2.02, 2: 3.0}


def test_tracker_forget_clears_last_x_and_pending():
    tracker = TopTracker(max_match_distance_m=0.1)
    tracker._last_x[1] = 2.0
    tracker.add_pending(2)
    tracker.forget(1)
    tracker.forget(2)
    assert tracker.last_x(1) is None
    assert tracker._pending == []
