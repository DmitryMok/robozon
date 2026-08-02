"""Тесты детектора моментов старт/центр/конец (Фаза 2.2, без Webots) —
sim/cv_moments.py. Запуск: python3 -m pytest tests/ (WSL .venv-cv или Windows
venv-cv-pytorch — модуль не тянет cv2, чистый numpy/math)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import load_layout                                   # noqa: E402
from sim.cv_moments import (                                          # noqa: E402
    MomentScheduler, estimate_x_from_mask, mask_row_bbox,
    mask_touches_row_edge, moment_thresholds, threshold_captures,
)


def test_moment_thresholds_match_documented_geometry():
    """Числа сверены с 03 Work/Расчёт перекладки CV-зоны и ramp.md §4/§6 —
    cv_rig.x=2.55, delta_max_top=0.876, delta_max_side=0.316."""
    rig = load_layout()["cv_rig"]
    t = moment_thresholds(rig)
    assert set(t) == {"start_top", "start_side", "center", "end_side", "end_top"}
    assert t["center"] == rig["x"]
    assert abs(t["start_top"] - (rig["x"] - 0.876)) < 0.01
    assert abs(t["start_side"] - (rig["x"] - 0.316)) < 0.01
    assert abs(t["end_side"] - (rig["x"] + 0.316)) < 0.01
    assert abs(t["end_top"] - (rig["x"] + 0.876)) < 0.01
    # Пороги строго по возрастанию X (объект их проходит по порядку,
    # top — самая широкая из старт/конец камер, см. заметку задачи).
    ordered = [t["start_top"], t["start_side"], t["center"], t["end_side"], t["end_top"]]
    assert ordered == sorted(ordered)


def test_threshold_captures_cover_nine_frames_per_object():
    """Итоговый выход Фазы 3 — 9 кадров/объект: top×3, side×3, diag×1,
    top_mirror×1, diag_mirror×1 (5 в центре + 2 старт + 2 конец), см. заметку
    задачи "Сборка сетки 9 ракурсов..." — зеркало отражает top+diag, не
    top+side (side зеркального отражения не имеет)."""
    captures = threshold_captures()
    all_pairs = [pair for pairs in captures.values() for pair in pairs]
    assert len(all_pairs) == 9
    assert sorted(all_pairs) == sorted([
        ("top", "start"), ("top", "center"), ("top", "end"),
        ("side", "start"), ("side", "center"), ("side", "end"),
        ("diag", "center"), ("top_mirror", "center"), ("diag_mirror", "center"),
    ])


def test_scheduler_fires_each_threshold_exactly_once_no_double_no_miss():
    """Проверка отсутствия двойных/пропущенных срабатываний (Приёмка, п.4) —
    объект едет монотонно по X одним и тем же шагом, каждый из 5 порогов
    должен сработать РОВНО один раз, дав ровно 7 (camera, moment) событий."""
    thresholds = {"start_top": 1.0, "start_side": 1.5, "center": 2.0,
                  "end_side": 2.5, "end_top": 3.0}
    scheduler = MomentScheduler(thresholds, threshold_captures())
    uid = 1
    events = []
    x = 0.5
    while x <= 3.5:
        events += scheduler.step({uid: round(x, 3)})
        x += 0.01   # мельче любого порога — не пропускаем пересечение
    assert len(events) == 9
    assert len(set(events)) == 9   # без повторов
    assert all(camera in ("top", "side", "diag", "top_mirror", "diag_mirror")
               for _, camera, _ in events)


def test_scheduler_does_not_fire_twice_across_calls_after_threshold_passed():
    thresholds = {"center": 2.0}
    scheduler = MomentScheduler(thresholds, {"center": [("top", "center")]})
    uid = 7
    scheduler.step({uid: 1.9})
    first = scheduler.step({uid: 2.1})
    second = scheduler.step({uid: 2.2})
    assert first == [(uid, "top", "center")]
    assert second == []


def test_scheduler_tracks_multiple_objects_independently():
    thresholds = {"center": 2.0}
    scheduler = MomentScheduler(thresholds, {"center": [("top", "center")]})
    scheduler.step({1: 1.9, 2: 2.5})   # объект 2 уже за порогом при первом наблюдении
    events = scheduler.step({1: 2.1, 2: 2.6})
    assert events == [(1, "top", "center")]   # объект 2 не должен ложно сработать


def test_mask_row_bbox_and_edge_touch():
    mask = np.zeros((100, 80), dtype=np.uint8)
    mask[40:60, 30:50] = 255   # объект в центре кадра — не касается краёв
    assert mask_row_bbox(mask) == (40, 59)
    assert not mask_touches_row_edge(mask, tolerance_px=6)

    mask_edge = np.zeros((100, 80), dtype=np.uint8)
    mask_edge[0:20, 30:50] = 255   # объект обрезан у верхней границы (старт/конец)
    assert mask_touches_row_edge(mask_edge, tolerance_px=6)


def test_mask_row_bbox_empty_mask_returns_none():
    mask = np.zeros((100, 80), dtype=np.uint8)
    assert mask_row_bbox(mask) is None
    assert not mask_touches_row_edge(mask, tolerance_px=6)
    assert estimate_x_from_mask(mask, camera_distance_m=2.0, fov_deg=68.0, cv_rig_x=2.55) is None


def test_estimate_x_from_mask_centered_object_equals_cv_rig_x():
    """Объект точно по центру кадра (по строкам) -> оценка X == cv_rig.x —
    это калибровочный случай момента "центр" (объект в точке наблюдения)."""
    mask = np.zeros((2592, 1944), dtype=np.uint8)
    mask[1280:1312, 900:1044] = 255   # bbox по строкам центрирован на 1296 (≈n_rows/2)
    est = estimate_x_from_mask(mask, camera_distance_m=2.0, fov_deg=68.0, cv_rig_x=2.55)
    assert est is not None
    assert abs(est - 2.55) < 0.01


def test_estimate_x_from_mask_offset_object_moves_estimate_away_from_center():
    n_rows = 2592
    mask_top_half = np.zeros((n_rows, 1944), dtype=np.uint8)
    mask_top_half[0:40, 900:1044] = 255   # bbox прижат к краю строк (не к центру)
    est = estimate_x_from_mask(mask_top_half, camera_distance_m=2.0, fov_deg=68.0, cv_rig_x=2.55)
    assert est is not None
    assert abs(est - 2.55) > 0.3   # заметно смещена от центра кадра/точки наблюдения
