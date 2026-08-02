"""Тесты `core/experiment.check_simple_camera_dims_v2` — per-point триангуляция контура сверху
вместо единой эталонной глубины `z_widest`/`near_edge` у `check_simple_camera_dims` (см. её
докстринг). Мотивация — обсуждение с пользователем: диск/пуфик, стоящий на закруглённом ободе
("на ребре", а не на плоской грани), даёт у v1 ~20% ошибку на той оси, где нет единой
представительной глубины (верхняя камера видит закруглённый обод, а не площадку).

Регрессия по пути (найдена и исправлена в этой же итерации, оба случая ниже её ловят):
1. Сетка перебора высоты `h`, растянутая на весь диапазон `h_max_ratio*camera_distance`
   (~1200мм), пропускала УЗКИЕ (единицы мм) окна согласованности у точек контура, отвечающих
   острым рёбрам/углам — точка тихо скатывалась в запасной h=0 (наибольшая наивная площадь),
   что ЗАВЫШАЛО габариты простой коробки сильнее, чем старый метод (регрессия). Исправлено
   сужением диапазона поиска до масштаба самого объекта (`h_max_margin`), не рига.
2. Точки на ДАЛЬНЕЙ от бокового ракурса стороне объекта недоопределены ОДНИМ боковым видом:
   согласованность остаётся истинной на широком плато высот далеко за пределами истинной
   границы (нехватка ракурсов, не решается более мелкой сеткой) — если брать максимум по ВСЕМ
   точкам, эти недоопределённые точки завышают высоту сильнее старого метода. Исправлено
   фильтром `well_constrained` (используются для высоты только точки с УЗКИМ окном
   согласованности), см. докстринг check_simple_camera_dims_v2.
"""

from __future__ import annotations

import trimesh

from roboson_tools.core.experiment import (
    Orientation,
    check_simple_camera_dims,
    check_simple_camera_dims_v2,
)
from roboson_tools.geometry.mesh_io import from_trimesh

_FOV_DEG = 80.0
_CAMERA_DISTANCE = 2000.0
_RESOLUTION_PX = 1024


def _box(ex: float, ey: float, ez: float):
    return from_trimesh(trimesh.creation.box(extents=(ex, ey, ez)))


def _disc_on_edge(radius: float, thickness: float):
    """Короткий толстый цилиндр, стоящий на закруглённом ободе (ось цилиндра вдоль Y, лежит на
    ленте своей кривой боковой поверхностью) — синтетический аналог "пуфика на ребре" из
    обсуждения с пользователем: у верхней камеры нет единой представительной глубины, потому что
    она видит закруглённый обод, а не плоскую площадку."""
    tm = trimesh.creation.cylinder(radius=radius, height=thickness, sections=64)
    tm.apply_transform(trimesh.transformations.rotation_matrix(1.5707963, [1, 0, 0]))
    return from_trimesh(tm)


def _max_abs_err_pct(dims, true_extents) -> float:
    d = sorted(dims, reverse=True)
    t = sorted(true_extents, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def test_box_stays_accurate_regression_guard():
    """Регрессия #1 из докстринга модуля: коробка не должна пострадать от перехода на
    per-point триангуляцию — v1 уже был точен (<2%), v2 не должен быть хуже него."""
    ex, ey, ez = 300.0, 200.0, 200.0
    mesh = _box(ex, ey, ez)
    orientation = Orientation()

    v1 = check_simple_camera_dims(mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX)
    v2 = check_simple_camera_dims_v2(mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX)

    assert v1 is not None and v2 is not None
    true_extents = (ex, ey, ez)
    err_v1 = _max_abs_err_pct([v1.length, v1.width, v1.height], true_extents)
    err_v2 = _max_abs_err_pct([v2.length, v2.width, v2.height], true_extents)
    assert err_v2 < 5.0, f"v2 регрессировал на простой коробке: {err_v2:.1f}%"
    assert err_v2 <= err_v1 + 2.0, f"v2 ({err_v2:.1f}%) заметно хуже v1 ({err_v1:.1f}%) на коробке"


def test_disc_on_edge_is_the_case_v2_was_built_for():
    """Основной сценарий из обсуждения — v1 систематически завышает габарит там, где у объекта
    нет единой представительной глубины (закруглённый обод); v2 должен быть заметно точнее."""
    mesh = _disc_on_edge(radius=244.5, thickness=264.0)
    orientation = Orientation()
    true_extents = (489.0, 489.0, 264.0)

    v1 = check_simple_camera_dims(mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX)
    v2 = check_simple_camera_dims_v2(mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX)

    assert v1 is not None and v2 is not None
    err_v1 = _max_abs_err_pct([v1.length, v1.width, v1.height], true_extents)
    err_v2 = _max_abs_err_pct([v2.length, v2.width, v2.height], true_extents)
    assert err_v1 > 10.0, f"синтетический кейс перестал воспроизводить исходную ошибку v1 ({err_v1:.1f}%)"
    assert err_v2 < 10.0, f"v2 не исправил кейс 'диск на ребре': {err_v2:.1f}%"
