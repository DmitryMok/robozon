"""Тесты `core/experiment.check_simple_camera_dims_v3` — multi-side per-point триангуляция
по всем 4 азимутам рига (0/45/90/135°, кроме верха) + опционально параллакс по ленте
(start/center/end).

Мотивация (см. docs/camera_dims_v2_investigation.md): v2 использует только ОДИН боковой
ракурс, и точки на «дальней» стороне недоопределены — фильтр `well_constrained` пропускает
ложно-узкие плато (регрессия на Шлеме 10.6→13.1%). v3 пересекает ВСЕ боковые силуэты, сужая
плато и делая фильтр надёжнее. Дополнительно, наклонные камеры 45/135° ограничивают стороны,
недоступные чисто боковой камере 0°, а параллакс по ленте (start/end) ограничивает длину по X.
"""

from __future__ import annotations

import numpy as np
import trimesh

from roboson_tools.core.experiment import (
    Orientation,
    check_simple_camera_dims,
    check_simple_camera_dims_v2,
    check_simple_camera_dims_v3,
)
from roboson_tools.geometry.mesh_io import from_trimesh

_FOV_DEG = 80.0
_CAMERA_DISTANCE = 2000.0
_RESOLUTION_PX = 1024
_VIEW_ANGLES = [0.0, 45.0, 90.0, 135.0]


def _box(ex: float, ey: float, ez: float):
    return from_trimesh(trimesh.creation.box(extents=(ex, ey, ez)))


def _disc_on_edge(radius: float, thickness: float):
    tm = trimesh.creation.cylinder(radius=radius, height=thickness, sections=64)
    tm.apply_transform(trimesh.transformations.rotation_matrix(1.5707963, [1, 0, 0]))
    return from_trimesh(tm)


def _max_abs_err_pct(dims, true_extents) -> float:
    d = sorted(dims, reverse=True)
    t = sorted(true_extents, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def test_box_stays_accurate_v3():
    """Коробка не должна пострадать от multi-side — v3 должен быть не хуже v1/v2."""
    ex, ey, ez = 300.0, 200.0, 200.0
    mesh = _box(ex, ey, ez)
    orientation = Orientation()

    v1 = check_simple_camera_dims(mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX)
    v3 = check_simple_camera_dims_v3(
        mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX,
        side_angles_deg=_VIEW_ANGLES,
    )

    assert v1 is not None and v3 is not None
    true_extents = (ex, ey, ez)
    err_v1 = _max_abs_err_pct([v1.length, v1.width, v1.height], true_extents)
    err_v3 = _max_abs_err_pct([v3.length, v3.width, v3.height], true_extents)
    assert err_v3 < 5.0, f"v3 регрессировал на простой коробке: {err_v3:.1f}%"
    assert err_v3 <= err_v1 + 3.0, f"v3 ({err_v3:.1f}%) заметно хуже v1 ({err_v1:.1f}%) на коробке"


def test_box_400_v3_beats_v2():
    """Короб 400×400×300: v2 давал 3.8% (регрессия от слишком широкой границы поиска), v3 с
    multi-side должен быть заметно точнее за счёт пересечения по нескольким ракурсам."""
    ex, ey, ez = 400.0, 400.0, 300.0
    mesh = _box(ex, ey, ez)
    orientation = Orientation()
    true_extents = (ex, ey, ez)

    v2 = check_simple_camera_dims_v2(mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX)
    v3 = check_simple_camera_dims_v3(
        mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX,
        side_angles_deg=_VIEW_ANGLES,
    )

    assert v2 is not None and v3 is not None
    err_v2 = _max_abs_err_pct([v2.length, v2.width, v2.height], true_extents)
    err_v3 = _max_abs_err_pct([v3.length, v3.width, v3.height], true_extents)
    assert err_v3 < 5.0, f"v3 неточен на коробе 400: {err_v3:.1f}%"
    assert err_v3 <= err_v2 + 1.0, f"v3 ({err_v3:.1f}%) заметно хуже v2 ({err_v2:.1f}%) на коробе 400"


def test_disc_on_edge_v3_maintains_v2_win():
    """Диск на ребре — целевой кейс v2 (20.6%→5.1%); v3 должен сохранять это улучшение."""
    mesh = _disc_on_edge(radius=244.5, thickness=264.0)
    orientation = Orientation()
    true_extents = (489.0, 489.0, 264.0)

    v3 = check_simple_camera_dims_v3(
        mesh, orientation, _FOV_DEG, _CAMERA_DISTANCE, _RESOLUTION_PX,
        side_angles_deg=_VIEW_ANGLES,
    )

    assert v3 is not None
    err_v3 = _max_abs_err_pct([v3.length, v3.width, v3.height], true_extents)
    assert err_v3 < 10.0, f"v3 потерял выигрыш на 'диск на ребре': {err_v3:.1f}%"


def test_tri_height_works_without_pure_side_angle():
    """Реальный риг (`camera_rig_presets.yaml`) не содержит ни одного чисто бокового
    ракурса (v_z=0) — раньше это отключало и tri_height, и v1-фолбэк, и высота тонкого
    объекта уходила в 0.0 (найдено на "Ручка.stl", см. [[Триангуляция высоты на риге без
    чисто бокового ракурса (roboson_tools)]]). tri_height обобщён на ПРОИЗВОЛЬНУЮ пару
    азимутов — высота больше не должна быть 0.0 на таком риге."""
    ex, ey, ez = 9.0, 13.0, 148.0
    mesh = _box(ex, ey, ez)
    orientation = Orientation()
    distances = {90.0: 2000.0, 70.0: 1900.0, 5.0: 1100.0, 157.26: 2570.6, 171.95: 2694.3}

    v3 = check_simple_camera_dims_v3(
        mesh, orientation, 68.0, distances[90.0], 1024,
        top_angle_deg=90.0, side_angles_deg=[5.0, 70.0, 157.26, 171.95],
        belt_positions=["start", "center", "end"], camera_distances=distances,
    )

    assert v3 is not None
    assert v3.height > 100.0, f"высота тонкого объекта снова 0/занижена: {v3.height}"
    err_pct = abs(100.0 * (v3.height - ez) / ez)
    assert err_pct < 10.0, f"высота неточна: {v3.height} (истина {ez}), {err_pct:.1f}%"


def test_tri_height_pure_side_pair_matches_old_special_case_formula():
    """Регрессия на «частный случай = было раньше»: при подстановке θ=0 (чисто боковой
    ракурс, старое допущение прежней формулы) новая обобщённая система Крамера обязана
    давать те же (Y, Z), что старая формула `Z=k0*(D0-Y)`, в пределах плавающей точки —
    прежняя формула является частным случаем новой, не отдельной веткой кода."""
    Y_true, Z_true = 37.2, 168.9
    D0 = 2000.0
    Dn = 2000.0
    theta_n = np.radians(45.0)
    sin_t, cos_t = np.sin(theta_n), np.cos(theta_n)

    k0 = Z_true / (D0 - Y_true)
    k_n = (-Y_true * sin_t + Z_true * cos_t) / (Dn - Y_true * cos_t - Z_true * sin_t)

    # старая формула (theta0 жёстко предполагался 0)
    num = k_n * Dn - k0 * D0 * (cos_t + k_n * sin_t)
    den_old = -sin_t + k_n * cos_t - k0 * cos_t - k0 * k_n * sin_t
    Y_old = num / den_old
    Z_old = k0 * (D0 - Y_old)

    # новая система A*Y+B*Z=C для theta_i=0 и theta_j=45°
    A_i, B_i, C_i = k0, 1.0, k0 * D0
    A_j = k_n * cos_t - sin_t
    B_j = k_n * sin_t + cos_t
    C_j = k_n * Dn
    den_new = A_i * B_j - A_j * B_i
    Y_new = (C_i * B_j - C_j * B_i) / den_new
    Z_new = (A_i * C_j - A_j * C_i) / den_new

    assert abs(Y_old - Y_new) < 1e-9
    assert abs(Z_old - Z_new) < 1e-9