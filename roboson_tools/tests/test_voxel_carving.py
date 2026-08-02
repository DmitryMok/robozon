"""Voxel carving (visual_hull/voxel_carving.py) — реконструкция по перспективным силуэтам
Camera Mode, несколько (азимут, положение на ленте) кадров. См. заметку задачи "Добавить
Camera Mode..." — обсуждение, почему start/end несут дополнительную информацию о форме, и
почему ОДНОГО азимута недостаточно для ограничения формы по всем трём осям (баг с "раздутой"
асимметричной реконструкцией, найденный пользователем по скриншоту)."""

from __future__ import annotations

import numpy as np
import trimesh

from roboson_tools.geometry.mesh_io import from_trimesh
from roboson_tools.visual_hull.voxel_carving import carve

FOV_DEG = 80.0
DISTANCE = 1000.0
ALL_ANGLES_DISTANCES = {a: DISTANCE for a in (0.0, 45.0, 90.0, 135.0)}


def _box():
    return from_trimesh(trimesh.creation.box(extents=(200.0, 100.0, 100.0)))


def test_single_azimuth_is_unbounded_along_view_axis():
    """Один азимут (даже с тремя belt_position) не ограничивает объект вдоль оси взгляда этой
    камеры (Y для view_angle=0) — контрпример на "раздутую" реконструкцию: диапазон по Y
    должен быть близок к габариту СЕТКИ вокселей (bbox+запас), а не к истинному габариту
    объекта (100 мм)."""
    mesh = _box()
    views = [(0.0, "start"), (0.0, "center"), (0.0, "end")]
    hull = carve(
        mesh, views, fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES,
        resolution_px=256, grid_resolution=24,
    )
    assert hull is not None
    y_extent = hull.points[:, 1].max() - hull.points[:, 1].min()
    # Истинный габарит по Y — 100 мм; при отсутствии реального ограничения по этой оси
    # облако растягивается почти на всю сетку вокселей (bbox+запас margin_ratio ~ 132 мм).
    assert y_extent > 125.0


def test_multiple_azimuths_bound_all_axes():
    """Несколько РАЗНЫХ азимутов (как в Analytical Mode) реально ограничивают форму по всем
    трём осям — в отличие от одного азимута (см. test_single_azimuth_is_unbounded_along_
    view_axis). Именно так исправлена "раздутая" асимметричная реконструкция."""
    mesh = _box()
    views = [(angle, "center") for angle in (0.0, 45.0, 90.0, 135.0)]
    hull = carve(
        mesh, views, fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES,
        resolution_px=512, grid_resolution=32,
    )
    assert hull is not None
    dims = hull.points.max(axis=0) - hull.points.min(axis=0)
    # Истина (200, 100, 100) — допускаем запас на грубость сетки вокселей/октагон-артефакт,
    # но никакая ось не должна быть радикально раздута (как в одноазимутальном случае).
    for measured, true in zip(dims, (200.0, 100.0, 100.0)):
        assert measured < true * 1.6


def test_adding_start_end_shrinks_or_keeps_volume():
    """Space carving: добавление кадров может только УБИРАТЬ воксели (пересечение), никогда
    не добавлять — облако точек center+start+end не может быть "больше" (по числу вокселей),
    чем только center, для того же набора азимутов."""
    mesh = _box()
    kwargs = dict(
        fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES, resolution_px=256, grid_resolution=24
    )
    angles = (0.0, 45.0, 90.0, 135.0)

    center_only = carve(mesh, [(a, "center") for a in angles], **kwargs)
    all_three = carve(
        mesh, [(a, pos) for a in angles for pos in ("start", "center", "end")], **kwargs
    )

    assert center_only is not None
    assert all_three is not None
    assert len(all_three.points) <= len(center_only.points)


def test_degenerate_case_returns_none():
    """Объект вне конуса обзора на всех кадрах -> None, а не падение/пустой мусор."""
    mesh = _box()
    hull = carve(
        mesh, [(0.0, "center")], fov_deg=1.0, camera_distances={0.0: 1.0},
        resolution_px=64, grid_resolution=16,
    )
    assert hull is None
