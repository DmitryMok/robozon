"""Обязательный тестовый сценарий (см. ТЗ): коробка w×w×L, roll=22.5°, view_angles=[0,45,90,135]
должна давать правильный восьмиугольник с Rin/Rout = cos(22.5°) = 0.9239."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from roboson_tools.geometry.mesh_io import from_trimesh, load_stl
from roboson_tools.geometry.transform import apply_orientation
from roboson_tools.metrics.roundness import compute_roundness
from roboson_tools.visual_hull.reconstruct import reconstruct_section

ASSETS_STL_DIR = Path(__file__).resolve().parents[1] / "assets" / "stl"
VIEW_ANGLES = [0, 45, 90, 135]
EXPECTED_K = np.cos(np.radians(22.5))


def test_box_octagon_synthetic():
    """Синтетическая коробка (trimesh.creation.box), длинная ось L — вдоль roll-оси X."""
    w, length = 100.0, 300.0
    mesh = from_trimesh(trimesh.creation.box(extents=(length, w, w)))
    oriented = apply_orientation(mesh, roll_deg=22.5, pitch_deg=0, yaw_deg=0)

    polygon = reconstruct_section(oriented, VIEW_ANGLES, axis_pos=0.0, resolution_px=2048)
    assert polygon is not None

    coords = list(polygon.exterior.coords)
    assert len(coords) - 1 == 8  # восьмиугольник (без учёта замыкающей вершины)

    result = compute_roundness(np.array(polygon.exterior.coords[:-1]), resolution_px=1024)
    assert abs(result.k - EXPECTED_K) / EXPECTED_K < 0.01
    assert result.passed


def test_box_octagon_real_stl_sanity():
    """Sanity-проверка на реальном Короб 400х400х300.stl — единственный из 11 объектов с
    околоквадратным сечением. У этого меша почти квадратная пара граней (~401x400 мм)
    перпендикулярна его собственной оси Y, а не X — поэтому перед roll-тестом мешу нужен
    предварительный поворот yaw=90°, чтобы эта ось совпала с roll-осью X. Допуск шире, чем
    у синтетического теста, из-за фасетной геометрии реального меша и отклонения 401 от 400."""
    mesh = load_stl(ASSETS_STL_DIR / "Короб 400х400х300.stl")
    aligned = apply_orientation(mesh, roll_deg=0, pitch_deg=0, yaw_deg=90)
    oriented = apply_orientation(aligned, roll_deg=22.5, pitch_deg=0, yaw_deg=0)

    polygon = reconstruct_section(oriented, VIEW_ANGLES, axis_pos=0.0, resolution_px=2048)
    assert polygon is not None

    result = compute_roundness(np.array(polygon.exterior.coords[:-1]), resolution_px=1024)
    assert abs(result.k - EXPECTED_K) / EXPECTED_K < 0.05
