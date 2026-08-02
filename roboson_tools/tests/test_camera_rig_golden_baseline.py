"""Golden-baseline инварианты для перехода camera_distance (скаляр на весь риг) ->
camera_distances (per-angle карта, см. заметку задачи "Добавить настройку рига камер...").

В проекте нет snapshot-файлов (весь стиль тестов — аналитические ассерты с допуском, см.
test_polytope_hull.py и др.) — вместо отдельного snapshot-механизма числа зафиксированы прямо
здесь как константы (посчитаны один раз на дефолтном пресете рига камер сразу после миграции
на camera_distances). Долг этого теста — поймать РЕГРЕСС: если последующий рефакторинг
случайно поменяет числа (например, перепутает, какая дистанция для какого азимута берётся),
эти assert'ы упадут. Он НЕ проверяет точность метода относительно истинных габаритов объекта
(это уже проверяют test_simple_camera_dims_v3.py/test_voxel_carving.py/test_polytope_hull.py)."""

from __future__ import annotations

import numpy as np
import trimesh

from roboson_tools.core.camera_rig import load_camera_rig_presets, rig_to_distances
from roboson_tools.core.experiment import Orientation, check_simple_camera_dims_v3
from roboson_tools.geometry.mesh_io import from_trimesh
from roboson_tools.visual_hull import polytope_hull, voxel_carving

_FOV_DEG = 68.0
_PRESET = load_camera_rig_presets()[0]  # "2 камеры + зеркало (дефолт)" — 0/45/90/135
_DISTANCES = rig_to_distances(_PRESET.cameras)
_ANGLES = list(_DISTANCES.keys())
_SIDE_ANGLES = [a for a in _ANGLES if a != 90.0]
_VIEWS = [(a, "center") for a in _ANGLES]


def _box():
    return from_trimesh(trimesh.creation.box(extents=(300.0, 200.0, 200.0)))


def test_default_rig_distances_unchanged():
    """Фиксирует сами значения дефолтного пресета — если кто-то случайно поменяет
    config/camera_rig_presets.yaml (например, при следующем пересчёте зеркала), это должно
    быть осознанным решением, а не тихой правкой файла."""
    assert _DISTANCES == {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}


def test_v3_dims_on_box_matches_golden_baseline():
    box = _box()
    result = check_simple_camera_dims_v3(
        box, Orientation(), _FOV_DEG, _DISTANCES[0.0], 1024,
        top_angle_deg=90.0, side_angles_deg=_SIDE_ANGLES, belt_positions=["center"],
        camera_distances=_DISTANCES,
    )
    assert result is not None
    assert np.allclose(
        [result.length, result.width, result.height],
        [197.7908935546875, 298.34466552734375, 200.28223232782787],
        rtol=1e-3,
    )


def test_voxel_carving_dims_on_box_matches_golden_baseline():
    box = _box()
    hull = voxel_carving.carve(
        box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=_DISTANCES,
        resolution_px=256, grid_resolution=32,
    )
    assert hull is not None
    dims = hull.points.max(axis=0) - hull.points.min(axis=0)
    assert np.allclose(dims, [325.5483871, 216.0, 216.0], rtol=1e-3)


def test_polytope_hull_dims_on_box_matches_golden_baseline():
    box = _box()
    hull = polytope_hull.carve(box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=_DISTANCES, resolution_px=1024)
    assert hull is not None
    dims = hull.vertices.max(axis=0) - hull.vertices.min(axis=0)
    # Baseline восстановлен 2026-07-18 после возврата `contour_epsilon_px=0.0` (дефолт
    # упрощения отключено) в `polytope_hull.carve` — обратная совместимость со всеми
    # вызывающими (`pose_carving` и др.). Значение совпадает с исходным baseline до Плана A.
    # При `contour_epsilon_px=1.5` (План A, этап 1) baseline был [304.34, 220.23, 220.28] —
    # см. bench_g4_endtoend.py (там передаётся явно 1.5).
    assert np.allclose(dims, [315.30426479, 220.22913134, 220.28489341], rtol=1e-3)
