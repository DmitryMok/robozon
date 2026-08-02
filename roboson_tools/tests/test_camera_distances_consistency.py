"""Регресс-тест на класс бага "добавили camera_distances, забыли реально прокинуть везде" —
именно это уже случилось один раз с прежним мёртвым полем `camera_distance_mm_by_angle`
(было в config/app_settings.yaml, но нигде не участвовало в 3D-реконструкции, см. заметку
задачи "Добавить настройку рига камер..."). Два независимых свойства:

1. Азимут, отсутствующий в `camera_distances`, должен явно падать (`KeyError`), а не тихо
   использовать какую-то заглушку — иначе рассинхрон дистанций и набора ракурсов остаётся
   незамеченным.
2. `camera_distances` реально влияет на результат: подмена корректных per-angle дистанций на
   одинаковую (неверную) для всех азимутов должна ЗАМЕТНО испортить реконструкцию — если бы
   где-то по пути camera_distances игнорировался и всегда использовалось одно и то же число,
   этот тест не отличил бы "правильный" вызов от "неправильного"."""

from __future__ import annotations

import pytest
import trimesh
from scipy.spatial import ConvexHull

from roboson_tools.geometry.mesh_io import from_trimesh
from roboson_tools.visual_hull import polytope_hull, voxel_carving

_FOV_DEG = 68.0
_CORRECT_DISTANCES = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}
_UNIFORM_WRONG_DISTANCES = {a: 2000.0 for a in _CORRECT_DISTANCES}
_VIEWS = [(a, "center") for a in _CORRECT_DISTANCES]


def _box():
    return from_trimesh(trimesh.creation.box(extents=(300.0, 200.0, 200.0)))


def test_voxel_carving_missing_angle_raises():
    box = _box()
    incomplete = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0}  # 135.0 отсутствует
    with pytest.raises(KeyError):
        voxel_carving.carve(box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=incomplete, resolution_px=128)


def test_polytope_hull_missing_angle_raises():
    box = _box()
    incomplete = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0}
    with pytest.raises(KeyError):
        polytope_hull.carve(box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=incomplete, resolution_px=256)


def test_wrong_uniform_distances_meaningfully_change_voxel_reconstruction():
    """Сравниваем ОБЪЁМ (число вокселей × их размер), а не bbox-габариты: внешние габариты
    насыщаются на границе сетки вокселей (margin_ratio) и не отличаются между корректным и
    неверным набором дистанций при этом grid_resolution — а вот сколько именно вокселей реально
    выжило (т.е. сам объём реконструкции) заметно отличается (см. докстринг модуля)."""
    box = _box()
    correct = voxel_carving.carve(
        box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=_CORRECT_DISTANCES,
        resolution_px=256, grid_resolution=32,
    )
    wrong = voxel_carving.carve(
        box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=_UNIFORM_WRONG_DISTANCES,
        resolution_px=256, grid_resolution=32,
    )
    assert correct is not None and wrong is not None
    vol_correct = len(correct.points) * correct.voxel_size**3
    vol_wrong = len(wrong.points) * wrong.voxel_size**3
    assert abs(vol_correct - vol_wrong) / vol_correct > 0.01


def test_wrong_uniform_distances_meaningfully_change_polytope_reconstruction():
    box = _box()
    correct = polytope_hull.carve(
        box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=_CORRECT_DISTANCES, resolution_px=1024
    )
    wrong = polytope_hull.carve(
        box, _VIEWS, fov_deg=_FOV_DEG, camera_distances=_UNIFORM_WRONG_DISTANCES, resolution_px=1024
    )
    assert correct is not None and wrong is not None
    vol_correct = ConvexHull(correct.vertices).volume
    vol_wrong = ConvexHull(wrong.vertices).volume
    assert abs(vol_correct - vol_wrong) / vol_correct > 0.003
