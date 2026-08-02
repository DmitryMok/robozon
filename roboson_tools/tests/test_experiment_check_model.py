"""check_model_roll: потенциал к перекату по круглой ПРОЕКЦИИ формы, восстановленной
по силуэтам (см. docs/method.md).

Ключевое отличие от старого критерия "круглое сечение": круглое сечение где-то в объекте
(например, наклонное сечение конуса на скосе горлышка бутылки) переката не означает; и
наоборот — у вертикального цилиндра нет ни одного круглого сечения поперёк X, но он катится.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from roboson_tools.core.experiment import Orientation, check_model_roll, mesh_true_dims
from roboson_tools.geometry.mesh_io import from_trimesh, load_stl
from roboson_tools.geometry.transform import apply_orientation

ASSETS_STL_DIR = Path(__file__).resolve().parents[1] / "assets" / "stl"
VIEW_ANGLES = [0, 45, 90, 135]
# Visual Hull выпуклого тела вращения из 4 ракурсов — восьмиугольная призма:
# k проекции вдоль оси не превышает cos(22.5°)
OCTAGON_K = np.cos(np.radians(22.5))


def _cylinder():
    return from_trimesh(trimesh.creation.cylinder(radius=50.0, height=300.0))  # ось = Z


def _box():
    return from_trimesh(trimesh.creation.box(extents=(300.0, 100.0, 100.0)))


def _azim_dist_to_axis(azim_deg: float, target_deg: float) -> float:
    """Расстояние между осями по азимуту: ось без знака, период 180°."""
    d = abs((azim_deg - target_deg) % 180.0)
    return min(d, 180.0 - d)


def test_cylinder_along_x_rolls():
    result = check_model_roll(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=90, yaw_deg=0),  # ось цилиндра -> X
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
        axis_step_deg=10.0,
    )
    assert result.found_round
    assert result.best is not None and result.best.passed
    assert result.best.k > 0.9
    assert abs(result.best.axis_elev_deg) <= 10.0
    assert _azim_dist_to_axis(result.best.axis_azim_deg, 0.0) <= 10.0


def test_cylinder_diagonal_axis_found():
    """Регрессия "бутылка по диагонали": ось переката должна находиться вдоль оси объекта,
    а не через случайное круглое сечение."""
    result = check_model_roll(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=90, yaw_deg=30),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
        axis_step_deg=10.0,
    )
    assert result.found_round
    assert _azim_dist_to_axis(result.best.axis_azim_deg, 30.0) <= 10.0
    assert abs(result.best.axis_elev_deg) <= 10.0


def test_upright_cylinder_rolls_without_round_sections():
    """Вертикальный цилиндр: сечения поперёк X — прямоугольники (ни одного круглого),
    но круглая проекция вдоль вертикали есть — объект катится."""
    result = check_model_roll(
        _cylinder(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
        axis_step_deg=10.0,
    )
    assert result.round_section_count == 0
    assert result.found_round
    assert abs(result.best.axis_elev_deg - 90.0) <= 10.0


def test_box_does_not_roll():
    for yaw in (0.0, 45.0):
        result = check_model_roll(
            _box(),
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=yaw),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=30,
            axis_step_deg=10.0,
        )
        assert not result.found_round
        assert result.best is not None and result.best.k < 0.8


def test_box_roll_22_5_is_known_hull_artifact():
    """Коробка с roll=22.5° для 4 камер неотличима от восьмиугольной призмы — проекция
    вдоль X даёт k = cos(22.5°) > 0.8. Это честный артефакт метода (ограничение числа
    ракурсов), фиксируем его поведение."""
    result = check_model_roll(
        _box(),
        Orientation(roll_deg=22.5, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
        axis_step_deg=10.0,
    )
    assert result.found_round
    assert abs(result.best.k - OCTAGON_K) / OCTAGON_K < 0.02


def test_dims_from_silhouettes_match_mesh():
    result = check_model_roll(
        _box(),
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=30,
        axis_step_deg=30.0,
    )
    assert result.dims is not None
    for measured, true in zip(result.dims, (300.0, 100.0, 100.0)):
        assert abs(measured - true) / true < 0.02


def test_true_dims_independent_of_orientation():
    """Истинные (минимальные) габариты должны оставаться близки к реальным размерам объекта
    независимо от поворота — в отличие от `dims` (bbox в ТЕКУЩЕЙ ориентации), который заметно
    растёт при повороте на произвольный угол (не геометрический артефакт реконструкции, а
    неизбежное свойство осевого bbox повёрнutого объекта — см. заметку задачи о Camera Mode)."""
    true_sorted = (300.0, 100.0, 100.0)
    for yaw in (0.0, 22.5, 45.0):
        result = check_model_roll(
            _box(),
            Orientation(roll_deg=0, pitch_deg=0, yaw_deg=yaw),
            VIEW_ANGLES,
            resolution_px=512,
            num_slices=30,
            axis_step_deg=10.0,
        )
        assert result.true_dims is not None
        for measured, true in zip(result.true_dims, true_sorted):
            assert abs(measured - true) / true < 0.05


def test_mesh_true_dims_fixed_across_roll():
    """`mesh_true_dims` считается по вершинам самого меша (не реконструкции), поэтому должен
    оставаться близким к реальному размеру при ЛЮБОМ повороте входного меша — в частности, не
    "ползти" при вращении по Roll вокруг оси X (жалоба пользователя: "Габариты STL" в GUI
    менялись при повороте Roll, хотя это истинный размер объекта и должен быть фиксирован)."""
    true_sorted = (300.0, 100.0, 100.0)
    for roll in (0.0, 15.0, 47.0, 90.0):
        oriented = apply_orientation(_box(), roll_deg=roll, pitch_deg=0.0, yaw_deg=0.0)
        dims = mesh_true_dims(oriented, axis_step_deg=10.0)
        assert dims is not None
        for measured, true in zip(dims, true_sorted):
            assert abs(measured - true) / true < 0.05


def test_bottle_axis_not_neck_bevel():
    """Бутылка (реальный STL): вердикт — катится, ось вдоль оси бутылки (в её STL это ось Y,
    азимут 90°), и вердикт не зависит от круглых сечений (их поперёк X нет вовсе)."""
    mesh = load_stl(ASSETS_STL_DIR / "Бутылка.stl")
    result = check_model_roll(
        mesh,
        Orientation(roll_deg=0, pitch_deg=0, yaw_deg=0),
        VIEW_ANGLES,
        resolution_px=512,
        num_slices=40,
        axis_step_deg=10.0,
    )
    assert result.found_round
    assert result.best.k > 0.9
    assert _azim_dist_to_axis(result.best.axis_azim_deg, 90.0) <= 10.0
    assert abs(result.best.axis_elev_deg) <= 10.0
    assert result.round_section_count == 0
