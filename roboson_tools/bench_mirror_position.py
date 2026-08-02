"""Точность габаритов (v3) в зависимости от положения зеркала (сдвиг t вдоль нормали,
см. docs/mirror_geometry_calc.md §3) на модели малой коробки (Короб 300х200х200).

Для каждого t берём пару расчётных оптических путей (45°/135°) из таблицы
mirror_geometry_calc.md §3 (путь камера->зеркало->объект) и подставляем как
camera_distances в check_simple_camera_dims_v3 при боевых параметрах камеры
(2592x1944, FOV 68 град, прямые ракурсы 2000мм, supersample=3). Плюс текущая
заглушка конфига (2600/2600 - единая дистанция) для сравнения.

Usage:
    python bench_mirror_position.py
"""
from __future__ import annotations

from pathlib import Path

from roboson_tools.core.experiment import (
    Orientation,
    check_simple_camera_dims_v3,
    mesh_true_dims,
)
from roboson_tools.geometry.mesh_io import load_stl

FOV_DEG = 68.0
CAMERA_DISTANCE = 2000.0
RESOLUTION_PX = (2592, 1944)
SIDE_ANGLES = [0.0, 45.0, 135.0]
TOP_ANGLE = 90.0
BELT_POSITIONS = ["start", "center", "end"]
SUPERSAMPLE = 3

YAWS = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]

# t, путь_45 (верх->зерк.), путь_135 (бок->зерк.) — расчётные пары из mirror_geometry_calc.md §3
# (единая дистанция 45/135=2600мм из старого конфига-заглушки НЕ используется: §2 того же
# документа доказывает, что путь через зеркало к 45° и к 135° физически не может быть одним
# числом ни при каком реальном положении зеркала — только обоснованные пары ниже).
MIRROR_POSITIONS = [
    ("t=-800", {45.0: 2028.0, 135.0: 3532.0}),
    ("t=-1000", {45.0: 2222.0, 135.0: 3923.0}),
    ("t=-1200", {45.0: 2467.0, 135.0: 4316.0}),
    ("t=-1500 (рекомендовано)", {45.0: 2900.0, 135.0: 4908.0}),
    ("t=-1800", {45.0: 3384.0, 135.0: 5501.0}),
    ("t=-2000", {45.0: 3725.0, 135.0: 5898.0}),
    ("t=-2500", {45.0: 4620.0, 135.0: 6890.0}),
]

STL_PATH = Path(__file__).parent / "assets" / "stl" / "Короб 300х200х200.stl"


def max_abs_err_pct(dims, true_extents):
    d = sorted(dims, reverse=True)
    t = sorted(true_extents, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def main():
    mesh = load_stl(STL_PATH)
    true_dims = mesh_true_dims(mesh, axis_step_deg=5.0)
    print(f"Короб 300х200х200, истинные габариты: {true_dims}\n")
    print(f"{'Положение зеркала':<32} {'max err%':>9} {'mean err%':>10}  err% по yaw")
    print("-" * 100)

    for label, distances in MIRROR_POSITIONS:
        errs = []
        for yaw in YAWS:
            ori = Orientation(roll_deg=0.0, pitch_deg=0.0, yaw_deg=yaw)
            r = check_simple_camera_dims_v3(
                mesh, ori, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
                top_angle_deg=TOP_ANGLE, side_angles_deg=SIDE_ANGLES,
                belt_positions=BELT_POSITIONS, camera_distances=distances,
                supersample=SUPERSAMPLE,
            )
            if r is None:
                errs.append(float("nan"))
            else:
                errs.append(max_abs_err_pct([r.length, r.width, r.height], true_dims))
        valid = [e for e in errs if e == e]  # drop NaN
        max_e = max(valid) if valid else float("nan")
        mean_e = sum(valid) / len(valid) if valid else float("nan")
        by_yaw = " ".join(f"{e:5.1f}" if e == e else "  NaN" for e in errs)
        print(f"{label:<32} {max_e:9.1f} {mean_e:10.1f}  {by_yaw}")


if __name__ == "__main__":
    main()
