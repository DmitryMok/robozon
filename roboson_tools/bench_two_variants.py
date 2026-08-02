"""Проверка двух вариантов рига «верх + вторая камера под углом + зеркало» (обсуждение задачи
[[Провести тесты с эмуляцией применения зеркала]]) на малой коробке и на Ручке.

Для каждого варианта — 4 реальных ракурса: прямой сверху (90°), прямой второй камеры (theta2),
и два ОТРАЖЁННЫХ (через зеркало) — на ИСТИННЫХ азимуте/дистанции виртуальной камеры (зеркальное
отражение реальной точки камеры через плоскость зеркала), а НЕ по приближённой формуле 2*phi-theta
(которая для конечного расстояния камеры даёт другой азимут — см. обсуждение задачи).

Usage: python bench_two_variants.py
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
RESOLUTION_PX = (2592, 1944)
SUPERSAMPLE = 3
BELT_POSITIONS = ["start", "center", "end"]
YAWS = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]

VARIANTS = {
    "A (компактный, H=2019мм)": dict(
        top_angle=90.0, top_dist=2018.9106305595487,
        side_angle=80.0, side_dist=2500.0,
        virt_top_az=104.83, virt_top_dist=2144.1,
        virt_side_az=108.92, virt_side_dist=2739.8,
    ),
    "B (разнесённый, H=3000мм)": dict(
        top_angle=90.0, top_dist=3000.0,
        side_angle=60.0, side_dist=3000.0,
        virt_top_az=102.94, virt_top_dist=3149.0,
        virt_side_az=127.24, virt_side_dist=3725.7,
    ),
}

OBJECTS = [
    ("Короб 300х200х200.stl", "Короб 300х200х200 (малая коробка)"),
    ("Ручка.stl", "Ручка"),
]

ASSETS = Path(__file__).parent / "assets" / "stl"


def max_abs_err_pct(dims, true_extents):
    d = sorted(dims, reverse=True)
    t = sorted(true_extents, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def main():
    for stl_name, label in OBJECTS:
        mesh = load_stl(ASSETS / stl_name)
        true_dims = mesh_true_dims(mesh, axis_step_deg=5.0)
        print(f"\n=== {label}: истинные габариты {true_dims} ===")
        for vname, v in VARIANTS.items():
            side_angles = [v["side_angle"], v["virt_top_az"], v["virt_side_az"]]
            camera_distances = {
                v["top_angle"]: v["top_dist"],
                v["side_angle"]: v["side_dist"],
                v["virt_top_az"]: v["virt_top_dist"],
                v["virt_side_az"]: v["virt_side_dist"],
            }
            errs = []
            for yaw in YAWS:
                ori = Orientation(roll_deg=0.0, pitch_deg=0.0, yaw_deg=yaw)
                r = check_simple_camera_dims_v3(
                    mesh, ori, FOV_DEG, v["top_dist"], RESOLUTION_PX,
                    top_angle_deg=v["top_angle"], side_angles_deg=side_angles,
                    belt_positions=BELT_POSITIONS, camera_distances=camera_distances,
                    supersample=SUPERSAMPLE,
                )
                if r is None:
                    errs.append(float("nan"))
                else:
                    errs.append(max_abs_err_pct([r.length, r.width, r.height], true_dims))
            valid = [e for e in errs if e == e]
            max_e = max(valid) if valid else float("nan")
            mean_e = sum(valid) / len(valid) if valid else float("nan")
            by_yaw = " ".join(f"{e:5.1f}" if e == e else "  NaN" for e in errs)
            print(f"  {vname:<28} max={max_e:5.1f}% mean={mean_e:5.1f}%  err%/yaw: {by_yaw}")


if __name__ == "__main__":
    main()
