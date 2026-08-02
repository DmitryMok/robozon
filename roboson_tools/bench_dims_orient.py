"""Benchmark v3 с перебором ориентаций (roll/pitch/yaw) на всём наборе assets/stl.

Для каждого объекта перебирает ориентации и берёт ЛУЧШИЙ результат v3 (минимальная ошибка),
имитируя реальный сценарий: объект на ленте может быть повёрнут как угодно, метод должен
работать в любой ориентации. Сравнивает с v1/v2 вOrientation() для контекста.

Usage:
    python bench_dims_orient.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from roboson_tools.core.experiment import (
    Orientation,
    check_simple_camera_dims,
    check_simple_camera_dims_v2,
    check_simple_camera_dims_v3,
    mesh_true_dims,
)
from roboson_tools.geometry.mesh_io import load_stl

FOV_DEG = 80.0
CAMERA_DISTANCE = 2000.0
RESOLUTION_PX = 1024
VIEW_ANGLES = [0.0, 45.0, 90.0, 135.0]
TOP_ANGLE = 90.0
BELT_POSITIONS = ["start", "center", "end"]

# Перебор ориентаций: roll/pitch/yaw в градусах
ORIENTATIONS = [
    Orientation(roll_deg=r, pitch_deg=p, yaw_deg=y)
    for r in [0.0]
    for p in [0.0]
    for y in [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
] + [
    Orientation(roll_deg=r, pitch_deg=p, yaw_deg=0.0)
    for r in [0.0]
    for p in [0.0]
]

ASSETS_DIR = Path(__file__).parent / "assets" / "stl"

_SIZE_TO_NAME = {
    326184: "Bottle",
    29684: "Box300x200x200",
    26884: "Box400x400x300",
    578784: "LunchBox",
    5675429: "Bag",
    3637684: "Detergent",
    644084: "Pouf",
    2046384: "Pen",
    125284: "Plate",
    107684: "Cylinder",
    2758034: "Helmet",
}


def max_abs_err_pct(dims, true_extents):
    d = sorted(dims, reverse=True)
    t = sorted(true_extents, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def main():
    seen = set()
    stl_files = []
    for p in sorted(ASSETS_DIR.iterdir()):
        if p.suffix.lower() == ".stl" and p.stem.lower() not in seen:
            seen.add(p.stem.lower())
            stl_files.append(p)
    if not stl_files:
        print(f"No STL files found in {ASSETS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"{'Object':<22} {'mesh_true':>20} {'v1@0':>7} {'v2@0':>7} "
          f"{'v3 best':>8} {'v3 worst':>9} {'v3@0':>6} {'best yaw':>9}")
    print("-" * 95)

    all_v1 = []
    all_v2 = []
    all_v3_best = []
    all_v3_worst = []

    for stl_path in stl_files:
        label = _SIZE_TO_NAME.get(stl_path.stat().st_size, stl_path.stem)
        mesh = load_stl(stl_path)

        true_dims = mesh_true_dims(mesh, axis_step_deg=5.0)
        if true_dims is None:
            print(f"{label:<22} FAILED mesh_true_dims")
            continue

        # v1/v2 в Orientation() (yaw=0)
        ori0 = Orientation()
        v1 = check_simple_camera_dims(
            mesh, ori0, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
            top_angle_deg=TOP_ANGLE, side_angle_deg=0.0,
        )
        v2 = check_simple_camera_dims_v2(
            mesh, ori0, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
            top_angle_deg=TOP_ANGLE, side_angle_deg=0.0,
        )
        v1_err = max_abs_err_pct([v1.length, v1.width, v1.height], true_dims) if v1 else float("nan")
        v2_err = max_abs_err_pct([v2.length, v2.width, v2.height], true_dims) if v2 else float("nan")

        # v3 по всем ориентациям
        v3_errs = []
        v3_best_yaw = 0.0
        for ori in ORIENTATIONS:
            v3 = check_simple_camera_dims_v3(
                mesh, ori, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
                top_angle_deg=TOP_ANGLE, side_angles_deg=VIEW_ANGLES,
                belt_positions=BELT_POSITIONS,
            )
            if v3 is not None:
                err = max_abs_err_pct([v3.length, v3.width, v3.height], true_dims)
                v3_errs.append((err, ori.yaw_deg))
            else:
                v3_errs.append((float("inf"), ori.yaw_deg))

        v3_best = min(e for e, _ in v3_errs)
        v3_worst = max(e for e, _ in v3_errs if e != float("inf")) if any(e != float("inf") for e, _ in v3_errs) else float("nan")
        v3_best_yaw = [y for e, y in v3_errs if e == v3_best][0]
        v3_at_0 = [e for e, y in v3_errs if y == 0.0][0]

        true_str = f"({true_dims[0]:.0f},{true_dims[1]:.0f},{true_dims[2]:.0f})"
        print(f"{label:<22} {true_str:>20} {v1_err:>7.1f} {v2_err:>7.1f} "
              f"{v3_best:>8.1f} {v3_worst:>9.1f} {v3_at_0:>6.1f} {v3_best_yaw:>8.1f}")

        all_v1.append(v1_err)
        all_v2.append(v2_err)
        all_v3_best.append(v3_best)
        all_v3_worst.append(v3_worst)

    print("-" * 95)
    print(f"{'MAX':<22} {'':>20} {max(all_v1):>7.1f} {max(all_v2):>7.1f} "
          f"{max(all_v3_best):>8.1f} {max(all_v3_worst):>9.1f}")
    print(f"{'MEAN':<22} {'':>20} {np.mean(all_v1):>7.1f} {np.mean(all_v2):>7.1f} "
          f"{np.mean(all_v3_best):>8.1f} {np.mean(all_v3_worst):>9.1f}")


if __name__ == "__main__":
    main()