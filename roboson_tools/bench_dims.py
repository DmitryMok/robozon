"""Benchmark габаритных методов v1/v2/v3 на всём наборе assets/stl.

Сравнивает check_simple_camera_dims (v1), check_simple_camera_dims_v2 (v2) и
check_simple_camera_dims_v3 (v3, multi-side триангуляция) с mesh_true_dims
(эталон — минимальный бокс по вершинам меша) на всём тестовом наборе.

Usage:
    python bench_dims.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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
RESOLUTION_PX = 4096
VIEW_ANGLES = [0.0, 45.0, 90.0, 135.0]
TOP_ANGLE = 90.0
BELT_POSITIONS = ["start", "center", "end"]

ASSETS_DIR = Path(__file__).parent / "assets" / "stl"

# ASCII labels for Cyrillic STL names (by file size — stable across encodings)
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

    print(f"{'Object':<22} {'mesh_true':>20} {'v1 err%':>8} {'v2 err%':>8} {'v3 err%':>8}")
    print("-" * 70)

    all_v1 = []
    all_v2 = []
    all_v3 = []

    for stl_path in stl_files:
        label = _SIZE_TO_NAME.get(stl_path.stat().st_size, stl_path.stem)
        mesh = load_stl(stl_path)
        orientation = Orientation()

        true_dims = mesh_true_dims(mesh, axis_step_deg=5.0)
        if true_dims is None:
            print(f"{label:<22} FAILED mesh_true_dims")
            continue

        v1 = check_simple_camera_dims(
            mesh, orientation, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
            top_angle_deg=TOP_ANGLE, side_angle_deg=0.0,
        )
        v2 = check_simple_camera_dims_v2(
            mesh, orientation, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
            top_angle_deg=TOP_ANGLE, side_angle_deg=0.0,
        )
        v3 = check_simple_camera_dims_v3(
            mesh, orientation, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
            top_angle_deg=TOP_ANGLE, side_angles_deg=VIEW_ANGLES,
            belt_positions=BELT_POSITIONS,
        )

        v1_err = max_abs_err_pct([v1.length, v1.width, v1.height], true_dims) if v1 else float("nan")
        v2_err = max_abs_err_pct([v2.length, v2.width, v2.height], true_dims) if v2 else float("nan")
        v3_err = max_abs_err_pct([v3.length, v3.width, v3.height], true_dims) if v3 else float("nan")

        true_str = f"({true_dims[0]:.0f},{true_dims[1]:.0f},{true_dims[2]:.0f})"
        print(f"{label:<22} {true_str:>20} {v1_err:>8.1f} {v2_err:>8.1f} {v3_err:>8.1f}")

        all_v1.append(v1_err)
        all_v2.append(v2_err)
        all_v3.append(v3_err)

    print("-" * 70)
    v1_max = max(all_v1) if all_v1 else float("nan")
    v2_max = max(all_v2) if all_v2 else float("nan")
    v3_max = max(all_v3) if all_v3 else float("nan")
    print(f"{'MAX':<22} {'':>20} {v1_max:>8.1f} {v2_max:>8.1f} {v3_max:>8.1f}")


if __name__ == "__main__":
    main()