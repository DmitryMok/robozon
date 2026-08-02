"""Стоимость supersample=3 vs 1 по времени, отдельно на больших и маленьких объектах.

Замеряет полное время check_simple_camera_dims_v3 (включая построение силуэтов)
на боевых параметрах. Usage: python bench_ss_cost.py
"""
from __future__ import annotations

import time
from pathlib import Path

from roboson_tools.core.experiment import Orientation, check_simple_camera_dims_v3
from roboson_tools.geometry.mesh_io import load_stl

FOV_DEG = 68.0
CAMERA_DISTANCE = 2000.0
RESOLUTION = (2592, 1944)
DISTANCES = {45.0: 2600.0, 135.0: 2600.0}
OBJECTS = [
    ("Ручка.stl", "Pen (маленький)"),
    ("Тарелка.stl", "Plate (плоский)"),
    ("Короб 400х400х300.stl", "Box400 (большой)"),
    ("Пуфик.stl", "Pouf (большой)"),
    ("Мешок.stl", "Bag (большой, тяжёлый меш)"),
]

for stl, label in OBJECTS:
    mesh = load_stl(Path("assets/stl") / stl)
    row = [label]
    for ss in (1, 3):
        # прогрев не делаем: важна относительная разница, JIT нет
        t0 = time.perf_counter()
        r = check_simple_camera_dims_v3(
            mesh, Orientation(0, 0, 0), FOV_DEG, CAMERA_DISTANCE, RESOLUTION,
            side_angles_deg=[0.0, 45.0, 135.0], belt_positions=["start", "center", "end"],
            camera_distances=DISTANCES, supersample=ss,
        )
        dt = time.perf_counter() - t0
        row.append(f"ss={ss}: {dt:.2f}s")
    print(" | ".join(row), flush=True)
