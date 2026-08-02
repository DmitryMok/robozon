"""Замеры скорости v3 (единственный метод габаритов) при параметрах РЕАЛЬНОЙ камеры
(2592×1944, FOV 68°, 2м). Разделяет: (1) построение силуэтов — общая часть; (2) расчёт
габаритов (сетка contained_all + tri_height + fallback) БЕЗ времени силуэтов.

Логика замера:
- Силуэты: повторно строим тот же набор, что v3 использует внутри (top + side×belt), и
  замеряем только это.
- Расчёт v3: полное время check_simple_camera_dims_v3 минус время построения силуэтов
  (повторный вызов build_silhouette для тех же ракурсов — честная оценка силуэтной части).
  Это изоляция «чистого расчёта»: оптимизация/сетка/триангуляция без растеризации.

Usage:
    python bench_speed_v3.py [имена объектов...]   # без аргументов — весь набор
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from roboson_tools.core.experiment import (
    Orientation,
    apply_orientation,
    bounding_box,
    check_simple_camera_dims_v3,
)
from roboson_tools.geometry.mesh_io import Mesh, load_stl
from roboson_tools.silhouette import camera as camera_silhouette

FOV = 68.0
DIST = 2000.0
RES = (2592, 1944)
MIRROR = {45.0: 2600.0, 135.0: 2600.0}
SS = 3

V3_SIDES = [0.0, 45.0, 135.0]
V3_BELT = ["start", "center", "end"]

ASSETS = Path(__file__).parent / "assets" / "stl"
_SIZE_TO_NAME = {
    326184: "Bottle", 29684: "Box300", 26884: "Box400", 578784: "LunchBox",
    5675429: "Bag", 644084: "Pouf", 2046384: "Pen", 125284: "Plate", 107684: "Cylinder",
    2758034: "Helmet", 3637684: "Detergent",
}


def _grounded(mesh: Mesh, orient: Orientation) -> Mesh:
    o = apply_orientation(mesh, orient.roll_deg, orient.pitch_deg, orient.yaw_deg)
    bb_min, _ = bounding_box(o.vertices)
    return Mesh(vertices=o.vertices - np.array([0.0, 0.0, bb_min[2]]), faces=o.faces)


def _time_silhouettes_v3(grounded: Mesh) -> float:
    """Время построения всех силуэтов v3 (top + side×belt) — растеризация с супсемплингом."""
    t0 = time.perf_counter()
    camera_silhouette.build_silhouette(grounded, 90.0, RES, FOV, DIST, "center", supersample=SS)
    for sa in V3_SIDES:
        d = float(MIRROR.get(sa, DIST))
        for bp in V3_BELT:
            camera_silhouette.build_silhouette(grounded, sa, RES, FOV, d, bp, supersample=SS)
    return time.perf_counter() - t0


def main():
    only = {a.lower() for a in sys.argv[1:]}
    files = []
    for p in sorted(ASSETS.iterdir()):
        if p.suffix.lower() != ".stl":
            continue
        name = _SIZE_TO_NAME.get(p.stat().st_size, p.stem)
        if only and name.lower() not in only:
            continue
        files.append((name, p))

    print(f"{'object':<12} {'yaw':>5} | {'силуэты':>8} {'расчёт':>7} {'итого':>7}")
    for name, p in files:
        mesh = load_stl(p)
        for yaw in [0.0, 45.0]:
            orient = Orientation(yaw_deg=yaw)
            grounded = _grounded(mesh, orient)
            t_sil = _time_silhouettes_v3(grounded)
            t0 = time.perf_counter()
            r = check_simple_camera_dims_v3(
                mesh, orient, FOV, DIST, RES, side_angles_deg=V3_SIDES,
                belt_positions=V3_BELT, camera_distances=MIRROR, supersample=SS,
                use_simple_shape_detector=False,
            )
            t_tot = time.perf_counter() - t0
            t_calc = max(0.0, t_tot - t_sil)
            ok = "ok" if r is not None else "None"
            print(
                f"{name:<12} {yaw:>5.1f} | {t_sil:>7.2f}s {t_calc:>6.2f}s {t_tot:>6.2f}s  [{ok}]",
                flush=True,
            )


if __name__ == "__main__":
    main()