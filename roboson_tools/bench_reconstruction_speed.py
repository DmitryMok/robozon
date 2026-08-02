"""Этап 1 — замер скорости 3D-реконструкции на 18 объектах × 3 метода × 2 режима перспективы.

Методы (все — Camera Mode, перспективная проекция; для baseline добавлен band-перебор
`reconstruct_shape_from_silhouettes` на ортографике):
  - reconstruct   — `reconstruct_shape_from_silhouettes` (band-пересечение).
                    Базовый путь G4 в текущей реализации. На перспективе с разными
                    `camera_distance` ломается на start/end (см. заметку задачи) — поэтому
                    замеряется только на ORTHO (baseline) и PERSP_CENTER (где случайно
                    работает); PERSP_PARALLAX для него помечен как «известно ломается».
  - voxel_carving — `visual_hull/voxel_carving.carve` (сетка вокселей в мировых координатах).
  - polytope_hull — `visual_hull/polytope_hull.carve` (точная выпуклая оболочка, qhull).
  - exact_hull    — `visual_hull/exact_polyhedral_hull.carve` (точная, сохраняет вогнутости,
                    manifold3d boolean).

Режимы (набор `views` для конусных методов):
  - PERSP_CENTER   — 4 вида: (azimuth, "center") для каждого из [0, 45, 90, 135].
  - PERSP_PARALLAX — 12 видов: (azimuth, bp) для bp ∈ [start, center, end] × 4 азимута.

Запуск: /tmp/robozon-bench-venv/bin/python bench_reconstruction_speed.py [out_json]
Отчёт: exports/reconstruction_speed.json (по умолчанию) + таблица в stdout.

Параметры Camera Mode — из config/app_settings.yaml + camera_rig_presets.yaml (дефолт).
RUNS = 3 на (объект, метод, режим) — медиана, чтобы сгладить шум. reconstruct (ORTHO) быстрый,
можно RUNS больше; voxel/polytope/exact медленнее, 3 достаточно для оценки порядка.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from roboson_tools.core.experiment import (
    Orientation,
    apply_orientation,
    build_silhouette_set,
)
from roboson_tools.geometry.mesh_io import load_stl, Mesh, bounding_box
from roboson_tools.visual_hull.reconstruct import reconstruct_shape_from_silhouettes
from roboson_tools.visual_hull import voxel_carving, polytope_hull, exact_polyhedral_hull

ASSETS_STL_DIR = Path(__file__).resolve().parent / "assets" / "stl"
DUMBBELL_STL = Path("/tmp/dumbbell.stl")
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
DEFAULT_OUT = EXPORTS_DIR / "reconstruction_speed.json"

VIEW_ANGLES = [0, 45, 90, 135]
RESOLUTION_PX = 512
NUM_SLICES = 40
RUNS = 3

# Camera Mode — из config (дефолтный пресет «2 камеры + зеркало»)
CAMERA_FOV_DEG = 68.0
CAMERA_DISTANCES = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}

# Voxel carving — сетка по умолчанию как в voxel_carving.carve (40³).
VOXEL_GRID = 40

BELT_POSITIONS_PARALLAX = ["start", "center", "end"]


def _stl_files() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for p in sorted(ASSETS_STL_DIR.iterdir()):
        if p.suffix.lower() == ".stl":
            items.append((p.stem, p))
    if DUMBBELL_STL.exists():
        items.append(("Гантель_синтетическая", DUMBBELL_STL))
    return items


def _bbox_extent(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) == 0:
        return None
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return tuple(float(v) for v in (mx - mn))


def _reconstruct_ortho_points(mesh: Mesh, belt_position: str) -> np.ndarray | None:
    """band-пересечение на ортографике (baseline). Возвращает облако (N,3) — та же сборка,
    что в check_model_roll_g4. belt_position игнорируется (ортографика не зависит от X)."""
    silhouettes = build_silhouette_set(mesh, VIEW_ANGLES, RESOLUTION_PX, use_camera=False)
    shape = reconstruct_shape_from_silhouettes(silhouettes, NUM_SLICES)
    if shape is None:
        return None
    chunks = []
    for x_pos, coords in shape.slices:
        arr = np.asarray(coords[:-1], dtype=np.float64)
        chunks.append(np.column_stack([np.full(len(arr), x_pos), arr[:, 0], arr[:, 1]]))
    if not chunks:
        return None
    return np.concatenate(chunks)


def _reconstruct_persp_band_points(mesh: Mesh, belt_position: str) -> np.ndarray | None:
    """band-пересечение на перспективе (то, что сейчас делает G4 в Camera Mode)."""
    silhouettes = build_silhouette_set(
        mesh, VIEW_ANGLES, RESOLUTION_PX, use_camera=True,
        camera_fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        belt_position=belt_position,
    )
    shape = reconstruct_shape_from_silhouettes(silhouettes, NUM_SLICES)
    if shape is None:
        return None
    chunks = []
    for x_pos, coords in shape.slices:
        arr = np.asarray(coords[:-1], dtype=np.float64)
        chunks.append(np.column_stack([np.full(len(arr), x_pos), arr[:, 0], arr[:, 1]]))
    if not chunks:
        return None
    return np.concatenate(chunks)


def _reconstruct_voxel_carving(mesh: Mesh, views) -> np.ndarray | None:
    hull = voxel_carving.carve(
        mesh, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX, grid_resolution=VOXEL_GRID,
    )
    return None if hull is None else hull.points


def _reconstruct_polytope_hull(mesh: Mesh, views) -> np.ndarray | None:
    hull = polytope_hull.carve(
        mesh, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
    )
    return None if hull is None else hull.vertices


def _reconstruct_exact_hull(mesh: Mesh, views) -> np.ndarray | None:
    hull = exact_polyhedral_hull.carve(
        mesh, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
    )
    return None if hull is None else hull.vertices


def _time_call(func, *args, runs=RUNS):
    """Прогон `runs` раз, возврат (последний результат, медиана elapsed_seconds, N точек или None)."""
    last = None
    elapsed = []
    for _ in range(runs):
        t0 = time.perf_counter()
        last = func(*args)
        elapsed.append(time.perf_counter() - t0)
    n_pts = 0 if last is None else len(last)
    return last, statistics.median(elapsed), n_pts


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    views_center = [(a, "center") for a in VIEW_ANGLES]
    views_parallax = [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX]

    print("=" * 160)
    print("Этап 1 — замер скорости 3D-реконструкции (18 объектов × 4 метода × 2-3 режима)")
    print(f"  view_angles={VIEW_ANGLES}  resolution_px={RESOLUTION_PX}  num_slices={NUM_SLICES}")
    print(f"  voxel_grid={VOXEL_GRID}  camera_fov_deg={CAMERA_FOV_DEG}")
    print(f"  camera_distances={CAMERA_DISTANCES}")
    print(f"  views_center={len(views_center)}  views_parallax={len(views_parallax)}")
    print(f"  runs={( 'reconstruct_ortho/persp: %d' % RUNS, 'voxel/polytope/exact: %d' % RUNS)}")
    print("=" * 160)
    header = (
        f"{'Объект':<24} | "
        f"{'ORTHO band t,мс':<16} {'N':<6} | "
        f"{'PERSP band CENTER':<18} {'N':<6} | "
        f"{'PERSP band PARALLAX':<19} {'N':<6} | "
        f"{'voxel CENTER':<13} {'N':<6} | "
        f"{'voxel PARALLAX':<15} {'N':<6} | "
        f"{'polytope CENTER':<16} {'N':<6} | "
        f"{'polytope PARALLAX':<18} {'N':<6} | "
        f"{'exact CENTER':<13} {'N':<6} | "
        f"{'exact PARALLAX':<15} {'N':<6}"
    )
    print(header)
    print("-" * 160)

    rows = []
    for name, path in _stl_files():
        mesh = load_stl(path)
        oriented = apply_orientation(mesh, 0.0, 0.0, 0.0)

        # ORTHO band (baseline)
        _, t_o, n_o = _time_call(_reconstruct_ortho_points, oriented, "center")
        # PERSP band CENTER (известно: работает только случайно)
        _, t_pc, n_pc = _time_call(_reconstruct_persp_band_points, oriented, "center")
        # PERSP band PARALLAX — отдельно по 3 позициям, берём max (худший из 3 прогонов
        # single-position; в G4 сейчас это 3 отдельных вызова). Для сравнения с конусными
        # методами, которые делают ОДНУ реконструкцию по 12 видам, здесь указываем худший
        # из 3 single-position вызовов — но это НЕ эквивалент (см. заметку задачи).
        t_pp_max = 0.0
        n_pp_max = 0
        for bp in BELT_POSITIONS_PARALLAX:
            _, t_bp, n_bp = _time_call(_reconstruct_persp_band_points, oriented, bp, runs=1)
            t_pp_max = max(t_pp_max, t_bp)
            n_pp_max = max(n_pp_max, n_bp)

        # voxel CENTER / PARALLAX
        _, t_vc, n_vc = _time_call(_reconstruct_voxel_carving, oriented, views_center)
        _, t_vp, n_vp = _time_call(_reconstruct_voxel_carving, oriented, views_parallax)

        # polytope CENTER / PARALLAX
        _, t_pc2, n_pc2 = _time_call(_reconstruct_polytope_hull, oriented, views_center)
        _, t_pp2, n_pp2 = _time_call(_reconstruct_polytope_hull, oriented, views_parallax)

        # exact CENTER / PARALLAX
        _, t_ec, n_ec = _time_call(_reconstruct_exact_hull, oriented, views_center)
        _, t_ep, n_ep = _time_call(_reconstruct_exact_hull, oriented, views_parallax)

        def _ms(t): return f"{t*1000:.0f}"
        def _n(n): return str(n) if n > 0 else "—"
        print(
            f"{name:<24} | "
            f"{_ms(t_o):<16} {_n(n_o):<6} | "
            f"{_ms(t_pc):<18} {_n(n_pc):<6} | "
            f"{_ms(t_pp_max):<19} {_n(n_pp_max):<6} | "
            f"{_ms(t_vc):<13} {_n(n_vc):<6} | "
            f"{_ms(t_vp):<15} {_n(n_vp):<6} | "
            f"{_ms(t_pc2):<16} {_n(n_pc2):<6} | "
            f"{_ms(t_pp2):<18} {_n(n_pp2):<6} | "
            f"{_ms(t_ec):<13} {_n(n_ec):<6} | "
            f"{_ms(t_ep):<15} {_n(n_ep):<6}"
        )

        rows.append({
            "object": name,
            "ortho_band": {"t_ms": t_o * 1000, "n_points": n_o},
            "persp_band_center": {"t_ms": t_pc * 1000, "n_points": n_pc},
            "persp_band_parallax_worst_of_3_singles": {"t_ms": t_pp_max * 1000, "n_points": n_pp_max},
            "voxel_center": {"t_ms": t_vc * 1000, "n_points": n_vc},
            "voxel_parallax": {"t_ms": t_vp * 1000, "n_points": n_vp},
            "polytope_center": {"t_ms": t_pc2 * 1000, "n_points": n_pc2},
            "polytope_parallax": {"t_ms": t_pp2 * 1000, "n_points": n_pp2},
            "exact_center": {"t_ms": t_ec * 1000, "n_points": n_ec},
            "exact_parallax": {"t_ms": t_ep * 1000, "n_points": n_ep},
        })

    # Сводка по медианам и худшим случаям
    print("-" * 160)
    print("Сводка (медиана / худший по 18 объектам, мс):")
    methods = [
        ("ORTHO band", [r["ortho_band"]["t_ms"] for r in rows]),
        ("PERSP band CENTER", [r["persp_band_center"]["t_ms"] for r in rows]),
        ("PERSP band PARALLAX (x3 worst)", [r["persp_band_parallax_worst_of_3_singles"]["t_ms"] for r in rows]),
        ("voxel CENTER", [r["voxel_center"]["t_ms"] for r in rows]),
        ("voxel PARALLAX", [r["voxel_parallax"]["t_ms"] for r in rows]),
        ("polytope CENTER", [r["polytope_center"]["t_ms"] for r in rows]),
        ("polytope PARALLAX", [r["polytope_parallax"]["t_ms"] for r in rows]),
        ("exact CENTER", [r["exact_center"]["t_ms"] for r in rows]),
        ("exact PARALLAX", [r["exact_parallax"]["t_ms"] for r in rows]),
    ]
    for label, ts in methods:
        med = statistics.median(ts)
        worst = max(ts)
        print(f"  {label:<32}  median={med:>7.1f}  worst={worst:>7.1f}")

    # Число точек (медиана) — оценка плотности облака для этапа 2
    print()
    print("Сводка (медиана N точек):")
    n_methods = [
        ("ORTHO band", [r["ortho_band"]["n_points"] for r in rows]),
        ("PERSP band CENTER", [r["persp_band_center"]["n_points"] for r in rows]),
        ("voxel CENTER", [r["voxel_center"]["n_points"] for r in rows]),
        ("voxel PARALLAX", [r["voxel_parallax"]["n_points"] for r in rows]),
        ("polytope CENTER", [r["polytope_center"]["n_points"] for r in rows]),
        ("polytope PARALLAX", [r["polytope_parallax"]["n_points"] for r in rows]),
        ("exact CENTER", [r["exact_center"]["n_points"] for r in rows]),
        ("exact PARALLAX", [r["exact_parallax"]["n_points"] for r in rows]),
    ]
    for label, ns in n_methods:
        med = statistics.median(ns)
        print(f"  {label:<32}  median N = {med:>7.0f}")

    report = {
        "config": {
            "view_angles": VIEW_ANGLES,
            "resolution_px": RESOLUTION_PX,
            "num_slices": NUM_SLICES,
            "voxel_grid": VOXEL_GRID,
            "camera_fov_deg": CAMERA_FOV_DEG,
            "camera_distances": CAMERA_DISTANCES,
            "runs": RUNS,
        },
        "rows": rows,
        "summary_median_ms": {label: statistics.median(ts) for label, ts in methods},
        "summary_worst_ms": {label: max(ts) for label, ts in methods},
        "summary_median_n_points": {label: statistics.median(ns) for label, ns in n_methods},
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON-отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())