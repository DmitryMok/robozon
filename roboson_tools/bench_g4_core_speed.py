"""Этап 2 — замер скорости обработки облака в логике G4 (без учёта реконструкции).

Для каждого из 18 объектов строится облако 3D-точек одним из 4 методов реконструкции, затем
вызывается `_g4_core_from_points` (вынесенная из `check_model_roll_g4` часть после получения
облака). Замеряется только время `_g4_core_from_points` — это «чистая» скорость логики G4
(PCA → 6 направлений → локальный поиск → вердикт) на облаке заданного размера.

Цель — понять, насколько размер облака (N точек) влияет на время G4, и какие методы
реконструкции дают облака, на которых G4 работает в бюджете.

Методы реконструкции (только дающие облако, время их работы не учитывается — это этап 1):
  - band_ortho     — `reconstruct_shape_from_silhouettes` на ортографике (baseline G4).
  - band_persp_c   — то же на перспективе, center (текущий G4 в Camera Mode).
  - voxel_c        — `voxel_carving.carve` на перспективе, center.
  - voxel_p        — `voxel_carving.carve` на перспективе, parallax (12 видов).
  - polytope_c     — `polytope_hull.carve`, center.
  - polytope_p     — `polytope_hull.carve`, parallax.
  - exact_c        — `exact_polyhedral_hull.carve`, center.
  - exact_p        — `exact_polyhedral_hull.carve`, parallax.

Запуск: /tmp/robozon-bench-venv/bin/python bench_g4_core_speed.py [out_json]
Отчёт: exports/g4_core_speed.json (по умолчанию) + таблица в stdout.

RUNS = 3 на (объект, метод) — медиана elapsed_seconds.
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
    _g4_core_from_points,
)
from roboson_tools.geometry.mesh_io import load_stl, Mesh, bounding_box
from roboson_tools.visual_hull.reconstruct import reconstruct_shape_from_silhouettes
from roboson_tools.visual_hull import voxel_carving, polytope_hull, exact_polyhedral_hull

ASSETS_STL_DIR = Path(__file__).resolve().parent / "assets" / "stl"
DUMBBELL_STL = Path("/tmp/dumbbell.stl")
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
DEFAULT_OUT = EXPORTS_DIR / "g4_core_speed.json"

VIEW_ANGLES = [0, 45, 90, 135]
RESOLUTION_PX = 512
NUM_SLICES = 40
RUNS = 3

CAMERA_FOV_DEG = 68.0
CAMERA_DISTANCES = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}

VOXEL_GRID = 40
BELT_POSITIONS_PARALLAX = ["start", "center", "end"]

# Параметры G4 (как в check_model_roll_g4 по умолчанию)
ROUNDNESS_THRESHOLD = 0.8
LOW_THRESHOLD = 0.65
SWEEP_RESOLUTION_PX = 512
LOCAL_SEARCH_RADIUS_DEG = 20.0
LOCAL_SEARCH_STEP_DEG = 5.0


def _stl_files() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for p in sorted(ASSETS_STL_DIR.iterdir()):
        if p.suffix.lower() == ".stl":
            items.append((p.stem, p))
    if DUMBBELL_STL.exists():
        items.append(("Гантель_синтетическая", DUMBBELL_STL))
    return items


def _band_points_ortho(oriented: Mesh) -> np.ndarray | None:
    silhouettes = build_silhouette_set(oriented, VIEW_ANGLES, RESOLUTION_PX, use_camera=False)
    shape = reconstruct_shape_from_silhouettes(silhouettes, NUM_SLICES)
    if shape is None:
        return None
    chunks = []
    for x_pos, coords in shape.slices:
        arr = np.asarray(coords[:-1], dtype=np.float64)
        chunks.append(np.column_stack([np.full(len(arr), x_pos), arr[:, 0], arr[:, 1]]))
    return np.concatenate(chunks) if chunks else None


def _band_points_persp(oriented: Mesh, belt_position: str) -> np.ndarray | None:
    silhouettes = build_silhouette_set(
        oriented, VIEW_ANGLES, RESOLUTION_PX, use_camera=True,
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
    return np.concatenate(chunks) if chunks else None


def _voxel_points(oriented: Mesh, views) -> np.ndarray | None:
    hull = voxel_carving.carve(
        oriented, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX, grid_resolution=VOXEL_GRID,
    )
    return None if hull is None else hull.points


def _polytope_points(oriented: Mesh, views) -> np.ndarray | None:
    hull = polytope_hull.carve(
        oriented, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
    )
    return None if hull is None else hull.vertices


def _exact_points(oriented: Mesh, views) -> np.ndarray | None:
    hull = exact_polyhedral_hull.carve(
        oriented, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
    )
    return None if hull is None else hull.vertices


def _build_cloud(name: str, oriented: Mesh) -> np.ndarray | None:
    builders = {
        "band_ortho":    lambda: _band_points_ortho(oriented),
        "band_persp_c":  lambda: _band_points_persp(oriented, "center"),
        "voxel_c":       lambda: _voxel_points(oriented, [(a, "center") for a in VIEW_ANGLES]),
        "voxel_p":       lambda: _voxel_points(oriented, [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX]),
        "polytope_c":    lambda: _polytope_points(oriented, [(a, "center") for a in VIEW_ANGLES]),
        "polytope_p":    lambda: _polytope_points(oriented, [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX]),
        "exact_c":       lambda: _exact_points(oriented, [(a, "center") for a in VIEW_ANGLES]),
        "exact_p":       lambda: _exact_points(oriented, [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX]),
    }
    return builders[name]()


def _time_g4_core(points_3d: np.ndarray | None, mesh_dims: tuple[float, float, float]):
    """Прогон `_g4_core_from_points` RUNS раз. Возвращает (медиана elapsed_seconds, verdict,
    k, fallback, N точек). Если points_3d None — все значения None/0 (облако не построено)."""
    if points_3d is None or len(points_3d) < 4:
        return None, None, None, None, 0
    elapsed = []
    last = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        last = _g4_core_from_points(
            points_3d,
            dims=None,
            mesh_dims=mesh_dims,
            start_time=t0,
            resolution_px=RESOLUTION_PX,
            roundness_threshold=ROUNDNESS_THRESHOLD,
            low_threshold=LOW_THRESHOLD,
            sweep_resolution_px=SWEEP_RESOLUTION_PX,
            local_search_radius_deg=LOCAL_SEARCH_RADIUS_DEG,
            local_search_step_deg=LOCAL_SEARCH_STEP_DEG,
        )
        elapsed.append(time.perf_counter() - t0)
    return (
        statistics.median(elapsed),
        last.verdict,
        last.best.k if last.best else 0.0,
        last.fallback_triggered,
        len(points_3d),
    )


METHODS = [
    "band_ortho", "band_persp_c",
    "voxel_c", "voxel_p",
    "polytope_c", "polytope_p",
    "exact_c", "exact_p",
]
METHOD_LABELS = {
    "band_ortho":   "band ORTHO",
    "band_persp_c": "band PERSP_c",
    "voxel_c":      "voxel PERSP_c",
    "voxel_p":      "voxel PERSP_p",
    "polytope_c":   "polytope PERSP_c",
    "polytope_p":   "polytope PERSP_p",
    "exact_c":      "exact PERSP_c",
    "exact_p":      "exact PERSP_p",
}


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 200)
    print("Этап 2 — замер скорости логики G4 на облаках от 8 (метод × режим) комбинаций")
    print(f"  resolution_px={RESOLUTION_PX}  sweep_resolution_px={SWEEP_RESOLUTION_PX}")
    print(f"  roundness_threshold={ROUNDNESS_THRESHOLD}  low_threshold={LOW_THRESHOLD}")
    print(f"  local_search: radius={LOCAL_SEARCH_RADIUS_DEG}° step={LOCAL_SEARCH_STEP_DEG}°")
    print(f"  runs={RUNS}")
    print("=" * 200)
    header = f"{'Объект':<24} | " + " | ".join(f"{METHOD_LABELS[m]:<15} {'v':<10} {'k':<6} {'fb':<3} {'t,мс':<6} {'N':<6}" for m in METHODS)
    print(header)
    print("-" * 200)

    rows = []
    for name, path in _stl_files():
        mesh = load_stl(path)
        oriented = apply_orientation(mesh, 0.0, 0.0, 0.0)
        bbox_min, bbox_max = bounding_box(oriented.vertices)
        mesh_dims = tuple((bbox_max - bbox_min).tolist())

        row = {"object": name}
        cells = []
        for m in METHODS:
            points = _build_cloud(m, oriented)
            t_med, verdict, k, fb, n = _time_g4_core(points, mesh_dims)
            v_label = verdict if verdict is not None else "—"
            fb_label = "+" if fb else ("-" if fb is False else " ")
            t_ms = f"{t_med*1000:.0f}" if t_med is not None else "—"
            k_s = f"{k:.3f}" if k is not None else "—"
            cells.append(f"{v_label:<10} {k_s:<6} {fb_label:<3} {t_ms:<6} {n:<6}")
            row[m] = {
                "verdict": verdict, "k": k, "fallback": fb,
                "t_ms": t_med * 1000 if t_med is not None else None,
                "n_points": n,
            }
        print(f"{name:<24} | " + " | ".join(f"{METHOD_LABELS[m]:<15} {cells[i]}" for i, m in enumerate(METHODS)))
        rows.append(row)

    print("-" * 200)
    # Сводка: медиана / худший по 18 объектам, отдельно для времени G4-ядра
    print("Сводка (медиана / худший / сумма по 18 объектам, мс G4-ядра; не учтено время реконструкции):")
    for m in METHODS:
        ts = [r[m]["t_ms"] for r in rows if r[m]["t_ms"] is not None]
        ns = [r[m]["n_points"] for r in rows if r[m]["n_points"] > 0]
        if not ts:
            print(f"  {METHOD_LABELS[m]:<22}  все None (облака не построены)")
            continue
        med = statistics.median(ts)
        worst = max(ts)
        n_med = statistics.median(ns) if ns else 0
        print(f"  {METHOD_LABELS[m]:<22}  median={med:>6.1f}  worst={worst:>6.1f}  median N={n_med:>6.0f}")

    # Число валидных вердиктов (облако построено)
    print()
    print("Число объектов с построенным облаком (вердикт не None):")
    for m in METHODS:
        n_valid = sum(1 for r in rows if r[m]["verdict"] is not None)
        print(f"  {METHOD_LABELS[m]:<22}  {n_valid}/18")

    report = {
        "config": {
            "view_angles": VIEW_ANGLES,
            "resolution_px": RESOLUTION_PX,
            "sweep_resolution_px": SWEEP_RESOLUTION_PX,
            "num_slices": NUM_SLICES,
            "voxel_grid": VOXEL_GRID,
            "camera_fov_deg": CAMERA_FOV_DEG,
            "camera_distances": CAMERA_DISTANCES,
            "roundness_threshold": ROUNDNESS_THRESHOLD,
            "low_threshold": LOW_THRESHOLD,
            "local_search_radius_deg": LOCAL_SEARCH_RADIUS_DEG,
            "local_search_step_deg": LOCAL_SEARCH_STEP_DEG,
            "runs": RUNS,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON-отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())