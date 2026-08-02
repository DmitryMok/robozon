"""Этап 3 — полный end-to-end бенчмарк G4 (реконструкция + обработка) × 4 метода × 2 режима.

Для каждого из 18 объектов считается вердикт G4 (реконструкция + `_g4_core_from_points`)
в каждой из 8 комбинаций (метод реконструкции × режим перспективы). Метрики:
  - verdict (round / not_round / uncertain), k, fallback
  - совпадение с эталоном (round→катится, not_round→не катится, uncertain=OK в обе стороны)
  - end-to-end время (реконструкция + G4-ядро), мс
  - пропусков катающихся (катится-эталон + verdict=not_round)
  - ложных «катится» (не-круг-эталон + verdict=round)

Методы:
  - band_ortho    — `reconstruct_shape_from_silhouettes` на ортографике (baseline).
  - band_persp_c  — то же на перспективе, center (текущий G4 в Camera Mode).
  - voxel_c / voxel_p
  - polytope_c / polytope_p
  - exact_c / exact_p

Запуск: /tmp/robozon-bench-venv/bin/python bench_g4_endtoend.py [out_json]
Отчёт: exports/g4_endtoend.json (по умолчанию) + таблица в stdout.

RUNS = 1 на (объект, метод) — end-to-end время уже включает шум обеих стадий; медиана по
RUNS=3 была бы медленнее (3× reconstructions), а порядок метрики нас интересует, не точность
до мс. Время — единичный прогон.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from roboson_tools.core.experiment import (
    Orientation,
    apply_orientation,
    build_silhouette_set,
    _g4_core_from_points,
    ROLL_VERDICT_NOT_ROUND,
    ROLL_VERDICT_ROUND,
    ROLL_VERDICT_UNCERTAIN,
)
from roboson_tools.geometry.mesh_io import load_stl, Mesh, bounding_box
from roboson_tools.visual_hull.reconstruct import reconstruct_shape_from_silhouettes
from roboson_tools.visual_hull import voxel_carving, polytope_hull, exact_polyhedral_hull

ASSETS_STL_DIR = Path(__file__).resolve().parent / "assets" / "stl"
DUMBBELL_STL = Path("/tmp/dumbbell.stl")
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
DEFAULT_OUT = EXPORTS_DIR / "g4_endtoend.json"

VIEW_ANGLES = [0, 45, 90, 135]
RESOLUTION_PX = 1024  # было 512 в исходном бенчмарке — повышено для этапа 3 (План A):
                      # упрощение контура + ROI позволяет ставить 1024 без штрафа (маска
                      # обрезана до bbox объекта, не полный кадр 1024²).
NUM_SLICES = 40

CAMERA_FOV_DEG = 68.0
CAMERA_DISTANCES = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}

VOXEL_GRID = 40
BELT_POSITIONS_PARALLAX = ["start", "center", "end"]

# План A: упрощение контура + ROI (см. bench_g4_endtoend.py этап 3)
CONTOUR_EPSILON_PX = 1.5
CROP_TO_OBJECT = True

ROUNDNESS_THRESHOLD = 0.8
LOW_THRESHOLD = 0.65
SWEEP_RESOLUTION_PX = 512
LOCAL_SEARCH_RADIUS_DEG = 20.0
LOCAL_SEARCH_STEP_DEG = 5.0

# Эталон «круглый» — тот же, что в bench_roll_methods.py / bench_g4_parallax.py
GROUND_TRUTH = {
    "Пуфик": True, "Бутылка": True, "Тарелка": True, "Шлем": True,
    "Короб 300х200х200": False, "ЛанчБокс": False, "Моющее средство": False, "Ручка": False,
    "Мешок": True, "Короб 400х400х300": False, "Цилиндр": False,
    "Гантель_синтетическая": False,
    "asym_cone": True, "asym_barrel": True, "asym_cyl": True,
    # cyl_skewed: ПРАВКА 2026-07-31 (визуальный осмотр пользователем) — физически НЕ
    # катится, эталон был ошибочным, см. docs/method.md и bench_roll_methods.py.
    "cyl_skewed": False,
    "ell_cyl": True, "hourglass_asym": True,
}


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
        crop_to_object=CROP_TO_OBJECT,
    )
    return None if hull is None else hull.points


def _polytope_points(oriented: Mesh, views) -> np.ndarray | None:
    hull = polytope_hull.carve(
        oriented, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
        contour_epsilon_px=CONTOUR_EPSILON_PX, crop_to_object=CROP_TO_OBJECT,
    )
    return None if hull is None else hull.vertices


def _exact_points(oriented: Mesh, views) -> np.ndarray | None:
    hull = exact_polyhedral_hull.carve(
        oriented, views, fov_deg=CAMERA_FOV_DEG, camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
        contour_epsilon_px=CONTOUR_EPSILON_PX, crop_to_object=CROP_TO_OBJECT,
    )
    return None if hull is None else hull.vertices


METHODS = [
    ("band_ortho",    "band ORTHO",     lambda o: _band_points_ortho(o)),
    ("band_persp_c",  "band PERSP_c",   lambda o: _band_points_persp(o, "center")),
    ("voxel_c",       "voxel PERSP_c",  lambda o: _voxel_points(o, [(a, "center") for a in VIEW_ANGLES])),
    ("voxel_p",       "voxel PERSP_p",  lambda o: _voxel_points(o, [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX])),
    ("polytope_c",    "polytope PERSP_c", lambda o: _polytope_points(o, [(a, "center") for a in VIEW_ANGLES])),
    ("polytope_p",    "polytope PERSP_p", lambda o: _polytope_points(o, [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX])),
    ("exact_c",       "exact PERSP_c",  lambda o: _exact_points(o, [(a, "center") for a in VIEW_ANGLES])),
    ("exact_p",       "exact PERSP_p",  lambda o: _exact_points(o, [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX])),
]


def _verdict_matches(verdict: str | None, name: str) -> bool | None:
    gt = GROUND_TRUTH.get(name)
    if gt is None or verdict is None:
        return None
    if gt:
        # эталон «катится»: not_round = MISMATCH, round/uncertain = OK
        return verdict != ROLL_VERDICT_NOT_ROUND
    else:
        # эталон «не катится»: round = MISMATCH, not_round/uncertain = OK
        return verdict != ROLL_VERDICT_ROUND


def _verdict_label(v: str | None) -> str:
    if v is None:
        return "—"
    return {ROLL_VERDICT_ROUND: "round", ROLL_VERDICT_NOT_ROUND: "not_round",
            ROLL_VERDICT_UNCERTAIN: "uncertain"}[v]


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("Этап 3 — end-to-end G4 × 4 метода × 2 режима (вердикт + k + время)")
    print(f"  resolution_px={RESOLUTION_PX}  sweep_resolution_px={SWEEP_RESOLUTION_PX}")
    print(f"  roundness_threshold={ROUNDNESS_THRESHOLD}  low_threshold={LOW_THRESHOLD}")
    print(f"  local_search: radius={LOCAL_SEARCH_RADIUS_DEG}° step={LOCAL_SEARCH_STEP_DEG}°")
    print(f"  План A: contour_epsilon_px={CONTOUR_EPSILON_PX}  crop_to_object={CROP_TO_OBJECT}")
    print("=" * 220)
    header = f"{'Объект':<24} {'эталон':<7} | " + " | ".join(
        f"{label:<15} {'v':<10} {'k':<6} {'fb':<3} {'t,мс':<5}" for _key, label, _f in METHODS
    ) + " | совпадения"
    print(header)
    print("-" * 220)

    rows = []
    n_match = {key: 0 for key, _, _ in METHODS}
    n_miss = {key: 0 for key, _, _ in METHODS}  # пропусков катающихся (not_round при круглом эталоне)
    n_false = {key: 0 for key, _, _ in METHODS}  # ложных «катится» (round при некруглом эталоне)
    n_total = 0

    for name, path in _stl_files():
        mesh = load_stl(path)
        oriented = apply_orientation(mesh, 0.0, 0.0, 0.0)
        bbox_min, bbox_max = bounding_box(oriented.vertices)
        mesh_dims = tuple((bbox_max - bbox_min).tolist())
        gt = GROUND_TRUTH.get(name)
        gt_label = "круглый" if gt else "не круг"

        row = {"object": name, "ground_truth_round": gt}
        cells = []
        match_labels = []
        for key, label, builder in METHODS:
            t0 = time.perf_counter()
            points = builder(oriented)
            t_recon = time.perf_counter() - t0

            if points is None or len(points) < 4:
                verdict, k, fb, t_core = None, None, None, 0.0
            else:
                t1 = time.perf_counter()
                result = _g4_core_from_points(
                    points,
                    dims=None,
                    mesh_dims=mesh_dims,
                    start_time=t1,
                    resolution_px=RESOLUTION_PX,
                    roundness_threshold=ROUNDNESS_THRESHOLD,
                    low_threshold=LOW_THRESHOLD,
                    sweep_resolution_px=SWEEP_RESOLUTION_PX,
                    local_search_radius_deg=LOCAL_SEARCH_RADIUS_DEG,
                    local_search_step_deg=LOCAL_SEARCH_STEP_DEG,
                )
                t_core = time.perf_counter() - t1
                verdict = result.verdict
                k = result.best.k if result.best else 0.0
                fb = result.fallback_triggered

            t_total_ms = (t_recon + t_core) * 1000
            k_s = f"{k:.3f}" if k is not None else "—"
            fb_label = "+" if fb else ("-" if fb is False else " ")
            v_label = _verdict_label(verdict)
            cells.append(f"{v_label:<10} {k_s:<6} {fb_label:<3} {t_total_ms:<5.0f}")

            match = _verdict_matches(verdict, name)
            if match is not None:
                n_total_aligned = True  # считаем в n_total только при gt есть
                if match:
                    n_match[key] += 1
                # Подсчёт пропусков/ложных
                if gt is True and verdict == ROLL_VERDICT_NOT_ROUND:
                    n_miss[key] += 1
                elif gt is False and verdict == ROLL_VERDICT_ROUND:
                    n_false[key] += 1
            match_labels.append("+" if match else ("-" if match is False else "·"))

            row[key] = {
                "verdict": verdict, "k": k, "fallback": fb,
                "t_recon_ms": t_recon * 1000, "t_core_ms": t_core * 1000,
                "t_total_ms": t_total_ms, "n_points": 0 if points is None else len(points),
                "verdict_matches_ground_truth": match,
            }

        n_total += 1
        match_str = " ".join(f"{k}:{m}" for (k, _l, _f), m in zip(METHODS, match_labels))
        print(f"{name:<24} {gt_label:<7} | " + " | ".join(
            f"{METHODS[i][1]:<15} {cells[i]}" for i in range(len(METHODS))
        ) + f" | {match_str}")
        rows.append(row)

    print("-" * 220)
    print("Сводка по совпадениям с эталоном (round→катится, not_round→не катится, uncertain=OK):")
    for key, label, _ in METHODS:
        n_valid = sum(1 for r in rows if r[key]["verdict"] is not None)
        print(f"  {label:<22}  совпадений {n_match[key]}/{n_total}  "
              f"пропусков катящихся={n_miss[key]}  ложных «катится»={n_false[key]}  "
              f"облаков построено={n_valid}/18")

    # Время end-to-end: медиана/худший
    print()
    print("Время end-to-end (реконструкция + G4-ядро), мс:")
    for key, label, _ in METHODS:
        ts = [r[key]["t_total_ms"] for r in rows if r[key]["t_total_ms"] is not None and r[key]["verdict"] is not None]
        if not ts:
            print(f"  {label:<22}  все None")
            continue
        import statistics as st
        print(f"  {label:<22}  median={st.median(ts):>7.1f}  worst={max(ts):>7.1f}")

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
            "contour_epsilon_px": CONTOUR_EPSILON_PX,
            "crop_to_object": CROP_TO_OBJECT,
        },
        "summary": {
            "matches": {key: n_match[key] for key, _, _ in METHODS},
            "missed_rolling": {key: n_miss[key] for key, _, _ in METHODS},
            "false_round": {key: n_false[key] for key, _, _ in METHODS},
            "n_total": n_total,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON-отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())