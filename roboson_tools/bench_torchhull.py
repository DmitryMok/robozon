"""Бенчмарк torchhull vs polytope_hull на 18 объектах × CENTER и PARALLAX.

Сравнивает вердикты G4 (round/not_round/uncertain), max k, время (реконструкция + G4-ядро)
между GPU-методом torchhull (sparse voxel octree + marching cubes) и CPU-методом polytope_hull
(qhull, текущий кандидат из Плана A).

Окружение (venv /home/mdm3/Projects/robozon/.venv-cv + CUDA toolkit /home/mdm3/opt/cuda-12.1):
  export CUDA_HOME=/home/mdm3/opt/cuda-12.1
  export PATH=<venv>/bin:$CUDA_HOME/bin:$PATH
  export CC=/home/mdm3/opt/gcc-12/usr/bin/gcc-12
  export CXX=/home/mdm3/opt/gcc-12/usr/bin/g++-12
  export CUDAHOSTCXX=/home/mdm3/opt/gcc-12/usr/bin/g++-12
  /home/mdm3/Projects/robozon/.venv-cv/bin/python bench_torchhull.py

Этот скрипт использует roboson_tools (exp-26) для построения силуэтов и G4-ядра, и
robozon/.venv-cv для torchhull + torch CUDA. Поэтому запускается с PYTHONPATH, включающим
roboson_tools, и venv-cv как интерпретатор.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

# roboson_tools — должен быть в PYTHONPATH или в sys.path
ROBOSON_TOOLS = Path("/mnt/c/Projects/exp-26/roboson_tools")
sys.path.insert(0, str(ROBOSON_TOOLS))

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
from roboson_tools.silhouette.camera import build_silhouette, belt_offset
from roboson_tools.visual_hull import polytope_hull
from roboson_tools.visual_hull.torchhull_adapter import (
    build_transforms_batch,
    visual_hull_points,
)

ASSETS_STL_DIR = ROBOSON_TOOLS / "assets" / "stl"
DUMBBELL_STL = Path("/tmp/dumbbell.stl")
EXPORTS_DIR = ROBOSON_TOOLS / "exports"
DEFAULT_OUT = EXPORTS_DIR / "torchhull_comparison.json"

VIEW_ANGLES = [0.0, 45.0, 90.0, 135.0]
RESOLUTION_PX = 512
CAMERA_FOV_DEG = 68.0
CAMERA_DISTANCES = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}
BELT_POSITIONS_PARALLAX = ["start", "center", "end"]

# Параметры G4 (как в bench_g4_endtoend.py)
ROUNDNESS_THRESHOLD = 0.8
LOW_THRESHOLD = 0.65
SWEEP_RESOLUTION_PX = 512
LOCAL_SEARCH_RADIUS_DEG = 20.0
LOCAL_SEARCH_STEP_DEG = 5.0

# Параметры polytope_hull (План A)
CONTOUR_EPSILON_PX = 1.5
CROP_TO_OBJECT = True

# Параметры torchhull
TORCHHULL_LEVEL = 7  # 128^3 voxels — баланс точности/скорости (8 = 256^3, 9 = 512^3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUNS = 1  # end-to-end, без медианы — оцениваем порядок, не точность до мс

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


def _verdict_label(v: str | None) -> str:
    if v is None:
        return "—"
    return {ROLL_VERDICT_ROUND: "round", ROLL_VERDICT_NOT_ROUND: "not_round",
            ROLL_VERDICT_UNCERTAIN: "uncertain"}[v]


def _verdict_matches(verdict: str | None, name: str) -> bool | None:
    gt = GROUND_TRUTH.get(name)
    if gt is None or verdict is None:
        return None
    if gt:
        return verdict != ROLL_VERDICT_NOT_ROUND
    else:
        return verdict != ROLL_VERDICT_ROUND


def _build_silhouettes(oriented: Mesh, views: list[tuple[float, str]]):
    """Строит маски и belt_offset для списка (азимут, belt_position). Возвращает
    (masks, dxs, view_angles) — маски одинакового размера (без crop_to_object, т.к. torchhull
    требует одинаковый размер масок; вместо ROI используем полный кадр 512²)."""
    masks = []
    dxs = []
    angles = []
    for ang, bp in views:
        sil = build_silhouette(
            oriented, ang, RESOLUTION_PX,
            fov_deg=CAMERA_FOV_DEG,
            camera_distance=CAMERA_DISTANCES[ang],
            belt_position=bp,
            # Без crop_to_object — torchhull требует одинаковый размер масок
            crop_to_object=False,
        )
        masks.append(sil.mask)
        dx = belt_offset(oriented, CAMERA_FOV_DEG, CAMERA_DISTANCES[ang], bp)
        dxs.append(dx)
        angles.append(ang)
    return masks, dxs, angles


def _torchhull_points(oriented: Mesh, views: list[tuple[float, str]]):
    masks, dxs, angles = _build_silhouettes(oriented, views)
    transforms = build_transforms_batch(
        [(a, bp, dx) for (a, bp), dx in zip(views, dxs)],
        fov_deg=CAMERA_FOV_DEG,
        camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
    )
    # cube — bbox объекта + margin
    bbox_min, bbox_max = bounding_box(oriented.vertices)
    margin = float((bbox_max - bbox_min).max()) * 0.1
    lo = (bbox_min - margin).tolist()
    length = float((bbox_max - bbox_max.min() + 2 * margin).max())  # cube, охватывающий bbox
    cube_length = float((bbox_max - bbox_min).max() + 2 * margin)
    # corner = bbox_min - margin (по всем осям), но длина = max extent + 2*margin (куб)
    cube_corner = [float(bbox_min[i] - margin) for i in range(3)]
    masks_partial = any(bp != "center" for _, bp in views)
    return visual_hull_points(
        masks=masks,
        transforms=transforms,
        cube_corner_bfl=cube_corner,
        cube_length=cube_length,
        level=TORCHHULL_LEVEL,
        masks_partial=masks_partial,
        device=DEVICE,
    )


def _polytope_points(oriented: Mesh, views: list[tuple[float, str]]):
    hull = polytope_hull.carve(
        oriented, views,
        fov_deg=CAMERA_FOV_DEG,
        camera_distances=CAMERA_DISTANCES,
        resolution_px=RESOLUTION_PX,
        contour_epsilon_px=CONTOUR_EPSILON_PX,
        crop_to_object=CROP_TO_OBJECT,
    )
    return None if hull is None else hull.vertices


def _run_g4_core(points_3d: np.ndarray | None, mesh_dims: tuple[float, float, float]):
    if points_3d is None or len(points_3d) < 4:
        return None, None, None, 0.0
    t0 = time.perf_counter()
    result = _g4_core_from_points(
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
    t_core = time.perf_counter() - t0
    return result.verdict, (result.best.k if result.best else 0.0), result.fallback_triggered, t_core


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 200, file=sys.stderr)
    print(f"torchhull (GPU, level={TORCHHULL_LEVEL}) vs polytope_hull (CPU, eps={CONTOUR_EPSILON_PX}, crop={CROP_TO_OBJECT})",
          file=sys.stderr)
    print(f"  device={DEVICE}  resolution_px={RESOLUTION_PX}  camera_fov_deg={CAMERA_FOV_DEG}",
          file=sys.stderr)
    print(f"  camera_distances={CAMERA_DISTANCES}", file=sys.stderr)
    print("=" * 200, file=sys.stderr)

    views_center = [(a, "center") for a in VIEW_ANGLES]
    views_parallax = [(a, bp) for a in VIEW_ANGLES for bp in BELT_POSITIONS_PARALLAX]

    print(f"{'Объект':<24} {'эталон':<7} | "
          f"{'torchhull CENTER':<35} | {'torchhull PARALLAX':<35} | "
          f"{'polytope CENTER':<35} | {'polytope PARALLAX':<35}",
          file=sys.stderr)
    print("-" * 200, file=sys.stderr)

    rows = []
    n_match = {"th_c": 0, "th_p": 0, "pt_c": 0, "pt_p": 0}
    n_miss = {"th_c": 0, "th_p": 0, "pt_c": 0, "pt_p": 0}  # пропусков катящихся
    n_false = {"th_c": 0, "th_p": 0, "pt_c": 0, "pt_p": 0}  # ложных «катится»
    n_total = 0

    for name, path in _stl_files():
        mesh = load_stl(path)
        oriented = apply_orientation(mesh, 0.0, 0.0, 0.0)
        bbox_min, bbox_max = bounding_box(oriented.vertices)
        mesh_dims = tuple((bbox_max - bbox_min).tolist())
        gt = GROUND_TRUTH.get(name)
        gt_label = "круглый" if gt else "не круг"

        # torchhull CENTER
        t0 = time.perf_counter()
        th_c_pts = _torchhull_points(oriented, views_center)
        th_c_recon_ms = (time.perf_counter() - t0) * 1000
        th_c_v, th_c_k, th_c_fb, th_c_core_ms = _run_g4_core(th_c_pts, mesh_dims)

        # torchhull PARALLAX
        t0 = time.perf_counter()
        th_p_pts = _torchhull_points(oriented, views_parallax)
        th_p_recon_ms = (time.perf_counter() - t0) * 1000
        th_p_v, th_p_k, th_p_fb, th_p_core_ms = _run_g4_core(th_p_pts, mesh_dims)

        # polytope CENTER
        t0 = time.perf_counter()
        pt_c_pts = _polytope_points(oriented, views_center)
        pt_c_recon_ms = (time.perf_counter() - t0) * 1000
        pt_c_v, pt_c_k, pt_c_fb, pt_c_core_ms = _run_g4_core(pt_c_pts, mesh_dims)

        # polytope PARALLAX
        t0 = time.perf_counter()
        pt_p_pts = _polytope_points(oriented, views_parallax)
        pt_p_recon_ms = (time.perf_counter() - t0) * 1000
        pt_p_v, pt_p_k, pt_p_fb, pt_p_core_ms = _run_g4_core(pt_p_pts, mesh_dims)

        # Совпадения
        for key, v in [("th_c", th_c_v), ("th_p", th_p_v), ("pt_c", pt_c_v), ("pt_p", pt_p_v)]:
            m = _verdict_matches(v, name)
            if m is None:
                continue
            if m:
                n_match[key] += 1
            if gt is True and v == ROLL_VERDICT_NOT_ROUND:
                n_miss[key] += 1
            elif gt is False and v == ROLL_VERDICT_ROUND:
                n_false[key] += 1
        n_total += 1

        def _fmt(v, k, fb, recon_ms, core_ms, n_pts):
            n = 0 if n_pts is None else len(n_pts)
            v_l = _verdict_label(v)
            fb_l = "+" if fb else ("-" if fb is False else " ")
            k_l = f"{k:.3f}" if k is not None else "—"
            return f"{v_l:<10} k={k_l:<6} fb={fb_l} N={n:>5} t={(recon_ms+core_ms):>5.0f}"

        print(f"{name:<24} {gt_label:<7} | "
              f"{_fmt(th_c_v, th_c_k, th_c_fb, th_c_recon_ms, th_c_core_ms, th_c_pts):<35} | "
              f"{_fmt(th_p_v, th_p_k, th_p_fb, th_p_recon_ms, th_p_core_ms, th_p_pts):<35} | "
              f"{_fmt(pt_c_v, pt_c_k, pt_c_fb, pt_c_recon_ms, pt_c_core_ms, pt_c_pts):<35} | "
              f"{_fmt(pt_p_v, pt_p_k, pt_p_fb, pt_p_recon_ms, pt_p_core_ms, pt_p_pts):<35}",
              file=sys.stderr)

        rows.append({
            "object": name,
            "ground_truth_round": gt,
            "torchhull_center": {
                "verdict": th_c_v, "k": th_c_k, "fallback": th_c_fb,
                "recon_ms": th_c_recon_ms, "core_ms": th_c_core_ms,
                "n_points": 0 if th_c_pts is None else len(th_c_pts),
                "matches_gt": _verdict_matches(th_c_v, name),
            },
            "torchhull_parallax": {
                "verdict": th_p_v, "k": th_p_k, "fallback": th_p_fb,
                "recon_ms": th_p_recon_ms, "core_ms": th_p_core_ms,
                "n_points": 0 if th_p_pts is None else len(th_p_pts),
                "matches_gt": _verdict_matches(th_p_v, name),
            },
            "polytope_center": {
                "verdict": pt_c_v, "k": pt_c_k, "fallback": pt_c_fb,
                "recon_ms": pt_c_recon_ms, "core_ms": pt_c_core_ms,
                "n_points": 0 if pt_c_pts is None else len(pt_c_pts),
                "matches_gt": _verdict_matches(pt_c_v, name),
            },
            "polytope_parallax": {
                "verdict": pt_p_v, "k": pt_p_k, "fallback": pt_p_fb,
                "recon_ms": pt_p_recon_ms, "core_ms": pt_p_core_ms,
                "n_points": 0 if pt_p_pts is None else len(pt_p_pts),
                "matches_gt": _verdict_matches(pt_p_v, name),
            },
        })

    print("-" * 200, file=sys.stderr)
    print("Сводка (round→катится, not_round→не катится, uncertain=OK в обе стороны):", file=sys.stderr)
    print(f"  torchhull CENTER  : совпадений {n_match['th_c']}/{n_total}  "
          f"пропусков катящихся={n_miss['th_c']}  ложных «катится»={n_false['th_c']}", file=sys.stderr)
    print(f"  torchhull PARALLAX: совпадений {n_match['th_p']}/{n_total}  "
          f"пропусков катящихся={n_miss['th_p']}  ложных «катится»={n_false['th_p']}", file=sys.stderr)
    print(f"  polytope CENTER   : совпадений {n_match['pt_c']}/{n_total}  "
          f"пропусков катящихся={n_miss['pt_c']}  ложных «катится»={n_false['pt_c']}", file=sys.stderr)
    print(f"  polytope PARALLAX : совпадений {n_match['pt_p']}/{n_total}  "
          f"пропусков катящихся={n_miss['pt_p']}  ложных «катится»={n_false['pt_p']}", file=sys.stderr)

    # Время: медиана/худший по 18 объектам
    print("\nВремя end-to-end (мс):", file=sys.stderr)
    for label, key in [("torchhull CENTER", "torchhull_center"),
                       ("torchhull PARALLAX", "torchhull_parallax"),
                       ("polytope CENTER", "polytope_center"),
                       ("polytope PARALLAX", "polytope_parallax")]:
        ts = []
        for r in rows:
            v = r[key]["verdict"]
            if v is None:
                continue
            t = r[key]["recon_ms"] + r[key]["core_ms"]
            ts.append(t)
        if not ts:
            print(f"  {label:<22}  все None", file=sys.stderr)
            continue
        print(f"  {label:<22}  median={statistics.median(ts):>7.1f}  worst={max(ts):>7.1f}",
              file=sys.stderr)

    report = {
        "config": {
            "view_angles": VIEW_ANGLES,
            "resolution_px": RESOLUTION_PX,
            "camera_fov_deg": CAMERA_FOV_DEG,
            "camera_distances": CAMERA_DISTANCES,
            "torchhull_level": TORCHHULL_LEVEL,
            "device": str(DEVICE),
            "polytope_contour_epsilon_px": CONTOUR_EPSILON_PX,
            "polytope_crop_to_object": CROP_TO_OBJECT,
            "roundness_threshold": ROUNDNESS_THRESHOLD,
            "low_threshold": LOW_THRESHOLD,
        },
        "summary": {
            "matches": n_match,
            "missed_rolling": n_miss,
            "false_round": n_false,
            "n_total": n_total,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON-отчёт: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())