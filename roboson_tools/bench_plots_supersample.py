"""Визуальный анализ разброса ошибки v3 БЕЗ субпикселя (supersample=1) vs С субпикселем
(supersample=3) по разным ориентациям — по аналогии с bench_plots.py (v1 vs v3), но на
БОЕВЫХ параметрах камеры: 2592x1944, FOV 68, 2м, зеркальные ракурсы 45/135 -> 2.6м.

Для каждого STL строит график: ошибка (%) по оси Y, ориентация (yaw) по оси X,
отдельные линии для v3(ss=1) и v3(ss=3). Сохраняет PNG в bench_plots/.

Usage:
    python bench_plots_supersample.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboson_tools.core.experiment import (
    Orientation,
    check_simple_camera_dims_v3,
    mesh_true_dims,
)
from roboson_tools.geometry.mesh_io import load_stl

FOV_DEG = 68.0
CAMERA_DISTANCE = 2000.0
RESOLUTION = (2592, 1944)
DISTANCES = {45.0: 2600.0, 135.0: 2600.0}
SIDE_ANGLES = [0.0, 45.0, 135.0]
TOP_ANGLE = 90.0
BELT_POSITIONS = ["start", "center", "end"]

YAWS = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]

ASSETS_DIR = Path(__file__).parent / "assets" / "stl"
PLOT_DIR = Path(__file__).parent / "bench_plots"

_SIZE_TO_NAME = {
    326184: "Bottle", 29684: "Box300", 26884: "Box400", 578784: "LunchBox",
    5675429: "Bag", 3637684: "Detergent", 644084: "Pouf", 2046384: "Pen",
    125284: "Plate", 107684: "Cylinder", 2758034: "Helmet",
}


def max_abs_err_pct(dims, true_extents):
    d = sorted(dims, reverse=True)
    t = sorted(true_extents, reverse=True)
    return max(abs(100.0 * (a - b) / b) for a, b in zip(d, t))


def run_v3(mesh, ori, supersample):
    v3 = check_simple_camera_dims_v3(
        mesh, ori, FOV_DEG, CAMERA_DISTANCE, RESOLUTION,
        top_angle_deg=TOP_ANGLE, side_angles_deg=SIDE_ANGLES,
        belt_positions=BELT_POSITIONS, camera_distances=DISTANCES,
        supersample=supersample,
    )
    return v3


def main():
    PLOT_DIR.mkdir(exist_ok=True)

    seen = set()
    stl_files = []
    for p in sorted(ASSETS_DIR.iterdir()):
        if p.suffix.lower() == ".stl" and p.stem.lower() not in seen:
            seen.add(p.stem.lower())
            stl_files.append(p)

    results = {}  # label -> {"true": dims, "ss1": [errs], "ss3": [errs]}
    for stl_path in stl_files:
        label = _SIZE_TO_NAME.get(stl_path.stat().st_size, stl_path.stem)
        mesh = load_stl(stl_path)
        true_dims = mesh_true_dims(mesh, axis_step_deg=5.0)
        if true_dims is None:
            continue

        ss1_errs = []
        ss3_errs = []
        for yaw in YAWS:
            ori = Orientation(yaw_deg=yaw)
            r1 = run_v3(mesh, ori, supersample=1)
            r3 = run_v3(mesh, ori, supersample=3)
            e1 = max_abs_err_pct([r1.length, r1.width, r1.height], true_dims) if r1 else float("nan")
            e3 = max_abs_err_pct([r3.length, r3.width, r3.height], true_dims) if r3 else float("nan")
            ss1_errs.append(e1)
            ss3_errs.append(e3)

        results[label] = {"true": true_dims, "ss1": ss1_errs, "ss3": ss3_errs}
        print(f"  {label}: ss1 mean={np.nanmean(ss1_errs):.1f} ss3 mean={np.nanmean(ss3_errs):.1f}", flush=True)

    # --- Plot 1: error vs yaw ---
    n = len(results)
    fig, axes = plt.subplots(4, 3, figsize=(18, 22), constrained_layout=True)
    axes_flat = axes.flatten()
    for idx, (label, data) in enumerate(results.items()):
        ax = axes_flat[idx]
        ax.plot(YAWS, data["ss1"], "o-", label="v3 ss=1 (бинарная маска)", color="tab:blue")
        ax.plot(YAWS, data["ss3"], "s-", label="v3 ss=3 (субпиксель)", color="tab:red")
        td = data["true"]
        ax.set_title(f"{label} ({td[0]:.0f}×{td[1]:.0f}×{td[2]:.0f})")
        ax.set_xlabel("yaw°")
        ax.set_ylabel("max err %")
        ax.legend()
        all_errs = [e for e in data["ss1"] + data["ss3"] if not np.isnan(e)]
        ax.set_ylim(0, max(max(all_errs) * 1.2, 10) if all_errs else 10)
        ax.grid(True, alpha=0.3)
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle(
        "Error vs Yaw — v3 supersample=1 vs 3 (боевая камера 2592×1944, FOV 68°, 2м, зеркала 2.6м)",
        fontsize=16,
    )
    fig.savefig(PLOT_DIR / "error_vs_yaw_supersample.png", dpi=150)
    plt.close(fig)
    print(f"Saved {PLOT_DIR / 'error_vs_yaw_supersample.png'}", flush=True)

    # --- Plot 2: scatter ss1 vs ss3 ---
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    for label, data in results.items():
        ax3.scatter(data["ss1"], data["ss3"], label=label, alpha=0.6, s=30)
    lim = max(
        [e for d in results.values() for e in d["ss1"] + d["ss3"] if not np.isnan(e)] + [10]
    ) * 1.1
    ax3.plot([0, lim], [0, lim], "k--", alpha=0.5, label="ss1=ss3")
    ax3.set_xlabel("v3 supersample=1 error %")
    ax3.set_ylabel("v3 supersample=3 error %")
    ax3.set_title(f"v3: без субпикселя vs с субпикселем — yaw sweep ({len(YAWS)} per object)")
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.legend(fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.3)
    fig3.savefig(PLOT_DIR / "scatter_v3ss1_vs_v3ss3.png", dpi=150)
    plt.close(fig3)
    print(f"Saved {PLOT_DIR / 'scatter_v3ss1_vs_v3ss3.png'}", flush=True)

    # --- Summary table ---
    print(f"\n{'Object':<22} {'ss1 mean':>9} {'ss3 mean':>9} {'ss1 med':>8} {'ss3 med':>8} {'ss1<ss3':>7} {'ss1>ss3':>7}")
    print("-" * 78)
    for label, data in results.items():
        a1 = np.array(data["ss1"])
        a3 = np.array(data["ss3"])
        print(f"{label:<22} {np.nanmean(a1):>9.1f} {np.nanmean(a3):>9.1f} "
              f"{np.nanmedian(a1):>8.1f} {np.nanmedian(a3):>8.1f} "
              f"{int((a1 < a3).sum()):>7} {int((a3 < a1).sum()):>7}")


if __name__ == "__main__":
    main()
