"""Визуальный анализ разброса ошибки v1 vs v3 по разным ориентациям.

Для каждого STL строит график: ошибка (%) по оси Y, ориентация (yaw) по оси X,
отдельные линии для v1 и v3. Сохраняет PNG в bench_plots/.

Usage:
    python bench_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboson_tools.core.experiment import (
    Orientation,
    check_simple_camera_dims,
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


def main():
    PLOT_DIR.mkdir(exist_ok=True)

    seen = set()
    stl_files = []
    for p in sorted(ASSETS_DIR.iterdir()):
        if p.suffix.lower() == ".stl" and p.stem.lower() not in seen:
            seen.add(p.stem.lower())
            stl_files.append(p)

    # Один проход: собираем все данные
    results = {}  # label -> {"true": dims, "v1": [errs], "v3": [errs]}
    for stl_path in stl_files:
        label = _SIZE_TO_NAME.get(stl_path.stat().st_size, stl_path.stem)
        mesh = load_stl(stl_path)
        true_dims = mesh_true_dims(mesh, axis_step_deg=5.0)
        if true_dims is None:
            continue

        v1_errs = []
        v3_errs = []
        for yaw in YAWS:
            ori = Orientation(yaw_deg=yaw)
            v1 = check_simple_camera_dims(
                mesh, ori, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
                top_angle_deg=TOP_ANGLE, side_angle_deg=0.0,
            )
            v3 = check_simple_camera_dims_v3(
                mesh, ori, FOV_DEG, CAMERA_DISTANCE, RESOLUTION_PX,
                top_angle_deg=TOP_ANGLE, side_angles_deg=VIEW_ANGLES,
                belt_positions=BELT_POSITIONS,
            )
            e1 = max_abs_err_pct([v1.length, v1.width, v1.height], true_dims) if v1 else float("nan")
            e3 = max_abs_err_pct([v3.length, v3.width, v3.height], true_dims) if v3 else float("nan")
            v1_errs.append(e1)
            v3_errs.append(e3)

        results[label] = {"true": true_dims, "v1": v1_errs, "v3": v3_errs}
        print(f"  {label}: v1 mean={np.nanmean(v1_errs):.1f} v3 mean={np.nanmean(v3_errs):.1f}", flush=True)

    # --- Plot 1: error vs yaw ---
    n = len(results)
    fig, axes = plt.subplots(4, 3, figsize=(18, 22), constrained_layout=True)
    axes_flat = axes.flatten()
    for idx, (label, data) in enumerate(results.items()):
        ax = axes_flat[idx]
        ax.plot(YAWS, data["v1"], "o-", label="v1", color="tab:blue")
        ax.plot(YAWS, data["v3"], "s-", label="v3", color="tab:red")
        td = data["true"]
        ax.set_title(f"{label} ({td[0]:.0f}×{td[1]:.0f}×{td[2]:.0f})")
        ax.set_xlabel("yaw°")
        ax.set_ylabel("max err %")
        ax.legend()
        all_errs = [e for e in data["v1"] + data["v3"] if not np.isnan(e)]
        ax.set_ylim(0, max(max(all_errs) * 1.2, 20) if all_errs else 20)
        ax.grid(True, alpha=0.3)
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle("Error vs Yaw (roll=0, pitch=0) — v1 vs v3", fontsize=16)
    fig.savefig(PLOT_DIR / "error_vs_yaw.png", dpi=150)
    plt.close(fig)
    print(f"Saved {PLOT_DIR / 'error_vs_yaw.png'}", flush=True)

    # --- Plot 2: scatter v1 vs v3 ---
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    for label, data in results.items():
        ax3.scatter(data["v1"], data["v3"], label=label, alpha=0.6, s=30)
    lim = 40
    ax3.plot([0, lim], [0, lim], "k--", alpha=0.5, label="v1=v3")
    ax3.set_xlabel("v1 error %")
    ax3.set_ylabel("v3 error %")
    ax3.set_title(f"v1 vs v3 — yaw sweep ({len(YAWS)} per object)")
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.legend(fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.3)
    fig3.savefig(PLOT_DIR / "scatter_v1_vs_v3.png", dpi=150)
    plt.close(fig3)
    print(f"Saved {PLOT_DIR / 'scatter_v1_vs_v3.png'}", flush=True)

    # --- Summary table ---
    print(f"\n{'Object':<22} {'v1 mean':>8} {'v3 mean':>8} {'v1 med':>8} {'v3 med':>8} {'v1<v3':>6} {'v1>v3':>6}")
    print("-" * 75)
    for label, data in results.items():
        v1_arr = np.array(data["v1"])
        v3_arr = np.array(data["v3"])
        v1_better = int((v1_arr < v3_arr).sum())
        v3_better = int((v3_arr < v1_arr).sum())
        print(f"{label:<22} {np.nanmean(v1_arr):>8.1f} {np.nanmean(v3_arr):>8.1f} "
              f"{np.nanmedian(v1_arr):>8.1f} {np.nanmedian(v3_arr):>8.1f} "
              f"{v1_better:>6} {v3_better:>6}")


if __name__ == "__main__":
    main()