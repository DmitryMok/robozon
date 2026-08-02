#!/usr/bin/env python3
"""Драйвер замера/отчёта для sim/cv_grid_v2.py::assemble_grid_v2 (Фаза 3.2).

Геометрия рига и сборщик — в sim/cv_grid_v2.py (переиспользуется
supervisor_main.py для endpoint /api/cv/moment_grid_v2/<uid>). Этот скрипт —
офлайн-замер на сохранённых кадрах (без живого Webots): грузит кадры в память
(I/O вне замера), прогоняет assemble_grid_v2 N раз, печатает тайминги и
метаданные ячеек.

Кадры (pen uid=5, box_large uid=6) — assets/cv_grid_test_frames/, фоны —
backgrounds/. I/O (imread) ВНЕ замера — кадры «только что из камеры».

Пример:
    python3 tools/experiment_grid_v2.py \\
        --frames-dir assets/cv_grid_test_frames/pen \\
        --backgrounds-dir assets/cv_grid_test_frames/backgrounds \\
        --uid 5 --out /tmp/pen_grid_v2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.config import load_layout  # noqa: E402
from sim.cv_grid import (  # noqa: E402
    CELL_CAMERA_MOMENT, CELL_NAMES, render_composite, cells_metadata,
    _downscale_bbox_fast, expected_pixel_row,
)
from sim.cv_grid_v2 import assemble_grid_v2, build_camera_geom, project_point  # noqa: E402


def _build_target_hint_from_ground_truth(frames_dir: Path, uid: int,
                                          rig: dict) -> dict | None:
    """Строит target_hint (cell_name -> (row, col)) из ground-truth X целевого
    объекта в каждом моменте: читает GT-файл per-события
    moment_<uid>_<th_name>_objects.json (th_name = start_top/start_side/
    center/end_side/end_top — уникален per-события, в отличие от moment "end",
    который снимается двумя событиями при РАЗНЫХ X: end_side=2.868, end_top=
    3.426; старый общий GT момент "end" перезаписывался последним событием и
    давал неверный X для side-кадра). Берёт X объекта с target_uid, считает
    expected_row (expected_pixel_row) и столбец = проекция (x, 0, 0) в кадр
    камеры. Для визуальной проверки сетки кропов на многообъектных кадрах —
    имитирует прод-источник target_hint (top-локализация + проекция), но из
    ground-truth. None если GT-файлы не найдены."""
    geoms = {name: build_camera_geom(name, rig)
             for name in {cam for cam, _m in CELL_CAMERA_MOMENT.values()}}
    # th_name -> (camera, moment) обратное отображение для CELL_CAMERA_MOMENT.
    # cell_name = th_name для top-ячеек (start_top/center_top/end_top), но
    # center_side/center_diag/center_top_mirror/center_diag_mirror — все
    # момент "center", одно событие "center" (5 камер одновременно, один GT).
    # Для start/end: th_name = start_top/start_side/end_side/end_top — каждый
    # своё событие со своим GT. Mapping cell_name -> th_name:
    cell_to_th = {
        "start_top": "start_top", "start_side": "start_side",
        "center_top": "center", "center_side": "center", "center_diag": "center",
        "center_top_mirror": "center", "center_diag_mirror": "center",
        "end_side": "end_side", "end_top": "end_top",
    }
    hint: dict[str, tuple[float, float]] = {}
    cv_rig_x = rig["x"]
    for cell_name, (camera, moment) in CELL_CAMERA_MOMENT.items():
        th_name = cell_to_th[cell_name]
        # Приоритет: per-th_name GT (точный X на момент снятия этой камеры);
        # fallback — старый moment GT (для старых данных без per-th_name).
        gt_path = frames_dir / f"moment_{uid}_{th_name}_objects.json"
        if not gt_path.is_file():
            gt_path = frames_dir / f"moment_{uid}_{moment}_objects.json"
            if not gt_path.is_file():
                return None
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        target_uid = gt.get("target_uid", uid)
        target_x = None
        for o in gt["objects"]:
            if o["uid"] == target_uid:
                target_x = o["x"]; break
        if target_x is None:
            return None
        # row: expected_pixel_row (обратная к estimate_x_from_mask, геометрия рига)
        g = geoms[camera]
        dist = float(__import__("numpy").linalg.norm(g.t))
        row = expected_pixel_row(target_x, cv_rig_x, dist, rig["fov_deg"], g.n_rows, g.n_cols)
        # col: проекция точки контакта (target_x - cv_rig_x, 0, 0) в кадр камеры.
        # top: col ~ центр (Y=0); боковые: тоже ~центр (объект по центру ленты).
        rig_pt = __import__("numpy").array([target_x - cv_rig_x, 0.0, 0.0])
        col, _ = project_point(rig_pt, g)
        if not __import__("math").isfinite(col):
            col = g.n_cols / 2.0
        hint[cell_name] = (float(row), float(col))
    return hint


def _load_frames_in_memory(frames_dir: Path, bg_dir: Path, uid: int
                            ) -> tuple[dict, dict, dict]:
    """Грузит кадры в память (ВНЕ замера)."""
    object_frames, prev_frames = {}, {}
    for cell_name, (camera, moment) in CELL_CAMERA_MOMENT.items():
        path = frames_dir / f"moment_{uid}_{camera}_{moment}.jpg"
        if path.is_file():
            object_frames[cell_name] = cv2.imread(str(path))
        prev_path = frames_dir / f"moment_{uid}_{camera}_{moment}_prev.jpg"
        if prev_path.is_file():
            prev_frames[cell_name] = cv2.imread(str(prev_path))
    background_frames = {}
    for camera in {cam for cam, _m in CELL_CAMERA_MOMENT.values()}:
        bg_path = bg_dir / f"{camera}_background.jpg"
        if bg_path.is_file():
            background_frames[camera] = cv2.imread(str(bg_path))
    return object_frames, prev_frames, background_frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--backgrounds-dir", required=True)
    ap.add_argument("--uid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=20,
                    help="число прогонов для усреднения (кадры в памяти)")
    ap.add_argument("--target-hint", action="store_true",
                    help="строить target_hint из ground-truth objects.json "
                         "(многообъектный кадр — сужение полосы + выбор цели)")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    bg_dir = Path(args.backgrounds_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rig = load_layout()["cv_rig"]
    grid_cfg = rig["grid"]

    # I/O вне замера
    object_frames, prev_frames, background_frames = _load_frames_in_memory(
        frames_dir, bg_dir, args.uid)

    # target_hint из ground-truth (многообъектный кадр)
    target_hint = None
    if args.target_hint:
        target_hint = _build_target_hint_from_ground_truth(frames_dir, args.uid, rig)
        if target_hint is None:
            print("WARN: ground-truth objects.json не найдены — target_hint=None",
                  file=sys.stderr)

    # один прогон для корректности + композит
    result = assemble_grid_v2(object_frames, prev_frames, background_frames, rig,
                               grid_cfg["cell_px"], grid_cfg["sam3_max_dim"],
                               grid_cfg["crop_margin_factor"],
                               target_hint=target_hint)
    grid_name = "grid_v2.png" if target_hint is None else "grid_v2_hint.png"
    cv2.imwrite(str(out_dir / grid_name), render_composite(result.cells))
    # Позы камер (R, t, K) в СК рига — для 3D-реконструкции вне сессии.
    # Каждая ячейка кропа ссылается на камеру по имени; здесь — полный набор
    # поз всех камер рига, чтобы visual hull / torchhull могли восстановить
    # K кропа (общий fx=fy + crop_x/crop_y/scale из cells_metadata) и позу.
    import numpy as _np
    geoms = {name: build_camera_geom(name, rig)
             for name in {cam for cam, _m in CELL_CAMERA_MOMENT.values()}}
    camera_poses = {}
    for name, g in geoms.items():
        camera_poses[name] = {
            "R": g.R.tolist(),
            "t": g.t.tolist(),
            "K": g.K.tolist(),
            "n_rows": g.n_rows, "n_cols": g.n_cols,
            "is_mirror": g.is_mirror,
            "distance_m": float(_np.linalg.norm(g.t)),
        }
    meta = {
        "scale": result.scale,
        "native_side": round(max(c.crop_side for c in result.cells.values()), 1),
        "cells": cells_metadata(result.cells),
        "camera_poses": camera_poses,
        "rig": {"x": rig["x"], "belt_z": rig.get("belt_z", 0.0),
                 "fov_deg": rig["fov_deg"],
                 "resolution": list(rig["resolution"])},
    }

    # замер: N прогонов, кадры в памяти
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        assemble_grid_v2(object_frames, prev_frames, background_frames, rig,
                          grid_cfg["cell_px"], grid_cfg["sam3_max_dim"],
                          grid_cfg["crop_margin_factor"],
                          target_hint=target_hint)
        times.append(time.perf_counter() - t0)
    meta["assemble_grid_v2_s"] = {
        "runs": args.runs,
        "min": round(min(times), 4),
        "mean": round(sum(times) / len(times), 4),
        "max": round(max(times), 4),
    }
    # отдельный замер ТОЛЬКО локализации по top (1 HSV)
    top_bg = background_frames.get("top")
    bs = rig["background_subtraction"]
    top_frame = object_frames["center_top"]
    t_loc = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _downscale_bbox_fast(top_frame, top_bg, bs)
        t_loc.append(time.perf_counter() - t0)
    meta["top_localization_s"] = {
        "runs": args.runs,
        "min": round(min(t_loc), 5),
        "mean": round(sum(t_loc) / len(t_loc), 5),
        "max": round(max(t_loc), 5),
    }
    (out_dir / "grid_v2.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"assemble_grid_v2: {meta['assemble_grid_v2_s']} -> {out_dir}")
    print(f"top_localization: {meta['top_localization_s']}")
    print(f"scale={result.scale:.4f} native_side={meta['native_side']}")
    for name in CELL_NAMES:
        c = result.cells[name]
        print(f"  {name:22s} src={c.bbox_source:8s} "
              f"crop=({c.crop_x:5d},{c.crop_y:5d}) side={c.crop_side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())