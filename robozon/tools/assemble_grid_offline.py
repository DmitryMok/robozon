#!/usr/bin/env python3
"""Собирает сетку 9 ракурсов из уже СОХРАНЁННЫХ на диске кадров, БЕЗ живого
Webots — для быстрой итерации на `sim/cv_grid.py` (не гонять симулятор на
каждое изменение алгоритма кропа) и как основа для будущего исследования
скорости кропа (см. заметку задачи "Сборка сетки 9 ракурсов...", раздел про
оптимизацию `assemble_grid`).

Кадры — по той же конвенции имён, что `Sorter._advance_moment_capture`
(`webots/controllers/supervisor_main/supervisor_main.py`):
`moment_<uid>_<camera>_<moment>[_prev].jpg`, фон — `<camera>_background.jpg`.
Готовый тестовый набор (pen — маленький вытянутый объект, box_large — крупный
короб, физическая X при захвате не сохранена — не нужна для одиночного
объекта в кадре без диcambiguation) — `assets/cv_grid_test_frames/`.

Пример:
    python3 tools/assemble_grid_offline.py \\
        --frames-dir assets/cv_grid_test_frames/pen \\
        --backgrounds-dir assets/cv_grid_test_frames/backgrounds \\
        --uid 5 --out /tmp/pen_grid
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.config import load_layout                                            # noqa: E402
from sim.cv_grid import CELL_CAMERA_MOMENT, assemble_grid, cells_metadata, render_composite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True, help="папка с moment_<uid>_<camera>_<moment>[_prev].jpg")
    ap.add_argument("--backgrounds-dir", required=True, help="папка с <camera>_background.jpg")
    ap.add_argument("--uid", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    bg_dir = Path(args.backgrounds_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rig = load_layout()["cv_rig"]
    grid_cfg = rig["grid"]

    object_frames, prev_frames = {}, {}
    for cell_name, (camera, moment) in CELL_CAMERA_MOMENT.items():
        path = frames_dir / f"moment_{args.uid}_{camera}_{moment}.jpg"
        if not path.is_file():
            print(f"нет файла {path}", file=sys.stderr)
            return 1
        object_frames[cell_name] = cv2.imread(str(path))
        prev_path = frames_dir / f"moment_{args.uid}_{camera}_{moment}_prev.jpg"
        if prev_path.is_file():
            prev_frames[cell_name] = cv2.imread(str(prev_path))

    background_frames = {}
    for camera in {cam for cam, _m in CELL_CAMERA_MOMENT.values()}:
        bg_path = bg_dir / f"{camera}_background.jpg"
        if bg_path.is_file():
            background_frames[camera] = cv2.imread(str(bg_path))

    t0 = time.perf_counter()
    result = assemble_grid(object_frames, prev_frames, background_frames, {}, rig,
                            grid_cfg["cell_px"], grid_cfg["sam3_max_dim"], grid_cfg["crop_margin_factor"])
    elapsed = time.perf_counter() - t0

    cv2.imwrite(str(out_dir / "grid.png"), render_composite(result.cells))
    meta = {"scale": result.scale, "assemble_grid_s": round(elapsed, 4),
             "cells": cells_metadata(result.cells)}
    (out_dir / "grid.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"assemble_grid: {elapsed:.3f}с -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
