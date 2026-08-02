#!/usr/bin/env python3
"""Тестовый прогон сборки сетки 9 ракурсов (Фаза 3 CV-пайплайна, заметка
задачи "Сборка сетки 9 ракурсов из кадров Фазы 2"). По образцу
`tools/capture_moments.py` (Фаза 2.2), но проверяет саму сетку
(`sim/cv_grid.py`), не точность детектора моментов.

Спавнит по одному объекту каждого типа (`rotation=identity`), ждёт 9
автоматических захватов (Фаза 2.2/3), затем один раз запрашивает
`GET /api/cv/moment_grid/<uid>` (композит, лениво собирается сервером) и
`GET /api/cv/moment_grid_meta/<uid>` (метаданные ячеек), сохраняет их в
`--out/<type>/` и строит отчёт:

1. Полнота: метаданные содержат ровно 9 ячеек (те же ключи, что
   `sim/cv_grid.py::CELL_CAMERA_MOMENT`).
2. Зеркальные ячейки не перепутаны местами с прямыми — camera в метаданных
   каждой ячейки совпадает с ОЖИДАЕМОЙ по имени ячейки.
3. Источник bbox на ячейку (contour/fallback) и общий scale сетки —
   информационно, для ручной проверки крупности/консистентности.
4. Неожиданная обрезка на центральных ячейках — crop касается границы
   ПОЛНОГО кадра камеры (аналог `center_unexpectedly_touches_edge` в
   `capture_moments.py`, но по bbox кропа, не по маске целиком): на
   старте/конце это ожидаемо (торцевой ракурс), на центре — нет.

Пример:
    python3 tools/assemble_grids.py --out /tmp/cv_grids --port 8008
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cv_grid import CELL_CAMERA_MOMENT                            # noqa: E402
from sim.config import load_objects                                   # noqa: E402
from tools.capture_cv_frames import api_post, spawn_and_wait           # noqa: E402
from tools.capture_moments import EXPECTED_COUNT, wait_for_moments     # noqa: E402

EDGE_TOLERANCE_PX = 6   # тот же допуск, что moment_detection.mask_edge_tolerance_px по умолчанию


def _fetch(port: int, url_path: str) -> bytes | None:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{url_path}", timeout=15) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _touches_edge(cell: dict, tolerance_px: int = EDGE_TOLERANCE_PX) -> bool:
    x0, y0 = cell["crop_x"], cell["crop_y"]
    x1, y1 = x0 + cell["crop_side"], y0 + cell["crop_side"]
    return (x0 <= tolerance_px or y0 <= tolerance_px
            or x1 >= cell["frame_w"] - tolerance_px or y1 >= cell["frame_h"] - tolerance_px)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--out", required=True, help="папка отчёта (JSON + txt + композиты)")
    ap.add_argument("--types", default=None, help="подмножество типов через запятую (по умолчанию все)")
    ap.add_argument("--v2", action="store_true",
                    help="использовать Фазу 3.2 (endpoint moment_grid_v2, sim/cv_grid_v2.py — "
                         "top-локализация+проекция). По умолчанию Фаза 3.1 (moment_grid).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    objects_cfg = load_objects()
    types = args.types.split(",") if args.types else list(objects_cfg)

    # Тот же принцип, что capture_moments.py — автозахват выключен по
    # умолчанию (Api.moments_enabled), включаем на время прогона, гарантированно
    # выключаем после (finally).
    api_post(args.port, "cv/moments_enabled", {"enabled": True})
    results = []
    try:
        for obj_type in types:
            obj_dir = out_dir / obj_type
            obj_dir.mkdir(exist_ok=True)

            api_post(args.port, "reset")
            try:
                uid = spawn_and_wait(args.port, obj_type, rotation="identity")
            except RuntimeError as exc:
                results.append({"type": obj_type, "error": str(exc)})
                print(f"{obj_type}: {exc}", file=sys.stderr)
                continue

            seen = wait_for_moments(args.port, uid, expected=EXPECTED_COUNT)
            if len(seen) != EXPECTED_COUNT:
                results.append({"type": obj_type, "uid": uid, "verdict": "count_mismatch",
                                 "count": len(seen)})
                continue

            png_bytes = _fetch(args.port, f"/api/cv/moment_grid_v2/{uid}" if args.v2 else f"/api/cv/moment_grid/{uid}")
            meta_bytes = _fetch(args.port, f"/api/cv/moment_grid_v2_meta/{uid}" if args.v2 else f"/api/cv/moment_grid_meta/{uid}")
            if png_bytes is None or meta_bytes is None:
                results.append({"type": obj_type, "uid": uid, "verdict": "grid_not_built"})
                continue
            (obj_dir / "grid.png").write_bytes(png_bytes)
            (obj_dir / "grid.json").write_bytes(meta_bytes)
            meta = json.loads(meta_bytes)
            cells = meta["cells"]

            # Композит ФОНА (те же окна/масштаб) — только v2, нужен для
            # 3D-реконструкции (вычитание object-background даёт силуэт),
            # см. заметку задачи "3D-реконструкция объекта по кропам сетки
            # ракурсов...". 404 допустим (фон недоступен для этого набора) —
            # не считается ошибкой прогона.
            if args.v2:
                bg_bytes = _fetch(args.port, f"/api/cv/moment_grid_v2_bg/{uid}")
                if bg_bytes is not None:
                    (obj_dir / "grid_bg.png").write_bytes(bg_bytes)

            entry = {"type": obj_type, "uid": uid, "scale": meta["scale"],
                      "timings_s": meta.get("timings_s")}
            missing = sorted(set(CELL_CAMERA_MOMENT) - set(cells))
            extra = sorted(set(cells) - set(CELL_CAMERA_MOMENT))
            swapped = sorted(name for name, (camera, _moment) in CELL_CAMERA_MOMENT.items()
                              if name in cells and cells[name]["camera"] != camera)
            center_touches_edge = sorted(
                name for name, cell in cells.items()
                if name.startswith("center_") and _touches_edge(cell))
            fallback_cells = sorted(name for name, cell in cells.items()
                                     if cell.get("bbox_source") in ("fallback", "geometry"))

            entry["missing_cells"] = missing
            entry["extra_cells"] = extra
            entry["swapped_camera_cells"] = swapped
            entry["center_touches_edge"] = center_touches_edge
            entry["fallback_cells"] = fallback_cells
            entry["verdict"] = ("ok" if not (missing or extra or swapped or center_touches_edge)
                                 else "flag")
            results.append(entry)
    finally:
        api_post(args.port, "cv/moments_enabled", {"enabled": False})

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    flagged = [r for r in results if r.get("verdict") != "ok" or "error" in r]
    lines = [f"Всего записей: {len(results)}, с проблемой: {len(flagged)}", ""]
    for r in results:
        if "error" in r:
            lines.append(f"{r['type']:15s} ERROR {r['error']}")
        elif r.get("verdict") == "count_mismatch":
            lines.append(f"{r['type']:15s} uid={r['uid']} COUNT_MISMATCH count={r['count']}/{EXPECTED_COUNT}")
        elif r.get("verdict") == "grid_not_built":
            lines.append(f"{r['type']:15s} uid={r['uid']} GRID_NOT_BUILT")
        else:
            t = r.get("timings_s") or {}
            assemble_key = "assemble_grid_v2" if args.v2 else "assemble_grid"
            lines.append(f"{r['type']:15s} uid={r['uid']} {r['verdict'].upper():5s} "
                         f"scale={r['scale']:.3f} fallback={r['fallback_cells']} "
                         f"swapped={r['swapped_camera_cells']} "
                         f"center_touches_edge={r['center_touches_edge']} "
                         f"io={t.get('io_load')} assemble={t.get(assemble_key)} "
                         f"composite={t.get('composite_write')} total={t.get('total')}")

    # Тайминги сборки сетки (сервер считает их сам в _build_moment_grid, см.
    # supervisor_main.py) — агрегат по всем успешно собранным сеткам, тот же
    # принцип, что замеры Фазы 2.2 (compute_mask был причиной фризов): здесь
    # это НЕ горячий путь симуляции (лениво, в потоке HTTP-обработчика), но
    # знать порядок величины важно для бюджета задержки CV-конвейера (§1 П.6
    # плана, ~0.5с на весь конвейер после выхода из кадра).
    all_timings = [r["timings_s"] for r in results if r.get("timings_s")]
    if all_timings:
        lines.append("")
        assemble_key = "assemble_grid_v2" if args.v2 else "assemble_grid"
        for key in ("io_load", assemble_key, "composite_write", "total"):
            values = [t[key] for t in all_timings if key in t]
            if values:
                lines.append(f"timings[{key}]: min={min(values):.3f} "
                             f"mean={sum(values) / len(values):.3f} max={max(values):.3f} "
                             f"(n={len(values)})")

    txt_path = out_dir / "report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nотчёт -> {report_path}, {txt_path}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
