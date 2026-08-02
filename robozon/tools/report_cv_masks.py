#!/usr/bin/env python3
"""Численная проверка масок фоновой субтракции (Фаза 2.1 CV-пайплайна,
заметка задачи "Фоновая субтракция и маска объекта на CV-камерах").

Прогоняет весь тестовый набор `config/objects.yaml` через живую симуляцию
(сначала запустить `scripts/run.ps1`/`run.sh --headless`), для каждого типа и
каждой ПРЯМОЙ камеры (top/diag/side — зеркальные вне Фазы 2.1) сравнивает
bbox маски (переведённый из пикселей в мм через известную геометрию камеры и
дистанцию наблюдения, см. `camera_scale_and_axes`) с истинными габаритами
объекта из конфига. Приближение "известная дистанция + FOV, без перспективного
искажения" оправдано геометрией рига (объект много меньше дистанции наблюдения,
узкий FOV 68°) — не полноценная проекция вершин через позу камеры.

Спавн — с `rotation=identity` (см. `supervisor_main.py::try_spawn`), НЕ
случайная ориентация как в обычной работе конвейера: истинные габариты объекта
без этого сравнивать не с чем (случайный разворот меняет видимый габарит
непредсказуемо). Отчёт (JSON + txt) сохраняется в `--out`.

Ожидаемое систематическое отклонение (обычно 5-25%, измеренное МЕНЬШЕ
номинала): осознанный компромисс порогов `h_threshold`/`s_threshold`
(`config/layout.yaml: cv_rig.background_subtraction`) в пользу устойчивости к
тени объекта — часть мягкого края/полутени у контура остаётся ниже порога.
Проверено эмпирически (`_diag_thresholds.py`, не входит в поставку): ослабление
порогов не даёт плавного восстановления кромки — за узким "обрывом" мгновенно
затягивает тень объекта в маску (даёт ПЕРЕОЦЕНКУ и ложный контур), то есть
текущие пороги — не непротестированная заглушка, а разумная точка на кривой
"недооценка силуэта" vs "ложный контур от тени". Отклонение сильнее для
маленьких/светлых (низкий контраст к ленте) объектов и на `side`-камере для
катящихся форм (объект физически докатывается/переворачивается на другую
грань к моменту cv_rig.x — это `verdict=flag`, а не баг вычисления). Отчёт
предназначен ловить ГРУБЫЕ провалы (no_mask, на порядок неверный габарит,
не тот объект), а не сертифицировать точность до единиц процентов.

Пример:
    python3 tools/report_cv_masks.py --out /tmp/cv_report --port 8008
"""
import argparse
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.config import load_layout, load_objects        # noqa: E402
from tools.capture_cv_frames import (                     # noqa: E402
    api_post, save_camera, spawn_and_wait,
)
from tools.gen_world import _camera_frame                 # noqa: E402


def canonical_points(bnd: dict) -> list[tuple[float, float, float]]:
    """Опорные точки boundingObject в локальной (без поворота) системе — та же
    геометрия, что `supervisor_main.py::_canonical_points`, продублирована
    здесь намеренно: этот скрипт — офлайн HTTP-клиент и не должен тянуть
    Webots `controller` (импортируется в supervisor_main.py на верхнем
    уровне модуля, недоступен вне запущенного Webots-процесса)."""
    if bnd["type"] == "box":
        sx, sy, sz = bnd["size"]
        return [(xs, ys, zs)
                for xs in (-sx / 2, sx / 2)
                for ys in (-sy / 2, sy / 2)
                for zs in (0, sz)]
    if bnd["type"] == "cylinder":
        r, h, axis = bnd["radius"], bnd["height"], bnd.get("axis", "z")
        n = 16
        rim = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
               for i in range(n)]
        if axis == "z":
            return [(cx, cy, zs) for cx, cy in rim for zs in (0, h)]
        if axis == "y":
            return [(cx, ys, r + cz) for cx, cz in rim for ys in (-h / 2, h / 2)]
        return [(xs, cx, r + cz) for cx, cz in rim for xs in (-h / 2, h / 2)]
    raise ValueError(f"неизвестный bounding: {bnd['type']}")


def camera_scale_and_axes(cam: dict, rig: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale_mm_per_px, right, up) для прямой камеры — `right`/`up` в мировых
    координатах (см. `tools/gen_world.py::_camera_frame`: `up`=(1,0,0) для
    ВСЕХ камер рига, вдоль ленты, `right`=-`left`). `resolution` в конфиге —
    [n_rows (вдоль ленты, ось `up`), n_cols (поперёк, ось `right`)]; FOV
    (`fov_deg`) в Webots действует по оси WIDTH (n_cols), см. config/layout.yaml
    cv_rig docstring — f_px выводится через n_cols (квадратные пиксели)."""
    theta = math.radians(cam["angle"])
    _forward, left, up = _camera_frame(theta)
    right = -left
    n_rows, n_cols = cam.get("resolution", rig["resolution"])
    fov = math.radians(cam.get("fov_deg", rig["fov_deg"]))
    f_px = (n_cols / 2.0) / math.tan(fov / 2.0)
    scale_mm_per_px = (cam["distance"] / f_px) * 1000.0
    return scale_mm_per_px, right, up


def expected_extents_mm(bnd: dict, right: np.ndarray, up: np.ndarray) -> tuple[float, float]:
    """Ожидаемые (cols, rows) габариты объекта в мм — проекция каноничных
    точек на оси кадра (ортографическое приближение, см. модуль docstring)."""
    points = canonical_points(bnd)
    cols = [float(np.dot(p, right)) for p in points]
    rows = [float(np.dot(p, up)) for p in points]
    return (max(cols) - min(cols)) * 1000.0, (max(rows) - min(rows)) * 1000.0


def mask_bbox_px(mask_png: bytes) -> tuple[int, int] | None:
    img = Image.open(io.BytesIO(mask_png)).convert("L")
    arr = np.array(img)
    ys, xs = np.nonzero(arr > 127)
    if len(xs) == 0:
        return None
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def fetch_mask(port: int, cam_name: str) -> bytes:
    with urllib.request.urlopen(f"http://localhost:{port}/api/cv/mask/{cam_name}", timeout=10) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--out", required=True, help="папка отчёта (JSON + txt + отладочные кадры)")
    ap.add_argument("--types", default=None, help="подмножество типов через запятую (по умолчанию все)")
    ap.add_argument("--tolerance-pct", type=float, default=30.0,
                    help="порог отклонения по каждой оси для вердикта ok, %% "
                         "(30% по умолчанию — см. docstring про ожидаемую недооценку "
                         "силуэта из-за консервативных порогов против тени)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout()
    objects_cfg = load_objects()
    rig = layout["cv_rig"]
    direct_cams = {n: c for n, c in rig["cameras"].items() if not c.get("mirror")}

    types = args.types.split(",") if args.types else list(objects_cfg)

    results = []
    for obj_type in types:
        cfg = objects_cfg[obj_type]
        bnd = cfg["bounding"]
        obj_dir = out_dir / obj_type
        obj_dir.mkdir(exist_ok=True)

        api_post(args.port, "reset")
        time.sleep(1)
        try:
            spawn_and_wait(args.port, obj_type, rotation="identity")
        except RuntimeError as exc:
            results.append({"type": obj_type, "error": str(exc)})
            print(f"{obj_type}: {exc}", file=sys.stderr)
            continue

        for cam_name, cam in direct_cams.items():
            save_camera(args.port, cam_name, obj_dir)
            try:
                mask_bytes = fetch_mask(args.port, cam_name)
            except Exception as exc:
                results.append({"type": obj_type, "camera": cam_name, "error": str(exc)})
                continue
            (obj_dir / f"{cam_name}_mask.png").write_bytes(mask_bytes)

            bbox_px = mask_bbox_px(mask_bytes)
            scale, right, up = camera_scale_and_axes(cam, rig)
            exp_cols_mm, exp_rows_mm = expected_extents_mm(bnd, right, up)
            if bbox_px is None:
                results.append({"type": obj_type, "camera": cam_name, "verdict": "no_mask",
                                "expected_mm": [round(exp_cols_mm, 1), round(exp_rows_mm, 1)]})
                continue

            w_px, h_px = bbox_px
            meas_cols_mm, meas_rows_mm = w_px * scale, h_px * scale
            dev_cols = (meas_cols_mm - exp_cols_mm) / exp_cols_mm * 100
            dev_rows = (meas_rows_mm - exp_rows_mm) / exp_rows_mm * 100
            verdict = "ok" if max(abs(dev_cols), abs(dev_rows)) <= args.tolerance_pct else "flag"
            results.append({
                "type": obj_type, "camera": cam_name, "verdict": verdict,
                "expected_mm": [round(exp_cols_mm, 1), round(exp_rows_mm, 1)],
                "measured_mm": [round(meas_cols_mm, 1), round(meas_rows_mm, 1)],
                "deviation_pct": [round(dev_cols, 1), round(dev_rows, 1)],
            })

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    flagged = [r for r in results if r.get("verdict") in ("flag", "no_mask") or "error" in r]
    lines = [f"Всего проверок: {len(results)}, вне порога/без маски/ошибка: {len(flagged)}", ""]
    for r in results:
        if "error" in r:
            lines.append(f"{r['type']:15s} {r.get('camera', '-'):6s} ERROR {r['error']}")
        elif r["verdict"] == "no_mask":
            lines.append(f"{r['type']:15s} {r['camera']:6s} NO_MASK  expected={r['expected_mm']}")
        else:
            lines.append(f"{r['type']:15s} {r['camera']:6s} {r['verdict'].upper():5s} "
                         f"expected={r['expected_mm']} measured={r['measured_mm']} "
                         f"dev%={r['deviation_pct']}")
    txt_path = out_dir / "report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nотчёт -> {report_path}, {txt_path}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
