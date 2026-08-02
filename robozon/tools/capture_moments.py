#!/usr/bin/env python3
"""Тестовый прогон + численная сверка детектора моментов старт/центр/конец
(Фаза 2.2 CV-пайплайна, заметка задачи "Детектор моментов старт-центр-конец
по CV-камерам").

Спавнит по одному объекту каждого типа (`rotation=identity`, как и
`tools/report_cv_masks.py` — детерминированная поза для воспроизводимой
сверки), ждёт 9 автоматических захватов (`sim/cv_moments.py` + Sorter в
`supervisor_main.py`: top×3 старт/центр/конец, side×3, diag×1 центр,
top_mirror×1 центр, diag_mirror×1 центр — зеркальные подключены Фазой 3),
скачивает кадры+маски через `GET /api/cv/moment_frame|moment_mask/<uid>/
<camera>/<moment>` и строит отчёт:

1. Полнота: ровно 9 моментов на объект (не меньше/не больше — Приёмка п.4,
   двойные/пропущенные срабатывания недопустимы).
2. Точность независимой (БЕЗ физики) оценки X по маске
   (`estimate_x_from_mask`) против фактической физической X, залогированной
   сервером В МОМЕНТ РЕАЛЬНОГО saveImage (событие `cv_moment` в
   `/api/status`, НЕ порог-цель — см. docstring
   `Sorter._advance_moment_capture`) — отклонение в мм, порог по умолчанию
   `--tolerance-mm`.
3. Сверка "касания края кадра" по маске (`mask_touches_row_edge`) — ожидаемо
   True на start/end (объект частично обрезан по построению окна), ожидаемо
   False на center (объект должен быть виден целиком) — расхождение с
   ожиданием сигнализирует, что окна старт/центр/конец подобраны неточно.

Пример:
    python3 tools/capture_moments.py --out /tmp/cv_moments --port 8008
"""
import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.config import load_layout, load_objects        # noqa: E402
from sim.cv_moments import (                              # noqa: E402
    estimate_x_from_mask, mask_touches_row_edge, moment_thresholds, threshold_captures,
)
from tools.capture_cv_frames import api_post, api_status, spawn_and_wait  # noqa: E402

EXPECTED_COUNT = 9   # top×3 + side×3 + diag×1 + top_mirror×1 + diag_mirror×1
                     # (Фаза 3 подключила зеркальные камеры к моменту "центр")


def _fetch(port: int, url_path: str) -> bytes | None:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{url_path}", timeout=10) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def wait_for_moments(port: int, uid: int, expected: int = EXPECTED_COUNT,
                      timeout: float = 15.0) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    """Опрашивает `/api/status` events, пока не наберётся `expected` разных
    (camera, moment) c событием `cv_moment` для данного uid, либо не истечёт
    таймаут (тогда возвращает то, что успело накопиться — недостача сама по
    себе диагностический результат, не ошибка скрипта). Значение —
    (x_cv, x_physics): `x_cv` — X, реально управлявшая триггером (CV-триггер
    по top-локализации, см. заметку задачи "CV-триггер момента по
    top-локализации..."), `x_physics` — физическая X В ТОТ ЖЕ МОМЕНТ
    (ground truth, доступна только в симуляторе) — для сверки, насколько
    CV-триггер отклоняется от точного физического порога."""
    seen: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        for e in api_status(port)["events"]:
            if e.get("event") == "cv_moment" and e.get("uid") == uid:
                seen[(e["camera"], e["moment"])] = (e["x"], e.get("x_physics"))
        if len(seen) >= expected:
            break
        time.sleep(0.05)
    if len(seen) >= expected:
        # Гонка: событие `cv_moment` логируется СИНХРОННО в момент постановки
        # JPEG-задания в очередь (`_moment_jpeg_jobs.put`), а само кодирование
        # +запись на диск — в фоновом потоке (`_jpeg_writer_loop`,
        # ~0.1-0.17с/кадр, см. её docstring в supervisor_main.py) — событие
        # НЕ гарантирует, что файл уже на диске. Живой прогон вскрыл реальное
        # проявление: `/api/cv/moment_grid_v2/<uid>`, запрошенный сразу после
        # этой функции, читал ЧУЖОЙ/устаревший `moment_<uid>_<camera>_<moment>.jpg`
        # (тот же uid переиспользуется в КАЖДОМ новом запуске Webots — не
        # обновлённый файл предыдущей сессии на диске ещё не был перезаписан
        # фоновым потоком) — композит сетки показывал не тот объект. Запас
        # с большим многократным резервом над задокументированной задержкой.
        time.sleep(0.5)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--out", required=True, help="папка отчёта (JSON + txt + кадры/маски)")
    ap.add_argument("--types", default=None, help="подмножество типов через запятую (по умолчанию все)")
    ap.add_argument("--tolerance-mm-center", type=float, default=20.0,
                    help="порог отклонения на моменте 'центр' (объект целиком в кадре, "
                         "ортографическое приближение точнее всего — на реальном прогоне "
                         "box_small даёт ~6мм на всех 3 прямых камерах)")
    ap.add_argument("--tolerance-mm-edge", type=float, default=250.0,
                    help="порог отклонения на старте/конце — заведомо мягче: окна delta_max "
                         "посчитаны под НАИХУДШИЙ (макс. по ТЗ, 450x320x320мм) габарит, "
                         "поэтому для меньших объектов старт/конец физически МЕНЬШЕ обрезаны, "
                         "чем расчётный запас, а сам центроид маски на частично обрезанном "
                         "силуэте — смещённая оценка истинного центра объекта не по вине кода "
                         "(см. Риски в заметке задачи). Ловит грубые провалы (порядок величины), "
                         "не сертифицирует точность на этих двух моментах")
    ap.add_argument("--tolerance-mm-cv-vs-physics-center", type=float, default=20.0,
                    help="допуск между X, реально управлявшей CV-триггером (top-локализация), "
                         "и физической X В ТОТ ЖЕ МОМЕНТ, на моменте 'центр' (Приёмка задачи "
                         "'CV-триггер момента по top-локализации...', п.1)")
    ap.add_argument("--tolerance-mm-cv-vs-physics-edge", type=float, default=250.0,
                    help="тот же допуск на старте/конце — заведомо мягче, по ТОЙ ЖЕ причине, что "
                         "--tolerance-mm-edge (см. её help): объект в кадре top ЧАСТИЧНО обрезан "
                         "у границы FOV, центроид ВИДИМОЙ части систематически смещён внутрь кадра "
                         "относительно истинного центра объекта — измерено живым прогоном "
                         "(box_small на входе: физика x=1.0, детектируемый bbox уже касается края "
                         "кадра, world_x по центроиду=1.37) — тот же класс погрешности, что для "
                         "estimate_x_from_mask на этих же двух моментах, не баг CV-триггера")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout()
    objects_cfg = load_objects()
    rig = layout["cv_rig"]
    cams = rig["cameras"]
    mirror_cameras = {name for name, cfg in cams.items() if cfg.get("mirror")}
    edge_tol_px = rig["moment_detection"]["mask_edge_tolerance_px"]
    thresholds = moment_thresholds(rig)
    captures = threshold_captures()
    # (camera, moment) -> имя порога (для "ожидаемой X по физике", информационно)
    expected_threshold_x = {pair: thresholds[name] for name, pairs in captures.items() for pair in pairs}

    types = args.types.split(",") if args.types else list(objects_cfg)

    # Автозахват выключен по умолчанию (см. Api.moments_enabled в
    # supervisor_main.py — обычная работа стенда не должна платить за
    # захват/JPEG-запись, которыми никто не пользуется без этого скрипта).
    # Включаем на время прогона, гарантированно выключаем после (finally) —
    # не оставляем стенд в "тестовом" режиме, если пользователь потом просто
    # продолжит смотреть симуляцию.
    api_post(args.port, "cv/moments_enabled", {"enabled": True})
    results = []
    try:
        for obj_type in types:
            obj_dir = out_dir / obj_type
            obj_dir.mkdir(exist_ok=True)

            api_post(args.port, "reset")
            time.sleep(1)
            try:
                uid = spawn_and_wait(args.port, obj_type, rotation="identity")
            except RuntimeError as exc:
                results.append({"type": obj_type, "error": str(exc)})
                print(f"{obj_type}: {exc}", file=sys.stderr)
                continue

            seen = wait_for_moments(args.port, uid)
            count_ok = len(seen) == EXPECTED_COUNT
            if not count_ok:
                missing = sorted(set(expected_threshold_x) - set(seen))
                extra = sorted(set(seen) - set(expected_threshold_x))
                results.append({"type": obj_type, "uid": uid, "verdict": "count_mismatch",
                                "count": len(seen), "missing": [f"{c}_{m}" for c, m in missing],
                                "extra": [f"{c}_{m}" for c, m in extra]})

            for (camera, moment), (actual_x_cv, actual_x) in seen.items():
                frame_bytes = _fetch(args.port, f"/api/cv/moment_frame/{uid}/{camera}/{moment}")
                if frame_bytes:
                    (obj_dir / f"{camera}_{moment}.jpg").write_bytes(frame_bytes)
                entry = {
                    "type": obj_type, "uid": uid, "camera": camera, "moment": moment,
                    "expected_x_threshold": round(expected_threshold_x[(camera, moment)], 4),
                    "actual_x_physics": round(actual_x, 4) if actual_x is not None else None,
                    "actual_x_cv": round(actual_x_cv, 4) if actual_x_cv is not None else None,
                }
                if actual_x_cv is None or actual_x is None:
                    entry["cv_vs_physics_verdict"] = "no_data"
                else:
                    cv_dev_mm = (actual_x_cv - actual_x) * 1000.0
                    entry["cv_vs_physics_deviation_mm"] = round(cv_dev_mm, 1)
                    cv_tolerance_mm = (args.tolerance_mm_cv_vs_physics_center if moment == "center"
                                       else args.tolerance_mm_cv_vs_physics_edge)
                    entry["cv_vs_physics_verdict"] = (
                        "ok" if abs(cv_dev_mm) <= cv_tolerance_mm else "flag")
                if camera in mirror_cameras:
                    # Маска зеркальных камер — вне объёма Фазы 2.2
                    # (`GET /api/cv/moment_mask` намеренно 404 для них, см. её
                    # docstring в supervisor_main.py; `cams[camera]` у зеркал
                    # к тому же не имеет ключа "distance"). Здесь фиксируем
                    # только сам факт захвата (Приёмка п.4 — количество/
                    # порядок моментов), не численную точность по маске —
                    # это проверяет отдельно tools/assemble_grids.py (Фаза 3).
                    entry["verdict"] = "mirror_frame_only"
                    results.append(entry)
                    continue
                mask_bytes = _fetch(args.port, f"/api/cv/moment_mask/{uid}/{camera}/{moment}")
                if mask_bytes is None:
                    entry["verdict"] = "no_mask"
                    results.append(entry)
                    continue
                (obj_dir / f"{camera}_{moment}_mask.png").write_bytes(mask_bytes)
                mask = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L")) > 127

                touches_edge = mask_touches_row_edge(mask, edge_tol_px)
                entry["touches_edge"] = touches_edge
                # "Центр" обязан быть далеко от края НЕЗАВИСИМО от размера
                # объекта (окно посчитано под наихудший габарит ТЗ, у любого
                # меньшего объекта запас только больше) — универсальная
                # проверка. На старте/конце касание края ОЖИДАЕМО только для
                # объектов, близких к наихудшему габариту; для меньших (как
                # box_small) окно даёт заметный запас — отсутствие касания
                # там НЕ баг, см. --help --tolerance-mm-edge, поэтому здесь
                # не судим, только фиксируем.
                if moment == "center":
                    entry["center_unexpectedly_touches_edge"] = touches_edge

                est_x = estimate_x_from_mask(mask, cams[camera]["distance"], rig["fov_deg"], rig["x"])
                entry["estimated_x_from_mask"] = round(est_x, 4) if est_x is not None else None
                tolerance_mm = args.tolerance_mm_center if moment == "center" else args.tolerance_mm_edge
                if est_x is None or actual_x is None:
                    entry["verdict"] = "no_estimate"
                else:
                    dev_mm = (est_x - actual_x) * 1000.0
                    entry["deviation_mm"] = round(dev_mm, 1)
                    entry["verdict"] = "ok" if abs(dev_mm) <= tolerance_mm else "flag"
                results.append(entry)
    finally:
        api_post(args.port, "cv/moments_enabled", {"enabled": False})

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    flagged = [r for r in results
               if r.get("verdict") in ("flag", "no_mask", "no_estimate", "count_mismatch")
               or r.get("cv_vs_physics_verdict") in ("flag", "no_data")
               or "error" in r or r.get("center_unexpectedly_touches_edge")]
    lines = [f"Всего записей: {len(results)}, с проблемой: {len(flagged)}", ""]
    for r in results:
        if "error" in r:
            lines.append(f"{r['type']:15s} ERROR {r['error']}")
        elif r.get("verdict") == "count_mismatch":
            lines.append(f"{r['type']:15s} uid={r['uid']} COUNT_MISMATCH count={r['count']}/{EXPECTED_COUNT} "
                         f"missing={r['missing']} extra={r['extra']}")
        else:
            edge_flag = " CENTER_TOUCHES_EDGE" if r.get("center_unexpectedly_touches_edge") else ""
            lines.append(f"{r['type']:15s} {r['camera']:5s} {r['moment']:6s} "
                         f"{r['verdict'].upper():12s} expected_x={r['expected_x_threshold']} "
                         f"actual_x={r['actual_x_physics']} "
                         f"est_x={r.get('estimated_x_from_mask')} "
                         f"dev_mm={r.get('deviation_mm')} touches_edge={r.get('touches_edge')}{edge_flag} "
                         f"cv_x={r.get('actual_x_cv')} "
                         f"cv_vs_physics_mm={r.get('cv_vs_physics_deviation_mm')} "
                         f"[{r.get('cv_vs_physics_verdict', '?').upper()}]")
    txt_path = out_dir / "report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nотчёт -> {report_path}, {txt_path}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
