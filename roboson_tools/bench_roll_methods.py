"""Бенчмарк-сравнение двух методов поиска круглой проекции:
  1. Текущий — перебор направлений оси на полусфере (`check_model_roll`).
  2. Новый — PCA: 3 главные оси инерции (`check_model_roll_pca`).

Запуск:
    /tmp/robozon-bench-venv/bin/python bench_roll_methods.py [out_json]

По умолчанию таблица печатается в stdout, JSON-отчёт сохраняется в
`exports/roll_methods_comparison.json` (или путь из аргумента).

Тестовый набор: 11 STL из assets/stl + синтетическая «гантель» (круглый стержень с
квадратными торцами — ключевой пограничный случай, мотивировавший задачу).

Сравниваются:
  - точность (вердикт «катится», max k, найденная ось);
  - скорость этапа «поиск оси + расчёт k» — БЕЗ построения силуэтов и реконструкции
    формы (это общая часть, оба метода её разделяют; измеряется elapsed_seconds,
    который стартует после реконструкции формы — см. check_model_roll*).

Каждый метод прогоняется N раз (по умолчанию 10), берётся медиана elapsed_seconds,
чтобы сгладить шум одного прогона.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from roboson_tools.core.experiment import (
    AxisHit,
    Orientation,
    check_model_roll,
    check_model_roll_pca,
)
from roboson_tools.geometry.mesh_io import from_trimesh, load_stl

ASSETS_STL_DIR = Path(__file__).resolve().parent / "assets" / "stl"
DUMBBELL_STL = Path("/tmp/dumbbell.stl")
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
DEFAULT_OUT = EXPORTS_DIR / "roll_methods_comparison.json"

VIEW_ANGLES = [0, 45, 90, 135]
RESOLUTION_PX = 512
NUM_SLICES = 40
AXIS_STEP_DEG = 5.0  # боевой шаг перебора полусферы (как в docs/method.md)
RUNS = 10

# Эталон «круглый» из docs/method.md (таблица результатов) + синтетические объекты
# (asym_* — асимметричные катящиеся, мотивировавшие расширение тестового набора;
# помечены как «круглый» — физически катятся, но F1 может пропускать по k<0.8 на
# восстановленной из 4 силуэтов форме; см. заметку задачи «Сравнить PCA-критерий...»)
GROUND_TRUTH = {
    "Пуфик": True,
    "Бутылка": True,
    "Тарелка": True,
    "Шлем": True,
    "Короб 300х200х200": False,
    "Ланчбокс": False,
    "Моющее средство": False,
    "Ручка": False,
    "Мешок": True,  # * — форма честно проходит, вопрос бизнес-правила
    "Короб 400х400х300": False,  # почти-куб, формально проходит порог — свойство ТЗ
    "Цилиндр": False,  # * — НЕ тело вращения (гранёный), ложное срабатывание Visual Hull
    "Гантель_синтетическая": False,  # круглый стержень + квадратные торцы — НЕ катится
    # Синтетические асимметричные катящиеся (физически катятся):
    "asym_cone": True,        # конус + боковой куб — конус катится по кругу
    "asym_barrel": True,      # бочонок + асимметричный край — цилиндр, катится
    "asym_cyl": True,         # цилиндр + боковой груз — круглый стержень, катится
    # cyl_skewed: ПРАВКА 2026-07-31 (пользователь, визуальный осмотр) — физически НЕ
    # катится, эталон был ошибочным (см. robozon/config/objects.yaml: category=ok уже
    # давно верно). G4 (F1+fallback, CPU-путь) даёт для него round — известный
    # ложноположительный результат, не желаемое поведение, см. docs/method.md.
    "cyl_skewed": False,      # диагональный клин физически стопорит перекат
    "ell_cyl": True,          # эллиптический цилиндр — катится с раскачиванием
    "hourglass_asym": True,   # песочные часы + боковой куб — двойной конус, катится
}


def _stl_files() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for p in sorted(ASSETS_STL_DIR.iterdir()):
        if p.suffix.lower() == ".stl":
            items.append((p.stem, p))
    if DUMBBELL_STL.exists():
        items.append(("Гантель_синтетическая", DUMBBELL_STL))
    return items


def _azim_dist(a: float, target: float) -> float:
    d = abs((a - target) % 180.0)
    return min(d, 180.0 - d)


def _verdict_match(found_round: bool, name: str) -> tuple[bool, str]:
    gt = GROUND_TRUTH.get(name)
    if gt is None:
        return True, "—"
    return found_round == gt, "OK" if found_round == gt else "MISMATCH"


def _run_method(func, mesh, orientation, runs: int):
    """Прогон `runs` раз, возврат результата последнего прогона + медиана elapsed_seconds."""
    last = None
    elapsed = []
    for _ in range(runs):
        last = func(
            mesh,
            orientation,
            VIEW_ANGLES,
            resolution_px=RESOLUTION_PX,
            num_slices=NUM_SLICES,
        )
        elapsed.append(last.elapsed_seconds)
    return last, statistics.median(elapsed)


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(
        f"{'Объект':<24} {'эталон':<7} | "
        f"{'SCAN verdict':<13} {'k':<6} {'ось(аз/эл)':<14} {'t,мс':<7} | "
        f"{'PCA verdict':<12} {'k':<6} {'ось(аз/эл)':<14} {'t,мс':<7} | "
        f"{'match':<8} {'speedup':<7}"
    )
    print("-" * 140)

    n_match = 0
    n_total = 0
    for name, path in _stl_files():
        mesh = load_stl(path)
        orientation = Orientation(0.0, 0.0, 0.0)

        r_scan, t_scan = _run_method(
            lambda *a, **kw: check_model_roll(*a, **kw, axis_step_deg=AXIS_STEP_DEG),
            mesh,
            orientation,
            RUNS,
        )
        r_pca, t_pca = _run_method(check_model_roll_pca, mesh, orientation, RUNS)

        scan_axis = (
            f"{r_scan.best.axis_azim_deg:.0f}/{r_scan.best.axis_elev_deg:.0f}"
            if r_scan.best
            else "—"
        )
        pca_axis = (
            f"{r_pca.best.axis_azim_deg:.0f}/{r_pca.best.axis_elev_deg:.0f}"
            if r_pca.best
            else "—"
        )
        scan_k = f"{r_scan.best.k:.3f}" if r_scan.best else "—"
        pca_k = f"{r_pca.best.k:.3f}" if r_pca.best else "—"

        gt_label = "круглый" if GROUND_TRUTH.get(name) else "не круг"
        scan_verdict = "катится" if r_scan.found_round else "не катится"
        pca_verdict = "катится" if r_pca.found_round else "не катится"

        # Совпадение вердикта PCA с эталоном (SCAN — отдельной колонкой не сравниваем,
        # он уже зафиксирован в docs/method.md)
        pca_match, match_label = _verdict_match(r_pca.found_round, name)
        n_total += 1
        if pca_match:
            n_match += 1

        speedup = t_scan / t_pca if t_pca > 0 else float("inf")

        print(
            f"{name:<24} {gt_label:<7} | "
            f"{scan_verdict:<13} {scan_k:<6} {scan_axis:<14} {t_scan*1000:<7.1f} | "
            f"{pca_verdict:<12} {pca_k:<6} {pca_axis:<14} {t_pca*1000:<7.1f} | "
            f"{match_label:<8} {speedup:<7.1f}"
        )

        rows.append({
            "object": name,
            "ground_truth_round": GROUND_TRUTH.get(name),
            "scan": {
                "found_round": r_scan.found_round,
                "k": r_scan.best.k if r_scan.best else None,
                "axis_azim_deg": r_scan.best.axis_azim_deg if r_scan.best else None,
                "axis_elev_deg": r_scan.best.axis_elev_deg if r_scan.best else None,
                "elapsed_ms_median": t_scan * 1000,
                "verdict_matches_ground_truth": _verdict_match(r_scan.found_round, name)[0],
            },
            "pca": {
                "found_round": r_pca.found_round,
                "k": r_pca.best.k if r_pca.best else None,
                "axis_azim_deg": r_pca.best.axis_azim_deg if r_pca.best else None,
                "axis_elev_deg": r_pca.best.axis_elev_deg if r_pca.best else None,
                "elapsed_ms_median": t_pca * 1000,
                "verdict_matches_ground_truth": pca_match,
                "all_axis_hits": [
                    {"azim": h.axis_azim_deg, "elev": h.axis_elev_deg, "k": h.k}
                    for h in r_pca.axis_hits
                ],
            },
            "speedup_scan_over_pca": speedup,
        })

    print("-" * 140)
    print(f"PCA совпадений с эталоном: {n_match}/{n_total}")

    report = {
        "config": {
            "view_angles": VIEW_ANGLES,
            "resolution_px": RESOLUTION_PX,
            "num_slices": NUM_SLICES,
            "axis_step_deg": AXIS_STEP_DEG,
            "runs": RUNS,
        },
        "summary": {
            "pca_matches_ground_truth": f"{n_match}/{n_total}",
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON-отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())