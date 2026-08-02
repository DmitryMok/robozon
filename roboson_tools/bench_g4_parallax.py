"""Бенчмарк G4 на 3D-реконструкции с параллаксом по ленте (Camera Mode).

Прогон `check_model_roll_g4` на 18 тестовых объектах (`assets/stl/` + `/tmp/dumbbell.stl`)
в трёх режимах силуэтов:

  1.ORTHO        — ортографические силуэты (baseline, уже проведённый бенчмарк G4).
  2. PERSP_CENTER — перспективные силуэты Camera Mode, `belt_position="center"`
                    (одна позиция на ленте, аналог одного кадра в центре зоны камер).
  3. PERSP_PARALLAX — перспективные силуэты с параллаксом: `belt_position ∈ {start, center,
                    end}` — три позиции, силуэты из каждой передаются в реконструкцию как
                    дополнительные ограничения (см. visual_hull/voxel_carving.py::carve — список
                    пар (view_angle_deg, belt_position); start/end несут реальную
                    дополнительную информацию о форме под перспективой — камера «заглядывает»
                    с торца, а не тот же кадр со сдвигом).

Запуск:
    /tmp/robozon-bench-venv/bin/python bench_g4_parallax.py [out_json]

По умолчанию отчёт сохраняется в `exports/g4_parallax_comparison.json`.
Параметры Camera Mode — из `config/app_settings.yaml` (camera_fov_deg) и пресета
«2 камеры + зеркало (дефолт)» из `config/camera_rig_presets.yaml` (camera_distances).

ВАЖНО: режим 3 (параллакс) требует передачи НЕСКОЛЬКИХ силуэтов на один азимут (start/center/
end). `reconstruct_shape_from_silhouettes` принимает `dict[float, Silhouette]` — один силуэт
на ключ, а ключ — азимут. Чтобы обойти это без правок в `core/experiment.py`, этот скрипт
работает следующим образом для режима 3:
  - для каждой позиции на ленте (start/center/end) ОТДЕЛЬНО строится набор силуэтов и
    запускается G4 (3 прогона × 4 силуэта), затем вердикты агрегируются по правилу:
      verdict = round       если хотя бы один из прогонов дал round
             = uncertain     иначе если хотя бы один дал uncertain
             = not_round     иначе
      k = max по трём прогонам
  Обоснование агрегации «max k» (а не «локальный поиск нашёл ось в одном из положений»): на
  реальном конвейере товар едет через зону камер, и каждый кадр — отдельное наблюдение; если
  в каком-то положении форма видится более круглой (k≥0.8), объект с этой осью качения
  действительно катится — это и есть смысл G4 (безопасная логика «лучше круглый»).
  Альтернатива (расширение `view_angles` до 12 с дробными ключами-азимутами) потребовала бы
  менять `reconstruct_shape_from_silhouettes`/`build_silhouette_set` — это правка рабочей
  реализации, явно отклонённая в постановке задачи.
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
    Orientation,
    check_model_roll_g4,
    ROLL_VERDICT_NOT_ROUND,
    ROLL_VERDICT_ROUND,
    ROLL_VERDICT_UNCERTAIN,
)
from roboson_tools.geometry.mesh_io import load_stl

ASSETS_STL_DIR = Path(__file__).resolve().parent / "assets" / "stl"
DUMBBELL_STL = Path("/tmp/dumbbell.stl")
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
DEFAULT_OUT = EXPORTS_DIR / "g4_parallax_comparison.json"

VIEW_ANGLES = [0, 45, 90, 135]
RESOLUTION_PX = 512
NUM_SLICES = 40
RUNS = 3  # меньше чем в bench_roll_methods.py (там 10) — G4 с fallback медленнее на перспективе

# Camera Mode — из config/app_settings.yaml + camera_rig_presets.yaml (пресет по умолчанию)
CAMERA_FOV_DEG = 68.0
CAMERA_DISTANCES = {0.0: 2000.0, 45.0: 2900.0, 90.0: 2000.0, 135.0: 4908.0}
BELT_POSITIONS_PARALLAX = ["start", "center", "end"]

# Эталон «круглый» — тот же, что в bench_roll_methods.py (см. там обоснование)
GROUND_TRUTH = {
    "Пуфик": True,
    "Бутылка": True,
    "Тарелка": True,
    "Шлем": True,
    "Короб 300х200х200": False,
    "ЛанчБокс": False,
    "Моющее средство": False,
    "Ручка": False,
    "Мешок": True,
    "Короб 400х400х300": False,
    "Цилиндр": False,
    "Гантель_синтетическая": False,
    "asym_cone": True,
    "asym_barrel": True,
    "asym_cyl": True,
    # cyl_skewed: ПРАВКА 2026-07-31 — физически НЕ катится, эталон был ошибочным,
    # см. docs/method.md и bench_roll_methods.py.
    "cyl_skewed": False,
    "ell_cyl": True,
    "hourglass_asym": True,
}

_VERDICT_PRIORITY = {
    ROLL_VERDICT_ROUND: 2,
    ROLL_VERDICT_UNCERTAIN: 1,
    ROLL_VERDICT_NOT_ROUND: 0,
}


def _stl_files() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for p in sorted(ASSETS_STL_DIR.iterdir()):
        if p.suffix.lower() == ".stl":
            items.append((p.stem, p))
    if DUMBBELL_STL.exists():
        items.append(("Гантель_синтетическая", DUMBBELL_STL))
    return items


def _verdict_label(v: str) -> str:
    return {ROLL_VERDICT_ROUND: "round", ROLL_VERDICT_NOT_ROUND: "not_round",
            ROLL_VERDICT_UNCERTAIN: "uncertain"}[v]


def _aggregate_verdicts(verdicts: list[str], ks: list[float]) -> tuple[str, float]:
    """Логика «лучше круглый»: max по приоритету вердикта (round > uncertain > not_round),
    k = max по всем прогонам."""
    best_v = max(verdicts, key=lambda v: _VERDICT_PRIORITY[v])
    return best_v, max(ks)


def _run_g4(mesh, orientation, use_camera, belt_position, runs=RUNS):
    """Прогон G4 `runs` раз, возврат результата последнего + медиана elapsed_seconds."""
    last = None
    elapsed = []
    for _ in range(runs):
        last = check_model_roll_g4(
            mesh,
            orientation,
            VIEW_ANGLES,
            resolution_px=RESOLUTION_PX,
            num_slices=NUM_SLICES,
            use_camera_silhouettes=use_camera,
            camera_fov_deg=CAMERA_FOV_DEG if use_camera else 0.0,
            camera_distances=CAMERA_DISTANCES if use_camera else None,
            belt_position=belt_position,
        )
        elapsed.append(last.elapsed_seconds)
    return last, statistics.median(elapsed)


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 150)
    print("G4 на 3D-реконструкции с параллаксом по ленте (Camera Mode)")
    print(f"  view_angles={VIEW_ANGLES}  resolution_px={RESOLUTION_PX}  num_slices={NUM_SLICES}")
    print(f"  camera_fov_deg={CAMERA_FOV_DEG}  camera_distances={CAMERA_DISTANCES}")
    print(f"  belt_positions(parallax)={BELT_POSITIONS_PARALLAX}")
    print("=" * 150)
    header = (
        f"{'Объект':<24} {'эталон':<7} | "
        f"{'ORTHO verdict':<12} {'k':<6} {'fb':<3} {'t,мс':<6} | "
        f"{'PERSP_CENTER v':<15} {'k':<6} {'fb':<3} {'t,мс':<6} | "
        f"{'PERSP_PARALLAX v':<16} {'k':<6} {'fb':<3} {'t,мс':<6} | "
        f"{'совпадение':<10}"
    )
    print(header)
    print("-" * 150)

    rows = []
    n_match_ortho = n_match_center = n_match_parallax = 0
    n_total = 0

    for name, path in _stl_files():
        mesh = load_stl(path)
        orientation = Orientation(0.0, 0.0, 0.0)

        # --- Режим 1: ортографика (baseline) ---
        r_o, t_o = _run_g4(mesh, orientation, use_camera=False, belt_position="center")
        verdict_o = r_o.verdict
        k_o = r_o.best.k if r_o.best else 0.0
        fb_o = "+" if r_o.fallback_triggered else "-"

        # --- Режим 2: перспектива, center ---
        r_c, t_c = _run_g4(mesh, orientation, use_camera=True, belt_position="center")
        verdict_c = r_c.verdict
        k_c = r_c.best.k if r_c.best else 0.0
        fb_c = "+" if r_c.fallback_triggered else "-"

        # --- Режим 3: перспектива с параллаксом start/center/end ---
        # Отдельный прогон G4 на каждой позиции, агрегация «max k / round > uncertain > not_round»
        verdicts_p, ks_p, fbs_p, ts_p = [], [], [], []
        for bp in BELT_POSITIONS_PARALLAX:
            r_p, t_p = _run_g4(mesh, orientation, use_camera=True, belt_position=bp)
            verdicts_p.append(r_p.verdict)
            ks_p.append(r_p.best.k if r_p.best else 0.0)
            fbs_p.append("+" if r_p.fallback_triggered else "-")
            ts_p.append(t_p)
        verdict_p, k_p = _aggregate_verdicts(verdicts_p, ks_p)
        fb_p = "+any" if any(b == "+" for b in fbs_p) else "-"
        t_p = max(ts_p)  # худший случай по 3 прогонам

        # Сравнение с эталоном
        gt = GROUND_TRUTH.get(name)
        gt_label = "круглый" if gt else "не круг"
        gt_round = gt is True  # эталон-«катится»

        # G4 verdict: round → «катится», not_round/uncertain → «не подтверждён как катящийся»
        def _matches(verdict: str) -> bool:
            if gt is None:
                return True
            if gt_round:
                # эталон «катится»: round = OK, uncertain = OK (оператор), not_round = MISMATCH
                return verdict != ROLL_VERDICT_NOT_ROUND
            else:
                # эталон «не катится»: not_round/uncertain = OK, round = MISMATCH
                return verdict != ROLL_VERDICT_ROUND

        m_o = _matches(verdict_o)
        m_c = _matches(verdict_c)
        m_p = _matches(verdict_p)
        n_total += 1
        if m_o:
            n_match_ortho += 1
        if m_c:
            n_match_center += 1
        if m_p:
            n_match_parallax += 1

        match_label = f"o:{'+' if m_o else '-'} c:{'+' if m_c else '-'} p:{'+' if m_p else '-'}"

        print(
            f"{name:<24} {gt_label:<7} | "
            f"{_verdict_label(verdict_o):<12} {k_o:<6.3f} {fb_o:<3} {t_o*1000:<6.0f} | "
            f"{_verdict_label(verdict_c):<15} {k_c:<6.3f} {fb_c:<3} {t_c*1000:<6.0f} | "
            f"{_verdict_label(verdict_p):<16} {k_p:<6.3f} {fb_p:<4} {t_p*1000:<6.0f} | "
            f"{match_label:<10}"
        )

        rows.append({
            "object": name,
            "ground_truth_round": GROUND_TRUTH.get(name),
            "ortho": {
                "verdict": verdict_o, "k": k_o, "fallback": r_o.fallback_triggered,
                "elapsed_ms_median": t_o * 1000,
                "verdict_matches_ground_truth": m_o,
            },
            "persp_center": {
                "verdict": verdict_c, "k": k_c, "fallback": r_c.fallback_triggered,
                "elapsed_ms_median": t_c * 1000,
                "verdict_matches_ground_truth": m_c,
            },
            "persp_parallax": {
                "verdict_aggregated": verdict_p,
                "k_max": k_p,
                "per_position_verdicts": verdicts_p,
                "per_position_ks": ks_p,
                "per_position_fallbacks": fbs_p,
                "elapsed_ms_worst": t_p * 1000,
                "verdict_matches_ground_truth": m_p,
            },
        })

    print("-" * 150)
    print(
        f"Совпадений с эталоном (round→катится, not_round→не катится, uncertain=OK в обе стороны):"
    )
    print(f"  ORTHO        : {n_match_ortho}/{n_total}")
    print(f"  PERSP_CENTER : {n_match_center}/{n_total}")
    print(f"  PERSP_PARALLAX: {n_match_parallax}/{n_total}")

    # Дельты между режимами — где вердикт изменился
    print()
    print("Изменения вердикта между режимами:")
    for r in rows:
        v_o = r["ortho"]["verdict"]
        v_c = r["persp_center"]["verdict"]
        v_p = r["persp_parallax"]["verdict_aggregated"]
        if v_o != v_c or v_c != v_p or v_o != v_p:
            print(
                f"  {r['object']:<24}  ORTHO={_verdict_label(v_o):<10} "
                f"-> PERSP_CENTER={_verdict_label(v_c):<10} "
                f"-> PERSP_PARALLAX={_verdict_label(v_p)}"
            )

    report = {
        "config": {
            "view_angles": VIEW_ANGLES,
            "resolution_px": RESOLUTION_PX,
            "num_slices": NUM_SLICES,
            "camera_fov_deg": CAMERA_FOV_DEG,
            "camera_distances": CAMERA_DISTANCES,
            "belt_positions_parallax": BELT_POSITIONS_PARALLAX,
            "runs": RUNS,
        },
        "summary": {
            "ortho_matches_ground_truth": f"{n_match_ortho}/{n_total}",
            "persp_center_matches_ground_truth": f"{n_match_center}/{n_total}",
            "persp_parallax_matches_ground_truth": f"{n_match_parallax}/{n_total}",
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON-отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())