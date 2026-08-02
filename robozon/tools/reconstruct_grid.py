#!/usr/bin/env python3
"""3D-реконструкция объекта по кропам сетки 9 ракурсов (Фаза 5 CV-пайплайна,
заметка задачи "3D-реконструкция объекта по кропам сетки ракурсов (Фаза 5,
CV-пайплайн Webots)").

Офлайн-скрипт, НЕ требует запущенного Webots — работает по уже сохранённым
`assets/cv_grid_test_frames/grids_v2_cv_trigger/<type>/{grid.png,grid_bg.png,
grid.json}` (Фаза 3/3.1/4). Сама геометрия (риг камер, конвертация в
CameraPose, HSV/SAM3-сегментация, hull/G4) — в `sim/grid_reconstruction.py`
(вынесено оттуда при добавлении GUI-стороны "Загрузить сетку..." в соседнем
`roboson_tools`, чтобы не дублировать геометрию в двух местах — см. докстринг
того модуля). Здесь остаются только CLI-обвязка и отрисовка PNG-отчётов
(matplotlib) — не нужны GUI-диалогу, у него своя отрисовка через Panel 1/2.

Два параллельных пути сегментации на ячейку:
  - **A (HSV)** — классическая фоновая субтракция (`sim/cv_localization.py`,
    уже используется в Фазах 2-4), быстрый путь, ЗАМЕРЯЕТСЯ на бюджет
    конвейера (~0.5с суммарно, из них сборка сетки уже ~118мс).
  - **B (SAM3)** — через уже готовый мост `roboson_tools.segmentation.
    sam3_subprocess_backend.Sam3SubprocessBackend` (subprocess к
    UavVisionLab), диагностика ВНЕ бюджета — интересен вопрос, спасает ли
    SAM3 ячейки, где HSV даёт `bbox_source=="geometry"` (типовой случай —
    `cylinder`/`side`, низкий контраст с лентой). Включается `--sam3`.

Методы пересечения на пути A (HSV, всегда на бюджетных наблюдениях obs_a) —
**C (voxel)** всегда, **D (torchhull, GPU)** опционально через `--torchhull`
(задача "3D-реконструкция объекта по кропам сетки ракурсов", запрос
пользователя 2026-07-25) — требует torch+torchhull+CUDA, которых нет в
обычном Windows-венве `venv-roboson-tools`; запускать ЭТОТ скрипт с флагом
`--torchhull` нужно через WSL `.venv-cv`
(`/home/mdm3/Projects/robozon/.venv-cv/bin/python`, тот же венв, что уже
использует `roboson_tools/bench_torchhull.py`), например:
    /home/mdm3/Projects/robozon/.venv-cv/bin/python \\
        /mnt/c/Projects/exp-26/robozon/tools/reconstruct_grid.py \\
        --type pen --torchhull --out /tmp/recon
(путь к скрипту — Windows-копия репозитория через /mnt/c, а не WSL-копия
`/home/mdm3/Projects/robozon` — та вторичная, см. `01 Sources/03
Conventions.md` вики-проекта). Без `--torchhull` скрипт по-прежнему
одинаково работает из обоих окружений (Windows-венв достаточно).

Для каждого объекта сохраняются (пункты 1-2 запроса пользователя 2026-07-25,
раздельно по пути сегментации CV/SAM3 — легче визуально сравнивать):
  - `<type>_grid_seg_cv.png` / `_sam3.png` — лёгкая сетка кропов + контур
    маски, без 3D (быстрый просмотр качества сегментации).
  - `<type>_report_cv.png` / `_sam3.png` — та же сетка + рендер(ы)
    реконструкции (hull, габариты, G4-вердикт) для этого пути (CV — A+C(+D),
    SAM3 — B).

Пример:
    .venv/bin/python3 tools/reconstruct_grid.py --type pen,box_small,cylinder --out /tmp/recon
    .venv/bin/python3 tools/reconstruct_grid.py --all --sam3 --out /tmp/recon
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROBOZON_ROOT = Path(__file__).resolve().parents[1]
ROBOSON_TOOLS_ROOT = ROBOZON_ROOT.parent / "roboson_tools"
for _p in (ROBOZON_ROOT, ROBOSON_TOOLS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sim.config import load_layout, load_objects  # noqa: E402
from sim.cv_grid_v2 import CamGeom                  # noqa: E402
import sim.grid_reconstruction as grid_reconstruction  # noqa: E402
from sim.grid_reconstruction import (               # noqa: E402
    MASK_COLOR_CV, MASK_COLOR_SAM3, build_camera_geoms, build_observations,
    build_observations_perspective, expected_verdict,
    ground_truth_dims_mm, hsv_masks, reconstruct_path, reconstruct_path_torchhull,
    reconstruct_path_voxel, render_cells_mosaic, sam3_masks, voxel_bbox_from_polytope,
)

DATA_DIR = ROBOZON_ROOT / "assets" / "cv_grid_test_frames" / "grids_v2_cv_trigger"

_MASK_COLOR_A = MASK_COLOR_CV
_MASK_COLOR_B = MASK_COLOR_SAM3


# ---------------------------------------------------------------------------
# Отчёт (CLI-специфика: PNG-рендеры, не нужны GUI-диалогу roboson_tools)
# ---------------------------------------------------------------------------

def save_segmentation_grid(
    cells_meta: dict, grid_png: np.ndarray, masks: dict[str, np.ndarray],
    color: tuple[int, int, int], mask_label: str, tile_px: int, out_path: Path,
) -> None:
    """Отдельный лёгкий артефакт (только сетка + контур маски, БЕЗ 3D-
    реконструкции) — пункт 1 запроса пользователя (2026-07-25): "сохранение
    для сетки каждого объекта отдельной сетки с сегментациями". В отличие от
    `render_variant_report` не требует matplotlib/hull — можно быстро
    просмотреть качество сегментации на любом количестве объектов, не ждя
    построения visual hull/G4 на каждом. Полное разрешение тайла (не
    280px-превью мозаики), `cv2.imwrite` — на порядок быстрее matplotlib."""
    canvas = render_cells_mosaic(cells_meta, grid_png, masks, color, mask_label, tile_px, thumb_px=tile_px)
    # cv2.imwrite не умеет в Unicode-пути на Windows (молча возвращает False,
    # не бросает) — путь к 05 Visuals содержит кириллицу/пробелы, поэтому
    # кодируем в память и пишем через Path.write_bytes (умеет Unicode).
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"cv2.imencode не смог закодировать {out_path}")
    out_path.write_bytes(buf.tobytes())


def render_variant_report(
    obj_type: str, cells_meta: dict, grid_png: np.ndarray,
    masks: dict[str, np.ndarray], color: tuple[int, int, int], mask_label: str, tile_px: int,
    hull_paths: list[tuple[str, dict]], out_path: Path,
) -> None:
    """Один PNG на (объект, путь сегментации): сверху — сетка кропов+маски
    этого пути (см. `render_cells_mosaic`), снизу — реконструкция(и) на
    основе этого пути (`hull_paths` — список (заголовок, result-словарь из
    `reconstruct_path`/`reconstruct_path_voxel`), обычно 1-2 варианта метода
    intersection на ОДНОЙ и той же маске). Аспект осей 3D-графика делается
    РАВНЫМ реальным пропорциям (`set_box_aspect`) — иначе matplotlib
    растягивает оси под квадратную область по умолчанию, и даже корректный
    вытянутый объект визуально выглядит кубом/искажённым.

    Раздельный файл на путь (CV / SAM3), а не общий комбинированный, как
    раньше, — по решению пользователя (2026-07-25), для независимой
    визуальной проверки каждого варианта сегментации целиком (маска +
    результат), без необходимости сопоставлять два цвета контура на одной
    картинке."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

    mosaic = render_cells_mosaic(cells_meta, grid_png, masks, color, mask_label, tile_px, thumb_px=280)
    mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)

    n_cols = max(2, len(hull_paths))
    fig = plt.figure(figsize=(6 * n_cols, 12))
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.3, 1])

    ax_mosaic = fig.add_subplot(gs[0, :])
    ax_mosaic.imshow(mosaic_rgb)
    ax_mosaic.axis("off")
    ax_mosaic.set_title(f"{obj_type}: кропы + маска ({mask_label})")

    for i, (title, res) in enumerate(hull_paths):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        if "vertices" not in res and "points" not in res:
            ax.set_title(f"{title}: {res.get('error', 'нет данных')}")
            continue
        if "faces" in res:
            verts = res["vertices"]
            mesh_polys = verts[res["faces"]]
            collection = Poly3DCollection(mesh_polys, alpha=0.4, edgecolor="k", linewidths=0.2)
            collection.set_facecolor((0.3, 0.6, 0.9))
            ax.add_collection3d(collection)
        else:
            verts = res["points"]
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=4, alpha=0.5, color=(0.2, 0.4, 0.8))
        extent = verts.max(axis=0) - verts.min(axis=0)
        pad = np.maximum(extent * 0.02, 1e-3)  # защита от вырожденного (нулевого) диапазона оси
        ax.set_xlim(verts[:, 0].min() - pad[0], verts[:, 0].max() + pad[0])
        ax.set_ylim(verts[:, 1].min() - pad[1], verts[:, 1].max() + pad[1])
        ax.set_zlim(verts[:, 2].min() - pad[2], verts[:, 2].max() + pad[2])
        ax.set_box_aspect(tuple(max(e, 1e-6) for e in extent))  # реальные пропорции, не куб
        dims = res.get("recon_dims_mm")
        dims_txt = "x".join(f"{d:.0f}" for d in dims) if dims else "?"
        ax.set_title(f"{title}: {res.get('verdict')} (k={res.get('k')})\ndims={dims_txt}мм")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def process_object(
    obj_type: str, geoms: dict[str, CamGeom], rig_cfg: dict, objects_cfg: dict,
    out_dir: Path, sam3_backend=None, use_torchhull: bool = False,
) -> dict:
    obj_dir = DATA_DIR / obj_type
    with open(obj_dir / "grid.json", encoding="utf-8") as f:
        grid_meta = json.load(f)
    cells_meta = grid_meta["cells"]
    grid_png = cv2.imread(str(obj_dir / "grid.png"))
    bg_png = cv2.imread(str(obj_dir / "grid_bg.png"))
    any_meta = next(iter(cells_meta.values()))
    tile_px = round(any_meta["crop_side"] * any_meta["scale"])

    t0 = time.perf_counter()
    masks_a, dropped_a = hsv_masks(cells_meta, grid_png, bg_png, tile_px, rig_cfg)
    # Итеративная перспективная коррекция ракурса (z_eff=h/2) — устраняет
    # систематическую ошибку ~50мм на start/end для высоких объектов,
    # искажавшую visual hull (сужение верхней грани по X в 1.7x — найдено
    # при диагностике box_small). См. reconstruct_path_perspective docstring.
    # Возвращает УТОЧНЁННЫЕ observations — переиспользуем для voxel/torchhull,
    # не только для polytope.
    obs_a, _persp_debug = grid_reconstruction.build_observations_perspective(
        masks_a, cells_meta, geoms, rig_cfg, tile_px)
    result_a = grid_reconstruction.reconstruct_path(obs_a, "A_hsv", budget_start=t0, also_exact=True)
    result_a["perspective_debug"] = _persp_debug
    result_a["dropped_cells"] = dropped_a

    result_b = None
    masks_b = None
    if sam3_backend is not None:
        prompt = "object"
        masks_b, dropped_b, sam3_elapsed = sam3_masks(sam3_backend, cells_meta, grid_png, tile_px, prompt)
        obs_b, _ = grid_reconstruction.build_observations_perspective(
            masks_b, cells_meta, geoms, rig_cfg, tile_px)
        result_b = grid_reconstruction.reconstruct_path(obs_b, "B_sam3", budget_start=None, also_exact=True)
        result_b["dropped_cells"] = dropped_b
        result_b["sam3_segment_batch_s"] = sam3_elapsed
        result_b["rescued_cells"] = sorted(set(dropped_a) & set(masks_b))

    # Путь C — voxel carving на ТЕХ ЖЕ наблюдениях (obs_a, HSV), что путь A —
    # изолирует переменную "метод" от переменной "маска" (см. reconstruct_path_voxel).
    # bbox — из carve_polytope (result_a), расширенный x3 (не generic worst-case ТЗ,
    # см. docstring voxel_bbox_from_polytope) — даёт разрешение сетки, адаптированное
    # под реальный масштаб объекта (критично для тонких pen/cylinder).
    bbox_mm = voxel_bbox_from_polytope(result_a, rig_cfg, margin_ratio=1.0)
    result_c = reconstruct_path_voxel(obs_a, bbox_mm, "C_voxel_hsv", grid_resolution=64)

    # Путь D — torchhull (GPU), опционально (--torchhull, требует WSL .venv-cv, см. докстринг
    # модуля) — на ТЕХ ЖЕ наблюдениях, что путь A/C, только другой bbox-запас (10%, не 100% —
    # torchhull не имеет проблемы разрешения тонких объектов, которую решал большой запас у C).
    result_d = None
    if use_torchhull:
        bbox_d_mm = voxel_bbox_from_polytope(result_a, rig_cfg, margin_ratio=0.1)
        result_d = reconstruct_path_torchhull(obs_a, bbox_d_mm, "D_torchhull_hsv", level=7)

    gt_dims_mm = ground_truth_dims_mm(obj_type)
    category = objects_cfg[obj_type]["category"]
    expected = expected_verdict(category)

    # Пункт 1 запроса: отдельные лёгкие сетки с сегментацией (без hull).
    save_segmentation_grid(cells_meta, grid_png, masks_a, _MASK_COLOR_A, "CV/HSV", tile_px,
                            out_dir / f"{obj_type}_grid_seg_cv.png")
    if masks_b is not None:
        save_segmentation_grid(cells_meta, grid_png, masks_b, _MASK_COLOR_B, "SAM3", tile_px,
                                out_dir / f"{obj_type}_grid_seg_sam3.png")

    # Пункт 2 запроса: раздельные отчёты на путь сегментации (CV/SAM3) — по
    # решению пользователя 2026-07-25 (см. docstring render_variant_report).
    # Пути C (voxel) и D (torchhull) переиспользуют ту же CV/HSV-маску, что
    # путь A — идут в CV-вариант, у SAM3 их нет (не запрашивалось).
    cv_hull_paths = [("A: HSV (polytope)", result_a), ("C: HSV (voxel)", result_c)]
    if result_d is not None:
        cv_hull_paths.append(("D: HSV (torchhull)", result_d))
    render_variant_report(
        obj_type, cells_meta, grid_png, masks_a, _MASK_COLOR_A, "CV/HSV", tile_px,
        cv_hull_paths,
        out_dir / f"{obj_type}_report_cv.png",
    )
    if result_b is not None:
        render_variant_report(
            obj_type, cells_meta, grid_png, masks_b, _MASK_COLOR_B, "SAM3", tile_px,
            [("B: SAM3 (polytope)", result_b)],
            out_dir / f"{obj_type}_report_sam3.png",
        )

    def _strip(d):
        return {k: v for k, v in d.items() if k not in (
            "vertices", "faces", "points", "exact_vertices", "exact_faces")}

    report = {
        "type": obj_type,
        "category": category,
        "expected_verdict": expected,
        "ground_truth_dims_mm": gt_dims_mm,
        "path_a_hsv_polytope": _strip(result_a),
        "path_b_sam3_polytope": _strip(result_b) if result_b is not None else None,
        "path_c_hsv_voxel": _strip(result_c),
        "path_d_hsv_torchhull": _strip(result_d) if result_d is not None else None,
    }
    if "verdict" in result_a:
        report["verdict_ok_a"] = result_a["verdict"] in expected
    if "verdict" in result_c:
        report["verdict_ok_c"] = result_c["verdict"] in expected
    if result_d is not None and "verdict" in result_d:
        report["verdict_ok_d"] = result_d["verdict"] in expected
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", default="pen,box_small,cylinder",
                         help="список типов через запятую (см. config/objects.yaml)")
    parser.add_argument("--all", action="store_true", help="прогнать все типы, найденные в data_dir")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out", required=True, help="папка для отчёта (JSON+PNG)")
    parser.add_argument("--sam3", action="store_true", help="включить путь B (SAM3), вне бюджета")
    parser.add_argument("--torchhull", action="store_true",
                         help="включить путь D (GPU torchhull) — требует torch+torchhull+CUDA, "
                              "запускать из WSL .venv-cv, см. докстринг модуля")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        obj_types = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    else:
        obj_types = [t.strip() for t in args.type.split(",") if t.strip()]

    rig_cfg = load_layout()["cv_rig"]
    objects_cfg = load_objects()
    geoms = build_camera_geoms(rig_cfg)

    sam3_backend = None
    if args.sam3:
        from roboson_tools.segmentation.sam3_subprocess_backend import Sam3SubprocessBackend
        sam3_backend = Sam3SubprocessBackend()
        # Прогрев ОДИН раз до цикла по объектам — иначе таймер первого объекта в отчёте
        # включает загрузку TRT-модели на GPU (~15-20с), не реальную скорость сегментации
        # (см. Sam3SubprocessBackend.warmup, по замечанию пользователя).
        warmup_s = sam3_backend.warmup()
        print(f"SAM3: прогрев модели — {warmup_s:.1f}с (не учитывается в тайминге объектов)")

    reports = []
    try:
        for obj_type in obj_types:
            print(f"=== {obj_type} ===")
            try:
                report = process_object(obj_type, geoms, rig_cfg, objects_cfg, out_dir,
                                         sam3_backend, use_torchhull=args.torchhull)
                reports.append(report)
                skip_keys = ("path_a_hsv_polytope", "path_b_sam3_polytope", "path_c_hsv_voxel",
                             "path_d_hsv_torchhull")
                print(json.dumps(
                    {k: v for k, v in report.items() if k not in skip_keys},
                    ensure_ascii=False, indent=2))
                pa = report["path_a_hsv_polytope"]
                print(f"  A/HSV/polytope: n_obs={pa.get('n_observations')} verdict={pa.get('verdict')} "
                      f"k={pa.get('k')} dims={pa.get('recon_dims_mm')} "
                      f"budget_s={pa.get('elapsed_total_budget_s')} "
                      f"dropped={list(pa.get('dropped_cells', {}))}")
                if report["path_b_sam3_polytope"] is not None:
                    pb = report["path_b_sam3_polytope"]
                    print(f"  B/SAM3/polytope: n_obs={pb.get('n_observations')} verdict={pb.get('verdict')} "
                          f"k={pb.get('k')} dims={pb.get('recon_dims_mm')} "
                          f"segment_batch_s={pb.get('sam3_segment_batch_s')} "
                          f"rescued={pb.get('rescued_cells')}")
                pc = report["path_c_hsv_voxel"]
                print(f"  C/HSV/voxel: n_obs={pc.get('n_observations')} verdict={pc.get('verdict')} "
                      f"k={pc.get('k')} dims={pc.get('recon_dims_mm')} "
                      f"n_voxels={pc.get('n_voxels')} voxel_size_mm={pc.get('voxel_size_mm')} "
                      f"elapsed_total_s={pc.get('elapsed_total_s')}")
                if report["path_d_hsv_torchhull"] is not None:
                    pd = report["path_d_hsv_torchhull"]
                    print(f"  D/HSV/torchhull: n_obs={pd.get('n_observations')} verdict={pd.get('verdict')} "
                          f"k={pd.get('k')} dims={pd.get('recon_dims_mm')} "
                          f"n_points={pd.get('n_points')} elapsed_total_s={pd.get('elapsed_total_s')}")
            except Exception:
                print(f"!!! {obj_type}: сбой, продолжаю со следующим объектом")
                traceback.print_exc()
                reports.append({"type": obj_type, "error": traceback.format_exc()})
    finally:
        if sam3_backend is not None:
            sam3_backend.shutdown()

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
