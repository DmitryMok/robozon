"""Движок 3D-реконструкции объекта по кропам сетки 9 ракурсов (Фаза 5
CV-пайплайна) — чистые функции без CLI/matplotlib, чтобы их мог
переиспользовать не только `tools/reconstruct_grid.py` (офлайн-скрипт), но и
GUI-диалог "Загрузить сетку..." в соседнем проекте `roboson_tools`
(`gui/dialogs/grid_reconstruction_dialog.py`, задача "3D-реконструкция
объекта по кропам сетки ракурсов", пункт 3 запроса пользователя 2026-07-25).

Извлечено из `tools/reconstruct_grid.py` при добавлении GUI-стороны — раньше
вся геометрия дублировалась бы в двух местах (Webots-риг + позы камер под
кроп сетки — источник уже трёх реальных багов в этой задаче, см. заметку),
теперь один источник истины.

Использует visual-hull реконструкцию соседнего проекта `roboson_tools`
(`pose_carving.carve_polytope`/`carve_exact`/`carve_voxels` — CPU, работают
в обычном Windows-венве `venv-roboson-tools`). Опционально — GPU-путь
`reconstruct_path_torchhull` (`roboson_tools.visual_hull.torchhull_adapter`,
задача "3D-реконструкция объекта по кропам сетки ракурсов", по запросу
пользователя 2026-07-25): `torchhull_adapter.build_transform_matrix` из
коробки не поддерживает сдвиг главной точки (жёстко центрирует cx/cy), а у
нас каждая из 9 ячеек сетки имеет свой crop_x/crop_y — поэтому K@T строится
вручную из уже готового `CameraPose` (та же геометрия, что у polytope/
exact/voxel, см. `_camera_pose_to_torchhull_transform`), в обход
`build_transform_matrix`. Требует torch+torchhull+CUDA — в Windows-венве
`venv-roboson-tools` их нет, запускать этот путь нужно через WSL
`.venv-cv` (`/home/mdm3/Projects/robozon/.venv-cv`, см. `roboson_tools/
bench_torchhull.py`) — `roboson_tools.visual_hull.torchhull_adapter`
импортируется ЛЕНИВО, внутри функции, а не на уровне модуля, чтобы CLI/GUI
на Windows-венве (без torch) продолжали работать без него.

Модуль сам добавляет соседний репозиторий `roboson_tools` в `sys.path`,
если его там ещё нет (симметрично тому, как `roboson_tools` добавляет
`robozon` при использовании этого модуля из GUI, см. диалог выше) —
работает независимо от того, кто импортирует первым.

Геометрия камер — `sim/cv_grid_v2.py::build_camera_geom` (риг Webots,
локальные оси X=вперёд/Y=влево/Z=вверх, метры) конвертируется в OpenCV-
конвенцию `roboson_tools.pose.camera_pose.CameraPose` (мировые мм — своя
локальная система, начало = точка наблюдения рига; абсолютные мировые
координаты Webots не нужны, все финальные метрики ротационно/трансляционно
инвариантны). ВАЖНО про зеркальные камеры (`top_mirror`/`diag_mirror`):
`_flip_mirror_frame` в `supervisor_main.py` физически переворачивает
сохранённый JPEG по вертикали ДО того, как его читает что-либо ещё — эта
компенсация (`down=+up`, `cy_native = n_rows-1-K[1,2]` вместо обычных
`down=-up`, `cy_native=K[1,2]`) НЕ отменяется тем, что 3D-поза зеркальной
камеры в `CamGeom` уже "прямая" (без runtime reflect) — это два независимых
факта. Без поправки конусы зеркальных камер целятся на ~200+px мимо цели
(численно проверено на этапе планирования).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROBOZON_ROOT = Path(__file__).resolve().parents[1]
ROBOSON_TOOLS_ROOT = ROBOZON_ROOT.parent / "roboson_tools"
for _p in (ROBOZON_ROOT, ROBOSON_TOOLS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sim.cv_grid import CELL_CAMERA_MOMENT, _COMPOSITE_LAYOUT  # noqa: E402
from sim.cv_grid_v2 import CamGeom, build_camera_geom          # noqa: E402
from sim.cv_localization import build_background_model, compute_mask  # noqa: E402

from roboson_tools.core.experiment import (  # noqa: E402
    _g4_core_from_points, _search_min_bounding_box, mesh_true_dims,
)
from roboson_tools.geometry.mesh_io import Mesh, load_stl  # noqa: E402
from roboson_tools.pose.camera_pose import CameraIntrinsics, CameraPose  # noqa: E402
from roboson_tools.visual_hull.pose_carving import (  # noqa: E402
    Observation, carve_exact, carve_polytope, carve_voxels,
)

# torchhull — ОПЦИОНАЛЬНАЯ GPU-зависимость (CUDA Toolkit + WSL .venv-cv, см. докстринг модуля
# и reconstruct_path_torchhull ниже). Тот же приём мягкого импорта, что уже используется в
# roboson_tools/gui/main_window.py для STL/Camera Mode — вызывающая сторона (CLI, GUI-диалог
# "Загрузить сетку...") проверяет этот флаг, чтобы показывать/скрывать путь D торчхалла, не
# падая там, где torch/torchhull не установлены (обычный Windows-венв venv-roboson-tools).
TORCHHULL_AVAILABLE = False
try:
    import torch as _torch  # noqa: F401
    import torchhull as _torchhull  # noqa: F401
    TORCHHULL_AVAILABLE = True
except Exception:
    pass

MESHES_DIR = ROBOZON_ROOT / "assets" / "meshes"

G4_KWARGS = dict(
    resolution_px=1024,
    roundness_threshold=0.8,
    low_threshold=0.65,
    sweep_resolution_px=512,
    local_search_radius_deg=20.0,
    local_search_step_deg=5.0,
)


# ---------------------------------------------------------------------------
# Геометрия: нарезка тайлов из композита, поза камеры на ячейку
# ---------------------------------------------------------------------------

def _cell_row_col(cell_name: str) -> tuple[int, int]:
    for ri, row in enumerate(_COMPOSITE_LAYOUT):
        if cell_name in row:
            return ri, row.index(cell_name)
    raise KeyError(f"неизвестная ячейка {cell_name!r}")


def slice_tile(composite: np.ndarray, cell_name: str, tile_px: int, pad: int = 4) -> np.ndarray:
    """Тайл `tile_px`x`tile_px` для `cell_name` из композита `render_composite`
    (`sim/cv_grid.py`) — та же геометрия раскладки (приватная в `cv_grid.py`,
    инлайнится здесь, а не импортируется)."""
    ri, ci = _cell_row_col(cell_name)
    y0 = ri * (tile_px + pad) + pad
    x0 = ci * (tile_px + pad) + pad
    return composite[y0:y0 + tile_px, x0:x0 + tile_px]


MASK_COLOR_CV = (0, 200, 0)      # HSV — зелёный (BGR)
MASK_COLOR_SAM3 = (200, 0, 200)  # SAM3 — пурпурный (BGR)


def render_cells_mosaic(
    cells_meta: dict, grid_png: np.ndarray, masks: dict[str, np.ndarray],
    color: tuple[int, int, int], mask_label: str, tile_px: int, thumb_px: int | None = None,
) -> np.ndarray:
    """Мозаика 3x3 (порядок `_COMPOSITE_LAYOUT`, как в самом `grid.png`) —
    на каждой ячейке её кроп с наложенным контуром ОДНОЙ маски (`masks`,
    один путь сегментации — CV или SAM3), подпись cell_name + bbox_source +
    что отброшено. `thumb_px=None` — нативный размер тайла (без даунскейла,
    для сохранения на диск CLI-скриптом); меньшее значение — превью в GUI.

    Вынесено из `tools/reconstruct_grid.py` при добавлении GUI-диалога
    "Загрузить сетку..." (`roboson_tools`) — тот же рендер маски+сетки
    нужен и там (предпросмотр после клика «Сегментировать»), не только в
    CLI-отчётах. По решению пользователя (2026-07-25) — раздельная мозаика
    на путь (не одна с двумя наложенными цветами): проще визуально
    сравнивать сегментацию и реконструкцию одного и того же пути между
    собой, не путая контуры двух методов на одном кропе."""
    if thumb_px is None:
        thumb_px = tile_px
    pad = 6
    n_rows, n_cols = len(_COMPOSITE_LAYOUT), len(_COMPOSITE_LAYOUT[0])
    canvas = np.full((n_rows * (thumb_px + pad) + pad, n_cols * (thumb_px + pad) + pad, 3),
                      40, dtype=np.uint8)
    for ri, row in enumerate(_COMPOSITE_LAYOUT):
        for ci, cell_name in enumerate(row):
            if cell_name not in cells_meta:
                continue
            tile = slice_tile(grid_png, cell_name, tile_px).copy()
            status_bits = [cells_meta[cell_name]["bbox_source"]]
            if cell_name in masks:
                contours, _ = cv2.findContours(
                    masks[cell_name].astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tile, contours, -1, color, max(2, tile_px // 200))
            else:
                status_bits.append(f"{mask_label}:нет маски")
            thumb = cv2.resize(tile, (thumb_px, thumb_px), interpolation=cv2.INTER_AREA)
            y0 = ri * (thumb_px + pad) + pad
            x0 = ci * (thumb_px + pad) + pad
            canvas[y0:y0 + thumb_px, x0:x0 + thumb_px] = thumb
            label = f"{cell_name} [{','.join(status_bits)}]"
            cv2.putText(canvas, label, (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (255, 255, 0), 1, cv2.LINE_AA)
    return canvas


def build_camera_geoms(rig_cfg: dict) -> dict[str, CamGeom]:
    camera_names = {cam for cam, _moment in CELL_CAMERA_MOMENT.values()}
    return {name: build_camera_geom(name, rig_cfg) for name in camera_names}


def build_cell_pose(
    cam_name: str, g: CamGeom, rig_cfg: dict,
    crop_x: float, crop_y: float, cell_scale: float, cell_px: int,
    x_shift_m: float = 0.0,
) -> CameraPose:
    """CamGeom (Webots-риг, м) -> CameraPose (OpenCV-конвенция, мм) для
    КОНКРЕТНОЙ ячейки сетки (со сдвигом главной точки под crop_x/crop_y и
    масштабом cell_scale). См. докстринг модуля про зеркальную поправку.

    `x_shift_m` — сдвиг СК рига вдоль X (вдоль ленты) под момент этой ячейки
    (см. `measured_shift_m`): 9 ячеек сетки сняты в РАЗНЫЕ физические
    моменты (объект физически движется между start/center/end — до 5 разных
    X, не 3, т.к. `start_top`/`start_side` и `end_top`/`end_side` —
    независимые события с разным `delta_max`, см. `sim/cv_moments.py`),
    поэтому позиции камер в фиксированной СК рига нельзя напрямую
    пересекать одним visual hull — нужно выразить их в СК, где объект
    приблизительно в начале координат независимо от момента. `center`
    (5 ячеек) — уже в начале координат по построению (момент = пересечение
    `cv_rig.x`), `x_shift_m≈0`."""
    forward, left, up = g.R[0], g.R[1], g.R[2]
    right = -left
    is_mirror = bool(rig_cfg["cameras"][cam_name].get("mirror"))
    if is_mirror:
        down = up
        cy_native = g.n_rows - 1.0 - g.K[1, 2]
    else:
        down = -up
        cy_native = g.K[1, 2]

    rotation_cv = np.stack([right, down, forward], axis=0)
    camera_pos_shifted = g.t - np.array([x_shift_m, 0.0, 0.0])
    camera_pos_mm = camera_pos_shifted * 1000.0
    translation_cv = -rotation_cv @ camera_pos_mm

    f_px = g.K[0, 0]
    fx = fy = f_px * cell_scale
    cx = (g.K[0, 2] - crop_x) * cell_scale
    cy = (cy_native - crop_y) * cell_scale

    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, resolution_px=(cell_px, cell_px))
    return CameraPose(rotation=rotation_cv, translation=translation_cv, intrinsics=intrinsics)


# ---------------------------------------------------------------------------
# Путь A: HSV-маски
# ---------------------------------------------------------------------------

def hsv_masks(
    cells_meta: dict, grid_png: np.ndarray, bg_png: np.ndarray, tile_px: int, rig_cfg: dict,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Возвращает (маски по уцелевшим ячейкам, причина исключения по остальным)."""
    bs = rig_cfg["background_subtraction"]
    masks: dict[str, np.ndarray] = {}
    dropped: dict[str, str] = {}
    for cell_name, meta in cells_meta.items():
        if meta["bbox_source"] == "geometry":
            dropped[cell_name] = "bbox_source=geometry"
            continue
        if not meta.get("has_background", True):
            dropped[cell_name] = "no_background"
            continue
        obj_tile = slice_tile(grid_png, cell_name, tile_px)
        bg_tile = slice_tile(bg_png, cell_name, tile_px)
        bg_model = build_background_model(bg_tile)
        scale = meta["scale"]
        min_area_px = bs["min_area_px"] * (scale ** 2)
        morph_kernel = max(1, round(bs["morph_kernel"] * scale))
        if morph_kernel % 2 == 0:
            morph_kernel += 1
        mask, bbox = compute_mask(
            obj_tile, bg_model, bs["h_threshold"], bs["s_threshold"],
            min_area_px, morph_kernel, v_threshold=bs.get("v_threshold"),
        )
        if bbox is None:
            dropped[cell_name] = "no_foreground"
            continue
        masks[cell_name] = mask.astype(bool)
    return masks, dropped


# ---------------------------------------------------------------------------
# Путь B: SAM3 (опционально, вне бюджета)
# ---------------------------------------------------------------------------

def sam3_masks(
    backend, cells_meta: dict, grid_png: np.ndarray, tile_px: int, prompt: str,
) -> tuple[dict[str, np.ndarray], dict[str, str], float]:
    """SAM3 гоняется на ВСЕХ 9 тайлах (включая bbox_source=='geometry') —
    именно это интересно сравнить с путём A."""
    cell_names = list(cells_meta.keys())
    tiles = [slice_tile(grid_png, name, tile_px) for name in cell_names]
    t0 = time.perf_counter()
    results = backend.segment_batch(tiles, prompt=prompt)
    elapsed = time.perf_counter() - t0
    masks: dict[str, np.ndarray] = {}
    dropped: dict[str, str] = {}
    for cell_name, result in zip(cell_names, results):
        if result is None:
            dropped[cell_name] = "sam3_no_detection"
            continue
        masks[cell_name] = result.mask.astype(bool)
    return masks, dropped, elapsed


# ---------------------------------------------------------------------------
# Общее: наблюдения -> hull -> габариты -> G4
# ---------------------------------------------------------------------------

def measured_shift_m(meta: dict, cam_geom: CamGeom, z_eff_m: float = 0.0) -> float:
    """Сдвиг СК рига вдоль X под момент ячейки — см. докстринг `build_cell_pose`.

    ИЗ РЕАЛЬНО ИЗМЕРЕННОЙ позиции объекта в этот момент (центр bbox, той же
    детекции, что уже дала сам кроп — `crop_y`+`crop_side`/2), а не из
    номинального `sim.cv_moments.moment_thresholds` (геометрический ХУДШИЙ
    СЛУЧАЙ порога момента, не реальная точка срабатывания конкретного
    объекта). Обратная формула к `sim/cv_grid.py::expected_pixel_row` —
    строка кодирует X ОДИНАКОВО для ВСЕХ камер рига (`up=(1,0,0)`, см.
    `sim/cv_moments.py::mask_row_bbox` докстринг), поэтому применима не
    только к top.

    Найдено эмпирически на этапе отладки задачи: наивная версия (сдвиг по
    номинальному порогу момента) давала систематическое рассогласование до
    ~200-250мм на `start_top`/`end_top` — тот же эффект, что уже
    задокументирован в задаче "CV-триггер момента по top-локализации..."
    (Фаза 4, реальный CV-триггер срабатывает не строго в номинальной
    геометрической точке) — из-за этого `carve_polytope` получал
    геометрически несовместные конусы и возвращал `None` на ВСЕХ 3 тестовых
    объектах. Измеренный сдвиг проверен диагностикой (мировое начало СК,
    приблизительно центр объекта в момент "центр") против полупространств
    каждого наблюдения — 0 нарушений на всех ячейках/объектах после фикса.

    `z_eff_m` — перспективная коррекция: bbox-центр маски соответствует
    z≈h/2 (середина высоты объекта), а не z=0 (уровень ленты). Формула:
    `x = -(row - n_rows/2) * (dist - z_eff) / f` вместо ортографической
    `x = -(row - n_rows/2) * dist / f`. Для center (объект в cv_rig.x,
    row≈n_rows/2) коррекция пренебрежимо мала; для start/end (объект
    сдвинут на ~0.8м) даёт поправку ~50мм, ранее искажавшую visual hull
    (сужение верхней грани по X в 1.7x — найдено при диагностике box_small).
    Двухпроходная схема (z=0 → оценка h → z=h/2) — в
    `build_observations_perspective`."""
    row_native = meta["crop_y"] + meta["crop_side"] / 2.0
    dist = float(np.linalg.norm(cam_geom.t))
    f_px = cam_geom.K[0, 0]
    scale_m_per_px = (dist - z_eff_m) / f_px
    return -(row_native - cam_geom.n_rows / 2.0) * scale_m_per_px


def build_observations(
    masks: dict[str, np.ndarray], cells_meta: dict, geoms: dict[str, CamGeom],
    rig_cfg: dict, tile_px: int,
) -> list[Observation]:
    observations = []
    for cell_name, mask in masks.items():
        meta = cells_meta[cell_name]
        cam_name = meta["camera"]
        x_shift = measured_shift_m(meta, geoms[cam_name])
        pose = build_cell_pose(
            cam_name, geoms[cam_name], rig_cfg,
            meta["crop_x"], meta["crop_y"], meta["scale"], tile_px,
            x_shift_m=x_shift,
        )
        observations.append(Observation(mask=mask, pose=pose, mirror_plane=None, margin_px=0.0))
    return observations


def _cam_z_eff_m(cam_geom: CamGeom, x_shift_m: float, obj_z_eff_m: float) -> float:
    """Глубина от камеры до точки на высоте `obj_z_eff_m` вдоль луча камеры.
    Для top (angle=90°, forward≈(0,0,-1)): depth = dist - z_eff.
    Для side/diag: forward имеет Z-компоненту, depth = (dist·cos_α - z_eff) / cos_α_z,
    где cos_α — косинус угла между forward и горизонталью; в общем случае
    depth = |camera_pos - (x_shift, 0, z_eff)|. Но для простоты и устойчивости
    используем Z-компоненту forward-вектора: depth_z = dist - z_eff / forward_z
    (forward_z = -sin(angle) для прямых камер; для top angle=90° → forward_z=-1).
    На практике коррекция значима только для top-камер (forward_z≈-1), для
    side/diag (forward_z≈-0.08..-0.34) она пренебрежимо мала — используем
    простую формулу depth = dist - z_eff для всех, что верно для top и
    консервативно (занижает коррекцию) для боковых."""
    # Простая и устойчивая аппроксимация: depth = dist - z_eff.
    # Для top (forward_z=-1) — точно. Для side/diag — занижает коррекцию
    # (реальная depth чуть больше из-за наклона), но коррекция там и так мала.
    return float(np.linalg.norm(cam_geom.t)) - obj_z_eff_m


def build_observations_perspective(
    masks: dict[str, np.ndarray], cells_meta: dict, geoms: dict[str, CamGeom],
    rig_cfg: dict, tile_px: int,
    max_iters: int = 3, z_tol_m: float = 0.005,
) -> tuple[list[Observation], dict]:
    """Итеративное уточнение x_shift через перспективную коррекцию высоты.

    Проблема: `measured_shift_m` (ортографика, z=0) даёт систематическую
    ошибку ~50мм на start/end для объектов высотой >0.1м — bbox-центр маски
    соответствует z≈h/2 (середина высоты), а не z=0 (уровень ленты).
    Перспективная формула: `x = -(row - n_rows/2) * (dist - z_eff) / f`,
    где z_eff — эффективная высота bbox-центра.

    Итеративная схема:
      1. Проход 0: z_eff=0 (ортографика) → построение халла → оценка h.
      2. Проход k: z_eff=h/2 (половина высоты халла из прохода k-1) →
         перестроение халла → новая оценка h.
      3. Сходимость: |h_k - h_{k-1}| < z_tol_m или max_iters.

    Возвращает (observations, debug_info). debug_info содержит историю
    изменения высоты и x_shift по итерациям."""
    from roboson_tools.visual_hull.pose_carving import carve_polytope as _carve

    debug = {"iters": [], "converged": False, "final_z_eff_m": 0.0}

    # Проход 0: z_eff=0 (ортографика)
    z_eff = 0.0
    observations = _build_obs_at_z(masks, cells_meta, geoms, rig_cfg, tile_px, z_eff)
    hull = _carve(observations)
    h_prev = _hull_height_m(hull)
    debug["iters"].append({"iter": 0, "z_eff_m": z_eff, "hull_h_m": h_prev})

    for it in range(1, max_iters + 1):
        z_eff = h_prev / 2.0 if h_prev is not None else 0.0
        observations = _build_obs_at_z(masks, cells_meta, geoms, rig_cfg, tile_px, z_eff)
        hull = _carve(observations)
        h_curr = _hull_height_m(hull)
        debug["iters"].append({"iter": it, "z_eff_m": z_eff, "hull_h_m": h_curr})
        debug["final_z_eff_m"] = z_eff

        if h_curr is None:
            # Халл сломался — откат к z_eff=0
            debug["converged"] = False
            observations = _build_obs_at_z(masks, cells_meta, geoms, rig_cfg, tile_px, 0.0)
            break
        if abs(h_curr - h_prev) < z_tol_m:
            debug["converged"] = True
            break
        h_prev = h_curr

    return observations, debug


def _build_obs_at_z(
    masks: dict[str, np.ndarray], cells_meta: dict, geoms: dict[str, CamGeom],
    rig_cfg: dict, tile_px: int, z_eff_m: float,
) -> list[Observation]:
    """Вспомогательная: строит observations с перспективной коррекцией z_eff."""
    observations = []
    for cell_name, mask in masks.items():
        meta = cells_meta[cell_name]
        cam_name = meta["camera"]
        g = geoms[cam_name]
        # Перспективная глубина для этой камеры
        depth_eff = _cam_z_eff_m(g, 0.0, z_eff_m)
        # measured_shift_m с коррекцией: scale = (depth_eff) / f
        row_native = meta["crop_y"] + meta["crop_side"] / 2.0
        f_px = g.K[0, 0]
        scale_m_per_px = depth_eff / f_px
        x_shift = -(row_native - g.n_rows / 2.0) * scale_m_per_px
        pose = build_cell_pose(
            cam_name, g, rig_cfg,
            meta["crop_x"], meta["crop_y"], meta["scale"], tile_px,
            x_shift_m=x_shift,
        )
        observations.append(Observation(mask=mask, pose=pose, mirror_plane=None, margin_px=0.0))
    return observations


def _hull_height_m(hull) -> float | None:
    """Высота халла по Z (м), или None если халл пустой."""
    if hull is None or not hasattr(hull, "vertices") or len(hull.vertices) == 0:
        return None
    z = hull.vertices[:, 2]
    return float((z.max() - z.min()) / 1000.0)  # мм -> м


def reconstruct_path(
    observations: list[Observation], label: str, budget_start: float | None,
    also_exact: bool = False,
) -> dict:
    """Строит hull (carve_polytope, приоритет — скорость) + опционально
    carve_exact (диагностика), считает габариты и G4-вердикт. `budget_start`
    — момент начала замера бюджетного пути (только путь A); `None` — вне
    бюджета (путь B)."""
    out: dict = {"path": label, "n_observations": len(observations)}
    if len(observations) < 2:
        out["error"] = "недостаточно наблюдений (<2)"
        return out

    t_a = time.perf_counter()
    hull = carve_polytope(observations)
    t_hull = time.perf_counter()
    if hull is None:
        out["error"] = "carve_polytope вернул None (пустое/невыпуклое пересечение)"
        return out

    recon_dims = mesh_true_dims(Mesh(vertices=hull.vertices, faces=hull.faces), axis_step_deg=5.0)
    mesh_dims = tuple((hull.vertices.max(axis=0) - hull.vertices.min(axis=0)).tolist())
    g4 = _g4_core_from_points(
        hull.vertices, dims=recon_dims, mesh_dims=mesh_dims,
        start_time=budget_start if budget_start is not None else t_a,
        **G4_KWARGS,
    )
    t_g4 = time.perf_counter()

    out.update(
        recon_dims_mm=recon_dims,
        verdict=g4.verdict,
        k=g4.best.k if g4.best is not None else None,
        fallback_triggered=g4.fallback_triggered,
        roll_axis_dir=g4.best_axis_dir,  # единичный вектор оси переката в СК рига, мм; None — не найдена
        elapsed_hull_s=t_hull - t_a,
        elapsed_g4_s=t_g4 - t_hull,
        vertices=hull.vertices,
        faces=hull.faces,
    )
    if budget_start is not None:
        out["elapsed_total_budget_s"] = t_g4 - budget_start

    if also_exact:
        t_e0 = time.perf_counter()
        hull_exact = carve_exact(observations)
        t_e1 = time.perf_counter()
        if hull_exact is not None:
            exact_dims = mesh_true_dims(
                Mesh(vertices=hull_exact.vertices, faces=hull_exact.faces), axis_step_deg=5.0)
            out["exact_dims_mm"] = exact_dims
            out["exact_elapsed_s"] = t_e1 - t_e0
            out["exact_vertices"] = hull_exact.vertices
            out["exact_faces"] = hull_exact.faces
        else:
            out["exact_error"] = "carve_exact вернул None"
    return out


def reconstruct_path_perspective(
    masks: dict[str, np.ndarray], cells_meta: dict, geoms: dict[str, CamGeom],
    rig_cfg: dict, tile_px: int, label: str, budget_start: float | None,
    also_exact: bool = False, max_iters: int = 3, z_tol_m: float = 0.005,
) -> dict:
    """Итеративная 3D-реконструкция с перспективной коррекцией ракурса.

    Проблема: `measured_shift_m` (ортографика, z=0) даёт систематическую
    ошибку ~50мм на start/end для объектов высотой >0.1м — bbox-центр маски
    соответствует z≈h/2 (середина высоты), а не z=0 (уровень ленты).
    Перспективная формула: `x = -(row - n_rows/2) * (dist - z_eff) / f`,
    где z_eff — эффективная высота bbox-центра.

    Итеративная схема (см. `build_observations_perspective`):
      1. Проход 0: z_eff=0 (ортографика) → халл → оценка h.
      2. Проход k: z_eff=h/2 → перестроение халла → новая оценка h.
      3. Сходимость: |h_k - h_{k-1}| < z_tol_m или max_iters.

    После сходимости — финальный `reconstruct_path` с уточнёнными observations
    (включая also_exact, габариты, G4-вердикт). Возвращает результат
    `reconstruct_path` + `perspective_debug` с историей итераций."""
    observations, debug = build_observations_perspective(
        masks, cells_meta, geoms, rig_cfg, tile_px, max_iters, z_tol_m)
    out = reconstruct_path(observations, label, budget_start, also_exact)
    out["perspective_debug"] = debug
    return out


def dims_from_points(points_3d: np.ndarray, axis_step_deg: float = 5.0) -> tuple[float, float, float] | None:
    """Минимальный охватывающий бокс облака точек — тот же расчёт, что
    `mesh_true_dims`, но без обёртки `Mesh` (voxel-облако не имеет faces)."""
    box = _search_min_bounding_box(points_3d, axis_step_deg)
    if box is None:
        return None
    return tuple(sorted(box.dims, reverse=True))


def voxel_bbox_mm(rig_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Запасной (fallback) bbox под сетку вокселей — из наихудшего габарита
    ТЗ (`moment_detection.worst_case_*`), с запасом 15%. Используется, только
    если `carve_polytope` не дал результата (см. `voxel_bbox_from_polytope`)
    — для реальных (некрайних) объектов этот box СИЛЬНО избыточен и даёт
    слишком грубый воксель на тонких объектах (pen: 9-13мм сечение против
    ~13мм вокселя при grid_resolution=48 на этом боксе — сечение просто не
    попадает ни в один воксель). Ось Z не симметрична — объект стоит НА
    ленте (Z=0 в СК рига), не вокруг неё."""
    md = rig_cfg["moment_detection"]
    half_x = md["worst_case_extent_x"] * 1000.0 * 1.15
    half_yz = md["worst_case_transverse_radius"] * 1000.0 * 1.15
    lo = np.array([-half_x, -half_yz, -30.0])
    hi = np.array([half_x, half_yz, 2.0 * half_yz])
    return lo, hi


def voxel_bbox_from_polytope(result_a: dict, rig_cfg: dict, margin_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Bbox под воксельную сетку из УЖЕ ПОСЧИТАННОГО `carve_polytope`
    (result_a), расширенный запасом `margin_ratio` (100% — на каждую сторону
    добавляется полный размер экстента, т.е. итоговый бокс втрое больше
    исходного) — не для точности центра/границ (polytope, как показано в
    этой же задаче, систематически занижает объём), а чтобы АДАПТИРОВАТЬ
    разрешение сетки к реальному масштабу объекта, а не тащить один на все
    типы худший габарит ТЗ (`voxel_bbox_mm`) — на нём тонкие объекты (pen,
    9-13мм сечение) не попадают ни в один воксель при разумном
    `grid_resolution`. Замер худшего наблюдавшегося занижения — ~50% по
    одной оси (cylinder/Z) — запас 100% с каждой стороны безопасен и с
    хорошим резервом. Falls back на `voxel_bbox_mm`, если `carve_polytope`
    не дал результата (`result_a` без `vertices`)."""
    if "vertices" not in result_a:
        return voxel_bbox_mm(rig_cfg)
    verts = result_a["vertices"]
    lo0, hi0 = verts.min(axis=0), verts.max(axis=0)
    extent = hi0 - lo0
    margin = np.maximum(extent * margin_ratio, 20.0)  # мин. 20мм запаса даже на вырожденных осях
    return lo0 - margin, hi0 + margin


def reconstruct_path_voxel(
    observations: list[Observation], bbox: tuple[np.ndarray, np.ndarray], label: str,
    grid_resolution: int = 48,
) -> dict:
    """Путь C — voxel carving (см. `carve_voxels` в roboson_tools) на ТЕХ ЖЕ
    наблюдениях, что путь A (HSV) — изолирует переменную "метод пересечения"
    от переменной "качество маски" (путь B/SAM3 меняет маску, этот путь —
    только метод). Мотивация — заметка задачи "Проверить G4 на
    3D-реконструкции с параллаксом по ленте (roboson_tools)": `polytope_hull`
    не выигрывает от start/end (16/18 и с параллаксом, и без), а воксельный
    метод (torchhull) — выигрывает (16/18 -> 18/18)."""
    out: dict = {"path": label, "n_observations": len(observations)}
    if len(observations) < 2:
        out["error"] = "недостаточно наблюдений (<2)"
        return out
    t0 = time.perf_counter()
    hull = carve_voxels(observations, bbox=bbox, grid_resolution=grid_resolution)
    t1 = time.perf_counter()
    if hull is None:
        out["error"] = "carve_voxels вернул None (пустая сетка после пересечения)"
        return out

    recon_dims = dims_from_points(hull.points, axis_step_deg=5.0)
    mesh_dims = tuple((hull.points.max(axis=0) - hull.points.min(axis=0)).tolist())
    g4 = _g4_core_from_points(
        hull.points, dims=recon_dims, mesh_dims=mesh_dims, start_time=t0, **G4_KWARGS)
    t2 = time.perf_counter()

    out.update(
        recon_dims_mm=recon_dims,
        verdict=g4.verdict,
        k=g4.best.k if g4.best is not None else None,
        fallback_triggered=g4.fallback_triggered,
        roll_axis_dir=g4.best_axis_dir,  # единичный вектор оси переката в СК рига, мм; None — не найдена
        elapsed_carve_s=t1 - t0,
        elapsed_g4_s=t2 - t1,
        elapsed_total_s=t2 - t0,
        points=hull.points,
        voxel_size_mm=hull.voxel_size,
        n_voxels=len(hull.points),
    )
    return out


def _camera_pose_to_torchhull_transform(pose: CameraPose) -> np.ndarray:
    """`CameraPose` (`x_cam = rotation @ x_world + translation`, строки rotation =
    [right, down, forward] — см. докстринг `roboson_tools.pose.camera_pose.CameraPose`)
    -> 4x4 `K@T` в OpenCV-конвенции, которую ожидает `torchhull.visual_hull`
    (см. докстринг `torchhull_adapter.py`: `T=[[R,t],[0,0,0,1]]`, `K` — 4x4 с fx/fy/cx/cy).
    Совпадает по построению с `build_transform_matrix` того модуля, но берёт K/R/t из уже
    готовой позы (с per-cell сдвигом cx/cy под кроп), а не строит их заново с центрированным
    cx/cy — см. докстринг модуля про то, почему `build_transform_matrix` здесь не годится."""
    intr = pose.intrinsics
    K = np.array([
        [intr.fx, 0.0, intr.cx, 0.0],
        [0.0, intr.fy, intr.cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = pose.rotation.astype(np.float32)
    T[:3, 3] = pose.translation.astype(np.float32)
    return K @ T


def reconstruct_path_torchhull(
    observations: list[Observation], bbox: tuple[np.ndarray, np.ndarray], label: str,
    level: int = 7,
) -> dict:
    """Путь D — GPU visual hull через torchhull (sparse voxel octree + marching
    cubes, `roboson_tools.visual_hull.torchhull_adapter.visual_hull_points`) на
    ТЕХ ЖЕ наблюдениях, что путь A (HSV) — CPU-альтернатива пути C (voxel
    carving), тот же приём "фиксируем маску, меняем только метод пересечения"
    (см. `reconstruct_path_voxel`). Требует torch+torchhull+CUDA — см.
    докстринг модуля, ленивый импорт ниже.

    `bbox` — (lo, hi) mm, например из `voxel_bbox_from_polytope`; torchhull
    принимает только КУБИЧЕСКИЙ объём (`cube_corner_bfl` + один `cube_length`,
    не отдельный per-axis bbox), поэтому здесь bbox расширяется до куба вокруг
    своего центра по наибольшей стороне (тот же приём, что уже использует
    `MainWindow._RecomputeWorker` в roboson_tools для торчхалла STL-режима)."""
    from roboson_tools.visual_hull.torchhull_adapter import visual_hull_points

    out: dict = {"path": label, "n_observations": len(observations)}
    if len(observations) < 2:
        out["error"] = "недостаточно наблюдений (<2)"
        return out

    t0 = time.perf_counter()
    masks = [obs.mask for obs in observations]
    transforms = np.stack(
        [_camera_pose_to_torchhull_transform(obs.pose) for obs in observations], axis=0
    )
    lo, hi = bbox
    center = (lo + hi) / 2.0
    half_extent = float(np.max(hi - lo)) / 2.0
    cube_corner = (center - half_extent).tolist()
    cube_length = 2.0 * half_extent

    # masks_partial=True — сетка ВСЕГДА содержит start/end ячейки (объект может быть
    # обрезан краем кадра, см. sim/cv_moments.py), не только center.
    points = visual_hull_points(
        masks=masks, transforms=transforms, cube_corner_bfl=cube_corner,
        cube_length=cube_length, level=level, masks_partial=True,
    )
    t1 = time.perf_counter()
    if points is None:
        out["error"] = "torchhull.visual_hull вернул None (пустая/вырожденная реконструкция)"
        return out

    recon_dims = dims_from_points(points, axis_step_deg=5.0)
    mesh_dims = tuple((points.max(axis=0) - points.min(axis=0)).tolist())
    # dense_points=True — только этот путь (torchhull, GPU sparse voxel octree + marching
    # cubes) даёт достаточно изотропно семплированное облако для признака «устойчивая
    # локальная круглость» (sustained_fraction) — регрессия подтвердила и на torchhull.STL-
    # режиме, и здесь (`reconstruct_path`/`reconstruct_path_voxel` — CPU carve_polytope/voxel —
    # НЕ dense_points, сложили тот же false positive, что и `_band_intersection_points`, см.
    # заметку задачи [[Устойчивая локальная круглость вдоль оси переката (G4)]], dev, 2026-07-31).
    g4 = _g4_core_from_points(
        points, dims=recon_dims, mesh_dims=mesh_dims, start_time=t0, dense_points=True, **G4_KWARGS)
    t2 = time.perf_counter()

    out.update(
        recon_dims_mm=recon_dims,
        verdict=g4.verdict,
        k=g4.best.k if g4.best is not None else None,
        fallback_triggered=g4.fallback_triggered,
        roll_axis_dir=g4.best_axis_dir,
        elapsed_carve_s=t1 - t0,
        elapsed_g4_s=t2 - t1,
        elapsed_total_s=t2 - t0,
        points=points,
        n_points=len(points),
    )
    return out


def ground_truth_dims_mm(obj_type: str) -> tuple[float, float, float]:
    mesh = load_stl(MESHES_DIR / f"{obj_type}.stl")
    dims_m = mesh_true_dims(mesh, axis_step_deg=5.0)
    return tuple(d * 1000.0 for d in dims_m)


def expected_verdict(category: str) -> tuple[str, ...]:
    if category == "round":
        return ("round",)
    return ("not_round", "uncertain")
