"""Сборка сетки 9 ракурсов с top-локализацией и проекцией в остальные камеры
(Фаза 3.2 CV-пайплайна). Переиспользуемая часть (геометрия рига + сборщик) —
без драйвера замера/I/O (он в tools/experiment_grid_v2.py).

Постановка, алгоритм по шагам и результаты —
`03 Work/Расчёт кропов сетки ракурсов — top-локализация и проекция (Фаза 3.2).md`.

Отличие от sim/cv_grid.py::assemble_grid (Фаза 3.1):
- Положение объекта (belt_x, belt_y) на ленте определяется ОДИН раз по top-
  камере (HSV-субтракция на ds8 + unproject центроида bbox на плоскость ленты).
- Для остальных 8 ячеек центр кропа = проекция (belt_x, belt_y, belt_z) в кадр
  этой камеры по позе из config/layout.yaml (в проде — PnP по ArUco).
- Уточнение bbox в остальных камерах — HSV-субтракцией в УЗКОЙ полосе ленты
  вокруг спроецированного центра (не по всему кадру).

Конвенция осей Webots (критично — починено в этом эксперименте, см. docstring
`project_point`): локальная X=ВПЕРЁД, Y=ВЛЕВО, Z=ВВЕРХ — минус перед cam_y/cam_z
в pinhole-формуле обязателен.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from sim.cv_grid import (
    CELL_CAMERA_MOMENT, CELL_NAMES, bbox_center_and_size, camera_distance_m,
    expected_pixel_row, _downscale_bbox_fast, _crop_and_resize_square,
    _downscale_to_max_dim, GridCell, GridResult,
)
from sim.cv_moments import moment_thresholds


# ---- Геометрия рига (повторяет tools/gen_world.py::_camera_frame, без импорта webots-генератора) ----


def _camera_frame(theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, left, up) прямой камеры на азимуте theta (рад) — копия
    tools/gen_world.py::_camera_frame. См. её docstring про конвенцию осей
    Webots (локальная +X вперёд)."""
    up = np.array([1.0, 0.0, 0.0])
    forward = np.array([0.0, -math.cos(theta), -math.sin(theta)])
    left = np.cross(up, forward)
    return forward, left, up


def _mirror_normal_3d(phi_deg: float) -> np.ndarray:
    phi = math.radians(phi_deg)
    return np.array([0.0, math.sin(phi), -math.cos(phi)])


def _reflect_dir(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - 2.0 * np.dot(v, n) * n


def _mirror_camera_basis(source_angle_deg: float, own_angle_deg: float,
                         own_distance: float, mirror_normal_deg: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, left, up) зеркальной камеры — копия логики
    tools/gen_world.py::_mirror_camera_rotation (прицел на объект + reflect left
    + восстановленный up), но возвращает базис, не VRML axis-angle."""
    theta_source = math.radians(source_angle_deg)
    _, real_left, _ = _camera_frame(theta_source)
    n = _mirror_normal_3d(mirror_normal_deg)
    theta_own = math.radians(own_angle_deg)
    virtual_pos_offset = np.array([0.0, own_distance * math.cos(theta_own),
                                    own_distance * math.sin(theta_own)])
    forward = -virtual_pos_offset
    forward = forward / np.linalg.norm(forward)
    left_raw = _reflect_dir(real_left, n)
    left = left_raw - np.dot(left_raw, forward) * forward
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)
    return forward, left, up


@dataclass
class CamGeom:
    """Поза камеры в СК рига (начало координат = точка наблюдения на ленте
    (rig.x, 0, belt_z)); нужна для проекции мировой точки в пиксели кадра."""
    R: np.ndarray  # 3x3, строки = (forward, left, up) в МИРОВЫХ осях
    t: np.ndarray  # 3, позиция камеры в СК рига
    K: np.ndarray  # 3x3 intrinsic
    n_rows: int
    n_cols: int
    is_mirror: bool


def build_camera_geom(name: str, rig: dict) -> CamGeom:
    """Строит CamGeom по config/layout.yaml: cv_rig.cameras[name]."""
    cam = rig["cameras"][name]
    theta = math.radians(cam["angle"])
    if cam.get("mirror"):
        d = rig["mirror_distance_by_angle"][float(cam["angle"])]
    else:
        d = cam["distance"]
    # позиция в СК рига (точка наблюдения = (rig.x, 0, belt_z) — начало СК)
    t = np.array([0.0, d * math.cos(theta), d * math.sin(theta)])
    if cam.get("mirror"):
        source_angle = rig["cameras"][cam["reflects"]]["angle"]
        forward, left, up = _mirror_camera_basis(source_angle, cam["angle"], d,
                                                  rig["mirror_normal_deg"])
    else:
        forward, left, up = _camera_frame(theta)
    # R: мир->камера, строки = базис камеры в МИРОВЫХ осях
    R = np.vstack([forward, left, up])
    # per-camera "resolution"/"fov_deg" (опционально) переопределяют общие —
    # нужно для top_track (узкое native-разрешение + отдельный fov_deg, см.
    # config/layout.yaml cv_rig.cameras.top_track). Без этого f_px считался бы
    # по общим параметрам рига, что для top_track давало бы ~3.86x занижение
    # (fov_deg=68 вместо 19.8525 при n_cols=336).
    n_rows, n_cols = cam.get("resolution", rig["resolution"])
    fov = math.radians(cam.get("fov_deg", rig["fov_deg"]))
    # Webots привязывает fieldOfView к оси WIDTH (= n_cols), см. config/layout.yaml
    # cv_rig docstring. f_px в пикселях квадратный, выводится через n_cols (ось,
    # к которой привязан FOV), а не n_rows (whose FOV выводится из отношения сторон).
    f_px = (n_cols / 2.0) / math.tan(fov / 2.0)
    K = np.array([[f_px, 0.0, n_cols / 2.0],
                  [0.0, f_px, n_rows / 2.0],
                  [0.0, 0.0, 1.0]])
    return CamGeom(R=R, t=t, K=K, n_rows=n_rows, n_cols=n_cols,
                   is_mirror=bool(cam.get("mirror")))


def project_point(world_pt: np.ndarray, g: CamGeom) -> tuple[float, float]:
    """Проекция 3D-точки (в СК рига, начало = точка наблюдения) в пиксели кадра.

    Конвенция осей Webots этого рига (см. tools/gen_world.py::_camera_frame
    docstring): локальная X=ВПЕРЁД, Y=ВЛЕВО, Z=ВВЕРХ (не OpenGL X-право/Y-вниз/
    Z-вперёд). Камера смотрит вдоль +X (forward), так что глубина = cam_x.
    Точка слева от оптической оси (cam_y>0 в Webots) попадает в кадре левее
    центра → u = cx − f·cam_y/depth. Точка выше оси (cam_z>0) — выше центра
    → v = cy − f·cam_z/depth. Знаки проверены сравнением с эталонным
    bbox-центром Фазы 3.1 на side/diag/mirror: без минусов side давал dcx=−111
    (объект основанием в центре кадра, а эталонный силуэт — правее), с минусами
    проекция центра по высоте совпадает с эталоном в пределах 3-26px.

    Для зеркальной камеры сохранённый кадр уже вертикально отброшен
    (_flip_mirror_frame в supervisor_main.py) → row = n_rows-1-row_raw."""
    cam = g.R @ (world_pt - g.t)
    depth = cam[0]
    if depth <= 1e-6:
        return float("nan"), float("nan")
    u = g.K[0, 0] * (-cam[1]) / depth + g.K[0, 2]
    v_raw = g.K[1, 1] * (-cam[2]) / depth + g.K[1, 2]
    if g.is_mirror:
        v = g.n_rows - 1.0 - v_raw
    else:
        v = v_raw
    return float(u), float(v)


def unproject_top_to_belt(cx_top: float, cy_top: float, top: CamGeom
                           ) -> tuple[float, float]:
    """Обратная проекция пикселя top-камеры (cx=столбец, cy=строка) на плоскость
    ленты в СК рига (начало = точка наблюдения (rig.x, 0, belt_z), плоскость
    ленты = Z=0 в этой СК). Для top (angle=90°, up=(1,0,0)) проекция
    ортографична: строка зависит только от X, столбец только от Y — используем
    ТОТ ЖЕ знак, что в cv_moments.py::estimate_x_from_mask (проверен реальным
    захватом): МЕНЬШЕМУ X соответствует БОЛЬШАЯ строка.
    Возвращает (x_rig, y_rig) — координаты в СК рига (x_rig=0 = объект в rig.x).
    """
    n_rows = top.n_rows
    n_cols = top.n_cols
    f_px = top.K[0, 0]
    d = float(np.linalg.norm(top.t))
    scale_m_per_px = d / f_px
    # строка растёт с уменьшением X → x_rig = -(cy - n_rows/2)*scale
    x_rig = -(cy_top - n_rows / 2.0) * scale_m_per_px
    # столбец: top смотрит вниз, left=(0,-1,0) → столбец растёт с уменьшением Y
    # → y_rig = -(cx - n_cols/2)*scale (объект на +Y → cx<n_cols/2 → y_rig>0)
    y_rig = -(cx_top - n_cols / 2.0) * scale_m_per_px
    return x_rig, y_rig


def compute_target_hint_from_top(
    object_frames: dict[str, np.ndarray],
    background_frames: dict[str, np.ndarray],
    rig_cfg: dict,
) -> dict[str, tuple[float, float]] | None:
    """Строит `target_hint` для `assemble_grid_v2` из НЕЗАВИСИМОЙ локализации
    КАЖДОГО top-кадра момента (`start_top`/`center_top`/`end_top`) — вместо
    единственной позиции из `center_top`, которую `assemble_grid_v2` без
    `target_hint` использует на все 9 ячеек сразу (см. её Step 1).

    Для однообъектного кадра, где top детектирует объект на всех трёх
    моментах (типичный случай — большинство объектов, включая `cylinder`,
    который НЕ детектируется на `side`, но ХОРОШО детектируется на `top`),
    это даёт: (1) верную позицию geometry-fallback на старте/конце, если
    детекция в конкретной ячейке (например, `side` для `cylinder`)
    проваливается — раньше центрировалось по позиции момента "центр",
    смещённой на сотни пикселей от истинной позиции старта/конца (найдено
    живым прогоном на `cylinder`, см. заметку задачи "3D-реконструкция
    объекта по кропам сетки ракурсов..."); (2) точную полосу поиска
    (`_project_belt_strip`/`_bbox_in_strip`) в остальных ячейках того же
    момента вместо полосы вокруг устаревшей позиции.

    Возвращает None, если фон `top` недоступен или top не увидел объект ни
    на одном из трёх моментов — тогда `assemble_grid_v2` использует прежнее
    поведение (единая позиция из `center_top`, если она есть, иначе (0,0))."""
    bs = rig_cfg["background_subtraction"]
    top_bg = background_frames.get("top")
    if top_bg is None:
        return None
    top_geom = build_camera_geom("top", rig_cfg)

    # Подсказка по строке для ПЕРВОГО прохода `_downscale_bbox_fast` на
    # КАЖДОМ top-моменте. Для "центра" — центр кадра (там же конкурент —
    # не шум, а сам объект по построению должен быть у центра; подсказка не
    # нужна для отличения от шума, но не вредит). Для "старт"/"конец" —
    # ИСКЛЮЧИТЕЛЬНО argmax по площади (`expected_row_hint=None`), БЕЗ
    # геометрической подсказки по порогу `delta_max`: живой прогон на
    # `cylinder` показал, что геометрическая оценка (из номинального порога,
    # не из реальной позиции объекта в момент триггера — та гуляет на
    # 100-300мм, см. заметку задачи) может случайно оказаться БЛИЖЕ к
    # мелкому шумовому фрагменту кадра, чем к настоящему объекту, и увести
    # выбор на артефакт. Реальный движущийся объект СТАБИЛЬНО оказывается
    # КРУПНЕЙШИМ кандидатом (у cylinder area=157 против 42/15/10 у
    # фрагментов шума на обоих прогонах) — argmax надёжнее подсказки здесь.
    thresholds = moment_thresholds(rig_cfg)
    moment_hint_row = {
        "start": None,
        "center": top_geom.n_rows / 2.0,
        "end": None,
    }

    hint: dict[str, tuple[float, float]] = {}
    world_x_by_moment: dict[str, float] = {}
    for moment, cell_name in (("start", "start_top"), ("center", "center_top"), ("end", "end_top")):
        frame = object_frames.get(cell_name)
        if frame is None:
            continue
        bbox = _downscale_bbox_fast(frame, top_bg, bs, expected_row_hint=moment_hint_row[moment],
                                    v_threshold=bs.get("v_threshold"))
        if bbox is None:
            continue
        cx, cy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0
        hint[cell_name] = (cy, cx)
        x_rig, _y_rig = unproject_top_to_belt(cx, cy, top_geom)
        world_x_by_moment[moment] = rig_cfg["x"] + x_rig
    if not world_x_by_moment:
        return None

    # "start"/"end" — это ДВА НЕЗАВИСИМЫХ физических события (top и side
    # пересекают СВОЙ порог `delta_max` в РАЗНОЕ время/положение на ленте, см.
    # DELTA_MAX_CAMERAS — у side заметно уже delta_max, чем у top): X,
    # измеренный по кадру `start_top`, НЕ является X объекта в момент
    # `start_side` — переиспользование одного world_x на оба привело к
    # реальному бага (окно кропа `start_side` оказалось ЦЕЛИКОМ за кадром,
    # найдено по вопросу пользователя "цилиндр отсутствует на side"). Для
    # side-ячеек момента старт/конец используем ГЕОМЕТРИЧЕСКИЙ порог самого
    # side (`thresholds["start_side"]`/`["end_side"]`) — независимая
    # детекция по кадру side ненадёжна (тот же низкий контраст, что и делает
    # эту ячейку fallback'ом в принципе), а геометрия рига — единственная
    # опора без физики, ВСЕГДА попадающая в кадр (в отличие от чужого X).
    # "center" — ОБЩИЙ порог cv_rig.x для всех 5 камер момента "центр"
    # (см. CENTER_CAMERAS) — тут world_x_by_moment["center"] остаётся верным
    # источником для center_side/center_diag/center_top_mirror/
    # center_diag_mirror, т.к. все они триггерятся ОДНОВРЕМЕННО с center_top.
    geometry_threshold_cell = {"start_side": "start_side", "end_side": "end_side"}

    fov_deg = rig_cfg["fov_deg"]
    n_rows, n_cols = rig_cfg["resolution"]
    for cell_name, (camera, moment) in CELL_CAMERA_MOMENT.items():
        if cell_name in hint:
            continue
        if cell_name in geometry_threshold_cell:
            world_x = thresholds[geometry_threshold_cell[cell_name]]
        else:
            world_x = world_x_by_moment.get(moment)
        if world_x is None:
            continue
        dist = camera_distance_m(rig_cfg, camera)
        row = expected_pixel_row(world_x, rig_cfg["x"], dist, fov_deg, n_rows, n_cols)
        hint[cell_name] = (row, n_cols / 2.0)
    return hint


# ---- v2 сборщик ----

# Ширина полосы ленты (м) — по запросу пользователя: ±0.25 м от центра объекта.
_BELT_HALF_WIDTH = 0.25
# Запас по высоте объекта в полосе (м) — худший габарит ТЗ 320мм + запас, чтобы
# высокий объект попал в полосу целиком. Найдено замером: для side-камеры (угол
# 5°, смотрит почти вдоль Y) РАЗМЕР ОБЪЕКТА ПО ВЫСОТЕ (Z) — главный размер по
# ГОРИЗОНТАЛИ кадра (столбцам), т.к. left≈-Z; полоса Z=0..0.22 резала box_large
# (626px wide в side) до 104px. Берём 0.40м = худший габарит 0.32 + запас.
_OBJECT_HEIGHT_MARGIN = 0.40


def _bbox_in_strip(frame: np.ndarray, bg_frame: np.ndarray, bs: dict,
                    row_lo: int, row_hi: int, col_lo: int, col_hi: int,
                    factor: int = 8,
                    expected_row_hint: float | None = None,
                    expected_col_hint: float | None = None,
                    ) -> tuple[int, int, int, int] | None:
    """HSV-субтракция в УЗКОЙ полосе кадра (row_lo..row_hi, col_lo..col_hi),
    ds8, connectedComponents → bbox в НАТИВНЫХ координатах всего кадра.
    Полоса уже маленькая → ds8 по полосе, не по всему кадру. Возвращает bbox
    (x, y, w, h) в native или None.

    `expected_row_hint`/`expected_col_hint` (опционально, НАТИВНЫЕ пиксели):
    при нескольких компонентах в полосе (многообъектный кадр — соседний объект
    попал в полосу; отражение) выбирает компоненту, чей центроид ближе к
    ожидаемому центру целевого, а не наибольшую по площади (argmax ловит
    крупного соседа — замеры на тройке triple_pen_boxlarge_boxsmall показали
    выбор box_large вместо целевого pen на ВСЕХ 9 ячейках). При `None` или
    одной компоненте — argmax по площади (прежнее поведение). Подсказки —
    геометрия (проекция точки контакта целевого из top-локализации), не
    ground-truth физики."""
    h, w = frame.shape[:2]
    row_lo = max(0, min(h, row_lo)); row_hi = max(0, min(h, row_hi))
    col_lo = max(0, min(w, col_lo)); col_hi = max(0, min(w, col_hi))
    if row_hi - row_lo < 4 or col_hi - col_lo < 4:
        return None
    strip = frame[row_lo:row_hi, col_lo:col_hi]
    strip_bg = bg_frame[row_lo:row_hi, col_lo:col_hi]
    # ds8 по полосе
    sh, sw = strip.shape[:2]
    f = factor
    if max(sh, sw) // f < 8:
        f = max(1, max(sh, sw) // 8)
    small = cv2.resize(strip, None, fx=1.0 / f, fy=1.0 / f,
                       interpolation=cv2.INTER_AREA)
    small_bg = cv2.resize(strip_bg, None, fx=1.0 / f, fy=1.0 / f,
                          interpolation=cv2.INTER_AREA)
    hsv1 = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv0 = cv2.cvtColor(small_bg, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = np.minimum(np.abs(hsv1[..., 0] - hsv0[..., 0]),
                    180.0 - np.abs(hsv1[..., 0] - hsv0[..., 0]))
    ds = np.abs(hsv1[..., 1] - hsv0[..., 1])
    fg = ((dh > bs["h_threshold"]) | (ds > bs["s_threshold"]))
    # V-критерий «объект ЯРЧЕ фона» (dv_signed > v_threshold) — тот же, что в
    # cv_localization.compute_mask: добавлен для нейтрально-серого `cylinder`,
    # почти неотличимого от ленты по H/S. Без него v2 даёт 7 fallback на
    # cylinder (Фаза 3.1 — 3, т.к. compute_mask использует V; v2 в полосе — нет).
    v_threshold = bs.get("v_threshold")
    if v_threshold is not None:
        dv_signed = hsv1[..., 2] - hsv0[..., 2]
        fg = fg | (dv_signed > v_threshold)
    fg = fg.astype(np.uint8) * 255
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area_ds = bs["min_area_px"] / (f * f)
    qualifying = [i for i in range(n - 1) if areas[i] >= min_area_ds]
    if not qualifying:
        return None
    # Выбор компоненты: при подсказке (row/col) — ближайшая к ожидаемому
    # центру целевого; иначе — наибольшая (прежнее поведение).
    if expected_row_hint is not None or expected_col_hint is not None:
        hint_row_ds = (expected_row_hint - row_lo) / f if expected_row_hint is not None else None
        hint_col_ds = (expected_col_hint - col_lo) / f if expected_col_hint is not None else None
        def _dist(i):
            cy_ds = stats[i + 1, cv2.CC_STAT_TOP] + stats[i + 1, cv2.CC_STAT_HEIGHT] / 2.0
            cx_ds = stats[i + 1, cv2.CC_STAT_LEFT] + stats[i + 1, cv2.CC_STAT_WIDTH] / 2.0
            d2 = 0.0
            if hint_row_ds is not None:
                d2 += (cy_ds - hint_row_ds) ** 2
            if hint_col_ds is not None:
                d2 += (cx_ds - hint_col_ds) ** 2
            return d2
        pick = min(qualifying, key=_dist)
    else:
        pick = max(qualifying, key=lambda i: areas[i])
    idx = pick + 1
    bleed = 1
    x_ds = stats[idx, cv2.CC_STAT_LEFT] - bleed
    y_ds = stats[idx, cv2.CC_STAT_TOP] - bleed
    w_ds = stats[idx, cv2.CC_STAT_WIDTH] + 2 * bleed
    h_ds = stats[idx, cv2.CC_STAT_HEIGHT] + 2 * bleed
    # обратно в координаты полосы (native), потом в координаты всего кадра
    x = col_lo + x_ds * f
    y = row_lo + y_ds * f
    return (int(x), int(y), int(w_ds * f), int(h_ds * f))


def _project_belt_strip(rig_pt: np.ndarray, g: CamGeom,
                         belt_half_width: float = _BELT_HALF_WIDTH,
                         height_margin: float = _OBJECT_HEIGHT_MARGIN,
                         target_half_extent_x: float | None = None,
                         ) -> tuple[int, int, int, int]:
    """Проекция полосы ленты вокруг rig_pt (точка контакта в СК рига) в кадр
    камеры g. Полоса по Y (поперёк ленты) = [rig_pt.y - hw, rig_pt.y + hw].
    По X (вдоль ленты): если `target_half_extent_x` задан (многообъектный кадр
    — цель локализована top-камерой) — УЗКАЯ полоса ±target_half_extent_x
    вокруг rig_pt (по умолчанию 0.225м = половина худшего габарита ТЗ; убирает
    соседей из полосы → HSV не сливает, выбор компоненты не врёт). Иначе —
    ±0.675м (worst_case_extent_x*3, запас под крупный объект, прежнее
    поведение Фазы 3.2 для однообъектного кадра). По высоте — от плоскости
    ленты (Z=0 в мире = rig_pt.z) до Z=height_margin над лентой. Возвращает
    (row_lo, row_hi, col_lo, col_hi) в native пикселях, охватывающий проекции
    ВСЕХ 4 углов полосы на двух высотях (0 и height_margin)."""
    hw = belt_half_width
    if target_half_extent_x is not None:
        ext_x = float(target_half_extent_x)
    else:
        ext_x = 0.675  # worst_case_extent_x * 3 — запас под крупный объект
    base_x, base_y, base_z = float(rig_pt[0]), float(rig_pt[1]), float(rig_pt[2])
    corners = []
    for sx in (-ext_x, ext_x):
        for sy in (-hw, hw):
            for sz in (0.0, height_margin):
                corners.append(np.array([base_x + sx, base_y + sy, base_z + sz]))
    us, vs = [], []
    for c in corners:
        u, v = project_point(c, g)
        if math.isfinite(u) and math.isfinite(v):
            us.append(u); vs.append(v)
    if not us:
        return 0, 0, 0, 0
    pad = 16
    col_lo = int(max(0, min(us) - pad))
    col_hi = int(min(g.n_cols, max(us) + pad))
    row_lo = int(max(0, min(vs) - pad))
    row_hi = int(min(g.n_rows, max(vs) + pad))
    return row_lo, row_hi, col_lo, col_hi


def _fallback_center_size_v2(row_center_px: float, col_center_px: float,
                              camera_distance_m_: float, fov_deg: float,
                              frame_shape: tuple[int, int],
                              worst_case_extent_x: float,
                              worst_case_transverse_radius: float
                              ) -> tuple[float, float, float]:
    """Та же формула, что fallback_center_and_size, но (cx=col, cy=row)
    порядок аргументов под v2 (project_point возвращает u=col, v=row)."""
    n_rows = frame_shape[0]
    n_cols = frame_shape[1]
    # FOV привязан к WIDTH (n_cols), см. config/layout.yaml cv_rig docstring.
    f_px = (n_cols / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    px_per_m = f_px / camera_distance_m_
    natural_size = 2.0 * max(worst_case_extent_x, worst_case_transverse_radius) * px_per_m
    return col_center_px, row_center_px, natural_size


def assemble_grid_v2(
    object_frames: dict[str, np.ndarray],
    prev_frames: dict[str, np.ndarray],
    background_frames: dict[str, np.ndarray],
    rig_cfg: dict,
    cell_px: int,
    sam3_max_dim: int,
    margin_factor: float = 0.15,
    target_hint: dict[str, tuple[float, float]] | None = None,
    narrow_strip: bool = True,
) -> GridResult:
    """Сборка сетки: локализация по top → проекция в 8 камер + полоса +
    уточнение bbox в полосе. I/O (imread) вне — кадры уже в памяти.

    `target_hint` (опционально): cell_name -> (expected_row, expected_col) —
    ожидаемый центр ЦЕЛЕВОГО объекта в кадре каждой ячейки (НАТИВНЫЕ пиксели).
    Для многообъектного кадра: по top-локализации знаем X целевого → проекция
    в кадр каждой камеры даёт ожидаемый центр; полоса сужается вокруг цели
    (±worst_case_extent_x вместо ±0.675м — убирает соседей), а выбор
    компоненты в полосе — по близости к hint, а не argmax площади (который
    ловит крупного соседа — замеры на triple_pen_boxlarge_boxsmall: выбор
    box_large на ВСЕХ 9 ячейках без hint). Для тестов — из ground-truth
    `target_x` → `expected_pixel_row`; в проде — из top-локализации + проекции.
    `None` — однообъектный кадр (прежнее поведение Фазы 3.2).

    `narrow_strip` (по умолчанию True — прежнее поведение при заданном
    `target_hint`): сужать ли полосу поиска до `target_hint`. `False` —
    `target_hint` используется ТОЛЬКО для выбора компоненты (`expected_row_hint`/
    `expected_col_hint`) и позиции geometry-fallback, полоса остаётся широкой
    (±0.675м). Нужно для однообъектного кадра с ПОЧТИ точным hint (из
    независимой локализации КАЖДОГО top-момента, см.
    `compute_target_hint_from_top`) — там сужение не нужно (нет соседа,
    которого убирать) и только повышает хрупкость: живой прогон показал
    регрессию (13 из 16 объектов получили fallback на `side` вместо 1 из 16
    без сужения) — узкая полоса нетерпима к малым неточностям hint,
    накопленным через цепочку приближений (row→X по одной оси, y_rig=0),
    широкая полоса всё ещё находит объект сигналом, hint нужен только чтобы
    не перепутать с чем-то ещё в этой же полосе."""
    bs = rig_cfg["background_subtraction"]
    # геометрия камер (один раз)
    geoms = {name: build_camera_geom(name, rig_cfg)
             for name in {cam for cam, _m in CELL_CAMERA_MOMENT.values()}}

    # Сужение полосы по X вокруг целевого (многообъектный кадр). Берём половину
    # худшего габарита ТЗ вдоль ленты — достаточно покрыть целевой объект
    # (до 0.45м → ±0.225м), не захватывая соседа через 0.5м (интервал спавна
    # 0.5с × 1.0 м/с = 0.5м между объектами).
    target_half_extent_x = None
    if target_hint is not None and narrow_strip:
        target_half_extent_x = float(rig_cfg["moment_detection"]["worst_case_extent_x"])

    # --- Шаг 1: локализация по top (ячейка center_top) ---
    top_frame = object_frames["center_top"]
    top_bg = background_frames.get("top")
    x_rig, y_rig = None, None
    if top_bg is not None:
        # hint = центр кадра (момент center = объект в rig.x → центр для top),
        # чтобы выбрать pen (маленький, в центре), а не крупный шум у края —
        # та же логика expected_row_hint, что в Фазе 3.1 для зеркальных камер.
        # При многообъектном кадре (target_hint задан) — используем hint из
        # target_hint (проекция целевого), не геометрию центра.
        if target_hint is not None and "center_top" in target_hint:
            hint_row, hint_col = target_hint["center_top"]
        else:
            hint_row = geoms["top"].n_rows / 2.0
            hint_col = None
        bbox = _downscale_bbox_fast(top_frame, top_bg, bs,
                                     expected_row_hint=hint_row)
        if bbox is not None:
            cx_top = bbox[0] + bbox[2] / 2.0
            cy_top = bbox[1] + bbox[3] / 2.0
            x_rig, y_rig = unproject_top_to_belt(cx_top, cy_top, geoms["top"])
    if x_rig is None:
        # fallback: центр кадра top → объект в точке наблюдения (0,0 в СК рига)
        x_rig, y_rig = 0.0, 0.0
    # точка контакта объекта с лентой в СК рига (Z=0 = плоскость ленты).
    # ВНИМАНИЕ: это rig_pt для момента CENTER (из center_top). Объект ДВИЖЕТСЯ
    # между моментами — на start он левее, на end правее. Для многообъектного
    # кадра (target_hint) rig_pt per-момент строится ниже из target_hint
    # top-ячеек (start_top/center_top/end_top), т.к. целевой смещается по X
    # и полоса/проекция для end_top, построенная вокруг center, указала бы на
    # соседа (он к center-точке ближе, чем уехавший целевой — найдено замером
    # на multi_1m: end_top ловил box_large вместо pen). Без target_hint —
    # один rig_pt на все ячейки (прежнее поведение, однообъектный кадр —
    # объект в кадре один, X-смещение между моментами не критично для полосы).
    rig_pt_center = np.array([x_rig, y_rig, 0.0])

    # rig_pt per-ячейки из target_hint. Каждая ячейка даёт (row, col) целевого
    # в кадре СВОЕЙ камеры → unproject через top-геометрию → (x_rig, y_rig).
    # Используем top-геометрию для unproject (проекция top ортографична: row→X,
    # col→Y), т.к. hint дан в координатах кадреса ячейки, а rig_pt нужен в СК
    # рига. Для top-ячеек hint_row/hint_col уже в кадре top → unproject напрямую.
    # Для side/diag/mirror — hint в ИХ кадре; row этих камер тоже зависит от X
    # (up=(1,0,0) для всех, см. cv_moments.py::mask_row_bbox), но масштаб
    # другой (distance отличается). Проще и точнее: rig_pt.x из X целевого,
    # который восстановим из hint top-ячейки того же момента — НО момент "end"
    # снимается двумя событиями (end_side на x=2.91, end_top на x=3.45) с
    # РАЗНЫМИ X. Поэтому строим rig_pt per cell_name напрямую: для top-ячеек —
    # unproject hint этой ячейки; для side/diag/mirror — unproject hint через
    # top-геометрию, т.к. hint_row пропорционален X через expected_pixel_row
    # этой камеры → обратная формула через top-геометрию даёт X.
    # Проще: восстановить X из hint_row камеры ячейки (expected_pixel_row
    # обратима), потом rig_pt = (X - cv_rig_x, 0, 0).
    rig_pt_per_cell: dict[str, np.ndarray] = {}
    if target_hint is not None:
        cv_rig_x = float(rig_cfg["x"])
        for cell_name in CELL_NAMES:
            if cell_name not in target_hint:
                continue
            camera, _moment = CELL_CAMERA_MOMENT[cell_name]
            g_cell = geoms[camera]
            hint_row, hint_col = target_hint[cell_name]
            # Обратная проекция: hint_row -> world X. expected_pixel_row:
            #   row = n_rows/2 - (x - cv_rig_x)/scale, scale = dist/f_px.
            # => x = cv_rig_x - (row - n_rows/2)*scale. Та же формула, что
            # unproject_top_to_belt, но для ЛЮБОЙ камеры (scale = dist/f_px
            # этой камеры). up=(1,0,0) для всех камер рига → row зависит от X
            # одинаково (знак), отличается только масштаб (distance).
            f_px = g_cell.K[0, 0]
            dist = float(np.linalg.norm(g_cell.t))
            scale_m_per_px = dist / f_px
            x_rig = -(hint_row - g_cell.n_rows / 2.0) * scale_m_per_px
            # Y из col через топ-геометрию (col зависит от Y; для top — прямо,
            # для боковых col ~ Z/Y, но объект на Y≈0 по центру ленты —
            # берём y_rig=0 как разумное приближение для полосы).
            y_rig = 0.0
            rig_pt_per_cell[cell_name] = np.array([x_rig, y_rig, 0.0])

    # --- Шаг 2: для каждой ячейки — центр по проекции, bbox в полосе ---
    centers: dict[str, tuple[float, float]] = {}
    natural_sizes: dict[str, float] = {}
    sources: dict[str, str] = {}
    for cell_name in CELL_NAMES:
        camera, moment = CELL_CAMERA_MOMENT[cell_name]
        frame = object_frames[cell_name]
        g = geoms[camera]
        # rig_pt для этой ячейки: per-cell из target_hint если есть, иначе
        # общий из center_top (однообъектный кадр / нет hint на эту ячейку).
        rig_pt = rig_pt_per_cell.get(cell_name, rig_pt_center)
        # проекция точки контакта (belt_x, belt_y, 0) — ожидаемый центр объекта
        # в кадре (для top это совпадёт с bbox-центром, для других — ожидаемое
        # положение, уточняемое bbox в полосе)
        cx_exp, cy_exp = project_point(rig_pt.copy(), g)
        if not math.isfinite(cx_exp) or not math.isfinite(cy_exp):
            cx_exp, cy_exp = g.n_cols / 2.0, g.n_rows / 2.0

        # bbox в узкой полосе ленты.
        # Если проекция целевого ВНЕ кадра (объект уехал за край — end_side
        # при узком FOV side-камеры: целевой на x=3.33 → v=-66, вне кадра;
        # сосед box_large на x=2.1 в кадре — bbox в полосе поймал бы ЕГО),
        # bbox не ищем — сразу geometry fallback по проекции. Кроп уйдёт за
        # край (паддинг серым), но не выдаст чужой объект — найдено на
        # boxsmall_boxlarge end_side: кроп с box_large вместо отсутствующего
        # целевого ломал бы 3D-реконструкцию.
        bg_frame = background_frames.get(camera)
        bbox = None
        target_in_frame = (0.0 <= cx_exp < g.n_cols and 0.0 <= cy_exp < g.n_rows)
        if bg_frame is not None and target_in_frame:
            row_lo, row_hi, col_lo, col_hi = _project_belt_strip(
                rig_pt, g, target_half_extent_x=target_half_extent_x)
            if row_hi > row_lo and col_hi > col_lo:
                # Подсказка для выбора компоненты: при многообъектном кадре —
                # ожидаемый центр целевого (проекция rig_pt); иначе None
                # (прежнее поведение, argmax площади — один объект в полосе).
                hint_row = float(cy_exp) if target_hint is not None else None
                hint_col = float(cx_exp) if target_hint is not None else None
                bbox = _bbox_in_strip(frame, bg_frame, bs,
                                       row_lo, row_hi, col_lo, col_hi,
                                       expected_row_hint=hint_row,
                                       expected_col_hint=hint_col)
        if bbox is not None:
            cx, cy, natural = bbox_center_and_size(bbox, margin_factor)
            sources[cell_name] = "strip"
        else:
            # fallback: геометрический центр + worst-case размер
            shape = frame.shape[:2]
            dist = float(np.linalg.norm(g.t))
            fov = rig_cfg["fov_deg"]
            md = rig_cfg["moment_detection"]
            cx, cy, natural = _fallback_center_size_v2(
                cy_exp, cx_exp, dist, fov, shape,
                md["worst_case_extent_x"], md["worst_case_transverse_radius"])
            sources[cell_name] = "geometry"
        centers[cell_name] = (cx, cy)
        natural_sizes[cell_name] = natural

    # --- Шаг 3: единый квадратный размер + кроп ---
    native_side = max(natural_sizes.values())
    scale = cell_px / native_side
    cells: dict[str, GridCell] = {}
    sam3_crops: dict[str, np.ndarray] = {}
    for cell_name in CELL_NAMES:
        camera, moment = CELL_CAMERA_MOMENT[cell_name]
        cx, cy = centers[cell_name]
        object_crop = _crop_and_resize_square(object_frames[cell_name], cx, cy,
                                               native_side, cell_px, scale)
        bg_frame = background_frames.get(camera)
        background_crop = (_crop_and_resize_square(bg_frame, cx, cy, native_side,
                                                    cell_px, scale)
                           if bg_frame is not None else None)
        prev_frame = prev_frames.get(cell_name)
        prev_crop = (_crop_and_resize_square(prev_frame, cx, cy, native_side,
                                              cell_px, scale)
                     if prev_frame is not None else None)
        frame_h, frame_w = object_frames[cell_name].shape[:2]
        cells[cell_name] = GridCell(
            camera=camera, moment=moment,
            crop_x=round(cx - native_side / 2.0),
            crop_y=round(cy - native_side / 2.0),
            crop_side=round(native_side), frame_w=frame_w, frame_h=frame_h,
            scale=scale, bbox_source=sources[cell_name],
            object_crop=object_crop, background_crop=background_crop,
            prev_crop=prev_crop)
        sam3_crops[cell_name] = _downscale_to_max_dim(object_crop, sam3_max_dim)
    return GridResult(cells=cells, scale=scale, sam3_crops=sam3_crops)