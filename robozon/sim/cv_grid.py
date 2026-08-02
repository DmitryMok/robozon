"""Сборка сетки 9 ракурсов (Фаза 3 CV-пайплайна) — геометрия кропа/масштаба,
чистая логика (numpy+cv2, без Webots). Постановка и принятые решения — заметка
задачи "Сборка сетки 9 ракурсов из кадров Фазы 2 (Фаза 3, CV-пайплайн Webots)".

9 ракурсов = 5 в центре (`top`, `side`, `diag`, `top_mirror`, `diag_mirror`) +
2 старт (`top`, `side`) + 2 конец (`top`, `side`) — см.
`sim/cv_moments.py::CENTER_CAMERAS`/`DELTA_MAX_CAMERAS`. Зеркало отражает
ИМЕННО `top`+`diag` (не `side`), см. `config/layout.yaml: cv_rig.cameras`.
Композит — квадрат 3×3 (`_COMPOSITE_LAYOUT`), без пустых клеток.

Масштаб/размер сетки (решение принято с пользователем, уточнено по итогам
визуальной проверки первого прогона): у КАЖДОЙ ячейки свой ЦЕНТР кропа (bbox
объекта в этой ячейке — меняет только cx/cy виртуальной камеры кропа), но
РАЗМЕР окна кропа (в нативных пикселях, квадратный) — ОДИН на всю сетку,
подобранный так, чтобы окно самого крупного (в пикселях) объекта стало
целевым размером ячейки `cell_px` после масштабирования. Так фиксируется сразу
два инварианта: (1) fx/fy остаются ОДИНАКОВЫМИ по всем 9 видам (совместимо с
`roboson_tools/visual_hull/torchhull_adapter.py`, который строит K с одним
fx=fy на весь батч, без чего 18/18 на G4-бенчмарке пришлось бы перепроверять)
— меняется только cx/cy; (2) сами кропы (`object_crop` и т.д.) ВСЕГДА ровно
`cell_px`×`cell_px`, различается только относительный размер объекта внутри
— критично для единообразной последующей обработки (SAM3-батч, torchhull),
не для одной лишь визуальной эстетики. Если окно (в нативных пикселях) выходит
за границу кадра (объект у края — ожидаемо на старте/конце) — недостающая
часть паддится серым (`crop_square_padded`), а не обрезается неравномерно.

Подключение per-cell метаданных (`crop_x`/`crop_y`/`crop_side`/`scale`) в сам
torchhull-адаптер — дело Фазы 5, не этой задачи, здесь только считаем и
сохраняем их в удобном виде (`cells_metadata`).

Разрешение: кроп остаётся в исходном (native) пиксельном масштабе камеры для
classical CV/torchhull; отдельная уменьшенная копия (`sam3_crops`, до
`sam3_max_dim`) — только для SAM3, не подменяет основной канал.

Фаза 3.1 (оптимизация скорости): полный `compute_mask` на full-res
(2592×1944, ~1027мс на 9 ячеек — 88% всего `assemble_grid`) убран из сборки
сетки — он уже отработал в детекторе моментов Фазы 2.2, а здесь его результат
(центр+размер кропа) получается дешёвой локализацией на уменьшенной копии ×8
(`_downscale_bbox_fast`, ~80мс) — обе координаты (cx, cy) из центроида bbox,
без зависимости от ground-truth физики (в реальной системе positions_x не
будет, CV должен сам найти объект). Итог: `assemble_grid` ~100мс (×12-17
ускорение), 0 потери качества кропов (замерено на `pen`/`box_large`: side
343/884px vs 398/880px текущей — разрешение не хуже, на `pen` даже лучше).
`compute_mask`
остаётся в `cv_localization.py`/`cv_moments.py` где он действительно нужен.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# compute_mask/select_target_contour/build_background_model остаются в
# cv_localization.py (используются в cv_moments.py для детектора моментов и в
# tests/test_cv_grid.py для синтетических тестов); в assemble_grid они больше
# не нужны (Фаза 3.1 — см. docstring модуля).

# cell_name -> (camera, moment) — см. sim/cv_moments.py::threshold_captures().
CELL_CAMERA_MOMENT: dict[str, tuple[str, str]] = {
    "start_top": ("top", "start"),
    "start_side": ("side", "start"),
    "center_top": ("top", "center"),
    "center_side": ("side", "center"),
    "center_diag": ("diag", "center"),
    "center_top_mirror": ("top_mirror", "center"),
    "center_diag_mirror": ("diag_mirror", "center"),
    "end_side": ("side", "end"),
    "end_top": ("top", "end"),
}
CELL_NAMES: tuple[str, ...] = tuple(CELL_CAMERA_MOMENT)

# Раскладка для render_composite — строго 3x3 (9 ячеек, ни одной пустой):
# зеркальные ячейки (снимаются только в момент "центр") уходят в свободные
# углы старт/конец-рядов, не в отдельный 4-й ряд.
_COMPOSITE_LAYOUT = (
    ("start_top", "start_side", "center_top_mirror"),
    ("center_top", "center_side", "center_diag"),
    ("end_top", "end_side", "center_diag_mirror"),
)


def camera_distance_m(rig_cfg: dict, camera: str) -> float:
    """Дистанция камеры (прямой — из `cameras[name].distance`, зеркальной —
    из `mirror_distance_by_angle[angle]`, TRUE azimuth, см.
    `config/layout.yaml: cv_rig` docstring)."""
    cam = rig_cfg["cameras"][camera]
    if cam.get("mirror"):
        return rig_cfg["mirror_distance_by_angle"][cam["angle"]]
    return cam["distance"]


def expected_pixel_row(x_world: float, cv_rig_x: float, camera_distance_m_: float,
                        fov_deg: float, n_rows: int, n_cols: int | None = None) -> float:
    """Обратная формула к `cv_moments.py::estimate_x_from_mask` — по известной
    (из физики) мировой X объекта даёт ОЖИДАЕМУЮ строку кадра. Тот же
    проверенный знак (см. docstring `estimate_x_from_mask`): БОЛЬШЕМУ X
    соответствует МЕНЬШАЯ строка.

    `n_cols` — число столбцов кадра (ось, к которой Webots привязывает
    fieldOfView). Если None — предполагается квадратный кадр (n_cols=n_rows),
    что неверно для текущего рига; передавать явно. f_px считается через n_cols
    (см. config/layout.yaml cv_rig docstring про привязку FOV к WIDTH)."""
    if n_cols is None:
        n_cols = n_rows
    f_px = (n_cols / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    scale_m_per_px = camera_distance_m_ / f_px
    return n_rows / 2.0 - (x_world - cv_rig_x) / scale_m_per_px


def bbox_center_and_size(bbox: tuple[int, int, int, int],
                          margin_factor: float) -> tuple[float, float, float]:
    """Центр bbox (cx, cy) + "естественный" размер объекта с запасом
    (`max(w,h) * (1 + 2*margin_factor)`) — квадратное окно кропа строится
    вокруг ЭТОГО центра, размером в размер САМОГО КРУПНОГО такого окна по
    всей сетке (см. docstring модуля), не размером именно этого bbox."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    natural_size = max(w, h) * (1.0 + 2.0 * margin_factor)
    return cx, cy, natural_size


def fallback_center_and_size(row_center_px: float, col_center_px: float,
                             camera_distance_m_: float, fov_deg: float,
                             frame_shape: tuple[int, int],
                             worst_case_extent_x: float,
                             worst_case_transverse_radius: float) -> tuple[float, float, float]:
    """Геометрический центр+размер под НАИХУДШИЙ габарит ТЗ (та же пара
    чисел, что `cv_moments.py::compute_delta_max` использует для окон
    старт/центр/конец) — когда bbox недоступен (фон не откалиброван для
    камеры, объект неотличим от фона по HSV и т.п.). `row_center_px`/
    `col_center_px` — ожидаемый центр объекта в кадре (`expected_pixel_row`
    для строки; для столбца, при отсутствии бокового смещения по построению
    рига, разумный дефолт — центр кадра)."""
    n_rows = frame_shape[0]
    n_cols = frame_shape[1]
    # FOV привязан к WIDTH (n_cols), см. config/layout.yaml cv_rig docstring.
    f_px = (n_cols / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    px_per_m = f_px / camera_distance_m_
    natural_size = 2.0 * max(worst_case_extent_x, worst_case_transverse_radius) * px_per_m
    return col_center_px, row_center_px, natural_size


def crop_square_padded(frame: np.ndarray, cx: float, cy: float, side: float,
                        pad_value: int = 128) -> np.ndarray:
    """Квадратный `side`×`side` кроп вокруг (cx, cy) в НАТИВНЫХ пикселях кадра.
    Часть окна за границей кадра — паддинг серым (128), НЕ неравномерный
    clip — гарантирует ОДИНАКОВЫЙ side×side для каждой ячейки независимо от
    того, насколько объект близко к краю (ожидаемо на старте/конце, см.
    `cv_moments.py::mask_touches_row_edge`)."""
    side_i = max(1, round(side))
    h, w = frame.shape[:2]
    x0 = round(cx - side_i / 2.0)
    y0 = round(cy - side_i / 2.0)
    canvas = np.full((side_i, side_i, frame.shape[2]), pad_value, dtype=frame.dtype)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(w, x0 + side_i), min(h, y0 + side_i)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        canvas[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = \
            frame[src_y0:src_y1, src_x0:src_x1]
    return canvas


def _crop_and_resize_square(frame: np.ndarray, cx: float, cy: float, native_side: float,
                             cell_px: int, scale: float) -> np.ndarray:
    square = crop_square_padded(frame, cx, cy, native_side)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(square, (cell_px, cell_px), interpolation=interp)


def _downscale_to_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Уменьшает копию под SAM3 — НИКОГДА не увеличивает (native-разрешение
    для classical CV/torchhull остаётся отдельным, полноразмерным каналом)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    factor = max_dim / longest
    new_size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


@dataclass
class GridCell:
    camera: str
    moment: str
    crop_x: int      # верхний левый угол native-окна (может быть <0 — см. crop_square_padded)
    crop_y: int
    crop_side: int    # сторона native-окна (одна и та же величина по всей сетке)
    frame_w: int
    frame_h: int
    scale: float
    bbox_source: str   # "contour" | "fallback"
    object_crop: np.ndarray    # всегда cell_px x cell_px
    background_crop: np.ndarray | None
    prev_crop: np.ndarray | None


@dataclass
class GridResult:
    cells: dict[str, GridCell]
    scale: float
    sam3_crops: dict[str, np.ndarray]


# Коэффициент уменьшения кадра для дешёвой локализации объекта (Фаза 3.1).
# HSV-субтракция на копии ~×8 (по площади) дешевле full-res compute_mask —
# INTER_AREA-ресайз сам по себе подавляет шум, поэтому морфология НЕ нужна
# (на full-res она была для удаления мелких шумовых компонент; на ds8 эрозия
# 3×3 убивает тонкие объекты вроде `pen` на `side` — найдено замерами, см.
# заметку задачи "Оптимизировать скорость..."). `min_area_px` пересчитывается
# пропорционально площади (×f²), фильтр шума по площади остаётся.
_DOWNSCALE_FACTOR = 8


def _downscale_components(
    frame: np.ndarray, bg_frame: np.ndarray, bs: dict, factor: int = _DOWNSCALE_FACTOR,
    v_threshold: float | None = None,
) -> list[tuple[tuple[int, int, int, int], int]]:
    """Общее ядро для `_downscale_bbox_fast` (одна лучшая компонента для
    сборки сетки) и `cv_top_tracking.py` (нужны ВСЕ компоненты —
    многообъектный кадр top-камеры, см. заметку задачи "CV-триггер момента по
    top-локализации..."). HSV-субтракция фона на уменьшенной копии кадра
    (Фаза 3.1) — те же пороги H/S, что в `cv_localization.compute_mask`, без
    морфологии (см. docstring `_DOWNSCALE_FACTOR`). Возвращает [(bbox в
    НАТИВНЫХ координатах, ds8-площадь), ...] для компонент, прошедших
    `min_area_px`, по убыванию площади.

    `v_threshold` (опционально, по умолчанию None = не проверяется): для
    `_downscale_bbox_fast`/сборки сетки сознательно опущен (см. её docstring —
    там объект уже локализован hint'ом, задача только в размере/центре кропа).
    Для `cv_top_tracking.py` — ОБЯЗАТЕЛЕН: там детекция сама решает, найден
    ли объект вообще (заменяет физику), без него нейтрально-серый `cylinder`
    (тот же пограничный случай, что в `cv_localization.compute_mask`) не
    детектируется по H/S вовсе — 0 компонент, CV-триггер никогда не
    срабатывает (найдено живым прогоном на восстановленной симуляции:
    `cylinder` дал 0/9 моментов вместо 9/9).

    `bleed_ds_px` (1): расширяет bbox на 1px в ds8-координатах (= factor px в
    native) с каждой стороны перед возвратом. Компенсирует сужение bbox от
    INTER_AREA-сглаживания границы (порог HSV срабатывает чуть внутри
    реального контура объекта), без которого тонкие/длинные объекты (pen на
    diag: 306×138px) получают bbox на ~8px уже full-res и край объекта
    попадает в паддинг кропа. Подобрано замерами на pen/box_large: +1px в ds8
    покрывает bleed без существенного увеличения окна (native_side +16px
    = +2% для box_large, +5% для pen — в пределах margin_factor=0.15)."""
    bleed_ds_px = 1
    small = cv2.resize(frame, None, fx=1.0 / factor, fy=1.0 / factor,
                       interpolation=cv2.INTER_AREA)
    small_bg = cv2.resize(bg_frame, None, fx=1.0 / factor, fy=1.0 / factor,
                          interpolation=cv2.INTER_AREA)
    hsv1 = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv0 = cv2.cvtColor(small_bg, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = np.minimum(np.abs(hsv1[..., 0] - hsv0[..., 0]),
                    180.0 - np.abs(hsv1[..., 0] - hsv0[..., 0]))
    ds = np.abs(hsv1[..., 1] - hsv0[..., 1])
    fg = (dh > bs["h_threshold"]) | (ds > bs["s_threshold"])
    if v_threshold is not None:
        dv_signed = hsv1[..., 2] - hsv0[..., 2]   # знак важен — см. cv_localization.py docstring
        fg = fg | (dv_signed > v_threshold)
    fg = fg.astype(np.uint8) * 255
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area_ds = bs["min_area_px"] / (factor * factor)
    small_h, small_w = small.shape[:2]
    results = []
    for i in range(n - 1):
        if areas[i] < min_area_ds:
            continue
        idx = i + 1
        x = stats[idx, cv2.CC_STAT_LEFT] - bleed_ds_px
        y = stats[idx, cv2.CC_STAT_TOP] - bleed_ds_px
        w = stats[idx, cv2.CC_STAT_WIDTH] + 2 * bleed_ds_px
        h = stats[idx, cv2.CC_STAT_HEIGHT] + 2 * bleed_ds_px
        # clip to small frame bounds (negative после bleed — норма, upscale ниже)
        x = max(0, x); y = max(0, y)
        w = min(w, small_w - x); h = min(h, small_h - y)
        bbox_native = (int(x * factor), int(y * factor), int(w * factor), int(h * factor))
        results.append((bbox_native, int(areas[i])))
    results.sort(key=lambda r: -r[1])
    return results


def _downscale_bbox_fast(
    frame: np.ndarray, bg_frame: np.ndarray, bs: dict, factor: int = _DOWNSCALE_FACTOR,
    expected_row_hint: float | None = None, v_threshold: float | None = None,
) -> tuple[int, int, int, int] | None:
    """Быстрый bbox объекта на уменьшенной копии кадра (Фаза 3.1) — вместо
    `compute_mask` на full-res 2592×1944. Один "лучший" bbox из
    `_downscale_components` — по умолчанию наибольший по площади.

    `expected_row_hint` (опционально, НАТИВНЫЕ пиксели): если в кадре
    НЕСКОЛЬКО компонент (зеркальные камеры box_large — отражение объекта в
    зеркале даёт 2 крупных контура: реальный объект + отражение), выбирает
    компонент, чей центроид ближе к ожидаемой строке объекта, а не наибольший
    по площади. Найдено живым прогоном: на `top_mirror`/`diag_mirror` в
    момент `center` видны ДВА больших контура — реальный объект (вверху) и
    отражение (внизу); без подсказки argmax берёт отражение (большая
    площадь, т.к. отражение в кадре зеркальной камеры крупнее/контрастнее), и
    кроп уходит за нижний край кадра. Подсказка — это НЕ ground-truth физики
    (недоступный в реальной системе), а геометрия РИГА: момент `center`
    определён как прохождение объектом `cv_rig.x` → `expected_pixel_row` даёт
    ~центр кадра для всех камер, независимо от физики конкретного объекта.
    Для моментов старт/конец подсказка = None (объект у края, ожидаемая
    строка по геометрии менее надёжна); выбор по наибольшей площади там
    работает (один объект в кадре, нет конкуренции с отражением).

    `v_threshold` (опционально, по умолчанию None = как раньше, без V):
    для ПЕРВИЧНОЙ локализации объекта БЕЗ предварительного hint'а (не для
    уточнения кропа при уже известной позиции, для чего эта функция
    исходно писалась) без V нейтрально-серый объект (`cylinder`) может НЕ
    дать вообще ни одной qualifying-компоненты на part кадра — найдено
    живым прогоном (`sim/cv_grid_v2.py::compute_target_hint_from_top`):
    единственный "кандидат" на `top` при старте/конце оказался статическим
    шумовым артефактом кадра (одинаковый bbox на start И end одновременно —
    не может быть движущимся объектом), а реальный цилиндр не прошёл порог
    вовсе — hint тут бессилен (нечего выбирать, кандидат один)."""
    components = _downscale_components(frame, bg_frame, bs, factor, v_threshold=v_threshold)
    if not components:
        return None
    if expected_row_hint is not None and len(components) > 1:
        def _row_dist(item):
            (x, y, w, h), _area = item
            return abs((y + h / 2.0) - expected_row_hint)
        bbox, _area = min(components, key=_row_dist)
    else:
        bbox, _area = components[0]   # уже отсортированы по убыванию площади
    return bbox


def assemble_grid(
    object_frames: dict[str, np.ndarray],
    prev_frames: dict[str, np.ndarray],
    background_frames: dict[str, np.ndarray],
    positions_x: dict[str, float],
    rig_cfg: dict,
    cell_px: int,
    sam3_max_dim: int,
    margin_factor: float = 0.15,
) -> GridResult:
    """Собирает 9-ракурсную сетку из уже сохранённых полных кадров камер.
    Каждая ячейка (`object_crop`/`background_crop`/`prev_crop`) — РОВНО
    `cell_px`×`cell_px` (см. docstring модуля).

    `object_frames`/`prev_frames`: cell_name -> BGR-кадр (`prev_frames` может
    не содержать часть/все ключи — захват "предыдущего" кадра best-effort).
    `background_frames`: camera_name -> BGR калиброванный фон (только для
    камер, где фон откалиброван — сейчас все 5 прямых+зеркальных, см.
    `Sorter.__init__`/`_bg_calib_countdowns`; ключ может отсутствовать).
    `positions_x`: cell_name -> физическая X объекта В МОМЕНТ ЭТОГО кадра
    (ground-truth из физики симулятора). В РЕАЛЬНОЙ системе этого источника
    не будет — положение объекта в кадре должен определять сам CV. Поэтому
    `positions_x` НЕ используется для центра кропа (только сохраняется как
    метаданные для совместимости с живым HTTP-вызовом); обе координаты
    (cx, cy) берутся из дешёвой ds8-локализации. Если bbox не найден —
    fallback на центр кадра + геометрический размер `worst_case_extent`.

    Центр и размер кропа (Фаза 3.1, оптимизация скорости):
    - cx И cy: обе координаты из центроида bbox на уменьшенной копии ×8
      (`_downscale_bbox_fast`) — без зависимости от физики/ground-truth,
      работает идентично в симуляции и реальной системе. Полный
      `compute_mask` не вызывается — он избыточен здесь (отработал в
      детекторе моментов Фазы 2.2 для триггера, здесь нужен только центр+размер).
    - expected_row_hint для выбора компонента на зеркальных камерах: в
      момент `center` объект в `cv_rig.x` → `expected_pixel_row` даёт ~центр
      кадра (геометрия рига, не физика конкретного объекта). Без подсказки
      на зеркальных камерах видны реальный объект + отражение, argmax
      берёт отражение (большая площадь) → кроп уходит за край. Подсказка
      только для `center` (на старте/конце объект у края, ожидаемая строка
      менее надёжна, выбор по площади работает).
    - natural_size: из того же ds8 bbox (адаптивный, per-object — сохраняет
      разрешение, в отличие от фиксированного `worst_case_extent`); fallback
      = геометрический `worst_case_extent` (макс. габарит ТЗ), если bbox
      не найден (тонкий объект на ds8, шум — нереально для ТЗ мин. 10мм).
    """
    bs = rig_cfg["background_subtraction"]
    md = rig_cfg["moment_detection"]
    fov = rig_cfg["fov_deg"]
    cv_rig_x = rig_cfg["x"]

    centers: dict[str, tuple[float, float]] = {}
    natural_sizes: dict[str, float] = {}
    sources: dict[str, str] = {}
    for cell_name in CELL_NAMES:
        camera, moment = CELL_CAMERA_MOMENT[cell_name]
        frame = object_frames[cell_name]
        shape = frame.shape[:2]
        dist = camera_distance_m(rig_cfg, camera)
        # Подсказка для выбора контура на зеркальных камерах в момент center:
        # геометрия рига (не физика объекта) — момент center = объект в cv_rig.x
        # → expected_pixel_row даёт ~центр кадра. На старт/конец None (выбор
        # по площади — один объект в кадре, конкуренции с отражением нет).
        hint = None
        if moment == "center":
            hint = expected_pixel_row(cv_rig_x, cv_rig_x, dist, fov, shape[0], shape[1])

        bbox = None
        bg_frame = background_frames.get(camera)
        if bg_frame is not None:
            bbox = _downscale_bbox_fast(frame, bg_frame, bs, expected_row_hint=hint)

        if bbox is not None:
            cx, cy, natural = bbox_center_and_size(bbox, margin_factor)
            sources[cell_name] = "ds8"
        else:
            cx, cy, natural = fallback_center_and_size(
                shape[0] / 2.0, shape[1] / 2.0, dist, fov, shape,
                md["worst_case_extent_x"], md["worst_case_transverse_radius"])
            sources[cell_name] = "geometry"
        centers[cell_name] = (cx, cy)
        natural_sizes[cell_name] = natural

    # ОДИН общий (native, до масштаба) размер квадратного окна на всю сетку —
    # см. docstring модуля.
    native_side = max(natural_sizes.values())
    scale = cell_px / native_side

    cells: dict[str, GridCell] = {}
    sam3_crops: dict[str, np.ndarray] = {}
    for cell_name in CELL_NAMES:
        camera, moment = CELL_CAMERA_MOMENT[cell_name]
        cx, cy = centers[cell_name]

        object_crop = _crop_and_resize_square(object_frames[cell_name], cx, cy, native_side,
                                               cell_px, scale)

        bg_frame = background_frames.get(camera)
        background_crop = (_crop_and_resize_square(bg_frame, cx, cy, native_side, cell_px, scale)
                            if bg_frame is not None else None)

        prev_frame = prev_frames.get(cell_name)
        prev_crop = (_crop_and_resize_square(prev_frame, cx, cy, native_side, cell_px, scale)
                     if prev_frame is not None else None)

        frame_h, frame_w = object_frames[cell_name].shape[:2]
        cells[cell_name] = GridCell(
            camera=camera, moment=moment,
            crop_x=round(cx - native_side / 2.0), crop_y=round(cy - native_side / 2.0),
            crop_side=round(native_side), frame_w=frame_w, frame_h=frame_h,
            scale=scale, bbox_source=sources[cell_name], object_crop=object_crop,
            background_crop=background_crop, prev_crop=prev_crop)
        sam3_crops[cell_name] = _downscale_to_max_dim(object_crop, sam3_max_dim)

    return GridResult(cells=cells, scale=scale, sam3_crops=sam3_crops)


def cells_metadata(cells: dict[str, GridCell]) -> dict:
    """Метаданные ячеек без пиксельных массивов — вход для Фазы 5 (сдвиг
    cx/cy на ячейку: crop_x/crop_y/scale достаточно, чтобы восстановить K
    кропа при общем для сетки fx=fy)."""
    return {
        name: {
            "camera": c.camera, "moment": c.moment,
            "crop_x": c.crop_x, "crop_y": c.crop_y, "crop_side": c.crop_side,
            "frame_w": c.frame_w, "frame_h": c.frame_h,
            "scale": c.scale, "bbox_source": c.bbox_source,
            "has_background": c.background_crop is not None,
            "has_prev": c.prev_crop is not None,
        }
        for name, c in cells.items()
    }


def render_composite(cells: dict[str, GridCell], field: str = "object_crop") -> np.ndarray:
    """Подписанная плитка 3×3 (не основной канал данных — только для
    визуальной проверки Фазы 3, см. заметку задачи "приёмка"). Все ячейки
    гарантированно одного размера (`assemble_grid`), поэтому раскладка —
    простое тайлирование без паддинга под разный размер.

    `field` — какой кроп ячейки рендерить: "object_crop" (по умолчанию) или
    "background_crop" (та же сетка окон/масштаба, но фон — для 3D-
    реконструкции нужна ПАРА кропов объект+фон с идентичными окнами, чтобы
    вычитание давало силуэт, см. заметку задачи "3D-реконструкция объекта по
    кропам сетки ракурсов")."""
    tile_px = next(iter(cells.values())).object_crop.shape[0]
    pad = 4
    n_rows = len(_COMPOSITE_LAYOUT)
    n_cols = len(_COMPOSITE_LAYOUT[0])
    canvas = np.full((n_rows * (tile_px + pad) + pad, n_cols * (tile_px + pad) + pad, 3),
                      40, dtype=np.uint8)
    for ri, row in enumerate(_COMPOSITE_LAYOUT):
        for ci, name in enumerate(row):
            crop = getattr(cells[name], field)
            y0 = ri * (tile_px + pad) + pad
            x0 = ci * (tile_px + pad) + pad
            canvas[y0:y0 + tile_px, x0:x0 + tile_px] = crop
            cv2.putText(canvas, name, (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
