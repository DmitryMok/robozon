"""Детектор моментов старт/центр/конец по CV-камерам (Фаза 2.2 CV-пайплайна).

Постановка — заметка задачи "Детектор моментов старт-центр-конец по
CV-камерам (Фаза 2.2, CV-пайплайн Webots)". Три момента вдоль ленты снимаются
`top` и `side` НЕЗАВИСИМО друг от друга (у каждой свой старт/конец по СВОЕЙ
границе FOV — синхронизация по одной камере обрезала бы торцевой ракурс у
другой раньше времени, решение пользователя); `diag` даёт только центр.
Зеркальные `top_mirror`/`diag_mirror` (Фаза 3, см. заметку задачи "Сборка
сетки 9 ракурсов...") тоже участвуют только в моменте "центр", своего
старт/конец не имеют — итог 9 кадров/объект (5 центр + 2 старт + 2 конец).

Окна FOV (`delta_max`) — положение ЦЕНТРА объекта от `cv_rig.x` на
старте/конце, формула из `roboson_tools/silhouette/camera.py::belt_offset`
(см. `03 Work/Расчёт перекладки CV-зоны и ramp.md` §3-4): худший ok/round
габарит по ТЗ (450×320×320мм) при случайной ориентации, не габарит
конкретного объекта — иначе окно зависело бы от ещё не определённой на этом
шаге категории (реальная система узнаёт габариты только ПОСЛЕ 3D-реконструкции,
т.е. после того, как объект уже проехал зону CV).

Момент фиксируется по ИЗВЕСТНОЙ (в симуляторе — из физики Webots) X-позиции
центра объекта, пересекающей заранее посчитанный порог — не по факту
"маска целиком видна"/"обрезана": на старте/конце объект в кадре ЧАСТИЧНО, и
это ожидаемо (см. `mask_touches_row_edge` ниже — используется только для
офлайн-сверки, не как продакшн-триггер в этой реализации). Практический
триггер по физике не подменяет детектор по маске, а проверяется против него
(`estimate_x_from_mask`) — реальная система (без доступа к физике) обязана
уметь работать по одной лишь маске.
"""
import math
from dataclasses import dataclass, field

import numpy as np

# Камеры со своим независимым стартом/концом (см. постановку выше). `diag` —
# только центр, здесь не участвует как отдельный порог. Зеркальные камеры
# тоже не имеют своего старт/конец (снимаются только вместе с "центром").
DELTA_MAX_CAMERAS = ("top", "side")
# Камеры, снимающие момент "центр" (Фаза 3 — добавлены зеркальные
# top_mirror/diag_mirror, см. заметку задачи "Сборка сетки 9 ракурсов...";
# зеркало отражает ИМЕННО top+diag, не top+side — см. config/layout.yaml:
# cv_rig.cameras и заметку задачи "Рассчитать геометрию 3-камерного рига...").
# Итог — 9 ракурсов: 5 в центре (эта константа) + 2 старт + 2 конец
# (DELTA_MAX_CAMERAS).
CENTER_CAMERAS = ("top", "side", "diag", "top_mirror", "diag_mirror")


def compute_delta_max(distance_m: float, fov_deg: float, margin_factor: float,
                       worst_case_extent_x: float, worst_case_transverse_radius: float) -> float:
    """Положение центра объекта (м) от точки наблюдения на границе безопасного
    окна кадра — та же формула, что `roboson_tools/silhouette/camera.py::
    belt_offset` (перенесена сюда как есть, не импортируется — `roboson_tools`
    отдельный репозиторий/venv, не зависимость этого проекта)."""
    half_fov = math.radians(fov_deg) / 2.0
    safe_depth = distance_m - worst_case_transverse_radius
    return math.tan(half_fov) * margin_factor * safe_depth - worst_case_extent_x


def moment_thresholds(rig: dict) -> dict[str, float]:
    """5 X-порогов момента: start_top, start_side, center, end_side, end_top
    (в порядке возрастания X при движении объекта вдоль ленты)."""
    md = rig["moment_detection"]
    cams = rig["cameras"]
    dm = {
        name: compute_delta_max(cams[name]["distance"], rig["fov_deg"], md["margin_factor"],
                                 md["worst_case_extent_x"], md["worst_case_transverse_radius"])
        for name in DELTA_MAX_CAMERAS
    }
    x = rig["x"]
    return {
        "start_top": x - dm["top"],
        "start_side": x - dm["side"],
        "center": x,
        "end_side": x + dm["side"],
        "end_top": x + dm["top"],
    }


def threshold_captures() -> dict[str, list[tuple[str, str]]]:
    """Порог -> список (камера, момент), которые нужно снять при его
    пересечении. "center" — общий порог для 5 камер (снимаются одним тиком,
    см. CENTER_CAMERAS)."""
    return {
        "start_top": [("top", "start")],
        "start_side": [("side", "start")],
        "center": [(cam, "center") for cam in CENTER_CAMERAS],
        "end_side": [("side", "end")],
        "end_top": [("top", "end")],
    }


@dataclass
class MomentScheduler:
    """Отслеживает пересечение порогов `moment_thresholds` по каждому
    отслеживаемому объекту. Объект на ленте A движется только в сторону +X
    (без отката) — простое "было меньше порога, стало >=" ловит момент без
    пропусков/повторов и не требует истории длиннее одного предыдущего тика."""
    thresholds: dict[str, float]
    captures: dict[str, list[tuple[str, str]]]
    _last_x: dict = field(default_factory=dict)
    _fired: dict = field(default_factory=dict)

    def step(self, positions_x: dict[int, float]) -> list[tuple[int, str, str]]:
        """positions_x: uid -> текущая X. Возвращает [(uid, camera, moment), ...]
        для порогов, впервые пересечённых на этом вызове."""
        events: list[tuple[int, str, str]] = []
        for uid, x in positions_x.items():
            if x is None:
                continue
            prev = self._last_x.get(uid)
            fired = self._fired.setdefault(uid, set())
            if prev is not None:
                for name, threshold in self.thresholds.items():
                    if name in fired or prev >= threshold or x < threshold:
                        continue
                    fired.add(name)
                    events.extend((uid, camera, moment) for camera, moment in self.captures[name])
            self._last_x[uid] = x
        return events

    def forget(self, uid: int) -> None:
        """Очистка состояния объекта, покинувшего трекинг (доставлен/потерян) —
        не обязательна для корректности (все пороги лежат в зоне CV, задолго
        до доставки), только чтобы словари не росли неограниченно за сессию."""
        self._last_x.pop(uid, None)
        self._fired.pop(uid, None)


def mask_row_bbox(mask: np.ndarray) -> tuple[int, int] | None:
    """(row_min, row_max) непустых пикселей маски — None если маска пуста.
    "Строка" — не случайная ось: `up`=(1,0,0) (мировая X, вдоль ленты) для
    ВСЕХ камер рига (`tools/gen_world.py::_camera_frame`), а `resolution` в
    конфиге — [n_rows(вдоль ленты), n_cols(поперёк)] — поэтому положение
    объекта вдоль ленты всегда читается по оси строк, независимо от азимута
    конкретной камеры."""
    ys, _xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max())


def mask_touches_row_edge(mask: np.ndarray, tolerance_px: int) -> bool:
    """True, если связная область маски касается верхней ИЛИ нижней границы
    кадра (в допуске `tolerance_px`) — сигнал "объект частично обрезан по оси
    ленты", ожидаемый на старте/конце, не ожидаемый в центре (см. docstring
    модуля и Приёмку задачи, п.4 "явная проверка отсутствия двойных/
    пропущенных срабатываний")."""
    bbox = mask_row_bbox(mask)
    if bbox is None:
        return False
    row_min, row_max = bbox
    h = mask.shape[0]
    return row_min <= tolerance_px or row_max >= (h - 1 - tolerance_px)


def estimate_x_from_mask(mask: np.ndarray, camera_distance_m: float, fov_deg: float,
                          cv_rig_x: float) -> float | None:
    """Независимая (БЕЗ доступа к физике) оценка мировой X объекта по одной
    маске: центроид bbox по строкам, домноженный на масштаб мм/px камеры
    (то же ортографическое приближение "d >> объект", что и
    `tools/report_cv_masks.py::camera_scale_and_axes` — оправдано той же
    геометрией рига, FOV 68°, дистанции 1.1-2.0м против объектов до 0.45м).
    None, если маска пуста (объект вне кадра — не должно происходить в
    заявленных окнах старт/центр/конец, сигнал ошибки калибровки порогов).

    Знак направления — ПРОВЕРЕН реальным захватом (`tools/capture_moments.py`
    на живом Webots, box_small): изначальная версия (без минуса) давала
    систематическое (не шумовое) огромное расхождение при старте/конце —
    оценка X уезжала в СТОРОНУ, ПРОТИВОПОЛОЖНУЮ фактическому движению объекта
    (тот же класс ошибки о переворотах осей, что не раз уже случался в этом
    проекте, см. `tools/gen_world.py::_camera_frame` docstring). Причина:
    камера смотрит по forward, а её локальная ось Z ("up", см. `_camera_frame`)
    — это МИРОВАЯ +X; при стандартном формировании изображения точка со
    смещением в СВОЮ +up (к большему мировому X) проецируется БЛИЖЕ К ВЕРХУ
    кадра, т.е. на МЕНЬШУЮ строку — поэтому меньшая строка соответствует
    БОЛЬШЕМУ X, отсюда минус перед смещением строки ниже. Подтверждено на
    моменте "центр" (где ортографическое приближение точнее всего): после
    исправления знака отклонение оценки от физической X упало с ~71мм до
    ~6мм на всех 3 прямых камерах box_small."""
    bbox = mask_row_bbox(mask)
    if bbox is None:
        return None
    row_min, row_max = bbox
    row_center = (row_min + row_max) / 2.0
    n_rows = mask.shape[0]
    n_cols = mask.shape[1]
    # FOV привязан к WIDTH (n_cols), см. config/layout.yaml cv_rig docstring.
    f_px = (n_cols / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    scale_m_per_px = camera_distance_m / f_px
    return cv_rig_x - (row_center - n_rows / 2.0) * scale_m_per_px
