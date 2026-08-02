"""Оркестрация расчётов — единая точка входа для GUI и headless CLI.

Вся расчётная логика собрана здесь в виде чистых функций: GUI и CLI используют её одинаково,
не дублируя пайплайн "ориентация -> силуэты -> Visual Hull -> метрики".

Критерий формы (см. docs/method.md): объект имеет потенциал к перекату, если существует ось,
вдоль которой ПРОЕКЦИЯ восстановленной по силуэтам формы близка к кругу (k = Rin/Rout >= 0.8).
Проверка отдельных сечений оставлена как справочная: круглое сечение само по себе перекат не
означает (наклонное сечение конуса — круг: скос горлышка бутылки давал ложные срабатывания).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from ..geometry.mesh_io import Mesh, bounding_box
from ..geometry.transform import apply_orientation
from ..metrics.roundness import RoundnessResult, compute_projection_roundness, compute_roundness
from ..silhouette import camera as camera_silhouette
from ..silhouette.analytical import build_silhouette, common_extent, view_axes
from ..silhouette.base import Silhouette
from ..visual_hull.reconstruct import (
    ReconstructedShape,
    band_half_length,
    reconstruct_section,
    reconstruct_section_from_silhouettes,
    reconstruct_shape_from_silhouettes,
)
from ..visual_hull import exact_polyhedral_hull, polytope_hull
from ..visual_hull.vertical_prism_fit import PrismFitResult, fit_vertical_prism

# torchhull — ОПЦИОНАЛЬНАЯ GPU-зависимость (CUDA Toolkit + venv-cv, см.
# visual_hull/torchhull_adapter.py и bench_torchhull.py). Мягкий импорт: если torch/torchhull
# нет в окружении — `_TORCHHULL_AVAILABLE = False`, и `check_model_roll_g4` при
# `use_camera_silhouettes=True` использует старый путь (`reconstruct_shape_from_silhouettes`).
# На GPU-окружении (robozon/.venv-cv) — `_TORCHHULL_AVAILABLE = True`, и G4 в Camera Mode
# использует GPU visual hull (sparse voxel octree + marching cubes), который на параллаксе
# даёт 18/18 точность vs 16/18 у band-пересечения (см. bench_torchhull.py).
_TORCHHULL_AVAILABLE = False
try:
    import torch as _torch  # noqa: F401
    import torchhull as _torchhull  # noqa: F401
    from ..visual_hull.torchhull_adapter import build_transforms_batch, visual_hull_points
    _TORCHHULL_AVAILABLE = True
except Exception:
    pass


@dataclass
class Orientation:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class EvaluationResult:
    view_angles_deg: list[float]
    axis_pos: float
    silhouettes: dict[float, Silhouette]
    polygon_coords: list[tuple[float, float]] | None  # None, если сечение вырождено
    roundness: RoundnessResult | None
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    # Реконструкция по силуэтам: стопка сечений [(x, контур), ...] и габариты (X, Y, Z);
    # None — объект не виден в силуэтах (вырожденный случай)
    hull_slices: list[tuple[float, list[tuple[float, float]]]] | None
    hull_dims: tuple[float, float, float] | None
    mesh_dims: tuple[float, float, float]  # истинные габариты меша, для сравнения


@dataclass
class ConfigCompareEntry:
    view_angles_deg: list[float]
    roundness: RoundnessResult | None


@dataclass
class SectionCheckHit:
    """Одно сечение реконструкции — справочная информация в режиме "Проверить модель"."""

    axis_pos: float
    k: float
    passed: bool
    polygon_coords: list[tuple[float, float]]


@dataclass
class AxisHit:
    """Одно проверенное направление оси переката (проекция формы вдоль этой оси)."""

    axis_azim_deg: float  # азимут оси: угол от X в плоскости XY
    axis_elev_deg: float  # подъём оси над плоскостью XY
    k: float
    passed: bool


@dataclass
class BoxHit:
    """Один проверенный кандидат минимального охватывающего бокса: направление одной из трёх
    осей бокса (azim/elev — как у AxisHit); две другие оси решаются точно через
    cv2.minAreaRect в плоскости, перпендикулярной этому направлению (см. _box_metrics)."""

    axis_azim_deg: float
    axis_elev_deg: float
    dims: tuple[float, float, float]  # (толщина вдоль оси, стороны найденного 2D-прямоугольника)
    volume: float


@dataclass
class RollCheckResult:
    """Итог "Проверить модель": потенциал к перекату по круглой проекции формы,
    восстановленной из силуэтов (см. check_model_roll)."""

    dims: tuple[float, float, float] | None  # габариты по силуэтам В ТЕКУЩЕЙ ориентации (X, Y, Z)
    mesh_dims: tuple[float, float, float]  # истинные габариты меша, для сравнения
    checked_directions: int
    axis_step_deg: float
    best: AxisHit | None  # направление с максимальным k
    best_axis_dir: tuple[float, float, float] | None  # то же направление, единичный вектор
    best_projection_coords: list[tuple[float, float]] | None  # контур лучшей проекции
    best_roundness: RoundnessResult | None  # метрики лучшей проекции (для отрисовки окружностей)
    found_round: bool  # есть направление с k >= порога -> объект может катиться
    section_hits: list[SectionCheckHit]  # справочно: круглость сечений реконструкции
    round_section_count: int
    # Истинные (минимальные) габариты — НЕ зависят от текущего Roll/Pitch/Yaw, в отличие от
    # `dims` выше: минимальный охватывающий бокс восстановленной формы, найденный тем же
    # перебором направлений (см. _box_metrics/BoxHit). Для сортировки по размерной категории
    # сравнивать нужно именно эту тройку (или её отсортированный вариант), а не `dims`.
    true_dims: tuple[float, float, float] | None
    true_dims_axis_dir: tuple[float, float, float] | None  # одна из осей найденного бокса
    elapsed_seconds: float  # анализ по готовым силуэтам (реконструкция + перебор осей)


def default_axis_pos(mesh: Mesh) -> float:
    """Позиция вдоль roll-оси X по умолчанию — середина bounding box ориентированного меша."""
    lo, hi = bounding_box(mesh.vertices)
    return float((lo[0] + hi[0]) / 2.0)


def build_silhouette_set(
    mesh: Mesh,
    view_angles_deg: list[float],
    resolution_px: int,
    use_camera: bool = False,
    camera_fov_deg: float = 0.0,
    camera_distances: dict[float, float] | None = None,
    belt_position: str = "center",
) -> dict[float, Silhouette]:
    """Силуэты по всем углам.

    По умолчанию (`use_camera=False`) — Analytical Mode, единый масштаб (общий extent ->
    одинаковый px_per_unit). `use_camera=True` — те же ракурсы строятся перспективной
    (pinhole, Camera Mode, см. silhouette/camera.py) моделью вместо ортографической; общего
    extent не нужно — масштаб общий по построению (один и тот же fov/resolution для всех
    углов, но КАЖДЫЙ угол — своя дистанция, см. `camera_distances`/`core/camera_rig.py`).
    Используется, чтобы оценить, насколько перспектива реальной камеры искажает габариты по
    силуэтам относительно ортографического ядра (см. core/experiment.evaluate,
    check_model_roll). `camera_distances` обязателен при `use_camera=True` (иначе `ValueError`
    — без скрытого фоллбека на 0.0); `KeyError` при отсутствии угла в карте."""
    if use_camera:
        if camera_distances is None:
            raise ValueError("camera_distances обязателен при use_camera=True")
        return {
            angle: camera_silhouette.build_silhouette(
                mesh,
                angle,
                resolution_px,
                fov_deg=camera_fov_deg,
                camera_distance=camera_distances[angle],
                belt_position=belt_position,
            )
            for angle in view_angles_deg
        }
    extent = common_extent(mesh, view_angles_deg)
    return {
        angle: build_silhouette(mesh, angle, resolution_px, extent_override=extent)
        for angle in view_angles_deg
    }


def evaluate(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    axis_pos: float | None = None,
    resolution_px: int = 1024,
    roundness_threshold: float = 0.8,
    hull_slices_count: int = 50,
    use_camera_silhouettes: bool = False,
    camera_fov_deg: float = 0.0,
    camera_distances: dict[float, float] | None = None,
    belt_position: str = "center",
) -> EvaluationResult:
    """Полный расчёт для GUI: силуэты по каждому углу (Panel 2), сечение (Panel 3),
    метрики (Panel 4), реконструкция формы и габариты по силуэтам (Panel 1/4).

    `use_camera_silhouettes=True` — силуэты строятся перспективной (Camera Mode) моделью
    вместо ортографической (см. build_silhouette_set); `mesh_dims` в результате остаётся
    истинным габаритом меша, поэтому разница с `hull_dims` показывает ошибку габаритов от
    перспективы реальной камеры при заданных `camera_fov_deg`/`camera_distances`."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )

    if axis_pos is None:
        axis_pos = default_axis_pos(oriented)

    silhouettes = build_silhouette_set(
        oriented,
        view_angles_deg,
        resolution_px,
        use_camera=use_camera_silhouettes,
        camera_fov_deg=camera_fov_deg,
        camera_distances=camera_distances,
        belt_position=belt_position,
    )

    bbox_min, bbox_max = bounding_box(oriented.vertices)
    half_length = band_half_length(bbox_min, bbox_max)
    polygon = reconstruct_section_from_silhouettes(silhouettes, axis_pos, half_length)

    roundness = None
    polygon_coords = None
    if polygon is not None:
        polygon_coords = list(polygon.exterior.coords)
        roundness = compute_roundness(
            np.array(polygon.exterior.coords[:-1], dtype=np.float64),
            resolution_px=resolution_px, threshold=roundness_threshold
        )

    shape = reconstruct_shape_from_silhouettes(silhouettes, hull_slices_count)

    return EvaluationResult(
        view_angles_deg=list(view_angles_deg),
        axis_pos=axis_pos,
        silhouettes=silhouettes,
        polygon_coords=polygon_coords,
        roundness=roundness,
        bbox_min=tuple(bbox_min.tolist()),
        bbox_max=tuple(bbox_max.tolist()),
        hull_slices=shape.slices if shape is not None else None,
        hull_dims=shape.dims if shape is not None else None,
        mesh_dims=tuple((bbox_max - bbox_min).tolist()),
    )


def check_vertical_prism(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    fov_deg: float,
    camera_distances: dict[float, float],
    resolution_px: int,
) -> PrismFitResult | None:
    """Геометрическая проверка "это вертикальная призма" (коробка/N-угольная колонна) + точные
    габариты по верхней+боковым камерам (см. visual_hull/vertical_prism_fit.py) — ИСКЛЮЧИТЕЛЬНО
    Camera Mode (перспектива), у Analytical Mode эта неоднозначность масштаба не возникает и
    метод ей не нужен. None — либо не сошлось (объект не похож на вертикальную призму, либо
    среди view_angles_deg нет камеры строго сверху/не хватает боковых, см. докстринг модуля).

    Единственное место в кодовой базе, где ориентированный меш сдвигается по Z: обычно оси
    центрированы по bbox объекта и это не нужно, но `fit_vertical_prism` ожидает, что Z=0 меша —
    это уровень ленты/земли (калибровочная константа рига, а не то, что выводится из силуэтов —
    как и camera_distance). Сдвиг здесь берётся из bbox САМОГО меша (единственная информация о
    "низе" объекта, которая вообще есть в этой программе — нигде в кодовой базе belt-калибровка
    отдельно не моделируется), поэтому эта проверка ЧАСТИЧНО использует истинную геометрию меша
    (только сдвиг по Z, не форму/размеры) — как и для остальных силуэтных метрик, `mesh_dims`
    остаётся отдельно посчитанным по мешу для сравнения."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    bbox_min, _bbox_max = bounding_box(oriented.vertices)
    grounded = Mesh(
        vertices=oriented.vertices - np.array([0.0, 0.0, bbox_min[2]]),
        faces=oriented.faces,
    )
    result = fit_vertical_prism(grounded, view_angles_deg, fov_deg, camera_distances, resolution_px)
    if result is None:
        return None
    # ground_z нужен только для 3D-визуализации (см. visual_hull.vertical_prism_fit.prism_mesh):
    # Panel 1 рисует oriented-меш БЕЗ сдвига по Z, а footprint_xy/height результата — в сдвинутой
    # (grounded) системе координат, в которой их и искал fit_vertical_prism.
    return replace(result, ground_z=float(bbox_min[2]))


def prism_dims(result: PrismFitResult) -> tuple[float, float, float]:
    """(сторона1, сторона2, высота) по результату check_vertical_prism/fit_vertical_prism —
    тот же формат, что hull_dims/true_dims, но не раздутые Visual Hull-конусами (см. докстринг
    check_vertical_prism): используется вместо них в GUI, когда объект прошёл геометрическую
    проверку "это вертикальная призма".

    `cv2.minAreaRect`, а НЕ наивный max-min по осям мира — та же причина, что у `_box_metrics`
    (см. её докстринг): footprint_xy лежит в мировых (X, Y) БЕЗ учёта текущего yaw объекта,
    поэтому max-min по осям даёт габарит повёрнутого прямоугольника по ДИАГОНАЛИ, а не по
    стороне (тот самый баг "короб 300×200 при 22.5° даёт 351×296", уже решённый для true_dims
    в docs/method.md, — здесь его повторили и он проявлялся резким ростом чисел при повороте)."""
    pts = np.asarray(result.footprint_xy, dtype=np.float32)
    (_cx, _cy), (side_a, side_b), _angle_deg = cv2.minAreaRect(pts)
    return (float(side_a), float(side_b), result.height)


@dataclass
class SimpleCameraDims:
    length: float
    width: float
    height: float
    top_angle_deg: float
    side_angle_deg: float


def _largest_contour_px(mask: np.ndarray) -> np.ndarray | None:
    """(N,2) пиксели (col, row) наибольшего внешнего контура маски, либо None (маска пуста)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    return contour.reshape(-1, 2).astype(np.float64)


def _sil_contour(sil: Silhouette) -> np.ndarray | None:
    """Контур силуэта: субпиксельный (`Silhouette.contour_px`, есть при supersample > 1),
    иначе — обычный бинарный по маске. Оба — float (N,2) [col, row] в нативных пикселях."""
    if sil.contour_px is not None:
        return sil.contour_px
    return _largest_contour_px(sil.mask)


def _camera_intrinsics(
    fov_deg: float, resolution_px: camera_silhouette.Resolution
) -> tuple[int, int, float, float, float]:
    """(n_rows, n_cols, f_px, cx, cy) — та же математика, что в
    `silhouette.camera.camera_frame` (fov_deg — вдоль ленты/строк, пиксели квадратные);
    для квадратного int-разрешения воспроизводит прежние формулы побитово."""
    n_rows, n_cols = camera_silhouette.frame_shape(resolution_px)
    f_px = (n_rows / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return n_rows, n_cols, f_px, n_cols / 2.0, n_rows / 2.0




_WIDEST_COLUMN_MARGIN = 1.15  # см. докстринг _widest_column — насколько заметно пик должен
# быть шире верхней ОБЛАСТИ, чтобы ему доверять больше, чем самому верху
_WIDEST_COLUMN_TOP_FRACTION = 0.25  # доля высоты у самого верха, в пределах которой ищем
# опорную ширину верха (а не берём буквально один крайний столбец, см. докстринг) — калибровано
# так, чтобы захватывать угловой артефакт ПОВЁРНУТОЙ коробки (наблюдался вплоть до ~20% высоты
# на yaw=45°, самый тяжёлый случай) и одновременно оставаться заметно уже настоящего сужения
# тарелки/мешка (на Мешок.stl сужение растянуто ощутимо больше — запас есть даже на 30%)


def _widest_column(mask: np.ndarray) -> int:
    """Индекс столбца бокового кадра с наибольшей протяжённостью по строкам — высота самой
    широкой части объекта, для "летающей тарелки"/мешка (см. докстринг check_simple_camera_dims).

    Доверяем этому пику, только если он ЗАМЕТНО (в >= `_WIDEST_COLUMN_MARGIN` раз) шире ширины
    В ОБЛАСТИ ВЕРХА объекта — иначе возвращаем верхний столбец (прежнее поведение, верно
    работавшее для коробки). "Область верха" — верхние `_WIDEST_COLUMN_TOP_FRACTION` высоты, а
    НЕ единственный крайний верхний столбец: у ПОВЁРНУТОЙ (yaw != 0/90°) коробки крайний верхний
    столбец — это буквально один верхний угол (точка), видимый под углом, его ширина в пикселях
    почти нулевая, хотя чуть ниже по высоте — там же, где полное верхнее ребро объекта уже видно
    целиком — ширина совпадает с максимумом по всему кадру. Сравнение с ОДНИМ крайним столбцом
    (более ранняя версия) давало ложный "разрыв" (полученное: до ×27) и всегда выбирало
    середину кадра вместо верха, даже для обычной коробки — баг, найденный пользователем на
    короб 300×200×200 при yaw=75° (и воспроизводимый на большинстве непрямых углов). Область в
    15% у верха достаточно узкая, чтобы не зацепить настоящий "экватор" тарелки/мешка (там
    сужение куда более плавное и растянуто на десятки процентов высоты — см. докстринг
    check_simple_camera_dims, проверено на Мешок.stl), но достаточно широкая, чтобы не попасть
    ровно в угловой артефакт одного столбца."""
    widths = mask.sum(axis=0)
    occupied = np.flatnonzero(widths)
    top_col = int(occupied.max())
    bottom_col = int(occupied.min())
    span = max(top_col - bottom_col, 1)
    top_region_start = max(int(top_col - _WIDEST_COLUMN_TOP_FRACTION * span), bottom_col)
    top_region_width = float(widths[top_region_start : top_col + 1].max())

    max_width = float(widths.max())
    if max_width < top_region_width * _WIDEST_COLUMN_MARGIN:
        return top_col
    candidates = np.flatnonzero(widths >= max_width - 0.5)  # с запасом на растровый шум
    return int(candidates.max())


def check_simple_camera_dims(
    mesh: Mesh,
    orientation: Orientation,
    fov_deg: float,
    camera_distance: float,
    resolution_px: camera_silhouette.Resolution,
    top_angle_deg: float = 90.0,
    side_angle_deg: float = 0.0,
    camera_distances: dict[float, float] | None = None,
    supersample: int = 1,
) -> SimpleCameraDims | None:
    """Простая оценка габаритов по ДВУМ фиксированным перспективным кадрам — вид сверху и вид
    сбоку — без распознавания формы и без итеративного поиска (сравни с точным, но более
    дорогим и ограниченным вертикальными призмами `visual_hull/vertical_prism_fit.py`). Длина/
    ширина — `cv2.minAreaRect` контура сверху (минимальная ширина фигуры, а не осевой bbox —
    та же причина, что у `prism_dims`/`_box_metrics`). Высота — протяжённость контура сбоку.
    None — вырожденный силуэт (объект вне кадра) с любой из двух камер.

    ДЛИНА/ШИРИНА СКОРРЕКТИРОВАНЫ (по просьбе пользователя — раз причина завышения известна,
    глубина до грани, которая формирует силуэт сверху, её можно скомпенсировать): камера сверху
    физически видит не обязательно самую верхнюю точку объекта, а ту грань, что даёт наибольший
    охват кадра — при том же реальном размере более близкая к камере грань занимает БОЛЬШЕ
    пикселей, поэтому наивный пересчёт по номинальному масштабу (как для точки на уровне земли
    Z=0) завышает длину/ширину.

    ВАЖНО (второй раунд обсуждения с пользователем — "летающая тарелка"/"мешок"): опорная
    высота для этой поправки — НЕ общая высота объекта, а высота его САМОЙ ШИРОКОЙ части. Для
    коробки (постоянное сечение) это одно и то же — верх; но для купольных/выпуклых форм
    (мешок-картофелина, летающая тарелка) самая широкая часть обычно ГДЕ-ТО В СЕРЕДИНЕ по
    высоте, а не наверху — именно она, а не узкая/скруглённая верхушка, определяет реальный
    размер силуэта сверху (см. `_widest_column`: ищется по ПРОТЯЖЁННОСТИ бокового силуэта
    построчно, а не по общему верх/низ). Обратное умножение на `(camera_distance - z_widest) /
    camera_distance` снимает почти всю ошибку (проверено на коробе: наивные 344×230 ->
    скорректированные 299×200 против истинных 301×200.5, вместо +14% ошибка падает до ~1%; для
    коробки `_widest_column` и общая высота совпадают, так что это регрессионный тест заодно и
    на "частный случай = коробка"). Это НЕ полная триангуляция, как в `check_vertical_prism`
    (там высота ищется перебором, соответствие кандидата боковым силуэтам проверяется явно) —
    здесь `z_widest` берётся из одного-единственного упрощённого измерения ниже, поэтому
    остаточная ошибка передаётся и в длину/ширину, но на порядок слабее, чем полное отсутствие
    поправки.

    ТРЕБУЕТСЯ, чтобы меш был размещён основанием на Z=0 — как и `check_vertical_prism`,
    `camera_distance` здесь это калибровочная высота камеры НАД ЛЕНТОЙ, а не над центром
    объекта; функция сама сдвигает ориентированный меш по bbox (см. код), а не полагается на
    то, что вызывающая сторона уже это сделала.

    Высота — тем же приёмом, что описан выше для длины/ширины, только для БОКОВОЙ камеры: её
    масштаб тоже зависит от глубины, а глубина — от того, НАСКОЛЬКО БЛИЗКО к боковой камере
    физически находится грань объекта, порождающая верхнюю границу силуэта. Вместо номинальной
    глубины `camera_distance` берётся глубина по Y ближнего к боковой камере края КОНТУРА
    СВЕРХУ (проекция контура на направление обзора боковой камеры, максимум) — тот же контур,
    что даёт длину/ширину. Это не полная триангуляция высоты, как в `check_vertical_prism`:
    собственная небольшая систематическая ошибка контура сверху по X/Y частично передаётся в
    высоту, но эффект от неё на порядок меньше, чем полное отсутствие поправки на глубину.

    `camera_distances` — необязательная карта {азимут -> реальная дистанция}: на риге
    «2 камеры + 1 зеркало» часть ракурсов снимается через зеркало, их оптический путь ДЛИННЕЕ
    базового `camera_distance`; отсутствующие в карте азимуты используют `camera_distance`.
    `supersample` > 1 — субпиксельные силуэты (см. silhouette/camera.build_silhouette):
    контуры берутся с точностью ~1/S px вместо ±0.5px, что резко снижает растровый шум на
    маленьких в кадре объектах (Pen)."""
    v_top, r_theta_top = view_axes(top_angle_deg)
    if abs(v_top[0]) > 1e-9 or abs(v_top[1]) > 1e-9 or abs(v_top[2] - 1.0) > 1e-9:
        raise ValueError(
            "top_angle_deg должен задавать камеру строго сверху (v_theta=(0,0,1)), обычно 90°"
        )
    v_side, _r_side = view_axes(side_angle_deg)
    if abs(v_side[2]) > 1e-9:
        raise ValueError(
            "side_angle_deg должен задавать чисто боковую камеру (v_theta без Z-компоненты, "
            "иначе столбец кадра — смесь Y и Z, а не чистая высота), обычно 0° или 180°"
        )

    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    bbox_min, _bbox_max = bounding_box(oriented.vertices)
    grounded = Mesh(
        vertices=oriented.vertices - np.array([0.0, 0.0, bbox_min[2]]),
        faces=oriented.faces,
    )
    distances = camera_distances or {}
    d_top = float(distances.get(top_angle_deg, camera_distance))
    d_side = float(distances.get(side_angle_deg, camera_distance))

    top_sil = camera_silhouette.build_silhouette(
        grounded, top_angle_deg, resolution_px, fov_deg, d_top, "center", supersample=supersample
    )
    side_sil = camera_silhouette.build_silhouette(
        grounded, side_angle_deg, resolution_px, fov_deg, d_side, "center", supersample=supersample
    )

    top_contour = _sil_contour(top_sil)
    side_contour = _sil_contour(side_sil)
    if top_contour is None or side_contour is None:
        return None

    (_cx, _cy), (side_a, side_b), _angle_deg = cv2.minAreaRect(top_contour.astype(np.float32))
    naive_length = float(side_a) / top_sil.px_per_unit
    naive_width = float(side_b) / top_sil.px_per_unit

    top_cx = top_sil.mask.shape[1] / 2.0
    u = (top_contour[:, 0] - top_cx) / top_sil.px_per_unit  # проекция на r_theta_top
    yz = np.outer(u, r_theta_top)  # (N,3) мировые смещения точек контура сверху (X-компонента 0)
    near_edge_naive = float((yz @ v_side).max())  # ближайшая к боковой камере точка контура сверху

    _n_rows, _n_cols, f_px, _cx_i, _cy_i = _camera_intrinsics(fov_deg, resolution_px)
    cols = side_contour[:, 0]  # столбец бокового кадра = высота (Z), см. view_axes/camera.py
    col_min = float(cols.min())
    col_span = float(cols.max() - col_min)
    col_widest = _widest_column(side_sil.mask)

    def _height_and_z_widest(near_edge: float) -> tuple[float, float] | None:
        side_depth = d_side - near_edge
        if side_depth <= 0:
            return None
        px_per_unit_side = f_px / side_depth
        h = col_span / px_per_unit_side
        zw = float(np.clip((col_widest - col_min) / px_per_unit_side, 0.0, h))
        return h, zw

    # ДВА ПРОХОДА, а не один (найдено пользователем — заниженная высота даже без "тарелочной"
    # эвристики выше): near_edge сам берётся из НЕОТКОРРЕКТИРОВАННОГО (завышенного по масштабу)
    # контура сверху, поэтому первый проход занижает near_edge->depth->масштаб->height. Второй
    # проход использует уже посчитанную высоту первого прохода, чтобы уменьшить near_edge на тот
    # же множитель, что и длину/ширину ниже, и пересчитать high точнее (проверено на "Короб
    # 400×400×300" при близкой камере: 351 -> 392 против истинных 400, ошибка 12% -> 2%). Это не
    # полная итерация до сходимости (см. `check_vertical_prism` для точного варианта), а один
    # дополнительный шаг уточнения — соответствует духу "простого" расчёта.
    pass1 = _height_and_z_widest(near_edge_naive)
    if pass1 is None:
        return None
    height_1, z_widest_1 = pass1
    near_edge_corrected = near_edge_naive * (d_top - z_widest_1) / d_top
    pass2 = _height_and_z_widest(near_edge_corrected)
    if pass2 is None:
        return None
    height, z_widest = pass2

    # Обратная поправка длины/ширины — НЕ по общей высоте, а по высоте САМОЙ ШИРОКОЙ части
    # объекта в боковом кадре ("летающая тарелка"/"мешок": именно она физически ближе всего к
    # верхней камере в среднем по кадру и определяет размер её силуэта — не обязательно самая
    # верхняя точка, см. докстринг). Для объектов с постоянным сечением (коробка) `_widest_column`
    # сводится к верхней точке — прежнее поведение воспроизводится как частный случай.
    if z_widest >= d_top:
        return None
    top_scale_correction = (d_top - z_widest) / d_top
    length = naive_length * top_scale_correction
    width = naive_width * top_scale_correction

    return SimpleCameraDims(
        length=length,
        width=width,
        height=height,
        top_angle_deg=top_angle_deg,
        side_angle_deg=side_angle_deg,
    )


@dataclass
class TriangulatedCameraDims:
    length: float
    width: float
    height: float
    top_angle_deg: float
    side_angle_deg: float
    triangulated_points: int  # сколько точек контура сверху дали УЗКО согласованную высоту
    # (используются для максимума высоты, см. `well_constrained` в докстринге
    # check_simple_camera_dims_v2) — низкое значение относительно длины контура означает, что
    # боковой ракурс плохо ограничивает бо́льшую часть объекта одним видом


def check_simple_camera_dims_v2(
    mesh: Mesh,
    orientation: Orientation,
    fov_deg: float,
    camera_distance: float,
    resolution_px: camera_silhouette.Resolution,
    top_angle_deg: float = 90.0,
    side_angle_deg: float = 0.0,
    h_max_margin: float = 1.3,
    coarse_steps: int = 96,
    refine_iterations: int = 16,
    camera_distances: dict[float, float] | None = None,
) -> TriangulatedCameraDims | None:
    """`check_simple_camera_dims`, но БЕЗ единой эталонной глубины (`z_widest`/`near_edge`) для
    всего контура — по мотивам обсуждения с пользователем: объект "на ребре" (диск/пуфик,
    стоящий на закруглённом ободе, а не на плоской грани) даёт ~20-30% ошибку у наивной
    поправки, потому что у него просто нет единой представительной глубины — верхняя камера
    видит закруглённый обод, а не плоскую площадку.

    Для КАЖДОЙ точки контура сверху отдельно ищется наибольшая высота h, при которой 3D-точка
    на луче камеры сверху через эту точку (см. `visual_hull.vertical_prism_fit._footprint_at_height`
    — та же формула пиксель->мир при гипотезе высоты h, только применённая к произвольной точке
    контура, а не только к углам многогранника) ЕЩЁ проецируется внутрь бокового силуэта — тот
    же инвариант "силуэт всегда содержит объект", что и во всей методике (docs/method.md), и та
    же идея консистентности, что использует `fit_vertical_prism`, но применённая per-point, а
    не к одной высоте для всего (полигонального) кандидата — поэтому работает и для
    круглых/скруглённых объектов, которые `fit_vertical_prism` сознательно отбраковывает
    (`_footprint_roundness_ratio`).

    Длина/ширина — `cv2.minAreaRect` по восстановленным (X, Y) точек контура сверху (каждая на
    СВОЕЙ найденной высоте), а не по наивному контуру с одной общей поправкой масштаба.

    Высота — max(а, б): (а) максимум Z среди найденных точек контура сверху — точен для
    объектов, у которых наивысшая точка лежит НА контуре сверху (коробка, цилиндр на боку —
    см. пример "диск на ребре" в обсуждении: контур сверху там включает торцевые рёбра,
    очерчивающие всю высоту от 0 до истинного максимума); (б) старая two-pass оценка
    `check_simple_camera_dims` по боковому контуру — наоборот верна для купольных форм
    (мешок, тарелка, сфера), где пик высоты интерьерный (проецируется в СЕРЕДИНУ контура
    сверху, не на его границу) и (а) его бы систематически занижал. `max` берёт лучшее из двух
    без регрессии уже проверенных на этих формах случаев.

    ГРАНИЦА ПОИСКА ПО h — не фиксированная доля camera_distance (как в `fit_vertical_prism`), а
    оценка масштаба ОБЪЕКТА (наивная высота по боковому силуэту на номинальной глубине, см.
    `Silhouette.px_per_unit`, с запасом `h_max_margin`). Найдено эмпирически на коробке
    300×200×200: у точек контура сверху, отвечающих острым РЁБРАМ/УГЛАМ объекта (не купольным
    скруглённым участкам), допустимый диапазон h, при котором точка всё ещё проецируется внутрь
    бокового силуэта, может быть УЗКИМ (считанные мм при camera_distance порядка 1500) —
    сетка `coarse_steps`, растянутая на фиксированные 0.8·camera_distance (~1200мм), пропускала
    такой диапазон между соседними узлами и точка тихо скатывалась в запасной h=0 (полная
    номинальная глубина, наибольшая возможная площадь) — именно это, а не общая неточность
    метода, давало +14% по обеим горизонтальным осям на простой коробке, где триангуляция должна
    быть почти точной. Сужение диапазона поиска до масштаба самого объекта вместо масштаба рига
    даёт ту же сетку `coarse_steps` на порядок мельче без роста числа шагов.

    None — вырожденный силуэт (объект вне кадра) у любой из двух камер, либо боковая маска
    пуста по столбцам (не удалось оценить масштаб для границы поиска).

    `camera_distances` — необязательная карта {азимут -> реальная дистанция}, тот же паттерн,
    что в `check_simple_camera_dims`: отсутствующие в карте азимуты используют базовый
    `camera_distance`."""
    v_top, r_theta_top = view_axes(top_angle_deg)
    if abs(v_top[0]) > 1e-9 or abs(v_top[1]) > 1e-9 or abs(v_top[2] - 1.0) > 1e-9:
        raise ValueError(
            "top_angle_deg должен задавать камеру строго сверху (v_theta=(0,0,1)), обычно 90°"
        )
    v_side, _r_side = view_axes(side_angle_deg)
    if abs(v_side[2]) > 1e-9:
        raise ValueError(
            "side_angle_deg должен задавать чисто боковую камеру (v_theta без Z-компоненты), "
            "обычно 0° или 180°"
        )

    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    bbox_min, _bbox_max = bounding_box(oriented.vertices)
    grounded = Mesh(
        vertices=oriented.vertices - np.array([0.0, 0.0, bbox_min[2]]),
        faces=oriented.faces,
    )
    distances = camera_distances or {}
    d_top = float(distances.get(top_angle_deg, camera_distance))
    d_side = float(distances.get(side_angle_deg, camera_distance))

    top_sil = camera_silhouette.build_silhouette(
        grounded, top_angle_deg, resolution_px, fov_deg, d_top, "center"
    )
    side_sil = camera_silhouette.build_silhouette(
        grounded, side_angle_deg, resolution_px, fov_deg, d_side, "center"
    )

    top_contour = _largest_contour_px(top_sil.mask)
    if top_contour is None:
        return None

    n_rows, n_cols, f_px, cx, cy = _camera_intrinsics(fov_deg, resolution_px)

    cols = top_contour[:, 0]
    rows = top_contour[:, 1]
    u_px = cols - cx
    x_px = cy - rows
    n = len(top_contour)

    def positions_at(h_arr: np.ndarray) -> np.ndarray:
        depth = d_top - h_arr
        u_world = u_px * depth / f_px
        x_world = x_px * depth / f_px
        y_world = u_world * r_theta_top[1]
        return np.stack([x_world, y_world, h_arr], axis=1)

    def contained(pts_xyz: np.ndarray) -> np.ndarray:
        col, row, depth = camera_silhouette.project_points(
            pts_xyz, side_angle_deg, fov_deg, d_side, resolution_px
        )
        col_i = np.round(col).astype(int)
        row_i = np.round(row).astype(int)
        in_bounds = (
            (depth > 0)
            & (col_i >= 0) & (col_i < n_cols)
            & (row_i >= 0) & (row_i < n_rows)
        )
        ok = np.zeros(len(pts_xyz), dtype=bool)
        ok[in_bounds] = side_sil.mask[row_i[in_bounds], col_i[in_bounds]]
        return ok

    side_mask_cols = np.flatnonzero(side_sil.mask.any(axis=0))
    if side_mask_cols.size == 0:
        return None
    naive_height_bound = (side_mask_cols[-1] - side_mask_cols[0]) / side_sil.px_per_unit
    h_max = naive_height_bound * h_max_margin
    grid = np.linspace(0.0, h_max, coarse_steps + 1)
    contained_grid = np.zeros((coarse_steps + 1, n), dtype=bool)
    for i, h in enumerate(grid):
        contained_grid[i] = contained(positions_at(np.full(n, float(h))))

    step_idx = np.arange(coarse_steps + 1)
    true_idx = np.where(contained_grid, step_idx[:, None], -1)
    last_true = true_idx.max(axis=0)  # -1 -> ни разу не contained (боковой ракурс не видит эту
    # часть контура сверху ни на одной высоте — остаётся h=0, консервативный запасной вариант)
    found = last_true >= 0
    lo = np.where(found, grid[np.clip(last_true, 0, coarse_steps)], 0.0)
    hi = np.where(found, grid[np.clip(last_true + 1, 0, coarse_steps)], 0.0)

    # Точки, у которых "содержится" верно почти на всём диапазоне h (не узкое окно вокруг
    # истинного ребра, а широкое плато) — недоопределены ОДНИМ боковым видом: луч камеры
    # сверху через такую точку (обычно с ДАЛЬНЕЙ от боковой камеры стороны объекта, см.
    # докстринг) при любой высоте продолжает проецироваться внутрь бокового силуэта просто
    # потому что тот большой и плохо ограничивает эту сторону — классический недобор Visual
    # Hull от нехватки ракурсов (docs/method.md), не решаемый более мелкой сеткой. `lo` для
    # таких точек может оказаться сильно выше истинной высоты (например, у коробки 400×400×300
    # такая точка дала h≈480 вместо истинных 400) — используются для X/Y (minAreaRect), но
    # НЕ для максимума по Z, иначе завышают высоту сильнее старого метода вместо его улучшения.
    true_count = contained_grid.sum(axis=0)
    well_constrained = found & (true_count <= max(1, coarse_steps // 5))

    points = positions_at(lo)
    xy = points[:, :2].astype(np.float32)
    (_cx, _cy), (side_a, side_b), _angle_deg = cv2.minAreaRect(xy)

    fallback = check_simple_camera_dims(
        mesh, orientation, fov_deg, camera_distance, resolution_px, top_angle_deg, side_angle_deg,
        camera_distances=camera_distances,
    )
    fallback_height = fallback.height if fallback is not None else 0.0
    confident_z = points[well_constrained, 2]
    triangulated_height = float(confident_z.max()) if confident_z.size else 0.0
    height = max(triangulated_height, fallback_height)

    return TriangulatedCameraDims(
        length=float(side_a),
        width=float(side_b),
        height=height,
        top_angle_deg=top_angle_deg,
        side_angle_deg=side_angle_deg,
        triangulated_points=int(well_constrained.sum()),
    )


@dataclass
class MultiSideCameraDims:
    length: float
    width: float
    height: float
    top_angle_deg: float
    side_angles_deg: list[float]
    triangulated_points: int


def check_simple_camera_dims_v3(
    mesh: Mesh,
    orientation: Orientation,
    fov_deg: float,
    camera_distance: float,
    resolution_px: camera_silhouette.Resolution,
    top_angle_deg: float = 90.0,
    side_angles_deg: list[float] | None = None,
    belt_positions: list[camera_silhouette.BeltPosition] | None = None,
    h_max_margin: float = 1.3,
    coarse_steps: int = 96,
    camera_distances: dict[float, float] | None = None,
    supersample: int = 1,
    use_simple_shape_detector: bool = True,
) -> MultiSideCameraDims | None:
    """Multi-side per-point триангуляция — обобщение `check_simple_camera_dims_v2` на
    ПРОИЗВОЛЬНЫЙ набор ограничивающих ракурсов (по умолчанию все 4 азимута рига 0/45/90/135°,
    кроме верха), а не один `side_angle_deg`. Дополнительно — параллакс по ленте: для каждого
    азимута можно использовать несколько положений на ленте (`belt_positions`, по умолчанию
    только "center"), добавляя силуэты из start/end как дополнительные ограничения.

    В отличие от v2, ракурсы не обязаны быть чисто боковыми (v_z=0): азимуты 45/135° имеют
    Z-компоненту в направлении взгляда, но для per-point триангуляции это не проблема —
    3D-точка (x, y, h) проецируется в кадр и проверяется по маске, наклон камеры корректно
    ограничивает точку под своим углом. Камера 45° видит «дальнюю» сторону объекта сверху-сбоку,
    камера 135° — другую сторону сверху-сбоку, камера 0° — чисто сбоку. Пересечение по всем
    трём ограничивает 3D-точку гораздо сильнее, чем один вид.

    Параллакс по ленте (`belt_positions` содержит "start"/"end" в дополнение к "center"):
    товар едет по ленте (ось X), и положения start/center/end дают разные перспективные
    проекции одной камеры — сдвиг ~±19° по оси X (сдвиг ~520мм при distance=1500, FOV=80°).
    Это НЕ новые камеры — те же физические ракурсы, но сдвинутые по X. Параллакс ограничивает
    протяжённость объекта по X (roll-ось), которую азимуты 0/45/90/135° (перпендикулярные X)
    не ограничивают. Особенно полезно для асимметричных по X объектов (Шлем, Ручка).

    Ключевое отличие от v2: `contained()` проверяет, что 3D-точка проецируется внутрь ВСЕХ
    силуэтов одновременно (пересечение по всем азимутам × belt_positions), а не одного.
    Точка на «дальней» стороне от камеры 0°, недоопределённая одним видом (широкое плато
    согласованности), хорошо ограничена камерами 45/135°. Это сужает плато до истинной границы
    и делает фильтр `well_constrained` надёжнее — ложно-узкие плато (регрессия на Шлеме
    10.6→13.1% у v2) отфильтровываются, т.к. «узкое по одному виду» уже не значит «узкое по
    всем».

    Высота — max(триангулированный максимум Z, fallback по v1 через `check_simple_camera_dims`
    с первым чисто боковым ракурсом). Fallback нужен для купольных форм (мешок, тарелка), где
    пик высоты интерьерный и per-point триангуляция его занижает — тот же приём, что в v2.

    Длина/ширина — `cv2.minAreaRect` по восстановленным (X, Y) точек контура сверху, каждая на
    СВОЕЙ найденной высоте (как в v2, но с более точной высотой благодаря multi-side).

    None — вырожденный силуэт (объект вне кадра) у верхней камеры или любого ракурса,
    либо маска пуста по столбцам (не удалось оценить масштаб для границы поиска).

    Новое (доработка под реальную камеру 2592×1944 / FOV 68° / зеркальные ракурсы):
    - `camera_distances` — карта {азимут -> реальная дистанция}: ракурсы, снимаемые через
      зеркало, имеют более длинный оптический путь, чем прямые; отсутствующие азимуты
      используют `camera_distance`. Триангуляция высоты парой ракурсов (tri_height) обобщена
      на разные D двух камер (см. вывод системы в коде).
    - `supersample` > 1 — субпиксельные силуэты: контур сверху (`Silhouette.contour_px`) и
      границы боковых силуэтов в tri_height локализованы с точностью ~1/S px — растровый шум
      (главный источник ошибки на маленьких в кадре объектах) подавлен. Containment
      (contained_all) при этом СОЗНАТЕЛЬНО остаётся растровым с допуском 1px — строгий
      субпиксельный порог сужает окна согласованности и роняет точки в h=0 (см. докстринг
      contained_all).
    - `belt_positions` принимает дробные позиции (float [-1..1]) — плотный параллакс по ленте
      (N кадров по движению, конвейер = известное движение), а не только 3 позиции.
    - `use_simple_shape_detector=False` — отключить детектор простой формы (Эксперимент 9):
      с субпиксельными масками растровый шум наклонных ракурсов, ради которого детектор
      вводился, подавлен, и его необходимость проверяется бенчмарком отдельно."""
    if side_angles_deg is None:
        side_angles_deg = [0.0, 45.0, 135.0]
    if belt_positions is None:
        belt_positions = ["center"]
    # Исключаем top_angle_deg, если он случайно попал в список — верхняя камера используется
    # отдельно как top_sil, её силуэт тривиально содержит все кандидаты (точка лежит на луче
    # верхней камеры) и не добавляет информации в contained().
    side_angles_deg = [a for a in side_angles_deg if a != top_angle_deg]
    if not side_angles_deg:
        raise ValueError("side_angles_deg must contain at least one non-top angle")

    v_top, r_theta_top = view_axes(top_angle_deg)
    if abs(v_top[0]) > 1e-9 or abs(v_top[1]) > 1e-9 or abs(v_top[2] - 1.0) > 1e-9:
        raise ValueError(
            "top_angle_deg должен задавать камеру строго сверху (v_theta=(0,0,1)), обычно 90°"
        )

    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    bbox_min, _bbox_max = bounding_box(oriented.vertices)
    grounded = Mesh(
        vertices=oriented.vertices - np.array([0.0, 0.0, bbox_min[2]]),
        faces=oriented.faces,
    )

    distances = camera_distances or {}

    def dist(angle: float) -> float:
        return float(distances.get(angle, camera_distance))

    d_top = dist(top_angle_deg)
    top_sil = camera_silhouette.build_silhouette(
        grounded, top_angle_deg, resolution_px, fov_deg, d_top, "center",
        supersample=supersample,
    )
    # Все ограничивающие силуэты: декартово произведение side_angles_deg × belt_positions.
    # Каждая пара (азимут, belt_position) даёт отдельный силуэт с своим сдвигом по X
    # (belt_offset), создавая параллакс, ограничивающий протяжённость по X.
    constraint_sils: list[tuple[float, camera_silhouette.BeltPosition, Silhouette]] = []
    for sa in side_angles_deg:
        for bp in belt_positions:
            sil = camera_silhouette.build_silhouette(
                grounded, sa, resolution_px, fov_deg, dist(sa), bp,
                supersample=supersample,
            )
            constraint_sils.append((sa, bp, sil))

    # Все силуэты (без фильтрации) — для tri_height, которой нужны наклонные ракурсы.
    all_constraint_sils = list(constraint_sils)

    # Детектор простой формы: если боковой силуэт (чисто боковой, v_z=0) имеет постоянную
    # ширину по строкам (X) — это призма/коробка, один боковой вид уже даёт точный силуэт.
    # Наклонные камеры (45/135°) в этом случае только вносят растровый шум в contained_all
    # (их столбцы смешивают Y и Z, 1px допуск работает по-разному). Критерий: CV ширины
    # силуэта по строкам < 0.1 (Box300=0.03, Box400=0.05, LunchBox=0.09 — простые;
    # Bag=0.27, Helmet=0.26, Pouf=0.25 — сложные). Дополнительно: если средняя ширина
    # силуэта < 10px — объект слишком тонкий для multi-side (Cylinder=6px), используем v1.
    # При простой форме отсекаем наклонные ракурсы из constraint_sils (для contained_all),
    # оставляя только чисто боковые (v_z≈0). all_constraint_sils (для tri_height) не трогаем.
    is_simple = False
    if use_simple_shape_detector:
        for sa, _bp, ssil in constraint_sils:
            v_s, _ = view_axes(sa)
            if abs(v_s[2]) > 1e-9:
                continue
            side_rows = np.flatnonzero(ssil.mask.any(axis=1))
            if len(side_rows) >= 3:
                widths = np.array([ssil.mask[r].sum() for r in side_rows], dtype=float)
                cv = widths.std() / widths.mean() if widths.mean() > 0 else 0.0
                mean_w = widths.mean()
                if cv < 0.1 or mean_w < 10.0:
                    is_simple = True
                    break

    if is_simple:
        constraint_sils = [
            (sa, bp, sil) for sa, bp, sil in constraint_sils
            if abs(view_axes(sa)[0][2]) < 1e-9
        ]

    top_contour = _sil_contour(top_sil)
    if top_contour is None:
        return None

    n_rows, n_cols, f_px, cx, cy = _camera_intrinsics(fov_deg, resolution_px)

    cols = top_contour[:, 0]
    rows = top_contour[:, 1]
    u_px = cols - cx
    x_px = cy - rows
    n = len(top_contour)

    def positions_at(h_arr: np.ndarray) -> np.ndarray:
        depth = d_top - h_arr
        u_world = u_px * depth / f_px
        x_world = x_px * depth / f_px
        y_world = u_world * r_theta_top[1]
        return np.stack([x_world, y_world, h_arr], axis=1)

    # Кеш belt_offset: результат зависит только от (азимут, belt_position), не от h и не от
    # pts_xyz — это константа для пары (sa, bp). Без кеша belt_offset (обход ВСЕХ вершин меша,
    # camera.py:151) пересчитывался бы coarse_steps × len(constraint_sils) раз (~873 на боевых
    # параметрах), и на тяжёлых мешах (Detergent 218k вершин) это доминировало в времени v3.
    _belt_dx_cache: dict[tuple[float, camera_silhouette.BeltPosition], float] = {}

    def _belt_dx(sa: float, bp: camera_silhouette.BeltPosition) -> float:
        key = (sa, bp)
        v = _belt_dx_cache.get(key)
        if v is None:
            v = camera_silhouette.belt_offset(grounded, fov_deg, dist(sa), bp)
            _belt_dx_cache[key] = v
        return v

    def contained_all(pts_xyz: np.ndarray) -> np.ndarray:
        """Точка contained, если она проецируется внутрь ВСЕХ силуэтов (по бинарной маске с
        допуском 1px на растровые ошибки на границе). Каждый силуэт — пара (азимут,
        belt_position): belt_position задаёт сдвиг камеры по X (belt_offset), создавая
        параллакс.

        ВАЖНО (найдено на бенчмарке боевой камеры): проверять принадлежность СТРОГО по
        субпиксельному покрытию (допуск ~0.2px вместо 1px) НЕЛЬЗЯ — сужение допуска сужает
        окна согласованности per-point, и у повёрнутого короба (yaw 22.5-67.5°) окна точек
        на рёбрах проваливаются МЕЖДУ узлами сетки высот: точка тихо падает в h=0 (полная
        номинальная глубина) и раздувает L/W на десятки процентов — та же механика, что
        задокументированный баг №1 в v2 (грубая сетка). Субпиксельный выигрыш в v3 живёт в
        другом месте: contour_px (контур сверху) и субпиксельные границы силуэтов в
        tri_height; containment остаётся растровым с допуском, супсемплинг маску при этом
        всё равно локализует лучше."""
        ok = np.ones(len(pts_xyz), dtype=bool)
        for sa, bp, ssil in constraint_sils:
            # belt_offset сдвигает объект по X; эквивалентно сдвигу камеры на -dx по X.
            # project_points использует несдвинутые точки, но силуэт строился со сдвигом,
            # поэтому нужно сдвинуть точки перед проекцией (тот же эффект, что в
            # polytope_hull._view_halfspaces: apex = camera_pos - dx * PRINCIPAL_AXIS).
            dx = _belt_dx(sa, bp)
            shifted = pts_xyz.copy()
            shifted[:, 0] += dx  # сдвигаем точки по X, как при построении силуэта
            d_sa = dist(sa)
            col, row, depth = camera_silhouette.project_points(
                shifted, sa, fov_deg, d_sa, resolution_px
            )
            col_i = np.round(col).astype(int)
            row_i = np.round(row).astype(int)
            in_bounds = (
                (depth > 0)
                & (col_i >= 0) & (col_i < n_cols)
                & (row_i >= 0) & (row_i < n_rows)
            )
            this_ok = np.zeros(len(pts_xyz), dtype=bool)
            if in_bounds.any():
                this_ok[in_bounds] = ssil.mask[row_i[in_bounds], col_i[in_bounds]]
            not_ok = ~this_ok & in_bounds
            if not_ok.any():
                for idx in np.flatnonzero(not_ok):
                    r, c = row_i[idx], col_i[idx]
                    r0, r1 = max(0, r - 1), min(n_rows, r + 2)
                    c0, c1 = max(0, c - 1), min(n_cols, c + 2)
                    if ssil.mask[r0:r1, c0:c1].any():
                        this_ok[idx] = True
            ok &= this_ok
            if not ok.any():
                break
        return ok

    # Граница поиска по h — по самому широкому силуэту из всех ограничивающих.
    max_height_bound = 0.0
    for _sa, _bp, ssil in constraint_sils:
        side_mask_cols = np.flatnonzero(ssil.mask.any(axis=0))
        if side_mask_cols.size == 0:
            return None
        bound = (side_mask_cols[-1] - side_mask_cols[0]) / ssil.px_per_unit
        max_height_bound = max(max_height_bound, bound)
    h_max = max_height_bound * h_max_margin
    grid = np.linspace(0.0, h_max, coarse_steps + 1)
    contained_grid = np.zeros((coarse_steps + 1, n), dtype=bool)
    for i, h in enumerate(grid):
        contained_grid[i] = contained_all(positions_at(np.full(n, float(h))))

    step_idx = np.arange(coarse_steps + 1)
    true_idx = np.where(contained_grid, step_idx[:, None], -1)
    last_true = true_idx.max(axis=0)
    found = last_true >= 0
    lo = np.where(found, grid[np.clip(last_true, 0, coarse_steps)], 0.0)

    # Фильтр well_constrained: точка ограничена, если окно согласованности УЗКО по ВСЕМ видам.
    # При multi-side это надёжнее, чем в v2: ложно-узкое плато по одному виду почти всегда
    # широко по другому, и пересечение сужает его — фильтр пропускает только истинно
    # ограниченные точки (ребра/углы, видимые с нескольких ракурсов).
    true_count = contained_grid.sum(axis=0)
    well_constrained = found & (true_count <= max(1, coarse_steps // 5))

    points = positions_at(lo)
    xy = points[:, :2].astype(np.float32)
    (_cx2, _cy2), (side_a, side_b), _angle_deg = cv2.minAreaRect(xy)

    # Fallback по v1 (через первый ЧИСТО БОКОВОЙ ракурс, v_z≈0) — для купольных форм, где
    # per-point триангуляция занижает высоту (пик интерьерный, не на контуре сверху). v1
    # требует чисто боковую камеру (v_side[2]==0), поэтому из side_angles_deg выбираем первый
    # угол, удовлетворяющий этому условию. Если такого нет (все наклонные) — fallback
    # пропускается (triangulated_height используется один).
    fallback_angle = None
    for sa in side_angles_deg:
        v_s, _ = view_axes(sa)
        if abs(v_s[2]) < 1e-9:
            fallback_angle = sa
            break
    fallback_height = 0.0
    if fallback_angle is not None:
        fallback = check_simple_camera_dims(
            mesh, orientation, fov_deg, camera_distance, resolution_px,
            top_angle_deg, fallback_angle,
            camera_distances=camera_distances, supersample=supersample,
        )
        fallback_height = fallback.height if fallback is not None else 0.0

    # Триангуляция высоты через пару ПРОИЗВОЛЬНЫХ азимутов — математически корректная оценка
    # высоты верхней точки объекта БЕЗ эвристического near_edge. Обобщение прежней версии (та
    # требовала, чтобы один из пары был строго боковым v_z=0 — на реальном риге такого ракурса
    # нет, side_angles_deg=[5,70,157.26,171.95], высота тонких объектов уходила в 0.0).
    # Верхняя граница силуэта под азимутом θ (col_bound, выбор границы — то же правило
    # θ<90→col_max, θ≥90→col_min, что и раньше, теперь применяется к ОБОИМ членам пары) даёт
    # k_θ = (col_bound-cx)/f_px = (-Y*sinθ + Z*cosθ) / (D_θ - Y*cosθ - Z*sinθ), линейное по (Y,Z):
    #   A*Y + B*Z = C,  A = k*cosθ - sinθ,  B = k*sinθ + cosθ,  C = k*D_θ
    # Пара разных азимутов θ_i≠θ_j даёт систему 2x2, решаемую по Крамеру. При θ_i=0 (старый
    # частный случай) A=k, B=1 → Z=k*(D-Y), т.е. прежняя формула — частный случай этой системы.
    # Берётся min по всем парам — верхняя граница visual hull по кратчайшему ограничению, не
    # завышает для коробок (где visual hull раздут), т.к. min выбирает более тесное ограничение.
    def _col_bounds(ssil: Silhouette) -> tuple[float, float] | None:
        """(col_min, col_max) силуэта — субпиксельные (по contour_px), иначе по маске."""
        if ssil.contour_px is not None:
            c = ssil.contour_px[:, 0]
            return float(c.min()), float(c.max())
        cc = np.flatnonzero(ssil.mask.any(axis=0))
        if cc.size == 0:
            return None
        return float(cc[0]), float(cc[-1])

    def _tri_coeffs(sa: float, ssil: Silhouette) -> tuple[float, float, float, float] | None:
        """(A, B, C, D) коэффициенты линейного уравнения A*Y+B*Z=C для азимута sa, либо None,
        если силуэт вырожден (пуст по столбцам)."""
        bounds = _col_bounds(ssil)
        if bounds is None:
            return None
        D_sa = dist(sa)
        col_bound = bounds[1] if sa < 90.0 else bounds[0]
        k = (col_bound - cx) / f_px
        theta = np.radians(sa)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        A = k * cos_t - sin_t
        B = k * sin_t + cos_t
        C = k * D_sa
        return A, B, C, D_sa

    tri_heights: list[float] = []
    for idx_i in range(len(all_constraint_sils)):
        sa_i, _bp_i, ssil_i = all_constraint_sils[idx_i]
        coeffs_i = _tri_coeffs(sa_i, ssil_i)
        if coeffs_i is None:
            continue
        A_i, B_i, C_i, D_i = coeffs_i
        theta_i = np.radians(sa_i)
        sin_i, cos_i = np.sin(theta_i), np.cos(theta_i)
        for idx_j in range(idx_i + 1, len(all_constraint_sils)):
            sa_j, _bp_j, ssil_j = all_constraint_sils[idx_j]
            if sa_j == sa_i:
                # разные belt_position одного азимута — вырожденная/повторная система
                continue
            coeffs_j = _tri_coeffs(sa_j, ssil_j)
            if coeffs_j is None:
                continue
            A_j, B_j, C_j, D_j = coeffs_j
            theta_j = np.radians(sa_j)
            sin_j, cos_j = np.sin(theta_j), np.cos(theta_j)
            den = A_i * B_j - A_j * B_i
            if abs(den) < 1e-12:
                continue
            Y_sol = (C_i * B_j - C_j * B_i) / den
            Z_sol = (A_i * C_j - A_j * C_i) / den
            if Z_sol <= 0:
                continue
            # Валидность: положительная глубина перед ОБЕИМИ камерами пары (надёжнее одной
            # границы min(D_i, D_j) — прежняя проверка полагалась на асимметрию 0°-члена).
            if D_i - Y_sol * cos_i - Z_sol * sin_i <= 0:
                continue
            if D_j - Y_sol * cos_j - Z_sol * sin_j <= 0:
                continue
            tri_heights.append(float(Z_sol))

    tri_height = min(tri_heights) if tri_heights else 0.0

    # Для простых форм (призма/коробка) tri_height через наклонные ракурсы завышает
    # высоту (visual hull с 4 ракурсами раздут для прямоугольных форм), а fallback v1
    # уже точен — не используем tri_height в этом случае.
    if is_simple:
        tri_height = 0.0

    confident_z = points[well_constrained, 2]
    triangulated_height = float(confident_z.max()) if confident_z.size else 0.0
    if is_simple:
        # Для простых форм (призма/коробка) per-point с одним боковым видом завышает
        # высоту (та же проблема, что в v2 на Box400), а tri_height отключён —
        # используем только fallback v1, который точен для постоянного сечения.
        height = fallback_height
    else:
        height = max(triangulated_height, fallback_height, tri_height)

    return MultiSideCameraDims(
        length=float(side_a),
        width=float(side_b),
        height=height,
        top_angle_deg=top_angle_deg,
        side_angles_deg=list(side_angles_deg),
        triangulated_points=int(well_constrained.sum()),
    )


def check_camera_hull_dims(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    fov_deg: float,
    camera_distances: dict[float, float],
    resolution_px: int,
    belt_position: camera_silhouette.BeltPosition = "center",
) -> tuple[float, float, float] | None:
    """Габариты из точной 3D-реконструкции по ВСЕМ реальным ракурсам рига (`polytope_hull` —
    точное пересечение конусов обзора), а не по двум фиксированным проекциям + эвристической
    поправке (`check_simple_camera_dims`/`_v2`, см. их докстринги и
    `docs/camera_dims_v2_investigation.md`). Не требует новых камер/зеркал — использует ровно
    то, что уже снимает риг «2 камеры + 1 зеркало» (`view_angles_deg` из
    `config/app_settings.yaml`, по умолчанию 0/45/90/135).

    Ключевое отличие от check_simple_camera_dims_v2: там ОДИН боковой ракурс недоопределяет
    дальнюю от него сторону сложных объектов (см. эксперимент 1 в investigation-документе —
    Ручка, Шлем). Здесь КАЖДЫЙ азимут одновременно ограничивает форму пересечением
    полупространств — тот же принцип, что даёт точность `true_dims` в Analytical Mode, но с
    настоящей перспективной геометрией камеры (camera_distance/fov), а не идеализированной
    ортографикой на бесконечном удалении.

    Ограничение (см. докстринг `polytope_hull`): пересечение конусов ВСЕГДА выпукло — для
    вогнутых объектов (Ручка — зажим+тонкий стержень со ступенькой) даёт выпуклую оболочку,
    не точную форму; для таких случаев нужен `exact_polyhedral_hull` (не подключено здесь).

    Возвращает осевой (X, Y, Z) bbox восстановленного многогранника в системе координат
    ориентированного меша, либо None — вырожденный случай (пустой силуэт на каком-то ракурсе,
    см. `polytope_hull.carve`)."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    hull = polytope_hull.carve(
        oriented,
        [(angle, belt_position) for angle in view_angles_deg],
        fov_deg,
        camera_distances,
        resolution_px,
    )
    if hull is None:
        return None
    lo, hi = hull.vertices.min(axis=0), hull.vertices.max(axis=0)
    dims = hi - lo
    return (float(dims[0]), float(dims[1]), float(dims[2]))


def evaluate_metrics_only(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    axis_pos: float | None = None,
    resolution_px: int = 512,
    roundness_threshold: float = 0.8,
) -> RoundnessResult | None:
    """Облегчённый расчёт без построения/хранения масок силуэтов — для Search
    (перебор тысяч ориентаций), где важны только метрики."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )

    if axis_pos is None:
        axis_pos = default_axis_pos(oriented)

    polygon = reconstruct_section(oriented, view_angles_deg, axis_pos, resolution_px)
    if polygon is None:
        return None
    return compute_roundness(
        np.array(polygon.exterior.coords[:-1], dtype=np.float64),
        resolution_px=resolution_px, threshold=roundness_threshold,
    )


def compare_configurations(
    mesh: Mesh,
    orientation: Orientation,
    angle_sets: list[list[float]],
    axis_pos: float | None = None,
    resolution_px: int = 1024,
    roundness_threshold: float = 0.8,
) -> list[ConfigCompareEntry]:
    """Одна и та же ориентация, несколько наборов view_angles — режим Compare Configurations."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )

    if axis_pos is None:
        axis_pos = default_axis_pos(oriented)

    results = []
    for angles in angle_sets:
        polygon = reconstruct_section(oriented, angles, axis_pos, resolution_px)
        roundness = (
            compute_roundness(
                np.array(polygon.exterior.coords[:-1], dtype=np.float64),
                resolution_px=resolution_px, threshold=roundness_threshold,
            )
            if polygon is not None
            else None
        )
        results.append(ConfigCompareEntry(view_angles_deg=list(angles), roundness=roundness))
    return results


def axis_direction(azim_deg: float, elev_deg: float) -> np.ndarray:
    """Единичный вектор оси по азимуту (от X в плоскости XY) и подъёму над плоскостью XY."""
    azim, elev = np.radians(azim_deg), np.radians(elev_deg)
    return np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )


def axis_to_angles(direction: np.ndarray) -> tuple[float, float]:
    """Обратное преобразование к axis_direction: азимут (0..360°) и подъём единичного вектора.
    Используется в GUI для отображения найденной оси переката (см. gui/main_window)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) or 1.0)
    elev = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(d[1], d[0])))
    return azim % 360.0, elev


def _projection_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ортонормированный базис плоскости, перпендикулярной направлению d."""
    e1 = np.cross(d, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(d, np.array([1.0, 0.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2


def _sweep_grid(step_deg: float) -> list[tuple[float, float]]:
    """Полусфера направлений (ось без знака: d и -d дают одну проекцию): азимут [0, 180),
    подъём (-90, 90), плюс полюс (вертикальная ось) один раз."""
    grid: list[tuple[float, float]] = [(0.0, 90.0)]
    for elev in np.arange(-90.0 + step_deg, 90.0 - step_deg / 2.0, step_deg):
        for azim in np.arange(0.0, 180.0, step_deg):
            grid.append((float(azim), float(elev)))
    return grid


def _projection_k(
    points_3d: np.ndarray,
    azim_deg: float,
    elev_deg: float,
    resolution_px: int,
    threshold: float,
) -> tuple[RoundnessResult | None, np.ndarray | None]:
    """Метрики круглости проекции облака точек вдоль оси (azim, elev) и сами 2D-точки."""
    d = axis_direction(azim_deg, elev_deg)
    e1, e2 = _projection_basis(d)
    projected = points_3d @ np.stack([e1, e2], axis=1)  # (N, 2)
    return compute_projection_roundness(projected, resolution_px, threshold), projected


def _box_metrics(
    points_3d: np.ndarray, azim_deg: float, elev_deg: float
) -> tuple[float, float, float] | None:
    """Габариты охватывающего бокса при направлении одной из его осей (azim, elev): толщина
    вдоль этой оси (точно — max-min проекции) и стороны минимального прямоугольника
    (`cv2.minAreaRect`, точное решение 2D-подзадачи) в перпендикулярной плоскости.

    Разумный (не полный, но практичный) поиск минимального 3D-бокса: полный перебор в SO(3)
    избыточен, а перебор направлений одной оси + точное решение оставшихся двух через
    minAreaRect — стандартный приём (аналог "rotating calipers" для полиэдров), достаточный
    для box-подобных объектов, которые и мотивировали эту метрику. None — вырожденный случай
    (< 3 точек в проекции)."""
    d = axis_direction(azim_deg, elev_deg)
    axis_vals = points_3d @ d
    thickness = float(axis_vals.max() - axis_vals.min())

    e1, e2 = _projection_basis(d)
    projected = (points_3d @ np.stack([e1, e2], axis=1)).astype(np.float32)  # (N, 2)
    if len(projected) < 3:
        return None
    (_cx, _cy), (side_a, side_b), _angle_deg = cv2.minAreaRect(projected)
    return thickness, float(side_a), float(side_b)


def _search_min_bounding_box(points_3d: np.ndarray, axis_step_deg: float) -> BoxHit | None:
    """Минимальный охватывающий бокс облака точек: перебор направлений на полусфере
    (`_sweep_grid`) + уточнение мелким шагом вокруг найденного оптимума (см. `_box_metrics`).
    Общая часть `check_model_roll` (по восстановленной форме) и `mesh_true_dims` (по вершинам
    самого меша) — вынесена, чтобы не тащить в `mesh_true_dims` несвязанный поиск круглой
    проекции, который в `check_model_roll` идёт тем же перебором ради общего прохода по сетке.
    None — вырожденное облако точек (< 3 точек в проекции при любом направлении).

    Точка ВНУТРИ 3D-выпуклой оболочки облака никогда не может дать более широкий bounding box
    (`_box_metrics` — только max-min проекции вдоль оси + `cv2.minAreaRect` в перпендикулярной
    плоскости, оба зависят только от крайних/оболочечных точек) — тот же математически точный
    приём, что уже применён к G4-перебору (см. [[Оптимизация скорости G4-перебора (свести к
    вершинам convex hull)]]). На плотных облаках torchhull (десятки тысяч точек, ~800
    направлений перебора) это меняет стоимость с O(n·800) на O(hull·800), где hull на 1-2
    порядка меньше n — не приближение, точный тот же результат."""
    try:
        hull_points = points_3d[ConvexHull(points_3d).vertices]
    except Exception:
        # Вырожденное облако (плоское/коллинеарное) — безопасный откат на полное облако.
        hull_points = points_3d
    best_box: BoxHit | None = None

    def try_box_direction(azim: float, elev: float) -> None:
        nonlocal best_box
        dims = _box_metrics(hull_points, azim, elev)
        if dims is None:
            return
        volume = dims[0] * dims[1] * dims[2]
        if best_box is None or volume < best_box.volume:
            best_box = BoxHit(axis_azim_deg=azim, axis_elev_deg=elev, dims=dims, volume=volume)

    for azim, elev in _sweep_grid(axis_step_deg):
        try_box_direction(azim, elev)

    if best_box is not None:
        fine_step = max(axis_step_deg / 5.0, 0.5)
        azim0, elev0 = best_box.axis_azim_deg, best_box.axis_elev_deg
        for elev in np.arange(elev0 - axis_step_deg, elev0 + axis_step_deg + 1e-9, fine_step):
            if not -90.0 <= elev <= 90.0:
                continue
            for azim in np.arange(azim0 - axis_step_deg, azim0 + axis_step_deg + 1e-9, fine_step):
                try_box_direction(float(azim % 360.0), float(elev))

    return best_box


def mesh_true_dims(
    mesh: Mesh, axis_step_deg: float = 5.0
) -> tuple[float, float, float] | None:
    """Истинные габариты STL — минимальный охватывающий бокс вершин САМОГО МЕША (не
    реконструкции), тот же перебор направлений, что и `true_dims` в `check_model_roll` (см.
    `_search_min_bounding_box`). В ОТЛИЧИЕ от `bounding_box(mesh.vertices)` (осевой bbox,
    растёт при повороте объекта — это неизбежное свойство осевого bbox, не баг, см.
    `test_true_dims_independent_of_orientation`), результат — тройка размеров стороны бокса,
    НЕ привязанная к мировым осям, поэтому не зависит от того, к какому меша (уже повёрнутому
    Roll/Pitch/Yaw или нет) её применили — вызывающая сторона может считать её один раз при
    загрузке STL и не пересчитывать при повороте (см. GUI: `self._stl_true_dims` в
    `main_window._load_mesh`). None — вырожденный меш (< 3 вершин в проекции)."""
    box = _search_min_bounding_box(mesh.vertices.astype(np.float64), axis_step_deg)
    if box is None:
        return None
    return tuple(sorted(box.dims, reverse=True))


def check_model_roll(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    resolution_px: int = 1024,
    roundness_threshold: float = 0.8,
    num_slices: int = 50,
    axis_step_deg: float = 5.0,
    sweep_resolution_px: int = 512,
    use_camera_silhouettes: bool = False,
    camera_fov_deg: float = 0.0,
    camera_distances: dict[float, float] | None = None,
    belt_position: str = "center",
) -> RollCheckResult:
    """"Проверить модель": может ли объект катиться + истинные (независимые от поворота) габариты.

    1. По силуэтам восстанавливается форма (стопка сечений) и габариты В ТЕКУЩЕЙ ориентации.
    2. Перебираются направления оси на полусфере с шагом `axis_step_deg`; для каждого
       направления считаются ДВЕ независимые метрики по одному и тому же облаку точек формы:
       - круглость проекции (k = Rin/Rout выпуклой оболочки) — для потенциала к перекату;
       - минимальный охватывающий бокс (толщина вдоль направления + `cv2.minAreaRect` в
         перпендикулярной плоскости, см. _box_metrics) — для истинных габаритов.
       У каждой метрики свой оптимум (лучшее направление для круглости обычно НЕ совпадает
       с направлением минимального объёма бокса), поэтому окрестность каждого найденного
       оптимума уточняется мелким шагом отдельно.
    3. Объект "круглый" (потенциал к перекату), если max k >= порога.

    Круглость отдельных сечений считается только справочно (section_hits): круглое сечение
    без круглой проекции переката не даёт (пример — скос горлышка бутылки).

    `use_camera_silhouettes=True` — как и в evaluate(), реконструкция идёт по перспективным
    (Camera Mode) силуэтам вместо ортографических; `mesh_dims` остаётся истинным, поэтому
    отклонение `dims`/`true_dims` от `mesh_dims` показывает ошибку от перспективы реальной
    камеры при заданных `camera_fov_deg`/`camera_distances`.
    """
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    bbox_min, bbox_max = bounding_box(oriented.vertices)
    mesh_dims = tuple((bbox_max - bbox_min).tolist())

    silhouettes = build_silhouette_set(
        oriented,
        view_angles_deg,
        resolution_px,
        use_camera=use_camera_silhouettes,
        camera_fov_deg=camera_fov_deg,
        camera_distances=camera_distances,
        belt_position=belt_position,
    )

    start = time.perf_counter()
    shape = reconstruct_shape_from_silhouettes(silhouettes, num_slices)
    if shape is None:
        return RollCheckResult(
            dims=None,
            mesh_dims=mesh_dims,
            checked_directions=0,
            axis_step_deg=axis_step_deg,
            best=None,
            best_axis_dir=None,
            best_projection_coords=None,
            best_roundness=None,
            found_round=False,
            section_hits=[],
            round_section_count=0,
            true_dims=None,
            true_dims_axis_dir=None,
            elapsed_seconds=time.perf_counter() - start,
        )

    # Справочно: круглость каждого сечения реконструкции (контуры для Panel 3)
    section_hits: list[SectionCheckHit] = []
    for x_pos, coords in shape.slices:
        roundness = compute_roundness(
            np.asarray(coords, dtype=np.float64),
            resolution_px=sweep_resolution_px, threshold=roundness_threshold,
        )
        section_hits.append(
            SectionCheckHit(
                axis_pos=x_pos, k=roundness.k, passed=roundness.passed, polygon_coords=coords
            )
        )

    # Облако точек формы: вершины контуров сечений в 3D (x — позиция среза)
    chunks = []
    for x_pos, coords in shape.slices:
        arr = np.asarray(coords[:-1], dtype=np.float64)  # без замыкающей точки
        chunks.append(np.column_stack([np.full(len(arr), x_pos), arr[:, 0], arr[:, 1]]))
    points_3d = np.concatenate(chunks)

    best: AxisHit | None = None
    checked = 0

    def try_direction(azim: float, elev: float) -> None:
        nonlocal best, checked
        result, _ = _projection_k(
            points_3d, azim, elev, sweep_resolution_px, roundness_threshold
        )
        if result is None:
            return
        checked += 1
        if best is None or result.k > best.k:
            best = AxisHit(
                axis_azim_deg=azim, axis_elev_deg=elev, k=result.k, passed=result.passed
            )

    for azim, elev in _sweep_grid(axis_step_deg):
        try_direction(azim, elev)

    # Уточнение вокруг лучшего направления мелким шагом — у круглости свой оптимум, обычно не
    # совпадающий с направлением минимального бокса (см. _search_min_bounding_box ниже, у неё
    # уточнение уже встроено).
    fine_step = max(axis_step_deg / 5.0, 0.5)
    if best is not None:
        azim0, elev0 = best.axis_azim_deg, best.axis_elev_deg
        for elev in np.arange(elev0 - axis_step_deg, elev0 + axis_step_deg + 1e-9, fine_step):
            if not -90.0 <= elev <= 90.0:
                continue
            for azim in np.arange(azim0 - axis_step_deg, azim0 + axis_step_deg + 1e-9, fine_step):
                try_direction(float(azim % 360.0), float(elev))

    best_box = _search_min_bounding_box(points_3d, axis_step_deg)

    # Лучшая проекция — финальный расчёт на полном разрешении, контур для Panel 3
    best_projection_coords = None
    best_roundness = None
    best_axis_dir = None
    if best is not None:
        best_roundness, projected = _projection_k(
            points_3d, best.axis_azim_deg, best.axis_elev_deg, resolution_px, roundness_threshold
        )
        if best_roundness is not None and projected is not None:
            hull = cv2.convexHull(projected.astype(np.float32))[:, 0, :]
            ring = np.vstack([hull, hull[:1]])
            best_projection_coords = [(float(p[0]), float(p[1])) for p in ring]
            best = AxisHit(
                axis_azim_deg=best.axis_azim_deg,
                axis_elev_deg=best.axis_elev_deg,
                k=best_roundness.k,
                passed=best_roundness.passed,
            )
        best_axis_dir = tuple(axis_direction(best.axis_azim_deg, best.axis_elev_deg).tolist())

    true_dims = None
    true_dims_axis_dir = None
    if best_box is not None:
        true_dims = tuple(sorted(best_box.dims, reverse=True))
        true_dims_axis_dir = tuple(
            axis_direction(best_box.axis_azim_deg, best_box.axis_elev_deg).tolist()
        )

    round_sections = sum(1 for h in section_hits if h.passed)
    return RollCheckResult(
        dims=shape.dims,
        mesh_dims=mesh_dims,
        checked_directions=checked,
        axis_step_deg=axis_step_deg,
        best=best,
        best_axis_dir=best_axis_dir,
        best_projection_coords=best_projection_coords,
        best_roundness=best_roundness,
        found_round=best is not None and best.passed,
        section_hits=section_hits,
        round_section_count=round_sections,
        true_dims=true_dims,
        true_dims_axis_dir=true_dims_axis_dir,
        elapsed_seconds=time.perf_counter() - start,
    )


@dataclass
class PcaRollResult:
    """Итог "Проверить модель" методом PCA (см. check_model_roll_pca): оси качения ищутся
    среди 3 главных осей инерции облака точек формы вместо перебора по полусфере.

    Структура намеренно проще RollCheckResult: PCA-метод не делает отдельный поиск
    минимального охватывающего бокса (true_dims) и не уточняет шагом — у него всего 3
    кандидата. Истинные габариты для сортировки считаются отдельно через mesh_true_dims
    по мешу объекта, как это уже делает GUI.

    Поля `true_dims`/`section_hits`/`axis_step_deg`/`round_section_count` — заглушки для
    совместимости с GUI (`metrics_panel.set_roll_check_result` через duck-typing)."""
    dims: tuple[float, float, float] | None
    mesh_dims: tuple[float, float, float]
    checked_directions: int
    best: AxisHit | None
    best_axis_dir: tuple[float, float, float] | None
    best_projection_coords: list[tuple[float, float]] | None
    best_roundness: RoundnessResult | None
    found_round: bool
    # Все 3 оси PCA с их k (для диагностики/отчёта — в RollCheckResult такого нет)
    axis_hits: list[AxisHit]
    elapsed_seconds: float
    method: str = "pca"
    # Заглушки для совместимости с GUI (RollCheckResult имеет эти поля, PCA их не считает):
    true_dims: tuple[float, float, float] | None = None
    true_dims_axis_dir: tuple[float, float, float] | None = None
    section_hits: list = field(default_factory=list)
    round_section_count: int = 0
    axis_step_deg: float = 0.0


def _principal_axes(points_3d: np.ndarray) -> np.ndarray:
    """Три ортонормированные главные оси инерции облака точек (столбцы (3,3)) по убыванию
    дисперсии. У тела вращения ось симметрии совпадает с одной из них (обычно с наименьшей
    дисперсией — «длинная» ось вытянутого объекта), поэтому проверка только 3 осей
    покрывает все физически осмысленные кандидаты на ось качения."""
    centered = points_3d - points_3d.mean(axis=0)
    # eigh сортирует по возрастанию дисперсии; возвращаем в порядке [σ_max, σ_mid, σ_min]
    _, eigvecs = np.linalg.eigh(centered.T @ centered / max(len(centered), 1))
    return eigvecs[:, ::-1]


def check_model_roll_pca(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    resolution_px: int = 1024,
    roundness_threshold: float = 0.8,
    num_slices: int = 50,
    sweep_resolution_px: int = 512,
    use_camera_silhouettes: bool = False,
    camera_fov_deg: float = 0.0,
    camera_distances: dict[float, float] | None = None,
    belt_position: str = "center",
) -> PcaRollResult:
    """«Проверить модель» методом PCA: вместо перебора направлений по полусфере (см.
    `check_model_roll`) ось качения ищется среди 3 главных осей инерции облака точек
    восстановленной формы.

    Мотивация и ожидаемое поведение см. в задаче
    «Сравнить PCA-критерий круглой проекции с текущим перебором полусферы (roboson_tools)».
    Кратко: ложные срабатывания перебора полусферы возникают на косых осях (диагональ куба,
    скос горлышка, косая ось гантели с круглой серединой и квадратными краями). PCA-оси —
    это оси симметрии объекта, а тело качения симметрично относительно своей оси → ось
    качения совпадает с одной из главных осей инерции. Ограничение до 3 осей отсекает
    ложные косые хиты и ускоряет расчёт на 1–2 порядка.

    Параметры силуэтов/реконструкции совпадают с `check_model_roll` — методы сравнимы
    напрямую (отличается только этап поиска оси). `elapsed_seconds` замеряет то же: от
    реконструкции формы до конца расчёта k (построение силуэтов не входит, как и в
    `check_model_roll`)."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    bbox_min, bbox_max = bounding_box(oriented.vertices)
    mesh_dims = tuple((bbox_max - bbox_min).tolist())

    silhouettes = build_silhouette_set(
        oriented,
        view_angles_deg,
        resolution_px,
        use_camera=use_camera_silhouettes,
        camera_fov_deg=camera_fov_deg,
        camera_distances=camera_distances,
        belt_position=belt_position,
    )

    start = time.perf_counter()
    shape = reconstruct_shape_from_silhouettes(silhouettes, num_slices)
    if shape is None:
        return PcaRollResult(
            dims=None,
            mesh_dims=mesh_dims,
            checked_directions=0,
            best=None,
            best_axis_dir=None,
            best_projection_coords=None,
            best_roundness=None,
            found_round=False,
            axis_hits=[],
            elapsed_seconds=time.perf_counter() - start,
        )

    chunks = []
    for x_pos, coords in shape.slices:
        arr = np.asarray(coords[:-1], dtype=np.float64)
        chunks.append(np.column_stack([np.full(len(arr), x_pos), arr[:, 0], arr[:, 1]]))
    points_3d = np.concatenate(chunks)

    axes = _principal_axes(points_3d)
    axis_hits: list[AxisHit] = []
    best: AxisHit | None = None
    best_roundness: RoundnessResult | None = None
    best_projection_coords: list[tuple[float, float]] | None = None
    best_axis_dir: tuple[float, float, float] | None = None
    best_projected: np.ndarray | None = None

    for d in axes.T:
        result, projected = _projection_k_for_axis(
            points_3d, d, sweep_resolution_px, roundness_threshold
        )
        if result is None:
            continue
        azim, elev = axis_to_angles(d)
        # axis_to_angles возвращает азимут [0,360); для AxisHit (ось без знака) приводим к [0,180)
        azim = azim % 180.0
        hit = AxisHit(axis_azim_deg=azim, axis_elev_deg=elev, k=result.k, passed=result.passed)
        axis_hits.append(hit)
        if best is None or result.k > best.k:
            best = hit
            best_roundness = result
            best_projected = projected
            best_axis_dir = tuple(float(v) for v in d.tolist())

    if best is not None and best_projected is not None and best_roundness is not None:
        hull = cv2.convexHull(best_projected.astype(np.float32))[:, 0, :]
        ring = np.vstack([hull, hull[:1]])
        best_projection_coords = [(float(p[0]), float(p[1])) for p in ring]
        # Пересчёт k на полном разрешении (как в check_model_roll) — для итоговой метрики
        best_roundness_full, _ = _projection_k_for_axis(
            points_3d,
            np.array(best_axis_dir),
            resolution_px,
            roundness_threshold,
        )
        if best_roundness_full is not None:
            best = AxisHit(
                axis_azim_deg=best.axis_azim_deg,
                axis_elev_deg=best.axis_elev_deg,
                k=best_roundness_full.k,
                passed=best_roundness_full.passed,
            )
            best_roundness = best_roundness_full

    return PcaRollResult(
        dims=shape.dims,
        mesh_dims=mesh_dims,
        checked_directions=len(axis_hits),
        best=best,
        best_axis_dir=best_axis_dir,
        best_projection_coords=best_projection_coords,
        best_roundness=best_roundness,
        found_round=best is not None and best.passed,
        axis_hits=axis_hits,
        elapsed_seconds=time.perf_counter() - start,
    )


def _projection_k_for_axis(
    points_3d: np.ndarray,
    direction: np.ndarray,
    resolution_px: int,
    threshold: float,
) -> tuple[RoundnessResult | None, np.ndarray | None]:
    """Метрики круглости проекции облака точек вдоль заданного единичного вектора и сами
    2D-точки. То же, что `_projection_k`, но принимает направление напрямую (а не азимут/
    подъём) — для PCA-осей, у которых есть готовый вектор."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) or 1.0)
    e1, e2 = _projection_basis(d)
    projected = points_3d @ np.stack([e1, e2], axis=1)
    return compute_projection_roundness(projected, resolution_px, threshold), projected


def _local_grid_search(
    points_3d: np.ndarray,
    d_center: np.ndarray,
    radius_deg: float,
    step_deg: float,
    resolution_px: int,
    threshold: float,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Локальный поиск оси с максимальным k в прямоугольной сетке азимут/подъём вокруг
    `d_center` (единичный вектор). Возвращает (max_k, best_direction, projected_2d).

    Используется в `check_model_roll_g4` как fallback для пограничных случаев: ось
    качения асимметричного объекта обычно близка (≤20°) к одной из 6 базовых
    направлений F1, но не точно совпадает — локальный поиск в окрестности находит её,
    не прибегая к полному перебору по полусфере (который медленный)."""
    az0, el0 = axis_to_angles(d_center)
    az0 = az0 % 180.0  # ось без знака — азимут в [0, 180)
    best_k = 0.0
    best_d: np.ndarray | None = None
    best_projected: np.ndarray | None = None
    for daz in np.arange(-radius_deg, radius_deg + step_deg, step_deg):
        for del_ in np.arange(-radius_deg, radius_deg + step_deg, step_deg):
            az = (az0 + daz) % 180.0
            el = float(np.clip(el0 + del_, -90.0, 90.0))
            d = axis_direction(az, el)
            result, projected = _projection_k_for_axis(
                points_3d, d, resolution_px, threshold
            )
            if result is None or result.k <= best_k:
                continue
            best_k = result.k
            best_d = d
            best_projected = projected
    return best_k, best_d, best_projected


# Вердикт G4 — три категории вместо двух (см. check_model_roll_g4):
# - "round"       — объект катится (max k ≥ порога)
# - "not_round"   — объект точно не катится (max k < low_threshold после fallback)
# - "uncertain"   — пограничный случай, требует перепроверки (оператор/ручной поток).
#                   Реализует логику пользователя: «лучше ошибиться в сторону круглого,
#                   чем пропустить катящийся в основной сортировщик» — никаких тихих
#                   пропусков; всё, что не подтверждено, идёт в «uncertain».
ROLL_VERDICT_ROUND = "round"
ROLL_VERDICT_NOT_ROUND = "not_round"
ROLL_VERDICT_UNCERTAIN = "uncertain"

# Устойчивость локальной круглости вдоль найденной оси (см. заметка задачи
# [[Устойчивая локальная круглость вдоль оси переката (G4)]]) — короб/куб вдоль диагонали даёт
# высокий АГРЕГИРОВАННЫЙ k, но локальное сечение круглое лишь на исчезающе малой доле длины
# оси; настоящий цилиндр/бутылка/конус на боку держат k высоким на большей части длины.
# Стартовые значения — требуют калибровки на тестовом наборе, как и 0.8/0.65 у самого G4.
# 24→16 (2026-08-01, по решению пользователя): эксперимент на боевом 16-объектном наборе
# (реальный GPU torchhull) показал, что M=24 демотирует "bottle" (round→uncertain) буквально на
# ОДИН бин (8/24=0.333 против нужных 9/24=0.35, `sustained_fraction` в этом случае — ТОЧНОЕ
# значение, не консервативная оценка от раннего выхода). M=16 чинит именно этот случай
# (6/16=0.375, порог пройден) БЕЗ единой регрессии по остальным 15 объектам (короба/`cyl_skewed`/
# `lunchbox` остаются `not_round`, `asym_barrel`/`ell_cyl`/`plate`/`pouf` остаются `round`, не
# становятся ложно `round`) — проверено дважды на реальном torchhull. `bag`/`asym_cone`/`helmet`
# при этом остаются демотированными при любом M∈{24,16,12} (их `sustained_fraction` держится
# далеко от порога, не близко к границе, как у bottle) — это НЕ калибровочная проблема числа
# бинов, а отдельный, нерешённый здесь вопрос (см. заметка задачи).
_SUSTAINED_BINS = 16
_SUSTAINED_LOCAL_RES_PX = 128
_SUSTAINED_MIN_RADIUS_FRAC = 0.5
_SUSTAINED_MIN_FRACTION = 0.35
# Минимум «круглых» бинов, при котором passed/_SUSTAINED_BINS >= _SUSTAINED_MIN_FRACTION —
# используется для детерминированного раннего выхода из цикла по бинам (см.
# _sustained_fraction_for_axis): решение предрешено раньше конца перебора, как только либо
# набрано это число, либо оставшихся бинов заведомо не хватит его набрать.
_SUSTAINED_MIN_PASSED = int(np.ceil(_SUSTAINED_MIN_FRACTION * _SUSTAINED_BINS - 1e-9))


@dataclass
class GRollResult:
    """Итог "Проверить модель" методом G4 — F1 + локальный поиск для пограничных
    случаев + трёхкатегорный вердикт (см. check_model_roll_g4). Расширение PcaRollResult
    полем `verdict` (категория) и `fallback_triggered` (запускался ли локальный поиск).

    Поля `true_dims`/`section_hits`/`axis_step_deg`/`round_section_count` добавлены
    как заглушки для совместимости с GUI (`metrics_panel.set_roll_check_result` работает
    с `Union[RollCheckResult, GRollResult]` через duck-typing). G4 не считает отдельный
    минимальный бокс и круглость сечений — для этого есть `mesh_true_dims` (считается
    отдельно по мешу, как уже делает GUI) и справочная `compute_roundness` по сечениям."""
    dims: tuple[float, float, float] | None
    mesh_dims: tuple[float, float, float]
    checked_directions: int
    best: AxisHit | None
    best_axis_dir: tuple[float, float, float] | None
    best_projection_coords: list[tuple[float, float]] | None
    best_roundness: RoundnessResult | None
    found_round: bool  # True только если verdict == "round" (uncertain — не round)
    verdict: str  # ROLL_VERDICT_*
    fallback_triggered: bool  # запускался ли локальный поиск (зона [low, high))
    axis_hits: list[AxisHit]  # все проверенные базовые направления (для диагностики)
    elapsed_seconds: float
    # Заглушки для совместимости с GUI (RollCheckResult имеет эти поля, G4 их не считает):
    true_dims: tuple[float, float, float] | None = None  # GUI считает через mesh_true_dims отдельно
    true_dims_axis_dir: tuple[float, float, float] | None = None
    section_hits: list = field(default_factory=list)  # G4 не считает круглость сечений
    round_section_count: int = 0
    axis_step_deg: float = 0.0  # не имеет смысла для G4 (нет перебора по полусфере)
    method: str = "g4"
    # Устойчивость локальной круглости вдоль оси (см. [[Устойчивая локальная круглость вдоль
    # оси переката (G4)]]) — считается только когда verdict на момент проверки был "round"
    # (иначе None, не тратим время впустую). sustained_demoted=True ⇒ verdict понижен
    # round → uncertain этим признаком (не not_round — новый непроверенный признак не должен
    # выбрасывать реально катящийся объект из безопасной зоны).
    sustained_fraction: float | None = None
    sustained_demoted: bool = False
    sustained_seconds: float | None = None
    # Сработало ли «Важное уточнение» (см. ТЗ): d_final заменён на другого из 6 базовых
    # кандидатов, потому что исходный d_final не прошёл sustained_fraction, а среди остальных
    # нашёлся кандидат с agregированным k ≥ порога И sustained_fraction ≥ порога.
    sustained_promoted_from_alt_axis: bool = False
    # Расширение sustained-check на зону uncertain (2026-07-31, по решению пользователя после
    # подтверждения базовой версии на round-ветке): найденная (лучшая) ось не только не
    # дотянула агрегированный k до порога, но и не держит устойчивую локальную круглость — и
    # ни у одного из остальных базовых кандидатов такой устойчивости тоже нет (иначе остались
    # бы в uncertain — см. _g4_core_from_points). Двойное подтверждение non-roundness даёт
    # уверенный not_round вместо ухода к оператору.
    sustained_confirmed_not_round: bool = False
    # Облако 3D-точек, реально использованное реконструкцией (torchhull GPU / CPU-параллакс /
    # band-пересечение) — для визуализации ТОГО ЖЕ облака, по которому принят вердикт (не
    # отдельного независимого CPU-халла, см. задача «Доработка CV-отчёта»). None у вызывающих
    # мест, которые это поле не заполняют (обратная совместимость).
    points_3d: np.ndarray | None = None


def _degenerate_g4_result(mesh_dims: tuple[float, float, float], start_time: float) -> GRollResult:
    """Вырожденный случай реконструкции (объект не виден ни в одном силуэте/ракурсе, либо
    пустое пересечение) — консервативный вердикт not_round. Общий для всех путей
    реконструкции `check_model_roll_g4` (torchhull, CPU-параллакс через exact hull,
    band-пересечение), чтобы не дублировать один и тот же блок трижды."""
    return GRollResult(
        dims=None,
        mesh_dims=mesh_dims,
        checked_directions=0,
        best=None,
        best_axis_dir=None,
        best_projection_coords=None,
        best_roundness=None,
        found_round=False,
        verdict=ROLL_VERDICT_NOT_ROUND,
        fallback_triggered=False,
        axis_hits=[],
        elapsed_seconds=time.perf_counter() - start_time,
        method="g4",
    )


def _band_intersection_points(
    silhouettes: dict[float, Silhouette], num_slices: int
) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    """Облако 3D-точек по band-пересечению (`reconstruct_shape_from_silhouettes`) — старый
    CPU-путь G4 (один набор силуэтов, один belt_position). Вынесена отдельно, чтобы её мог
    переиспользовать откат CPU-параллакса (см. `check_model_roll_g4`), если
    `exact_polyhedral_hull.carve` вырождается (пустое/невозможное пересечение на конкретном
    меше) — без этого отказ параллакса ронял бы результат в None вместо разумного отката.
    None — силуэты не пересекаются ни в одном срезе (объект не виден)."""
    shape = reconstruct_shape_from_silhouettes(silhouettes, num_slices)
    if shape is None:
        return None
    chunks = []
    for x_pos, coords in shape.slices:
        arr = np.asarray(coords[:-1], dtype=np.float64)
        chunks.append(np.column_stack([np.full(len(arr), x_pos), arr[:, 0], arr[:, 1]]))
    return np.concatenate(chunks), shape.dims


def check_model_roll_g4(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    resolution_px: int = 1024,
    roundness_threshold: float = 0.8,
    low_threshold: float = 0.65,
    num_slices: int = 50,
    sweep_resolution_px: int = 512,
    local_search_radius_deg: float = 20.0,
    local_search_step_deg: float = 5.0,
    use_camera_silhouettes: bool = False,
    camera_fov_deg: float = 0.0,
    camera_distances: dict[float, float] | None = None,
    belt_position: str = "center",
    torchhull_level: int = 7,
    torchhull_parallax: bool = False,
    mesh_convex_hull: bool = False,
) -> GRollResult:
    """«Проверить модель» методом G4 — F1 + локальный поиск для пограничных случаев +
    трёхкатегорный вердикт. Реализует логику пользователя «лучше ошибиться в сторону
    круглого»: никаких тихих пропусков катающихся объектов в основной сортировщик.

    Алгоритм:
    1. Базовые 6 направлений F1: 3 PCA-оси + 3 диагонали между парами. Считаем k для
       каждого, берём max → `k_base`, лучшее направление `d_base`.
    2. Если `k_base ≥ roundness_threshold` (по умолчанию 0.8) → вердикт "round"
       (катится), без fallback.
    3. Если `k_base < low_threshold` (по умолчанию 0.65) → вердикт "not_round"
       (явно не катится), без fallback.
    4. Если `k_base ∈ [low_threshold, roundness_threshold)` — **зона неуверенности**:
       запускается `_local_grid_search` вокруг `d_base` (радиус `local_search_radius_deg`,
       шаг `local_search_step_deg`). Если локальный поиск нашёл k ≥ порога → вердикт
       "round". Иначе → вердикт "uncertain" (требует перепроверки оператором / ручным
       потоком), НЕ "not_round" — это и есть страховка от пропуска катающегося.

    Параметры `low_threshold`/`local_search_radius_deg`/`local_search_step_deg` —
    компромисс между точностью и скоростью. По умолчанию: low=0.65 (ловит cyl_skewed с
    k_base=0.691), радиус 20° и шаг 5° — достаточно чтобы найти ось в ~20° от базовых
    (подтверждено на расширенном тестовом наборе из 18 объектов).

    **Реконструкция формы** (откуда берётся облако точек для PCA/roundness):
    - **Analytical Mode** (`use_camera_silhouettes=False`, по умолчанию) —
      `reconstruct_shape_from_silhouettes` (band-пересечение по ортографическим силуэтам).
      18/18 точность на идеальных STL, 0 GPU, без зависимостей.
    - **Camera Mode** (`use_camera_silhouettes=True`) + torchhull доступен
      (`_TORCHHULL_AVAILABLE=True`) — `visual_hull_points` (GPU sparse voxel octree +
      marching cubes) по перспективным силуэтам. На параллаксе (`torchhull_parallax=True`,
      start/center/end по ленте) даёт 18/18 точность vs 16/18 у band-пересечения (см.
      bench_torchhull.py). Параметр `torchhull_level` — уровень октре (7=128³ по умолчанию,
      8=256³ для повышенной точности, 9=512³ для максимума, но ~1.6 GB VRAM на сложных
      объектах). На 8 GB GPU при level=7 — ~43 MB VRAM, не мешает SAM3 (2.5 GB).
    - **Camera Mode** + torchhull НЕдоступен — fallback на
      `reconstruct_shape_from_silhouettes` по перспективным силуэтам (band-пересечение,
      известно ломается на разных camera_distance при start/end — см. заметку задачи, но
      center работает). Это путь по умолчанию в окружениях без GPU.

    `mesh_convex_hull=True` — перед растеризацией заменить меш на его выпуклую оболочку
    (через `trimesh.convex_hull`). Выпуклая оболочка даёт ТУ ЖЕ силуэт (выпуклая оболочка
    проекции меша = проекция выпуклой оболочки меша), но МЕНЬШЕ треугольников — на тяжёлых
    мешах (Моющее средство: 73к faces → 7к hull faces, ~3× ускорение растеризации). **НО
    ломает точность G4**: вогнутости, которые G4 использует для отличия «катится/не катится»
    (Моющее средство: без hull → uncertain/k=0.764, с hull → round/k=0.816 — ложное
    срабатывание, 17/18 вместо 18/18). Использовать ТОЛЬКО если вогнутости объекта не
    несут информации о круглости (например, простые выпуклые формы). По умолчанию False.
    На прод-сценарии (маски от SAM3) меша нет, параметр не применяется.

    Скорость:
    - Аналитический режим: 14–25 мс (явные), ~150 мс (с fallback), 0 GPU.
    - Camera Mode + torchhull (level=7, PARALLAX): ~50–200 мс на большинстве объектов,
      худший случай ~1.4с (Моющее средство, 73к faces — растеризация меша, не torchhull).
      С `mesh_convex_hull=True` — ~10× быстрее на тяжёлых мешах.

    Возвращает `GRollResult` с `verdict` ∈ {round, not_round, uncertain} и
    `fallback_triggered` (диагностика)."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    # `mesh_convex_hull`: заменяем меш на выпуклую оболочку — силуэт тот же (выпуклая
    # оболочка проекции = проекция выпуклой оболочки), но треугольников меньше. trimesh
    # импортируется мягко (не обязателен для аналитики без этого флага).
    if mesh_convex_hull:
        try:
            import trimesh as _trimesh
            _t = _trimesh.Trimesh(vertices=oriented.vertices, faces=oriented.faces)
            _hull = _t.convex_hull
            oriented = Mesh(
                vertices=np.asarray(_hull.vertices, dtype=oriented.vertices.dtype),
                faces=np.asarray(_hull.faces, dtype=oriented.faces.dtype),
            )
        except Exception:
            pass  # trimesh недоступен или hull вырожден — используем исходный меш

    bbox_min, bbox_max = bounding_box(oriented.vertices)
    mesh_dims = tuple((bbox_max - bbox_min).tolist())

    silhouettes = build_silhouette_set(
        oriented,
        view_angles_deg,
        resolution_px,
        use_camera=use_camera_silhouettes,
        camera_fov_deg=camera_fov_deg,
        camera_distances=camera_distances,
        belt_position=belt_position,
    )

    start = time.perf_counter()

    # Реконструкция формы → облако 3D-точек для PCA/roundness. Три пути:
    # 1. Camera Mode + torchhull доступен — GPU visual hull (sparse voxel octree +
    #    marching cubes) по перспективным силуэтам. На параллаксе 18/18 точность
    #    (см. bench_torchhull.py), ~43 MB VRAM при level=7. Параллакс start/center/end
    #    включается через `torchhull_parallax=True` — даёт реальную информацию о форме
    #    с торца (камера «заглядывает» под углом на start/end).
    # 2. Camera Mode + torchhull НЕдоступен, но параллакс запрошен (`torchhull_parallax=True`,
    #    тот же чекбокс «+ начало/конец ленты», что и для GPU-пути) — CPU-параллакс через
    #    `exact_polyhedral_hull.carve` по всем (азимут, belt_position). ДО этого чекбокс без
    #    GPU молча ни на что не влиял: старый путь (см. 3) строит силуэты только для ОДНОГО
    #    `belt_position` и никогда не видел start/end (найдено пользователем на боксе,
    #    k=0.72 в GUI vs k=0.82 в независимом CV-отчёте — разница объяснилась ориентацией
    #    объекта, а не этим багом, но сам факт «чекбокс мёртв без GPU» — реальный пробел).
    # 3. Аналитический режим, ИЛИ torchhull недоступен и параллакс не запрошен —
    #    band-пересечение по силуэтам (`reconstruct_shape_from_silhouettes`, один
    #    belt_position). 18/18 на ортографике; поведение при `torchhull_parallax=False`
    #    не меняется относительно прежней версии (обратная совместимость).
    use_torchhull = (
        use_camera_silhouettes
        and _TORCHHULL_AVAILABLE
        and camera_distances is not None
    )
    use_cpu_parallax_hull = (
        not use_torchhull
        and use_camera_silhouettes
        and torchhull_parallax
        and camera_distances is not None
    )
    if use_torchhull:
        # Список (азимут, belt_position) для torchhull. При `torchhull_parallax=True` —
        # 3 позиции (start/center/end) на каждый азимут, иначе только center (как
        # выбранный в `belt_position`, но для torchhull всегда center — параллакс
        # управляется этим флагом, а не `belt_position`, т.к. torchhull принимает все
        # кадры одним батчем).
        belt_positions = ["start", "center", "end"] if torchhull_parallax else ["center"]
        views = [(angle, bp) for angle in view_angles_deg for bp in belt_positions]
        # Маски — без crop_to_object (torchhull требует одинаковый размер масок в батче).
        # `build_silhouette_set` выше уже построил силуэты для `belt_position` (текущий
        # селектор GUI) — но для torchhull могут понадобиться все 3 позиции, поэтому
        # строим недостающие напрямую. Для center переиспользуем уже построенные.
        masks = []
        dxs = []
        for angle, bp in views:
            if bp == belt_position and angle in silhouettes:
                masks.append(silhouettes[angle].mask)
            else:
                sil = camera_silhouette.build_silhouette(
                    oriented, angle, resolution_px,
                    fov_deg=camera_fov_deg,
                    camera_distance=camera_distances[angle],
                    belt_position=bp,
                    crop_to_object=False,
                )
                masks.append(sil.mask)
            dxs.append(camera_silhouette.belt_offset(
                oriented, camera_fov_deg, camera_distances[angle], bp,
            ))
        transforms = build_transforms_batch(
            [(angle, bp, dx) for (angle, bp), dx in zip(views, dxs)],
            fov_deg=camera_fov_deg,
            camera_distances=camera_distances,
            resolution_px=resolution_px,
        )
        # cube — bbox объекта + 10% margin (torchhull ищет форму в этом кубе).
        margin = float((bbox_max - bbox_min).max()) * 0.1
        cube_corner = [float(bbox_min[i] - margin) for i in range(3)]
        cube_length = float((bbox_max - bbox_min).max() + 2 * margin)
        masks_partial = any(bp != "center" for _, bp in views)
        points_3d = visual_hull_points(
            masks=masks,
            transforms=transforms,
            cube_corner_bfl=cube_corner,
            cube_length=cube_length,
            level=torchhull_level,
            masks_partial=masks_partial,
        )
        dims = None  # torchhull не возвращает dims в формате ReconstructedShape
        if points_3d is None or len(points_3d) < 4:
            return _degenerate_g4_result(mesh_dims, start)
    elif use_cpu_parallax_hull:
        # CPU-параллакс: точное (невыпуклое) пересечение обобщённых конусов обзора по ВСЕМ
        # (азимут, belt_position) — та же геометрия, что уже используется для визуализации
        # халла (`set_camera_hull_exact`/CV-отчёт), но здесь облако точек идёт в PCA/roundness
        # вместо отрисовки. `contour_epsilon_px=1.5` — согласовано с остальными вызывающими
        # G4/полутоп-халл (см. докстринг `polytope_hull.carve`).
        belt_positions_cpu = ["start", "center", "end"]
        views = [(angle, bp) for angle in view_angles_deg for bp in belt_positions_cpu]
        hull = exact_polyhedral_hull.carve(
            oriented, views, camera_fov_deg, camera_distances, resolution_px,
            contour_epsilon_px=1.5,
        )
        if hull is not None and len(hull.vertices) >= 4:
            points_3d = hull.vertices
            dims = None  # ExactHull не отдаёт dims в формате ReconstructedShape (как torchhull)
        else:
            # Вырожденное пересечение (редко — сложная вогнутая форма под конкретным ригом) —
            # мягкий откат на band-пересечение одним belt_position, а не отказ в None.
            band = _band_intersection_points(silhouettes, num_slices)
            if band is None:
                return _degenerate_g4_result(mesh_dims, start)
            points_3d, dims = band
    else:
        # Старый путь: band-пересечение (аналитика, Camera Mode без torchhull и без
        # запрошенного параллакса). Поведение не изменилось относительно прежней версии.
        band = _band_intersection_points(silhouettes, num_slices)
        if band is None:
            return _degenerate_g4_result(mesh_dims, start)
        points_3d, dims = band

    return _g4_core_from_points(
        points_3d,
        dims=dims,
        mesh_dims=mesh_dims,
        start_time=start,
        resolution_px=resolution_px,
        roundness_threshold=roundness_threshold,
        low_threshold=low_threshold,
        sweep_resolution_px=sweep_resolution_px,
        local_search_radius_deg=local_search_radius_deg,
        local_search_step_deg=local_search_step_deg,
        dense_points=use_torchhull,
    )


def _sustained_fraction_for_axis(
    points_3d: np.ndarray,
    hull_points: np.ndarray,
    d: np.ndarray,
    sweep_resolution_px: int,
    roundness_threshold: float,
) -> tuple[float | None, np.ndarray | None]:
    """Доля бинов вдоль оси `d`, где локальная проекция круглая (шаг 3 ТЗ [[Устойчивая локальная
    круглость вдоль оси переката (G4)]]). Вынесена отдельно от `_g4_core_from_points`, чтобы её
    можно было применить не только к `d_final`, но и к остальным базовым кандидатам («Важное
    уточнение» — проверка перед понижением вердикта). Возвращает (sustained_fraction, projected)
    — `projected` те же 2D-проекции на `sweep_resolution_px`, что вернул бы `_projection_k_for_axis`
    для этой оси (переиспользуются при промоции кандидата, не пересчитываются лишний раз).
    None — вырожденная проекция (r_out=0), долю для этой оси оценить нельзя.

    `hull_points` — вершины 3D convex hull ПОЛНОГО `points_3d` (уже посчитаны один раз в
    `_g4_core_from_points`, см. [[Оптимизация скорости G4-перебора (свести к вершинам convex
    hull)]]): используются ТОЛЬКО для глобальной r_out/projected (точка внутри 3D-hull остаётся
    внутри любой её 2D-проекции — тот же довод, что и в основном G4-переборе). 24-бинный ЛОКАЛЬНЫЙ
    проход ниже обязан использовать оригинальное `points_3d` — сокращение недопустимо там, точка,
    внутренняя для ГЛОБАЛЬНОЙ оболочки, может лежать на границе ЛОКАЛЬНОГО среза (тонкая
    перемычка/талия формы).

    Если `sustained_fraction` возвращён благодаря детерминированному раннему выходу (см. ниже) —
    это КОНСЕРВАТИВНАЯ НИЖНЯЯ ОЦЕНКА (`passed_so_far/_SUSTAINED_BINS`, всегда ≤ истинного
    значения), а не точная доля — булево решение (>=/< порога) при этом уже гарантированно верно
    (см. вывод в ТЗ), но САМО ЧИСЛО нельзя показывать пользователю/организатору как точное."""
    roundness, projected = _projection_k_for_axis(hull_points, d, sweep_resolution_px, roundness_threshold)
    if roundness is None or roundness.r_out <= 0:
        return None, projected
    r_out_global = roundness.r_out
    e1, e2 = _projection_basis(d)
    axis_vals = points_3d @ d
    lo, hi = float(axis_vals.min()), float(axis_vals.max())
    extent = hi - lo
    if extent < 1e-6:
        # вырожденное плоское облако — проверку пропустить, не портить существующий вердикт
        return 1.0, projected
    edges = np.linspace(lo, hi, _SUSTAINED_BINS + 1)
    bin_idx = np.clip(np.digitize(axis_vals, edges) - 1, 0, _SUSTAINED_BINS - 1)
    # Разовая сортировка по бину вместо пересчёта булевой маски (bin_idx == i) по ПОЛНОМУ
    # points_3d на каждой из 24 итераций — тот же результат (каждая точка в том же бине), без
    # повторных O(n) проходов по всему облаку.
    order = np.argsort(bin_idx, kind="stable")
    sorted_points = points_3d[order]
    bin_bounds = np.searchsorted(bin_idx[order], np.arange(_SUSTAINED_BINS + 1))
    proj_basis = np.stack([e1, e2], axis=1)
    passed = 0
    for i in range(_SUSTAINED_BINS):
        pts_bin = sorted_points[bin_bounds[i]:bin_bounds[i + 1]]
        if len(pts_bin) >= 3:
            proj_bin = pts_bin @ proj_basis
            # Дешёвый пре-фильтр (P0.5): половина диагонали 2D bbox — верхняя граница r_out
            # (минимальный охватывающий круг не может быть больше половины диагонали bbox,
            # т.к. диаметр точек не превышает диагональ их bbox). Если даже эта верхняя
            # граница уже ниже порога — бин точно вырожденный «кончик», полный расчёт
            # (convexHull+fillPoly+distanceTransform) можно пропустить без риска ложно
            # отбросить непустой бин.
            bbox_span = proj_bin.max(axis=0) - proj_bin.min(axis=0)
            r_out_upper_bound = float(np.hypot(*bbox_span)) / 2.0
            if r_out_upper_bound >= _SUSTAINED_MIN_RADIUS_FRAC * r_out_global:
                local = compute_projection_roundness(
                    proj_bin, resolution_px=_SUSTAINED_LOCAL_RES_PX, threshold=roundness_threshold
                )
                if local is not None and local.r_out >= _SUSTAINED_MIN_RADIUS_FRAC * r_out_global:
                    # непустой, не вырожденный «кончик» бин (вершина/острие не учитывается ни
                    # за, ни против — короб и так почти везде "вырожденный кончик", это тоже
                    # часть сигнала, не шум)
                    if local.k >= roundness_threshold:
                        passed += 1
        # Детерминированный ранний выход (не меняет финальный булев результат, см. вывод в
        # ТЗ): решение passed/_SUSTAINED_BINS >= _SUSTAINED_MIN_FRACTION уже предрешено, если
        # либо порог уже набран, либо оставшихся бинов заведомо не хватит его набрать.
        remaining = _SUSTAINED_BINS - (i + 1)
        if passed >= _SUSTAINED_MIN_PASSED or passed + remaining < _SUSTAINED_MIN_PASSED:
            break
    return passed / _SUSTAINED_BINS, projected  # знаменатель — ВСЕ бины, не только counted


def _g4_core_from_points(
    points_3d: np.ndarray,
    *,
    dims: tuple[float, float, float] | None,
    mesh_dims: tuple[float, float, float],
    start_time: float,
    resolution_px: int,
    roundness_threshold: float,
    low_threshold: float,
    sweep_resolution_px: int,
    local_search_radius_deg: float,
    local_search_step_deg: float,
    dense_points: bool = False,
) -> GRollResult:
    """Логика G4, начиная с готового облака 3D-точек (после реконструкции формы). Вынесена
    отдельно от `check_model_roll_g4`, чтобы её можно было вызывать на облаках от разных
    методов реконструкции (band-пересечение, voxel_carving, polytope_hull, exact_polyhedral_hull)
    без дублирования кода. Параметры — те же, что у `check_model_roll_g4` после реконструкции;
    `start_time` — момент старта замера elapsed_seconds (в вызывающем коде, до реконструкции).

    `dense_points` — облако точек изотропно семплировано по объёму/поверхности (сейчас — только
    torchhull, GPU sparse voxel octree + marching cubes), а не вдоль одной фиксированной оси
    среза реконструкции (band-пересечение/CPU-параллакс) — регрессия подтвердила, что признак
    «устойчивая локальная круглость» (`sustained_fraction`) надёжно применим к verdict ТОЛЬКО
    при `dense_points=True` (см. заметку задачи 2026-07-31, dev)."""
    if len(points_3d) < 4:
        return GRollResult(
            dims=dims,
            mesh_dims=mesh_dims,
            checked_directions=0,
            best=None,
            best_axis_dir=None,
            best_projection_coords=None,
            best_roundness=None,
            found_round=False,
            verdict=ROLL_VERDICT_NOT_ROUND,
            fallback_triggered=False,
            axis_hits=[],
            elapsed_seconds=time.perf_counter() - start_time,
            method="g4",
            points_3d=points_3d,
        )

    # --- Оптимизация G4-перебора (см. заметка задачи [[Оптимизация скорости G4-перебора
    # (свести к вершинам convex hull)]]): точка ВНУТРИ 3D-выпуклой оболочки облака остаётся
    # внутри любой её 2D-проекции при любом направлении — значит агрегированный k (перебор 6
    # базовых + локальный поиск + финальный пересчёт) не изменится, если считать его по
    # вершинам 3D convex hull вместо всего облака (на плотном torchhull-облаке — 12к-69к точек
    # — вершин хватает на порядок меньше). НЕ применяется к sustained-check (бины вдоль оси
    # ниже) — точка, внутренняя для ГЛОБАЛЬНОЙ оболочки, может лежать на границе ЛОКАЛЬНОГО
    # среза (тонкая перемычка/талия формы), поэтому там используется оригинальное `points_3d`.
    # Тоже не применяется к самому PCA (`_principal_axes_with_variance` ниже) — направления
    # главных осей зависят от распределения МАССЫ точек, а не только от границы оболочки,
    # вершины hull дали бы смещённые (неверные) оси.
    try:
        hull_points = points_3d[ConvexHull(points_3d).vertices]
    except Exception:
        # Вырожденное облако (плоское/коллинеарное — qhull не строит 3D-оболочку) —
        # безопасный откат на полное облако, как было до оптимизации.
        hull_points = points_3d

    # --- Шаг 1: 6 базовых направлений F1 (3 PCA + 3 диагонали) ---
    axes, _eigvals = _principal_axes_with_variance(points_3d)
    base_candidates: list[np.ndarray] = [axes[:, i] for i in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            d = axes[:, i] + axes[:, j]
            d = d / (np.linalg.norm(d) or 1.0)
            base_candidates.append(d)

    axis_hits: list[AxisHit] = []
    # Направление, соответствующее каждой записи axis_hits (тот же индекс) — нужно для «Важного
    # уточнения» ниже (проверка остальных 5 базовых кандидатов перед понижением вердикта), т.к.
    # AxisHit хранит только azim/elev/k, не сам вектор.
    base_axis_by_hit: list[np.ndarray] = []
    k_base = 0.0
    d_base: np.ndarray | None = None
    d_base_hit_index: int | None = None
    best_projected: np.ndarray | None = None
    for d in base_candidates:
        result, projected = _projection_k_for_axis(
            hull_points, d, sweep_resolution_px, roundness_threshold
        )
        if result is None:
            continue
        azim, elev = axis_to_angles(d)
        azim = azim % 180.0
        hit = AxisHit(axis_azim_deg=azim, axis_elev_deg=elev, k=result.k, passed=result.passed)
        axis_hits.append(hit)
        base_axis_by_hit.append(d)
        if result.k > k_base:
            k_base = result.k
            d_base = d
            best_projected = projected
            d_base_hit_index = len(axis_hits) - 1

    # --- Шаги 2–4: вердикт по k_base + локальный поиск для пограничных ---
    fallback_triggered = False
    k_final = k_base
    d_final = d_base
    projected_final = best_projected
    local_search_count = 0  # сколько направлений проверил локальный поиск (для диагностики)

    if k_base >= roundness_threshold:
        verdict = ROLL_VERDICT_ROUND
    elif k_base < low_threshold:
        verdict = ROLL_VERDICT_NOT_ROUND
    else:
        # Зона неуверенности [low, high) — локальный поиск вокруг d_base
        fallback_triggered = True
        if d_base is not None:
            k_local, d_local, proj_local = _local_grid_search(
                hull_points,
                d_base,
                local_search_radius_deg,
                local_search_step_deg,
                sweep_resolution_px,
                roundness_threshold,
            )
            # _local_grid_search проверяет (2*radius/step+1)^2 направлений — фиксируем для отчёта
            n_per_axis = int(2 * local_search_radius_deg / local_search_step_deg) + 1
            local_search_count = n_per_axis * n_per_axis
            if k_local > k_base:
                k_final = k_local
                d_final = d_local
                projected_final = proj_local
        verdict = ROLL_VERDICT_ROUND if k_final >= roundness_threshold else ROLL_VERDICT_UNCERTAIN

    # --- Устойчивость локальной круглости вдоль оси (см. [[Устойчивая локальная круглость
    # вдоль оси переката (G4)]]) — при verdict "round": короб/куб вдоль диагонали даёт высокий
    # АГРЕГИРОВАННЫЙ k, хотя локальное сечение круглое лишь на исчезающе малой доле длины оси.
    # При verdict "uncertain" (после неудавшегося fallback) — расширение признака (2026-07-31,
    # по явному решению пользователя): низкий sustained_fraction у ЛУЧШЕЙ найденной оси — это
    # ДОПОЛНИТЕЛЬНОЕ подтверждение non-roundness поверх уже низкого агрегированного k, даёт
    # уверенный not_round вместо ухода к оператору. Не считается для not_round (уже отсеян
    # напрямую по k_base<low_threshold, без fallback, без этой доп. проверки).
    sustained_fraction: float | None = None
    sustained_demoted = False
    sustained_promoted_from_alt_axis = False
    sustained_confirmed_not_round = False
    sustained_seconds: float | None = None
    if verdict == ROLL_VERDICT_ROUND and d_final is not None:
        sustained_start = time.perf_counter()
        sustained_fraction, _ = _sustained_fraction_for_axis(
            points_3d, hull_points, d_final, sweep_resolution_px, roundness_threshold
        )
        if sustained_fraction is not None and sustained_fraction < _SUSTAINED_MIN_FRACTION:
            sustained_demoted = True
            # `dense_points` (2026-07-31, dev): понижаем verdict ТОЛЬКО когда облако точек
            # пришло из изотропной по плотности реконструкции (torchhull — GPU sparse voxel
            # octree + marching cubes). На band-пересечении (`_band_intersection_points`,
            # CPU-путь без GPU) регрессия показала системный false positive — точки там
            # плотно семплированы только вдоль ФИКСИРОВАННОЙ оси среза реконструкции
            # (num_slices=50 по X), и при переразбиении на бины вдоль произвольной d_final
            # плотность по бинам скачет от 0 до ~170 точек — локальный k получается шумовым
            # артефактом семплирования, а не признаком формы. На torchhull эта проблема не
            # воспроизвелась: полная регрессия (Шлем/Пуфик/Бутылка/Тарелка/Мешок/ell_cyl/
            # asym_cone держат sustained_fraction 0.54–1.0, короб/куб по диагонали —
            # 0.04–0.17, чистое разделение с большим запасом) — см. #llm в заметке задачи.
            # sustained_fraction/sustained_demoted всегда остаются в GRollResult как
            # диагностика, независимо от `dense_points`.
            if dense_points:
                # «Важное уточнение» (см. ТЗ): d_final — не единственный кандидат, которого
                # касается эта проверка. Прежде чем понижать вердикт — прогнать ту же локальную
                # проверку по ОСТАЛЬНЫМ 5 базовым кандидатам (агрегированный k для них уже
                # посчитан на шаге 1, axis_hits/base_axis_by_hit), отсортированным по убыванию
                # k. Если найдётся кандидат с k ≥ порога И sustained_fraction ≥ порога — он
                # становится новым d_final/best, вердикт остаётся round. Понижаем в uncertain
                # (не not_round — новый признак не должен выбрасывать реально катящийся объект
                # из безопасной зоны) только если ни один из 6 не проходит оба условия.
                other_indices = sorted(
                    (i for i in range(len(axis_hits)) if i != d_base_hit_index),
                    key=lambda i: axis_hits[i].k,
                    reverse=True,
                )
                promoted = None
                for i in other_indices:
                    hit_k = axis_hits[i].k
                    if hit_k < roundness_threshold:
                        break  # отсортировано по убыванию k — дальше по списку только хуже
                    hit_axis = base_axis_by_hit[i]
                    hit_fraction, hit_projected = _sustained_fraction_for_axis(
                        points_3d, hull_points, hit_axis, sweep_resolution_px, roundness_threshold
                    )
                    if hit_fraction is not None and hit_fraction >= _SUSTAINED_MIN_FRACTION:
                        promoted = (hit_axis, hit_k, hit_fraction, hit_projected)
                        break
                if promoted is not None:
                    d_final, k_final, sustained_fraction, projected_final = promoted
                    sustained_promoted_from_alt_axis = True
                    sustained_demoted = False  # verdict больше не понижается этим признаком
                else:
                    verdict = ROLL_VERDICT_UNCERTAIN
        sustained_seconds = time.perf_counter() - sustained_start
    elif (
        verdict == ROLL_VERDICT_UNCERTAIN
        and fallback_triggered
        and dense_points
        and d_final is not None
    ):
        # Расширение на зону uncertain (2026-07-31, по запросу пользователя). Первая версия
        # была временно отключена (verdict не менялся) из-за регрессии на `cyl_skewed.stl` —
        # на тот момент документированного как «реально катится» (k=0.752, sustained=0.083,
        # неотличимо от короба-ловушки). Пользователь ЛИЧНО визуально осмотрел реальную форму
        # объекта и подтвердил: `cyl_skewed` физически НЕ катится — исходный эталон был
        # ошибочным (`robozon/config/objects.yaml` уже давно правильно относит его к
        # `category: ok`; исправлено в `docs/method.md`/`Ограничения.md`/bench-скриптах и
        # переименованном тесте `test_g4_cyl_skewed_false_positive_via_fallback`). Блокер снят
        # — расширение снова понижает verdict в not_round. Проверка безопасности (см. ниже)
        # осталась без изменений: прежде чем объявить not_round — убедиться, что ни у одного
        # из 6 базовых кандидатов НЕТ высокой sustained_fraction (иначе шумная реконструкция
        # могла испортить агрегированный k настоящей круглой оси — остаёмся в uncertain).
        sustained_start = time.perf_counter()
        sustained_fraction, _ = _sustained_fraction_for_axis(
            points_3d, hull_points, d_final, sweep_resolution_px, roundness_threshold
        )
        if sustained_fraction is not None and sustained_fraction < _SUSTAINED_MIN_FRACTION:
            # Находка A (см. [[Устойчивая локальная круглость вдоль оси переката (G4)]],
            # раздел «Ускорение sustained-check», п.5/P1): та же сортировка по убыванию k +
            # ранний выход по `hit_k < roundness_threshold`, что уже применяется в ROUND-ветке
            # выше — кандидат с агрегированным k ниже порога не может стать «круглой осью»,
            # проверять его локальную устойчивость бессмысленно. Раньше здесь перебирались ВСЕ
            # axis_hits без сортировки/отсечения (asymметрия с ROUND-веткой) — самые дорогие
            # случаи замера (`box_large`, 56к точек) добирались до not_round именно через этот
            # путь. Полная регрессия по вердиктам обязательна (см. заметка задачи).
            other_axes_desc_by_k = sorted(
                range(len(axis_hits)), key=lambda i: axis_hits[i].k, reverse=True
            )
            any_other_axis_round_like = False
            for i in other_axes_desc_by_k:
                if axis_hits[i].k < roundness_threshold:
                    break  # отсортировано по убыванию k — дальше по списку только хуже
                frac, _ = _sustained_fraction_for_axis(
                    points_3d, hull_points, base_axis_by_hit[i], sweep_resolution_px, roundness_threshold
                )
                if frac is not None and frac >= _SUSTAINED_MIN_FRACTION:
                    any_other_axis_round_like = True
                    break
            if not any_other_axis_round_like:
                sustained_confirmed_not_round = True
                verdict = ROLL_VERDICT_NOT_ROUND
        sustained_seconds = time.perf_counter() - sustained_start

    # --- Финальная подготовка результата: пересчёт k на полном разрешении ---
    best: AxisHit | None = None
    best_roundness: RoundnessResult | None = None
    best_projection_coords: list[tuple[float, float, float]] | None = None
    best_axis_dir: tuple[float, float, float] | None = None

    if d_final is not None and projected_final is not None:
        best_roundness_full, _ = _projection_k_for_axis(
            hull_points, d_final, resolution_px, roundness_threshold
        )
        if best_roundness_full is not None:
            k_final = best_roundness_full.k
            hull = cv2.convexHull(projected_final.astype(np.float32))[:, 0, :]
            ring = np.vstack([hull, hull[:1]])
            best_projection_coords = [(float(p[0]), float(p[1])) for p in ring]
            azim, elev = axis_to_angles(d_final)
            best = AxisHit(
                axis_azim_deg=azim % 180.0,
                axis_elev_deg=elev,
                k=k_final,
                passed=k_final >= roundness_threshold,
            )
            best_axis_dir = tuple(float(v) for v in d_final.tolist())
            best_roundness = best_roundness_full
            # Пересчёт вердикта по финальному k (на полном разрешении — точнее)
            if k_final >= roundness_threshold:
                verdict = ROLL_VERDICT_ROUND
            elif fallback_triggered and not sustained_confirmed_not_round:
                # После fallback — только round/uncertain (not_round уже не возвращаем,
                # т.к. base уже был в зоне неуверенности, не явным «не круглым»), КРОМЕ
                # случая sustained_confirmed_not_round — там это уже двойное подтверждение
                # (агрегированный k низкий И локальная круглость не держится ни у одной
                # оси), пересчёт на полном разрешении не должен тихо откатывать его обратно
                # в uncertain (если только k_final не пересёк порог выше — тот случай уже
                # обработан веткой if выше и имеет приоритет, что и требуется: полное
                # разрешение сказало round — значит round, безопасное направление ошибки).
                verdict = ROLL_VERDICT_UNCERTAIN
            # если fallback не запускался и k < threshold — вердикт остался not_round
            if dense_points and sustained_demoted:
                # Пересчёт k на полном разрешении (выше) иначе безусловно вернул бы verdict
                # обратно в "round" для той же оси d_final — устойчивость локальной круглости
                # уже проверена именно для этой оси и не зависит от разрешения агрегированной
                # проекции, поэтому демоция имеет приоритет над пересчётом.
                verdict = ROLL_VERDICT_UNCERTAIN

    return GRollResult(
        dims=dims,
        mesh_dims=mesh_dims,
        checked_directions=len(axis_hits) + local_search_count,
        best=best,
        best_axis_dir=best_axis_dir,
        best_projection_coords=best_projection_coords,
        best_roundness=best_roundness,
        found_round=verdict == ROLL_VERDICT_ROUND,
        verdict=verdict,
        fallback_triggered=fallback_triggered,
        axis_hits=axis_hits,
        elapsed_seconds=time.perf_counter() - start_time,
        method="g4",
        sustained_fraction=sustained_fraction,
        sustained_demoted=sustained_demoted,
        sustained_seconds=sustained_seconds,
        sustained_promoted_from_alt_axis=sustained_promoted_from_alt_axis,
        sustained_confirmed_not_round=sustained_confirmed_not_round,
        points_3d=points_3d,
    )


def _principal_axes_with_variance(
    points_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Три ортонормированные главные оси (столбцы (3,3)) и соответствующие собственные
    значения дисперсии (3,), по убыванию. То же, что `_principal_axes`, но возвращает и
    собственные значения — нужны для ослабленного критерия радиальной симметрии в F1
    (см. `check_model_roll_f1`): тело вращения имеет 2 равные дисперсии по осям,
    перпендикулярным оси симметрии."""
    centered = points_3d - points_3d.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(centered.T @ centered / max(len(centered), 1))
    # eigh сортирует по возрастанию; возвращаем в порядке [σ_max, σ_mid, σ_min]
    return eigvecs[:, ::-1], eigvals[::-1]


def check_model_roll_f1(
    mesh: Mesh,
    orientation: Orientation,
    view_angles_deg: list[float],
    resolution_px: int = 1024,
    roundness_threshold: float = 0.8,
    symmetry_ratio_threshold: float = 0.8,
    num_slices: int = 50,
    sweep_resolution_px: int = 512,
    use_camera_silhouettes: bool = False,
    camera_fov_deg: float = 0.0,
    camera_distances: dict[float, float] | None = None,
    belt_position: str = "center",
) -> PcaRollResult:
    """«Проверить модель» методом F1 — ослабленный критерий радиальной симметрии +
    финальный k по проекции (см. заметку задачи «Сравнить PCA-критерий круглой проекции
    с текущим перебором полусферы», раздел «Дополнение после требования надёжно и <0.5с»).

    Мотивация: PCA-метод (`check_model_roll_pca`) даёт 11/12 совпадений с эталоном, но
    пропускает шлем — у него ось качения косая, не совпадает с главной осью инерции.
    Готовый `trimesh.inertia.radial_symmetry` слишком жёсток (требует равенства 2 из 3
    главных моментов инерции с порогом 1e-4) — ловит только идеальные тела вращения.

    F1 = компромисс:
    1. PCA → 3 оси + 3 собственных значения дисперсии σ_i.
    2. Для каждой оси d_i: ratio_i = min(σ_others) / max(σ_others) — насколько 2 другие
       оси «равны» (тело вращения относительно d_i имеет 2 равные σ).
    3. Кандидаты = PCA-оси с ratio_i ≥ `symmetry_ratio_threshold` (по умолчанию 0.8)
       + 3 диагонали между парами PCA-осей (нормированные суммы). Диагонали добавлены
       для «косой оси качения» (шлем: k=0.858 на PCA0+PCA2 vs 0.783 на чистой PCA0).
       Если ни одна PCA-ось не прошла порог — fallback на все 3 (как `check_model_roll_pca`).
    4. Финальный вердикт = max k_i по проекциям вдоль всех кандидатов.

    Финальный k по проекции (та же `compute_projection_roundness`, что и везде) —
    страховка от ложных хитов ослабленного критерия: гантель с круглым стержнем и
    квадратными торцами имеет ratio ≈ 1.0 (торцы симметричны), но проекция вдоль
    стержня = квадрат, k ≈ 0.707 → отбрасывается.

    Параметры силуэтов/реконструкции совпадают с `check_model_roll_pca`. Возвращает
    тот же тип `PcaRollResult` с `method="f1"` и диагностикой по 3 осям в `axis_hits`.
    `elapsed_seconds` замеряет то же: от реконструкции формы до конца расчёта k."""
    oriented = apply_orientation(
        mesh, orientation.roll_deg, orientation.pitch_deg, orientation.yaw_deg
    )
    bbox_min, bbox_max = bounding_box(oriented.vertices)
    mesh_dims = tuple((bbox_max - bbox_min).tolist())

    silhouettes = build_silhouette_set(
        oriented,
        view_angles_deg,
        resolution_px,
        use_camera=use_camera_silhouettes,
        camera_fov_deg=camera_fov_deg,
        camera_distances=camera_distances,
        belt_position=belt_position,
    )

    start = time.perf_counter()
    shape = reconstruct_shape_from_silhouettes(silhouettes, num_slices)
    if shape is None:
        return PcaRollResult(
            dims=None,
            mesh_dims=mesh_dims,
            checked_directions=0,
            best=None,
            best_axis_dir=None,
            best_projection_coords=None,
            best_roundness=None,
            found_round=False,
            axis_hits=[],
            elapsed_seconds=time.perf_counter() - start,
            method="f1",
        )

    chunks = []
    for x_pos, coords in shape.slices:
        arr = np.asarray(coords[:-1], dtype=np.float64)
        chunks.append(np.column_stack([np.full(len(arr), x_pos), arr[:, 0], arr[:, 1]]))
    points_3d = np.concatenate(chunks)

    axes, eigvals = _principal_axes_with_variance(points_3d)
    # eigvals = [σ_max, σ_mid, σ_min], axes — соответствующие столбцы.
    # Для оси d_i (i=0,1,2) «остальные» дисперсии — eigvals без i-го элемента.
    # ratio_i = min(eigvals_others) / max(eigvals_others) — насколько 2 перпендикулярные
    # оси равны (тело вращения относительно d_i). ratio_i ≈ 1 → строгая радиальная
    # симметрия; ratio_i < threshold → асимметрия поперёк d_i.
    others = [
        np.array([eigvals[1], eigvals[2]]),  # для d_0 (σ_max)
        np.array([eigvals[0], eigvals[2]]),  # для d_1 (σ_mid)
        np.array([eigvals[0], eigvals[1]]),  # для d_2 (σ_min)
    ]
    ratios = np.array([o.min() / max(o.max(), 1e-30) for o in others])

    # Кандидаты на ось качения. Два источника:
    # 1. PCA-оси с ratio_i ≥ порога — «ослабленная радиальная симметрия» (тело
    #    вращения в широком смысле). Если ни одной — fallback на все 3 PCA-оси.
    # 2. Диагонали между парами PCA-осей (нормированные суммы единичных векторов) —
    #    на случай «косой оси качения», как у шлема (k=0.858 на PCA0+PCA2 vs 0.783 на
    #    чистой PCA0). Проверяются всегда, без фильтра по ratio — для них ratio не
    #    определён (это не главная ось инерции), но финальный k по проекции отбрасывает
    #    ложные хиты: гантель с квадратными торцами и т.п. на диагоналях даёт низкий k.
    pca_idx = [i for i, r in enumerate(ratios) if r >= symmetry_ratio_threshold]
    if not pca_idx:
        pca_idx = [0, 1, 2]

    candidates: list[np.ndarray] = [axes[:, i] for i in pca_idx]
    for i in range(3):
        for j in range(i + 1, 3):
            d = axes[:, i] + axes[:, j]
            d = d / (np.linalg.norm(d) or 1.0)
            candidates.append(d)

    axis_hits: list[AxisHit] = []
    best: AxisHit | None = None
    best_roundness: RoundnessResult | None = None
    best_projection_coords: list[tuple[float, float]] | None = None
    best_axis_dir: tuple[float, float, float] | None = None
    best_projected: np.ndarray | None = None

    for d in candidates:
        result, projected = _projection_k_for_axis(
            points_3d, d, sweep_resolution_px, roundness_threshold
        )
        if result is None:
            continue
        azim, elev = axis_to_angles(d)
        azim = azim % 180.0
        hit = AxisHit(axis_azim_deg=azim, axis_elev_deg=elev, k=result.k, passed=result.passed)
        axis_hits.append(hit)
        if best is None or result.k > best.k:
            best = hit
            best_roundness = result
            best_projected = projected
            best_axis_dir = tuple(float(v) for v in d.tolist())

    if best is not None and best_projected is not None and best_roundness is not None:
        hull = cv2.convexHull(best_projected.astype(np.float32))[:, 0, :]
        ring = np.vstack([hull, hull[:1]])
        best_projection_coords = [(float(p[0]), float(p[1])) for p in ring]
        # Пересчёт k на полном разрешении (как в check_model_roll_pca) — для итоговой метрики
        best_roundness_full, _ = _projection_k_for_axis(
            points_3d,
            np.array(best_axis_dir),
            resolution_px,
            roundness_threshold,
        )
        if best_roundness_full is not None:
            best = AxisHit(
                axis_azim_deg=best.axis_azim_deg,
                axis_elev_deg=best.axis_elev_deg,
                k=best_roundness_full.k,
                passed=best_roundness_full.passed,
            )
            best_roundness = best_roundness_full

    return PcaRollResult(
        dims=shape.dims,
        mesh_dims=mesh_dims,
        checked_directions=len(axis_hits),
        best=best,
        best_axis_dir=best_axis_dir,
        best_projection_coords=best_projection_coords,
        best_roundness=best_roundness,
        found_round=best is not None and best.passed,
        axis_hits=axis_hits,
        elapsed_seconds=time.perf_counter() - start,
        method="f1",
    )


def roundness_to_dict(result: RoundnessResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "r_in": result.r_in,
        "r_out": result.r_out,
        "k": result.k,
        "passed": result.passed,
        "center_in": list(result.center_in),
        "center_out": list(result.center_out),
    }


def evaluation_to_dict(result: EvaluationResult) -> dict:
    """JSON-сериализуемое представление для headless CLI (без масок силуэтов — те не
    предназначены для JSON, при необходимости сохраняются как PNG через export/)."""
    return {
        "view_angles_deg": result.view_angles_deg,
        "axis_pos": result.axis_pos,
        "bbox_min": list(result.bbox_min),
        "bbox_max": list(result.bbox_max),
        "polygon_coords": result.polygon_coords,
        "roundness": roundness_to_dict(result.roundness),
        "hull_dims": list(result.hull_dims) if result.hull_dims is not None else None,
        "mesh_dims": list(result.mesh_dims),
    }


def roll_check_to_dict(result: RollCheckResult) -> dict:
    """JSON-сериализуемое представление результата "Проверить модель" для headless CLI."""
    return {
        "dims": list(result.dims) if result.dims is not None else None,
        "true_dims": list(result.true_dims) if result.true_dims is not None else None,
        "mesh_dims": list(result.mesh_dims),
        "found_round": result.found_round,
        "checked_directions": result.checked_directions,
        "axis_step_deg": result.axis_step_deg,
        "best": (
            {
                "axis_azim_deg": result.best.axis_azim_deg,
                "axis_elev_deg": result.best.axis_elev_deg,
                "axis_dir": list(result.best_axis_dir) if result.best_axis_dir else None,
                "k": result.best.k,
                "passed": result.best.passed,
            }
            if result.best is not None
            else None
        ),
        "round_section_count": result.round_section_count,
        "section_count": len(result.section_hits),
        "elapsed_seconds": result.elapsed_seconds,
    }
