"""Camera Mode — перспективная проекция (идеальная pinhole-камера, без дисторсии).

Отдельная, ПАРАЛЛЕЛЬНАЯ Analytical Mode возможность посмотреть/сравнить силуэт объекта под
реальной перспективой (FOV камеры, конечное расстояние) вместо ортографической проекции с
бесконечно удалённой камерой. НЕ подключена к реконструкции формы/круглой проекции (см. заметку
задачи "Добавить Camera Mode..." — это Фаза 2, отдельная будущая работа).

Геометрия переиспользует оси Analytical Mode без изменений (`analytical.view_axes`,
`analytical.PRINCIPAL_AXIS`): азимут view_angle_deg задаёт направление наблюдения v_theta и
поперечную ось силуэта r_theta в плоскости Y-Z; продольная ось X — направление движения объекта
по ленте (roll axis), в кадре соответствует вертикали (см. docstring `silhouette/analytical.py`).

Модель камеры: камера в точке `camera_distance * v_theta`, смотрит вдоль `-v_theta`
("вперёд"), "право" = r_theta, "верх" = PRINCIPAL_AXIS (X). Сенсор — квадратный, один и тот же
FOV по обеим осям (см. `config/app_settings.yaml`, `camera_fov_deg`). Объект дополнительно
сдвигается вдоль X на `belt_offset(...)`, чтобы эмулировать положения начало/центр/конец на
ленте в пределах кадра конечного FOV — см. заметку задачи, раздел "Ответы пользователя".

Точки за камерой (глубина <= 0) отбрасываются вместе с треугольником — иначе деление на
глубину проецирует их с "перевёрнутым" знаком (артефакт позади объектива).

`Silhouette.px_per_unit`/`u_min`/`axis_min` здесь НОМИНАЛЬНЫЕ: в отличие от Analytical Mode,
масштаб перспективной проекции не постоянен по кадру (зависит от глубины каждой точки) — эти
поля посчитаны на номинальной глубине `camera_distance` (плоскость, где был бы центр объекта
без сдвига по глубине) и пригодны для грубой оценки масштаба/тестов, но не для точных обратных
пересчётов пиксель->мировые координаты, в отличие от Analytical Mode.
"""

from __future__ import annotations

from typing import Literal, Union

import cv2
import numpy as np

from ..geometry.mesh_io import Mesh
from .analytical import PRINCIPAL_AXIS, view_axes
from .base import Silhouette

# Позиция на ленте: строковые пресеты (как раньше) ИЛИ дробная доля максимального безопасного
# сдвига в [-1, 1] (-1 = start, 0 = center, +1 = end) — для ПЛОТНОГО параллакса по ленте
# (N кадров по мере движения товара, а не только 3 фиксированные позиции). Дробные позиции —
# та же геометрия, что start/end, просто с промежуточным множителем сдвига.
BeltPosition = Union[Literal["start", "center", "end"], float]

# Разрешение кадра: int — квадратный сенсор (историческое поведение, все старые вызовы);
# (n_rows, n_cols) — прямоугольный сенсор, строки = продольная ось ленты X, столбцы =
# поперечная ось силуэта u. Реальная камера 2592×1944 монтируется ДЛИННОЙ стороной вдоль
# ленты (максимум параллакса по движению) => resolution_px = (2592, 1944).
Resolution = Union[int, tuple[int, int]]

_BELT_SIGN = {"start": -1.0, "center": 0.0, "end": 1.0}
_MARGIN_FACTOR = 0.92  # доля доступного углового бюджета tan(fov/2), реально используемая в
# belt_offset — запас, чтобы объект не оказывался ровно на границе кадра (см. её докстринг)
_MIN_DEPTH = 1e-6  # порог отсечения точек за камерой/в её плоскости


def frame_shape(resolution_px: Resolution) -> tuple[int, int]:
    """(n_rows, n_cols) кадра: int — квадрат (обратная совместимость), tuple — как есть."""
    if isinstance(resolution_px, (tuple, list)):
        n_rows, n_cols = int(resolution_px[0]), int(resolution_px[1])
        return n_rows, n_cols
    return int(resolution_px), int(resolution_px)


def _belt_fraction(belt_position: BeltPosition) -> float:
    """Доля максимального сдвига по ленте в [-1, 1] из строкового пресета или числа."""
    if isinstance(belt_position, str):
        return _BELT_SIGN[belt_position]
    frac = float(belt_position)
    if not -1.0 <= frac <= 1.0:
        raise ValueError(f"belt_position (доля сдвига) должна быть в [-1, 1], получено {frac}")
    return frac


def camera_frame(
    view_angle_deg: float, fov_deg: float, camera_distance: float, resolution_px: Resolution
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Параметры камеры для азимута view_angle_deg: (camera_pos, forward, r_theta, f_px, cx, cy).

    `fov_deg` — угол обзора ВДОЛЬ ЛЕНТЫ (ось X = строки кадра): именно эта ось определяет,
    когда объект входит/выходит из кадра. Для прямоугольного сенсора FOV по столбцам не задаётся
    отдельно — пиксели квадратные, один f_px на обе оси (fov_cols = 2*atan(tan(fov/2)*n_cols/
    n_rows) получается автоматически). Для квадратного сенсора поведение побитово прежнее.

    Вынесено отдельно от `build_silhouette`, чтобы ту же проекционную математику мог
    переиспользовать `visual_hull.voxel_carving` (carving по произвольным 3D-точкам, а не
    только по вершинам треугольников меша) — см. `project_points` ниже."""
    v_theta, r_theta = view_axes(view_angle_deg)
    forward = -v_theta
    camera_pos = camera_distance * v_theta
    fov = np.radians(fov_deg)
    n_rows, n_cols = frame_shape(resolution_px)
    f_px = (n_rows / 2.0) / np.tan(fov / 2.0)
    cx = n_cols / 2.0  # столбцы (поперечная ось u)
    cy = n_rows / 2.0  # строки (продольная ось X)
    return camera_pos, forward, r_theta, f_px, cx, cy


def project_points(
    points: np.ndarray,
    view_angle_deg: float,
    fov_deg: float,
    camera_distance: float,
    resolution_px: Resolution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Проекция произвольных 3D-точек (уже со сдвигом `belt_offset`, если нужен) в пиксели
    той же камеры, что и `build_silhouette`. Возвращает (px_col, px_row, depth) — depth <= 0
    означает точку за камерой/в её плоскости (см. докстринг модуля)."""
    camera_pos, forward, r_theta, f_px, cx, cy = camera_frame(
        view_angle_deg, fov_deg, camera_distance, resolution_px
    )
    rel = points - camera_pos
    depth = rel @ forward
    px_col = f_px * (rel @ r_theta) / depth + cx
    px_row = cy - f_px * (rel @ PRINCIPAL_AXIS) / depth
    return px_col, px_row, depth


def belt_offset(
    mesh: Mesh, fov_deg: float, camera_distance: float, belt_position: BeltPosition
) -> float:
    """Смещение объекта по X (roll-ось) для эмуляции положения на ленте в пределах кадра.

    Точка меша на локальном (до сдвига) x проецируется в строку кадра как
    `px_row = cy - f_px*(x+dx)/depth`, где `depth = camera_distance - d`, а `d` — компонента
    точки вдоль направления взгляда v_theta (в диапазоне +-transverse_radius: диаметр объекта
    в плоскости Y-Z). Прежняя формула считала `depth ~= camera_distance` без поправки на
    `transverse_radius` — для объекта с заметным диаметром (не тонкой пластины) точки на
    БЛИЖНЕЙ к камере стороне (`d = +transverse_radius`, `depth` МЕНЬШЕ camera_distance)
    проецируются дальше от центра кадра, чем эта линейная оценка предполагала, и на позиции
    start/end реально обрезались по краю кадра (например, узкое горлышко вертикально стоящей
    бутылки — см. заметку задачи). `Δx_max = tan(fov/2)*(camera_distance - transverse_radius)
    - extent_x_from_axis` учитывает оба члена явно, вместо одного условного margin_factor на
    extent_x.

    `_MARGIN_FACTOR` применяется к ДОСТУПНОМУ угловому бюджету (`tan(fov/2)`), а не к
    extent_x_from_axis: у тонких вытянутых объектов (например, тонкая ручка, extent_x всего
    несколько мм) масштабирование запаса от extent_x давало исчезающе малый абсолютный отступ —
    расчётный худший угол оказывался практически ВПРИТЫК к границе кадра (доли процента запаса),
    и растровое округление до целого пикселя срезало объект на этой самой границе. Запас должен
    быть в единицах доступного угла, а не в единицах (возможно крошечного) размера объекта.

    `belt_position` может быть дробной долей в [-1, 1] (см. BeltPosition) — тогда возвращается
    соответствующая доля максимального безопасного сдвига (для плотного параллакса по ленте).
    """
    sign = _belt_fraction(belt_position)
    if sign == 0.0:
        return 0.0

    axis_vals = mesh.vertices @ PRINCIPAL_AXIS
    extent_x_from_axis = float(max(abs(axis_vals.min()), abs(axis_vals.max())))
    transverse = mesh.vertices - np.outer(axis_vals, PRINCIPAL_AXIS)  # проекция в плоскость Y-Z
    transverse_radius = float(np.linalg.norm(transverse, axis=1).max())

    fov = np.radians(fov_deg)
    safe_depth = camera_distance - transverse_radius
    delta_max = np.tan(fov / 2.0) * _MARGIN_FACTOR * safe_depth - extent_x_from_axis
    return sign * max(delta_max, 0.0)


def build_silhouette(
    mesh: Mesh,
    view_angle_deg: float,
    resolution_px: Resolution,
    fov_deg: float,
    camera_distance: float,
    belt_position: BeltPosition = "center",
    supersample: int = 1,
    crop_to_object: bool = False,
    crop_margin_px: float = 4.0,
) -> Silhouette:
    """Перспективный силуэт. `supersample > 1` — субпиксельный (мягкий) режим: растеризация
    ведётся на сетке в `supersample` раз мельче по каждой оси, затем усредняется блоками
    S×S в float-маску покрытия `Silhouette.coverage` (доля пикселя внутри силуэта, [0..1]).

    Зачем: бинарная маска несёт ±0.5px неопределённости на границе — для объектов, занимающих
    мало пикселей в кадре, этот растровый шум сопоставим с самим геометрическим сигналом
    (задокументированные баги A/B в docs/camera_dims_v2_investigation.md, Эксперимент 7;
    Pen 32.5% ошибки при 1024px). Дробное покрытие локализует границу с точностью ~1/S px —
    эквивалент S-кратного роста разрешения по границе без роста стоимости даунстрим-обработки
    (сама маска остаётся в native-разрешении). На реальной камере ту же роль играет
    субпиксельная локализация границы по градациям серого (см. investigation-доку).

    `mask` при этом остаётся бинарной (coverage >= 0.5) — все существующие потребители
    работают без изменений; `coverage` заполняется только при supersample > 1.

    `crop_to_object=True` — обрезать кадр до bbox маски + `crop_margin_px` (по 2px с каждой
    стороны): вместо маски полного кадра 512²/1024² (где тонкий объект может занимать <1%
    площади, особенно на start/end у края) вернуть маску, плотно покрывающую объект.
    Оптический центр кадра (старые `cx`/`cy` полного кадра) сохраняется в полях
    `Silhouette.cx`/`Silhouette.cy` в координатах обрезанной маски — конусные методы
    (polytope_hull/exact_polyhedral_hull/voxel_carving) используют их для корректной проекции
    в обрезанную маску. `u_min`/`axis_min`/`px_per_unit` остаются НОМИНАЛЬНЫМИ (как в полном
    кадре) — это сознательное упрощение, т.к. конусные методы не используют `u_min`/`axis_min`
    (они работают через `project_points` в мировых координатах); `reconstruct_shape_from_
    silhouettes` (band-пересечение через эти поля) на перспективе всё равно не работает
    корректно (см. docs/method.md §Camera Mode), ROI эту проблему не чинит, а только
    ускоряет растеризацию/обработку для конусных методов. По умолчанию False — обратная
    совместимость со всеми существующими вызывающими (полный кадр, `cx`/`cy` не
    переопределены, потребители используют `camera_frame`)."""
    if supersample < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}")
    camera_pos, forward, r_theta, f_px, cx, cy = camera_frame(
        view_angle_deg, fov_deg, camera_distance, resolution_px
    )
    n_rows, n_cols = frame_shape(resolution_px)

    dx = belt_offset(mesh, fov_deg, camera_distance, belt_position)
    verts = mesh.vertices + dx * PRINCIPAL_AXIS

    tri_verts = verts[mesh.faces]  # (F,3,3)
    rel = tri_verts - camera_pos  # (F,3,3)
    depth = rel @ forward  # (F,3) — глубина каждой вершины треугольника вдоль forward

    valid = depth.min(axis=1) > _MIN_DEPTH  # весь треугольник отбрасывается, если хоть одна
    # вершина за камерой/в её плоскости — иначе деление на почти-нулевую/отрицательную глубину
    # даёт мусорную проекцию (см. докстринг модуля)
    rel = rel[valid]
    depth = depth[valid]

    u = rel @ r_theta  # (F',3)
    x = rel @ PRINCIPAL_AXIS  # (F',3)

    s = int(supersample)
    # Аффинное отображение "координата native-пикселя -> координата тонкой сетки":
    # p_fine = s*p + (s-1)/2 (центры блоков S×S совпадают с центрами native-пикселей),
    # поэтому достаточно домножить f_px и сдвинуть оптический центр.
    f_fine = f_px * s
    cx_fine = cx * s + (s - 1) / 2.0
    cy_fine = cy * s + (s - 1) / 2.0

    px_col = f_fine * u / depth + cx_fine
    px_row = cy_fine - f_fine * x / depth  # верх изображения = большой X, как в Analytical Mode
    # np.round, а не голый astype(int32): усечение (truncate-towards-zero) для положительных
    # пиксельных координат равносильно floor и систематически (не случайно) сдвигает силуэт
    # на ~0.5px в одну сторону — при малом объекте в кадре (несколько десятков px) это давало
    # заметную асимметрию силуэта относительно центра кадра (см. заметку задачи, разбор бага).
    tri_px = np.round(np.stack([px_col, px_row], axis=-1)).astype(np.int32)  # (F',3,2)

    # ROI с ранней обрезкой: при `crop_to_object=True` определяем bbox проекции ВСЕХ
    # треугольников ДО растеризации и растеризуем только в этот bbox (+margin), а не в полный
    # кадр n_rows×n_cols. На объектах с большим числом треугольников (Моющее средство — 72к
    # faces) это даёт 5-20× ускорение: cv2.fillConvexPoly вызывается для каждого треугольника,
    # и накладные расходы на полный кадр 1024² доминируют (75% времени build_silhouette по
    # профилю). bbox определяется по min/max tri_px (дешёвая операция), с защитой по границам
    # кадра (объект у края на start/end). При `crop_to_object=False` — старый путь: полный
    # кадр, обратная совместимость.
    cx_out: float | None = None
    cy_out: float | None = None
    if crop_to_object and len(tri_px) > 0:
        all_cols = tri_px[:, :, 0]
        all_rows = tri_px[:, :, 1]
        c_lo = max(int(all_cols.min()) - int(crop_margin_px) - 1, 0)
        c_hi = min(int(all_cols.max()) + int(crop_margin_px) + 2, n_cols * s)
        r_lo = max(int(all_rows.min()) - int(crop_margin_px) - 1, 0)
        r_hi = min(int(all_rows.max()) + int(crop_margin_px) + 2, n_rows * s)
        if c_hi > c_lo and r_hi > r_lo:
            # Сдвигаем координаты треугольников в систему обрезанной маски (тонкой сетки).
            tri_px_roi = tri_px.copy()
            tri_px_roi[:, :, 0] -= c_lo
            tri_px_roi[:, :, 1] -= r_lo
            fine = np.zeros((r_hi - r_lo, c_hi - c_lo), dtype=np.uint8)
            for tri in tri_px_roi:
                cv2.fillConvexPoly(fine, tri, 255)
            # Оптический центр в координатах обрезанной маски: cx_fine/cy_fine — это центр
            # ПОЛНОГО кадра в тонкой сетке; в обрезанной маске он смещён на -c_lo/-r_lo.
            # Обратное преобразование к native-координатам: (p_fine - (s-1)/2) / s, поэтому
            # cx_native = (cx_fine - c_lo - (s-1)/2) / s, а не (cx_fine - c_lo) / s — но
            # потребитель (конусные методы) использует cx/cy для проекции в ОБРЕЗАННУЮ маску
            # в native-координатах, поэтому нужно вернуть native-cx/cy. Делаем обратное
            # преобразование явно.
            cx_out = (cx_fine - c_lo - (s - 1) / 2.0) / s
            cy_out = (cy_fine - r_lo - (s - 1) / 2.0) / s
            # Маска в native-разрешении обрезанной области: усреднение блоков S×S (как в
            # полном пути, но на обрезанной тонкой сетке).
            coverage = None
            contour_px = None
            if s > 1:
                # Обрезанная тонкая сетка может быть не кратна s по размеру — дополняем до
                # кратного нулями справа/снизу перед reshape. Это не влияет на маску в
                # native-разрешении, т.к. padding остаётся за пределами исходных данных.
                fine_h, fine_w = fine.shape
                pad_h = (-fine_h) % s
                pad_w = (-fine_w) % s
                if pad_h or pad_w:
                    fine_padded = np.pad(fine, ((0, pad_h), (0, pad_w)), mode="constant")
                else:
                    fine_padded = fine
                # Число native-блоков по каждой оси (может быть больше исходного обрезанного
                # native-размера из-за padding — обрежем после усреднения).
                n_blocks_h = fine_padded.shape[0] // s
                n_blocks_w = fine_padded.shape[1] // s
                coverage = (
                    fine_padded.reshape(n_blocks_h, s, n_blocks_w, s)
                    .mean(axis=(1, 3)).astype(np.float32) / 255.0
                )
                # Native-размер обрезанной маски (без padding): ceil((r_hi-r_lo)/s), но
                # coverage может быть больше — обрежем до исходной обрезанной native-области.
                native_h = (r_hi - r_lo + s - 1) // s
                native_w = (c_hi - c_lo + s - 1) // s
                coverage = coverage[:native_h, :native_w]
                mask = coverage >= 0.5
                # Субпиксельный контур в native-координатах обрезанной маски.
                contours, _ = cv2.findContours(fine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest) > 0:
                        pts = largest.reshape(-1, 2).astype(np.float64)
                        contour_px = (pts - (s - 1) / 2.0) / s
            else:
                mask = fine.astype(bool)
            # u_min/axis_min — номинальные (для совместимости, конусные методы их не
            # используют; reconstruct_shape_from_silhouettes на перспективе не работает
            # независимо от ROI — см. докстринг функции).
            px_per_unit = f_px / camera_distance
            return Silhouette(
                mask=mask,
                view_angle_deg=view_angle_deg,
                u_min=-cx / px_per_unit,
                axis_min=-cy / px_per_unit,
                px_per_unit=px_per_unit,
                coverage=coverage,
                contour_px=contour_px,
                cx=cx_out,
                cy=cy_out,
            )

    # Полный кадр (crop_to_object=False или вырожденный bbox).
    fine = np.zeros((n_rows * s, n_cols * s), dtype=np.uint8)
    for tri in tri_px:
        cv2.fillConvexPoly(fine, tri, 255)

    coverage = None
    contour_px = None
    if s > 1:
        coverage = (
            fine.reshape(n_rows, s, n_cols, s).mean(axis=(1, 3)).astype(np.float32) / 255.0
        )
        mask = coverage >= 0.5
        # Субпиксельный контур: крупнейший внешний контур ТОНКОЙ сетки, пересчитанный в
        # нативные пиксельные координаты (обратное к p_fine = s*p + (s-1)/2) — граница
        # локализована с точностью ~1/s px, что и является главным выигрышем супсемплинга
        # для тонких объектов (Pen: ±0.5px бинарной границы = проценты размера).
        contours, _ = cv2.findContours(fine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0:
                pts = largest.reshape(-1, 2).astype(np.float64)
                contour_px = (pts - (s - 1) / 2.0) / s
    else:
        mask = fine.astype(bool)

    # ROI с поздней обрезкой (fallback): если ранняя обрезка не сработала (bbox вырожден,
    # tri_px пустой), а crop_to_object=True — пробуем обрезать готовую маску. Случай редкий
    # (только вырожденные меши), но оставлен для устойчивости.
    if crop_to_object:
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size > 0 and cols.size > 0:
            r_lo = max(int(rows.min() - crop_margin_px), 0)
            r_hi = min(int(rows.max() + crop_margin_px + 1), n_rows)
            c_lo = max(int(cols.min() - crop_margin_px), 0)
            c_hi = min(int(cols.max() + crop_margin_px + 1), n_cols)
            if r_hi > r_lo and c_hi > c_lo:
                mask = mask[r_lo:r_hi, c_lo:c_hi]
                if coverage is not None:
                    coverage = coverage[r_lo:r_hi, c_lo:c_hi]
                if contour_px is not None:
                    contour_px = contour_px - np.array([c_lo, r_lo], dtype=np.float64)
                cx_out = cx - c_lo
                cy_out = cy - r_lo

    px_per_unit = f_px / camera_distance
    return Silhouette(
        mask=mask,
        view_angle_deg=view_angle_deg,
        u_min=-cx / px_per_unit,
        axis_min=-cy / px_per_unit,
        px_per_unit=px_per_unit,
        coverage=coverage,
        contour_px=contour_px,
        cx=cx_out,
        cy=cy_out,
    )
