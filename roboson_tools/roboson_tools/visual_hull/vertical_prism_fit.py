"""Точные габариты вертикальной призмы (коробка, треугольная/N-угольная колонна и т.п.) по
Camera Mode — заметка задачи "камера сверху + под углом, расстояние от края ленты в пикселях".

Идея (обсуждение с пользователем): для объекта с вертикальными стенками (постоянное поперечное
сечение по высоте) верхняя камера даёт ТОЧНУЮ форму горизонтального сечения — но с точностью до
неизвестного пока однородного масштаба (см. ниже, почему). Высота находится БЕЗ явного
классического stereo-соответствия точек между кадрами — используется тот же инвариант, что и
во всей остальной методике (docs/method.md): силуэт всегда СОДЕРЖИТ объект. Кандидат-призма
(сечение с верхней камеры на высоте h, экструдированное от земли до h) должна помещаться внутри
силуэтов боковых камер при h <= истинной высоте и вылезать за них при h > истинной — бинарным
поиском находится наибольшее h, при котором кандидат ещё помещается во все боковые силуэты
(containment). Как только h найдено, тот же h снимает и масштабную неоднозначность верхнего
контура — получаются точные (X, Y) каждого угла.

Почему масштаб контура сверху НЕОДНОЗНАЧЕН без знания h: камера стоит строго над объектом,
смотрит вертикально вниз (v_theta=(0,0,1), depth = camera_distance - Z), поэтому весь контур на
высоте Z проецируется с ОДНИМ общим масштабом f/(camera_distance-Z) — форма контура точна
(углы/пропорции), но абсолютный размер линейно зависит от Z поверхности, которую камера
физически видит. Для сплошного объекта с вертикальными стенками сверху видна верхняя грань
(Z=h), а не основание (Z=0) — самый близкий к камере слой их и закрывает. Наивное допущение
"контур сверху — это основание" занизило бы размер на множитель (camera_distance-h)/camera_distance.

**Предусловие: Z=0 меша — это уровень ленты/земли** (объект должен быть уже размещён так, что
касается земли в Z=0 — как и `camera_distance` реального рига, это калибровочная константа
стенда, а не то, что выводится из самого меша/силуэтов; за пределами этого модуля нигде в
кодовой базе такое допущение не действует, ориентированный меш из core/experiment.py центрирован
по bbox и НЕ касается Z=0 — вызывающий код должен сдвинуть меш сам).

Отбраковка непризматических объектов (конус, шлем, бутылка и т.п.) — два независимых сигнала на
контуре сверху, ДО дорогого бинарного поиска: контур не сводится к <= `max_corners` углам через
`cv2.approxPolyDP` (гладкий/круглый контур не сжимается до малого числа вершин при разумном
epsilon), и/или отношение площади многоугольника к площади минимальной описанной окружности
(`_footprint_roundness_ratio`) выше `roundness_reject_ratio` (форма слишком похожа на круг). В
обоих случаях — `None`, вызывающий код использует общую (менее точную) реконструкцию. Полностью
не завязано на конкретное число углов — треугольная колонна (`sections=3` у
`trimesh.creation.cylinder`) работает так же, как коробка (N=4) или шестигранник (N=6): и то, и
другое — вертикальная призма с постоянным сечением, единственное отличие — сколько вершин найдёт
`cv2.approxPolyDP` на контуре сверху.

**Круглое сечение (вертикальный цилиндр) — N-угольник с N->бесконечность, не отдельный случай.**
Обсуждение с пользователем (2026-07, `docs/camera_dims_v2_investigation.md`) — поза "пуфик
плашмя" даёт у верхней камеры круглый контур, который раньше отбраковывался ЦЕЛИКОМ той же
проверкой `_footprint_roundness_ratio`, что защищает от ложной подгонки купольных/конических
объектов. Но у цилиндра сечение ТАК ЖЕ постоянно по высоте, как у любой призмы — разница только
в форме контура сверху (окружность вместо многоугольника), сама идея (верхняя камера даёт точную
форму сечения с точностью до масштаба, боковые камеры снимают масштабную неоднозначность через
бинарный поиск по высоте) не меняется. Замерен реальный радиальный профиль Пуфика в этой позе —
колебание радиуса < 3% по всей высоте (не купол/конус, где радиус монотонно падает к краям) —
конкретное эмпирическое основание считать эту форму цилиндром, а не отбраковывать по круглости.

Реализовано так: если контур НЕ сводится к <= `max_corners` углам (либо сводится, но получившийся
многоугольник сам оказался круглым — прежняя проверка `_footprint_roundness_ratio`), вместо
немедленного `None` пробуется ВТОРАЯ ветвь — `cv2.minEnclosingCircle` по НЕупрощённой выпуклой
оболочке контура; если ОНА достаточно круглая, генерируется плотная выборка точек по окружности
(`_CIRCLE_SAMPLE_POINTS`, по умолчанию 48) и используется как footprint ДАЛЬШЕ ПО ТОМУ ЖЕ
КОДУ, что и многоугольник (`_footprint_at_height`/`_prism_mesh`/`_excess_and_uncovered_pixels`/`_fit_height` не
знают и не должны знать, откуда взялись точки контура) — окружность и есть предельный случай
N-угольника для этой геометрии, отдельного алгоритма не требуется. Настоящие купольные/конические
объекты (Тарелка, Шлем, Бутылка) по-прежнему отбраковываются: `max_relative_excess` (остаточное
несовпадение кандидата-цилиндра с боковыми силуэтами ПОСЛЕ подбора высоты) не проверяет форму
контура сверху, а проверяет, действительно ли ПОСТОЯННОЕ по высоте сечение объясняет то, что видят
боковые камеры — для сужающегося к краям объекта результат не сойдётся, и это тот же защитный
механизм, что уже отбраковывает конус в тестах, независимо от круглости основания."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon

from ..geometry.mesh_io import Mesh, from_trimesh
from ..silhouette import camera as camera_silhouette
from ..silhouette.analytical import view_axes
from ..silhouette.base import Silhouette

_MIN_SIDE_ANGLES = 2  # хотя бы 2 боковых ракурса, иначе высота недостаточно ограничена
_CIRCLE_SAMPLE_POINTS = 48  # плотность выборки точек по окружности для круговой ветви — не
# физический смысл (не число ракурсов рига), а геометрическая дискретизация footprint, дальше
# используемая тем же кодом, что и многоугольник (см. докстринг модуля)


@dataclass
class PrismFitResult:
    footprint_xy: list[tuple[float, float]]  # мировые (X, Y) углов сечения (или точек
    # окружности при is_circle=True), N штук
    height: float  # мировая высота (вдоль Z) от уровня земли (Z=ground_z) до верхней грани
    corners: int  # для многоугольника — реальное число углов; для окружности —
    # _CIRCLE_SAMPLE_POINTS (плотность выборки, не физический смысл, см. is_circle)
    is_circle: bool  # True — footprint_xy аппроксимирует окружность (вертикальный цилиндр,
    # круглое сечение), а не настоящий многоугольник; см. докстринг модуля
    top_angle_deg: float
    ground_z: float = 0.0  # Z уровня земли/ленты В ИСХОДНОЙ (не сдвинутой) системе координат
    # меша, который передавали в fit_vertical_prism — сам fit_vertical_prism требует Z=0 на
    # входе, поэтому ничего о ground_z не знает; заполняется вызывающим кодом (см.
    # core/experiment.check_vertical_prism), нужен только для `prism_mesh` ниже (обратно
    # сдвинуть солид в кадр, где рисуется 3D-модель на Panel 1 — она НЕ сдвинута по Z)


def prism_mesh(result: PrismFitResult) -> Mesh:
    """Солид подобранной призмы (для 3D-визуализации, Panel 1) — та же геометрия, что
    использовалась внутри поиска высоты (`_prism_mesh`), но сдвинутая на `result.ground_z`
    обратно в исходную систему координат меша."""
    footprint = np.asarray(result.footprint_xy, dtype=np.float64)
    solid = _prism_mesh(footprint, result.height)
    return Mesh(
        vertices=solid.vertices + np.array([0.0, 0.0, result.ground_z]),
        faces=solid.faces,
    )


def _corners_pixel(
    mask: np.ndarray, max_corners: int, epsilon_ratio: float
) -> np.ndarray | None:
    """Углы контура силуэта в ПИКСЕЛЯХ (col, row), с точностью до центра пикселя. None —
    контур пуст/вырожден, либо не сводится к <= max_corners вершинам (не многогранник)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :].astype(np.float64)
    if not (3 <= len(approx) <= max_corners):
        return None
    return approx + 0.5  # центр пикселя, не угол


def _footprint_roundness_ratio(pixel_corners: np.ndarray) -> float:
    """Площадь многоугольника / площадь минимальной описанной окружности — близко к 1 у
    круга, заметно меньше у настоящего многогранника (квадрат ~0.64, треугольник ~0.41)."""
    pts = pixel_corners.astype(np.float32)
    poly_area = cv2.contourArea(pts)
    (_cx, _cy), radius = cv2.minEnclosingCircle(pts)
    circle_area = np.pi * radius * radius
    if circle_area <= 0:
        return 1.0
    return float(poly_area / circle_area)


def _contour_hull_pixel(mask: np.ndarray) -> np.ndarray | None:
    """Выпуклая оболочка контура силуэта сверху в ПИКСЕЛЯХ, БЕЗ упрощения до малого числа
    вершин (в отличие от `_corners_pixel`) — нужна, чтобы оценить круглость формы даже когда
    она не сводится к <= max_corners углам через `cv2.approxPolyDP` (гладкий контур). None —
    контур пуст/вырожден."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    hull = cv2.convexHull(contour)[:, 0, :].astype(np.float64)
    return hull + 0.5  # центр пикселя, та же конвенция, что у _corners_pixel


def _circle_footprint_pixel(hull_pixel: np.ndarray, n_points: int) -> np.ndarray:
    """Плотная выборка точек по окружности, вписывающей выпуклую оболочку контура сверху
    (`cv2.minEnclosingCircle`) — предельный случай N-угольника при N->бесконечность для
    вертикального цилиндра (см. докстринг модуля). Точки используются ДАЛЬШЕ ТЕМ ЖЕ КОДОМ,
    что и `_corners_pixel` (`_footprint_at_height`/`_prism_mesh`/`_excess_and_uncovered_pixels`) — они не
    различают, откуда взялся footprint."""
    pts = hull_pixel.astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(pts)
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    circle = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    return circle.astype(np.float64)


def _footprint_at_height(
    pixel_corners: np.ndarray,
    top_meta: tuple[float, float, float, np.ndarray],
    top_camera_distance: float,
    height: float,
) -> np.ndarray:
    """(N,2) мировые (X,Y) точек контура сверху ПРИ ГИПОТЕЗЕ, что видимая камерой поверхность
    лежит на высоте `height` (см. докстринг модуля — контур точен по форме, но масштаб зависит
    от того, на какой высоте физически лежит то, что камера видит). `top_camera_distance` —
    дистанция ИМЕННО верхней камеры (см. `core/camera_rig.py` — теперь может отличаться от
    дистанций боковых камер)."""
    cx, cy, f_px, r_theta = top_meta
    depth = top_camera_distance - height
    x = (cy - pixel_corners[:, 1]) * depth / f_px  # row -> X (PRINCIPAL_AXIS)
    u = (pixel_corners[:, 0] - cx) * depth / f_px  # col -> проекция на r_theta
    y = u * r_theta[1]  # r_theta = (0, r_y, 0) для строго вертикальной камеры (см. проверка выше)
    return np.stack([x, y], axis=1)


def _prism_mesh(footprint_xy: np.ndarray, height: float) -> Mesh:
    polygon = Polygon(footprint_xy)
    solid = trimesh.creation.extrude_polygon(polygon, height=height)
    return from_trimesh(solid, center=False)


def _excess_and_uncovered_pixels(
    pixel_corners: np.ndarray,
    top_meta: tuple[float, float, float, np.ndarray],
    top_camera_distance: float,
    side_silhouettes: dict[float, Silhouette],
    side_camera_distances: dict[float, float],
    fov_deg: float,
    resolution_px: int,
    height: float,
) -> tuple[int, int]:
    """(excess, uncovered) кандидата-призмы на высоте `height`, суммарно по всем боковым
    камерам, за ОДИН проход рендера (не два раздельных): `excess` — пиксели кандидата ВНЕ
    наблюдаемых силуэтов, `uncovered` — пиксели наблюдаемых силуэтов, которые кандидат НЕ
    покрыл. Вместе — симметрическая разность (XOR) кандидата и реальных силуэтов.

    ПОЧЕМУ НЕ ТОЛЬКО `excess` (было раньше — см. историю модуля): при ЗАНИЖЕННОМ h контур
    сверху пересчитывается с БОЛЬШИМ множителем масштаба (depth = camera_distance-h больше при
    меньшем h) — сечение получается ШИРЕ истинного, поэтому `excess` действительно растёт при
    занижении... но растёт МЕДЛЕННЕЕ, чем падает при завышении, и на объектах, маленьких в
    кадре (единицы-десятки пикселей — растровый шум сопоставим с сигналом, см. `docs/
    camera_dims_v2_investigation.md`, Эксперимент 7 баг A, найдено пользователем на маленькой
    коробке и на ЛанчБоксе — "верх призмы не должен быть ниже верха контура"), `excess` может
    иметь ЛОЖНЫЙ минимум ЗАМЕТНО НИЖЕ истинной высоты: заниженный кандидат просто не дотягивает
    до верхней части объекта (не покрывает её), не вылезая при этом никуда — `excess` там мал,
    хотя кандидат явно неверен. `uncovered` ловит именно эту ошибку (та же идея, что уже
    использовалась для отбраковки конуса через `min_relative_coverage`, — здесь применена не
    постфактум единожды, а НА КАЖДОМ шаге поиска, чтобы сам поиск не сходился в эту ложную
    точку). Сумма `excess+uncovered` эмпирически даёт минимум В ПРЕДЕЛАХ шага сетки от истинной
    высоты (проверено на коробке 80×60×40: старый минимум `excess` был на h≈32-38 при истине
    40, минимум `excess+uncovered` — на h≈38-41)."""
    h = max(height, 1e-6)
    footprint = _footprint_at_height(pixel_corners, top_meta, top_camera_distance, h)
    candidate_mesh = _prism_mesh(footprint, h)
    excess_total = 0
    uncovered_total = 0
    for angle, sil in side_silhouettes.items():
        candidate_sil = camera_silhouette.build_silhouette(
            candidate_mesh, angle, resolution_px, fov_deg, side_camera_distances[angle], "center"
        )
        excess_total += int(np.count_nonzero(candidate_sil.mask & ~sil.mask))
        uncovered_total += int(np.count_nonzero(sil.mask & ~candidate_sil.mask))
    return excess_total, uncovered_total


def _fit_height(
    pixel_corners: np.ndarray,
    top_meta: tuple[float, float, float, np.ndarray],
    top_camera_distance: float,
    side_silhouettes: dict[float, Silhouette],
    side_camera_distances: dict[float, float],
    fov_deg: float,
    resolution_px: int,
    h_max: float,
    coarse_steps: int,
    refine_iterations: int,
) -> tuple[float, int, int]:
    """Грубый скан + локальное тернарное уточнение минимума `excess+uncovered` (см. докстринг
    `_excess_and_uncovered_pixels` — почему не один только excess). Возвращает (высота, excess,
    uncovered на найденной высоте — используются вызывающим кодом как мера доверия к
    результату)."""

    def combined_score(h: float) -> int:
        excess, uncovered = _excess_and_uncovered_pixels(
            pixel_corners, top_meta, top_camera_distance, side_silhouettes,
            side_camera_distances, fov_deg, resolution_px, h,
        )
        return excess + uncovered

    grid = np.linspace(0.0, h_max, coarse_steps + 1)
    scores = [combined_score(float(h)) for h in grid]
    best_idx = int(np.argmin(scores))
    lo = float(grid[max(best_idx - 1, 0)])
    hi = float(grid[min(best_idx + 1, coarse_steps)])

    for _ in range(refine_iterations):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if combined_score(m1) <= combined_score(m2):
            hi = m2
        else:
            lo = m1

    best_h = (lo + hi) / 2.0
    excess, uncovered = _excess_and_uncovered_pixels(
        pixel_corners, top_meta, top_camera_distance, side_silhouettes, side_camera_distances,
        fov_deg, resolution_px, best_h,
    )
    return best_h, excess, uncovered


def fit_vertical_prism(
    mesh: Mesh,
    view_angles_deg: list[float],
    fov_deg: float,
    camera_distances: dict[float, float],
    resolution_px: int,
    top_angle_deg: float = 90.0,
    max_corners: int = 8,
    roundness_reject_ratio: float = 0.85,
    epsilon_ratio: float = 0.02,
    h_max_ratio: float = 0.8,
    h_max_margin: float = 1.3,
    coarse_steps: int = 24,
    refine_iterations: int = 16,
    max_relative_excess: float = 0.02,
    min_relative_coverage: float = 0.8,
    excess_pixel_margin: float = 5.0,
) -> PrismFitResult | None:
    """Точные (X, Y) сечения + высота для вертикальной призмы ИЛИ цилиндра (`result.is_circle`,
    см. докстринг модуля — окружность как предельный случай N-угольника), либо None (объект не
    похож ни на одно из двух, либо в наборе ракурсов нет камеры строго сверху / не хватает
    боковых, либо кандидат на лучшей высоте не проходит ОДНУ ИЗ ДВУХ проверок консистентности:
    `max_relative_excess` (кандидат не должен вылезать за боковые силуэты) и
    `min_relative_coverage` (кандидат должен объяснять БОЛЬШУЮ ЧАСТЬ боковых силуэтов, а не
    только маленький кусочек). Вторая проверка обязательна именно из-за круговой ветви: у конуса
    (апексом к камере) контур сверху математически консистентен с боковыми силуэтами уже при
    h≈0 — тонкий диск у широкого основания не вылезает никуда (`max_relative_excess` прошла бы),
    но и не объясняет сам конус выше основания (см. докстринг `_excess_and_uncovered_pixels`). Требует, чтобы
    `mesh` уже был размещён так, что касается земли в Z=0 (см. докстринг модуля), и работает
    только с `belt_position="center"` (без сдвига по X) — упрощение, начальная версия метода.
    `camera_distances` — обязательная карта {азимут -> дистанция}, по одной на КАЖДЫЙ угол из
    `view_angles_deg` включая `top_angle_deg` (см. `core/camera_rig.py`); `KeyError` при
    отсутствии угла — намеренно, без фоллбека.
    """
    v_theta, r_theta = view_axes(top_angle_deg)
    if abs(v_theta[0]) > 1e-9 or abs(v_theta[1]) > 1e-9 or abs(v_theta[2] - 1.0) > 1e-9:
        raise ValueError(
            "top_angle_deg должен задавать камеру строго сверху (v_theta=(0,0,1)), обычно 90°"
        )

    side_angles = [a for a in view_angles_deg if a != top_angle_deg]
    if len(side_angles) < _MIN_SIDE_ANGLES:
        return None

    top_camera_distance = camera_distances[top_angle_deg]
    top_sil = camera_silhouette.build_silhouette(
        mesh, top_angle_deg, resolution_px, fov_deg, top_camera_distance, "center"
    )
    polygon_corners = _corners_pixel(top_sil.mask, max_corners, epsilon_ratio)
    is_circle = False
    if polygon_corners is not None and _footprint_roundness_ratio(polygon_corners) < roundness_reject_ratio:
        # Настоящий N-угольник (вертикальные рёбра, не круглое сечение) — прежнее поведение.
        pixel_corners = polygon_corners
    else:
        # Не свёлся к <= max_corners углам, либо свёлся, но сам получившийся многоугольник
        # оказался круглым — пробуем вертикальный ЦИЛИНДР вместо отказа (см. докстринг модуля).
        # Круглость проверяется на НЕупрощённой оболочке контура (не на synthetic-точках
        # окружности, которые сгенерируем ниже, — они по построению круглые и ничего бы не
        # проверяли).
        hull_pixel = _contour_hull_pixel(top_sil.mask)
        if hull_pixel is None or _footprint_roundness_ratio(hull_pixel) < roundness_reject_ratio:
            return None  # ни многоугольник, ни окружность — не наша форма
        pixel_corners = _circle_footprint_pixel(hull_pixel, _CIRCLE_SAMPLE_POINTS)
        is_circle = True

    side_camera_distances = {angle: camera_distances[angle] for angle in side_angles}
    side_silhouettes = {
        angle: camera_silhouette.build_silhouette(
            mesh, angle, resolution_px, fov_deg, side_camera_distances[angle], "center"
        )
        for angle in side_angles
    }

    cx = cy = resolution_px / 2.0
    f_px = (resolution_px / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    top_meta = (cx, cy, f_px, r_theta)

    # Диапазон поиска по масштабу ОБЪЕКТА, а не рига (та же находка, что и в
    # check_simple_camera_dims_v2, см. docs/camera_dims_v2_investigation.md): `h_max_ratio *
    # camera_distance` (~1200мм при camera_distance=1500) с `coarse_steps` шагами давал шаг сетки
    # ~50мм — на объекте высотой 40-100мм это грубее самого объекта, окно согласованности вокруг
    # истинной высоты проваливалось между шагами, и `_fit_height` сходился к произвольной ЗАНИЖЕННОЙ
    # высоте вместо истинной (найдено пользователем на маленькой коробке — "верх призмы заметно
    # ниже реального"). Оценка — наибольший наивный вертикальный охват среди боковых силуэтов
    # (столбец кадра, делённый на |Z-компоненту r_theta| этого ракурса, т.к. не все боковые
    # ракурсы строго боковые — 45°/135° дают столбец как смесь Y/Z), с запасом `h_max_margin`;
    # `h_max_ratio * camera_distance` остаётся верхним пределом безопасности (min с ним).
    naive_height_bound = 0.0
    for angle, sil in side_silhouettes.items():
        cols = np.flatnonzero(sil.mask.any(axis=0))
        if cols.size == 0:
            continue
        _, r_theta_side = view_axes(angle)
        z_component = abs(r_theta_side[2])
        if z_component < 0.05:
            continue  # почти горизонтальный ракурс — столбец кадра почти не несёт высоты
        span_world = (cols[-1] - cols[0]) / sil.px_per_unit
        naive_height_bound = max(naive_height_bound, span_world / z_component)
    if naive_height_bound <= 0.0:
        return None
    h_max = min(h_max_ratio * top_camera_distance, naive_height_bound * h_max_margin)

    height, excess, uncovered = _fit_height(
        pixel_corners, top_meta, top_camera_distance, side_silhouettes, side_camera_distances,
        fov_deg, resolution_px, h_max, coarse_steps, refine_iterations,
    )
    if height <= 0.0 or height >= h_max * 0.999:
        return None  # не сошлось / упёрлось в границу поиска — результату нельзя доверять

    total_observed_area = sum(int(sil.mask.sum()) for sil in side_silhouettes.values())
    if total_observed_area <= 0:
        return None

    # Пороги на ФИКСИРОВАННОЙ доле площади не масштабируются с тем, сколько пикселей объект
    # реально занимает в кадре — растровый шум границы (антиалиасинг/округление до целого
    # пикселя) даёт примерно ПОСТОЯННОЕ число шумных пикселей независимо от разрешения объекта,
    # а его ДОЛЯ от площади растёт, когда объект мельче в кадре (дальше камера/меньше сам объект
    # — линейный размер ~ sqrt(площадь), периметр растёт как sqrt(площадь), площадь как сама
    # площадь). Найдено на Пуфике (`docs/camera_dims_v2_investigation.md`): при camera_distance
    # 1500мм excess=1.92% (проходит порог 2%), при 2000мм — та же геометрия, но объект вдвое
    # мельче в кадре (total_observed_area 103k -> 56k px) — excess=2.27%, ложный отказ. Порог
    # ослабляется на `excess_pixel_margin`/`sqrt(area на один боковой ракурс)` — не более, чем
    # нужно на несколько пикселей шума на ракурс, поэтому кардинально неправильные кандидаты
    # (конус, купол) он не спасает (см. test_cone_is_rejected_..., там дыра на порядок больше
    # пиксельного шума).
    pixel_scale = np.sqrt(total_observed_area / len(side_silhouettes))
    effective_max_relative_excess = max(max_relative_excess, excess_pixel_margin / pixel_scale)
    if excess / total_observed_area > effective_max_relative_excess:
        return None  # даже на лучшей высоте кандидат вылезает за боковые силуэты

    covered = total_observed_area - uncovered
    effective_min_relative_coverage = min(min_relative_coverage, 1.0 - excess_pixel_margin / pixel_scale)
    if covered / total_observed_area < effective_min_relative_coverage:
        return None  # кандидат умещается в силуэты, но объясняет лишь малую их часть (конус
        # апексом к камере — см. докстринг _excess_and_uncovered_pixels/fit_vertical_prism)

    footprint = _footprint_at_height(pixel_corners, top_meta, top_camera_distance, height)
    return PrismFitResult(
        footprint_xy=[(float(p[0]), float(p[1])) for p in footprint],
        height=float(height),
        corners=len(pixel_corners),
        is_circle=is_circle,
        top_angle_deg=top_angle_deg,
    )
