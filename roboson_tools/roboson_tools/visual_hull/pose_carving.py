"""3D-реконструкция по реальным фото — Visual Hull из набора наблюдений (маска силуэта + поза
камеры, опционально — через отражение в плоском зеркале), без привязки к кольцевой модели камеры
(`silhouette/camera.py`, `polytope_hull.py`/`exact_polyhedral_hull.py` в azimuth+distance
параметризации). Использует ту же геометрию конуса обзора (`_cone_geometry.py`), что и кольцевая
модель, но camera frame берётся из `pose.camera_pose.CameraPose` (apex/forward/right/down) и
`CameraIntrinsics` (fx/fy/cx/cy) вместо azimuth+`PRINCIPAL_AXIS`. См. заметку задачи
"3D-реконструкция объекта по реальным фото — от маски+позы к visual hull".

Регрессионная гарантия (`tests/test_pose_carving.py`): `CameraPose`, эквивалентная кольцевой
камере (тот же apex/forward/right/up, тот же fx=fy=f_px), даёт побитово тот же halfspace/cone,
что и `polytope_hull._view_halfspaces`/`exact_polyhedral_hull._view_cone_mesh` — генерализация
не меняет кольцевую модель, лишь снимает жёсткую привязку к ней.

В отличие от кольцевой модели `belt_offset`/`BeltPosition` (эмуляция позиции на конвейере для
синтетического меша) здесь не нужны вовсе — поза уже даёт реальное положение камеры в момент
съёмки.

**Наблюдения через отражение в зеркале** (`Observation.mirror_plane`): риг «камера + зеркало»
даёт дополнительный ракурс объекта БЕСПЛАТНО — намеренная цель зеркала (см. заметку задачи
"3D-реконструкция..." — обсуждение с пользователем: отражение обязательно нужно использовать в
реконструкции, а не выбрасывать, оно повышает точность габаритов лишним ракурсом). Мысленный
эксперимент: плоское зеркало отражает падающий на него луч так, что для наблюдателя (реальной
камеры) сцена ЗА зеркалом выглядит НЕОТЛИЧИМО от того, как если бы там стояла ВТОРАЯ, "виртуальная"
камера — в точке, зеркально отражённой относительно плоскости зеркала, с зеркально отражённым
направлением взгляда. Эта виртуальная камера имеет ЛЕВУЮ (не правую) тройку базисных векторов
(`right × down = -forward`, а не `+forward`, т.к. отражение — несобственное преобразование) —
но формулы `_cone_geometry.py` (направление луча = `forward + dx*right + dy*down`, знак нормали
полупространства определяется эвристикой "к центроиду", а не жёсткой ориентацией контура) не
требуют правой тройки и остаются корректны и для отражённого (левого) базиса — см. вывод формулы
в докстринге `reflect_camera_frame` и `tests/test_pose_carving.py` (синтетическая проверка: маска,
отрендеренная КАК ВИДНО ИЗ ВИРТУАЛЬНОЙ КАМЕРЫ, при обработке через реальную позу+плоскость
зеркала восстанавливает ту же геометрию, что и прямое наблюдение)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from scipy.optimize import linprog

from ..pose.camera_pose import CameraIntrinsics, CameraPose
from ._cone_geometry import cone_mesh_from_mask, halfspaces_from_mask
from .exact_polyhedral_hull import ExactHull
from .polytope_hull import PolytopeHull, _find_interior_point
from .voxel_carving import VoxelHull

MirrorPlane = tuple[np.ndarray, np.ndarray]  # (точка на плоскости, единичная нормаль), мировые мм


@dataclass
class Observation:
    """Один кадр реконструкции. `mirror_plane=None` — прямое наблюдение (`pose` — реальная поза
    камеры для этого кадра). `mirror_plane=(point, normal)` — `mask` снята ЧЕРЕЗ ОТРАЖЕНИЕ в
    плоском зеркале (`pose` — по-прежнему РЕАЛЬНАЯ поза камеры, не виртуальной; виртуальная
    камера строится внутри `carve_polytope`/`carve_exact` через `reflect_camera_frame`).

    `margin_px` — допуск (дилатация маски на N пикселей КОНТУРА, см. `_dilate_mask`) ПЕРЕД
    построением полупространств/конуса обзора. Зачем: `carve_polytope`/`carve_exact` берут
    ТОЧНОЕ пересечение без допуска — один чуть смещённый конус обзора (например, из-за
    накопленной ошибки позы+плоскости зеркала у отражённых наблюдений, см. заметку задачи
    "Использовать отражение в зеркале как полноценный ракурс реконструкции", диагностика
    2026-07-19 через `diagnose_photo2.py`/`sweep_mirror_margin.py`) режет реальный объём для
    ВСЕХ наблюдений сразу, а не только портит точность своего кадра. Пиксельный (не мм) допуск —
    физически верная модель: угловая (не метрическая) ошибка позы/зеркала даёт бОльшую
    позиционную ошибку на объекте при бОльшем расстоянии до апекса (объясняет, почему у
    зеркальных, более удалённых виртуальных камер эффект заметнее), но при обратной проекции на
    контур это ВСЕГДА постоянный пиксельный сдвиг (см. `_cone_geometry.py`: `dx=(col-cx)/fx`) —
    константный пиксельный допуск автоматически масштабируется в нужный мм-допуск на любой
    дистанции. Эмпирически подобранное значение (phone2, реальные фото) — 10px для отражённых
    наблюдений, 0 для прямых (см. `photo_capture_dialog.py::_build_observations`)."""

    mask: np.ndarray
    pose: CameraPose
    mirror_plane: MirrorPlane | None = None
    margin_px: float = 0.0


def _dilate_mask(mask: np.ndarray, margin_px: float) -> np.ndarray:
    if margin_px <= 0:
        return mask
    k = int(round(margin_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _undistort_mask(mask: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """Компенсирует дисторсию объектива в булевой маске силуэта ДО построения конуса/
    полупространств обзора. `_cone_geometry.py` (`halfspaces_from_mask`/`cone_mesh_from_mask`)
    считает идеальную pinhole-модель (`dx=(col-cx)/fx`), но контур маски получен из СЫРОГО
    (искажённого) фото — `intr.dist_coeffs`, уже посчитанный калибровкой
    (`pose/calibration.py::calibrate_camera`), раньше использовался только для оценки позы
    камеры (`pose/aruco_estimator.py`), не для самого силуэта объекта. Без коррекции положение
    точек контура систематически смещено (растёт с расстоянием от `cx`/`cy`) — на реальных
    фото `phone2` замерено ~20px у нижнего края кадра при `k1~-0.6`, при почти нулевом сдвиге
    у центра. `dist_coeffs=None`/все нули (кольцевая модель, синтетика) — маска не меняется.

    `cv2.undistort` на булевом (0/255) изображении с билинейной интерполяцией по умолчанию
    даёт полутона на границе — порог `>127` возвращает её к бинарному виду, огрубление на
    ~1px не существеннее уже принятого `contour_epsilon_px`."""
    if intr.dist_coeffs is None or not np.any(intr.dist_coeffs):
        return mask
    K = intr.matrix
    undistorted = cv2.undistort(mask.astype(np.uint8) * 255, K, intr.dist_coeffs, None, K)
    return undistorted > 127


def reflect_camera_frame(
    pose: CameraPose, mirror_plane: MirrorPlane
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Строит camera frame "виртуальной" зеркально-отражённой камеры — (apex, forward, right,
    down) — для наблюдения объекта ЧЕРЕЗ ПЛОСКОЕ ЗЕРКАЛО. `pose` — реальная (не виртуальная) поза
    камеры, `mirror_plane` — (точка, единичная нормаль) плоскости зеркала в тех же мировых мм.

    Вывод: луч из реальной камеры через пиксель (u,v) имеет направление
    `d = forward + dx*right + dy*down` (та же формула, что в `_cone_geometry.py`). После ОДНОГО
    отражения от зеркала этот луч продолжается в направлении `d' = d - 2*(d @ n)*n` (стандартная
    формула отражения направления относительно плоскости с нормалью `n`) из точки на зеркале —
    а поскольку отражение линейно, `d' = reflect(forward) + dx*reflect(right) + dy*reflect(down)`.
    Значит камера "апекс=реальная позиция камеры, отражённая через зеркало, оси=реальные оси
    камеры, ТОЖЕ отражённые" даёт луч из своего апекса в том же направлении `d'` для той же пары
    (dx,dy) — т.е. видит СЦЕНУ ЗА ЗЕРКАЛОМ (реальный объект) ровно так же, как реальная камера
    видит его отражение. Отражение всех 3 базисных векторов делает тройку (right,down,forward)
    ЛЕВОЙ (`right × down = -forward`) — `_cone_geometry.halfspaces_from_mask`/`cone_mesh_from_mask`
    от этого не ломаются (см. докстринг модуля и `tests/test_pose_carving.py`)."""
    plane_point, plane_normal = mirror_plane
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    def reflect_point(p: np.ndarray) -> np.ndarray:
        return p - 2.0 * ((p - plane_point) @ plane_normal) * plane_normal

    def reflect_dir(v: np.ndarray) -> np.ndarray:
        return v - 2.0 * (v @ plane_normal) * plane_normal

    apex = reflect_point(pose.camera_pos_world)
    forward = reflect_dir(pose.forward_world)
    right = reflect_dir(pose.right_world)
    down = reflect_dir(pose.down_world)
    return apex, forward, right, down


def _camera_frame_for_observation(
    obs: Observation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if obs.mirror_plane is None:
        return (
            obs.pose.camera_pos_world,
            obs.pose.forward_world,
            obs.pose.right_world,
            obs.pose.down_world,
        )
    return reflect_camera_frame(obs.pose, obs.mirror_plane)


def _halfspaces_from_observation(
    obs: Observation, contour_epsilon_px: float = 0.0
) -> tuple[np.ndarray, np.ndarray] | None:
    apex, forward, right, down = _camera_frame_for_observation(obs)
    intr = obs.pose.intrinsics
    mask = _dilate_mask(_undistort_mask(obs.mask, intr), obs.margin_px)
    return halfspaces_from_mask(
        mask, apex, forward, right, down, intr.fx, intr.fy, intr.cx, intr.cy,
        contour_epsilon_px=contour_epsilon_px,
    )


def carve_polytope(
    observations: list[Observation], contour_epsilon_px: float = 0.0
) -> PolytopeHull | None:
    """Точное пересечение конусов обзора (см. `polytope_hull.carve`) по произвольным позам
    камеры (в т.ч. через отражение в зеркале, см. докстринг модуля) — для ВЫПУКЛЫХ объектов
    (берёт `cv2.convexHull` контура каждой маски). `contour_epsilon_px` — тот же допуск
    упрощения контура (`cv2.approxPolyDP`), что и в `polytope_hull.carve`/
    `_cone_geometry.halfspaces_from_mask`. ДЕФОЛТ 0.0 (упрощение отключено) — обратная
    совместимость/побитовое совпадение с ring-моделью (см. `tests/test_pose_carving.py`);
    поднять до ~1.5px для ускорения на плотных контурах (тот же компромисс, что у
    `polytope_hull.carve`). None — вырожденный случай (пустая маска хотя бы на одном фото,
    недостаточно различных направлений взгляда, пустое/неограниченное пересечение
    полупространств)."""
    all_normals, all_offsets = [], []
    for obs in observations:
        result = _halfspaces_from_observation(obs, contour_epsilon_px)
        if result is None:
            return None
        normals, offsets = result
        all_normals.append(normals)
        all_offsets.append(offsets)

    A = np.concatenate(all_normals, axis=0)
    b = np.concatenate(all_offsets, axis=0)
    halfspaces = np.hstack([A, b[:, None]])

    # В отличие от кольцевой модели (там центроид синтетического mesh известен заранее) для
    # реальных фото объект a priori не известен — начинаем с мирового начала координат (по
    # обмеру рига объект снимается около него, см. заметку задачи "Заменить тестовый лист
    # ArUco..."); если это не строго внутри всех полупространств, `_find_interior_point`
    # всё равно найдёт центр Чебышева через LP, не полагаясь на догадку.
    interior_point = _find_interior_point(A, b, np.zeros(3))
    if interior_point is None:
        return None

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        vertices = hs.intersections
        if len(vertices) < 4:
            return None
        hull = ConvexHull(vertices)
    except QhullError:
        return None

    return PolytopeHull(vertices=vertices, faces=hull.simplices)


def carve_exact(
    observations: list[Observation],
    far_distance: float | None = None,
    contour_epsilon_px: float = 1.0,
) -> ExactHull | None:
    """Exact Polyhedral Visual Hull (см. `exact_polyhedral_hull.carve`) по произвольным позам
    камеры (в т.ч. через отражение в зеркале) — сохраняет вогнутости силуэта (не берёт
    `cv2.convexHull` контура). None — вырожденный случай (аналогично `carve_polytope`, либо
    ошибка триангуляции/булевого пересечения)."""
    if len(observations) < 2:
        # Один конус не ограничивает объект вдоль луча взгляда — тот же вырожденный случай, что
        # и у кольцевой модели (см. `exact_polyhedral_hull.carve`).
        return None
    if far_distance is None:
        # Аналог "самой длинной дистанции рига * 2" у кольцевой модели: здесь дистанция —
        # расстояние апекса (для зеркальных наблюдений — виртуального) от мирового начала
        # координат (объект снимается около него, см. `carve_polytope`).
        far_distance = (
            max(float(np.linalg.norm(_camera_frame_for_observation(obs)[0])) for obs in observations)
            * 2.0
        )

    cones = []
    for obs in observations:
        apex, forward, right, down = _camera_frame_for_observation(obs)
        intr = obs.pose.intrinsics
        mask = _dilate_mask(_undistort_mask(obs.mask, intr), obs.margin_px)
        cone = cone_mesh_from_mask(
            mask, apex, forward, right, down, intr.fx, intr.fy, intr.cx, intr.cy,
            far_distance, contour_epsilon_px,
        )
        if cone is None:
            return None
        cones.append(cone)

    if len(cones) == 1:
        result = cones[0]
    else:
        try:
            result = trimesh.boolean.intersection(cones, engine="manifold")
        except Exception:
            return None

    if result is None or len(result.vertices) == 0 or len(result.faces) == 0:
        return None
    return ExactHull(vertices=np.asarray(result.vertices), faces=np.asarray(result.faces))


def carve_voxels(
    observations: list[Observation],
    bbox: tuple[np.ndarray, np.ndarray],
    grid_resolution: int = 48,
    margin_ratio: float = 0.1,
) -> VoxelHull | None:
    """Voxel carving (space carving) по произвольным позам камеры — аналог
    `visual_hull/voxel_carving.py::carve`, но принимает готовые
    `Observation` (реальные маски+позы), а не рендерит силуэты САМ из
    известного `Mesh` (тот вариант — только для синтетики/визуализации, см.
    его докстринг).

    **Зачем нужен отдельно от `carve_polytope`/`carve_exact`**: те строят
    ТОЧНОЕ пересечение полупространств/конусов — добавление любого
    наблюдения может только СУЗИТЬ результат (или не изменить), никогда не
    расширить. Если входные маски систематически консервативны (уже
    отработанный случай — конвейерный риг с HSV-порогом против теней,
    заметка задачи "3D-реконструкция объекта по кропам сетки ракурсов
    (Фаза 5, CV-пайплайн Webots)" в вики robozon), то ЛЮБОЕ дополнительное
    наблюдение с той же консервативностью только режет объём дальше — не
    помогает, даже вредит. Задокументированный факт (заметка задачи
    "Проверить G4 на 3D-реконструкции с параллаксом по ленте" в этом же
    репозитории): на 18 тестовых объектах добавление start/end к center
    дало `polytope_hull` те же 16/18, что и один center, а `torchhull`
    (GPU, тоже воксельный/occupancy подход) — 18/18 против 16/18 у center.
    Огрубление на уровне вокселя действует как естественный допуск,
    параллакс (start/end) реально помогает именно здесь.

    `bbox` — (min, max) мировых мм, ОБЯЗАТЕЛЬНО передаётся вызывающей
    стороной (в отличие от `visual_hull/voxel_carving.py::carve`, который
    берёт его из `Mesh.vertices` — здесь меша нет). Разумный источник —
    геометрия ТЗ/рига (наихудший ожидаемый габарит объекта), НЕ bbox от
    `carve_polytope` на тех же наблюдениях — если полупространства уже
    занижают объём, затравка сеткой по их bbox рискует изначально не
    охватить истинный объект.

    Оптимизация: после каждого наблюдения тестируются только ещё
    уцелевшие точки сетки (не вся сетка заново) — на consecutive
    наблюдениях, режущих объём, стоимость быстро падает.

    `None` — вырожденный случай (объект не виден хотя бы в одном
    наблюдении — сетка опустела)."""
    lo, hi = bbox
    extent = hi - lo
    margin = float(extent.max()) * margin_ratio if extent.max() > 0 else 1.0
    lo = lo - margin
    hi = hi + margin
    axes = [np.linspace(lo[i], hi[i], grid_resolution) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    alive = np.arange(len(grid))
    for obs in observations:
        if len(alive) == 0:
            return None
        apex, forward, right, down = _camera_frame_for_observation(obs)
        intr = obs.pose.intrinsics
        mask = _dilate_mask(_undistort_mask(obs.mask, intr), obs.margin_px)
        pts = grid[alive]
        rel = pts - apex[None, :]
        depth = rel @ forward
        col = intr.fx * (rel @ right) / depth + intr.cx
        row = intr.fy * (rel @ down) / depth + intr.cy
        rows_i = np.round(row).astype(np.int64)
        cols_i = np.round(col).astype(np.int64)
        h, w = mask.shape
        in_bounds = (depth > 1e-6) & (rows_i >= 0) & (rows_i < h) & (cols_i >= 0) & (cols_i < w)
        inside = np.zeros(len(pts), dtype=bool)
        b_idx = np.flatnonzero(in_bounds)
        inside[b_idx] = mask[rows_i[b_idx], cols_i[b_idx]]
        alive = alive[inside]

    if len(alive) == 0:
        return None
    voxel_size = float(np.min((hi - lo) / max(grid_resolution - 1, 1)))
    return VoxelHull(points=grid[alive], voxel_size=voxel_size)


@dataclass
class CandidateObservation:
    """Наблюдение для консенсусной реконструкции (`carve_consensus`): вместо ОДНОЙ маски —
    список кандидатов (все детекции SAM3 на этой области кадра, прошедшие геометрический фильтр
    `region_filter.select_region_candidates`, по убыванию уверенности). Первый кандидат —
    рекомендуемый; остальные пробуются, только если он противоречит другим ракурсам."""

    candidates: list[np.ndarray]  # маски-кандидаты в полнокадровых координатах, len >= 1
    pose: CameraPose
    mirror_plane: MirrorPlane | None = None
    label: str = ""  # для отчёта пользователю, например "photo_3.jpg/отражение"
    margin_px: float = 0.0  # допуск на всех кандидатов этого наблюдения, см. Observation.margin_px


@dataclass
class ConsensusResult:
    polytope: PolytopeHull | None
    exact: ExactHull | None
    # Индекс выбранного кандидата на каждое наблюдение (в порядке входного списка);
    # None — наблюдение выброшено (ни один кандидат не согласуется с остальными ракурсами).
    chosen: list[int | None]
    switched_labels: list[str]  # наблюдения, где выбран НЕ первый кандидат
    dropped_labels: list[str]  # выброшенные наблюдения


_FEASIBLE_RADIUS_MM = 1.0  # порог радиуса Чебышева, ниже которого пересечение считаем пустым
_MAX_CONSENSUS_ITERS = 40  # защита от зацикливания (N наблюдений × кандидаты на каждом)


def _chebyshev_radius(A: np.ndarray, b: np.ndarray) -> float:
    """Радиус максимального шара, вписанного в пересечение полупространств `A@p + b <= 0`
    (центр Чебышева, тот же LP, что в `polytope_hull._find_interior_point`, но возвращает сам
    радиус) — быстрая (4 переменных) проверка "пересечение непусто и насколько": -inf, если LP
    неразрешим (пересечение пусто/неограничен радиус)."""
    n = A.shape[0]
    a_ub = np.hstack([A, np.ones((n, 1))])
    c = np.array([0.0, 0.0, 0.0, -1.0])  # max r == min -r
    res = linprog(c, A_ub=a_ub, b_ub=-b, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    return float(res.x[3]) if res.success else -np.inf


def carve_consensus(
    observations: list[CandidateObservation], contour_epsilon_px: float = 0.0
) -> ConsensusResult:
    """Реконструкция с выбором маски-кандидата по КРОСС-РАКУРСНОЙ согласованности.

    Мотивация (реальный кейс phone1, см. заметку задачи): в области отражения зеркала виден не
    только целевой объект, но и отражённый фоновый хлам (канистры/бутылки за ригом) — SAM3
    уверенно (0.98) детектит его по промпту, и В ОДНОМ кадре его геометрически не отличить от
    цели (внутри региона, разумный размер). Единственный различающий сигнал — согласованность с
    ДРУГИМИ ракурсами: конус обзора не-целевой детекции не пересекается с конусами остальных
    наблюдений. Причём выбрасывать всё наблюдение не нужно — правильная маска почти всегда есть
    среди менее уверенных кандидатов того же кадра.

    Алгоритм: жадный. Начинаем с первого (самого уверенного) кандидата каждого наблюдения. Пока
    пересечение всех конусов пусто (радиус Чебышева < `_FEASIBLE_RADIUS_MM`): находим
    наблюдение, чьё ИСКЛЮЧЕНИЕ даёт максимальный радиус остальным (наиболее несогласованное),
    и переключаем его на следующего кандидата; если кандидаты кончились — выбрасываем
    наблюдение совсем. Терминируется за `_MAX_CONSENSUS_ITERS` (защита) либо когда осталось < 2
    наблюдений.

    Возвращает `ConsensusResult` с обоими халлами (polytope + exact по ОДНОМУ И ТОМУ ЖЕ
    финальному набору масок) и отчётом switched/dropped для UI."""
    n = len(observations)
    chosen: list[int | None] = [0 if obs.candidates else None for obs in observations]

    def hs_for(i: int) -> tuple[np.ndarray, np.ndarray] | None:
        idx = chosen[i]
        if idx is None:
            return None
        obs = Observation(
            mask=observations[i].candidates[idx],
            pose=observations[i].pose,
            mirror_plane=observations[i].mirror_plane,
            margin_px=observations[i].margin_px,
        )
        return _halfspaces_from_observation(obs, contour_epsilon_px)

    hs_list: list[tuple[np.ndarray, np.ndarray] | None] = [hs_for(i) for i in range(n)]
    # Вырожденная маска (halfspaces None) — наблюдение сразу небоеспособно, пробуем следующих
    # кандидатов, потом выбрасываем.
    for i in range(n):
        while hs_list[i] is None and chosen[i] is not None:
            chosen[i] = chosen[i] + 1 if chosen[i] + 1 < len(observations[i].candidates) else None
            hs_list[i] = hs_for(i) if chosen[i] is not None else None

    for _ in range(_MAX_CONSENSUS_ITERS):
        alive = [i for i in range(n) if hs_list[i] is not None]
        if len(alive) < 2:
            break
        A = np.concatenate([hs_list[i][0] for i in alive], axis=0)
        b = np.concatenate([hs_list[i][1] for i in alive], axis=0)
        if _chebyshev_radius(A, b) >= _FEASIBLE_RADIUS_MM:
            break

        # Виновник — наблюдение, чьё удаление даёт максимальный радиус Чебышева остальным.
        best_i, best_r = None, -np.inf
        for j in alive:
            rest = [i for i in alive if i != j]
            A2 = np.concatenate([hs_list[i][0] for i in rest], axis=0)
            b2 = np.concatenate([hs_list[i][1] for i in rest], axis=0)
            r = _chebyshev_radius(A2, b2)
            if r > best_r:
                best_i, best_r = j, r

        if best_i is None:
            break
        i = best_i
        chosen[i] = chosen[i] + 1 if chosen[i] + 1 < len(observations[i].candidates) else None
        hs_list[i] = hs_for(i) if chosen[i] is not None else None
        while hs_list[i] is None and chosen[i] is not None:
            chosen[i] = chosen[i] + 1 if chosen[i] + 1 < len(observations[i].candidates) else None
            hs_list[i] = hs_for(i) if chosen[i] is not None else None

    final_observations = [
        Observation(
            mask=observations[i].candidates[chosen[i]],
            pose=observations[i].pose,
            mirror_plane=observations[i].mirror_plane,
            margin_px=observations[i].margin_px,
        )
        for i in range(n)
        if chosen[i] is not None
    ]
    polytope = carve_polytope(final_observations, contour_epsilon_px) if len(final_observations) >= 2 else None
    exact = carve_exact(final_observations) if len(final_observations) >= 2 else None
    switched = [observations[i].label for i in range(n) if chosen[i] not in (None, 0)]
    dropped = [observations[i].label for i in range(n) if chosen[i] is None]
    return ConsensusResult(
        polytope=polytope, exact=exact, chosen=chosen, switched_labels=switched, dropped_labels=dropped
    )


def rotation_aligning_up(up_normal: np.ndarray) -> np.ndarray:
    """(3,3) поворот, переводящий `up_normal` в мировую `+Z` — ТОЛЬКО для отображения (Panel 1,
    `gui/panels/model_panel.py::ModelPanel`, `pyqtgraph.opengl.GLViewWidget`, орбита камеры
    неявно считает "верх" сцены `+Z`). Мировая система координат ArUco-рига (референсная доска
    `belt_left`, см. `_build_observations`/`_REFERENCE_BOARD`) определяется ПРОИЗВОЛЬНО тем, как
    физически напечатана и уложена эта доска — её локальная ось Z НЕ обязана указывать вверх
    (на реальном стенде phone2 указывает вниз, антипараллельна `fit_belt_up_normal`, см. заметку
    задачи "Использовать отражение в зеркале..."), поэтому реконструкция по фото, показанная в
    Panel 1 БЕЗ поворота, выглядела перевёрнутой — не баг геометрии (габариты/пересечение
    полупространств считаются в исходных координатах и не меняются), баг ТОЛЬКО отображения.
    Стандартная формула Родрига для поворота, совмещающего два единичных вектора."""
    up = up_normal / np.linalg.norm(up_normal)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, target)
    sin_angle = np.linalg.norm(axis)
    cos_angle = float(up @ target)
    if sin_angle < 1e-9:
        return np.eye(3) if cos_angle > 0 else np.diag([1.0, -1.0, -1.0])
    axis_hat = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + axis_hat + axis_hat @ axis_hat * ((1.0 - cos_angle) / (sin_angle**2))


def top_view_outline(vertices: np.ndarray, up_normal: np.ndarray) -> list[tuple[float, float]]:
    """Контур (выпуклая оболочка) проекции `vertices` на плоскость, перпендикулярную
    `up_normal` — "вид сверху" на 3D-реконструкцию по фото для Panel 3 (`gui/panels/
    visual_hull_panel.py::set_photo_reconstruction`). В отличие от кольцевой модели, у мировых
    координат рига (ArUco-калибровка) нет заранее известной продольной оси X, вдоль которой
    строится обычное сечение (`silhouette/base.py` — конвенция `axis_min`/`row=X`) — здесь
    просто ортографическая проекция вдоль нормали ленты (см. `segmentation/region_filter.
    fit_belt_up_normal`), а не срез конкретной плоскостью."""
    normal = up_normal / np.linalg.norm(up_normal)
    e1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    projected = np.column_stack([vertices @ e1, vertices @ e2]).astype(np.float32)
    hull = cv2.convexHull(projected).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in hull]
