"""Геометрическая маскировка кадра по проекции меток ленты/зеркала — перед сегментацией SAM3
закрашивает всё, что НЕ входит ни в область ленты (маркеры бортов, см.
`assets/belt_board/generate_sheet.py`), ни в область зеркала (маркеры краёв зеркала, см.
`assets/mirror_board/generate_sheet.py`), спроецированные в кадр по уже известной позе камеры.

Зачем: реальные фото рига содержат фоновый "бутылко-подобный" мусор (см. заметку задачи —
отражённые в зеркале посторонние бутылки/канистры на заднем плане), который конкурирует с
целевым объектом за текстовый промпт SAM3 ("bottle"/"object"/...) и часто выигрывает (непрозрачный,
контрастный — в отличие от целевого прозрачного объекта). Протокол
`segmentation/sam3_subprocess_backend.py` -> `UavVisionLab/tools/segment_cli.py` теперь отдаёт ВСЕ
найденные детекции на кадр (см. `SegmentationResult.all_masks`), но всё равно нужен выбор ОДНОЙ,
однозначно привязанной к ленте/зеркалу маски для 3D-реконструкции (см. `classify_mask_region`) —
закрашивание нерелевантной области ДО отправки в SAM3 остаётся лучшим способом: модель физически
не видит фон, поэтому "самая уверенная" детекция гарантированно относится к объекту на ленте и
(если не исключено отдельно) его отражению в зеркале, а не к фоновому мусору.

Границы регионов — по УЖЕ ИЗВЕСТНЫМ мировым координатам меток (в слитой раскладке `MarkerLayout`,
после калибровки/регистрации рига, см. `pose/board_registration.py`), спроецированным в конкретный
кадр по позе `CameraPose`, посчитанной для этого фото. Область отражения — это НЕ граница самого
зеркала (внутри неё может быть что угодно, если зеркало направлено не идеально на ленту — то же
фоновое, что и проблема без фильтра), а проекция ЗЕРКАЛЬНОГО ОТРАЖЕНИЯ КОНКРЕТНО меток ленты
относительно плоскости зеркала (плоскость — МНК-приближение по меткам краёв зеркала): так область
отражения указывает именно туда, где должен быть виден объект, стоящий на ленте, а не любая точка
внутри рамки зеркала."""

from __future__ import annotations

import cv2
import numpy as np

from ..pose.camera_pose import CameraPose
from ..pose.marker_layout import MarkerLayout

# ID меток по ролям (см. assets/mirror_board/, assets/belt_board/generate_sheet.py::MARKERS) —
# фиксированная конвенция проекта, не читается из layout (там метки уже единым плоским словарём).
BELT_MARKER_IDS: tuple[int, ...] = (26, 27, 28, 29)
MIRROR_MARKER_IDS: tuple[int, ...] = (22, 23, 24, 25)

# Экструзия меток ленты вдоль нормали ленты ПЕРЕД отражением в зеркале (см. belt_and_mirror_
# regions) — объект имеет реальную высоту над лентой, а метки ленты лежат плашмя (Z=0 её
# собственной плоскости), поэтому отражение ТОЛЬКО плоских меток даёт полигон, ограничивающий
# отражение ПОЛА ленты (тонкую полосу у нижнего края зеркала), а не отражение самого объекта —
# оно физически видно ВЫШЕ в зеркале, пропорционально своей высоте. Диагностировано на реальных
# фото (phone1/phone2, см. заметку задачи "3D-реконструкция объекта по реальным фото..."):
# посчитанный без экструзии регион — полоса высотой ~25px из 1280px кадра, отражение объекта
# (~100мм) фактически попадало ЗА её пределы и стиралось (закрашивалось белым) до подачи в SAM3,
# что и объясняло почти всегда пустую/edge-case маску отражения. 300мм — с запасом на объекты
# этого стенда (см. roboson_tools/assets/stl, самый высокий — единицы сотен мм).
_MAX_OBJECT_HEIGHT_MM = 300.0


def _project_points(points_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(pose.rotation)
    intr = pose.intrinsics
    dist = intr.dist_coeffs if intr.dist_coeffs is not None else np.zeros(5)
    projected, _ = cv2.projectPoints(
        points_world, rvec, pose.translation.reshape(3, 1), intr.matrix, dist
    )
    return projected.reshape(-1, 2)


def _marker_points(layout: MarkerLayout, marker_ids: tuple[int, ...]) -> np.ndarray | None:
    points = [layout.corners_world_mm[i] for i in marker_ids if i in layout.corners_world_mm]
    if not points:
        return None
    return np.concatenate(points, axis=0)


def _hull_polygon_px(points_world: np.ndarray, pose: CameraPose) -> np.ndarray:
    projected = _project_points(points_world, pose).astype(np.float32)
    return cv2.convexHull(projected).reshape(-1, 2)


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """МНК-плоскость через `points` (мировые координаты меток зеркала — они физически наклеены
    на его плоскую поверхность). Возвращает (точка_на_плоскости, единичная_нормаль)."""
    centroid = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centroid)
    normal = vt[-1]  # направление наименьшей дисперсии среди центрированных точек — нормаль
    return centroid, normal / np.linalg.norm(normal)


def _reflect_points(points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Зеркально отражает мировые точки `points` относительно плоскости (plane_point, plane_normal)."""
    signed_distance = (points - plane_point) @ plane_normal
    return points - 2.0 * signed_distance[:, None] * plane_normal


def fit_mirror_plane(layout: MarkerLayout) -> tuple[np.ndarray, np.ndarray] | None:
    """МНК-плоскость зеркала (точка, единичная нормаль) по его меткам в мировых координатах —
    экспортируется отдельно от `belt_and_mirror_regions`, чтобы 3D-реконструкция
    (`visual_hull/pose_carving.py::Observation.mirror_plane`) могла построить "виртуальную"
    зеркально-отражённую камеру для наблюдений, полученных через отражение (см. заметку задачи —
    отражение ленты в зеркале обязано ПОВЫШАТЬ точность реконструкции лишним ракурсом, а не
    просто использоваться для отсева фона). `None` — меток зеркала нет в `layout`."""
    mirror_points = _marker_points(layout, MIRROR_MARKER_IDS)
    if mirror_points is None:
        return None
    return _fit_plane(mirror_points)


def _fit_belt_plane_oriented(
    layout: MarkerLayout, pose: CameraPose
) -> tuple[np.ndarray, np.ndarray] | None:
    """МНК-плоскость ленты (точка, единичная нормаль), нормаль ориентирована "вверх" — в сторону
    камеры, а не куда попало (SVD даёт нормаль с точностью до знака). `None` — меток ленты нет в
    `layout`."""
    belt_points = _marker_points(layout, BELT_MARKER_IDS)
    if belt_points is None:
        return None
    point, normal = _fit_plane(belt_points)
    if np.dot(normal, pose.camera_pos_world - point) < 0:
        normal = -normal
    return point, normal


def fit_belt_up_normal(layout: MarkerLayout, pose: CameraPose) -> np.ndarray | None:
    """Единичная нормаль плоскости ленты, ориентированная "вверх" (см. `_fit_belt_plane_oriented`
    — та же ориентация, что использует `belt_and_mirror_regions` для экструзии региона
    отражения). Наружу — как ось "вид сверху" для отображения 3D-реконструкции по фото на Panel 3
    (`visual_hull/pose_carving.top_view_outline`, `gui/dialogs/photo_capture_dialog.py::
    _on_reconstruct_clicked`): у мировых координат рига, в отличие от кольцевой модели, нет
    заранее известной продольной оси X. `None` — меток ленты нет в `layout`."""
    result = _fit_belt_plane_oriented(layout, pose)
    return None if result is None else result[1]


def belt_and_mirror_regions(
    layout: MarkerLayout, pose: CameraPose, max_object_height_mm: float = _MAX_OBJECT_HEIGHT_MM
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(область_ленты_px, область_отражения_ленты_в_зеркале_px) — выпуклые оболочки в пиксельных
    координатах кадра (порядок обхода готов для `cv2.fillConvexPoly`). Первая — прямая проекция
    меток бортов ленты. Вторая — проекция ОТРАЖЕНИЯ меток ленты, ЭКСТРУДИРОВАННЫХ вдоль нормали
    ленты на `max_object_height_mm` (см. `_MAX_OBJECT_HEIGHT_MM` про то, почему без экструзии
    регион ловит только отражение ПОЛА ленты, а не объекта на ней), относительно плоскости
    зеркала (МНК-плоскость по меткам его краёв) — а не граница самого зеркала, см. докстринг
    модуля про разницу. `None` на любой из двух позиций — не хватает меток соответствующей доски
    в `layout` (например ещё не зарегистрирована, см. `board_registration.register_boards`);
    отражение дополнительно требует ОБЕИХ групп меток (ленты и зеркала) сразу."""
    belt_points = _marker_points(layout, BELT_MARKER_IDS)
    mirror_points = _marker_points(layout, MIRROR_MARKER_IDS)

    belt_polygon = _hull_polygon_px(belt_points, pose) if belt_points is not None else None

    mirror_reflection_polygon = None
    if belt_points is not None and mirror_points is not None:
        mirror_plane_point, mirror_normal = _fit_plane(mirror_points)
        _belt_plane_point, belt_normal = _fit_belt_plane_oriented(layout, pose)
        extruded_belt_points = np.concatenate(
            [belt_points, belt_points + belt_normal * max_object_height_mm], axis=0
        )
        reflected_points = _reflect_points(extruded_belt_points, mirror_plane_point, mirror_normal)
        mirror_reflection_polygon = _hull_polygon_px(reflected_points, pose)

    return belt_polygon, mirror_reflection_polygon


def _expand_polygon(polygon: np.ndarray, margin_px: float) -> np.ndarray:
    """Расширяет выпуклый многоугольник наружу от центроида на `margin_px` — грубый, но простой
    способ дать запас на неточность позы/обмера, не требующий вычисления нормалей рёбер."""
    center = polygon.mean(axis=0)
    directions = polygon - center
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return polygon + directions / lengths * margin_px


def classify_mask_region(
    mask: np.ndarray, belt_region: np.ndarray | None, mirror_region: np.ndarray | None
) -> str | None:
    """"belt"/"mirror" по центроиду маски против регионов из `belt_and_mirror_regions` — какой
    физический объект реально сегментирован: на ленте или его отражение в зеркале (см.
    `SegmentationResult.region` — 3D-реконструкция должна использовать только "belt"). `None` —
    маска пуста, регионы неизвестны (поза недоступна), либо центроид вне обеих областей."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    centroid = (float(xs.mean()), float(ys.mean()))
    if belt_region is not None and cv2.pointPolygonTest(belt_region.astype(np.float32), centroid, False) >= 0:
        return "belt"
    if mirror_region is not None and cv2.pointPolygonTest(mirror_region.astype(np.float32), centroid, False) >= 0:
        return "mirror"
    return None


def mask_outside_regions(
    image_bgr: np.ndarray,
    regions: tuple[np.ndarray | None, ...],
    margin_px: float = 0.0,
    fill_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Возвращает КОПИЮ `image_bgr`, где всё вне объединения `regions` (полигоны из
    `belt_and_mirror_regions`) закрашено `fill_color`. `margin_px` — запас (расширение каждого
    полигона от центроида, см. `_expand_polygon`) на неточность позы. Если все `regions` — None
    (метки не найдены/не зарегистрированы на этом фото) — возвращает `image_bgr` БЕЗ ИЗМЕНЕНИЙ:
    нечем ограничивать область, лучше дать SAM3 работать по всему кадру, чем закрасить всё."""
    valid = [r for r in regions if r is not None]
    if not valid:
        return image_bgr

    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    for polygon in valid:
        expanded = _expand_polygon(polygon, margin_px) if margin_px > 0 else polygon
        cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 255)

    result = np.full_like(image_bgr, fill_color)
    result[mask > 0] = image_bgr[mask > 0]
    return result


def crop_to_regions(
    image_bgr: np.ndarray,
    regions: tuple[np.ndarray | None, ...],
    margin_px: float = 30.0,
    fill_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Как `mask_outside_regions` (закрашивает фон вне `regions`), но ДОПОЛНИТЕЛЬНО физически
    КАДРИРУЕТ результат по bbox объединения `regions` + `margin_px`, вместо того чтобы оставлять
    полноразмерный кадр с закрашенным фоном.

    Зачем: SAM3 (`segment_cli.py`) масштабирует ВЕСЬ входной кадр под `imgsz` перед инференсом —
    маленький объект (особенно отражение в зеркале: оно физически дальше и меньше в кадре, чем
    прямой вид, см. заметку задачи) в НЕкадрированном кадре 1280×960 после такого масштабирования
    теряет большую часть своего исходного разрешения, даже если фон вокруг уже закрашен белым.
    Подтверждено на реальных фото (phone1/phone2): на кадрах, где `mask_outside_regions` давал
    "0 детекций" отражения, кадрирование до `crop_to_regions` восстанавливало детекцию
    (confidence 0.33-0.49) — тот же объект, та же поза, разница только в том, что SAM3 реально
    видел на входе.

    Возвращает `(кадрированное_изображение, (x0, y0))` — `(x0, y0)` (левый верхний угол кадра в
    координатах ИСХОДНОГО изображения) обязателен вызывающей стороне, чтобы вернуть маску
    сегментации обратно в полнокадровые пиксельные координаты (`uncrop_mask`), в которых
    откалиброваны `CameraIntrinsics.cx`/`cy` и работает вся остальная геометрия (`pose_carving`,
    `classify_mask_region`, отрисовка превью). `None` — все `regions` пустые, кадрировать нечем
    (тот же случай, что и `mask_outside_regions`)."""
    valid = [r for r in regions if r is not None]
    if not valid:
        return None

    all_points = np.concatenate(valid, axis=0)
    h, w = image_bgr.shape[:2]
    x0 = max(0, int(np.floor(all_points[:, 0].min() - margin_px)))
    y0 = max(0, int(np.floor(all_points[:, 1].min() - margin_px)))
    x1 = min(w, int(np.ceil(all_points[:, 0].max() + margin_px)))
    y1 = min(h, int(np.ceil(all_points[:, 1].max() + margin_px)))
    cropped = image_bgr[y0:y1, x0:x1]

    offset = np.array([x0, y0], dtype=np.float64)
    shifted_regions = tuple(None if r is None else r - offset for r in regions)
    masked = mask_outside_regions(cropped, shifted_regions, margin_px=margin_px, fill_color=fill_color)
    return masked, (x0, y0)


def select_region_candidates(
    masks_fullframe: list[np.ndarray],
    region: np.ndarray,
    min_inside_fraction: float = 0.5,
    max_region_area_fraction: float = 0.4,
) -> list[np.ndarray]:
    """Геометрический отбор кандидатов из всех детекций SAM3 (`SegmentationResult.all_masks`,
    уже в полнокадровых координатах) для области `region` (полигон из `belt_and_mirror_regions`).
    Возвращает подсписок масок в ИСХОДНОМ порядке (по убыванию уверенности SAM3), прошедших два
    фильтра:

    - `min_inside_fraction` — доля площади маски внутри полигона региона: отсекает детекции
      соседних объектов, чей центроид случайно рядом (прежняя классификация по одному центроиду
      этого не гарантировала).
    - `max_region_area_fraction` — площадь маски относительно площади региона: отсекает "фон
      целиком" (картон ленты, лист с маркерами — SAM3 с промптом-существительным уверенно
      детектит их как объект, найдено на реальных фото phone1: маска площадью 130-210 тыс. px
      на объекте ~15 тыс. px, при этом полностью внутри региона).

    Первый элемент результата — рекомендуемая маска (самая уверенная из геометрически
    валидных); ХВОСТ списка нужен консенсусной реконструкции (`pose_carving.carve_consensus`):
    если самая уверенная детекция противоречит остальным ракурсам (типичный кейс — отражённый
    в зеркале фоновый предмет, неотличимый от целевого объекта в ОДНОМ кадре), она заменяется
    следующим кандидатом, а не выбрасывает всё наблюдение."""
    reg_mask = np.zeros(masks_fullframe[0].shape if masks_fullframe else (0, 0), dtype=np.uint8)
    if not masks_fullframe:
        return []
    cv2.fillConvexPoly(reg_mask, np.round(region).astype(np.int32), 255)
    reg_bool = reg_mask > 0
    reg_area = int(reg_bool.sum())
    if reg_area == 0:
        return []
    out = []
    for mask in masks_fullframe:
        area = int(mask.sum())
        if area == 0:
            continue
        inside = (mask & reg_bool).sum() / area
        if inside < min_inside_fraction:
            continue
        if area > max_region_area_fraction * reg_area:
            continue
        out.append(mask)
    return out


def uncrop_mask(
    mask: np.ndarray, offset_xy: tuple[int, int], full_shape: tuple[int, int]
) -> np.ndarray:
    """Обратная операция к `crop_to_regions` — вставляет маску (H_crop, W_crop), посчитанную SAM3
    на кадрированном изображении, обратно в маску полного размера `full_shape` (H, W) по смещению
    `offset_xy` = (x0, y0) из `crop_to_regions`. Без этого шага маска осталась бы в координатах
    кадрированного изображения, несовместимых с `CameraIntrinsics`/`CameraPose`, откалиброванными
    для ПОЛНОГО кадра (см. `pose_carving`, `classify_mask_region`)."""
    x0, y0 = offset_xy
    full = np.zeros(full_shape, dtype=bool)
    crop_h, crop_w = mask.shape
    full[y0 : y0 + crop_h, x0 : x0 + crop_w] = mask
    return full
