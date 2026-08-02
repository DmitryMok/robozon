"""`segmentation/region_filter.py` — геометрическая маскировка кадра по меткам ленты/зеркала
перед сегментацией SAM3 (см. заметку задачи про фоновый "бутылко-подобный" мусор, отражённый в
зеркале)."""

from __future__ import annotations

import cv2
import numpy as np

from roboson_tools.pose.camera_pose import CameraIntrinsics, CameraPose
from roboson_tools.pose.marker_layout import MarkerLayout
from roboson_tools.segmentation.region_filter import (
    BELT_MARKER_IDS,
    MIRROR_MARKER_IDS,
    _fit_plane,
    _reflect_points,
    belt_and_mirror_regions,
    classify_mask_region,
    crop_to_regions,
    fit_mirror_plane,
    mask_outside_regions,
    select_region_candidates,
    uncrop_mask,
)


def test_fit_plane_recovers_known_plane():
    # Плоскость x=1000 (нормаль вдоль X) — точки с произвольными y/z.
    rng = np.random.default_rng(0)
    yz = rng.uniform(-50, 50, size=(20, 2))
    points = np.column_stack([np.full(20, 1000.0), yz])

    point_on_plane, normal = _fit_plane(points)
    assert abs(point_on_plane[0] - 1000.0) < 1e-6
    # Нормаль может быть направлена в любую сторону вдоль оси X.
    assert abs(abs(normal[0]) - 1.0) < 1e-6
    assert abs(normal[1]) < 1e-6 and abs(normal[2]) < 1e-6


def test_reflect_points_matches_analytic_formula():
    plane_point = np.array([1000.0, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    points = np.array([[800.0, 10.0, 20.0], [1000.0, 5.0, 5.0], [1200.0, -3.0, 7.0]])

    reflected = _reflect_points(points, plane_point, normal)

    # Отражение относительно x=1000: x' = 2000 - x, y/z не меняются.
    expected = points.copy()
    expected[:, 0] = 2000.0 - points[:, 0]
    np.testing.assert_allclose(reflected, expected, atol=1e-9)


def _identity_pose(resolution_px=(480, 640)) -> CameraPose:
    intrinsics = CameraIntrinsics(fx=900.0, fy=900.0, cx=320.0, cy=240.0, resolution_px=resolution_px)
    return CameraPose(rotation=np.eye(3), translation=np.zeros(3), intrinsics=intrinsics)


def test_belt_and_mirror_regions_none_without_markers():
    layout = MarkerLayout(dictionary_name="DICT_4X4_50", corners_world_mm={})
    belt, mirror = belt_and_mirror_regions(layout, _identity_pose())
    assert belt is None
    assert mirror is None


def test_belt_region_present_mirror_region_requires_both_boards():
    belt_corners = {
        26: np.array([[-50.0, -50.0, 1000.0], [50.0, -50.0, 1000.0], [50.0, 50.0, 1000.0], [-50.0, 50.0, 1000.0]]),
    }
    layout = MarkerLayout(dictionary_name="DICT_4X4_50", corners_world_mm=belt_corners)
    belt, mirror = belt_and_mirror_regions(layout, _identity_pose())
    assert belt is not None
    assert mirror is None  # нет меток зеркала — отражение посчитать не из чего


def test_reflection_region_matches_manual_reflection():
    """Камера смотрит вдоль +Z (identity pose). Лента — маленький квадрат вокруг (0,0,1000).
    Зеркало — плоскость y=500 (нормаль вдоль Y), видимая теми же координатами Z, что и лента
    (упрощение для теста, не физическая раскладка). Отражение ленты относительно этой плоскости —
    квадрат вокруг (0, 1000, 1000) — должно спроецироваться в кадр в известном месте, отличном от
    прямой проекции ленты."""
    belt_corners = {
        26: np.array(
            [[-20.0, -20.0, 1000.0], [20.0, -20.0, 1000.0], [20.0, 20.0, 1000.0], [-20.0, 20.0, 1000.0]]
        ),
    }
    mirror_corners = {
        22: np.array(
            [[-100.0, 500.0, 900.0], [100.0, 500.0, 900.0], [100.0, 500.0, 1100.0], [-100.0, 500.0, 1100.0]]
        ),
        23: np.array(
            [[-100.0, 500.0, 1100.0], [100.0, 500.0, 1100.0], [100.0, 500.0, 1300.0], [-100.0, 500.0, 1300.0]]
        ),
    }
    layout = MarkerLayout(
        dictionary_name="DICT_4X4_50", corners_world_mm={**belt_corners, **mirror_corners}
    )
    pose = _identity_pose()

    belt_polygon, mirror_polygon = belt_and_mirror_regions(layout, pose)
    assert belt_polygon is not None
    assert mirror_polygon is not None

    # Ручной пересчёт: отражение точки (0, 500, 1000) (центр ленты) относительно плоскости y=500
    # — сама точка не сдвигается по Y (она уже на плоскости в этом синтетическом примере не лежит
    # ровно, лента на y~480-520, а зеркало на y=500 — отражение центра ленты (0,500,1000) даёт
    # (0,500,1000), т.к. он на плоскости). Проверим на угловой точке вместо центра.
    corner = np.array([[20.0, -20.0, 1000.0]])
    reflected_corner = _reflect_points(corner, np.array([0.0, 500.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    rvec, _ = cv2.Rodrigues(pose.rotation)
    projected_expected, _ = cv2.projectPoints(
        reflected_corner, rvec, pose.translation.reshape(3, 1), pose.intrinsics.matrix, np.zeros(5)
    )
    expected_px = projected_expected.reshape(2)

    # Эта спроецированная точка должна лежать внутри (или на границе) области отражения.
    assert cv2.pointPolygonTest(mirror_polygon.astype(np.float32), tuple(expected_px), False) >= -1.0

    # Область ленты и область отражения не должны совпадать (отражение сдвинуто по Y) — сравниваем
    # bbox, а не сами полигоны: экструзия вдоль нормали ленты (см. _MAX_OBJECT_HEIGHT_MM) может
    # дать области отражения ДРУГОЕ число вершин выпуклой оболочки, чем у плоской области ленты.
    def _bbox(polygon: np.ndarray) -> np.ndarray:
        return np.array([polygon.min(axis=0), polygon.max(axis=0)])

    assert not np.allclose(_bbox(belt_polygon), _bbox(mirror_polygon))


def test_mirror_region_covers_reflection_of_object_above_belt_not_just_belt_floor():
    """Регрессия на реальный баг (найден на фото phone1/phone2, см. заметку задачи): без
    экструзии меток ленты вдоль её нормали ПЕРЕД отражением, `mirror_reflection_polygon` ловит
    только отражение ПОЛА ленты (Z=0 её собственной плоскости) — тонкую полосу у ближнего к
    ленте края зеркала. Реальный объект стоит НАД лентой и его отражение физически видно ВЫШЕ
    в зеркале, за пределами этой полосы, поэтому `mask_outside_regions` стирало отражение
    объекта до подачи в SAM3 (наблюдаемый симптом — почти всегда пустая/крайне низкой
    уверенности маска отражения на реальных фото). Здесь — синтетическая версия той же
    геометрии: точка, изображающая объект высотой 150мм над лентой, должна проецироваться
    ВНУТРЬ области отражения; та же точка на уровне пола (высота 0) — тоже (полоса всё ещё
    покрыта, экструзия только добавляет площадь, не убирает её)."""
    belt_corners = {
        26: np.array(
            [[-20.0, -20.0, 1000.0], [20.0, -20.0, 1000.0], [20.0, 20.0, 1000.0], [-20.0, 20.0, 1000.0]]
        ),
    }
    mirror_corners = {
        22: np.array(
            [[-100.0, 500.0, 900.0], [100.0, 500.0, 900.0], [100.0, 500.0, 1100.0], [-100.0, 500.0, 1100.0]]
        ),
        23: np.array(
            [[-100.0, 500.0, 1100.0], [100.0, 500.0, 1100.0], [100.0, 500.0, 1300.0], [-100.0, 500.0, 1300.0]]
        ),
    }
    layout = MarkerLayout(
        dictionary_name="DICT_4X4_50", corners_world_mm={**belt_corners, **mirror_corners}
    )
    pose = _identity_pose()

    _belt_plane_point, belt_normal = _fit_plane(
        np.concatenate(list(belt_corners.values()), axis=0)
    )
    if np.dot(belt_normal, pose.camera_pos_world - _belt_plane_point) < 0:
        belt_normal = -belt_normal

    # "Объект" на ленте (центр, высота 150мм над лентой вдоль её нормали) — его отражение.
    object_point = np.array([[0.0, 0.0, 1000.0]]) + belt_normal * 150.0
    mirror_plane_point, mirror_normal = _fit_plane(
        np.concatenate(list(mirror_corners.values()), axis=0)
    )
    reflected_object = _reflect_points(object_point, mirror_plane_point, mirror_normal)
    rvec, _ = cv2.Rodrigues(pose.rotation)
    projected, _ = cv2.projectPoints(
        reflected_object, rvec, pose.translation.reshape(3, 1), pose.intrinsics.matrix, np.zeros(5)
    )
    object_reflection_px = tuple(projected.reshape(2))

    _belt_polygon, mirror_polygon_no_extrusion = belt_and_mirror_regions(
        layout, pose, max_object_height_mm=0.0
    )
    _belt_polygon, mirror_polygon_with_extrusion = belt_and_mirror_regions(
        layout, pose, max_object_height_mm=150.0
    )

    assert (
        cv2.pointPolygonTest(mirror_polygon_no_extrusion.astype(np.float32), object_reflection_px, False)
        < 0
    ), "без экструзии отражение объекта (150мм над лентой) должно быть ВНЕ региона — это и есть баг"
    assert (
        cv2.pointPolygonTest(mirror_polygon_with_extrusion.astype(np.float32), object_reflection_px, False)
        >= 0
    ), "с экструзией на ту же высоту отражение объекта должно попадать В регион"


def test_mask_outside_regions_blocks_background_keeps_regions():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)  # "фон" — должен исчезнуть
    region = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]], dtype=np.float32)

    masked = mask_outside_regions(image, (region, None), fill_color=(255, 255, 255))

    assert tuple(masked[50, 50]) == (10, 20, 30)  # внутри региона — сохранилось
    assert tuple(masked[5, 5]) == (255, 255, 255)  # снаружи — закрашено


def test_mask_outside_regions_noop_without_any_region():
    image = np.full((10, 10, 3), 7, dtype=np.uint8)
    result = mask_outside_regions(image, (None, None))
    np.testing.assert_array_equal(result, image)


def test_classify_mask_region_belt_vs_mirror_vs_neither():
    belt_region = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
    mirror_region = np.array([[60.0, 10.0], [90.0, 10.0], [90.0, 40.0], [60.0, 40.0]])

    mask_belt = np.zeros((100, 100), dtype=bool)
    mask_belt[15:25, 15:25] = True
    assert classify_mask_region(mask_belt, belt_region, mirror_region) == "belt"

    mask_mirror = np.zeros((100, 100), dtype=bool)
    mask_mirror[15:25, 65:75] = True
    assert classify_mask_region(mask_mirror, belt_region, mirror_region) == "mirror"

    mask_elsewhere = np.zeros((100, 100), dtype=bool)
    mask_elsewhere[80:90, 80:90] = True
    assert classify_mask_region(mask_elsewhere, belt_region, mirror_region) is None

    empty_mask = np.zeros((100, 100), dtype=bool)
    assert classify_mask_region(empty_mask, belt_region, mirror_region) is None

    assert classify_mask_region(mask_belt, None, None) is None


def test_fit_mirror_plane_matches_direct_fit_plane():
    mirror_corners = {
        22: np.array(
            [[-100.0, 500.0, 900.0], [100.0, 500.0, 900.0], [100.0, 500.0, 1100.0], [-100.0, 500.0, 1100.0]]
        ),
        23: np.array(
            [[-100.0, 500.0, 1100.0], [100.0, 500.0, 1100.0], [100.0, 500.0, 1300.0], [-100.0, 500.0, 1300.0]]
        ),
    }
    layout = MarkerLayout(dictionary_name="DICT_4X4_50", corners_world_mm=mirror_corners)
    plane = fit_mirror_plane(layout)
    assert plane is not None
    point, normal = plane
    assert abs(point[1] - 500.0) < 1e-6
    assert abs(abs(normal[1]) - 1.0) < 1e-6


def test_fit_mirror_plane_none_without_mirror_markers():
    layout = MarkerLayout(
        dictionary_name="DICT_4X4_50",
        corners_world_mm={26: np.zeros((4, 3))},
    )
    assert fit_mirror_plane(layout) is None


def test_crop_to_regions_none_without_any_region():
    image = np.full((100, 100, 3), 7, dtype=np.uint8)
    assert crop_to_regions(image, (None, None)) is None


def test_crop_to_regions_crops_tightly_and_keeps_region_masked():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)  # фон — должен исчезнуть внутри кадра, а сам кадр должен сузиться
    region = np.array([[80.0, 80.0], [140.0, 80.0], [140.0, 140.0], [80.0, 140.0]], dtype=np.float32)

    result = crop_to_regions(image, (region, None), margin_px=10.0, fill_color=(255, 255, 255))
    assert result is not None
    cropped, (x0, y0) = result

    # Кадр сузился (не остался 200x200) и сместился к региону, а не остался с (0, 0).
    assert cropped.shape[0] < 200 and cropped.shape[1] < 200
    assert x0 > 0 and y0 > 0
    # Внутри региона (в координатах кадра) — исходный фон сохранился.
    assert tuple(cropped[100 - y0, 100 - x0]) == (10, 20, 30)
    # У самого края кадрированного изображения (за пределами региона+margin) — закрашено.
    assert tuple(cropped[0, 0]) == (255, 255, 255)


def test_uncrop_mask_places_mask_back_at_full_frame_coordinates():
    crop_mask = np.zeros((20, 30), dtype=bool)
    crop_mask[5:10, 8:15] = True

    full_mask = uncrop_mask(crop_mask, offset_xy=(50, 40), full_shape=(200, 200))

    assert full_mask.shape == (200, 200)
    assert full_mask.sum() == crop_mask.sum()
    # Точка (5, 8) в кадре -> (40+5, 50+8) = (45, 58) в полном кадре.
    assert full_mask[45, 58]
    assert not full_mask[0, 0]


def test_select_region_candidates_filters_background_and_out_of_region():
    """Регрессия на реальный кейс phone1 (2026-07-19): SAM3 с промптом-существительным уверенно
    детектит и картон ленты ЦЕЛИКОМ (огромная маска, полностью внутри региона — фильтр только
    по 'inside' её не отсекает), и объекты вне региона. Валидный кандидат — маленькая маска
    внутри региона."""
    region = np.array([[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]])
    shape = (100, 100)

    target = np.zeros(shape, dtype=bool)
    target[40:55, 40:55] = True  # объект — маленький, внутри

    background = np.zeros(shape, dtype=bool)
    background[12:88, 12:88] = True  # "картон целиком" — внутри региона, но почти вся его площадь

    outside = np.zeros(shape, dtype=bool)
    outside[92:99, 92:99] = True  # чужой объект вне региона

    # Порядок = порядок уверенности SAM3: фон часто УВЕРЕННЕЕ цели.
    picked = select_region_candidates([background, outside, target], region)
    assert len(picked) == 1
    assert picked[0] is target


def test_select_region_candidates_keeps_confidence_order_of_valid_masks():
    region = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    shape = (100, 100)
    a = np.zeros(shape, dtype=bool)
    a[10:20, 10:20] = True
    b = np.zeros(shape, dtype=bool)
    b[60:75, 60:75] = True

    picked = select_region_candidates([a, b], region)
    assert len(picked) == 2
    assert picked[0] is a and picked[1] is b  # исходный порядок (по conf) сохранён


def test_marker_id_conventions_match_generator_scripts():
    """Регрессия на случай, если id в generate_sheet.py когда-нибудь поменяют, не обновив этот
    модуль — 22-25 зеркало, 26-29 лента (см. assets/mirror_board/, assets/belt_board/)."""
    assert BELT_MARKER_IDS == (26, 27, 28, 29)
    assert MIRROR_MARKER_IDS == (22, 23, 24, 25)
