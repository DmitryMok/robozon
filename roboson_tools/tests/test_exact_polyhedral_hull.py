"""Exact Polyhedral Visual Hull (visual_hull/exact_polyhedral_hull.py) — альтернатива
`polytope_hull.py` для невыпуклых объектов. См. докстринг модуля: `polytope_hull` берёт
`cv2.convexHull` контура силуэта и поэтому не может воспроизвести уступ (например, переход
"толстая зажимаемая часть -> тонкий стержень" у ручки) — реконструирует его как плавный конус.
Этот тест синтетически воспроизводит именно такую форму (ступенчатый стержень: два цилиндра
разного радиуса встык) и проверяет, что exact-метод сохраняет уступ, а `polytope_hull` — нет
(зная это заранее из диагностики на реальной "Ручка.stl", см. заметку задачи)."""

from __future__ import annotations

import numpy as np
import trimesh

from roboson_tools.geometry.mesh_io import bounding_box, from_trimesh
from roboson_tools.visual_hull import polytope_hull
from roboson_tools.visual_hull.exact_polyhedral_hull import carve

FOV_DEG = 80.0
DISTANCE = 1000.0
ALL_ANGLES_DISTANCES = {a: DISTANCE for a in (0.0, 45.0, 90.0, 135.0)}


def _box():
    return from_trimesh(trimesh.creation.box(extents=(200.0, 100.0, 60.0)))


def _stepped_rod(r_grip: float = 20.0, r_bare: float = 8.0, l_grip: float = 60.0, l_bare: float = 60.0):
    """Два соосных цилиндра разного радиуса, встык, ось вдоль X (roll axis) — синтетический
    аналог "толстая зажимаемая часть + тонкий стержень без зажима" у реальной "Ручка.stl"."""
    cyl1 = trimesh.creation.cylinder(radius=r_grip, height=l_grip, sections=48)
    cyl1.apply_translation([0, 0, l_grip / 2.0])
    cyl2 = trimesh.creation.cylinder(radius=r_bare, height=l_bare, sections=48)
    cyl2.apply_translation([0, 0, -l_bare / 2.0])
    combo = trimesh.util.concatenate([cyl1, cyl2])
    combo.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0]))
    return from_trimesh(combo)


def _radius_at_x(vertices: np.ndarray, faces: np.ndarray, x: float) -> float:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    section = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
    assert section is not None, f"пустое сечение на x={x}"
    pts = section.vertices
    return float(np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2).max())


def test_single_azimuth_is_unbounded_returns_none():
    """Один азимут не ограничивает конус вдоль оси взгляда — тот же вырожденный случай, что и
    у `polytope_hull` (см. его тесты); для одного вида пересекать нечего."""
    mesh = _box()
    result = carve(
        mesh, [(0.0, "center")], fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES,
        resolution_px=256,
    )
    assert result is None


def test_convex_object_matches_polytope_hull():
    """На ВЫПУКЛОМ объекте (короб) exact-метод обязан совпасть с `polytope_hull` (с точностью
    до растровой погрешности) — выпуклая оболочка выпуклого контура не меняет его форму, поэтому
    два метода на конусах одинаковой геометрии должны давать один и тот же многогранник."""
    mesh = _box()
    views = [(a, "center") for a in (0.0, 45.0, 90.0, 135.0)]
    exact_hull = carve(
        mesh, views, fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES, resolution_px=1024
    )
    poly_hull = polytope_hull.carve(
        mesh, views, fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES, resolution_px=1024
    )

    assert exact_hull is not None and poly_hull is not None
    exact_dims = exact_hull.vertices.max(axis=0) - exact_hull.vertices.min(axis=0)
    poly_dims = poly_hull.vertices.max(axis=0) - poly_hull.vertices.min(axis=0)
    assert np.allclose(exact_dims, poly_dims, rtol=0.02)

    exact_vol = trimesh.Trimesh(vertices=exact_hull.vertices, faces=exact_hull.faces, process=False).volume
    from scipy.spatial import ConvexHull

    poly_vol = ConvexHull(poly_hull.vertices).volume
    assert abs(exact_vol - poly_vol) < poly_vol * 0.02


def test_stepped_rod_preserves_shoulder_polytope_hull_does_not():
    """Ключевая проверка: на ступенчатом стержне (см. докстринг модуля) exact-метод должен
    сохранить резкий уступ (плоский профиль радиуса на каждом цилиндрическом участке), а
    `polytope_hull` — размазать его в плавный конус (как на реальной "Ручка.stl")."""
    mesh = _stepped_rod(r_grip=20.0, r_bare=8.0, l_grip=60.0, l_bare=60.0)
    bbox_min, bbox_max = bounding_box(mesh.vertices)
    views = [(a, "center") for a in (0.0, 45.0, 90.0, 135.0)]
    kwargs = dict(
        fov_deg=FOV_DEG,
        camera_distances={a: 400.0 for a in (0.0, 45.0, 90.0, 135.0)},
        resolution_px=1024,
    )

    exact_hull = carve(mesh, views, **kwargs)
    poly_hull = polytope_hull.carve(mesh, views, **kwargs)
    assert exact_hull is not None and poly_hull is not None

    # Профиль вдоль тонкого (безхватного) участка стержня — от начала до чуть перед уступом (x=0).
    xs = np.linspace(bbox_min[0] + 1.0, -2.0, 8)
    exact_radii = np.array([_radius_at_x(exact_hull.vertices, exact_hull.faces, x) for x in xs])
    poly_radii = np.array([_radius_at_x(poly_hull.vertices, poly_hull.faces, x) for x in xs])

    # exact: почти плоский профиль (цилиндр, не конус) — разброс на бесхватном участке мал.
    assert exact_radii.max() - exact_radii.min() < 2.0
    # polytope_hull: тот же участок размазан линейным скатом от радиуса стержня к радиусу
    # хвата — разброс на порядок больше и монотонно растёт к уступу.
    assert poly_radii.max() - poly_radii.min() > 8.0
    assert np.all(np.diff(poly_radii) > 0)  # монотонный рост радиуса вдоль всего "конуса"


def test_degenerate_case_returns_none():
    """Объект вне конуса обзора -> None, а не падение (тот же контракт, что у polytope_hull)."""
    mesh = _box()
    result = carve(mesh, [(0.0, "center")], fov_deg=1.0, camera_distances={0.0: 1.0}, resolution_px=64)
    assert result is None


def test_stable_across_yaw_sweep():
    """Регрессия на баг с дублирующей замыкающей точкой контура: `trimesh.creation.
    triangulate_polygon` возвращает вершины кольца силуэта КАК ЕСТЬ, включая точку, совпадающую
    с первой (шейпли хранит кольцо замкнутым); earcut эту дублирующую вершину в cap_faces не
    использует, но `cKDTree`-сопоставление границы (см. `_view_cone_mesh`) при нулевом допуске
    могло вернуть ЛЮБОЙ из двух индексов одной и той же точки — если не тот, что в cap_faces,
    получался разрыв 2-многообразия и `carve` тихо возвращал None. Баг проявлялся не всегда
    (зависело от порядка точек в конкретном контуре) — раньше на этом коробе ломался почти
    каждый 5-й yaw из 19 (реальный отчёт пользователя: "на коробке при yaw=0 не показывает,
    только на некоторых углах"), поэтому здесь прогоняется плотный перебор, а не пара углов."""
    from roboson_tools.geometry.transform import apply_orientation

    mesh = _box()
    views = [(a, "center") for a in (0.0, 45.0, 90.0, 135.0)]
    failures = []
    for yaw in range(0, 181, 5):
        rotated = apply_orientation(mesh, roll_deg=0.0, pitch_deg=0.0, yaw_deg=float(yaw))
        hull = carve(
            rotated, views, fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES, resolution_px=1024
        )
        if hull is None:
            failures.append(yaw)
    assert not failures, f"carve() вернул None на yaw={failures}"
