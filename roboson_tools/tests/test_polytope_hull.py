"""Точное пересечение конусов обзора (visual_hull/polytope_hull.py) — альтернатива voxel
carving для выпуклых объектов, см. заметку задачи "Добавить Camera Mode..."."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from roboson_tools.geometry.mesh_io import bounding_box, from_trimesh
from roboson_tools.visual_hull.polytope_hull import _view_halfspaces, carve

FOV_DEG = 80.0
DISTANCE = 1000.0
# Единая дистанция на все азимуты, встречающиеся в этом файле — эти тесты проверяют геометрию
# пересечения конусов, а не эффект РАЗНЫХ дистанций (см. tests/test_camera_distances_consistency.py
# для последнего), поэтому per-angle карта здесь везде плоская.
ALL_ANGLES_DISTANCES = {
    a: DISTANCE for a in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, -45.0)
}


def _box():
    return from_trimesh(trimesh.creation.box(extents=(200.0, 100.0, 60.0)))


def test_single_azimuth_is_unbounded_returns_none():
    """Один азимут (даже с belt_position) не ограничивает конус вдоль оси взгляда —
    полупространства образуют НЕограниченную область, HalfspaceIntersection не может
    построить конечный многогранник -> None, а не падение/мусор."""
    mesh = _box()
    result = carve(
        mesh, [(0.0, "center")], fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES,
        resolution_px=256,
    )
    assert result is None


def test_multiple_azimuths_give_tight_convex_bound():
    """4 азимута (как Analytical Mode) должны дать выпуклый многогранник, близкий по габариту
    и объёму к истинному коробу — без сеточной аппроксимации (в отличие от voxel carving)."""
    mesh = _box()
    views = [(a, "center") for a in (0.0, 45.0, 90.0, 135.0)]
    hull = carve(mesh, views, fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES, resolution_px=1024)

    assert hull is not None
    assert len(hull.vertices) >= 4
    dims = hull.vertices.max(axis=0) - hull.vertices.min(axis=0)
    for measured, true in zip(dims, (200.0, 100.0, 60.0)):
        assert measured < true * 1.15  # запас на артефакт конечного числа ракурсов (см. docs)
        assert measured > true * 0.95  # но не меньше истинного габарита (Visual Hull его содержит)

    true_volume = 200.0 * 100.0 * 60.0
    hull_volume = ConvexHull(hull.vertices).volume
    assert true_volume <= hull_volume < true_volume * 1.3


def test_adding_views_shrinks_or_keeps_volume():
    """Пересечение полупространств: добавление кадров может только УМЕНЬШАТЬ объём (или
    оставлять тем же), никогда не увеличивать."""
    mesh = _box()
    kwargs = dict(fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES, resolution_px=512)

    two_views = carve(mesh, [(0.0, "center"), (90.0, "center")], **kwargs)
    four_views = carve(mesh, [(a, "center") for a in (0.0, 45.0, 90.0, 135.0)], **kwargs)

    assert two_views is not None
    assert four_views is not None
    vol_two = ConvexHull(two_views.vertices).volume
    vol_four = ConvexHull(four_views.vertices).volume
    assert vol_four <= vol_two + 1e-6


def test_degenerate_case_returns_none():
    """Объект вне конуса обзора -> None, а не падение."""
    mesh = _box()
    result = carve(mesh, [(0.0, "center")], fov_deg=1.0, camera_distances={0.0: 1.0}, resolution_px=64)
    assert result is None


def test_mirrored_azimuths_give_mirrored_halfspaces():
    """Диагностика асимметричного "выступа" на скриншоте (короб приподнят над одним ребром,
    хотя противоположное ребро симметрично срезано): проверяем, что математика ОДНОГО конуса
    для зеркальных азимутов (+theta и -theta вокруг продольной оси X, отражение по Z) сама по
    себе корректна. v_theta=(0,cos theta,sin theta) при theta -> -theta даёт зеркало по Z —
    полупространства при -theta должны быть точным Z-зеркалом полупространств при +theta (тот
    же набор нормалей/offsets с точностью до порядка обхода контура). Если это НЕ так — ошибка
    в самой геометрии камеры/конуса (`_view_halfspaces`). Если это ТАК (как здесь) — видимая
    асимметрия реконструкции не в математике конуса, а в форме используемого набора азимутов
    (см. `test_asymmetric_default_views_give_asymmetric_bulge` ниже). Куб, а не произвольный
    короб: для куба контур силуэта попадает на пиксельную сетку одинаково при +-theta, поэтому
    растеризация не вносит собственный шум в сравнение (для несимметричного короба пиксельное
    квантование само по себе даёт разное число вершин contour hull при +-theta — это отдельный,
    ожидаемый шум растра, а не то, что здесь проверяется)."""
    cube = from_trimesh(trimesh.creation.box(extents=(400.0, 400.0, 400.0)))
    n1, o1 = _view_halfspaces(cube, 45.0, "center", FOV_DEG, DISTANCE, resolution_px=1024)
    n2, o2 = _view_halfspaces(cube, -45.0, "center", FOV_DEG, DISTANCE, resolution_px=1024)

    n2_mirrored = n2.copy()
    n2_mirrored[:, 2] *= -1.0

    # Полупространства из cv2-контура приходят в произвольном порядке обхода — сопоставляем
    # по ближайшей нормали, а не по индексу.
    dists = np.linalg.norm(n1[:, None, :] - n2_mirrored[None, :, :], axis=-1)
    match = dists.argmin(axis=1)

    assert np.allclose(n1, n2_mirrored[match], atol=1e-6)
    assert np.allclose(o1, o2[match], atol=1e-4)


def test_asymmetric_default_views_give_asymmetric_bulge():
    """Документирует ПРИЧИНУ асимметрии на скриншоте: дефолтный набор ракурсов рига "2 камеры
    + 1 зеркало" — [0, 45, 90, 135] (`config/app_settings.yaml`) — не имеет зеркальной пары по Z
    почти ни для одного угла (theta -> -theta даёт 315/270/225, их нет в наборе), поэтому
    пересечение конусов закономерно даёт РАЗНЫЙ избыток объёма по +Z и -Z даже для идеально
    симметричного объекта. Полный симметричный набор (обе половины круга) даёт побитово
    одинаковый избыток по +Z/-Z — то есть сама математика конуса не виновата (см. тест выше),
    виновата форма набора азимутов. Это НЕ баг, а известное свойство visual hull из ракурсов
    вокруг одной оси (без разброса по elevation) — полигон никогда не станет точным для угла
    короба, только теснее с числом ракурсов. Короб почти-куб (Y~Z в сечении, как реальный
    "Короб 400х400х300.stl" со скриншота) — эффект слабо заметен на вытянутых коробах, где
    сечение Y-Z далеко от квадрата."""
    mesh = from_trimesh(trimesh.creation.box(extents=(400.0, 400.0, 300.0)))
    bbox_min, bbox_max = bounding_box(mesh.vertices)
    kwargs = dict(fov_deg=FOV_DEG, camera_distances=ALL_ANGLES_DISTANCES, resolution_px=1024)

    default_hull = carve(mesh, [(a, "center") for a in (0.0, 45.0, 90.0, 135.0)], **kwargs)
    symmetric_hull = carve(
        mesh, [(a, "center") for a in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)], **kwargs
    )
    assert default_hull is not None and symmetric_hull is not None

    def overshoot(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return vertices.max(axis=0) - bbox_max, bbox_min - vertices.min(axis=0)

    over_hi, over_lo = overshoot(default_hull.vertices)
    # Z — асимметричная ось для набора [0,45,90,135]: избыток по +Z и -Z заметно различается.
    assert abs(over_hi[2] - over_lo[2]) > 5.0

    sym_over_hi, sym_over_lo = overshoot(symmetric_hull.vertices)
    # Полный симметричный набор -> избыток по +Z/-Z практически идентичен на всех осях.
    assert np.allclose(sym_over_hi, sym_over_lo, atol=1e-6)
