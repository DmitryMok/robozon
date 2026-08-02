"""3D-реконструкция по (маска, поза камеры) — `visual_hull/pose_carving.py`, обобщение
`polytope_hull.py`/`exact_polyhedral_hull.py` на произвольную позу (см. заметку задачи
"3D-реконструкция объекта по реальным фото — от маски+позы к visual hull").

Два вида проверок:
1. Регрессия: `CameraPose`, эквивалентная кольцевой камере (тот же apex/forward/right/up),
   должна дать ПОБИТОВО тот же halfspace/cone, что и кольцевая модель — генерализация не должна
   ничего изменить в уже работающем пути.
2. Произвольная (не кольцевая) поза: рендерим силуэты независимой (от кода под тестом) ручной
   pinhole-проекцией через look-at камеры с наклоном вне плоскости кольца — реконструкция должна
   дать габариты, близкие к истинным габаритам короба.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from roboson_tools.geometry.mesh_io import Mesh, from_trimesh
from roboson_tools.pose.camera_pose import CameraIntrinsics, CameraPose
from roboson_tools.silhouette.analytical import PRINCIPAL_AXIS
from roboson_tools.silhouette.camera import build_silhouette, camera_frame
from roboson_tools.visual_hull import exact_polyhedral_hull, polytope_hull, pose_carving

FOV_DEG = 80.0
DISTANCE = 1000.0
RESOLUTION = 1024
ANGLES = (0.0, 45.0, 90.0, 135.0)


def _box():
    return from_trimesh(trimesh.creation.box(extents=(200.0, 100.0, 60.0)))


def _ring_pose(view_angle_deg: float, fov_deg: float, camera_distance: float, resolution_px: int) -> CameraPose:
    """`CameraPose`, геометрически эквивалентная кольцевой камере на азимуте `view_angle_deg`
    (см. докстринг `_cone_geometry.py`: down=-PRINCIPAL_AXIS, right=r_theta, fx=fy=f_px)."""
    camera_pos, forward, r_theta, f_px, cx, cy = camera_frame(
        view_angle_deg, fov_deg, camera_distance, resolution_px
    )
    down = -PRINCIPAL_AXIS
    # rotation: мир->камера, x_cam = rotation @ x_world + translation. Столбцы rotation.T —
    # мировые оси камеры (right, down, forward), т.е. rotation = [right; down; forward]
    # (по строкам) — обратное свойство CameraPose.{right_world,down_world,forward_world}.
    rotation = np.stack([r_theta, down, forward], axis=0)
    translation = -rotation @ camera_pos
    intrinsics = CameraIntrinsics(
        fx=f_px, fy=f_px, cx=cx, cy=cy, resolution_px=(resolution_px, resolution_px)
    )
    return CameraPose(rotation=rotation, translation=translation, intrinsics=intrinsics)


def test_ring_equivalent_pose_matches_ring_model_polytope():
    mesh = _box()
    camera_distances = {a: DISTANCE for a in ANGLES}
    views = [(a, "center") for a in ANGLES]

    ring_hull = polytope_hull.carve(
        mesh, views, fov_deg=FOV_DEG, camera_distances=camera_distances, resolution_px=RESOLUTION
    )
    assert ring_hull is not None

    observations = []
    for angle in ANGLES:
        pose = _ring_pose(angle, FOV_DEG, DISTANCE, RESOLUTION)
        silhouette = build_silhouette(
            mesh, angle, RESOLUTION, fov_deg=FOV_DEG, camera_distance=DISTANCE, belt_position="center"
        )
        observations.append(pose_carving.Observation(mask=silhouette.mask, pose=pose))

    pose_hull = pose_carving.carve_polytope(observations)
    assert pose_hull is not None

    # HalfspaceIntersection не гарантирует одинаковый порядок вершин — сравниваем
    # лексикографически отсортированные массивы.
    ring_sorted = ring_hull.vertices[np.lexsort(ring_hull.vertices.T[::-1])]
    pose_sorted = pose_hull.vertices[np.lexsort(pose_hull.vertices.T[::-1])]
    assert ring_sorted.shape == pose_sorted.shape
    assert np.allclose(ring_sorted, pose_sorted, atol=1e-6)


def test_ring_equivalent_pose_matches_ring_model_exact():
    mesh = _box()
    camera_distances = {a: DISTANCE for a in ANGLES}
    views = [(a, "center") for a in ANGLES]

    ring_hull = exact_polyhedral_hull.carve(
        mesh, views, fov_deg=FOV_DEG, camera_distances=camera_distances, resolution_px=RESOLUTION
    )
    assert ring_hull is not None

    far_distance = max(camera_distances.values()) * 2.0
    observations = []
    for angle in ANGLES:
        pose = _ring_pose(angle, FOV_DEG, DISTANCE, RESOLUTION)
        silhouette = build_silhouette(
            mesh, angle, RESOLUTION, fov_deg=FOV_DEG, camera_distance=DISTANCE, belt_position="center"
        )
        observations.append(pose_carving.Observation(mask=silhouette.mask, pose=pose))

    pose_hull = pose_carving.carve_exact(observations, far_distance=far_distance)
    assert pose_hull is not None

    ring_vol = trimesh.Trimesh(vertices=ring_hull.vertices, faces=ring_hull.faces, process=False).volume
    pose_vol = trimesh.Trimesh(vertices=pose_hull.vertices, faces=pose_hull.faces, process=False).volume
    assert abs(ring_vol - pose_vol) < 1e-3 * max(ring_vol, 1.0)


def test_single_observation_polytope_is_unbounded_returns_none():
    mesh = _box()
    pose = _ring_pose(0.0, FOV_DEG, DISTANCE, RESOLUTION)
    silhouette = build_silhouette(
        mesh, 0.0, RESOLUTION, fov_deg=FOV_DEG, camera_distance=DISTANCE, belt_position="center"
    )
    assert pose_carving.carve_polytope([pose_carving.Observation(mask=silhouette.mask, pose=pose)]) is None


def test_single_observation_exact_returns_none():
    mesh = _box()
    pose = _ring_pose(0.0, FOV_DEG, DISTANCE, RESOLUTION)
    silhouette = build_silhouette(
        mesh, 0.0, RESOLUTION, fov_deg=FOV_DEG, camera_distance=DISTANCE, belt_position="center"
    )
    assert pose_carving.carve_exact([pose_carving.Observation(mask=silhouette.mask, pose=pose)]) is None


# --- произвольная (не кольцевая) поза: независимый рендер + реконструкция ----------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _look_at_pose(
    eye: np.ndarray, target: np.ndarray, up_ref: np.ndarray, intrinsics: CameraIntrinsics
) -> CameraPose:
    """Поза "смотрит из eye на target", наклон камеры относительно `up_ref` — независимая (от
    `_cone_geometry.py`) конструкция, см. вывод конвенции right/down/forward в докстринге
    `_cone_geometry.py` (right x down = forward, down = -up_ref для камеры без наклона)."""
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up_ref))
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    return CameraPose(rotation=rotation, translation=translation, intrinsics=intrinsics)


def _render_mask(mesh, pose: CameraPose) -> np.ndarray:
    """Ручная pinhole-растеризация силуэта меша по позе — намеренно НЕ переиспользует
    `_cone_geometry`/`silhouette.camera` (независимая проверка правильности знаков в
    `pose_carving`, а не тавтология «код тестирует сам себя»)."""
    intr = pose.intrinsics
    n_rows, n_cols = intr.resolution_px
    cam = (pose.rotation @ mesh.vertices.T).T + pose.translation  # (V,3), камерные координаты
    depth = cam[:, 2]
    tri_depth = depth[mesh.faces]  # (F,3)
    valid = tri_depth.min(axis=1) > 1e-6
    tri_cam = cam[mesh.faces][valid]  # (F',3,3)

    col = intr.fx * tri_cam[:, :, 0] / tri_cam[:, :, 2] + intr.cx
    row = intr.fy * tri_cam[:, :, 1] / tri_cam[:, :, 2] + intr.cy
    tri_px = np.round(np.stack([col, row], axis=-1)).astype(np.int32)  # (F',3,2)

    mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for tri in tri_px:
        cv2.fillConvexPoly(mask, tri, 255)
    return mask.astype(bool)


def test_arbitrary_tilted_poses_reconstruct_box_dims():
    """4 позы вокруг короба, каждая с ненулевым наклоном (не лежат в одной плоскости кольца,
    как ring model) — реконструкция всё равно должна дать габариты, близкие к истинным."""
    true_dims = np.array([200.0, 100.0, 60.0])
    mesh = from_trimesh(trimesh.creation.box(extents=tuple(true_dims)))
    intrinsics = CameraIntrinsics(
        fx=1200.0, fy=1200.0, cx=512.0, cy=512.0, resolution_px=(1024, 1024)
    )
    target = np.zeros(3)
    up_ref = PRINCIPAL_AXIS
    # Азимуты как у кольцевой модели, но каждая камера дополнительно приподнята/наклонена вдоль
    # X (не строго поперёк ленты, как ring model) — проверяет именно ту генерализацию, ради
    # которой заведён pose_carving (per-camera ориентация, а не общий "up").
    eyes = []
    for i, theta_deg in enumerate(ANGLES):
        theta = np.radians(theta_deg)
        v = np.array([0.0, np.cos(theta), np.sin(theta)])
        tilt = 150.0 * (-1) ** i  # чередующийся сдвиг по X — камера не лежит в плоскости Y-Z
        eyes.append(DISTANCE * v + np.array([tilt, 0.0, 0.0]))

    observations = []
    for eye in eyes:
        pose = _look_at_pose(eye, target, up_ref, intrinsics)
        mask = _render_mask(mesh, pose)
        assert mask.any(), "объект должен попадать в кадр"
        observations.append(pose_carving.Observation(mask=mask, pose=pose))

    hull = pose_carving.carve_polytope(observations)
    assert hull is not None
    dims = hull.vertices.max(axis=0) - hull.vertices.min(axis=0)
    for measured, true in zip(dims, true_dims):
        assert measured < true * 1.25  # запас на артефакт конечного числа ракурсов + наклон
        assert measured > true * 0.9

    true_volume = float(np.prod(true_dims))
    hull_volume = ConvexHull(hull.vertices).volume
    assert true_volume <= hull_volume < true_volume * 1.6


def _render_mask_distorted(mesh, pose: CameraPose, dist_coeffs: np.ndarray) -> np.ndarray:
    """Как `_render_mask`, но с реальной дисторсией объектива через `cv2.projectPoints`
    (forward-модель OpenCV — та же, что использует `cv2.calibrateCamera`/`solvePnP`) —
    симулирует то, что реально видит камера с неидеальной линзой (см.
    `pose_carving._undistort_mask`, заметка задачи "Учитывать дисторсию объектива...")."""
    intr = pose.intrinsics
    n_rows, n_cols = intr.resolution_px
    cam = (pose.rotation @ mesh.vertices.T).T + pose.translation
    depth = cam[:, 2]
    tri_depth = depth[mesh.faces]
    valid = tri_depth.min(axis=1) > 1e-6
    tri_cam = cam[mesh.faces][valid].reshape(-1, 3)

    px, _ = cv2.projectPoints(tri_cam, np.zeros(3), np.zeros(3), intr.matrix, dist_coeffs)
    tri_px = np.round(px.reshape(-1, 3, 2)).astype(np.int32)

    mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for tri in tri_px:
        cv2.fillConvexPoly(mask, tri, 255)
    return mask.astype(bool)


def test_undistort_mask_matches_ideal_mask_within_tolerance():
    """`pose_carving._undistort_mask` должен вернуть маску, близкую к идеальной (без дисторсии) —
    прямая проверка самой коррекции по IoU с эталонной (недисторсированной) маской, без
    confound-факторов пересечения нескольких конусов/контурной аппроксимации (см. заметку задачи
    "Учитывать дисторсию объектива в маске силуэта при 3D-реконструкции по фото").

    Дисторсия рендерится через `cv2.projectPoints` (forward-модель OpenCV, см.
    `_render_mask_distorted`) — независимо от `_undistort_mask` (та использует `cv2.undistort`,
    обратную операцию), поэтому это не тавтология "код тестирует сам себя"."""
    # Дисторсия растёт с расстоянием от главной точки кадра (см. заметку задачи — на реальных
    # фото phone2 у центра кадра сдвиг ~0px, у края маски бутылки ~20-23px) — сдвигаем короб
    # вдоль X (в конвенции ring model это "вверх/вниз в кадре", см. down=-PRINCIPAL_AXIS в
    # `_cone_geometry.py`), чтобы силуэт оказался вдали от центра, как реальный объект у
    # старта/конца ленты, а не в середине кадра, где дисторсия мала по построению.
    mesh = _box()
    shifted = Mesh(vertices=mesh.vertices + np.array([300.0, 0.0, 0.0]), faces=mesh.faces)
    pose = _ring_pose(30.0, FOV_DEG, DISTANCE, RESOLUTION)
    ideal_mask = build_silhouette(
        shifted, 30.0, RESOLUTION, fov_deg=FOV_DEG, camera_distance=DISTANCE, belt_position="center"
    ).mask

    dist_coeffs = np.array([-0.55, 0.25, 0.0, 0.0, 0.0])  # сопоставимо с реальной калибровкой phone2 (k1=-0.6)
    distorted_mask = _render_mask_distorted(shifted, pose, dist_coeffs)
    assert distorted_mask.any()

    intr_dist = CameraIntrinsics(
        fx=pose.intrinsics.fx, fy=pose.intrinsics.fy, cx=pose.intrinsics.cx, cy=pose.intrinsics.cy,
        resolution_px=pose.intrinsics.resolution_px, dist_coeffs=dist_coeffs,
    )
    corrected_mask = pose_carving._undistort_mask(distorted_mask, intr_dist)

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        return float((a & b).sum()) / float((a | b).sum())

    iou_before = _iou(distorted_mask, ideal_mask)
    iou_after = _iou(corrected_mask, ideal_mask)
    assert iou_after > iou_before + 0.03, (
        f"коррекция должна заметно приблизить искажённую маску к идеальной: "
        f"IoU до={iou_before:.3f} после={iou_after:.3f}"
    )
    assert iou_after > 0.9, f"после коррекции IoU с идеальной маской должен быть высоким: {iou_after:.3f}"


def test_undistort_mask_noop_without_dist_coeffs():
    """`dist_coeffs=None` (кольцевая модель/синтетика без дисторсии) — маска не меняется вообще
    (не просто "почти", а тот же массив/значения) — не должно быть накладных расходов или
    случайных искажений там, где дисторсии нет по определению."""
    mesh = _box()
    pose = _ring_pose(0.0, FOV_DEG, DISTANCE, RESOLUTION)
    mask = build_silhouette(
        mesh, 0.0, RESOLUTION, fov_deg=FOV_DEG, camera_distance=DISTANCE, belt_position="center"
    ).mask
    result = pose_carving._undistort_mask(mask, pose.intrinsics)
    assert result is mask


# --- наблюдения через отражение в зеркале (Observation.mirror_plane) --------------------------


def _reflect_pose_independent(
    pose: CameraPose, mirror_point: np.ndarray, mirror_normal: np.ndarray, intrinsics: CameraIntrinsics
) -> CameraPose:
    """Независимый (НЕ через `pose_carving.reflect_camera_frame`) пересчёт зеркально-отражённой
    камеры — та же формула, руками, чтобы тест не был тавтологией "код проверяет сам себя" (см.
    докстринг `_render_mask`). Отражение — инволюция (`reflect(reflect(X)) == X`), поэтому эта же
    функция годится в обе стороны: "дай виртуальную камеру по реальной" и "дай реальную камеру,
    которая при отражении дала бы вот эту желаемую виртуальную" (используется тестами ниже, чтобы
    гарантированно строить физически осмысленную "реальную" позу — см. их докстринги)."""

    def reflect_point(p: np.ndarray) -> np.ndarray:
        return p - 2.0 * ((p - mirror_point) @ mirror_normal) * mirror_normal

    def reflect_dir(v: np.ndarray) -> np.ndarray:
        return v - 2.0 * (v @ mirror_normal) * mirror_normal

    apex = reflect_point(pose.camera_pos_world)
    forward = reflect_dir(pose.forward_world)
    right = reflect_dir(pose.right_world)
    down = reflect_dir(pose.down_world)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ apex
    return CameraPose(rotation=rotation, translation=translation, intrinsics=intrinsics)


def test_mirror_observation_matches_equivalent_direct_pose_halfspaces():
    """`Observation(mask, real_pose, mirror_plane)` должен давать ПОБИТОВО тот же halfspace,
    что и `Observation(mask, virtual_pose)` без `mirror_plane` — где `virtual_pose` та же самая
    виртуальная камера, но упакованная как обычная (не помеченная) прямая поза. Проверяет саму
    проводку (`_camera_frame_for_observation`), а не только формулу отражения.

    `virtual_pose` строится как обычный рабочий ракурс (look-at на объект, как и в других
    тестах) — гарантированно видит объект; `real_pose` получается ОТРАЖЕНИЕМ этой желаемой
    виртуальной позы (отражение — инволюция, см. `_reflect_pose_independent`), поэтому при
    отражении `real_pose` обратно код должен восстановить исходную `virtual_pose`."""
    intrinsics = CameraIntrinsics(fx=900.0, fy=900.0, cx=320.0, cy=240.0, resolution_px=(480, 640))
    mirror_point = np.array([800.0, 0.0, 0.0])
    mirror_normal = np.array([1.0, 0.0, 0.0])

    virtual_pose = _look_at_pose(
        DISTANCE * np.array([0.0, np.cos(np.radians(225.0)), np.sin(np.radians(225.0))]),
        np.zeros(3), PRINCIPAL_AXIS, intrinsics,
    )
    real_pose = _reflect_pose_independent(virtual_pose, mirror_point, mirror_normal, intrinsics)

    mesh = _box()
    mask = _render_mask(mesh, virtual_pose)
    assert mask.any()

    via_mirror = pose_carving._halfspaces_from_observation(
        pose_carving.Observation(mask=mask, pose=real_pose, mirror_plane=(mirror_point, mirror_normal))
    )
    via_direct = pose_carving._halfspaces_from_observation(
        pose_carving.Observation(mask=mask, pose=virtual_pose)
    )
    assert via_mirror is not None and via_direct is not None
    np.testing.assert_allclose(via_mirror[0], via_direct[0], atol=1e-9)
    np.testing.assert_allclose(via_mirror[1], via_direct[1], atol=1e-9)


def test_mirror_reflection_observation_contributes_to_reconstruction():
    """Смешанный набор наблюдений (2 прямых + 1 через отражение в зеркале) должен восстановить
    габариты короба — доказывает, что отражение реально используется как полноценный
    дополнительный ракурс, а не игнорируется и не портит результат (пользовательское требование —
    отражение должно ПОВЫШАТЬ точность, не исключаться из реконструкции).

    Построение "реальной" позы — см. `test_mirror_observation_matches_equivalent_direct_pose_halfspaces`:
    через отражение заведомо рабочего ракурса (гарантирует, что рендер непустой и физически
    осмысленный, а не случайно смотрит мимо объекта)."""
    true_dims = np.array([200.0, 100.0, 60.0])
    mesh = from_trimesh(trimesh.creation.box(extents=tuple(true_dims)))
    intrinsics = CameraIntrinsics(fx=1200.0, fy=1200.0, cx=512.0, cy=512.0, resolution_px=(1024, 1024))
    target = np.zeros(3)

    observations = []
    for theta_deg in (0.0, 90.0):
        theta = np.radians(theta_deg)
        eye = DISTANCE * np.array([0.0, np.cos(theta), np.sin(theta)])
        pose = _look_at_pose(eye, target, PRINCIPAL_AXIS, intrinsics)
        mask = _render_mask(mesh, pose)
        assert mask.any()
        observations.append(pose_carving.Observation(mask=mask, pose=pose))

    mirror_point = np.array([800.0, 0.0, 0.0])
    mirror_normal = np.array([1.0, 0.0, 0.0])
    virtual_pose = _look_at_pose(
        DISTANCE * np.array([0.0, np.cos(np.radians(225.0)), np.sin(np.radians(225.0))]),
        target, PRINCIPAL_AXIS, intrinsics,
    )
    real_pose = _reflect_pose_independent(virtual_pose, mirror_point, mirror_normal, intrinsics)
    mirror_mask = _render_mask(mesh, virtual_pose)
    assert mirror_mask.any()
    observations.append(
        pose_carving.Observation(mask=mirror_mask, pose=real_pose, mirror_plane=(mirror_point, mirror_normal))
    )

    hull = pose_carving.carve_polytope(observations)
    assert hull is not None, "реконструкция с зеркальным наблюдением не должна давать None"
    dims = hull.vertices.max(axis=0) - hull.vertices.min(axis=0)
    for measured, true in zip(dims, true_dims):
        assert measured < true * 1.3
        assert measured > true * 0.85


def test_wrong_pose_for_mirror_mask_breaks_reconstruction():
    """Регрессия на изначально репортованный пользователем баг: если маску, снятую ЧЕРЕЗ
    отражение, скормить реконструкции БЕЗ mirror_plane (как будто это прямое наблюдение
    реальной камерой) — результат должен быть либо None, либо явно не совпадать с истинными
    габаритами (реальный случай на настоящих фото: HalfspaceIntersection не находит общий
    объём). Документирует, ПОЧЕМУ нужен mirror_plane, а не просто демонстрирует его пользу."""
    true_dims = np.array([200.0, 100.0, 60.0])
    mesh = from_trimesh(trimesh.creation.box(extents=tuple(true_dims)))
    intrinsics = CameraIntrinsics(fx=1200.0, fy=1200.0, cx=512.0, cy=512.0, resolution_px=(1024, 1024))
    target = np.zeros(3)

    observations = []
    for theta_deg in (0.0, 90.0):
        theta = np.radians(theta_deg)
        eye = DISTANCE * np.array([0.0, np.cos(theta), np.sin(theta)])
        pose = _look_at_pose(eye, target, PRINCIPAL_AXIS, intrinsics)
        observations.append(pose_carving.Observation(mask=_render_mask(mesh, pose), pose=pose))

    mirror_point = np.array([800.0, 0.0, 0.0])
    mirror_normal = np.array([1.0, 0.0, 0.0])
    virtual_pose = _look_at_pose(
        DISTANCE * np.array([0.0, np.cos(np.radians(225.0)), np.sin(np.radians(225.0))]),
        target, PRINCIPAL_AXIS, intrinsics,
    )
    real_pose = _reflect_pose_independent(virtual_pose, mirror_point, mirror_normal, intrinsics)
    mirror_mask = _render_mask(mesh, virtual_pose)

    # БАГ: маска отражения подана с РЕАЛЬНОЙ (не виртуальной) позой и БЕЗ mirror_plane.
    buggy_observations = observations + [pose_carving.Observation(mask=mirror_mask, pose=real_pose)]

    buggy_hull = pose_carving.carve_polytope(buggy_observations)
    if buggy_hull is not None:
        buggy_dims = buggy_hull.vertices.max(axis=0) - buggy_hull.vertices.min(axis=0)
        assert not np.allclose(buggy_dims, true_dims, rtol=0.3)


# --- консенсусная реконструкция (CandidateObservation / carve_consensus) ----------------------


def _consensus_setup():
    """3 прямых наблюдения короба + материал для 4-го с «чужим» первым кандидатом: маска другого
    объекта (сдвинутый бокс), несовместимая с остальными ракурсами — синтетический аналог
    реального кейса phone1 (отражённая в зеркале канистра детектится увереннее целевой банки,
    в одном кадре геометрически неотличима, отличает её только несогласованность с другими
    ракурсами)."""
    true_dims = np.array([200.0, 100.0, 60.0])
    mesh = from_trimesh(trimesh.creation.box(extents=tuple(true_dims)))
    # Сдвиг и вбок (x), и вдоль (y): конус обзора через decoy должен пройти МИМО пересечения
    # конусов остальных ракурсов (иначе консенсус сочтёт их совместимыми — конусы бесконечны,
    # чисто продольного сдвига недостаточно, проверено при разработке теста). center=False
    # ОБЯЗАТЕЛЕН: `from_trimesh` по умолчанию центрирует вершины по bbox и молча стёр бы сдвиг.
    decoy_mesh = from_trimesh(
        trimesh.creation.box(
            extents=(80.0, 80.0, 80.0),
            transform=trimesh.transformations.translation_matrix([250.0, 350.0, 0.0]),
        ),
        center=False,
    )
    intrinsics = CameraIntrinsics(fx=1200.0, fy=1200.0, cx=512.0, cy=512.0, resolution_px=(1024, 1024))
    target = np.zeros(3)

    poses = []
    for theta_deg in (0.0, 60.0, 120.0, 210.0):
        theta = np.radians(theta_deg)
        eye = DISTANCE * np.array([0.0, np.cos(theta), np.sin(theta)])
        poses.append(_look_at_pose(eye, target, PRINCIPAL_AXIS, intrinsics))
    return mesh, decoy_mesh, poses, true_dims


def test_carve_consensus_switches_to_consistent_candidate():
    mesh, decoy_mesh, poses, true_dims = _consensus_setup()

    observations = [
        pose_carving.CandidateObservation(
            candidates=[_render_mask(mesh, pose)], pose=pose, label=f"direct{i}"
        )
        for i, pose in enumerate(poses[:3])
    ]
    # 4-е наблюдение: первый кандидат — ЧУЖОЙ объект (уверенная детекция фона), второй — цель.
    decoy_mask = _render_mask(decoy_mesh, poses[3])
    true_mask = _render_mask(mesh, poses[3])
    assert decoy_mask.any() and true_mask.any()
    observations.append(
        pose_carving.CandidateObservation(
            candidates=[decoy_mask, true_mask], pose=poses[3], label="with-decoy"
        )
    )

    result = pose_carving.carve_consensus(observations)

    assert result.chosen == [0, 0, 0, 1]  # чужой кандидат заменён вторым, ничего не выброшено
    assert result.switched_labels == ["with-decoy"]
    assert result.dropped_labels == []
    assert result.polytope is not None
    dims = result.polytope.vertices.max(axis=0) - result.polytope.vertices.min(axis=0)
    for measured, true in zip(dims, true_dims):
        # 1.4 — запас на артефакт «выпуклый халл по 4 грубым ракурсам шире объекта» (см.
        # test_arbitrary_tilted_poses_reconstruct_box_dims — там 8+ ракурсов и допуск теснее).
        assert measured < true * 1.4
        assert measured > true * 0.85


def test_carve_consensus_drops_observation_without_valid_candidates():
    mesh, decoy_mesh, poses, _true_dims = _consensus_setup()

    observations = [
        pose_carving.CandidateObservation(
            candidates=[_render_mask(mesh, pose)], pose=pose, label=f"direct{i}"
        )
        for i, pose in enumerate(poses[:3])
    ]
    # У несогласованного наблюдения ЕДИНСТВЕННЫЙ кандидат — чужой объект: заменить нечем,
    # консенсус должен выбросить наблюдение целиком, а не уронить всю реконструкцию.
    observations.append(
        pose_carving.CandidateObservation(
            candidates=[_render_mask(decoy_mesh, poses[3])], pose=poses[3], label="decoy-only"
        )
    )

    result = pose_carving.carve_consensus(observations)

    assert result.chosen[:3] == [0, 0, 0]
    assert result.chosen[3] is None
    assert result.dropped_labels == ["decoy-only"]
    assert result.polytope is not None


def test_carve_consensus_no_conflict_keeps_all_first_candidates():
    mesh, _decoy_mesh, poses, _true_dims = _consensus_setup()
    observations = [
        pose_carving.CandidateObservation(
            candidates=[_render_mask(mesh, pose)], pose=pose, label=f"direct{i}"
        )
        for i, pose in enumerate(poses)
    ]
    result = pose_carving.carve_consensus(observations)
    assert result.chosen == [0, 0, 0, 0]
    assert result.switched_labels == [] and result.dropped_labels == []
    assert result.polytope is not None and result.exact is not None
