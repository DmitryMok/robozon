"""Юнит-тесты Camera Mode (перспективная проекция) — см. silhouette/camera.py и заметку задачи
"Добавить Camera Mode в roboson_tools (перспективные ракурсы вдоль конвейера)"."""

from __future__ import annotations

import numpy as np
import trimesh

from roboson_tools.geometry.mesh_io import from_trimesh
from roboson_tools.silhouette.analytical import build_silhouette as build_analytical_silhouette
from roboson_tools.silhouette.camera import build_silhouette


def _col_width_world(mask: np.ndarray, px_per_unit: float) -> float:
    cols = np.any(mask, axis=0)
    idx = np.flatnonzero(cols)
    return (idx[-1] - idx[0] + 1) / px_per_unit


def _row_width_world(mask: np.ndarray, px_per_unit: float) -> float:
    rows = np.any(mask, axis=1)
    idx = np.flatnonzero(rows)
    return (idx[-1] - idx[0] + 1) / px_per_unit


def _row_center(mask: np.ndarray) -> float:
    rows = np.any(mask, axis=1)
    idx = np.flatnonzero(rows)
    return (idx[0] + idx[-1]) / 2.0


def test_perspective_width_matches_similar_triangles():
    """Тонкая (по глубине от камеры) пластина на известном расстоянии — вся геометрия почти в
    одной плоскости, ширина силуэта должна следовать простому подобию треугольников:
    px_width = f_px * world_width / distance."""
    w_x, w_z, thin_y = 100.0, 60.0, 1.0
    mesh = from_trimesh(trimesh.creation.box(extents=(w_x, thin_y, w_z)))

    fov_deg = 80.0
    distance = 1000.0
    resolution = 4096  # тонкий силуэт занимает малую часть кадра — нужен запас пикселей, чтобы
    # ошибка округления int32-растеризации не перекрывала проверяемый эффект подобия треугольников

    sil = build_silhouette(
        mesh,
        view_angle_deg=0.0,  # v_theta=(0,1,0), r_theta=(0,0,1) — камера вдоль Y, смотрит на -Y
        resolution_px=resolution,
        fov_deg=fov_deg,
        camera_distance=distance,
        belt_position="center",
    )

    width_z = _col_width_world(sil.mask, sil.px_per_unit)  # вдоль r_theta (Z)
    width_x = _row_width_world(sil.mask, sil.px_per_unit)  # вдоль PRINCIPAL_AXIS (X)

    assert abs(width_z - w_z) / w_z < 0.02
    assert abs(width_x - w_x) / w_x < 0.02


def test_perspective_converges_to_orthographic_at_large_distance():
    """При camera_distance, много большей габарита объекта, перспективные искажения (foreshortening
    между ближней и дальней гранью куба) должны стать пренебрежимо малы, а ширина силуэта —
    приблизиться к ортографической (см. test_silhouette_analytical.py)."""
    w = 100.0
    mesh = from_trimesh(trimesh.creation.box(extents=(w, w, w)))
    fov_deg = 80.0
    resolution = 4096

    ortho = build_analytical_silhouette(mesh, view_angle_deg=0.0, resolution_px=resolution)
    ortho_width = _col_width_world(ortho.mask, ortho.px_per_unit)

    def perspective_width(distance: float) -> float:
        sil = build_silhouette(
            mesh,
            view_angle_deg=0.0,
            resolution_px=resolution,
            fov_deg=fov_deg,
            camera_distance=distance,
            belt_position="center",
        )
        return _col_width_world(sil.mask, sil.px_per_unit)

    err_near = abs(perspective_width(500.0) - ortho_width) / ortho_width
    err_far = abs(perspective_width(5000.0) - ortho_width) / ortho_width

    assert err_near > 0.05  # на близкой дистанции (полу-габарит/расстояние = 0.1) эффект заметен
    assert err_far < err_near
    assert err_far < 0.05  # на дальней дистанции — почти как ортографика


def test_silhouette_symmetric_for_centered_object():
    """Регрессия: короб, центрированный в начале координат, на view_angle=0/belt_position=
    center должен давать силуэт, СИММЕТРИЧНЫЙ относительно центра кадра (cx, cy) — раньше
    `np.round(...).astype(int32)`-растеризация была заменена голым `astype(int32)`
    (усечение к нулю = floor для положительных пиксельных координат), что систематически
    (не случайно — ровно на 1px при ЛЮБОМ разрешении) сдвигало силуэт в одну сторону.
    При маленьком объекте в кадре (типичная сцена Camera Mode: объект — небольшая часть
    512-1024px кадра) это давало заметную асимметрию 3D-реконструкции (voxel carving)."""
    mesh = from_trimesh(trimesh.creation.box(extents=(200.0, 100.0, 60.0)))
    resolution = 512

    sil = build_silhouette(
        mesh,
        view_angle_deg=0.0,
        resolution_px=resolution,
        fov_deg=80.0,
        camera_distance=1000.0,
        belt_position="center",
    )
    cx = cy = resolution / 2.0
    cols = np.flatnonzero(np.any(sil.mask, axis=0))
    rows = np.flatnonzero(np.any(sil.mask, axis=1))

    assert abs((cx - cols.min()) - (cols.max() - cx)) < 1e-9
    assert abs((cy - rows.min()) - (rows.max() - cy)) < 1e-9


def test_belt_position_shifts_silhouette():
    """start/center/end должны давать заметно различающиеся (и по разные стороны от center)
    положения силуэта по вертикали кадра — см. план задачи, п. belt_position."""
    w = 100.0
    mesh = from_trimesh(trimesh.creation.box(extents=(w, w, w)))
    fov_deg = 80.0
    distance = 1000.0
    resolution = 1024

    def build(belt_position):
        return build_silhouette(
            mesh,
            view_angle_deg=0.0,
            resolution_px=resolution,
            fov_deg=fov_deg,
            camera_distance=distance,
            belt_position=belt_position,
        )

    center = _row_center(build("center").mask)
    start = _row_center(build("start").mask)
    end = _row_center(build("end").mask)

    assert abs(start - center) > 1.0
    assert abs(end - center) > 1.0
    assert (start - center) * (end - center) < 0  # по разные стороны от center


def _touches_border(mask: np.ndarray) -> bool:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    height, width = mask.shape
    return bool(
        rows.min() == 0 or rows.max() == height - 1 or cols.min() == 0 or cols.max() == width - 1
    )


def test_belt_offset_keeps_object_with_diameter_off_frame_border():
    """Регрессия: старая формула `belt_offset` считала depth объекта ~= camera_distance без
    поправки на собственный диаметр объекта в плоскости Y-Z (transverse_radius). Для объекта с
    заметным диаметром (не тонкой пластины, например стоящая вертикально бутылка) точки на
    БЛИЖНЕЙ к камере стороне имеют depth МЕНЬШЕ camera_distance и на позиции start/end реально
    проецировались за край кадра — верхушка/горлышко пропадали из силуэта, хотя формально
    "безопасный" запас по старой формуле был соблюдён (см. заметку задачи, разбор бутылки)."""
    # Цилиндр: длинная ось вдоль X (roll axis) — как бутылка, стоящая вертикально в системе
    # координат этого стенда (см. докстринг analytical.py — X в кадре соответствует вертикали).
    mesh = from_trimesh(
        trimesh.creation.cylinder(radius=45.0, height=300.0, sections=32).apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        )
    )
    fov_deg, distance, resolution = 80.0, 1000.0, 512

    for belt_position in ("start", "end"):
        for angle in (0.0, 45.0, 90.0, 135.0):
            sil = build_silhouette(
                mesh, angle, resolution,
                fov_deg=fov_deg, camera_distance=distance, belt_position=belt_position,
            )
            assert sil.mask.any()
            assert not _touches_border(sil.mask), f"{belt_position} angle={angle}"


def test_belt_offset_keeps_thin_elongated_object_off_frame_border():
    """Регрессия второго порядка: после поправки на transverse_radius для ТОНКОГО вытянутого
    объекта (диаметр в мм при габарите вдоль X на порядки больше) добавка margin_factor,
    масштабированная на extent_x_from_axis, давала исчезающе малый абсолютный запас (доли
    процента от углового бюджета) — расчёт формально проходил, но растровое округление всё
    равно срезало объект вплотную к границе кадра (см. заметку задачи, разбор тонкой ручки).
    margin_factor должен масштабировать доступный угловой бюджет, а не размер объекта."""
    mesh = from_trimesh(
        trimesh.creation.cylinder(radius=4.5, height=150.0, sections=24).apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        )
    )
    fov_deg, distance, resolution = 80.0, 1000.0, 512

    for belt_position in ("start", "end"):
        for angle in (0.0, 45.0, 90.0, 135.0):
            sil = build_silhouette(
                mesh, angle, resolution,
                fov_deg=fov_deg, camera_distance=distance, belt_position=belt_position,
            )
            assert sil.mask.any()
            assert not _touches_border(sil.mask), f"{belt_position} angle={angle}"


def test_supersample_coverage_and_subpixel_contour():
    """Субпиксельный режим (supersample > 1): coverage — мягкая маска покрытия [0..1] той же
    формы, бинарная mask согласована с ней (coverage >= 0.5), суммарная площадь совпадает с
    бинарной в пределах периметра границы, а contour_px локализует границу с точностью
    лучше 1px (сходимость ширины к аналитическому подобию треугольников)."""
    w_x, thin_y, w_z = 100.0, 1.0, 60.0  # тонкая ось — по направлению взгляда (Y, азимут 0)
    mesh = from_trimesh(trimesh.creation.box(extents=(w_x, thin_y, w_z)))
    fov_deg, distance, resolution = 68.0, 2000.0, 512

    binary = build_silhouette(
        mesh, 0.0, resolution, fov_deg=fov_deg, camera_distance=distance, supersample=1
    )
    soft = build_silhouette(
        mesh, 0.0, resolution, fov_deg=fov_deg, camera_distance=distance, supersample=4
    )
    assert binary.coverage is None and binary.contour_px is None
    assert soft.coverage is not None and soft.contour_px is not None
    assert soft.coverage.shape == soft.mask.shape == binary.mask.shape
    assert soft.coverage.min() >= 0.0 and soft.coverage.max() <= 1.0
    assert np.array_equal(soft.mask, soft.coverage >= 0.5)
    # Площадь мягкой маски — почти аналитическая (подобие треугольников; пластина тонкая,
    # глубина ~= distance), тогда как бинарная на маленьком в кадре объекте раздута
    # растеризацией (~+0.5px на сторону — здесь это >10% площади). В этом и смысл режима.
    f_px = (resolution / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    expected_area_px = (f_px * w_x / distance) * (f_px * w_z / distance)
    soft_err = abs(float(soft.coverage.sum()) - expected_area_px)
    binary_err = abs(float(binary.mask.sum()) - expected_area_px)
    assert soft_err < 0.05 * expected_area_px
    assert soft_err < binary_err
    # Субпиксельный контур: протяжённость по строкам (мировая ось X) — с точностью < 1px
    expected_px = f_px * w_x / distance
    rows = soft.contour_px[:, 1]
    measured_px = float(rows.max() - rows.min())
    assert abs(measured_px - expected_px) < 1.0


def test_rectangular_resolution_frame():
    """Прямоугольный кадр (n_rows, n_cols): маска нужной формы, объект в центре, f_px
    определяется FOV вдоль ленты (строки), поперечный масштаб — те же квадратные пиксели,
    поэтому мировая ширина объекта по столбцам совпадает с квадратным кадром той же высоты."""
    mesh = from_trimesh(trimesh.creation.box(extents=(120.0, 80.0, 60.0)))
    fov_deg, distance = 68.0, 2000.0

    rect = build_silhouette(
        mesh, 0.0, (648, 486), fov_deg=fov_deg, camera_distance=distance
    )
    square = build_silhouette(
        mesh, 0.0, 648, fov_deg=fov_deg, camera_distance=distance
    )
    assert rect.mask.shape == (648, 486)
    assert rect.px_per_unit == square.px_per_unit  # f_px одинаков: FOV задан вдоль строк
    assert abs(
        _col_width_world(rect.mask, rect.px_per_unit)
        - _col_width_world(square.mask, square.px_per_unit)
    ) < 2.0 / rect.px_per_unit


def test_belt_offset_accepts_fractional_positions():
    """Дробные позиции ленты (плотный параллакс): 0.5 — ровно половина сдвига "end",
    -1.0 совпадает со "start", 0.0 — с "center"."""
    from roboson_tools.silhouette.camera import belt_offset

    mesh = from_trimesh(trimesh.creation.box(extents=(100.0, 80.0, 60.0)))
    fov_deg, distance = 68.0, 2000.0
    dx_end = belt_offset(mesh, fov_deg, distance, "end")
    assert dx_end > 0
    assert belt_offset(mesh, fov_deg, distance, 0.5) == dx_end * 0.5
    assert belt_offset(mesh, fov_deg, distance, -1.0) == belt_offset(mesh, fov_deg, distance, "start")
    assert belt_offset(mesh, fov_deg, distance, 0.0) == 0.0
