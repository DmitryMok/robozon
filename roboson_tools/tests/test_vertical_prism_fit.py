"""Тесты `visual_hull/vertical_prism_fit.py` — точная триангуляция габаритов вертикальной
призмы (верхняя камера даёт форму сечения, боковые камеры + подбор высоты снимают масштабную
неоднозначность, см. докстринг модуля). Отдельно проверяем, что метод работает не только для
коробки (N=4), но и для треугольной/шестиугольной колонны (N=3/6, см. обсуждение с пользователем
"а если у нас вертикальный треугольный столб") и для цилиндра (окружность — N->бесконечность, см.
"любой N-угольник, если есть вертикальные рёбра, должен работать"), и что конические/купольные
объекты (сечение НЕ постоянно по высоте) по-прежнему отбраковываются.
"""

from __future__ import annotations

import numpy as np
import trimesh

from roboson_tools.geometry.mesh_io import from_trimesh
from roboson_tools.visual_hull.vertical_prism_fit import fit_vertical_prism

_FOV_DEG = 80.0
_CAMERA_DISTANCE = 2000.0  # мм — соответствует реальному ригу пользователя (камера сверху)
_RESOLUTION_PX = 2048
_VIEW_ANGLES = [0.0, 45.0, 90.0, 135.0]  # 90° — камера строго сверху, см. docstring модуля
_ALL_ANGLES_DISTANCES = {a: _CAMERA_DISTANCE for a in _VIEW_ANGLES}


def _box_on_ground(ex: float, ey: float, ez: float):
    """Коробка exXeyXez, основание касается земли в Z=0 (предусловие модуля)."""
    tm = trimesh.creation.box(extents=(ex, ey, ez))
    tm.apply_translation([0.0, 0.0, ez / 2.0])
    return from_trimesh(tm, center=False)


def _prism_on_ground(radius: float, height: float, sections: int):
    """N-угольная колонна (approximated via trimesh.creation.cylinder(sections=N)), основание
    в Z=0. sections=3 — треугольная колонна, sections=6 — шестигранная, sections>=32 — круглая
    (для проверки отбраковки)."""
    tm = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    tm.apply_translation([0.0, 0.0, height / 2.0])
    return from_trimesh(tm, center=False)


def test_box_height_and_footprint_close_to_true_dims():
    ex, ey, ez = 300.0, 200.0, 150.0
    mesh = _box_on_ground(ex, ey, ez)

    result = fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG, _ALL_ANGLES_DISTANCES, _RESOLUTION_PX,
    )

    assert result is not None
    assert result.corners == 4
    assert abs(result.height - ez) / ez < 0.02

    xs = [p[0] for p in result.footprint_xy]
    ys = [p[1] for p in result.footprint_xy]
    assert abs((max(xs) - min(xs)) - ex) / ex < 0.02
    assert abs((max(ys) - min(ys)) - ey) / ey < 0.02


def test_triangular_column_matches_true_height_and_corner_count():
    """Ответ на вопрос "а если у нас вертикальный треугольный столб" — метод не завязан на
    четыре угла коробки, cv2.approxPolyDP находит ровно 3 угла контура сверху для треугольной
    колонны, и та же триангуляция высоты сходится к истинной высоте."""
    radius, height = 120.0, 180.0
    mesh = _prism_on_ground(radius, height, sections=3)

    result = fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG, _ALL_ANGLES_DISTANCES, _RESOLUTION_PX,
    )

    assert result is not None
    assert result.corners == 3
    assert abs(result.height - height) / height < 0.02


def test_hexagonal_column_generalizes_beyond_three_and_four_corners():
    radius, height = 120.0, 180.0
    mesh = _prism_on_ground(radius, height, sections=6)

    result = fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG, _ALL_ANGLES_DISTANCES, _RESOLUTION_PX,
    )

    assert result is not None
    assert result.corners == 6
    assert abs(result.height - height) / height < 0.02


def test_round_cylinder_is_fit_as_circle_not_rejected():
    """Круглый профиль (много сторон у cylinder, постоянное сечение по высоте — настоящий
    вертикальный цилиндр) — окружность как предельный случай N-угольника при N->бесконечность
    (см. докстринг модуля, обсуждение с пользователем 2026-07: "любой N-угольник, если есть
    вертикальные рёбра, должен работать", включая N->бесконечность). Раньше отбраковывался той
    же проверкой, что и купольные/конические объекты — теперь должен подгоняться точно, как и
    N-угольная колонна."""
    radius, height = 100.0, 200.0
    mesh = _prism_on_ground(radius=radius, height=height, sections=64)

    result = fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG, _ALL_ANGLES_DISTANCES, _RESOLUTION_PX,
    )

    assert result is not None
    assert result.is_circle
    assert abs(result.height - height) / height < 0.02
    radii = [np.hypot(x, y) for x, y in result.footprint_xy]
    assert abs(np.mean(radii) - radius) / radius < 0.02


def test_cone_is_rejected_not_treated_as_prism_or_cylinder():
    """Конус — заведомо НЕ вертикальная призма/цилиндр (сечение сужается к вершине); отбраковка
    должна сработать независимо от круглости самого основания (сама high-level причина, ради
    которой вообще нужна отбраковка по классу формы). Круглая ветвь (добавлена для цилиндра, см.
    test_round_cylinder_is_fit_as_circle_not_rejected) заставила контур конуса впервые дойти до
    поиска высоты — найдена и закрыта реальная дыра: апекс конуса направлен К камере, поэтому
    контур сверху (граница основания) математически консистентен с боковыми силуэтами уже при
    h≈0 (тонкий диск у широкого основания, `_excess_pixels`≈0 — не вылезает никуда), но НЕ
    объясняет сам конус выше основания — ловится проверкой `min_relative_coverage`
    (`_covered_pixels`), а не одной только `max_relative_excess`."""
    mesh = from_trimesh(trimesh.creation.cone(radius=100.0, height=200.0), center=False)

    result = fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG, _ALL_ANGLES_DISTANCES, _RESOLUTION_PX,
    )

    assert result is None


def test_cylinder_at_smaller_pixel_footprint_is_not_falsely_rejected():
    """Регрессия, найденная пользователем на Пуфике: тот же цилиндр, что и в
    test_round_cylinder_is_fit_as_circle_not_rejected, но камера сильно отодвинута (2800мм) при
    проде-подобном разрешении (1024px, а не тестовые 2048px по умолчанию в этом файле, см.
    `_RESOLUTION_PX` вверху) — объект занимает заметно меньше пикселей в кадре. С ФИКСИРОВАННЫМ
    порогом `max_relative_excess=0.02` это даёт ложный отказ (растровый шум границы даёт
    примерно ПОСТОЯННОЕ число шумных пикселей независимо от того, сколько пикселей занимает сам
    объект — доля от площади растёт, когда объект мельче в кадре; см. докстринг
    fit_vertical_prism/`excess_pixel_margin`, docs/camera_dims_v2_investigation.md — там же
    реальный кейс на Пуфике, camera_distance=2000мм, тот же механизм, но с меньшим запасом
    из-за неидеальной геометрии реального меша). Порог должен адаптироваться под то, сколько
    пикселей объект реально занимает, а не быть одной константой на все масштабы. Проверено:
    при `excess_pixel_margin=0` (старое поведение) этот же конкретный кейс отбраковывается —
    сама возможность подобрать масштаб, различающий старое/новое поведение на синтетике,
    подтверждает, что тест ловит именно этот механизм, а не что-то ещё."""
    radius, height = 100.0, 200.0
    mesh = _prism_on_ground(radius=radius, height=height, sections=64)

    assert fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG,
        camera_distances={a: 2800.0 for a in _VIEW_ANGLES}, resolution_px=1024,
        excess_pixel_margin=0.0,
    ) is None, "калибровка теста устарела — этот кейс больше не отбраковывается при старом пороге"

    result = fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG,
        camera_distances={a: 2800.0 for a in _VIEW_ANGLES}, resolution_px=1024,
    )

    assert result is not None
    assert result.is_circle
    assert abs(result.height - height) / height < 0.1


def test_small_box_height_not_undershot():
    """Регрессия, найденная пользователем на ЛанчБоксе ("верх призмы не должен быть ниже верха
    контура"): на объекте, маленьком в кадре (80x60x40мм при проде-подобном разрешении 1024px),
    `_excess_pixels`-only поиск сходился к высоте ~30мм при истинных 40мм (~25% ошибка) — минимум
    невязки был на ЛОЖНОЙ высоте, потому что заниженный кандидат не дотягивался до верха объекта,
    не вылезая при этом никуда (`excess`≈0 не видит недооценку, только переоценку). Исправлено
    переходом на симметрическую разность `excess+uncovered` в `_fit_height` (см. докстринг
    `_excess_and_uncovered_pixels`) — ошибка упала до ~6%."""
    ex, ey, ez = 80.0, 60.0, 40.0
    mesh = _box_on_ground(ex, ey, ez)

    result = fit_vertical_prism(
        mesh, _VIEW_ANGLES, _FOV_DEG,
        camera_distances={a: 2000.0 for a in _VIEW_ANGLES}, resolution_px=1024,
    )

    assert result is not None
    assert abs(result.height - ez) / ez < 0.1, (
        f"высота призмы ({result.height:.1f}) занижена относительно истинной ({ez}) сильнее, "
        "чем допускает растровый шум на этом масштабе"
    )
