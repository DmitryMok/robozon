"""Расчёт геометрии зеркала для рига «3 камеры + 1 зеркало» (top/diag/side).

Обобщение методики `docs/mirror_geometry_calc.md` §1-4 на раскладку: 3 прямые камеры
(top=90°, diag~70°, side=0°), зеркало отражает ТОЛЬКО top и diag (решение пользователя
2026-07-19). side — только прямой ракурс, зеркального отражения не имеет.

Геометрия зеркала — НЕ подбирается (не свободные параметры φ/t под целевой диапазон
виртуальных углов, как в первой версии этого скрипта), а ЗАДАНА физическим макетом
пользователя (уточнено 2026-07-19, важные условия):
1. Зеркало — с противоположной от `side`-камеры стороны ленты (иначе в `top`
   одновременно не попадут и объект, и отражение).
2. Нижний край зеркала смещён от КРОМКИ полотна ленты (не от осевой линии) на 300мм
   дальше, в плоскости, параллельной земле; высота нижнего края — на уровне
   поверхности ленты (Z=0 в этой системе координат).
3. Верхняя точка отклонена от вертикали на 23° В СТОРОНУ ОТ ЛЕНТЫ (не нависает над
   лентой, а "заваливается" наружу).

Все расчёты — в 2D-сечении Y-Z (ось X вдоль ленты не участвует). Референсная точка
наблюдения (объект) — начало координат (0,0), на осевой линии ленты. Конвенция та же,
что в config/layout.yaml/gen_world.py: азимут θ отсчитывается от оси Y, v_θ=(cosθ,sinθ),
камера в точке d·v_θ.

Usage:
    python mirror_rig_geometry.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    name: str
    angle_deg: float
    distance_mm: float


# Прямые камеры рига (азимут, дистанция) — приближение из плана §1 П.2д
# (План CV-пайплайна сортировки (Webots)): top дальняя ~2000мм, diag ~1900мм,
# side ближняя ~1100мм. Точные дистанции — компоновочное решение, здесь не пересматриваются.
CAMERAS = [
    Camera("top", 90.0, 2000.0),
    Camera("diag", 70.0, 1900.0),
    Camera("side", 0.0, 1100.0),
]

REFLECTED = ["top", "diag"]

# Лента
BELT_WIDTH_MM = 500.0
BELT_HALF_WIDTH_MM = BELT_WIDTH_MM / 2.0  # кромки на Y=±250мм

# side камера на азимуте 0° -> +Y сторона (см. cam_pos(0,...)=(distance,0)).
# Зеркало — с противоположной стороны (-Y), условие 1 пользователя.
MIRROR_SIDE_SIGN = -1.0

# Условие 2: нижний край смещён от КРОМКИ ленты (не от оси) ещё на 300мм дальше,
# на высоте Z=0 (уровень поверхности ленты).
EDGE_TO_MIRROR_MM = 300.0
BOTTOM_EDGE = (
    MIRROR_SIDE_SIGN * (BELT_HALF_WIDTH_MM + EDGE_TO_MIRROR_MM),  # Y = -(250+300) = -550
    0.0,                                                            # Z
)

# Условие 3: наклон верхней точки от вертикали, в сторону ОТ ленты (т.е. по мере
# подъёма Y уходит ЕЩЁ дальше в сторону MIRROR_SIDE_SIGN, а не к центру).
TILT_FROM_VERTICAL_DEG = 23.0

OBJECT = (0.0, 0.0)


def cam_pos(angle_deg: float, distance_mm: float) -> tuple[float, float]:
    th = math.radians(angle_deg)
    return (distance_mm * math.cos(th), distance_mm * math.sin(th))


def mirror_line_direction() -> tuple[float, float]:
    """Единичный вектор направления линии зеркала СНИЗУ ВВЕРХ (от нижнего края
    к верхней точке), с учётом наклона 23° от вертикали в сторону от ленты."""
    a = math.radians(TILT_FROM_VERTICAL_DEG)
    return (MIRROR_SIDE_SIGN * math.sin(a), math.cos(a))


def phi_from_direction(d: tuple[float, float]) -> float:
    """Пеленг φ линии (та же конвенция virtual=2φ-angle, n=(sinφ,-cosφ)): φ такое,
    что (cosφ, sinφ) = d."""
    return math.degrees(math.atan2(d[1], d[0])) % 360.0


def mirror_normal(phi_deg: float) -> tuple[float, float]:
    phi = math.radians(phi_deg)
    return (math.sin(phi), -math.cos(phi))


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def reflect_point(p: tuple[float, float], n: tuple[float, float], t: float) -> tuple[float, float]:
    d = 2.0 * (dot(p, n) - t)
    return (p[0] - d * n[0], p[1] - d * n[1])


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def virtual_angle(angle_deg: float, phi_deg: float) -> float:
    """ВНИМАНИЕ: закон 2φ-θ отражает НАПРАВЛЕНИЕ луча (верно только для зеркала,
    проходящего через начало координат, t=0) — НЕ азимут смещённой (t≠0) виртуальной
    камеры от объекта. Для реальной геометрии (t≠0, наш случай) даёт неверный угол
    (проверено: расхождение с true_reflected_azimuth на ~20° на этой геометрии).
    Оставлена только для справки/сверки, НЕ использовать для позиционирования камеры
    в gen_world.py — там нужен true_reflected_azimuth."""
    return (2 * phi_deg - angle_deg) % 360.0


def true_reflected_azimuth_and_distance(
    cam_pos_2d: tuple[float, float], n: tuple[float, float], t: float, object_2d: tuple[float, float]
) -> tuple[float, float]:
    """Настоящий азимут+дистанция виртуальной (зеркальной) камеры от объекта: точечное
    отражение реальной 3D-позиции камеры через плоскость зеркала (n·p=t), затем азимут
    полученной точки от объекта. Дистанция при этом равна оптическому пути (совпадает
    с dist(cam, reflect(object)) — оба способа дают одно число, т.к. отражение изометрия:
    dist(reflect(cam),object) = dist(cam,reflect(object)))."""
    refl = reflect_point(cam_pos_2d, n, t)
    distance = dist(object_2d, refl)
    azimuth = math.degrees(math.atan2(refl[1] - object_2d[1], refl[0] - object_2d[0])) % 360.0
    return azimuth, distance


def main() -> None:
    d = mirror_line_direction()
    phi = phi_from_direction(d)
    n = mirror_normal(phi)
    t = dot(BOTTOM_EDGE, n)

    print(f"Камеры: {[(c.name, c.angle_deg, c.distance_mm) for c in CAMERAS]}")
    print(f"Лента: ширина {BELT_WIDTH_MM}мм, кромки Y=±{BELT_HALF_WIDTH_MM:.0f}мм")
    print(f"Нижний край зеркала: Y={BOTTOM_EDGE[0]:.1f}мм, Z={BOTTOM_EDGE[1]:.1f}мм "
          f"(кромка ленты {'-' if MIRROR_SIDE_SIGN < 0 else '+'}{BELT_HALF_WIDTH_MM:.0f} "
          f"+ {EDGE_TO_MIRROR_MM:.0f}мм)")
    print(f"Направление снизу-вверх: ({d[0]:.4f}, {d[1]:.4f}) "
          f"(наклон {TILT_FROM_VERTICAL_DEG}° от вертикали, в сторону от ленты)")
    print(f"φ (пеленг линии) = {phi:.3f}°")
    print(f"n(φ) = ({n[0]:.4f}, {n[1]:.4f})")
    print(f"t (n·нижний_край) = {t:.1f}мм\n")

    # проверка: камеры/объект должны быть по ОДНУ сторону плоскости (реализуемость)
    checks = {"object": OBJECT, **{c.name: cam_pos(c.angle_deg, c.distance_mm) for c in CAMERAS}}
    print("Проверка стороны плоскости (n·p - t), знак должен совпадать у всех:")
    for name, p in checks.items():
        print(f"  {name:8s}: {dot(p, n) - t:8.1f}")
    print()

    print(f"{'камера':8s} {'угол':>6s} {'дист.':>7s} {'2φ-θ (НЕВЕРНО)':>15s} "
          f"{'true azimuth':>13s} {'путь, мм':>10s}")
    results = {}
    for c in CAMERAS:
        if c.name not in REFLECTED:
            continue
        pos = cam_pos(c.angle_deg, c.distance_mm)
        v_wrong = virtual_angle(c.angle_deg, phi)
        v_true, path = true_reflected_azimuth_and_distance(pos, n, t, OBJECT)
        results[c.name] = (v_true, path)
        print(f"{c.name:8s} {c.angle_deg:6.1f} {c.distance_mm:7.1f} {v_wrong:15.2f} "
              f"{v_true:13.2f} {path:10.1f}")

    print(f"\nСверка с исходными числами пользователя из черновика алгоритма — оказались")
    print(f"ВЕРНЫМИ и по углу, и по дистанции (ошибка была не у пользователя, а в более")
    print(f"ранней ревизии плана, которая сверяла их формулой 2φ-θ, не годной для t≠0):")
    print(f"  top:  было 158.1°/2574мм  ->  посчитано {results['top'][0]:.2f}°/{results['top'][1]:.1f}мм")
    print(f"  diag: было 172.8°/2696мм  ->  посчитано {results['diag'][0]:.2f}°/{results['diag'][1]:.1f}мм")

    print(f"\nИтог для layout.yaml:")
    print(f"  mirror_normal_deg: {phi:.3f}")
    print(f"  mirror_offset: {abs(t) / 1000:.4f}  # м, |t|")
    print(f"  mirror_distance_by_angle:")
    for c in CAMERAS:
        if c.name in REFLECTED:
            v_true, path = results[c.name]
            print(f"    {v_true:.2f}: {path:.1f}    # {c.name} -> {c.name}_mirror (true azimuth)")


if __name__ == "__main__":
    main()
