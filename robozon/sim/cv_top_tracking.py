"""Top-локализация нескольких объектов в кадре + трекинг по uid (часть 1
CV-триггера момента, замена физики). Постановка — заметка задачи "CV-триггер
момента по top-локализации (замена физики, Фаза 4, CV-пайплайн Webots)".

Заменяет источник X для `sim.cv_moments.MomentScheduler.step` — вместо
`node.getPosition()[0]` (физика симулятора, недоступная в прод-системе)
используется положение объекта, восстановленное по кадру top-камеры
(HSV-субтракция + `sim.cv_grid_v2.unproject_top_to_belt`). Часть 2
(разделение объектов в `assemble_grid_v2` через `target_hint`) уже сделана в
исходной задаче "CV-триггер момента по top-локализации + разделение объектов
в кадре" — здесь она не трогается, только источник триггера.

Многообъектный кадр: `localize_top_components` возвращает ВСЕ компоненты
(не одну, как `cv_grid._downscale_bbox_fast`), `TopTracker` сопоставляет их
с уже отслеживаемыми uid по близости к предыдущей X. Лента однонаправленная
(FIFO по X, без отката, см. `cv_moments.MomentScheduler` docstring) — простое
"ближайший к прошлому X" достаточно, венгерский алгоритм избыточен.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.cv_grid import _DOWNSCALE_FACTOR, _downscale_components
from sim.cv_grid_v2 import CamGeom, unproject_top_to_belt


def localize_top_components(
    frame: np.ndarray, bg_frame: np.ndarray, bs: dict, top_geom: CamGeom,
    cv_rig_x: float, factor: int = _DOWNSCALE_FACTOR,
) -> list[tuple[float, float]]:
    """Все объекты в кадре top-камеры (HSV-субтракция, ds8) -> список
    (world_x, belt_y) в МИРОВЫХ координатах (belt_y — поперёк ленты, в СК
    рига), по убыванию ds8-площади. `unproject_top_to_belt` возвращает X в СК
    рига (0 = объект в `cv_rig_x`) — здесь сразу переводим в мировую X, т.к.
    `MomentScheduler`/`moment_thresholds` работают в мировых координатах."""
    components = _downscale_components(frame, bg_frame, bs, factor, v_threshold=bs.get("v_threshold"))
    out = []
    for (x, y, w, h), _area in components:
        cx, cy = x + w / 2.0, y + h / 2.0
        x_rig, y_rig = unproject_top_to_belt(cx, cy, top_geom)
        out.append((cv_rig_x + x_rig, y_rig))
    return out


@dataclass
class TopTracker:
    """Сопоставление компонент кадра top-камеры с uid по близости к
    предыдущей мировой X. Состояние — только CV-наблюдения (никогда
    `node.getPosition()`).

    Новые (ещё ни разу не сопоставленные) uid ждут в очереди `_pending` (FIFO
    в порядке появления в зоне CV — на однонаправленной ленте это тот же
    порядок, что и по убыванию ожидаемой X). Компоненты, не подошедшие ни
    одному уже отслеживаемому uid, разбираются этой очередью по убыванию X
    (самая правая непойманная компонента — самый старый ожидающий uid),
    без допуска на расстояние (объект появляется в кадре top в произвольном
    месте окна старта, а не строго у одной точки)."""
    max_match_distance_m: float
    _last_x: dict[int, float] = field(default_factory=dict)
    _pending: list[int] = field(default_factory=list)

    def add_pending(self, uid: int) -> None:
        """Зарегистрировать uid как ожидающий первого попадания в кадр top
        (вызывается при появлении объекта в зоне CV, до какого-либо
        CV-наблюдения — не использует физическую позицию)."""
        if uid not in self._pending and uid not in self._last_x:
            self._pending.append(uid)

    def update(self, components: list[tuple[float, float]]) -> dict[int, float]:
        """`components`: [(world_x, belt_y), ...] этого кадра (см.
        `localize_top_components`). Возвращает uid -> world_x для
        сопоставленных на этом тике (частичный словарь — не для всех
        известных uid обязательно находится компонента на каждом кадре)."""
        remaining = sorted(range(len(components)), key=lambda i: -components[i][0])
        matched: dict[int, float] = {}
        for uid in sorted(self._last_x, key=lambda u: -self._last_x[u]):
            ex = self._last_x[uid]
            best_pos, best_d = None, None
            for pos, i in enumerate(remaining):
                d = abs(components[i][0] - ex)
                if d <= self.max_match_distance_m and (best_d is None or d < best_d):
                    best_pos, best_d = pos, d
            if best_pos is not None:
                i = remaining.pop(best_pos)
                matched[uid] = components[i][0]
        while self._pending and remaining:
            uid = self._pending.pop(0)
            i = remaining.pop(0)   # remaining ещё по убыванию X — самая правая первому pending
            matched[uid] = components[i][0]
        self._last_x.update(matched)
        return matched

    def last_x(self, uid: int) -> float | None:
        return self._last_x.get(uid)

    def forget(self, uid: int) -> None:
        """Очистка состояния объекта, покинувшего зону CV — см.
        `cv_moments.MomentScheduler.forget` (тот же повод вызова)."""
        self._last_x.pop(uid, None)
        if uid in self._pending:
            self._pending.remove(uid)
