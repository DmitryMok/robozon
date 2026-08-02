"""Маршрутизация: категория -> зона, контроль доставки.

Router не знает про Webots: ему сообщают положение объектов, он решает,
куда и когда поворачивать лоток, и фиксирует событие доставки.
Управляющее воздействие возвращается вызывающему коду (контроллеру).
"""
from dataclasses import dataclass, field

CATEGORY_TO_ZONE = {"ok": "B", "oversize": "C", "round": "D"}

# Состояния объекта в потоке
ON_BELT = "on_belt"          # едет по подающему конвейеру
DISPATCHED = "dispatched"    # лоток назначен и повёрнут под этот объект
DELIVERED = "delivered"      # зафиксирован в целевой зоне
MISSED = "missed"            # не попал в зону за отведённое время


@dataclass
class TrackedObject:
    uid: int
    type: str
    category: str
    zone: str                # целевая зона (B/C/D)
    state: str = ON_BELT
    spawn_time: float = 0.0
    dispatch_time: float = 0.0
    result_zone: str = ""    # фактическая зона доставки
    _zone_enter: tuple | None = None   # (зона, время входа) — для подтверждения


@dataclass
class Router:
    """Логика поворотных лотков.

    Лотки B и D — независимые физические устройства: каждый обслуживает
    один объект за раз (FIFO по ходу ленты для своей зоны), но объекты
    РАЗНЫХ зон обслуживаются одновременно и не блокируют друг друга — иначе
    при частом спавне объект, ожидающий позади чужой (другой зоны) занятости,
    успевает физически проехать точку срабатывания без команды и уезжает по
    умолчанию в C. Зона C — прямой проезд без лотка, никого не блокирует и
    сама никогда не занята.

    `trigger_x` — общий порог (fallback). `trigger_x_per_zone` — порог под
    конкретную зону (position.x ≥ порог → развернуть свой шибер). Per-zone
    нужен: шибер B (x=6.45) при общем раннем trigger_x=4.50 разворачивался
    задолго до прихода объекта и перехватывал чужие C-объекты, ехавшие
    следом. Если per-zone задан — используется он, иначе общий trigger_x.
    """
    trigger_x: float
    # зона -> {"enter": (ось, ">="/"<=", порог), "lateral": (ось, lo, hi)}
    # Доставка засчитывается по факту пересечения стенки бокса, прилегающей
    # к конвейеру (порог по оси входа) + попадания в её ширину (lateral) —
    # без проверки высоты: иначе штабель товаров в одном боксе не даёт
    # верхним засчитаться, хотя они физически уже на месте.
    zones: dict
    release_timeout_s: float = 6.0
    confirm_time_s: float = 0.6  # объект должен удержаться в зоне (не пролёт)
    trigger_x_per_zone: dict = field(default_factory=dict)   # зона -> порог
    # зона -> (x шибера, радиус опасной зоны вокруг него): шибер этой зоны
    # НЕ разворачивается, пока в радиусе есть объект ДРУГОЙ зоны (иначе при
    # частом спавне шибер D, разворачиваясь для своего D-объекта, перехватывает
    # B/C-объект, едущий следом и физически проезжающий точку D). Если None —
    # защита выключена.
    divert_positions: dict = field(default_factory=dict)   # зона -> x шибера
    divert_safe_radius: float = 0.0                          # радиус опасной зоны
    # зона -> (ось, ">="/"<=", порог) — РАННИЙ, отдельный от zones[], критерий
    # "объект уже закоммитился в свою зону" для освобождения лотка (_busy).
    # По умолчанию (зона отсутствует в словаре) освобождение идёт как раньше
    # — по факту входа в ПОЛНЫЙ zones[]-бокс. Нужен для D: щит держится
    # развёрнутым, пока D-объект не пройдёт весь путь до своего zones['D']
    # (y<=-0.33) — это даёт длинное окно (~1-1.5с), в которое щит физически
    # перекрывает ленту и может задеть B-объект, едущий следом (см. заметку
    # задачи "Тайминг шиберов B/D"). Более ранний порог (объект уже явно
    # пошёл в сторону D, но ещё не долетел до бокса) убирает лоток раньше —
    # объект всё равно долетит по инерции/наклону, подтверждение DELIVERED
    # по-прежнему считается по полному zones[] независимо от этого.
    release_enter: dict = field(default_factory=dict)
    objects: dict = field(default_factory=dict)   # uid -> TrackedObject
    _busy: dict = field(default_factory=dict)      # зона (B/D) -> uid, занявший её лоток

    def add(self, obj: TrackedObject) -> None:
        self.objects[obj.uid] = obj

    def _trigger_for(self, zone: str) -> float:
        return self.trigger_x_per_zone.get(zone, self.trigger_x)

    def _other_zone_object_near_divert(self, zone: str, positions: dict,
                                        obj_uid: int, on_belt_uids: set) -> bool:
        """Есть ли объект ДРУГОЙ зоны, который БЛИЖЕ к шиберу `zone`, чем
        наш объект этой зоны (obj_uid)? Если да — наш объект не ближайший к
        шиберу, зона не разворачивается (иначе перехватит чужой объект,
        который физически ещё не проехал сам щит).

        Логика: при частом спавне на ленте несколько объектов разных зон.
        Шибер разворачивается только когда СВОЙ объект — ближайший к нему
        (среди всех ON_BELT объектов). Если чужой объект ближе к шиберу — он
        дойдёт до него первым, шибер не должен разворачиваться (иначе
        перехватит чужой). Когда чужой объект проедет шибер (x > шибер.x +
        radius) — свой объект становится ближайшим, шибер разворачивается.

        Применяется к ОБОИМ щитам (D и B), не только к D: раньше защита была
        только для D с обоснованием "B — последний шибер, ему некого
        перехватывать" — но это неверно: C-объекты (без своего щита, прямой
        проезд) физически проезжают МИМО оси B по пути в C, и щит B,
        разворачиваясь для своего B-объекта, точно так же может задеть
        C-объект, едущий следом рядом (живой прогон: pouf(C, Ø489мм) едет
        перед cylinder(B) — щит B задевал пуфик, не давая ему проскочить).

        `on_belt_uids` — снимок ON_BELT-объектов на МОМЕНТ НАЧАЛА текущего
        такта (см. step()), а не текущее self.objects.values(). Это
        критично: `waiting` в step() обрабатывается от дальнего к ближнему,
        и объект дальней зоны может быть диспетчирован ПЕРВЫМ внутри того же
        вызова step() — его state станет DISPATCHED ещё до того, как очередь
        дойдёт до объекта ближней зоны. Проверка по живому self.objects
        тогда просто не увидит его (уже не ON_BELT) и не заблокирует —
        щиты срабатывают одновременно на одном и том же проезжающем объекте
        (найдено живым прогоном: cylinder(B)+helmet(D) с интервалом спавна
        1с, оба шибера разворачивались в один и тот же такт на цилиндре,
        толкая его в противоположные стороны — объект улетал мимо обеих
        зон). Снимок, зафиксированный ДО цикла диспетчеризации, не зависит
        от порядка обработки внутри такта.
        """
        if not self.divert_positions or self.divert_safe_radius <= 0:
            return False
        dx = self.divert_positions.get(zone)
        if dx is None:
            return False
        obj_x = positions.get(obj_uid, (0.0,))[0]
        for o in self.objects.values():
            if o.uid not in on_belt_uids or o.zone == zone or o.uid not in positions:
                continue
            ox = positions[o.uid][0]
            # чужой объект ближе к шиберу zone, чем наш объект, и ещё не проехал его
            if ox > obj_x and ox < dx + self.divert_safe_radius:
                return True
        return False

    def step(self, positions: dict, now: float) -> list[str]:
        """Один такт. positions: uid -> (x, y, z).

        Возвращает список зон, чьи лотки нужно развернуть на этом такте
        (по одной на каждый только что назначенный объект).
        """
        commands = []

        # 1. Фиксация доставки (с подтверждением удержания в зоне) и таймаутов,
        #    освобождение лотка занятой зоны. Лоток освобождаем сразу по факту
        #    входа объекта в какую-либо зону (а не по итоговому подтверждению
        #    confirm_time_s) — физически объект уже проехал сам лоток и от
        #    него больше не зависит; держать лоток развёрнутым дольше только
        #    расширяет окно, в которое он может перехватить чужой (не свой)
        #    объект, идущий следом по той же ленте (лоток D физически раньше
        #    лотка B — без этого B-объекты иногда утаскивались в D).
        for uid, obj in self.objects.items():
            if obj.state not in (ON_BELT, DISPATCHED) or uid not in positions:
                continue
            zone = self._zone_at(positions[uid])
            if zone:
                if obj._zone_enter is None or obj._zone_enter[0] != zone:
                    obj._zone_enter = (zone, now)
                    if self._busy.get(obj.zone) == uid:
                        self._busy[obj.zone] = None
                elif now - obj._zone_enter[1] >= self.confirm_time_s:
                    obj.state = DELIVERED
                    obj.result_zone = zone
            else:
                obj._zone_enter = None
                if (obj.state == DISPATCHED
                        and now - obj.dispatch_time > self.release_timeout_s):
                    obj.state = MISSED
                    if self._busy.get(obj.zone) == uid:
                        self._busy[obj.zone] = None

        # 1b. Досрочное освобождение лотка по release_enter (см. docstring
        #     поля) — НЕЗАВИСИМО от полного входа в zones[] выше. Не влияет
        #     на подтверждение DELIVERED (то по-прежнему только через блок 1
        #     и zones[]) — только снимает _busy раньше, чтобы сократить
        #     время, когда щит физически перекрывает ленту.
        for uid, obj in self.objects.items():
            if obj.state != DISPATCHED or uid not in positions:
                continue
            if self._busy.get(obj.zone) != uid:
                continue
            spec = self.release_enter.get(obj.zone)
            if spec is None:
                continue
            axis, op, threshold = spec
            coord = dict(zip("xyz", positions[uid]))[axis]
            committed = coord >= threshold if op == ">=" else coord <= threshold
            if committed:
                self._busy[obj.zone] = None

        # 2. Назначение: каждый объект, дошедший до линии упреждения — C сразу
        #    едет (лоток не нужен), B/D — только если свой лоток свободен.
        #    Порядок — от дальнего к ближнему по ленте, чтобы не перехватить
        #    лоток впереди идущего у того, что сзади (тот же лоток — одна
        #    очередь, FIFO по зоне). Порог срабатывания — per-zone, если задан.
        # Снимок ON_BELT-объектов НА НАЧАЛО такта — для _other_zone_object_near_divert
        # ниже. Нужен отдельно от live self.objects: цикл диспетчеризации сам
        # переводит объекты ON_BELT -> DISPATCHED по ходу обработки (в этом же
        # такте), а обрабатывается `waiting` от дальнего к ближнему — B-объект
        # (дальше по ленте) диспетчируется раньше D-объекта в том же вызове
        # step(). Проверка по живому state тогда перестаёт видеть уже
        # диспетчированный B-объект и не блокирует D — см. docstring
        # _other_zone_object_near_divert.
        on_belt_uids = {o.uid for o in self.objects.values() if o.state == ON_BELT}
        waiting = sorted(
            (o for o in self.objects.values()
             if o.state == ON_BELT and o.uid in positions
             and positions[o.uid][0] >= self._trigger_for(o.zone)),
            key=lambda o: -positions[o.uid][0],
        )
        for obj in waiting:
            if obj.zone != "C" and self._busy.get(obj.zone) is not None:
                continue
            # Защита от перехвата: не разворачивать шибер zone, если чужой
            # объект (другой зоны) ближе к шиберу, чем наш — иначе шибер D,
            # разворачиваясь для своего D-объекта, перехватит B/C-объект,
            # который дойдёт до D раньше.
            if obj.zone != "C" and self._other_zone_object_near_divert(obj.zone, positions, obj.uid, on_belt_uids):
                continue
            obj.state = DISPATCHED
            obj.dispatch_time = now
            if obj.zone != "C":
                self._busy[obj.zone] = obj.uid
                commands.append(obj.zone)
        return commands

    def _zone_at(self, pos) -> str | None:
        coord = dict(zip("xyz", pos))
        for zone, cfg in self.zones.items():
            axis, op, threshold = cfg["enter"]
            entered = coord[axis] >= threshold if op == ">=" else coord[axis] <= threshold
            if not entered:
                continue
            lat_axis, lo, hi = cfg["lateral"]
            if lo <= coord[lat_axis] <= hi:
                return zone
        return None

    def stats(self) -> dict:
        by_state: dict = {}
        for obj in self.objects.values():
            by_state[obj.state] = by_state.get(obj.state, 0) + 1
        correct = sum(1 for o in self.objects.values()
                      if o.state == DELIVERED and o.result_zone == o.zone)
        wrong = sum(1 for o in self.objects.values()
                    if o.state == DELIVERED and o.result_zone != o.zone)
        return {"by_state": by_state, "delivered_correct": correct,
                "delivered_wrong": wrong, "total": len(self.objects)}
