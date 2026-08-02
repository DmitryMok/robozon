"""Unit-тесты логики sim/ (без Webots): python3 -m pytest tests/ или python3 tests/test_sim.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.classifier import KnownTypeClassifier
from sim.config import load_layout, load_objects
from sim.router import (
    CATEGORY_TO_ZONE, DELIVERED, DISPATCHED, MISSED, ON_BELT, Router, TrackedObject,
)


def make_router(**kw):
    defaults = dict(
        trigger_x=5.0,
        zones={
            "B": {"enter": ("x", ">=", 8.0), "lateral": ("y", -0.4, 0.4)},
            "C": {"enter": ("x", ">=", 7.0), "lateral": ("y", 1.0, 2.4)},
            "D": {"enter": ("x", ">=", 7.0), "lateral": ("y", -2.4, -1.0)},
        },
        release_timeout_s=6.0,
    )
    defaults.update(kw)
    return Router(**defaults)


def test_config_loads_and_categories_valid():
    objects = load_objects()
    layout = load_layout()
    assert len(objects) == 11
    for name, cfg in objects.items():
        assert cfg["category"] in CATEGORY_TO_ZONE, name
    assert set(layout["zones"]) == {"B", "C", "D"}
    assert set(layout["diverters"]["positions"]) == {"B", "D"}


def test_known_type_classifier():
    clf = KnownTypeClassifier(load_objects())
    assert clf.classify({"type": "box_small"}) == "ok"
    assert clf.classify({"type": "box_large"}) == "oversize"
    assert clf.classify({"type": "bottle"}) == "round"


def test_router_dispatch_and_delivery():
    router = make_router()
    obj = TrackedObject(uid=1, type="bottle", category="round", zone="D")
    router.add(obj)

    # до линии упреждения — команды нет
    assert router.step({1: (3.0, 0, 0.7)}, now=1.0) == []
    assert obj.state == ON_BELT
    # пересёк линию — команда на зону D
    assert router.step({1: (5.1, 0, 0.7)}, now=2.0) == ["D"]
    assert obj.state == DISPATCHED
    # повторной команды нет, пока объект в пути
    assert router.step({1: (6.5, -0.5, 0.5)}, now=3.0) == []
    # вошёл в bbox зоны D — ещё не доставлен (нужно удержание confirm_time_s)
    router.step({1: (7.8, -1.7, 0.3)}, now=4.0)
    assert obj.state == DISPATCHED
    # удержался в зоне — доставлен
    router.step({1: (7.8, -1.7, 0.25)}, now=4.7)
    assert obj.state == DELIVERED
    assert obj.result_zone == "D"
    assert router.stats()["delivered_correct"] == 1


def test_router_flythrough_not_delivered():
    router = make_router()
    obj = TrackedObject(uid=1, type="bottle", category="round", zone="D")
    router.add(obj)
    router.step({1: (5.1, 0, 0.7)}, now=0.0)
    # пролетел сквозь зону D и вылетел — доставка не засчитана
    router.step({1: (7.8, -1.7, 0.4)}, now=1.0)
    router.step({1: (9.5, -2.6, 0.1)}, now=1.3)
    assert obj.state == DISPATCHED
    router.step({1: (9.5, -2.6, 0.1)}, now=9.0)
    assert obj.state == MISSED


def test_router_same_zone_serializes():
    """Один лоток — одна очередь: второй объект той же зоны ждёт первого."""
    router = make_router()
    first = TrackedObject(uid=1, type="bottle", category="round", zone="D")
    second = TrackedObject(uid=2, type="cylinder", category="round", zone="D")
    router.add(first)
    router.add(second)

    assert router.step({1: (5.5, 0, 0.7), 2: (2.0, 0, 0.7)}, now=1.0) == ["D"]
    # второй пересёк линию, но лоток D занят первым — команды нет
    assert router.step({1: (6.9, -1.7, 0.4), 2: (5.2, 0, 0.7)}, now=2.0) == []
    # первый вошёл в зону D — лоток освобождается сразу (объект уже физически
    # проехал его, ждать полного подтверждения незачем — иначе лоток лишнее
    # время может перехватить чужой объект следом по той же ленте), второй
    # (уже за линией упреждения) тут же получает свою команду на этом такте
    assert router.step({1: (7.8, -1.7, 0.3), 2: (5.3, 0, 0.7)}, now=3.0) == ["D"]
    assert first.state == DISPATCHED    # ещё не доставлен — только вошёл в зону
    assert second.state == DISPATCHED
    # первый удержался в зоне достаточно — доставлен
    assert router.step({1: (7.8, -1.7, 0.3), 2: (6.9, -1.2, 0.7)}, now=3.7) == []
    assert first.state == DELIVERED


def test_router_different_zones_run_concurrently():
    """Разные зоны — разные лотки: друг друга не блокируют (иначе при частом
    спавне объект соседней зоны успевает проехать точку срабатывания без
    команды и уезжает по умолчанию в C)."""
    router = make_router()
    first = TrackedObject(uid=1, type="bottle", category="round", zone="D")
    second = TrackedObject(uid=2, type="box_small", category="ok", zone="B")
    router.add(first)
    router.add(second)

    result = router.step({1: (5.5, -1.7, 0.7), 2: (6.0, 0, 0.7)}, now=1.0)
    assert set(result) == {"D", "B"}
    assert first.state == DISPATCHED
    assert second.state == DISPATCHED


def test_router_timeout_marks_missed():
    router = make_router()
    obj = TrackedObject(uid=1, type="pen", category="ok", zone="B")
    router.add(obj)
    assert router.step({1: (5.1, 0, 0.7)}, now=0.0) == ["B"]
    # объект застрял вне зон — по таймауту помечается промахом, лоток свободен
    router.step({1: (6.0, 0.6, 0.1)}, now=7.0)
    assert obj.state == MISSED
    nxt = TrackedObject(uid=2, type="bottle", category="round", zone="D")
    router.add(nxt)
    assert router.step({1: (6.0, 0.6, 0.1), 2: (5.3, 0, 0.7)}, now=8.0) == ["D"]


def main() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                print(f"FAIL  {name}: {exc}")
                fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
