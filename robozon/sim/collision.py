"""Форма коллизии товара — из кэша convex decomposition (tools/prepare_collision.py),
не ручной примитив. Кэш — assets/collision/<type>.json, по одному на STL из
assets/meshes (та же система координат, что у видимого меша: prepare_stl.py уже
отцентровал/сориентировал объект, origin у дна).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLLISION_DIR = ROOT / "assets" / "collision"


def load_parts(obj_type: str) -> list[dict]:
    path = COLLISION_DIR / f"{obj_type}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["parts"]


def collision_bounding_node(obj_type: str) -> str:
    """`boundingObject` — Group из выпуклых кусков (IndexedFaceSet).

    Каждый кусок геометрически выпуклый (гарантия convex decomposition) — это
    единственное, что нужно ODE/Webots для устойчивой динамики составного
    boundingObject (в отличие от полного конкав-меша). Проверено вручную
    (минимальный тестовый .wbt), что Webots считает mass/centerOfMass/
    inertiaMatrix из такой Group автоматически по Physics.density без
    предупреждений — явно считать инерцию самим не нужно.
    """
    parts = load_parts(obj_type)
    pieces = []
    for part in parts:
        points = " ".join(f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in part["points"])
        idx = " ".join(f"{a} {b} {c} -1" for a, b, c in part["faces"])
        pieces.append(
            f"IndexedFaceSet {{ coord Coordinate {{ point [ {points} ] }} "
            f"coordIndex [ {idx} ] }}"
        )
    children = "\n      ".join(pieces)
    return f"Group {{ children [\n      {children}\n    ] }}"
