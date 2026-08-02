#!/usr/bin/env python3
"""Автоматическая форма коллизии по STL — convex decomposition (coacd).

Заменяет ручной подбор `bounding: {type: box/cylinder/sphere, ...}` в
config/objects.yaml (см. комментарии там же — например `cylinder` получил
box вместо cylinder, а `helmet` — sphere, ЧТОБЫ не застревали у щита; такая
подгонка не переносится на новые STL). Вместо примитива-заменителя —
несколько выпуклых кусков, посчитанных прямо из формы объекта: подходит для
любого STL без ручной классификации.

Источник — assets/meshes/*.stl (уже отцентрованы/переориентированы
tools/prepare_stl.py, та же система координат, что у видимого меша).
Результат кэшируется в assets/collision/<name>.json — coacd на сложных
мешах (шлем, ~50k треугольников) считает больше минуты, при каждом запуске
мира это было бы слишком медленно.

Запуск:
    python3 tools/prepare_collision.py                  # все типы, только устаревший кэш
    python3 tools/prepare_collision.py --only helmet,cylinder
    python3 tools/prepare_collision.py --force           # пересчитать всё
"""
import argparse
import json
import sys
import time
from pathlib import Path

import coacd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT / "assets" / "meshes"
OUT_DIR = ROOT / "assets" / "collision"

# Мешаем не более этого числа треугольников в coacd — decimation держит время
# на сложных STL в пределах ~1-2 минут (см. обсуждение задачи, шлем ~50k
# треугольников без decimation считался >5 минут на одной MCTS-итерации).
DECIMATE_TARGET_FACES = 1500

COACD_KWARGS = dict(
    threshold=0.08,
    max_convex_hull=8,
    preprocess_resolution=20,
    mcts_nodes=10,
    mcts_iterations=60,
    mcts_max_depth=2,
    max_ch_vertex=64,
)


def decompose(stl_path: Path) -> list[dict]:
    mesh = trimesh.load(stl_path, process=True)
    if len(mesh.faces) > DECIMATE_TARGET_FACES:
        mesh = mesh.simplify_quadric_decimation(face_count=DECIMATE_TARGET_FACES)
    cmesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(cmesh, **COACD_KWARGS)
    return [
        {"points": v.tolist(), "faces": f.tolist()}
        for v, f in parts
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="список типов через запятую (по умолчанию — все)")
    p.add_argument("--force", action="store_true", help="пересчитать даже свежий кэш")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stl_paths = sorted(MESH_DIR.glob("*.stl"))
    if args.only:
        wanted = set(args.only.split(","))
        stl_paths = [p for p in stl_paths if p.stem in wanted]

    print(f"{'name':<14}{'куски':>7}{'грани':>8}{'сек':>8}")
    for stl_path in stl_paths:
        name = stl_path.stem
        out_path = OUT_DIR / f"{name}.json"
        if not args.force and out_path.exists() and out_path.stat().st_mtime > stl_path.stat().st_mtime:
            print(f"{name:<14}{'(кэш)':>7}")
            continue
        t0 = time.time()
        parts = decompose(stl_path)
        elapsed = time.time() - t0
        out_path.write_text(json.dumps({"parts": parts}), encoding="utf-8")
        n_faces = sum(len(part["faces"]) for part in parts)
        print(f"{name:<14}{len(parts):>7}{n_faces:>8}{elapsed:>8.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
