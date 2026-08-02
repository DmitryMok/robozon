#!/usr/bin/env python3
"""Подготовка STL-моделей для симуляции.

Читает STL из assets/stl (миллиметры, произвольное положение), выполняет:
  * пересчёт мм -> м;
  * центрирование: центр XY в нуле, минимум Z на нуле (объект «стоит» на origin);
  * запись бинарного STL в assets/meshes;
  * отчёт: габариты, коэффициент круга в сечении (r_впис/R_опис по XY-проекции),
    предлагаемая категория по правилам постановки.

Запуск:  python3 tools/prepare_stl.py
"""
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "stl"
DST = ROOT / "assets" / "meshes"

# Ограничения основного сортировщика, мм (из постановки задачи)
MAX_DIMS = (450.0, 320.0, 320.0)
MIN_DIMS = (10.0, 10.0, 10.0)
ROUNDNESS_K = 0.8  # порог r_впис/R_опис по ТЗ (Постановка задачи.pdf, "Правила классификации")


def read_stl(path: Path) -> np.ndarray:
    """Возвращает массив треугольников (N, 3, 3)."""
    data = path.read_bytes()
    # ASCII STL начинается со слова "solid" и не содержит бинарного счётчика
    if data[:5].lower() == b"solid" and b"facet" in data[:1000]:
        return _read_ascii(data)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
    tri = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
    return tri.astype(np.float64)


def _read_ascii(data: bytes) -> np.ndarray:
    pts = []
    for line in data.decode("ascii", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            pts.append([float(v) for v in line.split()[1:4]])
    return np.array(pts).reshape(-1, 3, 3)


def write_stl(path: Path, tri: np.ndarray) -> None:
    n = len(tri)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    normals = np.cross(b - a, c - a)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lens, out=np.zeros_like(normals), where=lens > 0)
    rec = np.zeros((n, 50), dtype=np.uint8)
    rec[:, 0:12] = normals.astype("<f4").view(np.uint8).reshape(n, 12)
    rec[:, 12:48] = tri.astype("<f4").view(np.uint8).reshape(n, 36)
    path.write_bytes(b"\0" * 80 + struct.pack("<I", n) + rec.tobytes())


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Выпуклая оболочка (алгоритм Эндрю), points (N,2) -> вершины CCW."""
    pts = np.unique(points.round(6), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


# Переориентация моделей в естественное «лежачее» положение (поворот вокруг X или Y на 90°)
REORIENT = {
    "plate": ("x", np.pi / 2),   # тарелка стоит на ребре -> положить плашмя
    "pen": ("y", np.pi / 2),     # ручка стоит вертикально -> положить вдоль X
    "pouf": ("x", np.pi / 2),    # пуфик лежит на боку («колесо») -> поставить на основание
}


def rotate(tri: np.ndarray, axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]
    return tri @ m.T


def roundness(tri: np.ndarray) -> float:
    """Максимум r_впис/R_опис по трём осевым проекциям (XY, XZ, YZ).

    Признак «круг в сечении» не зависит от ориентации модели, поэтому берём
    наихудшую (самую круглую) из трёх проекций.
    """
    pts3 = tri.reshape(-1, 3)
    return max(
        _roundness_2d(pts3[:, [0, 1]]),
        _roundness_2d(pts3[:, [0, 2]]),
        _roundness_2d(pts3[:, [1, 2]]),
    )


def _roundness_2d(pts: np.ndarray) -> float:
    hull = convex_hull_2d(pts)
    if len(hull) < 3:
        return 0.0
    centroid = hull.mean(axis=0)
    r_out = np.max(np.linalg.norm(hull - centroid, axis=1))
    # r_впис: минимальное расстояние от центроида до рёбер оболочки
    r_in = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ab = b - a
        t = np.clip(np.dot(centroid - a, ab) / np.dot(ab, ab), 0, 1)
        r_in = min(r_in, np.linalg.norm(a + t * ab - centroid))
    return float(r_in / r_out) if r_out > 0 else 0.0


def classify(dims_mm, k: float) -> str:
    """Правила постановки: сперва габариты, затем круг в сечении."""
    d = sorted(dims_mm, reverse=True)
    mx = sorted(MAX_DIMS, reverse=True)
    mn = sorted(MIN_DIMS, reverse=True)
    if any(d[i] >= mx[i] for i in range(3)) or any(d[i] <= mn[i] for i in range(3)):
        return "oversize"   # категория C
    if k > ROUNDNESS_K:
        return "round"      # категория D
    return "ok"             # категория B


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(SRC.glob("*.stl")):
        tri = read_stl(path) * 0.001  # мм -> м
        if path.stem in REORIENT:
            tri = rotate(tri, *REORIENT[path.stem])
        lo = tri.reshape(-1, 3).min(axis=0)
        hi = tri.reshape(-1, 3).max(axis=0)
        center = (lo + hi) / 2
        tri -= np.array([center[0], center[1], lo[2]])
        write_stl(DST / path.name, tri)
        dims_m = hi - lo
        k = roundness(tri)
        cat = classify(dims_m * 1000, k)
        rows.append((path.stem, dims_m, k, cat, len(tri)))

    print(f"{'name':<12}{'X м':>8}{'Y м':>8}{'Z м':>8}{'r/R':>7}{'категория':>11}{'tri':>9}")
    for name, d, k, cat, n in rows:
        print(f"{name:<12}{d[0]:>8.3f}{d[1]:>8.3f}{d[2]:>8.3f}{k:>7.2f}{cat:>11}{n:>9}")


if __name__ == "__main__":
    sys.exit(main())
