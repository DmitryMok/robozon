"""Поиск контрпримеров: брутфорс Roll/Pitch/Yaw для одного объекта.

Критерий "не цилиндр" — не пересчитывается геометрически на каждой ориентации (форма объекта
не меняется от того, как его повернули): это статичный признак `is_round_reference` в
`config/objects.yaml`, о котором этот модуль ничего не знает. Вызывающий код (CLI/GUI) должен
запускать поиск только для объектов с `is_round_reference=False` — тогда любая ориентация с
`k >= threshold` из результатов ниже уже является настоящим контрпримером.

Перебор распараллелен через ProcessPoolExecutor: на тяжёлых мешах (десятки тысяч треугольников,
например Шлем.stl) один расчёт занимает ~0.5с, полный перебор 360^3 с шагом 1° без
распараллеливания неприемлемо долог.
"""

from __future__ import annotations

import itertools
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from ..core.experiment import Orientation, evaluate_metrics_only
from ..geometry.mesh_io import Mesh


@dataclass
class CounterexampleHit:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    k: float


def _frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    v = start
    while v < stop:
        values.append(round(v, 6))
        v += step
    return values


def _evaluate_one(args: tuple) -> CounterexampleHit | None:
    mesh, view_angles_deg, resolution_px, roundness_threshold, roll, pitch, yaw = args
    orientation = Orientation(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
    roundness = evaluate_metrics_only(
        mesh,
        orientation,
        view_angles_deg,
        resolution_px=resolution_px,
        roundness_threshold=roundness_threshold,
    )
    if roundness is not None and roundness.passed:
        return CounterexampleHit(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, k=roundness.k)
    return None


def search_counterexamples(
    mesh: Mesh,
    view_angles_deg: list[float],
    step_deg: float,
    roundness_threshold: float = 0.8,
    resolution_px: int = 512,
    roll_range: tuple[float, float] = (0.0, 360.0),
    pitch_range: tuple[float, float] = (0.0, 360.0),
    yaw_range: tuple[float, float] = (0.0, 360.0),
    max_workers: int | None = None,
    cancel_event: threading.Event | None = None,
) -> list[CounterexampleHit]:
    """Перебирает ориентации в диапазоне/с шагом и возвращает те, где k >= threshold.

    `max_workers` — число процессов (по умолчанию os.cpu_count()); при малом числе
    комбинаций (<50) перебор всегда идёт в один поток — overhead запуска пула того не стоит.

    `cancel_event` — если передан и выставлен во время перебора, обход останавливается
    досрочно и пул процессов корректно останавливается (`shutdown`) перед возвратом,
    чтобы не оставлять осиротевшие процессы при отмене из GUI.
    """
    rolls = _frange(*roll_range, step_deg)
    pitches = _frange(*pitch_range, step_deg)
    yaws = _frange(*yaw_range, step_deg)

    combos = list(itertools.product(rolls, pitches, yaws))
    tasks = [
        (mesh, view_angles_deg, resolution_px, roundness_threshold, roll, pitch, yaw)
        for roll, pitch, yaw in combos
    ]

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if max_workers <= 1 or len(tasks) < 50:
        hits: list[CounterexampleHit] = []
        for task in tasks:
            if cancel_event is not None and cancel_event.is_set():
                break
            result = _evaluate_one(task)
            if result is not None:
                hits.append(result)
        return hits

    chunksize = max(1, len(tasks) // (max_workers * 4))
    hits = []
    executor = ProcessPoolExecutor(max_workers=max_workers)
    try:
        for result in executor.map(_evaluate_one, tasks, chunksize=chunksize):
            if cancel_event is not None and cancel_event.is_set():
                break
            if result is not None:
                hits.append(result)
    finally:
        # wait=True гарантирует, что все воркеры реально завершились к моменту
        # возврата из функции — иначе при отмене/закрытии окна процессы
        # ProcessPoolExecutor остаются висеть осиротевшими (см. утечку 2026-07-08).
        executor.shutdown(wait=True, cancel_futures=True)

    return hits
