"""Фоновая субтракция для прямых CV-камер (Фаза 2.1 CV-пайплайна).

HSV, а не RGB — тон (H)/насыщенность (S) меняются меньше яркости (V) при
попадании объекта в тень ленты, поэтому порог по H/S устойчивее к
освещению. Основной канал сегментации (не кадр-минус-предыдущий-кадр) —
модель фона строится один раз по пустой ленте, см. заметку задачи
"Фоновая субтракция и маска объекта на CV-камерах (Фаза 2.1, CV-пайплайн
Webots)".

Доп. НАПРАВЛЕННЫЙ критерий по V (яркость) — объект ЯРЧЕ фона на
`v_threshold` и более. Тень уменьшает V (объект ВСЕГДА темнее фона под
тенью), но никогда не увеличивает — поэтому порог "ярче фона" структурно
не может сработать на тени, в отличие от порога по |ΔV| в обе стороны.
Найдено на реальном случае: объект `cylinder` (тестовый набор) почти
неотличим от ленты по H/S (нейтральный серый на нейтральном сером фоне,
ΔH/ΔS в разы ниже порога) — без этого критерия терялось ~90% силуэта."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BackgroundModel:
    hsv: np.ndarray  # HxWx3 float32 — эталонный HSV-кадр пустой ленты


def build_background_model(frame_bgr: np.ndarray) -> BackgroundModel:
    return BackgroundModel(hsv=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32))


def compute_mask(
    frame_bgr: np.ndarray,
    background: BackgroundModel,
    h_threshold: float,
    s_threshold: float,
    min_area_px: float,
    morph_kernel: int,
    v_threshold: float | None = None,
    return_all_contours: bool = False,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None] | tuple[np.ndarray, tuple[int, int, int, int] | None, list]:
    """Маска переднего плана (uint8, 0/255) + bbox (x, y, w, h) наибольшей
    связной компоненты. Возвращает (пустая маска, None), если ни одна
    компонента не прошла порог площади `min_area_px` (шум/пустая лента).

    `return_all_contours=True` — доп. третий элемент кортежа: список ВСЕХ
    контуров, прошедших `min_area_px` (не только наибольший) — нужно, когда в
    кадре одновременно может быть виден другой объект (Фаза 3, `sim/cv_grid.py::
    select_target_contour` выбирает среди них нужный по ожидаемой позиции, не
    по площади) — см. заметку задачи "Сборка сетки 9 ракурсов..."."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = np.abs(hsv[..., 0] - background.hsv[..., 0])
    dh = np.minimum(dh, 180.0 - dh)  # H циклический (OpenCV: диапазон 0..179)
    ds = np.abs(hsv[..., 1] - background.hsv[..., 1])
    foreground = (dh > h_threshold) | (ds > s_threshold)
    if v_threshold is not None:
        dv_signed = hsv[..., 2] - background.hsv[..., 2]  # знак важен — см. docstring модуля
        foreground = foreground | (dv_signed > v_threshold)
    fg = foreground.astype(np.uint8) * 255

    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    qualifying = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not qualifying:
        empty = (np.zeros_like(fg), None)
        return (*empty, []) if return_all_contours else empty
    largest = max(qualifying, key=cv2.contourArea)

    mask = np.zeros_like(fg)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    bbox = cv2.boundingRect(largest)
    return (mask, bbox, qualifying) if return_all_contours else (mask, bbox)


def select_target_contour(contours: list, expected_row: float) -> tuple[int, int, int, int] | None:
    """Среди НЕСКОЛЬКИХ контуров (см. `compute_mask(..., return_all_contours=True)`)
    выбирает тот, чей bbox-центр по строкам БЛИЖЕ ВСЕГО к `expected_row`
    (ожидаемая строка объекта, посчитанная из известной физической X — см.
    `sim/cv_grid.py::expected_pixel_row`) — НЕ наибольший по площади.

    Нужна, когда в кадре одновременно виден другой объект (интервал появления
    товаров 0.7-0.5с меньше окна камеры `top` ~1.75с, реалистичный сценарий,
    не гипотетический — см. заметку задачи "Детектор моментов..." раздел "Не
    тронуто"): `compute_mask` без этой функции молча взял бы наибольший
    контур, который может принадлежать ЧУЖОМУ объекту, а не тому, чей момент
    сейчас обрабатывается. Возвращает bbox (x, y, w, h) выбранного контура,
    None если список пуст."""
    if not contours:
        return None

    def _row_distance(c) -> float:
        x, y, w, h = cv2.boundingRect(c)
        return abs((y + h / 2.0) - expected_row)
    best = min(contours, key=_row_distance)
    return cv2.boundingRect(best)
