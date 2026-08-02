"""Обнаружение ArUco ID, чей битовый узор совпадает САМ С СОБОЙ при зеркальном отражении.

Контекст: риг «2 камеры + 1 зеркало» — метки бортов ленты потенциально попадают в кадр И
напрямую, И отражёнными в зеркале одновременно (пользователь предложил это фильтровать). Изучено
эмпирически (см. `compute_mirror_ambiguous_ids`, перепроверяется в `tests/test_reflection_safety.py`):

1. `cv2.aruco.ArucoDetector` перебирает при сопоставлении 4 поворота метки, но НЕ отражения —
   поэтому отражённый (зеркальный) вид ОБЫЧНОЙ метки в подавляющем большинстве случаев вообще НЕ
   детектируется ни как какой-либо валидный ID (проверено по всему DICT_4X4_50: 46 из 50 ID).
   То есть для большинства меток отражение "само себя фильтрует" — детектор его просто не видит.
2. Исключение — узкий список ID (для DICT_4X4_50: **8, 9, 17, 47**), чей узор при отражении
   (по любой оси — проверено, детектор поворото-инвариантен, поэтому обе оси эквивалентны)
   случайно совпадает с ОДНИМ ИЗ 4 поворотов ТОГО ЖЕ ID. Для них порядок обхода вершин
   (winding order) НЕ помогает отличить отражение от прямого вида — эмпирически проверено, что
   он у этих ID НЕ меняется при отражении (в отличие от того, что можно было бы наивно
   предположить для обычной геометрической проекции без переотражения) — иными словами, для
   этих 4 ID отражение неотличимо от прямого вида ПО ЛЮБОЙ детектируемой геометрии.

Вывод: единственный надёжный способ борьбы — НЕ использовать эти ID для меток, потенциально
видимых через отражение (то есть практически для любой метки рига, раз зеркало есть в сцене).
`assert_no_mirror_ambiguous_ids` проверяет это при загрузке раскладки
(`marker_layout.load_marker_layout`/`load_boards`) — раньше, чем ошибка успеет незаметно
испортить реконструкцию."""

from __future__ import annotations

import cv2
import numpy as np

# Посчитано compute_mirror_ambiguous_ids("DICT_4X4_50") — см. docstring модуля и
# tests/test_reflection_safety.py, где значение перепроверяется свежим вычислением (защита от
# рассинхронизации, если словарь когда-либо будет заменён/обновлён в другой версии OpenCV).
MIRROR_AMBIGUOUS_IDS_DICT_4X4_50: frozenset[int] = frozenset({8, 9, 17, 47})

_KNOWN_AMBIGUOUS_IDS: dict[str, frozenset[int]] = {
    "DICT_4X4_50": MIRROR_AMBIGUOUS_IDS_DICT_4X4_50,
}


def compute_mirror_ambiguous_ids(dictionary_name: str, marker_px: int = 400) -> frozenset[int]:
    """Эмпирически находит ID словаря `dictionary_name`, чей рендер, отражённый по горизонтали
    ИЛИ по вертикали, детектируется тем же детектором (тот же словарь, `DetectorParameters()` по
    умолчанию) как ЛЮБОЙ валидный ID. Небыстро (рендерит + детектирует каждый ID словаря дважды,
    с полем вокруг метки для корректной детекции) — вызывать один раз при добавлении нового
    словаря/проверке в тесте, не в рантайме на каждую загрузку."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    n_markers = dictionary.bytesList.shape[0]
    margin = marker_px // 4

    ambiguous: set[int] = set()
    for marker_id in range(n_markers):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        canvas = np.full((marker_px + 2 * margin, marker_px + 2 * margin), 255, dtype=np.uint8)
        canvas[margin : margin + marker_px, margin : margin + marker_px] = marker_img
        img_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for flip_code in (0, 1):  # вертикально, горизонтально — покрывает любую ось зеркала
            flipped = cv2.flip(img_bgr, flip_code)
            _corners, ids, _rejected = detector.detectMarkers(flipped)
            if ids is not None:
                ambiguous.add(marker_id)
                break
    return frozenset(ambiguous)


def assert_no_mirror_ambiguous_ids(dictionary_name: str, marker_ids: set[int] | frozenset[int]) -> None:
    """Бросает `ValueError`, если среди `marker_ids` есть хотя бы один зеркально-неоднозначный ID
    для `dictionary_name` (см. докстринг модуля). Для известных словарей использует
    предпосчитанный `_KNOWN_AMBIGUOUS_IDS` (быстро); для незнакомых — считает на месте
    (`compute_mirror_ambiguous_ids`, дороже, но корректно)."""
    ambiguous_set = _KNOWN_AMBIGUOUS_IDS.get(dictionary_name)
    if ambiguous_set is None:
        ambiguous_set = compute_mirror_ambiguous_ids(dictionary_name)
    bad = set(marker_ids) & ambiguous_set
    if bad:
        raise ValueError(
            f"метки id={sorted(bad)} зеркально-неоднозначны в словаре {dictionary_name!r} — их "
            "отражение в зеркале рига детектируется как валидная метка ТОГО ЖЕ ID, неотличимо ни "
            "по какой детектируемой геометрии (см. roboson_tools/pose/reflection_safety.py). "
            "Выберите другие ID для этих меток."
        )
