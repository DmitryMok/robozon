"""Раскладка ArUco-меток в мировых (или локальных — см. `load_boards`) координатах (мм) — вход
для `pose/aruco_estimator.py`.

Два разных случая, ДВЕ разные функции загрузки — не путать:

- `load_marker_layout` — раскладка УЖЕ в одной общей системе координат (все "boards" в YAML —
  просто группировка для читаемости при обмере, метки сливаются в один плоский словарь). Годится
  для `config/aruco_markers.yaml` (текущий тестовый лист — один плоский лист, все метки в одной
  системе) и для результата `board_registration.merge_layout()` (после регистрации).
- `load_boards`/`load_all_boards` — "boards" в YAML физически НЕЗАВИСИМЫ (например
  `assets/mirror_board/mirror_board.yaml`: левая и правая колонки печатаются на одном листе, но
  ЛИСТ РАЗРЕЗАЕТСЯ пополам — точная геометрия известна только ВНУТРИ каждой колонки, взаимное
  положение колонок друг относительно друга неизвестно до регистрации по мостиковым фото). Эти
  функции НЕ сливают доски — возвращают `dict[board_name, MarkerLayout]`, готовый как `boards=`
  для `board_registration.register_boards()`.

ID меток уникальны глобально по всей сцене (по всем доскам сразу) — обе функции это проверяют.

ВАЖНО: порядок 4 углов в `corners_mm` должен совпадать с порядком, который возвращает
`cv2.aruco.ArucoDetector.detectMarkers` для конкретной метки — top-left, top-right,
bottom-right, bottom-left В СИСТЕМЕ ОТСЧЁТА САМОЙ МЕТКИ, КАК ОНА НАПЕЧАТАНА (не как её видно
с конкретного ракурса камеры). Перепутанный порядок — источник неверной/неоднозначной позы
при solvePnP, не всегда заметный по картинке.

Обе функции также проверяют (`reflection_safety.assert_no_mirror_ambiguous_ids`), что ни один ID
не является зеркально-неоднозначным для словаря раскладки — риг «2 камеры + 1 зеркало» может
показать метку и напрямую, и отражённой в зеркале одновременно, а для таких ID это неотличимо
по геометрии (см. `pose/reflection_safety.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..core.config import CONFIG_DIR, PROJECT_ROOT
from .reflection_safety import assert_no_mirror_ambiguous_ids

ARUCO_MARKERS_PATH = CONFIG_DIR / "aruco_markers.yaml"
# 4 доски реального рига (зеркало разрезано на 2 колонки, борта ленты — ещё 2 доски) — см.
# assets/mirror_board/, assets/belt_board/. Взаимное положение досок НЕ известно заранее — его
# считает `board_registration.register_boards()` по мостиковым фото, а не ручной обмер.
MIRROR_BOARD_PATH = PROJECT_ROOT / "assets" / "mirror_board" / "mirror_board.yaml"
BELT_BOARD_PATH = PROJECT_ROOT / "assets" / "belt_board" / "belt_board.yaml"


@dataclass(frozen=True)
class MarkerLayout:
    dictionary_name: str
    corners_world_mm: dict[int, np.ndarray]  # id -> (4, 3)


def _parse_marker_corners(marker: dict, seen_ids: set[int]) -> tuple[int, np.ndarray]:
    marker_id = int(marker["id"])
    if marker_id in seen_ids:
        raise ValueError(
            f"метка id={marker_id} встречается более одного раза в раскладке — ID меток должны "
            "быть уникальны глобально"
        )
    pts = np.array(marker["corners_mm"], dtype=np.float64)
    if pts.shape != (4, 3):
        raise ValueError(
            f"метка id={marker_id}: corners_mm должен быть 4x3 (4 угла, xyz), получено {pts.shape}"
        )
    return marker_id, pts


def load_marker_layout(path: Path | str = ARUCO_MARKERS_PATH) -> MarkerLayout:
    """Сливает ВСЕ "boards" YAML в один плоский `MarkerLayout` — верно только если они уже в
    одной системе координат (см. докстринг модуля)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    corners: dict[int, np.ndarray] = {}
    seen_ids: set[int] = set()
    for board in data["boards"]:
        for marker in board["markers"]:
            marker_id, pts = _parse_marker_corners(marker, seen_ids)
            seen_ids.add(marker_id)
            corners[marker_id] = pts
    if not corners:
        raise ValueError(f"раскладка меток {path} пуста")
    dictionary_name = str(data["dictionary"])
    assert_no_mirror_ambiguous_ids(dictionary_name, corners.keys())
    return MarkerLayout(dictionary_name=dictionary_name, corners_world_mm=corners)


def load_boards(path: Path | str) -> dict[str, MarkerLayout]:
    """Читает многодосочный YAML (см. `assets/mirror_board/`, `assets/belt_board/`), СОХРАНЯЯ
    доски отдельными `MarkerLayout` — в отличие от `load_marker_layout`, который слил бы их в
    один плоский словарь (верно только когда доски физически УЖЕ в одной системе координат — не
    тот случай для разрезаемых пополам листов). Результат годится напрямую как `boards=` для
    `board_registration.register_boards()`."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    dictionary_name = str(data["dictionary"])
    boards: dict[str, MarkerLayout] = {}
    seen_ids: set[int] = set()
    for board in data["boards"]:
        board_corners: dict[int, np.ndarray] = {}
        for marker in board["markers"]:
            marker_id, pts = _parse_marker_corners(marker, seen_ids)
            seen_ids.add(marker_id)
            board_corners[marker_id] = pts
        if not board_corners:
            raise ValueError(f"доска {board['name']!r} в {path} пуста")
        boards[str(board["name"])] = MarkerLayout(
            dictionary_name=dictionary_name, corners_world_mm=board_corners
        )
    if not boards:
        raise ValueError(f"раскладка {path} не содержит досок")
    assert_no_mirror_ambiguous_ids(dictionary_name, seen_ids)
    return boards


def load_all_boards(*paths: Path | str) -> dict[str, MarkerLayout]:
    """Объединяет несколько многодосочных YAML (см. `load_boards`) в один словарь
    `board_name -> MarkerLayout` — например `assets/mirror_board/mirror_board.yaml` +
    `assets/belt_board/belt_board.yaml` дают все 4 доски рига сразу (см. заметку задачи
    "3D-реконструкция объекта по реальным фото...", раздел про 4 доски и мостиковую регистрацию),
    готовые для `board_registration.register_boards()`. Проверяет уникальность имён досок и ID
    меток МЕЖДУ файлами (внутри одного файла это уже проверяет `load_boards`)."""
    merged: dict[str, MarkerLayout] = {}
    seen_ids: set[int] = set()
    for path in paths:
        for board_name, layout in load_boards(path).items():
            if board_name in merged:
                raise ValueError(f"доска {board_name!r} встречается в нескольких файлах ({path})")
            overlap = seen_ids & layout.corners_world_mm.keys()
            if overlap:
                raise ValueError(
                    f"метки id={sorted(overlap)} встречаются в нескольких файлах/досках ({path})"
                )
            seen_ids |= layout.corners_world_mm.keys()
            merged[board_name] = layout
    return merged


def cv2_dictionary(layout: MarkerLayout) -> cv2.aruco.Dictionary:
    dict_id = getattr(cv2.aruco, layout.dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)
