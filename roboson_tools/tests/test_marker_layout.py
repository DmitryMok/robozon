"""`pose/marker_layout.py` — загрузка раскладки ArUco-меток из YAML, без Qt."""

from __future__ import annotations

import cv2
import pytest
import yaml

from pathlib import Path

from roboson_tools.pose.marker_layout import (
    cv2_dictionary,
    load_all_boards,
    load_boards,
    load_marker_layout,
)

_VALID_LAYOUT = {
    "dictionary": "DICT_4X4_50",
    "boards": [
        {
            "name": "mirror",
            "markers": [
                {
                    "id": 0,
                    "corners_mm": [
                        [0.0, 0.0, 500.0],
                        [50.0, 0.0, 500.0],
                        [50.0, -50.0, 500.0],
                        [0.0, -50.0, 500.0],
                    ],
                },
            ],
        },
        {
            "name": "belt_sides",
            "markers": [
                {
                    "id": 10,
                    "corners_mm": [
                        [0.0, 150.0, 0.0],
                        [50.0, 150.0, 0.0],
                        [50.0, 150.0, -50.0],
                        [0.0, 150.0, -50.0],
                    ],
                },
            ],
        },
    ],
}


def _write_layout(tmp_path, data: dict):
    path = tmp_path / "aruco_markers.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_valid_layout_flattens_boards(tmp_path):
    path = _write_layout(tmp_path, _VALID_LAYOUT)
    layout = load_marker_layout(path)
    assert layout.dictionary_name == "DICT_4X4_50"
    assert set(layout.corners_world_mm.keys()) == {0, 10}
    assert layout.corners_world_mm[0].shape == (4, 3)
    assert layout.corners_world_mm[10][2].tolist() == [50.0, 150.0, -50.0]


def test_duplicate_id_across_boards_raises(tmp_path):
    data = {
        "dictionary": "DICT_4X4_50",
        "boards": [
            {"name": "mirror", "markers": [_VALID_LAYOUT["boards"][0]["markers"][0]]},
            {"name": "belt_sides", "markers": [_VALID_LAYOUT["boards"][0]["markers"][0]]},
        ],
    }
    path = _write_layout(tmp_path, data)
    with pytest.raises(ValueError, match="более одного раза"):
        load_marker_layout(path)


def test_wrong_corner_count_raises(tmp_path):
    data = {
        "dictionary": "DICT_4X4_50",
        "boards": [
            {
                "name": "mirror",
                "markers": [{"id": 0, "corners_mm": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}],
            }
        ],
    }
    path = _write_layout(tmp_path, data)
    with pytest.raises(ValueError, match="4x3"):
        load_marker_layout(path)


def test_empty_layout_raises(tmp_path):
    path = _write_layout(tmp_path, {"dictionary": "DICT_4X4_50", "boards": []})
    with pytest.raises(ValueError, match="пуста"):
        load_marker_layout(path)


def test_cv2_dictionary_usable_by_detector(tmp_path):
    path = _write_layout(tmp_path, _VALID_LAYOUT)
    layout = load_marker_layout(path)
    dictionary = cv2_dictionary(layout)
    # не должно падать — smoke-проверка, что раскладка даёт рабочий cv2.aruco.Dictionary
    cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def test_shipped_config_loads():
    """config/aruco_markers.yaml (пример, поставляемый с проектом) грузится без ошибок."""
    layout = load_marker_layout()
    assert len(layout.corners_world_mm) >= 2


def test_mirror_ambiguous_id_is_rejected_on_load(tmp_path):
    """id=8 зеркально-неоднозначен в DICT_4X4_50 (см. pose/reflection_safety.py) — загрузка
    раскладки с таким ID должна упасть, а не молча дать риск неотличимой от отражения метки."""
    data = {
        "dictionary": "DICT_4X4_50",
        "boards": [{"name": "mirror", "markers": [{**_VALID_LAYOUT["boards"][0]["markers"][0], "id": 8}]}],
    }
    path = _write_layout(tmp_path, data)
    with pytest.raises(ValueError, match="зеркально-неоднозначны"):
        load_marker_layout(path)


def test_load_boards_keeps_boards_separate(tmp_path):
    """В отличие от load_marker_layout, load_boards НЕ сливает доски — нужно для разрезаемых
    пополам листов (mirror_board/belt_board), где взаимное положение колонок неизвестно заранее."""
    path = _write_layout(tmp_path, _VALID_LAYOUT)
    boards = load_boards(path)
    assert set(boards.keys()) == {"mirror", "belt_sides"}
    assert set(boards["mirror"].corners_world_mm.keys()) == {0}
    assert set(boards["belt_sides"].corners_world_mm.keys()) == {10}
    assert boards["mirror"].dictionary_name == "DICT_4X4_50"


def test_load_boards_duplicate_id_across_boards_raises(tmp_path):
    data = {
        "dictionary": "DICT_4X4_50",
        "boards": [
            {"name": "mirror", "markers": [_VALID_LAYOUT["boards"][0]["markers"][0]]},
            {"name": "belt_sides", "markers": [_VALID_LAYOUT["boards"][0]["markers"][0]]},
        ],
    }
    path = _write_layout(tmp_path, data)
    with pytest.raises(ValueError, match="более одного раза"):
        load_boards(path)


def test_load_all_boards_merges_multiple_files(tmp_path):
    path_a = _write_layout(tmp_path, _VALID_LAYOUT)
    other = {
        "dictionary": "DICT_4X4_50",
        "boards": [
            {
                "name": "belt_left",
                "markers": [
                    {
                        "id": 26,
                        "corners_mm": [
                            [0.0, 0.0, 0.0], [45.0, 0.0, 0.0], [45.0, 45.0, 0.0], [0.0, 45.0, 0.0],
                        ],
                    }
                ],
            }
        ],
    }
    path_b = tmp_path.joinpath("other.yaml")
    path_b.write_text(yaml.safe_dump(other), encoding="utf-8")

    merged = load_all_boards(path_a, path_b)
    assert set(merged.keys()) == {"mirror", "belt_sides", "belt_left"}


def test_load_all_boards_rejects_duplicate_board_name(tmp_path):
    path_a = _write_layout(tmp_path, _VALID_LAYOUT)
    path_b = tmp_path.joinpath("dup.yaml")
    path_b.write_text(yaml.safe_dump(_VALID_LAYOUT), encoding="utf-8")
    with pytest.raises(ValueError, match="нескольких файлах"):
        load_all_boards(path_a, path_b)


def test_shipped_mirror_and_belt_boards_load_as_four_separate_boards():
    """Реальные assets/mirror_board/mirror_board.yaml + assets/belt_board/belt_board.yaml —
    ровно 4 доски (2 колонки зеркала + 2 борта ленты), как описано в заметке задачи."""
    root = Path(__file__).resolve().parents[1]
    merged = load_all_boards(
        root / "assets" / "mirror_board" / "mirror_board.yaml",
        root / "assets" / "belt_board" / "belt_board.yaml",
    )
    assert set(merged.keys()) == {"mirror_edge_left", "mirror_edge_right", "belt_left", "belt_right"}
    for board in merged.values():
        assert len(board.corners_world_mm) == 2  # каждая доска — пара меток
