"""`pose/reflection_safety.py` — обнаружение ArUco ID, неотличимых от собственного отражения в
зеркале. Проверяет, что предпосчитанная константа не разошлась со свежим вычислением (защита от
рассинхронизации при обновлении OpenCV/словаря), и что валидация ловит опасные ID."""

from __future__ import annotations

import pytest

from roboson_tools.pose.reflection_safety import (
    MIRROR_AMBIGUOUS_IDS_DICT_4X4_50,
    assert_no_mirror_ambiguous_ids,
    compute_mirror_ambiguous_ids,
)


def test_precomputed_constant_matches_fresh_computation():
    assert compute_mirror_ambiguous_ids("DICT_4X4_50") == MIRROR_AMBIGUOUS_IDS_DICT_4X4_50


def test_project_marker_ids_are_not_mirror_ambiguous():
    """Реальные ID, используемые в проекте (тестовый лист 20/21, mirror_board 22-25,
    belt_board 26-29) — не должны попадать в опасное множество."""
    project_ids = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
    assert_no_mirror_ambiguous_ids("DICT_4X4_50", project_ids)  # не должно бросить


def test_ambiguous_id_is_rejected():
    with pytest.raises(ValueError, match="зеркально-неоднозначны"):
        assert_no_mirror_ambiguous_ids("DICT_4X4_50", {8, 20})


def test_unknown_dictionary_falls_back_to_fresh_computation():
    # DICT_4X4_100 не в _KNOWN_AMBIGUOUS_IDS — должен посчитать на месте, не упасть.
    result = compute_mirror_ambiguous_ids("DICT_4X4_100")
    assert isinstance(result, frozenset)
