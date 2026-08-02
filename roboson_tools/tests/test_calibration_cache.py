"""`pose/calibration_cache.py` — сохранение/загрузка результата регистрации+калибровки рядом с
папкой фото, чтобы не пересчитывать при каждой загрузке диалога."""

from __future__ import annotations

import numpy as np
import pytest

from roboson_tools.pose.calibration_cache import (
    CalibrationCache,
    calibration_cache_path,
    load_calibration_cache,
    save_calibration_cache,
)
from roboson_tools.pose.camera_pose import CameraIntrinsics
from roboson_tools.pose.marker_layout import MarkerLayout

_INTRINSICS = CameraIntrinsics(
    fx=1000.0, fy=990.0, cx=480.0, cy=640.0, resolution_px=(1280, 960),
    dist_coeffs=np.array([-0.1, 0.7, -0.02, 0.007, -1.1]),
)
_LAYOUT = MarkerLayout(
    dictionary_name="DICT_4X4_50",
    corners_world_mm={
        22: np.array([[20.0, 20.0, 0.0], [65.0, 20.0, 0.0], [65.0, 65.0, 0.0], [20.0, 65.0, 0.0]]),
        26: np.array([[0.0, 100.0, 50.0], [45.0, 100.0, 50.0], [45.0, 145.0, 50.0], [0.0, 145.0, 50.0]]),
    },
)


def test_calibration_cache_path_is_next_to_photos(tmp_path):
    path = calibration_cache_path(tmp_path)
    assert path.parent == tmp_path
    assert path.name == "calibration_result.yaml"


def test_missing_cache_returns_none(tmp_path):
    assert load_calibration_cache(tmp_path / "calibration_result.yaml") is None


def test_round_trip_preserves_layout_and_intrinsics(tmp_path):
    cache = CalibrationCache(
        layout=_LAYOUT,
        intrinsics=_INTRINSICS,
        reference_board="belt_left",
        rms_reprojection_error_pct=0.24,
        source_photos=["a.jpg", "b.jpg"],
    )
    path = calibration_cache_path(tmp_path)
    save_calibration_cache(path, cache)

    loaded = load_calibration_cache(path)
    assert loaded is not None
    assert loaded.reference_board == "belt_left"
    assert loaded.rms_reprojection_error_pct == pytest.approx(0.24)
    assert loaded.source_photos == ["a.jpg", "b.jpg"]
    assert loaded.layout.dictionary_name == "DICT_4X4_50"
    assert set(loaded.layout.corners_world_mm.keys()) == {22, 26}
    np.testing.assert_allclose(loaded.layout.corners_world_mm[26], _LAYOUT.corners_world_mm[26])
    assert loaded.intrinsics.fx == pytest.approx(_INTRINSICS.fx)
    assert loaded.intrinsics.resolution_px == _INTRINSICS.resolution_px
    np.testing.assert_allclose(loaded.intrinsics.dist_coeffs, _INTRINSICS.dist_coeffs)


def test_round_trip_with_null_dist_coeffs(tmp_path):
    intrinsics_no_dist = CameraIntrinsics(fx=900.0, fy=900.0, cx=320.0, cy=240.0, resolution_px=(480, 640))
    cache = CalibrationCache(layout=_LAYOUT, intrinsics=intrinsics_no_dist, reference_board="belt_left")
    path = calibration_cache_path(tmp_path)
    save_calibration_cache(path, cache)

    loaded = load_calibration_cache(path)
    assert loaded.intrinsics.dist_coeffs is None


def test_corrupted_cache_raises(tmp_path):
    path = calibration_cache_path(tmp_path)
    path.write_text("not: [valid, structure", encoding="utf-8")
    with pytest.raises(Exception):
        load_calibration_cache(path)
