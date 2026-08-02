"""`segmentation/sam3_subprocess_backend.py` — протокол JSON/temp-файлов с постоянным фоновым
процессом UavVisionLab (`subprocess.Popen`, не одноразовый `subprocess.run` — модель должна
грузиться один раз и переживать несколько вызовов `segment_batch()`), без реального subprocess'а
(не тянем GPU/torch/tensorrt в тесты — `subprocess.Popen` подменяется фейковым процессом).
Реальная сквозная проверка (модель грузится, TRT-движок подходит под GPU) сделана вручную при
реализации задачи, не автоматизирована здесь."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from roboson_tools.segmentation.config import SegmentationConfig
from roboson_tools.segmentation.sam3_subprocess_backend import Sam3SubprocessBackend


def _config() -> SegmentationConfig:
    return SegmentationConfig(
        prompt="object",
        conf_threshold=0.30,
        imgsz=1008,
        uavvisionlab_root=Path("C:/Projects/er-26/UavVisionLab"),
        python_exe=Path("C:/Projects/er-26/UavVisionLab/.venv/Scripts/python.exe"),
        timeout_s=180,
    )


class _FakeStdin:
    def __init__(self, on_write) -> None:
        self._on_write = on_write
        self.closed = False

    def write(self, text: str) -> None:
        self._on_write(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    def __init__(self) -> None:
        self.queue: list[str] = []

    def readline(self) -> str:
        return self.queue.pop(0) if self.queue else ""


class _FakeProcess:
    """Имитирует живой (не завершившийся) `Popen`: запись в stdin синхронно кладёт ответ
    `handler` в очередь stdout — без потоков/асинхронности, тестам этого достаточно (вызовы
    `segment_batch` в тесте тоже синхронные)."""

    def __init__(self, handler) -> None:
        self.stdout = _FakeStdout()
        self.stdin = _FakeStdin(self._handle_write)
        self._handler = handler
        self.call_count = 0
        self._alive = True
        self.killed = False

    def _handle_write(self, text: str) -> None:
        self.call_count += 1
        request = json.loads(text)
        self.stdout.queue.append(self._handler(request))

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False
        self.killed = True


def _write_detection_mask(mask_path: Path, rank: int, region: tuple[slice, slice]) -> None:
    det_path = mask_path.with_name(f"{mask_path.stem}_{rank:02d}{mask_path.suffix}")
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[region] = 255
    cv2.imwrite(str(det_path), mask)


def _handler_ok(request: dict) -> str:
    """Одна детекция на кадр (обратно совместимый случай — большинство тестов ниже проверяют
    именно primary-детекцию, не мульти-детекцию)."""
    results = []
    for item in request["items"]:
        mask_path = Path(item["mask_path"])
        _write_detection_mask(mask_path, 0, (slice(2, 5), slice(3, 8)))
        results.append(
            {"ok": True, "detections": [{"confidence": 0.87, "class_name": "object", "bbox": [3, 2, 5, 3]}]}
        )
    return json.dumps({"ok": True, "results": results}) + "\n"


def _handler_multi_detection(request: dict) -> str:
    """Протокол теперь отдаёт ВСЕ детекции на кадре (по убыванию уверенности), не только
    самую уверенную — регрессия на баг: выключение чекбокса "Скрывать объекты вне ленты" не
    показывало сегментацию других объектов в кадре, т.к. backend раньше просто отбрасывал их
    (см. заметку задачи и docstring `SegmentationResult.all_masks`)."""
    results = []
    for item in request["items"]:
        mask_path = Path(item["mask_path"])
        _write_detection_mask(mask_path, 0, (slice(2, 5), slice(3, 8)))
        _write_detection_mask(mask_path, 1, (slice(6, 9), slice(10, 15)))
        results.append(
            {
                "ok": True,
                "detections": [
                    {"confidence": 0.87, "class_name": "object", "bbox": [3, 2, 5, 3]},
                    {"confidence": 0.41, "class_name": "object", "bbox": [10, 6, 5, 3]},
                ],
            }
        )
    return json.dumps({"ok": True, "results": results}) + "\n"


def test_segment_batch_empty_input_does_not_spawn_process():
    backend = Sam3SubprocessBackend(_config())
    with patch("subprocess.Popen") as mock_popen:
        results = backend.segment_batch([])
    assert results == []
    mock_popen.assert_not_called()


def test_segment_batch_parses_ok_result_and_reads_mask_file():
    backend = Sam3SubprocessBackend(_config())
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    with patch("subprocess.Popen", return_value=_FakeProcess(_handler_ok)):
        results = backend.segment_batch([image], prompt="object")

    assert len(results) == 1
    result = results[0]
    assert result is not None
    assert result.confidence == pytest.approx(0.87)
    assert result.class_name == "object"
    assert result.mask.shape == (10, 20)
    assert result.mask[3, 5]
    assert not result.mask[0, 0]
    assert len(result.all_masks) == 1
    assert result.all_masks[0] is result.mask


def test_segment_batch_returns_all_detections_not_just_best():
    backend = Sam3SubprocessBackend(_config())
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    with patch("subprocess.Popen", return_value=_FakeProcess(_handler_multi_detection)):
        results = backend.segment_batch([image], prompt="object")

    assert len(results) == 1
    result = results[0]
    assert result is not None
    # Primary остаётся самой уверенной детекцией (обратная совместимость).
    assert result.confidence == pytest.approx(0.87)
    assert result.mask[3, 5]
    # А вторая (менее уверенная) детекция теперь доступна через all_masks, не потеряна.
    assert len(result.all_masks) == 2
    assert result.all_masks[0] is result.mask
    assert result.all_masks[1][7, 12]
    assert not result.all_masks[1][3, 5]


def test_segment_batch_reuses_same_process_across_calls():
    """Регрессия на исходный запрос пользователя: повторная сегментация не должна поднимать
    процесс заново (модель грузилась бы повторно)."""
    backend = Sam3SubprocessBackend(_config())
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_process = _FakeProcess(_handler_ok)
    with patch("subprocess.Popen", return_value=fake_process) as mock_popen:
        backend.segment_batch([image])
        backend.segment_batch([image])
    mock_popen.assert_called_once()
    assert fake_process.call_count == 2


def test_segment_batch_maps_per_item_failure_to_none():
    backend = Sam3SubprocessBackend(_config())
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    def handler(request: dict) -> str:
        return json.dumps({"ok": True, "results": [{"ok": False, "error": "объект не найден"}]}) + "\n"

    with patch("subprocess.Popen", return_value=_FakeProcess(handler)):
        results = backend.segment_batch([image])

    assert results == [None]


def test_segment_batch_raises_on_dead_process():
    backend = Sam3SubprocessBackend(_config())
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    def handler(request: dict) -> str:
        return ""  # процесс "упал" — пустая строка вместо ответа

    with patch("subprocess.Popen", return_value=_FakeProcess(handler)):
        with pytest.raises(RuntimeError):
            backend.segment_batch([image])


def test_segment_batch_raises_on_fatal_backend_error():
    backend = Sam3SubprocessBackend(_config())
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    def handler(request: dict) -> str:
        return json.dumps({"ok": False, "error": "model load error: engine not found"}) + "\n"

    with patch("subprocess.Popen", return_value=_FakeProcess(handler)):
        with pytest.raises(RuntimeError):
            backend.segment_batch([image])


def test_shutdown_closes_stdin_and_clears_process():
    backend = Sam3SubprocessBackend(_config())
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    fake_process = _FakeProcess(_handler_ok)
    with patch("subprocess.Popen", return_value=fake_process):
        backend.segment_batch([image])
        backend.shutdown()

    assert fake_process.stdin.closed
    assert backend._process is None

    # После shutdown новый вызов поднимает процесс заново.
    fake_process_2 = _FakeProcess(_handler_ok)
    with patch("subprocess.Popen", return_value=fake_process_2) as mock_popen:
        backend.segment_batch([image])
    mock_popen.assert_called_once()
