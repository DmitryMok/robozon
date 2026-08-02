"""Backend сегментации через subprocess к Windows `.venv` соседнего проекта UavVisionLab
(`c:\\Projects\\er-26\\UavVisionLab`, путь — `config/segmentation.yaml`). Там уже установлены
torch/tensorrt/sam3 (~7 ГБ, gated HuggingFace-веса) и уже лежат все файлы модели
(`data/models/sam3/`) — решение не ставить этот стек второй раз в `venv-roboson-tools`, а
вызывать готовый Windows-процесс как чёрный ящик через stdin/stdout JSON. Обратная сторона
протокола — `UavVisionLab/tools/segment_cli.py`.

Оба процесса — Windows `python.exe`, WSL не участвует (архитектурное решение от 2026-07-18, см.
заметку задачи в вики). Модель — `pt_trt` (TRT-backbone + облегчённый pt-скелет), как в
продовых сценариях UavVisionLab.

Процесс держится живым между вызовами `segment_batch()` (загрузка TRT-модели занимает заметное
время — незачем платить за неё на каждый клик «Сегментация»). Владелец backend'а (сейчас —
`PhotoCaptureDialog`) должен вызвать `shutdown()` при закрытии приложения; смена классов между
вызовами не требует перезапуска процесса (`set_classes()` в `segment_cli.py` дешёвая).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .base import SegmentationResult
from .config import SegmentationConfig, load_segmentation_config


class Sam3SubprocessBackend:
    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self._config = config or load_segmentation_config()
        self._cli_script = self._config.uavvisionlab_root / "tools" / "segment_cli.py"
        self._process: subprocess.Popen | None = None
        self._stderr_log_path: Path | None = None

    def _ensure_process(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process

        stderr_log = tempfile.NamedTemporaryFile(
            prefix="roboson_sam3_stderr_", suffix=".log", delete=False
        )
        self._stderr_log_path = Path(stderr_log.name)
        self._process = subprocess.Popen(
            [str(self._config.python_exe), str(self._cli_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_log,
            text=True,
            encoding="utf-8",
            bufsize=1,  # построчная буферизация — нужна для readline() запрос-за-запросом
            cwd=str(self._config.uavvisionlab_root),
        )
        # Popen дублирует дескриптор для дочернего процесса — наш файловый объект больше не
        # нужен. На Windows незакрытый handle не даёт потом удалить файл в shutdown().
        stderr_log.close()
        return self._process

    _MAX_STRAY_LINES = 50  # запас на случай постороннего вывода от torch/sam3/TRT в stdout

    def _read_response(self, process: subprocess.Popen) -> dict:
        """`segment_cli.py` глушит через `_stdout_muted()` посторонний `print()` от sam3/TRT
        (см. его докстринг) — это основной фикс. Здесь — защитная сетка на случай, если что-то
        всё же просочится в stdout (например, из кода до входа в цикл обработки запросов):
        пропускаем строки, не парсящиеся как JSON, вместо падения на первой же из них."""
        for _ in range(self._MAX_STRAY_LINES):
            line = process.stdout.readline()
            if not line:
                returncode = process.poll()
                self._process = None
                raise RuntimeError(
                    f"segment_cli.py завершился (код {returncode}) без ответа. "
                    f"stderr:\n{self._stderr_tail()}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise RuntimeError(
            f"segment_cli.py: не удалось найти JSON-ответ в потоке постороннего вывода. "
            f"stderr:\n{self._stderr_tail()}"
        )

    def _stderr_tail(self) -> str:
        if self._stderr_log_path is None:
            return ""
        try:
            return self._stderr_log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            return ""

    def segment_batch(
        self, images_bgr: list[np.ndarray], prompt: str | None = None
    ) -> list[SegmentationResult | None]:
        if not images_bgr:
            return []
        effective_prompt = prompt or self._config.prompt
        process = self._ensure_process()

        with tempfile.TemporaryDirectory(prefix="roboson_sam3_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            items = []
            for i, image_bgr in enumerate(images_bgr):
                image_path = tmp_path / f"frame_{i:03d}.png"
                mask_path = tmp_path / f"mask_{i:03d}.png"
                cv2.imwrite(str(image_path), image_bgr)
                items.append({"image_path": str(image_path), "mask_path": str(mask_path)})

            request = {
                "prompt": effective_prompt,
                "conf_threshold": self._config.conf_threshold,
                "imgsz": self._config.imgsz,
                "items": items,
            }
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                response = self._read_response(process)
            except (BrokenPipeError, OSError) as exc:
                self._process = None
                raise RuntimeError(
                    f"segment_cli.py: процесс упал ({exc}). stderr:\n{self._stderr_tail()}"
                ) from exc
            if not response.get("ok"):
                raise RuntimeError(f"segment_cli.py: {response.get('error')}")

            results: list[SegmentationResult | None] = []
            for i, item_result in enumerate(response["results"]):
                if not item_result.get("ok"):
                    results.append(None)
                    continue
                # По одному PNG-файлу НА ДЕТЕКЦИЮ (см. докстринг segment_cli.py::_run_batch —
                # `detections[rank]` <-> `mask_{i:03d}_{rank:02d}.png`), по убыванию уверенности.
                # Детекция без читаемого файла маски отбрасывается ЦЕЛИКОМ (не только маска) —
                # иначе `all_masks[0]` могла бы оказаться маской ДРУГОЙ детекции, чем та, чьи
                # confidence/class_name используются как primary ниже.
                base_mask_path = tmp_path / f"mask_{i:03d}.png"
                detections = []
                all_masks: list[np.ndarray] = []
                for rank, detection in enumerate(item_result["detections"]):
                    det_mask_path = base_mask_path.with_name(
                        f"{base_mask_path.stem}_{rank:02d}{base_mask_path.suffix}"
                    )
                    mask_img = cv2.imread(str(det_mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_img is None:
                        continue
                    detections.append(detection)
                    all_masks.append(mask_img > 127)
                if not all_masks:
                    results.append(None)
                    continue
                primary = detections[0]
                results.append(
                    SegmentationResult(
                        mask=all_masks[0],
                        confidence=float(primary["confidence"]),
                        class_name=str(primary["class_name"]),
                        all_masks=all_masks,
                    )
                )
            return results

    def warmup(self) -> float:
        """Явный холостой запрос на маленьком синтетическом кадре — поднимает процесс (если
        ещё не поднят) и форсирует загрузку TRT-модели на GPU (~15-20с на первом реальном
        запросе, см. докстринг класса), результат отбрасывается. Возвращает время прогрева
        (с) — вызывающая сторона должна вызвать это ОДИН раз сразу после создания backend'а
        и показать пользователю ОТДЕЛЬНО от времени реальной сегментации (по замечанию
        пользователя, задача "3D-реконструкция объекта по кропам сетки ракурсов": "скорость
        сегментации SAM3 надо показывать без учёта загрузки и прогрева модели" — иначе первый
        замер несопоставим с последующими и вводит в заблуждение о реальной скорости метода).
        Идемпотентно по протоколу (`segment_batch` на уже прогретой модели просто отработает
        быстро), но вызывающая сторона обычно вызывает один раз за время жизни backend'а —
        см. `GridReconstructionDialog._on_segment_clicked`/`tools/reconstruct_grid.py::main`."""
        t0 = time.perf_counter()
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.segment_batch([dummy], prompt="warmup")
        return time.perf_counter() - t0

    def shutdown(self) -> None:
        """Останавливает фоновый процесс (модель выгружается) — вызывать при закрытии
        приложения, не при закрытии диалога (диалог переживает многократные show/hide, см.
        `gui/main_window.py::_on_photo_capture_button_clicked`)."""
        if self._process is not None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                self._process.kill()
            self._process = None
        if self._stderr_log_path is not None:
            self._stderr_log_path.unlink(missing_ok=True)
            self._stderr_log_path = None
