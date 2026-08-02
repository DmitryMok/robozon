"""Диалог "Загрузить сетку..." — тестирование 3D-реконструкции (visual hull +
G4 круглая проекция) на уже готовых 9-ракурсных сетках кропов, собранных
Webots-симулятором соседнего проекта `robozon` (Фаза 3-4 CV-пайплайна,
`assets/cv_grid_test_frames/grids_v2_cv_trigger/<type>/{grid.png,grid_bg.png,
grid.json}`). Пункт 3 запроса пользователя 2026-07-25 в задаче "3D-
реконструкция объекта по кропам сетки ракурсов (Фаза 5, CV-пайплайн
Webots)".

Сценарий (по уточнению пользователя 2026-07-25, вторая итерация) —
СЕГМЕНТАЦИЯ и РЕКОНСТРУКЦИЯ раздельные шаги, а не один клик:
  1. Загрузить папку сетки (файл/папка/drag&drop).
  2. Отдельно нажать «Сегментировать CV/HSV» и/или «Сегментировать SAM3» —
     каждая кнопка независимо считает маски + время сегментации, показывает
     превью сетки с контуром маски.
  3. С уже посчитанными масками — сколько угодно раз нажимать
     «Реконструировать» с разными галочками метода (polytope всегда,
     +exact, +voxel), не пересчитывая сегментацию заново — время
     реконструкции репортится отдельно от времени сегментации.

Немодальный — халл рисуется в Panel 1 главного окна (`model_panel`, как и
`PhotoCaptureDialog`), поэтому диалог не блокирует MainWindow.

Вся геометрия (риг камер Webots, конвертация в CameraPose, HSV/SAM3-
сегментация, visual hull, G4) — в `sim/grid_reconstruction.py` СОСЕДНЕГО
проекта `robozon` (уже используется его CLI-скриптом `tools/
reconstruct_grid.py`) — этот файл лишь тонкий Qt-слой поверх него, чтобы не
дублировать геометрию (уже трижды становившуюся источником реальных багов в
этой задаче) в двух местах. Требует `robozon` СИБЛИНГ-репозиторием (см.
`_ROBOZON_ROOT` ниже) — если его нет рядом, диалог остаётся доступным из
меню, но сообщает об отсутствии движка и не даёт загрузить сетку.

Источник данных сетки — `robozon/assets/cv_grid_test_frames/
grids_v2_cv_trigger/<type>/` (НЕ `05 Visuals/` — та папка содержит уже
готовые PNG-отчёты CLI-скрипта, не исходные `grid.json`/`grid.png`/
`grid_bg.png`)."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..panels.model_panel import ModelPanel

# --- Опциональная зависимость на соседний репозиторий robozon -------------
# roboson_tools/roboson_tools/gui/dialogs/<файл> -> repo root на 3 уровня выше.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROBOZON_ROOT = _REPO_ROOT.parent / "robozon"
_GRID_DATA_HINT = str(_ROBOZON_ROOT / "assets" / "cv_grid_test_frames" / "grids_v2_cv_trigger")
_GRID_ENGINE_IMPORT_ERROR: str | None = None
grid_reconstruction = None  # заполняется ниже при успешном импорте
load_layout = None
load_objects = None

if _ROBOZON_ROOT.is_dir():
    if str(_ROBOZON_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROBOZON_ROOT))
    try:
        from sim import grid_reconstruction  # type: ignore[no-redef]  # noqa: E402
        from sim.config import load_layout, load_objects  # type: ignore[no-redef]  # noqa: E402
    except Exception as exc:  # ImportError и всё, что тянет за собой окружение robozon
        _GRID_ENGINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _GRID_ENGINE_IMPORT_ERROR = f"репозиторий robozon не найден рядом ({_ROBOZON_ROOT})"

from ...segmentation.sam3_subprocess_backend import Sam3SubprocessBackend  # noqa: E402

_SEG_LABELS = {"cv": "CV/HSV", "sam3": "SAM3"}
# Варианты реконструкции, чей результат — облако точек (`points`), а не полигональный меш
# (`vertices`/`faces`) — voxel carving и torchhull (marching cubes без сохранённых граней в
# нашем возвращаемом словаре, см. sim/grid_reconstruction.py). Показываются в Panel 1 как
# GLScatterPlotItem (`model_panel.set_camera_hull_points`), не GLMeshItem.
_POINT_CLOUD_VARIANTS = {"cv_voxel", "cv_torchhull"}


def _bgr_to_pixmap(image_bgr: np.ndarray, max_side: int = 420) -> QPixmap:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    height, width, _c = rgb.shape
    qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class _SegmentationWorker(QThread):
    """Считает маски+наблюдения ОДНОГО пути сегментации (`path` — "cv" или
    "sam3") в фоне — SAM3 (subprocess к UavVisionLab) может занимать секунды
    даже на прогретой модели. HSV быстрый (доли секунды), но тоже уводится в
    поток для единообразия и чтобы не подвешивать GUI на всякий случай
    (первый холодный вызов на объекте с крупным изображением). Реконструкция
    (`_run_reconstruction`, метод диалога) — НЕ здесь, она использует уже
    готовые наблюдения синхронно (быстро, без подпроцесса)."""

    # path, masks, observations, dropped, elapsed_s, warmup_s (float|None — object, т.к. PyQt-
    # сигнал не пропускает None в слот float)
    finished_ok = pyqtSignal(str, dict, list, dict, float, object)
    finished_error = pyqtSignal(str, str)  # path, traceback

    def __init__(
        self,
        path: str,
        cells_meta: dict,
        grid_png: np.ndarray,
        bg_png: np.ndarray | None,
        tile_px: int,
        rig_cfg: dict,
        geoms: dict,
        sam3_backend: Sam3SubprocessBackend | None,
        sam3_prompt: str,
        needs_warmup: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._cells_meta = cells_meta
        self._grid_png = grid_png
        self._bg_png = bg_png
        self._tile_px = tile_px
        self._rig_cfg = rig_cfg
        self._geoms = geoms
        self._sam3_backend = sam3_backend
        self._sam3_prompt = sam3_prompt
        # Только для path=="sam3": первый вызов на свежесозданном backend'е должен явно
        # прогреть модель (~15-20с, загрузка TRT на GPU) ОТДЕЛЬНО от замера elapsed ниже —
        # см. Sam3SubprocessBackend.warmup, по замечанию пользователя ("скорость сегментации
        # SAM3 надо показывать без учёта загрузки и прогрева модели").
        self._needs_warmup = needs_warmup

    def run(self) -> None:
        try:
            warmup_elapsed: float | None = None
            if self._path == "sam3" and self._needs_warmup:
                warmup_elapsed = self._sam3_backend.warmup()

            t0 = time.perf_counter()
            if self._path == "cv":
                masks, dropped = grid_reconstruction.hsv_masks(
                    self._cells_meta, self._grid_png, self._bg_png, self._tile_px, self._rig_cfg
                )
                elapsed = time.perf_counter() - t0
            else:
                masks, dropped, elapsed = grid_reconstruction.sam3_masks(
                    self._sam3_backend, self._cells_meta, self._grid_png, self._tile_px, self._sam3_prompt
                )
            observations = grid_reconstruction.build_observations(
                masks, self._cells_meta, self._geoms, self._rig_cfg, self._tile_px
            )
            self.finished_ok.emit(self._path, masks, observations, dropped, elapsed, warmup_elapsed)
        except Exception:
            self.finished_error.emit(self._path, traceback.format_exc())


class _ReconstructionWorker(QThread):
    """Строит халл(ы) по УЖЕ готовым сегментациям (шаг 1) — в фоне, а не синхронно, как раньше:
    путь D (torchhull, GPU) может занимать секунды (JIT-компиляция CUDA-ядра при первом вызове
    процесса + сам carving — до нескольких секунд на объектах покрупнее, см. заметку задачи),
    заметно дольше polytope/exact/voxel (доли секунды на CPU). По образцу `_RecomputeWorker`
    (`main_window.py`) — та же логика, что там же строит torchhull для STL/Camera Mode."""

    finished_ok = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(
        self,
        seg: dict[str, dict],
        rig_cfg: dict,
        run_exact: bool,
        run_voxel: bool,
        run_torchhull: bool,
        torchhull_level: int,
        cells_meta: dict | None = None,
        geoms: dict | None = None,
        tile_px: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._seg = seg
        self._rig_cfg = rig_cfg
        self._run_exact = run_exact
        self._run_voxel = run_voxel
        self._run_torchhull = run_torchhull
        self._torchhull_level = torchhull_level
        # Для перспективной коррекции (reconstruct_path_perspective) нужны
        # masks+cells_meta+geoms+tile_px, а не готовые observations. Если
        # переданы — используем итеративный подход; иначе — старый путь
        # (build_observations уже вызван в _SegmentationWorker).
        self._cells_meta = cells_meta
        self._geoms = geoms
        self._tile_px = tile_px

    def run(self) -> None:
        try:
            results: dict[str, dict] = {}
            if "cv" in self._seg:
                # Уточнённые observations (с перспективной коррекцией) —
                # переиспользуем для voxel/torchhull, не только polytope.
                obs_cv_persp = self._build_perspective_observations("cv")
                res = grid_reconstruction.reconstruct_path(
                    obs_cv_persp, "CV/HSV (polytope)", budget_start=None,
                    also_exact=self._run_exact)
                results["cv_polytope"] = res
                res["dropped_cells"] = self._seg["cv"]["dropped"]
                res["segmentation_elapsed_s"] = self._seg["cv"]["elapsed"]
                # exact — отдельный вариант для Panel 1 (vertices/faces)
                if "exact_vertices" in res:
                    results["cv_exact"] = {
                        "vertices": res["exact_vertices"],
                        "faces": res["exact_faces"],
                        "recon_dims_mm": res.get("exact_dims_mm"),
                        "elapsed_total_s": res.get("exact_elapsed_s"),
                        "n_observations": res.get("n_observations"),
                        "dropped_cells": res.get("dropped_cells"),
                        "segmentation_elapsed_s": res.get("segmentation_elapsed_s"),
                    }
                if self._run_voxel:
                    bbox_mm = grid_reconstruction.voxel_bbox_from_polytope(res, self._rig_cfg, margin_ratio=1.0)
                    results["cv_voxel"] = grid_reconstruction.reconstruct_path_voxel(
                        obs_cv_persp, bbox_mm, "CV/HSV (voxel)", grid_resolution=64
                    )
                if self._run_torchhull:
                    bbox_d_mm = grid_reconstruction.voxel_bbox_from_polytope(res, self._rig_cfg, margin_ratio=0.1)
                    results["cv_torchhull"] = grid_reconstruction.reconstruct_path_torchhull(
                        obs_cv_persp, bbox_d_mm, "CV/HSV (torchhull)", level=self._torchhull_level
                    )
            if "sam3" in self._seg:
                obs_sam3_persp = self._build_perspective_observations("sam3")
                res = grid_reconstruction.reconstruct_path(
                    obs_sam3_persp, "SAM3 (polytope)", budget_start=None,
                    also_exact=self._run_exact)
                res["dropped_cells"] = self._seg["sam3"]["dropped"]
                res["segmentation_elapsed_s"] = self._seg["sam3"]["elapsed"]
                results["sam3_polytope"] = res
                if "exact_vertices" in res:
                    results["sam3_exact"] = {
                        "vertices": res["exact_vertices"],
                        "faces": res["exact_faces"],
                        "recon_dims_mm": res.get("exact_dims_mm"),
                        "elapsed_total_s": res.get("exact_elapsed_s"),
                        "n_observations": res.get("n_observations"),
                        "dropped_cells": res.get("dropped_cells"),
                        "segmentation_elapsed_s": res.get("segmentation_elapsed_s"),
                    }
            self.finished_ok.emit(results)
        except Exception:
            self._recon_error = traceback.format_exc()
            self.finished_error.emit(traceback.format_exc())

    def _build_perspective_observations(self, path: str) -> list:
        """Строит observations с перспективной коррекцией ракурса, если
        доступны cells_meta/geoms/tile_px; иначе — готовые observations из
        _SegmentationWorker (без коррекции). Возвращает список Observation —
        используется для ВСЕХ методов (polytope, exact, voxel, torchhull),
        чтобы коррекция применялась единообразно."""
        seg = self._seg[path]
        if (self._cells_meta is not None and self._geoms is not None
                and self._tile_px is not None
                and hasattr(grid_reconstruction, "build_observations_perspective")):
            obs, _ = grid_reconstruction.build_observations_perspective(
                seg["masks"], self._cells_meta, self._geoms, self._rig_cfg,
                self._tile_px)
            return obs
        return seg["observations"]


class GridReconstructionDialog(QDialog):
    """См. докстринг модуля. Хранится как атрибут MainWindow (как
    `PhotoCaptureDialog`) и переиспользуется между кликами кнопки."""

    shown = pyqtSignal()
    closing = pyqtSignal()

    def __init__(self, model_panel: ModelPanel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Загрузить сетку — тест реконструкции по кропам Webots-рига")
        self.resize(820, 780)
        self.setAcceptDrops(True)

        self._model_panel = model_panel
        self._grid_dir: Path | None = None
        self._obj_type: str | None = None
        self._rig_cfg: dict | None = None
        self._geoms: dict | None = None
        self._objects_cfg: dict | None = None
        self._sam3_backend: Sam3SubprocessBackend | None = None
        # True после первого успешного прогрева/сегментации SAM3 на ТЕКУЩЕМ backend'е — пока
        # backend не пересоздан (см. shutdown_segmentation_backend), последующие сегментации
        # (даже на других загруженных сетках) уже быстрые, прогревать заново не нужно.
        self._sam3_warmed_up = False
        self._seg_worker: _SegmentationWorker | None = None
        self._recon_worker: _ReconstructionWorker | None = None

        self._cells_meta: dict | None = None
        self._grid_png: np.ndarray | None = None
        self._bg_png: np.ndarray | None = None
        self._tile_px: int | None = None

        # Результаты сегментации по пути — "cv"/"sam3" (см. докстринг класса) — переживают
        # сколько угодно кликов «Реконструировать» подряд, сбрасываются только новой загрузкой
        # сетки или повторным кликом «Сегментировать» на том же пути.
        self._seg: dict[str, dict] = {}
        self._recon_results: dict[str, dict] = {}
        self._shown_variant: str | None = None

        root = QVBoxLayout(self)

        if _GRID_ENGINE_IMPORT_ERROR is not None:
            warn = QLabel(
                "Движок реконструкции недоступен — не найден или не импортируется "
                f"репозиторий robozon рядом с roboson_tools ({_ROBOZON_ROOT}):\n"
                f"{_GRID_ENGINE_IMPORT_ERROR}"
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #c62828;")
            root.addWidget(warn)

        load_row = QHBoxLayout()
        self.load_button = QPushButton("Загрузить сетку...")
        self.load_button.clicked.connect(self._on_load_clicked)
        self.load_button.setEnabled(_GRID_ENGINE_IMPORT_ERROR is None)
        load_row.addWidget(self.load_button)
        self.grid_status_label = QLabel(
            "Не загружено — выберите папку объекта с grid.json+grid.png (обычно "
            f"{_GRID_DATA_HINT}\\<тип>), либо перетащите её сюда. НЕ путать с "
            "готовыми отчётами в «05 Visuals» — там нет исходных данных для реконструкции."
        )
        self.grid_status_label.setWordWrap(True)
        load_row.addWidget(self.grid_status_label, stretch=1)
        root.addLayout(load_row)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(140)
        root.addWidget(self.preview_label)

        # --- Шаг 1: сегментация (раздельные кнопки CV/SAM3, каждая со своим статусом+превью) --
        seg_box = QGroupBox("1. Сегментация (независимо по каждому пути)")
        seg_layout = QHBoxLayout(seg_box)
        self._seg_widgets: dict[str, dict] = {}
        for path in ("cv", "sam3"):
            col = QVBoxLayout()
            button = QPushButton(f"Сегментировать {_SEG_LABELS[path]}")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, p=path: self._on_segment_clicked(p))
            status = QLabel("не посчитано")
            status.setWordWrap(True)
            preview = QLabel()
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setMinimumSize(240, 240)
            col.addWidget(button)
            col.addWidget(status)
            col.addWidget(preview, stretch=1)
            seg_layout.addLayout(col)
            self._seg_widgets[path] = {"button": button, "status": status, "preview": preview}
        root.addWidget(seg_box)

        # --- Шаг 2: реконструкция по уже посчитанным маскам ---------------------------------
        recon_box = QGroupBox("2. Реконструкция (по уже посчитанным маскам — пробуйте разные "
                               "методы без пересчёта сегментации)")
        recon_layout = QVBoxLayout(recon_box)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Метод пересечения:"))
        self.exact_checkbox = QCheckBox("+ exact (невыпуклый)")
        self.voxel_checkbox = QCheckBox("+ voxel carving (только CV/HSV)")
        self.torchhull_checkbox = QCheckBox("+ torchhull (GPU, только CV/HSV)")
        torchhull_available = grid_reconstruction is not None and grid_reconstruction.TORCHHULL_AVAILABLE
        if not torchhull_available:
            self.torchhull_checkbox.setEnabled(False)
            self.torchhull_checkbox.setToolTip(
                "Недоступно в этом Python-окружении: нужны torch+torchhull+CUDA. "
                "В обычном Windows-венве roboson_tools их нет — запустите GUI из окружения, "
                "где они установлены (см. roboson_tools/bench_torchhull.py про WSL .venv-cv "
                "+ CUDA Toolkit + gcc-12), либо используйте CLI tools/reconstruct_grid.py "
                "--torchhull из WSL."
            )
        method_row.addWidget(self.exact_checkbox)
        method_row.addWidget(self.voxel_checkbox)
        method_row.addWidget(self.torchhull_checkbox)
        method_row.addStretch(1)
        recon_layout.addLayout(method_row)

        run_row = QHBoxLayout()
        self.reconstruct_button = QPushButton("Реконструировать")
        self.reconstruct_button.clicked.connect(self._on_reconstruct_clicked)
        self.reconstruct_button.setEnabled(False)
        run_row.addWidget(self.reconstruct_button)
        run_row.addWidget(QLabel("Показать в Panel 1:"))
        self.display_combo = QComboBox()
        self.display_combo.setEnabled(False)
        self.display_combo.currentIndexChanged.connect(self._on_display_changed)
        run_row.addWidget(self.display_combo, stretch=1)
        recon_layout.addLayout(run_row)

        roll_row = QHBoxLayout()
        self.roll_axis_checkbox = QCheckBox("Показать ось + плоскость переката (для выбранного варианта)")
        self.roll_axis_checkbox.toggled.connect(lambda _checked: self._update_roll_axis_display())
        roll_row.addWidget(self.roll_axis_checkbox)
        roll_row.addStretch(1)
        recon_layout.addLayout(roll_row)
        root.addWidget(recon_box)

        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        root.addWidget(self.results_text, stretch=1)

    # --- жизненный цикл SAM3-подпроцесса (по образцу PhotoCaptureDialog) --------------

    def shutdown_segmentation_backend(self) -> None:
        if self._sam3_backend is not None:
            self._sam3_backend.shutdown()
            self._sam3_backend = None
            self._sam3_warmed_up = False

    def showEvent(self, event) -> None:  # noqa: ANN001 — сигнатура Qt
        super().showEvent(event)
        self.shown.emit()

    def closeEvent(self, event) -> None:  # noqa: ANN001 — сигнатура Qt
        self.closing.emit()
        super().closeEvent(event)

    # --- drag & drop --------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        self._load_grid_path(Path(urls[0].toLocalFile()))

    # --- загрузка сетки -------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        path_str = QFileDialog.getExistingDirectory(
            self, "Выбрать папку сетки (с grid.json+grid.png)", _GRID_DATA_HINT
        )
        if path_str:
            self._load_grid_path(Path(path_str))

    def _load_grid_path(self, path: Path) -> None:
        """Принимает файл grid.json/grid.png ИЛИ папку объекта (`<type>/`,
        как в `assets/cv_grid_test_frames/grids_v2_cv_trigger/<type>/`) —
        drag&drop может принести и то, и другое. НЕ принимает `05 Visuals/`
        (там только готовые PNG-отчёты, без grid.json/grid_bg.png)."""
        grid_dir = path if path.is_dir() else path.parent
        grid_json = grid_dir / "grid.json"
        grid_png_path = grid_dir / "grid.png"
        if not grid_json.exists() or not grid_png_path.exists():
            self.grid_status_label.setText(
                f"В {grid_dir} нет grid.json+grid.png — не похоже на папку сетки ракурсов. "
                f"Исходные данные обычно лежат в {_GRID_DATA_HINT}\\<тип>, "
                "НЕ в «05 Visuals» (там только готовые отчёты)."
            )
            return

        try:
            import json
            with open(grid_json, encoding="utf-8") as f:
                grid_meta = json.load(f)
            cells_meta = grid_meta["cells"]
        except Exception as exc:
            self.grid_status_label.setText(f"Не удалось прочитать {grid_json}: {exc}")
            return

        grid_png = cv2.imread(str(grid_png_path))
        if grid_png is None:
            self.grid_status_label.setText(f"Не удалось прочитать {grid_png_path}")
            return
        bg_png_path = grid_dir / "grid_bg.png"
        bg_png = cv2.imread(str(bg_png_path)) if bg_png_path.exists() else None

        if self._rig_cfg is None:
            try:
                self._rig_cfg = load_layout()["cv_rig"]
                self._objects_cfg = load_objects()
                self._geoms = grid_reconstruction.build_camera_geoms(self._rig_cfg)
            except Exception as exc:
                self.grid_status_label.setText(
                    f"Не удалось загрузить конфиг рига robozon (config/layout.yaml): {exc}"
                )
                return

        self._grid_dir = grid_dir
        self._obj_type = grid_dir.name
        self._cells_meta = cells_meta
        self._grid_png = grid_png
        self._bg_png = bg_png
        any_meta = next(iter(cells_meta.values()))
        self._tile_px = round(any_meta["crop_side"] * any_meta["scale"])

        has_bg = bg_png is not None
        self.grid_status_label.setText(
            f"Загружено: {grid_dir} — {len(cells_meta)} ячеек"
            + ("" if has_bg else " (нет grid_bg.png — CV/HSV недоступна, только SAM3)")
        )
        self.preview_label.setPixmap(_bgr_to_pixmap(grid_png))

        self._seg_widgets["cv"]["button"].setEnabled(has_bg)
        self._seg_widgets["cv"]["status"].setText("не посчитано" if has_bg else "нет grid_bg.png")
        self._seg_widgets["cv"]["preview"].clear()
        self._seg_widgets["sam3"]["button"].setEnabled(True)
        self._seg_widgets["sam3"]["status"].setText("не посчитано")
        self._seg_widgets["sam3"]["preview"].clear()

        self._seg = {}
        self._recon_results = {}
        self.results_text.setPlainText("")
        self.display_combo.clear()
        self.display_combo.setEnabled(False)
        self.reconstruct_button.setEnabled(False)
        self.clear_hull_display()

    # --- шаг 1: сегментация -----------------------------------------------------------

    def _on_segment_clicked(self, path: str) -> None:
        if self._grid_dir is None or self._rig_cfg is None:
            return
        needs_warmup = False
        if path == "sam3":
            if self._sam3_backend is None:
                self._sam3_backend = Sam3SubprocessBackend()
                self._sam3_warmed_up = False
            needs_warmup = not self._sam3_warmed_up

        prompt = "object"

        self._seg_widgets[path]["button"].setEnabled(False)
        self._seg_widgets[path]["status"].setText(
            "Прогреваю модель (однократно, ~15-20с)..." if needs_warmup else "Считаю..."
        )
        self.setCursor(Qt.CursorShape.WaitCursor)

        worker = _SegmentationWorker(
            path, self._cells_meta, self._grid_png, self._bg_png, self._tile_px,
            self._rig_cfg, self._geoms, self._sam3_backend, prompt,
            needs_warmup=needs_warmup, parent=self,
        )
        worker.finished_ok.connect(self._on_segment_finished)
        worker.finished_error.connect(self._on_segment_error)
        # Держим ссылку на воркер по пути — второй путь можно запустить, пока первый ещё
        # считается (SAM3 может идти секунды, HSV — доли секунды, независимые кнопки).
        setattr(self, f"_seg_worker_{path}", worker)
        worker.start()

    def _on_segment_error(self, path: str, message: str) -> None:
        self.unsetCursor()
        self._seg_widgets[path]["button"].setEnabled(True)
        self._seg_widgets[path]["status"].setText(f"Ошибка: {message.splitlines()[-1]}")

    def _on_segment_finished(
        self, path: str, masks: dict, observations: list, dropped: dict, elapsed: float,
        warmup_elapsed: float | None,
    ) -> None:
        self.unsetCursor()
        self._seg_widgets[path]["button"].setEnabled(True)
        if path == "sam3":
            self._sam3_warmed_up = True
        self._seg[path] = {
            "masks": masks, "observations": observations, "dropped": dropped, "elapsed": elapsed,
        }
        n_ok = len(masks)
        n_total = len(self._cells_meta)
        # elapsed — ТОЛЬКО сама сегментация (без прогрева, см. _SegmentationWorker.run) —
        # по замечанию пользователя, время прогрева показываем отдельной строкой, один раз.
        status_lines = [f"{n_ok}/{n_total} ячеек, {elapsed:.3f}с"]
        if warmup_elapsed is not None:
            status_lines.append(f"(+ прогрев модели {warmup_elapsed:.1f}с, однократно)")
        if dropped:
            status_lines.append("отброшено: " + ", ".join(sorted(dropped)))
        self._seg_widgets[path]["status"].setText("\n".join(status_lines))

        color = grid_reconstruction.MASK_COLOR_CV if path == "cv" else grid_reconstruction.MASK_COLOR_SAM3
        mosaic = grid_reconstruction.render_cells_mosaic(
            self._cells_meta, self._grid_png, masks, color, _SEG_LABELS[path], self._tile_px, thumb_px=90
        )
        self._seg_widgets[path]["preview"].setPixmap(_bgr_to_pixmap(mosaic, max_side=260))

        self.reconstruct_button.setEnabled(True)

    # --- шаг 2: реконструкция (фон — путь D/torchhull может занимать секунды) -----------

    def _on_reconstruct_clicked(self) -> None:
        if not self._seg:
            self.results_text.setPlainText("Сначала посчитайте сегментацию (шаг 1)")
            return

        run_exact = self.exact_checkbox.isChecked()
        run_voxel = self.voxel_checkbox.isChecked() and "cv" in self._seg
        run_torchhull = (
            self.torchhull_checkbox.isChecked() and self.torchhull_checkbox.isEnabled()
            and "cv" in self._seg
        )

        self.reconstruct_button.setEnabled(False)
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.results_text.setPlainText("Считаю...")

        self._recon_worker = _ReconstructionWorker(
            self._seg, self._rig_cfg, run_exact, run_voxel, run_torchhull,
            torchhull_level=7,
            cells_meta=self._cells_meta, geoms=self._geoms, tile_px=self._tile_px,
            parent=self,
        )
        self._recon_worker.finished_ok.connect(self._on_reconstruct_finished)
        self._recon_worker.finished_error.connect(self._on_reconstruct_error)
        self._recon_worker.start()

    def _on_reconstruct_error(self, message: str) -> None:
        self.unsetCursor()
        self.reconstruct_button.setEnabled(True)
        self.results_text.setPlainText(f"Ошибка реконструкции:\n{message}")

    def _on_reconstruct_finished(self, results: dict) -> None:
        self.unsetCursor()
        self.reconstruct_button.setEnabled(True)
        self._show_results(results)

    def _show_results(self, results: dict) -> None:
        self._recon_results = results

        expected = None
        gt_dims = None
        if self._objects_cfg is not None and self._obj_type in self._objects_cfg:
            try:
                category = self._objects_cfg[self._obj_type]["category"]
                expected = grid_reconstruction.expected_verdict(category)
                gt_dims = grid_reconstruction.ground_truth_dims_mm(self._obj_type)
            except Exception:
                pass  # объект не найден в meshes/ или objects.yaml — просто без эталона

        lines = [f"Объект: {self._obj_type}"]
        if gt_dims is not None:
            lines.append(
                f"Эталон STL: {gt_dims[0]:.0f}x{gt_dims[1]:.0f}x{gt_dims[2]:.0f} мм, "
                f"ожидаемый вердикт: {'/'.join(expected)}"
            )
        lines.append("")

        display_items: list[tuple[str, str]] = []
        for key, label in (
            ("cv_polytope", "CV/HSV — polytope"),
            ("cv_exact", "CV/HSV — exact"),
            ("cv_voxel", "CV/HSV — voxel"),
            ("cv_torchhull", "CV/HSV — torchhull"),
            ("sam3_polytope", "SAM3 — polytope"),
            ("sam3_exact", "SAM3 — exact"),
        ):
            res = results.get(key)
            if res is None:
                continue
            if "error" in res:
                lines.append(f"[{label}] ошибка: {res['error']}")
                continue
            dims = res.get("recon_dims_mm")
            dims_txt = "x".join(f"{d:.0f}" for d in dims) + " мм" if dims else "?"
            verdict = res.get("verdict", "—")
            ok_txt = ""
            if expected is not None and verdict != "—":
                ok_txt = " OK" if verdict in expected else " НЕ СОВПАДАЕТ С ЭТАЛОНОМ"
            # У reconstruct_path(budget_start=None) готовой суммы нет — считаем сами
            # (hull+G4, БЕЗ времени сегментации — оно репортится отдельно, см. ниже).
            elapsed = res.get("elapsed_total_s")
            if elapsed is None and "elapsed_hull_s" in res and "elapsed_g4_s" in res:
                elapsed = res["elapsed_hull_s"] + res["elapsed_g4_s"]
            elapsed_txt = f"{elapsed:.3f}с" if elapsed is not None else "?"
            seg_elapsed = res.get("segmentation_elapsed_s")
            seg_txt = f", сегментация={seg_elapsed:.3f}с" if seg_elapsed is not None else ""
            lines.append(
                f"[{label}] n_obs={res.get('n_observations')} verdict={verdict} "
                f"(k={res.get('k')}){ok_txt} dims={dims_txt} реконструкция={elapsed_txt}{seg_txt}"
            )
            if res.get("dropped_cells"):
                lines.append(f"    отброшено при сегментации: {list(res['dropped_cells'])}")
            # Перекат (G4): время — ОТДЕЛЬНО от общего времени реконструкции выше (elapsed_g4_s
            # — только сама проверка круглой проекции на уже построенном халле, без hull/carve).
            axis = res.get("roll_axis_dir")
            g4_elapsed = res.get("elapsed_g4_s")
            g4_txt = f"{g4_elapsed:.3f}с" if g4_elapsed is not None else "?"
            if axis is not None:
                lines.append(
                    f"    перекат: ось=({axis[0]:.3f},{axis[1]:.3f},{axis[2]:.3f}) время={g4_txt}"
                )
            else:
                lines.append(f"    перекат: ось не найдена (not_round) время={g4_txt}")
            if "exact_dims_mm" in res:
                ed = res["exact_dims_mm"]
                lines.append(
                    f"    + exact dims={ed[0]:.0f}x{ed[1]:.0f}x{ed[2]:.0f} мм "
                    f"время={res.get('exact_elapsed_s', 0):.3f}с"
                )
            display_items.append((key, f"{label}: {verdict} ({dims_txt})"))

        self.results_text.setPlainText("\n".join(lines))

        self.display_combo.blockSignals(True)
        self.display_combo.clear()
        for key, label in display_items:
            self.display_combo.addItem(label, key)
        self.display_combo.setEnabled(bool(display_items))
        self.display_combo.blockSignals(False)
        if display_items:
            self.display_combo.setCurrentIndex(0)
            self._show_variant(display_items[0][0])
        else:
            self.clear_hull_display()

    # --- отображение в Panel 1 --------------------------------------------------------

    def _on_display_changed(self, index: int) -> None:
        if index < 0:
            return
        key = self.display_combo.itemData(index)
        if key:
            self._show_variant(key)

    def clear_hull_display(self) -> None:
        self._model_panel.set_camera_hull_polytope(None, None)
        self._model_panel.set_camera_hull_exact(None, None)
        self._model_panel.set_camera_hull_points(None)
        self._model_panel.set_roll_axis(None)
        self._model_panel.set_slice_plane(None)

    def _show_variant(self, key: str) -> None:
        """Показывает ОДИН выбранный вариант реконструкции (CV/HSV-polytope,
        CV/HSV-exact, CV/HSV-voxel, CV/HSV-torchhull, SAM3-polytope, SAM3-exact)
        в Panel 1 — раздельно, не все сразу, чтобы не путать пользователя
        вершинами разных методов в одной сцене (тот же принцип, что раздельные
        PNG-отчёты по CV/SAM3 в CLI, `robozon/tools/reconstruct_grid.py`).

        exact — отдельный вариант (сохранённые vertices/faces из carve_exact,
        не только габариты) — показывает вогнутости силуэта, не упрощая до
        выпуклой оболочки, как polytope.

        Дополнительно прячет STL-объект и плоскость среза (по замечанию
        пользователя 2026-07-25 — при показе реконструкции по сетке в
        Panel 1 они мешают, это разные, не связанные системы координат:
        халл — в СК рига камер, STL/срез — в СК самой модели). Восстанавли-
        ваются при закрытии диалога (`MainWindow._on_grid_dialog_closing`),
        не здесь — пока диалог открыт, Panel 1 целиком в его распоряжении."""
        self._shown_variant = key
        self.clear_hull_display()
        res = self._recon_results.get(key)
        if res is None:
            return
        self._model_panel.set_object_visible(False)
        self._model_panel.set_hull_slices(None)
        if key in _POINT_CLOUD_VARIANTS:
            self._model_panel.set_camera_hull_points(res.get("points"))
        elif "vertices" in res and "faces" in res:
            self._model_panel.set_camera_hull_polytope(res["vertices"], res["faces"])
        self._update_roll_axis_display()

    def _update_roll_axis_display(self) -> None:
        """Ось+плоскость переката (G4, `roll_axis_dir` из `sim/grid_reconstruction.py::
        reconstruct_path`/`reconstruct_path_voxel`) для ТЕКУЩЕГО показанного варианта
        (`_shown_variant`) — переключается чекбоксом независимо от смены варианта в
        `display_combo` (тот вызывает `_show_variant`, который в конце тоже вызывает этот
        метод, чтобы ось следовала за выбором варианта). Центр — bbox-центр САМОГО халла,
        не мировое начало координат сцены (то — точка наблюдения рига камер, далеко от
        объекта, см. docstring `ModelPanel.set_roll_axis`)."""
        if not self.roll_axis_checkbox.isChecked() or self._shown_variant is None:
            self._model_panel.set_roll_axis(None)
            self._model_panel.set_slice_plane(None)
            return
        res = self._recon_results.get(self._shown_variant)
        axis = res.get("roll_axis_dir") if res is not None else None
        if axis is None:
            self._model_panel.set_roll_axis(None)
            self._model_panel.set_slice_plane(None)
            return
        verts = res.get("points") if self._shown_variant in _POINT_CLOUD_VARIANTS else res.get("vertices")
        if verts is not None and len(verts) > 0:
            verts = np.asarray(verts)
            center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        else:
            center = np.zeros(3)
        self._model_panel.set_roll_axis(axis, center)
        self._model_panel.set_slice_plane(center, np.asarray(axis))
