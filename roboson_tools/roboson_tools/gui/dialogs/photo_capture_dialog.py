"""Диалог загрузки реальных фото, расчёта позы камеры по ArUco-меткам (этап 1), сегментации
объекта SAM3 (этап 2) и 3D-реконструкции по маске+позе (этап 3, задача "3D-реконструкция объекта
по реальным фото — от маски+позы к visual hull"). Немодальный — результат реконструкции рисуется
в Panel 1 главного окна (`model_panel`, передаётся в конструктор), поэтому диалог не должен
блокировать MainWindow.

См. `roboson_tools/pose/` — вся геометрия позы, `roboson_tools/segmentation/` — сегментация,
`roboson_tools/visual_hull/pose_carving.py` — сама реконструкция; этот файл — тонкий Qt-слой
поверх них.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...pose.aruco_estimator import PoseEstimate, detect_markers, estimate_camera_pose
from ...pose.board_registration import bundle_adjust_registration, register_boards, merge_layout
from ...pose.calibration import calibrate_camera
from ...pose.calibration_cache import (
    CalibrationCache,
    calibration_cache_path,
    load_calibration_cache,
    save_calibration_cache,
)
from ...pose.camera_pose import CAMERA_INTRINSICS_PATH, CameraIntrinsics, load_camera_intrinsics
from ...pose.marker_layout import (
    ARUCO_MARKERS_PATH,
    BELT_BOARD_PATH,
    MIRROR_BOARD_PATH,
    MarkerLayout,
    load_all_boards,
    load_marker_layout,
)
from ...pose.photo_item import PhotoItem
from ...segmentation.base import SegmentationResult
from ...segmentation.config import load_classes_text, load_segmentation_config, save_classes_text
from ...segmentation.region_filter import (
    belt_and_mirror_regions,
    classify_mask_region,
    crop_to_regions,
    fit_belt_up_normal,
    fit_mirror_plane,
    select_region_candidates,
    uncrop_mask,
)
from ...segmentation.sam3_subprocess_backend import Sam3SubprocessBackend
from ...visual_hull import pose_carving
from ..panels.metrics_panel import MetricsPanel
from ..panels.model_panel import ModelPanel
from ..panels.silhouette_panel import SilhouettePanel
from ..panels.visual_hull_panel import VisualHullPanel

# Порог ошибки репроекции (% от короткой стороны кадра — см. pose/aruco_estimator.py), выше
# которого рекомендуется (пере)калибровка. Подобран по опыту: у реально откалиброванной камеры
# на этом риге ошибка обычно 0.1-0.3% (см. заметку задачи), у грубой прикидки интринсиков —
# заметно выше.
_CALIBRATION_RECOMMENDED_THRESHOLD_PCT = 0.5
_REFERENCE_BOARD = "belt_left"  # см. assets/belt_board/belt_board.yaml — произвольный, но
# фиксированный выбор опорной доски для board_registration.register_boards

# Допуск (px) для отражённых наблюдений в 3D-реконструкции — см. _build_observations.
_MIRROR_MARGIN_PX = 10.0

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_STATUS_LABELS = {
    "ok": "OK",
    "insufficient_markers": "мало меток",
    "solve_failed": "ошибка PnP",
    "read_failed": "не читается",
}
_AXIS_LENGTH_MM = 50.0


class _PoseEstimationWorker(QThread):
    """Загружает фото (если ещё не загружено — см. ниже) и считает позу в фоне — по образцу
    `_MeshTrueDimsWorker` (`gui/main_window.py`): входы захватываются в конструкторе, `run()` не
    трогает виджеты, результат уходит одним сигналом. Отмена не нужна — детекция ArUco+solvePnP
    на фото занимает миллисекунды, это не тяжёлый подпроцесс (в отличие от `search_dialog.py`).
    """

    results_ready = pyqtSignal(list)  # list[tuple[PhotoItem, PoseEstimate, np.ndarray | None]]

    def __init__(
        self,
        items: list[PhotoItem],
        layout: MarkerLayout,
        intrinsics: CameraIntrinsics,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._items = items
        self._layout = layout
        self._intrinsics = intrinsics

    def run(self) -> None:
        results: list[tuple[PhotoItem, PoseEstimate, np.ndarray | None]] = []
        for item in self._items:
            # При пересчёте под обновлённый конфиг (см. _on_reload_config_clicked) фото уже
            # загружено — не читаем файл с диска второй раз.
            image = item.image_bgr if item.image_bgr is not None else cv2.imread(str(item.path))
            if image is None:
                estimate = PoseEstimate(
                    pose=None, status="read_failed", message="не удалось прочитать файл"
                )
            else:
                estimate = estimate_camera_pose(image, self._layout, self._intrinsics)
            results.append((item, estimate, image))
        self.results_ready.emit(results)


def _rough_intrinsics_for(resolution_px: tuple[int, int]) -> CameraIntrinsics:
    """Грубая стартовая прикидка интринсиков ТОЛЬКО из размера кадра (fx=fy=max(сторона),
    главная точка в центре) — используется как затравка для `board_registration.register_boards`
    ДО калибровки (solvePnP устойчив к не идеально точной K) и как `initial_intrinsics` для
    `calibrate_camera`, если ничего лучше пока нет (см. `pose/calibration.py` про то, почему
    неплоской раскладке обязательно нужно начальное приближение)."""
    n_rows, n_cols = resolution_px
    max_dim = float(max(n_rows, n_cols))
    return CameraIntrinsics(fx=max_dim, fy=max_dim, cx=n_cols / 2.0, cy=n_rows / 2.0, resolution_px=resolution_px)


class _CalibrationWorker(QThread):
    """Регистрация 4 досок рига (`board_registration.register_boards`) + калибровка камеры
    (`calibration.calibrate_camera`) по уже загруженным фото, в фоне — по образцу
    `_PoseEstimationWorker`. Два прохода регистрации: первый — с грубыми/текущими интринсиками
    (нужны только чтобы решить solvePnP каждой доски отдельно), второй — с уже откалиброванными
    (уточняет положение досок) — см. проверку на реальных фото при разработке этой задачи,
    заметка "3D-реконструкция объекта по реальным фото...".

    Третий проход — `bundle_adjust_registration` (задача "Реализовать bundle adjustment для
    уточнения МНК-плоскости зеркала", 2026-07-26): совместная МНК-минимизация репроекционной
    ошибки по ВСЕМ загруженным фото и ВСЕМ доскам сразу поверх результата `reg2` (даёт начальное
    приближение) — точнее, чем попарное усреднение переходов `register_boards`, особенно для
    зеркала (меньше всего мостиковых фото, самая длинная цепочка до опорной доски ленты). Именно
    отсюда берётся точность плоскости зеркала для `region_filter.fit_mirror_plane`/
    `pose_carving` при реконструкции через отражение — см. заметку задачи про systematic ошибку
    зеркальных наблюдений на phone2."""

    finished_calibration = pyqtSignal(object, str)  # CalibrationCache | None, error

    def __init__(
        self,
        images_bgr: list[np.ndarray],
        boards: dict[str, MarkerLayout],
        current_intrinsics: CameraIntrinsics,
        source_photo_names: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._images = images_bgr
        self._boards = boards
        self._current_intrinsics = current_intrinsics
        self._source_photo_names = source_photo_names

    def run(self) -> None:
        if not self._images:
            self.finished_calibration.emit(None, "нет загруженных фото с прочитанным изображением")
            return
        resolution_px = self._images[0].shape[:2]
        bootstrap_intrinsics = (
            self._current_intrinsics
            if self._current_intrinsics.resolution_px == resolution_px
            else _rough_intrinsics_for(resolution_px)
        )

        reg1 = register_boards(self._images, self._boards, bootstrap_intrinsics, reference_board=_REFERENCE_BOARD)
        merged1 = merge_layout(self._boards, reg1)
        if len(merged1.corners_world_mm) == 0:
            self.finished_calibration.emit(
                None, f"регистрация не нашла ни одной доски: {reg1.message}"
            )
            return

        calib = calibrate_camera(self._images, merged1, initial_intrinsics=bootstrap_intrinsics)
        if calib.intrinsics is None:
            self.finished_calibration.emit(None, f"калибровка не удалась: {calib.message}")
            return

        # Уточняющий проход: те же доски, но уже с откалиброванными интринсиками — точнее
        # определяет положение досок (особенно тех, что видны не на всех фото).
        reg2 = register_boards(self._images, self._boards, calib.intrinsics, reference_board=_REFERENCE_BOARD)

        # Bundle adjustment поверх reg2 — совместная МНК-минимизация репроекции по всем фото и
        # всем доскам сразу (точнее попарного усреднения переходов), см. докстринг класса.
        reg3, ba_report = bundle_adjust_registration(
            self._images, self._boards, calib.intrinsics, reg2
        )
        merged3 = merge_layout(self._boards, reg3)

        cache = CalibrationCache(
            layout=merged3,
            intrinsics=calib.intrinsics,
            reference_board=_REFERENCE_BOARD,
            rms_reprojection_error_pct=calib.rms_reprojection_error_pct,
            source_photos=self._source_photo_names,
        )
        message = calib.message
        if reg3.unregistered_boards:
            message += f"; не зарегистрированы (мало мостиковых фото): {reg3.unregistered_boards}"
        if ba_report.n_frames_used > 0:
            message += (
                f"; bundle adjustment по {ba_report.n_frames_used} фото: RMS репроекции "
                f"{ba_report.rms_reprojection_px_before:.2f}px -> "
                f"{ba_report.rms_reprojection_px_after:.2f}px"
            )
        # Второй элемент сигнала — всегда информационное сообщение (не обязательно ошибка);
        # `cache is None` в обработчике отличает провал от успеха с предупреждением.
        self.finished_calibration.emit(cache, message)


@dataclass
class _SegmentationJob:
    """Один запрос к SAM3 — привязан к КОНКРЕТНОЙ области кадра (belt/mirror/unfiltered), не к
    фото целиком. См. докстринг модуля `_SegmentationWorker` — раньше лента и зеркало
    конкурировали за ЕДИНСТВЕННУЮ лучшую детекцию в одном запросе, и прямой вид (обычно
    увереннее — крупнее, лучше освещён) побеждал ВСЕГДА, из-за чего отражение никогда не
    сегментировалось, даже когда оно физически видно на каждом фото (см. заметку задачи —
    реальный кейс пользователя, phone2). Раздельные запросы дают каждой области честный шанс."""

    item: PhotoItem
    region: str  # "belt" / "mirror" / "unfiltered" (чекбокс маскировки выключен)
    image_bgr: np.ndarray  # уже кадрированный (region_filter.crop_to_regions) кадр для SAM3
    # (x0, y0) кадра `image_bgr` в координатах ПОЛНОГО фото (`item.image_bgr`) — None для
    # "unfiltered" (там `image_bgr` и есть полный кадр, кадрировать было не по чему). Нужен,
    # чтобы вернуть маску результата обратно в полнокадровые координаты, см.
    # `region_filter.uncrop_mask` и `PhotoCaptureDialog._on_segmentation_finished`.
    offset: tuple[int, int] | None = None


class _SegmentationWorker(QThread):
    """Считает маски SAM3 для пачки заданий (`_SegmentationJob`) за один запрос к уже
    поднятому backend'у (модель грузится один раз при первом запросе и держится живой между
    кликами — см. `segmentation/sam3_subprocess_backend.py`; backend — общий на диалог, НЕ
    создаётся заново здесь), по образцу `_PoseEstimationWorker`. В отличие от неё, весь батч
    может упасть целиком (subprocess не поднялся/умер, GPU занят и т.п.) — тогда `error`
    непустой, а `results` пуст."""

    finished_batch = pyqtSignal(list, str)  # list[tuple[_SegmentationJob, SegmentationResult | None]], error

    def __init__(
        self,
        jobs: list[_SegmentationJob],
        backend: Sam3SubprocessBackend,
        prompt: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._backend = backend
        self._prompt = prompt

    def run(self) -> None:
        try:
            results = self._backend.segment_batch(
                [job.image_bgr for job in self._jobs], prompt=self._prompt
            )
        except Exception as exc:  # subprocess/окружение — фатально для всего батча
            self.finished_batch.emit([], str(exc))
            return
        self.finished_batch.emit(list(zip(self._jobs, results)), "")


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    height, width, _channels = rgb.shape
    qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
    # .copy() — QImage не копирует буфер `rgb` сам, а он локальный и будет собран GC сразу
    # после return, иначе pixmap ссылается на уже освобождённую память.
    return QPixmap.fromImage(qimage.copy())


def _rotation_to_euler_deg(rotation: np.ndarray) -> np.ndarray:
    euler_angles, _mtx_r, _mtx_q, _qx, _qy, _qz = cv2.RQDecomp3x3(rotation)
    return np.array(euler_angles)


class PhotoCaptureDialog(QDialog):
    # Немодальный диалог живёт весь сеанс работы GUI (show/hide, не создаётся заново — см.
    # MainWindow._on_photo_capture_button_clicked) — эти сигналы дают MainWindow знать, когда
    # диалог реально открыт/закрыт, чтобы на это время блокировать панель управления STL
    # (см. заметку задачи: изменение ориентации/STL-настроек во время работы с фото не имеет
    # смысла и конфликтует с общим 3D-каналом Panel 1, см. _on_reconstruct_clicked).
    shown = pyqtSignal()
    closing = pyqtSignal()

    def __init__(
        self,
        marker_layout: MarkerLayout,
        intrinsics: CameraIntrinsics,
        model_panel: ModelPanel,
        silhouette_panel: SilhouettePanel,
        visual_hull_panel: VisualHullPanel,
        metrics_panel: MetricsPanel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Фото с реальной камеры — поза по ArUco-меткам")
        self.resize(920, 620)
        self.setAcceptDrops(True)

        self._layout_cfg = marker_layout
        self._intrinsics = intrinsics
        self._config_source = f"{ARUCO_MARKERS_PATH} / {CAMERA_INTRINSICS_PATH} (по умолчанию)"
        self._model_panel = model_panel
        # Panel 2/3/4 главного окна — заполняются реальными данными реконструкции по клику
        # «3D-реконструкция...» (см. _on_reconstruct_clicked): какие проекции реально
        # использовались (Panel 2), контур "вид сверху" и габариты (Panel 3 + метрики), а не
        # только халл в Panel 1 (тот и раньше рисовался через model_panel, общий канал с
        # синтетической Camera Mode реконструкцией STL — см. заметку задачи).
        self._silhouette_panel = silhouette_panel
        self._visual_hull_panel = visual_hull_panel
        self._metrics_panel = metrics_panel
        self._worker: _PoseEstimationWorker | None = None
        self._segmentation_worker: _SegmentationWorker | None = None
        # Промпт, которым были посчитаны ТЕКУЩИЕ item.segmentation — если пользователь поменял
        # поле «Классы:» с прошлого запуска, старые маски устарели и нужно пересчитать ВСЕ фото,
        # не только ещё не сегментированные (см. _on_segment_clicked).
        self._last_segmentation_prompt: str | None = None
        # То же для чекбокса "Скрывать объекты вне ленты" — его переключение тоже делает старые
        # маски устаревшими (см. _on_segment_clicked, тот же приём, что и для промпта).
        self._last_segmentation_used_filter: bool | None = None
        self._calibration_worker: _CalibrationWorker | None = None
        # True после первого успешного клика «3D-реконструкция...» — пока реконструкции ещё не
        # было, включение/выключение чекбокса фото нечего пересчитывать (см.
        # _on_item_checked_changed и заметку задачи "Выбор и исключение фото для
        # 3D-реконструкции"). Сбрасывается при закрытии диалога вместе с остальным состоянием
        # реконструкции (см. closeEvent/_on_photo_dialog_closing в MainWindow).
        self._has_reconstructed = False
        # Папка первого загруженного фото — по ней ищем/сохраняем calibration_result.yaml (см.
        # pose/calibration_cache.py). None, пока ни одного фото не загружено.
        self._photo_dir: Path | None = None
        # 4 доски реального рига (зеркало + борта ленты) — грузятся один раз, используются
        # только кнопкой калибровки (см. _on_calibrate_clicked); НЕ то же самое, что
        # self._layout_cfg (та — уже СЛИТАЯ раскладка для оценки поз одного фото).
        try:
            self._rig_boards = load_all_boards(MIRROR_BOARD_PATH, BELT_BOARD_PATH)
        except (OSError, KeyError, ValueError) as exc:
            self._rig_boards = {}
            self._rig_boards_error = str(exc)
        else:
            self._rig_boards_error = ""
        # Backend держится живым между кликами «Сегментация» (модель грузится один раз) —
        # создаётся лениво на первый клик (см. _get_segmentation_backend), останавливается
        # только в shutdown_segmentation_backend() (вызывает MainWindow при закрытии приложения,
        # НЕ closeEvent этого диалога — диалог переживает многократные show/hide).
        self._segmentation_backend: Sam3SubprocessBackend | None = None

        # Панель контроля конфигурации — что именно сейчас загружено (интринсики + раскладка
        # меток) и откуда, плюс кнопки "открыть файл" для правки и "обновить" для повторной
        # загрузки без перезапуска диалога (см. _on_reload_config_clicked).
        self.config_summary_label = QLabel()
        self.config_summary_label.setWordWrap(True)
        self.config_summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.open_intrinsics_button = QPushButton("Открыть camera_intrinsics.yaml")
        self.open_markers_button = QPushButton("Открыть aruco_markers.yaml")
        self.reload_config_button = QPushButton("Обновить конфиг и пересчитать")

        # Регистрация 4 досок рига + калибровка камеры по загруженным фото (см.
        # pose/board_registration.py, pose/calibration.py, pose/calibration_cache.py). Кнопка
        # всегда доступна (можно пересчитать в любой момент); статусная строка — авто-проверка
        # по средней ошибке репроекции уже загруженных/оценённых фото.
        self.calibration_status_label = QLabel("Фото не загружены.")
        self.calibration_status_label.setWordWrap(True)
        self.calibrate_button = QPushButton("Калибровать риг по загруженным фото...")
        self.calibrate_button.setToolTip(
            "Регистрирует 4 доски рига (зеркало + борта ленты) и калибрует камеру по всем "
            "загруженным фото; результат сохраняется рядом с фото и переиспользуется при "
            "следующей загрузке из той же папки."
        )

        self.photo_list = QListWidget()
        self.load_button = QPushButton("Загрузить фото...")
        self.remove_button = QPushButton("Удалить")

        self.preview_label = QLabel("Перетащите фото сюда или нажмите «Загрузить фото...»")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(420, 320)
        self.preview_label.setStyleSheet("background-color: #202020; color: #999;")

        self.status_value_label = QLabel("—")
        self.status_value_label.setWordWrap(True)
        self.position_value_label = QLabel("—")
        self.orientation_value_label = QLabel("—")
        self.markers_value_label = QLabel("—")
        self.reprojection_value_label = QLabel("—")
        self.segmentation_value_label = QLabel("—")
        self.segmentation_value_label.setWordWrap(True)

        self.info_form = QFormLayout()
        self.info_form.addRow("Статус:", self.status_value_label)
        self.info_form.addRow("Позиция камеры, мм (x,y,z):", self.position_value_label)
        self.info_form.addRow("Ориентация, ° (углы Эйлера):", self.orientation_value_label)
        self.info_form.addRow("Метки использованы:", self.markers_value_label)
        self.info_form.addRow("Ошибка репроекции, % кадра (px):", self.reprojection_value_label)
        self.info_form.addRow("Сегментация (SAM3):", self.segmentation_value_label)

        self.classes_label = QLabel("Классы:")
        self.classes_edit = QLineEdit(self._load_initial_classes_text())
        self.classes_edit.setToolTip(
            "Текстовый промпт SAM3 — классы объектов через запятую (например: object, box). "
            "Сохраняется между запусками."
        )
        self.classes_edit.setPlaceholderText("object")

        self.segment_button = QPushButton("Сегментация (SAM3)...")
        self.segment_button.setToolTip(
            "Посчитать маску объекта для всех загруженных фото без маски (subprocess к "
            "UavVisionLab, модель pt_trt)"
        )
        self.hide_offbelt_checkbox = QCheckBox("Скрывать объекты вне ленты (по маркерам)")
        self.hide_offbelt_checkbox.setChecked(True)
        self.hide_offbelt_checkbox.setToolTip(
            "Перед сегментацией закрашивает всё вне области ленты и области отражения ленты в "
            "зеркале (по уже посчитанной позе камеры и маркерам бортов/зеркала) — устраняет "
            "ложные детекции фонового «бутылко-подобного» мусора. Требует посчитанной позы "
            "камеры на фото (маркеры ленты/зеркала в кадре)."
        )
        self.reconstruct_button = QPushButton("3D-реконструкция...")
        self.reconstruct_button.setEnabled(False)
        self.reconstruct_button.setToolTip(
            "Нужно минимум 2 отмеченных фото с посчитанной позой и маской"
        )
        self.reconstruct_status_label = QLabel("—")
        self.reconstruct_status_label.setWordWrap(True)

        # Какие наблюдения участвуют в реконструкции — фильтр по регистрации (belt/mirror), не
        # по отдельным фото (см. _ready_observations_source/_selected_region_filter). Нужен, чтобы
        # сравнить, что именно зеркало добавляет/портит в реконструкции (см. заметку задачи про
        # зеркальный шум phone2), не переключая чекбоксы каждого фото по одному.
        self.view_filter_label = QLabel("Наблюдения:")
        self.view_filter_combo = QComboBox()
        self.view_filter_combo.addItems([
            "Все ракурсы",
            "Без отражений (только лента)",
            "Только отражения (зеркало)",
        ])
        self.view_filter_combo.setToolTip(
            "Какие наблюдения включать в 3D-реконструкцию — все, только прямые (лента) или "
            "только через отражение в зеркале."
        )

        # Видимость уже посчитанных халлов в Panel 1 — независимо от реконструкции (setVisible,
        # без пересчёта). Оба слоя рисуются additive и почти сливаются на выпуклых объектах, не
        # переключать нечем было раньше (см. заметку задачи).
        self.show_polytope_checkbox = QCheckBox("Выпуклый (polytope)")
        self.show_polytope_checkbox.setChecked(True)
        self.show_exact_checkbox = QCheckBox("Вогнутый (exact)")
        self.show_exact_checkbox.setChecked(True)

        self._build_layout()
        self._connect_signals()
        self._update_config_summary()

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)

        config_box = QGroupBox("Конфигурация (для контроля)")
        config_layout = QVBoxLayout(config_box)
        config_layout.addWidget(self.config_summary_label)
        config_buttons_row = QHBoxLayout()
        config_buttons_row.addWidget(self.open_intrinsics_button)
        config_buttons_row.addWidget(self.open_markers_button)
        config_buttons_row.addWidget(self.reload_config_button)
        config_layout.addLayout(config_buttons_row)
        outer.addWidget(config_box)

        calibration_box = QGroupBox("Калибровка рига (доски + камера)")
        calibration_layout = QVBoxLayout(calibration_box)
        calibration_layout.addWidget(self.calibration_status_label)
        calibration_layout.addWidget(self.calibrate_button)
        outer.addWidget(calibration_box)

        root = QHBoxLayout()
        outer.addLayout(root, stretch=1)

        left = QVBoxLayout()
        left.addWidget(self.photo_list, stretch=1)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.load_button)
        buttons_row.addWidget(self.remove_button)
        left.addLayout(buttons_row)
        root.addLayout(left, stretch=1)

        right = QVBoxLayout()
        right.addWidget(self.preview_label, stretch=1)
        right.addLayout(self.info_form)
        classes_row = QHBoxLayout()
        classes_row.addWidget(self.classes_label)
        classes_row.addWidget(self.classes_edit, stretch=1)
        right.addLayout(classes_row)
        right.addWidget(self.hide_offbelt_checkbox)
        view_filter_row = QHBoxLayout()
        view_filter_row.addWidget(self.view_filter_label)
        view_filter_row.addWidget(self.view_filter_combo, stretch=1)
        right.addLayout(view_filter_row)
        future_row = QHBoxLayout()
        future_row.addWidget(self.segment_button)
        future_row.addWidget(self.reconstruct_button)
        right.addLayout(future_row)
        right.addWidget(self.reconstruct_status_label)
        show_hull_row = QHBoxLayout()
        show_hull_row.addWidget(QLabel("Показать в 3D:"))
        show_hull_row.addWidget(self.show_polytope_checkbox)
        show_hull_row.addWidget(self.show_exact_checkbox)
        right.addLayout(show_hull_row)
        root.addLayout(right, stretch=2)

    def _connect_signals(self) -> None:
        self.load_button.clicked.connect(self._on_load_clicked)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.photo_list.currentItemChanged.connect(self._on_selection_changed)
        self.photo_list.itemChanged.connect(self._on_item_checked_changed)
        self.open_intrinsics_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(CAMERA_INTRINSICS_PATH)))
        )
        self.open_markers_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(ARUCO_MARKERS_PATH)))
        )
        self.reload_config_button.clicked.connect(self._on_reload_config_clicked)
        self.calibrate_button.clicked.connect(self._on_calibrate_clicked)
        self.segment_button.clicked.connect(self._on_segment_clicked)
        self.classes_edit.editingFinished.connect(self._on_classes_edited)
        self.reconstruct_button.clicked.connect(self._on_reconstruct_clicked)
        self.view_filter_combo.currentIndexChanged.connect(self._on_view_filter_changed)
        self.show_polytope_checkbox.toggled.connect(self._model_panel.set_polytope_hull_visible)
        self.show_exact_checkbox.toggled.connect(self._model_panel.set_exact_hull_visible)

    # --- конфигурация ------------------------------------------------------------------

    @staticmethod
    def _load_initial_classes_text() -> str:
        saved = load_classes_text()
        if saved:
            return saved
        try:
            return load_segmentation_config().prompt
        except (OSError, KeyError, ValueError):
            return "object"

    def _get_segmentation_backend(self) -> Sam3SubprocessBackend:
        if self._segmentation_backend is None:
            self._segmentation_backend = Sam3SubprocessBackend()
        return self._segmentation_backend

    def shutdown_segmentation_backend(self) -> None:
        """Останавливает фоновый процесс SAM3, если он был поднят. Вызывать при закрытии
        приложения (`MainWindow.closeEvent`) — НЕ при закрытии этого диалога, см. комментарий
        у `self._segmentation_backend` в `__init__`."""
        if self._segmentation_backend is not None:
            self._segmentation_backend.shutdown()
            self._segmentation_backend = None

    def _update_config_summary(self) -> None:
        dist = self._intrinsics.dist_coeffs
        dist_text = "нет (этап 1)" if dist is None else ", ".join(f"{c:g}" for c in dist)
        marker_ids = sorted(self._layout_cfg.corners_world_mm.keys())
        self.config_summary_label.setText(
            f"Интринсики: fx={self._intrinsics.fx:g}, fy={self._intrinsics.fy:g}, "
            f"cx={self._intrinsics.cx:g}, cy={self._intrinsics.cy:g}, "
            f"resolution_px={self._intrinsics.resolution_px} (rows, cols), "
            f"дисторсия: {dist_text}\n"
            f"Раскладка меток: {self._layout_cfg.dictionary_name}, "
            f"{len(marker_ids)} меток, id={marker_ids}\n"
            f"Источник: {self._config_source}"
        )

    def _on_reload_config_clicked(self) -> None:
        try:
            self._layout_cfg = load_marker_layout()
            self._intrinsics = load_camera_intrinsics()
        except (OSError, KeyError, ValueError) as exc:
            self.config_summary_label.setText(f"Ошибка загрузки конфига: {exc}")
            return
        self._config_source = f"{ARUCO_MARKERS_PATH} / {CAMERA_INTRINSICS_PATH}"
        self._update_config_summary()
        self._rerun_pose_estimation_on_all()

    def _all_items(self) -> list[PhotoItem]:
        return [
            self.photo_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.photo_list.count())
        ]

    def _rerun_pose_estimation_on_all(self) -> None:
        """Пересчитать позы уже загруженных фото под текущие self._layout_cfg/self._intrinsics —
        общий код для «Обновить конфиг» и завершения калибровки рига."""
        all_items = self._all_items()
        if not all_items:
            return
        self._worker = _PoseEstimationWorker(all_items, self._layout_cfg, self._intrinsics, self)
        self._worker.results_ready.connect(self._on_pose_results_ready)
        self._worker.start()

    # --- калибровка рига (регистрация досок + камера) -----------------------------------

    def _update_calibration_status(self) -> None:
        """Авто-проверка "нужна ли калибровка": средняя ошибка репроекции по фото с уже
        посчитанной позой, сравнённая с порогом (см. `_CALIBRATION_RECOMMENDED_THRESHOLD_PCT`).
        Кнопка калибровки при этом ВСЕГДА доступна — это только подсказка."""
        all_items = self._all_items()
        if not all_items:
            self.calibration_status_label.setText("Фото не загружены.")
            return
        ok_items = [
            item
            for item in all_items
            if item.pose_estimate is not None and item.pose_estimate.status == "ok"
        ]
        if not ok_items:
            self.calibration_status_label.setText(
                f"Поза не посчитана ни на одном из {len(all_items)} фото — нужна калибровка/"
                "регистрация досок рига."
            )
            return
        mean_err_pct = float(
            np.mean([item.pose_estimate.reprojection_error_pct for item in ok_items])
        )
        if mean_err_pct > _CALIBRATION_RECOMMENDED_THRESHOLD_PCT:
            verdict = "рекомендуется калибровка"
        else:
            verdict = "калибровка в норме"
        self.calibration_status_label.setText(
            f"{verdict}: средняя ошибка репроекции {mean_err_pct:.2f}% "
            f"по {len(ok_items)}/{len(all_items)} фото (порог {_CALIBRATION_RECOMMENDED_THRESHOLD_PCT:g}%)"
        )

    def _on_calibrate_clicked(self) -> None:
        if not self._rig_boards:
            self.calibration_status_label.setText(
                f"Раскладки досок рига не загрузились: {self._rig_boards_error}"
            )
            return
        all_items = self._all_items()
        ready = [item for item in all_items if item.image_bgr is not None]
        if len(ready) < 3:
            self.calibration_status_label.setText(
                f"Недостаточно фото с прочитанным изображением ({len(ready)}, нужно минимум 3)."
            )
            return
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.setText("Калибровка...")
        self.calibration_status_label.setText(f"Калибровка по {len(ready)} фото...")
        self._calibration_worker = _CalibrationWorker(
            [item.image_bgr for item in ready],
            self._rig_boards,
            self._intrinsics,
            [item.path.name for item in ready],
            self,
        )
        self._calibration_worker.finished_calibration.connect(self._on_calibration_finished)
        self._calibration_worker.start()

    def _on_calibration_finished(self, cache: CalibrationCache | None, message: str) -> None:
        self.calibrate_button.setEnabled(True)
        self.calibrate_button.setText("Калибровать риг по загруженным фото...")
        if cache is None:
            self.calibration_status_label.setText(f"Калибровка не удалась: {message}")
            return

        self._layout_cfg = cache.layout
        self._intrinsics = cache.intrinsics
        if self._photo_dir is not None:
            self._config_source = f"калибровка рига, только что посчитана ({calibration_cache_path(self._photo_dir)})"
        self._update_config_summary()

        if self._photo_dir is not None:
            try:
                save_calibration_cache(calibration_cache_path(self._photo_dir), cache)
                saved_note = f"сохранено в {calibration_cache_path(self._photo_dir)}"
            except OSError as exc:
                saved_note = f"не удалось сохранить кэш: {exc}"
        else:
            saved_note = "папка фото неизвестна, кэш не сохранён"

        err_text = (
            f"{cache.rms_reprojection_error_pct:.2f}%"
            if cache.rms_reprojection_error_pct is not None
            else "?"
        )
        note = f"; {message}" if message else ""
        self.calibration_status_label.setText(
            f"Калибровка выполнена: RMS {err_text}, {len(cache.layout.corners_world_mm)} меток "
            f"в раскладке, {saved_note}{note}"
        )
        self._rerun_pose_estimation_on_all()

    # --- сегментация (SAM3) -------------------------------------------------------------

    def _on_classes_edited(self) -> None:
        save_classes_text(self.classes_edit.text().strip())

    def _segmentation_jobs_for_item(self, item: PhotoItem, stale: bool) -> list[_SegmentationJob]:
        """Задания сегментации для ОДНОГО фото. С включённым чекбоксом — ДО ДВУХ отдельных
        запросов (belt/mirror region), каждый на СВОЁМ кадрированном (`region_filter.
        crop_to_regions`, не просто замаскированном на полном кадре — см. его докстринг про то,
        почему маленький объект/отражение теряет разрешение при масштабировании SAM3 без
        кадрирования) кадре — лента и зеркало больше не конкурируют за одну детекцию (см.
        докстринг `_SegmentationJob` и заметку задачи: раньше отражение никогда не побеждало
        прямой вид, даже когда оно видно на каждом фото). Без чекбокса — один запрос по всему
        кадру (старое поведение, регион неизвестен заранее — определяется постфактум в
        `_on_segmentation_finished`; кадрировать без знания региона не по чему)."""
        if item.image_bgr is None:
            return []
        if not self.hide_offbelt_checkbox.isChecked():
            if stale or item.segmentation is None:
                return [_SegmentationJob(item=item, region="unfiltered", image_bgr=item.image_bgr)]
            return []

        pose_estimate = item.pose_estimate
        if pose_estimate is None or pose_estimate.pose is None:
            return []
        belt_region, mirror_region = belt_and_mirror_regions(self._layout_cfg, pose_estimate.pose)
        jobs = []
        if belt_region is not None and (stale or item.segmentation is None):
            cropped = crop_to_regions(item.image_bgr, (belt_region,))
            if cropped is not None:
                image, offset = cropped
                jobs.append(_SegmentationJob(item=item, region="belt", image_bgr=image, offset=offset))
        if mirror_region is not None and (stale or item.segmentation_mirror is None):
            cropped = crop_to_regions(item.image_bgr, (mirror_region,))
            if cropped is not None:
                image, offset = cropped
                jobs.append(_SegmentationJob(item=item, region="mirror", image_bgr=image, offset=offset))
        return jobs

    def _on_segment_clicked(self) -> None:
        all_items = self._all_items()
        prompt = self.classes_edit.text().strip()
        use_filter = self.hide_offbelt_checkbox.isChecked()
        # Обычно обрабатываем только ещё не сегментированные фото/области — повторный клик не
        # пересчитывает уже готовые маски (типичный случай: догрузили новые фото). НО если
        # промпт («Классы:») или чекбокс «Скрывать объекты вне ленты» изменились с прошлого
        # запуска — старые маски посчитаны под другие условия и устарели, пересчитываем ВСЕ
        # фото, а не только новые (баг с промптом: клик после смены класса тихо ничего не делал,
        # если все фото уже были сегментированы).
        stale = prompt != self._last_segmentation_prompt or use_filter != self._last_segmentation_used_filter
        jobs = [job for item in all_items for job in self._segmentation_jobs_for_item(item, stale)]
        if not jobs:
            return
        save_classes_text(prompt)
        try:
            backend = self._get_segmentation_backend()
        except (OSError, KeyError, ValueError) as exc:
            self.segmentation_value_label.setText(f"Ошибка конфигурации: {exc}")
            return
        self.segment_button.setEnabled(False)
        self.segment_button.setText("Сегментация...")
        self._last_segmentation_prompt = prompt
        self._last_segmentation_used_filter = use_filter
        self._segmentation_worker = _SegmentationWorker(jobs, backend, prompt, self)
        self._segmentation_worker.finished_batch.connect(self._on_segmentation_finished)
        self._segmentation_worker.start()

    def _on_segmentation_finished(
        self, results: list[tuple[_SegmentationJob, SegmentationResult | None]], error: str
    ) -> None:
        self.segment_button.setEnabled(True)
        self.segment_button.setText("Сегментация (SAM3)...")
        if error:
            self.segmentation_value_label.setText(f"Ошибка: {error}")
            return
        touched: set[int] = set()
        for job, result in results:
            if result is not None:
                if job.offset is not None:
                    # Задание считалось на кадрированном (crop_to_regions), не полном кадре (см.
                    # `_segmentation_jobs_for_item`) — возвращаем маски в полнокадровые координаты
                    # ДО любого дальнейшего использования (классификация региона по центроиду,
                    # реконструкция, отрисовка превью — все ожидают координаты `item.image_bgr`).
                    full_shape = job.item.image_bgr.shape[:2]
                    result.all_masks = [
                        uncrop_mask(m, job.offset, full_shape) for m in result.all_masks
                    ]
                    result.mask = result.all_masks[0] if result.all_masks else uncrop_mask(
                        result.mask, job.offset, full_shape
                    )
                if job.region == "unfiltered":
                    # Регион неизвестен заранее (чекбокс маскировки был выключен) — определяем
                    # постфактум по центроиду маски, как раньше.
                    pose_estimate = job.item.pose_estimate
                    if pose_estimate is not None and pose_estimate.pose is not None:
                        belt_region, mirror_region = belt_and_mirror_regions(
                            self._layout_cfg, pose_estimate.pose
                        )
                        result.region = classify_mask_region(result.mask, belt_region, mirror_region)
                else:
                    result.region = job.region
                    # Геометрический отбор кандидатов (см. `select_region_candidates` и
                    # `SegmentationResult.candidates`): primary-маска — самая уверенная из
                    # ГЕОМЕТРИЧЕСКИ валидных для этой области, а не просто самая уверенная
                    # (та может быть фоном/чужим объектом — реальный кейс phone1: детекция
                    # картона ленты целиком с conf выше, чем у объекта). Пустой список
                    # кандидатов — ни одна детекция не подходит области, наблюдение
                    # непригодно (result сбрасывается в None ниже через candidates).
                    pose_estimate = job.item.pose_estimate
                    if pose_estimate is not None and pose_estimate.pose is not None:
                        belt_region, mirror_region = belt_and_mirror_regions(
                            self._layout_cfg, pose_estimate.pose
                        )
                        region_poly = belt_region if job.region == "belt" else mirror_region
                        if region_poly is not None:
                            result.candidates = select_region_candidates(
                                result.all_masks, region_poly
                            )
                            if result.candidates:
                                result.mask = result.candidates[0]
                            else:
                                result = None
            if job.region == "mirror":
                job.item.segmentation_mirror = result
            else:
                job.item.segmentation = result
            self._update_row_for_item(job.item)
            touched.add(id(job.item))
        current = self.photo_list.currentItem()
        if current is not None and id(current.data(Qt.ItemDataRole.UserRole)) in touched:
            self._show_item(current.data(Qt.ItemDataRole.UserRole))
        self._update_reconstruct_button_state()

    # --- 3D-реконструкция ---------------------------------------------------------------

    def _selected_region_filter(self) -> str | None:
        """`None` — все наблюдения (дефолт, прежнее поведение), `"belt"`/`"mirror"` — только
        прямые/только через отражение (см. `view_filter_combo`, порядок пунктов в `_build_layout`
        задаёт это же сопоставление индекс->регион)."""
        return {0: None, 1: "belt", 2: "mirror"}.get(self.view_filter_combo.currentIndex())

    def _on_view_filter_changed(self, _index: int) -> None:
        self._update_reconstruct_button_state()
        # Тот же приём, что и _on_item_checked_changed — пересчитать, только если реконструкция
        # уже запускалась в этом сеансе диалога.
        if self._has_reconstructed and self.reconstruct_button.isEnabled():
            self._on_reconstruct_clicked()

    def _ready_observations_source(self) -> list[tuple[PhotoItem, SegmentationResult]]:
        """(item, result) для реконструкции — ОТДЕЛЬНО прямой вид (`item.segmentation`) и
        отражение (`item.segmentation_mirror`): теперь это НЕЗАВИСИМЫЕ сегментации одного и
        того же фото (см. `PhotoItem`), а не взаимоисключающие альтернативы — оба, если есть,
        должны участвовать (пользовательское требование: отражение обязано повышать точность,
        а не просто использоваться, когда прямой вид не сегментировался).

        `_selected_region_filter()` дополнительно сужает набор до одного региона (belt/mirror) —
        для визуального/численного сравнения вклада зеркала в реконструкцию (см. `view_filter_combo`)."""
        region_filter = self._selected_region_filter()
        ready: list[tuple[PhotoItem, SegmentationResult]] = []
        for item in self._all_items():
            if not item.enabled or item.pose_estimate is None or item.pose_estimate.pose is None:
                continue
            if (
                item.segmentation is not None and item.segmentation.region == "belt"
                and region_filter in (None, "belt")
            ):
                ready.append((item, item.segmentation))
            if (
                item.segmentation_mirror is not None and item.segmentation_mirror.region == "mirror"
                and region_filter in (None, "mirror")
            ):
                ready.append((item, item.segmentation_mirror))
        return ready

    def _update_reconstruct_button_state(self) -> None:
        ready = len(self._ready_observations_source())
        self.reconstruct_button.setEnabled(ready >= 2)
        if ready < 2:
            self.reconstruct_button.setToolTip(
                f"Нужно минимум 2 наблюдения (прямых или через отражение) с посчитанной позой и "
                f"маской — сейчас {ready}"
            )

    def _build_observations(
        self, ready: list[tuple[PhotoItem, SegmentationResult]]
    ) -> list[pose_carving.CandidateObservation] | None:
        """Наблюдения для консенсусной реконструкции (`pose_carving.carve_consensus`) — прямые
        (region="belt") как есть, отражённые (region="mirror") с `mirror_plane` (см.
        `Observation`/`reflect_camera_frame`). Каждое наблюдение несёт СПИСОК кандидатов маски
        (`SegmentationResult.candidates` — геометрически валидные детекции по убыванию
        уверенности; fallback — одна primary-маска, если кандидаты не посчитаны, например
        сегментация шла без чекбокса маскировки): кандидата, противоречащего остальным
        ракурсам, консенсус заменит следующим, а не выбросит всё наблюдение (реальный кейс —
        отражённый в зеркале фоновый хлам, в одном кадре неотличимый от целевого объекта).
        `None` — среди `ready` есть "mirror", но плоскость зеркала посчитать не из чего (нет
        меток зеркала в `self._layout_cfg`).

        Отражённые наблюдения получают допуск `_MIRROR_MARGIN_PX` (дилатация маски перед
        построением полупространств/конуса, см. `pose_carving.Observation.margin_px`) — прямые
        не получают (уже хорошо согласованы, см. диагностику ниже). Без допуска (margin=0)
        отражения ТОЧНО пересекаются с прямыми наблюдениями — при накопленной ошибке позы+
        плоскости зеркала (виртуальная камера дальше от объекта, та же угловая ошибка даёт
        БОЛЬШУЮ позиционную) это подрезает реальный объём для ВСЕГО набора, а не только портит
        точность своего кадра. Эмпирически подобрано на реальных фото phone2
        (`sweep_mirror_margin.py`, диагностика 2026-07-19): margin=10px поднимает среднее IoU
        проекции халла против реальной маски SAM3 с 0.76 до ~0.79-0.80 без потери информации
        (в отличие от полного исключения зеркала) — см. заметку задачи "Использовать отражение в
        зеркале как полноценный ракурс реконструкции"."""
        mirror_plane = None
        if any(result.region == "mirror" for _item, result in ready):
            mirror_plane = fit_mirror_plane(self._layout_cfg)
            if mirror_plane is None:
                return None
        observations = []
        for item, result in ready:
            plane = mirror_plane if result.region == "mirror" else None
            region_label = "лента" if result.region == "belt" else "отражение"
            observations.append(
                pose_carving.CandidateObservation(
                    candidates=result.candidates or [result.mask],
                    pose=item.pose_estimate.pose,
                    mirror_plane=plane,
                    label=f"{item.path.name}/{region_label}",
                    margin_px=_MIRROR_MARGIN_PX if result.region == "mirror" else 0.0,
                )
            )
        return observations

    def _on_reconstruct_clicked(self) -> None:
        self._has_reconstructed = True
        ready = self._ready_observations_source()
        observations = self._build_observations(ready)
        if observations is None:
            self.reconstruct_status_label.setText(
                "Не удалось получить плоскость зеркала (нет меток зеркала в раскладке) — "
                "отражённые наблюдения нельзя использовать"
            )
            return
        n_belt = sum(1 for _item, result in ready if result.region == "belt")
        n_mirror = len(ready) - n_belt

        consensus = pose_carving.carve_consensus(observations)
        polytope_hull = consensus.polytope
        exact_hull = consensus.exact

        # Отчёт консенсуса: где выбран не самый уверенный кандидат / что выброшено — пользователь
        # должен видеть, что реконструкция «подправила» выбор SAM3 (см. carve_consensus docstring).
        consensus_notes = []
        if consensus.switched_labels:
            consensus_notes.append("заменён кандидат: " + ", ".join(consensus.switched_labels))
        if consensus.dropped_labels:
            consensus_notes.append("исключено: " + ", ".join(consensus.dropped_labels))
        consensus_note = ("; " + "; ".join(consensus_notes)) if consensus_notes else ""

        # Объект уже загруженного STL прячем — он в СВОЕЙ, никак не связанной с реальным ригом
        # системе координат (см. заметку задачи: пользователь запросил это явно, чтобы не путать
        # синтетический меш с реконструкцией по фото в общей 3D-сцене Panel 1). Восстанавливается
        # в MainWindow при закрытии диалога (см. PhotoCaptureDialog.closing).
        self._model_panel.set_object_visible(False)

        # Поворот ТОЛЬКО для отображения в Panel 1 (см. pose_carving.rotation_aligning_up) —
        # мировая ось Z рига произвольна (задана печатной доской), не обязана указывать вверх;
        # без поворота реконструкция выглядела перевёрнутой в GLViewWidget (тот неявно считает
        # верх сцены +Z). Габариты/консенсус/Panel 3 (top_view_outline) считаются НИЖЕ по
        # исходным (неповёрнутым) вершинам — поворот не влияет на геометрию, только на то, как
        # она нарисована в Panel 1.
        display_up_normal = fit_belt_up_normal(self._layout_cfg, ready[0][0].pose_estimate.pose)
        display_rotation = (
            pose_carving.rotation_aligning_up(display_up_normal)
            if display_up_normal is not None
            else None
        )

        def _for_display(vertices: np.ndarray) -> np.ndarray:
            return vertices @ display_rotation.T if display_rotation is not None else vertices

        if polytope_hull is not None:
            self._model_panel.set_camera_hull_polytope(
                _for_display(polytope_hull.vertices), polytope_hull.faces
            )
        if exact_hull is not None:
            self._model_panel.set_camera_hull_exact(
                _for_display(exact_hull.vertices), exact_hull.faces
            )
        # Новый халл создаётся видимым по умолчанию (см. model_panel.set_camera_hull_*) —
        # синхронизировать с текущим состоянием чекбоксов «Показать в 3D» сразу после пересчёта,
        # иначе снятая до клика «3D-реконструкция...» галочка молча слетала бы.
        self._model_panel.set_polytope_hull_visible(self.show_polytope_checkbox.isChecked())
        self._model_panel.set_exact_hull_visible(self.show_exact_checkbox.isChecked())

        # Panel 2 — какие проекции (маски) РЕАЛЬНО вошли в реконструкцию: маска — выбранный
        # консенсусом кандидат (не обязательно candidates[0]), выброшенные наблюдения помечены.
        tiles = []
        for (item, result), obs, chosen_idx in zip(ready, observations, consensus.chosen):
            region_label = "лента" if result.region == "belt" else "отражение"
            if chosen_idx is None:
                tiles.append((f"{item.path.name} — {region_label} (исключено)", result.mask))
            else:
                suffix = f" (кандидат #{chosen_idx})" if chosen_idx != 0 else ""
                tiles.append((f"{item.path.name} — {region_label}{suffix}", obs.candidates[chosen_idx]))
        self._silhouette_panel.set_photo_observations(tiles)

        if polytope_hull is None and exact_hull is None:
            status = (
                "Реконструкция не удалась (вырожденный случай — см. docstring "
                "pose_carving.carve_*)" + consensus_note
            )
            self.reconstruct_status_label.setText(status)
            self._visual_hull_panel.set_photo_reconstruction(None, None, None)
            self._metrics_panel.set_photo_reconstruction(status, n_belt, n_mirror)
            return

        # "Вид сверху" (Panel 3) вдоль нормали ленты — своей продольной оси у мировых координат
        # рига нет (в отличие от кольцевой STL-модели), см. pose_carving.top_view_outline.
        # (та же нормаль, что уже посчитана выше для display_rotation — переиспользуем.)
        up_normal = display_up_normal
        polytope_outline = (
            pose_carving.top_view_outline(polytope_hull.vertices, up_normal)
            if polytope_hull is not None and up_normal is not None
            else None
        )
        exact_outline = (
            pose_carving.top_view_outline(exact_hull.vertices, up_normal)
            if exact_hull is not None and up_normal is not None
            else None
        )

        parts = []
        polytope_dims = None
        if polytope_hull is not None:
            polytope_dims = tuple(polytope_hull.vertices.max(axis=0) - polytope_hull.vertices.min(axis=0))
            parts.append(f"polytope: {polytope_dims[0]:.0f}×{polytope_dims[1]:.0f}×{polytope_dims[2]:.0f} мм")
        else:
            parts.append("polytope: не удалось")
        exact_dims = None
        if exact_hull is not None:
            exact_dims = tuple(exact_hull.vertices.max(axis=0) - exact_hull.vertices.min(axis=0))
            parts.append(f"exact: {exact_dims[0]:.0f}×{exact_dims[1]:.0f}×{exact_dims[2]:.0f} мм")
        else:
            parts.append("exact: не удалось")
        n_used = sum(1 for c in consensus.chosen if c is not None)
        status = (
            f"Реконструкция по {n_used} наблюдениям ({n_belt} прямых + {n_mirror} через "
            f"отражение) — " + ", ".join(parts) + consensus_note
        )
        self.reconstruct_status_label.setText(status)
        self._visual_hull_panel.set_photo_reconstruction(polytope_outline, exact_outline, status)
        self._metrics_panel.set_photo_reconstruction(status, n_belt, n_mirror, polytope_dims, exact_dims)

    # --- drag & drop -----------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self._add_photos(paths)

    # --- загрузка фото -----------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        paths_str, _filter = QFileDialog.getOpenFileNames(
            self, "Выбрать фото", filter="Изображения (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        self._add_photos([Path(p) for p in paths_str])

    def _add_photos(self, paths: list[Path]) -> None:
        new_items = [PhotoItem(path=p) for p in paths if p.suffix.lower() in _IMAGE_EXTENSIONS]
        if not new_items:
            return

        # Папка первого загруженного фото — по ней ищем calibration_result.yaml (см.
        # pose/calibration_cache.py). Только на первой загрузке в этом диалоге: если уже
        # калибровали/загружали кэш в этой сессии, повторный поиск ничего не меняет намеренно.
        if self._photo_dir is None:
            self._photo_dir = new_items[0].path.resolve().parent
            cache = None
            try:
                cache = load_calibration_cache(calibration_cache_path(self._photo_dir))
            except (OSError, KeyError, ValueError) as exc:
                self.calibration_status_label.setText(
                    f"Кэш калибровки в {self._photo_dir} повреждён, не загружен: {exc}"
                )
            if cache is not None:
                self._layout_cfg = cache.layout
                self._intrinsics = cache.intrinsics
                self._config_source = f"кэш калибровки ({calibration_cache_path(self._photo_dir)})"
                self._update_config_summary()
                err_text = (
                    f"{cache.rms_reprojection_error_pct:.2f}%"
                    if cache.rms_reprojection_error_pct is not None
                    else "?"
                )
                self.calibration_status_label.setText(
                    f"Загружен кэш калибровки из {calibration_cache_path(self._photo_dir)} "
                    f"(RMS {err_text}, {len(cache.layout.corners_world_mm)} меток)."
                )

        for item in new_items:
            list_item = QListWidgetItem(f"{item.path.name} — обработка...")
            list_item.setFlags(list_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            list_item.setCheckState(Qt.CheckState.Checked)
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            self.photo_list.addItem(list_item)

        # Предыдущий воркер (если ещё не закончил) не отменяем и не ждём — просто теряем на
        # него ссылку, он тихо довыполнится и будет собран GC. Тот же паттерн, что
        # `_MeshTrueDimsWorker`/`_RecomputeWorker` в `gui/main_window.py`: реюз безопасен, т.к.
        # каждый воркер эмитит результат только за СВОИ фото (`new_items`), не пересекается с
        # предыдущим запуском.
        self._worker = _PoseEstimationWorker(new_items, self._layout_cfg, self._intrinsics, self)
        self._worker.results_ready.connect(self._on_pose_results_ready)
        self._worker.start()

    def _on_pose_results_ready(
        self, results: list[tuple[PhotoItem, PoseEstimate, np.ndarray | None]]
    ) -> None:
        for item, estimate, image in results:
            item.pose_estimate = estimate
            item.image_bgr = image
            self._update_row_for_item(item)
        current = self.photo_list.currentItem()
        if current is not None:
            self._show_item(current.data(Qt.ItemDataRole.UserRole))
        self._update_calibration_status()
        self._update_reconstruct_button_state()

    def _find_row(self, item: PhotoItem) -> QListWidgetItem | None:
        for row in range(self.photo_list.count()):
            list_item = self.photo_list.item(row)
            if list_item.data(Qt.ItemDataRole.UserRole) is item:
                return list_item
        return None

    def _update_row_for_item(self, item: PhotoItem) -> None:
        list_item = self._find_row(item)
        if list_item is None:
            return
        status = item.pose_estimate.status if item.pose_estimate else "insufficient_markers"
        text = f"{item.path.name} — {_STATUS_LABELS.get(status, status)}"
        if item.segmentation is not None:
            text += f", лента {item.segmentation.confidence:.2f}"
        if item.segmentation_mirror is not None:
            text += f", отражение {item.segmentation_mirror.confidence:.2f}"
        list_item.setText(text)

    def _on_remove_clicked(self) -> None:
        row = self.photo_list.currentRow()
        if row >= 0:
            self.photo_list.takeItem(row)
            self.preview_label.setText("Перетащите фото сюда или нажмите «Загрузить фото...»")
            self.preview_label.setPixmap(QPixmap())
            self._update_info(None)
        self._update_reconstruct_button_state()

    def _on_item_checked_changed(self, list_item: QListWidgetItem) -> None:
        item: PhotoItem | None = list_item.data(Qt.ItemDataRole.UserRole)
        if item is not None:
            item.enabled = list_item.checkState() == Qt.CheckState.Checked
        self._update_reconstruct_button_state()
        # Пересчёт по аналогии с «Обновить конфиг и пересчитать» (_on_reload_config_clicked) —
        # только если реконструкция уже запускалась хотя бы раз в этом сеансе диалога (см.
        # _has_reconstructed): до первого клика «3D-реконструкция...» панели ещё пустые, нечего
        # обновлять. Если после снятия галочки наблюдений стало < 2, кнопка сама блокируется
        # (см. _update_reconstruct_button_state) — реконструкцию просто не запускаем, старый
        # результат остаётся на экране до следующего валидного набора фото.
        if self._has_reconstructed and self.reconstruct_button.isEnabled():
            self._on_reconstruct_clicked()

    # --- превью/инфо для выбранного фото ------------------------------------------------

    def _on_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        self._show_item(current.data(Qt.ItemDataRole.UserRole))

    def _show_item(self, item: PhotoItem | None) -> None:
        self._update_preview(item)
        self._update_info(item)

    def _update_preview(self, item: PhotoItem | None) -> None:
        if item is None or item.image_bgr is None:
            self.preview_label.setText("Обработка...")
            return
        display = item.image_bgr.copy()
        if item.segmentation is not None or item.segmentation_mirror is not None:
            # Полупрозрачная заливка масок поверх фото — рисуем ДО осей/меток, чтобы они
            # оставались хорошо видны сверху. Обе маски — прямой вид (зелёный) И отражение
            # (жёлтый) — независимые сегментации одного и того же объекта, обе нужны видеть
            # одновременно (не альтернативы друг друга, см. PhotoItem docstring).
            overlay = display.copy()
            # Остальные найденные SAM3 объекты (не выбранные как "лента"/"отражение") — розовым,
            # рисуются ПЕРВЫМИ (под основными цветами), чтобы там, где они пересекаются с
            # выбранной маской, победил зелёный/жёлтый. Раньше backend отдавал только САМУЮ
            # уверенную детекцию — выключение чекбокса "Скрывать объекты вне ленты" никогда не
            # показывало эти маски, даже когда SAM3 их находил (см. заметку задачи,
            # SegmentationResult.all_masks).
            for result in (item.segmentation, item.segmentation_mirror):
                if result is None:
                    continue
                for other_mask in result.all_masks[1:]:
                    overlay[other_mask] = (180, 0, 220)  # BGR — розово-пурпурный, "другие объекты"
            if item.segmentation is not None:
                overlay[item.segmentation.mask] = (0, 255, 0)  # BGR — зелёный, лента (прямое)
            if item.segmentation_mirror is not None:
                overlay[item.segmentation_mirror.mask] = (0, 255, 255)  # BGR — жёлтый, отражение
            display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
        estimate = item.pose_estimate
        if estimate is not None and estimate.pose is not None:
            detected = detect_markers(item.image_bgr, self._layout_cfg)
            corners = [
                detected[marker_id].reshape(1, 4, 2).astype(np.float32)
                for marker_id in estimate.marker_ids_used
                if marker_id in detected
            ]
            ids = np.array(estimate.marker_ids_used, dtype=np.int32).reshape(-1, 1)
            if corners:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
            rvec, _ = cv2.Rodrigues(estimate.pose.rotation)
            tvec = estimate.pose.translation.reshape(3, 1)
            dist_coeffs = self._intrinsics.dist_coeffs
            if dist_coeffs is None:
                dist_coeffs = np.zeros(5)
            cv2.drawFrameAxes(
                display, self._intrinsics.matrix, dist_coeffs, rvec, tvec, _AXIS_LENGTH_MM
            )
        pixmap = _bgr_to_pixmap(display)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_info(self, item: PhotoItem | None) -> None:
        estimate = item.pose_estimate if item is not None else None
        if estimate is None:
            for label in (
                self.status_value_label,
                self.position_value_label,
                self.orientation_value_label,
                self.markers_value_label,
                self.reprojection_value_label,
            ):
                label.setText("—")
            self._update_segmentation_label(item)
            return

        self.status_value_label.setText(
            f"{_STATUS_LABELS.get(estimate.status, estimate.status)} — {estimate.message}"
        )
        self.markers_value_label.setText(
            ", ".join(str(marker_id) for marker_id in estimate.marker_ids_used) or "—"
        )
        if estimate.pose is not None:
            pos = estimate.pose.camera_pos_world
            self.position_value_label.setText(f"{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}")
            euler = _rotation_to_euler_deg(estimate.pose.rotation)
            self.orientation_value_label.setText(f"{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}")
        else:
            self.position_value_label.setText("—")
            self.orientation_value_label.setText("—")
        if estimate.reprojection_error_pct is not None:
            self.reprojection_value_label.setText(
                f"{estimate.reprojection_error_pct:.2f}% ({estimate.reprojection_error_px:.2f}px)"
            )
        else:
            self.reprojection_value_label.setText("—")
        self._update_segmentation_label(item)

    def _update_segmentation_label(self, item: PhotoItem | None) -> None:
        if item is None or (item.segmentation is None and item.segmentation_mirror is None):
            self.segmentation_value_label.setText("—")
            return
        parts = []
        if item.segmentation is not None:
            r = item.segmentation
            note = "" if r.region == "belt" else f", область неизвестна ({r.region}) — не участвует в 3D"
            other = f", +{len(r.all_masks) - 1} др. объект(ов) в кадре (розовым)" if len(r.all_masks) > 1 else ""
            parts.append(f"лента: conf={r.confidence:.2f}, класс={r.class_name}{note}{other}")
        if item.segmentation_mirror is not None:
            r = item.segmentation_mirror
            note = "" if r.region == "mirror" else f", область неизвестна ({r.region}) — не участвует в 3D"
            other = f", +{len(r.all_masks) - 1} др. объект(ов) в кадре (розовым)" if len(r.all_masks) > 1 else ""
            parts.append(f"отражение: conf={r.confidence:.2f}, класс={r.class_name}{note}{other}")
        self.segmentation_value_label.setText("; ".join(parts))

    def showEvent(self, event) -> None:  # noqa: ANN001 — QShowEvent, не импортирован ради типа
        super().showEvent(event)
        self.shown.emit()

    def closeEvent(self, event: QCloseEvent) -> None:
        # Диалог немодальный и может быть закрыт, пока фоновый воркер ещё обрабатывает пачку
        # фото — дожидаемся его, чтобы не удалить QThread, пока он выполняется (тот же риск,
        # что описан в search_dialog.py, только без отдельного шага отмены: работа быстрая).
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        if self._segmentation_worker is not None and self._segmentation_worker.isRunning():
            self._segmentation_worker.wait()
        if self._calibration_worker is not None and self._calibration_worker.isRunning():
            self._calibration_worker.wait()
        self.closing.emit()
        super().closeEvent(event)
