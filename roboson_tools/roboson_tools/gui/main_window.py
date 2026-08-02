"""Главное окно: 4 панели + панель управления, единый debounce-пересчёт при любом изменении."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QSettings, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.camera_rig import CameraSpec, load_camera_rig_presets, rig_angles, rig_to_distances
from ..core.config import load_app_settings, load_objects
from ..core.experiment import (
    EvaluationResult,
    GRollResult,
    MultiSideCameraDims,
    Orientation,
    RollCheckResult,
    SimpleCameraDims,
    axis_to_angles,
    check_model_roll,
    check_model_roll_f1,
    check_model_roll_g4,
    check_simple_camera_dims,
    check_simple_camera_dims_v3,
    check_vertical_prism,
    evaluate,
    mesh_true_dims,
    prism_dims,
    ROLL_VERDICT_NOT_ROUND,
    ROLL_VERDICT_ROUND,
    ROLL_VERDICT_UNCERTAIN,
)
from ..geometry.mesh_io import Mesh, load_stl
from ..geometry.transform import apply_orientation, rotation_matrix
from ..pose.camera_pose import load_camera_intrinsics
from ..pose.marker_layout import load_marker_layout
from ..silhouette import camera as camera_silhouette
from ..silhouette.base import Silhouette
from ..visual_hull import exact_polyhedral_hull, polytope_hull, voxel_carving
# torchhull — ОПЦИОНАЛЬНАЯ GPU-зависимость (CUDA Toolkit + venv-cv в robozon, см.
# bench_torchhull.py / visual_hull/torchhull_adapter.py). Импортируем мягко: если torch/torchhull
# нет в окружении — чек-бокс в GUI не показывается (как будто метод недоступен), остальные
# 3D-реконструкции (CPU voxel/polytope/exact) работают как обычно.
_TORCHHULL_AVAILABLE = False
try:
    import torch as _torch  # noqa: F401
    import torchhull as _torchhull  # noqa: F401
    from ..visual_hull.torchhull_adapter import build_transforms_batch, visual_hull_points
    _TORCHHULL_AVAILABLE = True
except Exception:
    pass
from ..visual_hull.vertical_prism_fit import PrismFitResult, prism_mesh
from .dialogs.camera_rig_dialog import CameraRigDialog
from .dialogs.compare_dialog import CompareDialog
from .dialogs.export_dialog import ExportDialog
from .dialogs.grid_reconstruction_dialog import GridReconstructionDialog
from .dialogs.photo_capture_dialog import PhotoCaptureDialog
from .dialogs.search_dialog import SearchDialog
from .panels.metrics_panel import STL_DIMS_PENDING, MetricsPanel
from .panels.model_panel import ModelPanel
from .panels.silhouette_panel import SilhouettePanel
from .panels.visual_hull_panel import VisualHullPanel
from .widgets.orientation_controls import OrientationControls

DEBOUNCE_MS = 50

# Наборы углов (без дистанций) для CompareDialog — сравнение конфигураций Analytical Mode при
# текущей ориентации, ортогонально ригу камер (angle+distance) ниже. Раньше жили в
# app_settings.yaml::angle_set_presets — вынесены сюда при миграции на именованные пресеты рига
# (config/camera_rig_presets.yaml), т.к. это разные сущности (просто углы vs угол+дистанция).
_COMPARE_ANGLE_PRESETS: list[list[float]] = [
    [0.0, 90.0],
    [0.0, 45.0, 90.0, 135.0],
    [0.0, 30.0, 60.0, 90.0, 120.0, 150.0],
    [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5],
]

# Camera Mode (см. заметку задачи "Добавить Camera Mode...") использует ту же азимутальную
# ось наблюдения, что и Analytical Mode, но не добавляет отдельного контрола для неё — берётся
# первый угол текущего набора view_angles (обычно 0° — "фронтальная" камера схемы ТЗ).
_BELT_POSITION_LABELS = {"start": "Начало", "center": "Центр", "end": "Конец"}
_BELT_POSITION_BY_LABEL = {v: k for k, v in _BELT_POSITION_LABELS.items()}

# Азимуты пресета "Симулятор Webots (cv_rig, 5 ракурсов)" (config/camera_rig_presets.yaml,
# 1:1 из robozon/config/layout.yaml::cv_rig.cameras) — по решению пользователя (2026-07-25),
# когда выбран ИМЕННО этот риг И включён camera_hull_use_ends_checkbox ("+ начало/конец
# ленты"), Panel 2 вместо одной силуэт-сетки на текущий belt_position показывает полную
# 3x3-сетку (5 центр + start/end для top/side — только у этих двух камер реальный риг вообще
# снимает старт/конец, см. robozon/sim/cv_grid.py::CELL_CAMERA_MOMENT), в ТОЙ ЖЕ раскладке,
# что `_COMPOSITE_LAYOUT` в robozon (`sim/cv_grid.py`) — прямое визуальное сравнение с
# `grid.png` реальной сетки. Для любого другого рига (angles не совпадают) — старое поведение
# (одна силуэт-сетка на выбранный belt_position), т.к. для generic-рига нет понятия "top"/
# "side" камеры со start/end съёмкой.
_SIM_RIG_TOP_ANGLE = 90.0
_SIM_RIG_SIDE_ANGLE = 5.0
_SIM_RIG_DIAG_ANGLE = 70.0
_SIM_RIG_TOP_MIRROR_ANGLE = 157.26
_SIM_RIG_DIAG_MIRROR_ANGLE = 171.95
_SIM_RIG_ANGLES = {
    _SIM_RIG_TOP_ANGLE, _SIM_RIG_SIDE_ANGLE, _SIM_RIG_DIAG_ANGLE,
    _SIM_RIG_TOP_MIRROR_ANGLE, _SIM_RIG_DIAG_MIRROR_ANGLE,
}


class _MeshTrueDimsWorker(QThread):
    """Считает `mesh_true_dims` (перебор направлений оси на полусфере, см. core/experiment) в
    фоновом потоке — на плотных мешах это заметно (на глаз) подвисало UI при загрузке STL,
    посчитано один раз на весь перебор координат. Меш захватывается в конструкторе (не читает
    `MainWindow._mesh` во время run()), поэтому безопасен, даже если пользователь успеет
    загрузить другой STL до завершения этого потока — см. _on_stl_true_dims_ready, где
    устаревший результат отбрасывается по идентичности воркера."""

    result_ready = pyqtSignal(object)  # tuple[float, float, float] | None

    def __init__(self, mesh: Mesh, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self._mesh = mesh

    def run(self) -> None:
        self.result_ready.emit(mesh_true_dims(self._mesh))


@dataclass
class _RecomputeRequest:
    """Входные параметры одного `_recompute()` — сняты с виджетов на GUI-потоке ДО передачи в
    `_RecomputeWorker` (виджеты Qt нельзя трогать из фонового потока)."""

    mesh: Mesh
    oriented_mesh: Mesh
    orientation: Orientation
    angles: list[float]
    axis_pos_override: float | None
    use_camera: bool
    belt_position: str
    resolution_px: int
    camera_fov_deg: float
    camera_distances: dict[float, float]
    roundness_threshold: float
    check_prism: bool
    show_voxel: bool
    show_polytope: bool
    show_exact: bool
    show_torchhull: bool  # GPU-реконструкция (опциональная зависимость, см. _TORCHHULL_AVAILABLE)
    torchhull_level: int  # уровень октре (7=128³ по умолчанию), для визуализации и G4
    camera_hull_use_ends: bool


@dataclass
class _RecomputeResult:
    """Всё, что раньше синхронно вычислялось в `_recompute()`/`_update_camera_mode()`, одним
    объектом — применяется к панелям на GUI-потоке в `_on_recompute_ready`."""

    request: _RecomputeRequest
    evaluation: EvaluationResult
    prism_result: PrismFitResult | None
    simple_dims: SimpleCameraDims | None
    simple_dims_v3: MultiSideCameraDims | None
    camera_silhouettes: dict[float, Silhouette] | None
    camera_caption_suffix: str | None
    # Не None -> Panel 2 показывает ЭТУ явную 3x3-сетку (start/center/end top+side + center
    # остальных) вместо одной силуэт-сетки camera_silhouettes — см. _SIM_RIG_ANGLES выше.
    camera_grid_tiles: list[tuple[str, np.ndarray]] | None
    belt_offset_dx: float | None
    voxel_points: np.ndarray | None
    polytope_vertices: np.ndarray | None
    polytope_faces: np.ndarray | None
    exact_vertices: np.ndarray | None
    exact_faces: np.ndarray | None
    torchhull_points: np.ndarray | None  # GPU visual hull (torchhull.visual_hull), облако точек
    # Время каждой 3D-реконструкции в мс (только сама reconstruction, без построения силуэтов и
    # G4-ядра) — для метки в GUI «3D-реконструкция: voxel=12мс, polytope=85мс, ...». None —
    # метод не был запрошен (чек-бокс выключен).
    recon_times_ms: dict[str, float] | None


def _base_distance(camera_distances: dict[float, float]) -> float:
    """Скаляр-заглушка для v1/v3 API (`check_simple_camera_dims`/`_v3`, см. core/experiment.py):
    эти функции берут обязательный базовый `camera_distance` + необязательный per-angle
    `camera_distances`, используя базовый только как fallback для азимутов, отсутствующих в
    самом `camera_distances`. При полном риге камер (всегда покрывает все используемые азимуты)
    этот fallback не должен реально сработать — значение здесь не более чем валидная заглушка."""
    return next(iter(camera_distances.values()))


class _RecomputeWorker(QThread):
    """Тяжёлая часть пересчёта (бывшие тела `_recompute()`/`_update_camera_mode()`) — раньше
    выполнялась синхронно в GUI-потоке по debounce-таймеру на КАЖДОЕ изменение (выбор модели,
    сдвиг любого слайдера), из-за чего окно на время "зависало" (Windows помечает как "не
    отвечает"). Особенно заметно на тяжёлых мешах — "Моющее средство" (218k вершин / 73k
    граней, самый тяжёлый объект в наборе) даёт ~4.6с на один `check_simple_camera_dims_v3` и
    ещё до ~4с на три Camera Mode реконструкции (voxel/polytope/exact), итого 8-10+с блокировки
    event loop за один пересчёт — пользователь наблюдал это как зависание, а при плотном потоке
    таких зависаний подряд (например, Windows решает, что окно не отвечает) — как "вылет без
    ошибки" (см. обсуждение с пользователем, docs/camera_dims_v2_investigation.md не заводился
    отдельно, т.к. это GUI-, а не расчётная проблема).

    Тот же паттерн, что уже был у `_MeshTrueDimsWorker`: все входные данные захватываются в
    конструкторе через `_RecomputeRequest` (не читаются из `self`/виджетов во время run()) —
    поток безопасен, даже если пользователь успеет сменить меш/ориентацию/настройки до
    завершения текущего расчёта (см. `_start_recompute_worker`, где устаревший сигнал
    отсоединяется, а не результат отбрасывается постфактум)."""

    result_ready = pyqtSignal(object)  # _RecomputeResult

    def __init__(self, request: _RecomputeRequest, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self._req = request

    def run(self) -> None:
        req = self._req

        evaluation = evaluate(
            req.mesh,
            req.orientation,
            req.angles,
            axis_pos=req.axis_pos_override,
            resolution_px=req.resolution_px,
            roundness_threshold=req.roundness_threshold,
            use_camera_silhouettes=req.use_camera,
            camera_fov_deg=req.camera_fov_deg,
            camera_distances=req.camera_distances,
            belt_position=req.belt_position,
        )

        prism_result = None
        if req.check_prism:
            prism_result = check_vertical_prism(
                req.mesh,
                req.orientation,
                req.angles,
                fov_deg=req.camera_fov_deg,
                camera_distances=req.camera_distances,
                resolution_px=req.resolution_px,
            )

        simple_dims = None
        simple_dims_v3 = None
        if req.use_camera:
            simple_dims = check_simple_camera_dims(
                req.mesh,
                req.orientation,
                fov_deg=req.camera_fov_deg,
                camera_distance=_base_distance(req.camera_distances),
                resolution_px=req.resolution_px,
                camera_distances=req.camera_distances,
            )
            simple_dims_v3 = check_simple_camera_dims_v3(
                req.mesh,
                req.orientation,
                fov_deg=req.camera_fov_deg,
                camera_distance=_base_distance(req.camera_distances),
                resolution_px=req.resolution_px,
                top_angle_deg=90.0,
                side_angles_deg=req.angles,
                belt_positions=["start", "center", "end"],
                camera_distances=req.camera_distances,
            )

        camera_silhouettes = None
        camera_caption_suffix = None
        camera_grid_tiles: list[tuple[str, np.ndarray]] | None = None
        belt_offset_dx = None
        voxel_points = None
        polytope_vertices = polytope_faces = None
        exact_vertices = exact_faces = None
        torchhull_pts = None
        recon_times_ms: dict[str, float] | None = None
        if req.use_camera:
            camera_silhouettes = {
                angle: camera_silhouette.build_silhouette(
                    req.oriented_mesh,
                    angle,
                    req.resolution_px,
                    fov_deg=req.camera_fov_deg,
                    camera_distance=req.camera_distances[angle],
                    belt_position=req.belt_position,
                )
                for angle in req.angles
            }
            camera_caption_suffix = (
                f"{req.camera_fov_deg:g}° FOV, "
                f"{_BELT_POSITION_LABELS[req.belt_position].lower()}"
            )
            # Только для отображения смещения на Panel 1 — берётся дистанция первого ракурса
            # набора (само значение не зависит от того, какую камеру рига показывать, разброс
            # между камерами здесь не принципиален).
            belt_offset_dx = camera_silhouette.belt_offset(
                req.oriented_mesh, req.camera_fov_deg, req.camera_distances[req.angles[0]],
                req.belt_position,
            )

            if req.camera_hull_use_ends and set(req.angles) == _SIM_RIG_ANGLES:
                def _grid_mask(angle: float, pos: str) -> np.ndarray:
                    return camera_silhouette.build_silhouette(
                        req.oriented_mesh, angle, req.resolution_px,
                        fov_deg=req.camera_fov_deg,
                        camera_distance=req.camera_distances[angle],
                        belt_position=pos,
                    ).mask
                # Порядок строго row-major под ту же раскладку, что robozon
                # `sim/cv_grid.py::_COMPOSITE_LAYOUT`: [[start_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diag],
                # [end_top, end_side, center_diag_mirror]].
                camera_grid_tiles = [
                    (f"top {_SIM_RIG_TOP_ANGLE:g}° — начало", _grid_mask(_SIM_RIG_TOP_ANGLE, "start")),
                    (f"side {_SIM_RIG_SIDE_ANGLE:g}° — начало", _grid_mask(_SIM_RIG_SIDE_ANGLE, "start")),
                    (f"top_mirror {_SIM_RIG_TOP_MIRROR_ANGLE:g}° — центр",
                     _grid_mask(_SIM_RIG_TOP_MIRROR_ANGLE, "center")),
                    (f"top {_SIM_RIG_TOP_ANGLE:g}° — центр", _grid_mask(_SIM_RIG_TOP_ANGLE, "center")),
                    (f"side {_SIM_RIG_SIDE_ANGLE:g}° — центр", _grid_mask(_SIM_RIG_SIDE_ANGLE, "center")),
                    (f"diag {_SIM_RIG_DIAG_ANGLE:g}° — центр", _grid_mask(_SIM_RIG_DIAG_ANGLE, "center")),
                    (f"top {_SIM_RIG_TOP_ANGLE:g}° — конец", _grid_mask(_SIM_RIG_TOP_ANGLE, "end")),
                    (f"side {_SIM_RIG_SIDE_ANGLE:g}° — конец", _grid_mask(_SIM_RIG_SIDE_ANGLE, "end")),
                    (f"diag_mirror {_SIM_RIG_DIAG_MIRROR_ANGLE:g}° — центр",
                     _grid_mask(_SIM_RIG_DIAG_MIRROR_ANGLE, "center")),
                ]

            if req.show_voxel or req.show_polytope or req.show_exact or req.show_torchhull:
                # ВСЕ азимуты набора (одного недостаточно — см. докстринг voxel_carving.py/
                # polytope_hull.py).
                belt_positions = (
                    ["start", "center", "end"] if req.camera_hull_use_ends else ["center"]
                )
                views = [(angle, pos) for angle in req.angles for pos in belt_positions]

                # Замер времени каждой реконструкции (только сама reconstruction, без силуэтов
                # и G4-ядра) — для метки в GUI «3D-реконструкция: voxel=12мс, ...».
                import time as _time
                recon_times_ms: dict[str, float] = {}

                if req.show_voxel:
                    _t0 = _time.perf_counter()
                    hull = voxel_carving.carve(
                        req.oriented_mesh,
                        views,
                        fov_deg=req.camera_fov_deg,
                        camera_distances=req.camera_distances,
                        resolution_px=req.resolution_px,
                    )
                    recon_times_ms["voxel"] = (_time.perf_counter() - _t0) * 1000
                    voxel_points = hull.points if hull is not None else None

                if req.show_polytope:
                    _t0 = _time.perf_counter()
                    polytope = polytope_hull.carve(
                        req.oriented_mesh,
                        views,
                        fov_deg=req.camera_fov_deg,
                        camera_distances=req.camera_distances,
                        resolution_px=req.resolution_px,
                    )
                    recon_times_ms["polytope"] = (_time.perf_counter() - _t0) * 1000
                    if polytope is not None:
                        polytope_vertices, polytope_faces = polytope.vertices, polytope.faces

                if req.show_exact:
                    _t0 = _time.perf_counter()
                    exact = exact_polyhedral_hull.carve(
                        req.oriented_mesh,
                        views,
                        fov_deg=req.camera_fov_deg,
                        camera_distances=req.camera_distances,
                        resolution_px=req.resolution_px,
                    )
                    recon_times_ms["exact"] = (_time.perf_counter() - _t0) * 1000
                    if exact is not None:
                        exact_vertices, exact_faces = exact.vertices, exact.faces

                if req.show_torchhull and _TORCHHULL_AVAILABLE:
                    # GPU visual hull через torchhull. Маски должны быть одинакового размера
                    # (torchhull требует [B, H, W, 1]) — строим без crop_to_object (полный кадр).
                    # torchhull принимает transforms = K @ T в OpenCV convention (см. адаптер).
                    masks = [
                        camera_silhouette.build_silhouette(
                            req.oriented_mesh, angle, req.resolution_px,
                            fov_deg=req.camera_fov_deg,
                            camera_distance=req.camera_distances[angle],
                            belt_position=pos,
                            crop_to_object=False,
                        ).mask
                        for angle, pos in views
                    ]
                    dxs = [
                        camera_silhouette.belt_offset(
                            req.oriented_mesh, req.camera_fov_deg,
                            req.camera_distances[angle], pos,
                        )
                        for angle, pos in views
                    ]
                    transforms = build_transforms_batch(
                        [(angle, pos, dx) for (angle, pos), dx in zip(views, dxs)],
                        fov_deg=req.camera_fov_deg,
                        camera_distances=req.camera_distances,
                        resolution_px=req.resolution_px,
                    )
                    # cube — bbox объекта + 10% margin
                    from ..geometry.mesh_io import bounding_box
                    bbox_min, bbox_max = bounding_box(req.oriented_mesh.vertices)
                    margin = float((bbox_max - bbox_min).max()) * 0.1
                    cube_corner = [float(bbox_min[i] - margin) for i in range(3)]
                    cube_length = float((bbox_max - bbox_min).max() + 2 * margin)
                    masks_partial = any(pos != "center" for _, pos in views)
                    _t0 = _time.perf_counter()
                    torchhull_pts = visual_hull_points(
                        masks=masks,
                        transforms=transforms,
                        cube_corner_bfl=cube_corner,
                        cube_length=cube_length,
                        level=req.torchhull_level,
                        masks_partial=masks_partial,
                    )
                    recon_times_ms["torchhull"] = (_time.perf_counter() - _t0) * 1000

        self.result_ready.emit(
            _RecomputeResult(
                request=req,
                evaluation=evaluation,
                prism_result=prism_result,
                simple_dims=simple_dims,
                simple_dims_v3=simple_dims_v3,
                camera_silhouettes=camera_silhouettes,
                camera_caption_suffix=camera_caption_suffix,
                camera_grid_tiles=camera_grid_tiles,
                belt_offset_dx=belt_offset_dx,
                voxel_points=voxel_points,
                polytope_vertices=polytope_vertices,
                polytope_faces=polytope_faces,
                exact_vertices=exact_vertices,
                exact_faces=exact_faces,
                torchhull_points=torchhull_pts,
                recon_times_ms=recon_times_ms,
            )
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("roboson_tools — стенд Visual Hull / силуэтов")
        self.resize(1200, 800)

        self._settings = load_app_settings()
        self._objects = load_objects()
        self._current_stl_path: Path | None = None
        self._current_is_round_reference: bool | None = None
        self._mesh = None
        # Фиксированные истинные габариты STL (core/experiment.mesh_true_dims) — минимальный
        # бокс по вершинам НЕповёрнутого меша, посчитан один раз при загрузке (см. _load_mesh) и
        # НЕ пересчитывается при повороте Roll/Pitch/Yaw: в отличие от bbox текущей ориентации,
        # не растёт при повороте (см. mesh_true_dims docstring) — стабильное число для GUI.
        # Дорогой перебор направлений считается в фоновом потоке (_MeshTrueDimsWorker), пока он
        # не завершится — значение STL_DIMS_PENDING (см. panels/metrics_panel.py), метрики
        # показывают "рассчитывается…" вместо обманчивого "—"/нулевой ошибки.
        self._stl_true_dims: tuple[float, float, float] | None = None
        self._stl_dims_worker: _MeshTrueDimsWorker | None = None
        # Тяжёлая часть _recompute() (evaluate/vertical_prism/v1/v3/Camera Mode реконструкции)
        # считается в фоновом потоке — см. _RecomputeWorker/_start_recompute_worker/
        # _on_recompute_ready. Тот же паттерн отсоединения сигнала устаревшего воркера, что и
        # у _stl_dims_worker выше.
        self._recompute_worker: _RecomputeWorker | None = None
        self._axis_pos_override: float | None = None
        self._last_result: EvaluationResult | None = None
        self._last_orientation = Orientation()
        self._last_roll_check: RollCheckResult | GRollResult | None = None
        # Результат последней geometric-проверки "вертикальная призма" (см. _recompute) — читается
        # в _on_check_model_clicked, чтобы подменить "Истинные габариты" точным числом, если
        # "Проверить модель" запускают уже после того, как объект прошёл проверку призмы.
        self._last_prism_result: PrismFitResult | None = None
        # Найденная ось переката в НЕПОВЁРНУТОЙ (базовой) системе координат меша — на каждый
        # _recompute() заново поворачивается текущей ориентацией, поэтому 3D-линия и плоскость
        # среза следуют за поворотом объекта, а не остаются «прибитыми» к старому положению.
        self._roll_axis_base: np.ndarray | None = None
        self._show_round_projection = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._recompute)

        self.model_panel = ModelPanel()
        self.silhouette_panel = SilhouettePanel()
        self.visual_hull_panel = VisualHullPanel()
        self.metrics_panel = MetricsPanel()

        self.object_combo = QComboBox()
        self.object_combo.addItems(list(self._objects.keys()))
        self.open_button = QPushButton("Open...")

        self.orientation_controls = OrientationControls()

        # Риг камер (угол+дистанция на каждую камеру, см. core/camera_rig.py) — заменяет
        # прежний список углов (AngleSetInput) + общий camera_distance_spinbox: реальный риг
        # "2 камеры + 1 зеркало" имеет РАЗНУЮ дистанцию на прямых/зеркальных ракурсах.
        # Восстанавливается из QSettings в _restore_camera_settings; дефолт — первый (встроенный)
        # пресет из config/camera_rig_presets.yaml.
        self._camera_rig: list[CameraSpec] = load_camera_rig_presets()[0].cameras
        self.camera_rig_button = QPushButton("Настроить риг камер...")
        self.camera_rig_summary_label = QLabel()
        self.camera_rig_summary_label.setWordWrap(True)
        self._update_camera_rig_summary()

        # Диалог загрузки реальных фото + расчёта позы камеры по ArUco-меткам (см.
        # roboson_tools/pose/) — немодальный, хранится как атрибут и переиспользуется между
        # кликами (см. _on_photo_capture_button_clicked), т.к. на будущих этапах в нём же будет
        # сегментация/реконструкция, а результат нужно видеть в Panel 1 одновременно с диалогом.
        self._photo_capture_dialog: PhotoCaptureDialog | None = None
        self.photo_capture_button = QPushButton("Фото с реальной камеры...")
        self.photo_capture_status_label = QLabel("")
        self.photo_capture_status_label.setWordWrap(True)
        self.photo_capture_status_label.setStyleSheet("color: #c62828;")

        # Диалог загрузки готовой 9-ракурсной сетки кропов из Webots-симулятора соседнего
        # robozon (немодальный, тот же паттерн, что photo_capture_dialog выше) — тест
        # 3D-реконструкции/G4 на реальных данных CV-пайплайна, задача "3D-реконструкция
        # объекта по кропам сетки ракурсов", пункт 3 запроса пользователя 2026-07-25.
        self._grid_reconstruction_dialog: GridReconstructionDialog | None = None
        self.grid_reconstruction_button = QPushButton("Загрузить сетку...")

        self.camera_mode_checkbox = QCheckBox("Проекция (камера FOV, реалистичные ракурсы)")
        self.belt_position_combo = QComboBox()
        self.belt_position_combo.addItems(list(_BELT_POSITION_LABELS.values()))
        self.belt_position_combo.setCurrentText(_BELT_POSITION_LABELS["center"])
        self.belt_position_combo.setEnabled(False)

        # Подбор FOV, высоты и разрешения камеры реального рига вживую — чтобы смотреть, как
        # ошибка габаритов от перспективы (Camera Mode) меняется с этими параметрами. Начальные
        # значения — из config/app_settings.yaml, но в GUI это уже не константы, а то, что можно
        # варьировать для оценки ошибки при разной геометрии рига.
        self.camera_fov_spinbox = QDoubleSpinBox()
        self.camera_fov_spinbox.setRange(1.0, 179.0)
        self.camera_fov_spinbox.setValue(self._settings.camera_fov_deg)
        self.camera_fov_spinbox.setSuffix("°")
        self.camera_fov_spinbox.setEnabled(False)

        self.camera_resolution_spinbox = QSpinBox()
        self.camera_resolution_spinbox.setRange(64, 4096)
        self.camera_resolution_spinbox.setSingleStep(64)
        self.camera_resolution_spinbox.setValue(self._settings.raster_resolution_px)
        self.camera_resolution_spinbox.setSuffix(" px")
        self.camera_resolution_spinbox.setEnabled(False)

        # Voxel carving по перспективным силуэтам (Panel 1) — см. заметку задачи: start/end
        # несут реальную дополнительную информацию о форме (торцевая грань видна под углом),
        # не только center. Две ступени: сначала один кадр (грубо), потом + начало/конец.
        self.show_camera_hull_checkbox = QCheckBox("Показать 3D-реконструкцию по проекциям (воксели)")
        self.show_camera_hull_checkbox.setEnabled(False)
        # Всегда доступен независимо от того, какой из чек-боксов реконструкции включён —
        # используется одинаково всеми тремя (voxel/polytope/exact), не только voxel carving.
        self.camera_hull_use_ends_checkbox = QCheckBox("+ начало/конец ленты")

        # Точное пересечение конусов обзора (visual_hull/polytope_hull.py) — альтернатива
        # voxel carving для сравнения: независимый чекбокс, оба можно включить одновременно
        # (используют один и тот же belt_positions/views, см. _RecomputeWorker.run).
        self.show_camera_polytope_checkbox = QCheckBox(
            "Показать точную 3D-реконструкцию (пересечение конусов)"
        )
        self.show_camera_polytope_checkbox.setEnabled(False)

        # Exact Polyhedral Visual Hull (visual_hull/exact_polyhedral_hull.py) — альтернатива
        # polytope_hull для НЕВЫПУКЛЫХ объектов (сохраняет вогнутости силуэта, например уступ
        # на ручке между зажимаемой частью и тонким стержнем — polytope_hull размазывает его в
        # конус через cv2.convexHull контура, см. заметку задачи). Независимый чекбокс, можно
        # включать одновременно с voxel/polytope для сравнения — тот же набор кадров.
        self.show_camera_exact_checkbox = QCheckBox(
            "Показать точную 3D-реконструкцию (невыпуклая, exact polyhedral hull)"
        )
        self.show_camera_exact_checkbox.setEnabled(False)

        # GPU visual hull через torchhull (опциональная CUDA-зависимость, см. _TORCHHULL_AVAILABLE).
        # Sparse voxel octree + marching cubes — в 10-30× быстрее CPU-polytope на PARALLAX, точнее
        # на параллаксе (18/18 vs 16/18 в бенчмарке, см. bench_torchhull.py). Отображается как
        # облако точек (как voxel_carving) — torchhull.visual_hull возвращает verts меша, но для
        # визуализации Panel 1 удобнее точки.
        self.show_camera_torchhull_checkbox = QCheckBox(
            "Показать 3D-реконструкцию (GPU torchhull, sparse voxel octree)"
        )
        # Если torchhull недоступен (нет torch/CUDA/torchhull) — чек-бокс скрыт, не просто
        # отключён: пользователь не должен видеть метод, который нельзя включить.
        if not _TORCHHULL_AVAILABLE:
            self.show_camera_torchhull_checkbox.setVisible(False)
        self.show_camera_torchhull_checkbox.setEnabled(False)

        # Уровень октре torchhull (7=128³, 8=256³, 9=512³) — баланс точности/скорости/VRAM.
        # По умолчанию 7 (18/18 точность в бенчмарке, ~43 MB VRAM). Виден только при
        # доступности torchhull. Влияет на G4 (через `torchhull_level` в check_model_roll_g4)
        # и на визуализацию Panel 1 (через тот же параметр в _RecomputeWorker).
        self.torchhull_level_spinbox = QSpinBox()
        self.torchhull_level_spinbox.setRange(5, 10)
        self.torchhull_level_spinbox.setValue(7)
        self.torchhull_level_spinbox.setSuffix(" (octree level)")
        self.torchhull_level_spinbox.setToolTip(
            "Уровень октре torchhull: 7=128³ (по умолчанию, 43 MB VRAM, 18/18 точность), "
            "8=256³ (91 MB, точнее), 9=512³ (до 1.6 GB, максимум). Влияет на 3D-реконструкцию "
            "(Panel 1, чек-бокс 'GPU torchhull') и на G4 (Проверить модель) в Camera Mode."
        )
        if not _TORCHHULL_AVAILABLE:
            self.torchhull_level_spinbox.setVisible(False)
        self.torchhull_level_spinbox.setEnabled(False)

        # Геометрическая проверка "это вертикальная призма" (коробка/N-угольная колонна) по
        # верхней+боковым камерам (см. visual_hull/vertical_prism_fit.py, core/experiment.
        # check_vertical_prism) — точные габариты БЕЗ допущения выпуклости/масштабной
        # неоднозначности, отличной от Visual Hull-реконструкций выше: не показ существующей
        # реконструкции, а отдельный расчёт (может вернуть None — объект не похож на призму,
        # тогда используется обычная реконструкция). Результат — Panel 4, блок "Вертикальная
        # призма".
        self.check_prism_checkbox = QCheckBox(
            "Проверить как вертикальную призму (геометрически, Panel 4)"
        )
        self.check_prism_checkbox.setEnabled(False)

        # exact_polyhedral_hull почти точно совпадает с поверхностью объекта (в отличие от
        # выпуклого polytope_hull, который заметно вылезает наружу) — тест глубины прячет
        # полупрозрачный халл ЗА непрозрачным объектом, снаружи не видно ничего. Скрыть объект —
        # единственный надёжный способ увидеть такой почти-точный халл целиком.
        self.show_object_checkbox = QCheckBox("Показывать сам объект (сплошной меш)")
        self.show_object_checkbox.setChecked(True)

        self.show_slice_plane_checkbox = QCheckBox("Показывать плоскость среза на 3D-модели")
        self.show_slice_plane_checkbox.setChecked(True)
        self.show_hull_checkbox = QCheckBox("Показывать форму по силуэтам (3D)")
        self.show_roll_axis_checkbox = QCheckBox("Показывать ось переката (3D)")
        self.show_roll_axis_checkbox.setChecked(True)
        self.show_roll_axis_checkbox.setEnabled(False)

        self.search_button = QPushButton("Search counterexamples...")
        self.compare_button = QPushButton("Compare configurations...")
        self.export_button = QPushButton("Export...")

        self.check_model_button = QPushButton("Проверить модель")
        # Метод поиска оси переката: G4 (рабочий, по умолчанию) / F1 (диагностический,
        # без локального fallback) / SCAN (эталон-оракул, медленный ~1.5 с — для валидации
        # G4 на новых объектах при сомнениях). См. docs/method.md.
        self.roll_method_combo = QComboBox()
        self.roll_method_combo.addItem("G4 (PCA + локальный поиск)", "g4")
        self.roll_method_combo.addItem("F1 (PCA + диагонали, без fallback)", "f1")
        self.roll_method_combo.addItem("SCAN (перебор полусферы, медленно)", "scan")
        self.axis_step_spinbox = QDoubleSpinBox()
        self.axis_step_spinbox.setRange(1.0, 30.0)
        self.axis_step_spinbox.setValue(self._settings.axis_step_deg)
        self.axis_step_spinbox.setSuffix("°")
        # Шаг перебора нужен только SCAN — для G4/F1 скрываем (не используется)
        self.axis_step_label = QLabel("Шаг перебора осей:")
        self.axis_step_spinbox.setVisible(False)
        self.axis_step_label.setVisible(False)

        self._build_layout()
        self._connect_signals()
        self._restore_camera_settings()

        if self._objects:
            self._load_object(next(iter(self._objects)))

    def _build_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        panels_grid = QGridLayout()
        panels_grid.addWidget(self._boxed("Panel 1 — 3D модель", self.model_panel), 0, 0)
        panels_grid.addWidget(self._boxed("Panel 2 — Силуэты", self.silhouette_panel), 0, 1)
        panels_grid.addWidget(
            self._boxed("Panel 3 — Сечение / Visual Hull", self.visual_hull_panel), 1, 0
        )
        panels_grid.addWidget(
            self._boxed("Panel 4 — Метрики", self.metrics_panel, scrollable=True), 1, 1
        )
        root_layout.addLayout(panels_grid, stretch=3)

        controls_box = QGroupBox("Управление")
        controls_layout = QVBoxLayout(controls_box)

        object_row = QHBoxLayout()
        object_row.addWidget(QLabel("Объект:"))
        object_row.addWidget(self.object_combo, stretch=1)
        object_row.addWidget(self.open_button)
        controls_layout.addLayout(object_row)

        controls_layout.addWidget(self.orientation_controls)
        controls_layout.addWidget(self.camera_rig_button)
        controls_layout.addWidget(self.camera_rig_summary_label)
        controls_layout.addWidget(self.photo_capture_button)
        controls_layout.addWidget(self.grid_reconstruction_button)
        controls_layout.addWidget(self.photo_capture_status_label)

        camera_mode_row = QHBoxLayout()
        camera_mode_row.addWidget(self.camera_mode_checkbox, stretch=1)
        camera_mode_row.addWidget(self.belt_position_combo)
        controls_layout.addLayout(camera_mode_row)

        camera_fov_row = QHBoxLayout()
        camera_fov_row.addWidget(QLabel("FOV камеры:"))
        camera_fov_row.addWidget(self.camera_fov_spinbox)
        controls_layout.addLayout(camera_fov_row)

        camera_resolution_row = QHBoxLayout()
        camera_resolution_row.addWidget(QLabel("Разрешение камеры:"))
        camera_resolution_row.addWidget(self.camera_resolution_spinbox)
        controls_layout.addLayout(camera_resolution_row)

        camera_hull_row = QHBoxLayout()
        camera_hull_row.addWidget(self.show_camera_hull_checkbox, stretch=1)
        camera_hull_row.addWidget(self.camera_hull_use_ends_checkbox)
        controls_layout.addLayout(camera_hull_row)
        controls_layout.addWidget(self.show_camera_polytope_checkbox)
        controls_layout.addWidget(self.show_camera_exact_checkbox)
        controls_layout.addWidget(self.show_camera_torchhull_checkbox)
        controls_layout.addWidget(self.torchhull_level_spinbox)
        controls_layout.addWidget(self.check_prism_checkbox)
        controls_layout.addWidget(self.show_object_checkbox)

        controls_layout.addWidget(self.show_slice_plane_checkbox)
        controls_layout.addWidget(self.show_hull_checkbox)
        controls_layout.addWidget(self.show_roll_axis_checkbox)
        controls_layout.addWidget(self.search_button)
        controls_layout.addWidget(self.compare_button)
        controls_layout.addWidget(self.export_button)

        self.check_model_box = QGroupBox("Проверить модель (перекат)")
        check_model_layout = QVBoxLayout(self.check_model_box)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Метод поиска оси:"))
        method_row.addWidget(self.roll_method_combo, stretch=1)
        check_model_layout.addLayout(method_row)
        axis_step_row = QHBoxLayout()
        axis_step_row.addWidget(self.axis_step_label)
        axis_step_row.addWidget(self.axis_step_spinbox)
        check_model_layout.addLayout(axis_step_row)
        check_model_layout.addWidget(self.check_model_button)
        controls_layout.addWidget(self.check_model_box)

        controls_layout.addStretch(1)

        controls_box.setMaximumWidth(360)
        root_layout.addWidget(controls_box, stretch=1)

        # Виджеты панели управления STL, блокируемые на время работы с диалогом фото (см.
        # _set_stl_controls_enabled) — ИСКЛЮЧАЯ чекбоксы «3D-реконструкция по проекциям»
        # (show_camera_hull/_polytope/_exact/_torchhull, camera_hull_use_ends): они делят с
        # реконструкцией по фото один и тот же канал отображения в Panel 1 и должны оставаться
        # доступны. photo_capture_button тоже не в списке — сам открывает/поднимает этот же
        # диалог, блокировать его нечем и незачем.
        self._stl_control_widgets: list[QWidget] = [
            self.object_combo,
            self.open_button,
            self.orientation_controls,
            self.camera_rig_button,
            self.camera_mode_checkbox,
            self.belt_position_combo,
            self.camera_fov_spinbox,
            self.camera_resolution_spinbox,
            self.check_prism_checkbox,
            self.show_object_checkbox,
            self.show_slice_plane_checkbox,
            self.show_hull_checkbox,
            self.show_roll_axis_checkbox,
            self.search_button,
            self.compare_button,
            self.export_button,
            self.check_model_box,
        ]

    @staticmethod
    def _boxed(title: str, widget: QWidget, scrollable: bool = False) -> QGroupBox:
        """`scrollable=True` (Panel 4 — раскрывающаяся "Эталон формы", см. заметку задачи):
        без QScrollArea раскрытый аккордеон растягивает QGroupBox/грид панелей до размера
        содержимого — нижний край окна уходит за экран, а при повторном сворачивании назад не
        уменьшается (Qt не сжимает уже увеличенные виджеты автоматически). QScrollArea фиксирует
        размер ЯЧЕЙКИ грида под то, что реально помещается, и прокручивает содержимое ВНУТРИ себя
        — снаружи размер бокса не меняется ни при раскрытии, ни при сворачивании."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        if scrollable:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(widget)
            layout.addWidget(scroll)
        else:
            layout.addWidget(widget)
        return box

    def _connect_signals(self) -> None:
        self.object_combo.currentTextChanged.connect(self._on_object_selected)
        self.open_button.clicked.connect(self._on_open_clicked)
        self.orientation_controls.orientationChanged.connect(lambda *_: self._schedule_recompute())
        self.camera_rig_button.clicked.connect(self._on_camera_rig_button_clicked)
        self.photo_capture_button.clicked.connect(self._on_photo_capture_button_clicked)
        self.grid_reconstruction_button.clicked.connect(self._on_grid_reconstruction_button_clicked)
        self.camera_mode_checkbox.toggled.connect(self._on_camera_mode_toggled)
        self.belt_position_combo.currentTextChanged.connect(lambda _text: self._on_camera_params_changed())
        self.camera_fov_spinbox.valueChanged.connect(lambda _value: self._on_camera_params_changed())
        self.camera_resolution_spinbox.valueChanged.connect(lambda _value: self._on_camera_params_changed())
        self.show_camera_hull_checkbox.toggled.connect(self._on_show_camera_hull_toggled)
        self.camera_hull_use_ends_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.show_camera_polytope_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.show_camera_exact_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.show_camera_torchhull_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.torchhull_level_spinbox.valueChanged.connect(lambda _value: self._schedule_recompute())
        self.check_prism_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.show_object_checkbox.toggled.connect(self.model_panel.set_object_visible)
        self.visual_hull_panel.axisPosChanged.connect(self._on_axis_pos_changed)
        self.visual_hull_panel.yawChanged.connect(self._on_section_yaw_changed)
        self.show_slice_plane_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.show_hull_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.show_roll_axis_checkbox.toggled.connect(lambda _checked: self._schedule_recompute())
        self.search_button.clicked.connect(self._on_search_clicked)
        self.compare_button.clicked.connect(self._on_compare_clicked)
        self.export_button.clicked.connect(self._on_export_clicked)
        self.check_model_button.clicked.connect(self._on_check_model_clicked)
        self.roll_method_combo.currentIndexChanged.connect(self._on_roll_method_changed)
        self.metrics_panel.showRoundProjectionToggled.connect(self._on_show_projection_toggled)

    def _restore_camera_settings(self) -> None:
        """Восстановить блок Camera Mode (вкл/выкл, позиция на ленте, FOV, риг камер,
        разрешение) из прошлого запуска — пользователь обычно подбирает их под конкретный риг
        и не хочет вводить заново каждый раз (см. обсуждение с пользователем). Остальные
        настройки (объект, ориентация) сессия от сессии обычно разные, поэтому не сохраняются.
        Дефолты при первом запуске/пустом QSettings — как и раньше, из config/app_settings.yaml
        (self._settings) для fov/разрешения, из первого (встроенного) пресета
        config/camera_rig_presets.yaml для рига камер.

        Org/app name переданы явно (не через QSettings() без аргументов) — не хотим зависеть от
        того, что main.py (единственное место с QCoreApplication.setOrganizationName/
        setApplicationName) успел выполниться раньше первого использования, например в тестах,
        которые создают QApplication напрямую."""
        settings = QSettings("roboson_tools", "roboson_tools_gui")
        self.camera_mode_checkbox.setChecked(
            settings.value("camera/mode_enabled", False, type=bool)
        )
        belt_label = settings.value(
            "camera/belt_position", _BELT_POSITION_LABELS["center"], type=str
        )
        if belt_label in _BELT_POSITION_BY_LABEL:
            self.belt_position_combo.setCurrentText(belt_label)
        self.camera_fov_spinbox.setValue(
            settings.value("camera/fov_deg", self._settings.camera_fov_deg, type=float)
        )
        self.camera_resolution_spinbox.setValue(
            settings.value(
                "camera/resolution_px", self._settings.raster_resolution_px, type=int
            )
        )
        rig_json = settings.value("camera/rig_cameras_json", "", type=str)
        if rig_json:
            try:
                pairs = json.loads(rig_json)
                self._camera_rig = [
                    CameraSpec(angle_deg=float(a), distance_mm=float(d)) for a, d in pairs
                ]
            except (ValueError, TypeError):
                pass  # сохранённое значение повреждено — остаётся дефолтный риг из __init__
        self._update_camera_rig_summary()

    def _save_camera_settings(self) -> None:
        settings = QSettings("roboson_tools", "roboson_tools_gui")
        settings.setValue("camera/mode_enabled", self.camera_mode_checkbox.isChecked())
        settings.setValue("camera/belt_position", self.belt_position_combo.currentText())
        settings.setValue("camera/fov_deg", self.camera_fov_spinbox.value())
        settings.setValue("camera/resolution_px", self.camera_resolution_spinbox.value())
        settings.setValue(
            "camera/rig_cameras_json",
            json.dumps([[c.angle_deg, c.distance_mm] for c in self._camera_rig]),
        )

    def _update_camera_rig_summary(self) -> None:
        self.camera_rig_summary_label.setText(
            ", ".join(f"{c.angle_deg:g}°/{c.distance_mm:g}мм" for c in self._camera_rig)
        )

    def _on_camera_rig_button_clicked(self) -> None:
        dialog = CameraRigDialog(self._camera_rig, self)
        dialog.rigChanged.connect(self._on_camera_rig_changed)
        dialog.exec()

    def _on_camera_rig_changed(self, cameras: list[CameraSpec]) -> None:
        self._camera_rig = cameras
        self._update_camera_rig_summary()
        self._on_camera_params_changed()

    def _on_photo_capture_button_clicked(self) -> None:
        if self._photo_capture_dialog is None:
            try:
                marker_layout = load_marker_layout()
                intrinsics = load_camera_intrinsics()
            except (OSError, KeyError, ValueError) as exc:
                self.photo_capture_status_label.setText(
                    f"Не удалось загрузить config/aruco_markers.yaml или "
                    f"config/camera_intrinsics.yaml: {exc}"
                )
                return
            self.photo_capture_status_label.setText("")
            self._photo_capture_dialog = PhotoCaptureDialog(
                marker_layout,
                intrinsics,
                self.model_panel,
                self.silhouette_panel,
                self.visual_hull_panel,
                self.metrics_panel,
                self,
            )
            self._photo_capture_dialog.shown.connect(self._on_photo_dialog_shown)
            self._photo_capture_dialog.closing.connect(self._on_photo_dialog_closing)
        self._photo_capture_dialog.show()
        self._photo_capture_dialog.raise_()
        self._photo_capture_dialog.activateWindow()

    def _on_grid_reconstruction_button_clicked(self) -> None:
        if self._grid_reconstruction_dialog is None:
            self._grid_reconstruction_dialog = GridReconstructionDialog(self.model_panel, self)
            self._grid_reconstruction_dialog.shown.connect(self._on_grid_dialog_shown)
            self._grid_reconstruction_dialog.closing.connect(self._on_grid_dialog_closing)
        self._grid_reconstruction_dialog.show()
        self._grid_reconstruction_dialog.raise_()
        self._grid_reconstruction_dialog.activateWindow()

    def _on_grid_dialog_shown(self) -> None:
        # Диалог рисует халл в Panel 1 через тот же канал (model_panel.set_camera_hull_*),
        # что Camera Mode STL и PhotoCaptureDialog — блокируем панель STL, пока диалог
        # открыт, по тому же образцу, что _on_photo_dialog_shown (см. docstring
        # _set_stl_controls_enabled).
        self._set_stl_controls_enabled(False)

    def _on_grid_dialog_closing(self) -> None:
        self._set_stl_controls_enabled(True)
        # Диалог мог спрятать STL-объект/плоскость среза, пока показывал реконструкцию по
        # сетке (см. GridReconstructionDialog._show_variant — по замечанию пользователя
        # 2026-07-25, эти два независимых объекта в разных СК только мешали друг другу в
        # Panel 1). Восстанавливаем: object_visible — сразу (дёшево, без debounce), плоскость
        # среза/форму по силуэтам — через обычный пересчёт (используют текущую ориентацию/
        # ось, которую проще получить пересчётом, чем восстанавливать вручную).
        self.model_panel.set_object_visible(self.show_object_checkbox.isChecked())
        if self._grid_reconstruction_dialog is not None:
            self._grid_reconstruction_dialog.clear_hull_display()
        self._schedule_recompute()

    def _on_photo_dialog_shown(self) -> None:
        self._set_stl_controls_enabled(False)

    def _on_photo_dialog_closing(self) -> None:
        self._set_stl_controls_enabled(True)
        # Восстанавливаем обычный вид: объект STL (спрятанный при клике «3D-реконструкция...»,
        # см. PhotoCaptureDialog._on_reconstruct_clicked) и Panel 2/3/4, занятые данными
        # реконструкции по фото, — иначе они молча оставались бы в состоянии "по фото" даже
        # после того, как панель управления STL снова стала интерактивной.
        self.model_panel.set_object_visible(self.show_object_checkbox.isChecked())
        self.silhouette_panel.set_photo_observations(None)
        self.visual_hull_panel.set_photo_reconstruction(None, None, None)
        self.metrics_panel.set_photo_reconstruction(None)

    def _set_stl_controls_enabled(self, enabled: bool) -> None:
        """Блокирует панель управления STL, пока открыт немодальный диалог фото — изменение
        ориентации/объекта/большинства настроек во время работы с реальными фото не имеет
        смысла (см. заметку задачи). Исключение — чекбоксы «3D-реконструкция по проекциям»
        (`self._stl_controls_exempt`): они управляют ТЕМ ЖЕ каналом отображения в Panel 1
        (`model_panel.set_camera_hull_polytope`/`_exact`/...), которым пользуется и реконструкция
        по фото (`PhotoCaptureDialog._on_reconstruct_clicked`) — пользователь должен иметь
        возможность переключать их, не закрывая диалог. Не трогаем `controls_box` целиком одним
        `setEnabled` — Qt каскадно блокирует ДЕТЕЙ независимо от их собственного enabled-состояния,
        поэтому список виджетов формируется явно, а их состояние ДО блокировки запоминается и
        восстанавливается (а не просто выставляется в True) — иначе виджеты, отключённые другой
        бизнес-логикой (например `belt_position_combo`, когда Camera Mode выключен), ошибочно
        стали бы кликабельными после закрытия диалога."""
        if not enabled:
            self._stl_controls_saved_enabled = {
                widget: widget.isEnabled() for widget in self._stl_control_widgets
            }
            for widget in self._stl_control_widgets:
                widget.setEnabled(False)
        else:
            saved = getattr(self, "_stl_controls_saved_enabled", {})
            for widget in self._stl_control_widgets:
                widget.setEnabled(saved.get(widget, True))

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_camera_settings()
        # Диалог фото может быть скрыт (не уничтожен — см. _on_photo_capture_button_clicked),
        # поэтому фоновый процесс SAM3 (если поднимался) останавливаем только здесь, при
        # закрытии всего приложения, а не в closeEvent самого диалога.
        if self._photo_capture_dialog is not None:
            self._photo_capture_dialog.shutdown_segmentation_backend()
        if self._grid_reconstruction_dialog is not None:
            self._grid_reconstruction_dialog.shutdown_segmentation_backend()
        super().closeEvent(event)

    def _on_object_selected(self, key: str) -> None:
        if key:
            self._load_object(key)

    def _on_open_clicked(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Открыть STL", "", "STL files (*.stl *.STL)"
        )
        if path_str:
            self._current_is_round_reference = None
            self._load_mesh(Path(path_str))

    def _load_object(self, key: str) -> None:
        self._current_is_round_reference = self._objects[key].is_round_reference
        self._load_mesh(self._objects[key].path)

    def _load_mesh(self, path: Path) -> None:
        self._current_stl_path = path
        self._mesh = load_stl(path)
        self._stl_true_dims = STL_DIMS_PENDING
        self._start_stl_true_dims_worker(self._mesh)
        self._axis_pos_override = None
        self._last_roll_check = None
        self._roll_axis_base = None
        self._show_round_projection = False
        self.show_roll_axis_checkbox.setEnabled(False)
        self.model_panel.load_object(self._mesh)
        self.metrics_panel.set_roll_check_result(None)
        self.visual_hull_panel.set_checked_sections([])
        self.visual_hull_panel.set_projection(None, None)
        self._schedule_recompute()

    def _start_stl_true_dims_worker(self, mesh: Mesh) -> None:
        """Запускает фоновый расчёт истинных габаритов STL для только что загруженного меша
        (см. _MeshTrueDimsWorker). Если предыдущий воркер (для прошлого STL) ещё не успел
        завершиться — не ждём и не убиваем его насильно (мог бы утащить UI-поток в блокировку
        на wait()), просто отсоединяем его сигнал: _on_stl_true_dims_ready всё равно проверяет
        идентичность отправителя и отбросил бы устаревший результат, но без disconnect он смог
        бы прилететь ПОСЛЕ того, как атрибут уже переиспользован новым воркером."""
        if self._stl_dims_worker is not None:
            self._stl_dims_worker.result_ready.disconnect(self._on_stl_true_dims_ready)
        worker = _MeshTrueDimsWorker(mesh, self)
        worker.result_ready.connect(self._on_stl_true_dims_ready)
        self._stl_dims_worker = worker
        worker.start()

    def _on_stl_true_dims_ready(self, dims: tuple[float, float, float] | None) -> None:
        self._stl_true_dims = dims
        self._schedule_recompute()
        if self._last_roll_check is not None:
            prism_override = (
                prism_dims(self._last_prism_result)
                if self._last_prism_result is not None
                else None
            )
            self.metrics_panel.set_roll_check_result(
                self._last_roll_check, prism_override, self._stl_true_dims
            )

    def _schedule_recompute(self) -> None:
        # Отсоединяем сигнал уже запущенного воркера СРАЗУ, а не только когда сработает
        # debounce и стартует новый (см. `_start_recompute_worker`) — иначе в окне между
        # "что-то изменилось" и "прошло DEBOUNCE_MS" быстро завершившийся старый воркер
        # успевает применить УСТАРЕВШИЙ результат (для прежнего меша/настроек) поверх уже
        # обновлённого UI. Баг, найденный пользователем: чекбокс "плоскость среза" вело себя
        # непредсказуемо именно после быстрой смены модели на маленьких мешах — там
        # предыдущий воркер чаще всего успевает завершиться раньше, чем истечёт 50мс
        # debounce и `_recompute()` доберётся до собственного disconnect в `_start_recompute_
        # worker`, так что тот disconnect срабатывал СЛИШКОМ ПОЗДНО.
        self._disconnect_recompute_worker()
        self._debounce.start()

    def _disconnect_recompute_worker(self) -> None:
        """Идемпотентно отсоединяет `result_ready` текущего `_recompute_worker`, если он есть
        и ещё подключён (не убивает сам поток — он доработает в фоне и просто не будет
        услышан, тот же принцип, что и у `_start_stl_true_dims_worker`)."""
        if self._recompute_worker is None:
            return
        try:
            self._recompute_worker.result_ready.disconnect(self._on_recompute_ready)
        except TypeError:
            pass  # уже отсоединён (например, _schedule_recompute вызвали дважды подряд)

    def _on_camera_params_changed(self) -> None:
        """Чек-бокс Camera Mode/FOV/позиция на ленте меняют то, ЧТО измеряет "Проверить модель"
        (`check_model_roll` перечитывает их заново при каждом клике, см. `_on_check_model_clicked`)
        — но сам расчёт там дорогой (перебор осей) и не входит в debounce-пересчёт `_recompute()`,
        поэтому последний показанный результат "Истинные габариты" при таких изменениях
        становится обманчиво устаревшим (посчитан для старого режима силуэтов). Явно сбрасываем
        его на "—", а не оставляем цифры, которые уже не соответствуют текущим настройкам —
        пользователь должен нажать "Проверить модель" заново, чтобы получить актуальное число.
        Тот же набор сбросов, что и при загрузке нового меша (см. `_load_mesh`) — старая ось
        переката тоже посчитана для старого режима силуэтов и больше не действительна."""
        self._last_roll_check = None
        self._roll_axis_base = None
        self._show_round_projection = False
        self.show_roll_axis_checkbox.setEnabled(False)
        self.metrics_panel.set_roll_check_result(None)
        self.visual_hull_panel.set_checked_sections([])
        self.visual_hull_panel.set_projection(None, None)
        self._schedule_recompute()

    def _on_camera_mode_toggled(self, checked: bool) -> None:
        self.belt_position_combo.setEnabled(checked)
        self.camera_fov_spinbox.setEnabled(checked)
        self.camera_resolution_spinbox.setEnabled(checked)
        self.show_camera_hull_checkbox.setEnabled(checked)
        self.show_camera_polytope_checkbox.setEnabled(checked)
        self.show_camera_exact_checkbox.setEnabled(checked)
        if _TORCHHULL_AVAILABLE:
            self.show_camera_torchhull_checkbox.setEnabled(checked)
            self.torchhull_level_spinbox.setEnabled(checked)
        self.check_prism_checkbox.setEnabled(checked)
        if not checked:
            self.show_camera_hull_checkbox.setChecked(False)
            self.show_camera_polytope_checkbox.setChecked(False)
            self.show_camera_exact_checkbox.setChecked(False)
            self.show_camera_torchhull_checkbox.setChecked(False)
            self.check_prism_checkbox.setChecked(False)
        self._on_camera_params_changed()

    def _on_show_camera_hull_toggled(self, _checked: bool) -> None:
        self._schedule_recompute()

    def _on_axis_pos_changed(self, pos: float) -> None:
        self._axis_pos_override = pos
        self._schedule_recompute()

    def _on_section_yaw_changed(self, yaw_deg: float) -> None:
        roll, pitch, _yaw = self.orientation_controls.orientation()
        self.orientation_controls.set_orientation(roll, pitch, yaw_deg)

    def _recompute(self) -> None:
        """Быстрая часть пересчёта — только то, что дёшево и должно откликаться мгновенно
        (отображение повёрнутого меша). Всё остальное (силуэты/реконструкция/габариты/Camera
        Mode 3D) уходит в фоновый `_RecomputeWorker` (см. его докстринг — раньше вся эта работа
        шла синхронно здесь же и подвешивала GUI на секунды на тяжёлых мешах)."""
        if self._mesh is None:
            return

        roll, pitch, yaw = self.orientation_controls.orientation()
        orientation = Orientation(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
        angles = rig_angles(self._camera_rig)

        oriented_mesh = apply_orientation(self._mesh, roll, pitch, yaw)
        self.model_panel.set_orientation_mesh(oriented_mesh)
        self.visual_hull_panel.set_yaw(yaw)

        use_camera = self.camera_mode_checkbox.isChecked()
        belt_position = _BELT_POSITION_BY_LABEL[self.belt_position_combo.currentText()]
        # Разрешение — общий параметр evaluate()/check_model_roll() для ОБОИХ режимов силуэтов
        # (один resolution_px на вызов): при выключенном Camera Mode Analytical Mode не трогаем
        # — оставляем значение из конфига, а не спинбокс Camera Mode.
        resolution_px = (
            self.camera_resolution_spinbox.value() if use_camera else self._settings.raster_resolution_px
        )

        request = _RecomputeRequest(
            mesh=self._mesh,
            oriented_mesh=oriented_mesh,
            orientation=orientation,
            angles=angles,
            axis_pos_override=self._axis_pos_override,
            use_camera=use_camera,
            belt_position=belt_position,
            resolution_px=resolution_px,
            camera_fov_deg=self.camera_fov_spinbox.value(),
            camera_distances=rig_to_distances(self._camera_rig),
            roundness_threshold=self._settings.roundness_threshold,
            check_prism=use_camera and self.check_prism_checkbox.isChecked(),
            show_voxel=use_camera and self.show_camera_hull_checkbox.isChecked(),
            show_polytope=use_camera and self.show_camera_polytope_checkbox.isChecked(),
            show_exact=use_camera and self.show_camera_exact_checkbox.isChecked(),
            show_torchhull=use_camera and _TORCHHULL_AVAILABLE and self.show_camera_torchhull_checkbox.isChecked(),
            torchhull_level=self.torchhull_level_spinbox.value() if _TORCHHULL_AVAILABLE else 7,
            camera_hull_use_ends=self.camera_hull_use_ends_checkbox.isChecked(),
        )
        self._start_recompute_worker(request)

    def _start_recompute_worker(self, request: _RecomputeRequest) -> None:
        """Как `_start_stl_true_dims_worker` — не ждём и не убиваем предыдущий воркер
        насильно (мог бы утащить GUI-поток в блокировку на wait()). Реальное отсоединение уже
        произошло в `_schedule_recompute()` (см. её докстринг про гонку); повторный вызов
        здесь — просто подстраховка (идемпотентна)."""
        self._disconnect_recompute_worker()
        worker = _RecomputeWorker(request, self)
        worker.result_ready.connect(self._on_recompute_ready)
        self._recompute_worker = worker
        worker.start()

    def _on_recompute_ready(self, payload: _RecomputeResult) -> None:
        """Применяет результат `_RecomputeWorker` к панелям — тело в точности бывшего
        `_recompute()` (после `evaluate(...)`) + бывшего `_update_camera_mode`, просто
        выполняется на GUI-потоке ПОСЛЕ того, как фоновый поток посчитал всё тяжёлое."""
        req = payload.request
        result = payload.evaluation
        self._last_result = result
        self._last_orientation = req.orientation

        current_axis_pos = (
            req.axis_pos_override if req.axis_pos_override is not None else result.axis_pos
        )
        self.visual_hull_panel.set_axis_range(
            result.bbox_min[0], result.bbox_max[0], current=current_axis_pos
        )
        self.silhouette_panel.set_silhouettes(result.silhouettes)

        if req.use_camera:
            # Явная 3x3-сетка (см. _SIM_RIG_ANGLES) приоритетнее обычной силуэт-сетки на один
            # belt_position — при её показе список Начало/Центр/Конец теряет смысл (визуализи-
            # руются все три сразу), поэтому блокируется (решение пользователя, 2026-07-25).
            self.silhouette_panel.set_camera_silhouette_grid(
                payload.camera_grid_tiles, payload.camera_caption_suffix
            )
            if payload.camera_grid_tiles is None:
                self.silhouette_panel.set_camera_silhouettes(
                    payload.camera_silhouettes, payload.camera_caption_suffix
                )
            self.belt_position_combo.setEnabled(payload.camera_grid_tiles is None)
            self.model_panel.set_belt_offset(payload.belt_offset_dx)
            self.model_panel.set_camera_hull_points(payload.voxel_points if req.show_voxel else None)
            self.model_panel.set_camera_hull_polytope(
                payload.polytope_vertices if req.show_polytope else None,
                payload.polytope_faces if req.show_polytope else None,
            )
            self.model_panel.set_camera_hull_exact(
                payload.exact_vertices if req.show_exact else None,
                payload.exact_faces if req.show_exact else None,
            )
            self.model_panel.set_camera_hull_torchhull(
                payload.torchhull_points if req.show_torchhull else None
            )
            # Блок «3D-реконструкция» в Panel 4 (Метрики) — время и N точек каждого включённого
            # метода. Помогает сравнивать скорость CPU (voxel/polytope/exact) и GPU (torchhull)
            # прямо в GUI, не запуская bench_*.py.
            n_points = {}
            if payload.voxel_points is not None:
                n_points["voxel"] = len(payload.voxel_points)
            if payload.polytope_vertices is not None:
                n_points["polytope"] = len(payload.polytope_vertices)
            if payload.exact_vertices is not None:
                n_points["exact"] = len(payload.exact_vertices)
            if payload.torchhull_points is not None:
                n_points["torchhull"] = len(payload.torchhull_points)
            self.metrics_panel.set_recon_times(payload.recon_times_ms, n_points)
        else:
            self.silhouette_panel.set_camera_silhouette_grid(None)
            self.silhouette_panel.set_camera_silhouettes(None)
            self.model_panel.set_belt_offset(None)
            self.model_panel.set_camera_hull_points(None)
            self.model_panel.set_camera_hull_polytope(None, None)
            self.model_panel.set_camera_hull_exact(None, None)
            self.model_panel.set_camera_hull_torchhull(None)
            self.metrics_panel.set_recon_times(None)

        self.visual_hull_panel.set_result(result.polygon_coords, result.roundness)
        self.metrics_panel.set_reference(self._current_is_round_reference)
        self.metrics_panel.set_result(result.roundness)

        self._last_prism_result = payload.prism_result
        self.metrics_panel.set_prism_result(payload.prism_result, self._stl_true_dims)

        if payload.prism_result is not None:
            self.metrics_panel.set_dims(
                prism_dims(payload.prism_result), self._stl_true_dims, from_prism=True
            )
            solid = prism_mesh(payload.prism_result)
            self.model_panel.set_prism_hull(solid.vertices, solid.faces)
        else:
            self.metrics_panel.set_dims(result.hull_dims, self._stl_true_dims)
            self.model_panel.set_prism_hull(None, None)

        self.metrics_panel.set_simple_camera_dims(payload.simple_dims, self._stl_true_dims)
        self.metrics_panel.set_simple_camera_dims_v3(payload.simple_dims_v3, self._stl_true_dims)

        show_hull = self.show_hull_checkbox.isChecked()
        self.model_panel.set_hull_slices(result.hull_slices if show_hull else None)

        # Ось переката (3D): хранится в НЕПОВЁРНУТОЙ системе координат меша (см. атрибут) и
        # каждый пересчёт заново поворачивается текущей ориентацией — иначе линия оставалась бы
        # неподвижной при повороте объекта слайдерами Roll/Pitch/Yaw.
        axis_world = None
        if self._roll_axis_base is not None:
            axis_world = (
                rotation_matrix(req.orientation.roll_deg, req.orientation.pitch_deg, req.orientation.yaw_deg)
                @ self._roll_axis_base
            )
        show_axis = self.show_roll_axis_checkbox.isChecked()
        self.model_panel.set_roll_axis(axis_world if show_axis else None)

        # Плоскость среза: по умолчанию перпендикулярна X на позиции Slice X. Если одновременно
        # включены «показывать плоскость среза» и «показать круглую проекцию» (Panel 4) —
        # плоскость вместо этого перпендикулярна найденной оси переката и проходит через центр
        # объекта (начало координат инвариантно относительно поворота вокруг него).
        show_plane = self.show_slice_plane_checkbox.isChecked()
        if not show_plane:
            self.model_panel.set_slice_plane(None)
        elif self._show_round_projection and axis_world is not None:
            self.model_panel.set_slice_plane(np.zeros(3), axis_world)
        else:
            self.model_panel.set_slice_plane(np.array([current_axis_pos, 0.0, 0.0]))

        # Слайдеры Panel 3 (Slice X / Yaw): пока активен показ круглой проекции, отображают (не
        # интерактивно) положение и азимут найденной оси, а не последний обычный срез — иначе
        # после выключения показа проекции слайдеры остаются в положении, не связанном с осью.
        if self._show_round_projection and axis_world is not None:
            azim_deg, _elev_deg = axis_to_angles(axis_world)
            self.visual_hull_panel.show_round_axis_marker(azim_deg, x_pos=0.0)

    def _on_roll_method_changed(self) -> None:
        """Показать/скрыть шаг перебора осей — нужен только SCAN (перебор полусферы).
        G4/F1 не используют шаг (у G4 своя сетка локального поиска, у F1 фиксированные
        6 направлений)."""
        is_scan = self.roll_method_combo.currentData() == "scan"
        self.axis_step_spinbox.setVisible(is_scan)
        self.axis_step_label.setVisible(is_scan)

    def _on_check_model_clicked(self) -> None:
        if self._mesh is None:
            return
        roll, pitch, yaw = self.orientation_controls.orientation()
        orientation = Orientation(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
        angles = rig_angles(self._camera_rig)

        use_camera = self.camera_mode_checkbox.isChecked()
        belt_position = _BELT_POSITION_BY_LABEL[self.belt_position_combo.currentText()]
        resolution_px = (
            self.camera_resolution_spinbox.value() if use_camera else self._settings.raster_resolution_px
        )
        method = self.roll_method_combo.currentData()
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            common_kwargs = dict(
                mesh=self._mesh,
                orientation=orientation,
                view_angles_deg=angles,
                resolution_px=resolution_px,
                roundness_threshold=self._settings.roundness_threshold,
                num_slices=self._settings.model_check_samples,
                use_camera_silhouettes=use_camera,
                camera_fov_deg=self.camera_fov_spinbox.value(),
                camera_distances=rig_to_distances(self._camera_rig),
                belt_position=belt_position,
            )
            if method == "g4":
                # torchhull_level/parallax — только при Camera Mode + доступности torchhull.
                # `torchhull_parallax` привязан к чекбоксу «+ начало/конец ленты» — тот же
                # смысл (3 позиции на ленте для уточнения формы).
                g4_kwargs = dict(common_kwargs)
                if use_camera and _TORCHHULL_AVAILABLE:
                    g4_kwargs["torchhull_level"] = self.torchhull_level_spinbox.value()
                    g4_kwargs["torchhull_parallax"] = self.camera_hull_use_ends_checkbox.isChecked()
                result = check_model_roll_g4(**g4_kwargs)
            elif method == "f1":
                result = check_model_roll_f1(**common_kwargs)
            else:  # scan
                result = check_model_roll(
                    **common_kwargs,
                    axis_step_deg=self.axis_step_spinbox.value(),
                )
        finally:
            self.unsetCursor()

        self._last_roll_check = result
        if result.found_round and result.best_axis_dir is not None:
            # best_axis_dir найден в СЕЙЧАС повёрнутой системе координат (roll/pitch/yaw на
            # момент клика) — переводим в базовую (неповёрнутую) систему меша, чтобы
            # _recompute() мог заново поворачивать её текущей ориентацией на каждый пересчёт.
            r_mat = rotation_matrix(roll, pitch, yaw)
            self._roll_axis_base = r_mat.T @ np.asarray(result.best_axis_dir)
        else:
            self._roll_axis_base = None
        self.show_roll_axis_checkbox.setEnabled(self._roll_axis_base is not None)

        prism_override = (
            prism_dims(self._last_prism_result) if self._last_prism_result is not None else None
        )
        self.metrics_panel.set_roll_check_result(result, prism_override, self._stl_true_dims)
        self.visual_hull_panel.set_checked_sections(result.section_hits)
        self._schedule_recompute()

    def _on_show_projection_toggled(self, checked: bool) -> None:
        """Чек-бокс в Panel 4: показать в Panel 3 проекцию формы вдоль найденной оси переката
        (вместо текущего сечения), а в Panel 1 — плоскость среза вдоль этой же оси
        (см. _recompute)."""
        self._show_round_projection = checked and self._last_roll_check is not None
        if not self._show_round_projection:
            self.visual_hull_panel.set_projection(None, None)
        else:
            self.visual_hull_panel.set_projection(
                self._last_roll_check.best_projection_coords,
                self._last_roll_check.best_roundness,
            )
        self._schedule_recompute()

    def _on_search_clicked(self) -> None:
        if self._mesh is None:
            return
        angles = rig_angles(self._camera_rig)
        key = self.object_combo.currentText()
        is_round_reference = self._objects[key].is_round_reference if key in self._objects else None

        dialog = SearchDialog(self._mesh, angles, is_round_reference, self)
        dialog.orientationPicked.connect(self._on_orientation_picked)
        dialog.exec()

    def _on_orientation_picked(self, roll: float, pitch: float, yaw: float) -> None:
        self.orientation_controls.set_orientation(roll, pitch, yaw)

    def _on_compare_clicked(self) -> None:
        if self._mesh is None:
            return
        roll, pitch, yaw = self.orientation_controls.orientation()
        orientation = Orientation(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
        dialog = CompareDialog(
            self._mesh,
            orientation,
            self._axis_pos_override,
            _COMPARE_ANGLE_PRESETS,
            self,
        )
        dialog.exec()

    def _on_export_clicked(self) -> None:
        if self._mesh is None or self._last_result is None or self._current_stl_path is None:
            return
        default_dir = Path("exports") / self._current_stl_path.stem
        dialog = ExportDialog(
            self._current_stl_path, self._last_orientation, self._last_result, default_dir, self
        )
        dialog.exec()
