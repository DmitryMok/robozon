"""Panel 4 — метрики: Rin, Rout, k=Rin/Rout, PASS/FAIL + эталонная метка формы из конфига,
габариты (по силуэтам и истинные), плюс блок результатов "Проверить модель" — потенциал
к перекату по круглой проекции (см. core/experiment.check_model_roll и docs/method.md).

is_round_reference (эталон) — статичный признак ИСТИННОЙ формы объекта (см. config/objects.yaml),
не зависит от текущей ориентации. PASS/FAIL — результат для ТЕКУЩЕГО ракурса/ориентации. Эти два
показателя могут расходиться (например, если ось X сейчас не совпадает с осью симметрии круглого
объекта в его исходной STL-ориентации) — панель явно показывает такое расхождение.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from ...core.experiment import (
    GRollResult,
    MultiSideCameraDims,
    PcaRollResult,
    RollCheckResult,
    ROLL_VERDICT_NOT_ROUND,
    ROLL_VERDICT_ROUND,
    ROLL_VERDICT_UNCERTAIN,
    SimpleCameraDims,
    prism_dims,
)
from ...metrics.roundness import RoundnessResult
from ...visual_hull.vertical_prism_fit import PrismFitResult

# Сентинел для `mesh_dims`/`stl_true_dims`-параметров ниже: mesh_true_dims (core/experiment) —
# дорогой перебор направлений на полусфере, в GUI считается в фоновом QThread (см.
# main_window._MeshTrueDimsWorker), а не при каждом _recompute(). Пока фоновый расчёт не
# завершился, вызывающая сторона передаёт этот сентинел вместо None — отличаем "ещё считается"
# от "данных нет" (None), чтобы не показывать пользователю обманчивый "—"/ошибку "нет данных".
STL_DIMS_PENDING = object()

_PENDING_TEXT = "рассчитывается…"


def _dims_text(dims: tuple[float, float, float] | None) -> str:
    if dims is STL_DIMS_PENDING:
        return _PENDING_TEXT
    if dims is None:
        return "—"
    return " × ".join(f"{v:.1f}" for v in dims)


def _dims_error_text(
    dims: tuple[float, float, float] | None,
    mesh_dims: tuple[float, float, float] | None,
    sort: bool = False,
) -> str:
    """Ошибка габаритов: `dims` (по силуэтам/реконструкции) относительно `mesh_dims` (истина,
    STL). `sort=True` — оба варианта сортируются по убыванию перед сравнением: нужно для троек,
    не привязанных к мировым осям (например, `true_dims`/`prism_dims` — см. их докстринги),
    иначе разница считается по случайно совпавшим индексам, а не по факту размера."""
    if mesh_dims is STL_DIMS_PENDING:
        return _PENDING_TEXT
    if dims is None or mesh_dims is None:
        return "—"
    a = sorted(dims, reverse=True) if sort else list(dims)
    b = sorted(mesh_dims, reverse=True) if sort else list(mesh_dims)
    diffs = [x - y for x, y in zip(a, b)]
    max_pct = max((abs(d) / y * 100.0 if y > 1e-9 else 0.0) for d, y in zip(diffs, b))
    diff_text = " / ".join(f"{d:+.1f}" for d in diffs)
    return f"Δ {diff_text} мм (макс. {max_pct:.1f}%)"


class MetricsPanel(QWidget):
    showRoundProjectionToggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._is_round_reference: bool | None = None
        self._last_passed: bool | None = None

        self.reference_label = QLabel("—")
        self.r_in_label = QLabel("—")
        self.r_out_label = QLabel("—")
        self.k_label = QLabel("—")
        self.pass_label = QLabel("—")
        self.hull_dims_label = QLabel("—")
        self.mesh_dims_label = QLabel("—")
        self.hull_dims_error_label = QLabel("—")
        self.hull_dims_error_label.setWordWrap(True)
        self.mismatch_label = QLabel("")
        self.mismatch_label.setWordWrap(True)

        self.check_status_label = QLabel("не выполнялась")
        self.check_dims_label = QLabel("—")
        self.check_dims_error_label = QLabel("—")
        self.check_dims_error_label.setWordWrap(True)
        self.true_dims_label = QLabel("—")
        self.true_dims_error_label = QLabel("—")
        self.true_dims_error_label.setWordWrap(True)
        self.check_axis_label = QLabel("—")
        self.check_axis_label.setWordWrap(True)
        self.check_sections_label = QLabel("—")
        self.check_time_label = QLabel("—")

        self.show_projection_checkbox = QCheckBox("Показать круглую проекцию (Panel 3)")
        self.show_projection_checkbox.setEnabled(False)
        self.show_projection_checkbox.toggled.connect(self.showRoundProjectionToggled)

        self.prism_status_label = QLabel("не выполнялась")
        self.prism_height_label = QLabel("—")
        self.prism_footprint_label = QLabel("—")
        self.prism_footprint_label.setWordWrap(True)
        self.prism_dims_error_label = QLabel("—")
        self.prism_dims_error_label.setWordWrap(True)

        self.simple_dims_label = QLabel("—")
        self.simple_dims_error_label = QLabel("—")
        self.simple_dims_error_label.setWordWrap(True)

        self.v3_dims_label = QLabel("—")
        self.v3_dims_error_label = QLabel("—")
        self.v3_dims_error_label.setWordWrap(True)

        # 3D-реконструкции Camera Mode (Panel 1, чек-боксы voxel/polytope/exact/torchhull) —
        # отдельный блок с временем каждого включённого метода и числом точек/вершин. Помогает
        # сравнивать скорость CPU (voxel/polytope/exact) и GPU (torchhull) методов прямо в GUI,
        # не запуская bench_*.py. Метка обновляется в `set_recon_times` из `_on_recompute_ready`.
        # Свёрнуто по умолчанию (QGroupBox.setChecked(False) ниже) — как «Эталон формы», т.к.
        # это справочная/диагностическая информация, а не основные габариты/вердикт G4.
        self.recon_status_label = QLabel("не выполнялась")
        self.recon_voxel_label = QLabel("—")
        self.recon_polytope_label = QLabel("—")
        self.recon_exact_label = QLabel("—")
        self.recon_torchhull_label = QLabel("—")
        self.recon_group = QGroupBox("3D-реконструкция (Camera Mode, скорость/качество методов)")
        self.recon_group.setCheckable(True)
        recon_content = QWidget()
        recon_form = QFormLayout(recon_content)
        recon_form.addRow("Итог", self.recon_status_label)
        recon_form.addRow("Voxel carving (CPU)", self.recon_voxel_label)
        recon_form.addRow("Polytope hull (CPU, qhull)", self.recon_polytope_label)
        recon_form.addRow("Exact polyhedral (CPU, manifold3d)", self.recon_exact_label)
        recon_form.addRow("TorchHull (GPU, sparse voxel octree)", self.recon_torchhull_label)
        recon_group_layout = QVBoxLayout(self.recon_group)
        recon_group_layout.addWidget(recon_content)
        self.recon_group.toggled.connect(recon_content.setVisible)
        self.recon_group.setChecked(False)

        # 3D-реконструкция по реальным фото (PhotoCaptureDialog._on_reconstruct_clicked) — не
        # связана с загруженным STL (нет "истины" для сравнения, объект снят камерой, а не
        # смоделирован), поэтому отдельный блок без Δ-ошибки, только то, что реально посчитано.
        self.photo_status_label = QLabel("не выполнялась")
        self.photo_observations_label = QLabel("—")
        self.photo_polytope_dims_label = QLabel("—")
        self.photo_exact_dims_label = QLabel("—")
        self.photo_group = QGroupBox("Объект из фото (3D-реконструкция по реальным фото)")
        self.photo_group.setCheckable(True)
        photo_content = QWidget()
        photo_form = QFormLayout(photo_content)
        photo_form.addRow("Итог", self.photo_status_label)
        photo_form.addRow("Наблюдения", self.photo_observations_label)
        photo_form.addRow("Габариты (polytope)", self.photo_polytope_dims_label)
        photo_form.addRow("Габариты (exact)", self.photo_exact_dims_label)
        photo_group_layout = QVBoxLayout(self.photo_group)
        photo_group_layout.addWidget(photo_content)
        self.photo_group.toggled.connect(photo_content.setVisible)
        self.photo_group.setChecked(True)

        # Свёрнуто по умолчанию (QGroupBox.setChecked(False) ниже) — иначе на панели фиксированной
        # высоты остальные, более востребованные блоки (габариты, "Проверить модель", призма)
        # уходят за пределы видимой области и их не прокрутить.
        self.reference_group = QGroupBox("Эталон формы (справочно, круглость текущего сечения)")
        self.reference_group.setCheckable(True)
        reference_content = QWidget()
        reference_form = QFormLayout(reference_content)
        reference_form.addRow("Эталон формы (конфиг)", self.reference_label)
        reference_form.addRow("Rin", self.r_in_label)
        reference_form.addRow("Rout", self.r_out_label)
        reference_form.addRow("Rin / Rout", self.k_label)
        reference_form.addRow("Сечение (текущий срез)", self.pass_label)
        reference_form.addRow(self.mismatch_label)
        group_layout = QVBoxLayout(self.reference_group)
        group_layout.addWidget(reference_content)
        self.reference_group.toggled.connect(reference_content.setVisible)
        self.reference_group.setChecked(False)

        # Гармошка, как у reference_group выше, но по умолчанию РАЗВЁРНУТА (setChecked(True)) —
        # это более востребованные блоки (см. комментарий над reference_group), просто с
        # возможностью свернуть, если они мешают.
        self.check_group = QGroupBox("Проверка модели (перекат по круглой проекции)")
        self.check_group.setCheckable(True)
        check_content = QWidget()
        check_form = QFormLayout(check_content)
        check_form.addRow("Итог", self.check_status_label)
        check_form.addRow("Габариты по силуэтам (текущая ориентация)", self.check_dims_label)
        check_form.addRow("Ошибка габаритов (текущая ориентация)", self.check_dims_error_label)
        check_form.addRow("Истинные габариты (мин. бокс, не зависят от поворота)", self.true_dims_label)
        check_form.addRow("Ошибка габаритов (мин. бокс)", self.true_dims_error_label)
        check_form.addRow("Лучшая ось / k", self.check_axis_label)
        check_form.addRow("Круглых сечений (справочно)", self.check_sections_label)
        check_form.addRow("Время анализа", self.check_time_label)
        check_form.addRow(self.show_projection_checkbox)
        check_group_layout = QVBoxLayout(self.check_group)
        check_group_layout.addWidget(check_content)
        self.check_group.toggled.connect(check_content.setVisible)
        self.check_group.setChecked(True)

        self.prism_group = QGroupBox("Вертикальная призма (Camera Mode, геометрическая проверка)")
        self.prism_group.setCheckable(True)
        prism_content = QWidget()
        prism_form = QFormLayout(prism_content)
        prism_form.addRow("Итог", self.prism_status_label)
        prism_form.addRow("Высота (Z)", self.prism_height_label)
        prism_form.addRow("Сечение (мировые X,Y)", self.prism_footprint_label)
        prism_form.addRow("Ошибка габаритов", self.prism_dims_error_label)
        prism_group_layout = QVBoxLayout(self.prism_group)
        prism_group_layout.addWidget(prism_content)
        self.prism_group.toggled.connect(prism_content.setVisible)
        self.prism_group.setChecked(True)

        layout = QFormLayout(self)
        layout.addRow(self.reference_group)
        layout.addRow("Габариты по силуэтам", self.hull_dims_label)
        layout.addRow("Габариты STL (истина)", self.mesh_dims_label)
        layout.addRow("Ошибка габаритов", self.hull_dims_error_label)

        layout.addRow(self.check_group)
        layout.addRow(self.prism_group)
        layout.addRow(self.photo_group)
        layout.addRow(self.recon_group)

        simple_section_label = QLabel(
            "<b>Просто по двум кадрам (v1: сверху + сбоку, без учёта формы)</b>"
        )
        simple_hint_label = QLabel(
            "Длина/ширина/высота скорректированы по глубине (близость к камере), не полная "
            "триангуляция — точность ниже, чем у проверки призмы"
        )
        simple_hint_label.setWordWrap(True)
        simple_hint_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addRow(simple_section_label)
        layout.addRow(simple_hint_label)
        layout.addRow("Длина × ширина × высота", self.simple_dims_label)
        layout.addRow("Ошибка габаритов", self.simple_dims_error_label)

        v3_section_label = QLabel(
            "<b>Multi-side триангуляция (v3: все ракурсы + параллакс по ленте)</b>"
        )
        v3_hint_label = QLabel(
            "Per-point триангуляция по всем азимутам рига + аналитическая высота через "
            "пару ракурсов. Точнее v1/v2 на сложных/асимметричных формах."
        )
        v3_hint_label.setWordWrap(True)
        v3_hint_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addRow(v3_section_label)
        layout.addRow(v3_hint_label)
        layout.addRow("Длина × ширина × высота", self.v3_dims_label)
        layout.addRow("Ошибка габаритов", self.v3_dims_error_label)

    def set_reference(self, is_round_reference: bool | None) -> None:
        self._is_round_reference = is_round_reference
        if is_round_reference is None:
            self.reference_label.setText("не задан (произвольный STL)")
        elif is_round_reference:
            self.reference_label.setText("круглый объект")
        else:
            self.reference_label.setText("НЕ круглый объект")
        self._update_mismatch_hint()

    def set_result(self, roundness: RoundnessResult | None) -> None:
        if roundness is None:
            self.r_in_label.setText("—")
            self.r_out_label.setText("—")
            self.k_label.setText("—")
            self.pass_label.setText("Сечение вырождено")
            self.pass_label.setStyleSheet("font-weight: bold; color: gray;")
            self._last_passed = None
            self._update_mismatch_hint()
            return

        self.r_in_label.setText(f"{roundness.r_in:.2f}")
        self.r_out_label.setText(f"{roundness.r_out:.2f}")
        self.k_label.setText(f"{roundness.k:.4f}")
        self._last_passed = roundness.passed
        if roundness.passed:
            self.pass_label.setText("круглое (k ≥ порога)")
            self.pass_label.setStyleSheet("font-weight: bold; color: #2e7d32;")
        else:
            self.pass_label.setText("не круглое")
            self.pass_label.setStyleSheet("font-weight: bold; color: #c62828;")
        self._update_mismatch_hint()

    def set_dims(
        self,
        hull_dims: tuple[float, float, float] | None,
        mesh_dims: tuple[float, float, float] | None,
        from_prism: bool = False,
    ) -> None:
        """Габариты для текущей ориентации: по реконструкции из силуэтов и истинные (по мешу).
        `from_prism=True` — `hull_dims` подменён точным результатом геометрической проверки
        "вертикальная призма" (см. core/experiment.check_vertical_prism/prism_dims) вместо
        обычной (более грубой) Camera Mode Visual Hull-реконструкции — явно помечаем, чтобы не
        путать с обычным силуэтным расчётом."""
        suffix = " · призма" if from_prism and hull_dims is not None else ""
        self.hull_dims_label.setText(_dims_text(hull_dims) + suffix)
        self.mesh_dims_label.setText(_dims_text(mesh_dims))
        # `mesh_dims` здесь — фиксированные истинные габариты STL (см. core/experiment.
        # mesh_true_dims), тройка размеров стороны минимального бокса, не привязанная к
        # мировым осям (в отличие от `hull_dims`, который зависит от текущей ориентации) —
        # сравниваем отсортированными тройками.
        self.hull_dims_error_label.setText(_dims_error_text(hull_dims, mesh_dims, sort=True))

    def set_roll_check_result(
        self,
        result: RollCheckResult | GRollResult | PcaRollResult | None,
        prism_dims_override: tuple[float, float, float] | None = None,
        stl_true_dims: tuple[float, float, float] | object | None = None,
    ) -> None:
        """`prism_dims_override` — если объект прошёл геометрическую проверку "вертикальная
        призма" (см. core/experiment.check_vertical_prism/prism_dims), показываем эти точные
        числа вместо `result.true_dims` (минимальный бокс по обычной, более грубой
        реконструкции) — с явной пометкой, аналогично `set_dims(from_prism=...)`.

        `stl_true_dims` — фиксированные истинные габариты STL (core/experiment.mesh_true_dims),
        НЕ `result.mesh_dims` (тот считается по мешу В ТЕКУЩЕЙ ориентации и меняется при
        повороте Roll/Pitch/Yaw — так и задумано для его внутреннего назначения, но вводит в
        заблуждение как "истина" в UI, см. обсуждение с пользователем).

        Поддерживает три типа результата:
        - `RollCheckResult` (SCAN, `check_model_roll`) — бинарный вердикт found_round.
        - `GRollResult` (G4, `check_model_roll_g4`) — трёхкатегорный verdict
          {round, not_round, uncertain} + fallback_triggered.
        - `PcaRollResult` (PCA/F1, `check_model_roll_pca`/`check_model_roll_f1`) —
          бинарный found_round, без fallback."""
        has_projection = result is not None and result.best_projection_coords is not None
        self.show_projection_checkbox.blockSignals(True)
        self.show_projection_checkbox.setChecked(False)
        self.show_projection_checkbox.blockSignals(False)
        self.show_projection_checkbox.setEnabled(has_projection)

        if result is None:
            self.check_status_label.setText("не выполнялась")
            self.check_status_label.setStyleSheet("")
            self.check_dims_label.setText("—")
            self.check_dims_error_label.setText("—")
            self.true_dims_label.setText("—")
            self.true_dims_error_label.setText("—")
            self.check_axis_label.setText("—")
            self.check_sections_label.setText("—")
            self.check_time_label.setText("—")
            return

        # Трёхкатегорный вердикт для G4; бинарный для SCAN/PCA/F1.
        is_g4 = isinstance(result, GRollResult)
        if is_g4:
            g4 = result  # type: GRollResult
            if g4.verdict == ROLL_VERDICT_ROUND:
                if g4.fallback_triggered:
                    self.check_status_label.setText("МОЖЕТ КАТИТЬСЯ (локальный поиск нашёл ось)")
                else:
                    self.check_status_label.setText("МОЖЕТ КАТИТЬСЯ (основной проход: 6 направлений)")
                self.check_status_label.setStyleSheet("font-weight: bold; color: #2e7d32;")
            elif g4.verdict == ROLL_VERDICT_UNCERTAIN:
                self.check_status_label.setText("ТРЕБУЕТ ПЕРЕПРОВЕРКИ (пограничный случай)")
                self.check_status_label.setStyleSheet("font-weight: bold; color: #ef6c00;")
            else:  # ROLL_VERDICT_NOT_ROUND
                self.check_status_label.setText("Не катится (явно не круглый)")
                self.check_status_label.setStyleSheet("font-weight: bold; color: #c62828;")
        elif result.found_round:
            self.check_status_label.setText("МОЖЕТ КАТИТЬСЯ (круглая проекция найдена)")
            self.check_status_label.setStyleSheet("font-weight: bold; color: #2e7d32;")
        else:
            self.check_status_label.setText("Не катится (круглой проекции нет)")
            self.check_status_label.setStyleSheet("font-weight: bold; color: #c62828;")

        self.check_dims_label.setText(_dims_text(result.dims))
        # И `result.dims` (bbox по силуэтам В ТЕКУЩЕЙ ориентации), и `stl_true_dims`
        # (фиксированная тройка сторон минимального бокса, не привязанная к мировым осям) —
        # сравниваем отсортированными тройками.
        self.check_dims_error_label.setText(_dims_error_text(result.dims, stl_true_dims, sort=True))

        true_ref = prism_dims_override if prism_dims_override is not None else result.true_dims
        if prism_dims_override is not None:
            self.true_dims_label.setText(_dims_text(prism_dims_override) + " · призма")
        else:
            self.true_dims_label.setText(_dims_text(result.true_dims))
        # `true_ref` — тройка сторон минимального бокса/призмы, НЕ привязанная к мировым осям
        # (см. докстринг true_dims/prism_dims), поэтому сравниваем с STL-истиной отсортированной.
        self.true_dims_error_label.setText(_dims_error_text(true_ref, stl_true_dims, sort=True))

        best = result.best
        if best is None:
            self.check_axis_label.setText("—")
        else:
            verdict = "PASS" if best.passed else "FAIL"
            # Источник вердикта для G4: основной проход (6 направлений) или локальный поиск
            # (~81 направление). Для SCAN — шаг перебора по полусфере. Для F1/PCA — 3/6
            # фиксированных направлений.
            if is_g4:
                source = "локальный поиск" if g4.fallback_triggered else "базовые 6 направлений"
                self.check_axis_label.setText(
                    f"k={best.k:.4f} ({verdict}), азимут {best.axis_azim_deg:.1f}°, "
                    f"наклон {best.axis_elev_deg:.1f}°; источник: {source} "
                    f"({result.checked_directions} направлений проверено)"
                )
            else:
                axis_step = getattr(result, "axis_step_deg", None)
                step_info = f" (шаг {axis_step:g}°)" if axis_step else ""
                self.check_axis_label.setText(
                    f"k={best.k:.4f} ({verdict}), азимут {best.axis_azim_deg:.1f}°, "
                    f"наклон {best.axis_elev_deg:.1f}°; проверено направлений: "
                    f"{result.checked_directions}{step_info}"
                )

        section_hits = getattr(result, "section_hits", []) or []
        round_section_count = getattr(result, "round_section_count", 0)
        self.check_sections_label.setText(f"{round_section_count} / {len(section_hits)}")

        # Время анализа — отдельно время реконструкции+поиска оси (elapsed_seconds) и,
        # для G4, отметка, запускался ли fallback (локальный поиск в зоне неуверенности).
        elapsed_ms = result.elapsed_seconds * 1000
        if is_g4 and g4.fallback_triggered:
            self.check_time_label.setText(f"{elapsed_ms:.0f} мс (с локальным поиском)")
        else:
            self.check_time_label.setText(f"{elapsed_ms:.0f} мс")

    def set_prism_result(
        self,
        result: PrismFitResult | None,
        mesh_dims: tuple[float, float, float] | None = None,
    ) -> None:
        """`mesh_dims` — истинные габариты STL (см. core/experiment.evaluate/mesh_dims), для
        сравнения с точными габаритами призмы (core/experiment.prism_dims) и оценки ошибки."""
        if result is None:
            self.prism_status_label.setText("не призма / не выполнялась")
            self.prism_status_label.setStyleSheet("")
            self.prism_height_label.setText("—")
            self.prism_footprint_label.setText("—")
            self.prism_dims_error_label.setText("—")
            return

        shape_label = "вертикальный цилиндр" if result.is_circle else f"вертикальная призма, {result.corners} угла(ов)"
        self.prism_status_label.setText(
            f"{shape_label} (верхняя камера {result.top_angle_deg:g}°)"
        )
        self.prism_status_label.setStyleSheet("font-weight: bold; color: #2e7d32;")
        self.prism_height_label.setText(f"{result.height:.1f}")
        self.prism_footprint_label.setText(
            ", ".join(f"({x:.1f}, {y:.1f})" for x, y in result.footprint_xy)
        )
        # footprint_xy лежит в мировых (X, Y) без учёта текущего yaw объекта (см. докстринг
        # prism_dims), поэтому стороны призмы не привязаны к осям mesh_dims — сравниваем
        # отсортированными тройками, как и true_dims/prism в блоке "Проверка модели".
        self.prism_dims_error_label.setText(
            _dims_error_text(prism_dims(result), mesh_dims, sort=True)
        )

    def set_simple_camera_dims(
        self,
        result: SimpleCameraDims | None,
        mesh_dims: tuple[float, float, float] | None = None,
    ) -> None:
        """`mesh_dims` — истинные габариты STL, для оценки ошибки."""
        if result is None:
            self.simple_dims_label.setText("—")
            self.simple_dims_error_label.setText("—")
            return
        dims = (result.length, result.width, result.height)
        self.simple_dims_label.setText(_dims_text(dims))
        self.simple_dims_error_label.setText(_dims_error_text(dims, mesh_dims, sort=True))

    def set_simple_camera_dims_v3(
        self,
        result: MultiSideCameraDims | None,
        mesh_dims: tuple[float, float, float] | None = None,
    ) -> None:
        """v3 — multi-side per-point триангуляция по всем ракурсам рига + параллакс по ленте."""
        if result is None:
            self.v3_dims_label.setText("—")
            self.v3_dims_error_label.setText("—")
            return
        dims = (result.length, result.width, result.height)
        self.v3_dims_label.setText(_dims_text(dims))
        self.v3_dims_error_label.setText(_dims_error_text(dims, mesh_dims, sort=True))

    def set_photo_reconstruction(
        self,
        status_text: str | None,
        n_belt: int = 0,
        n_mirror: int = 0,
        polytope_dims: tuple[float, float, float] | None = None,
        exact_dims: tuple[float, float, float] | None = None,
    ) -> None:
        """Результат последней 3D-реконструкции по фото (`PhotoCaptureDialog.
        _on_reconstruct_clicked`). `status_text=None` — сбросить блок в исходное состояние
        (диалог фото закрыт/реконструкция ещё не запускалась)."""
        if status_text is None:
            self.photo_status_label.setText("не выполнялась")
            self.photo_status_label.setStyleSheet("")
            self.photo_observations_label.setText("—")
            self.photo_polytope_dims_label.setText("—")
            self.photo_exact_dims_label.setText("—")
            return
        self.photo_status_label.setText(status_text)
        self.photo_status_label.setStyleSheet(
            "font-weight: bold; color: #2e7d32;"
            if (polytope_dims is not None or exact_dims is not None)
            else "font-weight: bold; color: #c62828;"
        )
        self.photo_observations_label.setText(f"{n_belt} прямых + {n_mirror} через отражение")
        self.photo_polytope_dims_label.setText(_dims_text(polytope_dims))
        self.photo_exact_dims_label.setText(_dims_text(exact_dims))

    def set_recon_times(
        self,
        times_ms: dict[str, float] | None,
        n_points: dict[str, int] | None = None,
    ) -> None:
        """Обновляет блок «3D-реконструкция (Camera Mode)» временем и числом точек каждого
        включённого метода. `times_ms=None` — сбросить в исходное состояние (Camera Mode
        выключен или ни один чек-бокс не включён). Ключи: 'voxel', 'polytope', 'exact',
        'torchhull' — только включённые методы присутствуют в `times_ms`."""
        if not times_ms:
            self.recon_status_label.setText("не выполнялась")
            self.recon_status_label.setStyleSheet("")
            self.recon_voxel_label.setText("—")
            self.recon_polytope_label.setText("—")
            self.recon_exact_label.setText("—")
            self.recon_torchhull_label.setText("—")
            return
        methods = [("voxel", self.recon_voxel_label),
                   ("polytope", self.recon_polytope_label),
                   ("exact", self.recon_exact_label),
                   ("torchhull", self.recon_torchhull_label)]
        n_points = n_points or {}
        for key, label in methods:
            if key in times_ms:
                n = n_points.get(key)
                n_str = f", N={n}" if n is not None else ""
                label.setText(f"{times_ms[key]:.0f} мс{n_str}")
                # Подсветка: GPU (torchhull) — синим, CPU методы — серым (визуально отличить).
                if key == "torchhull":
                    label.setStyleSheet("color: #1565c0; font-weight: bold;")
                else:
                    label.setStyleSheet("color: #555;")
            else:
                label.setText("—")
                label.setStyleSheet("")
        n_methods = len(times_ms)
        self.recon_status_label.setText(
            f"выполнено: {n_methods} метод{'ов' if n_methods != 1 else ''}"
        )
        self.recon_status_label.setStyleSheet("font-weight: bold; color: #2e7d32;")

    def _update_mismatch_hint(self) -> None:
        if self._is_round_reference is None or self._last_passed is None:
            self.mismatch_label.setText("")
            return

        if self._is_round_reference and not self._last_passed:
            self.mismatch_label.setText(
                "Эталонно круглый объект даёт FAIL в этой ориентации/наборе ракурсов — "
                "это нормально для отдельного сечения: итоговый вердикт даёт «Проверить "
                "модель» (круглая проекция), а не текущий срез."
            )
            self.mismatch_label.setStyleSheet("color: #e65100;")
        elif not self._is_round_reference and self._last_passed:
            self.mismatch_label.setText(
                "НЕ круглый (по конфигу) объект даёт круглое сечение — например, наклонное "
                "сечение конуса или артефакт Visual Hull. Проверьте вердикт по круглой "
                "проекции («Проверить модель»)."
            )
            self.mismatch_label.setStyleSheet("color: #e65100;")
        else:
            self.mismatch_label.setText("")
