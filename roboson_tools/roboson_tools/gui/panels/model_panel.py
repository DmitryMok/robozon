"""Panel 1 — 3D модель на куске ленты конвейера (pyqtgraph.opengl.GLViewWidget).

Лента фиксирована в мировых координатах (её размер считается один раз при загрузке объекта —
по описанной сфере исходного меша, инвариантной к повороту). При изменении Roll/Pitch/Yaw
обновляется только сам объект — визуально видно, как он лежит/катится/крутится на ленте.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader

from ...geometry.mesh_io import Mesh

MAX_DISPLAY_FACES = 60000  # OpenGL тянет намного больше граней, чем прежний matplotlib-рендер
DRAW_EDGES_MAX_FACES = 5000  # на тяжёлых мешах отрисовка рёбер заметно тормозит живое вращение

MESH_COLOR = (0.55, 0.65, 0.82, 0.95)
EDGE_COLOR = (0.15, 0.15, 0.18, 0.4)
BELT_COLOR = (0.30, 0.30, 0.33, 1.0)
SLICE_PLANE_COLOR = (1.0, 0.65, 0.0, 0.35)  # полупрозрачный, чтобы не перекрывать объект
HULL_RING_COLOR = (0.3, 0.95, 0.55, 0.9)  # кольца формы, восстановленной по силуэтам
ROLL_AXIS_COLOR = (1.0, 0.55, 0.1, 1.0)  # найденная ось переката (круглая проекция)
CAMERA_HULL_COLOR = (0.95, 0.35, 0.85, 0.85)  # voxel carving по перспективным силуэтам (Camera Mode)
POLYTOPE_HULL_COLOR = (0.35, 0.85, 0.95, 0.35)  # точное пересечение конусов — полупрозрачная
POLYTOPE_EDGE_COLOR = (0.35, 0.85, 0.95, 0.9)  # оболочка, для сравнения с voxel carving
EXACT_HULL_COLOR = (0.95, 0.75, 0.2, 0.35)  # Exact Polyhedral Visual Hull — сохраняет вогнутости,
EXACT_EDGE_COLOR = (0.95, 0.75, 0.2, 0.9)  # тёплый цвет, чтобы отличать от выпуклого polytope_hull
TORCHHULL_POINTS_COLOR = (0.45, 0.95, 0.55, 0.7)  # GPU visual hull (torchhull) — холодно-зелёный,
#  отличается от розово-малинового voxel (CAMERA_HULL_COLOR), чтобы видеть оба одновременно
PRISM_HULL_COLOR = (0.25, 1.0, 0.35, 0.35)  # geometric vertical-prism fit — плоские верх/низ,
PRISM_EDGE_COLOR = (0.25, 1.0, 0.35, 0.95)  # ярко-зелёный, не пересекается с остальной палитрой

# Фон остаётся чёрным (по умолчанию у GLViewWidget) — вместо этого объект освещается ярче,
# чем стандартный встроенный шейдер "shaded" (у него ambient всего 0.2, тёмная сторона почти
# не видна на чёрном фоне). Добавлены более высокий ambient и второй, слабый fill-light с
# противоположной стороны, чтобы тень тоже была различима.
_BRIGHT_SHADED = ShaderProgram(
    "roboson_bright_shaded",
    [
        VertexShader(
            """
            uniform mat4 u_mvp;
            uniform mat3 u_normal;
            attribute vec4 a_position;
            attribute vec3 a_normal;
            attribute vec4 a_color;
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                v_normal = normalize(u_normal * a_normal);
                v_color = a_color;
                gl_Position = u_mvp * a_position;
            }
            """
        ),
        FragmentShader(
            """
            #ifdef GL_ES
            precision mediump float;
            #endif
            varying vec4 v_color;
            varying vec3 v_normal;
            void main() {
                vec3 n = normalize(v_normal);
                float key = max(dot(n, normalize(vec3(1.0, -1.0, -1.0))), 0.0);
                float fill = max(dot(n, normalize(vec3(-0.6, 0.6, 0.4))), 0.0);
                float light = 0.55 + key * 0.7 + fill * 0.25;
                vec3 rgb = v_color.rgb * light;
                gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), v_color.a);
            }
            """
        ),
    ],
)


_BOX_FACES = np.array(
    [
        [0, 1, 2], [0, 2, 3],  # низ
        [4, 6, 5], [4, 7, 6],  # верх
        [0, 4, 5], [0, 5, 1],  # стороны
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ]
)


def _box_mesh(center: np.ndarray, half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ]
    )
    return verts, _BOX_FACES


def _oriented_plane_mesh(
    center: np.ndarray, normal: np.ndarray, half_size: float, thickness: float
) -> tuple[np.ndarray, np.ndarray]:
    """Тонкий бокс (плоскость среза), развёрнутый так, что толщина — вдоль `normal`, а не
    обязательно вдоль мировой оси X (частный случай normal=X эквивалентен `_box_mesh` с
    half_extents=(thickness, half_size, half_size)) — используется, когда плоскость должна
    следовать за найденной осью переката, а не только за обычным продольным срезом."""
    e1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    signs = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    local_half = np.array([thickness, half_size, half_size])
    basis = np.stack([normal, e1, e2])  # строки — базисные векторы (normal, e1, e2)
    verts = center + (signs * local_half) @ basis
    return verts, _BOX_FACES


class ModelPanel(gl.GLViewWidget):
    def __init__(self) -> None:
        super().__init__()
        self._object_item: gl.GLMeshItem | None = None
        self._belt_item: gl.GLMeshItem | None = None
        self._slice_plane_item: gl.GLMeshItem | None = None
        self._hull_item: gl.GLLinePlotItem | None = None
        self._roll_axis_item: gl.GLLinePlotItem | None = None
        self._camera_hull_item: gl.GLScatterPlotItem | None = None
        self._polytope_hull_item: gl.GLMeshItem | None = None
        self._exact_hull_item: gl.GLMeshItem | None = None
        self._torchhull_item: gl.GLScatterPlotItem | None = None
        self._prism_hull_item: gl.GLMeshItem | None = None
        self._belt_radius = 100.0
        self._display_faces: np.ndarray | None = None
        self._last_oriented_vertices: np.ndarray | None = None

    def load_object(self, mesh: Mesh) -> None:
        """Новый объект: пересчитать неподвижную ленту конвейера под его габарит и камеру,
        пересоздать item объекта (топология faces фиксируется здесь на всё время, пока не
        сменится объект)."""
        self.set_slice_plane(None)  # старая плоскость привязана к габариту прежнего объекта
        self.set_hull_slices(None)
        self.set_roll_axis(None)

        radius = float(np.linalg.norm(mesh.vertices, axis=1).max()) or 1.0
        self._belt_radius = radius
        self._rebuild_belt(radius)
        self._fit_camera(radius)

        faces = mesh.faces
        if len(faces) > MAX_DISPLAY_FACES:
            step = len(faces) // MAX_DISPLAY_FACES
            faces = faces[::step]
        self._display_faces = faces
        self._last_oriented_vertices = mesh.vertices

        if self._object_item is not None:
            self.removeItem(self._object_item)

        self._object_item = gl.GLMeshItem(
            vertexes=mesh.vertices,
            faces=faces,
            smooth=False,
            drawEdges=len(faces) <= DRAW_EDGES_MAX_FACES,
            edgeColor=EDGE_COLOR,
            color=MESH_COLOR,
            shader=_BRIGHT_SHADED,
        )
        self.addItem(self._object_item)

    def set_object_visible(self, visible: bool) -> None:
        """Скрыть/показать сам объект (сплошной непрозрачный меш) — не трогая ленту/халлы.
        Нужно при сравнении Camera Mode реконструкций (`set_camera_hull_exact`/`_polytope`):
        exact-халл почти точно совпадает с поверхностью объекта (в отличие от выпуклого
        polytope_hull, который заметно вылезает наружу на рёбрах/уступах) — тест буфера
        глубины прячет полупрозрачный халл ЗА непрозрачным объектом, снаружи ничего не видно
        ("рисует будто внутри объекта"). Скрыть объект — единственный надёжный способ увидеть
        такой почти-точный халл целиком."""
        if self._object_item is not None:
            self._object_item.setVisible(visible)

    def set_orientation_mesh(self, oriented_mesh: Mesh) -> None:
        """Обновить положение вершин объекта после поворота — лента, камера и топология
        faces не трогаются (дешевле, чем пересоздавать item на каждый пересчёт)."""
        self._last_oriented_vertices = oriented_mesh.vertices
        if self._object_item is None or self._display_faces is None:
            self.load_object(oriented_mesh)
            return
        self._object_item.setMeshData(vertexes=oriented_mesh.vertices, faces=self._display_faces)

    def set_belt_offset(self, dx: float | None) -> None:
        """Camera Mode: визуально сдвинуть объект вдоль X (лента), чтобы показать положение
        начало/центр/конец в кадре камеры (см. заметку задачи "Добавить Camera Mode...").
        Сдвиг ЧИСТО визуальный — расчёты (габариты, круглая проекция, evaluate,
        check_model_roll) по-прежнему используют несдвинутый объект. `dx=None`/0 — без сдвига."""
        if (
            self._object_item is None
            or self._display_faces is None
            or self._last_oriented_vertices is None
        ):
            return
        offset = dx or 0.0
        vertexes = self._last_oriented_vertices + np.array([offset, 0.0, 0.0])
        self._object_item.setMeshData(vertexes=vertexes, faces=self._display_faces)

    def set_slice_plane(
        self, center: np.ndarray | None, normal: np.ndarray | None = None
    ) -> None:
        """Полупрозрачная плоскость среза, проходящая через `center`. `normal` по умолчанию —
        продольная ось X (обычный срез Slice X из Panel 3); при одновременном показе плоскости
        среза и круглой проекции сюда передаётся найденная ось переката (см. MainWindow.
        _recompute) — плоскость разворачивается вдоль неё и поворачивается вместе с объектом.
        `center=None` — убрать плоскость."""
        if self._slice_plane_item is not None:
            self.removeItem(self._slice_plane_item)
            self._slice_plane_item = None

        if center is None:
            return

        if normal is None:
            normal = np.array([1.0, 0.0, 0.0])
        normal = np.asarray(normal, dtype=np.float64)
        normal = normal / (np.linalg.norm(normal) or 1.0)

        half_size = self._belt_radius * 1.1
        thickness = self._belt_radius * 0.01
        verts, faces = _oriented_plane_mesh(
            np.asarray(center, dtype=np.float64), normal, half_size, thickness
        )

        self._slice_plane_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            smooth=False,
            color=SLICE_PLANE_COLOR,
            shader="shaded",
            glOptions="translucent",
        )
        self.addItem(self._slice_plane_item)

    def set_hull_slices(
        self, slices: list[tuple[float, list[tuple[float, float]]]] | None
    ) -> None:
        """Форма, восстановленная по силуэтам: кольца сечений [(x, [(y, z), ...]), ...] поверх
        объекта. `slices=None` — убрать. Кольца рисуются одним GLLinePlotItem в режиме 'lines'
        (парами точек-сегментов), чтобы не плодить item на каждое сечение."""
        if self._hull_item is not None:
            self.removeItem(self._hull_item)
            self._hull_item = None

        if not slices:
            return

        segments = []
        for x_pos, coords in slices:
            ring = np.asarray(coords, dtype=np.float64)  # замкнутый контур (Y, Z)
            pts = np.column_stack([np.full(len(ring), x_pos), ring[:, 0], ring[:, 1]])
            # пары (p_i, p_i+1) для режима 'lines'
            segments.append(np.repeat(pts, 2, axis=0)[1:-1])
        self._hull_item = gl.GLLinePlotItem(
            pos=np.concatenate(segments),
            color=HULL_RING_COLOR,
            width=1.5,
            mode="lines",
            antialias=True,
            glOptions="translucent",
        )
        self.addItem(self._hull_item)

    def set_camera_hull_points(self, points: np.ndarray | None) -> None:
        """Voxel carving по перспективным силуэтам Camera Mode (см. заметку задачи и
        `visual_hull/voxel_carving.py`) — облако уцелевших вокселей поверх объекта.
        Отдельная, исследовательская визуализация: НЕ участвует в габаритах/круглой проекции
        (те по-прежнему считаются Analytical Mode). `points=None` — убрать."""
        if self._camera_hull_item is not None:
            self.removeItem(self._camera_hull_item)
            self._camera_hull_item = None

        if points is None or len(points) == 0:
            return

        self._camera_hull_item = gl.GLScatterPlotItem(
            pos=np.asarray(points, dtype=np.float64),
            color=CAMERA_HULL_COLOR,
            size=4.0,
            pxMode=True,
            glOptions="translucent",
        )
        self.addItem(self._camera_hull_item)

    def set_camera_hull_torchhull(self, points: np.ndarray | None) -> None:
        """GPU visual hull через torchhull (`visual_hull/torchhull_adapter.py`) — облако
        вершин восстановленного меша (torchhull.visual_hull возвращает verts+faces, но для
        сравнения в Panel 1 удобнее точки, как у voxel_carving). Отдельный цвет от voxel
        (TORCHHULL_POINTS_COLOR) — чтобы видеть оба одновременно. Отдельная, исследовательская
        визуализация: НЕ участвует в габаритах/круглой проекции. `points=None` — убрать."""
        if self._torchhull_item is not None:
            self.removeItem(self._torchhull_item)
            self._torchhull_item = None

        if points is None or len(points) == 0:
            return

        self._torchhull_item = gl.GLScatterPlotItem(
            pos=np.asarray(points, dtype=np.float64),
            color=TORCHHULL_POINTS_COLOR,
            size=3.0,
            pxMode=True,
            glOptions="translucent",
        )
        self.addItem(self._torchhull_item)

    def set_camera_hull_polytope(
        self, vertices: np.ndarray | None, faces: np.ndarray | None
    ) -> None:
        """Точное пересечение конусов обзора (`visual_hull/polytope_hull.py`) — полупрозрачный
        выпуклый многогранник, для сравнения с voxel carving (`set_camera_hull_points`) в той
        же сцене. Отдельная, исследовательская визуализация: НЕ участвует в габаритах/круглой
        проекции. `vertices=None` — убрать.

        `glOptions="additive"`, а не "translucent": на выпуклых объектах этот халл почти
        совпадает с exact-халлом (`set_camera_hull_exact`) — при обычном "translucent" (тест
        глубины включён, pyqtgraph требует ручной сортировки back-to-front, которой здесь нет)
        второй отрисованный полупрозрачный слой полностью проваливает тест глубины и становится
        невидимым ("на коробке жёлтого нет" — реальный кейс). additive тест глубины не использует
        вовсе, поэтому оба халла видны независимо от порядка отрисовки."""
        if self._polytope_hull_item is not None:
            self.removeItem(self._polytope_hull_item)
            self._polytope_hull_item = None

        if vertices is None or faces is None or len(vertices) == 0:
            return

        self._polytope_hull_item = gl.GLMeshItem(
            vertexes=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            smooth=False,
            drawEdges=True,
            edgeColor=POLYTOPE_EDGE_COLOR,
            color=POLYTOPE_HULL_COLOR,
            shader="shaded",
            glOptions="additive",
        )
        self.addItem(self._polytope_hull_item)

    def set_polytope_hull_visible(self, visible: bool) -> None:
        """Показать/скрыть уже посчитанный polytope-халл, не пересчитывая его (см.
        `set_camera_hull_polytope`) — нужно, чтобы отдельно смотреть выпуклый/вогнутый халл
        (иначе оба additive-слоя сливаются в один плохо читаемый цвет, особенно на вогнутых
        объектах, см. заметку задачи про фото-реконструкцию)."""
        if self._polytope_hull_item is not None:
            self._polytope_hull_item.setVisible(visible)

    def set_camera_hull_exact(
        self, vertices: np.ndarray | None, faces: np.ndarray | None
    ) -> None:
        """Exact Polyhedral Visual Hull (`visual_hull/exact_polyhedral_hull.py`) —
        полупрозрачный, возможно НЕВЫПУКЛЫЙ многогранник (в отличие от
        `set_camera_hull_polytope`, сохраняет вогнутости силуэта — например, уступ на ручке
        между зажимаемой частью и тонким стержнем). Отдельная, исследовательская визуализация:
        НЕ участвует в габаритах/круглой проекции. `vertices=None` — убрать.

        `glOptions="additive"` — см. докстринг `set_camera_hull_polytope`: без него этот слой
        (второй нарисованный полупрозрачный меш) проваливает тест глубины против уже
        нарисованного polytope-халла и становится невидимым на выпуклых объектах/ориентациях,
        где оба халла почти совпадают с поверхностью объекта."""
        if self._exact_hull_item is not None:
            self.removeItem(self._exact_hull_item)
            self._exact_hull_item = None

        if vertices is None or faces is None or len(vertices) == 0:
            return

        self._exact_hull_item = gl.GLMeshItem(
            vertexes=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            smooth=False,
            drawEdges=True,
            edgeColor=EXACT_EDGE_COLOR,
            color=EXACT_HULL_COLOR,
            shader="shaded",
            glOptions="additive",
        )
        self.addItem(self._exact_hull_item)

    def set_exact_hull_visible(self, visible: bool) -> None:
        """Показать/скрыть уже посчитанный exact-халл, не пересчитывая его — см.
        `set_polytope_hull_visible`."""
        if self._exact_hull_item is not None:
            self._exact_hull_item.setVisible(visible)

    def set_prism_hull(self, vertices: np.ndarray | None, faces: np.ndarray | None) -> None:
        """Геометрически подобранная вертикальная призма (`visual_hull/vertical_prism_fit.py`,
        `core/experiment.check_vertical_prism`) — плоские верх/низ вместо конусного скоса
        обычных Camera Mode халлов (`set_camera_hull_polytope`/`_exact`), поскольку это не
        Visual Hull пересечением конусов, а отдельный точный расчёт по верхней+боковым камерам.
        `vertices=None` — убрать. `glOptions="additive"` по той же причине, что у остальных
        полупрозрачных Camera Mode халлов (см. `set_camera_hull_polytope`) — без него слой может
        полностью провалить тест глубины и стать невидимым при наложении на другие халлы."""
        if self._prism_hull_item is not None:
            self.removeItem(self._prism_hull_item)
            self._prism_hull_item = None

        if vertices is None or faces is None or len(vertices) == 0:
            return

        self._prism_hull_item = gl.GLMeshItem(
            vertexes=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            smooth=False,
            drawEdges=True,
            edgeColor=PRISM_EDGE_COLOR,
            color=PRISM_HULL_COLOR,
            shader="shaded",
            glOptions="additive",
        )
        self.addItem(self._prism_hull_item)

    def set_roll_axis(
        self,
        direction: np.ndarray | tuple[float, float, float] | None,
        center: np.ndarray | tuple[float, float, float] | None = None,
    ) -> None:
        """Найденная ось переката (направление круглой проекции) через `center`.
        `direction` — в ТЕКУЩЕЙ (уже повёрнутой) системе координат объекта: вызывающая сторона
        (MainWindow._recompute) обязана каждый раз пересчитывать её текущей ориентацией, иначе
        линия не будет следовать за поворотом объекта. `direction=None` — убрать.

        `center=None` — через мировое начало координат (старое поведение, верно для STL-мешей
        этого приложения — они центрированы у начала). Явный `center` нужен для реконструкции
        по сетке ракурсов (`GridReconstructionDialog`) — там начало координат сцены это точка
        наблюдения рига камер, а НЕ центр восстановленного объекта, поэтому ось без сдвига
        рисовалась бы в стороне от самого халла."""
        if self._roll_axis_item is not None:
            self.removeItem(self._roll_axis_item)
            self._roll_axis_item = None

        if direction is None:
            return

        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) or 1.0)
        c = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
        length = self._belt_radius * 1.4
        pts = np.array([c - d * length, c + d * length])
        self._roll_axis_item = gl.GLLinePlotItem(
            pos=pts, color=ROLL_AXIS_COLOR, width=3.0, mode="lines", antialias=True
        )
        self.addItem(self._roll_axis_item)

    def _rebuild_belt(self, radius: float) -> None:
        if self._belt_item is not None:
            self.removeItem(self._belt_item)

        length_x = radius * 4.0
        width_y = radius * 2.4
        thickness = radius * 0.12
        top_z = -radius * 1.05

        center = np.array([0.0, 0.0, top_z - thickness / 2])
        half_extents = np.array([length_x / 2, width_y / 2, thickness / 2])
        verts, faces = _box_mesh(center, half_extents)

        self._belt_item = gl.GLMeshItem(
            vertexes=verts, faces=faces, smooth=False, color=BELT_COLOR, shader="shaded"
        )
        self.addItem(self._belt_item)

    def _fit_camera(self, radius: float) -> None:
        self.setCameraPosition(distance=radius * 6.0, elevation=22, azimuth=-55)
