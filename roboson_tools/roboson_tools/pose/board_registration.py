"""Автоматическая регистрация НЕСКОЛЬКИХ независимых ArUco-досок (зеркало, борта ленты, ...) в
одну общую систему координат по "мостиковым" фото — БЕЗ ручного обмера их взаимного положения.

Идея (по предложению пользователя, 2026-07-18): камера рига не обязана видеть все доски сразу.
Часть фото видит только доску зеркала, часть — только доску(и) ленты, часть — сразу несколько
("мостиковые" фото). Каждая доска обмеряется независимо в СВОЕЙ локальной плоской системе
координат (см. `assets/mirror_board/`, `assets/belt_board/` — их `*.yaml` дают именно такие
локальные координаты, ДО регистрации). На мостиковом фото можно решить `solvePnP` отдельно по
меткам каждой видимой доски (в её локальных координатах) и получить позу камеры относительно
каждой из них ОДНОВРЕМЕННО — а значит и фиксированный (не зависящий от фото) жёсткий переход
между локальными системами этих досок. Несколько мостиковых фото усредняются для устойчивости.

Доски, не связанные с опорной ни напрямую, ни через цепочку мостиковых фото — остаются
незарегистрированными (см. `BoardRegistration.unregistered_boards`); для них нужно больше
мостиковых фото.

После регистрации все доски можно слить в один `MarkerLayout` в системе опорной доски (обычно —
доска ленты, `X` вдоль ленты, см. `docs/method.md`) — далее обычный путь: `estimate_camera_pose`
по слитой раскладке работает так же, как раньше с единой вручную обмеренной раскладкой.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from .aruco_estimator import detect_markers
from .camera_pose import CameraIntrinsics
from .marker_layout import MarkerLayout


@dataclass(frozen=True)
class RigidTransform:
    """`X_out = rotation @ X_local + translation` — жёсткое преобразование координат."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def apply(self, points_local: np.ndarray) -> np.ndarray:
        return (self.rotation @ points_local.T).T + self.translation


IDENTITY_TRANSFORM = RigidTransform(rotation=np.eye(3), translation=np.zeros(3))


@dataclass
class BoardRegistration:
    transforms: dict[str, RigidTransform]  # board_name -> (board_local -> reference_board_local)
    reference_board: str
    bridge_photo_counts: dict[tuple[str, str], int]  # {(a,b) a<b: сколько фото дали образец пары}
    unregistered_boards: list[str]
    message: str


def _solve_board_pose_in_camera(
    detected_on_board: dict[int, np.ndarray],
    board_layout: MarkerLayout,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Поза камеры относительно ОДНОЙ доски (в её локальных координатах) ->
    (rotation, translation), `x_cam = rotation @ x_board_local + translation` (тот же формат, что
    `CameraPose`). None — solvePnP не сошёлся.

    В отличие от `aruco_estimator.estimate_camera_pose` (там неоднозначность плоской мишени
    снимается объединением НЕСКОЛЬКИХ, как правило некомпланарных, досок в одном solvePnP), здесь
    мы намеренно решаем позу ОДНОЙ доски — а её метки лежат в ОДНОЙ плоскости (реальный печатный
    лист), поэтому классическая planar pose ambiguity (два похожих по ошибке репроекции решения)
    актуальна и для доски из 2+ меток, не только одной. `cv2.SOLVEPNP_IPPE` возвращает ОБА
    кандидата — берём с меньшей ошибкой репроекции, а не то, что попало первым (обычный
    `cv2.solvePnP(..., SOLVEPNP_ITERATIVE)` может итеративно сойтись к худшему из двух локальных
    оптимумов в зависимости от геометрии кадра — воспроизведено на синтетике при регистрации
    досок, см. `tests/test_board_registration.py`)."""
    marker_ids = sorted(detected_on_board.keys())
    object_points = np.concatenate([board_layout.corners_world_mm[i] for i in marker_ids], axis=0)
    image_points = np.concatenate([detected_on_board[i] for i in marker_ids], axis=0)
    dist_coeffs = intrinsics.dist_coeffs
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    try:
        retval, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points, image_points, intrinsics.matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        retval = 0

    if not retval:
        # Фоллбек — вырожденная для IPPE геометрия (не должно случаться для честно плоских
        # меток, но не полагаемся на это молча).
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, intrinsics.matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not success:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        return rotation, tvec.reshape(3)

    best_idx = int(np.argmin(np.asarray(errors).reshape(-1)))
    rotation, _ = cv2.Rodrigues(rvecs[best_idx])
    return rotation, tvecs[best_idx].reshape(3)


def _pairwise_sample(
    board_a: str,
    pose_a: tuple[np.ndarray, np.ndarray],
    board_b: str,
    pose_b: tuple[np.ndarray, np.ndarray],
) -> tuple[str, str, RigidTransform]:
    """Из двух поз камеры (относительно доски A и относительно доски B на ОДНОМ фото) — один
    образец жёсткого перехода между локальными системами досок, в каноническом направлении
    (по алфавиту имён досок): `A -> B` если `board_a < board_b`, иначе `B -> A`.

    Вывод: `x_cam = R_a@x_a + t_a = R_b@x_b + t_b`, и `x_b = R_tr@x_a + t_tr` (переход A->B) =>
    `R_a = R_b@R_tr`, `t_a = R_b@t_tr + t_b` => `R_tr = R_b^T@R_a`, `t_tr = R_b^T@(t_a - t_b)`."""
    r_a, t_a = pose_a
    r_b, t_b = pose_b
    if board_a < board_b:
        source, target = board_a, board_b
        r_tr = r_b.T @ r_a
        t_tr = r_b.T @ (t_a - t_b)
    else:
        source, target = board_b, board_a
        r_tr = r_a.T @ r_b
        t_tr = r_a.T @ (t_b - t_a)
    return source, target, RigidTransform(rotation=r_tr, translation=t_tr)


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Среднее нескольких поворотов — не покомпонентное усреднение матриц (не ортонормировано),
    а проекция среднего на ближайшую ортогональную матрицу через SVD (стандартный приём;
    равносилен минимизации суммы `||R_avg - R_i||_F^2` по всем поворотам)."""
    mean = np.mean(rotations, axis=0)
    u, _s, vt = np.linalg.svd(mean)
    r = u @ vt
    if np.linalg.det(r) < 0:  # избежать отражения (det=-1) — не поворот
        u = u.copy()
        u[:, -1] *= -1
        r = u @ vt
    return r


def _invert(t: RigidTransform) -> RigidTransform:
    r_inv = t.rotation.T
    return RigidTransform(rotation=r_inv, translation=-r_inv @ t.translation)


def _compose(outer: RigidTransform, inner: RigidTransform) -> RigidTransform:
    """`outer ∘ inner`: сперва `inner`, затем `outer` (как обычная композиция функций)."""
    return RigidTransform(
        rotation=outer.rotation @ inner.rotation,
        translation=outer.rotation @ inner.translation + outer.translation,
    )


def register_boards(
    images_bgr: list[np.ndarray],
    boards: dict[str, MarkerLayout],
    intrinsics: CameraIntrinsics,
    reference_board: str,
    min_markers_per_board: int = 2,
) -> BoardRegistration:
    """Регистрирует все `boards` (`board_name -> MarkerLayout` в ЛОКАЛЬНЫХ координатах доски,
    см. докстринг модуля) в систему координат `reference_board` по фото из `images_bgr`.
    `reference_board` должен быть ключом `boards`. Доска считается видимой на фото, если на нём
    детектировано >= `min_markers_per_board` её собственных меток (нужно >= 2 для устойчивого
    непланарного solvePnP — одна метка даёт классическую planar pose ambiguity, см.
    `aruco_estimator.estimate_camera_pose`)."""
    if reference_board not in boards:
        return BoardRegistration(
            transforms={}, reference_board=reference_board, bridge_photo_counts={},
            unregistered_boards=list(boards),
            message=f"reference_board={reference_board!r} отсутствует в boards",
        )

    id_to_board: dict[int, str] = {}
    for board_name, layout in boards.items():
        for marker_id in layout.corners_world_mm:
            if marker_id in id_to_board:
                raise ValueError(
                    f"метка id={marker_id} есть и в доске {id_to_board[marker_id]!r}, и в "
                    f"{board_name!r} — ID меток должны быть уникальны по всей сцене"
                )
            id_to_board[marker_id] = board_name
    merged_layout = MarkerLayout(
        dictionary_name=next(iter(boards.values())).dictionary_name,
        corners_world_mm={
            i: pts for layout in boards.values() for i, pts in layout.corners_world_mm.items()
        },
    )

    samples: dict[tuple[str, str], list[RigidTransform]] = {}
    bridge_counts: dict[tuple[str, str], int] = {}
    for image in images_bgr:
        detected = detect_markers(image, merged_layout)
        if not detected:
            continue
        by_board: dict[str, dict[int, np.ndarray]] = {}
        for marker_id, corners_px in detected.items():
            by_board.setdefault(id_to_board[marker_id], {})[marker_id] = corners_px

        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for board_name, detected_on_board in by_board.items():
            if len(detected_on_board) < min_markers_per_board:
                continue
            pose = _solve_board_pose_in_camera(detected_on_board, boards[board_name], intrinsics)
            if pose is not None:
                poses[board_name] = pose

        visible = sorted(poses.keys())
        for i in range(len(visible)):
            for j in range(i + 1, len(visible)):
                board_a, board_b = visible[i], visible[j]
                source, target, transform = _pairwise_sample(
                    board_a, poses[board_a], board_b, poses[board_b]
                )
                key = (source, target)
                samples.setdefault(key, []).append(transform)
                bridge_counts[key] = bridge_counts.get(key, 0) + 1

    edges: dict[tuple[str, str], RigidTransform] = {}
    for (source, target), transform_samples in samples.items():
        rotation = _average_rotation([t.rotation for t in transform_samples])
        translation = np.mean([t.translation for t in transform_samples], axis=0)
        edges[(source, target)] = RigidTransform(rotation=rotation, translation=translation)

    adjacency: dict[str, list[str]] = {name: [] for name in boards}
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)

    resolved: dict[str, RigidTransform] = {reference_board: IDENTITY_TRANSFORM}
    queue: deque[str] = deque([reference_board])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in resolved:
                continue
            source, target = (node, neighbor) if node < neighbor else (neighbor, node)
            edge_transform = edges[(source, target)]  # source -> target
            if node == source:
                # известно source->ref (=resolved[node]); target->ref = (source->ref) ∘ (target->source)
                resolved[neighbor] = _compose(resolved[node], _invert(edge_transform))
            else:
                # node == target, известно target->ref; source->ref = (target->ref) ∘ (source->target)
                resolved[neighbor] = _compose(resolved[node], edge_transform)
            queue.append(neighbor)

    unregistered = sorted(set(boards) - set(resolved))
    message = (
        f"зарегистрировано {len(resolved)}/{len(boards)} досок относительно "
        f"{reference_board!r}, {len(edges)} мостиковых пар (по {sum(bridge_counts.values())} "
        "фото-образцам суммарно)"
    )
    if unregistered:
        message += f"; не связаны с опорной доской (нужно больше мостиковых фото): {unregistered}"

    return BoardRegistration(
        transforms=resolved,
        reference_board=reference_board,
        bridge_photo_counts=bridge_counts,
        unregistered_boards=unregistered,
        message=message,
    )


def merge_layout(boards: dict[str, MarkerLayout], registration: BoardRegistration) -> MarkerLayout:
    """Переносит координаты меток всех ЗАРЕГИСТРИРОВАННЫХ (см. `registration.unregistered_boards`)
    досок в систему `registration.reference_board`, возвращая один плоский `MarkerLayout` — с этого
    момента дальше можно пользоваться `estimate_camera_pose` как с обычной, вручную обмеренной,
    единой раскладкой (см. докстринг модуля)."""
    dictionary_name = next(iter(boards.values())).dictionary_name
    merged: dict[int, np.ndarray] = {}
    for board_name, layout in boards.items():
        transform = registration.transforms.get(board_name)
        if transform is None:
            continue
        for marker_id, corners_local in layout.corners_world_mm.items():
            merged[marker_id] = transform.apply(corners_local)
    return MarkerLayout(dictionary_name=dictionary_name, corners_world_mm=merged)


@dataclass
class BundleAdjustmentReport:
    """Диагностика `bundle_adjust_registration` — сколько фото реально вошло в оптимизацию и как
    изменилась суммарная (по всем меткам всех досок на всех этих фото) RMS ошибка репроекции,
    px. `rms_reprojection_px_before` — с НАЧАЛЬНЫМ приближением (`initial.transforms` +
    отдельный solvePnP на фото под затравку позы камеры), `_after` — после совместной
    оптимизации. `boards_refined` — какие доски реально уточнялись (не reference, не
    unregistered)."""

    n_frames_used: int
    rms_reprojection_px_before: float | None
    rms_reprojection_px_after: float | None
    boards_refined: list[str]


def _rigid_to_vec(t: RigidTransform) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(t.rotation)
    return np.concatenate([rvec.reshape(3), t.translation])


def _vec_to_rigid(v: np.ndarray) -> RigidTransform:
    rotation, _ = cv2.Rodrigues(v[:3].reshape(3, 1))
    return RigidTransform(rotation=rotation, translation=v[3:6].copy())


def bundle_adjust_registration(
    images_bgr: list[np.ndarray],
    boards: dict[str, MarkerLayout],
    intrinsics: CameraIntrinsics,
    initial: BoardRegistration,
    min_markers_per_board: int = 2,
) -> tuple[BoardRegistration, BundleAdjustmentReport]:
    """Уточняет `initial.transforms` совместной нелинейной МНК-минимизацией (bundle adjustment,
    `scipy.optimize.least_squares`) СУММАРНОЙ репроекционной ошибки по ВСЕМ фото и ВСЕМ уже
    зарегистрированным доскам сразу.

    В отличие от `register_boards` (независимый `solvePnP` НА КАЖДОМ фото под каждую пару
    видимых досок + попарное усреднение перехода доска-доска, затем BFS-композиция вдоль дерева
    мостиковых пар — ошибка вдоль длинной цепочки досок накапливается, а сами усреднения не
    учитывают фото, где board видна вместе с ДРУГИМИ, не смежными в дереве, досками), здесь ОДНА
    совместная оптимизация одновременно по:
      - позе камеры НА КАЖДОМ фото, где видна хотя бы одна уже зарегистрированная доска (только
        внутренние переменные этой функции — не переиспользуются потом при позиционировании
        фото реконструкции через `estimate_camera_pose`, нужны лишь чтобы объяснить наблюдения
        конкретного фото любой комбинацией досок сразу, в т.ч. когда видно 3+ доски);
      - `board_local -> reference_board` переходу КАЖДОЙ не-опорной доски из `initial.transforms`.

    Требует `initial` (обычно результат `register_boards()`) — даёт начальное приближение и
    связность (какие доски вообще разрешимы); доски из `initial.unregistered_boards` не
    участвуют (для них нет стартовой точки). Если после начального приближения не набралось ни
    одного пригодного фото (нет ни одной доски с >= `min_markers_per_board` метками хотя бы на
    одном кадре) — возвращает `initial` без изменений и пустой отчёт (`n_frames_used=0`).

    Возвращает (уточнённый `BoardRegistration`, диагностика `BundleAdjustmentReport` — в
    частности RMS репроекции до/после, полезно показать пользователю, что оптимизация реально
    что-то улучшила)."""
    reference_board = initial.reference_board
    optimized_boards = sorted(b for b in initial.transforms if b != reference_board and b in boards)
    empty_report = BundleAdjustmentReport(
        n_frames_used=0, rms_reprojection_px_before=None, rms_reprojection_px_after=None,
        boards_refined=[],
    )
    if not optimized_boards or reference_board not in initial.transforms:
        return initial, empty_report

    registered_boards = [reference_board, *optimized_boards]
    id_to_board: dict[int, str] = {}
    for board_name in registered_boards:
        for marker_id in boards[board_name].corners_world_mm:
            id_to_board[marker_id] = board_name
    # Раскладка только для фильтрации детекций по id (см. detect_markers) — мировые координаты
    # внутри неё не используются нигде дальше, реальные (локальные) координаты берутся из
    # boards[board_name] в _residuals ниже.
    filter_layout = MarkerLayout(
        dictionary_name=boards[reference_board].dictionary_name,
        corners_world_mm={
            marker_id: boards[board_name].corners_world_mm[marker_id]
            for marker_id, board_name in id_to_board.items()
        },
    )

    dist_coeffs = intrinsics.dist_coeffs if intrinsics.dist_coeffs is not None else np.zeros(5)

    frames: list[dict[str, dict[int, np.ndarray]]] = []
    frame_seed_poses: list[tuple[np.ndarray, np.ndarray]] = []
    for image in images_bgr:
        detected = detect_markers(image, filter_layout)
        if not detected:
            continue
        by_board: dict[str, dict[int, np.ndarray]] = {}
        for marker_id, corners_px in detected.items():
            by_board.setdefault(id_to_board[marker_id], {})[marker_id] = corners_px
        by_board = {b: pts for b, pts in by_board.items() if len(pts) >= min_markers_per_board}
        if not by_board:
            continue

        # Затравка позы камеры этого фото (в системе reference_board) — через ЛЮБУЮ видимую
        # доску с уже известным (начальным) переходом в reference; порядок перебора
        # детерминирован (алфавитный), первый успешный solvePnP используется.
        seed = None
        for board_name in sorted(by_board):
            pose = _solve_board_pose_in_camera(by_board[board_name], boards[board_name], intrinsics)
            if pose is None:
                continue
            r_board_img, t_board_img = pose
            transform = initial.transforms[board_name]
            r_cam_ref = r_board_img @ transform.rotation.T
            t_cam_ref = t_board_img - r_cam_ref @ transform.translation
            seed = (r_cam_ref, t_cam_ref)
            break
        if seed is None:
            continue

        frames.append(by_board)
        frame_seed_poses.append(seed)

    if not frames:
        return initial, empty_report

    n_boards = len(optimized_boards)
    n_frames = len(frames)
    x0 = np.concatenate(
        [_rigid_to_vec(initial.transforms[b]) for b in optimized_boards]
        + [np.concatenate([cv2.Rodrigues(r)[0].reshape(3), t]) for r, t in frame_seed_poses]
    )
    cam_offset = n_boards * 6

    def _unpack(x: np.ndarray) -> tuple[dict[str, RigidTransform], list[tuple[np.ndarray, np.ndarray]]]:
        board_transforms = {reference_board: IDENTITY_TRANSFORM}
        for i, board_name in enumerate(optimized_boards):
            board_transforms[board_name] = _vec_to_rigid(x[i * 6:(i + 1) * 6])
        cam_poses = []
        for j in range(n_frames):
            v = x[cam_offset + j * 6: cam_offset + (j + 1) * 6]
            rotation, _ = cv2.Rodrigues(v[:3].reshape(3, 1))
            cam_poses.append((rotation, v[3:6]))
        return board_transforms, cam_poses

    def _residuals(x: np.ndarray) -> np.ndarray:
        board_transforms, cam_poses = _unpack(x)
        out = []
        for by_board, (r_cam, t_cam) in zip(frames, cam_poses):
            rvec_cam, _ = cv2.Rodrigues(r_cam)
            for board_name, detected_on_board in by_board.items():
                marker_ids = sorted(detected_on_board.keys())
                local_corners = np.concatenate(
                    [boards[board_name].corners_world_mm[i] for i in marker_ids], axis=0
                )
                world_pts = board_transforms[board_name].apply(local_corners)
                image_pts = np.concatenate([detected_on_board[i] for i in marker_ids], axis=0)
                projected, _ = cv2.projectPoints(
                    world_pts, rvec_cam, t_cam, intrinsics.matrix, dist_coeffs
                )
                out.append((projected.reshape(-1, 2) - image_pts).ravel())
        return np.concatenate(out)

    residuals_before = _residuals(x0)
    rms_before = float(np.sqrt(np.mean(residuals_before ** 2)))

    method = "lm" if residuals_before.size >= x0.size else "trf"
    result = least_squares(_residuals, x0, method=method)

    residuals_after = _residuals(result.x)
    rms_after = float(np.sqrt(np.mean(residuals_after ** 2)))

    board_transforms, _cam_poses = _unpack(result.x)
    new_transforms = dict(initial.transforms)
    new_transforms.update(board_transforms)

    message = (
        f"{initial.message}; bundle adjustment по {n_frames} фото "
        f"({', '.join(optimized_boards)}): RMS репроекции {rms_before:.3f}px -> {rms_after:.3f}px"
    )
    refined = BoardRegistration(
        transforms=new_transforms,
        reference_board=initial.reference_board,
        bridge_photo_counts=initial.bridge_photo_counts,
        unregistered_boards=initial.unregistered_boards,
        message=message,
    )
    report = BundleAdjustmentReport(
        n_frames_used=n_frames,
        rms_reprojection_px_before=rms_before,
        rms_reprojection_px_after=rms_after,
        boards_refined=optimized_boards,
    )
    return refined, report
