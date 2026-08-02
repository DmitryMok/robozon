#!/usr/bin/env python3
"""Непрерывный поток кадров top-камеры + моментные кадры side/diag (Фаза 4).

Гибридная схема захвата для отладки CV-триггера/трекинга:
  * **top** — непрерывный поток ~30fps (CV-триггер работает по top-локализации,
    ему нужен каждый кадр). Файлы:
    `stream_<uid>_top_<frame_idx:05d>.jpg` + `_objects.json`.
  * **side/diag** — снимаются КОНКУРЕНТНО одним тиком в момент пересечения
    целевым объектом порогов момента (start_side/center/end_side для side,
    center для diag; пороги — sim/cv_moments.py). top-кадр этого же тика
    идёт в поток (как обычно), а side/diag — в moment-файлы (как
    capture_multi_object_frames.py):
    `moment_<uid>_<camera>_<moment>.jpg` + `_objects.json`.
    Зеркальные камеры НЕ снимаются (не нужны для CV-триггера — см.
    постановку; для сетки 9 ракурсов с зеркальными есть отдельный скрипт
    capture_multi_object_frames.py).

Дополнительно `stream_<uid>_meta.json` — {frame_rate, total_frames,
duration_s, cameras, objects: [{uid, type, spawn_x}], zone,
moment_events: {moment_name: {frame_idx, x}}} — привязка моментов к кадрам
top-потока (для синхронизации при отладке триггера).

Архитектура захвата — вариант (A) из заметки задачи: через существующий
HTTP API `/api/cv/frame/<camera>` (каждый запрос включает камеру на settle
3 тика = 24мс, потом выключает). Реальная частота top-потока упирается в
settle+рендер 2592×1944 одной камеры — засекается фактическая (фиксируется в
meta.json). При <10fps — WARN (рассмотреть вариант (B) — непрерывный режим
в supervisor_main.py, камера enable()'ута всё время — отдельная задача).

Наборы — те же 4 что в capture_multi_object_frames.py (целевой — первый):
    stream/pen_boxlarge/              — pen + box_large позади
    stream/boxsmall_boxlarge/         — box_small + box_large
    stream/cylinder_boxsmall/         — cylinder + box_small (низкоконтрастный)
    stream/triple_pen_boxlarge_boxsmall/ — 3 объекта

Требует запущенной симуляции (scripts/run.ps1, порт 8008). Грабли (см.
01 Sources/03 Conventions.md): прокси Windows ломает HTTP — в скрипте
прокси-переменные снимаются явно (urllib opener без proxy). 503 на первый
запрос CV-камер после старта (холодный рендер) — ретрай.

Пример (все 4 набора):
    python3 tools/capture_cv_stream.py --out assets/cv_grid_test_frames/multi/stream

Один набор, без side/diag (только top-поток):
    python3 tools/capture_cv_stream.py \\
        --sets pen_boxlarge --no-moments --out /tmp/stream
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.config import load_layout                                                     # noqa: E402
from sim.cv_moments import moment_thresholds, threshold_captures                        # noqa: E402

# Прокси ломает локальные запросы (см. 01 Sources/03 Conventions.md) — opener
# без proxy. Глобально для всех urllib-вызовов в скрипте.
_PROXY_HANDLER = urllib.request.ProxyHandler({})
_OPENER = urllib.request.build_opener(_PROXY_HANDLER)

# Сеты — те же что в capture_multi_object_frames.py (целевой — ПЕРВЫЙ,
# первый едет по ленте; соседи позади — отстанут на 0.5м при 0.5с интервале
# и 1.0 м/с ленты).
SETS = {
    "pen_boxlarge":               ["pen", "box_large"],
    "boxsmall_boxlarge":          ["box_small", "box_large"],
    "cylinder_boxsmall":          ["cylinder", "box_small"],
    "triple_pen_boxlarge_boxsmall": ["pen", "box_large", "box_small"],
}

# Целевая частота top-потока (между циклами). 30 fps — как прод, реальная
# упрётся в Webots-рендер/settle (фиксируется в meta.json).
_TARGET_FPS = 30.0

# Зона top-потока: от cv_rig.x - WINDOW_BEFORE до cv_rig.x + WINDOW_AFTER. При
# 1.0 м/с ~2с движения → ~60 кадров при 30fps. Симметричная (по постановке
# задачи — от cv_rig.x - 1.0 до cv_rig.x + 1.0, ~3с — расширено для запаса
# под трекинг слияний/разделений на краях зоны). Пороги моментов
# (start_top..end_top) лежат ВНУТРИ этого окна —moments ловятся по пути.
_WINDOW_BEFORE_M = 1.0
_WINDOW_AFTER_M = 1.0

# Допуск: снимаем, как только целевой вошёл в зону (x >= start_x).
_EVENT_TOLERANCE_M = 0.05

# Ретрай при 503 (холодный рендер CV-камер после старта Webots — см.
# Конвенции). На каждый запрос кадра — отдельный ретрай (503 возможен
# посреди потока, не только на первом кадре).
_HTTP_RETRIES = 3
_HTTP_RETRY_DELAY = 0.5


def _get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url)
    last_exc = None
    for _ in range(_HTTP_RETRIES):
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (502, 503):
                time.sleep(_HTTP_RETRY_DELAY)
                continue
            raise
        except urllib.error.URLError as e:
            last_exc = e
            time.sleep(_HTTP_RETRY_DELAY)
            continue
    raise RuntimeError(f"HTTP {url} failed after {_HTTP_RETRIES} retries: {last_exc}")


def _post(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_status(port: int) -> dict:
    return json.loads(_get(f"http://localhost:{port}/api/status"))


def api_reset(port: int) -> None:
    _post(f"http://localhost:{port}/api/reset")


def api_spawn(port: int, obj_type: str, rotation: str = "identity") -> None:
    _post(f"http://localhost:{port}/api/spawn", {"type": obj_type, "rotation": rotation})


def save_camera(port: int, name: str, out_path: Path) -> None:
    data = _get(f"http://localhost:{port}/api/cv/frame/{name}", timeout=15.0)
    out_path.write_bytes(data)


def spawn_and_get_uid(port: int, obj_type: str, before_uids: set[int],
                      timeout: float = 15.0) -> int:
    api_spawn(port, obj_type)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for o in api_status(port)["objects"]:
            if o["uid"] not in before_uids:
                return o["uid"]
        time.sleep(0.05)
    raise RuntimeError(f"объект {obj_type} не заспавнился за {timeout}с")


def _find_uid(status: dict, uid: int) -> dict | None:
    for o in status["objects"]:
        if o["uid"] == uid:
            return o
    return None


def _ground_truth(status: dict) -> list[dict]:
    return [{"uid": o["uid"], "type": o["type"],
             "x": o["x"], "y": o["y"], "z": o["z"]}
            for o in status["objects"]]


def _write_objects_json(out_dir: Path, stem: str, target_uid: int,
                         frame_idx: int, gt: list[dict]) -> None:
    payload = {"target_uid": target_uid, "frame_idx": frame_idx, "objects": gt}
    (out_dir / f"{stem}_objects.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_stream(port: int, set_name: str, obj_types: list[str],
               out_root: Path, with_moments: bool,
               spawn_interval_s: float = 0.5) -> dict:
    """Один поток: reset, спавн N объектов, непрерывный top-поток пока
    целевой проедет зону [cv_rig.x - WINDOW_BEFORE, cv_rig.x + WINDOW_AFTER];
    по пути — моментные кадры side/diag в порогах start_side/center/end_side
    (side) и center (diag), если with_moments. Возвращает meta-словарь (так же
    пишется в <out>/stream_<uid>_meta.json)."""
    out_dir = out_root / set_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rig = load_layout()["cv_rig"]
    cv_x = rig["x"]
    start_x = cv_x - _WINDOW_BEFORE_M
    end_x = cv_x + _WINDOW_AFTER_M

    # Пороги и привязка камер к моментам — из sim/cv_moments. Зеркальные
    # исключаем (не нужны для CV-триггера). top исключаем из моментных камер —
    # top снимается в потоке, момент top лишь отмечается в meta.
    th = moment_thresholds(rig)
    captures = threshold_captures()
    # Пороги в порядке движения (+X): start_top, start_side, center,
    # end_side, end_top. Каждый порог -> (moment_label, moment_cameras),
    # где moment_cameras — side/diag (БЕЗ top, БЕЗ зеркальных), которые надо
    # снять по тику в этом пороге. top-моменты (start_top/end_top/center-top)
    # — только отметка в meta (кадр уже в потоке).
    moments: list[tuple[str, float, str, list[str]]] = []
    for th_name in ("start_top", "start_side", "center", "end_side", "end_top"):
        moment_label = captures[th_name][0][1]   # "start"/"center"/"end"
        cams = [c for c, _m in captures[th_name]
                if c in ("side", "diag")]  # без top (в потоке) и без зеркальных
        moments.append((th_name, th[th_name], moment_label, cams))
    moments.sort(key=lambda m: m[1])

    print(f"\n=== {set_name}: {obj_types} ===")
    print(f"  зона top-потока: x in [{start_x:.3f}, {end_x:.3f}]")
    if with_moments:
        print(f"  моментные камеры (side/diag): "
              f"{sorted({c for _, _, _, cs in moments for c in cs})}")
    else:
        print("  моментные кадры side/diag: выключены (--no-moments)")

    api_reset(port)
    time.sleep(1.0)

    # Спавн N объектов с интервалом. Целевой — первый. rotation=identity для
    # детерминированной позы (как capture_multi_object_frames.py).
    target_uid = None
    spawn_x: list[dict] = []
    before = {o["uid"] for o in api_status(port)["objects"]}
    for i, obj_type in enumerate(obj_types):
        uid = spawn_and_get_uid(port, obj_type, before)
        before.add(uid)
        st = api_status(port)
        o = _find_uid(st, uid)
        sx = o["x"] if (o and o["x"] is not None) else None
        spawn_x.append({"uid": uid, "type": obj_type, "spawn_x": sx})
        if i == 0:
            target_uid = uid
        print(f"  спавн {obj_type} -> uid={uid}"
              f"{' (целевой)' if i == 0 else ''}")
        if i < len(obj_types) - 1:
            time.sleep(spawn_interval_s)

    if target_uid is None:
        raise RuntimeError("целевой объект не заспавнен")

    # Ждём, пока целевой войдёт в зону (x >= start_x - tolerance). Объект
    # движется в +X.
    print(f"  ждём целевого uid={target_uid} до x={start_x:.3f} ...")
    deadline = time.time() + 60.0
    while time.time() < deadline:
        o = _find_uid(api_status(port), target_uid)
        if o is None or o["x"] is None:
            time.sleep(0.03)
            continue
        if o["x"] >= start_x - _EVENT_TOLERANCE_M:
            break
        time.sleep(0.03)
    else:
        raise RuntimeError(
            f"целевой uid={target_uid} не дошёл до x={start_x:.3f} за 60с")

    # Непрерывный top-поток + моментные side/diag. На каждом кадре:
    #   1. /api/status (snapshot тика).
    #   2. Проверяем пороги: впервые пересечённые — снимаем moment-камеры
    #      этого порога (side/diag) конкурентно ВМЕСТЕ с top этого тика
    #      (одним тиком — иначе объект уедёт между камерами). top-кадр идёт
    #      в stream-файл, side/diag — в moment-файлы. Пороги top
    #      (start_top/end_top/center) — только отметка в meta (кадр уже в
    #      потоке).
    #   3. Если моментных камер нет (или момента нет) — снимаем только top.
    frame_idx = 0
    t_loop_start = time.monotonic()
    t_first = None
    t_last = None
    fired: set[str] = set()
    moment_events: dict[str, dict] = {}

    while True:
        status = api_status(port)
        o = _find_uid(status, target_uid)
        x_now = o["x"] if (o and o["x"] is not None) else None

        # Какие моментные камеры снять на этом тике ( впервые пересечённые
        # пороги, чьи moment-камеры непусты). top-пороги дают пустой список
        # (кадр top идёт в поток), но фиксируем событие в meta.
        fire_moment_cams: list[str] = []
        fire_moment_labels: list[str] = []
        if x_now is not None:
            for th_name, th_x, moment_label, cams in moments:
                if th_name in fired:
                    continue
                if x_now >= th_x - _EVENT_TOLERANCE_M:
                    fired.add(th_name)
                    moment_events[th_name] = {"frame_idx": frame_idx,
                                              "x": x_now,
                                              "moment": moment_label,
                                              "threshold_x": th_x}
                    if cams:
                        fire_moment_cams.extend(cams)
                        fire_moment_labels.append(moment_label)

        # Камеры этого тика: top (всегда, в поток) + moment-камеры (side/diag
        # по тику). Конкурентно одним тиком — иначе объект уедет между
        # камерами (как capture_cv_frames.py / capture_multi_object_frames.py).
        # Собираем список (kind, cam, moment_label) для запроса.
        requests: list[tuple[str, str, str | None]] = [("stream", "top", None)]
        if with_moments:
            for cam in fire_moment_cams:
                ml = next(ml for tn, _, ml, cs in moments
                           if cam in cs and tn in fired and ml in fire_moment_labels)
                requests.append(("moment", cam, ml))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as ex:
            fut_map = {ex.submit(_get,
                                 f"http://localhost:{port}/api/cv/frame/{cam}",
                                 15.0): (kind, cam, ml)
                       for kind, cam, ml in requests}
            results = {}
            for fut in concurrent.futures.as_completed(fut_map):
                kind, cam, ml = fut_map[fut]
                results[(kind, cam, ml)] = fut.result()

        # Сохраняем top (stream).
        top_bytes = results[("stream", "top", None)]
        (out_dir / f"stream_{target_uid}_top_{frame_idx:05d}.jpg").write_bytes(top_bytes)
        _write_objects_json(out_dir, f"stream_{target_uid}_top_{frame_idx:05d}",
                            target_uid, frame_idx, _ground_truth(status))

        # Сохраняем moment-камеры (side/diag).
        if with_moments:
            for (_kind, cam, ml), data in results.items():
                if _kind != "moment":
                    continue
                stem = f"moment_{target_uid}_{cam}_{ml}"
                (out_dir / f"{stem}.jpg").write_bytes(data)
                _write_objects_json(out_dir, stem, target_uid, frame_idx,
                                    _ground_truth(status))

        now = time.monotonic()
        if t_first is None:
            t_first = now
        t_last = now

        tag = f" +moments={sorted(set(fire_moment_labels))}" if fire_moment_labels else ""
        print(f"  кадр {frame_idx:05d} x={x_now:.3f}{tag}")

        # Конец зоны — выходим.
        if o is not None and o["x"] is not None and o["x"] > end_x:
            break

        frame_idx += 1
        dt = time.monotonic() - t_loop_start
        sleep_s = max(0.0, 1.0 / _TARGET_FPS - dt)
        if sleep_s > 0:
            time.sleep(sleep_s)
        t_loop_start = time.monotonic()

        if frame_idx > 2000:
            print("  WARN: достигнут лимит 2000 кадров, останов", file=sys.stderr)
            break

    total_frames = frame_idx + 1
    duration_s = (t_last - t_first) if (t_first and t_last) else 0.0
    actual_fps = (total_frames - 1) / duration_s if duration_s > 0 else 0.0

    # Перечитываем spawn_x для meta (после спавна X мог быть None).
    for it in spawn_x:
        st = api_status(port)
        o = _find_uid(st, it["uid"])
        if o and o["x"] is not None:
            it["spawn_x"] = o["x"]

    meta = {
        "frame_rate": round(actual_fps, 2),
        "frame_rate_target": _TARGET_FPS,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 3),
        "cameras": {"stream": ["top"],
                    "moments": sorted({c for _, _, _, cs in moments for c in cs})
                    if with_moments else []},
        "target_uid": target_uid,
        "objects": spawn_x,
        "zone": {"start_x": start_x, "end_x": end_x, "cv_rig_x": cv_x},
        "moment_events": moment_events,
    }
    meta_path = out_dir / f"stream_{target_uid}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    n_moment_files = sum(1 for p in out_dir.iterdir() if p.name.startswith("moment_"))
    print(f"  готово: top-поток {total_frames} кадров за {duration_s:.2f}с "
          f"({actual_fps:.1f} fps), moment-кадров side/diag: {n_moment_files // 2} "
          f"-> {out_dir}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--out", required=True,
                    help="корневая папка для наборов (создаются подпапки по имени сета)")
    ap.add_argument("--sets", default="all",
                    help="наборы через запятую (по умолчанию all = все 4): "
                         + ",".join(SETS))
    ap.add_argument("--no-moments", action="store_true",
                    help="не снимать side/diag в порогах момента (только top-поток)")
    ap.add_argument("--spawn-interval", type=float, default=0.5,
                    help="интервал между спавнами, с (0.5 = как spawn.min_interval_s)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.sets == "all":
        sets = list(SETS.items())
    else:
        names = [s.strip() for s in args.sets.split(",")]
        missing = [n for n in names if n not in SETS]
        if missing:
            print(f"неизвестные наборы: {missing}; доступные: {list(SETS)}",
                  file=sys.stderr)
            return 1
        sets = [(n, SETS[n]) for n in names]

    # Проверка связи с симулятором.
    try:
        api_status(args.port)
    except Exception as e:
        print(f"симулятор не отвечает на localhost:{args.port}: {e}",
              file=sys.stderr)
        print("запустите: scripts/run.ps1", file=sys.stderr)
        return 1

    summary = []
    rc = 0
    for set_name, obj_types in sets:
        try:
            meta = run_stream(args.port, set_name, obj_types, out_root,
                              with_moments=not args.no_moments,
                              spawn_interval_s=args.spawn_interval)
            fps = meta["frame_rate"]
            ok = fps >= 10.0
            tag = "ok" if ok else f"ok (LOW FPS {fps:.1f})"
            summary.append((set_name, meta["target_uid"], tag))
            if fps < 10.0:
                print(f"  WARN: фактическая частота top-потока {fps:.1f} fps < 10 — "
                      "рассмотреть вариант (B) непрерывного режима в supervisor "
                      "или уменьшить разрешение", file=sys.stderr)
        except Exception as e:
            summary.append((set_name, None, f"FAIL: {e}"))
            print(f"  ОШИБКА: {e}", file=sys.stderr)
            rc = 1

    print("\n=== Итог ===")
    for set_name, uid, status in summary:
        print(f"  {set_name:40s} uid={uid} {status}")
    return rc


if __name__ == "__main__":
    sys.exit(main())