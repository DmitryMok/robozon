#!/usr/bin/env python3
"""Стресс-тест исполнительного механизма: спавн товаров каждые 0.7с в
случайных ориентациях, отлов застреваний и неверных доставок.

Требует запущенной симуляции (scripts/run.ps1). Логирует каждый спавн
(тип, rot_deg) и финальный результат (delivered/missed, целевая/фактическая
зона). По итогам — сводка: всего, доставлено верно, неверно, missed,
разбивка по типам + список проблемных спавнов (uid/тип/rot_deg/результат).

    python scripts/test_stress.py [--port 8008] [--interval 0.7] [--count 40]
    python scripts/test_stress.py --types bottle,box_small
"""
import argparse
import json
import random
import sys
import time
import urllib.request

DEFAULT_INTERVAL_S = 0.7
DEFAULT_COUNT = 40
DELIVERY_TIMEOUT_S = 60


def api(host: str, port: int, path: str, body: dict | None = None) -> dict:
    url = f"http://{host}:{port}/api/{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def spawn(host: str, port: int, obj_type: str) -> dict:
    return api(host, port, "spawn", {"type": obj_type})


def status(host: str, port: int) -> dict:
    return api(host, port, "status")


def objects(host: str, port: int) -> list[dict]:
    return api(host, port, "objects")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="хост симуляции (из WSL — IP Windows-хоста, напр. 172.29.48.1)")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--types", help="список типов через запятую (по умолчанию все)")
    parser.add_argument("--seed", type=int, default=None, help="seed ГПСЧ для воспроизводимости")
    parser.add_argument("--no-reset", action="store_true", help="не делать reset перед стартом")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    catalog = {o["type"]: o for o in objects(args.host, args.port)}
    types = args.types.split(",") if args.types else list(catalog)

    if not args.no_reset:
        api(args.host, args.port, "reset", {})
        time.sleep(1.5)

    spawns: list[dict] = []     # что заспавнили (uid, тип, rot_deg, t_spawn)
    next_uid = 1
    t_start = time.time()

    # Спавним count товаров с интервалом interval, параллельно ждём доставки.
    spawned = 0
    while spawned < args.count or any(s["result"] is None for s in spawns):
        now = time.time()
        if spawned < args.count and now - t_start >= spawned * args.interval:
            obj_type = random.choice(types)
            before = {o["uid"] for o in status(args.host, args.port)["objects"]}
            spawn(args.host, args.port, obj_type)
            # найти uid нового объекта
            uid = None
            deadline = now + 5
            while uid is None and time.time() < deadline:
                for o in status(args.host, args.port)["objects"]:
                    if o["uid"] not in before:
                        uid = o["uid"]
                if uid is None:
                    time.sleep(0.2)
            # rot_deg — из events последнего spawn
            rot_deg = None
            for e in reversed(status(args.host, args.port)["events"]):
                if e.get("event") == "spawn" and e.get("uid") == uid:
                    rot_deg = e.get("rot_deg")
                    break
            spawns.append({"uid": uid, "type": obj_type, "rot_deg": rot_deg,
                            "t_spawn": round(now - t_start, 2), "result": None})
            spawned += 1
            print(f"[{now - t_start:5.1f}s] spawn #{spawned:<3} uid={uid} type={obj_type:<10} rot={rot_deg}")

        # опросить состояние, обновить результаты
        st = status(args.host, args.port)
        by_uid = {o["uid"]: o for o in st["objects"]}
        for s in spawns:
            if s["result"] is not None:
                continue
            o = by_uid.get(s["uid"])
            if o is None:
                # объект исчез из списка (устаревший) — пропускаем, ждём
                continue
            if o["state"] in ("delivered", "missed"):
                s["result"] = {"state": o["state"], "result": o["result"],
                                "target": o["target"]}
                mark = "ok " if (o["state"] == "delivered" and o["result"] == o["target"]) else "FAIL"
                print(f"  -> uid={s['uid']:<4} {mark} state={o['state']} target={o['target']} result={o['result'] or '-'}")

        if spawned >= args.count and all(s["result"] is not None for s in spawns):
            break
        time.sleep(0.2)

    # --- сводка ---
    print("\n" + "=" * 60)
    total = len(spawns)
    delivered_ok = sum(1 for s in spawns
                       if s["result"] and s["result"]["state"] == "delivered"
                       and s["result"]["result"] == s["result"]["target"])
    delivered_wrong = sum(1 for s in spawns
                          if s["result"] and s["result"]["state"] == "delivered"
                          and s["result"]["result"] != s["result"]["target"])
    missed = sum(1 for s in spawns if s["result"] and s["result"]["state"] == "missed")
    unresolved = sum(1 for s in spawns if s["result"] is None)
    print(f"Всего: {total}  доставлено_верно: {delivered_ok}  "
          f"доставлено_неверно: {delivered_wrong}  missed: {missed}  нет_результата: {unresolved}")

    # разбивка по типам
    by_type: dict[str, dict] = {}
    for s in spawns:
        d = by_type.setdefault(s["type"], {"total": 0, "ok": 0, "wrong": 0, "missed": 0})
        d["total"] += 1
        if s["result"]:
            if s["result"]["state"] == "delivered" and s["result"]["result"] == s["result"]["target"]:
                d["ok"] += 1
            elif s["result"]["state"] == "missed":
                d["missed"] += 1
            else:
                d["wrong"] += 1
    print("\nПо типам:")
    for t in sorted(by_type):
        d = by_type[t]
        print(f"  {t:<12} total={d['total']:>3} ok={d['ok']:>3} wrong={d['wrong']:>3} missed={d['missed']:>3}")

    # проблемные спавны
    problems = [s for s in spawns
                if s["result"] and (s["result"]["state"] == "missed"
                                     or s["result"]["state"] == "delivered"
                                     and s["result"]["result"] != s["result"]["target"])]
    if problems:
        print(f"\nПроблемные спавны ({len(problems)}):")
        for s in problems:
            print(f"  uid={s['uid']:<4} type={s['type']:<12} rot={s['rot_deg']:>7.1f}° "
                  f"t={s['t_spawn']:>5.1f}s target={s['result']['target']} "
                  f"-> state={s['result']['state']} result={s['result']['result'] or '-'}")
    return 0 if (delivered_wrong == 0 and missed == 0 and unresolved == 0) else 1


if __name__ == "__main__":
    sys.exit(main())