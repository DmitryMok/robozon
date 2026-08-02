#!/usr/bin/env python3
"""Пакетная ручная отладка: прогоняет список объектов подряд БЕЗ reset
(как в реальной работе), в авто-режиме, печатает результат каждого.

    python scripts/manual_batch.py --host 172.29.48.1 \
        --types detergent,plate,bottle,box_small,pouf,bag,lunchbox
    python scripts/manual_batch.py --types bottle,bottle,bottle,bottle  # стресс по одному типу
"""
import argparse
import json
import sys
import time
import urllib.request


def api(host: str, port: int, path: str, body: dict | None = None) -> dict:
    url = f"http://{host}:{port}/api/{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--types", required=True, help="список типов через запятую")
    p.add_argument("--route", choices=["auto", "B", "C", "D"], default="auto",
                   help="режим шибера: B/C/D (ручной, всё в эту зону) или auto")
    p.add_argument("--reset", action="store_true", help="reset перед стартом")
    p.add_argument("--gap", type=float, default=2.0, help="пауза между спавнами, с")
    p.add_argument("--timeout", type=float, default=40.0, help="таймаут на объект")
    args = p.parse_args()

    types = args.types.split(",")
    if args.reset:
        api(args.host, args.port, "reset", {})
        time.sleep(1.5)
        print("[reset]")

    # режим шибера
    if args.route == "auto":
        api(args.host, args.port, "auto", {})
        print("[mode] auto")
    else:
        api(args.host, args.port, "route", {"zone": args.route})
        print(f"[mode] manual route={args.route} (все объекты -> {args.route})")

    results = []
    for i, obj_type in enumerate(types):
        before = {o["uid"] for o in api(args.host, args.port, "status")["objects"]}
        api(args.host, args.port, "spawn", {"type": obj_type})
        # найти uid
        uid = None
        deadline = time.time() + 8
        while uid is None and time.time() < deadline:
            st = api(args.host, args.port, "status")
            for o in st["objects"]:
                if o["uid"] not in before:
                    uid = o["uid"]
            if uid is None:
                time.sleep(0.2)
        if uid is None:
            print(f"  [{i+1}/{len(types)}] {obj_type:<12} FAIL не заспавнился")
            results.append((obj_type, "no_spawn", None))
            continue

        # ждать финала
        t0 = time.time()
        final = None
        while time.time() - t0 < args.timeout:
            st = api(args.host, args.port, "status")
            obj = next((o for o in st["objects"] if o["uid"] == uid), None)
            if obj and obj["state"] in ("delivered", "missed"):
                final = obj
                break
            time.sleep(0.3)
        if final is None:
            print(f"  [{i+1}/{len(types)}] {obj_type:<12} uid={uid} TIMEOUT")
            results.append((obj_type, "timeout", None))
        else:
            ok = final["state"] == "delivered" and final["result"] == final["target"]
            mark = "ok " if ok else "FAIL"
            print(f"  [{i+1}/{len(types)}] {obj_type:<12} uid={uid} {mark} "
                  f"state={final['state']} target={final['target']} result={final['result']}")
            results.append((obj_type, final["state"], final["result"]))
        # пауза перед следующим
        time.sleep(args.gap)

    # сводка
    ok = sum(1 for _, s, r in results if s == "delivered" and r is not None)
    fails = [(t, s, r) for t, s, r in results if not (s == "delivered" and r is not None)]
    print(f"\nВсего: {len(results)}  ok: {ok}  fails: {len(fails)}")
    if fails:
        for t, s, r in fails:
            print(f"  FAIL {t}: {s} -> {r}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())