#!/usr/bin/env python3
"""Ручная отладка шиберов через API.

Спавнит один объект, опрашивает статус каждые 0.3с, печатает события и
позиции объекта (x,y,z) + состояние шиберов. По финалу (delivered/missed)
— сводка. Управление шиберами: --route B/C/D (ручной режим) или --auto.

    python scripts/manual_debug.py --host 172.29.48.1 --type detergent --route B
    python scripts/manual_debug.py --type bottle --auto
    python scripts/manual_debug.py --type pouf --route C --reset
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
    p.add_argument("--type", default="box_small", help="тип объекта")
    p.add_argument("--route", choices=["B", "C", "D", "auto"], default="auto",
                   help="режим шибера: B/C/D (ручной) или auto")
    p.add_argument("--reset", action="store_true", help="reset перед спавном")
    p.add_argument("--timeout", type=float, default=40.0, help="таймаут ожидания")
    p.add_argument("--poll", type=float, default=0.4, help="интервал опроса")
    p.add_argument("--show-pos", action="store_true", help="печатать позицию каждый опрос")
    args = p.parse_args()

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
        print(f"[mode] manual route={args.route}")

    # спавн
    before = {o["uid"] for o in api(args.host, args.port, "status")["objects"]}
    api(args.host, args.port, "spawn", {"type": args.type})
    print(f"[spawn] type={args.type}")

    # найти uid
    uid = None
    deadline = time.time() + 8
    while uid is None and time.time() < deadline:
        st = api(args.host, args.port, "status")
        for o in st["objects"]:
            if o["uid"] not in before:
                uid = o["uid"]
        if uid is None:
            time.sleep(0.3)
    if uid is None:
        print("FAIL: объект не заспавнился")
        return 1
    print(f"[uid] {uid}")

    # опрос до финала
    seen_events = set()
    t0 = time.time()
    last_pos = None
    while time.time() - t0 < args.timeout:
        st = api(args.host, args.port, "status")
        # события
        for e in st.get("events", []):
            key = (e.get("t"), e.get("event"), e.get("uid"))
            if key not in seen_events and e.get("uid") == uid:
                seen_events.add(key)
                # только события нашего объекта
                print(f"  [t={e.get('t')}] {e.get('event')} {e.get('target') or e.get('result') or ''}".rstrip())
        # объект
        obj = next((o for o in st["objects"] if o["uid"] == uid), None)
        if obj is None:
            time.sleep(args.poll)
            continue
        pos = (obj["x"], obj["y"], obj["z"])
        if args.show_pos and pos != last_pos:
            print(f"  pos x={obj['x']} y={obj['y']} z={obj['z']} state={obj['state']} paddles={st['paddles']}")
            last_pos = pos
        if obj["state"] in ("delivered", "missed"):
            print(f"\n[RESULT] state={obj['state']} target={obj['target']} result={obj['result']}")
            ok = obj["state"] == "delivered" and obj["result"] == obj["target"]
            print("ok" if ok else "FAIL")
            return 0 if ok else 1
        time.sleep(args.poll)
    print(f"\n[TIMEOUT] state={obj['state'] if obj else 'gone'}")
    return 2


if __name__ == "__main__":
    sys.exit(main())