#!/usr/bin/env bash
# Запуск симуляции участка сортировки.
#   ./scripts/run.sh            — обычный запуск (окно Webots через WSLg + web-стрим)
#   ./scripts/run.sh --headless — без окна (только web-стрим), нужен xvfb-run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBOTS_HOME="${WEBOTS_HOME:-$HOME/opt/webots}"
export WEBOTS_HOME
export PYTHONPATH="$WEBOTS_HOME/lib/controller/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Прокси мешает первому обращению Webots к своему серверу ассетов и может
# перехватывать запросы к локальной панели/мешам — снимаем прокси полностью
# и явно исключаем localhost из проксирования (на случай, если окружение
# сервера организаторов подставит прокси заново через /etc/environment).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export no_proxy="127.0.0.1,localhost,0.0.0.0"
export NO_PROXY="$no_proxy"

STREAM_PORT="$(python3 -c "import sys; sys.path.insert(0, '$ROOT'); from sim.config import load_layout; print(load_layout()['http']['stream_port'])")"
PANEL_PORT="$(python3 -c "import sys; sys.path.insert(0, '$ROOT'); from sim.config import load_layout; print(load_layout()['http']['port'])")"

echo "== Генерация мира из config/layout.yaml =="
python3 "$ROOT/tools/gen_world.py"

WEBOTS_ARGS=(
  --stream
  "--port=$STREAM_PORT"
  --batch
  --mode=realtime
  --stdout
  --stderr
  "$ROOT/webots/worlds/sorting_line.wbt"
)

echo "== Запуск Webots (стрим: ws://localhost:$STREAM_PORT) =="
echo "== Панель управления: http://localhost:$PANEL_PORT/ =="

if [[ "${1:-}" == "--headless" ]]; then
  export ROBOZON_HEADLESS=1
  exec xvfb-run -a "$WEBOTS_HOME/webots" "${WEBOTS_ARGS[@]}"
else
  # Есть нативное окно Webots — 3D-вид в браузере по умолчанию не рендерим
  # (иначе Webots кодирует сцену ещё и для web-стрима вхолостую, задача
  # панели тут — только управление); включить его можно вручную в панели.
  export ROBOZON_HEADLESS=0
  exec "$WEBOTS_HOME/webots" "${WEBOTS_ARGS[@]}"
fi
