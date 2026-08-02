#!/usr/bin/env bash
# WSL-порт run_demo_mechanism.ps1 — запуск автономной демо-сцены demo_mechanism
# (исполнительный механизм — отдельная сцена для финальной сдачи, см. вики "План
# финализации сдачи (раздельные демо CV и исполнительного механизма)") — БЕЗ
# gen_world.py и без основной HTTP-панели (это отдельная сцена от sorting_line.wbt).
#   ./scripts/run_demo_mechanism.sh            — обычный запуск (окно Webots через WSLg)
#   ./scripts/run_demo_mechanism.sh --headless — без окна (xvfb-run), 3D-стрим в браузере
#
# Веб-панель (спавн + встроенный 3D-стрим) — http://localhost:8022/ после запуска Webots.
# Независимый CV-отчёт (GPU/torchhull) — http://localhost:8010/ (поднимается этим скриптом).
#
# Правка геометрии — tools/gen_demo_mechanism.py, затем перезапуск этого скрипта
# (генерирует .wbt заново сам, как и run.sh для sorting_line.wbt).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEBOTS_HOME="${WEBOTS_HOME:-$HOME/opt/webots}"
export WEBOTS_HOME
export PYTHONPATH="$WEBOTS_HOME/lib/controller/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Те же прокси-грабли, что в run.sh — снимаем прокси полностью, localhost явно исключаем.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export no_proxy="127.0.0.1,localhost,0.0.0.0"
export NO_PROXY="$no_proxy"

# CV-отчёт требует GPU/torchhull для проверенной схемы torchhull_parallax=True (см. заметку
# задачи [[CV-отчёт по произвольному STL (математика без Webots и без SAM3)]]) — venv с
# torch+CUDA+torchhull уже настроен в WSL-копии (деплой-таргет), roboson_tools подхватывается
# автоматически через sys.path в cv_report_server.py (sibling-репозиторий по TOOLS_ROOT,
# путь не зависит от того, откуда запущен интерпретатор).
# Путь — относительно расположения этого скрипта (не абсолютный путь машины разработчика),
# см. [[Упаковка демо для Linux headless и Windows]]: у организатора репозиторий будет по
# другому пути после git clone.
CV_REPORT_PYTHON="${CV_REPORT_PYTHON:-$ROOT/.venv-cv/bin/python3}"
CV_REPORT_PORT=8010
STREAM_PORT=1235

if [[ ! -x "$CV_REPORT_PYTHON" ]]; then
  echo "Не найден Python для CV-отчёта: $CV_REPORT_PYTHON" >&2
  echo "Задайте CV_REPORT_PYTHON явно, если venv-cv лежит в другом месте." >&2
  exit 1
fi

echo "== Генерация demo_mechanism.wbt из tools/gen_demo_mechanism.py =="
python3 "$ROOT/tools/gen_demo_mechanism.py"

echo "== Запуск независимого CV-отчёта (GPU/torchhull) =="
# PATH с bin/ venv-cv ВПЕРЕДИ — нужен cmake/ninja для JIT-сборки torchhull (charonload),
# иначе первый запрос к /api/report падает с CommandNotFoundError: cmake (см. заметку задачи).
PATH="$(dirname "$CV_REPORT_PYTHON"):$PATH" \
  "$CV_REPORT_PYTHON" -B "$ROOT/tools/cv_report_server.py" \
    --host 127.0.0.1 --port "$CV_REPORT_PORT" &
CV_REPORT_PID=$!
# CV-отчёт не должен остаться фоновым процессом после завершения демо (аналог
# try/finally { Stop-Process } в run_demo_mechanism.ps1).
trap 'kill "$CV_REPORT_PID" 2>/dev/null || true' EXIT

echo "== Запуск Webots =="
echo "== Панель (спавн + встроенный 3D-стрим): http://localhost:8022/ =="
echo "== 3D-стрим: ws://localhost:$STREAM_PORT (виден на панели 8022, не отдельная страница) =="
echo "== CV-отчёт: http://localhost:$CV_REPORT_PORT/ =="

# --clear-cache: у пользователя мир зависает на "Downloading assets: Texture
# 'gtao_noise_texture.png'" (переживает перезапуск Webots) — подозрение на испорченный/
# залоченный кэш ассетов (~/.cache/Cyberbotics/Webots/assets); штатный флаг Webots для
# сброса такого кэша при старте. Файл gtao_noise_texture.png физически ЕСТЬ в установке
# Webots (~/opt/webots/resources/wren/textures/) — сетевая часть не должна быть нужна
# вовсе, если кэш почему-то не даёт использовать локальную копию напрямую.
WEBOTS_ARGS=(
  --stream
  "--port=$STREAM_PORT"
  --batch
  --mode=realtime
  --clear-cache
  --stdout
  --stderr
  "$ROOT/webots/worlds/demo_mechanism.wbt"
)

# Не exec — фоновый CV-отчёт нужно погасить трапом при выходе (exec заменил бы этот
# процесс образом Webots, и EXIT-трап выше никогда не сработал бы). Webots тоже запускаем
# фоновым job'ом и ждём wait, чтобы трап дотянулся и до его PID при явном kill скрипта
# (не только при Ctrl+C, который и так бьёт по всей группе процессов терминала).
#
# ROBOZON_HEADLESS — тот же сигнал, что в run.sh для supervisor_main.py / web/index.html:
# панель demo_panel_mechanism отдаёт его в /api/status (поле "headless"), JS по нему решает,
# показывать 3D-вид в браузере по умолчанию (1 — headless, нативного окна нет, стрим нужен;
# 0 — оконный запуск, у Webots уже есть окно, повторный рендер в браузере незачем — чекбокс
# остаётся для ручного включения, как в web/index.html панели sorting_line).
if [[ "${1:-}" == "--headless" ]]; then
  export ROBOZON_HEADLESS=1
  xvfb-run -a "$WEBOTS_HOME/webots" "${WEBOTS_ARGS[@]}" &
else
  export ROBOZON_HEADLESS=0
  "$WEBOTS_HOME/webots" "${WEBOTS_ARGS[@]}" &
fi
WEBOTS_PID=$!
trap 'kill "$CV_REPORT_PID" "$WEBOTS_PID" 2>/dev/null || true' EXIT
wait "$WEBOTS_PID"
# Известное ограничение (проверено живым прогоном): trap выше надёжно гасит CV-отчёт и
# прямой webots-процесс, НО если скрипт убит напрямую (kill <pid>, не Ctrl+C в терминале —
# тот бьёт по всей группе процессов и чистит всё сам), xvfb-run в --headless может оставить
# осиротевшими СВОИ дочерние Xvfb/webots-bin (xvfb-run не проксирует сигнал вложенным детям).
# Тот же класс проблемы уже задокументирован для sorting_line/run.sh в
# 01 Instructions.md (раздел «Грабли») — восстановление: `pkill -f webots-bin; pkill -f
# "xvfb-run.*webots"`.
