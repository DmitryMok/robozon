#!/usr/bin/env bash
# Установка Webots R2025a в ~/opt/webots (tarball, без root).
# Повторный запуск безопасен: существующая установка не перекачивается.
set -euo pipefail

WEBOTS_HOME="${WEBOTS_HOME:-$HOME/opt/webots}"
VERSION="R2025a"
URL="https://github.com/cyberbotics/webots/releases/download/${VERSION}/webots-${VERSION}-x86-64.tar.bz2"

if [[ -x "$WEBOTS_HOME/webots" ]]; then
  echo "Webots уже установлен: $WEBOTS_HOME"
  exit 0
fi

# Прокси из окружения может ломать доступ к github — качаем напрямую
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

echo "== Скачивание Webots ${VERSION} (~150 МБ) =="
mkdir -p "$(dirname "$WEBOTS_HOME")"
tmp="$(mktemp -d)"
curl -L --fail -o "$tmp/webots.tar.bz2" "$URL"

echo "== Распаковка в $(dirname "$WEBOTS_HOME") =="
tar xjf "$tmp/webots.tar.bz2" -C "$(dirname "$WEBOTS_HOME")"
rm -rf "$tmp"

echo "OK: $WEBOTS_HOME"
"$WEBOTS_HOME/webots" --version || true
