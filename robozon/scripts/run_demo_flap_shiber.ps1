# Запуск автономной демо-сцены demo_flap_shiber (шибер перенесён на вход в
# зону и развёрнут на 180°: закрыто=барьер у кромки, открыто=проём) —
# отдельно от основной линии, БЕЗ gen_world.py и без основной HTTP-панели
# (сцена не читает config/layout.yaml для мира, только для геометрии щита).
#   scripts\run_demo_flap_shiber.ps1              — обычный запуск, окно Webots
#   scripts\run_demo_flap_shiber.ps1 -Headless    — без окна/рендера, для
#                                                    быстрой проверки по логу
#
# Веб-панель (спавн одиночно/потоком, PASS/FAIL) — http://localhost:8021/
# после запуска Webots.
#
# Правка геометрии/тайминга цикла — tools/gen_demo_flap_shiber.py, затем:
#   C:\Projects\.venvs\venv-cv-pytorch\Scripts\python.exe tools\gen_demo_flap_shiber.py
param(
    [switch]$Headless
)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path "$PSScriptRoot\..").Path
$WebotsHome = "C:\Program Files\Webots"
$VenvPython = "C:\Projects\.venvs\venv-cv-pytorch\Scripts\python.exe"

$env:WEBOTS_HOME = $WebotsHome
$env:PYTHONPATH = "$WebotsHome\lib\controller\python;$Root"

# Webots запускает .py-контроллеры через "python" из PATH.
$env:Path = "C:\Projects\.venvs\venv-cv-pytorch\Scripts;" + $env:Path

Write-Output "== Генерация demo_flap_shiber.wbt из tools/gen_demo_flap_shiber.py =="
& $VenvPython "$Root\tools\gen_demo_flap_shiber.py"

Write-Output "== Запуск Webots =="
Write-Output "== Панель: http://localhost:8021/ =="

$WebotsArgs = @("--batch", "--mode=realtime", "--stdout", "--stderr")
if ($Headless) {
    $WebotsArgs = @("--batch", "--minimize", "--no-rendering", "--mode=fast", "--stdout", "--stderr")
}

& "$WebotsHome\msys64\mingw64\bin\webots.exe" $WebotsArgs "$Root\webots\worlds\demo_flap_shiber.wbt"
