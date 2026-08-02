# roboson_tools — стенд для исследования геометрии силуэтов и Visual Hull

Лёгкое настольное приложение для проверки математической состоятельности метода определения
«круглых» (способных катиться) объектов по набору силуэтов (viewpoints), до переноса схемы
в реальную Webots-симуляцию/железо. Это НЕ симулятор конвейера, НЕ CV-система и НЕ замена
Webots — только быстрое исследование геометрии, проверка гипотез и поиск контрпримеров.

Ключевые расчёты — **габариты по силуэтам** и **«круглая проекция»** (потенциал к перекату:
существует ось, вдоль которой проекция формы близка к кругу, k = Rin/Rout ≥ 0.8) — описаны
в `docs/method.md`. Круглость отдельных сечений считается только справочно: круглое сечение
переката не означает (наклонное сечение конуса — круг: скос горлышка бутылки).

Полная постановка задачи: `02 Kanban notes/Реализовать геометрический стенд для исследования
силуэтов.md` в Obsidian-vault `c:\projects_wiki\robozon`.

## Установка (Windows)

```
c:\Users\<user>\AppData\Local\Programs\Python\Python312\python.exe -m venv c:\Projects\.venvs\venv-roboson-tools
c:\Projects\.venvs\venv-roboson-tools\Scripts\pip install -r requirements-dev.txt
```

## Запуск

GUI:

```
c:\Projects\.venvs\venv-roboson-tools\Scripts\python.exe -m roboson_tools.gui.main
```

Headless CLI (для скриптов/LLM):

```
c:\Projects\.venvs\venv-roboson-tools\Scripts\python.exe -m roboson_tools.cli evaluate ^
  --stl assets\stl\Цилиндр.stl --roll 0 --pitch 0 --yaw 0 --angles 0,45,90,135 --out result.json

:: «Проверить модель»: габариты по силуэтам + потенциал к перекату (круглая проекция)
c:\Projects\.venvs\venv-roboson-tools\Scripts\python.exe -m roboson_tools.cli check ^
  --stl bottle --axis-step 5
```

Тесты:

```
c:\Projects\.venvs\venv-roboson-tools\Scripts\python.exe -m pytest tests
```

## Структура

- `geometry/` — загрузка STL (trimesh), ориентация объекта (roll/pitch/yaw).
- `silhouette/` — построение силуэтов (Analytical Mode — ортографическая проекция;
  Camera Mode — заглушка под перспективную проекцию, v2).
- `visual_hull/` — восстановление 2D-сечения пересечением полос (bands) от каждого ракурса
  и полной формы (стопка сечений + габариты по силуэтам).
- `metrics/` — Rin (вписанная окружность), Rout (описанная окружность), k=Rin/Rout, PASS/FAIL;
  круглость проекции облака точек (по выпуклой оболочке).
- `search/` — брутфорс-поиск контрпримеров по ориентациям + сравнение наборов ракурсов.
- `export/` — сохранение эксперимента (STL, ориентация, углы, силуэты, сечение, метрики) в директорию.
- `cli/` — headless-интерфейс (evaluate/check/search/compare/export), JSON на входе/выходе.
- `docs/method.md` — как считаются габариты по силуэтам и круглая проекция (перекат).
- `gui/` — PyQt6-интерфейс, 4 панели (3D-модель на куске ленты конвейера через
  `pyqtgraph.opengl` / силуэты / сечение / метрики).
- `core/experiment.py` — вся расчётная логика в виде чистых функций, общая для GUI и CLI.

## Обязательный тестовый сценарий

Квадратная коробка `w×w×L`, `roll=22.5°`, ракурсы `[0,45,90,135]` → Visual Hull должен дать
правильный восьмиугольник, `Rin/Rout = cos(22.5°) ≈ 0.9239`. Численно проверяется в
`tests/test_visual_hull_box_octagon.py`.
