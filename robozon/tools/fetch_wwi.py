#!/usr/bin/env python3
"""Векторинг веб-вьюера Webots (wwi) в web/wwi для офлайн-работы.

Собранный WebotsView.js и его модули раздаются только с cyberbotics.com —
в tarball дистрибутива их нет. Скрипт скачивает дерево зависимостей
(JS-импорты, CSS, картинки, wasm) и переписывает абсолютные CDN-ссылки
на относительные. Запускается один раз при подготовке репозитория:

    python3 tools/fetch_wwi.py
"""
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://cyberbotics.com/wwi/R2025a/"
DST = Path(__file__).resolve().parent.parent / "web" / "wwi"

TEXT_EXT = {".js", ".css", ".html"}
# import ... from '...' | import('...') | url(...) в css | абсолютные ссылки на BASE
RE_IMPORT = re.compile(r"""import[^;'"]{0,120}?from\s*['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]""")
RE_ABS = re.compile(re.escape(BASE) + r"""([A-Za-z0-9_\-./]+)""")
RE_CSS_URL = re.compile(r"""url\(\s*['"]?([^'")]+)['"]?\s*\)""")
RE_STR_REF = re.compile(r"""['"]((?:\./)?(?:images|css|dependencies)/[A-Za-z0-9_\-./]+|wrenjs\.(?:js|wasm|data))['"]""")


def norm(base_dir: str, ref: str) -> str | None:
    ref = ref.split("?")[0].split("#")[0]
    if ref.startswith(("http://", "https://", "data:")):
        return None
    parts = []
    for p in (f"{base_dir}/{ref}" if not ref.startswith("/") else ref).split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    return "/".join(parts)


def fetch(path: str) -> bytes | None:
    url = BASE + path
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {path}: {exc}")
        return None


def main() -> int:
    # прокси мешает прямому доступу
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        import os
        os.environ.pop(var, None)

    queue = ["WebotsView.js"]
    # Текстуры пост-обработки Wren (GTAO/SMAA) НЕ видны из дерева JS/CSS-импортов —
    # webotsJS (Parser.js::DefaultUrl.wrenImagesUrl()) грузит их в рантайме через
    # fetch по вычисляемому пути `<wwi>/images/post_processing/<name>.png`, не через
    # `import`/`url()`. Без них в `web/wwi/images/post_processing/` браузерный вьюер
    # при --stream виснет на «Downloading assets: Texture 'gtao_noise_texture.png'»
    # (404 на стрим-сервере :1235, у которого этих файлов нет). Явные seed'ы ниже
    # заставляют fetch_wwi.py скачать их при повторном прогоне (например, на новой
    # машине или после очистки web/wwi/), не полагаясь на обход импортов.
    for _seed in (
        "images/post_processing/gtao_noise_texture.png",
        "images/post_processing/smaa_area_texture.png",
        "images/post_processing/smaa_search_texture.png",
    ):
        queue.append(_seed)
    done: set[str] = set()
    while queue:
        path = queue.pop()
        if path in done:
            continue
        done.add(path)
        data = fetch(path)
        if data is None:
            continue
        target = DST / path
        target.parent.mkdir(parents=True, exist_ok=True)

        refs: set[str] = set()
        if target.suffix in TEXT_EXT:
            text = data.decode("utf-8", errors="replace")
            base_dir = str(Path(path).parent).replace("\\", "/")
            base_dir = "" if base_dir == "." else base_dir
            for m in RE_IMPORT.finditer(text):
                refs.add(m.group(1) or m.group(2))
            for m in RE_ABS.finditer(text):
                refs.add("/" + m.group(1))
            if target.suffix == ".css":
                for m in RE_CSS_URL.finditer(text):
                    refs.add(m.group(1))
            if target.suffix == ".js":
                for m in RE_STR_REF.finditer(text):
                    refs.add(m.group(1))
            # CDN-ссылки -> относительные (страница подключает /wwi/...)
            text = text.replace(BASE, "/wwi/")
            target.write_bytes(text.encode())
        else:
            target.write_bytes(data)
        print(f"  + {path} ({len(data)} B)")

        for ref in refs:
            p = norm(base_dir if target.suffix in TEXT_EXT else "", ref)
            if p:
                queue.append(p)

    print(f"Готово: {len(done)} файлов в {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
