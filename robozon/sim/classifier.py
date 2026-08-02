"""Определение категории товара.

Этап 1: категория известна по типу объекта (KnownTypeClassifier).
Этап CV: сюда добавляется классификатор по изображению с камеры,
реализующий тот же интерфейс Classifier — исполнительная часть
и маршрутизация при замене не меняются.

Категории (постановка задачи, взаимоисключающие):
  ok       — подходит для сортировки            -> зона B
  oversize — не подходит по габаритам           -> зона C
  round    — круг в сечении, нужна доупаковка   -> зона D
"""
from abc import ABC, abstractmethod

CATEGORIES = ("ok", "oversize", "round")


class Classifier(ABC):
    """Интерфейс: по описанию объекта вернуть одну из категорий CATEGORIES."""

    @abstractmethod
    def classify(self, obj: dict) -> str:
        """obj — словарь с известными признаками объекта (type, dims, ...)."""


class KnownTypeClassifier(Classifier):
    """Категория задана конфигом для известного типа (без CV)."""

    def __init__(self, objects_cfg: dict):
        self._by_type = {name: cfg["category"] for name, cfg in objects_cfg.items()}

    def classify(self, obj: dict) -> str:
        category = self._by_type[obj["type"]]
        if category not in CATEGORIES:
            raise ValueError(f"Неизвестная категория '{category}' у типа '{obj['type']}'")
        return category
