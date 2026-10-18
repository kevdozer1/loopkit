"""Adapter registry: models.yaml ``adapter`` name to class.

Implemented: ``openrouter`` (the default route), ``elorian``, ``jev`` and ``fake``.
The optional direct adapters that ``config.ADAPTERS`` also lists are not built; an
entry that uses one raises ModelNotConfigured (route it through OpenRouter instead).
"""

from __future__ import annotations

import threading

from ..errors import ModelNotConfigured
from .base import Adapter, AdapterRequest, AdapterResult, reasoning_level
from .elorian import ElorianAdapter
from .fake import FakeAdapter
from .jev import JevAdapter
from .openrouter import OpenRouterAdapter

_REGISTRY: dict[str, type[Adapter]] = {}
_INSTANCES: dict[str, Adapter] = {}
_LOCK = threading.Lock()


def register_adapter(name: str, cls: type[Adapter]) -> None:
    with _LOCK:
        key = name.strip().lower()
        _REGISTRY[key] = cls
        _INSTANCES.pop(key, None)


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(name: str) -> Adapter:
    key = (name or "").strip().lower()
    with _LOCK:
        if key not in _REGISTRY:
            raise ModelNotConfigured(
                f"adapter {name!r} is not implemented in loopkit.router (available: "
                f"{', '.join(sorted(_REGISTRY))}). Route the model through openrouter in config/models.yaml.")
        if key not in _INSTANCES:
            _INSTANCES[key] = _REGISTRY[key]()
        return _INSTANCES[key]


register_adapter("openrouter", OpenRouterAdapter)
register_adapter("elorian", ElorianAdapter)
register_adapter("jev", JevAdapter)
register_adapter("fake", FakeAdapter)

__all__ = [
    "Adapter", "AdapterRequest", "AdapterResult", "ElorianAdapter", "FakeAdapter", "JevAdapter",
    "OpenRouterAdapter", "available_adapters", "get_adapter", "reasoning_level", "register_adapter",
]
