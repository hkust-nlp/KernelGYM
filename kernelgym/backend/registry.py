"""Backend registry and lookup helpers."""

from __future__ import annotations

from typing import Dict, Type

from kernelgym.core import Registry

from .base import Backend
from .kernelbench import KernelBenchBackend

_BACKEND_REGISTRY = Registry()
_BACKEND_REGISTRY.register("kernelbench", KernelBenchBackend)


def get_backend(name: str) -> Backend:
    key = (name or "kernelbench").strip().lower()
    return _BACKEND_REGISTRY.get(key)()


def register_backend(name: str, backend_cls: Type[Backend]) -> None:
    key = name.strip().lower()
    _BACKEND_REGISTRY.register(key, backend_cls)


def list_backends() -> Dict[str, Type[Backend]]:
    return _BACKEND_REGISTRY.items()
