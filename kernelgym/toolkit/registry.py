"""Toolkit registry and lookup helpers."""

from __future__ import annotations

from typing import Dict, Type

from kernelgym.core import Registry

from .base import Toolkit

_TOOLKIT_REGISTRY = Registry()


def _ensure_default_toolkits() -> None:
    items = _TOOLKIT_REGISTRY.items()
    if "kernelbench" not in items:
        from .kernelbench.toolkit import KernelBenchToolkit
        _TOOLKIT_REGISTRY.register("kernelbench", KernelBenchToolkit)
    if "kernel_simple" not in items:
        from .kernel_simple.toolkit import KernelSimpleToolkit
        _TOOLKIT_REGISTRY.register("kernel_simple", KernelSimpleToolkit)


def get_toolkit(name: str) -> Toolkit:
    key = (name or "kernelbench").strip().lower()
    _ensure_default_toolkits()
    return _TOOLKIT_REGISTRY.get(key)()


def register_toolkit(name: str, toolkit_cls: Type[Toolkit]) -> None:
    key = name.strip().lower()
    _TOOLKIT_REGISTRY.register(key, toolkit_cls)


def list_toolkits() -> Dict[str, Type[Toolkit]]:
    _ensure_default_toolkits()
    return _TOOLKIT_REGISTRY.items()
