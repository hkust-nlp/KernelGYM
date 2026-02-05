"""KernelGym toolkit package.

Keep this module import-light to avoid circular imports during schema/model
imports. Public symbols are provided via lazy attribute access.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Toolkit", "get_toolkit", "list_toolkits", "register_toolkit"]


def __getattr__(name: str) -> Any:
    if name == "Toolkit":
        from .base import Toolkit  # local import to avoid import-time side effects

        return Toolkit
    if name in ("get_toolkit", "list_toolkits", "register_toolkit"):
        from .registry import get_toolkit, list_toolkits, register_toolkit

        return {
            "get_toolkit": get_toolkit,
            "list_toolkits": list_toolkits,
            "register_toolkit": register_toolkit,
        }[name]
    raise AttributeError(name)
