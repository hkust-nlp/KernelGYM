"""Toolkit abstraction (evaluation logic)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..backend import Backend


class Toolkit(ABC):
    name: str = "unknown"

    @abstractmethod
    def evaluate(self, task: Dict[str, Any], backend: Backend, **kwargs: Any) -> Dict[str, Any]:
        """Run evaluation logic against a backend."""
