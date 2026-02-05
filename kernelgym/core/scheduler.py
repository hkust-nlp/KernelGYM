"""Scheduler API for task submission and tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import TaskSpec


class SchedulerAPI(ABC):
    @abstractmethod
    async def submit(self, task: TaskSpec) -> str:
        """Submit a task and return its task_id."""

    @abstractmethod
    async def wait(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for a task result and return the raw result payload."""

    @abstractmethod
    async def get_status(self, task_id: str) -> Dict[str, Any]:
        """Return status metadata for a task."""

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Cancel a task if possible."""
