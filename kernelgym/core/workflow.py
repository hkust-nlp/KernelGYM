"""Workflow controller abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .scheduler import SchedulerAPI


@dataclass
class WorkflowState:
    data: Dict[str, Any] = field(default_factory=dict)


class WorkflowController(ABC):
    @abstractmethod
    async def handle_request(self, input_data: Dict[str, Any], scheduler: SchedulerAPI) -> Dict[str, Any]:
        """Run the workflow and return the final response payload."""

    async def validate_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optional request validation hook."""
        return {"valid": True}

    async def on_task_finished(
        self,
        state: WorkflowState,
        task_id: str,
        result: Dict[str, Any],
        scheduler: SchedulerAPI,
    ) -> Optional[Dict[str, Any]]:
        """Optional hook for incremental decision making."""
        return None

    async def aggregate(self, state: WorkflowState) -> Dict[str, Any]:
        """Aggregate state into a final response payload."""
        return dict(state.data)
