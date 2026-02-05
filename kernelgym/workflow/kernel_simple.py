"""Kernel simple workflow controller (single-task, kernel-only evaluation)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from kernelgym.common import ErrorCode
from kernelgym.schema import KernelEvaluationResult, KernelSimpleTask
from kernelgym.toolkit.validation import validate_code
from kernelgym.core.types import TaskSpec
from kernelgym.core.workflow import WorkflowController, WorkflowState
from kernelgym.core.scheduler import SchedulerAPI


def _resolve_entry_point(kernel_code: str, entry_point: Optional[str]) -> str:
    if entry_point and entry_point != "Model":
        return entry_point
    if "class ModelNew" in kernel_code:
        return "ModelNew"
    return entry_point or "ModelNew"


class KernelSimpleWorkflowController(WorkflowController):
    """Workflow controller for kernel-only evaluation."""

    async def validate_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        task = KernelSimpleTask.from_dict(input_data)
        entry_point = _resolve_entry_point(task.kernel_code, task.entry_point)
        valid, error = validate_code(task.kernel_code, entry_point)
        errors = []
        if not valid:
            errors.append(f"Kernel code validation failed: {error}")
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "task_id": task.task_id,
            "workflow": "kernel_simple",
            "kernel": {"valid": valid, "error": error, "entry_point": entry_point},
        }

    async def handle_request(self, input_data: Dict[str, Any], scheduler: SchedulerAPI) -> Dict[str, Any]:
        task = KernelSimpleTask.from_dict(input_data)
        task.entry_point = _resolve_entry_point(task.kernel_code, task.entry_point)
        state = WorkflowState({"base_task_id": task.task_id})

        validation = await self.validate_request(task.to_dict())
        if not validation["valid"]:
            message = validation["errors"][0] if validation["errors"] else "Validation failed"
            result = self._failed_result(task.task_id, message, ErrorCode.VALIDATION_ERROR)
            state.data["result"] = result
            return result

        payload = task.to_dict()
        payload["task_type"] = "kernel_simple"
        payload["toolkit"] = payload.get("toolkit") or "kernel_simple"
        payload["backend_adapter"] = payload.get("backend_adapter") or "kernelbench"

        task_spec = TaskSpec(
            kind="kernel_simple",
            payload=payload,
            resources=task.resources,
            metadata={"base_task_id": task.task_id},
        )
        task_id = await scheduler.submit(task_spec)
        result = await scheduler.wait(task_id)

        if not result:
            return self._failed_result(task.task_id, "kernel simple result missing", ErrorCode.RUNTIME_ERROR)

        if "error_message" in result and "compiled" not in result:
            return self._failed_result(
                task.task_id,
                result.get("error_message", "kernel simple task failed"),
                ErrorCode.RUNTIME_ERROR,
            )

        required = {"task_id", "compiled", "kernel_runtime", "metadata", "decoy_kernel"}
        if not required.issubset(result.keys()):
            missing = sorted(required - set(result.keys()))
            return self._failed_result(
                task.task_id,
                f"kernel simple result missing required fields: {missing}",
                ErrorCode.RUNTIME_ERROR,
            )

        result.setdefault("task_id", task.task_id)
        return result

    def _failed_result(self, task_id: str, message: str, error_code: ErrorCode) -> Dict[str, Any]:
        result = KernelEvaluationResult(
            task_id=task_id,
            base_task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            kernel_runtime=-1.0,
            metadata={"error": message},
            status="failed",
            error_message=message,
            error_code=error_code,
        )
        return result.to_dict()
