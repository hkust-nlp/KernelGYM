"""KernelBench workflow controller (server-side orchestration)."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import json
from datetime import datetime, timezone

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from .kernelbench_types import (
    EvaluationTask,
    ReferenceTimingTask,
    ReferenceTimingResult,
    KernelEvaluationResult,
    EvaluationResult,
)
from .kernelbench_helpers import (
    _combine_results,
    _create_paired_tasks,
    _get_cached_reference_runtime,
    _validate_code,
)

from ..core.types import TaskSpec
from ..core.workflow import WorkflowController, WorkflowState
from ..core.scheduler import SchedulerAPI


class KernelBenchWorkflowController(WorkflowController):
    """Main controller for KernelBench evaluation workflow."""

    async def validate_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        eval_task = EvaluationTask.from_dict(input_data)
        validation = self._validate_inputs(eval_task)
        validation["task_id"] = eval_task.task_id
        validation["workflow"] = "kernelbench"
        return validation

    async def handle_request(self, input_data: Dict[str, Any], scheduler: SchedulerAPI) -> Dict[str, Any]:
        eval_task = EvaluationTask.from_dict(input_data)
        state = WorkflowState({"base_task_id": eval_task.task_id})

        if eval_task.reference_backend:
            print(
                f"[Workflow] task={eval_task.task_id} reference_backend={eval_task.reference_backend}"
            )

        validation = self._validate_inputs(eval_task)
        if not validation["valid"]:
            message = validation["errors"][0] if validation["errors"] else "Validation failed"
            result = self._validation_failed_result(eval_task.task_id, message)
            self._persist_result(eval_task, result)
            return result

        ref_task, kernel_task = _create_paired_tasks(eval_task)

        kernel_payload = kernel_task.to_dict()
        kernel_payload["task_type"] = "kernel_evaluation"
        kernel_payload["toolkit"] = kernel_payload.get("toolkit", "kernelbench")
        kernel_payload["backend_adapter"] = kernel_payload.get("backend_adapter", "kernelbench")
        run_correctness = eval_task.run_correctness
        if run_correctness is None:
            run_correctness = True
        run_triton_detection = eval_task.run_triton_detection
        if run_triton_detection is None:
            run_triton_detection = eval_task.enable_triton_detection
        if run_triton_detection is None:
            run_triton_detection = eval_task.backend == "triton"
        run_performance = eval_task.run_performance
        if run_performance is None:
            run_performance = eval_task.measure_performance
        if run_performance is None:
            run_performance = True
        kernel_payload["run_correctness"] = run_correctness
        kernel_payload["run_triton_detection"] = run_triton_detection
        kernel_payload["run_performance"] = run_performance
        kernel_payload["enable_triton_detection"] = run_triton_detection
        kernel_payload["measure_performance"] = run_performance
        enable_profiling = eval_task.enable_profiling
        if enable_profiling is None:
            enable_profiling = settings.enable_profiling
        kernel_payload["enable_profiling"] = enable_profiling
        kernel_task_spec = TaskSpec(
            kind="kernelbench.kernel",
            payload=kernel_payload,
            resources=eval_task.resources,
            metadata={"base_task_id": eval_task.task_id},
        )
        kernel_task_id = await scheduler.submit(kernel_task_spec)
        kernel_result_dict = await scheduler.wait(kernel_task_id)

        if not kernel_result_dict:
            result = self._failed_result(eval_task.task_id, "kernel result missing")
            self._persist_result(eval_task, result)
            return result
        if "error_message" in kernel_result_dict and "compiled" not in kernel_result_dict:
            result = self._failed_result(
                eval_task.task_id,
                kernel_result_dict.get("error_message", "kernel task failed"),
            )
            self._persist_result(eval_task, result)
            return result

        required_kernel_fields = {
            "task_id",
            "base_task_id",
            "compiled",
            "correctness",
            "decoy_kernel",
            "kernel_runtime",
            "metadata",
        }
        if not required_kernel_fields.issubset(kernel_result_dict.keys()):
            missing = sorted(required_kernel_fields - set(kernel_result_dict.keys()))
            result = self._failed_result(
                eval_task.task_id,
                f"kernel result missing required fields: {missing}",
            )
            self._persist_result(eval_task, result)
            return result

        kernel_result = KernelEvaluationResult.from_dict(kernel_result_dict)
        state.data["kernel_result"] = kernel_result.to_dict()

        if not (kernel_result.compiled and kernel_result.correctness):
            result = self._kernel_only_result(eval_task, kernel_result)
            self._persist_result(eval_task, result)
            return result

        ref_result: Optional[ReferenceTimingResult] = None
        if ref_task is None:
            cached_runtime = _get_cached_reference_runtime(
                eval_task.uuid, eval_task.reference_code, eval_task.is_valid
            )
            if cached_runtime is not None:
                ref_result = self._cached_reference_result(eval_task, cached_runtime)
            else:
                    ref_task = ReferenceTimingTask(
                        task_id=f"{eval_task.task_id}_ref",
                        base_task_id=eval_task.task_id,
                        reference_code=eval_task.reference_code,
                        backend=eval_task.backend,
                        num_perf_trials=eval_task.num_perf_trials,
                        timeout=eval_task.timeout,
                        device=eval_task.device,
                        priority=eval_task.priority,
                        entry_point=eval_task.entry_point,
                        reference_backend=eval_task.reference_backend,
                        device_preference=eval_task.device_preference,
                    )

        if ref_result is None and ref_task is not None:
            ref_payload = ref_task.to_dict()
            ref_payload["task_type"] = "reference_timing"
            ref_payload["toolkit"] = ref_payload.get("toolkit", "kernelbench")
            ref_payload["backend_adapter"] = ref_payload.get("backend_adapter", "kernelbench")
            ref_task_spec = TaskSpec(
                kind="kernelbench.ref",
                payload=ref_payload,
                resources=eval_task.resources,
                metadata={"base_task_id": eval_task.task_id},
            )
            ref_task_id = await scheduler.submit(ref_task_spec)
            ref_result_dict = await scheduler.wait(ref_task_id)
            if ref_result_dict:
                if "error_message" in ref_result_dict and "reference_runtime" not in ref_result_dict:
                    result = self._kernel_only_result(eval_task, kernel_result)
                    self._persist_result(eval_task, result)
                    return result
                ref_result = ReferenceTimingResult.from_dict(ref_result_dict)

        if ref_result is None:
            result = self._kernel_only_result(eval_task, kernel_result)
            self._persist_result(eval_task, result)
            return result

        combined = _combine_results(ref_result, kernel_result)
        result = combined.to_dict()
        self._persist_result(eval_task, result)
        return result

    def _cached_reference_result(self, eval_task: EvaluationTask, runtime: float) -> ReferenceTimingResult:
        return ReferenceTimingResult(
            task_id=f"{eval_task.task_id}_ref",
            base_task_id=eval_task.task_id,
            reference_runtime=runtime,
            metadata={
                "cached": True,
                "uuid": eval_task.uuid,
                "device": "cached",
                "backend": eval_task.backend,
                "cache_type": "validation" if eval_task.is_valid else "regular",
            },
            status="completed",
        )

    def _kernel_only_result(self, eval_task: EvaluationTask, kernel_result: KernelEvaluationResult) -> Dict[str, Any]:
        metadata = dict(kernel_result.metadata or {})
        metadata["kernel_task_id"] = kernel_result.task_id
        result = EvaluationResult(
            task_id=eval_task.task_id,
            compiled=kernel_result.compiled,
            correctness=kernel_result.correctness,
            decoy_kernel=kernel_result.decoy_kernel,
            reference_runtime=-1.0,
            kernel_runtime=kernel_result.kernel_runtime,
            speedup=0.0,
            metadata=metadata,
            status=kernel_result.status,
            error_message=kernel_result.error_message,
            error_code=kernel_result.error_code,
        )
        return result.to_dict()

    def _failed_result(self, task_id: str, message: str) -> Dict[str, Any]:
        result = EvaluationResult(
            task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=-1.0,
            kernel_runtime=-1.0,
            speedup=0.0,
            metadata={"error": message},
            status="failed",
            error_message=message,
        )
        return result.to_dict()

    def _validation_failed_result(self, task_id: str, message: str) -> Dict[str, Any]:
        result = EvaluationResult(
            task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=-1.0,
            kernel_runtime=-1.0,
            speedup=0.0,
            metadata={"error": message},
            status="failed",
            error_message=message,
            error_code=ErrorCode.VALIDATION_ERROR.value,
        )
        return result.to_dict()

    def _validate_inputs(self, eval_task: EvaluationTask) -> Dict[str, Any]:
        errors = []

        resources = eval_task.resources or {}
        if resources:
            gpus = resources.get("gpus")
            if gpus is not None:
                try:
                    gpus_int = int(gpus)
                    if gpus_int < 1:
                        errors.append("resources.gpus must be >= 1")
                except (TypeError, ValueError):
                    errors.append("resources.gpus must be an integer")

        if eval_task.use_reference_cache and not eval_task.uuid:
            errors.append("UUID is required when use_reference_cache is True")

        ref_valid, ref_error = _validate_code(eval_task.reference_code, eval_task.entry_point)
        if not ref_valid:
            errors.append(f"Reference code validation failed: {ref_error}")

        kernel_entry_point = f"{eval_task.entry_point}New"
        kernel_valid, kernel_error = _validate_code(eval_task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            errors.append(f"Kernel code validation failed: {kernel_error}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "reference": {"valid": ref_valid, "error": ref_error, "entry_point": eval_task.entry_point},
            "kernel": {"valid": kernel_valid, "error": kernel_error, "entry_point": kernel_entry_point},
            "cache": {"use_reference_cache": eval_task.use_reference_cache, "uuid": eval_task.uuid},
            "resources": resources,
        }

    def _persist_result(self, eval_task: EvaluationTask, result: Dict[str, Any]) -> None:
        if not settings.save_eval_results:
            return
        try:
            path = Path(settings.eval_results_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": eval_task.task_id,
                "base_task_id": eval_task.task_id,
                "toolkit": eval_task.toolkit,
                "backend": eval_task.backend,
                "result": result,
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:
            return
