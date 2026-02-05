"""KernelBench workflow helpers (server-side orchestration utilities)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .kernelbench_types import (
    EvaluationTask,
    KernelEvaluationResult,
    KernelEvaluationTask,
    EvaluationResult,
    ReferenceTimingResult,
    ReferenceTimingTask,
)

_reference_cache: Any = None


def set_reference_cache(cache: Any) -> None:
    """Register a reference cache provider used by workflow orchestration."""
    global _reference_cache
    _reference_cache = cache


def _get_cached_reference_runtime(
    uuid: Optional[str], reference_code: str, is_valid: bool
) -> Optional[float]:
    if _reference_cache is None:
        return None
    return _reference_cache.get(uuid, reference_code, is_valid)


def _validate_code(code: str, entry_point: str = "Model") -> Tuple[bool, str]:
    try:
        if not code:
            return False, "Code is required"
        if f"class {entry_point}" not in code:
            return False, f"Code must contain a '{entry_point}' class"
        return True, ""
    except Exception as exc:
        return False, f"Code validation error: {exc}"


def _create_paired_tasks(
    task: EvaluationTask,
) -> Tuple[Optional[ReferenceTimingTask], KernelEvaluationTask]:
    ref_device = task.device_preference or task.device
    kernel_device = task.device

    reference_task: Optional[ReferenceTimingTask] = None
    if task.use_reference_cache and task.uuid:
        cached_runtime = _get_cached_reference_runtime(
            task.uuid, task.reference_code, task.is_valid
        )
        if cached_runtime is None:
            reference_task = ReferenceTimingTask(
                task_id=f"{task.task_id}_ref",
                base_task_id=task.task_id,
                reference_code=task.reference_code,
                toolkit=task.toolkit,
                backend_adapter=task.backend_adapter,
                backend=task.backend,
                num_perf_trials=task.num_perf_trials,
                timeout=task.timeout,
                device=ref_device,
                priority=task.priority,
                entry_point=task.entry_point,
                reference_backend=task.reference_backend,
                device_preference=task.device_preference,
                resources=task.resources,
            )
    else:
        reference_task = ReferenceTimingTask(
            task_id=f"{task.task_id}_ref",
            base_task_id=task.task_id,
            reference_code=task.reference_code,
            toolkit=task.toolkit,
            backend_adapter=task.backend_adapter,
            backend=task.backend,
            num_perf_trials=task.num_perf_trials,
            timeout=task.timeout,
            device=ref_device,
            priority=task.priority,
            entry_point=task.entry_point,
            reference_backend=task.reference_backend,
            device_preference=task.device_preference,
            resources=task.resources,
        )

    kernel_task = KernelEvaluationTask(
        task_id=f"{task.task_id}_kernel",
        base_task_id=task.task_id,
        reference_code=task.reference_code,
        kernel_code=task.kernel_code,
        toolkit=task.toolkit,
        backend_adapter=task.backend_adapter,
        backend=task.backend,
        num_correct_trials=task.num_correct_trials,
        num_perf_trials=task.num_perf_trials,
        timeout=task.timeout,
        device=kernel_device,
        priority=task.priority,
        entry_point=task.entry_point,
        device_preference=task.device_preference,
        enable_profiling=task.enable_profiling,
        enable_triton_detection=task.enable_triton_detection,
        measure_performance=task.measure_performance,
        run_correctness=task.run_correctness,
        run_triton_detection=task.run_triton_detection,
        run_performance=task.run_performance,
        resources=task.resources,
    )

    return reference_task, kernel_task


def _combine_results(
    reference_result: ReferenceTimingResult,
    kernel_result: KernelEvaluationResult,
) -> EvaluationResult:
    if reference_result.base_task_id != kernel_result.base_task_id:
        raise ValueError(
            f"Task ID mismatch: {reference_result.base_task_id} != {kernel_result.base_task_id}"
        )
    return EvaluationResult.from_paired_results(
        reference_result.base_task_id, reference_result, kernel_result
    )
