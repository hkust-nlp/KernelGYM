"""Kernel simple toolkit implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.schema import KernelEvaluationResult, KernelSimpleTask
from kernelgym.toolkit.kernelbench.exec_types import set_seed, get_error_name
from kernelgym.toolkit.kernelbench.timing import get_timing_stats, time_execution_with_cuda_event
from kernelgym.toolkit.validation import validate_code
from kernelgym.toolkit.base import Toolkit


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _normalize_case(case: Any, idx: int) -> Dict[str, Any]:
    if isinstance(case, dict):
        inputs = case.get("inputs", case.get("input"))
        outputs = case.get("outputs", case.get("output"))
        return {
            "name": case.get("name", f"case_{idx}"),
            "inputs": inputs,
            "outputs": outputs,
            "rtol": case.get("rtol"),
            "atol": case.get("atol"),
        }
    return {
        "name": f"case_{idx}",
        "inputs": case,
        "outputs": None,
        "rtol": None,
        "atol": None,
    }


def _normalize_cases(raw_cases: Any) -> List[Dict[str, Any]]:
    if raw_cases is None:
        return []
    if isinstance(raw_cases, dict):
        raw_cases = [raw_cases]
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a list of case objects")
    return [_normalize_case(case, idx) for idx, case in enumerate(raw_cases)]


def _load_cases_from_code(code: str) -> Tuple[List[Dict[str, Any]], Any]:
    context: Dict[str, Any] = {}
    compile(code, "<string>", "exec")
    exec(code, context)
    get_cases = context.get("get_cases")
    get_inputs = context.get("get_inputs")
    get_init_inputs = context.get("get_init_inputs")

    init_inputs = get_init_inputs() if callable(get_init_inputs) else []
    if callable(get_cases):
        raw_cases = get_cases()
        return _normalize_cases(raw_cases), init_inputs
    if callable(get_inputs):
        return _normalize_cases([{"inputs": get_inputs()}]), init_inputs
    return [], init_inputs


def _load_init_inputs_from_code(code: str) -> Any:
    context: Dict[str, Any] = {}
    compile(code, "<string>", "exec")
    exec(code, context)
    get_init_inputs = context.get("get_init_inputs")
    return get_init_inputs() if callable(get_init_inputs) else []


def _run_model(model: Any, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def _compare_tensors(expected: torch.Tensor, actual: torch.Tensor, rtol: float, atol: float) -> bool:
    try:
        return torch.allclose(actual, expected, rtol=rtol, atol=atol)
    except Exception:
        return False


def _compare_outputs(expected: Any, actual: Any, rtol: float, atol: float) -> bool:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return _compare_tensors(expected, actual, rtol, atol)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(
            _compare_outputs(exp, act, rtol, atol) for exp, act in zip(expected, actual)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return False
        return all(
            _compare_outputs(expected[k], actual[k], rtol, atol) for k in expected.keys()
        )
    return expected == actual


class KernelSimpleToolkit(Toolkit):
    """Kernel-only evaluation toolkit (cases + profiling)."""

    name = "kernel_simple"

    def evaluate(self, task: Dict[str, Any], backend=None, **kwargs: Any) -> Dict[str, Any]:
        task_obj = KernelSimpleTask.from_dict(task)
        device = torch.device(task_obj.device)

        if not torch.cuda.is_available() or device.type != "cuda":
            return KernelEvaluationResult(
                task_id=task_obj.task_id,
                base_task_id=task_obj.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata={"error": "CUDA is required for kernel_simple"},
                status="failed",
                error_message="CUDA is required for kernel_simple",
                error_code=ErrorCode.RUNTIME_ERROR,
            ).to_dict()

        entry_point = task_obj.entry_point or "ModelNew"
        valid, error = validate_code(task_obj.kernel_code, entry_point)
        if not valid:
            return KernelEvaluationResult(
                task_id=task_obj.task_id,
                base_task_id=task_obj.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata={"validation_error": error},
                status="failed",
                error_message=f"Kernel code validation failed: {error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).to_dict()

        set_seed(42)

        cases: List[Dict[str, Any]] = []
        init_inputs: Any = []
        cases_source = "inline"
        try:
            if task_obj.cases is not None:
                cases = _normalize_cases(task_obj.cases)
                init_inputs = _load_init_inputs_from_code(task_obj.kernel_code)
            elif task_obj.cases_code:
                cases, init_inputs = _load_cases_from_code(task_obj.cases_code)
                cases_source = "cases_code"
            else:
                cases, init_inputs = _load_cases_from_code(task_obj.kernel_code)
                cases_source = "kernel_code"
        except Exception as e:
            return KernelEvaluationResult(
                task_id=task_obj.task_id,
                base_task_id=task_obj.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Failed to load cases: {e}",
                error_code=ErrorCode.RUNTIME_ERROR,
            ).to_dict()

        if not cases:
            return KernelEvaluationResult(
                task_id=task_obj.task_id,
                base_task_id=task_obj.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata={"error": "No cases or inputs provided"},
                status="failed",
                error_message="No cases or inputs provided",
                error_code=ErrorCode.VALIDATION_ERROR,
            ).to_dict()

        has_expected = any(case.get("outputs") is not None for case in cases)
        run_correctness = task_obj.run_correctness
        if run_correctness is None:
            run_correctness = has_expected
        run_performance = task_obj.run_performance
        if run_performance is None:
            run_performance = True

        enable_profiling = task_obj.enable_profiling
        if enable_profiling is None:
            enable_profiling = settings.enable_profiling

        metadata: Dict[str, Any] = {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "backend": task_obj.backend,
            "cases_source": cases_source,
            "num_cases": len(cases),
        }

        artifact = backend.compile(
            task_obj.kernel_code,
            device=device,
            backend=task_obj.backend,
            entry_point=entry_point,
        )
        if not artifact.get("compiled"):
            error_msg = artifact.get("error", "Unknown compile error")
            return KernelEvaluationResult(
                task_id=task_obj.task_id,
                base_task_id=task_obj.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata={"compilation_error": error_msg},
                status="failed",
                error_message=f"Kernel compilation failed: {error_msg}",
                error_code=ErrorCode.COMPILATION_ERROR,
            ).to_dict()

        handle = None
        session = None
        try:
            handle = backend.load(artifact, device=device, context={})
            session = backend.open_session(handle, device=device)
            model = session.create_model(init_inputs, no_grad=True, synchronize=False)

            correctness: Optional[bool] = None
            if run_correctness and has_expected:
                failed_cases: List[str] = []
                with torch.no_grad():
                    for case in cases:
                        expected = case.get("outputs")
                        if expected is None:
                            continue
                        inputs = _move_to_device(case.get("inputs"), device)
                        expected = _move_to_device(expected, device)
                        actual = _run_model(model, inputs)
                        rtol = case.get("rtol", 1e-4)
                        atol = case.get("atol", 1e-5)
                        if not _compare_outputs(expected, actual, rtol, atol):
                            failed_cases.append(case.get("name", "unknown"))
                correctness = len(failed_cases) == 0
                metadata["correctness_failed_cases"] = failed_cases
            elif run_correctness and not has_expected:
                metadata["correctness_skipped"] = "no_expected_outputs"
                correctness = None
            else:
                metadata["correctness_skipped"] = True
                correctness = None

            kernel_runtime = -1.0
            if run_performance:
                perf_inputs = cases[0].get("inputs")
                if perf_inputs is None:
                    raise ValueError("Performance inputs are missing in first case")
                perf_inputs = _move_to_device(perf_inputs, device)
                if isinstance(perf_inputs, dict):
                    kernel_fn = lambda: _run_model(model, perf_inputs)
                    args: Tuple[Any, ...] = ()
                else:
                    kernel_fn = model
                    if isinstance(perf_inputs, (list, tuple)):
                        args = tuple(perf_inputs)
                    else:
                        args = (perf_inputs,)

                elapsed_times, profiling_metrics = time_execution_with_cuda_event(
                    kernel_fn,
                    *args,
                    num_warmup=task_obj.num_warmup,
                    num_trials=task_obj.num_perf_trials,
                    verbose=False,
                    device=device,
                    enable_profiling=bool(enable_profiling),
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)
                metadata["runtime_stats"] = runtime_stats
                kernel_runtime = runtime_stats["mean"]
                if enable_profiling and profiling_metrics:
                    metadata["profiling"] = profiling_metrics
            else:
                metadata["performance_skipped"] = True

            return KernelEvaluationResult(
                task_id=task_obj.task_id,
                base_task_id=task_obj.task_id,
                compiled=True,
                correctness=correctness,
                decoy_kernel=False,
                kernel_runtime=kernel_runtime,
                metadata=metadata,
                status="completed",
            ).to_dict()

        except Exception as e:
            error_code = ErrorCode.RUNTIME_ERROR
            return KernelEvaluationResult(
                task_id=task_obj.task_id,
                base_task_id=task_obj.task_id,
                compiled=True,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=-1.0,
                metadata={"error": str(e), "error_name": get_error_name(e)},
                status="failed",
                error_message=f"Kernel simple evaluation failed: {e}",
                error_code=error_code,
            ).to_dict()
        finally:
            if session is not None:
                session.close()
            elif handle is not None:
                backend.cleanup(handle)
