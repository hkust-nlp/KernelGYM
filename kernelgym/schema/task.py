"""Shared task models for KernelBench workflows."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class EvaluationTask:
    task_id: str
    reference_code: str
    kernel_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    reference_backend: Optional[str] = None
    device_preference: Optional[str] = None
    force_refresh: bool = False
    uuid: Optional[str] = None
    use_reference_cache: bool = False
    is_valid: bool = False
    enable_profiling: Optional[bool] = None
    enable_triton_detection: Optional[bool] = None
    measure_performance: Optional[bool] = None
    run_correctness: Optional[bool] = None
    run_triton_detection: Optional[bool] = None
    run_performance: Optional[bool] = None
    resources: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class ReferenceTimingTask:
    task_id: str
    base_task_id: str
    reference_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    reference_backend: Optional[str] = None
    device_preference: Optional[str] = None
    resources: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTimingTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class KernelEvaluationTask:
    task_id: str
    base_task_id: str
    reference_code: str
    kernel_code: str
    toolkit: str = "kernelbench"
    backend_adapter: str = "kernelbench"
    backend: str = "triton"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    timeout: int = 300
    device: str = "cuda:0"
    priority: str = "normal"
    entry_point: str = "Model"
    device_preference: Optional[str] = None
    enable_profiling: Optional[bool] = None
    enable_triton_detection: Optional[bool] = None
    measure_performance: Optional[bool] = None
    run_correctness: Optional[bool] = None
    run_triton_detection: Optional[bool] = None
    run_performance: Optional[bool] = None
    resources: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelEvaluationTask":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
