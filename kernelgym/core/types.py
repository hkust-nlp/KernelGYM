"""Core data models for KernelGym refactor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Artifact:
    name: str
    uri: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    task_id: str
    status: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metrics: List[Metric] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class TaskSpec:
    """Minimal task spec for scheduler submission."""

    kind: str
    payload: Dict[str, Any]
    resources: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGroup:
    """Container for multiple tasks with optional dependencies."""

    tasks: List[TaskSpec]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
