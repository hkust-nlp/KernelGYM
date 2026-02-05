"""Workflow registry and lookup helpers."""

from __future__ import annotations

from typing import Dict, Type

from kernelgym.core import Registry

from .kernelbench import KernelBenchWorkflowController
from .kernel_simple import KernelSimpleWorkflowController
from ..core.workflow import WorkflowController

_WORKFLOW_REGISTRY = Registry()
_WORKFLOW_REGISTRY.register("kernelbench", KernelBenchWorkflowController)
_WORKFLOW_REGISTRY.register("kernel_simple", KernelSimpleWorkflowController)


def get_workflow_controller(name: str) -> WorkflowController:
    key = (name or "kernelbench").strip().lower()
    return _WORKFLOW_REGISTRY.get(key)()


def register_workflow(name: str, controller_cls: Type[WorkflowController]) -> None:
    key = name.strip().lower()
    _WORKFLOW_REGISTRY.register(key, controller_cls)


def list_workflows() -> Dict[str, Type[WorkflowController]]:
    return _WORKFLOW_REGISTRY.items()
