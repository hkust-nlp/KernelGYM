"""Shared execution types for KernelBench (toolkit layer)."""

from __future__ import annotations

import torch
from pydantic import BaseModel


def get_error_name(e: Exception) -> str:
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class KernelExecResult(BaseModel):
    """Single Kernel Execution result."""

    compiled: bool = False
    correctness: bool = False
    decoy_kernel: bool = False
    metadata: dict = {}
    runtime: float = -1.0
    runtime_stats: dict = {}
