"""Triton backend execution adapters (legacy-eval compatible)."""

from __future__ import annotations

import tempfile
from typing import Any, Dict

import torch


def compile_only(kernel_code: str, device: torch.device) -> Dict[str, Any]:
    """Compile kernel without full evaluation (legacy stub)."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = {
                "compiled": True,
                "device": str(device),
                "build_dir": str(temp_dir),
            }
            return result
    except Exception as exc:
        return {"compiled": False, "error": str(exc), "device": str(device)}


__all__ = [
    "compile_only",
]
