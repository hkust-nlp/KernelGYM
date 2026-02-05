"""Triton-specific KernelBench backend implementation."""

from __future__ import annotations

import os
from typing import Any, Dict

from kernelgym.toolkit.kernelbench.loading import load_custom_model_with_tempfile
from kernelgym.toolkit.validation import validate_code
from kernelgym.backend.triton import compile_only as _compile_only

from .base import KernelBenchBackendBase


class KernelBenchTritonBackend(KernelBenchBackendBase):
    name = "kernelbench.triton"

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        build_dir = kwargs.get("build_dir")

        valid, error = validate_code(code, entry_point)
        if not valid:
            return {
                "compiled": False,
                "error": error,
                "device": str(device),
                "entry_point": entry_point,
                "backend": "triton",
                "build_dir": build_dir,
            }

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as exc:
            return {
                "compiled": False,
                "error": f"Syntax error in kernel code: {exc}",
                "device": str(device),
                "entry_point": entry_point,
                "backend": "triton",
                "build_dir": build_dir,
            }

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        artifact = _compile_only(code, device)
        artifact.update(
            {
                "code": code,
                "entry_point": entry_point,
                "backend": "triton",
                "build_dir": build_dir,
                "device": str(device),
            }
        )
        return artifact

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        build_dir = artifact.get("build_dir")
        context = kwargs.get("context") or {}
        tempfile_handle = None

        if not code:
            raise ValueError("KernelBenchTritonBackend.load requires kernel code in artifact")

        device = self._normalize_device(kwargs.get("device"))
        self._maybe_set_cuda_device(device)
        self._maybe_set_triton_env(device)

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        try:
            model_cls, tempfile_handle = load_custom_model_with_tempfile(
                code, entry_point=entry_point
            )
        except AttributeError as exc:
            raise ValueError(
                f"Failed to load model class '{entry_point}' from code"
            ) from exc

        if model_cls is None:
            raise ValueError(f"Failed to load model class '{entry_point}' from code")

        return {
            "model_cls": model_cls,
            "tempfile_handle": tempfile_handle,
            "context": context,
            "backend": "triton",
            "entry_point": entry_point,
            "device": device,
            "build_dir": build_dir,
        }
