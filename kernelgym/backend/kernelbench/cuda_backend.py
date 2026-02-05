"""CUDA/PyTorch backend implementation for KernelBench."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

from kernelgym.toolkit.kernelbench.loading import load_custom_model
from kernelgym.toolkit.kernelbench.compile import build_compile_cache
from kernelgym.toolkit.validation import validate_code

from .base import KernelBenchBackendBase


class KernelBenchCudaBackend(KernelBenchBackendBase):
    name = "kernelbench.cuda"

    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        device = self._normalize_device(kwargs.get("device"))
        entry_point = kwargs.get("entry_point", "ModelNew")
        backend = kwargs.get("backend", "cuda")
        build_dir = kwargs.get("build_dir")

        valid, error = validate_code(code, entry_point)
        if not valid:
            return {
                "compiled": False,
                "error": error,
                "device": str(device),
                "entry_point": entry_point,
                "backend": backend,
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
                "backend": backend,
                "build_dir": build_dir,
            }

        if build_dir is None:
            build_dir = tempfile.mkdtemp(prefix="kernelgym_cuda_")

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        cache_result = build_compile_cache(code, build_dir, verbose=False)
        artifact = {
            "compiled": cache_result["compiled"],
            "error": cache_result.get("error"),
            "stdout": cache_result.get("stdout"),
            "stderr": cache_result.get("stderr"),
            "device": str(device),
            "entry_point": entry_point,
            "backend": backend,
            "build_dir": build_dir,
            "code": code,
        }
        return artifact

    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        code = artifact.get("code")
        entry_point = artifact.get("entry_point", "ModelNew")
        build_dir = artifact.get("build_dir")
        backend = artifact.get("backend", "cuda")
        context = kwargs.get("context") or {}

        if not code:
            raise ValueError("KernelBenchCudaBackend.load requires kernel code in artifact")

        device = self._normalize_device(kwargs.get("device"))
        self._maybe_set_cuda_device(device)

        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        model_cls = load_custom_model(code, context, build_dir)

        if model_cls is None:
            raise ValueError(f"Failed to load model class '{entry_point}' from code")

        return {
            "model_cls": model_cls,
            "tempfile_handle": None,
            "context": context,
            "backend": backend,
            "entry_point": entry_point,
            "device": device,
            "build_dir": build_dir,
        }
