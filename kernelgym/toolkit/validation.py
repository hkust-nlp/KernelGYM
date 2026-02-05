"""Validation helpers for KernelBench toolkit."""

from __future__ import annotations

from typing import Optional, Tuple

from kernelgym.common import ErrorCode


def validate_code(code: str, entry_point: str = "Model") -> Tuple[bool, str]:
    """Basic validation of PyTorch code."""
    try:
        if not code:
            return False, "Code is required"
        if f"class {entry_point}" not in code:
            return False, f"Code must contain a '{entry_point}' class"
        return True, ""
    except Exception as exc:
        return False, f"Code validation error: {exc}"


def early_kernel_validation(
    kernel_code: str,
    backend: str = "triton",
    entry_point: str = "Model",
) -> Tuple[bool, str, Optional[ErrorCode]]:
    """Perform early kernel code validation without GPU resources."""
    try:
        kernel_entry_point = f"{entry_point}New"
        is_valid, error_msg = validate_code(kernel_code, kernel_entry_point)
        if not is_valid:
            return False, error_msg, ErrorCode.VALIDATION_ERROR

        try:
            compile(kernel_code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error in kernel code: {str(e)}", ErrorCode.SYNTAX_ERROR

        if backend == "triton":
            required_imports = ["import triton", "from triton import"]
            if not any(imp in kernel_code for imp in required_imports):
                return False, "Kernel code must import triton for triton backend", ErrorCode.IMPORT_ERROR
        elif backend == "cuda":
            cuda_indicators = [
                "torch.cuda",
                "cuda_kernel",
                "@cuda.jit",
                "from numba import cuda",
            ]
            if not any(indicator in kernel_code for indicator in cuda_indicators):
                return False, "Kernel code must contain CUDA kernel code for cuda backend", ErrorCode.IMPORT_ERROR

        kernel_patterns = [
            "@triton.jit",
            "def.*kernel.*\\(",
            "torch\\.cuda",
            "\\.cuda\\(",
        ]

        import re

        has_kernel_pattern = any(re.search(pattern, kernel_code) for pattern in kernel_patterns)
        if not has_kernel_pattern and backend == "triton":
            return True, "", None

        try:
            test_code = f"""
            import torch
            import torch.nn as nn
            {kernel_code}

            try:
                model = {kernel_entry_point}()
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate {kernel_entry_point}: {{e}}")
            """
            compile(test_code, "<test>", "exec")
        except Exception as e:
            error_str = str(e)
            if "Failed to instantiate" in error_str:
                return False, error_str, ErrorCode.INSTANTIATION_ERROR

        return True, "", None

    except Exception as e:
        return False, f"Early validation error: {str(e)}", ErrorCode.VALIDATION_ERROR
