"""KernelBench compile helpers (CUDA cache build)."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Dict

from kernelgym.toolkit.kernelbench.loading import load_custom_model


def build_compile_cache(custom_model_src: str, build_dir: str | None, verbose: bool = False) -> Dict[str, Any]:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    context: Dict[str, Any] = {}

    if verbose:
        print("[Compilation] Pre-compile custom CUDA binaries")

    try:
        if build_dir:
            custom_model_src = (
                "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
            ) + custom_model_src

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            load_custom_model(custom_model_src, context, build_dir)

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
        return {
            "compiled": True,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "error": None,
        }
    except Exception as exc:
        if verbose:
            print(
                f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, Error: {exc}"
            )
        return {
            "compiled": False,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "error": str(exc),
        }
