"""Shared serialization helpers for schema objects."""

from __future__ import annotations

import ast
from enum import Enum
from typing import Any, Optional

from kernelgym.common import ErrorCode


def make_json_safe(obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """Recursively convert objects to JSON-serializable forms."""
    if depth > max_depth:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, ast.AST):
        try:
            return ast.unparse(obj)
        except Exception:
            return f"<{type(obj).__name__}>"
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x, depth + 1, max_depth) for x in obj]
    return str(obj)


def coerce_error_code(value: Any) -> Optional[ErrorCode | str]:
    if value is None:
        return None
    if isinstance(value, ErrorCode):
        return value
    if isinstance(value, Enum):
        try:
            return ErrorCode(value.value)
        except Exception:
            return value.value
    if isinstance(value, str):
        try:
            return ErrorCode(value)
        except Exception:
            return value
    return value


def serialize_error_code(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value
