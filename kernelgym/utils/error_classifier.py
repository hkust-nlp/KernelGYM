"""Error classification utilities for KernelGym."""

import re
from typing import Optional

from kernelgym.common import ErrorCode


def classify_error(error_message: str, context: Optional[str] = None) -> ErrorCode:
    """Classify error message into appropriate error code."""
    if not error_message:
        return ErrorCode.UNKNOWN_ERROR

    error_lower = error_message.lower()

    validation_patterns = [
        r"validation failed",
        r"dangerous pattern detected",
        r"code must contain",
        r"invalid code format",
        r"missing.*class",
        r"invalid.*entry.*point",
        r"code validation error",
    ]
    if any(re.search(pattern, error_lower) for pattern in validation_patterns):
        return ErrorCode.VALIDATION_ERROR

    compilation_patterns = [
        r"compilation failed",
        r"compile.*error",
        r"syntax error",
        r"nvcc.*error",
        r"cuda.*compilation",
        r"triton.*compilation",
        r"kernel.*compilation",
        r"build.*failed",
        r"linker.*error",
    ]
    if any(re.search(pattern, error_lower) for pattern in compilation_patterns):
        return ErrorCode.COMPILATION_ERROR

    runtime_patterns = [
        r"runtime error",
        r"kernel.*execution.*failed",
        r"cuda.*runtime",
        r"out of memory",
        r"device.*error",
        r"gpu.*error",
        r"invalid.*device",
        r"cuda.*error",
        r"execution.*failed",
    ]
    if any(re.search(pattern, error_lower) for pattern in runtime_patterns):
        return ErrorCode.RUNTIME_ERROR

    correctness_patterns = [
        r"correctness.*check.*failed",
        r"output.*mismatch",
        r"result.*incorrect",
        r"assertion.*failed",
        r"accuracy.*test.*failed",
        r"numerical.*error",
        r"precision.*error",
    ]
    if any(re.search(pattern, error_lower) for pattern in correctness_patterns):
        return ErrorCode.CORRECTNESS_ERROR

    timeout_patterns = [
        r"timeout",
        r"task.*timed.*out",
        r"execution.*timeout",
        r"time.*limit.*exceeded",
        r"hung.*task",
        r"stuck.*task",
    ]
    if any(re.search(pattern, error_lower) for pattern in timeout_patterns):
        return ErrorCode.TIMEOUT_ERROR

    system_patterns = [
        r"system.*error",
        r"internal.*server.*error",
        r"redis.*error",
        r"database.*error",
        r"connection.*failed",
        r"service.*unavailable",
        r"initialization.*failed",
    ]
    if any(re.search(pattern, error_lower) for pattern in system_patterns):
        return ErrorCode.SYSTEM_ERROR

    resource_patterns = [
        r"resource.*error",
        r"insufficient.*memory",
        r"queue.*full",
        r"no.*available.*workers",
        r"gpu.*unavailable",
        r"device.*busy",
        r"memory.*exhausted",
        r"disk.*space",
        r"resource.*exhausted",
    ]
    if any(re.search(pattern, error_lower) for pattern in resource_patterns):
        return ErrorCode.RESOURCE_ERROR

    if context:
        context_lower = context.lower()
        if "validation" in context_lower:
            return ErrorCode.VALIDATION_ERROR
        if "compilation" in context_lower or "compile" in context_lower:
            return ErrorCode.COMPILATION_ERROR
        if "runtime" in context_lower or "execution" in context_lower:
            return ErrorCode.RUNTIME_ERROR
        if "correctness" in context_lower:
            return ErrorCode.CORRECTNESS_ERROR
        if "timeout" in context_lower:
            return ErrorCode.TIMEOUT_ERROR
        if "system" in context_lower:
            return ErrorCode.SYSTEM_ERROR
        if "resource" in context_lower:
            return ErrorCode.RESOURCE_ERROR

    return ErrorCode.UNKNOWN_ERROR


def get_error_description(error_code: ErrorCode) -> str:
    """Get human-readable description for error code."""
    descriptions = {
        ErrorCode.VALIDATION_ERROR: "Code validation failed - invalid or dangerous code detected",
        ErrorCode.COMPILATION_ERROR: "Kernel compilation failed - syntax or build errors",
        ErrorCode.RUNTIME_ERROR: "Kernel runtime error - execution or GPU errors",
        ErrorCode.CORRECTNESS_ERROR: "Correctness check failed - output doesn't match reference",
        ErrorCode.TIMEOUT_ERROR: "Task timeout - execution took too long",
        ErrorCode.SYSTEM_ERROR: "System error - internal service or infrastructure issue",
        ErrorCode.RESOURCE_ERROR: "Resource error - insufficient memory or unavailable resources",
        ErrorCode.UNKNOWN_ERROR: "Unknown error - unclassified error type",
    }
    return descriptions.get(error_code, "Unknown error type")


def get_error_category(error_code: ErrorCode) -> str:
    """Get error category for grouping similar errors."""
    categories = {
        ErrorCode.VALIDATION_ERROR: "input",
        ErrorCode.COMPILATION_ERROR: "compilation",
        ErrorCode.RUNTIME_ERROR: "runtime",
        ErrorCode.CORRECTNESS_ERROR: "correctness",
        ErrorCode.TIMEOUT_ERROR: "timeout",
        ErrorCode.SYSTEM_ERROR: "system",
        ErrorCode.RESOURCE_ERROR: "resource",
        ErrorCode.UNKNOWN_ERROR: "unknown",
    }
    return categories.get(error_code, "unknown")
