"""Common types and enums shared across modules."""

from enum import Enum


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Backend(str, Enum):
    """Supported backend types."""

    CUDA = "cuda"
    TRITON = "triton"


class Priority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ErrorCode(str, Enum):
    """Error code enumeration for different error types."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    CORRECTNESS_ERROR = "CORRECTNESS_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    INSTANTIATION_ERROR = "INSTANTIATION_ERROR"
