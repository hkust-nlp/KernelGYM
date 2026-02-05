"""Backend abstraction (compile/load/run)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Backend(ABC):
    name: str = "unknown"

    @abstractmethod
    def compile(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        """Compile kernel code and return build metadata."""

    @abstractmethod
    def load(self, artifact: Dict[str, Any], **kwargs: Any) -> Any:
        """Load compiled artifact for execution."""

    @abstractmethod
    def run(self, handle: Any, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Execute and return runtime metrics."""

    def create_model(self, handle: Any, init_inputs: Any, **kwargs: Any) -> Any:
        """Optional hook to construct a model instance from a loaded handle."""
        raise NotImplementedError("create_model is not implemented for this backend.")

    def open_session(self, handle: Any, device: Any | None = None) -> "BackendSession":
        return BackendSession(self, handle, device=device)

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        """Optional cleanup hook for resources created by load/run."""

    def clean(self, handle: Any, **kwargs: Any) -> None:
        """Alias for cleanup, provided for compatibility with older call sites."""
        self.cleanup(handle, **kwargs)

    def close(self, handle: Any, **kwargs: Any) -> None:
        """Alias for cleanup, provided for explicit lifecycle management."""
        self.cleanup(handle, **kwargs)


class BackendSession:
    """Lightweight lifecycle wrapper around a backend handle."""

    def __init__(self, backend: Backend, handle: Any, device: Any | None = None) -> None:
        self.backend = backend
        self.handle = handle
        self.device = device

    def create_model(self, init_inputs: Any, **kwargs: Any) -> Any:
        return self.backend.create_model(
            self.handle, init_inputs, device=self.device, **kwargs
        )

    def run(self, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return self.backend.run(self.handle, inputs, device=self.device, **kwargs)

    def cleanup(self) -> None:
        self.backend.cleanup(self.handle)

    def close(self) -> None:
        self.cleanup()

    def __enter__(self) -> "BackendSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()
