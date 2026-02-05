"""
Agent environment module with base classes and implementations.

This module ensures that all exec_tool_call methods in environments
inheriting from BaseEnv are decorated with with_timeout_and_retry.
"""

import functools
import inspect
from typing import get_type_hints

from .base_env import BaseEnv, FinishReasonTypeEnum, with_timeout_and_retry
from .code_sandbox_env import CodeSandboxEnv
from .file_search_env import FileSearchEnv
from .local_search_env import LocalSearchEnv
from .math_sandbox_env import MathSandboxEnv
from .swe_file_location_env import SWEFileLocationEnv

# Export all environment classes
__all__ = [
    "BaseEnv",
    "MathSandboxEnv",
    "CodeSandboxEnv",
    "LocalSearchEnv",
    "FileSearchEnv",
    "SWEFileLocationEnv",
    "with_timeout_and_retry",
    "FinishReasonTypeEnum",
    "create_environment",
]


def create_environment(env_type: str, max_turns: int, extra_info: dict = None) -> BaseEnv:
    """Factory function to create environment instances based on configuration.

    Args:
        env_type: Type of environment to create (e.g., 'MathSandboxEnv', 'CodeSandboxEnv')
        max_turns: Maximum number of turns for the environment
        extra_info: Extra information from dataset (e.g., prompt-dependent import code)

    Returns:
        BaseEnv instance

    Raises:
        ValueError: If env_type is not supported
    """
    env_map = {
        'MathSandboxEnv': MathSandboxEnv,
        'CodeSandboxEnv': CodeSandboxEnv,
        'LocalSearchEnv': LocalSearchEnv,
        'FileSearchEnv': FileSearchEnv,
        'SWEFileLocationEnv': SWEFileLocationEnv,
    }

    if env_type not in env_map:
        raise ValueError(f"Unsupported environment type: {env_type}. Supported types: {list(env_map.keys())}")

    return env_map[env_type](max_turns=max_turns, extra_info=extra_info)


def _check_exec_tool_call_decorator(cls):
    """
    Check if a class's exec_tool_call method is decorated with with_timeout_and_retry.

    This function verifies that any override of exec_tool_call in a BaseEnv subclass
    has the required timeout and retry decorator applied.
    """
    if not issubclass(cls, BaseEnv):
        return

    # Skip BaseEnv itself
    if cls is BaseEnv:
        return

    # Check if the class overrides exec_tool_call
    if 'exec_tool_call' in cls.__dict__:
        method = cls.__dict__['exec_tool_call']

        # Check if the method is decorated by looking at its wrapper attributes
        # The with_timeout_and_retry decorator adds retry and timeout functionality
        if not (
            hasattr(method, '__wrapped__')
            or (hasattr(method, '__name__') and 'wrapper' in str(method))
            or (hasattr(method, '__code__') and method.__code__.co_name == 'wrapper')
        ):
            raise AssertionError(
                f"{cls.__name__}.exec_tool_call must be decorated with @with_timeout_and_retry. "
                f"Add the decorator like this:\n"
                f"@with_timeout_and_retry(timeout_seconds=30.0, max_attempts=3)\n"
                f"async def exec_tool_call(self, action: str) -> tuple[str, float, bool, dict]:\n"
                f"    ..."
            )


# Perform runtime checks on all imported environment classes
for name in __all__:
    obj = globals().get(name)
    if obj and inspect.isclass(obj) and issubclass(obj, BaseEnv):
        _check_exec_tool_call_decorator(obj)
