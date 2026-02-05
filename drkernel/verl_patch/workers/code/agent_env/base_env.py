import asyncio
import functools
import logging
import random
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar, cast

import ray
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random

from verl_patch.workers.code.reward_manager.reward_cache import (
    DistributedCacheWrapper,
    DistributedRewardCache,
)

# Type variable for return type annotations
T = TypeVar('T')


def on_retry_error(retry_state):
    """Callback function for when all retries are exhausted"""
    e = retry_state.outcome.exception()
    logging.error(f'Give up retrying after {retry_state.attempt_number} attempts. Error: {e}')
    raise e


def before_retry_sleep(retry_state):
    """Callback function before each retry sleep"""
    msg = f'Function call error for {retry_state.attempt_number} time(s), will retry... Error: {retry_state.outcome.exception()}'
    if retry_state.attempt_number > 2:
        logging.warning(msg)
    else:
        logging.debug(msg)


def configurable_retry(max_attempts: int = 3, timeout_seconds: int = 20):
    """Decorator to add retry logic with constant wait to async/sync functions

    Args:
        max_attempts: Maximum number of attempts before giving up
        timeout_seconds: Timeout in seconds, used to calculate wait time and jitter

    Returns:
        Decorated function with retry logic
    """

    def decorator(func):
        # Calculate wait time and jitter based on timeout_seconds
        # Use a percentage of timeout_seconds for wait time (e.g., 10-20%)
        wait_time = min(10, timeout_seconds * 0.25)  # 25% of timeout, max 10 seconds
        jitter_time = min(10, timeout_seconds * 0.75)  # 75% of wait time as jitter, max 10 seconds

        @wraps(func)
        @retry(
            wait=wait_fixed(wait_time) + wait_random(0, jitter_time),
            stop=stop_after_attempt(max_attempts),
            before_sleep=before_retry_sleep,
            retry_error_callback=on_retry_error,
        )
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        @retry(
            wait=wait_fixed(wait_time) + wait_random(0, jitter_time),
            stop=stop_after_attempt(max_attempts),
            before_sleep=before_retry_sleep,
            retry_error_callback=on_retry_error,
        )
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to async functions

    Args:
        timeout_seconds: Timeout in seconds

    Returns:
        Decorated function with timeout
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise

        if asyncio.iscoroutinefunction(func):
            return wrapper
        else:
            raise ValueError("with_timeout decorator can only be used with async functions")

    return decorator


def with_timeout_and_retry(timeout_seconds: float = 30.0, max_attempts: int = 3):
    """Decorator combining both timeout and retry logic

    Args:
        timeout_seconds: Timeout in seconds for each attempt
        max_attempts: Maximum number of retry attempts

    Returns:
        Decorated function with both timeout and retry
    """

    def decorator(func):
        @configurable_retry(max_attempts, timeout_seconds)
        @with_timeout(timeout_seconds)
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return wrapper
        else:
            raise ValueError("with_timeout_and_retry decorator can only be used with async functions")

    return decorator


def get_env_reward_cache(max_retries: int = 5) -> DistributedCacheWrapper:
    """
    使用重试机制处理并发创建
    """

    for attempt in range(max_retries):
        try:
            # 尝试获取现有 actor
            actor = ray.get_actor("env_reward_cache")
            logging.info("Connected to existing reward cache actor")
            return DistributedCacheWrapper(actor)

        except ValueError:
            # Actor 不存在，尝试创建
            try:
                logging.info(f"Creating reward cache actor (attempt {attempt + 1})")
                actor = DistributedRewardCache.options(
                    name="env_reward_cache", lifetime="detached", max_restarts=3
                ).remote(name="env_reward_cache", max_size=512 * 128, persist_threshold=3)

                logging.info("Successfully created reward cache actor")
                return DistributedCacheWrapper(actor)

            except Exception as e:
                error_msg = str(e).lower()

                if "already exists" in error_msg or "name is already taken" in error_msg:
                    # 另一个进程已经创建，下次循环会获取到
                    logging.info("Actor created by another process, will retry getting it")
                    # 随机延迟避免所有进程同时重试
                    time.sleep(random.uniform(0.01, 0.1))
                    continue
                else:
                    # 其他错误
                    logging.error(f"Failed to create actor: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(random.uniform(0.1, 0.5))

    raise RuntimeError(f"Failed to get reward cache after {max_retries} attempts")


class BaseEnv:
    def __init__(self, max_turns: int = 2):
        self.max_turns = max_turns
        self.num_turns = 0

        # Initialize distributed cache using global actor pattern
        self.cache = get_env_reward_cache()

    async def reset(self, extra_info: dict | None = None) -> None:
        self.num_turns = 0

    async def step(self, action: str | None) -> tuple[str | None, bool, bool, float, dict]:
        """
        Args:
            action: str, the action from the agent
        Returns:
            str, the tool response
            bool, whether the episode is done
            bool, whether the episode is truncated
            float, the reward of the current step
            dict, additional info
        """
        self.num_turns += 1
        done, truncate, reward = False, False, 0
        tool_response, tool_info = None, {}

        if action is None:
            done = True
            tool_info["finish_type"] = FinishReasonTypeEnum.NO_TOOL_CALL
        else:
            # execute tool call and obtain relative information
            try:
                exec_result = await self.exec_tool_call(action)
                tool_response, reward, done, tool_info = exec_result
            except asyncio.TimeoutError:
                # Handle timeout case
                tool_response = "Execution timed out."
                reward = 0.0
                done = True
                tool_info["finish_type"] = FinishReasonTypeEnum.ERROR
                tool_info["error_type"] = "timeout"
                tool_info["error_message"] = "Operation timed out after exhausting all retry attempts"
            except Exception as e:
                # Handle other exceptions that might occur during execution
                tool_response = f"Execution failed with error: {str(e)}"
                reward = 0.0
                done = True
                tool_info["finish_type"] = FinishReasonTypeEnum.ERROR
                tool_info["error_type"] = "execution_error"
                tool_info["error_message"] = str(e)
            else:
                if self.num_turns >= self.max_turns:
                    done = True
                    truncate = True
                    tool_info["finish_type"] = FinishReasonTypeEnum.MAX_TOOL_CALL

        if tool_response is None or tool_response.strip() == "":
            tool_response = "The tool did not return any response."

        return tool_response, done, truncate, reward, tool_info

    async def exec_tool_call(self, action: str) -> tuple[str, float, bool, dict]:
        """
        Args:
            action: str, the action from the agent
        Returns:
            str, the tool response
            float, the reward of the current step
            bool, whether the episode is done
            dict, additional info
        """
        raise NotImplementedError("exec_tool_call must be implemented")


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"
    NO_TOOL_CALL = "no_tool_call"
    ANSWER = "answer"
    MAX_TOOL_CALL = "max_tool_call"
    ERROR = "error"
    ASYNC_TIMEOUT = "async_timeout"
    SKIPPED = "skipped"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        elif value == "no_tool_call":
            return cls.NO_TOOL_CALL
        elif value == "answer":
            return cls.ANSWER
        elif value == "max_tool_call":
            return cls.MAX_TOOL_CALL
        elif value == "async_timeout":
            return cls.ASYNC_TIMEOUT
        elif value == "error":
            return cls.ERROR
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")
