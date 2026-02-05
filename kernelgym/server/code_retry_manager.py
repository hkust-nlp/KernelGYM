"""
Code Retry Manager for handling problematic kernels.
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import redis.asyncio as redis

from kernelgym.config import settings

logger = logging.getLogger("kernelgym.code_retry")


class CodeRetryManager:
    """Manages retry logic for kernels that cause illegal memory access errors."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.max_retries = 2  # Maximum retry attempts per code
        self.retry_delay = 30  # Seconds between retries
        self.error_pattern_ttl = 86400  # 24 hours TTL for error patterns
        self.key_prefix = settings.redis_key_prefix
        self.legacy_prefix = settings.redis_key_prefix_legacy

    def _prefixes_for_read(self):
        prefixes = [self.key_prefix]
        if self.legacy_prefix and self.legacy_prefix != self.key_prefix:
            prefixes.append(self.legacy_prefix)
        return prefixes

    def _key(self, suffix: str, prefix: Optional[str] = None) -> str:
        return f"{prefix or self.key_prefix}:{suffix}"
        
    async def should_retry_code(self, task_id: str, error_message: str) -> bool:
        """Decide whether to retry based on error type and history.
        Policy: DO NOT retry on CUDA/memory illegal access; treat as hard failure.
        """
        # Never retry memory/illegal access errors
        if self._is_memory_error(error_message):
            logger.info(f"Not retrying task {task_id} due to memory/illegal access error")
            return False

        # Retry on profiler dropouts (no CUDA events captured), with bounded attempts.
        if self._is_profiler_error(error_message):
            original_task_id = self._get_original_task_id(task_id)
            retry_count = await self.get_retry_count(original_task_id)
            if retry_count >= self.max_retries:
                logger.info(
                    f"Not retrying task {task_id} due to profiler dropout retry limit"
                )
                return False
            return True
        
        # For now, default: no automatic retries for other errors either
        return False
    
    async def record_memory_error(self, task_id: str, code_hash: str, error_details: Dict[str, Any]):
        """Record a memory error for tracking patterns."""
        error_key = self._key(f"memory_errors:{code_hash}")
        error_data = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "error_message": error_details.get("error_message", ""),
            "worker_id": error_details.get("worker_id", ""),
            "device": error_details.get("device", "")
        }
        
        # Add to error list
        await self.redis.rpush(error_key, json.dumps(error_data))
        await self.redis.expire(error_key, self.error_pattern_ttl)
        
        # Increment error count
        count_key = self._key(f"memory_error_count:{code_hash}")
        await self.redis.incr(count_key)
        await self.redis.expire(count_key, self.error_pattern_ttl)
    
    async def get_error_count(self, code_hash: str) -> int:
        """Get total error count for a specific code hash."""
        for prefix in self._prefixes_for_read():
            count_key = self._key(f"memory_error_count:{code_hash}", prefix=prefix)
            count = await self.redis.get(count_key)
            if count:
                return int(count)
        return 0
    
    async def increment_retry_count(self, task_id: str) -> int:
        """Increment and return the retry count for a task."""
        original_task_id = self._get_original_task_id(task_id)
        retry_key = self._key(f"retry_count:{original_task_id}")
        new_count = await self.redis.incr(retry_key)
        await self.redis.expire(retry_key, 3600)  # 1 hour TTL
        return new_count
    
    async def get_retry_count(self, task_id: str) -> int:
        """Get current retry count for a task."""
        original_task_id = self._get_original_task_id(task_id)
        for prefix in self._prefixes_for_read():
            retry_key = self._key(f"retry_count:{original_task_id}", prefix=prefix)
            count = await self.redis.get(retry_key)
            if count:
                return int(count)
        return 0
    
    async def schedule_retry(self, task_data: Dict[str, Any], delay_seconds: Optional[int] = None):
        """Schedule a task for retry."""
        if delay_seconds is None:
            delay_seconds = self.retry_delay
        
        task_id = task_data["task_id"]
        original_task_id = task_data.get("original_task_id") or self._get_original_task_id(task_id)
        retry_count = await self.increment_retry_count(original_task_id)
        
        # Add retry metadata
        task_data["retry_count"] = retry_count
        task_data["retry_after"] = (datetime.now() + timedelta(seconds=delay_seconds)).isoformat()
        task_data["original_task_id"] = original_task_id
        
        # Store in retry queue
        retry_key = self._key("retry_queue")
        await self.redis.zadd(
            retry_key,
            {json.dumps(task_data): datetime.now().timestamp() + delay_seconds}
        )
        
        logger.info(
            f"Scheduled retry for task {task_id} (attempt {retry_count}/{self.max_retries}) "
            f"in {delay_seconds}s"
        )
    
    async def get_ready_retries(self) -> List[Dict[str, Any]]:
        """Get tasks that are ready for retry."""
        retry_key = self._key("retry_queue")
        current_time = datetime.now().timestamp()
        
        # Get tasks with score (timestamp) <= current time
        ready_tasks = await self.redis.zrangebyscore(
            retry_key, 
            "-inf", 
            current_time,
            withscores=False
        )
        
        tasks = []
        for task_data in ready_tasks:
            try:
                task = json.loads(task_data)
                tasks.append(task)
                # Remove from retry queue
                await self.redis.zrem(retry_key, task_data)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode retry task: {task_data}")
        
        return tasks
    
    async def clear_retry_history(self, task_id: str):
        """Clear retry history for a successfully completed task."""
        original_task_id = self._get_original_task_id(task_id)
        for prefix in self._prefixes_for_read():
            retry_key = self._key(f"retry_count:{original_task_id}", prefix=prefix)
            await self.redis.delete(retry_key)
    
    def _is_memory_error(self, error_message: str) -> bool:
        """Check if error is related to memory access issues."""
        memory_error_patterns = [
            "illegal memory access",
            "an illegal memory access was encountered",
            "illegal address",
            "cuda error",
            "cuda_error",
            "cudaerror",
            "cudaerrorillegalaccess",
            "cuda_error_illegal_address",
            "unspecified launch failure",
            "device-side assert triggered",
            "misaligned address",
            "memory access violation",
            "segmentation fault",
            "invalid memory",
            "out of memory"
        ]
        
        error_lower = (error_message or "").lower()
        return any(p in error_lower for p in memory_error_patterns)

    def _is_profiler_error(self, error_message: str) -> bool:
        """Check if error is related to profiler dropouts."""
        return "PROFILER_NO_CUDA_EVENTS" in (error_message or "")

    def _get_original_task_id(self, task_id: str) -> str:
        """Normalize task id for retry counting."""
        if "_retry" in task_id:
            return task_id.rsplit("_retry", 1)[0]
        return task_id
    
    async def get_problematic_codes(self, min_errors: int = 3) -> List[Dict[str, Any]]:
        """Get list of code hashes that frequently cause memory errors."""
        problematic: List[Dict[str, Any]] = []

        for prefix in self._prefixes_for_read():
            pattern = f"{prefix}:memory_error_count:*"
            cursor = b"0"
            while cursor:
                cursor, keys = await self.redis.scan(cursor, match=pattern.encode(), count=100)
                for key in keys:
                    count = await self.redis.get(key)
                    if count and int(count) >= min_errors:
                        code_hash = key.decode().split(":")[-1]
                        problematic.append({"code_hash": code_hash, "error_count": int(count)})

        return sorted(problematic, key=lambda x: x["error_count"], reverse=True)
    
    async def is_code_problematic(self, code_hash: str, threshold: int = 5) -> bool:
        """Check if a code hash has too many memory errors."""
        error_count = await self.get_error_count(code_hash)
        return error_count >= threshold
