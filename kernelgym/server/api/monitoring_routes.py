"""Monitoring routes for KernelGym API."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends

from kernelgym.server.task_manager import TaskManager
from kernelgym.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/problematic-codes")
async def get_problematic_codes(
    min_errors: int = 3,
    task_manager: TaskManager = Depends(lambda: None),
) -> Dict[str, Any]:
    try:
        from .server import get_task_manager

        task_manager = await get_task_manager()

        if not task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        problematic_codes = await task_manager.retry_manager.get_problematic_codes(min_errors)

        return {
            "status": "success",
            "min_errors_threshold": min_errors,
            "problematic_codes": problematic_codes,
            "total_count": len(problematic_codes),
        }

    except Exception as exc:
        logger.error(f"Failed to get problematic codes: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/retry-queue")
async def get_retry_queue_status(
    task_manager: TaskManager = Depends(lambda: None),
) -> Dict[str, Any]:
    try:
        from .server import get_task_manager

        task_manager = await get_task_manager()

        retry_key = f"{settings.redis_key_prefix}:retry_queue"
        retry_count = await task_manager.redis.zcard(retry_key)

        next_retries = await task_manager.redis.zrange(retry_key, 0, 9, withscores=True)

        retry_tasks = []
        for task_data, score in next_retries:
            try:
                import json
                from datetime import datetime

                task = json.loads(task_data)
                retry_time = datetime.fromtimestamp(score)
                retry_tasks.append(
                    {
                        "task_id": task.get("task_id"),
                        "retry_count": task.get("retry_count", 0),
                        "scheduled_for": retry_time.isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "success", "retry_queue_size": retry_count, "next_retries": retry_tasks}

    except Exception as exc:
        logger.error(f"Failed to get retry queue status: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/worker-health")
async def get_worker_health(
    task_manager: TaskManager = Depends(lambda: None),
) -> Dict[str, Any]:
    try:
        from .server import get_task_manager

        task_manager = await get_task_manager()

        workers_status = await task_manager.get_workers_status()

        for worker_id in workers_status:
            worker_key = f"{settings.redis_key_prefix}:worker:{worker_id}"
            worker_data = await task_manager.redis.hgetall(worker_key)

            if worker_data:
                cuda_error_shutdown = worker_data.get(b"cuda_error_shutdown", b"false").decode()
                workers_status[worker_id]["cuda_error_shutdown"] = cuda_error_shutdown == "true"

                if b"shutdown_time" in worker_data:
                    workers_status[worker_id]["shutdown_time"] = worker_data[b"shutdown_time"].decode()

        return {"status": "success", "workers": workers_status}

    except Exception as exc:
        logger.error(f"Failed to get worker health: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/clear-error-history/{code_hash}")
async def clear_error_history(
    code_hash: str,
    task_manager: TaskManager = Depends(lambda: None),
) -> Dict[str, Any]:
    try:
        from .server import get_task_manager

        task_manager = await get_task_manager()

        count_key = f"{settings.redis_key_prefix}:memory_error_count:{code_hash}"
        await task_manager.redis.delete(count_key)

        error_key = f"{settings.redis_key_prefix}:memory_errors:{code_hash}"
        await task_manager.redis.delete(error_key)

        return {"status": "success", "message": f"Cleared error history for code hash {code_hash}"}

    except Exception as exc:
        logger.error(f"Failed to clear error history: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
