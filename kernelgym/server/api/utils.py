"""Utility functions for KernelGym API server."""

import logging
from datetime import datetime
from typing import Dict, Any

import psutil
import torch

from kernelgym.config import settings

logger = logging.getLogger(__name__)


def format_timestamp(dt: datetime) -> str:
    return dt.isoformat() + "Z"


async def get_gpu_info() -> Dict[str, Any]:
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            if i in settings.gpu_devices:
                device = f"cuda:{i}"
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)

                    memory_used_percent = (memory_allocated / memory_total) * 100

                    gpu_info[device] = {
                        "name": gpu_name,
                        "memory_total": f"{memory_total / (1024**3):.1f}GB",
                        "memory_allocated": f"{memory_allocated / (1024**3):.1f}GB",
                        "memory_reserved": f"{memory_reserved / (1024**3):.1f}GB",
                        "memory_used_percent": f"{memory_used_percent:.1f}%",
                        "available": True,
                    }
                except Exception as exc:
                    gpu_info[device] = {"error": str(exc), "available": False}

        return gpu_info

    except Exception as exc:
        logger.error(f"Error getting GPU info: {exc}")
        return {"error": str(exc)}


async def get_system_health() -> Dict[str, Any]:
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        gpu_status = await get_gpu_info()

        queue_status = {"pending": 0, "processing": 0, "completed": 0}

        return {
            "status": "healthy",
            "timestamp": format_timestamp(datetime.now()),
            "gpu_status": gpu_status,
            "queue_status": queue_status,
            "memory_usage": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": f"{memory.available / (1024**3):.1f}GB",
                "memory_total": f"{memory.total / (1024**3):.1f}GB",
            },
            "active_tasks": 0,
            "total_processed": 0,
            "uptime": 0.0,
        }

    except Exception as exc:
        logger.error(f"Error getting system health: {exc}")
        return {
            "status": "unhealthy",
            "timestamp": format_timestamp(datetime.now()),
            "error": str(exc),
        }


async def get_system_metrics() -> Dict[str, Any]:
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        gpu_info = await get_gpu_info()
        avg_gpu_utilization = 0.0
        if gpu_info and "error" not in gpu_info:
            utilizations = []
            for info in gpu_info.values():
                if "memory_used_percent" in info:
                    utilizations.append(float(info["memory_used_percent"].rstrip("%")))
            if utilizations:
                avg_gpu_utilization = sum(utilizations) / len(utilizations)

        return {
            "timestamp": format_timestamp(datetime.now()),
            "performance_metrics": {
                "avg_processing_time": 0.0,
                "throughput_per_hour": 0.0,
                "success_rate": 0.0,
            },
            "resource_metrics": {
                "avg_gpu_utilization": avg_gpu_utilization,
                "memory_usage_percent": memory.percent,
                "cpu_usage_percent": cpu_percent,
            },
            "queue_metrics": {"pending_tasks": 0, "active_tasks": 0, "completed_tasks": 0},
            "error_metrics": {"compilation_errors": 0, "runtime_errors": 0, "timeout_errors": 0},
        }

    except Exception as exc:
        logger.error(f"Error getting system metrics: {exc}")
        return {"timestamp": format_timestamp(datetime.now()), "error": str(exc)}


async def validate_gpu_availability() -> bool:
    try:
        if not torch.cuda.is_available():
            return False

        available_devices = torch.cuda.device_count()
        required_devices = max(settings.gpu_devices) + 1 if settings.gpu_devices else 1
        return available_devices >= required_devices

    except Exception as exc:
        logger.error(f"Error validating GPU availability: {exc}")
        return False


async def cleanup_old_tasks(redis_client, max_age_hours: int = 24) -> int:
    try:
        return 0
    except Exception as exc:
        logger.error(f"Error cleaning up old tasks: {exc}")
        return 0


async def get_task_statistics(redis_client) -> Dict[str, Any]:
    try:
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0,
        }
    except Exception as exc:
        logger.error(f"Error getting task statistics: {exc}")
        return {"error": str(exc)}
