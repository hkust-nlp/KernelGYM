"""KernelBench profiling helpers (toolkit layer)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch

from kernelgym.config import settings

logger = logging.getLogger("kernelgym.toolkit.kernelbench.profiling")


def compute_triton_kernel_coverage(matched_triton_kernels: List[str], profilling_result: Dict[str, Any]):
    """Compute the coverage of the matched triton kernels in the profiling result."""

    def _matches_profiler_name(captured: str, profiler_name: str) -> bool:
        cap = captured.lower()
        prof = profiler_name.lower()
        if cap == prof:
            return True
        if cap in prof or prof in cap:
            return True
        return False

    kernels = matched_triton_kernels
    num_custom_kernels = 0
    kernel_names = [kernel.split(" ")[0] for kernel in kernels]

    kernels_in_profiling = profilling_result["kernels"]

    total_time = 0.0
    matched_cuda_time = 0.0
    triton_kernels_in_profiling = []

    for prof_kernel in kernels_in_profiling:
        prof_name = prof_kernel["name"]
        cuda_time = float(prof_kernel["cuda_time_us"])
        cpu_time = float(prof_kernel["cpu_time_us"])
        total_time += cuda_time + cpu_time

        if any(_matches_profiler_name(kernel_name, prof_name) for kernel_name in kernel_names):
            triton_kernels_in_profiling.append(prof_name)
            num_custom_kernels += 1
            matched_cuda_time += cuda_time

    triton_kernels_not_in_profiling = [
        kernel_name
        for kernel_name in kernel_names
        if not any(_matches_profiler_name(kernel_name, prof_name) for prof_name in triton_kernels_in_profiling)
    ]

    return {
        "num_custom_kernels": num_custom_kernels,
        "num_total_kernels": len(kernels_in_profiling),
        "total_kernel_run_time_in_profiling_us": total_time,
        "custom_kernel_cuda_time_in_profiling_us": matched_cuda_time,
        "triton_kernels_not_in_profiling": triton_kernels_not_in_profiling,
        "triton_kernels_in_profiling": triton_kernels_in_profiling,
    }


@contextmanager
def profiling_context(enabled: bool = True):
    if not enabled:
        yield None
        return

    try:
        import torch.profiler as profiler

        activities = []
        if "cpu" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CPU)
        if "cuda" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CUDA)

        print(f"[Profiler] Initializing with activities: {[str(a) for a in activities]}")

        if not activities:
            print("[Profiler] No activities configured, profiler will return no data")
            yield None
            return

        prof = profiler.profile(
            activities=activities,
            record_shapes=settings.profiling_record_shapes,
            profile_memory=settings.profiling_profile_memory,
            with_stack=settings.profiling_with_stack,
            on_trace_ready=None,
        )

        prof.__enter__()
        try:
            print("[Profiler] Profiler started successfully")
            cuda_available = torch.cuda.is_available()
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            device_info = "cuda:unavailable"
            if cuda_available:
                try:
                    current_device = torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(current_device)
                    device_info = f"cuda:{current_device} ({device_name})"
                except Exception as e:
                    device_info = f"cuda:unknown (error={e})"
            print(
                "[Profiler] Context pid=%s cuda_available=%s device=%s CUDA_VISIBLE_DEVICES=%s",
                os.getpid(),
                cuda_available,
                device_info,
                cuda_visible,
            )
            if cuda_available:
                try:
                    test = torch.ones((1024,), device="cuda")
                    _ = test.sum()
                    torch.cuda.synchronize()
                    print("[Profiler] Self-test CUDA op executed")
                except Exception as e:
                    print(f"[Profiler] Self-test failed: {e}")
            yield prof
        finally:
            try:
                prof.__exit__(None, None, None)
                print("[Profiler] Profiler stopped successfully")
            except Exception as e:
                print(f"[Profiler] Error during profiler cleanup: {e}")

    except Exception as e:
        logger.warning(f"[Profiler] Failed to initialize profiler: {e}. Continuing without profiling.")
        yield None


def extract_profiling_metrics(prof: Optional["torch.profiler.profile"]) -> Dict[str, Any]:
    if prof is None:
        return {}

    try:
        import torch.profiler as profiler

        events = prof.key_averages()
        print(f"[Profiler] key_averages: {events}")
        total_events = len(events)
        cuda_device_event_count = 0
        cuda_time_event_count = 0
        self_cuda_time_event_count = 0

        logger.debug(f"[Profiler] Captured {total_events} total events")

        def _safe_metric(evt: Any, names: Tuple[str, ...], default: float = 0.0) -> float:
            for name in names:
                if hasattr(evt, name):
                    value = getattr(evt, name)
                    if callable(value):
                        try:
                            value = value()
                        except Exception:
                            continue
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
            return default

        def _safe_int_metric(evt: Any, names: Tuple[str, ...], default: int = 0) -> int:
            for name in names:
                if hasattr(evt, name):
                    value = getattr(evt, name)
                    if callable(value):
                        try:
                            value = value()
                        except Exception:
                            continue
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            return default

        cuda_kernels = []
        total_cpu_time = 0.0
        total_self_cuda_time = 0.0
        for evt in events:
            cpu_time_us = _safe_metric(evt, ("cpu_time_total", "cpu_time"), 0.0)
            total_cpu_time += cpu_time_us

            cuda_time_us = _safe_metric(
                evt,
                ("device_time_total", "device_time", "cuda_time_total", "cuda_time"),
                0.0,
            )
            self_cuda_time_us = _safe_metric(
                evt,
                ("self_cuda_time_total", "self_cuda_time"),
                0.0,
            )
            if self_cuda_time_us > 0.0:
                self_cuda_time_event_count += 1
                total_self_cuda_time += self_cuda_time_us
            if cuda_time_us <= 0.0:
                continue
            device_type = getattr(evt, "device_type", None)
            if device_type is not None and device_type != profiler.DeviceType.CUDA:
                pass
            elif device_type == profiler.DeviceType.CUDA:
                cuda_device_event_count += 1
            cuda_time_event_count += 1

            kernel_entry = {
                "name": getattr(evt, "key", "unknown"),
                "cuda_time_us": cuda_time_us,
                "cpu_time_us": cpu_time_us,
                "count": _safe_int_metric(evt, ("count",), 0),
            }
            memory_usage = _safe_metric(evt, ("cuda_memory_usage",), 0.0)
            if memory_usage > 0.0:
                kernel_entry["cuda_memory_usage"] = memory_usage
            cuda_kernels.append(kernel_entry)

        cuda_kernels.sort(key=lambda x: x["cuda_time_us"], reverse=True)

        logger.debug(
            f"[Profiler] Filtered to {len(cuda_kernels)} CUDA kernels (from {len(events)} total)"
        )
        if len(cuda_kernels) == 0 and len(events) > 0:
            logger.warning(
                f"[Profiler] Captured events but no CUDA kernels! Event types: {[getattr(evt, 'device_type', 'unknown') for evt in list(events)[:5]]}"
            )

        memory_stats = {}
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_stats = {
                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
                    "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
                    "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
                }
        except Exception as e:
            logger.warning(f"[Profiler] Failed to collect memory stats: {e}")

        profiling_metrics = {
            "kernels": cuda_kernels,
            "kernel_count": len(cuda_kernels),
            "total_cpu_time_us": total_cpu_time,
            "total_cuda_time_us": sum(k["cuda_time_us"] for k in cuda_kernels),
            "total_self_cuda_time_us": total_self_cuda_time,
            "cuda_device_event_count": cuda_device_event_count,
            "cuda_time_event_count": cuda_time_event_count,
            "self_cuda_time_event_count": self_cuda_time_event_count,
            "memory_stats": memory_stats,
        }

        if len(cuda_kernels) == 0:
            profiling_metrics["profiling_warning"] = (
                "Profiler captured no CUDA kernels. This may indicate a profiler failure."
            )

        return profiling_metrics

    except Exception as e:
        logger.warning(f"[Profiler] Failed to extract profiling metrics: {e}")
        return {"profiling_error": str(e)}
