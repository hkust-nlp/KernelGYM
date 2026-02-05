"""
Task Executor with Subprocess Isolation
任务执行器 - 使用subprocess完全隔离CUDA Error

这个模块提供完全隔离的任务执行能力：
1. 每个任务在独立的subprocess中执行
2. 使用spawn context确保不继承CUDA状态
3. 每个subprocess直接使用分配的GPU设备（cuda:0, cuda:1等）
4. CUDA Error只影响当前subprocess，不会影响主进程或其他子进程
5. 支持torch.profiler（在subprocess中启用）

Author: KernelGym Team
Date: 2025-10-29
Version: v0.3.3-alpha
"""

import sys
import time
import logging
import traceback
import multiprocessing as mp
import queue
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("kernelgym.task_executor")


@dataclass
class TaskExecutionMetrics:
    """任务执行指标"""
    subprocess_spawn_time: float  # subprocess启动时间
    task_execution_time: float    # 任务执行时间
    total_time: float              # 总时间
    profiling_overhead: float = 0.0  # profiling开销（如果启用）
    success: bool = True
    error_type: Optional[str] = None


class IsolatedTaskExecutor:
    """
    完全隔离的任务执行器
    
    核心特性：
    1. 每个任务在新的subprocess中执行（spawn模式）
    2. CUDA Error完全隔离，不影响主进程
    3. 支持torch.profiler（在subprocess中启用）
    4. 完整的错误处理和超时机制
    5. GPU资源自动清理
    
    设计原则：
    - 主进程不使用CUDA
    - Subprocess在spawn后初始化CUDA，直接使用分配的GPU设备
    - Profiler在subprocess中启用（可选）
    - 所有数据通过Queue传递
    """
    
    @staticmethod
    def execute_task(
        task_data: Dict[str, Any],
        device_id: int,
        timeout: int = 60,
    ) -> Tuple[Dict[str, Any], TaskExecutionMetrics]:
        """
        在隔离的subprocess中执行通用任务（toolkit + backend）

        Args:
            task_data: task payload 字典（必须包含 toolkit 与 backend_adapter）
            device_id: GPU设备ID（物理ID，如0-7）
            timeout: 超时时间（秒）

        Returns:
            (result_dict, metrics): 结果字典和执行指标

        Raises:
            TimeoutError: 任务超时
            RuntimeError: 任务执行失败
        """
        start_time = time.time()
        
        # 创建spawn context的进程
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        spawn_start = time.time()
        
        process = ctx.Process(
            target=_toolkit_worker,
            args=(task_data, device_id, result_queue),
        )
        
        process.start()
        spawn_time = time.time() - spawn_start
        
        try:
            # 等待结果
            exec_start = time.time()
            result_data = result_queue.get(timeout=timeout)
            exec_time = time.time() - exec_start
            
            # 获取profiling数据（如果有）
            # 等待进程结束
            process.join(timeout=5)
            if process.is_alive():
                logger.warning("Process did not terminate, forcing kill")
                process.terminate()
                process.join(timeout=2)
            
            total_time = time.time() - start_time
            
            # 检查结果
            if not result_data.get('success', False):
                # 任务失败
                error_type = result_data.get('error_type', 'Unknown')
                error_message = result_data.get('error_message', 'Unknown error')
                
                metrics = TaskExecutionMetrics(
                    subprocess_spawn_time=spawn_time,
                    task_execution_time=exec_time,
                    total_time=total_time,
                    success=False,
                    error_type=error_type
                )
                
                raise RuntimeError(f"{error_type}: {error_message}")
            
            # 任务成功
            result = result_data['result']
            
            # 计算profiling开销
            profiling_overhead = 0.0
            if task_data.get("enable_profiling"):
                profiling_overhead = exec_time * 0.1  # 粗略估计
            
            metrics = TaskExecutionMetrics(
                subprocess_spawn_time=spawn_time,
                task_execution_time=exec_time,
                total_time=total_time,
                profiling_overhead=profiling_overhead,
                success=True
            )
            
            logger.info(
                f"Task {task_data.get('task_id', 'unknown')} completed: "
                f"spawn={spawn_time:.3f}s, exec={exec_time:.3f}s, total={total_time:.3f}s"
            )
            
            return result, metrics
            
        except queue.Empty:
            # 超时
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            
            total_time = time.time() - start_time
            metrics = TaskExecutionMetrics(
                subprocess_spawn_time=spawn_time,
                task_execution_time=timeout,
                total_time=total_time,
                success=False,
                error_type='TimeoutError'
            )
            
            raise TimeoutError(
                f"Task {task_data.get('task_id', 'unknown')} timeout after {timeout}s"
            )
            
        except Exception as e:
            # 其他错误
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            
            logger.error(f"Task execution failed: {e}")
            raise
            
        finally:
            # 确保进程被清理
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
    

# ============================================================================
# Subprocess Worker Functions (模块级别，可以被pickle)
# ============================================================================

def _toolkit_worker(
    task_data: Dict[str, Any],
    device_id: int,
    result_queue: mp.Queue,
):
    try:
        import torch
        import torch.cuda
        from kernelgym.backend import get_backend
        from kernelgym.toolkit import get_toolkit

        torch.cuda.init()
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        task_data["device"] = str(device)

        toolkit_name = task_data.get("toolkit")
        backend_adapter = task_data.get("backend_adapter")
        if not toolkit_name:
            raise ValueError("Task payload missing required 'toolkit'")
        if not backend_adapter:
            raise ValueError("Task payload missing required 'backend_adapter'")

        toolkit = get_toolkit(toolkit_name)
        backend = get_backend(backend_adapter)

        result = toolkit.evaluate(task_data, backend=backend)
        result_queue.put({"success": True, "result": result})
    except Exception as e:
        error_info = {
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        result_queue.put(error_info)
    finally:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as cleanup_error:
            print(f"[WARNING] GPU cleanup failed: {cleanup_error}", file=sys.stderr)
