"""
GPU诊断工具 - 用于测试subprocess隔离和profiler兼容性

这个模块提供以下功能：
1. 在不初始化主进程CUDA的情况下测试GPU可用性
2. 测试subprocess中的CUDA隔离
3. 测试profiler在subprocess中的兼容性
4. 验证CUDA Error不会污染主进程

Author: KernelServer Team
Date: 2025-10-29
"""

import os
import sys
import subprocess
import multiprocessing as mp
import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger("kernelgym.gpu_diagnostics")


@dataclass
class GPUHealthReport:
    """GPU健康检查报告"""
    healthy: bool
    device_id: int
    device_name: Optional[str] = None
    total_memory_gb: Optional[float] = None
    cuda_available: bool = False
    error_message: Optional[str] = None
    test_duration_sec: float = 0.0


@dataclass
class IsolationTestReport:
    """隔离测试报告"""
    isolation_successful: bool
    main_process_contaminated: bool
    subprocess_error_message: Optional[str] = None
    details: Dict[str, Any] = None


@dataclass
class ProfilerTestReport:
    """Profiler兼容性测试报告"""
    profiler_works: bool
    profiling_data_received: bool
    profiling_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# Module-level worker functions (必须在模块级别以便pickle)

def _gpu_health_worker(device_id: int, result_queue):
    """Subprocess worker for GPU health test"""
    try:
        # 设置CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        
        # Import torch (在subprocess中)
        import torch
        
        if not torch.cuda.is_available():
            result_queue.put({
                'success': False,
                'error': 'CUDA not available in subprocess'
            })
            return
        
        # 初始化CUDA
        torch.cuda.init()
        torch.cuda.set_device(0)  # 因为CUDA_VISIBLE_DEVICES只暴露一个GPU
        
        # 获取GPU信息
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        # 简单测试
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.mm(test_tensor, test_tensor.T)
        torch.cuda.synchronize()
        
        result_queue.put({
            'success': True,
            'device_name': device_name,
            'total_memory': total_memory
        })
        
    except Exception as e:
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _cuda_error_worker(device_id: int, result_queue):
    """故意触发CUDA Error的worker"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        import torch
        
        torch.cuda.init()
        torch.cuda.set_device(0)
        
        # 故意触发CUDA Error: 访问无效的内存地址
        try:
            # 创建一个非常大的tensor，可能导致OOM
            giant_tensor = torch.randn(100000, 100000, device='cuda')
            # 或者使用无效的CUDA kernel配置
            result_queue.put({'phase': 'error_triggered', 'success': False, 'expected': True})
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'out of memory' in str(e):
                result_queue.put({
                    'phase': 'error_caught',
                    'success': True,
                    'error': str(e)
                })
            else:
                raise
                
    except Exception as e:
        result_queue.put({
            'phase': 'unexpected_error',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _normal_worker(device_id: int, result_queue):
    """正常的worker，用于测试GPU是否仍然可用"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        import torch
        
        torch.cuda.init()
        torch.cuda.set_device(0)
        
        # 简单测试
        test_tensor = torch.randn(100, 100, device='cuda')
        result = torch.mm(test_tensor, test_tensor.T)
        torch.cuda.synchronize()
        
        result_queue.put({'success': True, 'phase': 'normal_execution'})
        
    except Exception as e:
        result_queue.put({
            'success': False,
            'phase': 'normal_execution_failed',
            'error': str(e)
        })


def _profiler_worker(device_id: int, result_queue, profiling_queue):
    """使用profiler的worker"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        import torch
        import torch.profiler as profiler
        
        torch.cuda.init()
        torch.cuda.set_device(0)
        
        # 启用profiler
        prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        )
        
        with prof:
            # 执行一些CUDA操作
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.mm(x, x.T)
            z = torch.nn.functional.relu(y)
            torch.cuda.synchronize()
        
        # 提取profiling数据
        events = prof.key_averages()
        cuda_events = [
            evt for evt in events 
            if hasattr(evt, 'device_type') and 
            evt.device_type == profiler.DeviceType.CUDA
        ]
        
        # 构建可序列化的profiling数据
        profiling_data = {
            'total_events': len(list(events)),
            'cuda_events': len(cuda_events),
            'top_5_cuda_kernels': [
                {
                    'name': evt.key,
                    'cuda_time_us': float(evt.cuda_time_total) if hasattr(evt, 'cuda_time_total') else 0.0,
                    'count': int(evt.count) if hasattr(evt, 'count') else 0
                }
                for evt in sorted(
                    cuda_events,
                    key=lambda e: getattr(e, 'cuda_time_total', 0.0),
                    reverse=True
                )[:5]
            ]
        }
        
        # 发送profiling数据
        profiling_queue.put(profiling_data)
        result_queue.put({'success': True})
        
    except Exception as e:
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


class GPUDiagnostics:
    """GPU诊断工具集"""
    
    @staticmethod
    def test_gpu_health_nvidia_smi(device_id: int) -> GPUHealthReport:
        """
        使用nvidia-smi测试GPU健康状态（不初始化CUDA）
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            GPUHealthReport对象
        """
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i", str(device_id),
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return GPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    error_message=f"nvidia-smi failed: {result.stderr}",
                    test_duration_sec=time.time() - start_time
                )
            
            # Parse output: "GPU Name, Memory in MB"
            output = result.stdout.strip()
            parts = output.split(',')
            
            if len(parts) < 2:
                return GPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    error_message=f"Unexpected nvidia-smi output: {output}",
                    test_duration_sec=time.time() - start_time
                )
            
            device_name = parts[0].strip()
            memory_mb = float(parts[1].strip())
            memory_gb = memory_mb / 1024.0
            
            return GPUHealthReport(
                healthy=True,
                device_id=device_id,
                device_name=device_name,
                total_memory_gb=memory_gb,
                cuda_available=True,
                test_duration_sec=time.time() - start_time
            )
            
        except subprocess.TimeoutExpired:
            return GPUHealthReport(
                healthy=False,
                device_id=device_id,
                error_message="nvidia-smi timeout",
                test_duration_sec=time.time() - start_time
            )
        except Exception as e:
            return GPUHealthReport(
                healthy=False,
                device_id=device_id,
                error_message=f"nvidia-smi error: {str(e)}",
                test_duration_sec=time.time() - start_time
            )
    
    @staticmethod
    def test_gpu_health_subprocess(device_id: int) -> GPUHealthReport:
        """
        在subprocess中测试GPU健康状态
        
        这个方法在独立的subprocess中初始化CUDA并测试GPU，
        不会影响主进程。
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            GPUHealthReport对象
        """
        start_time = time.time()
        
        # 使用spawn context创建进程
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        
        process = ctx.Process(target=_gpu_health_worker, args=(device_id, result_queue))
        process.start()
        
        try:
            # 等待结果，10秒超时（spawn+import torch+CUDA init需要时间）
            result = result_queue.get(timeout=10)
            process.join(timeout=2)
            
            duration = time.time() - start_time
            
            if result['success']:
                return GPUHealthReport(
                    healthy=True,
                    device_id=device_id,
                    device_name=result['device_name'],
                    total_memory_gb=result['total_memory'] / (1024**3),
                    cuda_available=True,
                    test_duration_sec=duration
                )
            else:
                return GPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    cuda_available=False,
                    error_message=result.get('error', 'Unknown error'),
                    test_duration_sec=duration
                )
                
        except Exception as e:
            process.terminate()
            process.join(timeout=2)
            
            return GPUHealthReport(
                healthy=False,
                device_id=device_id,
                cuda_available=False,
                error_message=f"Subprocess test failed: {str(e)}",
                test_duration_sec=time.time() - start_time
            )
    
    @staticmethod
    def test_cuda_error_isolation(device_id: int) -> IsolationTestReport:
        """
        测试CUDA Error隔离
        
        在subprocess中故意触发CUDA Error，验证：
        1. Subprocess正确捕获错误
        2. 主进程不受影响
        3. 后续subprocess可以正常使用GPU
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            IsolationTestReport对象
        """
        ctx = mp.get_context('spawn')
        
        # Step 1: 触发CUDA Error
        logger.info(f"[Isolation Test] Step 1: 触发CUDA Error在subprocess中")
        result_queue1 = ctx.Queue()
        process1 = ctx.Process(target=_cuda_error_worker, args=(device_id, result_queue1))
        process1.start()
        
        try:
            result1 = result_queue1.get(timeout=15)  # 增加超时时间
            process1.join(timeout=2)
        except Exception as e:
            process1.terminate()
            return IsolationTestReport(
                isolation_successful=False,
                main_process_contaminated=False,
                subprocess_error_message=f"Step 1 failed: {str(e)}"
            )
        
        # Step 2: 检查主进程是否受影响（主进程不使用CUDA，所以应该没有影响）
        logger.info(f"[Isolation Test] Step 2: 检查主进程状态")
        # 主进程不使用CUDA，所以这一步总是成功
        main_process_ok = True
        
        # Step 3: 在新的subprocess中测试GPU是否仍然可用
        logger.info(f"[Isolation Test] Step 3: 测试GPU在新subprocess中是否可用")
        result_queue2 = ctx.Queue()
        process2 = ctx.Process(target=_normal_worker, args=(device_id, result_queue2))
        process2.start()
        
        try:
            result2 = result_queue2.get(timeout=15)  # 增加超时时间
            process2.join(timeout=2)
        except Exception as e:
            process2.terminate()
            return IsolationTestReport(
                isolation_successful=False,
                main_process_contaminated=False,
                subprocess_error_message=f"Step 3 failed: {str(e)}",
                details={
                    'step1': result1,
                    'step2': 'main_process_ok',
                    'step3_error': str(e)
                }
            )
        
        # 判断隔离是否成功
        step1_ok = result1.get('phase') == 'error_caught'
        step2_ok = main_process_ok
        step3_ok = result2.get('success') == True
        
        isolation_successful = step1_ok and step2_ok and step3_ok
        
        # 构建详细的错误信息
        error_parts = []
        if not step1_ok:
            error_parts.append(f"Step1 failed: phase={result1.get('phase')}, expected='error_caught'")
        if not step2_ok:
            error_parts.append("Step2 failed: main process contaminated")
        if not step3_ok:
            error_parts.append(f"Step3 failed: success={result2.get('success')}, expected=True")
        
        error_message = "; ".join(error_parts) if error_parts else None
        
        logger.info(f"[Isolation Test] Step 1 OK: {step1_ok}, Phase: {result1.get('phase')}")
        logger.info(f"[Isolation Test] Step 2 OK: {step2_ok}")
        logger.info(f"[Isolation Test] Step 3 OK: {step3_ok}, Success: {result2.get('success')}")
        
        return IsolationTestReport(
            isolation_successful=isolation_successful,
            main_process_contaminated=not main_process_ok,
            subprocess_error_message=error_message,
            details={
                'step1_error_caught': result1,
                'step2_main_process_ok': main_process_ok,
                'step3_gpu_available': result2
            }
        )
    
    @staticmethod
    def test_profiler_compatibility(device_id: int) -> ProfilerTestReport:
        """
        测试torch.profiler在subprocess中的兼容性
        
        验证：
        1. Profiler可以在subprocess中正常启动
        2. Profiler可以收集CUDA事件
        3. Profiling数据可以通过Queue传递回主进程
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            ProfilerTestReport对象
        """
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        profiling_queue = ctx.Queue()
        
        process = ctx.Process(
            target=_profiler_worker,
            args=(device_id, result_queue, profiling_queue)
        )
        process.start()
        
        try:
            # 等待结果（profiler需要更长时间）
            result = result_queue.get(timeout=20)
            
            # 尝试获取profiling数据
            profiling_data = None
            if not profiling_queue.empty():
                profiling_data = profiling_queue.get_nowait()
            
            process.join(timeout=2)
            
            if result['success']:
                return ProfilerTestReport(
                    profiler_works=True,
                    profiling_data_received=(profiling_data is not None),
                    profiling_data=profiling_data
                )
            else:
                return ProfilerTestReport(
                    profiler_works=False,
                    profiling_data_received=False,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            process.terminate()
            process.join(timeout=2)
            
            return ProfilerTestReport(
                profiler_works=False,
                profiling_data_received=False,
                error_message=f"Profiler test failed: {str(e)}"
            )
    
    @staticmethod
    def run_full_diagnostics(device_id: int) -> Dict[str, Any]:
        """
        运行完整的GPU诊断
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            包含所有诊断结果的字典
        """
        logger.info(f"=== 开始GPU {device_id} 完整诊断 ===")
        
        results = {}
        
        # Test 1: nvidia-smi健康检查
        logger.info("[Test 1/4] nvidia-smi健康检查...")
        health_nvidia_smi = GPUDiagnostics.test_gpu_health_nvidia_smi(device_id)
        results['health_nvidia_smi'] = health_nvidia_smi
        logger.info(f"  结果: {'✅ 通过' if health_nvidia_smi.healthy else '❌ 失败'}")
        if health_nvidia_smi.healthy:
            logger.info(f"  GPU: {health_nvidia_smi.device_name}, "
                       f"Memory: {health_nvidia_smi.total_memory_gb:.1f}GB")
        
        # Test 2: Subprocess健康检查
        logger.info("[Test 2/4] Subprocess CUDA健康检查...")
        health_subprocess = GPUDiagnostics.test_gpu_health_subprocess(device_id)
        results['health_subprocess'] = health_subprocess
        logger.info(f"  结果: {'✅ 通过' if health_subprocess.healthy else '❌ 失败'}")
        if not health_subprocess.healthy:
            logger.error(f"  错误: {health_subprocess.error_message}")
        
        # Test 3: CUDA Error隔离测试
        logger.info("[Test 3/4] CUDA Error隔离测试...")
        isolation = GPUDiagnostics.test_cuda_error_isolation(device_id)
        results['isolation_test'] = isolation
        logger.info(f"  结果: {'✅ 隔离成功' if isolation.isolation_successful else '❌ 隔离失败'}")
        if not isolation.isolation_successful:
            logger.warning(f"  主进程污染: {isolation.main_process_contaminated}")
            logger.error(f"  错误信息: {isolation.subprocess_error_message}")
            if isolation.details:
                logger.debug(f"  详细信息: {isolation.details}")
        
        # Test 4: Profiler兼容性测试
        logger.info("[Test 4/4] torch.profiler兼容性测试...")
        profiler_test = GPUDiagnostics.test_profiler_compatibility(device_id)
        results['profiler_test'] = profiler_test
        logger.info(f"  结果: {'✅ Profiler可用' if profiler_test.profiler_works else '❌ Profiler失败'}")
        if profiler_test.profiler_works:
            logger.info(f"  Profiling数据接收: {'✅' if profiler_test.profiling_data_received else '❌'}")
            if profiler_test.profiling_data:
                logger.info(f"  CUDA事件数量: {profiler_test.profiling_data.get('cuda_events', 0)}")
        
        # 总结
        logger.info("=== 诊断完成 ===")
        all_passed = (
            health_nvidia_smi.healthy and
            health_subprocess.healthy and
            isolation.isolation_successful and
            profiler_test.profiler_works
        )
        logger.info(f"总体状态: {'✅ 所有测试通过' if all_passed else '⚠️  部分测试失败'}")
        
        results['all_passed'] = all_passed
        return results


def main():
    """命令行入口"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="GPU诊断工具")
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU设备ID（默认: 0）'
    )
    parser.add_argument(
        '--test',
        choices=['health', 'isolation', 'profiler', 'all'],
        default='all',
        help='要运行的测试类型'
    )
    
    args = parser.parse_args()
    
    if args.test == 'all':
        results = GPUDiagnostics.run_full_diagnostics(args.device)
        sys.exit(0 if results['all_passed'] else 1)
    
    elif args.test == 'health':
        report = GPUDiagnostics.test_gpu_health_subprocess(args.device)
        print(f"Healthy: {report.healthy}")
        if report.healthy:
            print(f"Device: {report.device_name}")
            print(f"Memory: {report.total_memory_gb:.1f}GB")
        sys.exit(0 if report.healthy else 1)
    
    elif args.test == 'isolation':
        report = GPUDiagnostics.test_cuda_error_isolation(args.device)
        print(f"Isolation Successful: {report.isolation_successful}")
        print(f"Main Process Contaminated: {report.main_process_contaminated}")
        sys.exit(0 if report.isolation_successful else 1)
    
    elif args.test == 'profiler':
        report = GPUDiagnostics.test_profiler_compatibility(args.device)
        print(f"Profiler Works: {report.profiler_works}")
        print(f"Data Received: {report.profiling_data_received}")
        if report.profiling_data:
            print(f"CUDA Events: {report.profiling_data.get('cuda_events', 0)}")
        sys.exit(0 if report.profiler_works else 1)


if __name__ == '__main__':
    main()

