# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel 训练监控指标收集器
提供详细的性能指标跟踪和分析
"""

import time
import logging
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import statistics
import json


@dataclass
class BatchMetrics:
    """批次处理指标"""
    batch_size: int
    processing_time: float
    success_rate: float
    compilation_rate: float
    correctness_rate: float
    decoy_kernel_rate: float
    avg_speedup: float
    error_count: int
    server_response_time: float
    timestamp: float = field(default_factory=time.time)


class MetricsTracker:
    """
    Kernel 训练指标跟踪器
    提供实时监控和历史指标分析
    """
    
    def __init__(self, window_size: int = 1000, log_interval: int = 100):
        """
        初始化指标跟踪器
        
        Args:
            window_size: 滑动窗口大小
            log_interval: 日志输出间隔
        """
        self.window_size = window_size
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        
        # 指标存储
        self.batch_metrics = deque(maxlen=window_size)
        self.request_metrics = defaultdict(lambda: deque(maxlen=window_size))
        
        # 累积统计
        self.total_requests = 0
        self.total_batches = 0
        self.total_errors = 0
        self.total_successes = 0
        
        # 性能指标
        self.compilation_successes = 0
        self.correctness_successes = 0
        self.performance_improvements = []
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 开始时间
        self.start_time = time.time()
        
        self.logger.info("MetricsTracker initialized with window_size=%d", window_size)
    
    def record_batch_metrics(self, batch_metrics: BatchMetrics):
        """记录批次指标"""
        with self._lock:
            self.batch_metrics.append(batch_metrics)
            self.total_batches += 1
            self.total_requests += batch_metrics.batch_size
            self.total_errors += batch_metrics.error_count
            self.total_successes += int(batch_metrics.batch_size - batch_metrics.error_count)
            
            # 更新性能指标
            if batch_metrics.compilation_rate > 0:
                self.compilation_successes += int(batch_metrics.batch_size * batch_metrics.compilation_rate)
            if batch_metrics.correctness_rate > 0:
                self.correctness_successes += int(batch_metrics.batch_size * batch_metrics.correctness_rate)
            if batch_metrics.avg_speedup > 0:
                self.performance_improvements.append(batch_metrics.avg_speedup)
            
            # 定期输出日志
            if self.total_batches % self.log_interval == 0:
                self._log_current_stats()
    
    def record_request_metrics(self, request_type: str, duration: float, 
                             success: bool, metadata: Optional[Dict] = None):
        """记录单个请求指标"""
        with self._lock:
            metric = {
                'duration': duration,
                'success': success,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self.request_metrics[request_type].append(metric)
    
    @contextmanager
    def measure_batch_processing(self, batch_size: int):
        """上下文管理器，用于测量批次处理性能"""
        start_time = time.time()
        error_count = 0
        success_count = 0
        compilation_count = 0
        correctness_count = 0
        decoy_kernel_count = 0
        speedups = []
        
        try:
            yield {
                'add_error': lambda: self._increment_counter('error_count', locals()),
                'add_success': lambda speedup=0, compiled=False, correct=False, decoy_kernel=False: 
                    self._record_success(locals(), speedup, compiled, correct, decoy_kernel),
            }
        finally:
            processing_time = time.time() - start_time
            
            # 构造批次指标
            batch_metrics = BatchMetrics(
                batch_size=batch_size,
                processing_time=processing_time,
                success_rate=success_count / batch_size if batch_size > 0 else 0,
                compilation_rate=compilation_count / batch_size if batch_size > 0 else 0,
                correctness_rate=correctness_count / batch_size if batch_size > 0 else 0,
                decoy_kernel_rate=decoy_kernel_count / batch_size if batch_size > 0 else 0,
                avg_speedup=statistics.mean(speedups) if speedups else 0,
                error_count=error_count,
                server_response_time=processing_time  # 简化处理
            )
            
            self.record_batch_metrics(batch_metrics)
    
    def _increment_counter(self, counter_name: str, local_vars: dict):
        """增加错误计数"""
        local_vars['error_count'] += 1
    
    def _record_success(self, local_vars: dict, speedup: float, compiled: bool, correct: bool, decoy_kernel: bool):
        """记录成功结果"""
        local_vars['success_count'] += 1
        if compiled:
            local_vars['compilation_count'] += 1
        if correct:
            local_vars['correctness_count'] += 1
        if decoy_kernel:
            local_vars['decoy_kernel_count'] += 1
        if speedup > 0:
            local_vars['speedups'].append(speedup)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        with self._lock:
            runtime = time.time() - self.start_time
            
            # 基础统计
            stats = {
                'runtime_seconds': runtime,
                'total_requests': self.total_requests,
                'total_batches': self.total_batches,
                'total_errors': self.total_errors,
                'total_successes': self.total_successes,
                'overall_success_rate': self.total_successes / self.total_requests if self.total_requests > 0 else 0,
                'requests_per_second': self.total_requests / runtime if runtime > 0 else 0,
            }
            
            # 性能指标
            stats.update({
                'compilation_success_rate': self.compilation_successes / self.total_requests if self.total_requests > 0 else 0,
                'correctness_success_rate': self.correctness_successes / self.total_requests if self.total_requests > 0 else 0,
                'avg_performance_improvement': statistics.mean(self.performance_improvements) if self.performance_improvements else 0,
                'performance_improvement_count': len(self.performance_improvements),
            })
            
            # 最近批次统计
            if self.batch_metrics:
                recent_batches = list(self.batch_metrics)[-50:]  # 最近50个批次
                stats.update({
                    'recent_avg_batch_size': statistics.mean([b.batch_size for b in recent_batches]),
                    'recent_avg_processing_time': statistics.mean([b.processing_time for b in recent_batches]),
                    'recent_avg_success_rate': statistics.mean([b.success_rate for b in recent_batches]),
                    'recent_avg_compilation_rate': statistics.mean([b.compilation_rate for b in recent_batches]),
                    'recent_avg_correctness_rate': statistics.mean([b.correctness_rate for b in recent_batches]),
                    'recent_avg_speedup': statistics.mean([b.avg_speedup for b in recent_batches if b.avg_speedup > 0]),
                })
            
            return stats
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """获取性能趋势数据"""
        with self._lock:
            if not self.batch_metrics:
                return {}
            
            recent_batches = list(self.batch_metrics)[-100:]  # 最近100个批次
            
            return {
                'timestamps': [b.timestamp for b in recent_batches],
                'success_rates': [b.success_rate for b in recent_batches],
                'compilation_rates': [b.compilation_rate for b in recent_batches],
                'correctness_rates': [b.correctness_rate for b in recent_batches],
                'avg_speedups': [b.avg_speedup for b in recent_batches],
                'processing_times': [b.processing_time for b in recent_batches],
                'batch_sizes': [b.batch_size for b in recent_batches],
            }
    
    def _log_current_stats(self):
        """输出当前统计信息到日志"""
        stats = self.get_current_stats()
        
        self.logger.info(
            "Kernel Training Metrics - "
            "Batches: %d, Requests: %d, Success Rate: %.2f%%, "
            "Compilation Rate: %.2f%%, Correctness Rate: %.2f%%, "
            "Avg Speedup: %.2fx, RPS: %.2f",
            stats['total_batches'],
            stats['total_requests'],
            stats['overall_success_rate'] * 100,
            stats['compilation_success_rate'] * 100,
            stats['correctness_success_rate'] * 100,
            stats['avg_performance_improvement'],
            stats['requests_per_second']
        )
    
    def export_metrics(self, filepath: str):
        """导出指标到文件"""
        with self._lock:
            export_data = {
                'stats': self.get_current_stats(),
                'trends': self.get_performance_trends(),
                'batch_metrics': [
                    {
                        'batch_size': b.batch_size,
                        'processing_time': b.processing_time,
                        'success_rate': b.success_rate,
                        'compilation_rate': b.compilation_rate,
                        'correctness_rate': b.correctness_rate,
                        'avg_speedup': b.avg_speedup,
                        'timestamp': b.timestamp
                    }
                    for b in self.batch_metrics
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info("Metrics exported to %s", filepath)
    
    def reset_metrics(self):
        """重置所有指标"""
        with self._lock:
            self.batch_metrics.clear()
            self.request_metrics.clear()
            self.total_requests = 0
            self.total_batches = 0
            self.total_errors = 0
            self.total_successes = 0
            self.compilation_successes = 0
            self.correctness_successes = 0
            self.performance_improvements.clear()
            self.start_time = time.time()
            
            self.logger.info("All metrics reset")


class PerformanceAnalyzer:
    """性能分析器，提供深度分析功能"""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        self.metrics_tracker = metrics_tracker
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """分析性能模式"""
        trends = self.metrics_tracker.get_performance_trends()
        
        if not trends or not trends.get('timestamps'):
            return {'error': 'Insufficient data for analysis'}
        
        analysis = {}
        
        # 成功率趋势分析
        success_rates = trends['success_rates']
        if success_rates:
            analysis['success_rate_trend'] = {
                'current': success_rates[-1],
                'average': statistics.mean(success_rates),
                'improving': success_rates[-1] > statistics.mean(success_rates[-10:]) if len(success_rates) >= 10 else False
            }
        
        # 性能改进趋势
        speedups = [s for s in trends['avg_speedups'] if s > 0]
        if speedups:
            analysis['speedup_trend'] = {
                'current': speedups[-1],
                'average': statistics.mean(speedups),
                'best': max(speedups),
                'improving': speedups[-1] > statistics.mean(speedups[-5:]) if len(speedups) >= 5 else False
            }
        
        # 处理时间分析
        processing_times = trends['processing_times']
        if processing_times:
            analysis['processing_time_trend'] = {
                'current': processing_times[-1],
                'average': statistics.mean(processing_times),
                'p95': sorted(processing_times)[int(len(processing_times) * 0.95)],
                'stable': statistics.stdev(processing_times[-10:]) < 1.0 if len(processing_times) >= 10 else False
            }
        
        return analysis
    
    def get_performance_recommendations(self) -> List[str]:
        """获取性能优化建议"""
        stats = self.metrics_tracker.get_current_stats()
        recommendations = []
        
        # 基于成功率的建议
        if stats['overall_success_rate'] < 0.8:
            recommendations.append("Overall success rate is low. Consider reviewing kernel generation prompts.")
        
        # 基于编译率的建议
        if stats['compilation_success_rate'] < 0.9:
            recommendations.append("Compilation success rate is low. Consider improving code generation quality.")
        
        # 基于正确性的建议
        if stats['correctness_success_rate'] < 0.7:
            recommendations.append("Correctness rate is low. Consider adjusting reward weights or training data.")
        
        # 基于性能改进的建议
        if stats['avg_performance_improvement'] < 1.2:
            recommendations.append("Average performance improvement is low. Consider focusing on optimization strategies.")
        
        # 基于处理时间的建议
        if 'recent_avg_processing_time' in stats and stats['recent_avg_processing_time'] > 30:
            recommendations.append("Processing time is high. Consider optimizing batch size or server configuration.")
        
        return recommendations or ["Performance is looking good! Keep up the training."]