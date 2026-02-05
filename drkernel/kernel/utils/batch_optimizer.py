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
客户端批量处理优化器
提供智能批次管理和请求去重功能
"""

import time
import hashlib
import asyncio
import logging
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import statistics
import threading


@dataclass
class BatchPerformance:
    """批次性能记录"""
    batch_size: int
    response_time: float
    success_rate: float
    timestamp: float
    server_load: float = 0.0


class AdaptiveBatchManager:
    """
    自适应批次管理器
    根据服务器性能动态调整批次大小
    """
    
    def __init__(self, 
                 initial_batch_size: int = 10,
                 min_batch_size: int = 1,
                 max_batch_size: int = 50,
                 adjustment_factor: float = 1.2,
                 performance_window: int = 20):
        """
        初始化自适应批次管理器
        
        Args:
            initial_batch_size: 初始批次大小
            min_batch_size: 最小批次大小
            max_batch_size: 最大批次大小
            adjustment_factor: 调整因子
            performance_window: 性能窗口大小
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.adjustment_factor = adjustment_factor
        self.performance_window = performance_window
        
        # 性能历史记录
        self.performance_history = deque(maxlen=performance_window)
        
        # 调整策略参数
        self.target_response_time = 10.0  # 目标响应时间（秒）
        self.target_success_rate = 0.9    # 目标成功率
        self.adjustment_cooldown = 5      # 调整冷却时间（批次）
        self.last_adjustment = 0
        
        # 线程安全
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("AdaptiveBatchManager initialized with batch_size=%d", initial_batch_size)
    
    def get_optimal_batch_size(self) -> int:
        """获取当前最优批次大小"""
        with self._lock:
            return self.current_batch_size
    
    def record_batch_performance(self, batch_size: int, response_time: float, 
                                success_rate: float, server_load: float = 0.0):
        """记录批次性能"""
        with self._lock:
            performance = BatchPerformance(
                batch_size=batch_size,
                response_time=response_time,
                success_rate=success_rate,
                timestamp=time.time(),
                server_load=server_load
            )
            
            self.performance_history.append(performance)
            
            # 检查是否需要调整批次大小
            if len(self.performance_history) >= 3 and \
               len(self.performance_history) - self.last_adjustment >= self.adjustment_cooldown:
                self._adjust_batch_size()
    
    def _adjust_batch_size(self):
        """调整批次大小"""
        if len(self.performance_history) < 3:
            return
        
        # 分析最近的性能数据
        recent_performances = list(self.performance_history)[-5:]
        avg_response_time = statistics.mean([p.response_time for p in recent_performances])
        avg_success_rate = statistics.mean([p.success_rate for p in recent_performances])
        
        adjustment_needed = False
        new_batch_size = self.current_batch_size
        
        # 响应时间过长，减少批次大小
        if avg_response_time > self.target_response_time * 1.5:
            new_batch_size = max(self.min_batch_size, 
                                int(self.current_batch_size / self.adjustment_factor))
            adjustment_needed = True
            reason = f"High response time: {avg_response_time:.2f}s"
        
        # 成功率过低，减少批次大小
        elif avg_success_rate < self.target_success_rate * 0.8:
            new_batch_size = max(self.min_batch_size, 
                                int(self.current_batch_size / self.adjustment_factor))
            adjustment_needed = True
            reason = f"Low success rate: {avg_success_rate:.2f}"
        
        # 性能良好，尝试增加批次大小
        elif (avg_response_time < self.target_response_time * 0.7 and 
              avg_success_rate > self.target_success_rate):
            new_batch_size = min(self.max_batch_size, 
                                int(self.current_batch_size * self.adjustment_factor))
            adjustment_needed = True
            reason = f"Good performance: {avg_response_time:.2f}s, {avg_success_rate:.2f}"
        
        # 应用调整
        if adjustment_needed and new_batch_size != self.current_batch_size:
            old_batch_size = self.current_batch_size
            self.current_batch_size = new_batch_size
            self.last_adjustment = len(self.performance_history)
            
            self.logger.info(
                "Batch size adjusted: %d -> %d (reason: %s)",
                old_batch_size, new_batch_size, reason
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        with self._lock:
            if not self.performance_history:
                return {}
            
            performances = list(self.performance_history)
            return {
                'current_batch_size': self.current_batch_size,
                'avg_response_time': statistics.mean([p.response_time for p in performances]),
                'avg_success_rate': statistics.mean([p.success_rate for p in performances]),
                'total_batches': len(performances),
                'recent_trend': self._get_recent_trend()
            }
    
    def _get_recent_trend(self) -> str:
        """获取最近的性能趋势"""
        if len(self.performance_history) < 6:
            return "insufficient_data"
        
        recent_5 = list(self.performance_history)[-5:]
        previous_5 = list(self.performance_history)[-10:-5]
        
        recent_avg_time = statistics.mean([p.response_time for p in recent_5])
        previous_avg_time = statistics.mean([p.response_time for p in previous_5])
        
        recent_avg_success = statistics.mean([p.success_rate for p in recent_5])
        previous_avg_success = statistics.mean([p.success_rate for p in previous_5])
        
        if recent_avg_time < previous_avg_time and recent_avg_success > previous_avg_success:
            return "improving"
        elif recent_avg_time > previous_avg_time and recent_avg_success < previous_avg_success:
            return "degrading"
        else:
            return "stable"


class RequestDeduplicator:
    """
    请求去重器
    合并相同的请求，避免重复评估
    """
    
    def __init__(self, cache_ttl: int = 3600):
        """
        初始化请求去重器
        
        Args:
            cache_ttl: 缓存过期时间（秒）
        """
        self.cache_ttl = cache_ttl
        
        # 正在处理的请求
        self.pending_requests: Dict[str, List[asyncio.Future]] = defaultdict(list)
        
        # 本地缓存
        self.local_cache: Dict[str, Tuple[Any, float]] = {}
        
        # 请求统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.duplicate_requests = 0
        
        # 线程安全
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("RequestDeduplicator initialized with cache_ttl=%d", cache_ttl)
    
    def _generate_request_key(self, reference_code: str, kernel_code: str) -> str:
        """生成请求的唯一键"""
        content = f"{reference_code}|||{kernel_code}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """检查缓存是否有效"""
        return time.time() - timestamp < self.cache_ttl
    
    async def deduplicate_request(self, reference_code: str, kernel_code: str, 
                                 request_func, *args, **kwargs) -> Any:
        """
        去重单个请求
        
        Args:
            reference_code: 参考代码
            kernel_code: 内核代码
            request_func: 请求函数
            *args, **kwargs: 传递给请求函数的参数
            
        Returns:
            请求结果
        """
        request_key = self._generate_request_key(reference_code, kernel_code)
        
        # 检查本地缓存
        with self._lock:
            if request_key in self.local_cache:
                result, timestamp = self.local_cache[request_key]
                if self._is_cache_valid(timestamp):
                    self.cache_hits += 1
                    self.logger.debug("Cache hit for request %s", request_key[:8])
                    return result
                else:
                    # 缓存过期，删除
                    del self.local_cache[request_key]
            
            self.cache_misses += 1
        
        # 检查是否有正在处理的相同请求
        with self._lock:
            if request_key in self.pending_requests:
                # 有相同请求正在处理，等待结果
                self.duplicate_requests += 1
                future = asyncio.Future()
                self.pending_requests[request_key].append(future)
                self.logger.debug("Duplicate request detected, waiting for result %s", request_key[:8])
                return await future
            else:
                # 新请求，添加到待处理列表
                self.pending_requests[request_key] = []
        
        try:
            # 执行实际请求
            result = await request_func(*args, **kwargs)
            
            # 缓存结果
            with self._lock:
                self.local_cache[request_key] = (result, time.time())
                
                # 通知等待的请求
                futures = self.pending_requests.pop(request_key, [])
                for future in futures:
                    if not future.done():
                        future.set_result(result)
            
            self.logger.debug("Request completed and cached %s", request_key[:8])
            return result
            
        except Exception as e:
            # 请求失败，通知等待的请求
            with self._lock:
                futures = self.pending_requests.pop(request_key, [])
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
            
            self.logger.error("Request failed %s: %s", request_key[:8], str(e))
            raise
    
    def deduplicate_batch_requests(self, tasks: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[int, str]]:
        """
        去重批次请求
        
        Args:
            tasks: 任务列表，每个任务包含 reference_code, kernel_code
            
        Returns:
            (unique_tasks, duplicate_mapping): 去重后的任务列表和重复映射
        """
        unique_tasks = []
        duplicate_mapping = {}
        seen_keys = {}
        
        for i, task in enumerate(tasks):
            request_key = self._generate_request_key(
                task["reference_code"], 
                task["kernel_code"]
            )
            
            # 检查本地缓存
            with self._lock:
                if request_key in self.local_cache:
                    result, timestamp = self.local_cache[request_key]
                    if self._is_cache_valid(timestamp):
                        self.cache_hits += 1
                        continue
                    else:
                        del self.local_cache[request_key]
            
            if request_key in seen_keys:
                # 重复请求
                duplicate_mapping[i] = seen_keys[request_key]
                self.duplicate_requests += 1
            else:
                # 新请求
                seen_keys[request_key] = len(unique_tasks)
                duplicate_mapping[i] = len(unique_tasks)
                unique_tasks.append(task)
                self.cache_misses += 1
        
        if len(unique_tasks) < len(tasks):
            self.logger.info(
                "Batch deduplication: %d/%d unique requests (%.1f%% reduction)",
                len(unique_tasks), len(tasks), 
                (1 - len(unique_tasks) / len(tasks)) * 100
            )
        
        return unique_tasks, duplicate_mapping
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'duplicate_requests': self.duplicate_requests,
                'cache_size': len(self.local_cache),
                'pending_requests': len(self.pending_requests)
            }
    
    def clear_cache(self):
        """清理缓存"""
        with self._lock:
            self.local_cache.clear()
            self.pending_requests.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.duplicate_requests = 0
            
            self.logger.info("Cache cleared")
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, (_, timestamp) in self.local_cache.items():
                if current_time - timestamp > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.local_cache[key]
        
        if expired_keys:
            self.logger.info("Cleaned up %d expired cache entries", len(expired_keys))


class BatchOptimizer:
    """
    批量优化器
    综合管理批次大小和请求去重
    """
    
    def __init__(self, 
                 initial_batch_size: int = 10,
                 max_batch_size: int = 50,
                 cache_ttl: int = 3600):
        """
        初始化批量优化器
        
        Args:
            initial_batch_size: 初始批次大小
            max_batch_size: 最大批次大小
            cache_ttl: 缓存过期时间
        """
        self.batch_manager = AdaptiveBatchManager(
            initial_batch_size=initial_batch_size,
            max_batch_size=max_batch_size
        )
        self.deduplicator = RequestDeduplicator(cache_ttl=cache_ttl)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("BatchOptimizer initialized")
    
    def optimize_batch(self, tasks: List[Dict[str, str]]) -> Tuple[List[List[Dict[str, str]]], Dict[int, Tuple[int, int]]]:
        """
        优化批次处理
        
        Args:
            tasks: 任务列表
            
        Returns:
            (optimized_batches, task_mapping): 优化后的批次列表和任务映射
        """
        # 1. 请求去重
        unique_tasks, duplicate_mapping = self.deduplicator.deduplicate_batch_requests(tasks)
        
        # 2. 分批处理
        optimal_batch_size = self.batch_manager.get_optimal_batch_size()
        optimized_batches = []
        
        for i in range(0, len(unique_tasks), optimal_batch_size):
            batch = unique_tasks[i:i + optimal_batch_size]
            optimized_batches.append(batch)
        
        # 3. 构建任务映射
        task_mapping = {}
        for original_idx, unique_idx in duplicate_mapping.items():
            batch_idx = unique_idx // optimal_batch_size
            task_idx = unique_idx % optimal_batch_size
            task_mapping[original_idx] = (batch_idx, task_idx)
        
        self.logger.info(
            "Batch optimization: %d tasks -> %d unique tasks in %d batches",
            len(tasks), len(unique_tasks), len(optimized_batches)
        )
        
        return optimized_batches, task_mapping
    
    def record_batch_performance(self, batch_size: int, response_time: float, success_rate: float):
        """记录批次性能"""
        self.batch_manager.record_batch_performance(batch_size, response_time, success_rate)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'batch_manager': self.batch_manager.get_performance_stats(),
            'deduplicator': self.deduplicator.get_cache_stats()
        }