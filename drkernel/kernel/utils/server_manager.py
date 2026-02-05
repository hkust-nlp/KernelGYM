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
动态服务器管理器
支持热插拔服务器、健康检查、自动故障切换和负载均衡
"""

import asyncio
import time
import logging
import httpx
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
import random


class ServerStatus(Enum):
    """服务器状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"  # 正在排空连接
    OFFLINE = "offline"


@dataclass
class ServerInfo:
    """服务器信息"""
    url: str
    status: ServerStatus = ServerStatus.UNKNOWN
    last_check: float = field(default_factory=time.time)
    response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    active_connections: int = 0
    total_requests: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def health_score(self) -> float:
        """计算健康分数 (0-1)"""
        if self.status == ServerStatus.OFFLINE:
            return 0.0
        elif self.status == ServerStatus.UNHEALTHY:
            return 0.1
        elif self.status == ServerStatus.DRAINING:
            return 0.3
        elif self.status == ServerStatus.HEALTHY:
            # 基于响应时间和成功率计算分数
            time_score = max(0, 1 - (self.response_time / 30.0))  # 30秒为满分界限
            success_score = self.success_rate
            return (time_score + success_score) / 2
        return 0.0


class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    FASTEST_RESPONSE = "fastest_response"
    HEALTH_BASED = "health_based"


class ServerPool:
    """服务器池管理"""
    
    def __init__(self, load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED):
        self.servers: Dict[str, ServerInfo] = {}
        self.strategy = load_balancing_strategy
        self.round_robin_index = 0
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_server(self, server_url: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加服务器"""
        with self._lock:
            if server_url in self.servers:
                self.logger.warning(f"Server {server_url} already exists")
                return False
            
            self.servers[server_url] = ServerInfo(
                url=server_url,
                status=ServerStatus.HEALTHY,  # Set to healthy by default for immediate availability
                metadata=metadata or {}
            )
            self.logger.info(f"Added server: {server_url}")
            return True
    
    def remove_server(self, server_url: str, graceful: bool = True) -> bool:
        """移除服务器"""
        with self._lock:
            if server_url not in self.servers:
                self.logger.warning(f"Server {server_url} not found")
                return False
            
            if graceful:
                # 优雅下线：设置为排空状态
                self.servers[server_url].status = ServerStatus.DRAINING
                self.logger.info(f"Server {server_url} is draining")
                return True
            else:
                # 立即移除
                del self.servers[server_url]
                self.logger.info(f"Removed server: {server_url}")
                return True
    
    def get_healthy_servers(self) -> List[ServerInfo]:
        """获取健康的服务器"""
        with self._lock:
            return [
                server for server in self.servers.values()
                if server.status == ServerStatus.HEALTHY
            ]
    
    def get_available_servers(self) -> List[ServerInfo]:
        """获取可用的服务器（健康或排空中）"""
        with self._lock:
            return [
                server for server in self.servers.values()
                if server.status in [ServerStatus.HEALTHY, ServerStatus.DRAINING]
            ]
    
    def select_server(self) -> Optional[ServerInfo]:
        """根据负载均衡策略选择服务器"""
        available_servers = self.get_available_servers()
        
        if not available_servers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_servers)
        elif self.strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(available_servers)
        elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based_select(available_servers)
        else:
            return random.choice(available_servers)
    
    def _round_robin_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """轮询选择"""
        with self._lock:
            if not servers:
                return None
            server = servers[self.round_robin_index % len(servers)]
            self.round_robin_index += 1
            return server
    
    def _weighted_round_robin_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """加权轮询选择"""
        weights = [server.health_score for server in servers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(servers)
        
        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return servers[i]
        
        return servers[-1]
    
    def _least_connections_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """最少连接选择"""
        return min(servers, key=lambda s: s.active_connections)
    
    def _fastest_response_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """最快响应选择"""
        healthy_servers = [s for s in servers if s.status == ServerStatus.HEALTHY]
        if not healthy_servers:
            return random.choice(servers)
        
        return min(healthy_servers, key=lambda s: s.response_time)
    
    def _health_based_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """基于健康分数选择"""
        health_scores = [server.health_score for server in servers]
        total_score = sum(health_scores)
        
        if total_score == 0:
            return random.choice(servers)
        
        rand_val = random.uniform(0, total_score)
        cumulative = 0
        for i, score in enumerate(health_scores):
            cumulative += score
            if rand_val <= cumulative:
                return servers[i]
        
        return servers[-1]
    
    def update_server_stats(self, server_url: str, response_time: float, 
                          success: bool, active_connections: int = 0):
        """更新服务器统计信息"""
        with self._lock:
            if server_url not in self.servers:
                return
            
            server = self.servers[server_url]
            server.response_time = response_time
            server.active_connections = active_connections
            server.total_requests += 1
            
            if success:
                server.success_count += 1
                server.error_count = max(0, server.error_count - 1)  # 逐渐恢复
            else:
                server.error_count += 1
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取所有服务器统计信息"""
        with self._lock:
            return {
                "total_servers": len(self.servers),
                "healthy_servers": len([s for s in self.servers.values() if s.status == ServerStatus.HEALTHY]),
                "servers": {
                    url: {
                        "status": server.status.value,
                        "response_time": server.response_time,
                        "success_rate": server.success_rate,
                        "health_score": server.health_score,
                        "active_connections": server.active_connections,
                        "total_requests": server.total_requests
                    }
                    for url, server in self.servers.items()
                }
            }


class DynamicServerManager:
    """动态服务器管理器"""
    
    def __init__(self, 
                 initial_servers: Optional[List[str]] = None,
                 health_check_interval: float = 30.0,
                 health_check_timeout: float = 5.0,
                 max_retries: int = 3,
                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_BASED):
        """
        初始化动态服务器管理器
        
        Args:
            initial_servers: 初始服务器列表
            health_check_interval: 健康检查间隔（秒）
            health_check_timeout: 健康检查超时（秒）
            max_retries: 最大重试次数
            load_balancing_strategy: 负载均衡策略
        """
        self.server_pool = ServerPool(load_balancing_strategy)
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.max_retries = max_retries
        
        # 健康检查相关
        self.health_check_task = None
        self.health_check_running = False
        self.health_check_client = None  # Initialize later to avoid URL parsing issues
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # 统计信息
        self.stats = {
            "total_health_checks": 0,
            "failed_health_checks": 0,
            "server_switches": 0,
            "auto_failovers": 0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # 添加初始服务器
        if initial_servers:
            for server_url in initial_servers:
                self.add_server(server_url)
        
        self.logger.info(f"DynamicServerManager initialized with {len(initial_servers or [])} servers")
    
    def add_server(self, server_url: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """添加服务器"""
        result = self.server_pool.add_server(server_url, metadata)
        if result:
            self._trigger_event("server_added", {"server_url": server_url, "metadata": metadata})
        return result
    
    def remove_server(self, server_url: str, graceful: bool = True) -> bool:
        """移除服务器"""
        result = self.server_pool.remove_server(server_url, graceful)
        if result:
            self._trigger_event("server_removed", {"server_url": server_url, "graceful": graceful})
        return result
    
    def get_server(self) -> Optional[str]:
        """获取可用服务器URL"""
        server = self.server_pool.select_server()
        if server:
            self.stats["server_switches"] += 1
            return server.url
        return None
    
    def update_server_stats(self, server_url: str, response_time: float, success: bool):
        """更新服务器统计信息"""
        self.server_pool.update_server_stats(server_url, response_time, success)
    
    async def start_health_check(self):
        """启动健康检查"""
        if self.health_check_running:
            return
        
        # Initialize health check client when needed
        if self.health_check_client is None:
            self.health_check_client = httpx.AsyncClient(timeout=httpx.Timeout(self.health_check_timeout))
        
        self.health_check_running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("Health check started")
    
    async def stop_health_check(self):
        """停止健康检查"""
        if not self.health_check_running:
            return
        
        self.health_check_running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health check stopped")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self.health_check_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # 短暂等待后重试
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        servers = list(self.server_pool.servers.values())
        
        # 并发检查所有服务器
        tasks = [self._check_server_health(server) for server in servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for server, result in zip(servers, results):
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed for {server.url}: {result}")
                self._update_server_health(server, False, 0.0)
            else:
                healthy, response_time = result
                self._update_server_health(server, healthy, response_time)
        
        self.stats["total_health_checks"] += len(servers)
    
    async def _check_server_health(self, server: ServerInfo) -> Tuple[bool, float]:
        """检查单个服务器健康状态"""
        start_time = time.time()
        
        try:
            response = await self.health_check_client.get(f"{server.url}/health")
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # 可以进一步检查响应内容
                return True, response_time
            else:
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time
    
    def _update_server_health(self, server: ServerInfo, healthy: bool, response_time: float):
        """更新服务器健康状态"""
        server.last_check = time.time()
        server.response_time = response_time
        
        old_status = server.status
        
        if healthy:
            server.status = ServerStatus.HEALTHY
            server.error_count = max(0, server.error_count - 1)  # 逐渐恢复
        else:
            server.error_count += 1
            self.stats["failed_health_checks"] += 1
            
            # 根据错误次数决定状态
            if server.error_count >= self.max_retries:
                server.status = ServerStatus.UNHEALTHY
            elif server.status == ServerStatus.UNKNOWN:
                server.status = ServerStatus.UNHEALTHY
        
        # 触发状态变化事件
        if old_status != server.status:
            self._trigger_event("server_status_changed", {
                "server_url": server.url,
                "old_status": old_status.value,
                "new_status": server.status.value
            })
            
            # 自动故障切换
            if server.status == ServerStatus.UNHEALTHY:
                self.stats["auto_failovers"] += 1
                self._trigger_event("server_failed", {"server_url": server.url})
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """注册事件回调"""
        self.event_callbacks[event_type].append(callback)
    
    def _trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """触发事件"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(event_type, event_data)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        server_stats = self.server_pool.get_server_stats()
        return {
            "health_check_running": self.health_check_running,
            "health_check_interval": self.health_check_interval,
            "stats": self.stats,
            "server_pool": server_stats
        }
    
    async def close(self):
        """关闭管理器"""
        await self.stop_health_check()
        await self.health_check_client.aclose()
        self.logger.info("DynamicServerManager closed")