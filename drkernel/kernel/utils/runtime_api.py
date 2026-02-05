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
运行时服务器管理API
提供 REST API 接口用于动态管理服务器
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from uvicorn import Config, Server
import threading


class ServerAddRequest(BaseModel):
    """添加服务器请求"""
    url: str = Field(..., description="服务器URL")
    weight: float = Field(1.0, ge=0.1, le=10.0, description="权重")
    max_connections: int = Field(20, ge=1, le=1000, description="最大连接数")
    timeout: float = Field(300.0, ge=1.0, le=3600.0, description="超时时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    enabled: bool = Field(True, description="是否启用")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class ServerUpdateRequest(BaseModel):
    """更新服务器请求"""
    weight: Optional[float] = Field(None, ge=0.1, le=10.0, description="权重")
    max_connections: Optional[int] = Field(None, ge=1, le=1000, description="最大连接数")
    timeout: Optional[float] = Field(None, ge=1.0, le=3600.0, description="超时时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    enabled: Optional[bool] = Field(None, description="是否启用")


class ServerResponse(BaseModel):
    """服务器响应"""
    url: str
    status: str
    weight: float
    max_connections: int
    timeout: float
    metadata: Dict[str, Any]
    enabled: bool
    health_score: float
    response_time: float
    success_rate: float
    active_connections: int
    total_requests: int
    last_check: str


class ServerStatsResponse(BaseModel):
    """服务器统计响应"""
    total_servers: int
    healthy_servers: int
    unhealthy_servers: int
    total_requests: int
    total_errors: int
    average_response_time: float
    servers: List[ServerResponse]


class RuntimeServerAPI:
    """运行时服务器管理API"""
    
    def __init__(self, 
                 server_manager=None,
                 config_manager=None,
                 host: str = "0.0.0.0",
                 port: int = 8888,
                 enable_auth: bool = False,
                 api_key: Optional[str] = None):
        """
        初始化运行时API
        
        Args:
            server_manager: 动态服务器管理器
            config_manager: 配置管理器
            host: API服务器主机
            port: API服务器端口
            enable_auth: 是否启用认证
            api_key: API密钥
        """
        self.server_manager = server_manager
        self.config_manager = config_manager
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.api_key = api_key
        
        # 创建 FastAPI 应用
        self.app = FastAPI(
            title="Kernel Server Manager API",
            description="Dynamic server management API for kernel training",
            version="1.0.0"
        )
        
        # 注册路由
        self._register_routes()
        
        # API 服务器
        self.server: Optional[Server] = None
        self.server_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.get("/")
        async def root():
            """根路径"""
            return {"message": "Kernel Server Manager API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health():
            """健康检查"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/servers", response_model=List[ServerResponse])
        async def get_servers():
            """获取所有服务器"""
            if not self.server_manager:
                raise HTTPException(status_code=503, detail="Server manager not available")
            
            servers = []
            server_stats = self.server_manager.server_pool.get_server_stats()
            
            for url, stats in server_stats.get("servers", {}).items():
                server_info = self.server_manager.server_pool.servers.get(url)
                if server_info:
                    servers.append(ServerResponse(
                        url=url,
                        status=stats["status"],
                        weight=server_info.metadata.get("weight", 1.0),
                        max_connections=server_info.metadata.get("max_connections", 20),
                        timeout=server_info.metadata.get("timeout", 300.0),
                        metadata=server_info.metadata,
                        enabled=server_info.metadata.get("enabled", True),
                        health_score=stats["health_score"],
                        response_time=stats["response_time"],
                        success_rate=stats["success_rate"],
                        active_connections=stats["active_connections"],
                        total_requests=stats["total_requests"],
                        last_check=datetime.fromtimestamp(server_info.last_check).isoformat()
                    ))
            
            return servers
        
        @self.app.get("/servers/{server_url:path}", response_model=ServerResponse)
        async def get_server(server_url: str):
            """获取特定服务器信息"""
            if not self.server_manager:
                raise HTTPException(status_code=503, detail="Server manager not available")
            
            # URL 解码
            import urllib.parse
            server_url = urllib.parse.unquote(server_url)
            
            server_info = self.server_manager.server_pool.servers.get(server_url)
            if not server_info:
                raise HTTPException(status_code=404, detail="Server not found")
            
            server_stats = self.server_manager.server_pool.get_server_stats()
            stats = server_stats.get("servers", {}).get(server_url, {})
            
            return ServerResponse(
                url=server_url,
                status=stats.get("status", "unknown"),
                weight=server_info.metadata.get("weight", 1.0),
                max_connections=server_info.metadata.get("max_connections", 20),
                timeout=server_info.metadata.get("timeout", 300.0),
                metadata=server_info.metadata,
                enabled=server_info.metadata.get("enabled", True),
                health_score=stats.get("health_score", 0.0),
                response_time=stats.get("response_time", 0.0),
                success_rate=stats.get("success_rate", 0.0),
                active_connections=stats.get("active_connections", 0),
                total_requests=stats.get("total_requests", 0),
                last_check=datetime.fromtimestamp(server_info.last_check).isoformat()
            )
        
        @self.app.post("/servers", response_model=dict)
        async def add_server(request: ServerAddRequest):
            """添加服务器"""
            if not self.server_manager:
                raise HTTPException(status_code=503, detail="Server manager not available")
            
            metadata = request.metadata.copy()
            metadata.update({
                "weight": request.weight,
                "max_connections": request.max_connections,
                "timeout": request.timeout,
                "enabled": request.enabled
            })
            
            success = self.server_manager.add_server(request.url, metadata)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to add server")
            
            # 如果有配置管理器，更新配置
            if self.config_manager:
                from .config_manager import ServerConfig
                server_config = ServerConfig(
                    url=request.url,
                    weight=request.weight,
                    max_connections=request.max_connections,
                    timeout=request.timeout,
                    metadata=request.metadata,
                    enabled=request.enabled
                )
                self.config_manager.add_server(server_config)
            
            return {"message": f"Server {request.url} added successfully"}
        
        @self.app.put("/servers/{server_url:path}", response_model=dict)
        async def update_server(server_url: str, request: ServerUpdateRequest):
            """更新服务器配置"""
            if not self.server_manager:
                raise HTTPException(status_code=503, detail="Server manager not available")
            
            # URL 解码
            import urllib.parse
            server_url = urllib.parse.unquote(server_url)
            
            server_info = self.server_manager.server_pool.servers.get(server_url)
            if not server_info:
                raise HTTPException(status_code=404, detail="Server not found")
            
            # 更新元数据
            if request.weight is not None:
                server_info.metadata["weight"] = request.weight
            if request.max_connections is not None:
                server_info.metadata["max_connections"] = request.max_connections
            if request.timeout is not None:
                server_info.metadata["timeout"] = request.timeout
            if request.metadata is not None:
                server_info.metadata.update(request.metadata)
            if request.enabled is not None:
                server_info.metadata["enabled"] = request.enabled
            
            return {"message": f"Server {server_url} updated successfully"}
        
        @self.app.delete("/servers/{server_url:path}", response_model=dict)
        async def remove_server(server_url: str, graceful: bool = True):
            """移除服务器"""
            if not self.server_manager:
                raise HTTPException(status_code=503, detail="Server manager not available")
            
            # URL 解码
            import urllib.parse
            server_url = urllib.parse.unquote(server_url)
            
            success = self.server_manager.remove_server(server_url, graceful)
            if not success:
                raise HTTPException(status_code=404, detail="Server not found")
            
            # 如果有配置管理器，更新配置
            if self.config_manager:
                self.config_manager.remove_server(server_url)
            
            return {"message": f"Server {server_url} removed successfully"}
        
        @self.app.get("/stats", response_model=ServerStatsResponse)
        async def get_stats():
            """获取统计信息"""
            if not self.server_manager:
                raise HTTPException(status_code=503, detail="Server manager not available")
            
            status = self.server_manager.get_status()
            server_stats = status["server_pool"]
            
            servers = []
            total_requests = 0
            total_errors = 0
            response_times = []
            
            for url, stats in server_stats.get("servers", {}).items():
                server_info = self.server_manager.server_pool.servers.get(url)
                if server_info:
                    servers.append(ServerResponse(
                        url=url,
                        status=stats["status"],
                        weight=server_info.metadata.get("weight", 1.0),
                        max_connections=server_info.metadata.get("max_connections", 20),
                        timeout=server_info.metadata.get("timeout", 300.0),
                        metadata=server_info.metadata,
                        enabled=server_info.metadata.get("enabled", True),
                        health_score=stats["health_score"],
                        response_time=stats["response_time"],
                        success_rate=stats["success_rate"],
                        active_connections=stats["active_connections"],
                        total_requests=stats["total_requests"],
                        last_check=datetime.fromtimestamp(server_info.last_check).isoformat()
                    ))
                    
                    total_requests += stats["total_requests"]
                    total_errors += int(stats["total_requests"] * (1 - stats["success_rate"]))
                    if stats["response_time"] > 0:
                        response_times.append(stats["response_time"])
            
            return ServerStatsResponse(
                total_servers=server_stats["total_servers"],
                healthy_servers=server_stats["healthy_servers"],
                unhealthy_servers=server_stats["total_servers"] - server_stats["healthy_servers"],
                total_requests=total_requests,
                total_errors=total_errors,
                average_response_time=sum(response_times) / len(response_times) if response_times else 0.0,
                servers=servers
            )
        
        @self.app.post("/servers/{server_url:path}/health-check", response_model=dict)
        async def trigger_health_check(server_url: str):
            """触发健康检查"""
            if not self.server_manager:
                raise HTTPException(status_code=503, detail="Server manager not available")
            
            # URL 解码
            import urllib.parse
            server_url = urllib.parse.unquote(server_url)
            
            server_info = self.server_manager.server_pool.servers.get(server_url)
            if not server_info:
                raise HTTPException(status_code=404, detail="Server not found")
            
            # 触发单个服务器的健康检查
            healthy, response_time = await self.server_manager._check_server_health(server_info)
            self.server_manager._update_server_health(server_info, healthy, response_time)
            
            return {
                "message": f"Health check triggered for {server_url}",
                "healthy": healthy,
                "response_time": response_time
            }
        
        @self.app.post("/config/reload", response_model=dict)
        async def reload_config():
            """重新加载配置"""
            if not self.config_manager:
                raise HTTPException(status_code=503, detail="Config manager not available")
            
            success = self.config_manager.load_config()
            if not success:
                raise HTTPException(status_code=500, detail="Failed to reload config")
            
            return {"message": "Configuration reloaded successfully"}
        
        @self.app.get("/config", response_model=dict)
        async def get_config():
            """获取当前配置"""
            if not self.config_manager:
                raise HTTPException(status_code=503, detail="Config manager not available")
            
            config = self.config_manager.get_config()
            if not config:
                raise HTTPException(status_code=404, detail="No configuration available")
            
            return config.to_dict()
    
    async def start(self):
        """启动API服务器"""
        if self.server_task:
            self.logger.warning("API server already running")
            return
        
        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        self.server = Server(config)
        self.server_task = asyncio.create_task(self.server.serve())
        
        self.logger.info(f"API server started at http://{self.host}:{self.port}")
    
    async def stop(self):
        """停止API服务器"""
        if self.server:
            self.server.should_exit = True
            
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            self.server_task = None
        
        self.logger.info("API server stopped")
    
    def start_background(self):
        """在后台线程中启动API服务器"""
        def run_server():
            import uvicorn
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        self.logger.info(f"API server started in background at http://{self.host}:{self.port}")
        return thread


# 便捷函数
def create_server_api(server_manager=None, config_manager=None, **kwargs) -> RuntimeServerAPI:
    """创建服务器API实例"""
    return RuntimeServerAPI(
        server_manager=server_manager,
        config_manager=config_manager,
        **kwargs
    )