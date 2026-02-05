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
配置热重载机制
支持运行时配置更新和服务器动态管理
"""

import asyncio
import os
import time
import logging
import yaml
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib


@dataclass
class ServerConfig:
    """服务器配置"""
    url: str
    weight: float = 1.0
    max_connections: int = 20
    timeout: float = 300.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        # 验证配置
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid server URL: {self.url}")
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive: {self.weight}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive: {self.timeout}")


@dataclass
class HotReloadConfig:
    """热重载配置"""
    servers: List[ServerConfig] = field(default_factory=list)
    load_balancing_strategy: str = "health_based"
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    max_retries: int = 3
    auto_failover: bool = True
    graceful_shutdown: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HotReloadConfig":
        """从字典创建配置"""
        servers = []
        for server_data in config_dict.get("servers", []):
            if isinstance(server_data, str):
                servers.append(ServerConfig(url=server_data))
            elif isinstance(server_data, dict):
                servers.append(ServerConfig(**server_data))
        
        return cls(
            servers=servers,
            load_balancing_strategy=config_dict.get("load_balancing_strategy", "health_based"),
            health_check_interval=config_dict.get("health_check_interval", 30.0),
            health_check_timeout=config_dict.get("health_check_timeout", 5.0),
            max_retries=config_dict.get("max_retries", 3),
            auto_failover=config_dict.get("auto_failover", True),
            graceful_shutdown=config_dict.get("graceful_shutdown", True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "servers": [
                {
                    "url": server.url,
                    "weight": server.weight,
                    "max_connections": server.max_connections,
                    "timeout": server.timeout,
                    "metadata": server.metadata,
                    "enabled": server.enabled
                }
                for server in self.servers
            ],
            "load_balancing_strategy": self.load_balancing_strategy,
            "health_check_interval": self.health_check_interval,
            "health_check_timeout": self.health_check_timeout,
            "max_retries": self.max_retries,
            "auto_failover": self.auto_failover,
            "graceful_shutdown": self.graceful_shutdown
        }


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控器"""
    
    def __init__(self, config_path: Path, callback: Callable[[Path], None]):
        self.config_path = config_path
        self.callback = callback
        self.last_modified = 0
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """文件修改事件"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.name == self.config_path.name:
            # 防止重复触发
            current_time = time.time()
            if current_time - self.last_modified < 1.0:
                return
            
            self.last_modified = current_time
            self.logger.info(f"Config file modified: {file_path}")
            
            try:
                self.callback(file_path)
            except Exception as e:
                self.logger.error(f"Config reload callback failed: {e}")


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_server_config(config: ServerConfig) -> List[str]:
        """验证服务器配置"""
        errors = []
        
        # URL 验证
        if not config.url.startswith(('http://', 'https://')):
            errors.append(f"Invalid server URL: {config.url}")
        
        # 权重验证
        if config.weight <= 0:
            errors.append(f"Weight must be positive: {config.weight}")
        
        # 超时验证
        if config.timeout <= 0:
            errors.append(f"Timeout must be positive: {config.timeout}")
        
        # 连接数验证
        if config.max_connections <= 0:
            errors.append(f"Max connections must be positive: {config.max_connections}")
        
        return errors
    
    @staticmethod
    def validate_hot_reload_config(config: HotReloadConfig) -> List[str]:
        """验证热重载配置"""
        errors = []
        
        # 验证服务器列表
        if not config.servers:
            errors.append("At least one server must be configured")
        
        for i, server in enumerate(config.servers):
            server_errors = ConfigValidator.validate_server_config(server)
            for error in server_errors:
                errors.append(f"Server {i}: {error}")
        
        # 验证策略
        valid_strategies = ["round_robin", "weighted_round_robin", "least_connections", 
                          "fastest_response", "health_based"]
        if config.load_balancing_strategy not in valid_strategies:
            errors.append(f"Invalid load balancing strategy: {config.load_balancing_strategy}")
        
        # 验证健康检查参数
        if config.health_check_interval <= 0:
            errors.append(f"Health check interval must be positive: {config.health_check_interval}")
        
        if config.health_check_timeout <= 0:
            errors.append(f"Health check timeout must be positive: {config.health_check_timeout}")
        
        if config.max_retries <= 0:
            errors.append(f"Max retries must be positive: {config.max_retries}")
        
        return errors


class HotReloadManager:
    """热重载管理器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化热重载管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path) if config_path else None
        self.current_config: Optional[HotReloadConfig] = None
        self.config_hash = ""
        
        # 文件监控
        self.observer: Optional[Observer] = None
        self.file_watcher: Optional[ConfigFileWatcher] = None
        
        # 事件回调
        self.change_callbacks: List[Callable[[HotReloadConfig, HotReloadConfig], None]] = []
        
        # 线程安全
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        
        # 加载初始配置
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """加载配置文件"""
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            self.logger.error(f"Config file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    config_data = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    self.logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                    return False
            
            # 计算配置哈希
            config_str = json.dumps(config_data, sort_keys=True)
            new_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            # 检查是否有变化
            if new_hash == self.config_hash:
                self.logger.debug("Config unchanged, skipping reload")
                return True
            
            # 解析配置
            new_config = HotReloadConfig.from_dict(config_data.get('kernel', {}).get('servers', {}))
            
            # 验证配置
            errors = ConfigValidator.validate_hot_reload_config(new_config)
            if errors:
                self.logger.error(f"Config validation failed: {errors}")
                return False
            
            # 更新配置
            with self._lock:
                old_config = self.current_config
                self.current_config = new_config
                self.config_hash = new_hash
                
                # 触发变化回调
                if old_config:
                    for callback in self.change_callbacks:
                        try:
                            callback(old_config, new_config)
                        except Exception as e:
                            self.logger.error(f"Config change callback failed: {e}")
            
            self.logger.info(f"Config reloaded successfully: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return False
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """保存配置文件"""
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.current_config:
            self.logger.error("No config path or current config available")
            return False
        
        try:
            config_data = {
                'kernel': {
                    'servers': self.current_config.to_dict()
                }
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    yaml.safe_dump(config_data, f, default_flow_style=False)
                elif self.config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    self.logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                    return False
            
            self.logger.info(f"Config saved successfully: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False
    
    def start_watching(self) -> bool:
        """开始监控配置文件"""
        if not self.config_path:
            self.logger.error("No config path specified")
            return False
        
        if self.observer:
            self.logger.warning("Already watching config file")
            return True
        
        try:
            self.file_watcher = ConfigFileWatcher(self.config_path, self._on_config_changed)
            self.observer = Observer()
            self.observer.schedule(self.file_watcher, str(self.config_path.parent), recursive=False)
            self.observer.start()
            
            self.logger.info(f"Started watching config file: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start watching config: {e}")
            return False
    
    def stop_watching(self):
        """停止监控配置文件"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.file_watcher = None
            self.logger.info("Stopped watching config file")
    
    def _on_config_changed(self, config_path: Path):
        """配置文件变化回调"""
        self.logger.info(f"Config file changed, reloading: {config_path}")
        self.load_config()
    
    def register_change_callback(self, callback: Callable[[HotReloadConfig, HotReloadConfig], None]):
        """注册配置变化回调"""
        self.change_callbacks.append(callback)
    
    def update_servers(self, servers: List[ServerConfig]) -> bool:
        """更新服务器配置"""
        with self._lock:
            if not self.current_config:
                self.logger.error("No current config available")
                return False
            
            # 创建新配置
            new_config = HotReloadConfig(
                servers=servers,
                load_balancing_strategy=self.current_config.load_balancing_strategy,
                health_check_interval=self.current_config.health_check_interval,
                health_check_timeout=self.current_config.health_check_timeout,
                max_retries=self.current_config.max_retries,
                auto_failover=self.current_config.auto_failover,
                graceful_shutdown=self.current_config.graceful_shutdown
            )
            
            # 验证配置
            errors = ConfigValidator.validate_hot_reload_config(new_config)
            if errors:
                self.logger.error(f"Server config validation failed: {errors}")
                return False
            
            # 更新配置
            old_config = self.current_config
            self.current_config = new_config
            
            # 触发变化回调
            for callback in self.change_callbacks:
                try:
                    callback(old_config, new_config)
                except Exception as e:
                    self.logger.error(f"Config change callback failed: {e}")
            
            self.logger.info(f"Server config updated: {len(servers)} servers")
            return True
    
    def get_config(self) -> Optional[HotReloadConfig]:
        """获取当前配置"""
        with self._lock:
            return self.current_config
    
    def get_servers(self) -> List[ServerConfig]:
        """获取服务器配置列表"""
        with self._lock:
            return self.current_config.servers if self.current_config else []
    
    def add_server(self, server: ServerConfig) -> bool:
        """添加服务器"""
        with self._lock:
            if not self.current_config:
                return False
            
            # 检查是否已存在
            if any(s.url == server.url for s in self.current_config.servers):
                self.logger.warning(f"Server already exists: {server.url}")
                return False
            
            # 验证服务器配置
            errors = ConfigValidator.validate_server_config(server)
            if errors:
                self.logger.error(f"Server validation failed: {errors}")
                return False
            
            # 添加服务器
            new_servers = self.current_config.servers + [server]
            return self.update_servers(new_servers)
    
    def remove_server(self, server_url: str) -> bool:
        """移除服务器"""
        with self._lock:
            if not self.current_config:
                return False
            
            # 过滤服务器
            new_servers = [s for s in self.current_config.servers if s.url != server_url]
            
            if len(new_servers) == len(self.current_config.servers):
                self.logger.warning(f"Server not found: {server_url}")
                return False
            
            return self.update_servers(new_servers)
    
    def close(self):
        """关闭管理器"""
        self.stop_watching()
        self.logger.info("HotReloadManager closed")