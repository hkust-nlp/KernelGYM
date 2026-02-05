"""KernelGym configuration settings."""

import os
from pathlib import Path
from typing import List, Dict, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent
KERNELBENCH_ROOT = PROJECT_ROOT.parent


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=10907, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=False, env="API_RELOAD")

    gpu_devices: List[int] = Field(default_factory=lambda: list(range(8)), env="GPU_DEVICES")
    gpu_memory_limit: str = Field(default="16GB", env="GPU_MEMORY_LIMIT")
    node_id: str = Field(default="", env="NODE_ID")
    worker_name_prefix: str = Field(default="", env="WORKER_NAME_PREFIX")
    worker_only_mode: bool = Field(default=False, env="WORKER_ONLY_MODE")

    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_key_prefix: str = Field(default="kernelgym", env="REDIS_KEY_PREFIX")
    redis_key_prefix_legacy: str = Field(default="kernelserver", env="REDIS_KEY_PREFIX_LEGACY")

    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    celery_task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    celery_accept_content: List[str] = Field(default_factory=lambda: ["json"], env="CELERY_ACCEPT_CONTENT")
    celery_timezone: str = Field(default="UTC", env="CELERY_TIMEZONE")

    default_num_trials: int = Field(default=100, env="DEFAULT_NUM_TRIALS")
    default_timeout: int = Field(default=600, env="DEFAULT_TIMEOUT")
    default_backend: str = Field(default="triton", env="DEFAULT_BACKEND")
    default_toolkit: str = Field(default="kernelbench", env="DEFAULT_TOOLKIT")
    default_backend_adapter: str = Field(
        default="kernelbench", env="DEFAULT_BACKEND_ADAPTER"
    )
    max_concurrent_tasks: int = Field(default=4, env="MAX_CONCURRENT_TASKS")

    verbose_error_traceback: bool = Field(
        default=True,
        env="VERBOSE_ERROR_TRACEBACK",
        description="Return full error traceback in metadata. Set to False for production to reduce response size",
    )

    enable_profiling: bool = Field(
        default=True,
        env="ENABLE_PROFILING",
        description="Enable torch.profiler for performance diagnostics. Default False to minimize overhead.",
    )
    profiling_activities: List[str] = Field(
        default_factory=lambda: ["cpu", "cuda"],
        env="PROFILING_ACTIVITIES",
        description="Profiling activities: cpu, cuda. Use ['cpu', 'cuda'] for full profiling.",
    )
    profiling_record_shapes: bool = Field(
        default=True,
        env="PROFILING_RECORD_SHAPES",
        description="Record tensor shapes in profiler. Useful for debugging shape mismatches.",
    )
    profiling_profile_memory: bool = Field(
        default=True,
        env="PROFILING_PROFILE_MEMORY",
        description="Profile memory allocations. Adds ~5% overhead but provides memory insights.",
    )
    profiling_with_stack: bool = Field(
        default=False,
        env="PROFILING_WITH_STACK",
        description="Record stack traces in profiler. Adds significant overhead (~15-20%), use for deep debugging only.",
    )
    profiling_retry_count: int = Field(
        default=1,
        env="PROFILING_RETRY_COUNT",
        description="Retry count when profiler returns empty results (0 to disable).",
    )

    reference_cache_dataset_path: str = Field(default="", env="REFERENCE_CACHE_DATASET_PATH")
    val_data_cache_dataset_path: str = Field(default="", env="VAL_DATA_CACHE_DATASET_PATH")
    enable_reference_cache: bool = Field(default=False, env="ENABLE_REFERENCE_CACHE")

    enable_sandbox: bool = Field(default=True, env="ENABLE_SANDBOX")
    docker_image: str = Field(default="kernelserver:latest", env="DOCKER_IMAGE")
    max_memory_per_task: str = Field(default="4GB", env="MAX_MEMORY_PER_TASK")
    max_gpu_time_per_task: int = Field(default=600, env="MAX_GPU_TIME_PER_TASK")

    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")

    save_eval_results: bool = Field(
        default=False,
        env="SAVE_EVAL_RESULTS",
        description="Persist evaluation results to local JSONL file.",
    )
    eval_results_path: str = Field(
        default="logs/eval_results.jsonl",
        env="EVAL_RESULTS_PATH",
        description="JSONL file path for persisted evaluation results.",
    )
    metrics_port: int = Field(default=8001, env="METRICS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    worker_monitor_interval: int = Field(default=30, env="WORKER_MONITOR_INTERVAL")
    worker_monitor_heartbeat_timeout: int = Field(default=120, env="WORKER_MONITOR_HEARTBEAT_TIMEOUT")
    worker_monitor_restart_cooldown: int = Field(default=60, env="WORKER_MONITOR_RESTART_COOLDOWN")
    worker_queue_wait_timeout_sec: int = Field(default=180, env="WORKER_QUEUE_WAIT_TIMEOUT_SEC")
    worker_queue_wait_monitor_interval: int = Field(default=20, env="WORKER_QUEUE_WAIT_MONITOR_INTERVAL")
    worker_queue_wait_scan_limit: int = Field(default=200, env="WORKER_QUEUE_WAIT_SCAN_LIMIT")
    worker_execution_timeout_grace_sec: int = Field(default=60, env="WORKER_EXECUTION_TIMEOUT_GRACE_SEC")
    worker_execution_timeout_monitor_interval: int = Field(default=30, env="WORKER_EXECUTION_TIMEOUT_MONITOR_INTERVAL")
    worker_pool_size: int = Field(
        default=1,
        env="WORKER_POOL_SIZE",
        description="Number of persistent subprocess workers per GPU. Set to 1 for strict serial execution.",
    )
    max_tasks_per_worker: int = Field(
        default=1,
        env="MAX_TASKS_PER_WORKER",
        description="Max tasks a subprocess worker handles before restart. Set to 1 for per-task isolation.",
    )

    log_dir: str = Field(default="logs", env="LOG_DIR")
    log_to_file: bool = Field(default=True, env="LOG_TO_FILE")
    log_max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    enable_result_cache: bool = Field(default=True, env="ENABLE_RESULT_CACHE")

    kernelbench_path: str = Field(default=str(KERNELBENCH_ROOT), env="KERNELBENCH_PATH")
    gpu_arch: List[str] = Field(default_factory=lambda: ["Hopper"], env="GPU_ARCH")

    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")

    @validator("gpu_devices", pre=True)
    def validate_gpu_devices(cls, v):
        if isinstance(v, str):
            try:
                import json

                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
                return [int(parsed)]
            except Exception:
                try:
                    return [int(x.strip()) for x in v.split(",")]
                except Exception:
                    return list(range(8))
        if isinstance(v, list):
            return [int(x) for x in v]
        return list(range(8))

    @validator("gpu_arch", pre=True)
    def validate_gpu_arch(cls, v):
        if isinstance(v, str):
            try:
                import json

                return json.loads(v)
            except Exception:
                return [v]
        if isinstance(v, list):
            return v
        return ["Hopper"]

    def setup_log_directory(self) -> None:
        log_path = Path(self.log_dir)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / self.log_dir
        os.makedirs(log_path, exist_ok=True)

    def get_redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def redis_url(self) -> str:
        return self.get_redis_url()

    def get_celery_config(self) -> Dict[str, Any]:
        return {
            "broker_url": self.celery_broker_url,
            "result_backend": self.celery_result_backend,
            "task_serializer": self.celery_task_serializer,
            "accept_content": self.celery_accept_content,
            "timezone": self.celery_timezone,
            "task_routes": {
                "worker.tasks.evaluate_kernel": {"queue": "gpu_evaluation"},
                "worker.tasks.compile_kernel": {"queue": "compilation"},
            },
            "task_annotations": {
                "worker.tasks.evaluate_kernel": {"rate_limit": f"{self.max_concurrent_tasks}/h"}
            },
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        @classmethod
        def prepare_field_value(cls, field_name: str, field, field_value, value_is_complex: bool):
            if field_name == "gpu_devices" and isinstance(field_value, str):
                try:
                    import json

                    parsed = json.loads(field_value)
                    if isinstance(parsed, list):
                        return [int(x) for x in parsed]
                    return [int(parsed)]
                except Exception:
                    try:
                        return [int(x.strip()) for x in field_value.split(",")]
                    except Exception:
                        return list(range(8))
            if field_name == "gpu_arch" and isinstance(field_value, str):
                try:
                    import json

                    return json.loads(field_value)
                except Exception:
                    return [field_value]
            return field_value


settings = Settings()

GPU_DEVICE_MAP = {
    f"cuda:{i}": {
        "device_id": i,
        "memory_limit": settings.gpu_memory_limit,
        "worker_queue": f"gpu_{i}",
    }
    for i in settings.gpu_devices
}

TASK_CONFIGS = {
    "quick": {"num_correct_trials": 3, "num_perf_trials": 10, "timeout": 60, "priority": "high"},
    "standard": {"num_correct_trials": 5, "num_perf_trials": 100, "timeout": 300, "priority": "normal"},
    "thorough": {"num_correct_trials": 10, "num_perf_trials": 1000, "timeout": 600, "priority": "low"},
}


def get_logging_config() -> Dict[str, Any]:
    settings.setup_log_directory()

    log_path = Path(settings.log_dir)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / settings.log_dir

    handlers = {
        "console": {
            "level": settings.log_level,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        }
    }

    if settings.log_to_file:
        handlers.update(
            {
                "file_server": {
                    "level": settings.log_level,
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_path / "kernelgym.log"),
                    "maxBytes": 104857600,
                    "backupCount": settings.log_backup_count,
                    "encoding": "utf8",
                },
                "file_worker": {
                    "level": settings.log_level,
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_path / "workers.log"),
                    "maxBytes": 104857600,
                    "backupCount": settings.log_backup_count,
                    "encoding": "utf8",
                },
                "file_api": {
                    "level": settings.log_level,
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_path / "api.log"),
                    "maxBytes": 104857600,
                    "backupCount": settings.log_backup_count,
                    "encoding": "utf8",
                },
            }
        )

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d] - %(message)s"
            },
        },
        "handlers": handlers,
        "loggers": {
            "": {
                "handlers": ["console"] + (["file_server"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "kernelgym.api": {
                "handlers": ["console"] + (["file_api"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "kernelgym.worker": {
                "handlers": ["console"] + (["file_worker"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console"] + (["file_server"] if settings.log_to_file else []),
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console"] + (["file_api"] if settings.log_to_file else []),
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    import logging.handlers

    return config


def setup_logging(component_name: str = "server"):
    import logging.config

    config = get_logging_config()
    logging.config.dictConfig(config)

    if component_name == "api":
        logger_name = "kernelgym.api"
    elif component_name == "worker":
        logger_name = "kernelgym.worker"
    else:
        logger_name = ""

    logger = logging.getLogger(logger_name)
    logger.info(f"Logging configured for {component_name} - File logging: {settings.log_to_file}")

    if settings.log_to_file:
        log_path = Path(settings.log_dir)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / settings.log_dir
        logger.info(f"Log files will be written to: {log_path}")

    return logger
