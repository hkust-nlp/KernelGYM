"""FastAPI server for KernelGym."""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import redis.asyncio as redis
from contextlib import asynccontextmanager
import json
import time

from kernelgym.config import settings, setup_logging
from .models import (
    EvaluationRequest,
    EvaluationResponse,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    TaskStatusResponse,
    SystemHealthResponse,
    MetricsResponse,
    ErrorResponse,
    WorkflowRequest,
    WorkflowResponse,
)
from .utils import get_system_health, get_system_metrics, format_timestamp
from kernelgym.server.task_manager import TaskManager
from kernelgym.server.scheduler import TaskManagerScheduler
from kernelgym.workflow import get_workflow_controller
from redis.exceptions import BusyLoadingError, ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError, ResponseError as RedisResponseError
from kernelgym.utils.error_classifier import classify_error
from kernelgym.common import ErrorCode, TaskStatus
from .monitoring_routes import router as monitoring_router

# Configure logging with file support
logger = logging.getLogger("kernelgym.api")

# Global variables
task_manager: Optional[TaskManager] = None
redis_client: Optional[redis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global task_manager, redis_client
    
    # Setup logging first
    setup_logging("api")
    
    # Startup
    logger.info("Starting KernelGym...")
    
    # Initialize Redis connection with readiness wait (handle RDB/AOF loading)
    async def _wait_for_redis_ready(url: str, timeout_sec: float = 60.0, interval_sec: float = 0.5):
        start = asyncio.get_event_loop().time()
        client = redis.from_url(url)
        last_err = None
        while True:
            try:
                await client.ping()
                return client
            except (BusyLoadingError, RedisResponseError) as e:
                last_err = e
                logger.warning(f"Redis not ready (loading data): {e}. Retrying...")
            except (RedisConnectionError, RedisTimeoutError) as e:
                last_err = e
                logger.warning(f"Redis connection not ready: {e}. Retrying...")
            except Exception as e:
                last_err = e
                logger.warning(f"Redis ping error: {e}. Retrying...")
            # timeout check
            if (asyncio.get_event_loop().time() - start) > timeout_sec:
                raise RuntimeError(f"Redis not ready within {timeout_sec}s: {last_err}")
            await asyncio.sleep(interval_sec)

    redis_client = await _wait_for_redis_ready(settings.redis_url)
    logger.info("Redis connection established")
    
    # Initialize task manager
    task_manager = TaskManager(redis_client)
    await task_manager.initialize()
    logger.info("Task manager initialized")
    
    # Initialize GPU workers
    logger.info(f"Initializing GPU workers for devices: {settings.gpu_devices}")
    
    # Store task manager in app state for access in endpoints
    app.state.task_manager = task_manager
    
    # Currently we register workers via url. So we do not need to use this waiting.
    # Wait for at least one worker to register
    # if settings.gpu_devices:
    #     logger.info("Waiting for workers to register...")
    #     max_wait_time = 30  # seconds
    #     check_interval = 1  # second
    #     start_time = asyncio.get_event_loop().time()
        
    #     while True:
    #         workers_status = await task_manager.get_workers_status()
    #         online_workers = sum(1 for w in workers_status.values() if w.get("online"))
            
    #         if online_workers > 0:
    #             logger.info(f"✅ {online_workers} worker(s) online and ready")
    #             break
            
    #         elapsed = asyncio.get_event_loop().time() - start_time
    #         if elapsed > max_wait_time:
    #             logger.warning(f"⚠️  No workers registered after {max_wait_time}s. Starting anyway...")
    #             break
            
    #         await asyncio.sleep(check_interval)
    
    yield
    
    # Shutdown
    logger.info("Shutting down KernelGym...")
    if task_manager:
        await task_manager.shutdown()
    if redis_client:
        await redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="KernelGym",
    description="GPU Kernel Evaluation Service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(monitoring_router)
# Node ID allocation endpoint (server-assigned node identifiers)
@app.post("/node/allocate")
async def allocate_node_id(
    hostname: str,
    node_name: Optional[str] = None
) -> Dict[str, Any]:
    """Allocate or return a stable node_id for a given hostname.

    Args:
        hostname: The hostname of the requesting node
        node_name: Optional explicit node identifier. If provided, this will be used
                   as the node_id directly instead of auto-generating a sequential ID.
                   This allows for stable, human-readable node IDs in multi-node setups.

    Rules:
    - If node_name is provided:
      - Use it as the node_id directly (e.g., "node2", "worker-gpu-1")
      - Check if this node_name already exists and verify it's for the same hostname
      - Store the mapping for future idempotency
    - If node_name is not provided (backward compatible):
      - If the hostname already has a bound node_id, return it (idempotent)
      - Otherwise allocate a new node_id (node-<seq>), bind both maps and return

    Returns:
        Dictionary with node_id and hostname

    Examples:
        # Explicit node naming (multi-node deployment)
        POST /node/allocate?hostname=worker-01&node_name=node2
        → {"node_id": "node2", "hostname": "worker-01"}

        # Auto allocation (single-node or backward compatible)
        POST /node/allocate?hostname=worker-01
        → {"node_id": "node-1", "hostname": "worker-01"}
    """
    try:
        if redis_client is None:
            raise HTTPException(status_code=503, detail="Redis not available")

        primary_prefix = settings.redis_key_prefix
        legacy_prefix = settings.redis_key_prefix_legacy

        def _keys(prefix: str) -> Dict[str, str]:
            return {
                "nodes": f"{prefix}:nodes",
                "hosts": f"{prefix}:nodes_by_host",
                "names": f"{prefix}:node_names",
            }

        keys_primary = _keys(primary_prefix)
        keys_legacy = _keys(legacy_prefix) if legacy_prefix and legacy_prefix != primary_prefix else None

        # Handle explicit node_name (multi-node deployment)
        if node_name:
            # Check if this node_name is already assigned (primary, fallback legacy)
            existing_hostname = await redis_client.hget(keys_primary["names"], node_name)
            if not existing_hostname and keys_legacy:
                existing_hostname = await redis_client.hget(keys_legacy["names"], node_name)
            if existing_hostname:
                existing_hostname_str = existing_hostname.decode() if isinstance(existing_hostname, bytes) else existing_hostname
                if existing_hostname_str != hostname:
                    logger.warning(
                        f"node_name={node_name} already assigned to different hostname={existing_hostname_str}, "
                        f"requested by hostname={hostname}"
                    )
                    raise HTTPException(
                        status_code=409,
                        detail=f"node_name '{node_name}' already assigned to hostname '{existing_hostname_str}'"
                    )
                # Same node_name + hostname, idempotent return
                logger.info(f"Returning existing node_name={node_name} for hostname={hostname}")
                if keys_legacy and not await redis_client.hget(keys_primary["names"], node_name):
                    await redis_client.hset(keys_primary["names"], node_name, hostname)
                    await redis_client.hset(keys_primary["nodes"], node_name, hostname)
                    await redis_client.hset(keys_primary["hosts"], hostname, node_name)
                return {"node_id": node_name, "hostname": hostname}

            # New node_name assignment
            await redis_client.hset(keys_primary["nodes"], node_name, hostname)
            await redis_client.hset(keys_primary["names"], node_name, hostname)
            await redis_client.hset(keys_primary["hosts"], hostname, node_name)
            logger.info(f"Allocated explicit node_id={node_name} for hostname={hostname}")
            return {"node_id": node_name, "hostname": hostname}

        # Handle auto-allocation (backward compatible)
        # Check existing mapping by hostname
        existing = await redis_client.hget(keys_primary["hosts"], hostname)
        if not existing and keys_legacy:
            existing = await redis_client.hget(keys_legacy["hosts"], hostname)
        if existing:
            node_id = existing.decode() if isinstance(existing, bytes) else existing
            logger.info(f"Returning existing node_id={node_id} for hostname={hostname}")
            if keys_legacy and not await redis_client.hget(keys_primary["hosts"], hostname):
                await redis_client.hset(keys_primary["hosts"], hostname, node_id)
                await redis_client.hset(keys_primary["nodes"], node_id, hostname)
            return {"node_id": node_id, "hostname": hostname}

        # Allocate a new sequential node id
        seq = await redis_client.incr(f"{primary_prefix}:node_seq")
        node_id = f"node-{seq}"

        # Bind both directions
        await redis_client.hset(keys_primary["nodes"], node_id, hostname)
        await redis_client.hset(keys_primary["hosts"], hostname, node_id)

        logger.info(f"Auto-allocated node_id={node_id} for hostname={hostname}")
        return {"node_id": node_id, "hostname": hostname}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error allocating node id for hostname={hostname}, node_name={node_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # For POST requests, try to log the body (but don't log sensitive data)
    if request.method == "POST":
        try:
            body = await request.body()
            if body:
                # Parse JSON and log task_id if present
                try:
                    json_body = json.loads(body)
                    task_id = json_body.get("task_id", "unknown")
                    logger.info(f"Request body for task {task_id}: {len(body)} bytes")
                except:
                    logger.info(f"Request body: {len(body)} bytes")
        except:
            pass
    
    # Process the request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.4f}s")
    
    return response


async def get_task_manager() -> TaskManager:
    """Dependency to get task manager."""
    if task_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task manager not available"
        )
    return task_manager


async def get_redis_client() -> redis.Redis:
    """Dependency to get Redis client."""
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client not available"
        )
    return redis_client


def _result_status(payload: Dict[str, Any]) -> TaskStatus:
    return TaskStatus.FAILED if payload.get("status") == "failed" else TaskStatus.COMPLETED


def _strip_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(payload)
    cleaned.pop("status", None)
    return cleaned


async def _execute_workflow(
    task_mgr: TaskManager,
    workflow_name: str,
    payload: Dict[str, Any],
    task_id: Optional[str] = None,
    force_refresh: bool = False,
) -> tuple[str, Dict[str, Any], TaskStatus]:
    if task_id:
        payload = dict(payload)
        payload["task_id"] = task_id
    if isinstance(payload, dict) and payload.get("resources") is None:
        payload["resources"] = None
    task_id = task_id or payload.get("task_id")
    if not task_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="task_id is required")

    if not force_refresh:
        existing = await task_mgr.get_task_result(task_id)
        if existing:
            existing = dict(existing)
            existing.setdefault("task_id", task_id)
            return task_id, existing, _result_status(existing)

    scheduler = TaskManagerScheduler(task_mgr)
    try:
        controller = get_workflow_controller(workflow_name or "kernelbench")
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    result = await controller.handle_request(payload, scheduler)
    if isinstance(result, dict):
        result.setdefault("task_id", task_id)
    await task_mgr.complete_task(task_id, result)
    return task_id, result, _result_status(result)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    error_details = exc.errors()
    error_msg = f"Request validation failed: {error_details}"
    logger.error(f"Validation error for {request.url}: {error_msg}")
    
    error_code = ErrorCode.VALIDATION_ERROR
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="RequestValidationError",
            message=error_msg,
            error_code=error_code,
            timestamp=format_timestamp(datetime.now())
        ).dict(),
        headers={"X-Error-Code": error_code.value}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    error_code = classify_error(str(exc), "system")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc),
            error_code=error_code,
            timestamp=format_timestamp(datetime.now())
        ).dict(),
        headers={"X-Error-Code": error_code.value}
    )


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "KernelGym",
        "version": "1.0.0",
        "description": "GPU Kernel Evaluation Service",
        "timestamp": format_timestamp(datetime.now())
    }


@app.post("/debug/validate")
async def debug_validate(request: EvaluationRequest):
    """Debug endpoint to validate request format."""
    logger.info(f"Debug validation for task {request.task_id}")

    try:
        controller = get_workflow_controller(request.workflow or "kernelbench")
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    validation = await controller.validate_request(request.dict())
    validation["request_info"] = {
        "backend": request.backend,
        "num_correct_trials": request.num_correct_trials,
        "num_perf_trials": request.num_perf_trials,
        "timeout": request.timeout,
        "priority": request.priority,
        "device_preference": request.device_preference,
        "force_refresh": request.force_refresh,
        "workflow": request.workflow,
    }
    return validation




@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_kernel(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    task_mgr: TaskManager = Depends(get_task_manager)
):
    """Submit a kernel evaluation task."""
    try:
        _, result, status_value = await _execute_workflow(
            task_mgr=task_mgr,
            workflow_name=request.workflow or "kernelbench",
            payload=request.dict(),
            task_id=request.task_id,
            force_refresh=request.force_refresh,
        )
        return EvaluationResponse(status=status_value, **_strip_status(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting task {request.task_id}: {e}")
        error_code = classify_error(str(e), "system")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}",
            headers={"X-Error-Code": error_code.value}
        )


@app.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks,
    task_mgr: TaskManager = Depends(get_task_manager)
):
    """Submit a batch of evaluation tasks."""
    try:
        batch_results = []
        failed_tasks = 0
        
        for task_request in request.tasks:
            try:
                _, result, status_value = await _execute_workflow(
                    task_mgr=task_mgr,
                    workflow_name=task_request.workflow or "kernelbench",
                    payload=task_request.dict(),
                    task_id=task_request.task_id,
                    force_refresh=task_request.force_refresh,
                )
                if status_value == TaskStatus.FAILED:
                    failed_tasks += 1
                batch_results.append(EvaluationResponse(status=status_value, **_strip_status(result)))
                
            except Exception as e:
                logger.error(f"Error processing task {task_request.task_id}: {e}")
                error_code = classify_error(str(e), "system")
                batch_results.append(EvaluationResponse(
                    task_id=task_request.task_id,
                    status=TaskStatus.FAILED,
                    error_message=str(e),
                    error_code=error_code,
                    submitted_at=format_timestamp(datetime.now())
                ))
                failed_tasks += 1
        
        return BatchEvaluationResponse(
            batch_id=request.batch_id,
            total_tasks=len(request.tasks),
            completed_tasks=len(batch_results) - failed_tasks,
            failed_tasks=failed_tasks,
            results=batch_results,
            batch_status=TaskStatus.COMPLETED,
            submitted_at=format_timestamp(datetime.now())
        )
        
    except Exception as e:
        logger.error(f"Error processing batch {request.batch_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process batch: {str(e)}"
        )


@app.post("/workflow/submit", response_model=WorkflowResponse)
async def submit_workflow(
    request: WorkflowRequest,
    task_mgr: TaskManager = Depends(get_task_manager),
):
    """Submit a workflow task with generic payload."""
    try:
        task_id = request.task_id or request.payload.get("task_id") if isinstance(request.payload, dict) else None
        payload = request.payload
        if isinstance(payload, dict) and payload.get("resources") is None and request.resources is not None:
            payload = dict(payload)
            payload["resources"] = request.resources
        task_id, result, status_value = await _execute_workflow(
            task_mgr=task_mgr,
            workflow_name=request.workflow or "kernelbench",
            payload=payload,
            task_id=task_id,
            force_refresh=request.force_refresh,
        )
        payload = _strip_status(result)
        return WorkflowResponse(
            task_id=task_id,
            status=status_value,
            result=payload,
            error_message=result.get("error_message"),
            error_code=result.get("error_code"),
            completed_at=payload.get("completed_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting workflow {request.workflow}: {e}")
        error_code = classify_error(str(e), "system")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit workflow: {str(e)}",
            headers={"X-Error-Code": error_code.value},
        )


@app.get("/workflow/results/{task_id}", response_model=WorkflowResponse)
async def get_workflow_results(
    task_id: str,
    task_mgr: TaskManager = Depends(get_task_manager),
):
    """Get workflow result for a task."""
    try:
        result = await task_mgr.get_task_result(task_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found",
            )
        result = dict(result)
        result.setdefault("task_id", task_id)
        status_value = _result_status(result)
        payload = _strip_status(result)
        return WorkflowResponse(
            task_id=task_id,
            status=status_value,
            result=payload,
            error_message=result.get("error_message"),
            error_code=result.get("error_code"),
            completed_at=payload.get("completed_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow results for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow results: {str(e)}",
        )

@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    task_mgr: TaskManager = Depends(get_task_manager)
):
    """Get task status."""
    try:
        status_info = await task_mgr.get_task_status(task_id)
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return TaskStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@app.get("/results/{task_id}", response_model=EvaluationResponse)
async def get_task_results(
    task_id: str,
    task_mgr: TaskManager = Depends(get_task_manager)
):
    """Get task results."""
    try:
        result = await task_mgr.get_task_result(task_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        result = dict(result)
        result.setdefault("task_id", task_id)
        status_value = _result_status(result)
        return EvaluationResponse(status=status_value, **_strip_status(result))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results for task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task results: {str(e)}"
        )


@app.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    task_mgr: TaskManager = Depends(get_task_manager)
):
    """Cancel a task."""
    try:
        success = await task_mgr.cancel_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found or cannot be cancelled"
            )
        
        return {"message": f"Task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """System health check."""
    try:
        health_info = await get_system_health()
        return SystemHealthResponse(**health_info)
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    try:
        metrics = await get_system_metrics()
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get("/queue/status")
async def get_queue_status(
    task_mgr: TaskManager = Depends(get_task_manager)
):
    """Get queue status."""
    try:
        queue_info = await task_mgr.get_queue_status()
        return queue_info
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )


@app.get("/workers/status")
async def get_workers_status(
    task_mgr: TaskManager = Depends(get_task_manager)
):
    """Get workers status."""
    try:
        workers_info = await task_mgr.get_workers_status()
        return workers_info
        
    except Exception as e:
        logger.error(f"Error getting workers status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workers status: {str(e)}"
        )


@app.post("/worker/register")
async def register_worker(
    worker_id: str,
    device: str,
    node_id: Optional[str] = None,
    hostname: Optional[str] = None,
    task_manager: TaskManager = Depends(get_task_manager)
) -> Dict[str, Any]:
    """Register a worker with the server."""
    try:
        logger.info(f"/worker/register received: worker_id={worker_id}, device={device}, node_id={node_id}, hostname={hostname}")
        success = await task_manager.register_worker(worker_id, device, node_id=node_id, hostname=hostname)
        if success:
            # LB snapshot
            lb_keys = list(task_manager.worker_load_balancer.available_workers.keys())
            logger.info(f"Worker {worker_id} registered successfully with device {device}; LB now has: {lb_keys}")
            return {"success": True, "message": f"Worker {worker_id} registered", "node_id": node_id or ""}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register worker"
            )
    except Exception as e:
        logger.error(f"Error registering worker {worker_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/worker/unregister")
async def unregister_worker(
    worker_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
) -> Dict[str, Any]:
    """Unregister a worker from the server."""
    try:
        success = await task_manager.unregister_worker(worker_id)
        if success:
            logger.info(f"Worker {worker_id} unregistered successfully")
            return {"success": True, "message": f"Worker {worker_id} unregistered"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to unregister worker"
            )
    except Exception as e:
        logger.error(f"Error unregistering worker {worker_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/worker/heartbeat")
async def worker_heartbeat(
    worker_id: str,
    device: str | None = None,
    node_id: Optional[str] = None,
    hostname: Optional[str] = None,
    task_manager: TaskManager = Depends(get_task_manager)
) -> Dict[str, Any]:
    """Update worker heartbeat."""
    try:
        logger.info(f"/worker/heartbeat received: worker_id={worker_id}, device={device}, node_id={node_id}, hostname={hostname}")
        # If not known, only auto-register when device is provided and consistent
        if worker_id not in task_manager.worker_registry:
            # Read stored device (if any) from Redis
            worker_data = await task_manager.get_worker_data(worker_id)
            stored_device = worker_data.get(b"device", b"").decode() if worker_data else ""
            
            if not device and not stored_device:
                # Unknown device, refuse to auto-register
                logger.warning(f"Heartbeat auto-register refused: unknown device for {worker_id}")
                raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                    detail=f"Worker {worker_id} not registered and device unknown; please register explicitly")
            # Prefer provided device, fall back to stored device if provided is None
            resolved_device = device or stored_device

            # Enforce uniqueness: device cannot be already used by another worker ON THE SAME NODE
            # In multi-node deployments, different nodes can have workers using the same device name (e.g., cuda:0)
            for wid, info in task_manager.worker_registry.items():
                if info.get("device") == resolved_device and wid != worker_id:
                    # Check if they're on the same node
                    existing_node_id = info.get("node_id")
                    if existing_node_id == node_id:
                        # Same device on same node = conflict
                        logger.warning(f"Heartbeat auto-register refused: device {resolved_device} already used by {wid} on node {node_id}")
                        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                            detail=f"Device {resolved_device} already in use by {wid} on node {node_id}")
            
            # If stored device exists and differs from provided -> refuse
            if stored_device and device and stored_device != device:
                logger.warning(f"Heartbeat auto-register refused: device mismatch stored={stored_device}, provided={device}")
                raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                    detail=f"Device mismatch for {worker_id}: stored={stored_device}, provided={device}")
            
            ok = await task_manager.register_worker(worker_id, resolved_device, node_id=node_id, hostname=hostname)
            if not ok:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail=f"Failed to auto-register worker {worker_id}")
            logger.info(f"Auto-registered worker {worker_id} on heartbeat with device {resolved_device}")
        
        # 设备冲突与空node_id防呆：读取注册信息，校验 node_id/hostname/设备一致性
        try:
            reg = task_manager.worker_registry.get(worker_id)
            if reg:
                # 若注册表无 node_id/hostname，但这次心跳带了，则补写（遗留修复）
                if not reg.get("node_id") and node_id:
                    reg["node_id"] = node_id
                if not reg.get("hostname") and hostname:
                    reg["hostname"] = hostname
                # 如果设备不一致，拒绝心跳
                if device and reg.get("device") and reg.get("device") != device:
                    logger.warning(f"Heartbeat refused: device mismatch for {worker_id}, reg={reg.get('device')} req={device}")
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Device mismatch; please re-register")
        except Exception:
            pass

        # Update heartbeat in load balancer（仅当注册信息一致且未冲突时）
        await task_manager.worker_load_balancer.update_worker_heartbeat(worker_id)
        lb_keys = list(task_manager.worker_load_balancer.available_workers.keys())
        logger.info(f"Heartbeat updated for {worker_id}; LB now has: {lb_keys}")
        return {"success": True, "message": f"Heartbeat updated for {worker_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating heartbeat for worker {worker_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/worker/evict_from_lb")
async def evict_from_lb(
    worker_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
) -> Dict[str, Any]:
    """Evict a worker from in-memory load balancer without deleting its Redis state."""
    try:
        await task_manager.worker_load_balancer.unregister_worker(worker_id)
        # Mark as offline in registry if present
        if worker_id in task_manager.worker_registry:
            task_manager.worker_registry[worker_id]["status"] = "offline"
        return {"success": True, "message": f"Worker {worker_id} evicted from LB"}
    except Exception as e:
        logger.error(f"Error evicting {worker_id} from LB: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Setup logging for direct run
    setup_logging("api")
    
    uvicorn.run(
        "kernelgym.server.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
