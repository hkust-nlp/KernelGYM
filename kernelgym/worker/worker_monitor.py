"""
Worker Monitor for KernelGym.
Monitors worker health and restarts crashed workers.

Enhancement: Persistent monitoring mode (opt-in via --persistent or env)
- When enabled, the monitor maintains target workers based on Redis keys that
  are populated by the launcher (e.g., start_all_with_monitor.sh):
    - f"{KEY_PREFIX}:expected_workers" (SET of worker_ids)
    - f"{KEY_PREFIX}:expected_worker:{worker_id}" (HASH: device, node_id, hostname)
    - f"{KEY_PREFIX}:worker_process:{worker_id}" (HASH: pid, start_time, device)
  The monitor will restart a worker if:
    - Its heartbeat hash key is missing (heartbeat key expired after crash), or
    - Its recorded PID is not alive, or
    - It meets original restart conditions (CUDA error shutdown / heartbeat timeout).
"""
import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set
import argparse
import redis.asyncio as redis
import signal
from pathlib import Path

from kernelgym.config import settings
KEY_PREFIX = settings.redis_key_prefix
from kernelgym.config import setup_logging
from redis.exceptions import BusyLoadingError, ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError, ResponseError as RedisResponseError

logger = logging.getLogger("kernelgym.worker_monitor")


class WorkerMonitor:
    """Monitors worker health and manages restarts."""
    
    def __init__(self, redis_client: redis.Redis, persistent: bool = False):
        self.redis = redis_client
        self.running = False
        self.monitored_workers: Dict[str, Dict[str, Any]] = {}
        self.restart_queue: asyncio.Queue = asyncio.Queue()
        self.restart_in_progress: Set[str] = set()
        self.persistent: bool = persistent
        
        # Configuration
        self.heartbeat_timeout = max(5, settings.worker_monitor_heartbeat_timeout)
        self.monitor_interval = max(5, settings.worker_monitor_interval)
        self.max_restart_attempts = 3
        self.restart_cooldown = max(5, settings.worker_monitor_restart_cooldown)
        logger.info(
            "Worker monitor configured with heartbeat_timeout=%ss, monitor_interval=%ss, restart_cooldown=%ss",
            self.heartbeat_timeout,
            self.monitor_interval,
            self.restart_cooldown,
        )
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker monitor received signal {signum}")
        self.running = False
    
    async def start(self):
        """Start the worker monitor."""
        self.running = True
        logger.info("Starting worker monitor")
        
        try:
            # Start monitoring and restart tasks
            monitor_task = asyncio.create_task(self._monitor_loop())
            restart_task = asyncio.create_task(self._restart_loop())
            
            await asyncio.gather(monitor_task, restart_task)
            
        except Exception as e:
            logger.error(f"Error in worker monitor: {e}")
            raise
    
    async def stop(self):
        """Stop the worker monitor."""
        logger.info("Stopping worker monitor")
        self.running = False
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                await self._check_workers()
                await asyncio.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _check_workers(self):
        """Check health of all workers."""
        try:
            # Get all worker keys
            worker_keys = await self.redis.keys(f"{KEY_PREFIX}:worker:*")
            # In persistent mode, load expected workers set once per cycle
            expected_ids: Set[str] = set()
            if self.persistent:
                try:
                    raw = await self.redis.smembers(f"{KEY_PREFIX}:expected_workers")
                    expected_ids = {wid.decode() if isinstance(wid, bytes) else wid for wid in raw} if raw else set()
                except Exception:
                    expected_ids = set()
            
            for key in worker_keys:
                worker_id = key.decode().split(":")[-1]
                
                # Get worker status
                worker_data = await self.redis.hgetall(key)
                if not worker_data:
                    continue
                
                # Decode data
                worker_info = {
                    k.decode(): v.decode() for k, v in worker_data.items()
                }
                
                # Check if worker needs restart
                needs_restart = False
                restart_reason = ""
                
                # Check for CUDA error shutdown
                if worker_info.get("cuda_error_shutdown") == "true":
                    needs_restart = True
                    restart_reason = "CUDA error shutdown"
                
                # Check heartbeat timeout
                elif "last_heartbeat" in worker_info:
                    last_heartbeat = datetime.fromisoformat(worker_info["last_heartbeat"])
                    if datetime.now() - last_heartbeat > timedelta(seconds=self.heartbeat_timeout):
                        needs_restart = True
                        restart_reason = "Heartbeat timeout"
                
                # Check if worker is marked as offline
                elif worker_info.get("online") == "false":
                    # Check if it's been offline for too long
                    if worker_id in self.monitored_workers:
                        offline_since = self.monitored_workers[worker_id].get("offline_since")
                        if offline_since and datetime.now() - offline_since > timedelta(seconds=60):
                            needs_restart = True
                            restart_reason = "Worker offline"
                    else:
                        self.monitored_workers[worker_id] = {
                            "offline_since": datetime.now()
                        }
                
                # If in persistent mode and this worker is not in expected set,
                # do not enforce restart for it.
                if self.persistent and worker_id not in expected_ids:
                    needs_restart = False

                if needs_restart and worker_id not in self.restart_in_progress:
                    logger.warning(f"Worker {worker_id} needs restart: {restart_reason}")
                    
                    # Get device info from worker ID (e.g., worker_gpu_0 -> cuda:0)
                    # Prefer reading device from Redis if available
                    worker_key = f"{KEY_PREFIX}:worker:{worker_id}"
                    worker_data = await self.redis.hgetall(worker_key)
                    device = worker_data.get(b"device", b"").decode() or f"cuda:{worker_id.split('_')[-1]}"
                    
                    # Add to restart queue
                    await self.restart_queue.put({
                        "worker_id": worker_id,
                        "device": device,
                        "reason": restart_reason,
                        "timestamp": datetime.now()
                    })
                    
                    self.restart_in_progress.add(worker_id)
                
                # Update monitoring info
                self.monitored_workers[worker_id] = {
                    "last_check": datetime.now(),
                    "status": worker_info.get("online", "unknown"),
                    "device": worker_info.get("device", "unknown")
                }

            # Persistent mode: also ensure expected workers are running even if
            # their heartbeat keys are missing or their PIDs are dead.
            if self.persistent:
                await self._check_persistent_expectations()
                
        except Exception as e:
            logger.error(f"Error checking workers: {e}")

    async def _check_persistent_expectations(self) -> None:
        """In persistent mode, restart workers missing from heartbeat keys
        or with dead PIDs according to expected worker list and process map."""
        try:
            # Load expected workers set
            expected_ids_raw = await self.redis.smembers(f"{KEY_PREFIX}:expected_workers")
            expected_ids = {wid.decode() if isinstance(wid, bytes) else wid for wid in expected_ids_raw} if expected_ids_raw else set()

            # Build set of existing heartbeat worker ids
            existing_keys = await self.redis.keys(f"{KEY_PREFIX}:worker:*")
            existing_ids = {k.decode().split(":")[-1] for k in existing_keys}

            # Restart missing-heartbeat workers
            missing_ids = expected_ids - existing_ids
            for wid in missing_ids:
                if wid in self.restart_in_progress:
                    continue
                # Determine device from expected worker hash if available
                edata = await self.redis.hgetall(f"{KEY_PREFIX}:expected_worker:{wid}")
                device = edata.get(b"device", b"").decode() if edata else f"cuda:{wid.split('_')[-1]}"
                logger.warning(f"Worker {wid} missing heartbeat key; scheduling restart (persistent mode)")
                await self.restart_queue.put({
                    "worker_id": wid,
                    "device": device,
                    "reason": "Missing heartbeat key",
                    "timestamp": datetime.now()
                })
                self.restart_in_progress.add(wid)

            # Check PID liveness for all expected workers
            for wid in expected_ids:
                # Skip ones already queued
                if wid in self.restart_in_progress:
                    continue
                proc_info = await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{wid}")
                if not proc_info:
                    # No process info recorded; if also no heartbeat key, schedule restart
                    if wid not in existing_ids:
                        edata = await self.redis.hgetall(f"{KEY_PREFIX}:expected_worker:{wid}")
                        device = edata.get(b"device", b"").decode() if edata else f"cuda:{wid.split('_')[-1]}"
                        logger.warning(f"Worker {wid} has no process info and no heartbeat; restarting (persistent mode)")
                        await self.restart_queue.put({
                            "worker_id": wid,
                            "device": device,
                            "reason": "Missing process info & heartbeat",
                            "timestamp": datetime.now()
                        })
                        self.restart_in_progress.add(wid)
                    continue
                try:
                    pid_raw = proc_info.get(b"pid")
                    pid = int(pid_raw) if pid_raw else 0
                except Exception:
                    pid = 0
                # If pid recorded but process not alive, schedule restart
                if pid:
                    try:
                        os.kill(pid, 0)
                        # Alive
                    except OSError:
                        # Dead process
                        device = proc_info.get(b"device", b"").decode() or f"cuda:{wid.split('_')[-1]}"
                        logger.warning(f"Worker {wid} PID {pid} not alive; scheduling restart (persistent mode)")
                        await self.restart_queue.put({
                            "worker_id": wid,
                            "device": device,
                            "reason": "Process dead",
                            "timestamp": datetime.now()
                        })
                        self.restart_in_progress.add(wid)
        except Exception as e:
            logger.error(f"Error in persistent expectation check: {e}")
    
    async def _restart_loop(self):
        """Process worker restart requests."""
        while self.running:
            try:
                # Get restart request with timeout
                try:
                    restart_info = await asyncio.wait_for(
                        self.restart_queue.get(), 
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                worker_id = restart_info["worker_id"]
                device = restart_info["device"]
                reason = restart_info["reason"]
                
                logger.info(f"Attempting to restart worker {worker_id} on {device}: {reason}")
                
                # Restart worker
                success = await self._restart_worker(worker_id, device)

                if success:
                    logger.info(f"Successfully restarted worker {worker_id}")
                    # Clear restart flags IMMEDIATELY to prevent re-detection during initialization
                    # If worker crashes during init, we'll detect it via PID check or missing heartbeat
                    await self.redis.hdel(
                        f"{KEY_PREFIX}:worker:{worker_id}",
                        "cuda_error_shutdown",
                        "shutdown_time"
                    )
                    # Give worker time to initialize (API registration + GPU init can take 30-60s)
                    # This prevents the monitor from immediately detecting the worker as "missing"
                    # before it has a chance to send its first real heartbeat
                    logger.info(f"Waiting 45s for worker {worker_id} to complete initialization...")
                    await asyncio.sleep(45)
                else:
                    logger.error(f"Failed to restart worker {worker_id}")
                    # Retry later
                    await asyncio.sleep(self.restart_cooldown)
                    await self.restart_queue.put(restart_info)

                # Remove from in-progress set
                self.restart_in_progress.discard(worker_id)
                
            except Exception as e:
                logger.error(f"Error in restart loop: {e}")
                await asyncio.sleep(5)

    async def _reset_gpu_device(self, device: str):
        """Reset GPU device to clear CUDA error state.

        After a CUDA error (especially illegal memory access), the GPU context
        is corrupted and needs to be reset. This method attempts multiple
        strategies to reset the GPU:
        1. nvidia-smi --gpu-reset (requires root/sudo on some systems)
        2. PyTorch device reset (fallback if nvidia-smi fails)

        Args:
            device: Device string (e.g., "cuda:0")
        """
        try:
            device_id = int(device.split(':')[-1])
            logger.info(f"Attempting to reset GPU device {device_id}")

            # Strategy 1: Use nvidia-smi to reset GPU (most effective)
            import subprocess
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--gpu-reset', '-i', str(device_id)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"Successfully reset GPU {device_id} using nvidia-smi")
                    return
                else:
                    logger.warning(f"nvidia-smi GPU reset failed (code {result.returncode}): {result.stderr}")
                    # Note: nvidia-smi --gpu-reset often requires root permissions
                    # or may not be available in compute mode. Fall through to other methods.
            except FileNotFoundError:
                logger.warning("nvidia-smi not found, trying alternative GPU reset methods")
            except subprocess.TimeoutExpired:
                logger.warning("nvidia-smi --gpu-reset timed out")
            except Exception as e:
                logger.warning(f"nvidia-smi GPU reset error: {e}")

            # Strategy 2: Use PyTorch CUDA device reset (in a subprocess to avoid affecting monitor)
            # This is less effective but doesn't require special permissions
            try:
                reset_script = f"""
import torch
import sys
try:
    if torch.cuda.is_available() and {device_id} < torch.cuda.device_count():
        torch.cuda.set_device({device_id})
        # Synchronize and clear cache
        torch.cuda.synchronize(device={device_id})
        torch.cuda.empty_cache()
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device={device_id})
        torch.cuda.reset_accumulated_memory_stats(device={device_id})
        print(f"Reset GPU {device_id} via PyTorch")
        sys.exit(0)
    else:
        print(f"GPU {device_id} not available", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"PyTorch GPU reset failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
                result = subprocess.run(
                    [sys.executable, '-c', reset_script],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"Reset GPU {device_id} using PyTorch: {result.stdout.strip()}")
                else:
                    logger.warning(f"PyTorch GPU reset failed: {result.stderr.strip()}")
            except Exception as e:
                logger.error(f"PyTorch GPU reset error: {e}")

            # Log completion regardless of success
            logger.info(f"GPU {device_id} reset attempt completed (worker will attempt to reinitialize)")

        except Exception as e:
            logger.error(f"Error resetting GPU device {device}: {e}")

    async def _restart_worker(self, worker_id: str, device: str) -> bool:
        """Restart a specific worker."""
        try:
            # Kill existing worker process if any
            await self._kill_worker_process(worker_id)

            # Reset GPU device to clear CUDA error state
            await self._reset_gpu_device(device)

            # Wait a bit for cleanup
            await asyncio.sleep(5)
            
            # Start new worker process
            import subprocess
            import functools
            
            # Build command to start single worker
            cmd = [
                sys.executable,
                "-m", "kernelgym.worker.single_worker",
                "--worker-id", worker_id,
                "--device", device,
                "--persistent"
            ]
            
            # Ensure logs directory exists and append logs to the same pattern as manual start
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = logs_dir / f"{worker_id}.log"
            # Start worker as subprocess with stdout/stderr redirected to log file
            log_fh = open(log_file_path, "a", buffering=1)
            # new session so we can kill the whole process group later
            preexec_fn = None
            creationflags = 0
            if hasattr(os, "setsid"):
                preexec_fn = os.setsid
            elif os.name == "nt":
                # On Windows, use CREATE_NEW_PROCESS_GROUP if available
                try:
                    import subprocess as sp
                    creationflags = getattr(sp, "CREATE_NEW_PROCESS_GROUP", 0)
                except Exception:
                    creationflags = 0
            process = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env={**os.environ},
                preexec_fn=preexec_fn,
                creationflags=creationflags
            )
            
            # Wait a bit to ensure it started
            await asyncio.sleep(5)
            
            # Check if process is running
            if process.poll() is None:
                logger.info(f"Worker {worker_id} process started with PID {process.pid}")
                
                # Store process info
                await self.redis.hset(
                    f"{KEY_PREFIX}:worker_process:{worker_id}",
                    mapping={
                        "pid": str(process.pid),
                        "start_time": datetime.now().isoformat(),
                        "device": device
                    }
                )
                
                return True
            else:
                logger.error(f"Worker {worker_id} process exited immediately")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting worker {worker_id}: {e}")
            return False
    
    async def _kill_worker_process(self, worker_id: str):
        """Kill existing worker process."""
        try:
            # Get process info from Redis
            process_info = await self.redis.hgetall(f"{KEY_PREFIX}:worker_process:{worker_id}")
            if process_info:
                pid = int(process_info.get(b"pid", 0))
                if pid:
                    import os
                    import signal
                    try:
                        # Try to terminate the whole process group first
                        try:
                            if hasattr(os, "killpg"):
                                os.killpg(pid, signal.SIGTERM)
                                logger.info(f"Sent SIGTERM to PGID {pid} for worker {worker_id}")
                            else:
                                os.kill(pid, signal.SIGTERM)
                                logger.info(f"Sent SIGTERM to worker {worker_id} (PID {pid})")
                        except Exception:
                            os.kill(pid, signal.SIGTERM)
                            logger.info(f"Sent SIGTERM to worker {worker_id} (PID {pid})")
                        
                        # Wait up to 10s, then escalate to SIGKILL
                        import time
                        deadline = time.time() + 10
                        while time.time() < deadline:
                            try:
                                os.kill(pid, 0)
                                await asyncio.sleep(0.5)
                            except OSError:
                                # Process gone
                                break
                        else:
                            try:
                                if hasattr(os, "killpg"):
                                    os.killpg(pid, signal.SIGKILL)
                                    logger.info(f"Sent SIGKILL to PGID {pid} for worker {worker_id}")
                                else:
                                    os.kill(pid, signal.SIGKILL)
                                    logger.info(f"Sent SIGKILL to worker {worker_id} (PID {pid})")
                            except ProcessLookupError:
                                pass
                    except ProcessLookupError:
                        logger.info(f"Worker {worker_id} process (PID {pid}) not found")
                    
                    # Clean up process info
                    await self.redis.delete(f"{KEY_PREFIX}:worker_process:{worker_id}")
                    
        except Exception as e:
            logger.error(f"Error killing worker process {worker_id}: {e}")


async def main():
    """Main entry point for worker monitor."""
    # Parse CLI args
    parser = argparse.ArgumentParser(description="KernelGym Worker Monitor")
    parser.add_argument("--persistent", action="store_true", help="Enable persistent monitoring (restart workers even if heartbeat keys disappear)")
    args = parser.parse_args()

    # Configure logging
    logger = setup_logging("worker_monitor")
    
    # Initialize Redis connection with readiness wait
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
                logger.warning(f"[monitor] Redis not ready (loading data): {e}. Retrying...")
            except (RedisConnectionError, RedisTimeoutError) as e:
                last_err = e
                logger.warning(f"[monitor] Redis connection not ready: {e}. Retrying...")
            except Exception as e:
                last_err = e
                logger.warning(f"[monitor] Redis ping error: {e}. Retrying...")
            if (asyncio.get_event_loop().time() - start) > timeout_sec:
                raise RuntimeError(f"Redis not ready within {timeout_sec}s: {last_err}")
            await asyncio.sleep(interval_sec)

    redis_client = await _wait_for_redis_ready(settings.redis_url)
    logger.info("Redis connection established for worker monitor")
    
    # Create and start monitor
    monitor = WorkerMonitor(redis_client, persistent=bool(args.persistent))
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker monitor error: {e}")
        sys.exit(1)
    finally:
        await monitor.stop()
        await redis_client.close()


if __name__ == "__main__":
    import os
    asyncio.run(main())