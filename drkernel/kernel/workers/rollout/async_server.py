# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import heapq
import importlib
import logging
import os
import random
import socket
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
import fastapi
import httpx
import numpy as np
import ray
import torch
import uvicorn
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local

from kernel.workers.rollout.vllm_rollout.vllm_async_engine import (
    AsyncvLLMEngine,
    MultiTurnAsyncvLLMEngine,
)

from kernel.workers.rollout.vllm_rollout.vllm_async_engine_multi_iter import (
    MultiIterAsyncvLLMEngine
)



logger = logging.getLogger(__file__)


class AsyncLLMEngineManager:
    """AsyncLLMEngineManager manage a group of vllm instances, i.e AsyncvLLMEngine."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup,
        tokenizer,
        reward_fn=None,
        val_reward_fn=None,
        *,
        scheduler_kwargs: Dict[str, Any] = None,
    ):
        """Initialize AsyncLLMEngineManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
            scheduler_kwargs: Dict[str, Any], kwargs for chat scheduler.
        """
        self.config = config
        self.worker_group = worker_group
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.tokenizer = tokenizer
        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        # Simple configuration for overall safety timeout
        self.max_timeout = 86400  # Maximum timeout in seconds (~24 hours) as safety net

        # Simple configuration for overall safety timeout
        self.max_timeout = 86400  # Maximum timeout in seconds (~24 hours) as safety net

        workers_info = ray.get(
            [
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.worker_group.workers
            ]
        )
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size

        rollout_backend = self.config.rollout.get("backend", "vllm")
        if rollout_backend in ("openai", "openai_sdk"):
            from kernel.workers.rollout.vllm_rollout.openai_async_engine_multi_iter import (
                AsyncvLLMEngine as OpenAIAsyncEngine,
                MultiIterAsyncvLLMEngine as OpenAIMultiIterEngine,
            )

            engine_class = OpenAIMultiIterEngine if self.config.rollout.multi_turn.enable else OpenAIAsyncEngine
        else:
            engine_class = MultiTurnAsyncvLLMEngine if self.config.rollout.multi_turn.enable else AsyncvLLMEngine
            if self.config.rollout.multi_turn.multi_iteration.enable:
                engine_class = MultiIterAsyncvLLMEngine

        config.rollout.max_model_len = (
            config.rollout.max_model_len
            if config.rollout.max_model_len
            else config.rollout.prompt_length + config.rollout.response_length
        )

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: engine_class.options(
                    # make sure AsyncvLLMEngine colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(
                    config,
                    self.rollout_dp_size,
                    rollout_dp_rank,
                    self.worker_group.name_prefix,
                    self.tokenizer,
                    reward_fn,
                    val_reward_fn,
                )
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    # address = ray.get(server.get_server_address.remote())
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def wake_up(self):
        """Wake up all vllm instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""

        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.wake_up()

        chunkes = prompts.chunk(len(self.async_llm_servers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.async_llm_servers, chunkes, strict=True)
            ]
        )
        # filter out output which is None
        outputs = [output for output in outputs if output is not None]
        if len(outputs) == 0:
            return None

        output = DataProto.concat(outputs)
        if self.config.rollout.free_cache_engine:
            self.sleep()
        return output


class StandaloneVLLMEngineManager:
    """Standalone vLLM manager that does not rely on FSDP rollout workers."""

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        reward_fn=None,
        val_reward_fn=None,
        *,
        total_gpus: int,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        if total_gpus < 1:
            raise ValueError("standalone_vllm requires total_gpus >= 1")
        if total_gpus % self.rollout_tp_size != 0:
            raise ValueError(
                f"standalone_vllm requires total_gpus ({total_gpus}) divisible by tensor_model_parallel_size "
                f"({self.rollout_tp_size})"
            )
        self.rollout_dp_size = total_gpus // self.rollout_tp_size
        self.world_size = self.rollout_dp_size * self.rollout_tp_size

        rollout_backend = self.config.rollout.get("backend", "vllm")
        if rollout_backend in ("openai", "openai_sdk"):
            from kernel.workers.rollout.vllm_rollout.openai_async_engine_multi_iter import (
                AsyncvLLMEngine as OpenAIAsyncEngine,
                MultiIterAsyncvLLMEngine as OpenAIMultiIterEngine,
            )

            engine_class = OpenAIMultiIterEngine if self.config.rollout.multi_turn.enable else OpenAIAsyncEngine
        else:
            engine_class = MultiTurnAsyncvLLMEngine if self.config.rollout.multi_turn.enable else AsyncvLLMEngine
            if self.config.rollout.multi_turn.multi_iteration.enable:
                engine_class = MultiIterAsyncvLLMEngine

        self.config.rollout.max_model_len = (
            self.config.rollout.max_model_len
            if self.config.rollout.max_model_len
            else self.config.rollout.prompt_length + self.config.rollout.response_length
        )

        user = os.environ.get("USER", "user")
        cache_root = f"/tmp/{user}"
        runtime_env = {
            "env_vars": {
                "VERL_VLLM_DISTRIBUTED_BACKEND": "local",
                "XDG_CACHE_HOME": f"{cache_root}/.cache",
                "TORCHINDUCTOR_CACHE_DIR": f"{cache_root}/torchinductor",
            }
        }
        self.async_llm_servers = [
            engine_class.options(
                num_gpus=self.rollout_tp_size,
                runtime_env=runtime_env,
                name=f"standalone_async_llm_server_{rollout_dp_rank}",
            ).remote(
                self.config,
                self.rollout_dp_size,
                rollout_dp_rank,
                "standalone_vllm",
                self.tokenizer,
                reward_fn,
                val_reward_fn,
            )
            for rollout_dp_rank in range(self.rollout_dp_size)
        ]

        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def wake_up(self):
        """Wake up all vllm instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""
        assert self.config.rollout.free_cache_engine, "Only free cache engine is supported for now."
        if self.config.rollout.free_cache_engine:
            self.wake_up()

        chunkes = prompts.chunk(len(self.async_llm_servers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.async_llm_servers, chunkes, strict=True)
            ]
        )
        outputs = [output for output in outputs if output is not None]
        if len(outputs) == 0:
            return None

        output = DataProto.concat(outputs)
        if self.config.rollout.free_cache_engine:
            self.sleep()
        return output
