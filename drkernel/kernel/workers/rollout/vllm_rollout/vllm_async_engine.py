import asyncio
import json
import logging
import os
import pickle
import random
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import cloudpickle
import numpy as np
import ray
import torch
import zmq
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.async_server import AsyncServerBase
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase


from kernel.workers.agent import BaseAgent, KernelAgent
from verl_patch.workers.code.agent_env import (
    BaseEnv,
    FinishReasonTypeEnum,
    create_environment,
)
from verl_patch.workers.code.agent import (
    MathAgent,
    CodeAgent,
    SearchAgent,
    FileSearchAgent,
    MathNeuralInterpreterAgent,
)


def create_agent(agent_type: str, tokenizer) -> BaseAgent:
    """Factory function to create agent instances based on configuration."""
    agent_map = {
        'MathAgent': MathAgent,
        'MathNeuralInterpreterAgent': MathNeuralInterpreterAgent,
        'CodeAgent': CodeAgent,
        'SearchAgent': SearchAgent,
        'FileSearchAgent': FileSearchAgent,
        'KernelAgent': KernelAgent,
    }

    if agent_type not in agent_map:
        raise ValueError(f"Unsupported agent type: {agent_type}. Supported types: {list(agent_map.keys())}")

    return agent_map[agent_type](tokenizer)
# from verl_patch.workers.code.reward_manager import CodeRewardManager, MathRewardManager

# from verl_patch.workers.code.rollout.schemas import AsyncRolloutRequestWithLogProbs

from collections import defaultdict
import re


@ray.remote
class SlowestRequestTracker:
    """Ray actor to track the slowest request across all workers."""

    def __init__(self):
        self.slowest_request_time = 0.0
        self.current_step = -1

    def update_slowest_time(self, request_time: float, global_step: int) -> bool:
        """Update the slowest request time if this is slower.

        Args:
            request_time: The time taken for the current request
            global_step: The current global step

        Returns:
            bool: True if this is the new slowest request, False otherwise
        """
        # Reset tracking every 10 steps
        if global_step % 10 == 0 and global_step > self.current_step:
            self.slowest_request_time = 0.0
            self.current_step = global_step

        # Check if this is the slowest request
        if request_time > self.slowest_request_time + 5.0:
            self.slowest_request_time = request_time
            return True
        return False

    def get_slowest_time(self) -> float:
        """Get the current slowest request time."""
        return self.slowest_request_time


def _create_logfire_logger(service_name: str = "vllm-async-engine"):
    """Create and configure a logfire logger instance.

    Args:
        service_name: The service name for logfire configuration

    Returns:
        logfire logger instance or None if initialization failed
    """
    try:
        LOGFIRE_KEY = os.getenv('LOGFIRE_KEY')
        if LOGFIRE_KEY:
            import logfire

            # get all workers' ip address and log them into logfire as tag

            logfire.configure(token=LOGFIRE_KEY, service_name=service_name, service_version="v1.0.0", scrubbing=False)
            logging.info(f"Logfire initialized successfully with service_name: {service_name}")
            return logfire
    except Exception as e:
        logging.warning(f"Failed to initialize Logfire: {e}")

    return None


def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        print("initializing ray ...")
        runtime_environment = {
            "env_vars": {"VLLM_USE_V1": "1", "FLASH_ATTENTION_DETERMINISTIC": "1", "VERL_AUTO_PADDING": "1"}
        }
        ray.init(namespace=namespace, runtime_env=runtime_environment, address='auto')
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
    ]

    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )

    def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank

    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")

    return workers


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)

        del method

        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers]
        )
        return outputs

    def check_health(self):
        return


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    logprobs: list[float]
    num_turns: int = 0
    reward: float = None
    reward_extra_info: dict = None


def infer_entry_point(ground_truth: str, default: str = "Model") -> str:
    if not ground_truth:
        return default
    match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|:)", ground_truth)
    return match.group(1) if match else default

def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        ray.init(namespace=namespace, address='auto')
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
    ]
    # Fallback for environments where RayWorkerGroup does not wrap workers in WorkerDict.
    if len(actor_names) == 0:
        actor_names = [
            actor_name
            for actor_name in ray.util.list_named_actors()
            if actor_name.startswith(f"{wg_prefix}AsyncActorRolloutRefWorker")
        ]

    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )

    def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank

    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")

    return workers


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)

        del method

        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers]
        )
        return outputs

    def check_health(self):
        return


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    logprobs: list[float] = []  # Add log probabilities
    num_turns: int = 0
    reward_tensor: torch.Tensor
    reward_extra_info: dict = defaultdict(list)


@ray.remote(num_cpus=1)
class AsyncvLLMEngine:
    """
    AsyncvLLMEngine is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    2. AsyncLLM spawn EngineCore in subprocess.
    3. EngineCore initialize ExternalRayDistributedExecutor.
    4. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    5. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str, tokenizer,
                 reward_fn = None, val_reward_fn = None,
                 ):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        # super().__init__()

        self.config = config
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.tokenizer = tokenizer
        self.engine: AsyncLLM = None
        self.pad_token_id = self.tokenizer.pad_token_id
        
        self.ref_reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn  

    def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        self.max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            max_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                value = config.get(k)
                if isinstance(value, ListConfig):
                    value = list(value)
                elif isinstance(value, DictConfig):
                    value = OmegaConf.to_container(value, resolve=True)  # need import OmegaConf
                kwargs[k] = value
        print(f"override_generation_config: {kwargs}")

        self.sampling_params = SamplingParams(**kwargs)

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "zeromq")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # Qian: this is a known issue of verl, see PR: https://github.com/volcengine/verl/pull/2068/files
            # disable_mm_preprocessor_cache=False,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            # disable_log_stats=config.disable_log_stats,
            disable_log_stats=False,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False)
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()

    async def _async_rollout_a_prompt(
        self, messages: list[dict[str, Any]],
         tokens, 
         ground_truth,
         entry_point, 
         uuid, 
         sampling_params, 
         is_validate: bool = False, 
         **kwargs
    ) -> DataProto:
        loop = asyncio.get_running_loop()
        request_id = uuid4().hex
        prompt_ids = await loop.run_in_executor(
            None, lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=tokens)

        outputs = self.engine.generate(
            prompt=prompt,  # because we have already convert it to prompt token id
            sampling_params=sampling_params,
            request_id=request_id,
        )
        async for res in outputs:
            results = res
        content = results.outputs[0].text
        content_token_ids = results.outputs[0].token_ids

        # Collect log probabilities for each sampled token
        logprobs = []
        if results.outputs[0].logprobs:
            for i, logprob_dict in enumerate(results.outputs[0].logprobs):
                # Get the logprob of the actual sampled token
                sampled_token_id = content_token_ids[i]
                if sampled_token_id in logprob_dict:
                    logprobs.append(logprob_dict[sampled_token_id].logprob)
                else:
                    # Fallback if the token is not in the logprobs dict
                    logprobs.append(-1.0)  # Use -1.0 as default for missing logprobs

        response_ids = content_token_ids
        response_mask = [1] * len(content_token_ids)
        response_length = self.config.rollout.response_length
        
        # TODO: Tianjian
        # Add reward function
        reward_fn = self.val_reward_fn if is_validate else self.ref_reward_fn
        # TODO: Wei
        # reward_fn: compute_kernel_reward_batch(solution_strs, ground_truths, entry_points, **kwargs)
        reward_kwargs = {**kwargs, "response_length": response_length}
        reward_dict = await loop.run_in_executor(
            None,
            lambda: reward_fn(
                response_ids,
                content,
                ground_truth,
                entry_point,
                uuid,
                **reward_kwargs,
            ),
        )
        reward_tensor = reward_dict["reward_tensor"]
        reward_extra_info = reward_dict["reward_extra_info"]
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:response_length],
            response_mask=response_mask[:response_length],
            logprobs=logprobs[:response_length],  # Add logprobs
            num_turns=2,
            reward_tensor=reward_tensor,
            reward_extra_info=reward_extra_info,  # Add extra info
        )
        return output

    async def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        # req_list = self._preprocess_prompt_to_async_rollout_requests(
        #     prompts,
        #     n=1 if is_validate else self.config.rollout.n,
        # )

        if not is_validate:
            if "n" in prompts.meta_info:
                n = prompts.meta_info["n"]
            else:
                n = self.config.rollout.n
            prompts = prompts.repeat(repeat_times=n, interleave=True)

        config = self.config.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=1 if config.calculate_log_probs else None,  # Ensure logprobs are collected
        )
        # override sampling params for validation
        if prompts.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        stop_token_ids = config.stop_token_ids
        if stop_token_ids is not None:
            sampling_params["stop_token_ids"] = list(stop_token_ids)

        if is_validate:
            val_stop_token_ids = config.val_kwargs.stop_token_ids
            if val_stop_token_ids is not None:
                sampling_params["stop_token_ids"] = list(val_stop_token_ids)

        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        tokens_ids = prompts.non_tensor_batch["raw_prompt_ids"]

        # ground_truths = prompts.non_tensor_batch["ground_truth"]
        # entry_points = prompts.non_tensor_batch["entry_point"]
        # uuids = prompts.non_tensor_batch["uuid"]

        ground_truths = prompts.non_tensor_batch.get("ground_truth")
        if ground_truths is None and "reward_model" in prompts.non_tensor_batch:
            ground_truths = [rm.get("ground_truth") for rm in prompts.non_tensor_batch["reward_model"]]

        entry_points = prompts.non_tensor_batch.get("entry_point")
        if entry_points is None:
            extra_info = prompts.non_tensor_batch.get("extra_info", [{}] * len(raw_prompts))
            entry_points = [
                # info.get("entry_point", infer_entry_point(ground_truth))
                info.get("entry_point", "Model")
                for info, ground_truth in zip(extra_info, ground_truths, strict=True)
            ]

        uuids = prompts.non_tensor_batch.get("uuid")
        if uuids is None:
            extra_info = prompts.non_tensor_batch.get("extra_info", [{}] * len(raw_prompts))
            uuids = [info.get("uuid") or info.get("problem_id") or uuid4().hex for info in extra_info]

        tasks = []
        for messages, tokens, ground_truth, entry_point, uuid in zip(raw_prompts, tokens_ids, ground_truths, entry_points, uuids):
            if not isinstance(messages, list):
                messages = messages.tolist()
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            elif hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            tasks.append(asyncio.create_task(self._async_rollout_a_prompt(messages, 
                                                                            tokens, 
                                                                            ground_truth, 
                                                                            entry_point, 
                                                                            uuid, 
                                                                            sampling_params, 
                                                                            is_validate, 
                                                                            **kwargs)))

        outputs = await asyncio.gather(*tasks)

        return self._postprocess(outputs)

    def _postprocess(self, inputs: list[AgentLoopOutput]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # TODO: Wei
        # To pad reward tensor

        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in inputs],
            padding="max_length",
            max_length=self.config.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]
        assert (
            response_ids.shape == response_mask.shape
        ), f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        response_mask = response_mask * response_attention_mask

        # pad reward tensor
        reward_tensors = []
        for input in inputs:
            reward = input.reward_tensor
            if not isinstance(reward, torch.Tensor):
                reward = torch.as_tensor(reward, dtype=torch.float32)
            else:
                reward = reward.to(dtype=torch.float32)

            if reward.dim() == 0:
                reward = reward.unsqueeze(0)
            elif reward.dim() > 1:
                reward = reward.view(-1)

            reward_tensors.append(reward)

        reward_tensor = pad_sequence(
            reward_tensors,
            batch_first=True,
            padding_value=0.0,
        )

        max_reward_len = self.config.rollout.response_length
        if reward_tensor.shape[1] < max_reward_len:
            pad_size = max_reward_len - reward_tensor.shape[1]
            reward_tensor = torch.nn.functional.pad(reward_tensor, (0, pad_size))
        elif reward_tensor.shape[1] > max_reward_len:
            reward_tensor = reward_tensor[:, :max_reward_len]

        print(f"reward_tensor: {reward_tensor.shape}")
        print(f"response length vs. reward tensor nonzero: {response_ids.shape[1]} vs. {reward_tensor.nonzero()}")

        if self.config.rollout.calculate_log_probs:
            # rollout_log_probs
            # Pad and convert logprobs to float32 tensor
            max_response_length = self.config.rollout.response_length
            padded_logprobs = []
            for input in inputs:
                # Pad logprobs to max_response_length with -1.0 (same as vllm_rollout_spmd.py)
                padded_logprob = input.logprobs + [-1.0] * (max_response_length - len(input.logprobs))
                padded_logprobs.append(padded_logprob[:max_response_length])

            # Convert to tensor
            rollout_log_probs = torch.tensor(padded_logprobs, dtype=torch.float32, device=response_ids.device)

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask


        # TODO Tianjian
        # rewards
        # reward_tensor = torch.cat([torch.tensor([input.reward], dtype=torch.float32) for input in inputs], dim=0)

        # reward extra info (non tensor batch format)
        reward_extra_info_array = np.array(
            [input.reward_extra_info for input in inputs], dtype=object
        )

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
                "reward_tensor": reward_tensor,  # [bsz, response_length]
            },
            batch_size=len(input_ids),
        )
        if self.config.rollout.calculate_log_probs:
            # rollout_log_probs
            batch["rollout_log_probs"] = rollout_log_probs

        num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "__num_turns__": num_turns,
                "reward_extra_info": reward_extra_info_array,
            },
        )

class MultiTurnStats(BaseModel):
    """Statistics for multi-turn conversations."""

    num_turns: int = 0  # Total number of turns
    contain_void_turn: int = 0  # Whether any turn had no tool call (void turn)
    finish_reason: str = ""  # How the conversation ended (e.g., "stop", "length", "error", etc.)
    cache_hits: int = 0  # Number of turns where results were retrieved from cache
    cache_misses: int = 0  # Number of turns where results had to be computed


class MultiTurnRequest(BaseModel):
    """Agent request for multi-turn interactions."""

    messages: list[dict[str, str]]
    # store the response token id corresponding to each turn
    response_token_ids: list[list[int]]
    # how many messages are used as the prefix for this response
    response_turns: list[int]
    # Multi-turn statistics
    stats: MultiTurnStats = Field(default_factory=MultiTurnStats)
    # Configuration for masking void turns
    mask_void_turn: bool = True
    # Extra information from dataset
    extra_info: dict = Field(default_factory=dict)
    # Ground truth from dataset
    ground_truth: str = None
    # Entry point from dataset
    entry_point: str = None
    # Unique request id for this multi-turn request
    uuid: str = None

    def add_message(self, message: str, is_tool_call: bool = False, response_token_ids: List[int] = None):
        """Add a message to the conversation history."""
        role = "assistant" if not is_tool_call else "user"
        # For assistant messages, store the index BEFORE adding the message
        # This ensures the prompt doesn't include the response itself
        if not is_tool_call and response_token_ids is not None:
            # Turns are decided based on x-th assistant messages
            self.response_turns.append(len(self.messages))
            self.response_token_ids.append(response_token_ids)
        elif not is_tool_call:
            # response_token_ids is None
            assert False, "response_token_ids must be provided for assistant messages"

        # Add message after storing the index
        self.messages.append({"role": role, "content": message})

    def get_num_turns(self):
        return len(self.response_turns)

    def finalize(self, reward_scores: List[float], finish_reason_type: str):
        contain_void_turn = False
        if finish_reason_type == FinishReasonTypeEnum.NO_TOOL_CALL:
            contain_void_turn = True

        if finish_reason_type == FinishReasonTypeEnum.ERROR or finish_reason_type == FinishReasonTypeEnum.ASYNC_TIMEOUT:
            contain_error = True
        else:
            contain_error = False

        is_meaningful_turn = [True]
        max_rewards = reward_scores[0]
        for turn_idx in range(1, len(reward_scores)):
            if max_rewards > 0.0:
                if reward_scores[turn_idx] > 0.0:
                    is_meaningful_turn.append(True)
                else:
                    is_meaningful_turn.append(False)
            else:
                is_meaningful_turn.append(True)

            max_rewards = max(max_rewards, reward_scores[turn_idx])

        # Update stats before finalizing
        self.stats.num_turns = len(self.response_turns)
        self.stats.contain_void_turn = 1 if contain_void_turn else 0
        self.stats.finish_reason = finish_reason_type

        # Determine loss_mask based on the finalization logic
        # For now, mask out responses if contain_void_turn is True
        # This can be customized based on other criteria
        loss_mask: list[int] = []
        for turn_idx in range(len(self.response_turns)):
            # Only mask if mask_void_turn is enabled AND contain_void_turn is True
            # should_mask = (self.mask_void_turn and contain_void_turn) or contain_error
            should_mask = (self.mask_void_turn and not is_meaningful_turn[turn_idx]) or contain_error
            # 1 for keep, 0 for mask out
            should_keep = 1 if not should_mask else 0
            loss_mask.append(should_keep)

        """Finalize the agent request."""
        return {
            "messages": self.messages,
            "response_token_ids": self.response_token_ids,
            "response_turns": self.response_turns,
            "reward_scores": reward_scores,
            "finish_reason_type": finish_reason_type,
            "loss_mask": loss_mask,  # Per-turn loss mask
            "stats": self.stats,  # Return the stats object
        }


class MultiTurnOutput(BaseModel):
    multi_prompt_ids: list[list[int]]
    multi_response_ids: list[list[int]]
    multi_logprobs: list[list[float]]
    multi_loss_mask: list[int]  # Per-turn loss mask from finalization
    multi_rewards: list[float] = None

    # Multi-turn statistics
    stats: MultiTurnStats
    # Unique request id for this multi-turn request
    request_id: str
    # Complete multi-turn conversation messages for logging
    messages: list[dict] = None  # Contains the complete conversation messages
    # Extra info to keep for each turn (per-sample)
    multi_reward_extra_info: list[dict] = None


@ray.remote(num_cpus=1)
class MultiTurnAsyncvLLMEngine:
    """MultiTurnAsyncLLMEngine extends AsyncvLLMEngine with multi-turn agent capabilities.

    This engine supports multiple conversation turns with tool usage.
    """

    def __init__(
        self,
        config: DictConfig,
        vllm_dp_size: int,
        vllm_dp_rank: int,
        wg_prefix: str,
        tokenizer,
        reward_fn,
        val_reward_fn,
    ):
        """Initialize MultiTurnAsyncLLMEngine.

        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
            tokenizer: The tokenizer for encoding/decoding text.
        """
        # Initialize AsyncLLM engine attributes
        self.config = config
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.tokenizer = tokenizer
        self.engine = None
        self.pad_token_id = tokenizer.pad_token_id

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.use_async_reward = True    # in kernel grading, we use the async reward manager

        # if hasattr(reward_fn, "score_raw_responses_async") and hasattr(val_reward_fn, "score_raw_responses_async"):
        #     self.use_async_reward = True
        # else:
        #     self.use_async_reward = False

        # Initialize Logfire with run_name as service name
        run_name = config.rollout.get("experiment_name", "vllm-async-engine")
        self.logfire_logger = _create_logfire_logger(service_name=run_name)

        # Store configuration for agent and environment
        self.max_agent_turns = config.rollout.multi_turn.max_user_turns
        self.mask_void_turn = config.rollout.multi_turn.mask_void_turn
        # self.mask_void_turn = False    # in kernel grading, we do not have exact definition of void turn

        # Agent and environment configuration
        self.agent_type = config.rollout.multi_turn.get("agent_type", "MathAgent")
        self.env_type = config.rollout.multi_turn.get("env_type", "MathSandboxEnv")
        self.per_turn_prompts = self.load_per_turn_prompts()

        # Calculate expected environment max timeout, this is for final timeout handling, which includes max_retry and timeout
        self.env_max_timeout = 60.0

        # Track the slowest request
        self.slowest_request_info = None
        # Initialize or get the shared tracker
        try:
            self.slowest_tracker = ray.get_actor("SlowestRequestTracker")
        except ValueError:
            # If the tracker doesn't exist, create it
            try:
                self.slowest_tracker = SlowestRequestTracker.options(name="SlowestRequestTracker").remote()
            except ValueError:
                # Another instance may have created it concurrently
                try:
                    self.slowest_tracker = ray.get_actor("SlowestRequestTracker")
                except ValueError:
                    logging.warning("Failed to create or get SlowestRequestTracker, disabling slowest request tracking")
                    self.slowest_tracker = None

        # Timeout handling configuration
        self.slowest_tracker_timeout = config.rollout.multi_turn.get("slowest_tracker_timeout", 2.0)

        # Timeout handling configuration (track train/validation separately)
        self.token_rate_history_train = deque(maxlen=1000)
        self.token_rate_history_val = deque(maxlen=1000)
        self.timeout_buffer = 2.0  # Fixed buffer factor

        # Track request deadlines for cleanup
        self.request_deadlines = {}  # Map request_id to deadline timestamp

    def load_per_turn_prompts(self) -> dict[str, dict[str, Any]]:
        """Load turn selection rules for the multi-turn agent loop."""
        per_turn_prompts = {}

        # Get prompt config path from config, with fallback to default
        prompt_config_path = self.config.rollout.multi_turn.get("prompt_config_path", None)
        if prompt_config_path is None:
            return None

        # Load from configured path
        with open(prompt_config_path, encoding='utf-8') as fp:
            prompt_cfg = OmegaConf.create(fp.read())

        for prompt_config in prompt_cfg.per_turn_prompts:
            per_turn_prompts[prompt_config.name] = {
                "condition": prompt_config.condition,
                "history_mode": prompt_config.history_mode,
                "skip_env": prompt_config.skip_env,
                "response_truncation": prompt_config.get("response_truncation", None),
                "update_memory": prompt_config.get("update_memory", False),
                "template": prompt_config.template,
            }

        return per_turn_prompts

    def log_multiturn_messages(
        self,
        step: int,
        request_id: str,
        logging_messages: list[str],
        turn_rewards: list[float],
        turn_infos: list[dict | None],
        stats: MultiTurnStats,
        finish_reason: str,
        multi_turn_output: MultiTurnOutput,
        is_slowest: bool = False,
    ):
        """Log multi-turn conversation to Logfire with structured spans.

        Args:
            run_name: Experiment or run name
            request_id: Unique request identifier
            logging_messages: List of formatted log messages from the conversation
            turn_rewards: List of rewards for each turn
            stats: Multi-turn statistics
            finish_reason: How the conversation ended
        """
        if not self.logfire_logger:
            print("\n\n".join(logging_messages))
            return

        # Parse timing information from messages
        timings = []
        total_model_time = 0.0
        total_env_time = 0.0

        for msg in logging_messages:
            if "Model time:" in msg:
                parts = msg.split(" | ")
                if len(parts) >= 2:
                    model_time_str = parts[1].replace("Model time: ", "").replace("s", "")
                    try:
                        model_time = float(model_time_str)
                        total_model_time += model_time
                        timings.append(("model", model_time))
                    except:
                        pass
            elif "Env time:" in msg:
                parts = msg.split(" | ")
                if len(parts) >= 2:
                    env_time_str = parts[1].replace("Env time: ", "").replace("s", "")
                    try:
                        env_time = float(env_time_str)
                        total_env_time += env_time
                        timings.append(("env", env_time))
                    except:
                        pass

        # Create async interaction bar
        total_time = total_model_time + total_env_time
        if total_time > 0:
            # Dynamic bar width based on total time, max 20 chars
            # Scale: 1 char per 0.5 seconds, minimum 5 chars, maximum 20 chars
            bar_width = max(5, min(20, int(total_time)))
            bar_segments = []

            # Calculate segment widths proportionally
            accumulated_width = 0
            for i, (type_, duration) in enumerate(timings):
                # Calculate raw width
                raw_width = (duration / total_time) * bar_width

                # For the last segment, use remaining width to avoid rounding errors
                if i == len(timings) - 1:
                    segment_width = bar_width - accumulated_width
                else:
                    segment_width = round(raw_width)
                    accumulated_width += segment_width

                if segment_width > 0:
                    if type_ == "model":
                        bar_segments.append("▓" * segment_width)
                    elif type_ == "env":
                        bar_segments.append("░" * segment_width)

            interaction_bar = "".join(bar_segments)
            timing_summary = (
                f"[{interaction_bar}] {total_time:.2f}s (▓ Model: {total_model_time:.2f}s ░ Env: {total_env_time:.2f}s)"
            )
        else:
            timing_summary = "No timing data available"

        # Create a span for the entire multi-turn conversation with enhanced preview
        span_prefix = '🐌 SLOWEST ' if is_slowest else '🎯 RANDOM'

        with self.logfire_logger.span(
            '{prefix} - step {step} - {num_turns} turns - {finish_reason} - {timing_summary}',
            prefix=span_prefix,
            step=step,
            num_turns=stats.num_turns,
            finish_reason=finish_reason,
            timing_summary=timing_summary,
            request_id=request_id,
            total_turns=stats.num_turns,
            total_model_time=total_model_time,
            total_env_time=total_env_time,
            total_time=total_time,
            is_slowest=is_slowest,
        ):
            # Parse and log each turn
            current_turn = 0
            turn_info_index = 0
            for i, msg in enumerate(logging_messages):
                if "Model Response:" in msg:
                    # Extract turn number and timings from the message
                    parts = msg.split(" | ")
                    model_time = parts[1] if len(parts) > 1 else ""  # e.g., "Model time: 1.23s"
                    model_response = parts[2].replace("Model Response: ", "") if len(parts) > 2 else ""

                    # Create a span for this turn
                    self.logfire_logger.info(
                        '🔄 turn_{turn_number}: {response_preview} ...',
                        turn_number=current_turn + 1,
                        response_preview=model_response[:50],
                        timing=model_time,
                        response=model_response,
                        reward=turn_rewards[current_turn] if current_turn < len(turn_rewards) else None,
                    )
                elif "Tool Response:" in msg:
                    # Log tool response within the same turn
                    parts = msg.split(" | ")
                    env_time = parts[1] if len(parts) > 1 else ""  # e.g., "Env time: 0.45s"
                    tool_response = parts[2].replace("Tool Response: ", "") if len(parts) > 2 else ""

                    # Get turn_info for this turn
                    turn_info = turn_infos[turn_info_index] if turn_info_index < len(turn_infos) else None
                    turn_info_index += 1

                    # Log all tool_info key-value pairs
                    if turn_info:
                        # Create a dictionary with all turn_info data for logging
                        tool_info_data = {k: v for k, v in turn_info.items()}
                        self.logfire_logger.info(
                            '🔧 tool_execution: {response_preview} ...',
                            response_preview=tool_response[:50],
                            response=tool_response,
                            env_time=env_time,
                            **tool_info_data,  # Add all key-value pairs from turn_info
                        )
                    else:
                        self.logfire_logger.info(
                            '🔧 tool_execution: {response_preview} ...',
                            response_preview=tool_response[:50],
                            response=tool_response,
                            env_time=env_time,
                        )
                    current_turn += 1
                elif "Finalizing" in msg:
                    # Log the final summary
                    self.logfire_logger.info('📋 conversation_summary', message=msg)

            # Log aggregated statistics
            self.logfire_logger.info(
                '📊 conversation_stats',
                total_turns=stats.num_turns,
                has_void_turn=bool(stats.contain_void_turn),
                all_rewards=turn_rewards,
                cache_hits=stats.cache_hits,
                cache_misses=stats.cache_misses,
                cache_hit_rate=(
                    stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                    if (stats.cache_hits + stats.cache_misses) > 0
                    else 0.0
                ),
            )

            # Log the full multi-turn output for reference
            multi_turn_output_dict = multi_turn_output.dict()

            # decode the raw prompt and response in text for viewing
            decoded_prompts = [
                self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for ids in multi_turn_output.multi_prompt_ids
            ]
            decoded_responses = [
                self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                for ids in multi_turn_output.multi_response_ids
            ]
            multi_turn_output_dict['decoded_prompts'] = decoded_prompts
            multi_turn_output_dict['decoded_responses'] = decoded_responses
            self.logfire_logger.info('🗂️ multi_turn_output', output=multi_turn_output_dict)

    def init_engine(self):
        """Initialize vLLM engine and agent components."""
        # Initialize vLLM AsyncLLM engine - replicating AsyncvLLMEngine init_engine method
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        self.max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            max_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                value = config.get(k)
                # Convert OmegaConf types to native Python types
                if isinstance(value, ListConfig):
                    value = list(value)
                elif isinstance(value, DictConfig):
                    value = OmegaConf.to_container(value, resolve=True)
                kwargs[k] = value
        print(f"override_generation_config: {kwargs}")

        self.sampling_params = SamplingParams(**kwargs)

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "ray")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # Qian: this is a known issue of verl, see PR: https://github.com/volcengine/verl/pull/2068/files
            # disable_mm_preprocessor_cache=False,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            # disable_log_stats=config.disable_log_stats,
            disable_log_stats=False,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.vllm_config = vllm_config
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        """Create engine configuration for vLLM AsyncLLM.

        Args:
            engine_args: AsyncEngineArgs for configuring the engine.

        Returns:
            Engine configuration object.
        """
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False)
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    def _resolve_multi_turn_rewards(self, turn_rewards: list[float], turn_speedups: list[float], turn_correctness: list[bool]) -> list[float]:
        """Resolve multi-turn rewards according to turn speedups and correctness.

        We follow the design:
        The later turn must be better than the earlier turn.
        1. If previous turn is correct, the later turn must be faster than the previous turn.
        2. If previous turn is incorrect, the later turn must be at least correct.
        Match one of the above two conditions, the reward could be the original reward, otherwise the reward is 0.0.

        Args:
            turn_rewards: List of turn rewards.
            turn_speedups: List of turn speedups.
            turn_correctness: List of turn correctness.
        """
        #TODO weiliu: I am not sure whether we should use the speedups and correctness to resolve the rewards, 
        # but I think it is not necessary for now.
        # Since we already used the per-turn advantage computations

        finlized_turn_rewards = turn_rewards
        # max_speedup = -1.0
        # max_correctness = False
        # for turn_idx in range(len(turn_rewards)):
        #     if not max_correctness:
        #         if turn_correctness[turn_idx]:
        #             max_correctness = True
        #             max_speedup = turn_speedups[turn_idx]
        #             finlized_turn_rewards.append(turn_rewards[turn_idx])
        #         else:
        #             finlized_turn_rewards.append(0.0)
        #     else:
        #         if turn_speedups[turn_idx] > max_speedup:
        #             max_speedup = turn_speedups[turn_idx]
        #             finlized_turn_rewards.append(turn_rewards[turn_idx])
        #         else:
        #             finlized_turn_rewards.append(0.0)
        
        return finlized_turn_rewards

    async def wake_up(self):
        """Wake up the engine from sleep mode."""
        await self.engine.wake_up()
        if not hasattr(self, "_watchdog_task") or self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._deadline_watchdog())

    async def sleep(self):
        """Put the engine into sleep mode."""
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()
        if getattr(self, "_watchdog_task", None):
            self._watchdog_task.cancel()
            self._watchdog_task = None

    async def _process_single_turn(
        self,
        req: MultiTurnRequest,
        request_id: str,
        sampling_params: dict,
        agent,
        env,
        is_validate: bool,
        uuid: str,
        entry_point: str,
        ground_truth: str,
        global_step: int = 0,
        tool_as_user: bool = True,
        actual_max_turns: int = None
    ) -> tuple[str | None, str | None, float, float, bool, bool, float, dict, list[int], list[float], list[int], dict]:
        """Process a single conversation turn.

        Args:
            req: The agent request containing conversation state.
            sampling_params: Sampling parameters for generation.
            is_validate: Whether the request is from validation.
            global_step: Current global step for logging.

        Returns:
            Tuple of (
                model_response: str | None,
                tool_response: str | None,
                model_time: float,
                env_time: float,
                turn_done: bool,
                turn_truncate: bool,
                turn_reward: float,
                turn_info: dict,
                model_response_token_ids: list[int],
                model_logprobs: list[float],
                prompt_token_ids: list[int],  # the actual prompt ids used for this turn
                env_state: dict,  # environment state for this turn
            ).
        """
        current_turn = req.get_num_turns()
        update_memory = False
        skip_env = False
        response_truncation = None
        make_up_tool_response = False # makeup user response firstly

        if self.per_turn_prompts is None:
            messages = req.messages
        else:
            current_prompt_config = None
            for prompt_name, prompt_config in self.per_turn_prompts.items():
                condition_expr = prompt_config["condition"]
                if eval(condition_expr):
                    current_prompt_config = prompt_config
                    break
            if current_prompt_config is None:
                messages = req.messages
            else:
                history_mode = current_prompt_config["history_mode"]
                update_memory = current_prompt_config["update_memory"]
                skip_env_prob = current_prompt_config["skip_env"]
                assert skip_env_prob >= 0.0 and skip_env_prob <= 1.0, "skip_env_prob must be between 0 and 1"
                skip_env = random.random() < skip_env_prob
                response_truncation = current_prompt_config["response_truncation"]
                prompt_template = current_prompt_config["template"]

                print(f"Prompt Template: {prompt_template}")
                
                if history_mode == "all":
                    messages = req.messages
                elif history_mode.startswith("initial+recent"):
                    suffix = history_mode[len("initial+recent") :]
                    recent_count = int(suffix)
                    if len(req.messages) > recent_count + 1:
                        messages = req.messages[:1] + req.messages[-recent_count:]
                    else:
                        messages = req.messages
                else:
                    raise ValueError(f"Invalid history mode: {history_mode}")
                if prompt_template is not None:
                    if not tool_as_user:
                        if current_turn == 0 and messages[-1]["role"] == "user":
                            messages[-1]["content"] = f'{messages[-1]["content"]}\n\n{prompt_template}'
                        else:
                            messages.append({"role": "user", "content": prompt_template})
                    else:
                        print("tool_as_user is True")
                        if current_turn > 0:
                            assert messages[-1]["role"] == "user", "The last message should be a user turn"
                            feedback = messages[-1]["content"]
                            #TODO: weiliu: please check the prompt template format and write in config yaml file
                            env_feedback_with_prompt = prompt_template.format(feedback=feedback)
                            print(f"env_feedback_with_prompt: {env_feedback_with_prompt}")
                            # messages.append({"role": "user", "content": env_feedback_with_prompt})
                            messages[-1]["content"] = env_feedback_with_prompt
        
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

        # Prepare parameters
        # Respect both model length limit and configured response length
        max_tokens = min(max(1, self.max_model_len - len(prompt_ids)), self.config.rollout.response_length)
        if max_tokens == 1:
            print(f"[debug] prompt overlong, max_tokens: {max_tokens}, prompt length: {len(prompt_ids)}")
            return (
                # placeholder, will not be used in loss computation
                "",
                None,
                0.0,
                0.0,
                True,
                False,
                0.0,
                {"finish_type": FinishReasonTypeEnum.LENGTH, "error": "Prompt Overlong"},
                [],
                [],
                prompt_ids,
                {},  # empty env_state
            )
        params = dict(sampling_params)
        params["logprobs"] = 1  # Always get logprobs

        # Create prompt and parameters
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        llm_params = SamplingParams(max_tokens=max_tokens, **params)

        # Add a timer here
        agent_start_time = asyncio.get_event_loop().time()

        # Qian: sometimes async will be very slow, we add a timeout here
        async_timeout = self._compute_adaptive_timeout(max_tokens=max_tokens, is_validate=is_validate)
        if async_timeout is not None:
            total_timeout = async_timeout + self.env_max_timeout + 5.0  # add buffer for env step
            self.request_deadlines[request_id] = {
                "deadline": asyncio.get_event_loop().time() + total_timeout,
                "global_step": global_step,
            }
            logging.debug(f"Request {request_id} deadline extended by {total_timeout:.2f}s")

        if async_timeout is None:
            outputs = self.engine.generate(prompt=prompt, sampling_params=llm_params, request_id=request_id)
            async for res in outputs:
                results = res
        else:
            try:
                results = None
                async with vllm_timeout(async_timeout):
                    async for res in self.engine.generate(
                        prompt=prompt, sampling_params=llm_params, request_id=request_id
                    ):
                        results = res
            except asyncio.TimeoutError:
                # This block will now execute correctly when the timeout is reached.
                print(f"Request {request_id}: Timed out after {async_timeout} seconds. The task will be abandoned.")
                if self.logfire_logger:
                    self.logfire_logger.warning(
                        f"Request timeout (asyncio) | request_id: {request_id} | step {global_step}",
                        request_id=request_id,
                        global_step=global_step,
                        timeout_seconds=async_timeout,
                        total_timeout=total_timeout,
                    )
                # Try aborting the request cleanly
                await self.engine.abort(request_id)
                self.clear_request_tracking(request_id)

                finish_reason = FinishReasonTypeEnum.ASYNC_TIMEOUT
                return (
                    # placeholder, will not be used in loss computation
                    None,
                    None,
                    async_timeout,
                    0.0,
                    True,
                    False,
                    0.0,
                    {"finish_type": finish_reason, "error": "LLM generation timeout"},
                    [],
                    [],
                    prompt_ids,
                    {},  # empty env_state
                )

        if results is None or results.outputs[0].finish_reason == "abort":
            # This can happen if the request was aborted due to timeout
            finish_reason = FinishReasonTypeEnum.ASYNC_TIMEOUT
            return (
                # placeholder, will not be used in loss computation
                None,
                None,
                async_timeout if async_timeout is not None else 0.0,
                0.0,
                True,
                False,
                0.0,
                {"finish_type": finish_reason, "error": "LLM generation aborted"},
                [],
                [],
                prompt_ids,
                {},  # empty env_state
            )

        # Extract content and token IDs
        output = results.outputs[0]
        response = output.text
        response_token_ids = output.token_ids
        agent_end_time = asyncio.get_event_loop().time()
        agent_spend_time = agent_end_time - agent_start_time

        # Update token rate history
        self._record_generation_stats(
            output_tokens=len(response_token_ids),
            generation_time=agent_spend_time,
            is_validate=is_validate,
        )

        # Agent thought generation, usually it is very fast and deterministic
        agent_result = await agent.generate_thought_and_action(response_token_ids, response_truncation)
        # Unpack agent result
        #TODO weiliu: handle the agent done case, here I think we could just assign the max turns
        model_response, model_response_token_ids, action, agent_done, agent_info = agent_result

        # Extract logprobs efficiently
        model_logprobs = []
        if output.logprobs:
            for i in range(len(model_response_token_ids)):
                token_id = model_response_token_ids[i]
                model_logprobs.append(output.logprobs[i].get(token_id).logprob)
        assert len(model_response_token_ids) == len(model_logprobs), "Mismatch in token IDs and logprobs length"

        #TODO: weiliu: In kernel agent, every turn is defined as a complete code generation turn.
        # if agent_done:
            # If the agent is done, skip environment step
            # return (
            #     model_response,
            #     None,
            #     agent_spend_time,
            #     0.0,
            #     agent_done,
            #     False,
            #     0.0,
            #     agent_info,
            #     model_response_token_ids,
            #     model_logprobs,
            #     prompt_ids,
            # )

        # if skip_env:
        #     return (
        #         model_response,
        #         None,
        #         agent_spend_time,
        #         0.0,
        #         False,
        #         False,
        #         0.0,
        #         {},
        #         model_response_token_ids,
        #         model_logprobs,
        #         prompt_ids,
        #     )

        # Environment step (if action is None, just skip)
        env_start_time = asyncio.get_event_loop().time()
        #TODO: weiliu, here we leverage our own reward manager to compute the reward (kernelserver)
        #TODO: To unify the kernelclient into the agent env
        # env_result = await env.step(action)
        # env_result = await self.reward_fn.score_raw_responses_async(
        #     [model_response],
        #     [ground_truth],
        #     [data_source],
        #     [extra_info],
        # )
        reward_kwargs = {"response_length": max_tokens}

        # Use run_in_executor to run sync reward_fn in async context
        reward_fn = self.val_reward_fn if is_validate else self.reward_fn

        loop = asyncio.get_running_loop()
        env_result = await loop.run_in_executor(
            None,
            lambda: reward_fn(
                model_response_token_ids,
                model_response,
                ground_truth,
                entry_point,
                uuid,
                return_full_state=True,
                **reward_kwargs,
            ),
        )
        env_end_time = asyncio.get_event_loop().time()
        env_spend_time = env_end_time - env_start_time

        # Unpack environment result

        #TODO: weiliu: change the env results and change returns of kernel clients
        # tool_response, env_done, truncate, turn_reward, tool_info = env_result
        env_state = env_result["env_state"]

        try:
            tool_response_json = json.dumps(env_state, ensure_ascii=False, indent=2)
        except Exception:
            tool_response_json = str(env_state)
        tool_response = tool_response_json

        make_up_tool_response = True    # makeup tool response secondly
        
        current_prompt_config = None
        for prompt_name, prompt_config in self.per_turn_prompts.items():
            if prompt_name == "tool_response":
                current_prompt_config = prompt_config
                break
        if current_prompt_config is not None:
            history_mode = current_prompt_config["history_mode"]
            update_memory = current_prompt_config["update_memory"]
            skip_env_prob = current_prompt_config["skip_env"]
            current_prompt_template = current_prompt_config["template"]
            
            if current_prompt_template is not None:
                tool_response = current_prompt_template.format(feedback=tool_response)

                print(f"tool_response: {tool_response}")

        # Extract scalar reward from reward_tensor (sum of all token-level rewards)
        reward_tensor = env_result["reward_tensor"]
        if isinstance(reward_tensor, torch.Tensor):
            turn_reward = float(reward_tensor.sum().item())
        else:
            turn_reward = float(reward_tensor)

        tool_info = env_result["reward_extra_info"]
        truncate = False
        # Check if we've reached max_agent_turns
        # current_turn is 0-indexed before this turn completes, so +1 for actual turn count
        current_turn_count = req.get_num_turns() + 1
        # env_done = current_turn_count >= self.max_agent_turns
        env_done = current_turn_count >= actual_max_turns

        print(f"Env Result: {env_result}")


        return (
            model_response,
            tool_response,
            agent_spend_time,
            env_spend_time,
            env_done,
            truncate,
            turn_reward,
            tool_info,
            model_response_token_ids,
            model_logprobs,
            prompt_ids,
            env_state,
        )

    async def _async_agent_loop(
        self,
        messages: list[dict[str, Any]],
        tokens,
        sampling_params: dict,
        is_validate: bool,
        global_step: int = 0,
        extra_info: dict = None,
        ground_truth: str = None,
        entry_point: str = None,
        uuid: str = None,
        actual_max_turns: int = None,
        **kwargs,
    ) -> AgentLoopOutput:
        """Run the full agent loop for multi-turn conversation.

        Args:
            messages: List of conversation messages.
            sampling_params: Sampling parameters for generation.
            timeout: Maximum time in seconds to wait for operations (default: 60s).

        Returns:
            AgentLoopOutput object with the final result.
        """

        # Create new agent and environment instances for this request based on configuration
        agent = create_agent(self.agent_type, self.tokenizer)
        # env = create_environment(self.env_type, self.max_agent_turns, extra_info)
        env = create_environment(self.env_type, actual_max_turns, extra_info)
        await env.reset(extra_info)

        # Initialize agent request
        req = MultiTurnRequest(
            messages=messages,
            response_token_ids=[],
            response_turns=[],
            mask_void_turn=self.mask_void_turn,
            extra_info=extra_info or {},
            ground_truth=ground_truth,
            entry_point=entry_point,
            uuid=uuid,
            # stats will be initialized with default values from MultiTurnStats
        )

        # Track rewards and turns
        turn_rewards = []
        turn_speedups = []
        turn_correctness = []
        done = False
        truncated = False
        finish_reason_type = FinishReasonTypeEnum.STOP

        # Store all collected logprobs
        all_logprobs = []
        # Store actual per-turn prompts and responses
        all_turn_prompts = []
        all_turn_responses = []
        all_turn_lengths = []

        request_id = uuid4().hex
        logging_message = []
        turn_infos = []  # Store turn_info for each turn
        request_start_time = asyncio.get_event_loop().time()

        # Request tracking is handled when deadline is set in _process_single_turn

        # Run the multi-turn interaction loop
        while not done and req.get_num_turns() < actual_max_turns:
            # Process a single turn with timeout
            turn_result = await self._process_single_turn(
                # (Qian): we have benchmarked the deepcopy cost and it is good for now
                # Weiliu: We return env state here to compare speedup among different turns
                deepcopy(req),
                request_id,
                sampling_params,
                agent,
                env,
                is_validate,
                uuid,
                entry_point,
                ground_truth,
                global_step,
                actual_max_turns = actual_max_turns,
            )
            (
                model_response,
                tool_response,
                model_time,
                env_step_time,
                turn_done,
                turn_truncate,
                turn_reward,
                turn_info,
                model_response_token_ids,
                model_logprobs,
                prompt_token_ids,
                turn_env_state,
            ) = turn_result

            # (TODO) Qian: only when there is something wrong we get None response (e.g. async timeout)
            if model_response is None:
                return None

            req.add_message(message=model_response, is_tool_call=False, response_token_ids=model_response_token_ids)
            logging_message.append(
                f"Turn {req.get_num_turns()} | Model time: {model_time:.2f}s | Model Response: {model_response} "
            )
            #TODO: weiliu: here I think we should append user turn with profiler info
            if tool_response is not None:
                req.add_message(message=tool_response, is_tool_call=True)
                logging_message.append(
                    f"Turn {req.get_num_turns()} | Env time: {env_step_time:.2f}s | Tool Response: {tool_response} "
                )

            # Always store turn_info (reward_extra_info), even if tool_response is None
            # When tool_response is None (timeout/error), turn_info contains error details
            turn_infos.append(turn_info if turn_info is not None else {})

            # Track cache hits/misses
            if tool_response is not None and "from_cache" in turn_info:
                if turn_info["from_cache"]:
                    req.stats.cache_hits += 1
                else:
                    req.stats.cache_misses += 1

            # Store per-turn actual prompt/response/logprobs
            all_turn_prompts.append(prompt_token_ids)
            all_turn_responses.append(model_response_token_ids)
            all_logprobs.append(model_logprobs)
            all_turn_lengths.append(len(model_response_token_ids))

            # Check if we got an error response
            # if turn_done and "error" in turn_info:
            #     logging.warning(f"Turn ended with error: {turn_info['error']}")
            #     finish_reason_type = FinishReasonTypeEnum.from_str(turn_info.get("finish_type", "error"))
            #     break

            turn_rewards.append(turn_reward)

            # speedup
            if "speedup" in turn_env_state:
                turn_speedups.append(turn_env_state["speedup"])
            else:
                turn_speedups.append(0.0)

            if "correctness" in turn_env_state and "decoy_kernel" in turn_env_state:
                turn_correctness.append(turn_env_state["correctness"] and not turn_env_state["decoy_kernel"])
            elif "correctness" in turn_env_state:
                logging.warning(f"Decoy kernel is not found in turn env state: {turn_env_state}")
                turn_correctness.append(turn_env_state["correctness"])
            else:
                logging.warning(f"Correctness is not found in turn env state: {turn_env_state}")
                turn_correctness.append(False)


            done = turn_done
            truncated = turn_truncate

            # Determine finish reason
            if done:
                if "finish_type" in turn_info:
                    try:
                        finish_reason_type = FinishReasonTypeEnum.from_str(turn_info["finish_type"])
                    except ValueError:
                        logging.warning(f"Unknown finish_type: {turn_info['finish_type']}")
                        finish_reason_type = FinishReasonTypeEnum.STOP
                elif truncated:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH

        logging_message.append(
            f"Finalizing after {req.get_num_turns()} turns. Finish reason: {finish_reason_type.value}"
        )

        # resolve multi-turn rewards according to turn speedups
        finlized_turn_rewards = self._resolve_multi_turn_rewards(turn_rewards, turn_speedups, turn_correctness)

        # Finalize the agent request
        final_output = agent.finalize(req, finlized_turn_rewards, finish_reason_type.value)

        # Build per-turn outputs using actual prompts/responses collected during the loop
        multi_prompt_ids = all_turn_prompts
        multi_response_ids = all_turn_responses
        multi_logprobs = all_logprobs

        # # Compute scalar reward with reward_fn only for the last turn; others set to 0.0
        # response_ids_for_reward = [multi_response_ids[-1]]
        # Concatenate all response ids for reward
        response_ids_for_reward = [[item for sublist in multi_response_ids for item in sublist]]
        response_strs = self.tokenizer.batch_decode(response_ids_for_reward, skip_special_tokens=True)
        ground_truths = [extra_info['ground_truth']]
        data_sources = [extra_info['data_source']]
        extra_infos = [extra_info['extra_info']]
        
        if self.use_async_reward:
            # We've already computed the rewards in _process_single_turn

            # weiliu: shape: [turn_num, 1]
            multi_rewards = finlized_turn_rewards
            multi_reward_extra_info = turn_infos

            # if is_validate:
            #     return_multi_output = MultiTurnOutput(
            #         multi_prompt_ids=[multi_prompt_ids[-1]],
            #         multi_response_ids=[multi_response_ids[-1]],
            #         multi_logprobs=[multi_logprobs[-1]],
            #         multi_loss_mask=[final_output["loss_mask"][-1]],  # Last turn only
            #         multi_rewards=[reward_result["scores"][0]],
            #         stats=final_output["stats"],
            #         request_id=request_id,
            #         messages=final_output["messages"],
            #         multi_reward_extra_info=[reward_result["extra_info"]],
            #     )
            # else:
                # fill only on last turn row per sample, others 0.0
            # multi_rewards = []
            # for turn_idx, turn_length in enumerate(all_turn_lengths):
            #     single_rewards = [0.0] * (turn_length - 1) + [finlized_turn_rewards[turn_idx]]
            #     multi_rewards.extend(single_rewards)
                
            # multi_reward_extra_info = []
            # for turn_idx, turn_length in enumerate(all_turn_lengths):
            #     single_extra_info = [{}] * (turn_length - 1) + [turn_infos[turn_idx]]
            #     multi_reward_extra_info.extend(single_extra_info)

            return_multi_output = MultiTurnOutput(
                multi_prompt_ids=multi_prompt_ids,
                multi_response_ids=multi_response_ids,
                multi_logprobs=multi_logprobs,
                multi_loss_mask=final_output["loss_mask"],
                multi_rewards=multi_rewards,
                stats=final_output["stats"],
                request_id=request_id,
                messages=final_output["messages"],
                multi_reward_extra_info=multi_reward_extra_info,
            )
        else:
            raise NotImplementedError("We only support async reward for multi-turn rewards")
            # if is_validate:
            #     return_multi_output = MultiTurnOutput(
            #         multi_prompt_ids=[multi_prompt_ids[-1]],
            #         multi_response_ids=[multi_response_ids[-1]],
            #         multi_logprobs=[multi_logprobs[-1]],
            #         multi_loss_mask=[final_output["loss_mask"][-1]],  # Last turn only
            #         stats=final_output["stats"],
            #         request_id=request_id,
            #         messages=final_output["messages"],
            #     )
            # else:
                # fill only on last turn row per sample, others 0.0
                # return_multi_output = MultiTurnOutput(
                #     multi_prompt_ids=multi_prompt_ids,
                #     multi_response_ids=multi_response_ids,
                #     multi_logprobs=multi_logprobs,
                #     multi_loss_mask=final_output["loss_mask"],
                #     stats=final_output["stats"],
                #     request_id=request_id,
                # )

        # Calculate total request time
        total_request_time = asyncio.get_event_loop().time() - request_start_time

        # Check with the shared tracker if this is the slowest request
        is_slowest = False
        if self.slowest_tracker is not None:
            try:
                update_ref = self.slowest_tracker.update_slowest_time.remote(total_request_time, global_step)
                is_slowest = await asyncio.wait_for(asyncio.shield(update_ref), timeout=self.slowest_tracker_timeout)
            except asyncio.TimeoutError:
                is_slowest = False
                logging.debug(
                    "Slowest tracker update timed out after %.2fs, skipping slowest-request logging.",
                    self.slowest_tracker_timeout,
                )
            except Exception as e:
                logging.debug(f"Failed to update slowest tracker: {e}")

        # Different logging probability for validation and training
        log_probability = 0.02 if is_validate else 0.005  # 1% for both validation and training
        should_log = is_slowest or random.random() < log_probability

        # Log if it's the slowest or randomly sampled
        if should_log:
            self.log_multiturn_messages(
                step=global_step,
                request_id=request_id,
                logging_messages=logging_message,
                turn_rewards=turn_rewards,
                turn_infos=turn_infos,
                stats=final_output["stats"],
                finish_reason=finish_reason_type.value,
                multi_turn_output=return_multi_output,
                is_slowest=is_slowest,
            )

        # Remove request from active set and deadline tracking when done
        self.clear_request_tracking(request_id)

        # Create agent loop output - using the collected logprobs
        return return_multi_output

    def _compute_adaptive_timeout(self, max_tokens: int, is_validate: bool) -> float:
        """
        Based on token generation rates, compute an adaptive timeout for LLM generation.
        """
        history = self.token_rate_history_val if is_validate else self.token_rate_history_train
        if len(history) <= 1000:
            # If no history, use default timeout
            return None

        # Use a conservative (75th percentile) estimate of recent generation speed
        rates = sorted(history)
        safe_rate = float(np.quantile(rates, 0.75))
        if safe_rate <= 0:
            return None

        # Predict time based on token rate
        predicted_time = max_tokens / safe_rate

        # Apply buffer
        timeout = predicted_time * self.timeout_buffer
        return timeout

    def _record_generation_stats(self, output_tokens: int, generation_time: float, is_validate: bool):
        """Record the token generation rate for adaptive timeout computation."""
        if generation_time > 0 and output_tokens > 0:
            rate = output_tokens / generation_time
            history = self.token_rate_history_val if is_validate else self.token_rate_history_train
            history.append(rate)

    async def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        global_step = prompts.meta_info.get("global_step", 0)  # Extract global_step for logfire
        tgt_device = prompts.batch["input_ids"].device

        # Support multiple sampling for both training and validation
        if is_validate:
            # For validation, use a separate config for number of samples
            val_n_samples = self.config.rollout.get("val_n_samples", 1)
            prompts = prompts.repeat(repeat_times=val_n_samples, interleave=True)
        else:
            prompts = prompts.repeat(repeat_times=self.config.rollout.n, interleave=True)

        config = self.config.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=1,  # Ensure logprobs are collected
        )

        # override sampling params for validation
        if prompts.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature
            # actual_max_turns = config.val_kwargs.get('max_user_turns', self.max_agent_turns)
            # actual_max_turns = prompts.meta_info.get("actual_max_turns", self.max_agent_turns)
        

        stop_token_ids = config.stop_token_ids
        if stop_token_ids is not None:
            sampling_params["stop_token_ids"] = list(stop_token_ids)

        actual_max_turns = self.max_agent_turns
        if is_validate:
            val_stop_token_ids = config.val_kwargs.stop_token_ids
            if val_stop_token_ids is not None:
                sampling_params["stop_token_ids"] = list(val_stop_token_ids)
            actual_max_turns = config.val_kwargs.get('max_user_turns', self.max_agent_turns)

        print(f"Actual max turns: {actual_max_turns}")
        print("stop_token_ids: ", sampling_params["stop_token_ids"])

        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        tokens_ids = prompts.non_tensor_batch["raw_prompt_ids"]

        ground_truths = prompts.non_tensor_batch.get("ground_truth")
        entry_points = prompts.non_tensor_batch.get("entry_point")
        uuids = prompts.non_tensor_batch.get("uuid")

        # CRITICAL: Ensure uuids are strings to pass Pydantic validation
        # Convert any integer/numeric uuids to strings
        if uuids is not None:
            if isinstance(uuids, np.ndarray):
                uuids = [str(u) for u in uuids]
            elif isinstance(uuids, list):
                uuids = [str(u) for u in uuids]
            elif not isinstance(uuids, (list, tuple)):
                # Handle scalar case
                uuids = [str(uuids)]

        # Handle None values by extracting from reward_model or extra_info
        batch_size = len(raw_prompts)
        if ground_truths is None:
            ground_truths = []
            for i in range(batch_size):
                if "reward_model" in prompts.non_tensor_batch:
                    rm = prompts.non_tensor_batch["reward_model"][i]
                    if isinstance(rm, str):
                        rm = json.loads(rm)
                    ground_truths.append(rm.get("ground_truth", ""))
                else:
                    ground_truths.append("")

        if entry_points is None:
            entry_points = []
            extra_info_list = prompts.non_tensor_batch.get("extra_info", [{}] * batch_size)
            for i in range(batch_size):
                if extra_info_list is not None and i < len(extra_info_list) and extra_info_list[i]:
                    # entry_points.append(extra_info_list[i].get("entry_point", infer_entry_point(ground_truths[i])))
                    entry_points.append(extra_info_list[i].get("entry_point", "Model"))
                else:
                    # entry_points.append(infer_entry_point(ground_truths[i]))
                    entry_points.append("Model")

        if uuids is None:
            uuids = []
            extra_info_list = prompts.non_tensor_batch.get("extra_info", [{}] * batch_size)
            for i in range(batch_size):
                if extra_info_list is not None and i < len(extra_info_list) and extra_info_list[i]:
                    uuids.append(extra_info_list[i].get("uuid") or extra_info_list[i].get("problem_id") or uuid4().hex)
                else:
                    uuids.append(uuid4().hex)

        # Extract uid from non_tensor_batch
        uids = prompts.non_tensor_batch["uid"]

        tasks = []
        for i, (messages, tokens, ground_truth, entry_point, uuid) in enumerate(zip(raw_prompts, tokens_ids, ground_truths, entry_points, uuids)):
            # Extract prompt-dependent extra_info
            if self.env_type == "MathSandboxEnv":
                extra_info = {
                    'ground_truth': prompts[i].non_tensor_batch['reward_model']['ground_truth'],
                    'data_source': prompts[i].non_tensor_batch['data_source'],
                    'extra_info': prompts[i].non_tensor_batch.get("extra_info", None),
                }
            elif self.env_type == "CodeSandboxEnv":
                if isinstance(prompts[i].non_tensor_batch["reward_model"], str):
                    ground_truth = json.loads(prompts[i].non_tensor_batch["reward_model"])["ground_truth"]
                else:
                    ground_truth = prompts[i].non_tensor_batch["reward_model"]["ground_truth"]
                extra_info = {
                    'ground_truth': ground_truth,
                    'data_source': prompts[i].non_tensor_batch['data_source'],
                    'extra_info': prompts[i].non_tensor_batch.get("extra_info", None),
                }
            elif self.env_type == "FileSearchEnv":
                # provide root dir for file search
                extra_info = {"root_dir": self.config.env.root_dir}
            elif self.env_type == "SWEFileLocationEnv":
                # provide root dir for file search
                extra_info = {
                    "root_dir": self.config.env.root_dir,  # In SWEFileLocationEnv, root_dir is the where we place all github repos
                    "repo": prompts[i].non_tensor_batch['reward_model']['repo'],
                    "base_commit": prompts[i].non_tensor_batch['reward_model']['base_commit'],
                }
            elif self.env_type == "KernelEnv":
                # For Kernel training, extract ground_truth from reward_model
                if "reward_model" in prompts[i].non_tensor_batch:
                    rm = prompts[i].non_tensor_batch["reward_model"]
                    if isinstance(rm, str):
                        rm = json.loads(rm)
                    gt = rm.get("ground_truth", ground_truth)
                else:
                    gt = ground_truth
                extra_info = {
                    'ground_truth': gt,
                    'data_source': prompts[i].non_tensor_batch.get('data_source', 'kernel'),
                    'extra_info': prompts[i].non_tensor_batch.get("extra_info", {}),
                }
            else:
                raise ValueError(f"Unsupported environment type: {self.env_type}")

            if not isinstance(messages, list):
                messages = messages.tolist()
            tasks.append(
                asyncio.create_task(
                    self._async_agent_loop(
                        messages, tokens, sampling_params, is_validate, global_step, extra_info, ground_truth, entry_point, uuid, actual_max_turns, **kwargs
                    )
                )
            )

        outputs = await asyncio.gather(*tasks)
        # filter out None of outputs
        if None in outputs:
            # for training, we remove the whole group if one of the outputs is None
            if not is_validate:
                # if one of the outputs is None, we need to filter out the corresponding prompts and uids
                removed_uids = {uids[i] for i, output in enumerate(outputs) if output is None}
                removed_indices = set()
                # remove from all responses if the uids match
                for i in range(len(outputs)):
                    if uids[i] in removed_uids:
                        removed_indices.add(i)
                keep_indices = [i for i in range(len(outputs)) if i not in removed_indices]

                if self.logfire_logger:
                    self.logfire_logger.warning(
                        f"Some training requests returned None output, possibly due to timeouts or errors. Keeped responses ({len(keep_indices)} / {len(outputs)}) ",
                        global_step=global_step,
                    )

                filtered_outputs = [outputs[i] for i in keep_indices]
                filtered_uids = [uids[i] for i in keep_indices]
                assert None not in filtered_outputs, "Filtered outputs should not contain None"

                uids = np.array(filtered_uids)
                outputs = filtered_outputs
            # for validation, we just keep the valid outputs, and add placeholder responses for None outputs to avoid
            # over-estimating the validation performance
            else:
                placeholder_number = 0
                for i in range(len(outputs)):
                    if outputs[i] is None:
                        # create a placeholder MultiTurnOutput with empty content
                        placeholder_output = MultiTurnOutput(
                            multi_prompt_ids=[1],
                            multi_response_ids=[1],
                            multi_logprobs=[1.0],
                            multi_loss_mask=[1],
                            stats=MultiTurnStats(),
                            request_id=f"placeholder_{i}",
                        )
                        outputs[i] = placeholder_output
                        placeholder_number += 1
                if self.logfire_logger:
                    self.logfire_logger.warning(
                        f"Some validation requests returned None output, possibly due to timeouts or errors. Replaced with {placeholder_number} placeholder outputs.",
                        global_step=global_step,
                    )

        return self._postprocess(outputs, is_validate, uids, actual_max_turns = actual_max_turns)

    def _postprocess(self, inputs: list[MultiTurnOutput], is_validate: bool, uids: np.ndarray, actual_max_turns: int = None) -> DataProto:
        """
        This function pads multi-turn data to have uniform shapes across batches, specifically:
        - prompt_ids as [batch_size * rollout_n * max_multi_turn, max_prompt_length]
        - response_ids as [batch_size * rollout_n * max_multi_turn, max_response_length]
        - And similar shapes for other tensors
        """
        # Skip processing if no inputs
        if not inputs or len(inputs) == 0:
            return None

        # Get max lengths from config
        max_prompt_length = self.config.rollout.prompt_length
        max_response_length = self.config.rollout.response_length

        # Calculate max valid turns across the batch
        # (TODO) Qian: originally we want to use a dynamic multi_turn based on the batch, but it seems to cause some issues in training due to
        # the data distribution across different ray workers. So we just use a fixed max_agent_turns for now.
        # if is_validate:
        #     max_multi_turn = 1
        # else:
        max_multi_turn = actual_max_turns
        # Flatten all turns into single collections
        all_prompts = []
        all_responses = []
        all_response_masks = []
        all_logprobs = []
        all_loss_mask = []  # Per-turn loss mask from finalization

        if self.use_async_reward:
            all_rewards = []
            all_reward_extra_info = []

        # Add indices for tracking
        all_sample_indices = []
        # (TODO) Qian: star from 1 for turn indices
        all_turn_indices = []

        # Multi-turn statistics
        all_num_turns = []
        all_contain_void_turn = []
        all_finish_reasons = []

        # Expanded uids for multi-turn
        all_uids = []

        # Complete conversation messages for each sample (only stored once per sample)
        all_messages = []

        # Get padding token
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Process each input sample and flatten
        # The all_prompts has the following shape: batch_size * rollout_n * max_multi_turn, max_prompt_length
        # [response_1_turn_1, response_1_turn_2, ..., response_1_turn_n, response_2_turn_1, ..., response_m_turn_n]

        # (TODO) Qian: here we have a special handling for samples with fewer turns than max_multi_turn, we add fake padding turns first, then the actual turns
        # This is to ensure that we can always get the last turn by indexing with -1
        for sample_idx, input in enumerate[MultiTurnOutput](inputs):
            num_turns = len(input.multi_prompt_ids)

            # Add fake padding turns to ensure consistent shape for each sample
            # For multi-turn, it is [LEFT] padding + actual prompt
            #TODO weiliu: I think actually we do not need this but we will definitely meet 
            # the issue the length of model is not enough for max_multi_turn, 
            # so we need to add fake padding turns
            for _ in range(num_turns, max_multi_turn):
                # Create fake prompt with padding tokens
                fake_prompt = [pad_token_id] * 1  # Minimal length, will be padded later
                # Create fake response with padding tokens
                fake_response = [pad_token_id] * 1  # Minimal length, will be padded later
                # Create fake response mask (all zeros)
                fake_mask = [0] * 1
                # Create fake logprobs (all -1.0)
                fake_logprobs = [-1.0] * 1
                # Create fake turn idx
                fake_turn_idx = -1

                all_prompts.append(fake_prompt)
                all_responses.append(fake_response)
                all_response_masks.append(fake_mask)
                all_logprobs.append(fake_logprobs)
                all_sample_indices.append(sample_idx)
                all_turn_indices.append(fake_turn_idx)
                all_loss_mask.append(0)  # Padding turns should not contribute to loss
                if self.use_async_reward:
                    all_rewards.append(0.0)  # Padding turns should not contribute to reward
                    all_reward_extra_info.append({})  # Empty dict for padding turns

                # Add stats for padding turns
                all_num_turns.append(input.stats.num_turns)
                all_contain_void_turn.append(input.stats.contain_void_turn)
                all_finish_reasons.append(input.stats.finish_reason)

                # Add uid for padding turn
                if uids is not None:
                    all_uids.append(uids[sample_idx])
                else:
                    all_uids.append(None)

                # Add None for messages in padding turns
                all_messages.append(None)

            # Process each turn for this input
            for turn_idx in range(num_turns):
                all_prompts.append(input.multi_prompt_ids[turn_idx])
                all_responses.append(input.multi_response_ids[turn_idx])
                all_logprobs.append(input.multi_logprobs[turn_idx])
                all_loss_mask.append(input.multi_loss_mask[turn_idx])  # Use per-turn mask from finalization
                # if self.use_async_reward:
                assert self.use_async_reward, "We only support async reward for multi-turn rewards"
                all_rewards.append(input.multi_rewards[turn_idx])  # Add rewards for actual turns
                all_reward_extra_info.append(input.multi_reward_extra_info[turn_idx])

                # Sample and turn indices
                all_sample_indices.append(sample_idx)
                all_turn_indices.append(turn_idx + 1)

                # Add stats for actual turns
                all_num_turns.append(input.stats.num_turns)
                all_contain_void_turn.append(input.stats.contain_void_turn)
                all_finish_reasons.append(input.stats.finish_reason)

                # Add uid for actual turn
                if uids is not None:
                    all_uids.append(uids[sample_idx])
                else:
                    all_uids.append(None)

                # Add messages only for the first actual turn (to avoid duplication)
                if turn_idx == 0:
                    all_messages.append(input.messages)
                else:
                    all_messages.append(None)

        # Manually truncate prompts that exceed max_prompt_length
        truncated_prompts = []
        num_truncated = 0
        for i, prompt_ids in enumerate(all_prompts):
            if len(prompt_ids) > max_prompt_length:
                num_truncated += 1
                logging.warning(
                    f"Prompt at index {i} exceeds max length ({len(prompt_ids)} > {max_prompt_length}). "
                    f"Truncating to {max_prompt_length} tokens."
                )
                # Truncate from the left (beginning) since padding_side is left
                truncated_prompts.append(prompt_ids[-max_prompt_length:])
            else:
                truncated_prompts.append(prompt_ids)

        # Pad prompts (left padding)
        self.tokenizer.padding_side = "left"
        prompt_outputs = self.tokenizer.pad(
            [{"input_ids": prompt_ids} for prompt_ids in truncated_prompts],
            padding="max_length",
            max_length=max_prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids = prompt_outputs["input_ids"]
        prompt_mask = prompt_outputs["attention_mask"]

        # Pad responses (right padding)
        self.tokenizer.padding_side = "right"
        response_outputs = self.tokenizer.pad(
            [{"input_ids": response_ids} for response_ids in all_responses],
            padding="max_length",
            max_length=max_response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids = response_outputs["input_ids"]
        response_mask = response_outputs["attention_mask"]

        # Pad logprobs
        padded_logprobs = []
        for logprob in all_logprobs:
            padded = logprob + [-1.0] * (max_response_length - len(logprob))
            padded_logprobs.append(padded[:max_response_length])
        rollout_log_probs = torch.tensor(padded_logprobs, dtype=torch.float32, device=response_ids.device)

        # Create combined tensors
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        # Create loss mask from the per-turn loss mask values
        # Shape: [batch_size * rollout_n * max_multi_turn]
        loss_mask = torch.tensor(all_loss_mask, dtype=torch.long, device=response_ids.device)

        # Create metadata tensors
        turn_indices = torch.tensor(all_turn_indices, dtype=torch.long, device=response_ids.device)
        assert max(all_turn_indices) <= max_multi_turn, "Turn indices are not equal to max_multi_turn"
        sample_indices = torch.tensor(all_sample_indices, dtype=torch.long, device=response_ids.device)

        batch_dict = {
            "prompts": prompt_ids,
            "responses": response_ids,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "rollout_log_probs": rollout_log_probs,
            "turn_indices": turn_indices,
            "sample_indices": sample_indices,   # note: do not use this for any aggregation metrics. Use uid instead.
            # Loss mask for filtered turns
            "loss_mask": loss_mask,
        }

        if self.use_async_reward:
            # Build reward tensor from scalar rewards computed in _async_agent_loop via reward_fn
            reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
            reward_extra_info_list = [{} for _ in range(len(reward_tensor))]
            valid_response_length = attention_mask[:, prompt_ids.shape[1] :].sum(dim=-1)

            # Determine last actual turn row index for each sample (ignore padding turns)
            # last_row_of_sample = {}
            valid_row_indices = {}
            s_t_format = "{sampe_idx}_{turn_idx}"
            for row_idx, (s_idx, t_idx) in enumerate(zip(all_sample_indices, all_turn_indices)):
                if t_idx != -1:
                    valid_row_indices[s_t_format.format(sampe_idx=s_idx, turn_idx=t_idx)] = row_idx
            # Assign reward only for the last turn of each sample at its last valid token
            for row_idx_format in valid_row_indices:
                row_idx = valid_row_indices[row_idx_format]
                reward_tensor[row_idx, valid_response_length[row_idx].item() - 1] = float(all_rewards[row_idx])
                assert all_reward_extra_info[row_idx] is not None and isinstance(all_reward_extra_info[row_idx], dict)
                reward_extra_info_list[row_idx] = all_reward_extra_info[row_idx]

            batch_dict["token_level_scores"] = reward_tensor
            reward_extra_info_array = np.array(reward_extra_info_list, dtype=object)

        # Create the batch
        batch = TensorDict(
            batch_dict,
            batch_size=len(input_ids),
        )

        # Multi-turn statistics in non-tensor batch
        non_tensor_batch = {
            "num_turns": np.array(all_num_turns, dtype=np.int32),
            "contain_void_turn": np.array(all_contain_void_turn, dtype=np.int32),
            "finish_reasons": np.array(all_finish_reasons, dtype=object),
            "multiturn_messages": np.array(all_messages, dtype=object),
        }

        # Add expanded uid to non_tensor_batch
        non_tensor_batch["uid"] = np.array(all_uids, dtype=object)

        if self.use_async_reward:
            non_tensor_batch["reward_extra_info"] = reward_extra_info_array

        print(f"reward_extra_info_array: {reward_extra_info_array}")

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def get_expired_requests(self):
        """Get list of request IDs that have exceeded their deadline."""
        current_time = asyncio.get_event_loop().time()
        expired = []
        for request_id, info in list(self.request_deadlines.items()):
            # Handle both old format (just deadline) and new format (dict with deadline and global_step)
            deadline = info["deadline"]
            global_step = info["global_step"]

            if current_time > deadline:
                expired.append((request_id, global_step))
        return expired

    async def _deadline_watchdog(self):
        while True:
            try:
                expired = self.get_expired_requests()
                if expired:
                    # Extract just the request IDs for aborting
                    request_ids = [rid for rid, _ in expired]

                    # Abort all expired requests
                    await asyncio.gather(*[self.engine.abort(rid) for rid in request_ids], return_exceptions=True)

                    # Clear tracking and log for all expired requests
                    for rid, step in expired:
                        if self.logfire_logger:
                            self.logfire_logger.warning(
                                f"Request timeout (watchdog) | request_id: {rid} | step {step}",
                                request_id=rid,
                                global_step=step,
                            )
                        self.clear_request_tracking(rid)
            except Exception as e:
                logging.warning(f"watchdog error: {e}")
            await asyncio.sleep(1.0)

    def clear_request_tracking(self, request_id):
        """Clear tracking for a completed or aborted request."""
        self.request_deadlines.pop(request_id, None)
