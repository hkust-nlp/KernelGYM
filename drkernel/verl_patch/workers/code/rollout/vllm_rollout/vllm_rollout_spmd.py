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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import os
import pickle
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl_patch.utils.random import save_random_states, set_random_states
from verl_patch.workers.code.rollout.vllm_rollout.vllm_config_helper import (
    get_vllm_config_kwargs,
    verify_importance_sampling_safety,
)

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert (
            tensor_parallel_size <= torch.distributed.get_world_size()
        ), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os

            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp
            )

        assert model_hf_config.max_position_embeddings >= config.prompt_length + max(
            config.response_length, config.val_kwargs.response_length
        ), "model context length should be greater than total sequence length"

        # if you specify the max_model_len in validation, we use the max one to initialize the max_model_len
        max_model_len = (
            self.config.max_model_len
            if self.config.max_model_len
            else config.prompt_length + max(config.response_length, config.val_kwargs.response_length)
        )
        print(f"config.val_kwargs.response_length {config.val_kwargs.response_length}")
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                'Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill'
            )

        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # Verify configuration is safe for importance sampling
        verify_importance_sampling_safety(config)

        # Get additional kwargs for correct logprobs handling
        vllm_kwargs = get_vllm_config_kwargs(config)

        # Initialize with all parameters
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **vllm_kwargs,  # Add the configuration-specific kwargs
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop('raw_prompt_ids'), non_tensor_batch.pop('multi_modal_data')
            ):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [
                {'prompt_token_ids': raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)

        # Initialize kwargs based on sampling mode
        kwargs = {}
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'max_tokens': self.config.val_kwargs.response_length,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        if is_validate:
            rng_states = save_random_states()
            set_random_states(rng_states=None)
        response_length = self.config.val_kwargs.response_length if is_validate else self.config.response_length

        # Always collect logprobs for importance sampling correction
        # Request top 1 logprob to get the logprob of the sampled token
        kwargs['logprobs'] = 1 if self.config.calculate_log_probs else None

        # Check if a custom n value is specified in meta_info (for oversampling)
        if 'n' in prompts.meta_info and do_sample and not is_validate:
            kwargs['n'] = prompts.meta_info['n']

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            top_log_probs = []  # Store maximum log probabilities for Gate 2
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    # Collect logprobs for each sampled token and max logprob
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        curr_top_log_prob = []
                        for i, logprob_dict in enumerate(output.outputs[sample_id].logprobs):
                            # logprob_dict is Dict[int, Logprob] where int is token_id
                            # With logprobs=1, dict contains top-1 token + sampled token (if different)

                            # Get the sampled token's logprob
                            sampled_token_id = response_ids[i]
                            sampled_logprob = logprob_dict[sampled_token_id].logprob
                            curr_log_prob.append(sampled_logprob)

                            # Get the maximum logprob (top token with rank=1)
                            # When logprobs=1, the top token is the one with highest logprob
                            if logprob_dict:
                                # Find the token with rank=1 or highest logprob
                                top_logprob = max(logprob_dict.values(), key=lambda x: x.logprob).logprob
                                curr_top_log_prob.append(top_logprob)
                            else:
                                # Shouldn't happen with logprobs=1, but use sampled as fallback
                                curr_top_log_prob.append(sampled_logprob)
                        rollout_log_probs.append(curr_log_prob)
                        top_log_probs.append(curr_top_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=response_length).to(idx.device)

            if self.config.calculate_log_probs:
                # Pad and convert logprobs to float32
                rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=response_length).to(
                    idx.device
                )
                rollout_log_probs = rollout_log_probs.to(torch.float32)

                # Pad and convert top_log_probs to float32 (for Gate 2)
                top_log_probs = pad_2d_list_to_length(top_log_probs, -1, max_length=response_length).to(idx.device)
                top_log_probs = top_log_probs.to(torch.float32)

            # debug prints
            # try:
            #     print(
            #         f"[vllm_rollout] responses_shape={tuple(response.shape)}, rollout_log_probs_shape={tuple(rollout_log_probs.shape)}, top_log_probs_shape={tuple(top_log_probs.shape)}"
            #     )
            #     # print a short preview of first row
            #     print(
            #         f"[vllm_rollout] rollout_log_probs[0][:5]={rollout_log_probs[0][:5].tolist() if rollout_log_probs.size(0) > 0 else []}"
            #     )
            #     print(
            #         f"[vllm_rollout] top_log_probs[0][:5]={top_log_probs[0][:5].tolist() if top_log_probs.size(0) > 0 else []}"
            #     )
            # except Exception as _e:
            #     print(f"[vllm_rollout] debug_print_error: {_e}")

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(
                        non_tensor_batch['multi_modal_inputs'], self.sampling_params.n
                    )
                if 'uid' in non_tensor_batch.keys():
                    non_tensor_batch['uid'] = _repeat_interleave(non_tensor_batch['uid'], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            },
            batch_size=batch_size,
        )

        if is_validate:
            set_random_states(rng_states=rng_states)

        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            # Always save rollout log probs aligned with responses
            batch['rollout_log_probs'] = rollout_log_probs
            # Save maximum log probs for Gate 2 filtering
            batch['top_log_probs'] = top_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
