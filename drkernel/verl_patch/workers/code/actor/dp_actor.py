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
Single Process Actor
"""

import logging
import os
from typing import Iterable, Tuple

import torch
import verl.utils.torch_functional as verl_F
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl import DataProto
from verl.utils.device import (
    get_device_id,
    get_device_name,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (
    get_reverse_idx,
    prepare_dynamic_batch,
    rearrange_micro_batches,
    restore_dynamic_batch,
)
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor

from verl_patch.trainer.code.ppo import core_algos
from verl_patch.utils.metric import PolicyOutput
from verl_patch.utils.torch_functional import compute_sum_pi_squared_from_logits
from verl_patch.workers.config.actor import ActorConfig

__all__ = ['CodeDataParallelPPOActor']

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CodeDataParallelPPOActor(BasePPOActor):

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # Sum of squared probabilities computation (for optimal baseline variance reduction)
        # Only initialize if compute_sum_pi_squared config is enabled
        if self.config.get('compute_sum_pi_squared', False):
            self.compute_sum_pi_squared_from_logits = (
                torch.compile(compute_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get('use_torch_compile', True)
                else compute_sum_pi_squared_from_logits
            )
            if torch.distributed.get_rank() == 0:
                print(
                    f"{role} Sum_pi_squared computation enabled: {'compiled' if self.config.get('use_torch_compile', True) else 'uncompiled'}"
                )

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, compute_sum_pi_squared=False
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            micro_batch: Input batch data
            temperature: Temperature for logits
            calculate_entropy: Whether to compute entropy
            compute_sum_pi_squared: Whether to compute sum of squared probabilities (for optimal baseline)

        Returns:
            entropy: (bs, response_len) or None if calculate_entropy=False
            log_probs: (bs, response_len) - always computed
            sum_pi_squared: (bs, response_len) or None if compute_sum_pi_squared=False
        """
        response_length = micro_batch["responses"].size(-1)
        sum_pi_squared = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import (
                        process_multi_modal_inputs_for_minicpmo,
                    )

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    if calculate_entropy:
                        entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    # Note: Fused kernels don't provide logits, so sum_pi_squared cannot be computed
                    # sum_pi_squared stays None (initialized at line 127)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # Compute sum_pi_squared if requested (for optimal baseline)
                    if compute_sum_pi_squared:
                        if not self.config.get('sum_pi_squared_checkpointing', False):
                            sum_pi_squared_rmpad = self.compute_sum_pi_squared_from_logits(logits_rmpad)
                        else:
                            sum_pi_squared_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_sum_pi_squared_from_logits, logits_rmpad
                            )
                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if compute_sum_pi_squared and not self.use_fused_kernels:
                        # sum_pi_squared_rmpad only exists if use_fused_kernels=False
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if compute_sum_pi_squared and not self.use_fused_kernels:
                    # sum_pi_squared_rmpad only exists if use_fused_kernels=False
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if compute_sum_pi_squared and not self.use_fused_kernels:
                    # full_sum_pi_squared only exists if use_fused_kernels=False
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    if calculate_entropy:
                        entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    # Note: Fused kernels don't provide logits, so sum_pi_squared cannot be computed
                    # sum_pi_squared stays None (initialized at line 127)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                    # Compute sum_pi_squared if requested (for optimal baseline)
                    if compute_sum_pi_squared:
                        if not self.config.get('sum_pi_squared_checkpointing', False):
                            sum_pi_squared = self.compute_sum_pi_squared_from_logits(logits)
                        else:
                            sum_pi_squared = torch.utils.checkpoint.checkpoint(
                                self.compute_sum_pi_squared_from_logits, logits
                            )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs, sum_pi_squared

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self, data: DataProto, calculate_entropy=False, compute_sum_pi_squared=False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

            calculate_entropy (bool): whether to compute entropy

            compute_sum_pi_squared (bool): whether to compute sum of squared probabilities

        Returns:
            tuple: (log_probs, entropys, sum_pi_squared) where entropys and sum_pi_squared can be None
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = [] if compute_sum_pi_squared else None

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                result = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    compute_sum_pi_squared=compute_sum_pi_squared,
                )

            log_probs_lst.append(result[1])
            if calculate_entropy:
                entropy_lst.append(result[0])
            if compute_sum_pi_squared:
                sum_pi_squared_lst.append(result[2])

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0) if compute_sum_pi_squared else None

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if entropys is not None:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if sum_pi_squared is not None:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        return log_probs, entropys, sum_pi_squared

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        # Rollout correction mode detection
        # bypass_mode: Uses rollout_log_prob as old_log_prob (set by trainer)
        # use_pure_rollout_correction: Selects loss function (PPO vs pure IS)
        bypass_mode = data.meta_info.get('bypass_old_logprob_for_rollout', False)
        use_pure_rollout_correction = data.meta_info.get('use_pure_rollout_correction', False)
        if use_pure_rollout_correction:
            # Get rollout correction config
            max_turns = data.meta_info.get('max_turns', 1)
            rollout_correction_kwargs = data.meta_info.get('rollout_correction_kwargs', {})

        # Modes:
        # 1. Legacy (bypass=False): old_log_prob computed by trainer, standard PPO
        # 2. PPO_IS (bypass=True, pure=False): old_log_prob=rollout_log_prob, PPO clips against rollout
        # 3. Pure IS (bypass=True, pure=True): No PPO clipping, pure policy gradient with IS correction

        # Include rollout_log_probs alongside old_log_probs since both are log probabilities from previous policies
        select_keys = [
            'responses',
            'response_mask',
            'input_ids',
            'attention_mask',
            'position_ids',
            'old_log_probs',
            'advantages',
        ]

        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')

        # In bypass mode, include rollout_log_probs for IS/RS computation
        # In pure rollout correction mode, we MUST have rollout_log_probs
        if (bypass_mode or use_pure_rollout_correction) and 'rollout_log_probs' in data.batch.keys():
            select_keys.append('rollout_log_probs')

        # Include pre-computed IS weights if present in batch (legacy mode)
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if 'rollout_is_weights' in data.batch.keys():
            select_keys.append('rollout_is_weights')

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    # Support all hardwares
                    if isinstance(micro_batch, DataProto):
                        data = {**micro_batch.batch.to(torch.cuda.current_device()), **micro_batch.non_tensor_batch}
                    else:
                        data = micro_batch.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = data['response_mask']
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    # Extract pre-computed rollout importance sampling weights (if present)
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    # When rollout_is=False (metrics-only mode), weights are not added → None here → no weight application
                    rollout_is_weights = data.get('rollout_is_weights', None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    clip_ratio_high = self.config.clip_ratio_high
                    clip_ratio_low = self.config.clip_ratio_low
                    entropy_coeff = self.config.entropy_coeff
                    clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                    entropy_clip_rate = self.config.get('entropy_clip_rate', 0.0)
                    use_gspo = self.config.get('use_gspo', False)
                    loss_agg_mode = self.config.get('loss_agg_mode', 'seq-mean-token-sum')
                    loss_scale_factor = self.config.get('loss_scale_factor', 1.0)
                    extreme_risk_prob_threshold = self.config.get('extreme_risk_prob_threshold', None)

                    # all return: (bsz, response_length)
                    entropy, log_prob, _ = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature, calculate_entropy=True
                    )

                    # Choose loss computation based on mode
                    if use_pure_rollout_correction:
                        # MODE 3: Pure rollout correction (no PPO clipping)
                        # Loss: L = -E[w * A] where w = exp(log_prob - rollout_log_prob).clamp(max=threshold)
                        # Computes IS/RS on-the-fly in this function
                        # Requires rollout_log_probs in batch (set by trainer in bypass mode)
                        rollout_log_prob = data.get('rollout_log_probs')
                        if rollout_log_prob is None:
                            raise ValueError(
                                "use_pure_rollout_correction=True requires rollout_log_probs in batch. "
                                "Ensure bypass_old_logprob_for_rollout=True in trainer."
                            )

                        policy_output = core_algos.compute_policy_loss_with_rollout_correction(
                            rollout_log_prob=rollout_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            loss_scale_factor=loss_scale_factor,
                            rollout_is=rollout_correction_kwargs['rollout_is'],
                            rollout_is_threshold=rollout_correction_kwargs['rollout_is_kwargs'].get('upper', None),
                            rollout_rs=rollout_correction_kwargs['rollout_rs'],
                            rollout_rs_threshold=rollout_correction_kwargs['rollout_rs_kwargs'].get('upper', None),
                            rollout_rs_threshold_lower=rollout_correction_kwargs['rollout_rs_kwargs'].get(
                                'lower', None
                            ),
                            rollout_token_veto_threshold=rollout_correction_kwargs['rollout_token_veto_threshold'],
                            max_turns=max_turns,
                        )
                    else:
                        # MODE 1 (legacy) or MODE 2 (PPO_IS): Standard PPO with clipping
                        # MODE 1: old_log_prob computed by trainer, standard PPO semantics
                        # MODE 2 (bypass): old_log_prob = rollout_log_prob (set by trainer)
                        #   - PPO clips ratio = π_current / π_rollout (instead of π_current / π_old)
                        #   - IS correction happens implicitly through the ratio
                        #   - No pre-computed rollout_is_weights (will be None)
                        policy_output = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange_high=clip_ratio_high,
                            cliprange_low=clip_ratio_low,
                            clip_ratio_c=clip_ratio_c,
                            entropy_clip_rate=entropy_clip_rate,
                            entropy=entropy if entropy_clip_rate > 0 else None,
                            use_gspo=use_gspo,
                            loss_agg_mode=loss_agg_mode,
                            loss_scale_factor=loss_scale_factor,
                            rollout_is_weights=rollout_is_weights,  # Pre-computed weights (legacy mode only)
                            extreme_risk_prob_threshold=extreme_risk_prob_threshold,
                        )

                    # Extract main loss
                    pg_loss = policy_output.loss
                    # compute entropy loss using the same aggregation mode
                    entropy_loss = core_algos.agg_loss(
                        loss_mat=entropy,
                        loss_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        scale_factor=loss_scale_factor,
                    )

                    # For metrics, compute average entropy for interpretability
                    avg_entropy = verl_F.masked_mean(entropy.detach(), response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        # Use the same aggregation mode for KL loss consistency
                        kl_loss = core_algos.agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            scale_factor=loss_scale_factor,
                        )

                        # For metrics, compute average KL for interpretability
                        avg_kl = verl_F.masked_mean(kld.detach(), response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        # Log KL metrics
                        metrics['actor/avg_kl'] = avg_kl.item()
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    # Collect all actor metrics using unified system
                    actor_metrics = policy_output.to_scalars(prefix='actor/')

                    # Add entropy metrics
                    actor_metrics.update(
                        {
                            'actor/avg_entropy': avg_entropy.item(),
                            'actor/entropy_loss': entropy_loss.detach().item(),
                            'actor/entropy_coeff': entropy_coeff,
                            'actor/policy_loss': policy_loss.detach().item(),
                        }
                    )
                    append_to_dict(metrics, actor_metrics)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
