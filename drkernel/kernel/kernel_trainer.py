import json
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_throughout_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator as BaseAdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer as _timer

# AIIC VERL utilities
from verl_patch.monkey_patch.monkey_patch import create_colocated_worker_cls_patch

# AIIC VERL filters
from verl_patch.trainer.code.filters import PPOBatchFilter, PPOFilterConfig
from verl_patch.trainer.code.metrics.multi_turn_metrics import compute_multi_turn_metrics

# AIIC VERL metrics
from verl_patch.trainer.code.metrics.unified_metrics import compute_all_training_metrics
from verl_patch.trainer.code.ppo import core_algos

# Import our extended AdvantageEstimator that includes OPTIMAL_BASELINE
# AIIC VERL PPO imports
from verl_patch.trainer.code.ppo.advantage_estimator import AdvantageEstimator
from verl_patch.trainer.code.ppo.mismatch_helper import (
    compute_rollout_importance_weights_and_rejection_mask,
)
from verl_patch.trainer.code.ppo.variance_reduction import apply_variance_reduction

# AIIC VERL datasets and samplers
from verl_patch.utils.dataset.rl_dataset import RLHFDataset, SolveRateDynamicRLHFDataset
from verl_patch.utils.metric.utils import reduce_metrics
from verl_patch.utils.samplers.always_moderate_sampler import DynamicSolveRateSampler
from verl_patch.utils.samplers.batch_sampler import DynamicBatchSampler
from verl_patch.utils.samplers.prioritized_batch_sampler import PrioritizedBatchSampler
from verl_patch.utils.samplers.refresh_moderate_sampler import RefreshSolveRateSampler

# Kernel-specific metrics
from kernel.metrics.kernel_multi_turn_metrics import compute_kernel_multi_turn_metrics
from kernel.trainer.ppo.core_algos import shape_rewards
from kernel.rewards.coverage_helper import compute_coverage_rejection_mask


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def apply_loss_mask_to_masks(data: DataProto) -> None:
    """Zero out response and attention masks for loss-masked samples.

    This helper mutates the provided ``DataProto`` in-place, ensuring that
    computations downstream (e.g., log-prob evaluation) skip turns whose
    ``loss_mask`` is 0.
    """
    assert (
        "loss_mask" in data.batch
    ), "Expected loss_mask in data batch to apply masking"

    loss_mask = data.batch["loss_mask"]
    if loss_mask.dim() != 1:
        raise ValueError(
            f"Expected `loss_mask` to be 1-D, got shape {tuple(loss_mask.shape)}"
        )
    loss_mask = loss_mask.unsqueeze(1)

    data.batch["response_mask"] = data.batch["response_mask"] * loss_mask
    data.batch["attention_mask"] = data.batch["attention_mask"] * loss_mask


def apply_loss_mask_to_rewards(data: DataProto) -> None:
    """Zero out token-level reward tensors for samples masked by ``loss_mask``."""
    if "loss_mask" not in data.batch:
        return

    loss_mask = data.batch["loss_mask"]
    if loss_mask.dim() != 1:
        raise ValueError(
            f"Expected `loss_mask` to be 1-D, got shape {tuple(loss_mask.shape)}"
        )

    mask = loss_mask.unsqueeze(1)
    masked_keys = ["token_level_scores", "token_level_rewards"]
    for key in masked_keys:
        if key not in data.batch:
            continue
        tensor = data.batch[key]
        data.batch[key] = tensor * mask.to(tensor.dtype)


def get_last_turn_data(data: DataProto, max_turns: int) -> DataProto:
    """
    Extract last turn data from multi-turn DataProto.
    Assumes every max_turns consecutive entries, the last one is the final turn.

    Args:
        data: Multi-turn DataProto
        max_turns: Maximum number of turns (N)

    Returns:
        DataProto containing only last turn data
    """
    if data.batch is None:
        raise ValueError("Expected tensor batch for multi-turn data")

    batch_size = next(iter(data.batch.values())).shape[0]
    if batch_size % max_turns != 0:
        raise ValueError(
            f"Batch size {batch_size} is not divisible by max_turns={max_turns}"
        )

    num_sequences = batch_size // max_turns

    # Extract last turn data for tensor batch keys using reshape to avoid expensive indexing
    last_turn_batch = {}
    for key, value in data.batch.items():
        reshaped = value.reshape(num_sequences, max_turns, *value.shape[1:])
        last_turn_batch[key] = reshaped[:, -1].clone()
    last_turn_batch = TensorDict(last_turn_batch, batch_size=(num_sequences,))

    # Extract last turn data for non-tensor batch keys
    last_turn_non_tensor_batch = {}
    if getattr(data, "non_tensor_batch", None):
        for key, value in data.non_tensor_batch.items():
            if isinstance(value, list):
                array = np.asarray(value, dtype=object).reshape(
                    num_sequences, max_turns
                )
                last_turn_non_tensor_batch[key] = array[:, -1].tolist()
            elif isinstance(value, np.ndarray):
                reshaped = value.reshape(num_sequences, max_turns, *value.shape[1:])
                last_turn_non_tensor_batch[key] = reshaped[:, -1].copy()
            else:
                raise TypeError(f"Unsupported type {type(value)} in non_tensor_batch")

    return DataProto(
        batch=last_turn_batch,
        non_tensor_batch=last_turn_non_tensor_batch,
        meta_info=data.meta_info,
    )


def broadcast_last_turn_to_multi_turn(
    last_turn_data: DataProto, multi_turn_data: DataProto, max_turns: int
) -> DataProto:
    """
    Broadcast computed values from last turn DataProto back to multi-turn DataProto.

    Args:
        last_turn_data: DataProto containing computed values (advantages, returns, etc.) for last turns only
        multi_turn_data: Original multi-turn DataProto to update
        max_turns: Maximum number of turns (N)

    Returns:
        Updated multi-turn DataProto with broadcasted values
    """
    # Basic UID alignment check to ensure ordering is consistent before broadcasting.
    if not (
        hasattr(multi_turn_data, "non_tensor_batch")
        and multi_turn_data.non_tensor_batch
    ):
        raise ValueError(
            "multi_turn_data must provide non_tensor_batch with UID information"
        )
    if "uid" not in multi_turn_data.non_tensor_batch:
        raise ValueError(
            "multi_turn_data.non_tensor_batch is missing 'uid' required for alignment"
        )

    last_uids = np.asarray(last_turn_data.non_tensor_batch["uid"])
    multi_uids = np.asarray(multi_turn_data.non_tensor_batch["uid"])

    if len(multi_uids) % max_turns != 0:
        raise ValueError(
            f"multi_turn_data UID length {len(multi_uids)} is not divisible by max_turns={max_turns}"
        )

    multi_uids_grouped = multi_uids.reshape(-1, max_turns)
    if multi_uids_grouped.shape[0] != len(last_uids):
        raise ValueError(
            "Mismatch between number of sequences in multi_turn_data and last_turn_data during UID alignment"
        )

    if not np.array_equal(multi_uids_grouped[:, -1], last_uids):
        raise ValueError(
            "UID ordering mismatch detected between last_turn_data and multi_turn_data; "
            "ensure filtering does not reorder samples."
        )

    # Identify keys that exist in last_turn_data but not in multi_turn_data
    # These are the computed values we need to broadcast
    keys_to_broadcast = []
    for key in last_turn_data.batch.keys():
        if key not in multi_turn_data.batch:
            keys_to_broadcast.append(key)

    # Get response masks for all turns
    assert (
        "response_mask" in multi_turn_data.batch
    ), "response_mask must be present in multi_turn_data for broadcasting"
    assert (
        "response_mask" in last_turn_data.batch
    ), "response_mask must be present in last_turn_data for broadcasting"

    last_turn_response_mask = last_turn_data.batch["response_mask"]
    # Broadcast each missing key
    for key in keys_to_broadcast:
        last_turn_tensor = last_turn_data.batch[key]
        num_samples = last_turn_tensor.shape[0]
        tensor_shape = last_turn_tensor.shape[1:]
        # Special handling for token_level_rewards, token_level_scores, advantages, and returns
        if key in ["advantages", "returns"]:
            multi_turn_response_mask = multi_turn_data.batch["response_mask"]
            # Validate advantages/returns share identical values across response tokens
            last_mask = last_turn_response_mask.to(last_turn_tensor.dtype)
            diff = (
                ((last_turn_tensor - last_turn_tensor[:, :1]) * last_mask)
                .abs()
                .max()
                .item()
            )
            assert (
                diff < 1e-10
            ), f"Cannot broadcast value for key {key} because not all tokens share the same value"

            values = last_turn_tensor[:, 0]  # (num_samples,)
            expanded_mask = multi_turn_response_mask.reshape(
                num_samples, max_turns, *tensor_shape
            )
            output = (
                expanded_mask.to(last_turn_tensor.dtype)
                * values.view(num_samples, 1, *([1] * len(tensor_shape)))
            ).reshape(num_samples * max_turns, *tensor_shape)
        elif key in {"token_level_rewards", "token_level_scores"}:
            # For other keys, broadcast to all turns as before
            expanded = last_turn_tensor.unsqueeze(1).expand(
                num_samples, max_turns, *tensor_shape
            )
            output = expanded.reshape(
                num_samples * max_turns, *tensor_shape
            ).contiguous()
        else:
            # raise error for unexpected keys
            print(keys_to_broadcast)
            raise ValueError(
                f"Unexpected key '{key}' for broadcasting from last turn to multi-turn data"
            )

        # Add to multi-turn data
        multi_turn_data.batch[key] = output

    # Handle non_tensor_batch - assert all keys already exist in multi_turn_data
    if hasattr(last_turn_data, "non_tensor_batch") and last_turn_data.non_tensor_batch:
        # Assert that multi_turn_data has non_tensor_batch
        assert (
            hasattr(multi_turn_data, "non_tensor_batch")
            and multi_turn_data.non_tensor_batch is not None
        ), "multi_turn_data must have non_tensor_batch if last_turn_data has non_tensor_batch"

        # Assert that all keys in last_turn_data already exist in multi_turn_data
        for key in last_turn_data.non_tensor_batch.keys():
            assert (
                key in multi_turn_data.non_tensor_batch
            ), f"Non-tensor key '{key}' from last_turn_data must already exist in multi_turn_data"

    # Also update any meta_info if needed
    if hasattr(last_turn_data, "meta_info") and last_turn_data.meta_info:
        if (
            not hasattr(multi_turn_data, "meta_info")
            or multi_turn_data.meta_info is None
        ):
            multi_turn_data.meta_info = {}
        multi_turn_data.meta_info.update(last_turn_data.meta_info)

    return multi_turn_data


def compute_rloo_advantages_for_metric_computation(
    batch: DataProto, max_turns: int
) -> DataProto:
    """Compute RLOO advantages for a batch of data.
    Only for metric computation.
    """
    compare_mtrloo = batch.meta_info.get("compare_mtrloo", False)
    if compare_mtrloo:
        rloo_advantages, _ = core_algos.compute_multi_turn_rloo_outcome_advantage(
            token_level_rewards=batch.batch["token_level_rewards"],
            eos_mask=batch.batch["response_mask"],
            loss_mask=batch.batch["loss_mask"],
            turn_indices=batch.batch["turn_indices"],
            index=batch.non_tensor_batch["uid"],
            max_turns=max_turns,
        )
    else:
        if max_turns > 1:
            last_turn_batch = get_last_turn_data(batch, max_turns)
        else:
            last_turn_batch = batch
        rloo_advantages, _ = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=last_turn_batch.batch["token_level_rewards"],
            eos_mask=last_turn_batch.batch["response_mask"],
            index=last_turn_batch.non_tensor_batch["uid"],
        )
        if max_turns > 1:
            response_mask = batch.batch["response_mask"]
            rloo_advantages = (
                rloo_advantages.repeat_interleave(max_turns, dim=0) * response_mask
            )

    batch.batch["rloo_advantages"] = rloo_advantages

    return batch


def compute_multi_turn_advantage(
    data: DataProto,
    max_turns,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    batch_std=False,
    use_multi_prompt_mvu=False,
    reward_shaping=False,
    unbiased_shaping=False,
):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    if adv_estimator == "grpo":
        # Compute turn-level scores
        token_level_rewards = data.batch["token_level_rewards"]
        turn_rewards = token_level_rewards.sum(dim=-1)

        if reward_shaping:
            turn_rewards = shape_rewards(
                turn_rewards, max_turns, gamma, unbiased_shaping
            )
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = core_algos.compute_multi_turn_returns(
            turn_rewards, gamma, max_turns
        )

        index = data.non_tensor_batch["uid"]
        loss_mask = data.batch["loss_mask"]
        turn_indices = data.batch["turn_indices"]

        with torch.no_grad():
            # Group trajectories by prompt
            prompt_groups = defaultdict(list)
            batch_size = turn_scores.shape[0]
            for i in range(batch_size):
                if turn_indices[i].item() == -1 or not loss_mask[i]:
                    continue
                idx = (index[i], turn_indices[i].item())
                prompt_groups[idx].append(i)

            # Compute optimal baseline for each prompt group
            baselines = torch.zeros_like(turn_scores)

            for _, trajectory_indices in prompt_groups.items():
                N = len(trajectory_indices)
                traj_idx = torch.tensor(trajectory_indices, device=turn_scores.device)

                if N == 1:
                    # Single trajectory - no baseline (keep original reward as advantage)
                    baselines[traj_idx[0]] = 0.0
                    continue

                # Extract group data
                R_group = turn_scores[traj_idx]
                # Direct mean value for all in group
                b_star = R_group.mean()
                # Convert to match baselines dtype (epsilon can cause float64 promotion)
                baselines[traj_idx] = b_star.to(baselines.dtype)

            # Compute advantages
            advantages = turn_scores - baselines

        # Expand advantages and returns to match token level rewards
        response_length = token_level_rewards.shape[-1]
        eos_mask = data.batch["response_mask"]
        advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        returns = turn_scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif adv_estimator == "turn_independent_grpo":

        # Compute turn-level scores
        token_level_rewards = data.batch["token_level_rewards"]
        turn_rewards = token_level_rewards.sum(dim=-1)
        # turn_scores = core_algos.compute_multi_turn_returns(turn_rewards, gamma, max_turns)

        if reward_shaping:
            turn_rewards = shape_rewards(
                turn_rewards, max_turns, gamma, unbiased_shaping
            )
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = turn_rewards  # turn-level scores, we do not compute return and use rewards directly

        index = data.non_tensor_batch["uid"]
        loss_mask = data.batch["loss_mask"]
        turn_indices = data.batch["turn_indices"]

        with torch.no_grad():
            # Group trajectories by prompt
            prompt_groups = defaultdict(list)
            batch_size = turn_scores.shape[0]
            for i in range(batch_size):
                if turn_indices[i].item() == -1 or not loss_mask[i]:
                    continue
                # idx = (index[i], turn_indices[i].item())
                # prompt_groups[idx].append(i)
                idx = index[i]
                prompt_groups[idx].append(i)

            # Compute optimal baseline for each prompt group
            baselines = torch.zeros_like(turn_scores)

            for _, trajectory_indices in prompt_groups.items():
                N = len(trajectory_indices)
                traj_idx = torch.tensor(trajectory_indices, device=turn_scores.device)

                if N == 1:
                    # Single trajectory - no baseline (keep original reward as advantage)
                    baselines[traj_idx[0]] = 0.0
                    continue

                # Extract group data
                R_group = turn_scores[traj_idx]
                # Direct mean value for all in group
                b_star = R_group.mean()
                # Convert to match baselines dtype (epsilon can cause float64 promotion)
                baselines[traj_idx] = b_star.to(baselines.dtype)

            # Compute advantages
            advantages = turn_scores - baselines

        # Expand advantages and returns to match token level rewards
        response_length = token_level_rewards.shape[-1]
        eos_mask = data.batch["response_mask"]
        advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        returns = turn_scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif adv_estimator == "egae":
        # GAE for multi-turn training
        # Compute turn-level scores based on the advantage mode
        token_level_rewards = data.batch["token_level_rewards"]
        turn_rewards = token_level_rewards.sum(dim=-1)

        if reward_shaping:
            turn_rewards = shape_rewards(
                turn_rewards, max_turns, gamma, unbiased_shaping
            )
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = core_algos.compute_multi_turn_returns(
            turn_rewards, gamma, max_turns
        )

        # Find the last token position of each response
        response_info = _compute_response_info(data)
        response_lengths = response_info["response_length"].to(torch.long)

        # CRITICAL: Place the turn score at the last token of each turn
        token_level_rewards[
            torch.arange(len(response_lengths)), response_lengths - 1
        ] = turn_scores

        # Now compute standard GAE with the modified token-level rewards
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=data.batch["values"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif adv_estimator == "reinforce":
        response_length = data.batch["token_level_rewards"].shape[-1]
        eos_mask = data.batch["response_mask"]
        scores = data.batch["token_level_rewards"].sum(dim=-1)
        loss_mask = data.batch["loss_mask"].bool()

        if reward_shaping:
            scores = shape_rewards(scores, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = scores

        returns = core_algos.compute_multi_turn_returns(scores, gamma, max_turns)

        with torch.no_grad():
            if len(returns[loss_mask]) == 1:
                return_mean = torch.tensor(0.0)
                return_std = torch.tensor(1.0)
            elif len(returns[loss_mask]) > 1:
                return_mean = torch.mean(returns[loss_mask])
                return_std = torch.std(returns[loss_mask])
            else:
                raise ValueError("No valid returns to compute advantages.")

            advantages = (returns - return_mean) / (return_std + 1e-6)
            advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
            returns = returns.unsqueeze(-1).tile([1, response_length]) * eos_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif adv_estimator == "erloo":
        # ompute advantages using mean of other samples with same prompt
        response_length = data.batch["token_level_rewards"].shape[-1]
        eos_mask = data.batch["response_mask"]
        scores = data.batch["token_level_rewards"].sum(dim=-1)
        loss_mask = data.batch["loss_mask"].bool()
        index = data.non_tensor_batch["uid"]

        if reward_shaping:
            scores = shape_rewards(scores, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = scores

        returns = core_algos.compute_multi_turn_returns(scores, gamma, max_turns)

        id2return = defaultdict(list)
        id2mean = {}

        with torch.no_grad():
            bsz = returns.shape[0]

            advantages = torch.zeros_like(returns)

            for i in range(bsz):
                if not loss_mask[i]:
                    continue
                id2return[index[i]].append(returns[i])

            for idx in id2return:
                if len(id2return[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                elif len(id2return[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2return[idx]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")

            for i in range(bsz):
                if not loss_mask[i]:
                    continue
                idx = index[i]
                response_num = len(id2return[idx])
                if response_num > 1:
                    advantages[i] = returns[i] * response_num / (
                        response_num - 1
                    ) - id2mean[idx] * response_num / (response_num - 1)
                else:
                    advantages[i] = returns[i]

            advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
            returns = returns.unsqueeze(-1).tile([1, response_length]) * eos_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif adv_estimator == "erloo_norm":
        # RLOO with standardization: compute advantages using LOO mean and std
        response_length = data.batch["token_level_rewards"].shape[-1]
        eos_mask = data.batch["response_mask"]
        scores = data.batch["token_level_rewards"].sum(dim=-1)
        loss_mask = data.batch["loss_mask"].bool()
        index = data.non_tensor_batch["uid"]

        if reward_shaping:
            scores = shape_rewards(scores, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = scores

        returns = core_algos.compute_multi_turn_returns(scores, gamma, max_turns)

        with torch.no_grad():
            prompt_groups = defaultdict(list)
            batch_size = returns.shape[0]
            for i in range(batch_size):
                if not loss_mask[i]:
                    # Skip padded prompts
                    continue
                prompt_groups[index[i]].append(i)

            advantages = torch.zeros_like(returns)

            for _, prompt_indices in prompt_groups.items():
                # Calculate LOO mean and std for each prompt
                N = len(prompt_indices)
                group_returns = returns[prompt_indices]

                if N == 1:
                    # Single sample: use return directly (no LOO possible)
                    loo_mean = torch.tensor(0.0)
                    loo_std = torch.tensor(1.0)
                elif N == 2:
                    loo_mean = group_returns.flip(dims=[-1])
                    loo_std = torch.tensor(
                        1.0
                    )  # Only one sample left, leave std to 1.0
                else:
                    # LOO means for each sample
                    total_sum = group_returns.sum()
                    loo_mean = (total_sum - group_returns) / (N - 1)

                    # LOO standard deviations
                    group_returns_repeat = group_returns.unsqueeze(0).repeat(N, 1)
                    loo_mask = torch.ones_like(group_returns_repeat, dtype=torch.bool)
                    loo_mask[torch.arange(N), torch.arange(N)] = False
                    loo_group_returns = group_returns_repeat[loo_mask].reshape(
                        -1, N - 1
                    )
                    loo_std = torch.std(loo_group_returns, dim=-1)

                advantages[prompt_indices] = (group_returns - loo_mean) / (
                    loo_std + 1e-8
                )

            advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
            returns = returns.unsqueeze(-1).tile([1, response_length]) * eos_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == "trloo":
        if reward_shaping:
            raise NotImplementedError("Reward shaping is not supported for trloo.")

        advantages, returns = core_algos.compute_multi_turn_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            loss_mask=data.batch["loss_mask"],
            turn_indices=data.batch["turn_indices"],
            index=data.non_tensor_batch["uid"],
            max_turns=max_turns,
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif adv_estimator == "token_optimal_baseline":
        # Import optimal baseline implementation
        from verl_patch.trainer.code.ppo.optimal_baseline import (
            compute_token_level_optimal_baseline_outcome_advantage,
        )

        # Check if sum_pi_squared is available
        if "sum_pi_squared" not in data.batch:
            raise ValueError(
                "Optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.compute_sum_pi_squared=True in config."
            )

        optimal_baseline_kwargs = data.meta_info.get("optimal_baseline_kwargs", {})
        uniform_weight = optimal_baseline_kwargs.get("uniform_weight", False)
        uniform_cumulative = optimal_baseline_kwargs.get("uniform_cumulative", False)
        rollout_correction = optimal_baseline_kwargs.get("rollout_correction", False)

        # Get pre-computed rollout IS weights if available
        rollout_is_weights = None
        if rollout_correction:
            rollout_is_weights = data.batch.get("rollout_is_weights", None)

        # Compute turn-level scores
        token_level_rewards = data.batch["token_level_rewards"]
        turn_rewards = token_level_rewards.sum(dim=-1)

        if reward_shaping:
            turn_rewards = shape_rewards(
                turn_rewards, max_turns, gamma, unbiased_shaping
            )
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = core_algos.compute_multi_turn_returns(
            turn_rewards, gamma, max_turns
        )
        # Compute token-level scores
        response_length = token_level_rewards.shape[-1]
        eos_mask = data.batch["response_mask"]
        token_scores = turn_scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        # Compute optimal baseline advantages (variance reduction applied separately below)
        advantages, returns = compute_token_level_optimal_baseline_outcome_advantage(
            token_rewards=token_scores,
            response_mask=data.batch["response_mask"],
            old_log_probs=data.batch["old_log_probs"],
            sum_pi_squared=data.batch["sum_pi_squared"],
            loss_mask=data.batch["loss_mask"],
            turn_indices=data.batch["turn_indices"],
            index=data.non_tensor_batch["uid"],
            max_turns=max_turns,
            rollout_is_weights=rollout_is_weights,
            uniform_weight=uniform_weight,
            uniform_cumulative=uniform_cumulative,
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == "turn_optimal_baseline":
        # Import optimal baseline implementation
        from verl_patch.trainer.code.ppo.optimal_baseline import (
            compute_turn_level_optimal_baseline_outcome_advantage,
        )

        # Check if sum_pi_squared is available
        if "sum_pi_squared" not in data.batch:
            raise ValueError(
                "Optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.compute_sum_pi_squared=True in config."
            )

        optimal_baseline_kwargs = data.meta_info.get("optimal_baseline_kwargs", {})
        uniform_weight = optimal_baseline_kwargs.get("uniform_weight", False)
        uniform_cumulative = optimal_baseline_kwargs.get("uniform_cumulative", False)
        rollout_correction = optimal_baseline_kwargs.get("rollout_correction", False)

        # Get pre-computed rollout IS weights if available
        rollout_is_weights = None
        if rollout_correction:
            rollout_is_weights = data.batch.get("rollout_is_weights", None)

        # Compute turn-level scores based on the advantage mode
        token_level_rewards = data.batch["token_level_rewards"]
        turn_rewards = token_level_rewards.sum(dim=-1)

        if reward_shaping:
            turn_rewards = shape_rewards(
                turn_rewards, max_turns, gamma, unbiased_shaping
            )
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = core_algos.compute_multi_turn_returns(
            turn_rewards, gamma, max_turns
        )

        # Compute optimal baseline advantages (variance reduction applied separately below)
        advantages, returns = compute_turn_level_optimal_baseline_outcome_advantage(
            turn_rewards=turn_scores,
            response_mask=data.batch["response_mask"],
            old_log_probs=data.batch["old_log_probs"],
            sum_pi_squared=data.batch["sum_pi_squared"],
            loss_mask=data.batch["loss_mask"],
            turn_indices=data.batch["turn_indices"],
            index=data.non_tensor_batch["uid"],
            max_turns=max_turns,
            rollout_is_weights=rollout_is_weights,
            uniform_weight=uniform_weight,
            uniform_cumulative=uniform_cumulative,
        )

        # Expand advantages and returns to match token level rewards
        response_length = token_level_rewards.shape[-1]
        eos_mask = data.batch["response_mask"]
        advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        returns = returns.unsqueeze(-1).tile([1, response_length]) * eos_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif adv_estimator == "optimal_baseline":
        # Import optimal baseline implementation
        from verl_patch.trainer.code.ppo.optimal_baseline import (
            compute_multi_turn_optimal_baseline,
        )

        # Check if sum_pi_squared is available
        if "sum_pi_squared" not in data.batch:
            raise ValueError(
                "Optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.compute_sum_pi_squared=True in config."
            )

        optimal_baseline_kwargs = data.meta_info.get("optimal_baseline_kwargs", {})
        uniform_weight = optimal_baseline_kwargs.get("uniform_weight", False)
        uniform_cumulative = optimal_baseline_kwargs.get("uniform_cumulative", False)
        rollout_correction = optimal_baseline_kwargs.get("rollout_correction", False)

        # Get pre-computed rollout IS weights if available
        rollout_is_weights = None
        if rollout_correction:
            rollout_is_weights = data.batch.get("rollout_is_weights", None)

        # Compute cumulative reward for each rollout
        token_level_rewards = data.batch[
            "token_level_rewards"
        ]  # [shape: (bs * n * turn, response_length)]
        turn_scores = token_level_rewards.sum(dim=-1)  # [shape: (bs * n * turn,)]

        if reward_shaping:
            turn_scores = shape_rewards(turn_scores, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = turn_scores

        turn_cum_rewards = core_algos.compute_multi_turn_cumulative_rewards(
            turn_scores, max_turns
        )  # [shape: (bs * n * turn,)]
        rewards = turn_cum_rewards.reshape(-1, max_turns)[:, -1]  # [shape: (bs * n,)]
        index = data.non_tensor_batch["uid"].reshape(-1, max_turns)[
            :, -1
        ]  # [shape: (bs * n,)]

        # Compute optimal baseline (variance reduction applied separately below)
        baselines = compute_multi_turn_optimal_baseline(
            rewards=rewards,
            response_mask=data.batch["response_mask"],
            old_log_probs=data.batch["old_log_probs"],
            sum_pi_squared=data.batch["sum_pi_squared"],
            loss_mask=data.batch["loss_mask"],
            index=index,
            max_turns=max_turns,
            rollout_is_weights=rollout_is_weights,
        )  # [shape: (bs * n,)]

        # Compute advantages, same for each turn in one rollout
        advantages = rewards - baselines  # [shape: (bs * n,)]
        advantages = (
            advantages.unsqueeze(-1).tile([1, max_turns]).reshape(-1)
        )  # [shape: (bs * n * turn,)]

        # Expand advantages and returns to match token level rewards
        response_length = token_level_rewards.shape[-1]
        eos_mask = data.batch["response_mask"]
        advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = advantages

    else:
        raise NotImplementedError

    # Apply multi-prompt MVU weighting if enabled (works for ALL estimators)
    if use_multi_prompt_mvu:
        modified_advantages, variance_info = apply_variance_reduction(
            data=data, use_batch_reweighting=False, use_multi_prompt_mvu=True,
        )
        data.batch["advantages"] = modified_advantages

        # Store variance reduction info for logging
        if variance_info is not None:
            data.meta_info["variance_reduction_info"] = variance_info

    # Apply batch standardization if enabled
    if batch_std:
        response_length = data.batch["advantages"].shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        standardized_advantages = core_algos.apply_batch_standardization(
            data.batch["advantages"], response_mask
        )
        data.batch["advantages"] = standardized_advantages

    return data


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    batch_std=False,
    use_multi_prompt_mvu=False,
):
    # Store the original estimator name for tracking
    if isinstance(adv_estimator, str):
        data.meta_info["advantage_estimator"] = adv_estimator
    else:
        data.meta_info["advantage_estimator"] = (
            adv_estimator.value
            if hasattr(adv_estimator, "value")
            else str(adv_estimator)
        )

    # Convert string to enum if needed
    if isinstance(adv_estimator, str):
        if adv_estimator == "optimal_baseline":
            adv_estimator = AdvantageEstimator.OPTIMAL_BASELINE
        elif adv_estimator == "optimal_baseline_step":
            adv_estimator = AdvantageEstimator.OPTIMAL_BASELINE_STEP
        else:
            # Try our extended enum first, then base
            try:
                adv_estimator = AdvantageEstimator(adv_estimator)
            except ValueError:
                adv_estimator = BaseAdvantageEstimator(adv_estimator)

    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator in [AdvantageEstimator.GAE, BaseAdvantageEstimator.GAE, "gae"]:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in [
        AdvantageEstimator.GRPO,
        BaseAdvantageEstimator.GRPO,
        "grpo",
    ]:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in [
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        BaseAdvantageEstimator.REINFORCE_PLUS_PLUS,
        "reinforce_plus_plus",
    ]:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in [
        AdvantageEstimator.REMAX,
        BaseAdvantageEstimator.REMAX,
        "remax",
    ]:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            eos_mask=data.batch["response_mask"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in [
        AdvantageEstimator.RLOO,
        BaseAdvantageEstimator.RLOO,
        "rloo",
    ]:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator in [AdvantageEstimator.OPTIMAL_BASELINE, "optimal_baseline"]:
        # Import optimal baseline implementation (outcome-level)
        from verl_patch.trainer.code.ppo.optimal_baseline import (
            compute_optimal_baseline_outcome_advantage,
        )

        # Check if sum_pi_squared is available
        if "sum_pi_squared" not in data.batch:
            raise ValueError(
                "Optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.compute_sum_pi_squared=True in config."
            )

        optimal_baseline_kwargs = data.meta_info.get("optimal_baseline_kwargs", {})
        uniform_weight = optimal_baseline_kwargs.get("uniform_weight", False)
        uniform_cumulative = optimal_baseline_kwargs.get("uniform_cumulative", False)
        rollout_correction = optimal_baseline_kwargs.get("rollout_correction", False)

        # Get pre-computed rollout IS weights if available
        rollout_is_weights = None
        if rollout_correction:
            rollout_is_weights = data.batch.get("rollout_is_weights", None)

        # Compute outcome baseline: single baseline per trajectory
        advantages, returns = compute_optimal_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            old_log_probs=data.batch["old_log_probs"],
            sum_pi_squared=data.batch["sum_pi_squared"],
            index=data.non_tensor_batch["uid"],
            rollout_is_weights=rollout_is_weights,
            uniform_weight=uniform_weight,
            uniform_cumulative=uniform_cumulative,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator in [
        AdvantageEstimator.OPTIMAL_BASELINE_STEP,
        "optimal_baseline_step",
    ]:
        # Import step-dependent optimal baseline implementation
        from verl_patch.trainer.code.ppo.optimal_baseline import (
            compute_optimal_baseline_step_advantage,
        )

        # Check if sum_pi_squared is available
        if "sum_pi_squared" not in data.batch:
            raise ValueError(
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.compute_sum_pi_squared=True in config."
            )

        optimal_baseline_kwargs = data.meta_info.get("optimal_baseline_kwargs", {})
        uniform_weight = optimal_baseline_kwargs.get("uniform_weight", False)
        uniform_cumulative = optimal_baseline_kwargs.get("uniform_cumulative", False)
        rollout_correction = optimal_baseline_kwargs.get("rollout_correction", False)

        # Get pre-computed rollout IS weights if available
        rollout_is_weights = None
        if rollout_correction:
            rollout_is_weights = data.batch.get("rollout_is_weights", None)

        # Compute step-dependent baseline: unique baseline per timestep
        advantages, returns = compute_optimal_baseline_step_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            old_log_probs=data.batch["old_log_probs"],
            sum_pi_squared=data.batch["sum_pi_squared"],
            index=data.non_tensor_batch["uid"],
            rollout_is_weights=rollout_is_weights,
            uniform_weight=uniform_weight,
            uniform_cumulative=uniform_cumulative,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    else:
        raise NotImplementedError

    # Apply multi-prompt MVU weighting if enabled (works for ALL estimators)
    if use_multi_prompt_mvu:
        modified_advantages, variance_info = apply_variance_reduction(
            data=data, use_batch_reweighting=False, use_multi_prompt_mvu=True,
        )
        data.batch["advantages"] = modified_advantages

        # Store variance reduction info for logging
        if variance_info is not None:
            data.meta_info["variance_reduction_info"] = variance_info

    # Apply batch standardization if enabled
    if batch_std:
        response_length = data.batch["advantages"].shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        standardized_advantages = core_algos.apply_batch_standardization(
            data.batch["advantages"], response_mask
        )
        data.batch["advantages"] = standardized_advantages

    return data


from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # According to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # Use the actual batch size for KL controller update
    # For multi-turn data, use the original batch size, not the number of turns
    # if 'sample_indices' in data.batch and 'turn_indices' in data.batch:
    if "uid" in data.non_tensor_batch:
        # Count unique samples (original batch size) instead of total turns
        uids = data.non_tensor_batch["uid"]
        unique_samples = torch.unique(uids).numel()
        print(
            f"[DEBUG] KL using unique samples count: {unique_samples} instead of {batch_size}"
        )
        kl_ctrl.update(current_kl=current_kl, n_steps=unique_samples)
    else:
        logger.warning("uids is not found in the data.non_tensor_batch")
        kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }

    return data, metrics


# Define a decorator to dynamically add functions to a specified class
def bind_to_class(target_class):
    def decorator(func):
        setattr(target_class, func.__name__, func)
        return func

    return decorator


class RayKernelTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # Keep PPO updates on policy by scaling mini-batch size when multi-turn rollouts are used.
        multi_turn_cfg = config.actor_rollout_ref.rollout.get("multi_turn", None)
        if multi_turn_cfg and multi_turn_cfg.enable:
            max_turns = multi_turn_cfg.max_user_turns
            original_ppo_mini_batch_size = (
                config.actor_rollout_ref.actor.ppo_mini_batch_size
            )
            config.actor_rollout_ref.actor.ppo_mini_batch_size *= max_turns
            print(
                f"[multi_turn] actor.ppo_mini_batch_size scaled from {original_ppo_mini_batch_size} "
                f"to {config.actor_rollout_ref.actor.ppo_mini_batch_size} (max_turns={max_turns}) to ensure that"
                " multi-turn training is on-policy when specifying the same batch_size and ppo_mini_batch_size."
            )

        # Ensure sum_pi_squared is enabled for optimal baseline (required for this estimator)
        # Note: compute_sum_pi_squared is now enabled by default for all algorithms to support variance proxy metrics
        if "optimal_baseline" in config.algorithm.adv_estimator:
            if not config.actor_rollout_ref.actor.get("compute_sum_pi_squared", True):
                OmegaConf.set_struct(config.actor_rollout_ref.actor, False)
                config.actor_rollout_ref.actor.compute_sum_pi_squared = True
                OmegaConf.set_struct(config.actor_rollout_ref.actor, True)
                print(
                    "Warning: Forcing compute_sum_pi_squared=True for optimal_baseline (required)"
                )

        # Temporarily change adv_estimator to bypass parent class check
        # This is necessary because open_verl's RayPPOTrainer only recognize BaseAdvantageEstimator
        use_external_adv_estimator = False
        if config.algorithm.adv_estimator not in list(BaseAdvantageEstimator):
            use_external_adv_estimator = True
            external_external_adv_estimator = config.algorithm.adv_estimator
            if "gae" in external_external_adv_estimator:
                config.algorithm.adv_estimator = "gae"
            else:
                config.algorithm.adv_estimator = "grpo"

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
        )

        if use_external_adv_estimator:
            self.config.algorithm.adv_estimator = external_external_adv_estimator

        # Convert algorithm config to dataclass to validate it
        self.algorithm_cfg = omega_conf_to_dataclass(self.config.algorithm)

        # Initialize batch filter after super().__init__
        self._initialize_batch_filter()

    def _initialize_batch_filter(self):
        """Initialize the unified batch filter with appropriate configuration."""
        self.batch_filter = None

        # Check if we need the filter
        needs_filter = (
            self.config.data.get("prompt_oversampling_factor", 1.0) > 1.0
            or self.config.data.get("sample_oversampling_factor", 1.0) > 1.0
            or self.config.get("rejection_sampling", {}).get(
                "enable_two_gate_filter", False
            )
            or self.config.trainer.rejection_sample
            or self.config.trainer.get("remove_clip", False)
        )

        if needs_filter:
            # Derive target number of groups; for multi-turn with turn-level advantages, we need groups per turn
            target_num_groups = self.config.data.train_batch_size
            multi_turn_cfg = self.config.actor_rollout_ref.rollout.get(
                "multi_turn", None
            )
            if (
                multi_turn_cfg
                and multi_turn_cfg.enable
                and not self.config.algorithm.is_get_last_turn
            ):
                target_num_groups *= multi_turn_cfg.max_user_turns

            # Create unified filter config
            filter_config = PPOFilterConfig(
                # Sample selection (filter doesn't handle oversampling - that's done at DataLoader/generation)
                sample_selection_strategy=self.config.data.get(
                    "sample_selection_strategy", "efficiency_stochastic"
                ),
                # Group management
                target_group_size=self.config.actor_rollout_ref.rollout.n,
                min_group_size=self.config.data.get(
                    "min_group_size", None
                ),  # None = auto-set to target_group_size // 2 + 1
                target_num_groups=target_num_groups,  # Number of groups (prompts or turns) to select after filtering
                # Rejection sampling
                reward_threshold=None,  # Not currently configured via command line
                max_response_length=(
                    self.config.data.get("max_response_length", None)
                    if self.config.trainer.rejection_sample
                    or self.config.trainer.get("remove_clip", False)
                    else None
                ),
                reject_low_variance_groups=self.config.trainer.get(
                    "rejection_sample", False
                ),  # Only reject low variance when rejection_sample is True
                remove_clip=self.config.trainer.get(
                    "remove_clip", False
                ),  # Whether to use remove_clip logic
                min_rollout_n=(
                    self.config.actor_rollout_ref.rollout.get("min_n", None)
                    if self.config.trainer.get("remove_clip", False)
                    else None
                ),
                # Two-gate filter
                enable_two_gate_filter=self.config.get("rejection_sampling", {}).get(
                    "enable_two_gate_filter", False
                ),
                gate1_enabled=self.config.get("rejection_sampling", {})
                .get("gate1", {})
                .get("enabled", True),
                gate1_bias_epsilon=self.config.get("rejection_sampling", {})
                .get("gate1", {})
                .get("bias_epsilon", 0.01),
                gate2_enabled=self.config.get("rejection_sampling", {})
                .get("gate2", {})
                .get("enabled", True),
                gate2_instability_threshold=self.config.get("rejection_sampling", {})
                .get("gate2", {})
                .get("instability_threshold", -15.0),
                # Metrics
                save_metrics=True,
                log_rejection_reasons=self.config.get("rejection_sampling", {}).get(
                    "log_rejected_samples", False
                ),
            )

            self.batch_filter = PPOBatchFilter(filter_config)
            print(f"PPO Batch Filter initialized:")
            print(f"  Selection strategy: {filter_config.sample_selection_strategy}")
            print(f"  Min group size: {filter_config.min_group_size}")
            print(
                f"  Two-gate filter: {'Enabled' if filter_config.enable_two_gate_filter else 'Disabled'}"
            )

        # Keep backward compatibility alias
        self.oversampling_filter = self.batch_filter

    def _compute_suggested_sample_factor(
        self, expected_samples: int, selected_samples: int
    ) -> Optional[float]:
        """
        Compute the minimum prompt_oversampling_factor required to reach the expected
        number of samples, based on the currently selected sample count.
        """
        if selected_samples <= 0 or expected_samples <= 0:
            return None

        current_factor = float(self.config.data.get("prompt_oversampling_factor", 1.0))
        ratio = expected_samples / selected_samples
        if ratio <= 1.0:
            return current_factor

        return max(current_factor, current_factor * ratio)

    def _save_to_buffer(
        self,
        buffer_batch: Optional[DataProto],
        batch: DataProto,
        disable_buffer: bool,
        multi_turn_batch: Optional[DataProto],
        use_multi_turn: bool,
    ) -> Optional[DataProto]:
        """Store skipped samples so we can reuse them once enough accumulate."""
        if disable_buffer:
            return None

        avaiable_keys = (
            multi_turn_batch.batch.keys() if use_multi_turn else batch.batch.keys()
        )
        select_keys = [
            "input_ids",
            "rollout_log_probs",
            "position_ids",
            "loss_mask",
            "turn_indices",
            "attention_mask",
            "sample_indices",
            "prompts",
            "responses",
            "response_mask",
            "top_log_probs",
            "token_level_scores",
            "uid",
        ]
        select_keys = [key for key in select_keys if key in avaiable_keys]
        if use_multi_turn:
            saved_batch = multi_turn_batch.select(batch_keys=select_keys)
        else:
            saved_batch = batch.select(batch_keys=select_keys)

        if buffer_batch is None:
            buffer_batch = saved_batch
        else:
            buffer_batch = DataProto.concat([buffer_batch, saved_batch])

        per_prompt = max(self.config.actor_rollout_ref.rollout.n, 1)
        current_size = batch.batch["input_ids"].shape[0] // per_prompt
        print(f"[Buffer] Save to buffer, current buffer size: {current_size}")

        return buffer_batch

    def _load_from_buffer(
        self, batch: DataProto, buffer_batch: Optional[DataProto], disable_buffer: bool,
    ) -> Tuple[DataProto, Optional[DataProto]]:
        """
        Load buffered samples back into the working batch if buffering is enabled.

        Returns the updated batch and the new buffer state (None when buffer is consumed).
        """
        if buffer_batch is None:
            return batch, buffer_batch

        if disable_buffer:
            print("Buffer is disabled, discarding buffered batch")
            return batch, None

        # (Qian): make sure that the buffered batch and current batch have the same keys
        keys_to_remove = set(buffer_batch.batch.keys()) - set(batch.batch.keys())
        if keys_to_remove:
            buffer_batch = buffer_batch.select(
                batch_keys=[
                    key
                    for key in buffer_batch.batch.keys()
                    if key not in keys_to_remove
                ]
            )

        combined_batch = DataProto.concat([buffer_batch, batch])
        per_prompt = max(self.config.actor_rollout_ref.rollout.n, 1)
        print(
            "[Buffer] Using buffered batch to form a new batch. The new batch size is:",
            combined_batch.batch["input_ids"].shape[0] // per_prompt,
        )

        return combined_batch, None

    def _get_max_world_size(self) -> int:
        """
        Calculate the maximum world size across all worker groups.

        Returns:
            Maximum world size needed for batch padding to ensure proper distributed chunking.
        """
        max_world_size = self.actor_rollout_wg.world_size
        if self.use_critic:
            max_world_size = max(max_world_size, self.critic_wg.world_size)
        if self.use_reference_policy:
            max_world_size = max(max_world_size, self.ref_policy_wg.world_size)
        if self.use_rm:
            max_world_size = max(max_world_size, self.rm_wg.world_size)

        # if we use multi_turn, the max_world_size must be multiple of max_turns
        if self.config.actor_rollout_ref.rollout.multi_turn.enable:
            max_turns = self.config.actor_rollout_ref.rollout.multi_turn.max_user_turns
            if max_world_size % max_turns != 0:
                print(
                    f"Warning: max_world_size {max_world_size} is not multiple of max_turns {max_turns}, adjusting it to be multiple of max_turns"
                )
                max_world_size = max_world_size * max_turns

        return max_world_size

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        if self.config.data.get("use_moderate_sampling", False) or self.config.data.get(
            "use_refresh_sampling", False
        ):
            self.train_dataset = SolveRateDynamicRLHFDataset(
                parquet_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                image_key=self.config.data.get("image_key", "images"),
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                apply_chat_template=self.config.data.apply_chat_template,
                return_raw_chat=self.config.data.get("return_raw_chat", False),
                truncation=self.config.data.get("truncation", "error"),
                filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                system_prompt_config=self.config.data.get("system_prompt_config", None),
            )
        else:
            self.train_dataset = RLHFDataset(
                parquet_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                image_key=self.config.data.get("image_key", "images"),
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                apply_chat_template=self.config.data.apply_chat_template,
                return_raw_chat=self.config.data.get("return_raw_chat", False),
                truncation=self.config.data.get("truncation", "error"),
                filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                system_prompt_config=self.config.data.get("system_prompt_config", None),
            )
        assert self.train_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'

        # use basic sampler for train dataloader
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            base_sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            base_sampler = SequentialSampler(data_source=self.train_dataset)

        train_batch_size = self.config.data.train_batch_size
        world_size = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        rollout_n = getattr(self.config.actor_rollout_ref.rollout, "n", None)

        # check if we use prioritized sampling
        if self.config.data.get("use_prioritized_sampling", False):
            # create a prioritized batch sampler
            batch_sampler = PrioritizedBatchSampler(
                sampler=base_sampler,
                target_batch_size=train_batch_size,
                oversampling_factor=self.config.data.get(
                    "prompt_oversampling_factor", 1.2
                ),
                default_filter_rate=self.config.data.get("default_filter_rate", 0.0),
                ema_alpha=self.config.data.get("ema_alpha", 1.0),
                shuffle=self.config.data.get("prompt_sampler_shuffle", True),
                seed=self.config.data.get("seed", 42),
                world_size=world_size,
            )
            self.prioritized_batch_sampler = batch_sampler
            self.dynamic_batch_sampler = batch_sampler
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )
        elif self.config.data.get("use_moderate_sampling", False):
            batch_sampler = DynamicSolveRateSampler(
                dataset=self.train_dataset,
                sampler=base_sampler,
                target_batch_size=train_batch_size,
                oversampling_factor=self.config.data.get(
                    "prompt_oversampling_factor", 1.2
                ),
                default_filter_rate=self.config.data.get("default_filter_rate", 0.0),
                ema_alpha=self.config.data.get("ema_alpha", 1.0),
                shuffle=True,
                seed=self.config.data.get("seed", 42),
                world_size=world_size,
                solverate_low=self.config.data.get("solverate_low", 0.1),
                solverate_high=self.config.data.get("solverate_high", 0.9),
            )
            self.moderate_batch_sampler = batch_sampler
            self.dynamic_batch_sampler = batch_sampler
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )
        elif self.config.data.get("use_refresh_sampling", False):
            # Early configuration check with helpful context
            oversampling_factor = self.config.data.get(
                "prompt_oversampling_factor", 1.2
            )
            num_prompts = train_batch_size * oversampling_factor

            # Provide informative warning if configuration might be problematic
            if num_prompts < world_size:
                min_oversampling = world_size / train_batch_size
                rollout_n = self.config.actor_rollout_ref.rollout.n
                total_trajectories = num_prompts * rollout_n
                # Note: The sampler will validate and raise its own error, but we provide
                # additional context here about rollout_n that the sampler doesn't have
                print(f"\n{'='*60}")
                print(f"Configuration Warning for RefreshSolveRateSampler:")
                print(
                    f"  Number of prompts = {train_batch_size} * {oversampling_factor} = {num_prompts}"
                )
                print(f"  World size = {world_size} GPUs")
                print(
                    f"  After rollout (n={rollout_n}): {total_trajectories} total trajectories"
                )
                print(f"  ({total_trajectories/world_size:.1f} trajectories per GPU)")
                print(
                    f"\nThe sampler requires num_prompts >= world_size, so it will fail."
                )
                print(f"Suggested oversampling_factor >= {min_oversampling:.1f}")
                print(f"{'='*60}\n")

            batch_sampler = RefreshSolveRateSampler(
                dataset=self.train_dataset,
                sampler=base_sampler,
                target_batch_size=train_batch_size,
                default_filter_rate=0.0,
                oversampling_factor=oversampling_factor,
                ema_alpha=self.config.data.get("ema_alpha", 1.0),
                shuffle=True,
                seed=self.config.data.get("seed", 42),
                solverate_high=self.config.data.get("solverate_high", 1.0),
                solverate_low=self.config.data.get("solverate_low", 0.0),
                solverate_mean=self.config.data.get("solverate_mean", 0.5),
                solverate_std=self.config.data.get("solverate_std", 0.1),
                freshness_balance=0.1,
                current_step=0,
                world_size=world_size,
                propagation_threshold=5,
                propagation_confidence=0.8,
            )
            self.refresh_batch_sampler = batch_sampler
            self.dynamic_batch_sampler = batch_sampler
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )
        # Check if dynamic batch sampling is enabled
        elif self.config.data.get("automatic_oversampling", False):
            # Create dynamic batch sampler
            batch_sampler = DynamicBatchSampler(
                sampler=base_sampler,
                target_batch_size=train_batch_size,
                oversampling_factor=self.config.data.get(
                    "prompt_oversampling_factor", 1.2
                ),
                default_filter_rate=self.config.data.get("default_filter_rate", 0.0),
                ema_alpha=self.config.data.get("ema_alpha", 1.0),
                shuffle=self.config.data.get("prompt_sampler_shuffle", True),
                seed=self.config.data.get("seed", 42),
                world_size=world_size,
            )
            self.dynamic_batch_sampler = batch_sampler

            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )
        else:
            # Apply prompt-level oversampling when rejection sampling is enabled
            if self.config.trainer.rejection_sample or self.config.trainer.remove_clip:
                # Use prompt_oversampling_factor from config (default 1.0 means no oversampling)
                prompt_oversample = self.config.data.get(
                    "prompt_oversampling_factor", 1.0
                )
                if prompt_oversample > 1.0:
                    train_batch_size = int(train_batch_size * prompt_oversample)
                    # Round batch size to world size multiple
                    train_batch_size = int(train_batch_size // world_size * world_size)

            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=train_batch_size,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=base_sampler,
            )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            sample_size=self.config.data.val_sample_size,
            apply_chat_template=self.config.data.apply_chat_template,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
            system_prompt_config=self.config.data.get("system_prompt_config", None),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation uses whole batch for memory scheduling
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert (
            len(self.val_dataloader) == 1
        ), "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        # Inject total_training_steps to optimizer configs
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _update_sampler_states(self, batch: DataProto, metrics: Dict[str, Any]) -> None:
        """Update sampler statistics and success rates after reward computation."""

        update_every = self.config.data.get("update_success_rates_every", 1)
        should_update_success = (
            update_every > 0 and self.global_steps % update_every == 0
        )

        if not should_update_success:
            return

        prioritized_sampler = getattr(self, "prioritized_batch_sampler", None)
        moderate_sampler = getattr(self, "moderate_batch_sampler", None)
        refresh_sampler = getattr(self, "refresh_batch_sampler", None)

        if not any((prioritized_sampler, moderate_sampler, refresh_sampler)):
            return

        problem_ids = np.asarray(batch.non_tensor_batch["prompt_index"])
        rewards_tensor = batch.batch["token_level_scores"].sum(-1)
        rewards = (
            rewards_tensor.detach().cpu().numpy()
            if isinstance(rewards_tensor, torch.Tensor)
            else np.asarray(rewards_tensor)
        )

        success_threshold = self.config.data.get("success_threshold", 1.0)
        success_rates = self._compute_success_rates(
            problem_ids, rewards, success_threshold
        )

        if prioritized_sampler:
            prioritized_sampler.update_success_rates(success_rates)
            metrics.update(prioritized_sampler.get_metrics())

        if moderate_sampler:
            moderate_sampler.update_success_rates(success_rates)
            metrics.update(moderate_sampler.get_metrics())

        if refresh_sampler:
            refresh_sampler.set_current_step(self.global_steps)
            refresh_sampler.update_success_rates(success_rates)
            refresh_sampler.print_solve_rate_bin_distribution()

    @staticmethod
    def _compute_success_rates(
        problem_ids, rewards, success_threshold: float
    ) -> dict[int, float]:
        problem_ids = np.asarray(problem_ids)
        rewards = np.asarray(rewards)

        success_rates: dict[int, float] = {}
        unique_problem_ids = np.unique(problem_ids)
        for problem_id in unique_problem_ids:
            mask = problem_ids == problem_id
            success_count = np.sum(rewards[mask] >= success_threshold)
            total_count = np.sum(mask)
            key = int(problem_id) if isinstance(problem_id, np.integer) else problem_id
            success_rates[key] = success_count / total_count if total_count > 0 else 0.0

        return success_rates

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # Initialize WorkerGroup
        # NOTE: For different parallel sizes per role, use separate resource pools instead of create_colocated_worker_cls
        # See: https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls_patch(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # Keep WorkerDict reference for Ray >= 2.31 compatibility
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # Create rollout last for better vLLM KV cache memory estimation
        self.actor_rollout_wg = all_wg["actor_rollout"]

        self.actor_rollout_wg.init_model()

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async_vllm":
            self.async_rollout_mode = True
            from kernel.workers.rollout.async_server import AsyncLLMEngineManager

            self.async_rollout_manager = AsyncLLMEngineManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_fn,
                val_reward_fn=self.val_reward_fn,
            )

        # IMPORTANT: This happens ONLY for sufficient batches (after buffering)
        # We find the maximum world_size across all worker groups to ensure compatibility
        # Why max()? If batch is divisible by 128, it's also divisible by 64, 32, 16, etc.
        max_world_size = self.actor_rollout_wg.world_size
        if self.use_critic:
            max_world_size = max(max_world_size, self.critic_wg.world_size)
        if self.use_reference_policy:
            max_world_size = max(max_world_size, self.ref_policy_wg.world_size)
        if self.use_rm:
            max_world_size = max(max_world_size, self.rm_wg.world_size)
        # if we use multi_turn, the max_world_size must be multiple of max_turns
        if self.config.actor_rollout_ref.rollout.multi_turn.enable:
            max_turns = self.config.actor_rollout_ref.rollout.multi_turn.max_user_turns
            if max_world_size % max_turns != 0:
                print(
                    f"Warning: max_world_size {max_world_size} is not multiple of max_turns {max_turns}, adjusting it to be multiple of max_turns"
                )
                max_world_size = max_world_size * max_turns

        self.max_world_size = max_world_size

    def _save_checkpoint(self):
        super()._save_checkpoint()

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        # save sampler state if using RefreshSolveRateSampler
        if hasattr(self.train_dataloader, "batch_sampler") and hasattr(
            self.train_dataloader.batch_sampler, "save_state"
        ):
            sampler_local_path = os.path.join(
                local_global_step_folder, "sampler_state.pkl"
            )
            self.train_dataloader.batch_sampler.save_state(sampler_local_path)
            print(f"Sampler state saved to {sampler_local_path}")

    def _load_checkpoint(self):
        super()._load_checkpoint()

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if not (
                self.config.trainer.resume_from_path and global_step_folder is not None
            ):
                assert isinstance(
                    self.config.trainer.resume_from_path, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_from_path
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        # load sampler state if using RefreshSolveRateSampler
        if hasattr(self.train_dataloader, "batch_sampler") and hasattr(
            self.train_dataloader.batch_sampler, "load_state"
        ):
            sampler_local_path = os.path.join(global_step_folder, "sampler_state.pkl")
            if os.path.exists(sampler_local_path):
                self.train_dataloader.batch_sampler.load_state(sampler_local_path)
                print(f"Sampler state loaded from {sampler_local_path}")
            else:
                print(
                    f"Warning: No sampler state found at {sampler_local_path}, sampler will start from initial state"
                )

        # IMPORTANT: This happens ONLY for sufficient batches (after buffering)
        # We find the maximum world_size across all worker groups to ensure compatibility
        # Why max()? If batch is divisible by 128, it's also divisible by 64, 32, 16, etc.
        self.max_world_size = self._get_max_world_size()

    def _save_checkpoint(self):
        super()._save_checkpoint()

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        # save sampler state if using RefreshSolveRateSampler
        if hasattr(self.train_dataloader, "batch_sampler") and hasattr(
            self.train_dataloader.batch_sampler, "save_state"
        ):
            sampler_local_path = os.path.join(
                local_global_step_folder, "sampler_state.pkl"
            )
            self.train_dataloader.batch_sampler.save_state(sampler_local_path)
            print(f"Sampler state saved to {sampler_local_path}")

    def _load_checkpoint(self):
        super()._load_checkpoint()

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if not (
                self.config.trainer.resume_from_path and global_step_folder is not None
            ):
                assert isinstance(
                    self.config.trainer.resume_from_path, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_from_path
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        # load sampler state if using RefreshSolveRateSampler
        if hasattr(self.train_dataloader, "batch_sampler") and hasattr(
            self.train_dataloader.batch_sampler, "load_state"
        ):
            sampler_local_path = os.path.join(global_step_folder, "sampler_state.pkl")
            if os.path.exists(sampler_local_path):
                self.train_dataloader.batch_sampler.load_state(sampler_local_path)
                print(f"Sampler state loaded from {sampler_local_path}")
            else:
                print(
                    f"Warning: No sampler state found at {sampler_local_path}, sampler will start from initial state"
                )

        if self.config.actor_rollout_ref.rollout.mode == "async_agent":
            self.async_rollout_mode = True
            from verl_patch.experimental.agent_loop.agent_loop import AgentLoopManager

            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg,
            )

    def compute_pass_at_k(self, results: list[list[bool]], k: int):
        """
        Compute the average pass@k metric for a list of problem results.

        Args:
            results: A list of lists of booleans, where each sublist represents the success of samples for a problem.
            k: The number of samples to consider (k in pass@k).

        Returns:
            The average pass@k score across all problems.
        """

        if k < 1:
            raise ValueError("k must be at least 1")

        pass_rates = []
        for problem in results:
            n = len(problem)
            if n < k:
                raise ValueError(
                    f"Each problem must have at least {k} samples, found {n}"
                )

            correct = sum(problem)
            if correct == 0:
                pass_rates.append(0.0)
                continue

            fail_prob = 1.0
            for i in range(k):
                fail_prob *= (n - correct - i) / (n - i)

            pass_rates.append(1 - fail_prob)

        return sum(pass_rates) / len(pass_rates)

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(
            self.config.trainer.logger, samples, self.global_steps
        )

    def _validate(self):
        reward_tensor_lst = []
        reward_extra_info_dict: Optional[Dict[str, list[list[float]]]] = (
            None  # the values are of shape (num_of_batch, batch_size)
        )
        data_source_lst = []

        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        async_rollout_diffs = []

        # For multi-turn metrics: accumulate all test_batches
        all_test_batches = []

        def _coerce_extra_metric_value(value: Any) -> Optional[float]:
            """Convert heterogeneous reward extra info entries into scalars when possible."""
            if value is None:
                return None

            if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
                return float(value)

            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return float(value.detach().cpu().item())
                return None

            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.reshape(-1)[0])
                return None

            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return _coerce_extra_metric_value(value[0])
                return None

            return None

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # For multi-turn, repeat by n * max_turns to match flattened output shape
            # if self.config.actor_rollout_ref.rollout.multi_turn.enable:
            #     max_turns = self.config.actor_rollout_ref.rollout.multi_turn.max_user_turns
            #     repeat_times = self.config.actor_rollout_ref.rollout.val_kwargs.n * max_turns
            # else:
            #     repeat_times = self.config.actor_rollout_ref.rollout.val_kwargs.n

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            test_batch.non_tensor_batch["uid"] = np.array(
                [
                    f"test_batch_{self.global_steps}_example_{uuid4().hex}"
                    for i in range(len(test_batch.batch))
                ],
                dtype=object,
            )

            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original input for later use (before repeat)
            # For multi-turn, we'll build complete conversations after generation
            if not self.config.actor_rollout_ref.rollout.multi_turn.enable:
                input_ids = test_batch.batch["input_ids"]
                input_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in input_ids
                ]
                sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "uid"]

            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # for uid, we assign it back from gen_batch to batch
            test_batch.non_tensor_batch["uid"] = test_gen_batch.non_tensor_batch["uid"]
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_step": self.global_steps,
                "actual_max_turns": self.config.actor_rollout_ref.rollout.val_kwargs.max_user_turns,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # Add reward_model to gen_batch if it exists in batch, but keep it in original batch
            # Only add for multi_turn async mode to avoid batch size mismatch
            if (
                "reward_model" in test_batch.non_tensor_batch
                and self.async_rollout_mode
            ):
                test_gen_batch.non_tensor_batch[
                    "reward_model"
                ] = test_batch.non_tensor_batch["reward_model"]
                test_gen_batch.non_tensor_batch[
                    "data_source"
                ] = test_batch.non_tensor_batch["data_source"]

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch_padded
                )
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                    test_gen_batch_padded
                )
                self.async_rollout_manager.sleep()

            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # test_batch has already been repeated to match test_output_gen_batch shape
            # (by n * max_turns for multi-turn, or just n for single-turn)
            async_rollout_diffs.append(
                len(test_batch.batch) - len(test_output_gen_batch.batch)
            )
            # if batch is larger than gen_batch_output, which means that some prompts have been filtered out because of async timeout
            if len(test_batch.batch) > len(test_output_gen_batch.batch):
                # use uid to filter
                gen_uids = set(test_output_gen_batch.non_tensor_batch["uid"])
                batch_mask = np.array(
                    [uid in gen_uids for uid in test_batch.non_tensor_batch["uid"]]
                )
                test_batch = test_batch[batch_mask]
                # guarantee uid alignment with generated outputs as in training
                assert False not in (
                    test_batch.non_tensor_batch["uid"]
                    == test_output_gen_batch.non_tensor_batch["uid"]
                )
                test_batch.non_tensor_batch[
                    "uid"
                ] = test_output_gen_batch.non_tensor_batch["uid"]

            # Build sample_inputs and sample_outputs
            if self.config.actor_rollout_ref.rollout.multi_turn.enable:
                # For multi-turn, use multiturn_messages to build complete conversations
                sample_indices = (
                    test_output_gen_batch.batch["sample_indices"].cpu().numpy()
                )

                # print(f"sample_indices before metrics: {sample_indices}")
                turn_indices = test_output_gen_batch.batch["turn_indices"].cpu().numpy()
                multiturn_messages = test_output_gen_batch.non_tensor_batch.get(
                    "multiturn_messages", None
                )

                # Find first turn (with messages) for each sample
                sample_first_turn = {}  # sample_id -> row_idx of first turn
                for i in range(len(test_output_gen_batch.batch)):
                    # s_idx = int(sample_indices[i])
                    s_idx = test_output_gen_batch.non_tensor_batch["uid"][i]
                    t_idx = int(turn_indices[i])

                    if t_idx == -1:  # Skip padding turns
                        continue

                    # First turn has the messages
                    if s_idx not in sample_first_turn:
                        sample_first_turn[s_idx] = i

                # Build input/output for each sample using multiturn_messages
                for s_idx in sorted(sample_first_turn.keys()):
                    first_idx = sample_first_turn[s_idx]

                    if (
                        multiturn_messages is not None
                        and multiturn_messages[first_idx] is not None
                    ):
                        messages = multiturn_messages[first_idx]

                        # Input: extract first user message
                        first_user_msg = ""
                        for msg in messages:
                            if msg.get("role") == "user":
                                first_user_msg = msg.get("content", "")
                                break
                        sample_inputs.append(first_user_msg)

                        # Output: build complete conversation string
                        full_output = ""
                        for msg in messages:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            full_output += f"[{role}]\n{content}\n\n"
                        sample_outputs.append(full_output)
                    else:
                        # Fallback to prompt/response decoding
                        first_prompt = self.tokenizer.decode(
                            test_output_gen_batch.batch["prompts"][first_idx],
                            skip_special_tokens=True,
                        )
                        sample_inputs.append(first_prompt)
                        sample_outputs.append("[No messages available]")
            else:
                # Original logic for single-turn
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for ids in output_ids
                ]
                sample_outputs.extend(output_texts)

            if self.config.actor_rollout_ref.rollout.multi_turn.enable:
                # max_turns = (
                #     self.config.actor_rollout_ref.rollout.multi_turn.max_user_turns
                # )
                max_turns = (
                    self.config.actor_rollout_ref.rollout.val_kwargs.max_user_turns
                )
                test_batch = test_batch.repeat(repeat_times=max_turns, interleave=True)

            test_batch = test_batch.union(test_output_gen_batch)

            if "token_level_scores" not in test_batch.batch:
                reward_result = self.val_reward_fn(test_batch)

                if isinstance(reward_result, dict):
                    reward_tensor = reward_result["reward_tensor"]
                    cur_data_source = test_batch.non_tensor_batch.get(
                        "data_source", ["unknown"] * reward_tensor.shape[0]
                    )
                    if "extra_info" in reward_result:
                        if reward_extra_info_dict is None:
                            reward_extra_info_dict = {}
                        for key, extra_reward in reward_result["extra_info"].items():
                            for i, data_source in enumerate(cur_data_source):
                                composed_key = f"{key}_{data_source}"
                                if composed_key not in reward_extra_info_dict:
                                    reward_extra_info_dict[composed_key] = []
                                reward_extra_info_dict[composed_key].append(
                                    extra_reward[i]
                                )
                else:
                    reward_tensor = reward_result
                    cur_data_source = test_batch.non_tensor_batch.get(
                        "data_source", ["unknown"] * reward_tensor.shape[0]
                    )
            else:
                reward_tensor = test_batch.batch.pop("token_level_scores")
                cur_data_source = test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
                reward_extra_info_raw = test_batch.non_tensor_batch.get(
                    "reward_extra_info"
                )
                if reward_extra_info_raw is None:
                    reward_extra_info_list = []
                elif hasattr(reward_extra_info_raw, "tolist"):
                    reward_extra_info_list = reward_extra_info_raw.tolist()
                else:
                    reward_extra_info_list = list(reward_extra_info_raw)

                # Filter out empty dicts and error-only dicts (those without kernel metrics)
                # Keep only dicts that have kernel-specific fields like 'correctness', 'performance', etc.
                valid_indices = []
                for i, d in enumerate(reward_extra_info_list):
                    if len(d) > 0:
                        # Check if dict has kernel-specific metrics (not just error info)
                        has_kernel_metrics = any(
                            key in d
                            for key in [
                                "correctness",
                                "performance",
                                "compiled",
                                "success",
                            ]
                        )
                        if has_kernel_metrics:
                            valid_indices.append(i)

                valid_reward_extra_info_list = [
                    reward_extra_info_list[i] for i in valid_indices
                ]
                valid_data_sources = [cur_data_source[i] for i in valid_indices]

                # convert list of dict to dict of list (only for valid entries with kernel metrics)
                if len(valid_reward_extra_info_list) > 0:
                    raw_reward_extra_info_dict = {
                        k: [d[k] for d in valid_reward_extra_info_list]
                        for k in valid_reward_extra_info_list[0].keys()
                    }

                    if reward_extra_info_dict is None:
                        reward_extra_info_dict = {}
                    for key, extra_reward in raw_reward_extra_info_dict.items():
                        for i, data_source in enumerate(valid_data_sources):
                            composed_key = f"{key}_{data_source}"
                            if composed_key not in reward_extra_info_dict:
                                reward_extra_info_dict[composed_key] = []
                            reward_extra_info_dict[composed_key].append(extra_reward[i])

            scores = reward_tensor.sum(-1).cpu().tolist()

            # For multi-turn, aggregate scores per sample
            if self.config.actor_rollout_ref.rollout.multi_turn.enable:
                # sample_indices = test_batch.batch['sample_indices'].cpu().numpy()
                sample_indices = test_batch.non_tensor_batch.get("uid", None)
                turn_indices = test_batch.batch["turn_indices"].cpu().numpy()

                # Aggregate scores by sample (sum of all turns)
                sample_score_map = {}
                for i in range(len(scores)):
                    s_idx = sample_indices[i]
                    t_idx = int(turn_indices[i])
                    if t_idx == -1:  # Skip padding turns
                        continue
                    if s_idx not in sample_score_map:
                        sample_score_map[s_idx] = 0.0
                    sample_score_map[s_idx] += scores[i]

                # Add aggregated scores in the same UID order used for sample inputs/outputs above
                for s_idx in sorted(sample_score_map.keys()):
                    sample_scores.append(sample_score_map[s_idx])
            else:
                sample_scores.extend(scores)

            # Log multi-turn conversations to JSONL during validation
            if (
                self.config.actor_rollout_ref.rollout.multi_turn.enable
                and self.config.actor_rollout_ref.rollout.multi_turn.rollout_save_jsonl
                is not None
            ):
                self._log_multiturn_to_jsonl(test_output_gen_batch, scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(cur_data_source)

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)

        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # (WARNING) we cannot guarantee len(rewards) is multiple of n when we use async rollout
            if len(rewards) % self.config.actor_rollout_ref.rollout.val_kwargs.n != 0:
                print(
                    "Warning: validation samples not divisible by n, padding with 0.0 rewards"
                )
                # Add some padding rewards with 0.0
                n_missing = self.config.actor_rollout_ref.rollout.val_kwargs.n - (
                    len(rewards) % self.config.actor_rollout_ref.rollout.val_kwargs.n
                )
                rewards.extend([0.0] * n_missing)

            assert (
                len(rewards) % self.config.actor_rollout_ref.rollout.val_kwargs.n == 0
            )
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)
            print(
                f"""Calculating pass@k rate for {data_source} with k={self.config.actor_rollout_ref.rollout.val_kwargs.k}"""
            )
            reward_per_test_sample = np.reshape(
                rewards, (-1, self.config.actor_rollout_ref.rollout.val_kwargs.n)
            )  # [N, n_val]
            pass_at_k_rate = self.compute_pass_at_k(
                reward_per_test_sample,
                k=self.config.actor_rollout_ref.rollout.val_kwargs.k,
            )
            print(f"[{data_source}]pass_at_k_rate:", pass_at_k_rate)
            metric_dict[
                f"val/test_score/{data_source}_pass@{self.config.actor_rollout_ref.rollout.val_kwargs.k}"
            ] = pass_at_k_rate

        if reward_extra_info_dict is not None:
            for key, extra_info_list in reward_extra_info_dict.items():
                # Normalize heterogeneous entries (dicts, tensors, scalars, etc.) to floats when possible
                coerced_values = [
                    _coerce_extra_metric_value(v) for v in extra_info_list
                ]
                valid_values = [v for v in coerced_values if v is not None]

                if valid_values:
                    metric_dict[f"val/test_score_extra/{key}"] = float(
                        np.mean(valid_values)
                    )
                else:
                    metric_dict[f"val/test_score_extra/{key}"] = 0.0

                if not key.startswith("score_"):
                    continue

                if not coerced_values or any(v is None for v in coerced_values):
                    print(
                        f"Skipping pass@k computation for extra metric {key} due to missing/non-numeric values "
                        f"(total={len(coerced_values)})"
                    )
                    continue

                extra_rewards = list(coerced_values)

                n_val = self.config.actor_rollout_ref.rollout.val_kwargs.n
                k_val = self.config.actor_rollout_ref.rollout.val_kwargs.k

                if len(extra_rewards) % n_val != 0:
                    print(
                        f"Warning: extra validation samples for {key} not divisible by n, padding with 0.0 rewards"
                    )
                    n_missing = n_val - (len(extra_rewards) % n_val)
                    extra_rewards.extend([0.0] * n_missing)

                assert len(extra_rewards) % n_val == 0

                print(
                    f"""Calculating pass@k rate for extra metric {key} with k={k_val}"""
                )
                extra_rewards_per_sample = np.reshape(extra_rewards, (-1, n_val))
                extra_pass_at_k_rate = self.compute_pass_at_k(
                    extra_rewards_per_sample, k=k_val
                )
                print(f"[extra:{key}]pass_at_k_rate:", extra_pass_at_k_rate)
                metric_dict[
                    f"val/test_score_extra/{key}_pass@{k_val}"
                ] = extra_pass_at_k_rate

        metric_dict["val/batch/rollout_timeout_samples"] = sum(async_rollout_diffs)

        if self.config.actor_rollout_ref.rollout.multi_turn.enable:
            # Add multi-turn metrics using the compute_multi_turn_metrics function
            multi_turn_metrics = compute_multi_turn_metrics(test_batch)
            for key, val in multi_turn_metrics.items():
                metric_dict[f"val/{key}"] = val

            # Add kernel-specific multi-turn metrics (per-turn and best-by-turn)
            kernel_multi_turn_metrics = compute_kernel_multi_turn_metrics(
                test_batch, prefix="kernel"
            )
            for key, val in kernel_multi_turn_metrics.items():
                metric_dict[f"val/{key}"] = val

        return metric_dict

    def compute_rollout_correction_and_add_to_batch(
        self, batch: DataProto, max_turns: int = 1
    ) -> tuple[DataProto, dict]:
        """Compute IS weights and rejection mask for rollout-training policy mismatch.

        Args:
            batch: DataProto containing required fields:
                - old_log_probs: Log probabilities from training policy
                - rollout_log_probs: Log probabilities from rollout policy
                - response_mask: Original valid token mask (1=valid, 0=padding)
            max_turns: Maximum number of conversation turns (for sequence-level aggregation)

        Returns:
            Tuple of (updated_batch, metrics):
                updated_batch: DataProto with:
                    - Modified `response_mask` (if `rollout_rs` is enabled)
                    - New `rollout_is_weights` (if `rollout_is` is enabled)
                metrics: Dictionary of mismatch/IS/rejection metrics (prefixed with "mismatch/"),
                    empty if rollout data is missing.
        """
        # Skip processing if rollout_log_probs are missing (no mismatch to correct)
        if "rollout_log_probs" not in batch.batch:
            return batch, {}

        # Store original mask for quality analysis
        original_response_mask = batch.batch["response_mask"].clone()

        rollout_rs = self.config.algorithm.get("rollout_rs", None)
        rollout_is = self.config.algorithm.get("rollout_is", None)
        coverage_rs = self.config.reward_model.get("coverage_rs", None)

        # Compute IS weights and rejection mask (for log-prob mismatch)
        (
            rollout_is_weights,
            modified_response_mask,
            rollout_metrics,
        ) = compute_rollout_importance_weights_and_rejection_mask(
            old_log_prob=batch.batch["old_log_probs"],
            rollout_log_prob=batch.batch["rollout_log_probs"],
            response_mask=batch.batch["response_mask"],
            max_turns=max_turns,
            rollout_is=rollout_is,
            rollout_is_threshold=self.config.algorithm.get("rollout_is_kwargs", {}).get(
                "upper", None
            ),
            rollout_rs=rollout_rs,
            rollout_rs_threshold=self.config.algorithm.get("rollout_rs_kwargs", {}).get(
                "upper", None
            ),
            rollout_rs_threshold_lower=self.config.algorithm.get(
                "rollout_rs_kwargs", {}
            ).get("lower", None),
            rollout_token_veto_threshold=self.config.algorithm.get(
                "rollout_token_veto_threshold", None
            ),
        )

        # Apply coverage-based rejection sampling (only for correct samples)
        coverage_metrics = {}
        if coverage_rs is not None and "reward_extra_info" in batch.non_tensor_batch:

            # Collect coverage data from reward_extra_info
            device = batch.batch["response_mask"].device
            batch_size = batch.batch["response_mask"].shape[0] // max_turns

            time_coverage_list = []
            num_coverage_list = []
            correctness_list = []
            performance_list = []

            for info in batch.non_tensor_batch["reward_extra_info"]:
                time_coverage_list.append(info.get("time_coverage", 0.0))
                num_coverage_list.append(info.get("num_coverage", 0.0))
                # Consider both correctness and decoy_kernel for filtering
                is_correct = info.get("correctness", False) and not info.get(
                    "decoy_kernel", False
                )
                correctness_list.append(is_correct)
                performance_list.append(info.get("performance", 0.0))

            # Convert to tensors with proper shapes
            time_coverage = torch.tensor(time_coverage_list, device=device).reshape(
                batch_size, max_turns
            )
            num_coverage = torch.tensor(num_coverage_list, device=device).reshape(
                batch_size, max_turns
            )
            correctness = torch.tensor(
                correctness_list, dtype=torch.bool, device=device
            )  # (batch_size * max_turns,)
            performance = torch.tensor(
                performance_list, dtype=torch.float32, device=device
            )  # (batch_size * max_turns,)


            # Apply coverage-based rejection sampling
            modified_response_mask, coverage_metrics = compute_coverage_rejection_mask(
                time_coverage=time_coverage,
                num_coverage=num_coverage,
                response_mask=modified_response_mask,  # Use already-modified mask from rollout_rs
                correctness=correctness,
                max_turns=max_turns,
                coverage_rs=coverage_rs,
                coverage_rs_key=self.config.reward_model.get(
                    "coverage_rs_key", "time_coverage"
                ),
                coverage_rs_threshold=self.config.reward_model.get(
                    "coverage_rs_threshold", 0.3
                ),
                coverage_rs_factor=self.config.reward_model.get(
                    "coverage_rs_factor", 0.1
                ),
                speedup=performance,
                speedup_threshold=self.config.reward_model.get(
                    "speedup_threshold", None
                ),
            )

        quality_metrics = {}
        # Compute quality metrics for masked samples (only if RS is enabled and reward info available)
        if rollout_rs is not None and "reward_extra_info" in batch.non_tensor_batch:
            from kernel.metrics.mismatch_quality_metrics import (
                compute_mismatch_quality_metrics,
            )

            quality_metrics = compute_mismatch_quality_metrics(
                batch=batch,
                original_response_mask=original_response_mask,
                modified_response_mask=modified_response_mask,
                prefix="mismatch_quality",
            )
            # rollout_metrics.update(quality_metrics)

        if rollout_rs is not None or coverage_rs is not None:
            # Update response_mask if either rollout_rs or coverage_rs is enabled
            # Rejected tokens/sequences are masked to 0 — excluded from loss
            batch.batch["response_mask"] = modified_response_mask
        if rollout_is is not None:
            # Add IS weights to batch only if explicitly enabled (for variance reduction in loss)
            batch.batch["rollout_is_weights"] = rollout_is_weights

        # Merge all metrics
        all_metrics = {**rollout_metrics, **coverage_metrics}
        return batch, all_metrics, quality_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl_patch.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Initialize early stopping variables
        self.best_val_score = None
        self.val_patience_counter = 0
        self.train_metric_history = []
        self.train_metric_drop_counter = 0

        # Get early stopping config - check new structure first, then fall back to old
        early_stop_config = self.config.trainer.get("early_stopping", {})

        # Validation early stopping
        self.val_metric = early_stop_config.get("val_metric")
        self.val_patience = early_stop_config.get("val_patience")
        self.val_mode = early_stop_config.get("val_mode")

        # Training metric early stopping
        self.train_metric = early_stop_config.get("train_metric")
        self.train_threshold = early_stop_config.get("train_threshold")
        self.train_window = early_stop_config.get("train_window")
        self.train_patience = early_stop_config.get("train_patience")

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            # Initialize validation early stopping with first validation score
            if self.val_metric and self.val_patience > 0:
                if self.val_metric in val_metrics:
                    self.best_val_score = val_metrics[self.val_metric]
                    print(
                        f"Validation early stopping: Initial best score for {self.val_metric}: {self.best_val_score}"
                    )
                else:
                    print(
                        f"Warning: Validation early stop metric '{self.val_metric}' not found in initial validation metrics"
                    )

            if self.config.trainer.get("val_only", False):
                return
        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # temp buffer if we skip batch
        buffer_batch = None

        for _ in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # (TODO) Qian: previously we set uid as batch_x_sample_i, but it is not unique if we have buffer.
                batch.non_tensor_batch["uid"] = np.array(
                    [
                        f"batch_{self.global_steps}_example_{uuid4().hex}"
                        for _ in range(len(batch.batch))
                    ],
                    dtype=object,
                )

                metrics["batch/oversampling_batch_size"] = batch.batch[
                    "input_ids"
                ].shape[0]
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "uid"]

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    if "raw_prompt" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("raw_prompt")
                    if "tools_kwargs" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("tools_kwargs")
                    gen_batch = batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )

                    # Apply sample-level oversampling by modifying generation parameters
                    target_n = self.config.actor_rollout_ref.rollout.n
                    sample_oversampling_factor = self.config.data.get(
                        "sample_oversampling_factor", 1.0
                    )
                    actual_n = int(target_n * sample_oversampling_factor)
                    gen_batch.meta_info["n"] = actual_n

                    # Pass the oversampled n value through meta_info
                    if sample_oversampling_factor != 1.0:
                        metrics[
                            "batch/sample_oversampling_factor"
                        ] = sample_oversampling_factor
                        metrics["batch/samples_per_prompt_generated"] = actual_n
                        metrics["batch/samples_per_prompt_target"] = target_n

                is_last_step = self.global_steps >= self.total_training_steps

                # for uid, we assign it back from gen_batch to batch
                batch.non_tensor_batch["uid"] = gen_batch.non_tensor_batch["uid"]

                # record if use multi turn training
                use_multi_turn = self.config.actor_rollout_ref.rollout.multi_turn.enable
                max_turns = (
                    self.config.actor_rollout_ref.rollout.multi_turn.max_user_turns
                    if use_multi_turn
                    else 1
                )

                # Add reward_model to gen_batch if it exists in batch, but keep it in original batch
                # Only add for multi_turn async mode to avoid batch size mismatch
                if "reward_model" in batch.non_tensor_batch and self.async_rollout_mode:
                    gen_batch.non_tensor_batch["reward_model"] = batch.non_tensor_batch[
                        "reward_model"
                    ]
                    gen_batch.non_tensor_batch["data_source"] = batch.non_tensor_batch[
                        "data_source"
                    ]

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # Pass global_step through meta_info for logfire logging
                        gen_batch.meta_info["global_step"] = self.global_steps
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                gen_batch
                            )
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(
                                gen_batch
                            )
                            self.async_rollout_manager.sleep()

                    # check on if generation is empty due to filtering in async mode
                    if gen_batch_output is None or len(gen_batch_output.batch) == 0:
                        # skip this batch
                        print(
                            f"Warning: all prompts in batch {self.global_steps} are filtered during generation, skipping this batch."
                        )
                        continue

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                gen_baseline_batch
                            )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    if not use_multi_turn:
                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n,
                            interleave=True,
                        )
                    else:
                        # For multi-turn, we will expand to rollout_n * current_max_turns
                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n
                            * max_turns,
                            interleave=True,
                        )

                    metrics["batch/rollout_timeout_samples"] = len(batch.batch) - len(
                        gen_batch_output.batch
                    )
                    # if batch is larger than gen_batch_output, which means that some prompts have been filtered out because of async timeout
                    if len(batch.batch) > len(gen_batch_output.batch):
                        # use uid to filter
                        gen_uids = set(gen_batch_output.non_tensor_batch["uid"])
                        batch_mask = np.array(
                            [uid in gen_uids for uid in batch.non_tensor_batch["uid"]]
                        )
                        batch = batch[batch_mask]
                        # guarantee that all uids are equalivant to gen_batch_output's uid
                        assert False not in (
                            batch.non_tensor_batch["uid"]
                            == gen_batch_output.non_tensor_batch["uid"]
                        )
                        # assign gen_batch_output to batch's uid
                        batch.non_tensor_batch[
                            "uid"
                        ] = gen_batch_output.non_tensor_batch["uid"]

                    # Union the generated responses with the batch
                    batch = batch.union(gen_batch_output)

                    # Check if buffer is disabled
                    disable_buffer = self.config.trainer.get("disable_buffer", False)
                    with _timer("buffer_load", timing_raw):
                        batch, buffer_batch = self._load_from_buffer(
                            batch=batch,
                            buffer_batch=buffer_batch,
                            disable_buffer=disable_buffer,
                        )

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if use_multi_turn:
                        # Apply loss mask for batch to avoid computing loss on padded turns
                        apply_loss_mask_to_masks(batch)

                    batch, pad_size = pad_dataproto_to_divisor(
                        batch, self.max_world_size
                    )
                    if pad_size > 0:
                        print(
                            f"[Log Prob Padding] Padded batch with {pad_size} samples to be divisible by world_size={self.max_world_size}"
                        )

                    # Rollout correction mode selection
                    bypass_mode = self.config.algorithm.get(
                        "bypass_old_logprob_for_rollout", False
                    )

                    if bypass_mode:
                        # BYPASS MODE: Use rollout_log_probs as old_log_probs
                        # This skips the expensive actor forward pass for old_log_prob computation
                        #
                        # Two sub-modes (controlled by use_pure_rollout_correction in actor):
                        # 1. PPO_IS mode (use_pure_rollout_correction=False, default):
                        #    - Actor uses standard PPO with old_log_prob=rollout_log_prob
                        #    - PPO clips ratio = π_current / π_rollout (not π_current / π_old)
                        #    - IS correction happens implicitly through the ratio
                        #
                        # 2. Pure rollout correction mode (use_pure_rollout_correction=True):
                        #    - Actor uses compute_policy_loss_with_rollout_correction()
                        #    - Pure policy gradient with IS correction (no PPO clipping)
                        #    - Computes IS/RS on-the-fly in actor

                        batch.meta_info["bypass_old_logprob_for_rollout"] = True

                        if "rollout_log_probs" not in batch.batch:
                            raise ValueError(
                                "bypass_old_logprob_for_rollout=True requires rollout_log_probs in batch. "
                                "Ensure rollout worker is configured to return log probabilities."
                            )

                        # Store temperature for update
                        batch.meta_info[
                            "temperature"
                        ] = self.config.actor_rollout_ref.rollout.temperature

                        # Use rollout log probs as old log probs (zero-cost substitution)
                        batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

                        # Skip trainer-level IS computation (will be done in actor if needed)
                        # Log that we're in bypass mode
                        metrics["rollout_correction/bypass_mode"] = 1.0
                        metrics[
                            "rollout_correction/old_logprob_computation_skipped"
                        ] = 1.0
                    else:
                        # LEGACY MODE: Compute old_log_probs from actor
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        # Compute rollout IS weights ONCE before advantage computation
                        # (optimal_baseline needs them to scale W(τ), others will use them for policy loss)
                        (
                            batch,
                            is_metrics,
                            mismatch_quality_metrics,
                        ) = self.compute_rollout_correction_and_add_to_batch(
                            batch, max_turns
                        )
                        metrics.update(is_metrics)
                        metrics.update(mismatch_quality_metrics)

                    # If we have reference log prob, it should be computed on the whole multi-turn batch
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            with _timer("reward_model", timing_raw):
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                        with _timer("reward_fn", timing_raw):
                            # we combine with rule-based rm
                            extra_rewards_info = None

                            use_final_reward = self.config.algorithm.get(
                                "use_final_reward", True
                            )
                            # if reward already computed in async loop, skip recomputation
                            if "token_level_scores" not in batch.batch.keys():
                                if use_final_reward and use_multi_turn:
                                    last_turn_batch = get_last_turn_data(
                                        batch, max_turns=max_turns
                                    )
                                    reward_result = self.reward_fn(last_turn_batch)
                                else:
                                    reward_result = self.reward_fn(batch)
                                if isinstance(reward_result, dict):
                                    token_level_scores = reward_result["reward_tensor"]
                                    if "extra_info" in reward_result:
                                        extra_rewards_info = reward_result["extra_info"]
                                else:
                                    token_level_scores = reward_result
                                # Apply final reward logic if enabled
                                if use_final_reward and use_multi_turn:
                                    # Expand scores to all turns but only keep last turn reward
                                    token_level_scores = token_level_scores.unsqueeze(
                                        1
                                    ).repeat(1, max_turns, 1)
                                    token_level_scores[:, :-1, :] = 0.0
                                    token_level_scores = token_level_scores.reshape(
                                        -1, token_level_scores.shape[-1]
                                    )
                                batch.batch["token_level_scores"] = token_level_scores
                            else:
                                if use_final_reward and use_multi_turn:
                                    token_level_scores = batch.batch.pop(
                                        "token_level_scores"
                                    )
                                    # Expand scores to all turns but only keep last turn reward
                                    token_level_scores = token_level_scores.reshape(
                                        -1, max_turns, token_level_scores.shape[-1]
                                    )
                                    token_level_scores[:, :-1, :] = 0.0
                                    token_level_scores = token_level_scores.reshape(
                                        -1, token_level_scores.shape[-1]
                                    )
                                    batch.batch[
                                        "token_level_scores"
                                    ] = token_level_scores
                                reward_extra_info_list = batch.non_tensor_batch[
                                    "reward_extra_info"
                                ].tolist()
                                # Convert list of dict to dict of list, filtering out None values and non-numeric values
                                extra_rewards_info = {}
                                if len(reward_extra_info_list) > 0:
                                    # Collect all keys from all dicts
                                    all_keys = set()
                                    for d in reward_extra_info_list:
                                        if len(d) > 0:
                                            all_keys.update(d.keys())

                                    # Build dict of lists, filtering None values, missing keys, and non-numeric values
                                    for k in all_keys:
                                        values = []
                                        for d in reward_extra_info_list:
                                            if (
                                                len(d) > 0
                                                and k in d
                                                and d[k] is not None
                                            ):
                                                # Only add numeric values (int, float, bool)
                                                # Skip string fields like 'error', 'finish_type', etc.
                                                if isinstance(
                                                    d[k], (int, float, bool, np.number)
                                                ):
                                                    values.append(d[k])
                                        if (
                                            values
                                        ):  # Only add key if there are valid numeric values
                                            extra_rewards_info[k] = values

                        # (TODO) Qian: we should be careful about here, to avoid "void turn" still getting non-zero rewards, we mask them here to zero
                        apply_loss_mask_to_rewards(batch)

                        # Extract last turn for filter stats update
                        if use_multi_turn:
                            # Keep a reference to the full multi-turn batch for later usage
                            multi_turn_batch = batch
                            # override batch with the last turn data only
                            if self.config.algorithm.is_get_last_turn:
                                batch = get_last_turn_data(
                                    multi_turn_batch, max_turns=max_turns
                                )
                            else:
                                batch = multi_turn_batch
                        else:
                            multi_turn_batch = None

                        prompt_filter_stats = None
                        if self.batch_filter is not None and (
                            self.config.trainer.rejection_sample
                            or self.config.trainer.remove_clip
                            or self.config.get("rejection_sampling", {}).get(
                                "enable_two_gate_filter", False
                            )
                            or self.config.data.get("prompt_oversampling_factor", 1.0)
                            > 1.0
                            or self.config.data.get("sample_oversampling_factor", 1.0)
                            > 1.0
                        ):
                            # Prepare batch data for the oversampling filter
                            filter_batch_data = {
                                "rewards": batch.batch["token_level_scores"].sum(-1),
                                "response_mask": batch.batch["response_mask"],
                            }

                            # Add optional data for two-gate filter if present
                            if "old_log_probs" in batch.batch:
                                filter_batch_data["old_log_probs"] = batch.batch[
                                    "old_log_probs"
                                ]
                            if "rollout_log_probs" in batch.batch:
                                filter_batch_data["rollout_log_probs"] = batch.batch[
                                    "rollout_log_probs"
                                ]
                            if "top_log_probs" in batch.batch:
                                filter_batch_data["top_log_probs"] = batch.batch[
                                    "top_log_probs"
                                ]
                            if "prompt_index" in batch.non_tensor_batch:
                                filter_batch_data[
                                    "prompt_index"
                                ] = batch.non_tensor_batch["prompt_index"]

                            # Get UIDs for group tracking; include turn_id when using multi-turn advantages
                            if (
                                use_multi_turn
                                and not self.config.algorithm.is_get_last_turn
                            ):
                                uids_list = [
                                    f"{uid}_t{int(turn)}"
                                    for uid, turn in zip(
                                        batch.non_tensor_batch["uid"],
                                        batch.batch["turn_indices"],
                                    )
                                ]
                            else:
                                uids_list = (
                                    batch.non_tensor_batch["uid"].tolist()
                                    if hasattr(batch.non_tensor_batch["uid"], "tolist")
                                    else list(batch.non_tensor_batch["uid"])
                                )

                            # Apply filtering and selection through the unified filter
                            with _timer("filter", timing_raw):
                                (
                                    selected_indices,
                                    filter_metrics,
                                ) = self.batch_filter.filter_batch(
                                    filter_batch_data,
                                    uids_list,
                                    global_step=self.global_steps,
                                    return_indices=True,  # Get indices to filter main batch
                                )

                            # CRITICAL FIX: Store filter statistics for later update (only if batch is used for training)
                            # This prevents poisoning DynamicBatchSampler with statistics from skipped batches
                            #
                            # Previous bug: Filter stats were updated BEFORE checking if batch would be skipped,
                            # causing a feedback loop where high filter rates from skipped batches led to
                            # excessive oversampling in subsequent iterations.
                            #
                            # Now: We save the stats here and only update DynamicBatchSampler after confirming
                            # the batch will be used for training (see line ~1484)
                            if (
                                hasattr(self, "dynamic_batch_sampler")
                                and "prompt_filter_stats" in filter_metrics
                            ):
                                # Save the statistics but don't update yet
                                prompt_filter_stats = filter_metrics[
                                    "prompt_filter_stats"
                                ]

                            if "prompt_filter_stats" in filter_metrics:
                                # Remove prompt_filter_stats from filter_metrics BEFORE updating metrics
                                # to avoid passing numpy.int64 keys to WandB
                                del filter_metrics["prompt_filter_stats"]

                            # Update metrics with filter statistics (after removing prompt_filter_stats)
                            metrics.update(filter_metrics)

                            # Reorder selected indices to maintain original batch order
                            # (TODO) Qian: this is very important for multi-turn broadcasting since it assumes the order is not changed
                            selected_indices = sorted(selected_indices)

                            # Calculate the size of the valid batch after filtering
                            valid_query_size = (
                                len(selected_indices)
                                // self.config.actor_rollout_ref.rollout.n
                            )

                            if (
                                use_multi_turn
                                and not self.config.algorithm.is_get_last_turn
                            ):
                                valid_query_size = valid_query_size // max_turns

                            assert (
                                valid_query_size <= self.config.data.train_batch_size
                            ), "The valid batch size after filtering should not exceed the expected batch size"

                            # Apply selection to the main batch
                            selected_indices_tensor = torch.as_tensor(
                                selected_indices, dtype=torch.long
                            )
                            batch = batch.select_idxs(selected_indices_tensor)

                            # Handle multi-turn batch filtering based on is_get_last_turn mode
                            if use_multi_turn and multi_turn_batch is not None:
                                if self.config.algorithm.is_get_last_turn:
                                    # When is_get_last_turn=True:
                                    # - batch is last-turn-only (size N)
                                    # - selected_indices are in range [0, N)
                                    # - Need to expand indices to select corresponding turns from multi_turn_batch (size N * max_turns)
                                    turn_offsets = torch.arange(
                                        max_turns, device=selected_indices_tensor.device
                                    )
                                    multi_turn_indices = (
                                        selected_indices_tensor.unsqueeze(1) * max_turns
                                        + turn_offsets
                                    ).reshape(-1)
                                    multi_turn_batch = multi_turn_batch.select_idxs(
                                        multi_turn_indices
                                    )
                                else:
                                    # When is_get_last_turn=False:
                                    # - batch IS multi_turn_batch (size N * max_turns)
                                    # - selected_indices are already in range [0, N * max_turns)
                                    # - batch.select_idxs() already filtered, just assign to multi_turn_batch
                                    multi_turn_batch = batch

                        # Calculate expected size for final training batch, no whether filtering is applied
                        expected_input_ids_size = (
                            self.config.actor_rollout_ref.rollout.n
                            * self.config.data.train_batch_size
                        )

                        if (
                            use_multi_turn
                            and not self.config.algorithm.is_get_last_turn
                        ):
                            print(
                                f"use_multi_turn and not is_get_last_turn, expected_input_ids_size: {expected_input_ids_size}"
                            )
                            expected_input_ids_size = (
                                expected_input_ids_size * max_turns
                            )

                        actual_input_ids_size = batch.batch["input_ids"].shape[0]

                        if actual_input_ids_size == 0:
                            raise RuntimeError(
                                "No valid samples were selected after filtering. "
                                "Increase rollout number to ensure that there are valid examples for training."
                            )
                        elif actual_input_ids_size < expected_input_ids_size:
                            current_sample_factor = float(
                                self.config.data.get("prompt_oversampling_factor", 1.0)
                            )
                            suggested_factor = (
                                self._compute_suggested_sample_factor(
                                    expected_input_ids_size, actual_input_ids_size
                                )
                                or current_sample_factor
                            )
                            skip_metrics = {
                                "over_sampling/suggested_min_oversample_factor": suggested_factor,
                                "over_sampling/current_sample_oversample_factor": current_sample_factor,
                            }
                            logger.log(data=skip_metrics, step=self.global_steps)
                            print(
                                f"[Oversampling] Selected {actual_input_ids_size} of {expected_input_ids_size} required samples. "
                                f"Suggested minimum prompt_oversampling_factor: {suggested_factor:.2f} "
                                f"(current {current_sample_factor})."
                            )
                            with _timer("buffer_save", timing_raw):
                                buffer_batch = self._save_to_buffer(
                                    buffer_batch=buffer_batch,
                                    batch=batch,
                                    disable_buffer=disable_buffer,
                                    multi_turn_batch=multi_turn_batch,
                                    use_multi_turn=use_multi_turn,
                                )
                            continue

                        # Update DynamicBatchSampler ONLY for batches we actually train on
                        # This prevents skipped batches from poisoning the filter statistics
                        skip_steps = self.config.trainer.get(
                            "automatic_oversampling_skip_steps", 0
                        )
                        if (
                            prompt_filter_stats is not None
                            and hasattr(self, "dynamic_batch_sampler")
                            and self.global_steps > skip_steps
                        ):
                            # Update batch sampler with exact per-prompt statistics
                            # Format: prompt_filter_stats[prompt_idx] =
                            #         {'before': total_groups, 'after': valid_groups, 'selected': used_groups}
                            self.dynamic_batch_sampler.update_filter_stats(
                                prompt_filter_stats
                            )
                            # Get and record sampler metrics
                            sampler_metrics = self.dynamic_batch_sampler.get_metrics()
                            metrics.update(sampler_metrics)

                        # Update other sampler if available
                        self._update_sampler_states(batch, metrics)

                        # Restore multi_turn_batch if use_multi_turn is True
                        if use_multi_turn:
                            batch = multi_turn_batch

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            with _timer("kl_penalty", timing_raw):
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        apply_loss_mask_to_rewards(batch)

                        batch.meta_info[
                            "optimal_baseline_kwargs"
                        ] = self.config.algorithm.get("optimal_baseline_kwargs", {})
                        # (TODO) Qian: old_log_prob must be computed before advantage computation for optimal baseline setup
                        with _timer("compute_adv", timing_raw):
                            adv_kwargs = dict(
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                batch_std=self.config.algorithm.get("batch_std", False),
                                use_multi_prompt_mvu=self.config.algorithm.get(
                                    "use_multi_prompt_mvu", False
                                ),
                                reward_shaping=self.config.algorithm.get(
                                    "reward_shaping", False
                                ),
                                unbiased_shaping=self.config.algorithm.get(
                                    "unbiased_shaping", False
                                ),
                            )
                            if self.config.algorithm.get("adv_by_last_turn", True):
                                print("[ADV] Using last turn to compute advantages")
                                last_turn_batch = (
                                    get_last_turn_data(batch, max_turns=max_turns)
                                    if use_multi_turn
                                    else batch
                                )
                                last_turn_batch = compute_advantage(
                                    last_turn_batch, **adv_kwargs
                                )
                                if use_multi_turn:
                                    # Filter the multi_turn_batch whose uid is in the current batch
                                    valid_uids = set(
                                        last_turn_batch.non_tensor_batch["uid"]
                                    )
                                    batch = batch[
                                        [
                                            uid in valid_uids
                                            for uid in batch.non_tensor_batch["uid"]
                                        ]
                                    ]
                                    batch = broadcast_last_turn_to_multi_turn(
                                        last_turn_data=last_turn_batch,
                                        multi_turn_data=batch,
                                        max_turns=max_turns,
                                    )
                                else:
                                    batch = last_turn_batch
                            else:
                                print("[ADV] Using multi turn to compute advantages")
                                batch = compute_multi_turn_advantage(
                                    batch, max_turns, **adv_kwargs,
                                )

                        batch = compute_rloo_advantages_for_metric_computation(
                            batch, max_turns
                        )

                        if use_multi_turn:
                            # Compute multi-turn metrics before filtering
                            multi_turn_metrics = compute_multi_turn_metrics(batch)
                            metrics.update(multi_turn_metrics)

                        # If the entire sample is masked, we remove it from the batch, use sum of response_mask to determine
                        masked_examples = batch.batch["response_mask"].sum(dim=1) == 0
                        # Get indices of samples to remove
                        remove_indices = torch.nonzero(
                            masked_examples, as_tuple=False
                        ).squeeze(1)

                        if len(remove_indices) > 0:
                            keep_indices = torch.tensor(
                                [
                                    i
                                    for i in range(len(batch.batch["input_ids"]))
                                    if i not in remove_indices
                                ],
                                dtype=torch.long,
                            )
                            originl_len = len(batch)
                            batch = batch.select_idxs(keep_indices)
                            print(
                                f"Filtered batch: {originl_len} -> {len(batch)} examples"
                            )

                        # Pad by duplicating first N samples to make batch % max_world_size == 0
                        # Duplicated samples contribute same gradients (equivalent to increased sample weight)
                        batch, pad_size = pad_dataproto_to_divisor(
                            batch, self.max_world_size
                        )
                        if pad_size > 0:
                            print(
                                f"[Example Removal Padding] Padded batch with {pad_size} samples to be divisible by world_size={self.max_world_size}"
                            )

                        # NOTE: Qian: we must move balance batch to later to guarantee the correctness of the batch order
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                    # Rollout correction mode selection
                    if self.config.algorithm.get("use_pure_rollout_correction", False):
                        rollout_correction_kwargs = {
                            "rollout_is": self.config.algorithm.get("rollout_is", None),
                            "rollout_is_kwargs": self.config.algorithm.get(
                                "rollout_is_kwargs", {}
                            ),
                            "rollout_rs": self.config.algorithm.get("rollout_rs", None),
                            "rollout_rs_kwargs": self.config.algorithm.get(
                                "rollout_rs_kwargs", {}
                            ),
                            "rollout_token_veto_threshold": self.config.algorithm.get(
                                "rollout_token_veto_threshold", None
                            ),
                        }
                        batch.meta_info[
                            "rollout_correction_kwargs"
                        ] = rollout_correction_kwargs
                        batch.meta_info["max_turns"] = max_turns
                        batch.meta_info["use_pure_rollout_correction"] = True
                        batch.meta_info["bypass_old_logprob_for_rollout"] = True

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # compute global_valid tokens after filtering
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    batch.meta_info["global_response_token_num"] = torch.sum(
                        batch.batch["response_mask"], dim=-1
                    ).tolist()

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Rollout IS weights already computed before advantage computation (line 1584)

                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                n_gpus = self.resource_pool_manager.get_n_gpus()

                # Extract gradient norm from actor metrics if available
                batch_gradient_norm_squared = None
                if "actor/grad_norm" in metrics:
                    # The gradient norm from dp_actor is already the L2 norm
                    # We need to square it for the variance proxy computations
                    grad_norm = metrics["actor/grad_norm"]
                    batch_gradient_norm_squared = grad_norm ** 2

                all_metrics = compute_all_training_metrics(
                    batch=batch,
                    use_critic=self.use_critic,
                    extra_rewards_info=extra_rewards_info,
                    timing_raw=timing_raw,
                    n_gpus=n_gpus,
                    global_step=self.global_steps,
                    batch_gradient_norm_squared=batch_gradient_norm_squared,
                )
                metrics.update(all_metrics)

                # Compute kernel-specific multi-turn training metrics
                if use_multi_turn:
                    kernel_training_multi_turn_metrics = compute_kernel_multi_turn_metrics(
                        batch, prefix="kernel"
                    )
                    for key, val in kernel_training_multi_turn_metrics.items():
                        metrics[f"train/{key}"] = val

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Check training metric early stopping
                if self.train_metric:
                    if self.train_metric in metrics:
                        current_value = metrics[self.train_metric]
                        self.train_metric_history.append(current_value)

                        # Only check after we have enough history
                        if len(self.train_metric_history) >= self.train_window:
                            # Keep only the most recent window
                            if len(self.train_metric_history) > self.train_window:
                                self.train_metric_history.pop(0)

                            # Calculate moving average
                            moving_avg = sum(self.train_metric_history) / len(
                                self.train_metric_history
                            )

                            # Check for sudden drop (compare current value to moving average)
                            drop_ratio = (
                                (moving_avg - current_value) / moving_avg
                                if moving_avg > 0
                                else 0
                            )

                            if drop_ratio > self.train_threshold:
                                self.train_metric_drop_counter += 1
                                print(
                                    f"Training metric {self.train_metric} dropped by {drop_ratio:.2%} "
                                    f"(current: {current_value:.4f}, moving avg: {moving_avg:.4f}). "
                                    f"Drop counter: {self.train_metric_drop_counter}/{self.train_patience}"
                                )

                                if (
                                    self.train_metric_drop_counter
                                    >= self.train_patience
                                ):
                                    print(
                                        f"Training metric early stopping triggered! "
                                        f"{self.train_metric} has dropped significantly for "
                                        f"{self.train_patience} consecutive steps."
                                    )
                                    progress_bar.close()
                                    print(
                                        f"Early stopping: metric={self.train_metric}, best_score={moving_avg}, patience={self.train_patience}, step={self.global_steps}"
                                    )
                                    return
                            else:
                                # Reset counter if no significant drop
                                if self.train_metric_drop_counter > 0:
                                    print(
                                        f"Training metric {self.train_metric} recovered. Resetting drop counter."
                                    )
                                self.train_metric_drop_counter = 0
                    else:
                        print(
                            f"Warning: Training early stop metric '{self.train_metric}' not found in training metrics"
                        )

                # Check for validation early stopping
                if (
                    self.val_metric
                    and self.val_patience > 0
                    and self.val_metric in metrics
                ):
                    current_score = metrics[self.val_metric]

                    # Check if this is the best score so far
                    if self.best_val_score is None:
                        self.best_val_score = current_score
                        self.val_patience_counter = 0
                        print(
                            f"Validation early stopping: Initial best score for {self.val_metric}: {self.best_val_score}"
                        )
                    else:
                        # Check if score improved
                        improved = False
                        if self.val_mode == "max":
                            improved = current_score > self.best_val_score
                        else:  # mode == 'min'
                            improved = current_score < self.best_val_score

                        if improved:
                            self.best_val_score = current_score
                            self.val_patience_counter = 0
                            print(
                                f"Validation early stopping: New best score for {self.val_metric}: {self.best_val_score}"
                            )
                        else:
                            self.val_patience_counter += 1
                            print(
                                f"Validation early stopping: No improvement for {self.val_patience_counter}/{self.val_patience} validations"
                            )

                        # Check if we should stop
                        if self.val_patience_counter >= self.val_patience:
                            print(
                                f"Validation early stopping triggered! Best score for {self.val_metric}: {self.best_val_score}"
                            )
                            progress_bar.close()
                            print(
                                f"Early stopping: metric={self.val_metric}, best_score={self.best_val_score}, patience={self.val_patience}, step={self.global_steps}"
                            )
                            return

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

        progress_bar.close()

    def _log_multiturn_to_jsonl(self, gen_batch_output: DataProto, scores: list):
        """Log multi-turn conversations to JSONL file.

        Args:
            gen_batch_output: DataProto containing multi-turn conversation data
            scores: Reward scores for each row (flattened format)
        """
        # Get the save path from config
        rollout_save_jsonl = (
            self.config.actor_rollout_ref.rollout.multi_turn.rollout_save_jsonl
        )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(rollout_save_jsonl), exist_ok=True)

        # Get sample and turn indices
        # sample_indices = gen_batch_output.batch.get('sample_indices', None)
        # if sample_indices is not None:
        #     sample_indices = sample_indices.cpu().numpy()
        turn_indices = gen_batch_output.batch["turn_indices"].cpu().numpy()
        uids = gen_batch_output.non_tensor_batch.get("uid", None)
        multiturn_messages = gen_batch_output.non_tensor_batch.get(
            "multiturn_messages", None
        )

        # Find first turn (with messages) for each sample and aggregate scores
        # Use uid as sample_id for uniqueness across batches
        # sample_indices are batch-local (reset to 0 in each batch), so they repeat when batches are concatenated
        sample_data = (
            {}
        )  # sample_id -> {'first_idx': row_idx, 'num_turns': int, 'total_score': float}
        for i in range(len(gen_batch_output.batch)):
            # Get sample identifier - prioritize uid over sample_indices
            if uids is not None:
                s_idx = uids[i]
            else:
                raise ValueError("uids is None")
            # elif sample_indices is not None:
            #     s_idx = int(sample_indices[i])
            # else:
            #     s_idx = i

            t_idx = int(turn_indices[i])

            if t_idx == -1:  # Skip padding turns
                continue

            if s_idx not in sample_data:
                sample_data[s_idx] = {
                    "first_idx": i,
                    "num_turns": t_idx,
                    "total_score": scores[i],
                }
            else:
                sample_data[s_idx]["total_score"] += scores[i]
                if t_idx > sample_data[s_idx]["num_turns"]:
                    sample_data[s_idx]["num_turns"] = t_idx

        # Write to JSONL file
        with open(rollout_save_jsonl, "a", encoding="utf-8") as f:
            for s_idx in sorted(sample_data.keys()):
                data = sample_data[s_idx]
                first_idx = data["first_idx"]

                # Use multiturn_messages if available
                if (
                    multiturn_messages is not None
                    and multiturn_messages[first_idx] is not None
                ):
                    messages = multiturn_messages[first_idx]
                    entry = {
                        "messages": messages,
                        "score": data["total_score"],
                        "num_turns": data["num_turns"],
                    }
                else:
                    # Fallback to prompt + response
                    prompt = self.tokenizer.decode(
                        gen_batch_output.batch["prompts"][first_idx],
                        skip_special_tokens=True,
                    )
                    response = self.tokenizer.decode(
                        gen_batch_output.batch["responses"][first_idx],
                        skip_special_tokens=True,
                    )
                    entry = {
                        "conversation": prompt + response,
                        "score": data["total_score"],
                        "num_turns": data["num_turns"],
                    }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
