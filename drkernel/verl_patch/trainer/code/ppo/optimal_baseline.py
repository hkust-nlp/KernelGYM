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
Optimal baseline advantage estimation for minimum variance REINFORCE.

This module implements the optimal baseline that minimizes gradient variance
using a score-norm proxy W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²].
"""

from collections import defaultdict
from typing import Tuple

import numpy as np
import torch


def compute_optimal_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-8,
    rollout_is_weights: torch.Tensor = None,
    uniform_weight: bool = False,
    uniform_cumulative: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using optimal baseline for minimum variance REINFORCE.

    The optimal baseline minimizes gradient variance by using a score-norm proxy W(τ)
    that approximates ||∇log π||² (the squared gradient norm of the score function).

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        epsilon: Small constant for numerical stability (default: 1e-8)
        rollout_is_weights: Pre-computed IS weights for W(τ) correction [shape: (bs, response_length)], None if not using IS
        uniform_weight: If True, use w_per_timestep = 1 instead of variance proxy.
            Results in W(τ) = length (length-proportional weighting). Default: False
        uniform_cumulative: If True, use w_per_timestep = 1/length instead of variance proxy.
            Results in W(τ) = 1 (length-independent weighting). Default: False

    Returns:
        advantages: Advantage estimates for each token [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W(τ) is scaled by ρ̄² to minimize MSE
        under truncated IS. The optimal baseline formula becomes:
            b* = Σ[R(τ) × ρ̄²(τ) × W(τ)] / Σ[ρ̄²(τ) × W(τ)]
        IS weights are pre-computed in ray_trainer to avoid redundant computation.
    """
    with torch.no_grad():
        # Compute returns (cumulative rewards)
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Extract total rewards for each trajectory (used in baseline computation)
        rewards = returns[:, 0]  # First position contains cumulative reward

        # Step 1: Compute w_per_timestep based on ablation mode
        # Formula: W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
        if uniform_weight:
            # Ablation: uniform weights (w_per_timestep = 1)
            # Results in W(τ) = length (length-proportional weighting)
            w_per_timestep = torch.ones_like(old_log_probs)
        elif uniform_cumulative:
            # Ablation: normalized uniform weights (w_per_timestep = 1/length)
            # Results in W(τ) = 1 (length-independent weighting)
            seq_lengths = response_mask.sum(dim=-1, keepdim=True)  # [batch, 1]
            w_per_timestep = torch.ones_like(old_log_probs) / seq_lengths
        else:
            # Standard: use variance proxy (w_per_timestep = 1 - 2π_t + Σπ²)
            pi_t = torch.exp(old_log_probs)
            w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by (IS weight)² to minimize MSE under truncated IS
            # This implements the optimal baseline for truncated importance sampling:
            # b* = Σ[R × ρ̄² × W] / Σ[ρ̄² × W], where ρ̄ = min(π_train/π_rollout, threshold)
            # Weights are pre-computed and already detached
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Sum across timesteps to get W(τ)
        w_values = (w_per_timestep * response_mask).sum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        batch_size = rewards.shape[0]
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(rewards)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=rewards.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = w_values[traj_idx]
            R_group = rewards[traj_idx]

            # Compute optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum() / (w_group.sum() + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

        # Expand baselines to match token dimension
        # Baseline is constant across all tokens in a trajectory
        baselines_expanded = baselines.unsqueeze(-1).expand_as(returns)

        # Compute advantages and mask out invalid positions
        advantages = (returns - baselines_expanded) * response_mask

    return advantages, returns


def compute_optimal_baseline_step_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-8,
    rollout_is_weights: torch.Tensor = None,
    uniform_weight: bool = False,
    uniform_cumulative: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Path-Weighted Reward Baseline (PWRB) with per-timestep baselines.

    Unlike the outcome baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory (from PWRB paper):
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures how "surprising" the trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        epsilon: Small constant for numerical stability (default: 1e-8)
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)], None if not using IS
        uniform_weight: If True, use w_per_timestep = 1 instead of variance proxy.
            Results in w_cumulative = cumsum(1) = t (grows linearly with timestep). Default: False
        uniform_cumulative: If True, use w_per_timestep = 1/min(t, length) instead of variance proxy.
            Results in w_cumulative = 1 at each timestep (constant weighting). Default: False

    Returns:
        advantages: PWRB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep based on ablation mode
        # Formula: ||s_t||² = 1 - 2π_t + Σπ²
        if uniform_weight:
            # Ablation: uniform weights (w_per_timestep = 1)
            # Results in w_cumulative = cumsum(1) = timestep index
            w_per_timestep = torch.ones_like(old_log_probs)
        elif uniform_cumulative:
            # Ablation: w_per_timestep = 1/min(t, length) so that w_cumulative = 1 at each timestep
            # Create position indices: [1, 2, 3, ..., seq_len]
            positions = (
                torch.arange(1, seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
            )  # [batch, seq_len]
            # Clip by actual sequence length
            seq_lengths = response_mask.sum(dim=-1, keepdim=True)  # [batch, 1]
            effective_positions = torch.minimum(positions.float(), seq_lengths)  # [batch, seq_len]
            # w_per_timestep = 1 / min(t, length)
            w_per_timestep = 1.0 / effective_positions
        else:
            # Standard: use variance proxy (w_per_timestep = 1 - 2π_t + Σπ²)
            pi_t = torch.exp(old_log_probs)
            w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


def compute_token_level_optimal_baseline_outcome_advantage(
    token_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    loss_mask: torch.Tensor,
    turn_indices: torch.Tensor,
    index: np.ndarray,
    max_turns: int,
    epsilon: float = 1e-8,
    rollout_is_weights: torch.Tensor = None,
    uniform_weight: bool = False,
    uniform_cumulative: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using optimal baseline for minimum variance REINFORCE.

    The optimal baseline minimizes gradient variance by using a score-norm proxy W(τ)
    that approximates ||∇log π||² (the squared gradient norm of the score function).

    Args:
        token_rewards: Rewards at each token position [shape: (bs * n * turn, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs * n * turn, response_length)]
        old_log_probs: Log probabilities from FSDP model during generation [shape: (bs * n * turn, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs * n * turn, response_length)]
        loss_mask: Binary mask for valid turns (1) vs padding (0) [shape: (bs * n * turn,)]
        turn_indices: Turn indices for grouping trajectories from same prompt [shape: (bs * n * turn,)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs * n * turn,)]
        max_turns: Maximum number of turns
        epsilon: Small constant for numerical stability (default: 1e-8)
        rollout_is_weights: Pre-computed IS weights for W(τ) correction [shape: (bs * n * turn, response_length)], None if not using IS
        uniform_weight: If True, use w_per_timestep = 1 instead of variance proxy.
            Results in W(τ) = length (length-proportional weighting). Default: False
        uniform_cumulative: If True, use w_per_timestep = 1/length instead of variance proxy.
            Results in W(τ) = 1 (length-independent weighting). Default: False

    Returns:
        advantages: Advantage estimates for each token [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    vLLM Importance Sampling Correction:
        When vLLM generates trajectories but FSDP computes gradients, we correct for the
        distribution mismatch. The importance weight is:
            ρ = π_FSDP(a|s) / π_vLLM(a|s) = exp(log_π_FSDP - log_π_vLLM)

        This weight is truncated to prevent instability:
            ρ̄ = min(ρ, threshold)

        The W score is then scaled by ρ̄² because this minimizes the MSE of the
        gradient estimator under truncated importance sampling. This follows from the
        optimal baseline theory for biased estimators:
            b* = Σ[R(τ) × ρ̄²(τ) × W(τ)] / Σ[ρ̄²(τ) × W(τ)]
        where ρ̄(τ) is the truncated IS ratio: ρ̄ = min(π_FSDP/π_vLLM, threshold)
    """
    with torch.no_grad():

        # Compute W(τ): score-norm proxy that approximates ||∇log π||²
        # Formula: W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
        if uniform_weight:
            # Ablation: uniform weights (w_per_timestep = 1)
            # Results in W(τ) = length (length-proportional weighting)
            w_per_timestep = torch.ones_like(old_log_probs)
        elif uniform_cumulative:
            # Ablation: normalized uniform weights (w_per_timestep = 1/length)
            # Results in W(τ) = 1 (length-independent weighting)
            seq_lengths = response_mask.sum(dim=-1, keepdim=True)  # [batch, 1]
            w_per_timestep = torch.ones_like(old_log_probs) / seq_lengths
        else:
            # Standard: use variance proxy (w_per_timestep = 1 - 2π_t + Σπ²)
            pi_t = torch.exp(old_log_probs)
            w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Apply rollout importance sampling correction when pre-computed weights provided
        if rollout_is_weights is not None:
            # Scale W by (IS weight)² to minimize MSE under truncated IS
            # This implements the optimal baseline for truncated importance sampling:
            # b* = Σ[R × ρ̄² × W] / Σ[ρ̄² × W], where ρ̄ = min(π_train/π_rollout, threshold)
            # Weights are pre-computed and already detached
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        response_length = response_mask.shape[-1]
        w_per_timestep = (w_per_timestep * response_mask) * loss_mask.unsqueeze(
            -1
        )  # [shape: (bs * n * turn, response_length)]
        w_per_timestep = w_per_timestep.reshape(
            -1, max_turns * response_length
        )  # [shape: (bs * n, turn * response_length)]

        # Compute cumulative sum of W(τ) over token level for each trajectory
        w_values = torch.cumsum(w_per_timestep, dim=-1)  # [shape: (bs * n, turn * response_length)]
        w_values = w_values.reshape(-1, response_length)  # [shape: (bs * n * turn, response_length)]
        w_values = (w_values * response_mask) * loss_mask.unsqueeze(-1)  # [shape: (bs * n * turn, response_length)]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        batch_size = token_rewards.shape[0]
        for i in range(batch_size):
            if turn_indices[i].item() == -1 or not loss_mask[i]:
                continue
            idx = (index[i], turn_indices[i].item())
            prompt_groups[idx].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(token_rewards)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=token_rewards.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = w_values[traj_idx]  # [shape: (N, response_length)]
            R_group = token_rewards[traj_idx]  # [shape: (N, response_length)]

            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

        # Compute advantages
        advantages = token_rewards - baselines
        advantages = (advantages * response_mask) * loss_mask.unsqueeze(-1)  # [shape: (bs * n * turn, response_length)]

    return advantages, token_rewards


def compute_turn_level_optimal_baseline_outcome_advantage(
    turn_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    loss_mask: torch.Tensor,
    turn_indices: torch.Tensor,
    index: np.ndarray,
    max_turns: int,
    epsilon: float = 1e-8,
    rollout_is_weights: torch.Tensor = None,
    uniform_weight: bool = False,
    uniform_cumulative: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using optimal baseline for minimum variance REINFORCE.

    The optimal baseline minimizes gradient variance by using a score-norm proxy W(τ)
    that approximates ||∇log π||² (the squared gradient norm of the score function).

    Args:
        turn_rewards: Rewards at each token position [shape: (bs * n * turn,)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs * n * turn, response_length)]
        old_log_probs: Log probabilities from FSDP model during generation [shape: (bs * n * turn, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs * n * turn, response_length)]
        loss_mask: Binary mask for valid turns (1) vs padding (0) [shape: (bs * n * turn,)]
        turn_indices: Turn indices for grouping trajectories from same prompt [shape: (bs * n * turn,)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs * n * turn,)]
        max_turns: Maximum number of turns
        epsilon: Small constant for numerical stability (default: 1e-8)
        rollout_is_weights: Pre-computed IS weights for W(τ) correction [shape: (bs * n * turn, response_length)], None if not using IS
        uniform_weight: If True, use w_per_timestep = 1 instead of variance proxy.
            Results in W(τ) = length (length-proportional weighting). Default: False
        uniform_cumulative: If True, use w_per_timestep = 1/length instead of variance proxy.
            Results in W(τ) = 1 (length-independent weighting). Default: False

    Returns:
        advantages: Advantage estimates for each token [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    vLLM Importance Sampling Correction:
        When vLLM generates trajectories but FSDP computes gradients, we correct for the
        distribution mismatch. The importance weight is:
            ρ = π_FSDP(a|s) / π_vLLM(a|s) = exp(log_π_FSDP - log_π_vLLM)

        This weight is truncated to prevent instability:
            ρ̄ = min(ρ, threshold)

        The W score is then scaled by ρ̄² because this minimizes the MSE of the
        gradient estimator under truncated importance sampling. This follows from the
        optimal baseline theory for biased estimators:
            b* = Σ[R(τ) × ρ̄²(τ) × W(τ)] / Σ[ρ̄²(τ) × W(τ)]
        where ρ̄(τ) is the truncated IS ratio: ρ̄ = min(π_FSDP/π_vLLM, threshold)
    """
    with torch.no_grad():

        # Compute W(τ): score-norm proxy that approximates ||∇log π||²
        # Formula: W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
        if uniform_weight:
            # Ablation: uniform weights (w_per_timestep = 1)
            # Results in W(τ) = length (length-proportional weighting)
            w_per_timestep = torch.ones_like(old_log_probs)
        elif uniform_cumulative:
            # Ablation: normalized uniform weights (w_per_timestep = 1/length)
            # Results in W(τ) = 1 (length-independent weighting)
            seq_lengths = response_mask.sum(dim=-1, keepdim=True)  # [batch, 1]
            w_per_timestep = torch.ones_like(old_log_probs) / seq_lengths
        else:
            # Standard: use variance proxy (w_per_timestep = 1 - 2π_t + Σπ²)
            pi_t = torch.exp(old_log_probs)
            w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Apply rollout importance sampling correction when pre-computed weights provided
        if rollout_is_weights is not None:
            # Scale W by (IS weight)² to minimize MSE under truncated IS
            # This implements the optimal baseline for truncated importance sampling:
            # b* = Σ[R × ρ̄² × W] / Σ[ρ̄² × W], where ρ̄ = min(π_train/π_rollout, threshold)
            # Weights are pre-computed and already detached
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        w_values = (w_per_timestep * response_mask).sum(dim=-1) * loss_mask  # [shape: (bs * n * turn,)]
        w_values = w_values.reshape(-1, max_turns)  # [shape: (bs * n, turn)]
        w_values = torch.cumsum(w_values, dim=-1)  # [shape: (bs * n, turn)]
        w_values = w_values.reshape(-1)  # [shape: (bs * n * turn,)]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        batch_size = turn_rewards.shape[0]
        for i in range(batch_size):
            if turn_indices[i].item() == -1 or not loss_mask[i]:
                continue
            idx = (index[i], turn_indices[i].item())
            prompt_groups[idx].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(turn_rewards)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=turn_rewards.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = w_values[traj_idx]
            R_group = turn_rewards[traj_idx]

            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum() / (w_group.sum() + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

        # Compute advantages
        advantages = turn_rewards - baselines

    return advantages, turn_rewards


def compute_multi_turn_optimal_baseline(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    loss_mask: torch.Tensor,
    index: np.ndarray,
    max_turns: int,
    epsilon: float = 1e-8,
    rollout_is_weights: torch.Tensor = None,
    uniform_weight: bool = False,
    uniform_cumulative: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using optimal baseline for minimum variance REINFORCE.

    The optimal baseline minimizes gradient variance by using a score-norm proxy W(τ)
    that approximates ||∇log π||² (the squared gradient norm of the score function).

    Args:
        rewards: Rewards of each rollout [shape: (bs * n)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs * n * turn, response_length)]
        old_log_probs: Log probabilities from FSDP model during generation [shape: (bs * n * turn, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs * n * turn, response_length)]
        loss_mask: Binary mask for valid turns (1) vs padding (0) [shape: (bs * n * turn,)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs * n,)]
        max_turns: Maximum number of turns
        epsilon: Small constant for numerical stability (default: 1e-8)
        rollout_is_weights: Pre-computed IS weights for W(τ) correction [shape: (bs * n * turn, response_length)], None if not using IS
        uniform_weight: If True, use w_per_timestep = 1 instead of variance proxy.
            Results in W(τ) = length (length-proportional weighting). Default: False
        uniform_cumulative: If True, use w_per_timestep = 1/length instead of variance proxy.
            Results in W(τ) = 1 (length-independent weighting). Default: False

    Returns:
        baselines: Optimal baselines for each completions [shape: (bs * n,)]

    vLLM Importance Sampling Correction:
        When vLLM generates trajectories but FSDP computes gradients, we correct for the
        distribution mismatch. The importance weight is:
            ρ = π_FSDP(a|s) / π_vLLM(a|s) = exp(log_π_FSDP - log_π_vLLM)

        This weight is truncated to prevent instability:
            ρ̄ = min(ρ, threshold)

        The W score is then scaled by ρ̄² because this minimizes the MSE of the
        gradient estimator under truncated importance sampling. This follows from the
        optimal baseline theory for biased estimators:
            b* = Σ[R(τ) × ρ̄²(τ) × W(τ)] / Σ[ρ̄²(τ) × W(τ)]
        where ρ̄(τ) is the truncated IS ratio: ρ̄ = min(π_FSDP/π_vLLM, threshold)
    """
    with torch.no_grad():

        # Compute W(τ): score-norm proxy that approximates ||∇log π||²
        # Formula: W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
        if uniform_weight:
            # Ablation: uniform weights (w_per_timestep = 1)
            # Results in W(τ) = length (length-proportional weighting)
            w_per_timestep = torch.ones_like(old_log_probs)
        elif uniform_cumulative:
            # Ablation: normalized uniform weights (w_per_timestep = 1/length)
            # Results in W(τ) = 1 (length-independent weighting)
            seq_lengths = response_mask.sum(dim=-1, keepdim=True)  # [batch, 1]
            w_per_timestep = torch.ones_like(old_log_probs) / seq_lengths
        else:
            # Standard: use variance proxy (w_per_timestep = 1 - 2π_t + Σπ²)
            pi_t = torch.exp(old_log_probs)
            w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Apply rollout importance sampling correction when pre-computed weights provided
        if rollout_is_weights is not None:
            # Scale W by (IS weight)² to minimize MSE under truncated IS
            # This implements the optimal baseline for truncated importance sampling:
            # b* = Σ[R × ρ̄² × W] / Σ[ρ̄² × W], where ρ̄ = min(π_train/π_rollout, threshold)
            # Weights are pre-computed and already detached
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        w_mask = response_mask * loss_mask.unsqueeze(1)  # [shape: (bs * n * turn, response_length)]
        w_values = (
            (w_per_timestep * w_mask).reshape(-1, max_turns, w_mask.shape[-1]).sum(dim=[-1, -2])
        )  # [shape: (bs * n,)]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        batch_size = rewards.shape[0]
        last_turn_loss_mask = loss_mask.reshape(-1, max_turns)[:, -1]  # [shape: (bs * n,)]
        for i in range(batch_size):
            if not last_turn_loss_mask[i]:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(rewards)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=rewards.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = w_values[traj_idx]
            R_group = rewards[traj_idx]

            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum() / (w_group.sum() + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

    return baselines
