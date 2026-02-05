# Copyright 2025 ByteDance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms with enhanced loss aggregation modes.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import verl.utils.torch_functional as verl_F

from verl_patch.utils.metric import PolicyOutput


def compute_multi_turn_returns(scores, gamma, max_turns):

    with torch.no_grad():
        shaped_scores = scores.reshape(-1, max_turns)

        returns = torch.zeros_like(shaped_scores)

        for i in reversed(range(max_turns)):
            if i == max_turns - 1:
                returns[:, i] = shaped_scores[:, i]
            else:
                returns[:, i] = shaped_scores[:, i] + gamma * returns[:, i + 1]

        returns = returns.reshape(-1)
    return returns


def compute_multi_turn_cumulative_rewards(scores, max_turns):

    with torch.no_grad():
        shaped_scores = scores.reshape(-1, max_turns)

        cum_scores = torch.cumsum(shaped_scores, dim=-1)

        cum_scores = cum_scores.reshape(-1)

    return cum_scores


def validate_loss_agg_mode(loss_agg_mode: str) -> None:
    """
    Validate the loss aggregation mode parameter.

    Args:
        loss_agg_mode: The loss aggregation mode to validate

    Raises:
        ValueError: If the mode is not supported
    """
    valid_modes = [
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
        "seq-sum-no-norm",
    ]
    if loss_agg_mode not in valid_modes:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}. " f"Supported modes: {valid_modes}")


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        # Fixed KL controller doesn't update
        _ = current_kl  # Mark as intentionally unused
        _ = n_steps  # Mark as intentionally unused
        pass


def agg_loss(
    loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str, scale_factor: float = 1.0
) -> torch.Tensor:
    """
    Aggregate the loss matrix into a scalar using different aggregation strategies.

    This function provides flexible loss aggregation modes that can significantly impact
    training dynamics, especially for algorithms like DrGRPO that require specific
    normalization strategies.

    Args:
        loss_mat: `(torch.Tensor)`
            Loss values per token, shape: (batch_size, response_length)
        loss_mask: `(torch.Tensor)`
            Binary mask indicating valid tokens, shape: (batch_size, response_length)
        loss_agg_mode: (str) Loss aggregation strategy. Choices:
            - "token-mean": Average across all valid tokens in the batch (length-biased)
            - "seq-mean-token-sum": Sum tokens per sequence, then average across sequences (REINFORCE-aligned, recommended)
            - "seq-mean-token-mean": Mean tokens per sequence, then average across sequences (length-biased)
            - "seq-mean-token-sum-norm": Sum tokens per sequence, normalize by max_seq_length (constant scaling)
            - "seq-sum-no-norm": Sum all tokens across all sequences (no normalization)
        scale_factor: (float) Optional scaling factor to divide the final loss by.
            Useful for controlling gradient magnitude with very long sequences.
            Default: 1.0 (no scaling)

    Returns:
        loss: `torch.Tensor`
            Aggregated scalar loss

    Mathematical Details:
        token-mean:
            loss = Σ(loss_mat * loss_mask) / Σ(loss_mask)

        seq-mean-token-sum:
            seq_losses[i] = Σ_j(loss_mat[i,j] * loss_mask[i,j])
            loss = mean(seq_losses)

        seq-mean-token-mean:
            seq_losses[i] = Σ_j(loss_mat[i,j] * loss_mask[i,j]) / Σ_j(loss_mask[i,j])
            loss = mean(seq_losses)

        seq-mean-token-sum-norm:
            seq_losses[i] = Σ_j(loss_mat[i,j] * loss_mask[i,j])
            loss = Σ(seq_losses) / max_seq_length

        seq-sum-no-norm:
            loss = Σ(loss_mat * loss_mask)

    Notes:
        - "token-mean" treats all tokens equally, regardless of sequence boundaries
        - "seq-mean-*" modes ensure each sequence contributes equally to the final loss
        - "seq-mean-token-sum-norm" maintains constant normalization for DrGRPO replication
        - "seq-sum-no-norm" provides raw unnormalized loss sum across all tokens
        - Use scale_factor parameter to control gradient magnitude for long sequences
        - The choice of aggregation mode can affect convergence and bias in RL training
    """
    # Validate the aggregation mode
    validate_loss_agg_mode(loss_agg_mode)

    if loss_agg_mode == "token-mean":
        # Standard token-level averaging across the entire batch
        loss = verl_F.masked_mean(loss_mat, loss_mask)

    elif loss_agg_mode == "seq-mean-token-sum":
        # Sum loss per sequence, then average across sequences
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # (batch_size,)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # (batch_size,) 1 if seq has valid tokens
        loss = verl_F.masked_mean(seq_losses, seq_mask)

    elif loss_agg_mode == "seq-mean-token-mean":
        # Mean loss per sequence (length-normalized), then average across sequences
        # Compute per-sequence token-mean, then average over valid sequences
        seq_losses = verl_F.masked_mean(loss_mat, loss_mask, axis=-1)  # (batch_size,) token-mean per seq
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # (batch_size,) 1 if seq has valid tokens
        loss = verl_F.masked_mean(seq_losses, seq_mask)  # seq-mean over valid seqs

    elif loss_agg_mode == "seq-mean-token-sum-norm":
        # Sum loss per sequence, normalize by maximum sequence length
        # This mode is designed for DrGRPO paper replication where the divisor
        # should remain constant throughout training
        # NOTE: Rejected sequences (with all tokens masked) contribute 0 to the sum,
        # which is the intended behavior for constant normalization
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # (batch_size,)
        max_seq_length = loss_mask.shape[-1]  # Maximum possible sequence length
        loss = torch.sum(seq_losses) / max_seq_length

    elif loss_agg_mode == "seq-sum-no-norm":
        # Sum all valid tokens across all sequences without any normalization
        # This gives the raw total loss across the entire batch
        # Formula: loss = Σ(loss_mat * loss_mask)
        loss = torch.sum(loss_mat * loss_mask)

    # Apply optional scaling factor to control gradient magnitude
    if scale_factor != 1.0:
        loss = loss / scale_factor

    return loss


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == 'fixed':
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == 'adaptive':
        assert kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {kl_ctrl.horizon}'
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """
    Compute advantage for GRPO using trajectory-level baseline.

    Mathematical formulation:
    A_t^(i) = (R_t^(i) - mean) / std
    where R_t is reward-to-go from position t, and mean/std are computed
    from total trajectory rewards within each prompt group.

    This correctly implements the decoupled policy gradient with trajectory-level
    baseline, since we only sample complete trajectories from the initial state.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # Compute returns (reward-to-go from each position)
    returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    with torch.no_grad():
        # Extract total trajectory rewards (R^(i) = R_1^(i))
        trajectory_rewards = returns[:, 0] if returns.shape[0] > 0 else returns
        bsz = trajectory_rewards.shape[0]

        # Group trajectories by prompt and compute normalization parameters
        id2rewards = defaultdict(list)
        id2mean = {}
        id2std = {}

        for i in range(bsz):
            id2rewards[index[i]].append(trajectory_rewards[i])

        for idx in id2rewards:
            if len(id2rewards[idx]) == 1:
                # Single trajectory: use standard normalization
                id2mean[idx] = torch.tensor(0.0, device=trajectory_rewards.device)
                id2std[idx] = torch.tensor(1.0, device=trajectory_rewards.device)
            else:
                rewards_tensor = torch.stack(id2rewards[idx])
                id2mean[idx] = rewards_tensor.mean()
                id2std[idx] = rewards_tensor.std()
                if id2std[idx] < epsilon:
                    id2std[idx] = torch.tensor(1.0, device=trajectory_rewards.device)

        # Apply normalization to reward-to-go at each position
        # A_t^(i) = (R_t^(i) - mean) / std
        advantages = torch.zeros_like(returns)
        for i in range(bsz):
            mean = id2mean[index[i]]
            std = id2std[index[i]]
            advantages[i] = (returns[i] - mean) / (std + epsilon)

        # Apply mask
        advantages = advantages * eos_mask

    return advantages, returns


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: np.ndarray):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    using trajectory-level leave-one-out baseline.

    Mathematical formulation:
    A_t^(i) = R_t^(i) - b_LOO^(i)
    where R_t is reward-to-go from position t, and b_LOO is the leave-one-out
    mean of total trajectory rewards from the same prompt group.

    This correctly implements the decoupled policy gradient with trajectory-level
    leave-one-out baseline, since we only sample complete trajectories.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # Compute returns (reward-to-go from each position)
    returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    with torch.no_grad():
        # Extract total trajectory rewards
        trajectory_rewards = returns[:, 0] if returns.shape[0] > 0 else returns
        bsz = trajectory_rewards.shape[0]

        # Group trajectories by prompt
        id2rewards = defaultdict(list)
        id2indices = defaultdict(list)

        for i in range(bsz):
            id2rewards[index[i]].append(trajectory_rewards[i])
            id2indices[index[i]].append(i)

        # Compute leave-one-out baselines for each trajectory
        trajectory_baselines = torch.zeros_like(trajectory_rewards)

        for idx in id2rewards:
            rewards_list = id2rewards[idx]
            indices_list = id2indices[idx]
            n_traj = len(rewards_list)

            if n_traj == 1:
                # Single trajectory: no baseline (keep original score)
                trajectory_baselines[indices_list[0]] = 0.0
            else:
                # Multiple trajectories: compute LOO baseline for each
                rewards_tensor = torch.stack(rewards_list)
                total_sum = rewards_tensor.sum()

                for i, traj_idx in enumerate(indices_list):
                    # Leave-one-out mean: (sum - current) / (n - 1)
                    loo_baseline = (total_sum - rewards_tensor[i]) / (n_traj - 1)
                    trajectory_baselines[traj_idx] = loo_baseline

        # Compute advantages: A_t = R_t - b_LOO
        # Expand baselines to match token dimension
        baselines_expanded = trajectory_baselines.unsqueeze(-1).expand_as(returns)
        advantages = (returns - baselines_expanded) * eos_mask

    return advantages, returns


def compute_multi_turn_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    turn_indices: torch.Tensor,
    index: np.ndarray,
    max_turns: int,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
):
    """
    Turn-aware REINFORCE Leave-one-out: compute advantages using mean of other samples
    with same prompt, same turn, and loss_mask == 1
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, )
        turn_indices: `(torch.Tensor)`
            shape: (bs, )

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    returns = compute_multi_turn_returns(scores, gamma, max_turns)

    id2return = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = returns.shape[0]

        advantages = torch.zeros_like(returns)

        for i in range(bsz):
            if turn_indices[i].item() == -1 or not loss_mask[i]:
                continue
            idx = (index[i], turn_indices[i].item())
            id2return[idx].append(returns[i])

        for idx in id2return:
            if len(id2return[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2return[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2return[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if turn_indices[i].item() == -1 or not loss_mask[i]:
                continue
            idx = (index[i], turn_indices[i].item())
            response_num = len(id2return[idx])
            if response_num > 1:
                advantages[i] = returns[i] * response_num / (response_num - 1) - id2mean[idx] * response_num / (
                    response_num - 1
                )
            else:
                advantages[i] = returns[i]

        advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        returns = returns.unsqueeze(-1).tile([1, response_length]) * eos_mask
        return advantages, returns


def apply_batch_standardization(advantages: torch.Tensor, response_mask: torch.Tensor, epsilon: float = 1e-6):
    """
    Apply batch-level standardization to advantages using the correct method based on advantage structure.

    This function automatically detects the modeling perspective and applies appropriate standardization:

    **Modeling Perspectives:**
    - **Sequence-Level Modeling**: GRPO/RLOO (uniform advantages within sequences)
      → Uses sequence-level standardization to avoid length bias
    - **Token-Level Modeling**: GAE (variable advantages) OR post-constraint advantages
      → Uses token-level standardization for proper credit assignment


    Args:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length) - Token-level advantages
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length) - Mask indicating valid tokens
        epsilon: (float) Small value to prevent division by zero

    Returns:
        standardized_advantages: `(torch.Tensor)`
            shape: (bs, response_length) - Standardized advantages
    """
    with torch.no_grad():
        if advantages.numel() == 0 or response_mask.sum() == 0:
            return advantages

        # Check if advantages are uniform within sequences (sequence-level modeling)
        is_uniform_within_sequences = _check_uniform_within_sequences(advantages, response_mask, epsilon)

        if is_uniform_within_sequences:
            # Sequence-level standardization for GRPO/RLOO (avoids length bias)
            # Compute sequence-level advantages (mean per sequence)
            sequence_lengths = response_mask.sum(dim=-1)
            valid_sequences = sequence_lengths > 0

            if not valid_sequences.any():
                return advantages

            # Compute mean advantage per sequence using masked_mean
            sequence_advantages = verl_F.masked_mean(advantages, response_mask, axis=-1)
            valid_seq_advantages = sequence_advantages[valid_sequences]

            # Standardize across sequences (not tokens!)
            seq_mean = torch.mean(valid_seq_advantages)
            seq_std = torch.std(valid_seq_advantages)

            if seq_std < epsilon:  # All sequences have same advantage
                return advantages

            # Apply standardization uniformly to all tokens in each sequence
            standardized_seq_advantages = (sequence_advantages - seq_mean) / seq_std
            standardized_advantages = standardized_seq_advantages.unsqueeze(-1) * response_mask

        else:
            # Token-level standardization for GAE or post-constraint advantages
            valid_mask = response_mask.bool()
            valid_advantages = advantages[valid_mask]

            # Standardize across all valid tokens
            token_mean = torch.mean(valid_advantages)
            token_std = torch.std(valid_advantages)

            if token_std < epsilon:  # All tokens have same advantage
                return advantages

            standardized_advantages = torch.zeros_like(advantages)
            standardized_advantages[valid_mask] = (valid_advantages - token_mean) / token_std

    return standardized_advantages


def _check_uniform_within_sequences(
    advantages: torch.Tensor, response_mask: torch.Tensor, epsilon: float = 1e-6
) -> bool:
    """
    Check if advantages are uniform within each sequence (indicating sequence-level modeling).

    Args:
        advantages: Token-level advantages [batch_size, response_length]
        response_mask: Valid token mask [batch_size, response_length]
        epsilon: Threshold for considering values as uniform

    Returns:
        bool: True if advantages are uniform within sequences (GRPO/RLOO), False if variable (GAE)
    """
    batch_size = advantages.shape[0]

    for i in range(batch_size):
        seq_mask = response_mask[i].bool()
        valid_seq_length = seq_mask.sum().item()

        if valid_seq_length <= 1:
            continue  # Skip sequences with 0 or 1 tokens

        seq_advantages = advantages[i][seq_mask]
        seq_variance = torch.var(seq_advantages).item()

        # If any sequence has non-uniform advantages, it's token-level modeling
        if seq_variance > epsilon:
            return False

    return True


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, eos_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange_low,
    cliprange_high,
    clip_ratio_c=3.0,
    entropy_clip_rate=0,
    entropy=None,
    use_gspo=False,
    loss_agg_mode="seq-mean-token-sum",
    loss_scale_factor=1.0,
    rollout_is_weights=None,
    extreme_risk_prob_threshold=None,
):
    """Compute policy loss for PPO or GSPO with optional rollout importance sampling correction.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            Log probabilities from the old policy. Shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            Log probabilities from the current policy. Shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            Advantage estimates for each token. Shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            Mask indicating valid tokens (1 for valid, 0 for padding). Shape: (bs, response_length)
        cliprange_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_high: (float)
            The higher clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_c: (float)
            THe lower bound of the ratio for dual-clip PPO, defalut 3. See https://arxiv.org/pdf/1912.09729
        entropy_clip_rate: (float)
            Fraction of tokens with low entropy to exclude from policy loss.
            For example, 0.8 means 80% of low-entropy tokens will not be trained on.
        entropy: `(torch.Tensor)` Optional
            Token-level entropy values. Shape: (bs, response_length). Required if entropy_clip_rate > 0.
        use_gspo: (bool)
            Whether to use GSPO (Group Sequence Policy Optimization) instead of standard PPO.
        loss_agg_mode: (str)
            Loss aggregation strategy. Choices:
            - "token-mean": Average across all valid tokens in the batch (length-biased)
            - "seq-mean-token-sum": Sum tokens per sequence, then average across sequences (REINFORCE-aligned, recommended)
            - "seq-mean-token-mean": Mean tokens per sequence, then average across sequences (length-biased)
            - "seq-mean-token-sum-norm": Sum tokens per sequence, normalize by max_seq_length (constant scaling)
            - "seq-sum-no-norm": Sum all tokens across all sequences (no normalization)
        loss_scale_factor: (float)
            Multiplicative scaling factor applied to final loss value. Default: 1.0
        rollout_is_weights: `(torch.Tensor)` Optional
            Pre-computed rollout importance sampling weights from the trainer.
            Shape: (bs, response_length). These weights correct for distribution mismatch
            between rollout policy (e.g., vLLM BFloat16) and training policy (FSDP FP32).
            Weights are computed centrally in ray_trainer.py using mismatch_helper.py.
        extreme_risk_prob_threshold: (float) Optional
            Probability threshold for masking extreme risk tokens in negative advantage trajectories.
            Tokens with π(a|s) < threshold AND negative advantages will be masked (zero loss).
            This prevents gradient explosion from very low probability tokens.
            Typical values: 1e-5, 1e-6, or 1e-7 for aggressive risk mitigation.

    Returns:
        PolicyOutput: A dataclass containing:
            - loss: Policy gradient loss computed via PPO
            - kl_divergence: KL between current and old policy
            - clip_fraction: Fraction of ratios being clipped
            - clip_fraction_lower/upper/dual: Detailed clipping statistics
            - algorithm: "ppo"
            - algorithm_metrics: Dict containing (when applicable):
                - extreme_risk_*: Risk masking statistics (if extreme_risk_prob_threshold set)
            - loss_config: Configuration used for loss computation

    Note:
        - Rollout IS weights and mismatch metrics are computed centrally in ray_trainer.py
          using mismatch_helper.compute_rollout_importance_weights() before distributing
          the batch to workers. This avoids duplicate computation and ensures consistency.
        - All mismatch metrics are prefixed with "mismatch/" and logged by the trainer.
    """
    if use_gspo:
        # Compute policy loss for GSPO (Group Sequence Policy Optimization).
        # Compute length-normalized sequence-level importance ratio
        # Use masked_mean for better numerical stability
        # This computes mean(log π) - mean(log μ) which is more stable than mean(log π - log μ)
        avg_log_prob = verl_F.masked_mean(log_prob, eos_mask, axis=-1)
        avg_old_log_prob = verl_F.masked_mean(old_log_prob, eos_mask, axis=-1)

        # Average log ratio per token (equivalent to seq_log_ratio but more numerically stable)
        seq_log_ratio = avg_log_prob - avg_old_log_prob

        # Compute sequence-level importance ratio s_i(θ)
        seq_ratio = torch.exp(seq_log_ratio)

        # Extract sequence-level advantages (use the first valid token's advantage)
        seq_advantages = advantages[:, 0]

        # Compute GSPO loss with clipping
        pg_losses = -seq_advantages * seq_ratio
        pg_losses2 = -seq_advantages * torch.clamp(seq_ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)

        # Choose the maximum (less negative) loss
        pg_losses_clipped = torch.max(pg_losses, pg_losses2)

        # Compute clip fraction
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).float())

        # Final loss is the mean over sequences
        pg_loss = torch.mean(pg_losses_clipped)

        # Compute KL divergence (average KL per token)
        # Using the stable computation: mean(log_old) - mean(log_new) = -seq_log_ratio
        gspo_kl = -seq_log_ratio.mean()

        # GSPO doesn't use lower bound clipping, so return 0.0 for compatibility
        pg_clipfrac_lower = torch.tensor(0.0)

        # Create unified output for GSPO
        return PolicyOutput(
            loss=pg_loss,
            kl_divergence=gspo_kl,
            clip_fraction=pg_clipfrac,
            clip_fraction_lower=pg_clipfrac_lower,
            clip_fraction_upper=torch.tensor(0.0),  # GSPO doesn't use upper clipping
            clip_fraction_dual=torch.tensor(0.0),  # GSPO doesn't use dual clipping
            algorithm="gspo",
            algorithm_metrics={},
            loss_config={
                "loss_agg_mode": loss_agg_mode,
                "loss_scale_factor": loss_scale_factor,
                "cliprange_low": cliprange_low,
                "cliprange_high": cliprange_high,
                "entropy_clip_rate": entropy_clip_rate,
            },
        )

    assert (
        clip_ratio_c > 1.0
    ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses_original = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)

    clip_pg_losses1 = torch.maximum(pg_losses_original, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
    # We only apply the dual-clip when the advantage is negative.
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
    # Correct clipping fraction calculations
    # Dual-clip activated: when dual constraint is binding (dual loss < standard loss)
    dual_clip_activated = torch.lt(pg_losses3, clip_pg_losses1) * (advantages < 0).float()

    # Standard PPO clipping bounds
    ratio_upper_bound = ratio > (1.0 + cliprange_high)
    ratio_lower_bound = ratio < (1.0 - cliprange_low)

    # Make categories mutually exclusive: dual-clip takes precedence
    # Upper bound clipping: ratio > 1 + cliprange_high AND NOT dual-clipped
    upper_clipped = ratio_upper_bound.float() * (1.0 - dual_clip_activated)

    # Lower bound clipping: ratio < 1 - cliprange_low OR dual-clip activated
    lower_clipped = torch.clamp(ratio_lower_bound.float() + dual_clip_activated, 0.0, 1.0)

    # Total clipping: upper + lower (now mutually exclusive)
    total_clipped = upper_clipped + lower_clipped

    pg_clipfrac = verl_F.masked_mean(total_clipped, eos_mask)
    pg_clipfrac_lower = verl_F.masked_mean(lower_clipped, eos_mask)
    pg_clipfrac_higher = verl_F.masked_mean(upper_clipped, eos_mask)
    pg_clipfrac_dual = verl_F.masked_mean(dual_clip_activated, eos_mask)

    # Compute entropy clip
    if entropy_clip_rate > 0:
        assert entropy is not None, "entropy should be provided when entropy_clip_rate > 0."
        # Note: This uses masked_quantile which may not be available in all versions
        # Consider implementing a fallback or ensuring this dependency is available
        try:
            entropy_quantile = verl_F.masked_quantile(entropy, eos_mask, entropy_clip_rate)
            pg_losses = torch.where(entropy > entropy_quantile, pg_losses, 0.0)
        except AttributeError:
            # Fallback: skip entropy clipping if masked_quantile is not available
            import warnings

            warnings.warn("masked_quantile not available, skipping entropy clipping")

    # Mask extreme risk tokens ONLY in negative advantage trajectories
    # This prevents gradient explosion from the policy gradient term: ∇L ∝ A/π
    # When A < 0 and π → 0, the gradient can explode catastrophically
    extreme_risk_masked_fraction = None
    extreme_risk_masked_advantage_mass = None
    extreme_risk_effective_batch_ratio = None
    if extreme_risk_prob_threshold is not None:
        # Work in log space for better numerical precision
        log_threshold = torch.log(torch.tensor(extreme_risk_prob_threshold))

        # Identify extreme risk tokens: BOTH conditions must be true:
        # 1. Low probability: log(π) < log(threshold) ⟺ π < threshold (high gradient amplification)
        # 2. Negative advantage: A < 0 (would cause large negative gradient)
        extreme_risk_mask = (old_log_prob < log_threshold) & (advantages < 0)

        # Essential monitoring metrics
        # 1. Fraction of tokens being masked
        extreme_risk_masked_fraction = verl_F.masked_mean(extreme_risk_mask.float(), eos_mask)

        # 2. Total advantage mass being masked (shows importance of masked tokens)
        masked_advantages = torch.where(
            extreme_risk_mask & eos_mask.bool(), advantages.abs(), torch.zeros_like(advantages)
        )
        total_advantages = torch.where(eos_mask.bool(), advantages.abs(), torch.zeros_like(advantages))
        masked_advantages_sum = verl_F.masked_sum(masked_advantages, eos_mask)
        total_advantages_sum = verl_F.masked_sum(total_advantages, eos_mask)
        extreme_risk_masked_advantage_mass = masked_advantages_sum / (total_advantages_sum + 1e-8)

        # 3. Effective batch size ratio (how much data we're actually using)
        total_valid_tokens = eos_mask.sum()
        unmasked_tokens = (eos_mask.bool() & ~extreme_risk_mask).float().sum()
        extreme_risk_effective_batch_ratio = unmasked_tokens / (total_valid_tokens + 1e-8)

        # Zero out loss for extreme risk tokens to prevent gradient explosion
        # This effectively removes these tokens from the gradient computation
        pg_losses = torch.where(extreme_risk_mask, torch.zeros_like(pg_losses), pg_losses)

    # vLLM Importance Sampling Correction
    # NOTE: IS weights are now pre-computed centrally in ray_trainer.py
    # This avoids duplicate computation and ensures consistency across workers
    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = rollout_is_weights * pg_losses

    # Use the flexible loss aggregation function instead of hardcoded masked_mean
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode, scale_factor=loss_scale_factor
    )

    # Create algorithm_metrics dict
    # Note: IS and mismatch metrics are now computed centrally in ray_trainer.py
    # to avoid duplication and ensure consistency
    algorithm_metrics = {}

    if extreme_risk_masked_fraction is not None:
        algorithm_metrics["extreme_risk_masked_fraction"] = extreme_risk_masked_fraction
        algorithm_metrics["extreme_risk_masked_advantage_mass"] = extreme_risk_masked_advantage_mass
        algorithm_metrics["extreme_risk_effective_batch_ratio"] = extreme_risk_effective_batch_ratio
        algorithm_metrics["extreme_risk_prob_threshold"] = extreme_risk_prob_threshold

    return PolicyOutput(
        loss=pg_loss,
        kl_divergence=ppo_kl,
        clip_fraction=pg_clipfrac,
        clip_fraction_lower=pg_clipfrac_lower,
        clip_fraction_upper=pg_clipfrac_higher,
        clip_fraction_dual=pg_clipfrac_dual,
        algorithm="ppo",
        algorithm_metrics=algorithm_metrics,
        loss_config={
            "loss_agg_mode": loss_agg_mode,
            "loss_scale_factor": loss_scale_factor,
            "cliprange_low": cliprange_low,
            "cliprange_high": cliprange_high,
            "clip_ratio_c": clip_ratio_c,
            "entropy_clip_rate": entropy_clip_rate,
            "rollout_is_weights_applied": rollout_is_weights is not None,
            "extreme_risk_prob_threshold": extreme_risk_prob_threshold,
        },
    )


def compute_policy_loss_with_rollout_correction(
    rollout_log_prob,
    log_prob,
    advantages,
    eos_mask,
    loss_agg_mode="seq-mean-token-sum",
    loss_scale_factor=1.0,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: float = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
    max_turns: int = 1,
):
    """Compute policy loss with pure rollout correction (no PPO clipping).

    This function implements policy gradient with importance sampling correction
    for rollout-training policy mismatch, without PPO's clipping mechanism.

    Mathematical formulation:
        Without IS (rollout_is=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is enabled):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length)
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        eos_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        loss_scale_factor: Multiplicative scaling factor applied to final loss.
        rollout_is: IS aggregation level ("token", "turn", "sequence", or None).
        rollout_is_threshold: Upper threshold for truncating IS weights.
        rollout_rs: Rejection sampling aggregation level (or None to disable).
        rollout_rs_threshold: Upper threshold for rejection sampling.
        rollout_rs_threshold_lower: Lower threshold for rejection sampling.
        rollout_token_veto_threshold: Per-token veto threshold for catastrophic outliers.
        max_turns: Maximum number of conversation turns (for multi-turn aggregation).

    Returns:
        PolicyOutput containing:
            - loss: Policy gradient loss with IS correction
            - kl_divergence: KL between current and rollout policy
            - clip_fraction: Always 0.0 (no clipping in this mode)
            - algorithm: "pure_rollout_correction"
            - algorithm_metrics: IS/RS statistics and mismatch metrics

    Note:
        Unlike compute_policy_loss (PPO), this function:
        - Does NOT use PPO clipping (no old_log_prob needed)
        - Directly applies IS correction computed from current vs rollout
        - Computes IS/RS on-the-fly during training

    Usage:
        This function is called by the actor when:
        - bypass_old_logprob_for_rollout=True (trainer uses rollout_log_prob as old_log_prob)
        - use_pure_rollout_correction=True (actor uses this function instead of compute_policy_loss)

    Example config:
        algorithm:
          bypass_old_logprob_for_rollout: true
          use_pure_rollout_correction: true
          rollout_is: "token"
          rollout_is_kwargs: {upper: 2.0}
          rollout_rs: "token"
          rollout_rs_kwargs: {upper: 2.0, lower: 0.5}

    Performance:
        - Memory: Avoids storing old_log_prob tensors
        - Speed: Skips expensive actor.compute_log_prob() forward pass
        - Variance: Higher than PPO (no clipping safety net)
    """
    # Import rollout correction helper
    from verl_patch.trainer.code.ppo.mismatch_helper import (
        compute_rollout_importance_weights_and_rejection_mask,
    )

    # Compute IS weights and rejection mask on-the-fly
    rollout_is_weights, modified_response_mask, rollout_metrics = compute_rollout_importance_weights_and_rejection_mask(
        old_log_prob=log_prob,  # Current policy
        rollout_log_prob=rollout_log_prob,  # Rollout policy
        response_mask=eos_mask,
        max_turns=max_turns,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
    )

    # Apply rejection mask (if RS is enabled)
    effective_mask = modified_response_mask if rollout_rs is not None else eos_mask

    # Compute pure policy gradient loss with IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    #
    # Note: rollout_is_weights already contains w = π_current / π_rollout
    # So we apply it to the standard log-prob trick formula

    if rollout_is_weights is not None:
        # With IS correction: weight the log-prob trick by IS weight
        # w = exp(log_prob - rollout_log_prob).clamp(max=threshold)
        # L = -E[w * log π * A]
        # Gradient: ∇L = -E[w * ∇log π * A] = -E[w * A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # No IS correction: standard REINFORCE with log-prob trick
        # L = -E[log π(a|s) * A]
        # Gradient: ∇L = -E[∇log π * A] = -E[A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=effective_mask,
        loss_agg_mode=loss_agg_mode,
        scale_factor=loss_scale_factor,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, effective_mask)

    # Prepare algorithm metrics (include all rollout correction metrics)
    algorithm_metrics = dict(rollout_metrics)  # Contains IS/RS/mismatch metrics

    return PolicyOutput(
        loss=pg_loss,
        kl_divergence=kl_divergence,
        clip_fraction=torch.tensor(0.0),  # No clipping in this mode
        clip_fraction_lower=torch.tensor(0.0),
        clip_fraction_upper=torch.tensor(0.0),
        clip_fraction_dual=torch.tensor(0.0),
        algorithm="pure_rollout_correction",
        algorithm_metrics=algorithm_metrics,
        loss_config={
            "loss_agg_mode": loss_agg_mode,
            "loss_scale_factor": loss_scale_factor,
            "rollout_is": rollout_is,
            "rollout_is_threshold": rollout_is_threshold,
            "rollout_rs": rollout_rs,
            "rollout_rs_threshold": rollout_rs_threshold,
            "rollout_token_veto_threshold": rollout_token_veto_threshold,
        },
    )


def compute_entropy_loss(logits, eos_mask, loss_agg_mode="seq-mean-token-sum", loss_scale_factor=1.0):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) Loss aggregation strategy. Choices:
            - "token-mean": Average across all valid tokens in the batch (length-biased)
            - "seq-mean-token-sum": Sum tokens per sequence, then average across sequences (REINFORCE-aligned, recommended)
            - "seq-mean-token-mean": Mean tokens per sequence, then average across sequences (length-biased)
            - "seq-mean-token-sum-norm": Sum tokens per sequence, normalize by max_seq_length (constant scaling)
            - "seq-sum-no-norm": Sum all tokens across all sequences (no normalization)

    Returns:
        entropy_loss: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(
        loss_mat=entropy, loss_mask=eos_mask, loss_agg_mode=loss_agg_mode, scale_factor=loss_scale_factor
    )
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
