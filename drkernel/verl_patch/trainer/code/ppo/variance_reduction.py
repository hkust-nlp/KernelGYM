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
General variance reduction techniques for all advantage estimators.

This module provides batch reweighting and multi-prompt MVU that can be
applied to ANY advantage estimator (GAE, GRPO, RLOO, optimal baseline, etc.).
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from verl import DataProto


def compute_batch_reweighting(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    w_values: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Compute MSE-optimal batch reweighting for variance reduction.

    Optimally weights trajectories to minimize Mean Squared Error (MSE) of the
    gradient estimator. Applies to any advantage estimator.

    Theory:
    - Gradient estimator: ḡ = (1/N) Σ_i A_i * S_θ(τ_i)
    - Optimal weight: w* = ||g_true||² / (||g_true||² + Var(ḡ))
    - Uses W(τ) as proxy for ||S_θ(τ)||²

    Args:
        advantages: Token-level advantages [batch, seq_len]
        response_mask: Valid token mask [batch, seq_len]
        w_values: W(τ) values as gradient norm proxy [batch] or [batch, seq_len]
        epsilon: Numerical stability constant

    Returns:
        Dict with batch_weights [batch], mse_scale, gradient_norm, gradient_variance
    """
    batch_size = advantages.shape[0]
    device = advantages.device

    # Get scalar advantages per trajectory
    advantages_scalar = (advantages * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + epsilon)

    if w_values is None:
        w_values = torch.ones(batch_size, device=device)
    elif w_values.dim() > 1:
        w_values = (w_values * response_mask).sum(dim=-1)

    # Estimate ||g_true||² using E[A² * W]
    gradient_norm_estimate = (advantages_scalar**2 * w_values).mean()

    # Estimate Var(ḡ) using Var(X) = E[X²] - E[X]² where X = A * √W
    aw = advantages_scalar * torch.sqrt(w_values + epsilon)
    gradient_variance = (aw**2).mean() - aw.mean() ** 2

    # MSE-optimal global scaling factor
    mse_scale = gradient_norm_estimate / (gradient_norm_estimate + gradient_variance + epsilon)

    # Per-trajectory weights: normalize W(τ) by mean, then apply MSE scaling
    batch_weights = w_values / (w_values.mean() + epsilon) * mse_scale

    return {
        'batch_weights': batch_weights,
        'mse_scale': mse_scale,
        'gradient_norm': gradient_norm_estimate,  # Keep key name for compatibility
        'gradient_variance': gradient_variance,
    }


def compute_multi_prompt_mvu(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    w_values: torch.Tensor,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-prompt Minimum Variance Unbiased (MVU) weighting.

    Uses inverse variance weighting across prompts to optimally combine gradient
    estimates. Prompts with lower variance receive higher weights.

    Theory:
    - MVU estimator: ḡ_MVU = Σ_k w_k * ḡ_k
    - Optimal weights: w_k = (1/σ_k²) / Σ_j(1/σ_j²)
    - Minimizes: Var(ḡ_MVU) = 1 / Σ_k(1/σ_k²)

    Implementation:
    - Groups trajectories by prompt index
    - Estimates σ_k² using Proxy 3: Var(ḡ) = (1/(N-1)) × [E[||ĝ||²] - ||ḡ||²]
    - Uses W(τ) as proxy for gradient norm squared (IS-corrected if rollout IS enabled)
    - Applies inverse variance weighting

    Args:
        advantages: Token-level advantages [batch, seq_len]
        response_mask: Valid token mask [batch, seq_len]
        index: Prompt indices for grouping [batch]
        w_values: W(τ) values for gradient norm proxy [batch] or [batch, seq_len]
                  Should be IS-corrected (scaled by ρ̄²) if rollout_is enabled
        epsilon: Numerical stability constant

    Returns:
        Dict with prompt_weights [batch], prompt_variances, prompt_means, prompt_counts
    """
    batch_size = advantages.shape[0]
    device = advantages.device

    # Get scalar advantages per trajectory
    advantages_scalar = (advantages * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + epsilon)

    # Convert w_values to scalar per trajectory if needed
    if w_values.dim() > 1:
        w_values = (w_values * response_mask).sum(dim=-1)

    # Group trajectories by prompt
    prompt_groups = defaultdict(list)
    for i in range(batch_size):
        prompt_groups[index[i]].append(i)

    prompt_variances = {}
    prompt_means = {}
    prompt_counts = {}

    for prompt_id, trajectory_indices in prompt_groups.items():
        traj_idx = torch.tensor(trajectory_indices, device=device)
        advantages_group = advantages_scalar[traj_idx]

        N = len(trajectory_indices)
        prompt_counts[prompt_id] = N
        prompt_means[prompt_id] = advantages_group.mean() if N > 1 else advantages_group[0]

        if N == 1:
            prompt_variances[prompt_id] = torch.tensor(float('inf'), device=device)
        else:
            # Estimate Var(ḡ_j) using Proxy 3: Var(ḡ) = (1/(N-1)) × [E[||ĝ||²] - ||ḡ||²]
            # Use W(τ) as gradient norm proxy (IS-corrected if rollout IS enabled)
            w_group = w_values[traj_idx]

            # Proxy 2: E[||ĝ||²], Proxy 1: ||ḡ||² (using signed advantages)
            mean_squared_magnitude = (advantages_group**2 * w_group).mean()
            mean_grad = (advantages_group * torch.sqrt(w_group + epsilon)).mean()
            squared_norm_mean_grad = mean_grad**2

            # Proxy 3 with Bessel's correction (N is group size)
            prompt_variances[prompt_id] = (mean_squared_magnitude - squared_norm_mean_grad) / (N - 1)

    # Compute inverse variance weights
    valid_prompts = [p for p in prompt_variances if prompt_variances[p] != float('inf')]

    if len(valid_prompts) == 0:
        # Cannot estimate variance - return uniform weights and skip reduction
        return {
            'prompt_weights': torch.ones(batch_size, device=device),
            'prompt_variances': prompt_variances,
            'prompt_means': prompt_means,
            'prompt_counts': prompt_counts,
            'skip_reduction': True,
        }

    # Compute normalized inverse variance weights
    prompt_weights_dict = {}
    if len(valid_prompts) == 1:
        prompt_weights_dict[valid_prompts[0]] = 1.0
    else:
        inv_vars = {p: 1.0 / (prompt_variances[p] + epsilon) for p in valid_prompts}
        total_inv_var = sum(inv_vars.values())
        prompt_weights_dict = {p: inv_var / total_inv_var for p, inv_var in inv_vars.items()}

    # Assign zero weight to single-trajectory prompts
    for prompt_id in prompt_groups:
        if prompt_id not in prompt_weights_dict:
            prompt_weights_dict[prompt_id] = 0.0

    # Create per-trajectory weight tensor
    prompt_weights = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        prompt_weights[i] = prompt_weights_dict[index[i]]

    return {
        'prompt_weights': prompt_weights,
        'prompt_variances': prompt_variances,
        'prompt_means': prompt_means,
        'prompt_counts': prompt_counts,
    }


def apply_variance_reduction(
    data: 'DataProto',
    use_batch_reweighting: bool = False,
    use_multi_prompt_mvu: bool = False,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Optional[Dict]]:
    """
    Apply variance reduction techniques to advantages.

    Args:
        data: DataProto containing batch data with fields:
              - batch['advantages']: Token-level advantages [batch, seq_len]
              - batch['response_mask']: Valid token mask [batch, seq_len]
              - non_tensor_batch['uid']: Prompt indices for grouping (required for MVU)
              - batch['old_log_probs']: Log probs from training policy (optional, for W(τ))
              - batch['sum_pi_squared']: Sum of π² (optional, for W(τ))
              - batch['rollout_is_weights']: IS weights (optional, for IS correction)
        use_batch_reweighting: Apply MSE-optimal batch reweighting
        use_multi_prompt_mvu: Apply multi-prompt MVU
        epsilon: Numerical stability constant

    Returns:
        Tuple of (modified_advantages, variance_info dict or None)
    """
    if not use_batch_reweighting and not use_multi_prompt_mvu:
        return data.batch['advantages'], None

    # Extract required fields from DataProto
    advantages = data.batch['advantages']
    response_mask = data.batch['response_mask']
    index = data.non_tensor_batch.get('uid', None)

    # Extract fields for W(τ) computation
    # Note: old_log_probs and sum_pi_squared should always be present when
    # actor.compute_sum_pi_squared=True (enabled by default for all algorithms)
    old_log_probs = data.batch.get('old_log_probs', None)
    sum_pi_squared = data.batch.get('sum_pi_squared', None)
    rollout_is_weights = data.batch.get('rollout_is_weights', None)

    # Assert required fields are present for W(τ) computation
    assert old_log_probs is not None, "old_log_probs must be present in batch"
    assert sum_pi_squared is not None, "sum_pi_squared must be present in batch (set actor.compute_sum_pi_squared=True)"

    # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
    pi_t = torch.exp(old_log_probs)
    w_values = 1 - 2 * pi_t + sum_pi_squared

    # Apply rollout importance sampling correction when IS weights present
    # This ensures MVU uses the same IS-corrected W(τ) as optimal baseline
    if rollout_is_weights is not None:
        # Scale W by (IS weight)² to minimize MSE under truncated IS
        w_values = w_values * (rollout_is_weights**2)

    variance_info = {}
    modified_advantages = advantages.clone()
    batch_size = advantages.shape[0]

    if use_batch_reweighting:
        batch_info = compute_batch_reweighting(advantages, response_mask, w_values, epsilon)
        variance_info['batch_reweighting'] = batch_info
        batch_weights_expanded = batch_info['batch_weights'].unsqueeze(-1)
        modified_advantages = modified_advantages * batch_weights_expanded

    if use_multi_prompt_mvu:
        if index is None:
            raise ValueError("index is required for multi-prompt MVU")

        mvu_info = compute_multi_prompt_mvu(advantages, response_mask, index, w_values, epsilon)
        variance_info['multi_prompt_mvu'] = mvu_info

        if not mvu_info.get('skip_reduction', False):
            prompt_weights_expanded = mvu_info['prompt_weights'].unsqueeze(-1)
            # MVU weights sum to 1, scale by batch_size to compensate for loss averaging
            modified_advantages = modified_advantages * prompt_weights_expanded * batch_size

    # Preserve masking in output
    modified_advantages = modified_advantages * response_mask

    return modified_advantages, variance_info
