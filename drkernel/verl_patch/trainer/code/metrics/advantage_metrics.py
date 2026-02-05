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
Comprehensive Advantage Metrics for RL Training in LLMs

This module provides advanced metrics for monitoring and diagnosing advantage computation
in Reinforcement Learning training for LLMs, with a focus on GRPO/RLOO
algorithms and bias detection.

Key Features:
- Zero-sum constraint validation for GRPO/RLOO
- Length bias detection via correlation
- Group-wise analysis for prompt-level validation
- Comprehensive statistical analysis
"""

import warnings
from collections import defaultdict
from typing import Dict, Optional, Union

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl import DataProto


def compute_verification_metrics(
    per_response_advantages: torch.Tensor, group_ids: Optional[Union[list, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute verification metrics for GRPO/RLOO zero-sum constraint validation.

    Args:
        per_response_advantages: Per-response advantage values [batch_size]
        group_ids: Optional group identifiers for per-group validation

    Returns:
        Dictionary containing verification metrics
    """
    metrics = {}

    # Core verification: batch-level zero-sum
    zero_sum_mean = per_response_advantages.mean().item()
    metrics["zero_sum_mean"] = zero_sum_mean

    # Group-wise zero-sum validation
    if group_ids is not None and len(group_ids) > 0:
        per_response_adv_np = per_response_advantages.cpu().numpy()

        # Validate group_ids length matches batch size
        if len(group_ids) != len(per_response_adv_np):
            warnings.warn(f"Group IDs length ({len(group_ids)}) doesn't match batch size ({len(per_response_adv_np)})")
            return metrics

        group_sums = defaultdict(float)

        for i, gid in enumerate(group_ids):
            group_sums[gid] += per_response_adv_np[i]

        if len(group_sums) > 0:
            group_sum_values = np.array(list(group_sums.values()))
            metrics.update(
                {
                    "group_sum_mean": float(group_sum_values.mean()),
                    "group_sum_std": float(group_sum_values.std()),
                    "group_sum_max": float(group_sum_values.max()),
                    "group_sum_min": float(group_sum_values.min()),
                    "group_count": len(group_sum_values),
                }
            )

    return metrics


def compute_distribution_metrics(
    per_response_advantages: torch.Tensor, all_token_advantages: torch.Tensor
) -> Dict[str, float]:
    """
    Compute distribution and spread metrics for advantages.

    Args:
        per_response_advantages: Per-response advantage values [batch_size]
        all_token_advantages: All valid token advantages [total_valid_tokens]

    Returns:
        Dictionary containing distribution metrics
    """
    metrics = {}

    # Per-response statistics
    metrics.update(
        {
            "per_response_mean": per_response_advantages.mean().item(),
            "per_response_min": per_response_advantages.min().item(),
            "per_response_max": per_response_advantages.max().item(),
        }
    )

    # Only compute std if we have more than one response
    if per_response_advantages.numel() > 1:
        metrics["per_response_std"] = per_response_advantages.std().item()
    else:
        metrics["per_response_std"] = 0.0

    # Per-token statistics
    if all_token_advantages.numel() > 0:
        # Note: per_token_mean/min/max removed as they duplicate critic/advantages/mean/min/max
        # Only compute std if we have more than one element
        if all_token_advantages.numel() > 1:
            metrics["per_token_std"] = all_token_advantages.std().item()
        else:
            metrics["per_token_std"] = 0.0

        # Signal balance
        per_response_adv = per_response_advantages.cpu().numpy()
        metrics.update(
            {
                "frac_pos": float((per_response_adv > 0).mean()),
                "frac_neg": float((per_response_adv < 0).mean()),
                "frac_zero": float((per_response_adv == 0).mean()),
            }
        )

        # Stability metrics - detect outliers as deviations from mean
        outlier_threshold = 3.0
        advantage_mean = all_token_advantages.mean()
        advantage_std = all_token_advantages.std()
        outlier_mask = (all_token_advantages - advantage_mean).abs() > outlier_threshold * advantage_std
        metrics["outlier_ratio"] = float(outlier_mask.float().mean())

    return metrics


def compute_bias_metrics(
    per_response_advantages: torch.Tensor,
    response_lengths: torch.Tensor,
    per_response_rewards: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute bias detection metrics, particularly length bias.

    Args:
        per_response_advantages: Per-response advantage values [batch_size]
        response_lengths: Response lengths [batch_size]
        per_response_rewards: Optional per-response rewards [batch_size]

    Returns:
        Dictionary containing bias metrics
    """
    metrics = {}

    per_response_adv_np = per_response_advantages.cpu().numpy()
    response_lengths_np = response_lengths.cpu().numpy()

    # Length correlation - the main metric for length bias detection
    try:
        from scipy.stats import pearsonr

        if len(per_response_adv_np) > 1:
            # Check for variance in both arrays
            if np.std(response_lengths_np) > 1e-8 and np.std(per_response_adv_np) > 1e-8:
                corr, pval = pearsonr(response_lengths_np, per_response_adv_np)
                if not np.isnan(corr):
                    metrics["length_corr"] = float(corr)
                    metrics["length_corr_pvalue"] = float(pval)
                else:
                    metrics["length_corr"] = 0.0
                    metrics["length_corr_pvalue"] = 1.0
            else:
                # No variance means no correlation
                metrics["length_corr"] = 0.0
                metrics["length_corr_pvalue"] = 1.0
        else:
            metrics["length_corr"] = 0.0
            metrics["length_corr_pvalue"] = 1.0
    except ImportError:
        warnings.warn("scipy not available for correlation computation")
        metrics["length_corr"] = 0.0
        metrics["length_corr_pvalue"] = 1.0

    # Reward correlation (only if provided)
    if per_response_rewards is not None:
        per_response_rewards_np = per_response_rewards.cpu().numpy()
        try:
            from scipy.stats import pearsonr

            if len(per_response_rewards_np) > 1 and np.std(per_response_rewards_np) > 1e-8:
                corr, pval = pearsonr(per_response_rewards_np, per_response_adv_np)
                if not np.isnan(corr):
                    metrics["reward_corr"] = float(corr)
                    metrics["reward_corr_pvalue"] = float(pval)
        except ImportError:
            pass

    return metrics


def compute_learning_dynamics_metrics(
    advantages: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute metrics related to learning dynamics and value function quality.

    Args:
        advantages: Advantage tensor [batch_size, seq_len]
        values: Value function estimates [batch_size, seq_len]
        returns: Return values [batch_size, seq_len]
        response_mask: Response mask [batch_size, seq_len]

    Returns:
        Dictionary containing learning dynamics metrics
    """
    metrics = {}

    if values is None or returns is None or response_mask is None:
        return metrics

    # Extract valid values (ensure mask is boolean for PyTorch 2.7+)
    valid_returns = torch.masked_select(returns, response_mask.bool())
    valid_values = torch.masked_select(values, response_mask.bool())

    if valid_returns.numel() <= 1:
        return metrics

    # Value function explained variance
    return_var = torch.var(valid_returns)
    if return_var > 1e-8:
        residual_var = torch.var(valid_returns - valid_values)
        vf_explained_var = 1.0 - residual_var / return_var
        metrics["vf_explained_var"] = float(vf_explained_var)

        # Value function bias
        vf_bias = (valid_values - valid_returns).mean()
        metrics["vf_bias"] = float(vf_bias)

        # Value function RMSE
        vf_rmse = torch.sqrt(torch.mean((valid_values - valid_returns) ** 2))
        metrics["vf_rmse"] = float(vf_rmse)

    return metrics


def compute_comprehensive_advantage_metrics(
    batch: DataProto, include_learning_dynamics: bool = True, include_group_analysis: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive advantage metrics for RL training monitoring.

    This is the main function that computes all advantage-related metrics
    for monitoring and diagnosing RL training in LLMs.

    Args:
        batch: DataProto object containing batch data
        include_learning_dynamics: Whether to include value function metrics
        include_group_analysis: Whether to include group-wise analysis

    Returns:
        Dictionary containing all computed metrics with consistent naming
    """
    try:
        # Extract core data with validation
        if "advantages" not in batch.batch:
            warnings.warn("Batch missing 'advantages' key")
            return {}
        if "responses" not in batch.batch:
            warnings.warn("Batch missing 'responses' key")
            return {}
        if "attention_mask" not in batch.batch:
            warnings.warn("Batch missing 'attention_mask' key")
            return {}

        advantages = batch.batch["advantages"]

        # Get response mask - prefer precomputed mask, otherwise compute it
        if 'response_mask' in batch.batch:
            response_mask = batch.batch['response_mask']
        else:
            # Fallback: compute response mask if not available
            max_response_length = batch.batch["responses"].shape[-1]
            response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

        response_lengths = response_mask.sum(dim=-1).float()

        # Check if we have valid data
        if response_mask.sum() == 0:
            warnings.warn("No valid tokens found in response_mask")
            return {}

        # Prepare data for metrics
        per_response_adv = verl_F.masked_mean(advantages, response_mask, axis=-1)
        all_token_adv = torch.masked_select(advantages, response_mask.bool())

        # Prepare optional data
        group_ids = None
        if include_group_analysis:
            group_ids = batch.non_tensor_batch.get("uid")

        per_response_rewards = None
        if "token_level_rewards" in batch.batch:
            per_response_rewards = verl_F.masked_mean(batch.batch["token_level_rewards"], response_mask, axis=-1)

        # Compute all metric categories
        all_metrics = {}

        # 1. Verification metrics
        verification_metrics = compute_verification_metrics(per_response_adv, group_ids)
        for k, v in verification_metrics.items():
            all_metrics[f"verification_{k}"] = v

        # 2. Distribution metrics
        distribution_metrics = compute_distribution_metrics(per_response_adv, all_token_adv)
        for k, v in distribution_metrics.items():
            all_metrics[f"distribution_{k}"] = v

        # 3. Bias metrics
        bias_metrics = compute_bias_metrics(per_response_adv, response_lengths, per_response_rewards)
        for k, v in bias_metrics.items():
            all_metrics[f"bias_{k}"] = v

        # 4. Learning dynamics metrics
        if include_learning_dynamics and "values" in batch.batch and "returns" in batch.batch:
            learning_metrics = compute_learning_dynamics_metrics(
                advantages, batch.batch["values"], batch.batch["returns"], response_mask
            )
            for k, v in learning_metrics.items():
                all_metrics[f"learning_{k}"] = v

        return all_metrics

    except Exception as e:
        warnings.warn(f"Error computing advantage metrics: {e}")
        return {}


def get_metric_thresholds() -> Dict[str, Dict[str, float]]:
    """
    Get predefined threshold values for metrics alerts.

    Returns:
        Dictionary mapping metric names to threshold dictionaries
    """
    return {
        "verification_zero_sum_mean": {"warning": 1e-6, "critical": 1e-4},
        "bias_length_corr": {"warning": 0.3, "critical": 0.5},
        "distribution_outlier_ratio": {"warning": 0.01, "critical": 0.05},
        "learning_vf_explained_var": {"warning": 0.5, "critical": 0.3},
    }


def check_metric_alerts(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Check metrics against thresholds and return alerts.

    Args:
        metrics: Dictionary of computed metrics

    Returns:
        Dictionary mapping metric names to alert levels ("warning" or "critical")
    """
    thresholds = get_metric_thresholds()
    alerts = {}

    for metric_name, value in metrics.items():
        if metric_name in thresholds:
            threshold = thresholds[metric_name]

            # Handle different alert patterns
            if "zero_sum_mean" in metric_name or "length_corr" in metric_name:
                if abs(value) > threshold["critical"]:
                    alerts[metric_name] = "critical"
                elif abs(value) > threshold["warning"]:
                    alerts[metric_name] = "warning"
            elif "outlier_ratio" in metric_name:
                if value > threshold["critical"]:
                    alerts[metric_name] = "critical"
                elif value > threshold["warning"]:
                    alerts[metric_name] = "warning"
            elif "vf_explained_var" in metric_name:
                if value < threshold["critical"]:
                    alerts[metric_name] = "critical"
                elif value < threshold["warning"]:
                    alerts[metric_name] = "warning"

    return alerts
