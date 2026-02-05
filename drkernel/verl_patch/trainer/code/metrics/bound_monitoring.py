# Copyright 2025 ByteDance Ltd. and/or its affiliates
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
Universal Advantage Bound Monitoring for RL Training in LLMs

This module provides universal advantage bound monitoring that works across
all advantage estimation algorithms (GAE, GRPO, RLOO) to provide theoretical
validation without applying constraints.

Key Features:
- Universal monitoring enabled by default with log_metrics=True
- Theoretical validation based on |A^π(s,a)| ≤ 1 - π(a|s) lemma
- Algorithm-agnostic design works with any advantage estimator
- Zero performance impact (monitoring-only, no constraint enforcement)
- Comprehensive 68 metrics including extreme probability monitoring
"""

from typing import Dict

import torch

# Risk threshold constants for low probability detection
LOW_PROB_THRESHOLDS = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9"]
LOW_PROB_VALUES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

# Risk threshold constants for high probability detection
HIGH_PROB_THRESHOLDS = ["0.5", "0.7", "0.9", "0.95", "0.99", "0.999"]
HIGH_PROB_VALUES = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]


def compute_advantage_bound_violations(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute advantage bound violation metrics without applying constraints.

    This function analyzes how often empirical advantages violate the
    theoretical bound |A^π(s,a)| ≤ 1 - π(a|s), providing insights for
    algorithm validation and theoretical compliance monitoring.

    Works universally with all advantage estimation methods:
    - GAE (Generalized Advantage Estimation)
    - GRPO (Group Relative Policy Optimization)
    - RLOO (REINFORCE Leave-One-Out)

    Args:
        advantages: [batch_size, response_length] - Token-level advantages
        log_probs: [batch_size, response_length] - Per-token log probabilities from the policy
                                                     at the beginning of PPO update (old_log_probs)
        response_mask: [batch_size, response_length] - Valid token mask
        epsilon: Small value to prevent numerical issues

    Returns:
        Dict[str, float] - Metrics organized by category:

        Bound Metrics (12):
        - bound/violation_rate: Fraction of tokens violating |A| ≤ 1-π bound
        - bound/mean_violation: Average violation severity
        - bound/max_violation: Maximum violation magnitude
        - bound/positive_violation_rate: Violation rate for positive advantages
        - bound/negative_violation_rate: Violation rate for negative advantages
        - bound/advantage_std: Standard deviation of all advantages
        - bound/positive_count: Number of positive advantage tokens
        - bound/negative_count: Number of negative advantage tokens
        - bound/positive_negative_ratio: Ratio of positive to negative advantages
        - bound/negative_bottom_10pct: Mean of bottom 10% negative advantages
        - bound/positive_mean_bound: Mean uncertainty bound (1-π) for positive advantages
        - bound/negative_mean_bound: Mean uncertainty bound (1-π) for negative advantages

        Risk Monitoring - Low Probability (per trajectory type):
        For both positive and negative advantages:
        - risk/{type}_prob_below_1e-2: Fraction with π < 1e-2 (moderate risk)
        - risk/{type}_prob_below_1e-3: Fraction with π < 1e-3 (high risk)
        - risk/{type}_prob_below_1e-4: Fraction with π < 1e-4 (extreme risk)
        - risk/{type}_prob_below_1e-5: Fraction with π < 1e-5 (ultra extreme risk)
        - risk/{type}_prob_below_1e-6: Fraction with π < 1e-6 (critical risk)
        - risk/{type}_prob_below_1e-7: Fraction with π < 1e-7 (catastrophic risk)
        - risk/{type}_prob_below_1e-8: Fraction with π < 1e-8 (apocalyptic risk)
        - risk/{type}_prob_below_1e-9: Fraction with π < 1e-9 (doomsday risk)
        - risk/{type}_adv_mag_below_1e-X: Mean |A| for tokens with π < 1e-X

        Risk Monitoring - High Probability (per trajectory type):
        For both positive and negative advantages:
        - risk/{type}_prob_above_0.5: Fraction with π > 0.5
        - risk/{type}_prob_above_0.7: Fraction with π > 0.7
        - risk/{type}_prob_above_0.9: Fraction with π > 0.9
        - risk/{type}_prob_above_0.95: Fraction with π > 0.95
        - risk/{type}_prob_above_0.99: Fraction with π > 0.99
        - risk/{type}_prob_above_0.999: Fraction with π > 0.999
        - risk/{type}_adv_mag_above_X: Mean |A| for tokens with π > X

        Note: All probability comparisons are performed in log space for numerical stability.
    """
    # Mask and flatten advantages and log_probs
    adv = advantages[response_mask.bool()]
    log_p = log_probs[response_mask.bool()]

    if len(adv) == 0:
        # Return zero metrics if no valid tokens
        metrics = {
            "bound/violation_rate": 0.0,
            "bound/mean_violation": 0.0,
            "bound/max_violation": 0.0,
            "bound/positive_violation_rate": 0.0,
            "bound/negative_violation_rate": 0.0,
            "bound/advantage_std": 0.0,
            "bound/positive_count": 0.0,
            "bound/negative_count": 0.0,
            "bound/positive_negative_ratio": 0.0,
            "bound/negative_bottom_10pct": 0.0,
            "bound/positive_mean_bound": 0.0,
            "bound/negative_mean_bound": 0.0,
        }

        # Add all risk metrics for both positive and negative trajectories
        for trajectory_type in ["positive", "negative"]:
            for threshold in LOW_PROB_THRESHOLDS:
                metrics[f"risk/{trajectory_type}_prob_below_{threshold}"] = 0.0
                metrics[f"risk/{trajectory_type}_adv_mag_below_{threshold}"] = 0.0
            for threshold in HIGH_PROB_THRESHOLDS:
                metrics[f"risk/{trajectory_type}_prob_above_{threshold}"] = 0.0
                metrics[f"risk/{trajectory_type}_adv_mag_above_{threshold}"] = 0.0

        return metrics

    # Compute probabilities and bounds
    probs = torch.exp(log_p).clamp(min=epsilon, max=1.0)
    bounds = 1.0 - probs  # uncertainty bound: 1 - π(a|s)

    # Check violations: |A| > bound
    advantage_magnitudes = adv.abs()
    violations = advantage_magnitudes > bounds

    # Core bound violation metrics
    metrics = {
        "bound/violation_rate": float(violations.float().mean()),
        "bound/mean_violation": float((advantage_magnitudes - bounds)[violations].mean() if violations.any() else 0.0),
        "bound/max_violation": float(torch.max(advantage_magnitudes - bounds).clamp(min=0)),
    }

    # Positive/negative advantage analysis
    positive_mask = adv > 0
    negative_mask = adv < 0

    # Positive advantages
    if positive_mask.any():
        pos_bounds = bounds[positive_mask]
        pos_violations = advantage_magnitudes[positive_mask] > pos_bounds

        metrics.update(
            {
                "bound/positive_mean_bound": float(pos_bounds.mean()),
                "bound/positive_violation_rate": float(pos_violations.float().mean()),
                "bound/positive_count": int(positive_mask.sum()),
            }
        )
    else:
        metrics.update(
            {
                "bound/positive_mean_bound": 0.0,
                "bound/positive_violation_rate": 0.0,
                "bound/positive_count": 0,
            }
        )

    # Negative advantages
    if negative_mask.any():
        neg_adv = adv[negative_mask]
        neg_bounds = bounds[negative_mask]
        neg_violations = advantage_magnitudes[negative_mask] > neg_bounds

        metrics.update(
            {
                "bound/negative_mean_bound": float(neg_bounds.mean()),
                "bound/negative_violation_rate": float(neg_violations.float().mean()),
                "bound/negative_count": int(negative_mask.sum()),
            }
        )

        # Bottom 10% analysis
        neg_adv_sorted, _ = torch.sort(neg_adv)
        bottom_10_percent_idx = max(1, len(neg_adv_sorted) // 10)
        metrics["bound/negative_bottom_10pct"] = float(neg_adv_sorted[:bottom_10_percent_idx].mean())
    else:
        metrics.update(
            {
                "bound/negative_mean_bound": 0.0,
                "bound/negative_violation_rate": 0.0,
                "bound/negative_count": 0,
                "bound/negative_bottom_10pct": 0.0,
            }
        )

    # Positive/negative ratio
    if negative_mask.any():
        metrics["bound/positive_negative_ratio"] = float(positive_mask.sum()) / float(negative_mask.sum())
    else:
        metrics["bound/positive_negative_ratio"] = float('inf') if positive_mask.any() else 0.0

    # Basic statistics
    metrics["bound/advantage_std"] = float(adv.std())

    # PPO Stability Risk Metrics (Critical)
    # These detect cases where small π can lead to gradient explosion: ∇L ∝ A/π
    # Also track high probability tokens (π > threshold) that may indicate over-confidence

    # Define log thresholds for numerical stability
    log_thresholds_low = [
        (torch.log(torch.tensor(val)), thresh_str) for val, thresh_str in zip(LOW_PROB_VALUES, LOW_PROB_THRESHOLDS)
    ]

    log_thresholds_high = [
        (torch.log(torch.tensor(val)), thresh_str) for val, thresh_str in zip(HIGH_PROB_VALUES, HIGH_PROB_THRESHOLDS)
    ]

    # Process negative advantage tokens
    if negative_mask.any():
        # Work in log space for better numerical precision
        neg_log_probs = log_p[negative_mask]
        neg_advantages = adv[negative_mask]

        # Monitor low probability thresholds for negative advantage tokens
        for log_threshold, threshold_str in log_thresholds_low:
            mask = neg_log_probs <= log_threshold
            metrics[f"risk/negative_prob_below_{threshold_str}"] = float(mask.float().mean())
            metrics[f"risk/negative_adv_mag_below_{threshold_str}"] = float(
                neg_advantages[mask].abs().mean() if mask.any() else 0.0
            )

        # Monitor high probability thresholds for negative advantage tokens
        for log_threshold, threshold_str in log_thresholds_high:
            mask = neg_log_probs >= log_threshold
            metrics[f"risk/negative_prob_above_{threshold_str}"] = float(mask.float().mean())
            metrics[f"risk/negative_adv_mag_above_{threshold_str}"] = float(
                neg_advantages[mask].abs().mean() if mask.any() else 0.0
            )
    else:
        # Initialize all negative metrics to 0
        for _, threshold_str in log_thresholds_low:
            metrics[f"risk/negative_prob_below_{threshold_str}"] = 0.0
            metrics[f"risk/negative_adv_mag_below_{threshold_str}"] = 0.0
        for _, threshold_str in log_thresholds_high:
            metrics[f"risk/negative_prob_above_{threshold_str}"] = 0.0
            metrics[f"risk/negative_adv_mag_above_{threshold_str}"] = 0.0

    # Process positive advantage tokens
    if positive_mask.any():
        # Work in log space for better numerical precision
        pos_log_probs = log_p[positive_mask]
        pos_advantages = adv[positive_mask]

        # Monitor low probability thresholds for positive advantage tokens
        for log_threshold, threshold_str in log_thresholds_low:
            mask = pos_log_probs <= log_threshold
            metrics[f"risk/positive_prob_below_{threshold_str}"] = float(mask.float().mean())
            metrics[f"risk/positive_adv_mag_below_{threshold_str}"] = float(
                pos_advantages[mask].abs().mean() if mask.any() else 0.0
            )

        # Monitor high probability thresholds for positive advantage tokens
        for log_threshold, threshold_str in log_thresholds_high:
            mask = pos_log_probs >= log_threshold
            metrics[f"risk/positive_prob_above_{threshold_str}"] = float(mask.float().mean())
            metrics[f"risk/positive_adv_mag_above_{threshold_str}"] = float(
                pos_advantages[mask].abs().mean() if mask.any() else 0.0
            )
    else:
        # Initialize all positive metrics to 0
        for _, threshold_str in log_thresholds_low:
            metrics[f"risk/positive_prob_below_{threshold_str}"] = 0.0
            metrics[f"risk/positive_adv_mag_below_{threshold_str}"] = 0.0
        for _, threshold_str in log_thresholds_high:
            metrics[f"risk/positive_prob_above_{threshold_str}"] = 0.0
            metrics[f"risk/positive_adv_mag_above_{threshold_str}"] = 0.0

    return metrics


def check_bound_alerts(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Check violation metrics against warning/critical thresholds.

    Args:
        metrics: Dictionary of metrics from compute_advantage_bound_violations

    Returns:
        Dictionary mapping metric names to alert levels ('warning' or 'critical')
    """
    thresholds = {
        "bound/violation_rate": {"warning": 0.3, "critical": 0.5},
        "bound/mean_violation": {"warning": 0.1, "critical": 0.2},
        "bound/positive_violation_rate": {"warning": 0.35, "critical": 0.55},
        "bound/negative_violation_rate": {"warning": 0.35, "critical": 0.55},
        "bound/negative_bottom_10pct": {"warning": -3.0, "critical": -5.0},
    }

    # Add thresholds for low probability metrics (both positive and negative)
    for trajectory_type in ["positive", "negative"]:
        thresholds.update(
            {
                f"risk/{trajectory_type}_prob_below_1e-2": {"warning": 0.15, "critical": 0.3},
                f"risk/{trajectory_type}_prob_below_1e-3": {"warning": 0.05, "critical": 0.15},
                f"risk/{trajectory_type}_prob_below_1e-4": {"warning": 0.01, "critical": 0.05},
                f"risk/{trajectory_type}_prob_below_1e-5": {"warning": 0.005, "critical": 0.02},
                f"risk/{trajectory_type}_prob_below_1e-6": {"warning": 0.001, "critical": 0.01},
                f"risk/{trajectory_type}_prob_below_1e-7": {"warning": 0.0005, "critical": 0.005},
                f"risk/{trajectory_type}_prob_below_1e-8": {"warning": 0.0001, "critical": 0.001},
                f"risk/{trajectory_type}_prob_below_1e-9": {"warning": 0.00005, "critical": 0.0005},
            }
        )

    # Add thresholds for high probability metrics (both positive and negative)
    for trajectory_type in ["positive", "negative"]:
        thresholds.update(
            {
                f"risk/{trajectory_type}_prob_above_0.5": {"warning": 0.3, "critical": 0.5},
                f"risk/{trajectory_type}_prob_above_0.7": {"warning": 0.2, "critical": 0.35},
                f"risk/{trajectory_type}_prob_above_0.9": {"warning": 0.1, "critical": 0.2},
                f"risk/{trajectory_type}_prob_above_0.95": {"warning": 0.05, "critical": 0.1},
                f"risk/{trajectory_type}_prob_above_0.99": {"warning": 0.02, "critical": 0.05},
                f"risk/{trajectory_type}_prob_above_0.999": {"warning": 0.005, "critical": 0.02},
            }
        )

    alerts = {}

    for metric_name, threshold in thresholds.items():
        if metric_name in metrics:
            value = metrics[metric_name]

            # Handle negative thresholds (for negative_bottom_10pct)
            if "negative_bottom_10pct" in metric_name:
                if value < threshold["critical"]:
                    alerts[metric_name] = "critical"
                elif value < threshold["warning"]:
                    alerts[metric_name] = "warning"
            else:
                # Normal thresholds (higher is worse)
                if value > threshold["critical"]:
                    alerts[metric_name] = "critical"
                elif value > threshold["warning"]:
                    alerts[metric_name] = "warning"

    return alerts
