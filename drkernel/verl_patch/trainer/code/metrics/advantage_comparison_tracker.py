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
Comprehensive tracking module for advantage estimator comparison analysis.

This module tracks and compares any advantage estimation method with RLOO baseline,
computing various metrics including cosine similarity, magnitude comparisons,
variance analysis, and more. This helps understand how different advantage estimators
(GAE, GRPO, optimal_baseline, etc.) behave relative to RLOO.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class AdvantageComparisonTracker:
    """
    Tracks and analyzes any advantage estimator's behavior compared to RLOO baseline.

    This tracker computes two sets of metrics:

    Per-Sequence Metrics (using one representative value per sequence):
    - Avoids length bias by giving equal weight to each sequence
    - More statistically correct for sequence-level advantages (RLOO, optimal_baseline)

    All-Token Metrics (using all valid tokens):
    - Provides overall alignment across all tokens
    - May have length bias but shows complete picture

    Metrics include:
    1. Cosine similarity (directional alignment)
    2. Magnitude ratio (scale comparison)
    3. Variance ratio (variance reduction)
    4. RMSE (overall difference)
    5. Distribution statistics (mean, min, max)

    This is useful for understanding how different advantage estimators
    (GAE, GRPO, optimal_baseline, etc.) compare to the RLOO baseline.
    """

    def __init__(self, enable_detailed_tracking: bool = False):
        """
        Initialize the tracker.

        Args:
            enable_detailed_tracking: If True, keep history for summary stats
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.reset()

    def reset(self):
        """Reset all tracking statistics."""
        self.metrics_history = []

    def compute_rloo_advantages(
        self,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute RLOO advantages for comparison using the core implementation.

        Args:
            token_level_rewards: shape (bs, response_length)
            response_mask: shape (bs, response_length) - same as eos_mask
            index: array of prompt indices

        Returns:
            RLOO advantages tensor
        """
        # Use the core implementation to avoid code duplication
        from verl_patch.trainer.code.ppo.core_algos import compute_rloo_outcome_advantage

        # response_mask is the same as eos_mask (different naming convention)
        advantages, _ = compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index  # response_mask IS eos_mask
        )

        return advantages

    def track_advantages(
        self,
        rloo_advantages: torch.Tensor,
        actual_advantages: torch.Tensor,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
        estimator_name: str = "unknown",
        old_log_probs: Optional[torch.Tensor] = None,
        sum_pi_squared: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Track and compare actual advantages (from any estimator) with RLOO baseline.

        Computes two sets of metrics:

        Per-Sequence Metrics (suffix: _per_sequence):
        - Uses one representative value per sequence
        - Avoids length bias, equal weight per sequence
        - Better for sequence-level advantages (RLOO, optimal_baseline)

        All-Token Metrics (no suffix):
        - Uses all valid tokens
        - May have length bias but shows complete picture

        Essential metrics tracked:
        - Cosine similarity (directional alignment)
        - Magnitude ratio (scale comparison)
        - Variance ratio (variance reduction)
        - RMSE (overall difference)
        - Mean statistics

        Args:
            rloo_advantages: RLOO advantages for comparison
            actual_advantages: Actual advantages being used (bs, response_length)
            token_level_rewards: Token-level rewards (bs, response_length)
            response_mask: Response mask (bs, response_length)
            index: Prompt indices array
            estimator_name: Name of the advantage estimator being used
            old_log_probs: Optional log probabilities (bs, response_length)
            sum_pi_squared: Optional sum of squared probabilities (bs, response_length)

        Returns:
            Dictionary of computed metrics (reduced set)
        """
        with torch.no_grad():
            # Flatten valid advantages for comparison
            valid_mask = response_mask.bool()
            actual_flat = actual_advantages[valid_mask]
            rloo_flat = rloo_advantages[valid_mask]

            metrics = {}

            # Skip if no valid tokens
            if actual_flat.numel() == 0:
                return metrics

            # ESSENTIAL METRICS ONLY
            # All metrics are computed on token-level advantages (flattened valid tokens)

            # 1. Cosine Similarity (most important - shows directional alignment)
            actual_norm = F.normalize(actual_flat.unsqueeze(0), dim=1)
            rloo_norm = F.normalize(rloo_flat.unsqueeze(0), dim=1)
            metrics['cosine_similarity_with_rloo'] = F.cosine_similarity(actual_norm, rloo_norm, dim=1).item()

            # 1b. Per-Sequence Cosine Similarity (using one representative value per sequence)
            # Since advantages are uniform within sequences, take the first valid token per sequence
            bs = actual_advantages.shape[0]
            actual_seq_values = []
            rloo_seq_values = []

            for i in range(bs):
                # Find first valid token in this sequence
                seq_mask = response_mask[i].bool()
                if seq_mask.any():
                    first_valid_idx = seq_mask.nonzero(as_tuple=True)[0][0]
                    actual_seq_values.append(actual_advantages[i, first_valid_idx])
                    rloo_seq_values.append(rloo_advantages[i, first_valid_idx])

            if actual_seq_values:
                actual_seq_tensor = torch.stack(actual_seq_values)
                rloo_seq_tensor = torch.stack(rloo_seq_values)

                # Normalize and compute cosine similarity
                actual_seq_norm = F.normalize(actual_seq_tensor.unsqueeze(0), dim=1)
                rloo_seq_norm = F.normalize(rloo_seq_tensor.unsqueeze(0), dim=1)
                metrics['cosine_similarity_per_sequence_with_rloo'] = F.cosine_similarity(
                    actual_seq_norm, rloo_seq_norm, dim=1
                ).item()

                # Per-sequence magnitude ratio
                actual_seq_l2 = torch.norm(actual_seq_tensor, p=2).item()
                rloo_seq_l2 = torch.norm(rloo_seq_tensor, p=2).item()
                metrics['magnitude_ratio_per_sequence_to_rloo'] = actual_seq_l2 / (rloo_seq_l2 + 1e-8)

                # Per-sequence variance ratio
                actual_seq_var = actual_seq_tensor.var().item() if actual_seq_tensor.numel() > 1 else 0.0
                rloo_seq_var = rloo_seq_tensor.var().item() if rloo_seq_tensor.numel() > 1 else 0.0
                metrics['variance_ratio_per_sequence_to_rloo'] = actual_seq_var / (rloo_seq_var + 1e-8)

                # Per-sequence RMSE
                seq_diff = actual_seq_tensor - rloo_seq_tensor
                metrics['rmse_per_sequence_from_rloo'] = torch.sqrt((seq_diff**2).mean()).item()

                # Per-sequence mean statistics
                metrics['actual_mean_per_sequence'] = actual_seq_tensor.mean().item()
                metrics['rloo_mean_per_sequence'] = rloo_seq_tensor.mean().item()

            # 2. Magnitude Ratio (shows scale changes) - ALL TOKENS
            actual_l2 = torch.norm(actual_flat, p=2).item()
            rloo_l2 = torch.norm(rloo_flat, p=2).item()
            metrics['magnitude_ratio_to_rloo'] = actual_l2 / (rloo_l2 + 1e-8)

            # 3. Variance Ratio (shows variance reduction achieved) - ALL TOKENS
            actual_var = actual_flat.var().item() if actual_flat.numel() > 1 else 0.0
            rloo_var = rloo_flat.var().item() if rloo_flat.numel() > 1 else 0.0
            metrics['variance_ratio_to_rloo'] = actual_var / (rloo_var + 1e-8)

            # 4. Mean and Min/Max (shows distribution shift and range) - ALL TOKENS
            metrics['actual_mean'] = actual_flat.mean().item()
            metrics['rloo_mean'] = rloo_flat.mean().item()
            metrics['actual_min'] = actual_flat.min().item()
            metrics['actual_max'] = actual_flat.max().item()
            metrics['rloo_min'] = rloo_flat.min().item()
            metrics['rloo_max'] = rloo_flat.max().item()

            # 5. RMSE (overall difference metric) - ALL TOKENS
            diff = actual_flat - rloo_flat
            metrics['rmse_from_rloo'] = torch.sqrt((diff**2).mean()).item()

            # 6. Store estimator name (for logging)
            metrics['estimator_name'] = estimator_name

            # Optional: Add W(τ) statistics only for optimal_baseline
            if old_log_probs is not None and sum_pi_squared is not None:
                pi_t = torch.exp(old_log_probs)
                w_values = (1 - 2 * pi_t + sum_pi_squared) * response_mask
                w_sum = w_values.sum(dim=-1)
                metrics['w_mean'] = w_sum.mean().item()

            # Store in lightweight history (only keep recent)
            if self.enable_detailed_tracking:
                self.metrics_history.append(metrics)
                # Keep only last 100 entries to avoid memory growth
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

            return metrics

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get lightweight summary statistics over recent batches.

        Returns:
            Dictionary of aggregated statistics (reduced set)
        """
        if not self.metrics_history:
            return {}

        summary = {}

        # Only aggregate the most important metrics
        key_metrics = [
            # All-token metrics
            'cosine_similarity_with_rloo',
            'magnitude_ratio_to_rloo',
            'variance_ratio_to_rloo',
            'rmse_from_rloo',
            # Per-sequence metrics
            'cosine_similarity_per_sequence_with_rloo',
            'magnitude_ratio_per_sequence_to_rloo',
            'variance_ratio_per_sequence_to_rloo',
            'rmse_per_sequence_from_rloo',
        ]

        for key in key_metrics:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)

        summary['n_batches_tracked'] = len(self.metrics_history)

        return summary
