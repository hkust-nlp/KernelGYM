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
Unified Metrics System for RL Training

Clear, simple interface for all training metrics with extensible architecture.
Replaces all embedded metrics computation with a unified system.

Design:
- Single entry point: compute_all_metrics()
- Clear I/O specification
- Easy to extend with new metrics
- Backward compatible with existing code
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl import DataProto


@dataclass
class MetricsContext:
    """
    Context information for metrics computation.

    This provides all the extra information that metrics might need
    beyond what's in the DataProto batch.
    """

    # Training context
    use_critic: bool = True
    global_step: Optional[int] = None

    # Extra rewards (if any)
    extra_rewards_info: Optional[Dict[str, Any]] = None

    # Timing information
    timing_raw: Optional[Dict[str, float]] = None
    n_gpus: Optional[int] = None

    # Monitoring metadata stored in batch.meta_info
    monitoring_metrics: Optional[Dict[str, float]] = None

    # For Proxy 3: batch gradient norm squared (if available from training loop)
    batch_gradient_norm_squared: Optional[float] = None

    # Policy output from compute_policy_loss (contains rollout_is_weights)
    policy_output: Optional[Any] = None  # PolicyOutput from core_algos.py


class UnifiedMetricsSystem:
    """
    Unified system for computing all training metrics.

    This system computes ALL metrics that were previously scattered
    across different functions and provides a clean interface.
    """

    def __init__(self):
        self.enabled_categories = {
            'core': True,  # Basic score/reward/advantage metrics
            'length': True,  # Response/prompt length metrics
            'rl_advanced': True,  # Advanced RL metrics (zero-sum, bias, bound monitoring)
            'monitoring': True,  # Custom monitoring metrics
            'timing': True,  # Performance timing metrics
            'throughput': True,  # Throughput analysis
            'prompt': True,  # Prompt-level coverage and entropy metrics
            'variance_proxy': True,  # Variance proxy metrics using W-score
            'mismatch': True,  # Training-inference mismatch metrics (vllm-kl, PPL)
        }

        # Inter-batch variance tracking with EMA
        self.inter_batch_variance_ema = None
        self.inter_batch_ema_beta = 0.99  # Smoothing factor for EMA (configurable)

    def compute_all_metrics(self, batch: DataProto, context: MetricsContext = None) -> Dict[str, float]:
        """
        Compute ALL metrics for the training batch.

        Args:
            batch: Training batch data
            context: Additional context information

        Returns:
            Dictionary with all metrics, properly prefixed and organized

        Example:
            ```python
            # Replace all existing metrics computation with:
            context = MetricsContext(
                use_critic=True,
                extra_rewards_info=extra_rewards_info,
                timing_raw=timing_raw,
                n_gpus=n_gpus
            )
            all_metrics = metrics_system.compute_all_metrics(batch, context)
            ```
        """
        if context is None:
            context = MetricsContext()

        all_metrics = {}

        # Core metrics (basic RL signals)
        if self.enabled_categories['core']:
            core_metrics = self._compute_core_metrics(batch, context)
            all_metrics.update(core_metrics)

        # Length metrics
        if self.enabled_categories['length']:
            length_metrics = self._compute_length_metrics(batch, context)
            all_metrics.update(length_metrics)

        # Advanced RL metrics (zero-sum, bias detection, etc.)
        if self.enabled_categories['rl_advanced']:
            rl_metrics = self._compute_rl_advanced_metrics(batch, context)
            all_metrics.update(rl_metrics)

        # Monitoring metrics (bound violations, etc.)
        if self.enabled_categories['monitoring']:
            monitoring_metrics = self._compute_monitoring_metrics(batch, context)
            all_metrics.update(monitoring_metrics)

        # Prompt-level metrics
        if self.enabled_categories['prompt']:
            prompt_metrics = self._compute_prompt_metrics(batch, context)
            all_metrics.update(prompt_metrics)

        # Extra rewards metrics
        if context.extra_rewards_info:
            extra_metrics = self._compute_extra_rewards_metrics(context.extra_rewards_info)
            all_metrics.update(extra_metrics)

        # Timing metrics
        if self.enabled_categories['timing'] and context.timing_raw:
            timing_metrics = self._compute_timing_metrics(batch, context)
            all_metrics.update(timing_metrics)

        # Throughput metrics
        if self.enabled_categories['throughput'] and context.timing_raw and context.n_gpus:
            throughput_metrics = self._compute_throughput_metrics(batch, context)
            all_metrics.update(throughput_metrics)

        # Advantage comparison metrics (compare final advantages with RLOO)
        if self.enabled_categories.get('advantage_comparison', True):
            advantage_comp_metrics = self._compute_advantage_comparison_metrics(batch, context)
            all_metrics.update(advantage_comp_metrics)

        # Variance proxy metrics (if W-score data is available)
        if self.enabled_categories.get('variance_proxy', True):
            variance_proxy_metrics = self._compute_variance_proxy_metrics(batch, context)
            all_metrics.update(variance_proxy_metrics)

        # Note: Training-inference mismatch metrics (KL, PPL, etc.) are now computed
        # centrally in ray_trainer.py via compute_rollout_importance_weights_and_add_to_batch()
        # to avoid duplication and ensure consistency across the training pipeline.

        return all_metrics

    def configure(self, **categories):
        """
        Configure which metric categories to compute.

        Args:
            **categories: Boolean flags for each category

        Example:
            metrics_system.configure(
                core=True,
                rl_advanced=True,
                timing=False  # Disable timing metrics
            )
        """
        for category, enabled in categories.items():
            if category in self.enabled_categories:
                self.enabled_categories[category] = enabled

    def _compute_core_metrics(self, batch: DataProto, context: MetricsContext) -> Dict[str, float]:
        """Core metrics that replace compute_data_metrics()."""
        metrics = {}

        # Score metrics
        if 'token_level_scores' in batch.batch:
            sequence_score = batch.batch['token_level_scores'].sum(-1)
            metrics.update(
                {
                    'critic/score/mean': torch.mean(sequence_score).detach().item(),
                    'critic/score/max': torch.max(sequence_score).detach().item(),
                    'critic/score/min': torch.min(sequence_score).detach().item(),
                }
            )
        
        if "shaped_turn_rewards" in batch.batch:
            shaped_turn_rewards = batch.batch["shaped_turn_rewards"]
            metrics.update(
                {
                    'critic/shaped_turn_rewards/mean': torch.mean(shaped_turn_rewards).detach().item(),
                    'critic/shaped_turn_rewards/max': torch.max(shaped_turn_rewards).detach().item(),
                    'critic/shaped_turn_rewards/min': torch.min(shaped_turn_rewards).detach().item(),
                }
            )

        # Reward metrics
        if 'token_level_rewards' in batch.batch:
            sequence_reward = batch.batch['token_level_rewards'].sum(-1)
            metrics.update(
                {
                    'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
                    'critic/rewards/max': torch.max(sequence_reward).detach().item(),
                    'critic/rewards/min': torch.min(sequence_reward).detach().item(),
                }
            )

        # Advantage metrics
        if 'advantages' in batch.batch:
            advantages = batch.batch['advantages']
            response_mask = self._get_response_mask(batch, advantages.shape[-1])
            valid_adv = advantages[response_mask.bool()]

            if len(valid_adv) > 0:
                metrics.update(
                    {
                        'critic/advantages/mean': torch.mean(valid_adv).detach().item(),
                        'critic/advantages/max': torch.max(valid_adv).detach().item(),
                        'critic/advantages/min': torch.min(valid_adv).detach().item(),
                    }
                )

        # Returns metrics
        if 'returns' in batch.batch:
            returns = batch.batch['returns']
            response_mask = self._get_response_mask(batch, returns.shape[-1])
            valid_returns = returns[response_mask.bool()]

            if len(valid_returns) > 0:
                metrics.update(
                    {
                        'critic/returns/mean': torch.mean(valid_returns).detach().item(),
                        'critic/returns/max': torch.max(valid_returns).detach().item(),
                        'critic/returns/min': torch.min(valid_returns).detach().item(),
                    }
                )

        # Value function metrics (if using critic)
        if context.use_critic and 'values' in batch.batch and 'returns' in batch.batch:
            values = batch.batch['values']
            returns = batch.batch['returns']
            response_mask = self._get_response_mask(batch, values.shape[-1])

            valid_values = values[response_mask.bool()]
            valid_returns = returns[response_mask.bool()]

            if len(valid_values) > 0 and len(valid_returns) > 0:
                return_diff_var = torch.var(valid_returns - valid_values)
                return_var = torch.var(valid_returns)

                metrics.update(
                    {
                        'critic/values/mean': torch.mean(valid_values).detach().item(),
                        'critic/values/max': torch.max(valid_values).detach().item(),
                        'critic/values/min': torch.min(valid_values).detach().item(),
                        # Explained variance - show actual value even if negative
                        'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
                    }
                )

        return metrics

    def _compute_length_metrics(self, batch: DataProto, _context: MetricsContext) -> Dict[str, float]:
        """Response and prompt length metrics."""
        metrics = {}

        try:
            from verl.trainer.ppo.metric_utils import _compute_response_info

            response_info = _compute_response_info(batch)

            prompt_length = response_info['prompt_length']
            response_length = response_info['response_length']

            # Response length metrics
            max_response_length = (
                batch.batch['responses'].shape[-1] if 'responses' in batch.batch else response_length.max()
            )

            metrics.update(
                {
                    'response_length/mean': torch.mean(response_length).detach().item(),
                    'response_length/max': torch.max(response_length).detach().item(),
                    'response_length/min': torch.min(response_length).detach().item(),
                    'response_length/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float())
                    .detach()
                    .item(),
                }
            )

            # Prompt length metrics
            if 'attention_mask' in batch.batch:
                max_prompt_length = batch.batch['attention_mask'].shape[-1] - max_response_length

                metrics.update(
                    {
                        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
                        'prompt_length/max': torch.max(prompt_length).detach().item(),
                        'prompt_length/min': torch.min(prompt_length).detach().item(),
                        'prompt_length/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float())
                        .detach()
                        .item(),
                    }
                )

            # Add effective response metrics
            from .prompt_metrics import compute_effective_response_metrics

            effective_metrics = compute_effective_response_metrics(batch, max_response_length)
            metrics.update(effective_metrics)

        except ImportError:
            # Fallback implementation
            pass

        return metrics

    def _compute_rl_advanced_metrics(self, batch: DataProto, context: MetricsContext) -> Dict[str, float]:
        """
        Advanced RL-specific metrics including advantage analysis and bound monitoring.

        This includes:
        - Zero-sum verification for GRPO/RLOO
        - Length bias detection
        - Advantage distribution analysis
        - Value function quality metrics
        - Advantage bound violation monitoring (15 metrics)
        """
        prefixed_metrics = {}

        try:
            # 1. Advantage metrics
            from .advantage_metrics import compute_comprehensive_advantage_metrics

            comprehensive_metrics = compute_comprehensive_advantage_metrics(
                batch, include_learning_dynamics=context.use_critic, include_group_analysis=True
            )

            # Add advantage/ prefix to all metrics
            for key, value in comprehensive_metrics.items():
                prefixed_metrics[f'advantage/{key}'] = value

            # 2. Bound monitoring metrics
            bound_metrics = self._compute_bound_monitoring_metrics(batch)
            prefixed_metrics.update(bound_metrics)

        except ImportError:
            pass

        return prefixed_metrics

    def _compute_monitoring_metrics(self, batch: DataProto, _context: MetricsContext) -> Dict[str, float]:
        """Monitoring metrics from meta_info."""
        metrics = {}

        # Add monitoring metrics if available in meta_info
        if 'monitoring_metrics' in batch.meta_info:
            monitoring_metrics = batch.meta_info['monitoring_metrics']
            if isinstance(monitoring_metrics, dict):
                # Add monitoring/ prefix to all custom metrics
                for key, value in monitoring_metrics.items():
                    if not key.startswith('monitoring/'):
                        metrics[f'monitoring/{key}'] = value
                    else:
                        metrics[key] = value

        return metrics

    def _compute_prompt_metrics(self, batch: DataProto, _context: MetricsContext) -> Dict[str, float]:
        """Prompt-level coverage and entropy metrics."""
        metrics = {}

        try:
            from .prompt_metrics import compute_prompt_coverage_metrics

            # Only compute if we have the required data
            if 'token_level_rewards' in batch.batch and 'uid' in batch.non_tensor_batch:
                coverage_metrics = compute_prompt_coverage_metrics(batch)
                metrics.update(coverage_metrics)
        except Exception:
            # Skip if computation fails
            pass

        return metrics

    def _compute_extra_rewards_metrics(self, extra_rewards_info: Dict[str, Any]) -> Dict[str, float]:
        """Extra rewards metrics."""
        metrics = {}

        for key, sequence_extra in extra_rewards_info.items():
            metrics.update(
                {
                    f'critic/rewards_extra/{key}/mean': np.mean(sequence_extra),
                    f'critic/rewards_extra/{key}/max': np.max(sequence_extra),
                    f'critic/rewards_extra/{key}/min': np.min(sequence_extra),
                    f'critic/rewards_extra/{key}/var': np.var(sequence_extra),
                }
            )

        return metrics

    def _compute_timing_metrics(self, batch: DataProto, context: MetricsContext) -> Dict[str, float]:
        """Timing metrics."""
        try:
            # Import response info helper
            from verl.trainer.ppo.metric_utils import _compute_response_info

            response_info = _compute_response_info(batch)
            num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
            num_response_tokens = torch.sum(response_info['response_length']).item()
            num_overall_tokens = num_prompt_tokens + num_response_tokens

            num_tokens_of_section = {
                'gen': num_response_tokens,
                **{name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']},
            }

            timing_metrics = {}
            # Add basic timing in seconds
            for name, value in context.timing_raw.items():
                timing_metrics[f'timing_s/{name}'] = value

            # Add per-token timing in milliseconds
            for name in set(num_tokens_of_section.keys()) & set(context.timing_raw.keys()):
                timing_metrics[f'timing_per_token_ms/{name}'] = (
                    context.timing_raw[name] * 1000 / num_tokens_of_section[name]
                )

            return timing_metrics

        except ImportError:
            # Fallback to basic timing
            return {f'timing_s/{name}': value for name, value in context.timing_raw.items()}

    def _compute_throughput_metrics(self, batch: DataProto, context: MetricsContext) -> Dict[str, float]:
        """Throughput metrics."""
        try:
            from verl.trainer.ppo.metric_utils import compute_throughout_metrics

            return compute_throughout_metrics(batch, context.timing_raw, context.n_gpus)
        except (ImportError, KeyError, AttributeError):
            # Throughput computation requires specific meta_info that may not be available
            return {}

    def _compute_bound_monitoring_metrics(self, batch: DataProto) -> Dict[str, float]:
        """
        Compute bound monitoring metrics for both old_log_probs and rollout_log_probs.

        Args:
            batch: Training batch containing advantages and log probabilities

        Returns:
            Dictionary of bound monitoring metrics with appropriate prefixes
        """
        metrics = {}

        # Early return if no advantages to monitor
        if 'advantages' not in batch.batch:
            return metrics

        # Import here to avoid circular dependency
        from .bound_monitoring import compute_advantage_bound_violations

        # Get response mask - prefer precomputed, otherwise compute
        response_mask = self._extract_response_mask(batch)
        if response_mask is None:
            return metrics

        advantages = batch.batch['advantages']

        # Monitor bounds for old_log_probs (policy at beginning of PPO update)
        if 'old_log_probs' in batch.batch:
            old_bound_metrics = compute_advantage_bound_violations(
                advantages=advantages,
                log_probs=batch.batch['old_log_probs'],
                response_mask=response_mask,
            )
            metrics.update(old_bound_metrics)

        # Monitor bounds for rollout_log_probs (from vLLM rollout when available)
        if 'rollout_log_probs' in batch.batch:
            rollout_bound_metrics = compute_advantage_bound_violations(
                advantages=advantages,
                log_probs=batch.batch['rollout_log_probs'],
                response_mask=response_mask,
            )
            # Prefix rollout metrics to distinguish from policy metrics
            metrics.update(self._prefix_rollout_metrics(rollout_bound_metrics))

        return metrics

    def _extract_response_mask(self, batch: DataProto) -> Optional[torch.Tensor]:
        """
        Extract or compute response mask from batch.

        Args:
            batch: Training batch

        Returns:
            Response mask tensor or None if cannot be determined
        """
        # Prefer precomputed mask
        if 'response_mask' in batch.batch:
            return batch.batch['response_mask']

        # Fallback to computing from attention mask
        if 'attention_mask' not in batch.batch:
            return None

        if 'responses' in batch.batch:
            max_response_length = batch.batch['responses'].shape[-1]
            return batch.batch['attention_mask'][:, -max_response_length:].bool()
        else:
            return batch.batch['attention_mask'].bool()

    def _prefix_rollout_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Add appropriate prefixes to rollout metrics.

        Args:
            metrics: Dictionary of metrics to prefix

        Returns:
            Dictionary with prefixed metric names
        """
        prefixed = {}
        for key, value in metrics.items():
            if key.startswith('bound/'):
                new_key = key.replace('bound/', 'rollout_bound/', 1)
            elif key.startswith('risk/'):
                new_key = key.replace('risk/', 'rollout_risk/', 1)
            else:
                new_key = f'rollout_{key}'
            prefixed[new_key] = value
        return prefixed

    def _get_response_mask(self, batch: DataProto, response_length: int) -> torch.Tensor:
        """Get response mask from batch."""
        if 'response_mask' in batch.batch:
            return batch.batch['response_mask']
        elif 'attention_mask' in batch.batch:
            return batch.batch['attention_mask'][:, -response_length:]
        else:
            # Get device and dtype from any existing tensor in batch
            device = torch.device('cpu')
            dtype = torch.float32
            for key, value in batch.batch.items():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    dtype = (
                        value.dtype if value.dtype in [torch.float16, torch.float32, torch.float64] else torch.float32
                    )
                    break
            return torch.ones(batch.batch.batch_size[0], response_length, device=device, dtype=dtype)

    def _compute_advantage_comparison_metrics(self, batch: DataProto, context: MetricsContext) -> Dict[str, float]:
        """
        Compute metrics comparing actual advantages with RLOO baseline.

        This helps understand how different advantage estimators and modifications
        (variance reduction, standardization) affect the final advantages used in training.
        """
        metrics = {}

        # Check if we have the required data
        if (
            'rloo_advantages' not in batch.batch
            or 'advantages' not in batch.batch
            or 'token_level_rewards' not in batch.batch
            or 'uid' not in batch.non_tensor_batch
        ):
            return metrics

        # Check if tracking is enabled
        if not batch.meta_info.get('track_advantage_comparison', True):
            return metrics

        # Import the tracker
        from verl_patch.trainer.code.metrics.advantage_comparison_tracker import (
            AdvantageComparisonTracker,
        )

        # Create tracker if not exists (using class attribute for persistence)
        if not hasattr(self, 'advantage_tracker'):
            self.advantage_tracker = AdvantageComparisonTracker(enable_detailed_tracking=True)

        # Get estimator name from meta_info
        estimator_name = batch.meta_info.get('advantage_estimator', 'unknown')

        # Track the final advantages
        tracking_metrics = self.advantage_tracker.track_advantages(
            rloo_advantages=batch.batch['rloo_advantages'],
            actual_advantages=batch.batch['advantages'],
            token_level_rewards=batch.batch['token_level_rewards'],
            response_mask=batch.batch.get(
                'response_mask', self._get_response_mask(batch, batch.batch['advantages'].shape[-1])
            ),
            index=batch.non_tensor_batch['uid'],
            estimator_name=estimator_name,
            old_log_probs=batch.batch.get('old_log_probs'),
            sum_pi_squared=batch.batch.get('sum_pi_squared'),
        )

        # Prefix all metrics
        prefixed_metrics = {f'advantage_comparison/{k}': v for k, v in tracking_metrics.items()}
        metrics.update(prefixed_metrics)

        # Add summary statistics periodically (every 100 steps)
        if context.global_step and context.global_step % 100 == 0:
            summary_stats = self.advantage_tracker.get_summary_statistics()
            summary_metrics = {f'advantage_comparison/summary/{k}': v for k, v in summary_stats.items()}
            metrics.update(summary_metrics)

        return metrics

    def _compute_variance_proxy_metrics(self, batch: DataProto, context: MetricsContext) -> Dict[str, float]:
        """
        Compute variance proxy metrics using the simplified expected squared norm approach.

        This metric provides a computationally efficient way to monitor gradient variance
        during training. It works for any advantage estimator as long as sum_pi_squared
        is available from the actor.

        Theory:
        - Full variance: Var(g̃) = E[||g̃||²] - ||g_true||²
        - Simplified proxy (when ||g_true||² ≈ 0): Var(g̃) ≈ E[||g̃||²]
        - Using W-score approximation: E[||g̃||²] ≈ E[A² × W(τ)]

        Where W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²] is the score-norm proxy.
        """
        metrics = {}

        # Check if we have the necessary data (sum_pi_squared is required for W-score)
        if 'sum_pi_squared' not in batch.batch or 'old_log_probs' not in batch.batch or 'advantages' not in batch.batch:
            return metrics

        # Compute W(τ) = Σ_t[1 - 2π_t(y_t) + Σπ²]
        pi_t = torch.exp(batch.batch['old_log_probs'])
        w_per_timestep = 1 - 2 * pi_t + batch.batch['sum_pi_squared']

        # Get response mask to only consider valid tokens
        response_mask = self._get_response_mask(batch, w_per_timestep.shape[-1])

        # Use pre-computed rollout IS weights from batch (for variance proxy consistency with training loss)
        # IS weights are computed centrally in ray_trainer.py to avoid duplication
        rollout_is_weights = None
        if 'rollout_is_weights' in batch.batch:
            # Extract pre-computed IS weights from batch (already computed in trainer)
            rollout_is_weights = batch.batch['rollout_is_weights']

            # Scale W by (rollout IS weight)² for optimal baseline under biased estimation
            w_per_timestep = w_per_timestep * (rollout_is_weights**2).detach()

            # Note: IS weight statistics and mismatch metrics are logged in ray_trainer.py

        # Get scalar advantages (mean over timesteps)
        advantages = batch.batch['advantages']
        # Compute mean advantage per trajectory using masked_mean
        advantages_scalar = verl_F.masked_mean(advantages, response_mask, axis=-1)

        # Compute W values (sum over timesteps)
        w_values = verl_F.masked_sum(w_per_timestep, response_mask, axis=-1)

        # ====== COMPUTE VARIANCE PROXIES (Following LaTeX nomenclature) ======
        # Variance proxy should match the actual gradient computation:
        # - If IS weights were computed/applied: use them in variance proxy calculation
        # - Otherwise: compute on-policy variance proxy
        use_is_weights = rollout_is_weights is not None

        # ====== PROXY 2 (LaTeX): Total Power E[||ĝ_τ||²] ======
        # Measures the average of squared gradient norms (Signal + Noise)
        if use_is_weights:
            # Off-policy with IS correction applied: use clamped weights consistently with actual gradient computation
            rollout_is_weights_scalar = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
            # Recover original W (before IS correction was applied in line 657)
            # Clamp to avoid division by zero when IS weights are zero
            w_original = verl_F.masked_sum(
                w_per_timestep / torch.clamp((rollout_is_weights**2).detach(), min=1e-10), response_mask, axis=-1
            )
            # Clamp W to avoid negative values (which would cause NaN in sqrt)
            w_original = torch.clamp(w_original, min=0.0)
            # Proxy 2 for off-policy: E[ρ̄² × A² × W]
            proxy2_total_power = ((rollout_is_weights_scalar**2) * (advantages_scalar**2) * w_original).mean()

            # For Proxy 4, also use clamped weights
            # Off-policy gradient magnitude for Proxy 4: |ρ̄(τ) · A(τ)| · √W(τ)
            gradient_magnitudes = torch.abs(rollout_is_weights_scalar * advantages_scalar) * torch.sqrt(w_original)
        else:
            # On-policy Proxy 2: E[A² × W]
            # Clamp W to avoid negative values (which would cause NaN in sqrt)
            w_values_clamped = torch.clamp(w_values, min=0.0)
            proxy2_total_power = (advantages_scalar**2 * w_values_clamped).mean()
            # On-policy gradient magnitude for Proxy 4: |A(τ)| · √W(τ)
            gradient_magnitudes = torch.abs(advantages_scalar) * torch.sqrt(w_values_clamped)

        # ====== PROXY 4 (LaTeX): Magnitude Inconsistency Var[|A|·√W] ======
        # Diagnoses inconsistency in update strengths across trajectories
        # Var(X) = E[X²] - E[X]²
        mean_magnitude = gradient_magnitudes.mean()
        proxy4_magnitude_inconsistency = (gradient_magnitudes**2).mean() - mean_magnitude**2

        # ====== PROXY 1 (LaTeX): Signal Strength ||ḡ||² ======
        # The squared norm of the mean gradient (provided from training loop)
        proxy1_signal_strength = (
            context.batch_gradient_norm_squared if context and context.batch_gradient_norm_squared is not None else None
        )

        # ====== PROXY 3 (LaTeX): Pure Noise - Variance of Mean Vector ======
        # Requires ||ḡ||² from actual batch gradient
        # Formula: (1/(N-1)) × (Proxy2 - Proxy1)
        proxy3_pure_noise = None
        if proxy1_signal_strength is not None:
            batch_size = advantages_scalar.shape[0]
            if batch_size > 1:
                proxy3_pure_noise = (1.0 / (batch_size - 1)) * (proxy2_total_power - proxy1_signal_strength)
                # Ensure non-negative (can be negative due to numerical errors)
                proxy3_pure_noise = max(
                    0.0, proxy3_pure_noise.item() if torch.is_tensor(proxy3_pure_noise) else proxy3_pure_noise
                )

        # ====== INTER-BATCH VARIANCE (with EMA) ======
        # Track the squared norm of batch gradients across training steps
        # According to LaTeX: v_t = β · v_{t-1} + (1 - β) · ||ĝ_t||²
        # This requires the actual batch gradient norm squared
        inter_batch_variance_ema = None
        current_grad_norm_squared = None
        if context and context.batch_gradient_norm_squared is not None:
            current_grad_norm_squared = context.batch_gradient_norm_squared

            # Update EMA for inter-batch variance
            if self.inter_batch_variance_ema is None:
                self.inter_batch_variance_ema = current_grad_norm_squared
            else:
                self.inter_batch_variance_ema = (
                    self.inter_batch_ema_beta * self.inter_batch_variance_ema
                    + (1 - self.inter_batch_ema_beta) * current_grad_norm_squared
                )
            inter_batch_variance_ema = self.inter_batch_variance_ema

        # Decompose into components for analysis
        expected_a_squared = (advantages_scalar**2).mean()
        expected_w = w_values.mean()

        # Coefficient of variation for gradient magnitudes
        # CV = std(gradient_magnitudes) / mean(gradient_magnitudes)
        # Clamp mean_magnitude to avoid division issues
        cv_gradient_magnitudes = torch.std(gradient_magnitudes) / torch.clamp(mean_magnitude, min=1e-6)

        # Average W per timestep (normalized by response length)
        response_lengths = verl_F.masked_sum(torch.ones_like(response_mask), response_mask, axis=-1)
        # Use clamped w_values if available from on-policy path
        w_for_avg = w_values_clamped if not use_is_weights else w_original
        w_per_timestep_avg = (w_for_avg / torch.clamp(response_lengths, min=1.0)).mean()

        # Signal-to-noise ratio approximation
        # If we had access to the true gradient, SNR ≈ ||g_true||² / Var(g̃)
        # Since we assume ||g_true||² ≈ 0, we track the reciprocal as a stability metric
        # Lower values indicate higher variance relative to signal
        # Clamp proxy4 to avoid inf when variance is extremely small, and clamp result to reasonable range
        proxy4_clamped = torch.clamp(proxy4_magnitude_inconsistency, min=1e-10)
        stability_metric = torch.clamp(1.0 / proxy4_clamped, max=1e10)

        metrics.update(
            {
                # ====== Four Proxies for Intra-Batch Variance (Following LaTeX) ======
                # Proxy 1 (LaTeX): Signal Strength ||ḡ||²
                'variance_proxy/proxy1_signal_strength': (
                    proxy1_signal_strength if proxy1_signal_strength is not None else 0.0
                ),
                # Proxy 2 (LaTeX): Total Power E[||ĝ_τ||²]
                'variance_proxy/proxy2_total_power': proxy2_total_power.detach().item(),
                # Proxy 3 (LaTeX): Pure Noise - Variance of Mean Vector
                'variance_proxy/proxy3_pure_noise': proxy3_pure_noise if proxy3_pure_noise is not None else 0.0,
                'variance_proxy/proxy3_available': proxy3_pure_noise is not None,
                # Proxy 4 (LaTeX): Magnitude Inconsistency Var[|A|·√W]
                'variance_proxy/proxy4_magnitude_inconsistency': proxy4_magnitude_inconsistency.detach().item(),
                # Inter-batch variance tracking (EMA of batch gradient norm squared across training steps)
                'variance_proxy/inter_batch_variance_ema': (
                    inter_batch_variance_ema if inter_batch_variance_ema is not None else 0.0
                ),
                'variance_proxy/inter_batch_variance_available': inter_batch_variance_ema is not None,
                # Component metrics for debugging
                'variance_proxy/expected_a_squared': expected_a_squared.detach().item(),
                'variance_proxy/expected_w': expected_w.detach().item(),
                # Stability metrics
                'variance_proxy/coefficient_of_variation': cv_gradient_magnitudes.detach().item(),
                'variance_proxy/stability_metric': stability_metric.detach().item(),
                # W-score statistics
                'variance_proxy/w_per_timestep': w_per_timestep_avg.detach().item(),
                'variance_proxy/w_total_mean': expected_w.detach().item(),
                'variance_proxy/w_total_std': torch.std(w_values).detach().item(),
                # Gradient magnitude statistics (new metrics matching LaTeX)
                'variance_proxy/gradient_magnitude_mean': mean_magnitude.detach().item(),
                'variance_proxy/gradient_magnitude_std': torch.std(gradient_magnitudes).detach().item(),
                # Current batch gradient norm squared (||ḡ||²) - used for Proxy 3 and inter-batch tracking
                'variance_proxy/batch_gradient_norm_squared': (
                    current_grad_norm_squared if current_grad_norm_squared is not None else 0.0
                ),
                'variance_proxy/batch_gradient_norm': (
                    (current_grad_norm_squared**0.5) if current_grad_norm_squared is not None else 0.0
                ),
            }
        )

        return metrics

    # Note: _compute_mismatch_metrics() has been removed as mismatch metrics are now
    # computed centrally in ray_trainer.py via compute_rollout_importance_weights_and_add_to_batch().
    # This eliminates duplication and ensures consistency across the training pipeline.


# Global instance for easy access
_global_metrics_system = UnifiedMetricsSystem()


def configure_variance_tracking(ema_beta: float = 0.99):
    """
    Configure the variance tracking parameters.

    Args:
        ema_beta: Smoothing factor for inter-batch variance EMA (0 < beta < 1).
                 Higher values give more weight to history.
    """
    _global_metrics_system.inter_batch_ema_beta = ema_beta


def reset_inter_batch_tracking():
    """Reset the inter-batch variance EMA tracking."""
    _global_metrics_system.inter_batch_variance_ema = None


def compute_all_training_metrics(
    batch: DataProto,
    use_critic: bool = True,
    extra_rewards_info: Optional[Dict[str, Any]] = None,
    timing_raw: Optional[Dict[str, float]] = None,
    n_gpus: Optional[int] = None,
    global_step: Optional[int] = None,
    batch_gradient_norm_squared: Optional[float] = None,
) -> Dict[str, float]:
    """
    Single function to compute ALL training metrics.

    This replaces all scattered metrics computation in ray_trainer.py

    Args:
        batch: Training batch
        use_critic: Whether critic is being used
        extra_rewards_info: Extra rewards information
        timing_raw: Timing information
        n_gpus: Number of GPUs for throughput calculation
        global_step: Current training step
        batch_gradient_norm_squared: Squared L2 norm of batch gradient (for Proxy 3 and inter-batch variance)

    Returns:
        Complete metrics dictionary ready for logging
    """
    context = MetricsContext(
        use_critic=use_critic,
        extra_rewards_info=extra_rewards_info,
        timing_raw=timing_raw,
        n_gpus=n_gpus,
        global_step=global_step,
        batch_gradient_norm_squared=batch_gradient_norm_squared,
    )

    return _global_metrics_system.compute_all_metrics(batch, context)


def configure_metrics(**categories):
    """
    Configure which metric categories to compute globally.

    Example:
        configure_metrics(
            core=True,
            rl_advanced=True,
            timing=False
        )
    """
    _global_metrics_system.configure(**categories)
