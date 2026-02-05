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

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", "rloo", "remax",
            "optimal_baseline" (outcome-level baseline), "optimal_baseline_step" (PWRB per-timestep baseline).
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl", "abs", "mse", "low_var_kl", or "full".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy.
        batch_std (bool): Enable intelligent batch standardization (sequence vs token level).
        use_multi_prompt_mvu (bool): Multi-prompt Minimum Variance Unbiased (MVU) weighting.
        adv_by_last_turn(bool): Only use the last turn's data for advantage calculation
        use_final_reward(bool): Restrict reward assignment to the last turn (ignore intermediate rewards)
        rollout_is_threshold (Optional[float]): Upper threshold for IS weights. null = disabled,
            float value = enabled (compute weights and metrics). This is the main on/off switch.
        rollout_is_threshold_lower (Optional[float]): Lower threshold for IS weights. If None, defaults to 1/upper.
        rollout_is_level (str): Aggregation level: "token", "sequence", or "geometric".
        rollout_is_mode (str): Bounding mode: "truncate" (cap upper only) or "mask" (zero outside bounds).
        rollout_is_veto_threshold (Optional[float]): Per-token veto threshold for catastrophic outliers.
            null = disabled, float = enabled. If enabled, rejects entire sequence if any token has ratio < threshold.
        rollout_is (bool): Whether to apply IS weights to policy loss. True = apply weights,
            False = compute metrics only (useful for monitoring before enabling correction). Default: False.
        optimal_baseline_kwargs (dict[str, Any]): Arguments forwarded to optimal-baseline estimators.
            Common keys:
              - ``uniform_weight``: bool, force w_per_timestep = 1 instead of variance proxy.
              - ``uniform_cumulative``: bool, normalize cumulative weight to 1 per turn/step.
              - ``rollout_correction``: bool, enable importance-sampling-aware weighting.
            Default: {}.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)
    filter_groups: Optional[FilterGroupsConfig] = None

    # Variance reduction techniques
    batch_std: bool = False  # Enable intelligent batch standardization (sequence vs token level)
    use_multi_prompt_mvu: bool = False  # Multi-prompt Minimum Variance Unbiased (MVU) weighting

    # Advantage estimation configuration
    adv_by_last_turn: bool = True  # Only use the last turn's data for advantage calculation
    use_final_reward: bool = True  # Restrict reward assignment to the last turn (ignore intermediate rewards)
    is_get_last_turn: bool = True  # Whether to extract only last turn data for filtering (all turns still used for training)
    reward_shaping: bool = False  # Whether to shape rewards to residual rewards
    unbiased_shaping: bool = False  # Whether to use unbiased shaping, we could add a terminal rewards 0 to keep optimal policy still.

    # Rollout Correction
    # Controls computation of IS weights, rejection mask and mismatch metrics
    rollout_is: Optional[str] = None  # "token", "turn", "sequence", or "null" (disabled)
    rollout_is_kwargs: dict[str, Any] = field(default_factory=dict)  # upper: float
    rollout_rs: Optional[str] = None  # "token", "turn", "turn_geo", "sequence", "geometric", or "null" (disabled)
    rollout_rs_kwargs: dict[str, Any] = field(default_factory=dict)  # upper: float, lower: float
    rollout_token_veto_threshold: Optional[float] = None  # null = disabled, float = enabled

    # Rollout Correction Mode Selection
    # These flags control how rollout-training policy mismatch is handled
    bypass_old_logprob_for_rollout: bool = False
    # When True: Uses rollout_log_prob as old_log_prob (skips expensive actor.compute_log_prob())
    # Benefit: Avoids extra forward pass for old_log_prob computation
    # Trade-off: PPO clips against rollout policy instead of true old policy

    use_pure_rollout_correction: bool = False
    # When True: Uses pure policy gradient with IS correction (no PPO clipping)
    # Requires: bypass_old_logprob_for_rollout=True
    # Formula: L = -E[w * A] where w = exp(log_prob - rollout_log_prob).clamp(max=threshold)
    # Use case: When you trust IS correction and don't need PPO's conservative updates
    # Warning: Higher variance than PPO, requires careful hyperparameter tuning

    # Optimal baseline configuration / ablations
    optimal_baseline_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate: bypass mode incompatible with features requiring sum_pi_squared
        if self.bypass_old_logprob_for_rollout:
            # Check 1: optimal_baseline estimators
            if 'optimal_baseline' in self.adv_estimator:
                raise ValueError(
                    f"bypass_old_logprob_for_rollout=True is incompatible with "
                    f"adv_estimator='{self.adv_estimator}'. Optimal baseline requires sum_pi_squared "
                    f"from actor.compute_log_prob(), which is skipped in bypass mode. "
                    f"Use bypass_old_logprob_for_rollout=False with optimal_baseline estimators."
                )

            # Check 2: multi-prompt MVU (requires sum_pi_squared for W(τ) computation)
            if self.use_multi_prompt_mvu:
                raise ValueError(
                    f"bypass_old_logprob_for_rollout=True is incompatible with "
                    f"use_multi_prompt_mvu=True. MVU requires sum_pi_squared "
                    f"from actor.compute_log_prob(), which is skipped in bypass mode. "
                    f"Use bypass_old_logprob_for_rollout=False with use_multi_prompt_mvu=True."
                )
        if self.use_pure_rollout_correction:
            if not self.bypass_old_logprob_for_rollout:
                raise ValueError(
                    f"use_pure_rollout_correction=True requires bypass_old_logprob_for_rollout=True. "
                    f"Set bypass_old_logprob_for_rollout=True to enable pure rollout correction."
                )

    # Threshold Recommendations for ~10000 token sequences:
    # - Token level: 2.0 (lower=0.5 by default) - stable but may introduce bias
    # - Sequence level: 10.0 (lower=0.1) or 5.0_0.2 - high variance, use with caution
    # - Geometric level (RECOMMENDED for long sequences):
    #   * Conservative: 1.0001_0.9999 (±0.01% per token, ±100% for 10k tokens)
    #   * Balanced: 1.0002_0.9998 (±0.02% per token, ±200% for 10k tokens)
    #   * Aggressive: 1.0005_0.9995 (±0.05% per token, ±500% for 10k tokens)
    #   Note: Geometric compounds over sequence length, so small per-token changes have large effects
