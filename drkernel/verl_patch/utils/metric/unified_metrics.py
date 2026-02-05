# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Unified metric computation system for RL algorithms.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch
import verl.utils.torch_functional as verl_F


@dataclass
class PolicyOutput:
    """Unified output for policy optimization algorithms."""

    # Primary loss for backpropagation
    loss: torch.Tensor

    # Core PPO metrics (always tensors for consistency)
    kl_divergence: torch.Tensor
    clip_fraction: torch.Tensor

    # Detailed clipping analysis
    clip_fraction_lower: torch.Tensor
    clip_fraction_upper: torch.Tensor
    clip_fraction_dual: torch.Tensor

    # Algorithm identifier
    algorithm: str  # "ppo", "grpo", etc.

    # Algorithm-specific metrics
    algorithm_metrics: Dict[str, Any] = field(default_factory=dict)

    # Loss configuration
    loss_config: Dict[str, Any] = field(default_factory=dict)

    def get_all_metrics(self, prefix: str = "") -> Dict[str, Union[torch.Tensor, float, str]]:
        """Get all metrics as a dictionary with optional prefix."""
        metrics = {
            f"{prefix}loss": self.loss,
            f"{prefix}kl_divergence": self.kl_divergence,
            f"{prefix}clip_fraction": self.clip_fraction,
            f"{prefix}clip_fraction_lower": self.clip_fraction_lower,
            f"{prefix}clip_fraction_upper": self.clip_fraction_upper,
            f"{prefix}clip_fraction_dual": self.clip_fraction_dual,
            f"{prefix}algorithm": self.algorithm,
        }

        # Add loss configuration
        for key, value in self.loss_config.items():
            metrics[f"{prefix}config/{key}"] = value

        # Add algorithm-specific metrics
        algo_prefix = f"{prefix}{self.algorithm}/"
        for key, value in self.algorithm_metrics.items():
            metrics[f"{algo_prefix}{key}"] = value

        return metrics

    def detach_all(self) -> 'PolicyOutput':
        """Create a new PolicyOutput with all tensors detached."""

        def detach_if_tensor(x):
            return x.detach() if isinstance(x, torch.Tensor) else x

        return PolicyOutput(
            loss=self.loss.detach(),
            kl_divergence=self.kl_divergence.detach(),
            clip_fraction=self.clip_fraction.detach(),
            clip_fraction_lower=self.clip_fraction_lower.detach(),
            clip_fraction_upper=self.clip_fraction_upper.detach(),
            clip_fraction_dual=self.clip_fraction_dual.detach(),
            algorithm=self.algorithm,
            algorithm_metrics={k: detach_if_tensor(v) for k, v in self.algorithm_metrics.items()},
            loss_config=self.loss_config.copy(),
        )

    def to_scalars(self, prefix: str = "") -> Dict[str, Union[float, str]]:
        """Convert all metrics to Python scalars for logging."""

        def to_scalar(x):
            if isinstance(x, torch.Tensor):
                return x.detach().item()
            return x

        metrics = {
            f"{prefix}loss": to_scalar(self.loss),
            f"{prefix}kl_divergence": to_scalar(self.kl_divergence),
            f"{prefix}clip_fraction": to_scalar(self.clip_fraction),
            f"{prefix}clip_fraction_lower": to_scalar(self.clip_fraction_lower),
            f"{prefix}clip_fraction_upper": to_scalar(self.clip_fraction_upper),
            f"{prefix}clip_fraction_dual": to_scalar(self.clip_fraction_dual),
            f"{prefix}algorithm": self.algorithm,
        }

        # Add loss configuration
        for key, value in self.loss_config.items():
            metrics[f"{prefix}config/{key}"] = to_scalar(value)

        # Add algorithm-specific metrics
        algo_prefix = f"{prefix}{self.algorithm}/"
        for key, value in self.algorithm_metrics.items():
            metrics[f"{algo_prefix}{key}"] = to_scalar(value)

        return metrics


@dataclass
class MetricStatistics:
    """Statistics for a metric."""

    mean: float
    std: float
    min: float
    max: float
    count: int

    @classmethod
    def from_tensor(cls, values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> 'MetricStatistics':
        """Compute statistics from a tensor with optional mask."""
        if mask is not None:
            values = values[mask]

        if len(values) == 0:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0, count=0)

        return cls(
            mean=values.mean().item(),
            std=values.std().item() if len(values) > 1 else 0.0,
            min=values.min().item(),
            max=values.max().item(),
            count=len(values),
        )

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """Convert to dictionary with optional prefix."""
        return {
            f"{prefix}mean": self.mean,
            f"{prefix}std": self.std,
            f"{prefix}min": self.min,
            f"{prefix}max": self.max,
            f"{prefix}count": self.count,
        }


def compute_clipping_metrics(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range_low: float,
    clip_range_high: float,
    dual_clip_coeff: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute clipping fractions for PPO."""
    # Lower bound clipping (advantages > 0)
    positive_mask = (advantages > 0) & mask
    if positive_mask.any():
        positive_ratios = ratio[positive_mask]
        clip_fraction_lower = torch.mean((positive_ratios > 1.0 + clip_range_high).float())
    else:
        clip_fraction_lower = torch.tensor(0.0, device=ratio.device)

    # Upper bound clipping (advantages < 0)
    negative_mask = (advantages < 0) & mask
    if negative_mask.any():
        negative_ratios = ratio[negative_mask]
        clip_fraction_upper = torch.mean((negative_ratios < 1.0 - clip_range_low).float())
    else:
        clip_fraction_upper = torch.tensor(0.0, device=ratio.device)

    # Dual clipping
    if dual_clip_coeff > 0:
        dual_clip_mask = (advantages < 0) & (ratio > dual_clip_coeff) & mask
        clip_fraction_dual = torch.mean(dual_clip_mask.float())
    else:
        clip_fraction_dual = torch.tensor(0.0, device=ratio.device)

    return clip_fraction_lower, clip_fraction_upper, clip_fraction_dual


def ensure_tensor(
    value: Union[float, torch.Tensor], device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Ensure a value is a tensor with proper device and dtype."""
    if isinstance(value, torch.Tensor):
        if device is not None and value.device != device:
            value = value.to(device)
        if value.dtype != dtype:
            value = value.to(dtype)
        return value
    else:
        return torch.tensor(value, device=device, dtype=dtype)
