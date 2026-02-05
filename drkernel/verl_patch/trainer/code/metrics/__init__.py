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
Unified Metrics Package for RL Training

This package provides comprehensive metrics computation and monitoring
for Reinforcement Learning training of Large Language Models.

Main Entry Points:
- compute_all_training_metrics: Single function for all metrics
- configure_metrics: Enable/disable metric categories

Usage:
    from verl_patch.trainer.code.metrics import compute_all_training_metrics

    metrics = compute_all_training_metrics(
        batch=batch,
        use_critic=True,
        extra_rewards_info=extra_rewards_info,
        timing_raw=timing_raw,
        n_gpus=n_gpus,
        global_step=global_step
    )
"""

from .advantage_metrics import (
    check_metric_alerts,
    compute_comprehensive_advantage_metrics,
    compute_verification_metrics,
)

# Monitoring utilities
from .bound_monitoring import compute_advantage_bound_violations

# Prompt-level metrics
from .prompt_metrics import (
    compute_effective_response_metrics,
    compute_prompt_coverage_metrics,
)

# Main unified system - primary interface
from .unified_metrics import (
    MetricsContext,
    UnifiedMetricsSystem,
    compute_all_training_metrics,
    configure_metrics,
)

__all__ = [
    # Main interface
    'compute_all_training_metrics',
    'configure_metrics',
    'UnifiedMetricsSystem',
    'MetricsContext',
    # Monitoring
    'compute_advantage_bound_violations',
    # Prompt-level metrics
    'compute_effective_response_metrics',
    'compute_prompt_coverage_metrics',
    # Advanced metrics
    'compute_comprehensive_advantage_metrics',
    'compute_verification_metrics',
    'check_metric_alerts',
]
