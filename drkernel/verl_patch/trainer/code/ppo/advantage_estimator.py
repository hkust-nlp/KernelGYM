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
Extended AdvantageEstimator enum for AIIC VERL.

This module extends the base AdvantageEstimator to include optimal baseline.
"""

from enum import Enum


class AdvantageEstimator(str, Enum):
    """
    Extended advantage estimation algorithms.

    Includes all base algorithms from verl plus optimal baseline.
    """

    # Base algorithms from verl
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"

    # AIIC VERL custom algorithms
    OPTIMAL_BASELINE = "optimal_baseline"
    OPTIMAL_BASELINE_STEP = "optimal_baseline_step"
