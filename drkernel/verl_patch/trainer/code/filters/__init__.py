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
Unified filter module for PPO batch processing.

This package provides a single, clean API for all filtering operations in PPO training:
- Rejection sampling (reward and length based)
- Two-gate precision filtering (FP32/BFloat16 mismatch)
- Dual-level oversampling (prompt and sample level)
- Smart sample selection strategies
- Group management and metrics tracking

The PPOBatchFilter is the ONLY public API that should be used by external code.
All other filters are internal implementation details.
"""

from .unified_filter import PPOBatchFilter, PPOFilterConfig

__all__ = ['PPOBatchFilter', 'PPOFilterConfig']
