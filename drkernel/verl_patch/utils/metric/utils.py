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
Metrics utils.
"""

from typing import Any, Dict, List

import numpy as np


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    IMPORTANT: This function expects only numeric values. Non-numeric values are filtered out
    with warnings to maintain system stability while providing debugging information.

    Args:
        metrics: A dictionary mapping metric names to lists of numeric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    import warnings

    for key, val in metrics.items():
        if not isinstance(val, list):
            warnings.warn(f"Metric '{key}' is not a list, converting to list")
            val = [val] if val is not None else []

        if len(val) == 0:
            metrics[key] = None
            continue

        # Filter to only numeric values
        numeric_values = []
        for v in val:
            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_values.append(float(v))
            else:
                # Log the problematic value for debugging
                warnings.warn(
                    f"Metric '{key}' contains non-numeric value: {v} (type: {type(v).__name__}). Filtering out."
                )

        if not numeric_values:
            warnings.warn(f"Metric '{key}' has no numeric values after filtering. Setting to None.")
            metrics[key] = None
            continue

        # Now safely apply numpy operations
        if "max" in key:
            metrics[key] = np.max(numeric_values)
        elif "min" in key:
            metrics[key] = np.min(numeric_values)
        else:
            metrics[key] = np.mean(numeric_values)

    return metrics
