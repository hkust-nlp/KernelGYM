"""
Mismatch quality metrics for analyzing masked samples.

This module computes comprehensive quality metrics for responses masked by
training-inference mismatch mechanisms (rollout RS/IS/veto), comparing their
characteristics (correctness, performance) to non-masked responses.
"""

from typing import Any

import numpy as np
import torch
from verl import DataProto

FAST_THRESHOLDS = (1.0, 1.2, 1.5, 2.0, 3.0)


def compute_mismatch_quality_metrics(
    batch: DataProto,
    original_response_mask: torch.Tensor,
    modified_response_mask: torch.Tensor,
    prefix: str = "mismatch_quality",
) -> dict[str, Any]:
    """
    Compute quality metrics for mismatch-masked responses.

    Analyzes which responses were masked by rollout RS/IS/veto and compares
    their quality characteristics (correctness, performance, fast@p rates) to
    non-masked responses.

    This function helps answer:
    - Are we masking correct or incorrect samples? (correctness distribution)
    - What's the performance of masked vs non-masked samples? (mean_performance)
    - Are high-quality samples (fast@p) being masked? (mask_rate by quality)

    Args:
        batch: DataProto containing:
            - non_tensor_batch['reward_extra_info']: List of dicts with:
                - 'correctness': bool (whether kernel is correct)
                - 'performance': float (speedup ratio)
                - 'compilation': bool (whether kernel compiled)
        original_response_mask: Original response mask before mismatch masking,
            shape (batch_size, seq_length)
        modified_response_mask: Response mask after RS/veto applied,
            shape (batch_size, seq_length)
        prefix: Metric prefix (default: "mismatch_quality")

    Returns:
        Dictionary with metrics organized into three groups:

        **A. Masked Group Stats** (responses that were masked):
            - {prefix}/masked/count: Total masked responses
            - {prefix}/masked/correct_count: Correct responses in masked group
            - {prefix}/masked/incorrect_count: Incorrect responses in masked group
            - {prefix}/masked/correct_rate: Proportion of correct samples in masked group
            - {prefix}/masked/mean_performance: Mean speedup (correct only)
            - {prefix}/masked/mean_performance_in_all: Mean speedup (all, incorrect=0)
            - {prefix}/masked/fast@{p}: Fraction >= threshold (correct only)
            - {prefix}/masked/fast@{p}_in_all: Fraction >= threshold (all)

        **B. Non-Masked Group Stats** (responses that passed):
            - {prefix}/non_masked/count: Total non-masked responses
            - {prefix}/non_masked/correct_count: Correct responses
            - {prefix}/non_masked/incorrect_count: Incorrect responses
            - {prefix}/non_masked/correct_rate: Proportion of correct samples in non-masked group
            - {prefix}/non_masked/mean_performance: Mean speedup (correct only)
            - {prefix}/non_masked/mean_performance_in_all: Mean speedup (all)
            - {prefix}/non_masked/fast@{p}: Fraction >= threshold (correct only)
            - {prefix}/non_masked/fast@{p}_in_all: Fraction >= threshold (all)

        **C. Mask Rate by Quality** (what percentage of each quality group was masked):
            - {prefix}/correct/mask_rate: Mask rate for correct samples
            - {prefix}/incorrect/mask_rate: Mask rate for incorrect samples
            - {prefix}/fast@{p}/mask_rate: Mask rate for fast@p samples (correct only)
            - {prefix}/fast@{p}_in_all/mask_rate: Mask rate for fast@p samples (all)

    Example:
        >>> metrics = compute_mismatch_quality_metrics(
        ...     batch, original_mask, modified_mask, prefix="mismatch_quality"
        ... )
        >>> print(metrics["mismatch_quality/masked/correct_count"])  # 12
        >>> print(metrics["mismatch_quality/correct/mask_rate"])     # 0.0065
    """
    # Check required fields
    if not hasattr(batch, 'non_tensor_batch') or 'reward_extra_info' not in batch.non_tensor_batch:
        return {}

    reward_extra_info = batch.non_tensor_batch['reward_extra_info']
    if reward_extra_info is None or len(reward_extra_info) == 0:
        return {}

    # Convert to numpy if needed
    if hasattr(original_response_mask, 'cpu'):
        original_response_mask_np = original_response_mask.cpu().numpy()
        modified_response_mask_np = modified_response_mask.cpu().numpy()
    else:
        original_response_mask_np = np.array(original_response_mask)
        modified_response_mask_np = np.array(modified_response_mask)

    # Identify masked samples (sequence-level masking)
    # A sample is masked if it had valid tokens originally but has none after masking
    original_valid = original_response_mask_np.sum(axis=-1) > 0  # (batch_size,)
    modified_valid = modified_response_mask_np.sum(axis=-1) > 0  # (batch_size,)
    is_masked = original_valid & ~modified_valid  # (batch_size,) bool array

    # Extract reward_extra_info as list
    if hasattr(reward_extra_info, 'tolist'):
        reward_info_list = reward_extra_info.tolist()
    else:
        reward_info_list = list(reward_extra_info)

    # Ensure consistent length
    batch_size = len(is_masked)
    if len(reward_info_list) != batch_size:
        # Truncate or pad if necessary
        if len(reward_info_list) > batch_size:
            reward_info_list = reward_info_list[:batch_size]
        else:
            # Pad with empty dicts
            reward_info_list.extend([{}] * (batch_size - len(reward_info_list)))

    # Collect data for masked and non-masked groups
    masked_data = _extract_quality_data(reward_info_list, is_masked)
    non_masked_data = _extract_quality_data(reward_info_list, ~is_masked)

    # Compute metrics
    metrics = {}

    # A. Masked group stats
    metrics.update(_compute_group_metrics(masked_data, f"{prefix}/masked"))

    # B. Non-masked group stats
    metrics.update(_compute_group_metrics(non_masked_data, f"{prefix}/non_masked"))

    # C. Mask rate by quality
    metrics.update(_compute_mask_rates(masked_data, non_masked_data, prefix))

    return metrics


def _extract_quality_data(reward_info_list: list, mask: np.ndarray) -> dict[str, list]:
    """
    Extract correctness and performance data for a subset of samples.

    Args:
        reward_info_list: List of dicts with 'correctness', 'performance', etc.
        mask: Boolean mask indicating which samples to include

    Returns:
        Dictionary with:
            - 'correctness': List of bool
            - 'performance': List of float (for correct samples only)
            - 'performance_all': List of float (0 for incorrect samples)
    """
    data = {
        'correctness': [],
        'performance': [],        # Only correct samples
        'performance_all': [],    # All samples (incorrect=0)
    }

    for i, include in enumerate(mask):
        if not include:
            continue

        info = reward_info_list[i]
        if not isinstance(info, dict) or len(info) == 0:
            continue

        # Extract correctness
        correctness = info.get('correctness', False)
        # Handle decoy_kernel if present
        if 'decoy_kernel' in info:
            correctness = correctness and not info['decoy_kernel']

        data['correctness'].append(correctness)

        # Extract performance
        if correctness:
            performance = info.get('performance', 0.0)
            if performance is not None:
                data['performance'].append(float(performance))
                data['performance_all'].append(float(performance))
            else:
                data['performance_all'].append(0.0)
        else:
            data['performance_all'].append(0.0)

    return data


def _compute_group_metrics(data: dict[str, list], prefix: str) -> dict[str, Any]:
    """
    Compute metrics for a group of samples (masked or non-masked).

    Args:
        data: Dictionary with 'correctness', 'performance', 'performance_all'
        prefix: Metric prefix (e.g., "mismatch_quality/masked")

    Returns:
        Dictionary of metrics for this group
    """
    metrics = {}

    # Count metrics
    total_count = len(data['correctness'])
    if total_count == 0:
        metrics[f"{prefix}/count"] = 0
        return metrics

    metrics[f"{prefix}/count"] = total_count

    correct_count = sum(data['correctness'])
    incorrect_count = total_count - correct_count
    metrics[f"{prefix}/correct_count"] = correct_count
    metrics[f"{prefix}/incorrect_count"] = incorrect_count
    metrics[f"{prefix}/correct_rate"] = correct_count / total_count

    # Performance metrics (correct only)
    if data['performance']:
        metrics[f"{prefix}/mean_performance"] = float(np.mean(data['performance']))
        metrics[f"{prefix}/max_performance"] = float(np.max(data['performance']))
        metrics[f"{prefix}/min_performance"] = float(np.min(data['performance']))

        # fast@p metrics (correct only)
        performance_array = np.array(data['performance'], dtype=float)
        for threshold in FAST_THRESHOLDS:
            threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
            metrics[f"{prefix}/fast@{threshold_label}"] = float(
                np.mean(performance_array >= threshold)
            )
    else:
        metrics[f"{prefix}/mean_performance"] = 0.0

    # Performance metrics (all samples, incorrect=0)
    if data['performance_all']:
        metrics[f"{prefix}/mean_performance_in_all"] = float(np.mean(data['performance_all']))

        # fast@p metrics (all samples)
        performance_all_array = np.array(data['performance_all'], dtype=float)
        for threshold in FAST_THRESHOLDS:
            threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
            metrics[f"{prefix}/fast@{threshold_label}_in_all"] = float(
                np.mean(performance_all_array >= threshold)
            )
    else:
        metrics[f"{prefix}/mean_performance_in_all"] = 0.0

    return metrics


def _compute_mask_rates(
    masked_data: dict[str, list],
    non_masked_data: dict[str, list],
    prefix: str,
) -> dict[str, float]:
    """
    Compute mask rates by quality characteristics.

    Args:
        masked_data: Quality data for masked samples
        non_masked_data: Quality data for non-masked samples
        prefix: Metric prefix (e.g., "mismatch_quality")

    Returns:
        Dictionary with mask_rate metrics
    """
    metrics = {}

    # Combine all samples
    all_correctness = masked_data['correctness'] + non_masked_data['correctness']
    all_performance = masked_data['performance'] + non_masked_data['performance']
    all_performance_all = masked_data['performance_all'] + non_masked_data['performance_all']

    masked_count = len(masked_data['correctness'])
    total_count = len(all_correctness)

    if total_count == 0:
        return metrics

    # Overall mask rate
    metrics[f"{prefix}/overall_mask_rate"] = masked_count / total_count

    # Mask rate for correct samples
    total_correct = sum(all_correctness)
    masked_correct = sum(masked_data['correctness'])
    if total_correct > 0:
        metrics[f"{prefix}/correct/mask_rate"] = masked_correct / total_correct
    else:
        metrics[f"{prefix}/correct/mask_rate"] = 0.0

    # Mask rate for incorrect samples
    total_incorrect = len(all_correctness) - total_correct
    masked_incorrect = len(masked_data['correctness']) - masked_correct
    if total_incorrect > 0:
        metrics[f"{prefix}/incorrect/mask_rate"] = masked_incorrect / total_incorrect
    else:
        metrics[f"{prefix}/incorrect/mask_rate"] = 0.0

    # Mask rate for fast@p samples (correct only)
    if all_performance:
        all_performance_array = np.array(all_performance, dtype=float)
        masked_performance_array = np.array(masked_data['performance'], dtype=float)

        for threshold in FAST_THRESHOLDS:
            threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)

            # Count samples >= threshold
            total_fast = np.sum(all_performance_array >= threshold)
            masked_fast = np.sum(masked_performance_array >= threshold) if len(masked_performance_array) > 0 else 0

            if total_fast > 0:
                metrics[f"{prefix}/fast@{threshold_label}/mask_rate"] = masked_fast / total_fast
            else:
                metrics[f"{prefix}/fast@{threshold_label}/mask_rate"] = 0.0

    # Mask rate for fast@p samples (all samples, incorrect=0)
    if all_performance_all:
        all_performance_all_array = np.array(all_performance_all, dtype=float)
        masked_performance_all_array = np.array(masked_data['performance_all'], dtype=float)

        for threshold in FAST_THRESHOLDS:
            threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)

            total_fast_all = np.sum(all_performance_all_array >= threshold)
            masked_fast_all = (
                np.sum(masked_performance_all_array >= threshold)
                if len(masked_performance_all_array) > 0
                else 0
            )

            if total_fast_all > 0:
                metrics[f"{prefix}/fast@{threshold_label}_in_all/mask_rate"] = masked_fast_all / total_fast_all
            else:
                metrics[f"{prefix}/fast@{threshold_label}_in_all/mask_rate"] = 0.0

    return metrics
