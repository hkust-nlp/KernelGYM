"""
Kernel-specific multi-turn validation metrics.

This module provides metrics computation for multi-turn kernel training validation,
including per-turn metrics and best-by-turn metrics for tracking improvement over turns.
"""

from collections import defaultdict
from typing import Any

import numpy as np
from verl import DataProto

FAST_THRESHOLDS = (1.0, 1.2, 1.5, 2.0, 3.0)


def _quartile_means(values: list[float]) -> dict[str, float]:
    """Return mean performance for four descending quartiles."""
    if not values:
        return {}

    sorted_vals = np.sort(np.array(values, dtype=float))[::-1]
    chunks = np.array_split(sorted_vals, 4)
    labels = ("top_25", "p25_50", "p50_75", "p75_100")
    return {
        label: float(np.mean(chunk)) if len(chunk) > 0 else 0.0
        for label, chunk in zip(labels, chunks)
    }


def compute_kernel_multi_turn_metrics(batch: DataProto, prefix: str = "kernel") -> dict[str, Any]:
    """
    Compute kernel-specific multi-turn metrics.

    This function computes two types of metrics:
    1. Per-turn metrics: Independent metrics for each turn (turn_1, turn_2, turn_3)
       - correctness_rate: Ratio of correct results for that turn
       - mean_performance: Average performance for that turn (only for correct samples)
       - compilation_rate: Ratio of successful compilations

    2. Best-by-turn metrics: Best result achieved by turn N across all previous turns
       - best_by_turn_N/correctness_rate: Best correctness by turn N
       - best_by_turn_N/mean_performance: Best performance by turn N

    Args:
        batch: DataProto containing results with:
            - batch['turn_indices']: Tensor of turn indices (1-indexed, -1 for padding)
            # - batch['sample_indices']: Tensor of sample indices
            - non_tensor_batch['reward_extra_info']: List of dicts with per-turn info
            - non_tensor_batch['uid']: Sample identifiers
        prefix: Prefix for metric keys (default: "kernel")

    Returns:
        Dictionary of metrics with keys like:
            - {prefix}/turn_1/correctness_rate
            - {prefix}/turn_1/mean_performance
            - {prefix}/turn_1/compilation_rate
            - {prefix}/best_by_turn_2/correctness_rate
            - {prefix}/best_by_turn_2/mean_performance
    """
    # Check required fields
    if not hasattr(batch, 'batch') or 'turn_indices' not in batch.batch:
        return {}

    if not hasattr(batch, 'non_tensor_batch') or 'reward_extra_info' not in batch.non_tensor_batch:
        return {}

    # Extract data
    turn_indices = batch.batch['turn_indices']
    # sample_indices = batch.batch.get('sample_indices', None)
    reward_extra_info = batch.non_tensor_batch['reward_extra_info']
    uids = batch.non_tensor_batch.get('uid', None)

    # Convert tensors to numpy if needed
    if hasattr(turn_indices, 'cpu'):
        turn_indices = turn_indices.cpu().numpy()
    # if sample_indices is not None and hasattr(sample_indices, 'cpu'):
    #     sample_indices = sample_indices.cpu().numpy()

    # Build per-sample per-turn data structure
    # Key: (sample_id, turn_idx), Value: reward_extra_info dict
    sample_turn_data = defaultdict(dict)

    # Track statistics for filtering
    total_actual_turns = 0
    filtered_abnormal_turns = 0
    padding_turns = 0
    empty_dict_count = 0
    has_keys_count = 0

    print(f"turn_indices: {turn_indices}")
    # print(f"sample_indices: {sample_indices}")
    print(f"reward_extra_info: {reward_extra_info}")
    print(f"uids: {uids}")

    total_timeout_turns = 0

    for row_idx in range(len(turn_indices)):
        turn_idx = int(turn_indices[row_idx])

        # Skip padding turns
        if turn_idx == -1:
            padding_turns += 1
            continue

        total_actual_turns += 1

        # Get sample identifier
        # Use uid as sample_id for uniqueness across batches
        # sample_indices are batch-local (reset to 0 in each batch), so they repeat when batches are concatenated
        if uids is not None:
            sample_id = uids[row_idx]
        else:
            raise ValueError("uids is None")
        # elif sample_indices is not None:
            # Fallback to sample_indices (but this may cause issues with multiple batches)
        #     sample_id = int(sample_indices[row_idx])
        # else:
        #     sample_id = row_idx

        # Get reward extra info
        extra_info = reward_extra_info[row_idx]

        if "status" in extra_info and extra_info["status"] == "timeout":
            total_timeout_turns += 1

        if extra_info is None:
            filtered_abnormal_turns += 1
            print(f"Abnormal turn w/o any extra info at row {row_idx}")
            continue

        # Check if it's an empty dict
        if len(extra_info) == 0:
            empty_dict_count += 1
            print(f"Abnormal turn w/ 0 length extra info at row {row_idx}")
            filtered_abnormal_turns += 1
            continue

        # Skip abnormal turns (those without kernel-specific metrics)
        # Abnormal turns typically contain only 'finish_type' and 'error' fields
        # Normal turns should have at least one of: 'correctness', 'performance', 'compiled'
        if not any(key in extra_info for key in ['correctness', 'performance', 'compilation', 'success']):
            filtered_abnormal_turns += 1
            print(f"Abnormal turn w/o any kernel-specific metrics at row {row_idx}: {extra_info}")
            continue

        has_keys_count += 1
        sample_turn_data[sample_id][turn_idx] = extra_info

    if not sample_turn_data:
        return {}

    # Add filtering statistics
    metrics = {}
    if total_actual_turns > 0:
        metrics[f'{prefix}/data/total_actual_turns'] = total_actual_turns
        metrics[f'{prefix}/data/filtered_abnormal_turns'] = filtered_abnormal_turns
        metrics[f'{prefix}/data/valid_turns'] = total_actual_turns - filtered_abnormal_turns
        metrics[f'{prefix}/data/abnormal_turns_rate'] = filtered_abnormal_turns / total_actual_turns
        metrics[f'{prefix}/data/valid_turns_rate'] = (total_actual_turns - filtered_abnormal_turns) / total_actual_turns
        metrics[f'{prefix}/data/empty_dict_count'] = empty_dict_count
        metrics[f'{prefix}/data/has_keys_count'] = has_keys_count
        metrics[f'{prefix}/data/total_timeout_turns'] = total_timeout_turns
        metrics[f'{prefix}/data/timeout_turns_rate'] = total_timeout_turns / total_actual_turns

    metrics[f'{prefix}/data/padding_turns'] = padding_turns
    metrics[f'{prefix}/data/unique_samples'] = len(sample_turn_data)

    # Determine max turns
    all_turns = set()
    for turns_dict in sample_turn_data.values():
        all_turns.update(turns_dict.keys())
    max_turn = max(all_turns) if all_turns else 0

    # ============================================
    # Type 1: Per-turn independent metrics
    # ============================================
    for turn_idx in range(1, max_turn + 1):
        turn_correctness = []
        turn_performances = []
        turn_all_performances = []
        turn_compiled = []
        turn_speedups = []

        turn_time_coverages = []
        turn_num_coverages = []

        turn_time_coverages_all = []
        turn_num_coverages_all = []

        for sample_id, turns_dict in sample_turn_data.items():
            if turn_idx not in turns_dict:
                continue

            info = turns_dict[turn_idx]

            # Correctness
            correctness = info.get('correctness', False)
            # Handle decoy_kernel if present
            if 'decoy_kernel' in info:
                correctness = correctness and not info['decoy_kernel']
            turn_correctness.append(1.0 if correctness else 0.0)

            # Performance (only for correct samples)
            if correctness:
                performance = info.get('performance', 0.0)
                turn_performances.append(performance)
                if performance is not None:
                    turn_speedups.append(float(performance))
                turn_all_performances.append(float(performance) if performance is not None else 0.0)
                turn_time_coverages.append(float(info.get('time_coverage', 0.0)) if info.get('time_coverage', 0.0) is not None else 0.0)
                turn_num_coverages.append(float(info.get('num_coverage', 0.0)) if info.get('num_coverage', 0.0) is not None else 0.0)

                turn_time_coverages_all.append(float(info.get('time_coverage', 0.0)) if info.get('time_coverage', 0.0) is not None else 0.0)
                turn_num_coverages_all.append(float(info.get('num_coverage', 0.0)) if info.get('num_coverage', 0.0) is not None else 0.0)
            else:
                # Incorrect or decoy: count performance as 0 for _in_all metrics
                turn_all_performances.append(0.0)
                turn_time_coverages_all.append(0.0)
                turn_num_coverages_all.append(0.0)

            # Compilation
            compiled = info.get('compilation', False)
            turn_compiled.append(1.0 if compiled else 0.0)

        if turn_correctness:
            metrics[f'{prefix}/turn_{turn_idx}/correctness_rate'] = np.mean(turn_correctness)
            metrics[f'{prefix}/turn_{turn_idx}/count'] = len(turn_correctness)

        if turn_performances:
            metrics[f'{prefix}/turn_{turn_idx}/mean_performance'] = np.mean(turn_performances)
            metrics[f'{prefix}/turn_{turn_idx}/max_performance'] = np.max(turn_performances)
            metrics[f'{prefix}/turn_{turn_idx}/min_performance'] = np.min(turn_performances)
            for label, mean_val in _quartile_means(turn_performances).items():
                metrics[f'{prefix}/turn_{turn_idx}/performance_percentile/{label}'] = mean_val
        else:
            metrics[f'{prefix}/turn_{turn_idx}/mean_performance'] = 0.0

        if turn_all_performances:
            metrics[f'{prefix}/turn_{turn_idx}/mean_performance_in_all'] = np.mean(turn_all_performances)
            speedup_array_all = np.array(turn_all_performances, dtype=float)
            for threshold in FAST_THRESHOLDS:
                threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
                metrics[f'{prefix}/turn_{turn_idx}/fast@{threshold_label}_in_all'] = float(
                    np.mean(speedup_array_all >= threshold)
                )
            for label, mean_val in _quartile_means(turn_all_performances).items():
                metrics[f'{prefix}/turn_{turn_idx}/performance_percentile_in_all/{label}'] = mean_val

            # coverage
            metrics[f'{prefix}/turn_{turn_idx}/mean_time_coverage_in_all'] = np.mean(turn_time_coverages_all)
            metrics[f'{prefix}/turn_{turn_idx}/max_time_coverage_in_all'] = np.max(turn_time_coverages_all)
            metrics[f'{prefix}/turn_{turn_idx}/min_time_coverage_in_all'] = np.min(turn_time_coverages_all)
            metrics[f'{prefix}/turn_{turn_idx}/mean_num_coverage_in_all'] = np.mean(turn_num_coverages_all)
            metrics[f'{prefix}/turn_{turn_idx}/max_num_coverage_in_all'] = np.max(turn_num_coverages_all)
            metrics[f'{prefix}/turn_{turn_idx}/min_num_coverage_in_all'] = np.min(turn_num_coverages_all)

        if turn_speedups:
            speedup_array = np.array(turn_speedups, dtype=float)
            for threshold in FAST_THRESHOLDS:
                threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
                metrics[f'{prefix}/turn_{turn_idx}/fast@{threshold_label}'] = float(
                    np.mean(speedup_array >= threshold)
                )

        if turn_compiled:
            metrics[f'{prefix}/turn_{turn_idx}/compilation_rate'] = np.mean(turn_compiled)

        if turn_time_coverages:
            metrics[f'{prefix}/turn_{turn_idx}/mean_time_coverage'] = np.mean(turn_time_coverages)
            metrics[f'{prefix}/turn_{turn_idx}/max_time_coverage'] = np.max(turn_time_coverages)
            metrics[f'{prefix}/turn_{turn_idx}/min_time_coverage'] = np.min(turn_time_coverages)

        if turn_num_coverages:
            metrics[f'{prefix}/turn_{turn_idx}/mean_num_coverage'] = np.mean(turn_num_coverages)
            metrics[f'{prefix}/turn_{turn_idx}/max_num_coverage'] = np.max(turn_num_coverages)
            metrics[f'{prefix}/turn_{turn_idx}/min_num_coverage'] = np.min(turn_num_coverages)

    # ============================================
    # Type 2: Best-by-turn cumulative metrics
    # ============================================
    # For each sample, track best result up to each turn
    for turn_idx in range(1, max_turn + 1):
        best_correctness = []
        best_performances = []
        best_all_performances = []

        for sample_id, turns_dict in sample_turn_data.items():
            # Find best result up to turn_idx
            best_correct = False
            best_performance = 0.0

            for t in range(1, turn_idx + 1):
                if t not in turns_dict:
                    continue

                info = turns_dict[t]

                # Check correctness
                correctness = info.get('correctness', False)
                if 'decoy_kernel' in info:
                    correctness = correctness and not info['decoy_kernel']

                # Update best
                if correctness:
                    performance = info.get('performance', 0.0)
                    if not best_correct or performance > best_performance:
                        best_correct = True
                        best_performance = performance

            best_correctness.append(1.0 if best_correct else 0.0)
            if best_correct:
                best_performances.append(best_performance)
                best_all_performances.append(float(best_performance))
            else:
                best_all_performances.append(0.0)

        if best_correctness:
            metrics[f'{prefix}/best_by_turn_{turn_idx}/correctness_rate'] = np.mean(best_correctness)
            metrics[f'{prefix}/best_by_turn_{turn_idx}/count'] = len(best_correctness)

        if best_performances:
            metrics[f'{prefix}/best_by_turn_{turn_idx}/mean_performance'] = np.mean(best_performances)
            metrics[f'{prefix}/best_by_turn_{turn_idx}/max_performance'] = np.max(best_performances)
            speedup_array = np.array(best_performances, dtype=float)
            for threshold in FAST_THRESHOLDS:
                threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
                metrics[f'{prefix}/best_by_turn_{turn_idx}/fast@{threshold_label}'] = float(
                    np.mean(speedup_array >= threshold)
                )
            for label, mean_val in _quartile_means(best_performances).items():
                metrics[f'{prefix}/best_by_turn_{turn_idx}/performance_percentile/{label}'] = mean_val
        else:
            metrics[f'{prefix}/best_by_turn_{turn_idx}/mean_performance'] = 0.0

        if best_all_performances:
            metrics[f'{prefix}/best_by_turn_{turn_idx}/mean_performance_in_all'] = np.mean(best_all_performances)
            speedup_array_all = np.array(best_all_performances, dtype=float)
            for threshold in FAST_THRESHOLDS:
                threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
                metrics[f'{prefix}/best_by_turn_{turn_idx}/fast@{threshold_label}_in_all'] = float(
                    np.mean(speedup_array_all >= threshold)
                )
            for label, mean_val in _quartile_means(best_all_performances).items():
                metrics[f'{prefix}/best_by_turn_{turn_idx}/performance_percentile_in_all/{label}'] = mean_val

    # ============================================
    # Overall statistics
    # ============================================
    # Final turn metrics (last turn for each sample)
    final_correctness = []
    final_performances = []

    for sample_id, turns_dict in sample_turn_data.items():
        if not turns_dict:
            continue
        final_turn = max(turns_dict.keys())
        info = turns_dict[final_turn]

        correctness = info.get('correctness', False)
        if 'decoy_kernel' in info:
            correctness = correctness and not info['decoy_kernel']
        final_correctness.append(1.0 if correctness else 0.0)

        if correctness:
            final_performances.append(info.get('performance', 0.0))

    if final_correctness:
        metrics[f'{prefix}/final/correctness_rate'] = np.mean(final_correctness)
    if final_performances:
        metrics[f'{prefix}/final/mean_performance'] = np.mean(final_performances)

    # Improvement metrics (comparing first and last turns)
    first_correct_count = 0
    last_correct_count = 0
    improvement_count = 0  # Samples that improved
    regression_count = 0   # Samples that regressed

    for sample_id, turns_dict in sample_turn_data.items():
        if not turns_dict:
            continue

        turns = sorted(turns_dict.keys())
        first_turn = turns[0]
        last_turn = turns[-1]

        first_info = turns_dict[first_turn]
        last_info = turns_dict[last_turn]

        first_correct = first_info.get('correctness', False)
        if 'decoy_kernel' in first_info:
            first_correct = first_correct and not first_info['decoy_kernel']

        last_correct = last_info.get('correctness', False)
        if 'decoy_kernel' in last_info:
            last_correct = last_correct and not last_info['decoy_kernel']

        if first_correct:
            first_correct_count += 1
        if last_correct:
            last_correct_count += 1

        # Track improvement
        if not first_correct and last_correct:
            improvement_count += 1
        elif first_correct and not last_correct:
            regression_count += 1

    total_samples = len(sample_turn_data)
    if total_samples > 0:
        metrics[f'{prefix}/improvement/first_turn_correct_rate'] = first_correct_count / total_samples
        metrics[f'{prefix}/improvement/last_turn_correct_rate'] = last_correct_count / total_samples
        metrics[f'{prefix}/improvement/improved_samples'] = improvement_count
        metrics[f'{prefix}/improvement/regressed_samples'] = regression_count
        metrics[f'{prefix}/improvement/net_improvement'] = improvement_count - regression_count

    return metrics


def extract_turn_progression_metrics(
    metrics_dict: dict[str, Any],
    prefix: str = "turn_stats",
    thresholds: tuple = FAST_THRESHOLDS
) -> dict[str, dict[str, dict[int, float]]]:
    """
    Extract turn progression data from compute_kernel_multi_turn_metrics output.

    This function extracts metrics for each turn to enable visualization of how metrics
    evolve across turns (turn_1 → turn_2 → ... → turn_n). It provides two views:
    - 'independent': Metrics for each turn independently
    - 'best_by_turn': Best result achieved up to and including each turn (cumulative)

    Args:
        metrics_dict: Output from compute_kernel_multi_turn_metrics
        prefix: Metric prefix (default: "kernel")
        thresholds: Performance thresholds for fast@p metrics (default: FAST_THRESHOLDS)

    Returns:
        Dictionary with structure:
        {
            'independent': {
                'count': {1: 512, 2: 512, 3: 480, ...},
                'correctness_rate': {1: 0.65, 2: 0.72, 3: 0.78, ...},
                'compilation_rate': {1: 0.95, 2: 0.96, 3: 0.97, ...},
                'mean_performance': {1: 1.5, 2: 1.8, 3: 2.1, ...},
                'mean_performance_in_all': {1: 1.0, 2: 1.3, 3: 1.6, ...},
                'mean_time_coverage': {1: 0.65, 2: 0.70, 3: 0.75, ...},
                'mean_time_coverage_in_all': {1: 0.42, 2: 0.50, 3: 0.59, ...},
                'mean_num_coverage': {1: 0.58, 2: 0.63, 3: 0.68, ...},
                'mean_num_coverage_in_all': {1: 0.38, 2: 0.45, 3: 0.53, ...},
                'fast@1': {1: 0.50, 2: 0.60, 3: 0.70, ...},
                'fast@1_in_all': {1: 0.35, 2: 0.45, 3: 0.55, ...},
                'performance_percentile/top_25': {1: 2.5, 2: 2.8, 3: 3.1, ...},
                ... (for each threshold and quartile)
            },
            'best_by_turn': {
                'count': {1: 512, 2: 512, 3: 480, ...},
                'correctness_rate': {1: 0.65, 2: 0.75, 3: 0.82, ...},
                'mean_performance': {1: 1.5, 2: 1.9, 3: 2.3, ...},
                'mean_performance_in_all': {1: 1.0, 2: 1.4, 3: 1.9, ...},
                'fast@1': {1: 0.50, 2: 0.65, 3: 0.75, ...},
                'fast@1_in_all': {1: 0.35, 2: 0.50, 3: 0.65, ...},
                'performance_percentile/top_25': {1: 2.5, 2: 2.9, 3: 3.2, ...},
                ... (for each threshold and quartile, no compilation/coverage)
            }
        }

    Example usage:
        # After computing kernel metrics
        kernel_metrics = compute_kernel_multi_turn_metrics(all_dataproto, prefix="kernel")
        turn_progression = extract_turn_progression_metrics(kernel_metrics, prefix="kernel")

        # Plot independent metrics
        import matplotlib.pyplot as plt
        for metric_name, turn_values in turn_progression['independent'].items():
            turns = sorted(turn_values.keys())
            values = [turn_values[t] for t in turns]
            plt.plot(turns, values, label=metric_name)
        plt.xlabel('Turn ID')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.savefig('turn_progression.png')
    """
    import re

    # Step 1: Find max_turn by scanning all keys
    max_turn = 0
    for key in metrics_dict.keys():
        # Extract turn numbers from patterns like "kernel/turn_3/..." or "kernel/best_by_turn_5/..."
        match = re.search(rf'{re.escape(prefix)}/(best_by_)?turn_(\d+)/', key)
        if match:
            max_turn = max(max_turn, int(match.group(2)))

    if max_turn == 0:
        return {'independent': {}, 'best_by_turn': {}}

    # Step 2: Initialize result structure
    result = {
        'independent': defaultdict(dict),
        'best_by_turn': defaultdict(dict)
    }

    # Step 3: Define metric names to extract
    simple_metrics_independent = [
        'count', 'correctness_rate', 'compilation_rate',
        'mean_performance', 'mean_performance_in_all',
        'mean_time_coverage', 'mean_time_coverage_in_all',
        'mean_num_coverage', 'mean_num_coverage_in_all'
    ]

    simple_metrics_best = [
        'count', 'correctness_rate',
        'mean_performance', 'mean_performance_in_all'
    ]

    quartiles = ['top_25', 'p25_50', 'p50_75', 'p75_100']

    # Step 4: Extract independent per-turn metrics
    for turn_idx in range(1, max_turn + 1):
        # Simple metrics
        for metric in simple_metrics_independent:
            key = f'{prefix}/turn_{turn_idx}/{metric}'
            if key in metrics_dict:
                result['independent'][metric][turn_idx] = metrics_dict[key]

        # Fast@p metrics
        for threshold in thresholds:
            threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)

            key = f'{prefix}/turn_{turn_idx}/fast@{threshold_label}'
            if key in metrics_dict:
                result['independent'][f'fast@{threshold_label}'][turn_idx] = metrics_dict[key]

            key_all = f'{prefix}/turn_{turn_idx}/fast@{threshold_label}_in_all'
            if key_all in metrics_dict:
                result['independent'][f'fast@{threshold_label}_in_all'][turn_idx] = metrics_dict[key_all]

        # Quartile metrics
        for quartile in quartiles:
            key = f'{prefix}/turn_{turn_idx}/performance_percentile/{quartile}'
            if key in metrics_dict:
                result['independent'][f'performance_percentile/{quartile}'][turn_idx] = metrics_dict[key]

            key_all = f'{prefix}/turn_{turn_idx}/performance_percentile_in_all/{quartile}'
            if key_all in metrics_dict:
                result['independent'][f'performance_percentile_in_all/{quartile}'][turn_idx] = metrics_dict[key_all]

    # Step 5: Extract best-by-turn cumulative metrics
    for turn_idx in range(1, max_turn + 1):
        for metric in simple_metrics_best:
            key = f'{prefix}/best_by_turn_{turn_idx}/{metric}'
            if key in metrics_dict:
                result['best_by_turn'][metric][turn_idx] = metrics_dict[key]

        for threshold in thresholds:
            threshold_label = str(int(threshold)) if float(threshold).is_integer() else str(threshold)

            key = f'{prefix}/best_by_turn_{turn_idx}/fast@{threshold_label}'
            if key in metrics_dict:
                result['best_by_turn'][f'fast@{threshold_label}'][turn_idx] = metrics_dict[key]

            key_all = f'{prefix}/best_by_turn_{turn_idx}/fast@{threshold_label}_in_all'
            if key_all in metrics_dict:
                result['best_by_turn'][f'fast@{threshold_label}_in_all'][turn_idx] = metrics_dict[key_all]

        for quartile in quartiles:
            key = f'{prefix}/best_by_turn_{turn_idx}/performance_percentile/{quartile}'
            if key in metrics_dict:
                result['best_by_turn'][f'performance_percentile/{quartile}'][turn_idx] = metrics_dict[key]

            key_all = f'{prefix}/best_by_turn_{turn_idx}/performance_percentile_in_all/{quartile}'
            if key_all in metrics_dict:
                result['best_by_turn'][f'performance_percentile_in_all/{quartile}'][turn_idx] = metrics_dict[key_all]

    # Step 6: Convert defaultdicts to regular dicts
    result['independent'] = dict(result['independent'])
    result['best_by_turn'] = dict(result['best_by_turn'])

    return result
