from typing import Any, Dict

import numpy as np
from verl import DataProto


def compute_multi_turn_metrics(batch: DataProto) -> Dict[str, Any]:
    """
    Computes metrics related to multi-turn conversations in training.
    """
    # Get multi-turn statistics from non-tensor batch
    if not hasattr(batch, 'non_tensor_batch') or 'num_turns' not in batch.non_tensor_batch:
        return {}

    stats_batch = batch.non_tensor_batch
    num_turns = stats_batch['num_turns']
    contain_void_turn = stats_batch['contain_void_turn']

    num_turns_mean = np.mean(num_turns)
    num_turns_max = np.max(num_turns)
    num_turns_min = np.min(num_turns)
    contain_void_turn_rate = np.mean(contain_void_turn)

    metrics = {
        "multiturn/num_turns/mean": num_turns_mean,
        "multiturn/num_turns/max": num_turns_max,
        "multiturn/num_turns/min": num_turns_min,
        "multiturn/contain_void_turn_rate": contain_void_turn_rate,
    }

    # Add finish_reason metrics if available
    if 'finish_reasons' in stats_batch:
        finish_reasons = stats_batch['finish_reasons']
        # Count occurrences of each finish reason
        unique_reasons, counts = np.unique(finish_reasons, return_counts=True)
        total_count = len(finish_reasons)

        # Add distribution of finish reasons
        for reason, count in zip(unique_reasons, counts):
            if reason:  # Skip empty strings
                metrics[f"multiturn/finish_reason/{reason}"] = count / total_count

    # Add prompt truncation ratio if available
    if 'prompt_truncation_ratio' in stats_batch:
        metrics['multiturn/prompt_truncation_ratio'] = stats_batch['prompt_truncation_ratio']

    # Add turn health statistics (normal vs abnormal turns)
    if 'finish_reasons' in stats_batch and 'turn_indices' in batch.batch:
        finish_reasons = stats_batch['finish_reasons']
        turn_indices = batch.batch['turn_indices'].cpu().numpy()

        # Define abnormal finish reasons (errors, timeouts, etc.)
        # Normal reasons include: stop, length, tool_calls, answer, max_tool_call, skipped
        abnormal_reasons = {'error', 'async_timeout', 'no_tool_call'}

        # Count normal and abnormal turns (exclude padding turns with turn_idx == -1)
        normal_count = 0
        abnormal_count = 0

        # Per-turn statistics
        turn_stats = {}  # {turn_idx: {'normal': count, 'abnormal': count}}

        for i, (reason, turn_idx) in enumerate(zip(finish_reasons, turn_indices)):
            if turn_idx == -1:  # Skip padding turns
                continue

            turn_idx = int(turn_idx)
            if turn_idx not in turn_stats:
                turn_stats[turn_idx] = {'normal': 0, 'abnormal': 0}

            normalized_reason = reason
            if normalized_reason is None:
                normalized_reason = ''
            elif isinstance(normalized_reason, (bytes, np.bytes_)):
                normalized_reason = normalized_reason.decode('utf-8')
            else:
                normalized_reason = str(normalized_reason)
            normalized_reason = normalized_reason.lower()
            is_abnormal = normalized_reason in abnormal_reasons
            if is_abnormal:
                abnormal_count += 1
                turn_stats[turn_idx]['abnormal'] += 1
            else:
                normal_count += 1
                turn_stats[turn_idx]['normal'] += 1

        # Add overall statistics
        total_turns = normal_count + abnormal_count
        if total_turns > 0:
            metrics['multiturn/normal_turns_total'] = normal_count
            metrics['multiturn/abnormal_turns_total'] = abnormal_count
            metrics['multiturn/normal_turns_rate'] = normal_count / total_turns
            metrics['multiturn/abnormal_turns_rate'] = abnormal_count / total_turns

            # Add per-turn statistics
            for turn_idx in sorted(turn_stats.keys()):
                stats = turn_stats[turn_idx]
                turn_total = stats['normal'] + stats['abnormal']
                if turn_total > 0:
                    metrics[f'multiturn/turn_{turn_idx}/normal_count'] = stats['normal']
                    metrics[f'multiturn/turn_{turn_idx}/abnormal_count'] = stats['abnormal']
                    metrics[f'multiturn/turn_{turn_idx}/normal_rate'] = stats['normal'] / turn_total
                    metrics[f'multiturn/turn_{turn_idx}/abnormal_rate'] = stats['abnormal'] / turn_total

    return metrics
