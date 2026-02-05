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
Prompt-level Metrics for RL Training

This module provides metrics that analyze training data at the prompt level,
including coverage metrics and effective prompt statistics.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from verl import DataProto


def compute_effective_response_metrics(batch: DataProto, max_response_length: Optional[int] = None) -> Dict[str, float]:
    """
    Compute effective response metrics based on prompts with non-zero reward variance.

    This identifies "effective" prompts - those that have variation in rewards across
    different response generations, indicating the model is exploring different responses.

    Args:
        batch: Training batch data
        max_response_length: Maximum response length for clip ratio calculation

    Returns:
        Dictionary containing:
        - response_length/effective_num_prompts: Number of prompts with reward variance > 0
        - response_length/effective_mean: Mean response length for effective prompts
        - response_length/effective_max: Max response length for effective prompts
        - response_length/effective_min: Min response length for effective prompts
        - response_length/effective_clip_ratio: Clip ratio for effective prompts
    """
    metrics = {}

    # Get response lengths
    try:
        from verl.trainer.ppo.metric_utils import _compute_response_info

        response_info = _compute_response_info(batch)
        response_length = response_info["response_length"]
    except ImportError:
        # Fallback: compute response lengths from attention mask
        if 'attention_mask' in batch.batch:
            response_length = batch.batch['attention_mask'].sum(dim=-1).float()
        else:
            # Use sequence length as fallback
            seq_len = batch.batch['token_level_rewards'].shape[-1]
            response_length = torch.full((batch.batch['token_level_rewards'].shape[0],), seq_len, dtype=torch.float)

    # Get or compute sequence rewards
    if "seq_reward" not in batch.non_tensor_batch:
        seq_reward = batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
    else:
        seq_reward = batch.non_tensor_batch["seq_reward"]

    # Collect rewards for each prompt UID
    prompt_uid2rewards = defaultdict(list)
    for uid, reward in zip(batch.non_tensor_batch["uid"], seq_reward):
        prompt_uid2rewards[uid].append(reward)

    # Calculate standard deviation for each prompt
    prompt_uid2std = {}
    for prompt_uid, rewards in prompt_uid2rewards.items():
        prompt_uid2std[prompt_uid] = np.std(rewards)

    # Identify effective prompts (std > 0 - prompts showing reward variance)
    # Note: Single response prompts are not effective for exploration metrics
    effective_prompt_uids = [uid for uid, std in prompt_uid2std.items() if std > 0]

    # Get indices of effective prompts
    effective_prompt_ids = []
    for i, uid in enumerate(batch.non_tensor_batch["uid"]):
        if uid in effective_prompt_uids:
            effective_prompt_ids.append(i)

    # Calculate metrics
    if len(effective_prompt_ids) == 0:
        # No effective prompts - return zero statistics
        return {
            "response_length/effective_num_prompts": 0,
            "response_length/effective_mean": 0.0,
            "response_length/effective_max": 0.0,
            "response_length/effective_min": 0.0,
            "response_length/effective_clip_ratio": 0.0,
        }
    else:
        num_effective_prompts = len(set(effective_prompt_uids))

    effective_response_lengths = response_length[effective_prompt_ids]

    metrics["response_length/effective_num_prompts"] = num_effective_prompts
    metrics["response_length/effective_mean"] = torch.mean(effective_response_lengths).detach().item()
    metrics["response_length/effective_max"] = torch.max(effective_response_lengths).detach().item()
    metrics["response_length/effective_min"] = torch.min(effective_response_lengths).detach().item()

    if max_response_length is not None:
        metrics["response_length/effective_clip_ratio"] = (
            torch.mean(torch.eq(effective_response_lengths, max_response_length).float()).detach().item()
        )

    return metrics


def compute_prompt_coverage_metrics(batch: DataProto) -> Dict[str, float]:
    """
    Compute prompt-level coverage and entropy metrics.

    These metrics analyze the distribution of positive/negative rewards across prompts,
    helping to identify:
    - Whether all prompts receive both positive and negative feedback
    - The diversity of the prompt distribution
    - Prompts that might be too easy (all positive) or too hard (all negative)

    Args:
        batch: Training batch data

    Returns:
        Dictionary containing:
        - prompt/positive_coverage_ratio: Fraction of prompts with at least one positive reward
        - prompt/all_positive_ratio: Fraction of prompts with only positive rewards
        - prompt/negative_coverage_ratio: Fraction of prompts with at least one negative reward
        - prompt/all_negative_ratio: Fraction of prompts with only negative rewards
        - prompt/entropy: Entropy of the prompt distribution
        - prompt/entropy_reduction: Reduction from maximum possible entropy
    """
    # Get sequence-level scores
    scores = batch.batch["token_level_rewards"].sum(dim=-1)
    bsz = scores.shape[0]

    # Build UID to ID mapping
    uids = batch.non_tensor_batch["uid"]
    uid2id = {}
    i = 0
    for uid in uids:
        if uid not in uid2id:
            uid2id[uid] = i
            i += 1
    num_of_prompts = i

    # Handle empty batch case
    if num_of_prompts == 0:
        return {
            "prompt/positive_coverage_ratio": 0.0,
            "prompt/all_positive_ratio": 0.0,
            "prompt/negative_coverage_ratio": 0.0,
            "prompt/all_negative_ratio": 0.0,
            "prompt/entropy": 0.0,
            "prompt/entropy_reduction": 0.0,
        }

    # Collect scores for each unique prompt
    id2scores = defaultdict(list)
    for i in range(bsz):
        idx = uid2id[uids[i]]
        id2scores[idx].append(scores[i])

    # Count positive and negative responses per prompt
    positive_counts = torch.zeros(num_of_prompts, dtype=torch.int32)
    negative_counts = torch.zeros(num_of_prompts, dtype=torch.int32)

    for i in range(num_of_prompts):
        for score in id2scores[i]:
            if score > 0:
                positive_counts[i] += 1
            elif score < 0:
                negative_counts[i] += 1
            # Note: zero scores are treated as neutral, not counted in either

    # Calculate coverage metrics
    positive_coverage_ratio = torch.sum(positive_counts > 0).float() / num_of_prompts
    all_positive_ratio = ((positive_counts > 0) & (negative_counts == 0)).float().mean()
    negative_coverage_ratio = torch.sum(negative_counts > 0).float() / num_of_prompts
    all_negative_ratio = ((negative_counts > 0) & (positive_counts == 0)).float().mean()

    # Calculate prompt entropy
    prompt_counts = positive_counts + negative_counts
    total_count = torch.sum(prompt_counts).item()
    prompt_probs = prompt_counts.float() / total_count

    prompt_entropy = 0.0
    for p in prompt_probs:
        if p > 0:
            prompt_entropy -= p * torch.log(p)

    # Maximum possible entropy (uniform distribution)
    max_entropy = torch.log(torch.tensor(float(num_of_prompts)))
    prompt_entropy_reduction = max_entropy - prompt_entropy

    return {
        "prompt/positive_coverage_ratio": positive_coverage_ratio.item(),
        "prompt/all_positive_ratio": all_positive_ratio.item(),
        "prompt/negative_coverage_ratio": negative_coverage_ratio.item(),
        "prompt/all_negative_ratio": all_negative_ratio.item(),
        "prompt/entropy": prompt_entropy.item() if torch.is_tensor(prompt_entropy) else prompt_entropy,
        "prompt/entropy_reduction": (
            prompt_entropy_reduction.item() if torch.is_tensor(prompt_entropy_reduction) else prompt_entropy_reduction
        ),
    }
