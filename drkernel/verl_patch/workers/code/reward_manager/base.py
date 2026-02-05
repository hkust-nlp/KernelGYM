# Copyright 2024 PRIME team and/or its affiliates
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

import traceback
from typing import Dict, List, Tuple

import ray
import torch
from ray.exceptions import GetTimeoutError
from verl import DataProto

from verl_patch.utils.reward_score import _default_compute_score

from .utils import reward_func_timeout_ray


class BaseRewardManager:
    """
    Base class for reward managers to reduce code duplication.
    It handles the parallel score computation using Ray and the main reward tensor assignment logic.
    Subclasses must implement the `_get_solution_strs` method.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, timeout_seconds: int = 30) -> None:
        """
        Initializes the base reward manager.

        Args:
            tokenizer: The tokenizer for decoding sequences.
            num_examine (int): The number of batches of decoded responses to print.
            compute_score (callable, optional): The function to compute scores. Defaults to _default_compute_score.
            timeout_seconds (int): Timeout for each reward computation task.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.timeout_seconds = timeout_seconds

    def _compute_scores_parallel(
        self, data_sources: List[str], solution_strs: List[str], ground_truths: List, extra_infos: List
    ) -> Tuple[List[float], Dict[str, List]]:
        """
        Computes scores in parallel using Ray with timeout handling.
        """
        scores: List[float] = [0.0] * len(solution_strs)
        extra_info_dict: Dict[str, List] = {}
        print(f"Scoring process started over {len(solution_strs)} samples, waiting for results...")

        futures = [
            reward_func_timeout_ray.remote(
                self.compute_score, data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i]
            )
            for i in range(len(solution_strs))
        ]

        default_fail_score = {"score": 0.0, "extra_info": {"is_filter": 1}}

        for i, future in enumerate(futures):
            try:
                task_result = ray.get(future, timeout=self.timeout_seconds)

                if isinstance(task_result, dict):
                    assert (
                        "extra_info" in task_result
                    ), f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                    score_result = task_result
                    if "is_filter" not in task_result["extra_info"]:
                        score_result["extra_info"]["is_filter"] = 0
                elif isinstance(task_result, (int, float)):
                    score_result = {"score": float(task_result), "extra_info": {"is_filter": 0}}
                else:
                    print(
                        f"Unexpected task_result type for item {i}: {type(task_result)}. Using default score. Result: {task_result}"
                    )
                    ray.cancel(future, force=True)
                    score_result = default_fail_score
            except GetTimeoutError:
                print(
                    f"Timeout processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'). Using default score."
                )
                score_result = default_fail_score
            except Exception as e:
                print(
                    f"Error processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'): {e}"
                )
                traceback.print_exc()
                ray.cancel(future, force=True)
                score_result = default_fail_score

            scores[i] = float(score_result.get('score', 0.0))

            if 'extra_info' in score_result and isinstance(score_result['extra_info'], dict):
                for key, value in score_result['extra_info'].items():
                    if key not in extra_info_dict:
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        return scores, extra_info_dict

    def _get_solution_strs(self, data: DataProto) -> List[str]:
        """
        Abstract method to extract solution strings from the data batch.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the `_get_solution_strs` method.")

    def __call__(self, data: DataProto) -> Dict:
        """
        Main entry point for computing rewards.
        """
        if 'rm_scores' in data.batch:
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        # Extract common metadata
        ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = [item.non_tensor_batch.get('extra_info', None) for item in data]

        # Get solution strings using the subclass-specific implementation
        solution_strs = self._get_solution_strs(data)

        assert len(solution_strs) == len(ground_truths) == len(data_sources)

        # Compute scores in parallel
        scores, extra_info_dict = self._compute_scores_parallel(data_sources, solution_strs, ground_truths, extra_infos)

        # Assign scores to the reward tensor at the last valid token position
        prompt_length = data.batch['prompts'].shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)

        for i in range(len(data)):
            data_source = data_sources[i]
            # Ensure the index is valid
            response_end_idx = valid_response_length[i].item() - 1
            if response_end_idx >= 0:
                reward_tensor[i, response_end_idx] = scores[i]

            # Logic for printing debug samples
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # (Original code had this block duplicated, corrected to one increment)

        return {'reward_tensor': reward_tensor, 'extra_info': extra_info_dict}
