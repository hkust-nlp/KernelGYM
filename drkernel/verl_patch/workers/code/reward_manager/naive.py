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
from typing import Dict, List

import ray
import torch
from ray.exceptions import GetTimeoutError  # 用于处理超时
from verl import DataProto

from verl_patch.utils.reward_score import _default_compute_score

from .utils import reward_func_timeout_ray


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.timeout_seconds = 5

    def compute_score_parallel_with_ray(self, data_sources, solution_strs, ground_truths, extra_infos):
        scores: List[float] = [0.0] * len(solution_strs)
        extra_info_dict: Dict[str, List[float]] = {}  # Key -> list of values for the batch
        print(f"Scoring process started over {len(solution_strs)} samples, waiting for results...")

        futures = []
        for i in range(len(solution_strs)):
            ground_truth = ground_truths[i]
            solution_str = solution_strs[i]
            data_source = data_sources[i]
            extra_info = extra_infos[i]

            # 提交任务给 Ray
            future = reward_func_timeout_ray.remote(
                self.compute_score, data_source, solution_str, ground_truth, extra_info
            )
            futures.append(future)

        default_fail_score = {"score": 0.0, "extra_info": {"is_filter": 1}}  # Default on error which should be filtered

        # 获取任务结果，处理超时逻辑
        for i, future in enumerate(futures):
            try:
                # 设置结果返回的超时时间。与 ProcessPoolExecutor 不同，Ray 在这里通过 ray.get 的 timeout 参数控制
                task_result = ray.get(future, timeout=self.timeout_seconds)

                # 标准化 task_result 的格式
                if isinstance(task_result, dict):
                    assert (
                        'extra_info' in task_result
                    ), f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                    score_result = task_result
                    # 如果计算结果未过滤，确保正确标记
                    if "is_filter" not in task_result["extra_info"]:
                        score_result["extra_info"].update({"is_filter": 0})
                elif isinstance(task_result, (int, float)):  # 处理标量返回结果
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
                import traceback

                traceback.print_exc()
                ray.cancel(future, force=True)
                score_result = default_fail_score

            # 存储最终得分
            scores[i] = float(score_result.get('score', 0.0))  # 确保 score 是 float 类型

            # 如果存在 extra_info，收集它
            if 'extra_info' in score_result and isinstance(score_result['extra_info'], dict):
                for key, value in score_result['extra_info'].items():
                    if key not in extra_info_dict:
                        # 初始化列表（例如默认值 0.0）以匹配所有项
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        return scores, extra_info_dict

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        response_ids = data.batch['responses']
        sequences_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = [data_item.non_tensor_batch.get('extra_info', None) for data_item in data]

        assert len(sequences_strs) == len(ground_truths) == len(data_sources)

        scores, extra_info_dict = self.compute_score_parallel_with_ray(
            data_sources, sequences_strs, ground_truths, extra_infos
        )

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        data_sources = data.non_tensor_batch['data_source']

        for i in range(len(data)):
            # data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        return {'reward_tensor': reward_tensor, 'extra_info': extra_info_dict}
