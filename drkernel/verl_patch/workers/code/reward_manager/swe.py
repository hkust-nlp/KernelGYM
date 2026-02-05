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

from typing import Any, Callable, Dict, List

import ray
import torch
from ray.exceptions import GetTimeoutError  # 用于处理超时
from verl import DataProto

from verl_patch.utils.reward_score import _default_compute_score


@ray.remote(num_cpus=4, max_calls=5000)
def reward_func_timeout_ray(func: Callable, *args: Any, **kwargs: Any):
    """A Ray remote function that executes the given function with arguments.

    The timeout is handled by Ray's ray.get() method when retrieving results,
    not within this function itself.

    Args:
        func: The function to execute
        timeout_seconds: This parameter is kept for backward compatibility but not used here.
                        The actual timeout is enforced by ray.get() in the calling code.
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        The result of func(*args, **kwargs)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # If an exception occurs during execution, return a default fail score
        print(f"Error in reward computation: {e}")
        return {"score": 0.0, "extra_info": {"is_filter": 1}}


class SWERewardManager:
    """
    The Reward Manager is borrowed from https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.timeout_seconds = 110

    # 修改原始方法，使用 Ray
    def swe_compute_score_parallel_with_ray(self, data_sources, solution_strs, ground_truths, extra_infos):
        if "swe-agent" in data_sources[0]:
            default_fail_score = {
                "score": 0.0,
                "extra_info": {"is_filter": 1.0, "current_score": 0.0, "max_score": 0.0},
            }
        else:
            default_fail_score = {
                "score": 0.0,
                "extra_info": {"is_filter": 1.0},
            }  # Default on error which should be filtered

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
                        score_result["extra_info"].update({"is_filter": 0.0})
                elif isinstance(task_result, (int, float)):  # 处理标量返回结果
                    score_result = {"score": float(task_result), "extra_info": {"is_filter": 0.0}}
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

            if 'swe-agent' in data_sources[i]:
                if 'max-exact' in data_sources[i]:
                    max_score = float(score_result.get('max_score', 0.0))
                    scores[i] = 1.0 if max_score > 0.95 else 0.0
                elif 'max' in data_sources[i]:
                    scores[i] = float(score_result.get('max_score', 0.0))
                else:
                    scores[i] = float(score_result.get('current_score', 0.0))
            else:
                scores[i] = float(score_result.get('score', 0.0))  # 确保 score 是 float 类型

            # 如果存在 extra_info，收集它
            if 'extra_info' in score_result and isinstance(score_result['extra_info'], dict):
                for key, value in score_result['extra_info'].items():
                    if key not in extra_info_dict:
                        # 初始化列表（例如默认值 0.0）以匹配所有项
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        return scores, extra_info_dict

    def swe_bench_compute_score_parallel_with_ray_grouped(
        self, data_sources, solution_strs, ground_truths, extra_infos
    ):
        """
        并行计算得分，但确保相同instance_id的任务不会并行执行，避免资源竞争
        """
        default_fail_score = {"score": 0.0, "extra_info": {"is_filter": 1.0, "patch_similarity": 0.0}}

        scores = [0.0] * len(solution_strs)
        extra_info_dict = {}
        print(f"Scoring process started over {len(solution_strs)} samples with grouping by instance_id...")

        # 1. 根据instance_id对任务进行分组
        task_groups = {}
        for i in range(len(solution_strs)):
            instance_id = extra_infos[i].get("instance_id", f"default_{i}")

            if instance_id not in task_groups:
                task_groups[instance_id] = []

            task_groups[instance_id].append(
                {
                    "index": i,
                    "ground_truth": ground_truths[i],
                    "solution_str": solution_strs[i],
                    "data_source": data_sources[i],
                    "extra_info": extra_infos[i],
                }
            )

        print(f"Tasks grouped into {len(task_groups)} distinct instance groups")

        # 2. 处理每个组的任务，组内串行，组间并行
        group_futures = {}  # 存储每个实例组的当前进行中的任务
        completed_results = {}  # 存储已完成任务的结果

        # 初始化：为每个组启动第一个任务
        for instance_id, tasks in task_groups.items():
            if tasks:
                task = tasks[0]
                future = reward_func_timeout_ray.options(runtime_env={"env_vars": {"INSTANCE_ID": instance_id}}).remote(
                    self.compute_score,
                    self.timeout_seconds,
                    task["data_source"],
                    task["solution_str"],
                    task["ground_truth"],
                    task["extra_info"],
                )
                group_futures[instance_id] = (future, task["index"], 1)

        # 处理所有任务
        while group_futures:
            # 等待任何一个任务完成
            done_ids, _ = ray.wait([f for f, _, _ in group_futures.values()], num_returns=1)

            # 找出哪个组的任务完成了
            completed_instance_id = None
            for instance_id, (future, _, _) in group_futures.items():
                if future in done_ids:
                    completed_instance_id = instance_id
                    break

            if completed_instance_id is None:
                continue

            # 处理完成的任务结果
            future, i, next_task_idx = group_futures[completed_instance_id]
            tasks = task_groups[completed_instance_id]

            try:
                task_result = ray.get(future)

                # 标准化结果格式
                if isinstance(task_result, dict):
                    score_result = task_result
                    if "is_filter" not in score_result.get("extra_info", {}):
                        score_result["extra_info"]["is_filter"] = 0.0
                elif isinstance(task_result, (int, float)):
                    score_result = {"score": float(task_result), "extra_info": {"is_filter": 0.0}}
                else:
                    score_result = default_fail_score
            except:
                score_result = default_fail_score

            # 存储结果
            completed_results[i] = score_result

            # 检查并启动下一个任务
            if next_task_idx < len(tasks):
                next_task = tasks[next_task_idx]
                next_future = reward_func_timeout_ray.remote(
                    self.compute_score,
                    self.timeout_seconds,
                    next_task["data_source"],
                    next_task["solution_str"],
                    next_task["ground_truth"],
                    next_task["extra_info"],
                )
                group_futures[completed_instance_id] = (next_future, next_task["index"], next_task_idx + 1)
            else:
                del group_futures[completed_instance_id]
                # print(f"Instance group {completed_instance_id} completed ({len(tasks)} tasks)")

        # 处理最终结果
        for i, score_result in completed_results.items():
            scores[i] = float(score_result.get('score', 0.0))

            # 收集extra_info
            if 'extra_info' in score_result and isinstance(score_result['extra_info'], dict):
                for key, value in score_result['extra_info'].items():
                    if key not in extra_info_dict:
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        return scores, extra_info_dict

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        response_ids = data.batch['responses']
        sequences_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = [data_item.non_tensor_batch.get('extra_info', None) for data_item in data]

        assert len(sequences_strs) == len(ground_truths) == len(data_sources)

        # (NOTE) Qian: it is very important to use ray to compute score in parallel!
        if data_sources[0] == 'swe-bench':
            # For SWE-Bench, we use a grouped approach to avoid parallel execution of the same instance_id
            scores, extra_info_dict = self.swe_bench_compute_score_parallel_with_ray_grouped(
                data_sources, sequences_strs, ground_truths, extra_infos
            )
        else:
            scores, extra_info_dict = self.swe_compute_score_parallel_with_ray(
                data_sources, sequences_strs, ground_truths, extra_infos
            )

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        data_sources = data.non_tensor_batch['data_source']

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return {'reward_tensor': reward_tensor, 'extra_info': extra_info_dict}
