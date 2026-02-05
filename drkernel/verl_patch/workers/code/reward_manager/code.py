import asyncio
import json
import os

import ray
import torch
from ray.exceptions import GetTimeoutError, RayTaskError
from verl import DataProto

from verl_patch.utils.reward_score import _default_compute_score

from .utils import reward_func_timeout_ray


class CodeRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, timeout_seconds=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        # Allow timeout to be configured via environment variable or parameter
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        else:
            self.timeout_seconds = int(os.environ.get('REWARD_TIMEOUT_SECONDS', 180))
        print(f"CodeRewardManager initialized with timeout: {self.timeout_seconds} seconds")

    def code_compute_score_parallel_with_ray(self, data_sources, solution_strs, ground_truths, extra_infos):
        # Compute rewards
        scores: list[float] = [0.0] * len(solution_strs)
        extra_info_dict: dict[str, list[float]] = {}  # Key -> list of values for the batch

        futures = []
        for i in range(len(solution_strs)):
            ground_truth = ground_truths[i]
            solution_str = solution_strs[i]
            data_source = data_sources[i]
            extra_info = extra_infos[i]

            future = reward_func_timeout_ray.remote(
                self.compute_score,
                self.timeout_seconds,
                data_source,
                solution_str,
                ground_truth,
                extra_info,
            )
            futures.append(future)

        default_fail_score = {
            "score": 0.0,
            "extra_info": {
                "score": 0.0,
                "has_code": 0,
                "valid_code": 0,
                "is_filter": 1,
                "sandbox_failed": 0,
            },
        }  # Default on error which should be filtered

        for i, future in enumerate(futures):
            try:
                task_result = ray.get(future, timeout=self.timeout_seconds)

                assert (
                    "extra_info" in task_result
                ), f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                score_result = task_result
                if "score" not in task_result["extra_info"]:
                    score_result = default_fail_score
                elif "is_filter" not in task_result["extra_info"]:
                    score_result["extra_info"].update({"is_filter": 0})
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

            scores[i] = float(score_result.get("score", 0.0))

            if "extra_info" in score_result and isinstance(score_result["extra_info"], dict):
                for key, value in score_result["extra_info"].items():
                    if key not in extra_info_dict:
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        return scores, extra_info_dict

    async def code_compute_score_parallel_with_ray_async(self, data_sources, solution_strs, ground_truths, extra_infos):
        scores: list[float] = [0.0] * len(solution_strs)
        extra_info_dict: dict[str, list[float]] = {}

        futures = []
        for i in range(len(solution_strs)):
            future = reward_func_timeout_ray.remote(
                self.compute_score,
                self.timeout_seconds,
                data_sources[i],
                solution_strs[i],
                ground_truths[i],
                extra_infos[i],
            )
            futures.append(future)

        default_fail_score = {
            "score": 0.0,
            "extra_info": {
                "score": 0.0,
                "has_code": 0,
                "valid_code": 0,
                "is_filter": 1,
                "sandbox_failed": 0,
            },
        }

        results = await asyncio.gather(*futures, return_exceptions=True)
        for i, task_result in enumerate(results):
            if isinstance(task_result, Exception):
                print(
                    f"Error processing item {i} (gold='{str(ground_truths[i])[:50]}...', "
                    f"target='{str(solution_strs[i])[:50]}...'): {task_result}"
                )
                if isinstance(task_result, RayTaskError):
                    print(task_result)
                score_result = default_fail_score
            else:
                assert (
                    "extra_info" in task_result
                ), f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                score_result = task_result
                if "score" not in task_result["extra_info"]:
                    score_result = default_fail_score
                elif "is_filter" not in task_result["extra_info"]:
                    score_result["extra_info"].update({"is_filter": 0})

            scores[i] = float(score_result.get("score", 0.0))

            if "extra_info" in score_result and isinstance(score_result["extra_info"], dict):
                for key, value in score_result["extra_info"].items():
                    if key not in extra_info_dict:
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        return scores, extra_info_dict

    def __call__(self, data: DataProto):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        response_ids = data.batch["responses"]
        sequences_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        if isinstance(data[0].non_tensor_batch["reward_model"], str):
            ground_truths = [
                json.loads(data_item.non_tensor_batch["reward_model"])["ground_truth"] for data_item in data
            ]
        else:
            ground_truths = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch["data_source"]
        extra_infos = [data_item.non_tensor_batch.get("extra_info", None) for data_item in data]

        assert len(sequences_strs) == len(ground_truths) == len(data_sources)

        print(f"Scoring process started over {len(sequences_strs)} samples, waiting for results...")
        scores, extra_info_dict = self.code_compute_score_parallel_with_ray(
            data_sources, sequences_strs, ground_truths, extra_infos
        )

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        data_sources = data.non_tensor_batch["data_source"]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return {"reward_tensor": reward_tensor, "extra_info": extra_info_dict}

    async def score_raw_responses_async(self, response_strs, ground_truths, data_sources, extra_infos):
        # Compute scalar scores only, not token-level tensor
        assert (
            len(response_strs) == len(ground_truths) == len(data_sources)
        ), f"length mismatch:\nresponse_strs: {response_strs}\nground_truths: {ground_truths}\ndata_sources: {data_sources}\nextra_infos: {extra_infos}"

        scores, extra_info_dict = await self.code_compute_score_parallel_with_ray_async(
            data_sources, response_strs, ground_truths, extra_infos
        )

        return {"scores": [float(s) for s in scores], "extra_info": extra_info_dict}


def _compute_score_with_binary_scoring(data_source, solution_str, ground_truth, extra_info=None):
    """包装函数，将小于1.0的分数直接赋值为0.0"""
    result = _default_compute_score(data_source, solution_str, ground_truth, extra_info)

    if result["score"] < 1.0:
        result["score"] = 0.0

    return result


class CodeBinaryRewardManager(CodeRewardManager):
    """继承自CodeRewardManager，小于1.0的分数直接赋值为0.0"""

    def __init__(self, tokenizer, num_examine, compute_score=None, timeout_seconds=None) -> None:
        # 调用父类构造函数，使用自定义的compute_score
        super().__init__(
            tokenizer, num_examine, compute_score=_compute_score_with_binary_scoring, timeout_seconds=timeout_seconds
        )

        print(
            f"CodeBinaryRewardManager initialized with timeout: {self.timeout_seconds} seconds (binary scoring: <1.0 -> 0.0)"
        )
