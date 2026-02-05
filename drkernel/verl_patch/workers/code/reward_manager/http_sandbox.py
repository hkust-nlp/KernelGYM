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

import asyncio

import torch
from verl import DataProto

from verl_patch.utils.reward_score import _default_compute_score

try:
    from sandbox_fusion import (
        RunCodeRequest,
        RunStatus,
        run_code_async,
        set_sandbox_endpoint,
    )
except ImportError:
    print(
        "sandbox_fusion package is required for HttpSandboxRewardManager. "
        "Please install it via pip: pip install sandbox-fusion"
    )

import re

python_pattern = r"```python[ \t]*[\r\n]+(.*?)[ \t]*[\r\n]+```"
python_re = re.compile(python_pattern, re.DOTALL | re.IGNORECASE)


# extract the first python code block from the solution string
def python_extract_first(solution_str: str) -> str:
    match = python_re.search(solution_str)
    if match:
        return match.group(1)
    else:
        return None


# extract the last python code block from the solution string
def python_extract_last(solution_str: str) -> str:
    match = python_re.findall(solution_str)
    return match[-1] if match else None


async def single_sandbox_fusion_async(code, language='python', compile_timeout=1.0, run_timeout=3.0, semaphore=None):
    if semaphore is None:
        raise ValueError("Semaphore must be provided for async execution.")
    if code is None:
        response = {'status': RunStatus.Failed}
    else:
        request = RunCodeRequest(code=code, language=language, compile_timeout=compile_timeout, run_timeout=run_timeout)
        async with semaphore:
            try:
                response = await run_code_async(request, client_timeout=5.0)
                response = response.dict()
            except Exception as e:
                print(f"Error occurred: {e}")
                response = {'status': RunStatus.Failed}
    await asyncio.sleep(0.5)
    return response


async def parallel_sandbox_fusion_async(
    evaluation_func, completions, references, tasks, num_processes=64, sandbox_url=None
):
    # Use semaphore to control concurrency
    semaphore = asyncio.Semaphore(num_processes)
    if sandbox_url is None:
        raise ValueError("A url for sandbox is required")
    set_sandbox_endpoint(sandbox_url)

    sols = [python_extract_last(comp) for comp in completions]
    sol_and_tests = [(s + '\n' + t) if s is not None else None for s, t in zip(sols, references)]

    tasks_async = [single_sandbox_fusion_async(code, semaphore=semaphore) for code in sol_and_tests]

    try:
        results = await asyncio.gather(*tasks_async, return_exceptions=False)
    except Exception as e:
        print(f"Unexpected error in parallel computation: {e}")

    scores = []
    for result in results:
        if result['status'] == RunStatus.Success:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


class HttpSandboxRewardManager:
    """
    The Sandbox Reward Manager, changed from PrimeRewardManager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, sandbox_url=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = (
            compute_score or _default_compute_score
        )  # TODO: change to different compute scores (sandbox) later.
        if sandbox_url is None:
            self.sandbox_url = 'https://seed-sandbox.byteintl.net/faas/sandbox/'
        else:
            self.sandbox_url = sandbox_url

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = asyncio.run(
                parallel_sandbox_fusion_async(
                    self.compute_score,
                    sequences_str,
                    ground_truth,
                    data_sources,
                    num_processes=64,
                    sandbox_url=self.sandbox_url,
                )
            )
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0.0 for _ in range(len(sequences_str))]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                if type(sequences_str) is list:
                    print(f"There are {len(sequences_str)} sequences_str")
                    print(f"Printing the first {self.num_examine} sequence_str:")
                    for i in range(self.num_examine):
                        print(sequences_str[i])
                        print("-" * 100)
                else:
                    print(sequences_str)

        return reward_tensor
