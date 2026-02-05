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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Default reward computation function.

    Returns:
        Union[float, Dict[str, Any]]: Either a float score or a dictionary with 'score' and optional 'extra_info'
    """
    if data_source == 'openai/gsm8k':
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math

        res = math.compute_score(solution_str, ground_truth)

    elif data_source in ["LeetCode", "taco"]:
        from . import code

        res = code.compute_score(
            solution_str, ground_truth, extra_info=extra_info, pattern=r"```(?:py|python)\r?\n([\s\S]*?)\n```"
        )
    elif data_source in ["codeio_generated"]:
        from . import codeio

        res = codeio.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif (
        data_source in ['simplelr_math_35', 'big_math', 'deepscaler']
        or data_source.startswith('dapo')
        or data_source.startswith('deepscaler')
    ):
        from . import hf_math_verify

        res = hf_math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ['deepseek_r1']:
        from . import deepseek_r1

        res = deepseek_r1.compute_score(solution_str, ground_truth)
    elif data_source in ['sweb-extra-execute']:
        from . import swe

        res = swe.compute_score_execute(solution_str, extra_info=extra_info)
    elif data_source in ['sweb-extra-similarity']:
        from . import swe

        res = swe.compute_score_patch_similarity(solution_str, ground_truth, extra_info=extra_info)
    elif 'swe-bench' in data_source:
        from . import swe

        res = swe.compute_score_sweb(solution_str, ground_truth, extra_info=extra_info)
    elif "swe-agent" in data_source:
        from . import swe_agent

        res = swe_agent.compute_score_function(solution_str, extra_info=extra_info)
    elif "searchR1" in data_source:
        from . import search

        res = search.local_search_compute_score(solution_str, ground_truth)
    elif "synthetic" in data_source:
        from . import search

        res = search.file_search_compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    elif isinstance(res, dict):
        return res
    else:
        return float(res[0])
