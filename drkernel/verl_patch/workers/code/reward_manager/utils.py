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

from typing import Any, Callable, Tuple

import ray


def _strip_optional_timeout(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Drop a leading timeout_seconds positional arg if present.

    We support both call styles:
    - reward_func_timeout_ray.remote(func, data_source, solution_str, ...)
    - reward_func_timeout_ray.remote(func, timeout_seconds, data_source, solution_str, ...)
    """
    if len(args) >= 2 and isinstance(args[0], (int, float)) and isinstance(args[1], str):
        return args[1:]
    return args


@ray.remote(num_cpus=4, max_calls=5000)
def reward_func_timeout_ray(func: Callable, *args: Any, **kwargs: Any):
    """Run reward compute function in Ray worker.

    Timeout is enforced by ray.get in the caller; this function simply executes.
    """
    _ = kwargs.pop("timeout_seconds", None)
    call_args = _strip_optional_timeout(args)
    try:
        return func(*call_args, **kwargs)
    except Exception as e:
        print(f"Error in reward computation: {e}")
        return {"score": 0.0, "extra_info": {"is_filter": 1}}
