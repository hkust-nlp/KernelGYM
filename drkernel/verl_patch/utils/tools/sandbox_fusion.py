import asyncio
import os
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

import sandbox_fusion
from sandbox_fusion.models import BaseModel, CommandRunResult, RunStatus


# Create a patched version of RunCodeResponse with sandbox_instance_name
class PatchedRunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}
    sandbox_instance_name: Optional[str] = None  # Added field


# Monkey patch the original class
sandbox_fusion.RunCodeResponse = PatchedRunCodeResponse

from sandbox_fusion import (
    CommandRunStatus,
    RunCodeRequest,
    run_code_async,
    set_sandbox_endpoint,
)

# 全局信号量，控制并发数
GLOBAL_SEMAPHORE = asyncio.Semaphore(500)

# 从环境变量获取 sandbox endpoint，如果没有则使用默认值
DEFAULT_SANDBOX_ENDPOINT = "https://seed-sandbox.byteintl.net/faas/sandbox/"
SANDBOX_ENDPOINT = os.getenv("SANDBOX_ENDPOINT", DEFAULT_SANDBOX_ENDPOINT)

print(f"Using sandbox endpoint: {SANDBOX_ENDPOINT}")

# 初始化sandbox端点
set_sandbox_endpoint(SANDBOX_ENDPOINT)


T = TypeVar('T')


async def execute_single_task(
    code,
    stdin: str = None,
    language="python",
    run_timeout=None,
    max_attempts=1,
    compile_timeout=1.0,
    client_timeout=None,
):
    """执行单个任务"""
    request = RunCodeRequest(
        code=code,
        stdin=stdin,
        language=language,
        compile_timeout=compile_timeout,
        run_timeout=run_timeout,
    )
    tool_t0 = time.time()
    async with GLOBAL_SEMAPHORE:
        tool_t1 = time.time()
        queue_time = tool_t1 - tool_t0
        response = await run_code_async(request, client_timeout=client_timeout, max_attempts=max_attempts)
        tool_t2 = time.time()

    response = response.dict()

    status = response["status"]
    run_result = response["run_result"]

    run_result["local_queue_time"] = queue_time
    run_result["sandbox_queue_time"] = tool_t2 - tool_t1 - run_result["execution_time"]

    # Add sandbox_instance_name if it exists in the response
    if "sandbox_instance_name" in response:
        run_result["sandbox_instance_name"] = response["sandbox_instance_name"]

    return run_result, status


async def single_sandbox(
    code, stdin: str = None, language="python", compile_timeout=1.0, run_timeout=3.0, semaphore=None
):
    request = RunCodeRequest(
        code=code,
        stdin=stdin,
        language=language,
        compile_timeout=compile_timeout,
        run_timeout=run_timeout,
    )
    async with semaphore:
        response = await run_code_async(request, client_timeout=30.0, max_attempts=2)
        response = response.dict()

    # Add sandbox_instance_name to run_result if it exists in the response
    if "sandbox_instance_name" in response and "run_result" in response:
        response["run_result"]["sandbox_instance_name"] = response["sandbox_instance_name"]

    # await asyncio.sleep(2)
    return response


async def parallel_sandbox(tasks, stdin_list=None, num_processes=200):
    semaphore = asyncio.Semaphore(num_processes)
    set_sandbox_endpoint(SANDBOX_ENDPOINT)
    if stdin_list is None:
        tasks_async = [single_sandbox(code, semaphore=semaphore) for code in tasks]
    else:
        tasks_async = [single_sandbox(code, stdin, semaphore=semaphore) for code, stdin in zip(tasks, stdin_list)]
    results = await asyncio.gather(*tasks_async, return_exceptions=True)

    # Process results, handling exceptions
    success_list = []
    stdout_list = []
    stderr_list = []

    for r in results:
        if isinstance(r, Exception):
            # Handle timeout or other exceptions
            success_list.append(False)
            stdout_list.append("")
            stderr_list.append(f"Exception: {str(r)}")
        else:
            success_list.append(r["status"] == RunStatus.Success)
            stdout_list.append(r["run_result"]["stdout"])
            stderr_list.append(r["run_result"]["stderr"])

    return success_list, stdout_list, stderr_list


def truncate_content(content: str, max_length: int) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )


def parse_sandbox_output(sandbox_stdout, sandbox_stderr, run_status: RunStatus, command_run_status: CommandRunStatus):
    """
    Args:
        sandbox_stdout: str, the stdout of the sandbox
        sandbox_stderr: str, the stderr of the sandbox
        run_status: RunStatus, [Success, Failed, SandboxError]
        command_run_status: CommandRunStatus, [Finished, Error, TimeLimitExceeded]

    Returns:
        obs: str, agent's next observation
        finish_reason: dict, the finish reason of the sandbox
    """
    stdout = str(sandbox_stdout)
    stderr = str(sandbox_stderr)

    if run_status == RunStatus.Success:
        if len(stdout) > 0:
            truncated_stdout = truncate_content(stdout, max_length=512)
            obs = truncated_stdout
        else:
            obs = ""
    elif run_status == RunStatus.Failed:
        if command_run_status == CommandRunStatus.Finished:
            if len(stderr) > 0:
                stderr_lines = stderr.splitlines()
                # only keep the last error line
                truncated_stderr = truncate_content("\n".join(stderr_lines[-1:]), max_length=512)
                obs = truncated_stderr
            else:
                obs = "Code execution failed"
        elif command_run_status == CommandRunStatus.Error:
            obs = "Code execution failed"
        elif command_run_status == CommandRunStatus.TimeLimitExceeded:
            obs = "Time limit exceeded"
    elif run_status == RunStatus.SandboxError:
        obs = "Sandbox service error"
    else:
        raise ValueError(f"Unknown execution status: {run_status} {command_run_status}")

    return obs


class CodeParser:
    def __init__(self) -> None:
        self.first_block_re = re.compile(
            r"""
            (?P<before>.*?)                       # ① 第一个代码块之前的所有文本
            (?P<block>                            # ② <block>：从 ``` … 到最后3个```，不含尾字
                ```[ \t]*(?:python|py)?[ \t]*(?:\r?\n)?   # —— 开围栏
                (?P<code>.*?)                     # ③ <code>：纯代码体
                (?:\r?\n)?```                     # —— 闭围栏，只吃到反引号本身
            )
            (?=[^\n]*(?:\r?\n|\Z))                # ④ 先行断言：窥视到行尾/文本尾
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

    def parse_tool_call(self, content) -> tuple[str, str, str] | None:
        match = self.first_block_re.search(content)
        if match:
            preceding_text = match.group("before").strip()
            full_code_block = match.group("block")
            code = match.group("code")
            # enable using `final_answer()` to output boxed result
            code = (
                """
def final_answer(result):
    print(f"\\\\boxed{{{result}}}")

"""
                + code
            )
            return preceding_text, full_code_block, code

        return None
