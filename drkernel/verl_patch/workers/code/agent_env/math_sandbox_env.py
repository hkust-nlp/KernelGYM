import hashlib
import os
import re

from sandbox_fusion import CommandRunStatus, RunStatus

from verl_patch.utils.tools.sandbox_fusion import (
    execute_single_task,
    parse_sandbox_output,
)
from verl_patch.workers.code.agent_env.base_env import (
    BaseEnv,
    FinishReasonTypeEnum,
    with_timeout_and_retry,
)

# 从环境变量获取超时配置
SANDBOX_RUN_TIMEOUT = int(os.getenv("SANDBOX_RUN_TIMEOUT", 5))
SANDBOX_CLIENT_TIMEOUT = int(os.getenv("SANDBOX_CLIENT_TIMEOUT", 30))
print(f"SANDBOX_RUN_TIMEOUT: {SANDBOX_RUN_TIMEOUT}, SANDBOX_CLIENT_TIMEOUT: {SANDBOX_CLIENT_TIMEOUT}")


class MathSandboxEnv(BaseEnv):
    """
    This environment is used to execute the code in the sandbox.
    The code is extracted from the action using a regular expression.
    """

    def __init__(self, max_turns: int = 2, extra_info: dict = None):
        super().__init__(max_turns=max_turns)

        # 从full_code_block中提取code的正则表达式
        self.code_extraction_re = re.compile(
            r"""
            ```[ \t]*(?:python|py)?[ \t]*(?:\r?\n)?   # 开围栏：```可选python/py标识
            (?P<code>.*?)                            # 捕获组：纯代码内容
            (?:\r?\n)?```                            # 闭围栏：```
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

    @with_timeout_and_retry(timeout_seconds=5000.0, max_attempts=1)
    async def exec_tool_call(self, action: str) -> tuple[str, float, bool, dict]:
        """
        Args:
            action: str, the code from the agent
        Returns:
            str, the tool response
            float, the reward of the current step
            bool, whether the episode is done
            dict, additional info
        """
        code_content = self.code_extraction_re.search(action).group("code")
        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        # This code is borrowed from ReTool codebase.
        lines = code_content.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print") and not lines[i].startswith("final_answer"):
                lines[i] = f"print({line})"
            break
        code_content = "\n".join(lines)
        # wrap the code to output final answer in \boxed{}
        code_content = (
            """
def final_answer(result):
    print(f"\\\\boxed{{{result}}}")

"""
            + code_content
        )

        # Try to get result from cache
        # cached_result = self.cache.get(code_content)
        cached_result = None  # Temporarily disable cache

        if cached_result is not None:
            # Use cached result
            code_execution_result = cached_result['execution_result']
            code_execution_status = cached_result['code_execution_status']
            tool_info = {"code_execution_status": code_execution_status, "from_cache": True}
        else:
            # Execute code
            run_result, run_status = await execute_single_task(
                code_content, run_timeout=SANDBOX_RUN_TIMEOUT, client_timeout=SANDBOX_CLIENT_TIMEOUT, max_attempts=3
            )
            sandbox_stdout = run_result["stdout"]
            sandbox_stderr = run_result["stderr"]
            command_run_status = run_result["status"]
            code_execution_result = parse_sandbox_output(sandbox_stdout, sandbox_stderr, run_status, command_run_status)

            tool_info = {
                "is_correct_code": run_status == RunStatus.Success,
                "is_error_code": run_status == RunStatus.Failed and command_run_status == CommandRunStatus.Finished,
                "is_other_error": run_status == RunStatus.Failed and command_run_status == CommandRunStatus.Error,
                "is_timeout_code": run_status == RunStatus.Failed
                and command_run_status == CommandRunStatus.TimeLimitExceeded,
                "is_sandbox_error": run_status == RunStatus.SandboxError,
                "local_queue_time": run_result["local_queue_time"],
                "sandbox_queue_time": run_result["sandbox_queue_time"],
                "execution_time": run_result["execution_time"],
                "sandbox_instance_name": run_result.get("sandbox_instance_name", ""),
                "from_cache": False,
            }

            # Store result in cache
            # cache_value = {
            #     'execution_result': code_execution_result,
            #     'is_correct': is_correct_code,
            #     'stdout': sandbox_stdout,
            #     'stderr': sandbox_stderr,
            #     'success': sandbox_success,
            # }
            # self.cache.put(code_content, cache_value)

        if "\\boxed" in code_execution_result:
            # use `final_answer` to output boxed answer
            done = True
            tool_info["finish_type"] = FinishReasonTypeEnum.ANSWER
        else:
            done = False

        return code_execution_result, 0, done, tool_info
