import asyncio
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


class CodeSandboxEnv(BaseEnv):
    """
    Sandbox environment that supports multi-turn code generation and self-test.
    Capable of executing Python code and handling code execution results
    """

    def __init__(self, max_turns: int = 2, extra_info: dict = None):
        super().__init__(max_turns=max_turns)
        self.extra_info = extra_info

        # Regular expression to extract code from code blocks (supports both ```python and ```answer)
        self.code_extraction_re = re.compile(
            r"""
            ```[ \t]*(?P<lang>python|py|answer)?[ \t]*(?:\r?\n)?   # Opening fence: ``` with optional python/py/answer identifier
            (?P<code>.*?)                            # Capture group: pure code content
            (?:\r?\n)?```                            # Closing fence: ```
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

    async def step(self, action: str | None) -> tuple[str | None, bool, bool, float, dict]:
        """
        Args:
            action: str, the action from the agent
        Returns:
            str, the tool response
            bool, whether the episode is done
            bool, whether the episode is truncated
            float, the reward of the current step
            dict, additional info
        """
        self.num_turns += 1
        done, truncate, reward = False, False, 0
        tool_response, tool_info = None, {}

        if action is None:
            done = True
            tool_info["finish_type"] = FinishReasonTypeEnum.NO_TOOL_CALL
        else:
            # execute tool call and obtain relative information
            try:
                exec_result = await self.exec_tool_call(action)
                tool_response, reward, done, tool_info = exec_result
            except asyncio.TimeoutError:
                # Handle timeout case
                tool_response = "Execution timed out."
                reward = 0.0
                done = True
                tool_info["finish_type"] = FinishReasonTypeEnum.ERROR
                tool_info["error_type"] = "timeout"
                tool_info["error_message"] = "Operation timed out after exhausting all retry attempts"
            except Exception as e:
                # Handle other exceptions that might occur during execution
                tool_response = f"Execution failed with error: {str(e)}"
                reward = 0.0
                done = True
                tool_info["finish_type"] = FinishReasonTypeEnum.ERROR
                tool_info["error_type"] = "execution_error"
                tool_info["error_message"] = str(e)
            else:
                if self.num_turns >= self.max_turns:
                    done = True
                    truncate = True
                    tool_info["finish_type"] = FinishReasonTypeEnum.MAX_TOOL_CALL

        return tool_response, done, truncate, reward, tool_info

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
        # Extract code block and language type
        match = self.code_extraction_re.search(action)
        lang = match.group("lang")
        code_content = match.group("code")

        # Check if it's an answer block
        if lang and lang.lower() == "answer":
            done = True
            tool_info = {"finish_type": FinishReasonTypeEnum.ANSWER}
            code_execution_result = None
        else:
            done = False
            code_to_execute = f"{self.extra_info['ground_truth']['import_code']}\n\n{code_content}"

            # Execute code
            run_result, run_status = await execute_single_task(
                code_to_execute, run_timeout=SANDBOX_RUN_TIMEOUT, client_timeout=SANDBOX_CLIENT_TIMEOUT, max_attempts=3
            )
            sandbox_stdout = run_result["stdout"]
            sandbox_stderr = run_result["stderr"]
            command_run_status = run_result["status"]
            code_execution_result = parse_sandbox_output(sandbox_stdout, sandbox_stderr, run_status, command_run_status)

            if code_execution_result is None or code_execution_result.strip() == "":
                code_execution_result = "The code is successfully executed, but did not print any output."

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
            }

        return code_execution_result, 0, done, tool_info
