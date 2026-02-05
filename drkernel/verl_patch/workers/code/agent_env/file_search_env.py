import asyncio
import os
import random
import re
import subprocess
import time

from verl_patch.workers.code.agent_env.base_env import BaseEnv, with_timeout_and_retry


class FileSearchEnv(BaseEnv):
    """
    A file-finding environment that operates on a real local filesystem.

    This environment confines operations to a "sandboxed" root directory
    to prevent access to or modification of external system files. Its role
    is purely to execute commands and maintain the current directory state.
    It supports multi-line commands in a single bash block.
    """

    def __init__(
        self,
        max_turns: int = 10,
        extra_info: dict = None,
    ):
        super().__init__(max_turns)
        self.root_dir = os.path.abspath(extra_info.get("root_dir", ""))
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"The specified root directory does not exist: {self.root_dir}")

        self.max_obs_words = extra_info.get("max_obs_words", 128)

        # Define a set of commands that are not allowed for security reasons
        self.BLOCKED_COMMANDS = {
            'rm',
            'mkdir',
            'mv',
            'cp',
            'touch',
            'chmod',
            'chown',
            'dd',
            'sudo',
            'pip',
            'conda',
            'ln',
            'git',
        }

        self.current_path = self.root_dir

        # Regular expression to extract code from a markdown-style block
        self.code_extraction_re = re.compile(
            r"""
            ```[ \t]*(?:bash|sh|zsh)?[ \t]*(?:\r?\n)?  # Optional language identifier
            (?P<code>.*?)                           # Capture group for the code
            (?:\r?\n)?```                             # Closing fence
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )

    def _truncate_observation(self, observation: str) -> str:
        """
        Truncates the observation string if it exceeds the word limit,
        preserving newlines and other whitespace.
        """
        # Find all non-whitespace sequences (words) and their positions
        matches = list(re.finditer(r'\S+', observation))

        if len(matches) <= self.max_obs_words:
            return observation

        half_limit = self.max_obs_words // 2

        # Find the character index to cut off the start
        start_cutoff_index = matches[half_limit - 1].end()

        # Find the character index to resume the end
        end_resume_index = matches[len(matches) - half_limit].start()

        start_content = observation[:start_cutoff_index]
        end_content = observation[end_resume_index:]

        return f"{start_content}\n---response truncated---\n{end_content}"

    async def reset(self, extra_info: dict) -> str:
        """
        Resets the environment to the root directory.
        """
        await super().reset(extra_info)
        self.current_path = self.root_dir

    @with_timeout_and_retry(timeout_seconds=100.0, max_attempts=1)
    async def exec_tool_call(self, action: str) -> tuple[str, float, bool, dict]:
        code_match = self.code_extraction_re.search(action)
        if code_match:
            action = code_match.group('code')
        action = action.strip()

        # The environment no longer handles 'submit' actions. This should be handled externally.
        assert not action.startswith('submit'), "Environment should not receive 'submit' actions."

        reward = 0.0  # Neutral reward, as the env's role is execution, not judgment.
        done = False
        info = {
            "execution_time": 0.0,
            "local_queue_time": 0.0,
            "sandbox_queue_time": 0.0,
        }

        # --- NEW LOGIC: Handle multi-line commands ---
        commands = action.strip().split('\n')
        all_outputs = []

        for command_line in commands:
            command_line = command_line.strip()
            if not command_line:
                continue

            commands_in_chain = [cmd.strip() for cmd in command_line.split('&&')]

            for command in commands_in_chain:

                output_prefix = f"$ {command}"
                command_output = ""

                # Command 1: cd (handled specially as it changes internal state)
                if command.startswith('cd '):
                    path = command.split(' ', 1)[1]
                    if path.startswith('"') and path.endswith('"'):
                        path = path[1:-1]

                    new_path = os.path.abspath(os.path.join(self.current_path, path))

                    if os.path.isdir(new_path) and new_path.startswith(self.root_dir):
                        self.current_path = new_path
                        command_output = (
                            f"Current directory changed to: {os.path.relpath(new_path, self.root_dir) or '.'}"
                        )
                    else:
                        command_output = f"Error: directory does not exist or access is denied '{path}'"

                # Command 2: All other bash commands (ls, find, cat, etc.)
                else:
                    # block "find /"
                    if command.startswith('find /'):
                        command_output = f"Error: Cannot search root directory."
                    else:
                        command_word = command.split()[0].replace('./', '')
                        if command_word in self.BLOCKED_COMMANDS:
                            command_output = f"Error: Command '{command_word}' is not allowed in this environment."
                            # Stop executing further commands in this block if a blocked one is found
                            all_outputs.append(f"{output_prefix}\n{command_output}")
                            break

                        try:
                            start_time = time.time()
                            process = await asyncio.create_subprocess_shell(
                                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.current_path
                            )
                            stdout, stderr = await process.communicate()
                            info["execution_time"] = time.time() - start_time

                            stdout_str = stdout.decode('utf-8', errors='ignore').strip()
                            stderr_str = stderr.decode('utf-8', errors='ignore').strip()

                            if process.returncode == 0:
                                command_output = (
                                    stdout_str if stdout_str else "Command executed successfully with no output."
                                )
                            else:
                                command_output = (
                                    f"Command failed with return code {process.returncode}:\n{stderr_str or stdout_str}"
                                )

                        except Exception as e:
                            command_output = f"An error occurred while executing the command: {e}"

                all_outputs.append(f"{output_prefix}\n{command_output}")

        observation = "\n\n".join(all_outputs)

        return self._truncate_observation(observation), reward, done, info


async def setup_test_directory(dir_name="numpy_sandbox"):
    """Creates a dummy directory for the main function to use."""
    if not os.path.exists(dir_name):
        print(f"Creating sandbox directory: '{dir_name}'")
        os.makedirs(os.path.join(dir_name, "numpy", "core"))
        with open(os.path.join(dir_name, "README.md"), "w") as f:
            f.write("This is a sandbox environment.")
    return dir_name


async def main():
    # Setup a dummy directory for the example
    sandbox_dir = await setup_test_directory()

    # Initialize the environment
    env = FileSearchEnv(max_turns=5, extra_info={"root_dir": sandbox_dir})
    observation = env.reset()

    print("--- Environment Initialized ---")
    print(observation)
    print("-----------------------------\n")

    # Simulate actions
    actions = [
        "```bash\nls -F\nls python```",
        "```bash\ncd numpy\ntree -L 2\n```",
    ]

    for action in actions:
        print(f">>> Executing action: {action}")
        observation, done, truncated, reward, info = await env.step(action)
        print(f"--- Observation ---\n{observation}\n")
        print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}\n")
        if done or truncated:
            break


if __name__ == '__main__':
    asyncio.run(main())
