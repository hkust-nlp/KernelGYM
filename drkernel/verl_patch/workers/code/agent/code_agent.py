import re

from verl_patch.workers.code.agent.base_agent import BaseAgent
from verl_patch.workers.code.agent_env.base_env import FinishReasonTypeEnum


class CodeAgent(BaseAgent):
    """
    Agent that supports multi-turn code generation, capable of handling code execution, self-test, and final answer extraction
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        # Regular expression to match Python code blocks
        self.response_truncation_re = {
            'python_code': re.compile(
                r"""
                (?P<before>.*?)                       # All text before the code block
                (?P<block>                            # Complete code block
                    ```[ \t]*(?:python|py)[ \t]*(?:\r?\n)?   # Opening fence (requires python/py)
                    (?P<code>.*?)                     # Code content
                    (?:\r?\n)?```                     # Closing fence
                )
                (?=[^\n]*(?:\r?\n|\Z))                # Lookahead: peek to end of line/text
                """,
                re.IGNORECASE | re.DOTALL | re.VERBOSE,
            ),
            # Regular expression to match answer blocks
            'answer_code': re.compile(
                r"""
                (?P<before>.*?)                       # All text before the answer block
                (?P<block>                            # Complete answer block
                    ```answer[ \t]*(?:\r?\n)?         # Opening fence: ```answer
                    (?P<code>.*?)                     # Code content
                    (?:\r?\n)?```                     # Closing fence
                )
                (?=[^\n]*(?:\r?\n|\Z))                # Lookahead: peek to end of line/text
                """,
                re.IGNORECASE | re.DOTALL | re.VERBOSE,
            ),
        }

    async def generate_thought_and_action(
        self, response_token_ids: list[int], response_truncation: str
    ) -> tuple[str | None, str | None, bool | None, dict]:
        # remove padding token ids
        response_token_ids = [id for id in response_token_ids if id != self.tokenizer.pad_token_id]
        # translate result_token_id back to string
        response = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)

        if response is None:
            return None, None, None, True, {}

        # convert response_truncation to list without whitespace
        if response_truncation is None:
            response_truncation = list(self.response_truncation_re.keys())
        else:
            response_truncation = [item.strip() for item in response_truncation.split(',')]

        # 综合所有模式，选取在字符串中最靠后出现的模式，不做截断
        candidates = []  # (start_index, mode, match)
        for idx, mode in enumerate(response_truncation):
            if mode not in self.response_truncation_re:
                raise ValueError(
                    f"Invalid response truncation: {mode}. Only {self.response_truncation_re.keys()} are supported."
                )
            matches = list(self.response_truncation_re[mode].finditer(response))
            if not matches:
                continue
            match = matches[-1]
            if mode in ("python_code", "answer_code"):
                start_pos = match.start("block")
            else:
                start_pos = match.start()
            candidates.append((start_pos, mode, match))

        if len(candidates) > 0:
            _, chosen_mode, chosen_match = max(candidates, key=lambda x: x[0])

            if chosen_mode == "python_code":
                # Found a code block, use it
                preceding_text = chosen_match.group("before")
                full_code_block = chosen_match.group("block")
                done = False
                truncated_token_ids = response_token_ids
                # # Calculate the target string: preceding_text + full_code_block
                # truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, preceding_text + full_code_block)
                agent_info = {}
            elif chosen_mode == "answer_code":
                # Found an answer block, use it
                preceding_text = chosen_match.group("before")
                full_code_block = chosen_match.group("block")
                done = True
                truncated_token_ids = response_token_ids
                # # Calculate the target string: preceding_text + full_answer_block
                # truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, preceding_text + full_code_block)
                agent_info = {'finish_type': FinishReasonTypeEnum.ANSWER}
            else:
                # Neither code block nor boxed answer found
                done = True
                full_code_block = None
                truncated_token_ids = response_token_ids
                agent_info = {'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL}
        else:
            # 无任何匹配
            done = True
            full_code_block = None
            truncated_token_ids = response_token_ids
            agent_info = {'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL}

        truncated_response_str = self.tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
        return truncated_response_str, truncated_token_ids, full_code_block, done, agent_info
