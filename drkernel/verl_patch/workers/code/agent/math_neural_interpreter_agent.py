import re

from verl_patch.workers.code.agent.base_agent import BaseAgent
from verl_patch.workers.code.agent_env.base_env import FinishReasonTypeEnum


class MathNeuralInterpreterAgent(BaseAgent):
    """
    Agent that supports multi-turn math solving with a python interpreter, capable of handling code execution and final answer extraction
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.response_truncation_re = {
            'python_code': re.compile(
                r"""
                (?P<before>.*?)                       # ① 第一个代码块之前的所有文本
                (?P<block>                            # ② <block>：从 ``` … 到最后3个```，不含尾字
                    ```[ \t]*(?:python|py)[ \t]*(?:\r?\n)?   # Opening fence (requires python/py)
                    (?P<code>.*?)                     # ③ <code>：纯代码体
                    (?:\r?\n)?```                     # —— 闭围栏，只吃到反引号本身
                )
                (?=[^\n]*(?:\r?\n|\Z))                # ④ 先行断言：窥视到行尾/文本尾
                """,
                re.IGNORECASE | re.DOTALL | re.VERBOSE,
            ),
            'output_prediction': re.compile(
                r"""
                (?P<before>.*?)                       # ① 第一个输出预测块之前的所有文本
                (?P<block>                            # ② <block>：从 ``` … 到最后3个```，不含尾字
                    ```output[ \t]*(?:\r?\n)?         # Opening fence: ```output
                    (?P<code>.*?)                     # ③ <code>：纯代码体
                    (?:\r?\n)?```                     # —— 闭围栏，只吃到反引号本身
                )
                (?=[^\n]*(?:\r?\n|\Z))                # ④ 先行断言：窥视到行尾/文本尾
                """,
                re.IGNORECASE | re.DOTALL | re.VERBOSE,
            ),
            'boxed_answer': re.compile(
                r"""
                (?P<before_boxed>.*?)                 # 在 \boxed 之前的所有内容
                (?P<boxed_part>                       # \boxed{...} 完整部分
                    \\boxed\s*                        # \boxed 可能带空白
                    \{                                # 开大括号
                    (?P<answer>                       # 括号内的答案内容
                        (?:                           # 非捕获组，匹配：
                            [^{}]                     # 非大括号字符
                            |                         # 或者
                            \{(?:[^{}]|\{[^{}]*\})*\} # 嵌套大括号（一层嵌套）
                        )*                            # 重复任意次
                    )
                    \}                                # 闭大括号
                )
                (?P<after_boxed>.*)                   # \boxed{...} 之后的所有内容
                """,
                re.DOTALL | re.VERBOSE,
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

        # 综合所有模式，选取在字符串中最靠前出现的模式并据此截断
        candidates = []  # (start_index, end_index, mode, match)
        for idx, mode in enumerate(response_truncation):
            if mode not in self.response_truncation_re:
                raise ValueError(
                    f"Invalid response truncation: {mode}. Only {self.response_truncation_re.keys()} are supported."
                )
            match = self.response_truncation_re[mode].search(response)
            if match is None:
                continue
            if mode in ("python_code", "output_prediction"):
                start_pos = match.start("block")
            elif mode == "boxed_answer":
                start_pos = match.start("boxed_part") if "boxed_part" in match.re.groupindex else match.start()
            else:
                start_pos = match.start()
            candidates.append((start_pos, mode, match))

        if len(candidates) > 0:
            candidates.sort(key=lambda x: x[0])
            _, chosen_mode, chosen_match = candidates[0]

            if chosen_mode == "python_code":
                # Found a code block, use it
                preceding_text = chosen_match.group("before")
                full_code_block = chosen_match.group("block")
                done = False
                # Calculate the target string: preceding_text + full_code_block
                truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, preceding_text + full_code_block)
                agent_info = {}
            elif chosen_mode == "output_prediction":
                # Found a output prediction block, use it
                preceding_text = chosen_match.group("before")
                full_code_block = None
                full_output_prediction_block = chosen_match.group("block")
                done = False
                # Calculate the target string: preceding_text + full_output_prediction_block
                truncated_token_ids = self._truncate_to_prefix_ids(
                    response_token_ids, preceding_text + full_output_prediction_block
                )
                agent_info = {}
            elif chosen_mode == "boxed_answer":
                done = True
                # Get everything up to and including the \boxed{...} part
                text_up_to_boxed = chosen_match.group("before_boxed") + chosen_match.group("boxed_part")

                # Truncate response_token_ids to match the text_up_to_boxed
                truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, text_up_to_boxed)
                # specify finish type
                full_code_block = None
                agent_info = {'finish_type': FinishReasonTypeEnum.ANSWER}
            else:
                # Neither code block nor boxed answer found
                done = True
                truncated_token_ids = response_token_ids
                full_code_block = None
                agent_info = {'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL}
        else:
            # 无任何匹配
            done = True
            truncated_token_ids = response_token_ids
            full_code_block = None
            agent_info = {'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL}

        truncated_response_str = self.tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
        return truncated_response_str, truncated_token_ids, full_code_block, done, agent_info
