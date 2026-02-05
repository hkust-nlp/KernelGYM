import re

from verl_patch.workers.code.agent.base_agent import BaseAgent
from verl_patch.workers.code.agent_env.base_env import FinishReasonTypeEnum


class MathAgent(BaseAgent):
    """
    Agent that supports multi-turn math solving with a python interpreter, capable of handling code execution and final answer extraction
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.first_block_re = re.compile(
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
        )
        self.boxed_re = re.compile(
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
        )

    async def generate_thought_and_action(
        self, response_token_ids: list[int], response_truncation: str
    ) -> tuple[str | None, str | None, bool | None, dict]:
        # remove padding token ids
        response_token_ids = [id for id in response_token_ids if id != self.tokenizer.pad_token_id]
        # translate result_token_id back to string
        response = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)

        if response is None:
            return None, None, None, True, {}

        # Detect the first code block
        code_match = self.first_block_re.search(response)
        if code_match:
            # Found a code block, use it
            preceding_text = code_match.group("before")
            full_code_block = code_match.group("block")
            done = False
            # Calculate the target string: preceding_text + full_code_block
            truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, preceding_text + full_code_block)
            agent_info = {}
        else:
            # No code block found, now check for the first boxed answer
            preceding_text = response
            full_code_block = None

            # detect boxed answer
            boxed_match = self.boxed_re.search(response)
            if boxed_match:
                done = True
                # Get everything up to and including the \boxed{...} part
                text_up_to_boxed = boxed_match.group("before_boxed") + boxed_match.group("boxed_part")

                # Truncate response_token_ids to match the text_up_to_boxed
                truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, text_up_to_boxed)
                # specify finish type
                agent_info = {'finish_type': FinishReasonTypeEnum.ANSWER}
            else:
                # Neither code block nor boxed answer found
                done = True
                truncated_token_ids = response_token_ids
                agent_info = {'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL}

        truncated_response_str = self.tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
        return truncated_response_str, truncated_token_ids, full_code_block, done, agent_info
