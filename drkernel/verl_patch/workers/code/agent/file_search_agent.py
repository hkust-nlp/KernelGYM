import re

from verl_patch.workers.code.agent.base_agent import BaseAgent
from verl_patch.workers.code.agent_env.base_env import FinishReasonTypeEnum


class FileSearchAgent(BaseAgent):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.first_block_re = re.compile(
            r"""
            (?P<before>.*?)                       # ① 第一个代码块之前的所有文本
            (?P<block>                            # ② <block>：从 ``` … 到最后3个```，不含尾字
                ```[ \t]*(?:bash|sh)?[ \t]*(?:\r?\n)?   # —— 开围栏
                (?P<code>.*?)                     # ③ <code>：纯代码体
                (?:\r?\n)?```                     # —— 闭围栏，只吃到反引号本身
            )
            (?=[^\n]*(?:\r?\n|\Z))                # ④ 先行断言：窥视到行尾/文本尾
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
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

        # Detect the first query block
        code_block = self.first_block_re.search(response)
        if code_block:
            # Found a query block, use it
            preceding_text = code_block.group("before")
            full_block = code_block.group("block")
            # Calculate the target string: preceding_text + full_query
            truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, preceding_text + full_block)
            agent_info = {}
            if "submit" in full_block:
                agent_info = {'finish_type': FinishReasonTypeEnum.ANSWER}
                done = True
            else:
                done = False
        else:
            # No action found
            done = True
            truncated_token_ids = response_token_ids
            agent_info = {'finish_type': FinishReasonTypeEnum.NO_TOOL_CALL}
            full_block = None

        truncated_response_str = self.tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

        return truncated_response_str, truncated_token_ids, full_block, done, agent_info
