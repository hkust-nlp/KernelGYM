import re

from verl_patch.workers.code.agent.base_agent import BaseAgent
from verl_patch.workers.code.agent_env.base_env import FinishReasonTypeEnum


class SearchAgent(BaseAgent):
    """
    Agent that supports multi-turn math solving with a python interpreter, capable of handling code execution and final answer extraction
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.query_re = re.compile(
            r"""
            (?P<before>.*?)                       # ① 第一个search query之前的所有文本
            (?P<block>
                <search>
                (?P<query>.*?)
                </search>
            )
            (?=[^\n]*(?:\r?\n|\Z))                # ④ 先行断言：窥视到行尾/文本尾
            """,
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        self.answer_re = re.compile(
            r"""
            (?P<before_answer>.*?)                 # 在<answer> 之前的所有内容
            (?P<answer_part>                       # <answer> 完整部分
                <answer>
                (?P<answer>.*?)
                </answer>                            # 闭标签
            )
            (?P<after_answer>.*)                   # <answer> 之后的所有内容
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

        # Detect the first query block
        query_match = self.query_re.search(response)
        if query_match:
            # Found a query block, use it
            preceding_text = query_match.group("before")
            full_query = query_match.group("block")
            done = False
            # Calculate the target string: preceding_text + full_query
            truncated_token_ids = self._truncate_to_prefix_ids(response_token_ids, preceding_text + full_query)
            agent_info = {}
        else:
            # No query block found, now check for the first answer
            preceding_text = response
            full_query = None

            # detect answer
            answer_match = self.answer_re.search(response)
            if answer_match:
                done = True
                # Get everything up to and including the <answer></answer> part
                text_up_to_boxed = answer_match.group("before_answer") + answer_match.group("answer_part")

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
        return truncated_response_str, truncated_token_ids, full_query, done, agent_info
