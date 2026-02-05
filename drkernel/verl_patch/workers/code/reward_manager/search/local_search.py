from typing import Any, Dict, List

from verl import DataProto

from verl_patch.utils.reward_score import _default_compute_score

# Import the base class and the provided scoring function
from verl_patch.workers.code.reward_manager.base import BaseRewardManager


class LocalSearchRewardManager(BaseRewardManager):
    """
    Reward Manager for local-searching agent tasks.

    This manager leverages an external scoring function that processes the entire
    agent response to find the final answer and determine the score.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        """
        Initializes the reward manager.

        Args:
            tokenizer: The tokenizer for decoding sequences.
            num_examine (int): The number of decoded responses to print.
            compute_score (callable, optional): The scoring function. Defaults to the
                adapter for the task-specific scorer.
        """
        scorer_to_use = compute_score or _default_compute_score

        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=scorer_to_use,
            timeout_seconds=10,  # A short timeout is sufficient for string parsing
        )

    def _get_solution_strs(self, data: DataProto) -> List[str]:
        """
        Provides the full, raw decoded response strings for scoring.

        The scoring function itself is responsible for parsing this string
        to find the submitted answer, so we don't need to do any extraction here.
        """
        response_ids = data.batch['responses']
        return self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
