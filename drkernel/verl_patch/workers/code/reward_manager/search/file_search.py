from typing import Any, Dict, List

from verl import DataProto

from verl_patch.utils.reward_score import _default_compute_score

# Import the base class and the provided scoring function
from verl_patch.workers.code.reward_manager.base import BaseRewardManager


def _file_search_score_adapter(
    data_source: str, solution_str: str, ground_truth: Dict, extra_info: Dict
) -> Dict[str, Any]:
    """
    Adapter to bridge the BaseRewardManager's scoring call with the provided compute_score.

    The BaseRewardManager calls with four arguments, but the specific `compute_score`
    for this task only needs two. This function handles the mapping.
    """
    # The 'solution_str' passed by the base manager is the full model response
    response = solution_str

    # The ground_truth from the dataset is a dictionary, e.g., {'target': [...]}.
    # We extract the list of paths for the scoring function.
    ground_truth_list = ground_truth.get('target', [])

    # Call the provided scoring function with the expected arguments
    return _default_compute_score(data_source, response, ground_truth_list)


class FileSearchRewardManager(BaseRewardManager):
    """
    Reward Manager for file searching agent tasks.

    This manager leverages an external scoring function that processes the entire
    agent response to find the final 'submit' action and determine the score.
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
        scorer_to_use = compute_score or _file_search_score_adapter

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
