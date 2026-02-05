import logging
from bisect import bisect_left

import torch

logger = logging.getLogger(__file__)


class BaseAgent:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def reset(self):
        """Reset the agent."""
        pass

    def finalize(self, req, turn_rewards, finish_reason_type: str):
        """Finalize the agent."""
        return req.finalize(turn_rewards, finish_reason_type)

    def _truncate_to_prefix_ids(self, response_ids: list[int], target_str: str) -> torch.Tensor:
        """
        Truncate the response to a target string prefix

        Args:
            response_ids: LongTensor of shape (L,) with padding
            target_str: Target prefix string that is guaranteed to be a prefix
                of the decoded full response string.

        Example:
            response_ids = tensor(tokenizer.tokenize("It is a good day"))
            target_str = "It is a"
            truncated_ids = truncate_to_prefix_ids(response_ids, target_str)
            # truncated_ids will be:
            # tensor(tokenizer.tokenize("It is a"))

        Returns:
            truncated_ids: LongTensor of shape (Lcut,) containing the truncated
                token IDs without padding.
        """
        tokenizer = self.tokenizer

        # The target str must match the prefix of the decoded response
        decoded_response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        assert decoded_response.startswith(
            target_str
        ), f"Target string '{target_str}' is not a prefix of decoded response '{decoded_response}'"

        # Remove padding tokens from response_ids if any
        processed_response_ids = []
        for id in response_ids:
            if id != tokenizer.pad_token_id:
                processed_response_ids.append(id)

        if len(processed_response_ids) != len(response_ids):
            print(
                f"WARNING: Removed {len(response_ids) - len(processed_response_ids)} padding tokens from response_ids."
            )

        # Find the smallest truncation idx such that decode(response_ids[:idx]) == target_str
        # Use binary search with a tiny linear fallback if there is no exact matching during search
        n = len(processed_response_ids)
        if n == 0:
            logger.warning("Response IDs are empty after truncation. Returning original response_ids.")
            idx = 0
        else:
            # Binary search over token prefix length
            left, right = 1, n
            best = None
            while left <= right:
                mid = (left + right) // 2
                dec = tokenizer.decode(
                    processed_response_ids[:mid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if dec == target_str:
                    best = mid  # try to minimize further
                    right = mid - 1
                elif target_str.startswith(dec):
                    left = mid + 1  # need more tokens to cover target
                else:
                    right = mid - 1  # decoded has diverged, reduce

            if best is not None:
                idx = best
            else:
                # Fallback (very short linear probe) to be extra safe against odd spacing cases
                idx = 0
                for k in range(1, n + 1):
                    dec = tokenizer.decode(
                        processed_response_ids[:k],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if dec == target_str or dec.startswith(target_str):
                        idx = k
                        break

                # If still no match found, log warning and return original
                if idx == 0:
                    logger.warning(
                        f"Could not find truncation point for target string '{target_str}' in response. Returning original response_ids."
                    )
                    return processed_response_ids

        # Return truncated tensor without padding
        if idx > 0:
            return processed_response_ids[:idx]
        else:
            # return the original if idx is 0
            return response_ids

    async def generate_thought_and_action(
        self, response_token_ids: list[int], response_truncation: str
    ) -> tuple[str | None, str | None, bool | None, dict]:
        """
        Args:
            response_token_ids: list[int], the response token ids from the inference engine
            response_truncation: str, the response truncation regex patterns to use
        Returns:
            str, the preceding text, i.e., the reasoning
            str, the tool call, i.e., the action
            bool, whether the task is done
            dict, additional info
        """
        raise NotImplementedError
