import torch
import verl.utils.torch_functional as verl_F

def compute_turn_level_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def shape_rewards(rewards: torch.Tensor, max_turns: int, gamma: float, unbiased: bool = False) -> torch.Tensor:

    """
        Shaping rewards to residual rewards, i.e. rewards - rewards of the previous turn.

        r_t = r_t - r_{t-1}

    Args:
        rewards: `(torch.Tensor)`
            shape: (bs, max_turns)
        max_turns: `(int)`
            maximum number of turns
        gamma: `(float)`
            discounted factor used in RL
        unbiased: `(bool)`
            whether to use unbiased shaping, we could add a -$\gamma$ * rewards[max_turns-1] term to last turn's rewards to keep the optimal policy still.
    Returns:
        shaped_rewards: `(torch.Tensor)`
        shape: (bs x max_turns)
    """
    
    with torch.no_grad():
        rewards_to_use = rewards.reshape(-1, max_turns)
        if unbiased:
            rewards_to_use[:, -1] = rewards_to_use[:, -1] -gamma * rewards_to_use[:, -1]

        shaped_rewards = torch.zeros_like(rewards_to_use)

        for i in range(max_turns):
            if i == 0:
                shaped_rewards[:, i] = rewards_to_use[:, i]
            else:
                shaped_rewards[:, i] = rewards_to_use[:, i] - rewards_to_use[:, i - 1]

        shaped_rewards = shaped_rewards.reshape(-1)
    return shaped_rewards

# def add_fixed_term_rewards(rewards: torch.Tensor, max_turns: int, gamma: float) -> torch.Tensor:
    
#     '''
#         We could add a -$\gamma$ * rewards[max_turns-1] term to last turn's rewards to keep the optimal policy still.
#     '''

#     with torch.no_grad():
#         rewards_to_use = rewards.reshape(-1, max_turns)

#         rewards_to_use[:, -1] = rewards_to_use[:, -1] + -gamma * rewards_to_use[:, -1]

#         rewards_to_use = rewards_to_use.reshape(-1)

#     return rewards_to_use