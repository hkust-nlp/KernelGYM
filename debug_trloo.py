# debug_trloo.py
import torch
import numpy as np
from drkernel.verl_patch.trainer.code.ppo.core_algos import (
    compute_multi_turn_rloo_outcome_advantage,
    compute_multi_turn_returns,
)

# 模拟 3 个 prompt，每个 2 个 sample，max_turns=3
bs = 6  # 3 prompts × 2 samples, 然后 × 3 turns = 18 rows
max_turns = 3
total_rows = 6 * max_turns  # 18
response_length = 32

# 构造输入
token_level_rewards = torch.zeros(total_rows, response_length)
# 在每行最后一个 token 放一个 reward
for i in range(total_rows):
    token_level_rewards[i, -1] = float(i % 5)  # 一些不同的 reward

eos_mask = torch.ones(total_rows, response_length)
loss_mask = torch.ones(total_rows)  # 所有 turn 都 valid
turn_indices = torch.tensor([t for _ in range(6) for t in range(max_turns)])

# uid: 每个 prompt 的 2 个 sample 相同
index = np.array([f"prompt_{i//2}" for i in range(6) for _ in range(max_turns)])

advantages, returns = compute_multi_turn_rloo_outcome_advantage(
    token_level_rewards=token_level_rewards,
    eos_mask=eos_mask,
    loss_mask=loss_mask,
    turn_indices=turn_indices,
    index=index,
    max_turns=max_turns,
    gamma=1.0,
)

print(f"advantages shape: {advantages.shape}")
print(f"returns shape: {returns.shape}")
# 检查每个 prompt group 的 LOO 逻辑
for i in range(total_rows):
    if advantages[i].sum() != 0:
        print(f"row {i}: uid={index[i]}, turn={turn_indices[i].item()}, "
              f"adv={advantages[i, -1].item():.4f}, ret_scalar={returns[i, -1].item():.4f}")
