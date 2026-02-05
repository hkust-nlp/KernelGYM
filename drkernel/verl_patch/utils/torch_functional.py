"""
Contain small torch utilities
"""

from typing import Dict, List, Optional, Union

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from verl.utils.torch_functional import logprobs_from_logits_v2

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels):
    output = cross_entropy_loss(logits, labels)
    assert isinstance(
        output, tuple
    ), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    return -output[0]


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels


def compute_sum_pi_squared_from_logits(logits: torch.Tensor):
    """
    Compute exact sum of squared probabilities from logits.
    Formula: Σπ² = exp(logsumexp(2*logits) - 2*logsumexp(logits))

    Used for optimal baseline variance reduction as described in
    "What Matters for Model Merging at Scale?" (arXiv:2410.03617)

    Args:
        logits: Logits tensor (..., vocab_size).

    Returns:
        Sum of squared probabilities tensor (...).
    """
    return torch.exp(torch.logsumexp(2.0 * logits, dim=-1) - 2.0 * torch.logsumexp(logits, dim=-1))


"""
Optimizer related
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_sigmoid_decay_schedule(
    optimizer: Optimizer,
    lr_low: float,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        k = (num_warmup_steps * 0.1) ** -1
        lr_high = 1
        lr = (lr_high - lr_low) / (1 + math.exp(k * (current_step - num_warmup_steps))) + lr_low
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_final_eos_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    """
    修改逻辑，让它截取到最后一个 EOS：
    1. 找到所有 eos_token 的位置
    2. 翻转后进行累加（cumsum）
    3. cumsum > 0 的位置表示在原序列中已经出现最后一个 eos（含）之后
    4. 再次翻转回去即得到：在最后一个 eos 之前(含)都为 1，否则为 0
    如果需要处理“没有 eos_token 时全为 1”的情况，可额外根据 cumsum 是否全零来处理
    """
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    # 找到所有的 eos 位置
    eos_mask = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_id.eq(token)

    # 转成 0/1
    eos_mask = eos_mask.long()

    # 翻转 -> 累加 -> 取 > 0 -> 翻转回去
    reversed_mask = torch.flip(eos_mask, dims=[1])
    reversed_cumsum = torch.cumsum(reversed_mask, dim=1)
    # 只要累加结果 > 0，说明在原序列中，该位置在最后一个 EOS 及其左侧
    new_mask = reversed_cumsum.gt(0)
    new_mask = torch.flip(new_mask, dims=[1])

    return new_mask.to(dtype)


def get_eos_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    '''
    end of sentence token can be int or list: 1 or [1, 2]
    e.g. eos_token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    '''
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    eos_mask = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_id.eq(token)

    eos_mask = eos_mask.long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def masked_max(values, mask, axis=None, return_indice=False):
    """Compute maximum of tensor with masked values."""
    values_masked = values.clone()
    values_masked[mask == 0] = float('-inf')
    if return_indice:
        assert axis is not None
        return values_masked.max(dim=axis)
    return values_masked.max(dim=axis)[0] if axis is not None else values_masked.max()


def masked_min(values, mask, axis=None, return_indice=False):
    """Compute minimum of tensor with masked values."""
    values_masked = values.clone()
    values_masked[mask == 0] = float('inf')
    if return_indice:
        assert axis is not None
        return values_masked.min(dim=axis)
    return values_masked.min(dim=axis)[0] if axis is not None else values_masked.min()
