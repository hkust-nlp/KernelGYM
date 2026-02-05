# vLLM Rollout Logprobs Collection

## Overview

This document explains how logprobs (log probabilities) are collected during vLLM rollout generation and what they represent.

## What are Rollout Logprobs?

Rollout logprobs are the log probabilities of tokens sampled during response generation. These values are crucial for:
- Computing likelihood ratios in RL training
- Evaluating sampling policies
- Debugging generation behavior

## Implementation Details

### Configuration

Logprobs collection is **always enabled** by default. The system automatically collects log probabilities for all sampled tokens during rollout generation.

### How Logprobs are Collected

1. **Request logprobs from vLLM**: Set `logprobs=1` in SamplingParams to request the top-1 logprob
2. **Extract sampled token logprobs**: For each generated token, extract its log probability
3. **Store in batch**: Save as `rollout_log_probs` aligned with response tokens

### What the Logprobs Represent

The collected logprobs represent the **actual sampling distribution** used to generate tokens:

1. **After Temperature Scaling**: `logits.div_(temperatures)`
2. **After Top-p/Top-k Filtering**: `_apply_top_k_top_p(logits, ...)`  
3. **Final Distribution**: `logprobs = torch.log_softmax(logits, dim=-1)`

These are the exact probabilities from which tokens were sampled during rollout.

## Code Structure

```python
# Key extraction logic in vllm_rollout_spmd.py
for i, logprob in enumerate(output.outputs[sample_id].logprobs):
    curr_log_prob.append(logprob[response_ids[i]].logprob)
```

Where:
- `response_ids[i]`: The sampled token ID at position i
- `logprob[token_id].logprob`: The log probability of that token

## Data Format

- **Type**: `torch.float32` tensor
- **Shape**: `(batch_size, response_length)`
- **Padding**: Padded positions use value `-1`
- **Storage**: Available in `batch['rollout_log_probs']`

## Debugging

Debug prints are included to verify collection:

```python
print(f"[vllm_rollout] responses_shape={tuple(response.shape)}, "
      f"rollout_log_probs_shape={tuple(rollout_log_probs.shape)}")
```

## Integration with PPO Training

The collected `rollout_log_probs` are automatically used for importance sampling correction in PPO training to handle off-policy issues when the vLLM rollout distribution differs from the policy being optimized. See the [vLLM IS documentation](../../../../docs/vllm_is/README.md) for details.

## Related Files

- `vllm_rollout_spmd.py`: Main implementation
- `verl_patch/docs/vllm_is/`: Complete importance sampling documentation
- vLLM source: `vllm/model_executor/layers/sampler.py` (upstream sampling logic)
