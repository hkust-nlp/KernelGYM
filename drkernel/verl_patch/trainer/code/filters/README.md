# PPO Batch Filtering System

Unified filtering pipeline for PPO training that ensures data quality while minimizing skip rates.

## Key Principles

1. **No Individual Length Filtering**: Samples are NEVER filtered by length individually
2. **Group-Level Filtering Only**: Length filtering applies to entire groups, not individual samples
3. **Proportional Selection**: Maintains model's natural positive/negative distribution
4. **Smart Selection**: Uses efficiency strategies instead of hard filtering

## Quick Start

```python
from verl_patch.trainer.code.filters import PPOBatchFilter, PPOFilterConfig

# Configure the unified filter
config = PPOFilterConfig(
    # Sample selection strategy
    sample_selection_strategy='efficiency_stochastic',  # Default: adds exploration

    # Group management
    target_group_size=8,                # Target samples per group (rollout.n)
    min_group_size=None,                # Auto-set to target_group_size // 2 + 1 (e.g., 5 for target=8)

    # Optional rejection thresholds
    reward_threshold=None,              # Min reward to keep sample (not commonly used)
    max_response_length=None,           # Max response length (set when rejection_sample=True)
    reject_low_variance_groups=True,    # Filter groups with reward variance < 1e-3

    # Two-gate precision filter
    enable_two_gate_filter=True,
    gate1_bias_epsilon=0.01,           # Systematic bias threshold (1% tolerance)
    gate2_instability_threshold=-15.0,  # Numerical instability threshold
)

# Create filter instance
filter = PPOBatchFilter(config)

# Apply filtering
batch_data = {
    'rewards': rewards_tensor,           # Already summed per sample (1D tensor)
    'response_lengths': lengths_tensor,
    'response_mask': mask_tensor,
    'old_log_probs': fsdp_log_probs,      # Optional: for Gate 1
    'rollout_log_probs': vllm_log_probs,  # Optional: for Gate 1
    'top_log_probs': max_log_probs,       # Optional: for Gate 2
}
uids = ['prompt_001'] * 12 + ['prompt_002'] * 12  # Group identifiers

# Get filtered data
filtered_batch, metrics = filter.filter_batch(batch_data, uids)

# Or get selected indices
selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)
```

## Features

### 1. Unified API
- Single entry point: `PPOBatchFilter`
- Automatic filter ordering
- Clean output with metrics

### 2. Two-Gate Precision Filter (Optional)
- **Gate 1**: Detects systematic bias (FP32/BFloat16 mismatch)
- **Gate 2**: Catches numerical instability
- Only active with oversampling

### 3. Filtering Pipeline

1. **Two-Gate Filter** (with oversampling only)
2. **Group-Level Filtering**:
   - Low variance groups (std < 1e-3, filtered by default)
   - All samples over `max_response_length`
   - Insufficient short samples (`remove_clip=True`)
3. **Individual Filtering**: Reward threshold only (NO length)
4. **Sample Selection**: Smart strategies within groups
5. **Output**: Returns incomplete groups intentionally (maximizes utilization)

### 4. Oversampling Support
- Prompt-level: `prompt_oversampling_factor`
- Sample-level: `sample_oversampling_factor`
- Enables aggressive filtering without data loss

### 5. Selection Strategies

- **`uniform`**: Random selection, no bias
- **`efficiency`**: Smart selection with proportional allocation
  - Maintains model's natural positive/negative ratio
  - Positive: by reward/length ratio
  - Negative: by shortest length
- **`efficiency_stochastic`** (default): Probabilistic efficiency for exploration

### 6. Group Management
- Groups = prompts (identified by UIDs)
- Min size: `min_group_size` (auto-set to `target_group_size // 2 + 1`, ensures <50% padding overhead)
- Target size: `target_group_size` (default: 16, typically set to `rollout.n`)
- Prioritizes complete groups, but returns incomplete groups intentionally
- Ray trainer pads incomplete groups with two-stage padding approach

## Configuration

```yaml
batch_filter:
  # Core settings
  sample_selection_strategy: efficiency_stochastic  # Default
  target_group_size: 8
  min_group_size: null  # Auto-set to 5 (> 8//2)

  # Optional filters
  max_response_length: 512   # Group-level only
  reward_threshold: null      # Individual-level

  # Oversampling
  prompt_oversampling_factor: 2.0
  sample_oversampling_factor: 1.5

  # Two-gate filter (optional)
  enable_two_gate_filter: true
  gate1_bias_epsilon: 0.01
  gate2_instability_threshold: -15.0
```

## Key Metrics

- **Batch**: `selection_rate`, `complete_groups_selected`
- **Rejection**: `reward_rejection_rate`, group-level length rejection
- **Two-Gate**: `gate1/gate2_rejection_rate`, `acceptance_rate`
- **Quality**: PPL drift, extreme token rates

## Testing

### Run All Tests

```bash
# Run comprehensive test suite (48 tests, 100% coverage)
python verl_patch/trainer/code/filters/tests/test_comprehensive_coverage.py

# Run all filter tests
python verl_patch/trainer/code/filters/tests/run_all_tests.py

# Individual test suites
python verl_patch/trainer/code/filters/tests/test_unified_filter.py
python verl_patch/trainer/code/filters/tests/test_two_gate_filter.py
python verl_patch/trainer/code/filters/tests/test_filter_integration.py
```

### Test Coverage

The comprehensive test suite covers:
- Edge cases for batch sizes (empty, single, odd)
- All selection strategies with corner cases
- Group management edge cases
- Precision filter edge cases
- Oversampling factor variations
- Reward/length extremes
- Error handling and invalid inputs
- Large batch performance (10,000+ samples)
- Deterministic behavior verification
- Memory efficiency checks

## Performance

- **Computational overhead**: <1% (efficient tensor operations)
- **Memory overhead**: Minimal (only stores metrics)
- **Large batch handling**: <10s for 10,000 samples
- **Effective batch size**: Reduced by rejection rate (typically 5-15%)
- **Recommendation**: Start with default settings, adjust based on metrics

## Code Quality

The implementation features:
- **100% test coverage** with 48 comprehensive tests
- **Complete type hints** for all methods and parameters
- **Comprehensive docstrings** with usage examples
- **Robust error handling** with input validation
- **Memory-efficient operations** with in-place updates
- **Performance optimizations** for large batches
- **Modular design** for easy extension

## Migration Note

The legacy aliases `OversamplingFilter` and `OversamplingConfig` have been removed. Use `PPOBatchFilter` and `PPOFilterConfig` directly.

## Theory and References

Based on research in rejection sampling for precision mismatch:
- BFloat16 precision in vLLM causes systematic bias
- Two-gate approach catches both bias and instability
- Dual-level oversampling maintains batch quality
- Smart selection strategies combat length inflation
- See `/verl_patch/docs/TWO_GATE_REJECTION_SAMPLING.md` for details

## Architecture

```
PPOBatchFilter (Public API)
├── Stage 0: Two-Gate Precision Filter (with oversampling only)
│   ├── TwoGateRejectionFilter (internal)
│   ├── Gate 1: Systematic bias check
│   └── Gate 2: Numerical instability check
├── Stage 1: Group-Level Filtering
│   ├── Low variance group detection (std < 1e-3)
│   ├── All-over-length group filtering
│   └── min_rollout_n checking (when remove_clip=True)
├── Stage 2: Individual Sample Filtering
│   └── Reward threshold only (NO length filtering)
├── Stage 3: Group Selection & Sample Selection Within Groups
│   ├── Group validity checking (min_group_size)
│   ├── Random group selection (target_num_groups)
│   ├── Sample selection strategies (with oversampling only)
│   └── Returns incomplete groups intentionally (maximizes utilization)
└── Stage 4: Metrics compilation

ray_trainer.py
└── Uses only PPOBatchFilter and PPOFilterConfig
```

## Contributing

When adding new filters:
1. Implement internally in the filters module
2. Integrate into `PPOBatchFilter.filter_batch()` pipeline
3. Add configuration to `PPOFilterConfig`
4. Update metrics in Stage 4
5. Add comprehensive tests
6. Update this documentation

The goal is to keep the external API simple while supporting complex filtering internally.
