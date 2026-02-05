"""
Helper for configuring vLLM with correct logprobs settings for importance sampling.

Configured for vLLM 0.10.2+ where processed_logprobs includes ALL transformations
(penalties, temperature, top-k, top-p) and always returns the correct sampling distribution.
This allows using any temperature/top_p/top_k values with V1 sampler.
"""

import os


def get_vllm_config_kwargs(config):
    """
    Get vLLM configuration kwargs for V1 sampler with processed_logprobs.

    V0 is deprecated in vLLM 0.10+, so this function enforces V1 usage.

    Args:
        config: The rollout configuration object

    Returns:
        dict: Kwargs to pass to LLM initialization
    """
    kwargs = {}

    # Check if V1 sampler is enabled
    use_v1 = os.getenv('VLLM_USE_V1', '0') == '1'

    if use_v1:
        # V1 sampler with processed_logprobs returns the actual sampling distribution
        # In vLLM 0.10.2+, processed_logprobs includes ALL transformations (penalties, temperature, top-k/top-p)
        # This means we can use any temperature/top_p values and still get correct importance sampling
        kwargs['logprobs_mode'] = 'processed_logprobs'

        print("✅ Using V1 sampler with processed_logprobs mode (vLLM 0.10.2+)")
        print("✅ Supports ANY temperature/top_p/top_k values with correct importance sampling")
        print("✅ processed_logprobs includes all transformations: penalties, temperature, top-k/top-p")

        # Log the sampling parameters being used
        temperature = config.get('temperature', 1.0)
        top_p = config.get('top_p', 1.0)
        top_k = config.get('top_k', -1)
        min_p = config.get('min_p', 0.0)

        print(f"📊 Sampling parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}")
    else:
        # V0 sampler is deprecated and being removed in vLLM 0.10+
        print("⚠️  WARNING: V0 sampler is deprecated and being removed in vLLM 0.10+")
        print("⚠️  Please set VLLM_USE_V1=1 to use the V1 engine")
        raise ValueError("V0 engine is not supported in vLLM 0.10+. " "Please set VLLM_USE_V1=1 to use the V1 engine.")

    return kwargs


def verify_importance_sampling_safety(config):
    """
    Verify that the current configuration is safe for importance sampling.

    With vLLM 0.10.2+, all configurations are safe with V1 sampler + processed_logprobs.

    Args:
        config: The rollout configuration object
    """
    use_v1 = os.getenv('VLLM_USE_V1', '0') == '1'

    if not use_v1:
        # V0 is deprecated in vLLM 0.10+
        print("⚠️  WARNING: V0 sampler is deprecated and being removed in vLLM 0.10+")
        raise ValueError("V0 engine is not supported in vLLM 0.10+. " "Please set VLLM_USE_V1=1 to use the V1 engine.")

    # With vLLM 0.10.2+, V1 sampler with processed_logprobs is always safe
    print("✅ Using V1 sampler with processed_logprobs (vLLM 0.10.2+)")
    print("✅ Configuration is safe for importance sampling with any temperature/top_p/top_k values")

    # Log current configuration for transparency
    temperature = config.get('temperature', 1.0)
    top_p = config.get('top_p', 1.0)
    top_k = config.get('top_k', -1)
    min_p = config.get('min_p', 0.0)

    print(f"📊 Current settings: temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}")

    # Optional: Add informational notes about special cases
    if min_p > 0:
        print(f"ℹ️  Note: min_p={min_p} is active (filters tokens below {min_p*100:.1f}% of top token's probability)")

    if temperature == 0:
        print("ℹ️  Note: temperature=0 means greedy sampling (deterministic)")
    elif temperature < 0.5:
        print(f"ℹ️  Note: Low temperature ({temperature}) makes sampling more focused")
    elif temperature > 1.5:
        print(f"ℹ️  Note: High temperature ({temperature}) makes sampling more diverse")
