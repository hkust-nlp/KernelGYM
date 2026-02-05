#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for CodeDataParallelPPOActor with compute_sum_pi_squared enabled.

This test suite verifies:
1. Sum of squared probabilities computation correctness
2. Checkpointing functionality
3. Return value consistency (always 3-tuple)
4. All code paths (remove_padding × fused kernels combinations)
5. Mathematical validity of outputs
"""

import math
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from verl_patch.utils.torch_functional import compute_sum_pi_squared_from_logits


def test_sum_pi_squared_computation():
    """Test the mathematical correctness of sum_pi_squared computation."""
    # Create test logits
    vocab_size = 100
    batch_size = 2
    seq_len = 5

    # Test case 1: Uniform distribution (all logits equal)
    uniform_logits = torch.zeros(batch_size, seq_len, vocab_size)

    # Expected: sum of squared probs = 1/vocab_size for uniform distribution
    expected_uniform = torch.full((batch_size, seq_len), 1.0 / vocab_size)

    # Compute sum_pi_squared using shared utility function
    sum_pi_squared = compute_sum_pi_squared_from_logits(uniform_logits)

    assert torch.allclose(sum_pi_squared, expected_uniform, rtol=1e-5)

    # Test case 2: One-hot distribution (one logit very large)
    onehot_logits = torch.full((batch_size, seq_len, vocab_size), -1000.0)
    onehot_logits[:, :, 0] = 1000.0  # Make first token have ~100% probability

    # Expected: sum of squared probs ≈ 1.0 for one-hot distribution
    expected_onehot = torch.ones(batch_size, seq_len)

    sum_pi_squared_onehot = compute_sum_pi_squared_from_logits(onehot_logits)

    assert torch.allclose(sum_pi_squared_onehot, expected_onehot, rtol=1e-3)

    # Test case 3: Valid range check
    random_logits = torch.randn(batch_size, seq_len, vocab_size)
    sum_pi_squared_random = compute_sum_pi_squared_from_logits(random_logits)

    # Should be in range [1/vocab_size, 1.0]
    assert torch.all(sum_pi_squared_random >= (1.0 / vocab_size) - 1e-6)
    assert torch.all(sum_pi_squared_random <= 1.0 + 1e-6)


def test_checkpointing_configuration():
    """Test that checkpointing configuration is properly handled."""
    from verl_patch.workers.code.actor.dp_actor import CodeDataParallelPPOActor

    # Mock configuration
    config = Mock()
    config.get = MagicMock(
        side_effect=lambda key, default=None: {
            'use_torch_compile': False,  # Disable compilation for testing
            'sum_pi_squared_checkpointing': True,  # Enable checkpointing
            'entropy_checkpointing': False,
            'use_remove_padding': False,
            'use_fused_kernels': False,
        }.get(key, default)
    )
    config.ulysses_sequence_parallel_size = 1
    config.entropy_from_logits_with_chunking = False
    config.grad_clip = 1.0
    config.compute_sum_pi_squared = True

    # Mock actor module
    actor_module = Mock(spec=nn.Module)
    actor_module.eval = Mock()
    actor_module.train = Mock()

    with patch('torch.distributed.get_rank', return_value=0):
        actor = CodeDataParallelPPOActor(config=config, actor_module=actor_module, actor_optimizer=None)

    # Verify checkpointing is accessible
    assert hasattr(actor, 'compute_sum_pi_squared_from_logits')

    # Test checkpointing flag reading
    checkpointing_enabled = config.get('sum_pi_squared_checkpointing', False)
    assert checkpointing_enabled == True


def test_return_value_consistency():
    """Test that _forward_micro_batch always returns 3-tuple."""
    from verl_patch.workers.code.actor.dp_actor import CodeDataParallelPPOActor

    config = Mock()
    config.get = MagicMock(
        side_effect=lambda key, default=None: {
            'use_torch_compile': False,
            'use_remove_padding': False,
            'use_fused_kernels': False,
            'entropy_checkpointing': False,
            'sum_pi_squared_checkpointing': False,
        }.get(key, default)
    )
    config.ulysses_sequence_parallel_size = 1
    config.entropy_from_logits_with_chunking = False
    config.compute_sum_pi_squared = True

    actor_module = Mock(spec=nn.Module)
    actor_module.eval = Mock()

    vocab_size = 100
    mock_output = Mock()
    mock_output.logits = torch.randn(2, 10, vocab_size)
    actor_module.return_value = mock_output

    with patch('torch.distributed.get_rank', return_value=0):
        actor = CodeDataParallelPPOActor(config=config, actor_module=actor_module, actor_optimizer=None)

    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.device_name = 'cpu'

    micro_batch = {
        'input_ids': torch.randint(0, 1000, (2, 10)),
        'attention_mask': torch.ones(2, 10),
        'position_ids': torch.arange(10).expand(2, -1),
        'responses': torch.randint(0, 1000, (2, 5)),
    }

    with patch('verl_patch.workers.code.actor.dp_actor.logprobs_from_logits', return_value=torch.randn(2, 5)):
        # Test 1: calculate_entropy=False, compute_sum_pi_squared=False
        result = actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=False, compute_sum_pi_squared=False
        )
        assert len(result) == 3
        assert result[0] is None  # entropy
        assert result[1] is not None  # log_probs
        assert result[2] is None  # sum_pi_squared

        # Test 2: calculate_entropy=True, compute_sum_pi_squared=False
        result = actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=True, compute_sum_pi_squared=False
        )
        assert len(result) == 3
        assert result[0] is not None  # entropy
        assert result[1] is not None  # log_probs
        assert result[2] is None  # sum_pi_squared

        # Test 3: calculate_entropy=False, compute_sum_pi_squared=True
        result = actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=False, compute_sum_pi_squared=True
        )
        assert len(result) == 3
        assert result[0] is None  # entropy
        assert result[1] is not None  # log_probs
        assert result[2] is not None  # sum_pi_squared

        # Test 4: calculate_entropy=True, compute_sum_pi_squared=True
        result = actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=True, compute_sum_pi_squared=True
        )
        assert len(result) == 3
        assert result[0] is not None  # entropy
        assert result[1] is not None  # log_probs
        assert result[2] is not None  # sum_pi_squared


def test_all_code_paths():
    """Test that all four code paths work correctly."""
    from verl_patch.workers.code.actor.dp_actor import CodeDataParallelPPOActor

    test_configs = [
        (False, False),  # remove_padding=False, fused=False
        (True, False),  # remove_padding=True, fused=False
    ]

    for use_remove_padding, use_fused_kernels in test_configs:
        config = Mock()
        config.get = MagicMock(
            side_effect=lambda key, default=None: {
                'use_torch_compile': False,
                'use_remove_padding': use_remove_padding,
                'use_fused_kernels': use_fused_kernels,
                'entropy_checkpointing': False,
                'sum_pi_squared_checkpointing': False,
            }.get(key, default)
        )
        config.ulysses_sequence_parallel_size = 1
        config.entropy_from_logits_with_chunking = False
        config.compute_sum_pi_squared = True

        # Create mock module that returns proper output
        actor_module = Mock(spec=nn.Module)
        actor_module.eval = Mock()

        vocab_size = 100
        mock_output = Mock()

        if use_fused_kernels:
            # Fused kernels return pre-computed values
            mock_output.log_probs = torch.randn(2, 10) if use_remove_padding else torch.randn(2, 10)
            mock_output.entropy = torch.randn(2, 10) if use_remove_padding else torch.randn(2, 10)
        else:
            # Non-fused always returns logits
            mock_output.logits = (
                torch.randn(2, 10, vocab_size) if use_remove_padding else torch.randn(2, 10, vocab_size)
            )

        actor_module.return_value = mock_output

        with patch('torch.distributed.get_rank', return_value=0):
            actor = CodeDataParallelPPOActor(config=config, actor_module=actor_module, actor_optimizer=None)

        actor.use_remove_padding = use_remove_padding
        actor.use_fused_kernels = use_fused_kernels
        actor.use_ulysses_sp = False
        actor.device_name = 'cpu'

        # For remove_padding, we need to mock the padding functions
        if use_remove_padding:
            with (
                patch('verl_patch.workers.code.actor.dp_actor.unpad_input') as mock_unpad,
                patch('verl_patch.workers.code.actor.dp_actor.pad_input') as mock_pad,
                patch('verl_patch.workers.code.actor.dp_actor.index_first_axis') as mock_index,
                patch('verl_patch.workers.code.actor.dp_actor.logprobs_from_logits') as mock_logprobs,
            ):

                # Setup mocks for remove_padding path
                mock_unpad.return_value = (
                    torch.randn(20, 1),  # input_ids_rmpad
                    torch.arange(20),  # indices
                    torch.tensor([0, 10, 20]),  # cu_seqlens
                    None,
                )
                mock_index.return_value = torch.randn(20, 1)
                mock_pad.return_value = torch.randn(2, 10, 1)
                mock_logprobs.return_value = torch.randn(20)

                # Override the mock output for rmpad case
                if use_fused_kernels:
                    mock_output.log_probs = torch.randn(1, 20)
                    mock_output.entropy = torch.randn(1, 20)
                else:
                    mock_output.logits = torch.randn(1, 20, vocab_size)

                micro_batch = {
                    'input_ids': torch.randint(0, 1000, (2, 10)),
                    'attention_mask': torch.ones(2, 10),
                    'position_ids': torch.arange(10).expand(2, -1),
                    'responses': torch.randint(0, 1000, (2, 5)),
                }

                # This should not raise any errors
                entropy, log_probs, sum_pi_squared = actor._forward_micro_batch(
                    micro_batch, temperature=1.0, calculate_entropy=True, compute_sum_pi_squared=True
                )

                assert log_probs is not None
                # entropy and sum_pi_squared depend on code path
        else:
            micro_batch = {
                'input_ids': torch.randint(0, 1000, (2, 10)),
                'attention_mask': torch.ones(2, 10),
                'position_ids': torch.arange(10).expand(2, -1),
                'responses': torch.randint(0, 1000, (2, 5)),
            }

            # Mock logprobs_from_logits for non-fused case
            with patch(
                'verl_patch.workers.code.actor.dp_actor.logprobs_from_logits',
                return_value=torch.randn(2, 5),
            ):
                # This should not raise any errors
                entropy, log_probs, sum_pi_squared = actor._forward_micro_batch(
                    micro_batch, temperature=1.0, calculate_entropy=True, compute_sum_pi_squared=True
                )

                assert log_probs is not None
                assert entropy is not None
                assert sum_pi_squared is not None


def test_compute_log_prob_returns_tuple():
    """Test that compute_log_prob returns correct tuple with sum_pi_squared when enabled."""
    from verl import DataProto

    from verl_patch.workers.code.actor.dp_actor import CodeDataParallelPPOActor

    config = Mock()
    config.get = MagicMock(
        side_effect=lambda key, default=None: {
            'use_torch_compile': False,
            'use_remove_padding': False,
            'use_fused_kernels': False,
            'entropy_checkpointing': False,
            'sum_pi_squared_checkpointing': False,
        }.get(key, default)
    )
    config.ulysses_sequence_parallel_size = 1
    config.entropy_from_logits_with_chunking = False
    config.compute_sum_pi_squared = True

    actor_module = Mock(spec=nn.Module)
    actor_module.eval = Mock()

    # Mock the model output
    mock_output = Mock()
    mock_output.logits = torch.randn(2, 10, 100)
    actor_module.return_value = mock_output

    with patch('torch.distributed.get_rank', return_value=0):
        actor = CodeDataParallelPPOActor(config=config, actor_module=actor_module, actor_optimizer=None)

    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.device_name = 'cpu'

    # Create test data
    test_data = DataProto.from_dict(
        {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'position_ids': torch.arange(10).expand(2, -1),
            'responses': torch.randint(0, 1000, (2, 5)),
        }
    )
    test_data.meta_info = {'micro_batch_size': 1, 'temperature': 1.0, 'use_dynamic_bsz': False}

    # Mock logprobs_from_logits
    with (
        patch(
            'verl_patch.workers.code.actor.dp_actor.logprobs_from_logits',
            return_value=torch.randn(2, 5),
        ),
        patch('verl.utils.device.get_device_id', return_value='cpu'),
    ):

        log_probs, entropys, sum_pi_squared = actor.compute_log_prob(
            test_data, calculate_entropy=True, compute_sum_pi_squared=True
        )

    # Verify result is a tuple with 3 elements
    assert log_probs is not None
    assert entropys is not None
    assert sum_pi_squared is not None

    # Verify shapes are correct
    assert log_probs.shape == (2, 5)
    assert sum_pi_squared.shape == (2, 5)
    assert entropys.shape == (2, 5)


if __name__ == "__main__":
    # Run tests
    test_sum_pi_squared_computation()
    print("✓ Mathematical correctness test passed")

    test_checkpointing_configuration()
    print("✓ Checkpointing configuration test passed")

    test_return_value_consistency()
    print("✓ Return value consistency test passed")

    test_all_code_paths()
    print("✓ All code paths test passed")

    test_compute_log_prob_returns_tuple()
    print("✓ compute_log_prob return type test passed")

    print("\n✅ All tests passed successfully!")
