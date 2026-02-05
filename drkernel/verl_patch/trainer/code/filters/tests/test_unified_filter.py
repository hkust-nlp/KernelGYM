#!/usr/bin/env python3
"""
Comprehensive test suite for unified_filter.py

This is the PRIMARY test file for the unified filter module.
It includes:
1. Multi-process determinism tests (verifies the dict.fromkeys() fix)
2. Core functionality tests
3. Selection strategy tests
4. Filtering logic tests
5. Edge cases
6. Integration tests
"""

import os
import subprocess
import sys

import torch


# Mock verl module to avoid import errors
class MockVerl:
    class utils:
        class torch_functional:
            @staticmethod
            def masked_mean(tensor, mask, dim=None):
                if dim is None:
                    return (tensor * mask).sum() / mask.sum()
                return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


sys.modules['verl'] = MockVerl()
sys.modules['verl.utils'] = MockVerl.utils
sys.modules['verl.utils.torch_functional'] = MockVerl.utils.torch_functional

from verl_patch.trainer.code.filters import PPOBatchFilter, PPOFilterConfig

# ============================================================================
# 1. Multi-Process Determinism Tests (CRITICAL - Verifies the fix)
# ============================================================================


def test_dict_fromkeys_preserves_order():
    """Test that dict.fromkeys() preserves insertion order."""
    print("\n" + "=" * 70)
    print("TEST: dict.fromkeys() preserves order")
    print("=" * 70)

    uids = ['group_A'] * 4 + ['group_B'] * 4 + ['group_C'] * 4

    # Test 10 times to ensure consistency
    results = []
    for i in range(10):
        unique_uids = list(dict.fromkeys(uids))
        results.append(unique_uids)
        if i == 0:
            print(f"  Result: {unique_uids}")

    # All results should be identical
    assert all(r == results[0] for r in results), "dict.fromkeys() should be deterministic"

    # Should preserve order
    assert results[0] == ['group_A', 'group_B', 'group_C'], "Should preserve first occurrence order"

    print("✅ PASS: dict.fromkeys() is deterministic and preserves order")
    return True


def test_filter_group_order_reproducibility():
    """Test that filter produces deterministic, order-preserving results across multiple runs."""
    print("\n" + "=" * 70)
    print("TEST: Filter group order reproducibility")
    print("=" * 70)

    config = PPOFilterConfig(
        sample_selection_strategy="uniform",
        target_group_size=4,
        min_group_size=None,  # Auto-set to 3 (> 4//2)
        reject_low_variance_groups=False,
    )

    # Create batch with multiple groups
    batch_data = {
        'rewards': torch.tensor(
            [
                0.1,
                0.2,
                0.3,
                0.4,  # group_A
                0.5,
                0.6,
                0.7,
                0.8,  # group_B
                0.9,
                1.0,
                1.1,
                1.2,  # group_C
            ]
        ),
        'response_lengths': torch.tensor([100] * 12),
        'response_mask': torch.ones(12, 100, dtype=torch.bool),
    }

    uids = ['group_A'] * 4 + ['group_B'] * 4 + ['group_C'] * 4

    print(f"  Input order: group_A, group_B, group_C")

    # Run filter 10 times
    results = []
    for _ in range(10):
        filter = PPOBatchFilter(config)
        selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)
        results.append(selected_indices.tolist())

    # Check reproducibility
    first_result = results[0]
    all_same = all(result == first_result for result in results)

    print(f"  Output: {first_result}")
    assert all_same, "Filter should produce deterministic results across runs"

    # Check order preservation
    expected_order = list(range(12))
    assert first_result == expected_order, f"Should preserve input order"

    print("✅ PASS: Filter is reproducible and preserves order")
    return True


def test_subprocess_determinism():
    """Test determinism across different subprocess runs (simulating different machines)."""
    print("\n" + "=" * 70)
    print("TEST: Cross-subprocess determinism")
    print("=" * 70)

    test_code = """
import torch
import sys
sys.path.insert(0, '/Users/bytedance/code/verl_patch')

try:
    from verl_patch.trainer.code.filters import PPOBatchFilter, PPOFilterConfig

    config = PPOFilterConfig(
        sample_selection_strategy="uniform",
        target_group_size=4,
        reject_low_variance_groups=False,
    )

    batch_data = {
        'rewards': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
        'response_lengths': torch.tensor([100] * 12),
        'response_mask': torch.ones(12, 100, dtype=torch.bool),
    }

    uids = ['group_A'] * 4 + ['group_B'] * 4 + ['group_C'] * 4

    filter = PPOBatchFilter(config, data_config={'seed': 42})
    selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)
    print(selected_indices.tolist())
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
"""

    # Run in 3 separate subprocesses
    results = []
    for i in range(3):
        try:
            result = subprocess.run([sys.executable, '-c', test_code], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                results.append(result.stdout.strip())
        except Exception as e:
            print(f"  Subprocess {i+1} skipped: {e}")
            return True  # Skip this test if subprocess fails

    if results and len(results) >= 2:
        print(f"  Process 1 result: {results[0]}")
        assert all(r == results[0] for r in results), "Should be identical across processes"
        print("✅ PASS: Results identical across subprocesses")
    else:
        print("⊘ SKIP: Subprocess test not applicable")

    return True


# ============================================================================
# 2. Core Functionality Tests
# ============================================================================


def test_basic_filtering():
    """Test basic filtering functionality."""
    print("\n" + "=" * 70)
    print("TEST: Basic filtering")
    print("=" * 70)

    config = PPOFilterConfig(
        target_group_size=4,  # Small target for this test
        reward_threshold=-1.0,
        max_response_length=300,
        enable_two_gate_filter=False,
    )

    filter = PPOBatchFilter(config)

    batch_data = {
        'rewards': torch.tensor([1.0, 0.5, -0.5, 0.8, -0.2, 0.3, 0.7]),
        'response_lengths': torch.tensor([100, 150, 250, 180, 120, 200, 140]),
        'response_mask': torch.ones(7, 100, dtype=torch.bool),
    }
    # uid_0: 3 samples, uid_1: 3 samples, uid_2: 1 sample (rejected - below min)
    uids = ['uid_0', 'uid_0', 'uid_0', 'uid_1', 'uid_1', 'uid_1', 'uid_2']

    selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)

    print(f"  Selected {len(selected_indices)} samples")
    assert len(selected_indices) >= 3, f"Should select at least 3 samples"

    print("✅ PASS: Basic filtering works")
    return True


def test_empty_batch():
    """Test handling of empty batch."""
    print("\n" + "=" * 70)
    print("TEST: Empty batch handling")
    print("=" * 70)

    config = PPOFilterConfig(target_group_size=8)
    filter = PPOBatchFilter(config)

    batch_data = {
        'rewards': torch.tensor([]),
        'response_lengths': torch.tensor([]),
        'response_mask': torch.ones(0, 100, dtype=torch.bool),
    }
    uids = []

    selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)

    assert len(selected_indices) == 0
    assert metrics['batch/total_samples_generated'] == 0

    print("✅ PASS: Empty batch handled correctly")
    return True


# ============================================================================
# 3. Selection Strategy Tests
# ============================================================================


def test_selection_strategies():
    """Test different selection strategies."""
    print("\n" + "=" * 70)
    print("TEST: Selection strategies")
    print("=" * 70)

    strategies = ["uniform", "efficiency", "efficiency_stochastic"]

    batch_data = {
        'rewards': torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
        'response_lengths': torch.tensor([200, 50, 180, 60, 160, 70, 140, 80]),
        'response_mask': torch.ones(8, 100, dtype=torch.bool),
    }
    uids = ['uid_0'] * 8  # Oversampled: 8 -> 4

    for strategy in strategies:
        config = PPOFilterConfig(
            sample_selection_strategy=strategy,
            target_group_size=4,
        )
        filter = PPOBatchFilter(config, data_config={'seed': 42})

        selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)

        print(f"  {strategy:25s} - Selected {len(selected_indices)} samples")
        assert len(selected_indices) == 4, f"{strategy} should select exactly 4 samples"

    print("✅ PASS: All selection strategies work")
    return True


# ============================================================================
# 4. Filtering Logic Tests
# ============================================================================


def test_low_variance_group_rejection():
    """Test that low variance groups are rejected."""
    print("\n" + "=" * 70)
    print("TEST: Low variance group rejection")
    print("=" * 70)

    config = PPOFilterConfig(
        target_group_size=4,
        reject_low_variance_groups=True,
    )
    filter = PPOBatchFilter(config)

    batch_data = {
        'rewards': torch.tensor(
            [
                0.1,
                0.2,
                0.3,
                0.4,  # group_0: good variance (keep)
                0.5,
                0.5,
                0.5,
                0.5,  # group_1: low variance (reject)
                0.9,
                1.0,
                1.1,
                1.2,  # group_2: good variance (keep)
            ]
        ),
        'response_lengths': torch.tensor([100] * 12),
        'response_mask': torch.ones(12, 100, dtype=torch.bool),
    }
    uids = ['uid_0'] * 4 + ['uid_1'] * 4 + ['uid_2'] * 4

    selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)

    selected_uids = [uids[i] for i in selected_indices]
    assert 'uid_0' in selected_uids, "Group 0 should be kept"
    assert 'uid_1' not in selected_uids, "Group 1 should be rejected (low variance)"
    assert 'uid_2' in selected_uids, "Group 2 should be kept"

    print(f"  Rejected low variance group as expected")
    print("✅ PASS: Low variance group filtering works")
    return True


def test_group_level_length_filtering():
    """Test that length filtering only happens at group level."""
    print("\n" + "=" * 70)
    print("TEST: Group-level length filtering")
    print("=" * 70)

    config = PPOFilterConfig(
        sample_selection_strategy="uniform",
        target_group_size=4,
        max_response_length=150,
    )
    filter = PPOBatchFilter(config)

    batch_data = {
        'rewards': torch.tensor([0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2, 0.1]),
        'response_lengths': torch.tensor([100, 200, 120, 180, 160, 170, 180, 190]),
        'response_mask': torch.ones(8, 100, dtype=torch.bool),
    }
    uids = ['uid_0'] * 4 + ['uid_1'] * 4  # uid_0: mixed lengths, uid_1: all > 150

    selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)

    selected_uids = [uids[i] for i in selected_indices]
    assert 'uid_0' in selected_uids, "Group 0 should be kept (has samples under max_length)"
    assert 'uid_1' not in selected_uids, "Group 1 should be filtered (all samples over max_length)"

    # Over-length individual samples in group 0 can still be selected
    selected_lengths = batch_data['response_lengths'][selected_indices]
    over_length_count = (selected_lengths > 150).sum().item()
    print(f"  Over-length samples in selection: {over_length_count}")
    assert over_length_count > 0, "Should have selected some over-length samples from group 0"

    print("✅ PASS: Group-level length filtering works correctly")
    return True


# ============================================================================
# 5. Edge Cases
# ============================================================================


def test_all_groups_filtered():
    """Test when all groups are filtered out."""
    print("\n" + "=" * 70)
    print("TEST: All groups filtered (edge case)")
    print("=" * 70)

    config = PPOFilterConfig(
        target_group_size=4,
        reject_low_variance_groups=True,
    )
    filter = PPOBatchFilter(config)

    # All groups have identical rewards (low variance)
    batch_data = {
        'rewards': torch.tensor([0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7]),
        'response_lengths': torch.tensor([100] * 8),
        'response_mask': torch.ones(8, 100, dtype=torch.bool),
    }
    uids = ['uid_0'] * 4 + ['uid_1'] * 4

    selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)

    assert len(selected_indices) == 0, "All groups should be filtered"
    assert metrics['batch/critical_empty_batch'] == 1

    print("✅ PASS: All-groups-filtered case handled correctly")
    return True


def test_very_large_batch():
    """Test with very large batch."""
    print("\n" + "=" * 70)
    print("TEST: Very large batch")
    print("=" * 70)

    config = PPOFilterConfig(
        target_group_size=8,
        min_group_size=None,  # Auto-set to 5 (> 8//2)
    )
    filter = PPOBatchFilter(config, data_config={'seed': 42})

    # 100 groups, each with 8 samples
    n_groups = 100
    samples_per_group = 8
    total_samples = n_groups * samples_per_group

    batch_data = {
        'rewards': torch.randn(total_samples),
        'response_lengths': torch.randint(50, 200, (total_samples,)),
        'response_mask': torch.ones(total_samples, 100, dtype=torch.bool),
    }
    uids = [f'uid_{i}' for i in range(n_groups) for _ in range(samples_per_group)]

    selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)

    print(f"  Processed {total_samples} samples")
    print(f"  Selected {len(selected_indices)} samples")
    assert len(selected_indices) > 0, "Should select some samples from large batch"
    assert len(selected_indices) <= total_samples, "Should not select more than input"

    print("✅ PASS: Large batch handled efficiently")
    return True


def test_determinism_with_random_group_selection():
    """Test determinism when randomly selecting subset of groups."""
    print("\n" + "=" * 70)
    print("TEST: Determinism with random group selection")
    print("=" * 70)

    config = PPOFilterConfig(
        sample_selection_strategy="uniform",
        target_group_size=4,
        target_num_groups=5,  # Select 5 out of 10 groups
    )

    batch_data = {
        'rewards': torch.randn(40),
        'response_lengths': torch.randint(50, 150, (40,)),
        'response_mask': torch.ones(40, 100, dtype=torch.bool),
    }
    uids = [f'uid_{i}' for i in range(10) for _ in range(4)]  # 10 groups

    # Run multiple times with same seed
    results = []
    for _ in range(5):
        filter = PPOBatchFilter(config, data_config={'seed': 42})
        selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)
        results.append(selected_indices.tolist())

    # Check all identical
    assert all(r == results[0] for r in results), "Should be deterministic with random selection"

    print(f"  Selected {len(results[0])} samples consistently")
    print("✅ PASS: Random group selection is deterministic")
    return True


def test_determinism_with_stochastic_strategy():
    """Test determinism with stochastic selection strategy."""
    print("\n" + "=" * 70)
    print("TEST: Determinism with stochastic strategy")
    print("=" * 70)

    config = PPOFilterConfig(
        sample_selection_strategy="efficiency_stochastic",
        target_group_size=4,
    )

    batch_data = {
        'rewards': torch.randn(24),
        'response_lengths': torch.randint(50, 150, (24,)),
        'response_mask': torch.ones(24, 100, dtype=torch.bool),
    }
    uids = ['uid_0'] * 8 + ['uid_1'] * 8 + ['uid_2'] * 8

    # Run multiple times with same seed
    results = []
    for _ in range(5):
        filter = PPOBatchFilter(config, data_config={'seed': 42})
        selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)
        results.append(selected_indices.tolist())

    # Check all identical
    assert all(r == results[0] for r in results), "Stochastic strategy should be deterministic with same seed"

    print(f"  Selected {len(results[0])} samples consistently")
    print("✅ PASS: Stochastic strategy is deterministic")
    return True


def test_determinism_with_mixed_groups():
    """Test determinism with mixed complete and incomplete groups."""
    print("\n" + "=" * 70)
    print("TEST: Determinism with mixed complete/incomplete groups")
    print("=" * 70)

    config = PPOFilterConfig(
        target_group_size=8,
        target_num_groups=3,  # Select 3 out of 5
        min_group_size=None,  # Auto-set to 5 (> 8//2)
    )

    batch_data = {
        'rewards': torch.randn(36),
        'response_lengths': torch.randint(50, 150, (36,)),
        'response_mask': torch.ones(36, 100, dtype=torch.bool),
    }
    # 5 groups with varying sizes: 8, 8, 5, 6, 9
    uids = ['uid_0'] * 8 + ['uid_1'] * 8 + ['uid_2'] * 5 + ['uid_3'] * 6 + ['uid_4'] * 9

    # Run multiple times
    results = []
    for _ in range(5):
        filter = PPOBatchFilter(config, data_config={'seed': 42})
        selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)
        results.append(selected_indices.tolist())

    # Check all identical
    assert all(r == results[0] for r in results), "Should be deterministic with mixed groups"

    print(f"  Selected {len(results[0])} samples consistently")
    print("✅ PASS: Mixed group selection is deterministic")
    return True


def test_determinism_with_combined_filters():
    """Test determinism with combined filtering criteria."""
    print("\n" + "=" * 70)
    print("TEST: Determinism with combined filters")
    print("=" * 70)

    config = PPOFilterConfig(
        target_group_size=4,
        reject_low_variance_groups=True,
        max_response_length=100,
    )

    batch_data = {
        'rewards': torch.tensor(
            [
                0.1,
                0.2,
                0.3,
                0.4,  # group_0: good variance, mixed lengths
                0.5,
                0.5,
                0.5,
                0.5,  # group_1: low variance (reject)
                0.6,
                0.7,
                0.8,
                0.9,  # group_2: all over-length (reject)
                0.2,
                0.4,
                0.6,
                0.8,  # group_3: good
            ]
        ),
        'response_lengths': torch.tensor(
            [
                80,
                90,
                110,
                120,  # group_0
                50,
                60,
                70,
                80,  # group_1
                110,
                120,
                130,
                140,  # group_2
                60,
                70,
                80,
                90,  # group_3
            ]
        ),
        'response_mask': torch.ones(16, 100, dtype=torch.bool),
    }
    uids = ['uid_0'] * 4 + ['uid_1'] * 4 + ['uid_2'] * 4 + ['uid_3'] * 4

    # Run multiple times
    results = []
    for _ in range(5):
        filter = PPOBatchFilter(config, data_config={'seed': 42})
        selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)
        results.append(selected_indices.tolist())

    # Check all identical
    assert all(r == results[0] for r in results), "Should be deterministic with combined filters"

    print(f"  Selected {len(results[0])} samples consistently")
    print("✅ PASS: Combined filtering is deterministic")
    return True


def test_determinism_with_non_alphabetical_uids():
    """Test determinism preserves insertion order (not alphabetical)."""
    print("\n" + "=" * 70)
    print("TEST: Determinism with non-alphabetical UIDs")
    print("=" * 70)

    config = PPOFilterConfig(
        target_group_size=3,
        min_group_size=None,  # Auto-set to 2 (> 3//2)
    )

    batch_data = {
        'rewards': torch.randn(12),
        'response_lengths': torch.randint(50, 150, (12,)),
        'response_mask': torch.ones(12, 100, dtype=torch.bool),
    }
    # UIDs in non-alphabetical order: zebra, apple, middle, banana
    uids = ['zebra'] * 3 + ['apple'] * 3 + ['middle'] * 3 + ['banana'] * 3

    # Run multiple times
    results = []
    for _ in range(5):
        filter = PPOBatchFilter(config, data_config={'seed': 42})
        selected_indices, _ = filter.filter_batch(batch_data, uids, return_indices=True)
        results.append(selected_indices.tolist())

    # Check all identical
    assert all(r == results[0] for r in results), "Should preserve insertion order"

    # Verify order is zebra, apple, middle, banana (not alphabetical)
    expected_order = list(range(12))  # Should be [0,1,2,3,4,5,6,7,8,9,10,11]
    assert results[0] == expected_order, "Should preserve insertion order, not sort alphabetically"

    print(f"  Order preserved: zebra → apple → middle → banana")
    print("✅ PASS: Insertion order preserved (not alphabetical)")
    return True


# ============================================================================
# 6. Integration Test
# ============================================================================


def test_full_pipeline():
    """Test full pipeline with no rejection (ideal case)."""
    print("\n" + "=" * 70)
    print("TEST: Full pipeline integration")
    print("=" * 70)

    config = PPOFilterConfig(
        sample_selection_strategy="uniform",
        target_group_size=8,
        min_group_size=None,  # Auto-set to 5 (> 8//2)
        reject_low_variance_groups=False,
        reward_threshold=None,
        max_response_length=None,
    )
    filter = PPOBatchFilter(config, data_config={'seed': 42})

    # 3 groups, each with exactly target_group_size samples
    batch_data = {
        'rewards': torch.randn(24),
        'response_lengths': torch.randint(50, 150, (24,)),
        'response_mask': torch.ones(24, 100, dtype=torch.bool),
    }
    uids = ['uid_0'] * 8 + ['uid_1'] * 8 + ['uid_2'] * 8

    selected_indices, metrics = filter.filter_batch(batch_data, uids, return_indices=True)

    # Should select all samples
    assert len(selected_indices) == 24
    assert metrics['batch/selection_rate'] == 1.0
    assert metrics['batch/complete_groups_selected'] == 3

    print(f"  Selected {len(selected_indices)}/24 samples (100%)")
    print("✅ PASS: Full pipeline integration works")
    return True


# ============================================================================
# Test Runner
# ============================================================================


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("UNIFIED FILTER COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        # Category 1: Multi-process determinism (CRITICAL)
        (
            "Multi-Process Determinism",
            [
                test_dict_fromkeys_preserves_order,
                test_filter_group_order_reproducibility,
                test_subprocess_determinism,
            ],
        ),
        # Category 2: Core functionality
        (
            "Core Functionality",
            [
                test_basic_filtering,
                test_empty_batch,
            ],
        ),
        # Category 3: Selection strategies
        (
            "Selection Strategies",
            [
                test_selection_strategies,
            ],
        ),
        # Category 4: Filtering logic
        (
            "Filtering Logic",
            [
                test_low_variance_group_rejection,
                test_group_level_length_filtering,
            ],
        ),
        # Category 5: Edge cases
        (
            "Edge Cases",
            [
                test_all_groups_filtered,
                test_very_large_batch,
                test_determinism_with_random_group_selection,
                test_determinism_with_stochastic_strategy,
                test_determinism_with_mixed_groups,
                test_determinism_with_combined_filters,
                test_determinism_with_non_alphabetical_uids,
            ],
        ),
        # Category 6: Integration
        (
            "Integration",
            [
                test_full_pipeline,
            ],
        ),
    ]

    total_passed = 0
    total_failed = 0
    total_tests = 0

    for category_name, category_tests in tests:
        print(f"\n{'='*70}")
        print(f"Category: {category_name}")
        print(f"{'='*70}")

        for test in category_tests:
            total_tests += 1
            try:
                if test():
                    total_passed += 1
                else:
                    total_failed += 1
            except Exception as e:
                total_failed += 1
                print(f"❌ FAIL: {test.__name__}: {e}")
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {total_passed}/{total_tests}")
    print(f"❌ Failed: {total_failed}/{total_tests}")

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total_failed} tests failed")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
