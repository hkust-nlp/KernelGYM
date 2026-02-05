"""
dynamic_solve_rate_dataset.py
Implementation of dynamic solve rate filtered RLHF dataset with test cases

Key Components:
1. SolveRateDynamicRLHFDataset - Dataset with dynamic solve rate tracking
2. RefreshSolveRateSampler - Batch sampler with adaptive filtering based on solve rates
3. Save/Restore functionality for checkpointing
4. Optimized propagation logic
"""

import json
import math
import os
import pickle
from collections import defaultdict
from typing import *

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.utils.data import RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl_patch.utils.dataset.rl_dataset import SolveRateDynamicRLHFDataset, collate_fn
from verl_patch.utils.samplers.prioritized_batch_sampler import PrioritizedBatchSampler


def print_dict(d):
    """Helper function to print dictionaries in readable JSON format"""
    print(json.dumps(d, indent=4, ensure_ascii=False))


class RefreshSolveRateSampler(PrioritizedBatchSampler):
    """
    Adaptive batch sampler with:
    1. Freshness-aware sampling (prioritize less-used samples)
    2. Target distribution matching via rejection sampling
    3. Adaptive solve rate updates for non-sampled items
    4. Save/Restore functionality for checkpointing
    """

    def __init__(
        self,
        dataset: SolveRateDynamicRLHFDataset,
        solverate_high=1.0,
        solverate_low=0.0,
        solverate_mean=0.5,
        solverate_std=0.1,
        freshness_balance=0.2,
        current_step=0,
        drop_few=True,
        propagation_threshold=5,
        propagation_confidence=0.8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.solverate_mean = solverate_mean
        self.solverate_std = solverate_std
        self.solverate_high = solverate_high
        self.solverate_low = solverate_low

        self.freshness_balance = freshness_balance
        self.current_step = current_step
        self.drop_few = drop_few

        # Propagation parameters
        self.propagation_threshold = propagation_threshold
        self.propagation_confidence = propagation_confidence

        # Track usage and solve rate changes
        self.usage_counts = defaultdict(int)
        self.last_update_steps = defaultdict(int)
        self.solve_rate_changes_by_bin = defaultdict(list)

        # Enhanced change tracking for better propagation
        self.global_change_history = []  # Track global trends
        self.bin_change_momentum = defaultdict(float)  # Momentum for each bin

        # set random seed for numpy if not provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Validate configuration at initialization
        raw_target = self.target_batch_size * self.oversampling_factor
        if raw_target < self.world_size:
            raise ValueError(
                f"Configuration Error: target_batch_size * oversampling_factor "
                f"({self.target_batch_size} * {self.oversampling_factor} = {raw_target}) "
                f"must be >= world_size ({self.world_size})\n"
                f"\nFor {self.world_size} GPUs, you need either:\n"
                f"  1. Increase train_batch_size (currently {self.target_batch_size})\n"
                f"  2. Increase oversampling_factor (currently {self.oversampling_factor})\n"
                f"  3. Use fewer nodes/GPUs\n"
                f"\nSuggested minimum oversampling_factor: {self.world_size / self.target_batch_size:.1f}"
            )
        else:
            np.random.seed(42)
        self._read_init_solve_rate()

    def _read_init_solve_rate(self):
        """Initialize solve rates and tracking info"""
        for data in self.dataset:
            data_id = data['prompt_index']
            self.success_rates[data_id] = data['solve_rate']
            self.usage_counts[data_id] = 0
            self.last_update_steps[data_id] = 0

        rates = list(self.success_rates.values())
        print(f"Initial solve rates - Mean: {np.mean(rates):.3f}, " f"Min: {min(rates):.3f}, Max: {max(rates):.3f}")

    def save_state(self, filepath: str) -> None:
        """
        Save the complete state of the sampler to disk

        Args:
            filepath: Path to save the state file
        """
        state = {
            'success_rates': dict(self.success_rates),
            'usage_counts': dict(self.usage_counts),
            'last_update_steps': dict(self.last_update_steps),
            'current_step': self.current_step,
            'solve_rate_changes_by_bin': {k: list(v) for k, v in self.solve_rate_changes_by_bin.items()},
            'global_change_history': list(self.global_change_history),
            'bin_change_momentum': dict(self.bin_change_momentum),
            'solverate_mean': self.solverate_mean,
            'solverate_std': self.solverate_std,
            'freshness_balance': self.freshness_balance,
            'ema_alpha': self.ema_alpha,
            'seed': self.seed,
            'version': '1.0',  # For future compatibility
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Sampler state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """
        Load the complete state of the sampler from disk

        Args:
            filepath: Path to load the state file from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Validate version compatibility
        if state.get('version', '1.0') != '1.0':
            self.logger.warning("Loading state from different version, compatibility not guaranteed")

        # Restore state
        self.success_rates = defaultdict(float, state['success_rates'])
        self.usage_counts = defaultdict(int, state['usage_counts'])
        self.last_update_steps = defaultdict(int, state['last_update_steps'])
        self.current_step = state['current_step']
        self.solve_rate_changes_by_bin = defaultdict(
            list, {k: list(v) for k, v in state['solve_rate_changes_by_bin'].items()}
        )
        self.global_change_history = list(state['global_change_history'])
        self.bin_change_momentum = defaultdict(float, state['bin_change_momentum'])

        # Restore configuration (optional, but useful for consistency)
        self.solverate_mean = state.get('solverate_mean', self.solverate_mean)
        self.solverate_std = state.get('solverate_std', self.solverate_std)
        self.freshness_balance = state.get('freshness_balance', self.freshness_balance)
        self.ema_alpha = state.get('ema_alpha', self.ema_alpha)

        # Restore random seed if available
        if 'seed' in state and state['seed'] is not None:
            np.random.seed(state['seed'])

        rates = list(self.success_rates.values())
        print(f"Sampler state loaded from {filepath}")
        print(f"Loaded solve rates - Mean: {np.mean(rates):.3f}, " f"Min: {min(rates):.3f}, Max: {max(rates):.3f}")

    def __iter__(self) -> iter:
        """
        Enhanced sampling strategy:
        1. Sort by usage count (freshness priority)
        2. Create candidate pool from fresh samples
        3. Use rejection sampling to match target distribution
        4. Update usage counts for sampled items
        """
        all_indices = np.array(list(self.success_rates.keys()))
        original_count = len(all_indices)
        if len(all_indices) > 0:
            # if solverate_high or solverate_low is set, filter indices
            if self.solverate_high < 1.0 or self.solverate_low > 0.0:
                origin_solve_rates = np.array([self.success_rates[idx] for idx in all_indices])
                mask = (origin_solve_rates >= self.solverate_low) & (origin_solve_rates <= self.solverate_high)
                filtered_indices = all_indices[mask]

                # Log filtering results
                print(
                    f"[RefreshSampler] Filtering: {len(filtered_indices)}/{original_count} samples in range "
                    f"[{self.solverate_low}, {self.solverate_high}]"
                )

                # If filtering removes all indices, this is a configuration/data issue
                if len(filtered_indices) == 0:
                    raise ValueError(
                        f"Data/Configuration Error: No samples found in solverate range [{self.solverate_low}, {self.solverate_high}].\n"
                        f"Current solve rate distribution: min={min(origin_solve_rates):.3f}, "
                        f"max={max(origin_solve_rates):.3f}, mean={np.mean(origin_solve_rates):.3f}\n"
                        f"\nPossible solutions:\n"
                        f"  1. Adjust solverate_low/solverate_high to match your data\n"
                        f"  2. Check if solve rates are being updated correctly\n"
                        f"  3. Verify your dataset contains samples in the expected difficulty range"
                    )
                else:
                    all_indices = filtered_indices

        target_size = self.target_batch_size * self.oversampling_factor
        target_size = int(target_size // self.world_size) * self.world_size

        # Check if we have any indices to sample from
        if len(all_indices) == 0:
            print("ERROR: No indices available for sampling after filtering. Cannot proceed.")
            return  # Return from iterator without yielding anything

        # Simple freshness-based sorting
        usage_counts = np.array([self.usage_counts[idx] for idx in all_indices])
        epsilon = 1e-2
        freshness_weights = 1.0 / (usage_counts + epsilon)

        # Simple target distribution filtering
        solve_rates = np.array([self.success_rates[idx] for idx in all_indices])
        target_diff = np.abs(solve_rates - self.solverate_mean)
        solverate_weights = np.exp(-0.5 * (target_diff / self.solverate_std) ** 2)

        # Balance parameter: 0.0 = pure solve rate, 1.0 = pure freshness
        freshness_balance = self.freshness_balance

        # Normalize weights to [0, 1] range for proper combination
        freshness_weights_norm = freshness_weights / np.max(freshness_weights)
        solverate_weights_norm = solverate_weights / np.max(solverate_weights)

        # Combine weights
        combined_weights = freshness_balance * freshness_weights_norm + (1 - freshness_balance) * solverate_weights_norm

        # Normalize combined weights for sampling
        if np.sum(combined_weights) > 0:
            combined_weights = combined_weights / np.sum(combined_weights)
        else:
            combined_weights = np.ones(len(all_indices)) / len(all_indices)

        # Sample without replacement using weights
        accepted_indices = np.random.choice(
            all_indices, size=min(target_size, len(all_indices)), replace=False, p=combined_weights
        )

        if len(accepted_indices) > 0:
            accepted_solve_rates = [self.success_rates[idx] for idx in accepted_indices]
            accepted_usage_counts = [self.usage_counts[idx] for idx in accepted_indices]

            # Print distribution diagnostics
            solve_rate_hist = np.histogram(accepted_solve_rates, bins=np.arange(0, 1.1, 0.1))
            print(f"[Batch] solve rate distribution: {solve_rate_hist}")
            usage_count_hist = np.histogram(accepted_usage_counts, bins=np.arange(0, max(accepted_usage_counts) + 1))
            print(f"[Batch] usage count distribution: {usage_count_hist}")

        # Update usage and yield
        for idx in accepted_indices:
            self.usage_counts[idx] += 1

        batch = list(accepted_indices)
        if len(batch) >= target_size:
            yield batch
        elif len(batch) > 0 and not self.drop_few:
            yield batch

    def update_success_rates(self, new_rates: Dict[int, float]):
        """
        Enhanced update with optimized adaptive propagation to non-sampled items
        """
        if not new_rates:
            return

        # Track solve rate changes for adaptive updates
        step_changes = []  # For global trend tracking

        for problem_id, new_rate in new_rates.items():
            new_rate = min(1.0, max(0.0, new_rate))  # Clamp to [0,1]
            old_rate = self.success_rates[problem_id]

            # Calculate change rate
            step_diff = max(1, self.current_step - self.last_update_steps[problem_id])
            change_rate = (new_rate - old_rate) / step_diff
            step_changes.append(change_rate)

            # Store change by solve rate bin for propagation
            rate_bin = round(old_rate, 1)  # 0.1 precision bins
            self.solve_rate_changes_by_bin[rate_bin].append(change_rate)

            # Update momentum for this bin
            momentum_decay = 0.9
            self.bin_change_momentum[rate_bin] = (
                momentum_decay * self.bin_change_momentum[rate_bin] + (1 - momentum_decay) * change_rate
            )

            # Update the sample
            self.success_rates[problem_id] = (1 - self.ema_alpha) * old_rate + self.ema_alpha * new_rate
            self.last_update_steps[problem_id] = self.current_step

        # Track global change trend
        if step_changes:
            global_change = np.mean(step_changes)
            self.global_change_history.append(global_change)
            # Keep only recent history
            if len(self.global_change_history) > 100:
                self.global_change_history = self.global_change_history[-100:]

        # Optimized propagation for non-sampled items
        self._propagate_solve_rate_changes()

        # Clear solve_rate_changes_by_bin to prevent memory leak
        self.solve_rate_changes_by_bin.clear()

        # Logging
        if self.success_rates:
            rates = list(self.success_rates.values())
            print(f"Updated solve rates - Mean: {np.mean(rates):.3f}, " f"Min: {min(rates):.3f}, Max: {max(rates):.3f}")

    def _propagate_solve_rate_changes(self):
        """
        Optimized propagation of solve rate changes to non-sampled items
        Uses vectorized operations and smarter interpolation
        """
        if not self.solve_rate_changes_by_bin:
            return

        # 1. Calculate weighted average changes per bin with momentum
        bin_changes = {}
        bin_weights = {}

        for rate_bin, changes in self.solve_rate_changes_by_bin.items():
            if changes:
                # Use recent changes with exponential weighting
                weights = np.exp(np.linspace(-1, 0, len(changes)))
                weighted_avg = np.average(changes, weights=weights)

                # Combine with momentum
                momentum_weight = 0.3
                final_change = (1 - momentum_weight) * weighted_avg + momentum_weight * self.bin_change_momentum[
                    rate_bin
                ]

                bin_changes[rate_bin] = final_change
                bin_weights[rate_bin] = len(changes)  # Confidence based on sample count

        if not bin_changes:
            return

        # 2. Create interpolation function for smooth transitions
        solve_rate_bins = np.array(sorted(bin_changes.keys()))
        change_values = np.array([bin_changes[bin] for bin in solve_rate_bins])
        # Note: confidence_weights could be used for weighted interpolation in future enhancements
        # confidence_weights = np.array([bin_weights[bin] for bin in solve_rate_bins])

        # Use weighted interpolation if we have multiple points
        if len(solve_rate_bins) > 1:
            # Create interpolation function with extrapolation
            interp_func = interp1d(
                solve_rate_bins, change_values, kind='linear', fill_value='extrapolate', bounds_error=False
            )
        else:
            # Single point - use constant value
            single_change = change_values[0]
            interp_func = lambda x: np.full_like(x, single_change)

        # 3. Vectorized update of stale samples
        stale_ids = []
        stale_rates = []
        stale_steps = []

        for problem_id, solve_rate in self.success_rates.items():
            step_diff = self.current_step - self.last_update_steps[problem_id]
            if step_diff > self.propagation_threshold:
                stale_ids.append(problem_id)
                stale_rates.append(solve_rate)
                stale_steps.append(step_diff)

        if not stale_ids:
            return

        # Vectorized interpolation
        stale_rates_array = np.array(stale_rates)
        stale_steps_array = np.array(stale_steps)

        # Get interpolated change rates
        interpolated_changes = interp_func(stale_rates_array)

        # Calculate confidence based on distance to nearest observed bin
        distances = np.abs(stale_rates_array[:, None] - solve_rate_bins[None, :])
        min_distances = np.min(distances, axis=1)
        distance_confidence = np.exp(-min_distances * 5)  # Exponential decay with distance

        # Apply changes with adaptive confidence
        for i, (problem_id, step_diff) in enumerate(zip(stale_ids, stale_steps_array)):
            change_rate = interpolated_changes[i]

            if abs(change_rate) > 1e-6:  # Only update if change is significant
                old_rate = stale_rates_array[i]

                # Calculate estimated change
                estimated_change = change_rate * step_diff
                new_rate = old_rate + estimated_change
                new_rate = np.clip(new_rate, 0.0, 1.0)  # Clamp to valid range

                # Adaptive confidence based on:
                # 1. Distance to observed data
                # 2. Time since last update (decay confidence over time)
                # 3. Global trend consistency
                base_confidence = self.propagation_confidence
                distance_factor = distance_confidence[i]
                time_factor = np.exp(-step_diff / 20.0)  # Decay over time

                # Check consistency with global trend
                global_trend = np.mean(self.global_change_history[-10:]) if self.global_change_history else 0
                trend_consistency = 1.0 - min(1.0, abs(change_rate - global_trend) / max(abs(global_trend), 0.1))

                final_confidence = base_confidence * distance_factor * time_factor * trend_consistency
                final_confidence = np.clip(final_confidence, 0.1, 0.9)  # Reasonable bounds

                # Apply update
                updated_rate = old_rate * (1 - final_confidence) + new_rate * final_confidence
                self.success_rates[problem_id] = updated_rate

                # Mark as updated (but not with full recency)
                self.last_update_steps[problem_id] = max(
                    self.last_update_steps[problem_id],
                    self.current_step - int(step_diff * 0.5),  # Partial recency credit
                )

    def set_current_step(self, step: int):
        """Update current training step"""
        self.current_step = step

    def get_freshness_stats(self) -> dict:
        """Get statistics about sample freshness"""
        usage_counts = list(self.usage_counts.values())
        step_diffs = [self.current_step - step for step in self.last_update_steps.values()]

        return {
            'avg_usage_count': np.mean(usage_counts) if usage_counts else 0,
            'max_usage_count': max(usage_counts) if usage_counts else 0,
            'fresh_samples': sum(1 for count in usage_counts if count == 0),
            'stale_samples': sum(1 for diff in step_diffs if diff > self.propagation_threshold),
            'total_samples': len(usage_counts),
            'avg_staleness': np.mean(step_diffs) if step_diffs else 0,
        }

    def print_solve_rate_bin_distribution(self) -> None:
        """Enhanced distribution printing with freshness info"""

        def count_values_in_bins(data, start=0.0, end=1.0, bin_size=0.2):
            bins = np.arange(start, end + bin_size, bin_size)
            bin_labels = [f"{round(bins[i], 2)}-{round(bins[i+1], 2)}" for i in range(len(bins) - 1)]
            binned_data = pd.cut(data, bins=bins, labels=bin_labels, include_lowest=True)
            counts = pd.Series(binned_data).value_counts(sort=False)
            return counts

        data = list(self.success_rates.values())
        counts = count_values_in_bins(data, bin_size=0.1)

        print(f"[Global] Solve rate distribution:")
        print(counts)

        # Print freshness stats
        freshness_stats = self.get_freshness_stats()
        print(f"[Global] Freshness stats: {freshness_stats}")

        # Print histogram of last update steps
        last_steps_stats = {
            'avg_last_update_step': np.mean(list(self.last_update_steps.values())),
            'max_last_update_step': max(self.last_update_steps.values()),
            'min_last_update_step': min(self.last_update_steps.values()),
        }
        print(f"[Global] Last update step stats: {last_steps_stats}")

    def __len__(self) -> int:
        """Estimate number of batches"""
        estimated_samples = len(self.success_rates)
        return max(1, estimated_samples // self.target_batch_size)


def test_dynamic_solve_rate_filtering(parquet_files: str, tokenizer) -> None:
    """
    End-to-end test of dynamic solve rate filtering system with save/restore

    Test Scenarios:
    1. Initial filtering with default rates
    2. Behavior after solve rate updates
    3. Save/restore functionality
    4. Multi-epoch training simulation
    """

    # Mock processor for image data handling
    class MockProcessor:
        image_processor = type('', (), {'return_tensors': 'pt'})()

    # Initialize dynamic dataset
    dataset = SolveRateDynamicRLHFDataset(
        parquet_files=parquet_files,
        tokenizer=tokenizer,
        processor=MockProcessor(),
        filter_overlong_prompts=True,
        sample_size=None,
        max_prompt_length=4096,
    )

    # Configure sampling strategy
    train_dataloader_generator = torch.Generator().manual_seed(1)
    base_sampler = RandomSampler(dataset, generator=train_dataloader_generator)

    batch_sampler = RefreshSolveRateSampler(
        dataset=dataset,
        sampler=base_sampler,
        target_batch_size=512,
        oversampling_factor=1.2,
        solverate_mean=0.5,
        solverate_std=0.1,
        freshness_balance=0.1,
        current_step=0,
        propagation_threshold=5,
        propagation_confidence=0.8,
        default_filter_rate=0.0,
        ema_alpha=1.0,
        shuffle=True,
        seed=42,
        world_size=32,
    )

    # Create stateful dataloader
    dataloader = StatefulDataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    # Training loop simulation with save/restore
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_steps = 0
    save_interval = 100  # Save every 10 steps

    for epoch in range(100):
        print(f"=== Epoch {epoch}, Step {total_steps} ===")

        for data in dataloader:
            # Simulate training updates
            success_rates = {}
            for idx in data["prompt_index"]:
                # Simulate gradual improvement with some noise
                current_rate = dataloader.batch_sampler.success_rates[idx]
                improvement = np.random.normal(0.02, 0.01)  # Small improvement with variance
                success_rates[idx] = np.clip(current_rate + improvement, 0.0, 1.0)

            # Update sampler state
            dataloader.batch_sampler.set_current_step(total_steps)
            dataloader.batch_sampler.update_success_rates(success_rates)

            # Periodic saving
            if total_steps % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"sampler_step_{total_steps}.pkl")
                dataloader.batch_sampler.save_state(checkpoint_path)
                print(f"Saved checkpoint at step {total_steps}")

                # Test restore functionality
                if total_steps > 0:
                    print("Testing restore functionality...")
                    # Create a new sampler and load the state
                    test_sampler = RefreshSolveRateSampler(
                        dataset=dataset,
                        sampler=base_sampler,
                        target_batch_size=512,
                        oversampling_factor=1.2,
                        default_filter_rate=0.0,
                        ema_alpha=1.0,
                        shuffle=True,
                        seed=42,
                        world_size=32,
                    )
                    test_sampler.load_state(checkpoint_path)
                    print("Restore test successful!")

            # Print statistics
            if total_steps % 5 == 0:
                dataloader.batch_sampler.print_solve_rate_bin_distribution()
            total_steps += 1


if __name__ == "__main__":
    # Initialize components for testing
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local

    model_path = copy_to_local("/mnt/hdfs/tiktok_aiic/user/liuqian/Qwen2.5-7B")
    tokenizer = hf_tokenizer(model_path, trust_remote_code=False)

    # Execute test suite
    test_dynamic_solve_rate_filtering(
        "/mnt/hdfs/tiktok_aiic/user/liuqian/rl_datasets/deepscaler/train_with_solverate_n16_0602.parquet", tokenizer
    )
