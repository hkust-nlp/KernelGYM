"""
dynamic_solve_rate_dataset.py
Implementation of dynamic solve rate filtered RLHF dataset with test cases

Key Components:
1. SolveRateDynamicRLHFDataset - Dataset with dynamic solve rate tracking
2. DynamicSolveRateSampler - Batch sampler with adaptive filtering based on solve rates
3. Test cases for validation
"""

import json
import os
import random
import time
from typing import *

import numpy as np
import pandas as pd
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl_patch.utils.dataset.rl_dataset import SolveRateDynamicRLHFDataset, collate_fn
from verl_patch.utils.samplers.prioritized_batch_sampler import PrioritizedBatchSampler


def print_dict(d):
    """Helper function to print dictionaries in readable JSON format"""
    print(json.dumps(d, indent=4, ensure_ascii=False))


def convert_to_native_types(obj):
    """Recursively convert numpy and torch types to native Python types while avoiding unnecessary list wrapping"""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(v) for v in obj]
    elif hasattr(obj, 'item') and hasattr(obj, 'dtype'):
        # Handle scalars and single-element arrays/tensors
        if (hasattr(obj, 'ndim') and obj.ndim == 0) or (hasattr(obj, 'dim') and obj.dim() == 0):
            return obj.item()  # True scalar
        elif getattr(obj, 'size', 1) == 1 or getattr(obj, 'numel', lambda: 1)() == 1:
            return obj.reshape(()).item()  # Single-element array/tensor
        else:
            return convert_to_native_types(obj.tolist())  # Recursive conversion for multi-element
    return obj


def parquet_to_list_of_dicts(input_file):
    # Load and convert data with type safety
    start_time = time.time()
    df = pd.read_parquet(input_file)

    # Convert DataFrame to native types using JSON serialization
    temp_json_path = "temp_data_conversion.jsonl"
    try:
        df.to_json(temp_json_path, orient='records', lines=True)
        with open(temp_json_path) as f:
            data = [convert_to_native_types(json.loads(line)) for line in f]
    finally:
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
    print(f'Data loading completed in {time.time()-start_time:.2f}s')
    return data


class DynamicSolveRateSampler(PrioritizedBatchSampler):
    """
    Adaptive batch sampler that dynamically filters samples based on solve rates

    Args:
        dataset: SolveRateDynamicRLHFDataset instance
        sampler: Base data sampler
        target_batch_size: Desired effective batch size
        oversampling_factor: Multiplier for initial sample selection
        default_filter_rate: Base rate for filtering unobserved samples
        default_success_rate: Base success rate for new samples
        ema_alpha: Exponential moving average parameter for rate updates
        shuffle: Whether to shuffle indices
        drop_last: Whether to drop incomplete batches
        seed: Random seed for reproducibility
        world_size: Distributed training world size
    """

    def __init__(self, dataset: SolveRateDynamicRLHFDataset, solverate_low=0.1, solverate_high=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self._read_init_solve_rate()
        self.solverate_low = solverate_low
        self.solverate_high = solverate_high

    def _read_init_solve_rate(self):
        # get initial solve rates for each data sample
        # index_set = set([data['prompt_index'] for data in self.dataset])
        # assert len(index_set) == len(self.dataset), f"""{len(index_set)} vs {len(self.dataset)}"""
        for data in self.dataset:
            data_id = data['prompt_index']
            self.success_rates[data_id] = data['solve_rate']

        # logging
        rates = list(self.success_rates.values())
        self.logger.info(
            f"Success rates - Mean: {np.mean(rates):.3f}, " f"Min: {min(rates):.3f}, Max: {max(rates):.3f}"
        )

    def __iter__(self) -> iter:
        """
        Generates batches of indices with dynamic solve rate filtering

        Strategy:
        1. Start with full candidate pool from base sampler
        2. Apply solve-rate based probabilistic filtering
        3. Build batches until reaching target effective sample count
        """
        # all data
        indices = list(self.success_rates.keys())

        # Check if we have any indices to work with
        if len(indices) == 0:
            raise ValueError(
                "ERROR: No indices available for sampling. Dataset may be empty or not properly initialized."
            )

        if self.shuffle:
            self.rng.shuffle(indices)

        batch = []
        effective_count = 0
        target = self.target_batch_size * self.oversampling_factor

        for idx in indices:

            # Get sample success probability
            success_rate = self.success_rates.get(idx, None)
            if success_rate is None:
                # Skip samples without a success rate
                continue

            # Probabilistic inclusion based on success rate
            if self.solverate_low <= success_rate <= self.solverate_high:
                # Calculate expected contribution to effective batch
                filter_rate = self.filter_rates.get(idx, self.last_batch_filter_rate)
                effective_contribution = 1.0 - filter_rate
                batch.append(idx)
                effective_count += effective_contribution
            elif self.solverate_low > success_rate:
                # print("aaaaaaaa: ", self.rng.random())
                if self.rng.random() < 0.05 * (1.0 - success_rate):
                    filter_rate = self.filter_rates.get(idx, self.last_batch_filter_rate)
                    effective_contribution = 1.0 - filter_rate
                    effective_count += effective_contribution
                    batch.append(idx)
            elif self.solverate_high < success_rate:
                if self.rng.random() < 0.1 * (1.0 - success_rate):
                    filter_rate = self.filter_rates.get(idx, self.last_batch_filter_rate)
                    effective_contribution = 1.0 - filter_rate
                    effective_count += effective_contribution
                    batch.append(idx)
            else:
                raise ValueError(f"Invalid solve rate: {success_rate}")

            # Yield batch when target is reached
            if effective_count >= target:
                if len(batch) % self.world_size == 0:
                    # print("effective_counttttttt:", effective_count)
                    # print("batchhhhhhhhhh:", batch)
                    # print("targetttttttttt:", target)
                    # print("self.world_sizeeeeeeeee:", self.world_size)
                    accepted_solve_rates = [self.success_rates[idx] for idx in batch]
                    solve_rate_hist = np.histogram(accepted_solve_rates, bins=np.arange(0, 1.1, 0.1))
                    print(f"[Batch] solve rate distribution: {solve_rate_hist}")
                    yield batch
                    batch = []
                    effective_count = 0

        # Handle final incomplete batch
        if batch and not self.drop_last:
            if len(batch) % self.world_size == 0:
                yield batch
            elif len(batch) > self.world_size:
                # Truncate to last full parallel batch
                yield batch[: (len(batch) // self.world_size) * self.world_size]

    def __len__(self) -> int:
        """Estimate number of batches based on current solve rates"""
        # Count samples in medium difficulty range (25%-75% solve rate)
        valid_samples = sum(
            1 for rate in list(self.success_rates.values()) if self.solverate_low <= rate <= self.solverate_high
        )
        # print("hhhhhhh: ", valid_samples)
        return valid_samples // self.target_batch_size

    def update_success_rates(self, new_rates: Dict[int, float]):
        """
        更新样本解决率。

        Args:
            new_rates: 新的解决率字典 {problem_id: success_rate}
        """
        # 更新每个样本的解决率
        for problem_id, new_rate in new_rates.items():
            try:
                assert new_rate <= 1.0, f"Invalid solve rate: {new_rate} > 1.0"
            except:
                new_rate = 1.0
            old_rate = self.success_rates[problem_id]
            self.success_rates[problem_id] = (1 - self.ema_alpha) * old_rate + self.ema_alpha * new_rate

        # 记录日志
        if self.success_rates:
            rates = list(self.success_rates.values())
            self.logger.info(
                f"Success rates - Mean: {np.mean(rates):.3f}, " f"Min: {min(rates):.3f}, Max: {max(rates):.3f}"
            )

    def print_solve_rate_bin_distribution(self) -> None:
        def count_values_in_bins(data, start=0.0, end=1.0, bin_size=0.2):
            """
            Count the number of values in each bin.
            Parameters:
            - data (list): List of float values.
            - start (float): Start of the range.
            - end (float): End of the range.
            - bin_size (float): Size of each bin.
            Returns:
            - counts (pd.Series): Counts for each bin.
            """
            # Create bin edges
            bins = np.arange(start, end + bin_size, bin_size)
            # Use `pd.cut` to assign values to bins
            bin_labels = [f"{round(bins[i], 2)} - {round(bins[i+1], 2)}" for i in range(len(bins) - 1)]
            binned_data = pd.cut(data, bins=bins, labels=bin_labels, include_lowest=True)
            # Convert to Series and count values
            counts = pd.Series(binned_data).value_counts(sort=False)  # Keep bins in order
            return counts

        data = list(self.success_rates.values())
        counts = count_values_in_bins(data, bin_size=0.1)
        print(f"Solve rate distribution:")
        print(counts)


def test_dynamic_solve_rate_filtering(parquet_files: str, tokenizer) -> None:
    """
    End-to-end test of dynamic solve rate filtering system

    Test Scenarios:
    1. Initial filtering with default rates
    2. Behavior after solve rate updates
    3. Multi-epoch training simulation
    """

    # Mock processor for image data handling
    class MockProcessor:
        image_processor = type('', (), {'return_tensors': 'pt'})()

    # dataset = parquet_to_list_of_dicts(parquet_files)
    # print_dict(convert_to_native_types(dataset[0]))

    # Initialize dynamic dataset
    dataset = SolveRateDynamicRLHFDataset(
        parquet_files=parquet_files,
        tokenizer=tokenizer,
        processor=MockProcessor(),
        filter_overlong_prompts=False,
        sample_size=None,
        max_prompt_length=2048,
    )
    # print(convert_to_native_types(dataset[0]))

    # Configure sampling strategy
    train_dataloader_generator = torch.Generator().manual_seed(1)
    base_sampler = RandomSampler(dataset, generator=train_dataloader_generator)
    # print(sorted(list(base_sampler)))

    batch_sampler = DynamicSolveRateSampler(
        dataset=dataset,
        sampler=base_sampler,
        target_batch_size=512,
        oversampling_factor=1.2,
        default_filter_rate=0,
        ema_alpha=1.0,
        shuffle=True,
        seed=42,
        world_size=32,  # Simulate 32-GPU setup
    )
    # breakpoint()
    # print("hahhahaah:", len(batch_sampler.sampler))

    # Create stateful dataloader
    dataloader = StatefulDataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    # Training loop simulation
    for epoch in range(1000):
        print()
        print(f"Number of batches: {len(dataloader)}")
        print(
            f"""Unsolved count: {
            len([r for r in list(dataloader.batch_sampler.success_rates.values())
                 if r < dataloader.batch_sampler.solverate_low])
            }"""
        )
        # print("wocaooooooo: ", dataloader.batch_sampler.success_rates[2])
        print("Epoch: ", epoch)
        if len(dataloader) > 0:
            for i, data in enumerate(dataloader):
                success_rates = {}
                for idx in data["prompt_index"]:
                    success_rates[idx] = dataloader.batch_sampler.success_rates[idx] + 0.05
                    dataloader.batch_sampler.update_success_rates(success_rates)
                    # Mock solve rate update
                    # if  dataloader.batch_sampler.success_rates[idx] + 0.1 > 1.0:
                    #     dataloader.batch_sampler.success_rates[idx] = 1
                    # else:
                    #     dataloader.batch_sampler.success_rates[idx] += 0.1
                # current_solve_rates = data["solve_rate_Qwen2.5-Math-7B_16"]
                # print(data)
                # print(data["solve_rate_Qwen2.5-Math-7B_16"])
                # success_rates = {data[""]}
                # dataloader.batch_sampler.update_success_rates(success_rates)

                # print(f"""{epoch}-{i}""")
        dataloader.batch_sampler.print_solve_rate_bin_distribution()


if __name__ == "__main__":
    # Initialize components for testing
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local

    model_path = copy_to_local("/mnt/hdfs/codeai/hf_models/Qwen2.5-Math-7B")
    tokenizer = hf_tokenizer(model_path, trust_remote_code=False)

    # Execute test suite
    test_dynamic_solve_rate_filtering(
        "/mnt/hdfs/codeai/rl_datasets/simplelr_math_35/train_with_solverate_n16_0602.parquet", tokenizer
    )
