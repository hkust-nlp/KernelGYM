import logging
import random
from typing import Dict

import numpy as np

from .batch_sampler import DynamicBatchSampler


class PrioritizedBatchSampler(DynamicBatchSampler):
    """
    在DynamicBatchSampler基础上添加基于解决率的优先级采样。
    除了正常的filter rate外, 还会根据问题的解决率决定是否保留该问题。
    """

    def __init__(
        self,
        sampler,
        target_batch_size: int,
        oversampling_factor: float = 1.5,
        default_filter_rate: float = 0.2,
        default_success_rate: float = 0.0,
        ema_alpha: float = 1.0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed=None,
        world_size: int = 8,
    ):
        super().__init__(
            sampler=sampler,
            target_batch_size=target_batch_size,
            oversampling_factor=oversampling_factor,
            default_filter_rate=default_filter_rate,
            ema_alpha=ema_alpha,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            world_size=world_size,
        )

        # 为每个样本索引维护解决率
        self.success_rates = {}
        self.default_success_rate = default_success_rate

        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)

    def update_success_rates(self, new_rates: Dict[int, float]):
        """
        更新样本解决率。

        Args:
            new_rates: 新的解决率字典 {problem_id: success_rate}
        """
        # 更新每个样本的解决率
        for problem_id, new_rate in new_rates.items():
            if problem_id in self.success_rates:
                old_rate = self.success_rates[problem_id]
                self.success_rates[problem_id] = (1 - self.ema_alpha) * old_rate + self.ema_alpha * new_rate
            else:
                self.success_rates[problem_id] = new_rate

        # 记录日志
        if self.success_rates:
            rates = list(self.success_rates.values())
            self.logger.info(
                f"Success rates - Mean: {np.mean(rates):.3f}, " f"Min: {min(rates):.3f}, Max: {max(rates):.3f}"
            )

    def __iter__(self):
        """
        在父类的基础上添加基于解决率的采样。
        """
        # 获取所有候选索引
        indices = list(self.sampler)

        # 根据shuffle参数决定是否随机打乱
        if self.shuffle:
            self.rng.shuffle(indices)

        # 分批次采样
        batch = []
        effective_count = 0
        target = self.target_batch_size * self.oversampling_factor

        for idx in indices:
            # 获取样本的预期有效率和成功率
            expected_effective_rate = 1.0 - self.filter_rates.get(idx, self.last_batch_filter_rate)
            success_rate = self.success_rates.get(idx, self.default_success_rate)

            # 根据成功率决定是否保留样本
            if self.rng.random() < (1.0 - success_rate):
                # 添加到当前批次
                batch.append(idx)
                effective_count += expected_effective_rate

            # 如果预期有效样本数已达到目标，返回批次
            if effective_count >= target:
                if len(batch) % self.world_size == 0:
                    yield batch
                    batch = []
                    effective_count = 0
                else:
                    continue

        # 返回最后不足一批的样本（如果有）
        if batch and not not self.drop_last:
            if len(batch) % self.world_size == 0:
                yield batch
            else:
                if len(batch) > self.world_size:
                    yield batch[: (len(batch) // self.world_size) * self.world_size]
                else:
                    pass

    def get_metrics(self) -> Dict[str, float]:
        """
        获取采样器指标，包括父类的filter rate指标和success rate指标。
        """
        metrics = super().get_metrics()

        if self.success_rates:
            rates = list(self.success_rates.values())
            keep_rates = [1.0 - rate for rate in rates]
            metrics.update(
                {
                    "sampler/success_rate_mean": np.mean(rates),
                    "sampler/success_rate_max": max(rates),
                    "sampler/success_rate_min": min(rates),
                    "sampler/keep_rate_mean": np.mean(keep_rates),
                    "sampler/keep_rate_max": max(keep_rates),
                    "sampler/keep_rate_min": min(keep_rates),
                }
            )
        else:
            metrics.update(
                {
                    "sampler/success_rate_mean": self.default_success_rate,
                    "sampler/success_rate_max": self.default_success_rate,
                    "sampler/success_rate_min": self.default_success_rate,
                    "sampler/keep_rate_mean": 1.0 - self.default_success_rate,
                    "sampler/keep_rate_max": 1.0 - self.default_success_rate,
                    "sampler/keep_rate_min": 1.0 - self.default_success_rate,
                }
            )

        return metrics
