import logging
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from torch.utils.data.sampler import Sampler


class DynamicBatchSampler(Sampler):
    """
    基于累加有效样本预期的批次采样器。

    采样逻辑：可选地随机打乱索引，然后按顺序累加(1-filter_rate)，
    直到总和达到目标批次大小。
    """

    def __init__(
        self,
        sampler: Sampler,
        target_batch_size: int,
        oversampling_factor: float = 1.5,
        default_filter_rate: float = 0.0,
        ema_alpha: float = 1.0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: Optional[int] = None,
        world_size: Optional[int] = 8,
    ):
        """
        初始化累加式批次采样器。

        Args:
            sampler: 基础采样器，用于获取样本ID
            target_batch_size: 过滤后期望的批次大小
            oversampling_factor: 目标放大系数
            default_filter_rate: 新样本的默认过滤率
            ema_alpha: 更新过滤率的EMA系数
            shuffle: 是否随机打乱样本顺序
            seed: 随机种子
        """
        self.sampler = sampler
        self.target_batch_size = target_batch_size
        self.oversampling_factor = oversampling_factor
        self.default_filter_rate = default_filter_rate
        self.ema_alpha = ema_alpha
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.world_size = world_size

        # 为每个样本索引维护过滤率
        self.filter_rates = defaultdict(lambda: self.default_filter_rate)

        # 全局过滤率
        self.global_filter_rate = self.default_filter_rate

        # 记录上一个batch的过滤率
        self.last_batch_filter_rate = self.default_filter_rate

        # 随机种子
        self.seed = seed
        # 初始化内部随机数生成器，不再调用 global random.seed
        self.rng = random.Random(self.seed) if self.seed is not None else random.Random()

        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 目标有效样本数
        self.target_effective_samples = self.target_batch_size

    def update_filter_stats(self, sample_filter_results: Dict[int, Dict[str, int]]):
        """
        更新样本过滤统计信息。

        Args:
            sample_filter_results: {样本索引: {'before': 采样数, 'after': 过滤后数}}
        """
        total_before = 0
        total_after = 0

        # 更新每个样本的过滤率
        for sample_idx, stats in sample_filter_results.items():
            if stats['before'] == 0:
                continue

            before_count = stats['before']
            after_count = stats['after']
            current_filter_rate = 1.0 - (after_count / before_count)

            total_before += before_count
            total_after += after_count

            # 使用EMA更新
            old_rate = self.filter_rates[sample_idx]
            self.filter_rates[sample_idx] = (1 - self.ema_alpha) * old_rate + self.ema_alpha * current_filter_rate

        # 更新全局过滤率和上一个batch过滤率
        if total_before > 0:
            current_global_rate = 1.0 - (total_after / total_before)
            self.global_filter_rate = (
                1 - self.ema_alpha
            ) * self.global_filter_rate + self.ema_alpha * current_global_rate
            self.last_batch_filter_rate = current_global_rate

        self.logger.info(
            f"Global filter rate: {self.global_filter_rate:.3f}, "
            + f"Last batch filter rate: {self.last_batch_filter_rate:.3f}, "
            + f"Tracked samples: {len(self.filter_rates)}"
        )

    def __iter__(self) -> Iterator[List[int]]:
        """
        按累加预期有效样本数的方式创建批次。
        """
        # 获取所有候选索引
        indices = list(self.sampler)

        # 根据shuffle参数决定是否随机打乱
        if self.shuffle:
            self.rng.shuffle(indices)

        # 分批次采样
        batch = []
        effective_count = 0
        target = self.target_batch_size * self.oversampling_factor  # 期望的有效样本数

        for idx in indices:
            # 获取样本的预期有效率 - 对于新样本使用上一个batch的过滤率
            expected_effective_rate = 1.0 - self.filter_rates.get(idx, self.last_batch_filter_rate)

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
        if batch and not self.drop_last:
            if len(batch) % self.world_size == 0:
                yield batch
            else:
                if len(batch) > self.world_size:
                    yield batch[: (len(batch) // self.world_size) * self.world_size]
                else:
                    pass

    def __len__(self) -> int:
        """
        估计批次总数。
        """
        # 基于平均有效率估计批次数
        avg_effective_rate = 1.0 - self.global_filter_rate
        if avg_effective_rate > 0:
            estimated_samples_per_batch = self.target_batch_size * self.oversampling_factor / avg_effective_rate
            return max(1, int(len(self.sampler) / estimated_samples_per_batch))
        else:
            return 1  # 避免除零

    def get_metrics(self) -> Dict[str, float]:
        """
        获取采样器指标。
        """
        if self.filter_rates:
            rates = list(self.filter_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            mean_rate = sum(rates) / len(rates)
        else:
            max_rate = min_rate = mean_rate = self.default_filter_rate

        # 估计每批的平均样本数
        avg_effective_rate = 1.0 - self.global_filter_rate
        avg_batch_size = (
            self.target_batch_size * self.oversampling_factor / avg_effective_rate
            if avg_effective_rate > 0
            else float('inf')
        )

        return {
            "sampler/tracked_samples": len(self.filter_rates),
            "sampler/global_filter_rate": self.global_filter_rate,
            "sampler/last_batch_filter_rate": self.last_batch_filter_rate,
            "sampler/max_filter_rate": max_rate,
            "sampler/min_filter_rate": min_rate,
            "sampler/mean_filter_rate": mean_rate,
            "sampler/estimated_avg_batch_size": avg_batch_size,
        }
