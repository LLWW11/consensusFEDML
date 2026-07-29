"""为知识图谱训练样本向量化计算频率子采样权重。"""

from __future__ import annotations

from typing import Tuple

import torch


class TripleFrequencySubsampler:
    """预计算头关系和尾关系模式频次，并提供批量权重查询。"""

    def __init__(
        self,
        train_triples: torch.Tensor,
        start_count: int = 4,
        num_relations: int = None,
    ):
        """一次性建立排序频次索引和与训练行严格对齐的权重张量。"""

        if train_triples.dtype != torch.long:
            raise TypeError("训练三元组必须使用torch.long")
        if train_triples.ndim != 2 or train_triples.shape[1] != 3:
            raise ValueError("训练三元组形状必须为[N, 3]")
        if int(train_triples.shape[0]) <= 0:
            raise ValueError("训练三元组不能为空")
        if int(start_count) <= 0:
            raise ValueError("start_count必须大于0")
        inferred_relations = int(train_triples[:, 1].max().item()) + 1
        self.num_relations = int(
            inferred_relations
            if num_relations is None
            else num_relations
        )
        if self.num_relations < inferred_relations:
            raise ValueError("num_relations小于训练三元组关系编号范围")
        self.start_count = int(start_count)
        source = train_triples.detach().cpu()
        head_keys, tail_keys = self._pattern_keys(source)
        (
            self._head_unique_keys,
            head_inverse,
            self._head_counts,
        ) = torch.unique(
            head_keys,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        (
            self._tail_unique_keys,
            tail_inverse,
            self._tail_counts,
        ) = torch.unique(
            tail_keys,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        frequencies = (
            self.start_count
            + self._head_counts.index_select(0, head_inverse)
            + self.start_count
            + self._tail_counts.index_select(0, tail_inverse)
        ).float()
        self.precomputed_weights = frequencies.pow(-0.5)

    def _pattern_keys(
        self, triples: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分别编码头关系与尾关系模式，避免两个命名空间相互碰撞。"""

        head_keys = (
            triples[:, 0] * self.num_relations + triples[:, 1]
        )
        tail_keys = (
            triples[:, 2] * self.num_relations + triples[:, 1]
        )
        return head_keys, tail_keys

    @staticmethod
    def _lookup_counts(
        query_keys: torch.Tensor,
        unique_keys: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """通过排序键二分查找频次，不存在的模式返回零。"""

        positions = torch.searchsorted(unique_keys, query_keys)
        safe_positions = positions.clamp(max=int(unique_keys.numel()) - 1)
        matched = (positions < int(unique_keys.numel())) & (
            unique_keys[safe_positions] == query_keys
        )
        values = torch.zeros_like(query_keys)
        values[matched] = counts[safe_positions[matched]]
        return values

    def weights(self, positive_triples: torch.Tensor) -> torch.Tensor:
        """向量化返回任意正三元组批次的平方根逆频率权重。"""

        if positive_triples.dtype != torch.long:
            raise TypeError("正三元组必须使用torch.long")
        if positive_triples.ndim != 2 or positive_triples.shape[1] != 3:
            raise ValueError("正三元组形状必须为[N, 3]")
        source = positive_triples.detach().cpu()
        head_keys, tail_keys = self._pattern_keys(source)
        head_counts = self._lookup_counts(
            head_keys,
            self._head_unique_keys,
            self._head_counts,
        )
        tail_counts = self._lookup_counts(
            tail_keys,
            self._tail_unique_keys,
            self._tail_counts,
        )
        frequencies = (
            self.start_count
            + head_counts
            + self.start_count
            + tail_counts
        ).float()
        return frequencies.pow(-0.5).to(
            device=positive_triples.device
        )


