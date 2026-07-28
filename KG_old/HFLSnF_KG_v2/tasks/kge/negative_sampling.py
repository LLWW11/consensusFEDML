"""排除已知真三元组的TransE负采样器。"""

from __future__ import annotations

from typing import Iterable, Set, Tuple

import numpy as np
import torch


IdTriple = Tuple[int, int, int]


class FilteredNegativeSampler:
    """通过替换头实体或尾实体生成确定性filtered负样本。"""

    def __init__(
        self,
        num_entities: int,
        true_triples: Iterable[IdTriple],
        seed: int,
    ):
        """保存真三元组集合并初始化独立NumPy随机数生成器。"""

        if int(num_entities) <= 1:
            raise ValueError("负采样至少需要两个候选实体")
        self.num_entities = int(num_entities)
        self.true_triples: Set[IdTriple] = {
            tuple(int(value) for value in triple)
            for triple in true_triples
        }
        self.rng = np.random.RandomState(int(seed))

    def _sample_one(
        self, triple: IdTriple, corruption_mode: str
    ) -> IdTriple:
        """按指定头尾替换模式寻找不属于已知真集合的负三元组。"""

        head, relation, tail = triple
        normalized_mode = str(corruption_mode).strip().lower()
        if normalized_mode == "head_tail":
            replace_head = bool(self.rng.randint(0, 2))
        elif normalized_mode == "tail":
            replace_head = False
        else:
            raise ValueError(
                "corruption_mode必须是head_tail或tail，实际为{}".format(
                    corruption_mode
                )
            )
        for _ in range(64):
            candidate = int(self.rng.randint(0, self.num_entities))
            negative = (
                (candidate, relation, tail)
                if replace_head
                else (head, relation, candidate)
            )
            if negative != triple and negative not in self.true_triples:
                return negative

        # 稠密微型图可能需要穷举；从随机起点扫描仍保持固定种子可复现。
        start = int(self.rng.randint(0, self.num_entities))
        for offset in range(self.num_entities):
            candidate = (start + offset) % self.num_entities
            negative = (
                (candidate, relation, tail)
                if replace_head
                else (head, relation, candidate)
            )
            if negative != triple and negative not in self.true_triples:
                return negative
        raise RuntimeError(
            "无法为三元组{}生成filtered负样本".format(triple)
        )

    def sample(
        self,
        positive_triples: torch.Tensor,
        negative_sample_count: int = 1,
        corruption_mode: str = "head_tail",
    ) -> torch.Tensor:
        """按指定替换方式为每个正三元组生成filtered负样本。"""

        if positive_triples.dtype != torch.long:
            raise TypeError("正三元组必须使用torch.long")
        if positive_triples.ndim != 2 or positive_triples.shape[1] != 3:
            raise ValueError("正三元组形状必须为[N, 3]")
        if int(negative_sample_count) <= 0:
            raise ValueError("negative_sample_count必须大于0")
        device = positive_triples.device
        source = positive_triples.detach().cpu().tolist()
        negatives = [
            self._sample_one(
                tuple(int(value) for value in triple),
                corruption_mode,
            )
            for triple in source
            for _ in range(int(negative_sample_count))
        ]
        return torch.tensor(
            negatives, dtype=torch.long, device=device
        )
