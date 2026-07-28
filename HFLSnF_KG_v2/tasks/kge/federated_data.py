"""知识图谱客户端分区类型与头实体均衡划分。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .data import KnowledgeGraphDataset


@dataclass(frozen=True)
class KnowledgeGraphClientPartition:
    """保存一个知识客户端独占的训练三元组及局部知识范围。"""

    client_id: int
    train_triples: torch.Tensor
    head_entity_ids: torch.Tensor
    entity_ids: torch.Tensor
    relation_ids: torch.Tensor

    def __post_init__(self) -> None:
        """校验客户端编号、训练三元组和知识范围张量。"""

        if int(self.client_id) < 0:
            raise ValueError("知识客户端编号不能小于0")
        if (
            self.train_triples.dtype != torch.long
            or self.train_triples.ndim != 2
            or self.train_triples.shape[1] != 3
        ):
            raise ValueError("客户端训练三元组必须是形状[N, 3]的long张量")
        if int(self.train_triples.shape[0]) <= 0:
            raise ValueError("知识客户端不能没有训练三元组")
        for field_name, values in (
            ("head_entity_ids", self.head_entity_ids),
            ("entity_ids", self.entity_ids),
            ("relation_ids", self.relation_ids),
        ):
            if values.dtype != torch.long or values.ndim != 1:
                raise ValueError("{}必须是一维long张量".format(field_name))
            if int(values.numel()) <= 0:
                raise ValueError("{}不能为空".format(field_name))

    @property
    def triple_count(self) -> int:
        """返回客户端持有的正训练三元组数量。"""

        return int(self.train_triples.shape[0])

    def summary(self) -> Dict[str, object]:
        """返回适合写入分区摘要的客户端统计。"""

        return {
            "client_id": int(self.client_id),
            "triple_count": self.triple_count,
            "head_entity_count": int(self.head_entity_ids.numel()),
            "entity_count": int(self.entity_ids.numel()),
            "relation_count": int(self.relation_ids.numel()),
        }


@dataclass(frozen=True)
class FederatedKnowledgeGraphData:
    """保存完整知识图谱和互不重复的客户端训练分区。"""

    dataset: KnowledgeGraphDataset
    partitions: Tuple[KnowledgeGraphClientPartition, ...]
    partition_strategy: str
    partition_seed: int
    partition_hash: str

    def __post_init__(self) -> None:
        """校验客户端编号连续且训练三元组严格无重复无遗漏。"""

        if not self.partitions:
            raise ValueError("联邦知识图谱至少需要一个客户端分区")
        client_ids = [int(partition.client_id) for partition in self.partitions]
        if client_ids != list(range(len(self.partitions))):
            raise ValueError("知识客户端编号必须从0开始连续排列")
        combined = torch.cat(
            [partition.train_triples for partition in self.partitions], dim=0
        )
        source_rows = sorted(map(tuple, self.dataset.train_triples.tolist()))
        combined_rows = list(map(tuple, combined.tolist()))
        if len(combined_rows) != len(set(combined_rows)):
            raise ValueError("客户端分区之间包含重复训练三元组")
        if sorted(combined_rows) != source_rows:
            raise ValueError("客户端分区没有完整覆盖训练三元组")

        head_owners: Dict[int, int] = {}
        for partition in self.partitions:
            for head_id in partition.head_entity_ids.tolist():
                head_id = int(head_id)
                if head_id in head_owners:
                    raise ValueError(
                        "头实体{}被分配给多个客户端".format(head_id)
                    )
                head_owners[head_id] = int(partition.client_id)

    @property
    def client_count(self) -> int:
        """返回联邦知识客户端数量。"""

        return len(self.partitions)

    @staticmethod
    def _overlap_statistics(
        partitions: Sequence[KnowledgeGraphClientPartition],
    ) -> Dict[str, float]:
        """计算客户端实体集合两两Jaccard重叠统计。"""

        entity_sets = [
            set(int(value) for value in partition.entity_ids.tolist())
            for partition in partitions
        ]
        overlaps = []
        for left_index in range(len(entity_sets)):
            for right_index in range(left_index + 1, len(entity_sets)):
                union = entity_sets[left_index].union(entity_sets[right_index])
                intersection = entity_sets[left_index].intersection(
                    entity_sets[right_index]
                )
                overlaps.append(
                    float(len(intersection)) / float(len(union))
                    if union
                    else 0.0
                )
        if not overlaps:
            return {
                "entity_jaccard_mean": 0.0,
                "entity_jaccard_min": 0.0,
                "entity_jaccard_max": 0.0,
            }
        return {
            "entity_jaccard_mean": float(np.mean(overlaps)),
            "entity_jaccard_min": float(np.min(overlaps)),
            "entity_jaccard_max": float(np.max(overlaps)),
        }

    def summary(self) -> Dict[str, object]:
        """返回数据集、划分方法、负载和实体重叠摘要。"""

        triple_counts = [
            partition.triple_count for partition in self.partitions
        ]
        return {
            "dataset": self.dataset.dataset_name,
            "partition_strategy": self.partition_strategy,
            "partition_seed": int(self.partition_seed),
            "partition_hash": self.partition_hash,
            "client_count": self.client_count,
            "total_train_triple_count": int(sum(triple_counts)),
            "min_client_triple_count": int(min(triple_counts)),
            "max_client_triple_count": int(max(triple_counts)),
            "mean_client_triple_count": float(np.mean(triple_counts)),
            **self._overlap_statistics(self.partitions),
            "clients": [
                partition.summary() for partition in self.partitions
            ],
        }


def _seeded_head_order(
    head_counts: Dict[int, int], seed: int
) -> List[int]:
    """按三元组数降序并使用固定种子打破相同频次头实体的顺序。"""

    rng = np.random.RandomState(int(seed))
    tie_breakers = {
        int(head_id): float(rng.random_sample())
        for head_id in sorted(head_counts.keys())
    }
    return sorted(
        head_counts.keys(),
        key=lambda head_id: (
            -int(head_counts[head_id]),
            tie_breakers[int(head_id)],
            int(head_id),
        ),
    )


def _partition_hash(
    partitions: Sequence[KnowledgeGraphClientPartition],
) -> str:
    """计算客户端编号和有序三元组内容的SHA-256划分指纹。"""

    digest = hashlib.sha256()
    for partition in partitions:
        digest.update(
            int(partition.client_id).to_bytes(
                8, byteorder="little", signed=False
            )
        )
        contiguous = partition.train_triples.detach().cpu().contiguous()
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def partition_train_triples_by_head(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seed: int,
) -> FederatedKnowledgeGraphData:
    """按头实体三元组数贪心均衡地构造互斥客户端分区。"""

    client_count = int(client_count)
    if client_count <= 0:
        raise ValueError("client_count必须大于0")
    rows = dataset.train_triples.tolist()
    head_to_row_indices: Dict[int, List[int]] = {}
    for row_index, row in enumerate(rows):
        head_id = int(row[0])
        head_to_row_indices.setdefault(head_id, []).append(row_index)
    if client_count > len(head_to_row_indices):
        raise ValueError(
            "客户端数{}超过训练头实体数{}".format(
                client_count, len(head_to_row_indices)
            )
        )

    head_counts = {
        head_id: len(row_indices)
        for head_id, row_indices in head_to_row_indices.items()
    }
    client_loads = [0 for _ in range(client_count)]
    client_heads: List[List[int]] = [
        [] for _ in range(client_count)
    ]
    for head_id in _seeded_head_order(head_counts, seed):
        client_id = min(
            range(client_count),
            key=lambda value: (client_loads[value], value),
        )
        client_heads[client_id].append(int(head_id))
        client_loads[client_id] += int(head_counts[head_id])

    partitions = []
    for client_id, head_ids in enumerate(client_heads):
        row_indices = sorted(
            row_index
            for head_id in head_ids
            for row_index in head_to_row_indices[head_id]
        )
        local_triples = dataset.train_triples.index_select(
            0, torch.tensor(row_indices, dtype=torch.long)
        )
        local_entities = torch.unique(
            local_triples[:, (0, 2)], sorted=True
        )
        local_relations = torch.unique(
            local_triples[:, 1], sorted=True
        )
        partitions.append(
            KnowledgeGraphClientPartition(
                client_id=client_id,
                train_triples=local_triples,
                head_entity_ids=torch.tensor(
                    sorted(head_ids), dtype=torch.long
                ),
                entity_ids=local_entities,
                relation_ids=local_relations,
            )
        )
    partition_tuple = tuple(partitions)
    return FederatedKnowledgeGraphData(
        dataset=dataset,
        partitions=partition_tuple,
        partition_strategy="balanced_head_entity",
        partition_seed=int(seed),
        partition_hash=_partition_hash(partition_tuple),
    )
