"""知识图谱客户端分区类型、互斥头实体和关系分层划分。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .data import KnowledgeGraphDataset


BALANCED_HEAD_ENTITY = "balanced_head_entity"
RELATION_STRATIFIED_TRIPLE_BALANCED = (
    "relation_stratified_triple_balanced"
)
SUPPORTED_PARTITION_STRATEGIES = frozenset(
    {
        BALANCED_HEAD_ENTITY,
        RELATION_STRATIFIED_TRIPLE_BALANCED,
    }
)


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
        expected_values = (
            (
                "head_entity_ids",
                torch.unique(self.train_triples[:, 0], sorted=True),
                self.head_entity_ids,
            ),
            (
                "entity_ids",
                torch.unique(
                    self.train_triples[:, (0, 2)], sorted=True
                ),
                self.entity_ids,
            ),
            (
                "relation_ids",
                torch.unique(self.train_triples[:, 1], sorted=True),
                self.relation_ids,
            ),
        )
        for field_name, expected, actual in expected_values:
            if not torch.equal(
                expected.detach().cpu(),
                actual.detach().cpu(),
            ):
                raise ValueError(
                    "{}必须与客户端训练三元组中的实际编号一致".format(
                        field_name
                    )
                )

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
        normalized_strategy = str(self.partition_strategy).strip().lower()
        if normalized_strategy not in SUPPORTED_PARTITION_STRATEGIES:
            raise ValueError(
                "不支持的客户端划分策略：{}".format(
                    self.partition_strategy
                )
            )
        if normalized_strategy != self.partition_strategy:
            raise ValueError("客户端划分策略必须使用规范化小写名称")
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

        # 旧策略的实验语义要求头实体互斥；关系分层策略则有意允许重叠。
        if normalized_strategy == BALANCED_HEAD_ENTITY:
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

    @staticmethod
    def _coverage_counts(
        partitions: Sequence[KnowledgeGraphClientPartition],
        field_name: str,
    ) -> List[int]:
        """统计每个知识编号出现在多少个客户端的局部集合中。"""

        coverage: Dict[int, int] = {}
        for partition in partitions:
            values = getattr(partition, field_name)
            for value in values.tolist():
                normalized_value = int(value)
                coverage[normalized_value] = (
                    coverage.get(normalized_value, 0) + 1
                )
        return list(coverage.values())

    @classmethod
    def _coverage_statistics(
        cls,
        partitions: Sequence[KnowledgeGraphClientPartition],
    ) -> Dict[str, object]:
        """汇总头实体、全部实体和关系的跨客户端覆盖程度。"""

        head_counts = cls._coverage_counts(
            partitions, "head_entity_ids"
        )
        entity_counts = cls._coverage_counts(partitions, "entity_ids")
        relation_counts = cls._coverage_counts(
            partitions, "relation_ids"
        )
        return {
            "head_entity_unique_count": len(head_counts),
            "head_entity_client_count_min": int(min(head_counts)),
            "head_entity_client_count_mean": float(
                np.mean(head_counts)
            ),
            "head_entity_client_count_max": int(max(head_counts)),
            "head_entity_multi_client_fraction": float(
                np.mean(
                    [
                        1.0 if count > 1 else 0.0
                        for count in head_counts
                    ]
                )
            ),
            "entity_unique_count": len(entity_counts),
            "entity_client_count_min": int(min(entity_counts)),
            "entity_client_count_mean": float(
                np.mean(entity_counts)
            ),
            "entity_client_count_max": int(max(entity_counts)),
            "relation_unique_count": len(relation_counts),
            "relation_client_count_min": int(min(relation_counts)),
            "relation_client_count_mean": float(
                np.mean(relation_counts)
            ),
            "relation_client_count_max": int(max(relation_counts)),
        }

    @staticmethod
    def _relation_balance_statistics(
        partitions: Sequence[KnowledgeGraphClientPartition],
        num_relations: int,
    ) -> Dict[str, float]:
        """计算每个关系在客户端之间的三元组数量极差。"""

        per_client_counts = []
        for partition in partitions:
            counts = torch.bincount(
                partition.train_triples[:, 1].detach().cpu(),
                minlength=int(num_relations),
            )
            per_client_counts.append(counts.to(dtype=torch.float64))
        count_matrix = torch.stack(per_client_counts, dim=0)
        relation_imbalances = (
            count_matrix.max(dim=0).values
            - count_matrix.min(dim=0).values
        )
        return {
            "relation_triple_count_imbalance_mean": float(
                relation_imbalances.mean().item()
            ),
            "relation_triple_count_imbalance_max": float(
                relation_imbalances.max().item()
            ),
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
            "std_client_triple_count": float(np.std(triple_counts)),
            **self._overlap_statistics(self.partitions),
            **self._coverage_statistics(self.partitions),
            **self._relation_balance_statistics(
                self.partitions,
                self.dataset.num_relations,
            ),
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


def _build_partitions_from_row_indices(
    dataset: KnowledgeGraphDataset,
    client_row_indices: Sequence[Sequence[int]],
) -> Tuple[KnowledgeGraphClientPartition, ...]:
    """把每个客户端的训练集行号转换为完整分区对象。"""

    partitions = []
    for client_id, raw_row_indices in enumerate(client_row_indices):
        row_indices = sorted(int(value) for value in raw_row_indices)
        if not row_indices:
            raise ValueError(
                "关系分层后客户端{}没有训练三元组".format(client_id)
            )
        local_triples = dataset.train_triples.index_select(
            0, torch.tensor(row_indices, dtype=torch.long)
        )
        # 三类局部编号都从最终三元组重新计算，避免分配元数据漂移。
        local_heads = torch.unique(local_triples[:, 0], sorted=True)
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
                head_entity_ids=local_heads,
                entity_ids=local_entities,
                relation_ids=local_relations,
            )
        )
    return tuple(partitions)


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

    client_row_indices = [
        sorted(
            row_index
            for head_id in head_ids
            for row_index in head_to_row_indices[head_id]
        )
        for head_ids in client_heads
    ]
    partition_tuple = _build_partitions_from_row_indices(
        dataset, client_row_indices
    )
    return FederatedKnowledgeGraphData(
        dataset=dataset,
        partitions=partition_tuple,
        partition_strategy=BALANCED_HEAD_ENTITY,
        partition_seed=int(seed),
        partition_hash=_partition_hash(partition_tuple),
    )


def partition_train_triples_by_relation_stratified(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seed: int,
) -> FederatedKnowledgeGraphData:
    """按关系分层并同时均衡关系计数和总行数构造客户端分区。"""

    client_count = int(client_count)
    if client_count <= 0:
        raise ValueError("client_count必须大于0")
    train_triple_count = int(dataset.train_triples.shape[0])
    if client_count > train_triple_count:
        raise ValueError(
            "客户端数{}超过训练三元组数{}".format(
                client_count, train_triple_count
            )
        )

    relation_to_row_indices: Dict[int, List[int]] = {}
    for row_index, relation_id in enumerate(
        dataset.train_triples[:, 1].tolist()
    ):
        relation_to_row_indices.setdefault(
            int(relation_id), []
        ).append(int(row_index))

    rng = np.random.RandomState(int(seed))
    client_loads = [0 for _ in range(client_count)]
    client_row_indices: List[List[int]] = [
        [] for _ in range(client_count)
    ]
    for relation_id in sorted(relation_to_row_indices):
        shuffled_rows = np.asarray(
            relation_to_row_indices[relation_id],
            dtype=np.int64,
        )
        rng.shuffle(shuffled_rows)
        client_tie_order = [
            int(value) for value in rng.permutation(client_count)
        ]
        client_tie_rank = {
            client_id: rank
            for rank, client_id in enumerate(client_tie_order)
        }
        relation_client_loads = [
            0 for _ in range(client_count)
        ]
        for row_index in shuffled_rows.tolist():
            # 先保证当前关系均匀，再用总负载和种子顺序稳定打破平局。
            client_id = min(
                range(client_count),
                key=lambda value: (
                    relation_client_loads[value],
                    client_loads[value],
                    client_tie_rank[value],
                    value,
                ),
            )
            client_row_indices[client_id].append(int(row_index))
            relation_client_loads[client_id] += 1
            client_loads[client_id] += 1

    partition_tuple = _build_partitions_from_row_indices(
        dataset, client_row_indices
    )
    return FederatedKnowledgeGraphData(
        dataset=dataset,
        partitions=partition_tuple,
        partition_strategy=RELATION_STRATIFIED_TRIPLE_BALANCED,
        partition_seed=int(seed),
        partition_hash=_partition_hash(partition_tuple),
    )
