"""知识图谱客户端分区类型、互斥头实体和关系分层划分。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import KnowledgeGraphDataset


BALANCED_HEAD_ENTITY = "balanced_head_entity"
BALANCED_HEAD_ENTITY_OVERLAP_TARGET = (
    "balanced_head_entity_overlap_target"
)
SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED = (
    "semantic_domain_graph_local_balanced"
)
RELATION_STRATIFIED_TRIPLE_BALANCED = (
    "relation_stratified_triple_balanced"
)
SUPPORTED_PARTITION_STRATEGIES = frozenset(
    {
        BALANCED_HEAD_ENTITY,
        BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
        SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
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
    target_entity_overlap: Optional[float] = None
    overlap_tolerance: Optional[float] = None
    load_tolerance: Optional[float] = None
    relation_overlap_tolerance: Optional[float] = None
    reference_relation_overlap: Optional[float] = None
    search_restarts: Optional[int] = None
    search_seed: Optional[int] = None
    partition_metadata: Optional[Dict[str, object]] = None

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

        # 两种头实体策略都要求头实体互斥；关系分层策略有意允许重叠。
        if normalized_strategy in {
            BALANCED_HEAD_ENTITY,
            BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
        }:
            head_owners: Dict[int, int] = {}
            for partition in self.partitions:
                for head_id in partition.head_entity_ids.tolist():
                    head_id = int(head_id)
                    if head_id in head_owners:
                        raise ValueError(
                            "头实体{}被分配给多个客户端".format(head_id)
                        )
                    head_owners[head_id] = int(partition.client_id)
        if normalized_strategy == BALANCED_HEAD_ENTITY_OVERLAP_TARGET:
            required_values = {
                "target_entity_overlap": self.target_entity_overlap,
                "overlap_tolerance": self.overlap_tolerance,
                "load_tolerance": self.load_tolerance,
                "relation_overlap_tolerance": (
                    self.relation_overlap_tolerance
                ),
                "reference_relation_overlap": (
                    self.reference_relation_overlap
                ),
                "search_restarts": self.search_restarts,
                "search_seed": self.search_seed,
            }
            if any(value is None for value in required_values.values()):
                raise ValueError("目标重叠划分缺少完整的校准元数据")
            if not 0.0 <= float(self.target_entity_overlap) <= 1.0:
                raise ValueError("目标实体重叠率必须位于[0, 1]")
            if float(self.overlap_tolerance) < 0.0:
                raise ValueError("实体重叠率容差不能为负数")
            if not 0.0 <= float(self.load_tolerance) < 1.0:
                raise ValueError("负载容差必须位于[0, 1)")
            if float(self.relation_overlap_tolerance) < 0.0:
                raise ValueError("关系重叠率容差不能为负数")
            if not 0.0 <= float(self.reference_relation_overlap) <= 1.0:
                raise ValueError("参考关系重叠率必须位于[0, 1]")
            if int(self.search_restarts) <= 0:
                raise ValueError("目标重叠搜索重启次数必须大于0")
        if normalized_strategy == SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED:
            if not isinstance(self.partition_metadata, dict):
                raise ValueError("图语义划分缺少分区元数据")
            required_fields = (
                "domain_extractor",
                "client_primary_domains",
            )
            if any(
                field not in self.partition_metadata
                for field in required_fields
            ):
                raise ValueError("图语义划分元数据字段不完整")
            if len(
                self.partition_metadata["client_primary_domains"]
            ) != len(self.partitions):
                raise ValueError("客户端主域数量必须与客户端数量一致")

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
    def _coverage_map(
        partitions: Sequence[KnowledgeGraphClientPartition],
        field_name: str,
    ) -> Dict[int, int]:
        """返回每个知识编号覆盖的客户端数量。"""

        coverage: Dict[int, int] = {}
        for partition in partitions:
            values = getattr(partition, field_name)
            for value in values.tolist():
                normalized_value = int(value)
                coverage[normalized_value] = (
                    coverage.get(normalized_value, 0) + 1
                )
        return coverage

    @classmethod
    def _coverage_counts(
        cls,
        partitions: Sequence[KnowledgeGraphClientPartition],
        field_name: str,
    ) -> List[int]:
        """统计每个知识编号出现在多少个客户端的局部集合中。"""

        return list(
            cls._coverage_map(partitions, field_name).values()
        )

    @staticmethod
    def _normalized_overlap(
        coverage_counts: Sequence[int],
        client_count: int,
    ) -> float:
        """把知识编号的平均客户端覆盖数归一化到零至一。"""

        if not coverage_counts or int(client_count) <= 1:
            return 0.0
        replication_factor = float(np.mean(coverage_counts))
        return (
            (replication_factor - 1.0)
            / float(int(client_count) - 1)
        )

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
        entity_replication_factor = float(np.mean(entity_counts))
        relation_replication_factor = float(np.mean(relation_counts))
        client_count = len(partitions)
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
            "entity_client_count_median": float(
                np.median(entity_counts)
            ),
            "entity_client_count_p90": float(
                np.percentile(entity_counts, 90)
            ),
            "entity_replication_factor": entity_replication_factor,
            "entity_normalized_overlap": cls._normalized_overlap(
                entity_counts, client_count
            ),
            "entity_multi_client_fraction": float(
                np.mean(
                    [
                        1.0 if count > 1 else 0.0
                        for count in entity_counts
                    ]
                )
            ),
            "relation_unique_count": len(relation_counts),
            "relation_client_count_min": int(min(relation_counts)),
            "relation_client_count_mean": float(
                np.mean(relation_counts)
            ),
            "relation_client_count_max": int(max(relation_counts)),
            "relation_client_count_median": float(
                np.median(relation_counts)
            ),
            "relation_client_count_p90": float(
                np.percentile(relation_counts, 90)
            ),
            "relation_replication_factor": relation_replication_factor,
            "relation_normalized_overlap": cls._normalized_overlap(
                relation_counts, client_count
            ),
        }

    @classmethod
    def _boundary_statistics(
        cls,
        partitions: Sequence[KnowledgeGraphClientPartition],
    ) -> Dict[str, float]:
        """统计共享实体及其关联正训练三元组的比例。"""

        entity_coverage = cls._coverage_map(partitions, "entity_ids")
        shared_entities = {
            entity_id
            for entity_id, count in entity_coverage.items()
            if int(count) > 1
        }
        total_triples = 0
        shared_triples = 0
        entity_frequency: Dict[int, int] = {}
        for partition in partitions:
            for head, _, tail in partition.train_triples.tolist():
                normalized_head = int(head)
                normalized_tail = int(tail)
                total_triples += 1
                if (
                    normalized_head in shared_entities
                    or normalized_tail in shared_entities
                ):
                    shared_triples += 1
                entity_frequency[normalized_head] = (
                    entity_frequency.get(normalized_head, 0) + 1
                )
                entity_frequency[normalized_tail] = (
                    entity_frequency.get(normalized_tail, 0) + 1
                )
        client_count = len(partitions)
        weighted_denominator = float(sum(entity_frequency.values()))
        weighted_numerator = float(
            sum(
                entity_frequency[entity_id]
                * (int(entity_coverage[entity_id]) - 1)
                for entity_id in entity_frequency
            )
        )
        normalized_denominator = (
            weighted_denominator * float(max(client_count - 1, 1))
        )
        return {
            "shared_entity_count": len(shared_entities),
            "shared_entity_fraction": (
                float(len(shared_entities)) / float(len(entity_coverage))
                if entity_coverage
                else 0.0
            ),
            "shared_entity_triple_fraction": (
                float(shared_triples) / float(total_triples)
                if total_triples
                else 0.0
            ),
            "frequency_weighted_entity_normalized_overlap": (
                weighted_numerator / normalized_denominator
                if normalized_denominator > 0.0
                else 0.0
            ),
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
        mean_triple_count = float(np.mean(triple_counts))
        std_triple_count = float(np.std(triple_counts))
        max_relative_load_deviation = max(
            abs(float(count) - mean_triple_count) / mean_triple_count
            for count in triple_counts
        )
        summary: Dict[str, object] = {
            "dataset": self.dataset.dataset_name,
            "partition_strategy": self.partition_strategy,
            "partition_seed": int(self.partition_seed),
            "partition_hash": self.partition_hash,
            "client_count": self.client_count,
            "total_train_triple_count": int(sum(triple_counts)),
            "min_client_triple_count": int(min(triple_counts)),
            "max_client_triple_count": int(max(triple_counts)),
            "mean_client_triple_count": mean_triple_count,
            "std_client_triple_count": std_triple_count,
            "client_triple_count_cv": (
                std_triple_count / mean_triple_count
            ),
            "max_relative_load_deviation": float(
                max_relative_load_deviation
            ),
            **self._overlap_statistics(self.partitions),
            **self._coverage_statistics(self.partitions),
            **self._boundary_statistics(self.partitions),
            **self._relation_balance_statistics(
                self.partitions,
                self.dataset.num_relations,
            ),
            "clients": [
                partition.summary() for partition in self.partitions
            ],
        }
        if self.partition_strategy == BALANCED_HEAD_ENTITY_OVERLAP_TARGET:
            actual_entity_overlap = float(
                summary["entity_normalized_overlap"]
            )
            actual_relation_overlap = float(
                summary["relation_normalized_overlap"]
            )
            summary.update(
                {
                    "target_entity_overlap": float(
                        self.target_entity_overlap
                    ),
                    "entity_overlap_absolute_error": abs(
                        actual_entity_overlap
                        - float(self.target_entity_overlap)
                    ),
                    "overlap_tolerance": float(
                        self.overlap_tolerance
                    ),
                    "load_tolerance": float(self.load_tolerance),
                    "reference_relation_overlap": float(
                        self.reference_relation_overlap
                    ),
                    "relation_overlap_absolute_error": abs(
                        actual_relation_overlap
                        - float(self.reference_relation_overlap)
                    ),
                    "relation_overlap_tolerance": float(
                        self.relation_overlap_tolerance
                    ),
                    "search_restarts": int(self.search_restarts),
                    "search_seed": int(self.search_seed),
                }
            )
        if self.partition_metadata is not None:
            summary.update(dict(self.partition_metadata))
        return summary


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


def _build_head_bundle_maps(
    dataset: KnowledgeGraphDataset,
) -> Tuple[
    Dict[int, List[int]],
    Dict[int, set],
    Dict[int, set],
]:
    """按头实体构造行号、实体足迹和关系足迹三个映射。"""

    head_to_rows: Dict[int, List[int]] = {}
    head_to_entities: Dict[int, set] = {}
    head_to_relations: Dict[int, set] = {}
    for row_index, raw_row in enumerate(dataset.train_triples.tolist()):
        head_id, relation_id, tail_id = map(int, raw_row)
        head_to_rows.setdefault(head_id, []).append(int(row_index))
        head_to_entities.setdefault(head_id, set()).update(
            {head_id, tail_id}
        )
        head_to_relations.setdefault(head_id, set()).add(relation_id)
    return head_to_rows, head_to_entities, head_to_relations


def _maximum_relative_load_deviation(
    client_loads: Sequence[int],
) -> float:
    """计算客户端三元组数量相对全局均值的最大偏差。"""

    mean_load = float(np.mean(client_loads))
    if mean_load <= 0.0:
        return 0.0
    return max(
        abs(float(load) - mean_load) / mean_load
        for load in client_loads
    )


def _search_overlap_partition_candidate(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seed: int,
    target_entity_overlap: float,
    target_relation_overlap: float,
    load_tolerance: float,
) -> Tuple[Tuple[KnowledgeGraphClientPartition, ...], Dict[str, float]]:
    """执行一次确定性贪心搜索并返回目标重叠候选分区。"""

    (
        head_to_rows,
        head_to_entities,
        head_to_relations,
    ) = _build_head_bundle_maps(dataset)
    head_counts = {
        head_id: len(row_indices)
        for head_id, row_indices in head_to_rows.items()
    }
    if int(client_count) > len(head_counts):
        raise ValueError(
            "客户端数{}超过训练头实体数{}".format(
                client_count, len(head_counts)
            )
        )

    train_entities = set(
        int(value)
        for value in dataset.train_triples[:, (0, 2)]
        .reshape(-1)
        .tolist()
    )
    train_relations = set(
        int(value) for value in dataset.train_triples[:, 1].tolist()
    )
    target_entity_memberships = float(len(train_entities)) * (
        1.0
        + float(max(client_count - 1, 0))
        * float(target_entity_overlap)
    )
    target_relation_memberships = float(len(train_relations)) * (
        1.0
        + float(max(client_count - 1, 0))
        * float(target_relation_overlap)
    )
    entity_membership_scale = float(
        max(len(train_entities) * (client_count - 1), 1)
    )
    relation_membership_scale = float(
        max(len(train_relations) * (client_count - 1), 1)
    )
    mean_load = float(dataset.train_triples.shape[0]) / float(client_count)
    upper_load_limit = int(
        math.floor(mean_load * (1.0 + float(load_tolerance)))
    )
    balance_window = max(
        1,
        int(math.floor(mean_load * float(load_tolerance) / 2.0)),
    )

    client_loads = [0 for _ in range(client_count)]
    client_rows: List[List[int]] = [[] for _ in range(client_count)]
    client_entities = [set() for _ in range(client_count)]
    client_relations = [set() for _ in range(client_count)]
    entity_memberships = 0
    relation_memberships = 0
    rng = np.random.RandomState(int(seed))
    client_tie_order = [
        int(value) for value in rng.permutation(client_count)
    ]
    client_tie_rank = {
        client_id: rank
        for rank, client_id in enumerate(client_tie_order)
    }

    for head_id in _seeded_head_order(head_counts, seed):
        bundle_load = int(head_counts[head_id])
        capacity_clients = [
            client_id
            for client_id in range(client_count)
            if client_loads[client_id] + bundle_load
            <= upper_load_limit
        ]
        if not capacity_clients:
            # 极少数原子头包超过上限时仍给出可诊断候选，最终合同会拒绝。
            capacity_clients = list(range(client_count))
        minimum_load = min(
            client_loads[client_id]
            for client_id in capacity_clients
        )
        balanced_clients = [
            client_id
            for client_id in capacity_clients
            if client_loads[client_id]
            <= minimum_load + balance_window
        ]
        bundle_entities = head_to_entities[head_id]
        bundle_relations = head_to_relations[head_id]

        def candidate_key(
            client_id: int,
        ) -> Tuple[float, float, float, int, int]:
            """按实体目标、关系目标和负载顺序评价一个客户端。"""

            new_entity_count = len(
                bundle_entities.difference(client_entities[client_id])
            )
            new_relation_count = len(
                bundle_relations.difference(client_relations[client_id])
            )
            entity_error = abs(
                float(entity_memberships + new_entity_count)
                - target_entity_memberships
            )
            relation_error = abs(
                float(relation_memberships + new_relation_count)
                - target_relation_memberships
            )
            # 关系覆盖是硬合同，因此使用归一化联合误差防止其被实体目标吞没。
            joint_error = (
                entity_error / entity_membership_scale
                + relation_error / relation_membership_scale
            )
            return (
                joint_error,
                relation_error,
                entity_error,
                int(client_loads[client_id] + bundle_load),
                int(client_tie_rank[client_id]),
            )

        selected_client = min(balanced_clients, key=candidate_key)
        new_entities = bundle_entities.difference(
            client_entities[selected_client]
        )
        new_relations = bundle_relations.difference(
            client_relations[selected_client]
        )
        client_rows[selected_client].extend(head_to_rows[head_id])
        client_entities[selected_client].update(bundle_entities)
        client_relations[selected_client].update(bundle_relations)
        client_loads[selected_client] += bundle_load
        entity_memberships += len(new_entities)
        relation_memberships += len(new_relations)

    partition_tuple = _build_partitions_from_row_indices(
        dataset, client_rows
    )
    coverage = FederatedKnowledgeGraphData._coverage_statistics(
        partition_tuple
    )
    metrics = {
        "entity_normalized_overlap": float(
            coverage["entity_normalized_overlap"]
        ),
        "relation_normalized_overlap": float(
            coverage["relation_normalized_overlap"]
        ),
        "max_relative_load_deviation": float(
            _maximum_relative_load_deviation(client_loads)
        ),
    }
    return partition_tuple, metrics


def partition_train_triples_by_overlap_target(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seed: int,
    target_entity_overlap: float,
    overlap_tolerance: float = 0.005,
    load_tolerance: float = 0.05,
    relation_overlap_tolerance: float = 0.02,
    search_restarts: int = 8,
    search_seed: Optional[int] = None,
    strict: bool = True,
) -> FederatedKnowledgeGraphData:
    """在头实体互斥前提下搜索接近目标实体重叠率的分区。"""

    client_count = int(client_count)
    target_entity_overlap = float(target_entity_overlap)
    overlap_tolerance = float(overlap_tolerance)
    load_tolerance = float(load_tolerance)
    relation_overlap_tolerance = float(relation_overlap_tolerance)
    search_restarts = int(search_restarts)
    normalized_search_seed = (
        int(seed) if search_seed is None else int(search_seed)
    )
    if client_count <= 1:
        raise ValueError("目标重叠划分至少需要两个客户端")
    if not 0.0 <= target_entity_overlap <= 1.0:
        raise ValueError("目标实体重叠率必须位于[0, 1]")
    if overlap_tolerance < 0.0:
        raise ValueError("实体重叠率容差不能为负数")
    if not 0.0 <= load_tolerance < 1.0:
        raise ValueError("负载容差必须位于[0, 1)")
    if relation_overlap_tolerance < 0.0:
        raise ValueError("关系重叠率容差不能为负数")
    if search_restarts <= 0:
        raise ValueError("确定性搜索重启次数必须大于0")

    reference_partition = partition_train_triples_by_head(
        dataset=dataset,
        client_count=client_count,
        seed=int(seed),
    )
    reference_relation_overlap = float(
        reference_partition.summary()["relation_normalized_overlap"]
    )
    candidates = []
    for restart_index in range(search_restarts):
        # 使用互素步长派生稳定搜索种子，保证同一配置可完全复现。
        restart_seed = (
            normalized_search_seed + int(restart_index) * 104729
        )
        evaluated_guidance: Dict[float, float] = {}

        def evaluate_guidance(guidance_overlap: float) -> float:
            """评价一个内部引导值并记录其相对外部目标的合同误差。"""

            normalized_guidance = float(
                min(max(guidance_overlap, 0.0), 1.0)
            )
            cache_key = round(normalized_guidance, 12)
            if cache_key in evaluated_guidance:
                return evaluated_guidance[cache_key]
            partition_tuple, metrics = _search_overlap_partition_candidate(
                dataset=dataset,
                client_count=client_count,
                seed=restart_seed,
                target_entity_overlap=normalized_guidance,
                target_relation_overlap=reference_relation_overlap,
                load_tolerance=load_tolerance,
            )
            actual_overlap = float(
                metrics["entity_normalized_overlap"]
            )
            entity_error = abs(
                actual_overlap - target_entity_overlap
            )
            relation_error = abs(
                float(metrics["relation_normalized_overlap"])
                - reference_relation_overlap
            )
            load_error = float(
                metrics["max_relative_load_deviation"]
            )
            candidates.append(
                (
                    entity_error,
                    relation_error,
                    load_error,
                    _partition_hash(partition_tuple),
                    partition_tuple,
                )
            )
            evaluated_guidance[cache_key] = actual_overlap
            return actual_overlap

        evaluate_guidance(target_entity_overlap)
        if bool(strict):
            low_guidance = 0.0
            high_guidance = 1.0
            low_actual = evaluate_guidance(low_guidance)
            high_actual = evaluate_guidance(high_guidance)
            if low_actual > high_actual:
                low_guidance, high_guidance = (
                    high_guidance,
                    low_guidance,
                )
                low_actual, high_actual = high_actual, low_actual
            # 在可达端点之间反解内部引导值，外部目标仍是用户定义的ρ_E。
            if low_actual < target_entity_overlap < high_actual:
                for _ in range(8):
                    middle_guidance = (
                        low_guidance + high_guidance
                    ) / 2.0
                    middle_actual = evaluate_guidance(middle_guidance)
                    if middle_actual < target_entity_overlap:
                        low_guidance = middle_guidance
                        low_actual = middle_actual
                    else:
                        high_guidance = middle_guidance
                        high_actual = middle_actual

    feasible_candidates = [
        candidate
        for candidate in candidates
        if candidate[1] <= relation_overlap_tolerance + 1e-12
        and candidate[2] <= load_tolerance + 1e-12
    ]
    if not feasible_candidates:
        best_candidate = min(candidates, key=lambda value: value[:4])
        raise ValueError(
            "未找到同时满足负载和关系重叠合同的分区；"
            "最佳实体误差={:.6f}，关系误差={:.6f}，负载偏差={:.6f}".format(
                best_candidate[0],
                best_candidate[1],
                best_candidate[2],
            )
        )
    best_candidate = min(feasible_candidates, key=lambda value: value[:4])
    if bool(strict) and best_candidate[0] > overlap_tolerance + 1e-12:
        raise ValueError(
            "目标实体重叠率不可在当前容差内达到；"
            "目标={:.6f}，最佳误差={:.6f}".format(
                target_entity_overlap, best_candidate[0]
            )
        )
    partition_tuple = best_candidate[4]
    return FederatedKnowledgeGraphData(
        dataset=dataset,
        partitions=partition_tuple,
        partition_strategy=BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
        partition_seed=int(seed),
        partition_hash=_partition_hash(partition_tuple),
        target_entity_overlap=target_entity_overlap,
        overlap_tolerance=overlap_tolerance,
        load_tolerance=load_tolerance,
        relation_overlap_tolerance=relation_overlap_tolerance,
        reference_relation_overlap=reference_relation_overlap,
        search_restarts=search_restarts,
        search_seed=normalized_search_seed,
    )


def _compact_calibration_summary(
    federated_data: FederatedKnowledgeGraphData,
) -> Dict[str, object]:
    """提取校准报告需要的分区合同和指纹字段。"""

    summary = federated_data.summary()
    keys = (
        "partition_seed",
        "partition_hash",
        "entity_normalized_overlap",
        "relation_normalized_overlap",
        "reference_relation_overlap",
        "entity_overlap_absolute_error",
        "relation_overlap_absolute_error",
        "max_relative_load_deviation",
        "min_client_triple_count",
        "max_client_triple_count",
        "entity_replication_factor",
        "entity_jaccard_mean",
    )
    return {key: summary[key] for key in keys if key in summary}


def calibrate_entity_overlap_levels(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seeds: Sequence[int] = (42, 2024, 2025),
    overlap_tolerance: float = 0.005,
    load_tolerance: float = 0.05,
    relation_overlap_tolerance: float = 0.02,
    search_restarts: int = 8,
    minimum_overlap_span: float = 0.06,
) -> Dict[str, object]:
    """无训练地校准三个种子共同可达的低中高实体重叠档位。"""

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("校准至少需要一个随机种子")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("校准随机种子不能重复")
    minimum_overlap_span = float(minimum_overlap_span)
    if minimum_overlap_span < 0.0:
        raise ValueError("最小重叠跨度不能为负数")

    baselines: Dict[str, Dict[str, object]] = {}
    low_endpoints: Dict[int, float] = {}
    high_endpoints: Dict[int, float] = {}
    for seed in normalized_seeds:
        baseline = partition_train_triples_by_head(
            dataset=dataset,
            client_count=int(client_count),
            seed=seed,
        )
        baseline_summary = baseline.summary()
        baselines[str(seed)] = {
            "partition_seed": seed,
            "partition_hash": baseline.partition_hash,
            "entity_normalized_overlap": float(
                baseline_summary["entity_normalized_overlap"]
            ),
            "relation_normalized_overlap": float(
                baseline_summary["relation_normalized_overlap"]
            ),
            "max_relative_load_deviation": float(
                baseline_summary["max_relative_load_deviation"]
            ),
        }
        low_partition = partition_train_triples_by_overlap_target(
            dataset=dataset,
            client_count=int(client_count),
            seed=seed,
            target_entity_overlap=0.0,
            overlap_tolerance=float(overlap_tolerance),
            load_tolerance=float(load_tolerance),
            relation_overlap_tolerance=float(
                relation_overlap_tolerance
            ),
            search_restarts=int(search_restarts),
            strict=False,
        )
        high_partition = partition_train_triples_by_overlap_target(
            dataset=dataset,
            client_count=int(client_count),
            seed=seed,
            target_entity_overlap=1.0,
            overlap_tolerance=float(overlap_tolerance),
            load_tolerance=float(load_tolerance),
            relation_overlap_tolerance=float(
                relation_overlap_tolerance
            ),
            search_restarts=int(search_restarts),
            strict=False,
        )
        low_endpoints[seed] = float(
            low_partition.summary()["entity_normalized_overlap"]
        )
        high_endpoints[seed] = float(
            high_partition.summary()["entity_normalized_overlap"]
        )

    common_low = max(low_endpoints.values())
    common_high = min(high_endpoints.values())
    common_span = common_high - common_low
    if common_span + 1e-12 < minimum_overlap_span:
        raise ValueError(
            "三个种子的共同可达区间不足；"
            "低端={:.6f}，高端={:.6f}，跨度={:.6f}，要求至少={:.6f}".format(
                common_low,
                common_high,
                common_span,
                minimum_overlap_span,
            )
        )
    level_targets = {
        "low": common_low,
        "medium": (common_low + common_high) / 2.0,
        "high": common_high,
    }
    levels: Dict[str, Dict[str, object]] = {}
    for level_name, target_overlap in level_targets.items():
        per_seed: Dict[str, Dict[str, object]] = {}
        for seed in normalized_seeds:
            calibrated = partition_train_triples_by_overlap_target(
                dataset=dataset,
                client_count=int(client_count),
                seed=seed,
                target_entity_overlap=target_overlap,
                overlap_tolerance=float(overlap_tolerance),
                load_tolerance=float(load_tolerance),
                relation_overlap_tolerance=float(
                    relation_overlap_tolerance
                ),
                search_restarts=int(search_restarts),
                strict=True,
            )
            per_seed[str(seed)] = _compact_calibration_summary(
                calibrated
            )
        levels[level_name] = {
            "target_entity_overlap": float(target_overlap),
            "per_seed": per_seed,
        }

    return {
        "status": "passed",
        "dataset": dataset.dataset_name,
        "client_count": int(client_count),
        "seeds": list(normalized_seeds),
        "constraints": {
            "overlap_tolerance": float(overlap_tolerance),
            "load_tolerance": float(load_tolerance),
            "relation_overlap_tolerance": float(
                relation_overlap_tolerance
            ),
            "search_restarts": int(search_restarts),
            "minimum_overlap_span": minimum_overlap_span,
        },
        "endpoint_search": {
            "per_seed_low": {
                str(seed): value for seed, value in low_endpoints.items()
            },
            "per_seed_high": {
                str(seed): value for seed, value in high_endpoints.items()
            },
        },
        "common_reachable_interval": {
            "low": float(common_low),
            "high": float(common_high),
            "span": float(common_span),
        },
        "baselines": baselines,
        "levels": levels,
    }


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
