"""Freebase语义域与实体图局部性联合约束的客户端划分。"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

from .data import KnowledgeGraphDataset
from .federated_data import (
    SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
    FederatedKnowledgeGraphData,
    KnowledgeGraphClientPartition,
    _build_partitions_from_row_indices,
    _maximum_relative_load_deviation,
    _partition_hash,
)


DOMAIN_EXTRACTOR = "freebase_top_level"
_RESTART_PRIME = 104729


@dataclass(frozen=True)
class _SemanticPacket:
    """保存同一语义域和头实体对应的不可拆分三元组包。"""

    domain: str
    head_id: int
    row_indices: Tuple[int, ...]
    entity_ids: frozenset

    @property
    def triple_count(self) -> int:
        """返回当前不可拆分包包含的训练三元组数。"""

        return len(self.row_indices)


def freebase_relation_domain(relation_name: str) -> str:
    """从Freebase关系路径提取稳定的顶级语义域。"""

    normalized = str(relation_name).strip()
    parts = [part for part in normalized.strip("/").split("/") if part]
    if not parts:
        raise ValueError("Freebase关系名称不能为空")
    if parts[0] == "user":
        return "user"
    return parts[0]


def relation_domains(dataset: KnowledgeGraphDataset) -> Dict[int, str]:
    """返回关系编号到顶级语义域的完整映射。"""

    inverse = {
        int(relation_id): str(relation_name)
        for relation_name, relation_id in dataset.relation_to_id.items()
    }
    expected = set(range(dataset.num_relations))
    if set(inverse) != expected:
        raise ValueError("关系编号必须从0开始连续排列")
    return {
        relation_id: freebase_relation_domain(inverse[relation_id])
        for relation_id in sorted(inverse)
    }


def _build_semantic_packets(
    dataset: KnowledgeGraphDataset,
    domain_by_relation: Mapping[int, str],
) -> Tuple[_SemanticPacket, ...]:
    """按语义域和头实体构造不可拆分训练数据包。"""

    packet_rows: Dict[Tuple[str, int], List[int]] = {}
    packet_entities: Dict[Tuple[str, int], Set[int]] = {}
    for row_index, raw_row in enumerate(dataset.train_triples.tolist()):
        head_id, relation_id, tail_id = map(int, raw_row)
        domain = str(domain_by_relation[relation_id])
        key = (domain, head_id)
        packet_rows.setdefault(key, []).append(int(row_index))
        packet_entities.setdefault(key, set()).update({head_id, tail_id})
    return tuple(
        _SemanticPacket(
            domain=domain,
            head_id=int(head_id),
            row_indices=tuple(sorted(packet_rows[(domain, head_id)])),
            entity_ids=frozenset(packet_entities[(domain, head_id)]),
        )
        for domain, head_id in sorted(packet_rows)
    )


def _largest_remainder_quotas(
    domain_triple_counts: Mapping[str, int],
    client_count: int,
) -> Dict[str, int]:
    """按最大余数法把客户端主域配额分配给各语义域。"""

    total = int(sum(int(value) for value in domain_triple_counts.values()))
    if total <= 0:
        raise ValueError("语义域三元组总数必须大于0")
    client_count = int(client_count)
    exact = {
        domain: float(client_count) * float(count) / float(total)
        for domain, count in domain_triple_counts.items()
    }
    quotas = {domain: int(math.floor(value)) for domain, value in exact.items()}
    remaining = client_count - int(sum(quotas.values()))
    order = sorted(
        exact,
        key=lambda domain: (
            -(exact[domain] - float(quotas[domain])),
            -int(domain_triple_counts[domain]),
            domain,
        ),
    )
    for domain in order[:remaining]:
        quotas[domain] += 1
    if int(sum(quotas.values())) != client_count:
        raise RuntimeError("主域配额总数与客户端数量不一致")
    return quotas


def _primary_domain_sequence(
    quotas: Mapping[str, int],
    restart_seed: int,
) -> Tuple[str, ...]:
    """按配额生成经固定种子打乱的客户端主域序列。"""

    values = [
        domain
        for domain in sorted(quotas)
        for _ in range(int(quotas[domain]))
    ]
    rng = np.random.RandomState(int(restart_seed))
    if values:
        order = rng.permutation(len(values)).tolist()
        values = [values[int(index)] for index in order]
    return tuple(values)


def _packet_order(
    packets: Sequence[_SemanticPacket],
    restart_seed: int,
) -> List[int]:
    """按包大小降序并以固定种子稳定打破平局。"""

    rng = np.random.RandomState(int(restart_seed) + 17)
    tie_values = rng.random_sample(len(packets))
    return sorted(
        range(len(packets)),
        key=lambda index: (
            -packets[index].triple_count,
            float(tie_values[index]),
            packets[index].domain,
            packets[index].head_id,
        ),
    )


def _assignment_score(
    packet: _SemanticPacket,
    client_id: int,
    primary_domains: Sequence[str],
    client_entities: Sequence[Set[int]],
    client_loads: Sequence[int],
    tie_ranks: Sequence[int],
) -> Tuple[int, int, int, int]:
    """返回语义一致性、图增量、负载和稳定平局次序。"""

    return (
        0 if primary_domains[client_id] == packet.domain else 1,
        len(packet.entity_ids.difference(client_entities[client_id])),
        int(client_loads[client_id]),
        int(tie_ranks[client_id]),
    )


def _assign_packets_once(
    packets: Sequence[_SemanticPacket],
    client_count: int,
    load_tolerance: float,
    restart_seed: int,
    primary_domains: Sequence[str],
) -> List[List[int]]:
    """执行一次带硬上界的语义和图局部性贪心分配。"""

    total_triples = int(sum(packet.triple_count for packet in packets))
    mean_load = float(total_triples) / float(client_count)
    upper_limit = int(math.floor(mean_load * (1.0 + load_tolerance)))
    client_packet_indices: List[List[int]] = [
        [] for _ in range(client_count)
    ]
    client_loads = [0 for _ in range(client_count)]
    client_entities: List[Set[int]] = [set() for _ in range(client_count)]
    rng = np.random.RandomState(int(restart_seed) + 31)
    tie_order = [int(value) for value in rng.permutation(client_count)]
    tie_ranks = [0 for _ in range(client_count)]
    for rank, client_id in enumerate(tie_order):
        tie_ranks[client_id] = int(rank)

    for packet_index in _packet_order(packets, restart_seed):
        packet = packets[packet_index]
        feasible = [
            client_id
            for client_id in range(client_count)
            if client_loads[client_id] + packet.triple_count <= upper_limit
        ]
        candidates = feasible if feasible else list(range(client_count))
        client_id = min(
            candidates,
            key=lambda value: _assignment_score(
                packet,
                value,
                primary_domains,
                client_entities,
                client_loads,
                tie_ranks,
            ),
        )
        client_packet_indices[client_id].append(int(packet_index))
        client_loads[client_id] += packet.triple_count
        client_entities[client_id].update(packet.entity_ids)

    _repair_underloaded_clients(
        packets=packets,
        client_packet_indices=client_packet_indices,
        primary_domains=primary_domains,
        load_tolerance=load_tolerance,
        restart_seed=restart_seed,
    )
    return client_packet_indices


def _move_penalty(
    packet: _SemanticPacket,
    source_id: int,
    target_id: int,
    primary_domains: Sequence[str],
    target_entities: Set[int],
    tie_rank: int,
) -> Tuple[int, int, int]:
    """衡量负载修复移动对语义纯度和图局部性的损失。"""

    before = 0 if primary_domains[source_id] == packet.domain else 1
    after = 0 if primary_domains[target_id] == packet.domain else 1
    return (
        int(after - before),
        len(packet.entity_ids.difference(target_entities)),
        int(tie_rank),
    )


def _repair_underloaded_clients(
    packets: Sequence[_SemanticPacket],
    client_packet_indices: List[List[int]],
    primary_domains: Sequence[str],
    load_tolerance: float,
    restart_seed: int,
) -> None:
    """按低负载客户端批量移动数据包，避免重复全量扫描。"""

    client_count = len(client_packet_indices)
    total = int(sum(packet.triple_count for packet in packets))
    mean_load = float(total) / float(client_count)
    lower_limit = int(math.ceil(mean_load * (1.0 - load_tolerance)))
    upper_limit = int(math.floor(mean_load * (1.0 + load_tolerance)))
    loads = [
        int(sum(packets[index].triple_count for index in values))
        for values in client_packet_indices
    ]
    entities: List[Set[int]] = []
    for values in client_packet_indices:
        entity_ids: Set[int] = set()
        for packet_index in values:
            entity_ids.update(packets[packet_index].entity_ids)
        entities.append(entity_ids)
    rng = np.random.RandomState(int(restart_seed) + 47)
    packet_ties = {
        index: rank
        for rank, index in enumerate(
            [int(value) for value in rng.permutation(len(packets))]
        )
    }

    targets = sorted(
        range(client_count),
        key=lambda value: (loads[value], value),
    )
    for target_id in targets:
        if loads[target_id] >= lower_limit:
            continue
        # 对一个目标只构造一次候选表；移动可行性仍按最新负载复核。
        moves = []
        for source_id in range(client_count):
            if source_id == target_id:
                continue
            for packet_index in tuple(client_packet_indices[source_id]):
                packet = packets[packet_index]
                moves.append(
                    (
                        _move_penalty(
                            packet,
                            source_id,
                            target_id,
                            primary_domains,
                            entities[target_id],
                            packet_ties[packet_index],
                        ),
                        -packet.triple_count,
                        source_id,
                        packet_index,
                    )
                )
        for _, _, source_id, packet_index in sorted(moves):
            if loads[target_id] >= lower_limit:
                break
            if packet_index not in client_packet_indices[source_id]:
                continue
            packet = packets[packet_index]
            if loads[source_id] - packet.triple_count < lower_limit:
                continue
            if loads[target_id] + packet.triple_count > upper_limit:
                continue
            client_packet_indices[source_id].remove(packet_index)
            client_packet_indices[target_id].append(packet_index)
            loads[source_id] -= packet.triple_count
            loads[target_id] += packet.triple_count
            entities[target_id].update(packet.entity_ids)


def _row_indices_from_packets(
    packets: Sequence[_SemanticPacket],
    client_packet_indices: Sequence[Sequence[int]],
) -> List[List[int]]:
    """把客户端包编号展开为训练集行号。"""

    return [
        sorted(
            row_index
            for packet_index in packet_indices
            for row_index in packets[int(packet_index)].row_indices
        )
        for packet_indices in client_packet_indices
    ]


def _largest_component_fraction(triples: torch.Tensor) -> float:
    """计算客户端无向实体图最大连通分量的实体占比。"""

    parents: Dict[int, int] = {}

    def find(value: int) -> int:
        """查找并压缩并查集中的实体根节点。"""

        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    def union(left: int, right: int) -> None:
        """合并一条三元组边两端实体所在的集合。"""

        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for head_id, _, tail_id in triples.tolist():
        union(int(head_id), int(tail_id))
    if not parents:
        return 0.0
    sizes = Counter(find(value) for value in parents)
    return float(max(sizes.values())) / float(len(parents))


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个离散分布的Jensen-Shannon散度。"""

    middle = 0.5 * (left + right)

    def kl(values: np.ndarray, reference: np.ndarray) -> float:
        """计算忽略零概率项的KL散度。"""

        mask = values > 0.0
        return float(
            np.sum(values[mask] * np.log(values[mask] / reference[mask]))
        )

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def semantic_graph_statistics(
    dataset: KnowledgeGraphDataset,
    partitions: Sequence[KnowledgeGraphClientPartition],
    primary_domains: Sequence[str],
) -> Dict[str, object]:
    """汇总语义纯度、关系异构性和局部图结构指标。"""

    if len(partitions) != len(primary_domains):
        raise ValueError("主域数量必须与客户端分区数量一致")
    domain_by_relation = relation_domains(dataset)
    global_relation_counts = torch.bincount(
        dataset.train_triples[:, 1],
        minlength=dataset.num_relations,
    ).numpy().astype(np.float64)
    global_relation_distribution = (
        global_relation_counts / float(global_relation_counts.sum())
    )
    client_rows = []
    total_dominant = 0
    total_primary = 0
    entropy_values = []
    js_values = []
    lcc_values = []
    local_entity_occurrences = 0
    total_triples = 0
    for partition, primary_domain in zip(partitions, primary_domains):
        domains = [
            domain_by_relation[int(relation_id)]
            for relation_id in partition.train_triples[:, 1].tolist()
        ]
        counts = Counter(domains)
        triple_count = partition.triple_count
        dominant_domain, dominant_count = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        primary_count = int(counts.get(str(primary_domain), 0))
        probabilities = np.asarray(
            [float(value) / float(triple_count) for value in counts.values()],
            dtype=np.float64,
        )
        entropy = (
            -float(np.sum(probabilities * np.log(probabilities)))
            / float(math.log(len(counts)))
            if len(counts) > 1
            else 0.0
        )
        relation_counts = torch.bincount(
            partition.train_triples[:, 1],
            minlength=dataset.num_relations,
        ).numpy().astype(np.float64)
        relation_distribution = relation_counts / float(relation_counts.sum())
        js_value = _js_divergence(
            relation_distribution,
            global_relation_distribution,
        )
        lcc_fraction = _largest_component_fraction(partition.train_triples)
        total_dominant += int(dominant_count)
        total_primary += primary_count
        total_triples += triple_count
        local_entity_occurrences += int(partition.entity_ids.numel())
        entropy_values.append(entropy)
        js_values.append(js_value)
        lcc_values.append(lcc_fraction)
        client_rows.append(
            {
                "client_id": int(partition.client_id),
                "primary_domain": str(primary_domain),
                "dominant_domain": str(dominant_domain),
                "dominant_domain_purity": (
                    float(dominant_count) / float(triple_count)
                ),
                "primary_domain_fraction": (
                    float(primary_count) / float(triple_count)
                ),
                "domain_count": len(counts),
                "normalized_domain_entropy": entropy,
                "relation_js_divergence": js_value,
                "local_entity_reuse_ratio": (
                    2.0 * float(triple_count)
                    / float(partition.entity_ids.numel())
                ),
                "largest_component_entity_fraction": lcc_fraction,
            }
        )
    return {
        "semantic_domain_count": len(set(domain_by_relation.values())),
        "triple_weighted_dominant_domain_purity": (
            float(total_dominant) / float(total_triples)
        ),
        "triple_weighted_primary_domain_fraction": (
            float(total_primary) / float(total_triples)
        ),
        "mean_normalized_domain_entropy": float(np.mean(entropy_values)),
        "mean_relation_js_divergence": float(np.mean(js_values)),
        "local_entity_reuse_ratio": (
            2.0 * float(total_triples)
            / float(local_entity_occurrences)
        ),
        "mean_largest_component_entity_fraction": float(np.mean(lcc_values)),
        "semantic_clients": client_rows,
    }


def _partition_candidate(
    dataset: KnowledgeGraphDataset,
    packets: Sequence[_SemanticPacket],
    primary_domains: Sequence[str],
    client_packet_indices: Sequence[Sequence[int]],
    seed: int,
    load_tolerance: float,
    search_restarts: int,
) -> FederatedKnowledgeGraphData:
    """把一次包分配转换为带图语义统计的正式分区候选。"""

    row_indices = _row_indices_from_packets(packets, client_packet_indices)
    partitions = _build_partitions_from_row_indices(dataset, row_indices)
    metadata = semantic_graph_statistics(
        dataset,
        partitions,
        primary_domains,
    )
    metadata.update(
        {
            "domain_extractor": DOMAIN_EXTRACTOR,
            "domain_head_packet_count": len(packets),
            "client_primary_domains": list(primary_domains),
            "load_tolerance": float(load_tolerance),
            "search_restarts": int(search_restarts),
            "search_seed": int(seed),
        }
    )
    return FederatedKnowledgeGraphData(
        dataset=dataset,
        partitions=partitions,
        partition_strategy=SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
        partition_seed=int(seed),
        partition_hash=_partition_hash(partitions),
        load_tolerance=float(load_tolerance),
        search_restarts=int(search_restarts),
        search_seed=int(seed),
        partition_metadata=metadata,
    )


def _candidate_score(
    candidate: FederatedKnowledgeGraphData,
    load_tolerance: float,
) -> Tuple[int, float, float, float, str]:
    """按预注册顺序计算图语义分区候选的选择元组。"""

    summary = candidate.summary()
    max_deviation = float(summary["max_relative_load_deviation"])
    violations = int(max_deviation > float(load_tolerance) + 1e-12)
    violations += int(
        any(partition.triple_count <= 0 for partition in candidate.partitions)
    )
    return (
        violations,
        max_deviation,
        1.0 - float(summary["triple_weighted_primary_domain_fraction"]),
        float(summary["entity_replication_factor"]),
        candidate.partition_hash,
    )


def _raw_assignment_score(
    dataset: KnowledgeGraphDataset,
    packets: Sequence[_SemanticPacket],
    primary_domains: Sequence[str],
    assignments: Sequence[Sequence[int]],
    load_tolerance: float,
) -> Tuple[int, float, float, float, str]:
    """不构造完整摘要地计算一次重启的候选选择元组。"""

    loads = []
    primary_triples = 0
    entity_occurrences = 0
    digest_parts = []
    for client_id, packet_indices in enumerate(assignments):
        row_indices = sorted(
            row_index
            for packet_index in packet_indices
            for row_index in packets[int(packet_index)].row_indices
        )
        loads.append(len(row_indices))
        entities: Set[int] = set()
        for packet_index in packet_indices:
            packet = packets[int(packet_index)]
            entities.update(packet.entity_ids)
            if packet.domain == primary_domains[client_id]:
                primary_triples += packet.triple_count
        entity_occurrences += len(entities)
        digest_parts.append((client_id, row_indices))
    max_deviation = _maximum_relative_load_deviation(loads)
    violations = int(max_deviation > float(load_tolerance) + 1e-12)
    violations += int(any(load <= 0 for load in loads))
    digest = hashlib.sha256()
    for client_id, row_indices in digest_parts:
        digest.update(
            int(client_id).to_bytes(8, byteorder="little", signed=False)
        )
        local = dataset.train_triples.index_select(
            0, torch.tensor(row_indices, dtype=torch.long)
        ).contiguous()
        digest.update(local.numpy().tobytes())
    return (
        violations,
        float(max_deviation),
        1.0 - float(primary_triples) / float(sum(loads)),
        float(entity_occurrences) / float(dataset.num_entities),
        digest.hexdigest(),
    )


def partition_train_triples_by_semantic_domain_graph_local(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seed: int,
    load_tolerance: float = 0.05,
    search_restarts: int = 8,
) -> FederatedKnowledgeGraphData:
    """构造语义域主导、实体图局部且负载均衡的客户端分区。"""

    client_count = int(client_count)
    seed = int(seed)
    load_tolerance = float(load_tolerance)
    search_restarts = int(search_restarts)
    if client_count <= 0:
        raise ValueError("client_count必须大于0")
    if not 0.0 <= load_tolerance < 1.0:
        raise ValueError("load_tolerance必须位于[0, 1)")
    if search_restarts <= 0:
        raise ValueError("search_restarts必须大于0")
    domains = relation_domains(dataset)
    packets = _build_semantic_packets(dataset, domains)
    if client_count > len(packets):
        raise ValueError("客户端数量不能超过域-头实体包数量")
    domain_counts = Counter()
    for packet in packets:
        domain_counts[packet.domain] += packet.triple_count
    quotas = _largest_remainder_quotas(domain_counts, client_count)
    raw_candidates = []
    for restart_index in range(search_restarts):
        restart_seed = seed + restart_index * _RESTART_PRIME
        primary_domains = _primary_domain_sequence(quotas, restart_seed)
        assignments = _assign_packets_once(
            packets=packets,
            client_count=client_count,
            load_tolerance=load_tolerance,
            restart_seed=restart_seed,
            primary_domains=primary_domains,
        )
        score = _raw_assignment_score(
            dataset,
            packets,
            primary_domains,
            assignments,
            load_tolerance,
        )
        raw_candidates.append((score, primary_domains, assignments))
    score, primary_domains, assignments = min(
        raw_candidates,
        key=lambda value: value[0],
    )
    if int(score[0]) != 0:
        raise RuntimeError(
            "图语义划分未满足负载合同：最大偏差{:.6f}，容差{:.6f}".format(
                float(score[1]),
                load_tolerance,
            )
        )
    best = _partition_candidate(
        dataset=dataset,
        packets=packets,
        primary_domains=primary_domains,
        client_packet_indices=assignments,
        seed=seed,
        load_tolerance=load_tolerance,
        search_restarts=search_restarts,
    )
    if best.partition_hash != str(score[4]):
        raise RuntimeError("图语义候选快速哈希与正式分区哈希不一致")
    return best
