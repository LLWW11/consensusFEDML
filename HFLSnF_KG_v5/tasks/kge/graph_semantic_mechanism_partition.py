"""实现语义主域与实体图局部性的两个隔离消融划分。"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .data import KnowledgeGraphDataset
from .federated_data import (
    DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
    SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
    FederatedKnowledgeGraphData,
    _build_partitions_from_row_indices,
    _maximum_relative_load_deviation,
    _partition_hash,
)
from .graph_semantic_partition import (
    DOMAIN_EXTRACTOR,
    _RESTART_PRIME,
    _SemanticPacket,
    _build_semantic_packets,
    _largest_remainder_quotas,
    _packet_order,
    _primary_domain_sequence,
    _row_indices_from_packets,
    relation_domains,
    semantic_graph_statistics,
)


PACKET_UNIT = "semantic_domain_head_entity"
GRAPH_ONLY_OBJECTIVES = (
    "new_entity_count",
    "current_load",
    "seeded_tie_rank",
)
SEMANTIC_ONLY_OBJECTIVES = (
    "primary_domain_mismatch",
    "current_load",
    "seeded_tie_rank",
)


def _validate_partition_arguments(
    client_count: int,
    load_tolerance: float,
    search_restarts: int,
) -> Tuple[int, float, int]:
    """规范化并校验两个机制消融共享的划分参数。"""

    normalized_client_count = int(client_count)
    normalized_load_tolerance = float(load_tolerance)
    normalized_search_restarts = int(search_restarts)
    if normalized_client_count <= 0:
        raise ValueError("client_count必须大于0")
    if not 0.0 <= normalized_load_tolerance < 1.0:
        raise ValueError("load_tolerance必须位于[0, 1)")
    if normalized_search_restarts <= 0:
        raise ValueError("search_restarts必须大于0")
    return (
        normalized_client_count,
        normalized_load_tolerance,
        normalized_search_restarts,
    )


def _load_limits(
    packets: Sequence[_SemanticPacket],
    client_count: int,
    load_tolerance: float,
) -> Tuple[int, int]:
    """返回负载合同对应的客户端三元组下界和上界。"""

    total = int(sum(packet.triple_count for packet in packets))
    mean_load = float(total) / float(client_count)
    return (
        int(math.ceil(mean_load * (1.0 - load_tolerance))),
        int(math.floor(mean_load * (1.0 + load_tolerance))),
    )


def _client_tie_ranks(client_count: int, restart_seed: int) -> List[int]:
    """生成固定种子的客户端稳定平局名次。"""

    rng = np.random.RandomState(int(restart_seed) + 31)
    tie_order = [int(value) for value in rng.permutation(client_count)]
    tie_ranks = [0 for _ in range(client_count)]
    for rank, client_id in enumerate(tie_order):
        tie_ranks[client_id] = int(rank)
    return tie_ranks


def _graph_only_packet_order(
    packets: Sequence[_SemanticPacket],
    restart_seed: int,
) -> List[int]:
    """按包大小和固定随机数排序且完全不读取语义域标签。"""

    rng = np.random.RandomState(int(restart_seed) + 17)
    tie_values = rng.random_sample(len(packets))
    return sorted(
        range(len(packets)),
        key=lambda index: (
            -packets[index].triple_count,
            float(tie_values[index]),
            packets[index].head_id,
            packets[index].row_indices,
        ),
    )


def _assignment_loads(
    packets: Sequence[_SemanticPacket],
    assignments: Sequence[Sequence[int]],
) -> List[int]:
    """计算每个客户端当前持有的训练三元组数量。"""

    return [
        int(sum(packets[index].triple_count for index in packet_indices))
        for packet_indices in assignments
    ]


def _assignment_entities(
    packets: Sequence[_SemanticPacket],
    assignments: Sequence[Sequence[int]],
) -> List[Set[int]]:
    """计算每个客户端当前包集合覆盖的实体集合。"""

    values: List[Set[int]] = []
    for packet_indices in assignments:
        entity_ids: Set[int] = set()
        for packet_index in packet_indices:
            entity_ids.update(packets[int(packet_index)].entity_ids)
        values.append(entity_ids)
    return values


def _packet_tie_ranks(
    packet_count: int,
    restart_seed: int,
) -> Dict[int, int]:
    """生成低负载修复阶段使用的稳定数据包名次。"""

    rng = np.random.RandomState(int(restart_seed) + 47)
    return {
        index: rank
        for rank, index in enumerate(
            [int(value) for value in rng.permutation(packet_count)]
        )
    }


def _repair_graph_only_underload(
    packets: Sequence[_SemanticPacket],
    assignments: List[List[int]],
    load_tolerance: float,
    restart_seed: int,
) -> None:
    """仅按目标端实体增量修复图局部消融臂的低负载客户端。"""

    lower_limit, upper_limit = _load_limits(
        packets,
        len(assignments),
        load_tolerance,
    )
    loads = _assignment_loads(packets, assignments)
    entities = _assignment_entities(packets, assignments)
    packet_ties = _packet_tie_ranks(len(packets), restart_seed)
    targets = sorted(range(len(assignments)), key=lambda value: (loads[value], value))
    for target_id in targets:
        if loads[target_id] >= lower_limit:
            continue
        moves = []
        for source_id in range(len(assignments)):
            if source_id == target_id:
                continue
            for packet_index in tuple(assignments[source_id]):
                packet = packets[packet_index]
                moves.append(
                    (
                        len(packet.entity_ids.difference(entities[target_id])),
                        packet_ties[packet_index],
                        -packet.triple_count,
                        source_id,
                        packet_index,
                    )
                )
        for _, _, _, source_id, packet_index in sorted(moves):
            if loads[target_id] >= lower_limit:
                break
            if packet_index not in assignments[source_id]:
                continue
            packet = packets[packet_index]
            if loads[source_id] - packet.triple_count < lower_limit:
                continue
            if loads[target_id] + packet.triple_count > upper_limit:
                continue
            assignments[source_id].remove(packet_index)
            assignments[target_id].append(packet_index)
            loads[source_id] -= packet.triple_count
            loads[target_id] += packet.triple_count
            entities[target_id].update(packet.entity_ids)
            # 源客户端后续也可能成为修复目标，必须移除已经迁出的实体影响。
            entities[source_id] = _assignment_entities(
                packets,
                [assignments[source_id]],
            )[0]


def _repair_semantic_only_underload(
    packets: Sequence[_SemanticPacket],
    assignments: List[List[int]],
    primary_domains: Sequence[str],
    load_tolerance: float,
    restart_seed: int,
) -> None:
    """只按主域一致性修复语义消融臂的低负载客户端。"""

    lower_limit, upper_limit = _load_limits(
        packets,
        len(assignments),
        load_tolerance,
    )
    loads = _assignment_loads(packets, assignments)
    packet_ties = _packet_tie_ranks(len(packets), restart_seed)
    targets = sorted(range(len(assignments)), key=lambda value: (loads[value], value))
    for target_id in targets:
        if loads[target_id] >= lower_limit:
            continue
        moves = []
        for source_id in range(len(assignments)):
            if source_id == target_id:
                continue
            for packet_index in tuple(assignments[source_id]):
                packet = packets[packet_index]
                before = int(primary_domains[source_id] != packet.domain)
                after = int(primary_domains[target_id] != packet.domain)
                moves.append(
                    (
                        after - before,
                        packet_ties[packet_index],
                        -packet.triple_count,
                        source_id,
                        packet_index,
                    )
                )
        for _, _, _, source_id, packet_index in sorted(moves):
            if loads[target_id] >= lower_limit:
                break
            if packet_index not in assignments[source_id]:
                continue
            packet = packets[packet_index]
            if loads[source_id] - packet.triple_count < lower_limit:
                continue
            if loads[target_id] + packet.triple_count > upper_limit:
                continue
            assignments[source_id].remove(packet_index)
            assignments[target_id].append(packet_index)
            loads[source_id] -= packet.triple_count
            loads[target_id] += packet.triple_count


def _assign_graph_only_once(
    packets: Sequence[_SemanticPacket],
    client_count: int,
    load_tolerance: float,
    restart_seed: int,
) -> List[List[int]]:
    """执行一次不读取语义域标签的实体图局部贪心分配。"""

    _, upper_limit = _load_limits(packets, client_count, load_tolerance)
    assignments: List[List[int]] = [[] for _ in range(client_count)]
    loads = [0 for _ in range(client_count)]
    entities: List[Set[int]] = [set() for _ in range(client_count)]
    tie_ranks = _client_tie_ranks(client_count, restart_seed)
    for packet_index in _graph_only_packet_order(packets, restart_seed):
        packet = packets[packet_index]
        feasible = [
            client_id
            for client_id in range(client_count)
            if loads[client_id] + packet.triple_count <= upper_limit
        ]
        candidates = feasible if feasible else list(range(client_count))
        client_id = min(
            candidates,
            key=lambda value: (
                len(packet.entity_ids.difference(entities[value])),
                loads[value],
                tie_ranks[value],
            ),
        )
        assignments[client_id].append(int(packet_index))
        loads[client_id] += packet.triple_count
        entities[client_id].update(packet.entity_ids)
    _repair_graph_only_underload(
        packets,
        assignments,
        load_tolerance,
        restart_seed,
    )
    return assignments


def _assign_semantic_only_once(
    packets: Sequence[_SemanticPacket],
    client_count: int,
    load_tolerance: float,
    restart_seed: int,
    primary_domains: Sequence[str],
) -> List[List[int]]:
    """执行一次完全不读取实体集合的主域优先贪心分配。"""

    _, upper_limit = _load_limits(packets, client_count, load_tolerance)
    assignments: List[List[int]] = [[] for _ in range(client_count)]
    loads = [0 for _ in range(client_count)]
    tie_ranks = _client_tie_ranks(client_count, restart_seed)
    for packet_index in _packet_order(packets, restart_seed):
        packet = packets[packet_index]
        feasible = [
            client_id
            for client_id in range(client_count)
            if loads[client_id] + packet.triple_count <= upper_limit
        ]
        candidates = feasible if feasible else list(range(client_count))
        client_id = min(
            candidates,
            key=lambda value: (
                int(primary_domains[value] != packet.domain),
                loads[value],
                tie_ranks[value],
            ),
        )
        assignments[client_id].append(int(packet_index))
        loads[client_id] += packet.triple_count
    _repair_semantic_only_underload(
        packets,
        assignments,
        primary_domains,
        load_tolerance,
        restart_seed,
    )
    return assignments


def _assignment_digest(
    dataset: KnowledgeGraphDataset,
    packets: Sequence[_SemanticPacket],
    assignments: Sequence[Sequence[int]],
) -> Tuple[List[int], str]:
    """返回候选负载列表和与正式分区一致的快速SHA-256哈希。"""

    loads = []
    digest = hashlib.sha256()
    for client_id, packet_indices in enumerate(assignments):
        row_indices = sorted(
            row_index
            for packet_index in packet_indices
            for row_index in packets[int(packet_index)].row_indices
        )
        loads.append(len(row_indices))
        digest.update(
            int(client_id).to_bytes(8, byteorder="little", signed=False)
        )
        local = dataset.train_triples.index_select(
            0,
            torch.tensor(row_indices, dtype=torch.long),
        ).contiguous()
        digest.update(local.numpy().tobytes())
    return loads, digest.hexdigest()


def _graph_only_raw_score(
    dataset: KnowledgeGraphDataset,
    packets: Sequence[_SemanticPacket],
    assignments: Sequence[Sequence[int]],
    load_tolerance: float,
) -> Tuple[int, float, float, str]:
    """按负载、实体复制和哈希计算消融A候选排序元组。"""

    loads, digest = _assignment_digest(dataset, packets, assignments)
    max_deviation = _maximum_relative_load_deviation(loads)
    violations = int(max_deviation > float(load_tolerance) + 1e-12)
    violations += int(any(load <= 0 for load in loads))
    entity_occurrences = sum(
        len(values) for values in _assignment_entities(packets, assignments)
    )
    return (
        violations,
        float(max_deviation),
        float(entity_occurrences) / float(dataset.num_entities),
        digest,
    )


def _semantic_only_raw_score(
    dataset: KnowledgeGraphDataset,
    packets: Sequence[_SemanticPacket],
    assignments: Sequence[Sequence[int]],
    primary_domains: Sequence[str],
    load_tolerance: float,
) -> Tuple[int, float, float, str]:
    """按负载、主域一致性和哈希计算消融B候选排序元组。"""

    loads, digest = _assignment_digest(dataset, packets, assignments)
    max_deviation = _maximum_relative_load_deviation(loads)
    violations = int(max_deviation > float(load_tolerance) + 1e-12)
    violations += int(any(load <= 0 for load in loads))
    primary_triples = 0
    for client_id, packet_indices in enumerate(assignments):
        for packet_index in packet_indices:
            packet = packets[int(packet_index)]
            if packet.domain == primary_domains[client_id]:
                primary_triples += packet.triple_count
    return (
        violations,
        float(max_deviation),
        1.0 - float(primary_triples) / float(sum(loads)),
        digest,
    )


def _observed_dominant_domains(
    dataset: KnowledgeGraphDataset,
    partitions: Sequence[object],
) -> Tuple[str, ...]:
    """仅为事后统计计算各客户端实际占比最高的语义域。"""

    domain_by_relation = relation_domains(dataset)
    values = []
    for partition in partitions:
        counts = Counter(
            domain_by_relation[int(relation_id)]
            for relation_id in partition.train_triples[:, 1].tolist()
        )
        values.append(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        )
    return tuple(values)


def _statistics_without_primary_domains(
    dataset: KnowledgeGraphDataset,
    partitions: Sequence[object],
) -> Dict[str, object]:
    """生成消融A统计并把所有预设主域字段明确置为null。"""

    observed_domains = _observed_dominant_domains(dataset, partitions)
    statistics = semantic_graph_statistics(
        dataset,
        partitions,
        observed_domains,
    )
    statistics["triple_weighted_primary_domain_fraction"] = None
    for client in statistics["semantic_clients"]:
        client["primary_domain"] = None
        client["primary_domain_fraction"] = None
    return statistics


def _build_ablation_candidate(
    dataset: KnowledgeGraphDataset,
    packets: Sequence[_SemanticPacket],
    assignments: Sequence[Sequence[int]],
    strategy: str,
    seed: int,
    load_tolerance: float,
    search_restarts: int,
    primary_domains: Optional[Sequence[str]],
    primary_domain_quotas: Optional[Mapping[str, int]],
) -> FederatedKnowledgeGraphData:
    """把一次机制消融包分配转换为正式联邦分区。"""

    row_indices = _row_indices_from_packets(packets, assignments)
    partitions = _build_partitions_from_row_indices(dataset, row_indices)
    if primary_domains is None:
        metadata = _statistics_without_primary_domains(dataset, partitions)
        has_primary_domain = False
        uses_entity_locality = True
        objective_order = GRAPH_ONLY_OBJECTIVES
        arm = "graph_only"
    else:
        metadata = semantic_graph_statistics(
            dataset,
            partitions,
            primary_domains,
        )
        has_primary_domain = True
        uses_entity_locality = False
        objective_order = SEMANTIC_ONLY_OBJECTIVES
        arm = "semantic_only"
    metadata.update(
        {
            "ablation_arm": arm,
            "domain_extractor": DOMAIN_EXTRACTOR,
            "domain_head_packet_count": len(packets),
            "packet_unit": PACKET_UNIT,
            "has_primary_domain": has_primary_domain,
            "uses_entity_locality_objective": uses_entity_locality,
            "assignment_objective_order": list(objective_order),
            "client_primary_domains": (
                None
                if primary_domains is None
                else list(primary_domains)
            ),
            "primary_domain_quotas": (
                None
                if primary_domain_quotas is None
                else {
                    str(domain): int(value)
                    for domain, value in sorted(primary_domain_quotas.items())
                }
            ),
            "load_tolerance": float(load_tolerance),
            "search_restarts": int(search_restarts),
            "search_seed": int(seed),
        }
    )
    return FederatedKnowledgeGraphData(
        dataset=dataset,
        partitions=partitions,
        partition_strategy=strategy,
        partition_seed=int(seed),
        partition_hash=_partition_hash(partitions),
        load_tolerance=float(load_tolerance),
        search_restarts=int(search_restarts),
        search_seed=int(seed),
        partition_metadata=metadata,
    )


def partition_train_triples_by_graph_local_no_primary(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seed: int,
    load_tolerance: float = 0.05,
    search_restarts: int = 8,
) -> FederatedKnowledgeGraphData:
    """构造无预设主域、保留实体图局部目标的消融A分区。"""

    client_count, load_tolerance, search_restarts = (
        _validate_partition_arguments(
            client_count,
            load_tolerance,
            search_restarts,
        )
    )
    seed = int(seed)
    packets = _build_semantic_packets(dataset, relation_domains(dataset))
    if client_count > len(packets):
        raise ValueError("客户端数量不能超过域-头实体包数量")
    candidates = []
    for restart_index in range(search_restarts):
        restart_seed = seed + restart_index * _RESTART_PRIME
        assignments = _assign_graph_only_once(
            packets,
            client_count,
            load_tolerance,
            restart_seed,
        )
        score = _graph_only_raw_score(
            dataset,
            packets,
            assignments,
            load_tolerance,
        )
        candidates.append((score, assignments))
    score, assignments = min(candidates, key=lambda value: value[0])
    if int(score[0]) != 0:
        raise RuntimeError(
            "消融A未满足负载合同：最大偏差{:.6f}，容差{:.6f}".format(
                float(score[1]),
                load_tolerance,
            )
        )
    candidate = _build_ablation_candidate(
        dataset=dataset,
        packets=packets,
        assignments=assignments,
        strategy=DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
        seed=seed,
        load_tolerance=load_tolerance,
        search_restarts=search_restarts,
        primary_domains=None,
        primary_domain_quotas=None,
    )
    if candidate.partition_hash != str(score[3]):
        raise RuntimeError("消融A快速哈希与正式分区哈希不一致")
    return candidate


def partition_train_triples_by_semantic_domain_no_graph_local(
    dataset: KnowledgeGraphDataset,
    client_count: int,
    seed: int,
    load_tolerance: float = 0.05,
    search_restarts: int = 8,
) -> FederatedKnowledgeGraphData:
    """构造保留语义主域、删除实体图局部目标的消融B分区。"""

    client_count, load_tolerance, search_restarts = (
        _validate_partition_arguments(
            client_count,
            load_tolerance,
            search_restarts,
        )
    )
    seed = int(seed)
    packets = _build_semantic_packets(dataset, relation_domains(dataset))
    if client_count > len(packets):
        raise ValueError("客户端数量不能超过域-头实体包数量")
    domain_counts = Counter()
    for packet in packets:
        domain_counts[packet.domain] += packet.triple_count
    quotas = _largest_remainder_quotas(domain_counts, client_count)
    candidates = []
    for restart_index in range(search_restarts):
        restart_seed = seed + restart_index * _RESTART_PRIME
        primary_domains = _primary_domain_sequence(quotas, restart_seed)
        assignments = _assign_semantic_only_once(
            packets,
            client_count,
            load_tolerance,
            restart_seed,
            primary_domains,
        )
        score = _semantic_only_raw_score(
            dataset,
            packets,
            assignments,
            primary_domains,
            load_tolerance,
        )
        candidates.append((score, primary_domains, assignments))
    score, primary_domains, assignments = min(
        candidates,
        key=lambda value: value[0],
    )
    if int(score[0]) != 0:
        raise RuntimeError(
            "消融B未满足负载合同：最大偏差{:.6f}，容差{:.6f}".format(
                float(score[1]),
                load_tolerance,
            )
        )
    candidate = _build_ablation_candidate(
        dataset=dataset,
        packets=packets,
        assignments=assignments,
        strategy=SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
        seed=seed,
        load_tolerance=load_tolerance,
        search_restarts=search_restarts,
        primary_domains=primary_domains,
        primary_domain_quotas=quotas,
    )
    if candidate.partition_hash != str(score[3]):
        raise RuntimeError("消融B快速哈希与正式分区哈希不一致")
    return candidate
