"""任务无关的静态、序列和MATLAB拓扑提供器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _normalize_groups(
    group_to_clients: Mapping[int, Iterable[int]]
) -> Dict[int, Tuple[int, ...]]:
    """把边缘分组规范化为排序稳定且无重复客户端的不可变映射。"""

    normalized: Dict[int, Tuple[int, ...]] = {}
    all_clients: List[int] = []
    for raw_group_id, raw_clients in sorted(
        group_to_clients.items(), key=lambda item: int(item[0])
    ):
        group_id = int(raw_group_id)
        clients = tuple(int(value) for value in raw_clients)
        if group_id < 0:
            raise ValueError("边缘组编号不能小于0")
        if any(client_id < 0 for client_id in clients):
            raise ValueError("客户端编号不能小于0")
        if len(set(clients)) != len(clients):
            raise ValueError("边缘组{}内部存在重复客户端".format(group_id))
        if clients:
            normalized[group_id] = clients
            all_clients.extend(clients)
    if len(set(all_clients)) != len(all_clients):
        raise ValueError("同一客户端不能同时出现在多个边缘组")
    return normalized


@dataclass(frozen=True)
class RoundTopology:
    """保存一个通信轮中的边缘分组、活跃客户端和可选物理边缘编号。"""

    group_to_client_indexes: Dict[int, Tuple[int, ...]]
    active_client_indexes: Tuple[int, ...]
    edge_node_ids: Dict[int, int]
    participant_count: int
    source_round_index: int

    @classmethod
    def from_groups(
        cls,
        group_to_clients: Mapping[int, Iterable[int]],
        round_index: int,
        edge_node_ids: Optional[Mapping[int, int]] = None,
    ) -> "RoundTopology":
        """从普通分组映射构造并校验一个通信轮拓扑。"""

        normalized = _normalize_groups(group_to_clients)
        active_clients = tuple(
            sorted(
                client_id
                for client_ids in normalized.values()
                for client_id in client_ids
            )
        )
        if edge_node_ids is None:
            normalized_edge_ids = {
                group_id: group_id for group_id in normalized.keys()
            }
        else:
            normalized_edge_ids = {
                int(group_id): int(edge_id)
                for group_id, edge_id in edge_node_ids.items()
                if int(group_id) in normalized
            }
        return cls(
            group_to_client_indexes=normalized,
            active_client_indexes=active_clients,
            edge_node_ids=normalized_edge_ids,
            participant_count=len(active_clients),
            source_round_index=int(round_index),
        )

    def copy_groups(self) -> Dict[int, List[int]]:
        """返回可供运行记录序列化的边缘分组列表副本。"""

        return {
            int(group_id): list(client_ids)
            for group_id, client_ids in self.group_to_client_indexes.items()
        }


class TopologyProvider(ABC):
    """为每个通信轮提供与具体学习任务无关的客户端分组。"""

    @abstractmethod
    def get_round(self, round_index: int) -> RoundTopology:
        """返回指定通信轮的拓扑。"""

    @abstractmethod
    def describe(self) -> Dict[str, object]:
        """返回可写入结果目录的拓扑元数据。"""


class StaticTopologyProvider(TopologyProvider):
    """在全部通信轮中重复使用同一组客户端—边缘映射。"""

    def __init__(self, group_to_clients: Mapping[int, Iterable[int]]):
        """保存经过校验的固定边缘分组。"""

        self._groups = _normalize_groups(group_to_clients)

    @classmethod
    def round_robin(
        cls, client_ids: Sequence[int], group_count: int
    ) -> "StaticTopologyProvider":
        """按客户端顺序轮转分配到指定数量的边缘组。"""

        group_count = int(group_count)
        if group_count <= 0:
            raise ValueError("group_count 必须大于0")
        groups: Dict[int, List[int]] = {
            group_id: [] for group_id in range(group_count)
        }
        for offset, raw_client_id in enumerate(client_ids):
            groups[offset % group_count].append(int(raw_client_id))
        return cls(groups)

    def get_round(self, round_index: int) -> RoundTopology:
        """返回任意通信轮都相同的固定分组。"""

        if int(round_index) < 0:
            raise ValueError("round_index 不能小于0")
        return RoundTopology.from_groups(self._groups, int(round_index))

    def describe(self) -> Dict[str, object]:
        """返回固定拓扑类型和边缘分组。"""

        return {
            "provider_type": "static",
            "group_to_client_indexes": {
                str(group_id): list(client_ids)
                for group_id, client_ids in self._groups.items()
            },
        }


class SequenceTopologyProvider(TopologyProvider):
    """按给定顺序为每个通信轮返回一份独立拓扑。"""

    def __init__(
        self, round_groups: Sequence[Mapping[int, Iterable[int]]]
    ):
        """预先校验并保存有限长度的逐轮分组序列。"""

        if not round_groups:
            raise ValueError("逐轮拓扑序列不能为空")
        self._rounds = [
            _normalize_groups(group_mapping) for group_mapping in round_groups
        ]

    def get_round(self, round_index: int) -> RoundTopology:
        """返回指定索引的逐轮拓扑，越界时给出明确错误。"""

        round_index = int(round_index)
        if round_index < 0 or round_index >= len(self._rounds):
            raise IndexError(
                "通信轮{}超出拓扑序列范围0..{}".format(
                    round_index, len(self._rounds) - 1
                )
            )
        return RoundTopology.from_groups(
            self._rounds[round_index], round_index
        )

    def describe(self) -> Dict[str, object]:
        """返回序列拓扑类型和总通信轮数。"""

        return {
            "provider_type": "sequence",
            "round_count": len(self._rounds),
        }


class FixedCountTopologyProvider(TopologyProvider):
    """按YAML指定人数构造随机、SnF投影或均匀轮换拓扑。"""

    def __init__(
        self,
        client_ids: Sequence[int],
        participant_count: int,
        architecture: str,
        group_count: int,
        selection_mode: str,
        seed: int,
        source_provider: Optional[TopologyProvider] = None,
    ):
        """校验固定人数合同并保存可选MAT来源拓扑。"""

        normalized_clients = tuple(int(value) for value in client_ids)
        if not normalized_clients or len(set(normalized_clients)) != len(
            normalized_clients
        ):
            raise ValueError("固定人数调度的客户端池必须非空且无重复")
        self._client_ids = normalized_clients
        self._client_set = set(normalized_clients)
        self._participant_count = int(participant_count)
        if not 0 < self._participant_count <= len(self._client_ids):
            raise ValueError("固定参与人数必须位于1和客户端池大小之间")
        self._architecture = str(architecture).strip().lower()
        if self._architecture not in {"hfl", "fl"}:
            raise ValueError("固定人数调度architecture必须是hfl或fl")
        self._group_count = int(group_count)
        if self._architecture == "fl":
            if self._group_count != 1:
                raise ValueError("FL固定人数调度必须只有1个组")
        elif not 0 < self._group_count <= self._participant_count:
            raise ValueError("HFL组数必须位于1和固定参与人数之间")
        self._selection_mode = str(selection_mode).strip().lower()
        if self._selection_mode not in {
            "seeded_random",
            "snf_mat_projected",
            "seeded_round_robin",
        }:
            raise ValueError(
                "fixed_count_selection_mode必须是"
                "seeded_random、snf_mat_projected或seeded_round_robin"
            )
        if (
            self._selection_mode == "snf_mat_projected"
            and source_provider is None
        ):
            raise ValueError("SnF固定人数投影必须提供MAT来源拓扑")
        if self._selection_mode in {
            "seeded_random",
            "seeded_round_robin",
        } and source_provider is not None:
            raise ValueError("noSnF随机或均匀轮换不能读取MAT选择结果")
        self._source_provider = source_provider
        self._seed = int(seed)
        seeded_order = list(self._client_ids)
        random.Random(self._seed).shuffle(seeded_order)
        self._round_robin_order = tuple(seeded_order)

    def _seeded_round_order(
        self,
        round_index: int,
        candidates: Sequence[int],
        salt: int,
    ) -> Tuple[int, ...]:
        """按轮次和盐值生成可复现的候选客户端顺序。"""

        ordered = list(int(value) for value in candidates)
        random.Random(
            self._seed + 1000003 * int(round_index) + int(salt)
        ).shuffle(ordered)
        return tuple(ordered)

    def _project_snf_participants(
        self,
        round_index: int,
        source_topology: RoundTopology,
    ) -> Tuple[int, ...]:
        """优先保留MAT选中客户端，并补齐或裁剪到YAML固定人数。"""

        source_active = tuple(
            int(value) for value in source_topology.active_client_indexes
        )
        if not set(source_active).issubset(self._client_set):
            raise ValueError("MAT来源拓扑包含客户端池之外的编号")
        active_order = self._seeded_round_order(
            round_index,
            source_active,
            salt=17,
        )
        if len(active_order) >= self._participant_count:
            return tuple(
                sorted(active_order[: self._participant_count])
            )
        inactive = tuple(
            client_id
            for client_id in self._client_ids
            if client_id not in set(source_active)
        )
        fill_order = self._seeded_round_order(
            round_index,
            inactive,
            salt=31,
        )
        selected = active_order + fill_order[
            : self._participant_count - len(active_order)
        ]
        return tuple(sorted(selected))

    def _round_robin_participants(
        self,
        round_index: int,
    ) -> Tuple[int, ...]:
        """按循环窗口选择固定人数，使长期参与次数尽量均衡。"""

        client_count = len(self._round_robin_order)
        start = (
            int(round_index) * self._participant_count
        ) % client_count
        selected = tuple(
            self._round_robin_order[(start + offset) % client_count]
            for offset in range(self._participant_count)
        )
        return tuple(sorted(selected))

    def _seeded_random_participants(
        self,
        round_index: int,
    ) -> Tuple[int, ...]:
        """按轮次独立随机抽取固定人数，并通过种子保证结果可复现。"""

        random_generator = random.Random(
            self._seed + 1000003 * int(round_index) + 47
        )
        selected = random_generator.sample(
            self._client_ids,
            self._participant_count,
        )
        return tuple(sorted(int(value) for value in selected))

    def _groups_for_participants(
        self,
        participants: Sequence[int],
    ) -> Dict[int, List[int]]:
        """把选中客户端稳定轮转分入HFL六组或FL单组。"""

        if self._architecture == "fl":
            return {0: [int(value) for value in participants]}
        groups: Dict[int, List[int]] = {
            group_id: [] for group_id in range(self._group_count)
        }
        for offset, client_id in enumerate(participants):
            groups[offset % self._group_count].append(int(client_id))
        return groups

    def get_round(self, round_index: int) -> RoundTopology:
        """返回客户端人数严格等于YAML设定值的一轮拓扑。"""

        round_index = int(round_index)
        if round_index < 0:
            raise ValueError("round_index不能小于0")
        source_round_index = round_index
        if self._selection_mode == "snf_mat_projected":
            source = self._source_provider.get_round(round_index)
            source_round_index = int(source.source_round_index)
            participants = self._project_snf_participants(
                round_index,
                source,
            )
        elif self._selection_mode == "seeded_random":
            participants = self._seeded_random_participants(round_index)
        else:
            participants = self._round_robin_participants(round_index)
        groups = self._groups_for_participants(participants)
        topology = RoundTopology.from_groups(
            groups,
            source_round_index,
        )
        if topology.participant_count != self._participant_count:
            raise RuntimeError("固定人数拓扑构造后参与人数发生漂移")
        return topology

    def describe(self) -> Dict[str, object]:
        """返回固定人数、选择语义和可选MAT来源元数据。"""

        source_metadata = (
            dict(self._source_provider.describe())
            if self._source_provider is not None
            else None
        )
        return {
            "provider_type": "fixed_count",
            "architecture": self._architecture,
            "snf_enabled": (
                self._selection_mode == "snf_mat_projected"
            ),
            "edge_mode": (
                "fixed" if self._architecture == "hfl" else "none"
            ),
            "fixed_participant_count": self._participant_count,
            "fixed_group_count": self._group_count,
            "fixed_count_selection_mode": self._selection_mode,
            "fixed_count_seed": self._seed,
            "source_topology": source_metadata,
            "mat_file": (
                source_metadata.get("mat_file")
                if isinstance(source_metadata, dict)
                else None
            ),
            "topology_util": (
                source_metadata.get("topology_util")
                if isinstance(source_metadata, dict)
                else None
            ),
            "round_count": (
                source_metadata.get("round_count")
                if isinstance(source_metadata, dict)
                else None
            ),
            "source_round_count": (
                source_metadata.get(
                    "source_round_count",
                    source_metadata.get("round_count"),
                )
                if isinstance(source_metadata, dict)
                else None
            ),
            "topology_schedule_policy": (
                source_metadata.get(
                    "topology_schedule_policy", "unbounded"
                )
                if isinstance(source_metadata, dict)
                else "unbounded"
            ),
        }


class MatlabTopologyProvider(TopologyProvider):
    """把原HFLSnF的MATLAB拓扑调度包装为任务无关提供器。"""

    def __init__(
        self,
        mat_path: Path,
        architecture: str,
        snf_enabled: bool,
        edge_mode: str,
        util: float,
        client_count: int,
        schedule_policy: str = "strict",
    ):
        """加载只读MAT调度，并配置越过源轮数时严格报错或循环复用。"""

        # 使用工程内置解析器，避免运行时依赖外部 HFLSnF_dynEdge 工程。
        from .matlab_topology_schedule import MatlabTopologySchedule

        normalized_policy = str(schedule_policy).strip().lower()
        if normalized_policy not in {"strict", "cycle"}:
            raise ValueError(
                "MAT拓扑调度策略必须是strict或cycle"
            )
        self._schedule_policy = normalized_policy
        self._schedule = MatlabTopologySchedule(
            mat_path=str(Path(mat_path).expanduser().resolve()),
            architecture=str(architecture),
            snf_enabled=bool(snf_enabled),
            edge_mode=str(edge_mode),
            util=float(util),
            client_num_in_total=int(client_count),
            candidate_client_count=int(client_count),
        )

    @property
    def round_count(self) -> int:
        """返回MAT文件中可用的拓扑轮数。"""

        return int(self._schedule.round_count)

    def get_round(self, round_index: int) -> RoundTopology:
        """按严格或循环策略读取MAT源行，并保留真实源轮次编号。"""

        requested_round_index = int(round_index)
        if requested_round_index < 0:
            raise ValueError("round_index不能小于0")
        source_round_index = requested_round_index
        if self._schedule_policy == "cycle":
            source_round_index %= self.round_count
        original = self._schedule.get_round(source_round_index)
        return RoundTopology.from_groups(
            original.copy_groups(),
            source_round_index,
            edge_node_ids=original.edge_node_ids,
        )

    def describe(self) -> Dict[str, object]:
        """返回旧MAT调度已经校验过的完整元数据。"""

        metadata = dict(self._schedule.to_metadata())
        metadata["provider_type"] = "matlab_adapter"
        metadata["slot_mapping"] = "identity"
        metadata["topology_schedule_policy"] = (
            self._schedule_policy
        )
        metadata["source_round_count"] = self.round_count
        return metadata
