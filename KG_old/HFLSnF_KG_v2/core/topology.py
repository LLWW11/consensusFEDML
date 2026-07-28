"""任务无关的静态、序列和MATLAB拓扑提供器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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
    ):
        """加载旧项目的只读MAT调度，并使用客户端编号直接对应候选槽位。"""

        from HFLSnF_dynEdge.topology_schedule import MatlabTopologySchedule

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
        """读取MAT指定行并转换为新模拟器使用的通信轮拓扑。"""

        original = self._schedule.get_round(int(round_index))
        return RoundTopology.from_groups(
            original.copy_groups(),
            int(round_index),
            edge_node_ids=original.edge_node_ids,
        )

    def describe(self) -> Dict[str, object]:
        """返回旧MAT调度已经校验过的完整元数据。"""

        metadata = dict(self._schedule.to_metadata())
        metadata["provider_type"] = "matlab_adapter"
        metadata["slot_mapping"] = "identity"
        return metadata
