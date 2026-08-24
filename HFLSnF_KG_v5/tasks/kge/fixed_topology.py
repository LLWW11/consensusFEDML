"""四种联邦TransE对照实验的固定参与客户端与分组拓扑。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ...core.device import as_bool


COMPARISON_SCENARIOS = {
    "flnosnf": {
        "name": "FLnoSnF",
        "architecture": "fl",
        "snf_enabled": False,
        "client_num_per_round": 5,
        "group_num": 1,
    },
    "flsnf": {
        "name": "FLSnF",
        "architecture": "fl",
        "snf_enabled": True,
        "client_num_per_round": 25,
        "group_num": 1,
    },
    "hflnosnf": {
        "name": "HFLnoSnF",
        "architecture": "hfl",
        "snf_enabled": False,
        "client_num_per_round": 15,
        "group_num": 6,
    },
    "hflsnf": {
        "name": "HFLSnF",
        "architecture": "hfl",
        "snf_enabled": True,
        "client_num_per_round": 35,
        "group_num": 6,
    },
}


def _normalize_scenario_name(value: str) -> str:
    """把大小写和分隔符不同的方案名称转换为稳定键。"""

    normalized = (
        str(value)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    if normalized not in COMPARISON_SCENARIOS:
        raise ValueError(
            "comparison_scenario必须是FLnoSnF、FLSnF、"
            "HFLnoSnF或HFLSnF，实际为{}".format(value)
        )
    return normalized


def _hash_payload(payload: Dict[str, object]) -> str:
    """对可JSON序列化的固定拓扑内容计算SHA-256指纹。"""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FixedParticipantTopology:
    """保存整次实验不变的客户端集合及FL/HFL聚合分组。"""

    scenario_name: str
    architecture: str
    snf_enabled: bool
    client_num_in_total: int
    client_num_per_round: int
    group_num: int
    selection_seed: int
    sampled_client_ids: Tuple[int, ...]
    group_client_ids: Tuple[Tuple[int, ...], ...]
    participant_set_hash: str
    topology_hash: str

    def __post_init__(self) -> None:
        """校验固定参与集合、组间互斥性和客户端编号范围。"""

        if self.architecture not in {"fl", "hfl"}:
            raise ValueError("architecture必须是fl或hfl")
        if self.client_num_in_total <= 0:
            raise ValueError("client_num_in_total必须大于0")
        if not 0 < self.client_num_per_round <= self.client_num_in_total:
            raise ValueError(
                "client_num_per_round必须位于1到client_num_in_total之间"
            )
        if len(self.sampled_client_ids) != self.client_num_per_round:
            raise ValueError("固定采样客户端数量与配置不一致")
        if len(set(self.sampled_client_ids)) != len(
            self.sampled_client_ids
        ):
            raise ValueError("固定采样客户端中存在重复编号")
        if any(
            client_id < 0 or client_id >= self.client_num_in_total
            for client_id in self.sampled_client_ids
        ):
            raise ValueError("固定采样客户端编号超出范围")
        if len(self.group_client_ids) != self.group_num:
            raise ValueError("固定分组数量与group_num不一致")
        grouped_ids = [
            client_id
            for group in self.group_client_ids
            for client_id in group
        ]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise ValueError("客户端不能同时出现在多个固定组中")
        if set(grouped_ids) != set(self.sampled_client_ids):
            raise ValueError("固定分组没有完整覆盖采样客户端")
        if any(not group for group in self.group_client_ids):
            raise ValueError("固定拓扑不能包含空组")
        if self.architecture == "fl" and self.group_num != 1:
            raise ValueError("FL固定拓扑必须只有一个云端直连组")

    @property
    def active_client_ids(self) -> Tuple[int, ...]:
        """返回按编号排序的固定活跃客户端。"""

        return tuple(sorted(self.sampled_client_ids))

    def group_mapping(self) -> Dict[int, Tuple[int, ...]]:
        """返回组编号到固定客户端编号元组的映射副本。"""

        return {
            group_id: tuple(client_ids)
            for group_id, client_ids in enumerate(self.group_client_ids)
        }

    def summary(self) -> Dict[str, object]:
        """返回适合保存到结果目录的固定拓扑摘要。"""

        return {
            "scenario": self.scenario_name,
            "architecture": self.architecture,
            "snf_enabled": bool(self.snf_enabled),
            "dynamic_client_selection": False,
            "participant_selection": "fixed_seeded_prefix",
            "snf_selection_applied": False,
            "client_num_in_total": int(self.client_num_in_total),
            "client_num_per_round": int(self.client_num_per_round),
            "group_num": int(self.group_num),
            "selection_seed": int(self.selection_seed),
            "sampled_client_ids": [
                int(value) for value in self.sampled_client_ids
            ],
            "active_client_ids": [
                int(value) for value in self.active_client_ids
            ],
            "group_to_client_indexes": {
                str(group_id): [int(value) for value in client_ids]
                for group_id, client_ids in self.group_mapping().items()
            },
            "group_client_counts": {
                str(group_id): len(client_ids)
                for group_id, client_ids in self.group_mapping().items()
            },
            "participant_set_hash": self.participant_set_hash,
            "topology_hash": self.topology_hash,
        }


def build_fixed_participant_topology(
    args,
    actual_client_count: int,
) -> FixedParticipantTopology:
    """按共同固定排列截取客户端，并为HFL执行稳定六组划分。"""

    configured_total = int(
        getattr(args, "client_num_in_total", actual_client_count)
    )
    if configured_total != int(actual_client_count):
        raise ValueError(
            "client_num_in_total={}与数据分区数{}不一致".format(
                configured_total, actual_client_count
            )
        )
    selection_mode = str(
        getattr(args, "participant_selection", "fixed_once")
    ).strip().lower()
    if selection_mode != "fixed_once":
        raise ValueError(
            "当前四方案只支持participant_selection=fixed_once"
        )

    scenario_key = _normalize_scenario_name(
        getattr(args, "comparison_scenario", "")
    )
    expected = COMPARISON_SCENARIOS[scenario_key]
    architecture = str(
        getattr(args, "topology_architecture", expected["architecture"])
    ).strip().lower()
    snf_enabled = as_bool(
        getattr(args, "topology_snf", expected["snf_enabled"])
    )
    participant_count = int(
        getattr(
            args,
            "client_num_per_round",
            expected["client_num_per_round"],
        )
    )
    group_num = int(getattr(args, "group_num", expected["group_num"]))
    enforce_budget = as_bool(
        getattr(args, "enforce_comparison_budget", True)
    )
    if architecture != expected["architecture"]:
        raise ValueError(
            "{}要求topology_architecture={}，实际为{}".format(
                expected["name"],
                expected["architecture"],
                architecture,
            )
        )
    if snf_enabled != bool(expected["snf_enabled"]):
        raise ValueError(
            "{}的topology_snf配置不正确".format(expected["name"])
        )
    if enforce_budget and participant_count != int(
        expected["client_num_per_round"]
    ):
        raise ValueError(
            "{}要求client_num_per_round={}，实际为{}".format(
                expected["name"],
                expected["client_num_per_round"],
                participant_count,
            )
        )
    if enforce_budget and group_num != int(expected["group_num"]):
        raise ValueError(
            "{}要求group_num={}，实际为{}".format(
                expected["name"], expected["group_num"], group_num
            )
        )
    if not 0 < participant_count <= configured_total:
        raise ValueError(
            "client_num_per_round必须位于1到{}之间".format(
                configured_total
            )
        )
    if architecture == "fl":
        if group_num != 1:
            raise ValueError("FL方案的group_num必须为1")
    elif not 0 < group_num <= participant_count:
        raise ValueError("HFL的group_num必须位于1到参与客户端数之间")

    selection_seed = int(
        getattr(args, "fixed_client_seed", getattr(args, "random_seed", 0))
    )
    # 四份配置使用相同种子时共享同一排列并截取不同前缀，减少身份混杂。
    permutation = np.random.RandomState(selection_seed).permutation(
        configured_total
    )
    sampled_client_ids = tuple(
        int(value) for value in permutation[:participant_count]
    )
    if architecture == "fl":
        groups = (sampled_client_ids,)
    else:
        groups = tuple(
            tuple(sampled_client_ids[group_id::group_num])
            for group_id in range(group_num)
        )

    participant_payload = {
        "client_num_in_total": configured_total,
        "selection_seed": selection_seed,
        "active_client_ids": sorted(sampled_client_ids),
    }
    topology_payload = {
        "scenario": expected["name"],
        "architecture": architecture,
        "snf_enabled": snf_enabled,
        "groups": [list(group) for group in groups],
        "participant_set_hash": _hash_payload(participant_payload),
    }
    return FixedParticipantTopology(
        scenario_name=str(expected["name"]),
        architecture=architecture,
        snf_enabled=snf_enabled,
        client_num_in_total=configured_total,
        client_num_per_round=participant_count,
        group_num=group_num,
        selection_seed=selection_seed,
        sampled_client_ids=sampled_client_ids,
        group_client_ids=groups,
        participant_set_hash=_hash_payload(participant_payload),
        topology_hash=_hash_payload(topology_payload),
    )
