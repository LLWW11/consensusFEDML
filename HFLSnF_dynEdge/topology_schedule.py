from dataclasses import dataclass
import math
import os

import numpy as np
from scipy.io import loadmat


def _coerce_integer(value, context):
    """将 MATLAB 数值安全转换为 Python 整数。"""
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("{} 包含非有限数值 {!r}".format(context, value))
    integer_value = int(value)
    if float(value) != integer_value:
        raise ValueError("{} 包含非整数数值 {!r}".format(context, value))
    return integer_value


def _flatten_numeric_values(value, context):
    """递归展开 MATLAB cell、NumPy 数组和标量中的整数值。"""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        flattened = []
        for item in value.reshape(-1):
            flattened.extend(_flatten_numeric_values(item, context))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item, context))
        return flattened
    return [_coerce_integer(value, context)]


def _as_object_items(value, context):
    """把 MATLAB cell 对应值转换为保留 cell 边界的 Python 列表。"""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.dtype == object:
            return list(value.reshape(-1))
        return list(value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    if np.isscalar(value):
        return [value]
    raise TypeError("{} 的 MATLAB cell 类型不受支持：{}".format(context, type(value).__name__))


def _extract_policy_value(value, policy_index, context):
    """从 MATLAB 策略 cell 中提取指定的零基策略位置。"""
    policy_items = _as_object_items(value, context)
    if len(policy_items) <= policy_index:
        raise ValueError(
            "{} 只有 {} 个策略位置，无法读取索引 {}".format(context, len(policy_items), policy_index)
        )
    return policy_items[policy_index]


def _split_group_values(value, expected_group_slots, context):
    """依据边缘槽位数恢复 HFL 映射中每个边缘对应的客户端 cell。"""
    if expected_group_slots == 0:
        if _flatten_numeric_values(value, context):
            raise ValueError("{} 没有边缘槽位却包含客户端".format(context))
        return []

    if expected_group_slots == 1:
        items = _as_object_items(value, context)
        if isinstance(value, np.ndarray) and value.dtype == object and len(items) == 1:
            return [items[0]]
        return [value]

    items = _as_object_items(value, context)
    if len(items) != expected_group_slots:
        raise ValueError(
            "{} 包含 {} 个映射槽位，期望 {} 个".format(
                context, len(items), expected_group_slots
            )
        )
    return items


def _matrix_value(matrix, round_index, util_index, context):
    """从 epoch×util MATLAB 矩阵中读取一个单元。"""
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("{} 必须是二维 epoch×util 矩阵，实际形状为 {}".format(context, array.shape))
    if round_index >= array.shape[0] or util_index >= array.shape[1]:
        raise IndexError(
            "{} 索引 ({}, {}) 超出形状 {}".format(context, round_index, util_index, array.shape)
        )
    return array[round_index, util_index]


def _parse_bool(value, context):
    """将 YAML 布尔值或常见字符串转换为 bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("{} 必须是布尔值，实际为 {!r}".format(context, value))


@dataclass(frozen=True)
class RoundTopology:
    """保存一个本地 epoch 中参与候选槽位及其边缘分组。"""

    round_index: int
    group_to_client_indexes: dict
    active_client_indexes: tuple
    edge_node_ids: dict
    participant_count: int

    @property
    def group_client_counts(self):
        """返回每个边缘槽位需要的候选客户端数量，包括人数为零的组。"""
        group_indexes = set(self.group_to_client_indexes.keys())
        group_indexes.update(self.edge_node_ids.keys())
        return {
            int(group_index): len(self.group_to_client_indexes.get(group_index, ()))
            for group_index in sorted(group_indexes)
        }

    @property
    def active_candidate_slots(self):
        """返回 MAT 当前 epoch 中启用的 37 人候选槽位。"""
        return self.active_client_indexes

    def copy_groups(self):
        """返回可由训练流程安全修改的候选槽位分组副本。"""
        return {
            int(group_index): list(client_indexes)
            for group_index, client_indexes in self.group_to_client_indexes.items()
        }


class MatlabTopologySchedule:
    """从正式 MATLAB 结果构建可供 Python 联邦训练使用的拓扑调度。"""

    def __init__(
            self,
            mat_path,
            architecture,
            snf_enabled,
            edge_mode,
            util,
            client_num_in_total=None,
            candidate_client_count=None,
    ):
        """加载 MAT，并解耦真实客户端池与 MAT 的 37 个候选槽位。"""
        self.mat_path = os.path.abspath(mat_path)
        if not os.path.isfile(self.mat_path):
            raise FileNotFoundError("找不到 MATLAB 拓扑文件：{}".format(self.mat_path))

        self.architecture = str(architecture).strip().lower()
        if self.architecture not in {"hfl", "fl"}:
            raise ValueError("topology_architecture 必须是 hfl 或 fl")
        self.snf_enabled = _parse_bool(snf_enabled, "topology_snf")
        self.edge_mode = str(edge_mode).strip().lower()
        if self.architecture == "hfl" and self.edge_mode not in {"fixed", "dynamic"}:
            raise ValueError("HFL 的 topology_edge_mode 必须是 fixed 或 dynamic")
        if self.architecture == "fl":
            self.edge_mode = "none"

        self.requested_util = float(util)
        if client_num_in_total is None and candidate_client_count is None:
            raise ValueError("必须提供 client_num_in_total 或 candidate_client_count")

        # 兼容旧调用：只传 client_num_in_total=37 时，将它同时视为候选槽位数。
        self.client_num_in_total = (
            int(client_num_in_total) if client_num_in_total is not None else None
        )
        if candidate_client_count is None:
            candidate_client_count = self.client_num_in_total
        self.candidate_client_count = int(candidate_client_count)
        if self.candidate_client_count <= 0:
            raise ValueError("candidate_client_count 必须大于 0")
        if (
                self.client_num_in_total is not None
                and self.client_num_in_total < self.candidate_client_count
        ):
            raise ValueError(
                "client_num_in_total={} 不能小于 candidate_client_count={}".format(
                    self.client_num_in_total, self.candidate_client_count
                )
            )
        self._data = loadmat(self.mat_path, simplify_cells=True)
        self._load_common_metadata()
        self._select_scenario_fields()
        self._rounds = tuple(self._build_round(round_index) for round_index in range(self.round_count))
        self.group_capacity = self._compute_group_capacity()

    def _load_common_metadata(self):
        """读取并校验轮数、利用率、节点数和云节点等公共元数据。"""
        required_fields = {"epoch_num", "total_util", "num_of_nodes", "Cloud"}
        missing_fields = sorted(required_fields.difference(self._data))
        if missing_fields:
            raise KeyError("MATLAB 拓扑文件缺少字段：{}".format(", ".join(missing_fields)))

        self.round_count = _coerce_integer(self._data["epoch_num"], "epoch_num")
        self.total_utils = tuple(float(value) for value in np.asarray(self._data["total_util"]).reshape(-1))
        matching_indexes = [
            index for index, value in enumerate(self.total_utils)
            if math.isclose(value, self.requested_util, rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(matching_indexes) != 1:
            raise ValueError(
                "topology_util={} 未唯一匹配 total_util={}".format(
                    self.requested_util, list(self.total_utils)
                )
            )
        self.util_index = matching_indexes[0]

        self.num_of_nodes = _coerce_integer(self._data["num_of_nodes"], "num_of_nodes")
        self.cloud_node_id = _coerce_integer(self._data["Cloud"], "Cloud")
        self.valid_physical_client_ids = tuple(
            node_id for node_id in range(1, self.num_of_nodes + 1)
            if node_id != self.cloud_node_id
        )
        self.mat_physical_client_count = len(self.valid_physical_client_ids)
        if self.mat_physical_client_count != self.candidate_client_count:
            raise ValueError(
                "MATLAB 中有 {} 个候选槽位，但 candidate_client_count={}".format(
                    self.mat_physical_client_count, self.candidate_client_count
                )
            )
        self._physical_to_candidate_slot = {
            physical_id: python_index
            for python_index, physical_id in enumerate(self.valid_physical_client_ids)
        }

    def _select_scenario_fields(self):
        """根据 HFL/FL、SnF 和边缘模式选择 MAT 中的映射及校验字段。"""
        if self.architecture == "hfl":
            method_name = "HFLSnF" if self.snf_enabled else "HFLnoSnF"
            mode_suffix = "fix" if self.edge_mode == "fixed" else "los"
            self.mapping_field = "actual_c2e_map_{}".format(method_name)
            self.edge_field = "DynEdgeSet_{}".format(method_name)
            self.group_count_field = "group_num_{}_{}".format(method_name, mode_suffix)
            self.client_count_field = "client_num_{}_{}".format(method_name, mode_suffix)
            self.policy_index = 2 if self.edge_mode == "fixed" else 0
        else:
            method_name = "FLSnF" if self.snf_enabled else "FLnoSnF"
            self.mapping_field = "c2cmap_{}_all".format(method_name)
            self.edge_field = None
            self.group_count_field = None
            self.client_count_field = "client_num_{}".format(method_name)
            self.policy_index = None

        required_fields = {self.mapping_field, self.client_count_field}
        if self.edge_field is not None:
            required_fields.update({self.edge_field, self.group_count_field})
        missing_fields = sorted(required_fields.difference(self._data))
        if missing_fields:
            raise KeyError(
                "场景 {} 缺少 MAT 字段：{}".format(self.scenario_name, ", ".join(missing_fields))
            )

    @property
    def scenario_name(self):
        """返回适合日志和结果目录使用的场景名称。"""
        snf_name = "snf" if self.snf_enabled else "no_snf"
        if self.architecture == "hfl":
            return "hfl_{}_{}".format(snf_name, self.edge_mode)
        return "fl_{}".format(snf_name)

    @property
    def util_label(self):
        """返回适合文件名使用的网络利用率文本。"""
        return ("{:g}".format(self.requested_util)).replace(".", "p")

    def matlab_id_to_candidate_slot(self, physical_client_id):
        """把 MATLAB 物理节点 ID 转换为 0 开始的候选槽位编号。"""
        physical_client_id = _coerce_integer(physical_client_id, "physical client id")
        if physical_client_id == self.cloud_node_id:
            raise ValueError("云节点 {} 不能作为客户端".format(self.cloud_node_id))
        if physical_client_id not in self._physical_to_candidate_slot:
            raise ValueError("非法 MATLAB 客户端节点 {}".format(physical_client_id))
        return self._physical_to_candidate_slot[physical_client_id]

    def matlab_id_to_python(self, physical_client_id):
        """兼容旧接口，返回 MATLAB 物理节点对应的候选槽位编号。"""
        return self.matlab_id_to_candidate_slot(physical_client_id)

    def _convert_client_ids(self, raw_mapping, context):
        """展开一个映射并将其中所有物理 ID 转换为 Python 客户端编号。"""
        physical_ids = _flatten_numeric_values(raw_mapping, context)
        python_ids = [
            self.matlab_id_to_candidate_slot(physical_id)
            for physical_id in physical_ids
        ]
        if len(set(python_ids)) != len(python_ids):
            raise ValueError("{} 中存在重复客户端 ID".format(context))
        return python_ids

    def _expected_scalar(self, field_name, round_index):
        """读取所选 util 下某个 epoch×util 数值字段的整数标量。"""
        value = _matrix_value(
            self._data[field_name], round_index, self.util_index, field_name
        )
        return _coerce_integer(value, "{} round {}".format(field_name, round_index))

    def _build_round(self, round_index):
        """解析并校验一个通信轮次的真实参与客户端映射。"""
        if self.architecture == "hfl":
            return self._build_hfl_round(round_index)
        return self._build_fl_round(round_index)

    def _build_hfl_round(self, round_index):
        """构建一个 HFL 轮次的边缘分组并核对组数和参与人数。"""
        context = "{} round {} util {}".format(
            self.scenario_name, round_index, self.requested_util
        )
        raw_mapping_cell = _matrix_value(
            self._data[self.mapping_field], round_index, self.util_index, self.mapping_field
        )
        raw_edge_cell = _matrix_value(
            self._data[self.edge_field], round_index, self.util_index, self.edge_field
        )
        policy_mapping = _extract_policy_value(raw_mapping_cell, self.policy_index, context + " mapping")
        policy_edges = _extract_policy_value(raw_edge_cell, self.policy_index, context + " edges")
        edge_ids = _flatten_numeric_values(policy_edges, context + " edges")
        group_values = _split_group_values(policy_mapping, len(edge_ids), context + " mapping")

        groups = {}
        edge_node_ids = {}
        all_clients = []
        for group_index, (edge_id, raw_group) in enumerate(zip(edge_ids, group_values)):
            group_clients = self._convert_client_ids(
                raw_group, "{} group {}".format(context, group_index)
            )
            edge_node_ids[group_index] = edge_id
            if group_clients:
                groups[group_index] = tuple(group_clients)
                all_clients.extend(group_clients)

        if len(set(all_clients)) != len(all_clients):
            raise ValueError("{} 的不同边缘组之间存在重复客户端".format(context))
        expected_group_count = self._expected_scalar(self.group_count_field, round_index)
        expected_client_count = self._expected_scalar(self.client_count_field, round_index)
        if len(groups) != expected_group_count:
            raise ValueError(
                "{} 映射得到 {} 个有效组，但 {} 记录为 {}".format(
                    context, len(groups), self.group_count_field, expected_group_count
                )
            )
        if len(all_clients) != expected_client_count:
            raise ValueError(
                "{} 映射得到 {} 个客户端，但 {} 记录为 {}".format(
                    context, len(all_clients), self.client_count_field, expected_client_count
                )
            )
        if expected_client_count > self.candidate_client_count:
            raise ValueError(
                "{} 的参与人数 {} 超过候选槽位数 {}".format(
                    context, expected_client_count, self.candidate_client_count
                )
            )
        return RoundTopology(
            round_index=round_index,
            group_to_client_indexes=groups,
            active_client_indexes=tuple(sorted(all_clients)),
            edge_node_ids=edge_node_ids,
            participant_count=len(all_clients),
        )

    def _build_fl_round(self, round_index):
        """构建一个 FL 轮次的直接云参与客户端列表并核对人数。"""
        context = "{} round {} util {}".format(
            self.scenario_name, round_index, self.requested_util
        )
        raw_mapping = _matrix_value(
            self._data[self.mapping_field], round_index, self.util_index, self.mapping_field
        )
        client_indexes = self._convert_client_ids(raw_mapping, context + " mapping")
        expected_client_count = self._expected_scalar(self.client_count_field, round_index)
        if len(client_indexes) != expected_client_count:
            raise ValueError(
                "{} 映射得到 {} 个客户端，但 {} 记录为 {}".format(
                    context, len(client_indexes), self.client_count_field, expected_client_count
                )
            )
        if expected_client_count > self.candidate_client_count:
            raise ValueError(
                "{} 的参与人数 {} 超过候选槽位数 {}".format(
                    context, expected_client_count, self.candidate_client_count
                )
            )
        groups = {0: tuple(client_indexes)} if client_indexes else {}
        return RoundTopology(
            round_index=round_index,
            group_to_client_indexes=groups,
            active_client_indexes=tuple(sorted(client_indexes)),
            edge_node_ids={},
            participant_count=len(client_indexes),
        )

    def _compute_group_capacity(self):
        """计算整个实验需要预建的最大边缘槽位数。"""
        if self.architecture == "fl":
            return 1
        maximum_slot = 0
        for round_topology in self._rounds:
            if round_topology.edge_node_ids:
                maximum_slot = max(maximum_slot, max(round_topology.edge_node_ids) + 1)
        return maximum_slot

    def get_round(self, round_index):
        """返回指定通信轮次的已校验拓扑。"""
        if round_index < 0 or round_index >= self.round_count:
            raise IndexError(
                "round {} 超出 MATLAB 拓扑的 0..{} 范围".format(
                    round_index, self.round_count - 1
                )
            )
        return self._rounds[round_index]

    def to_metadata(self):
        """返回可写入结果目录的拓扑实验元数据。"""
        participant_counts = [item.participant_count for item in self._rounds]
        return {
            "mat_file": self.mat_path,
            "schema_version": str(self._data.get("schema_version", "unknown")),
            "scenario": self.scenario_name,
            "architecture": self.architecture,
            "snf_enabled": self.snf_enabled,
            "edge_mode": self.edge_mode,
            "topology_util": self.requested_util,
            "util_index": self.util_index,
            "available_utils": list(self.total_utils),
            "round_count": self.round_count,
            "group_capacity": self.group_capacity,
            "cloud_node_id": self.cloud_node_id,
            "client_num_in_total": self.client_num_in_total,
            "candidate_client_count": self.candidate_client_count,
            "mat_physical_client_count": self.mat_physical_client_count,
            "physical_to_candidate_slot": {
                str(physical_id): python_index
                for physical_id, python_index
                in self._physical_to_candidate_slot.items()
            },
            "physical_to_python_client": {
                str(physical_id): python_index
                for physical_id, python_index
                in self._physical_to_candidate_slot.items()
            },
            "participant_count_min": min(participant_counts),
            "participant_count_max": max(participant_counts),
            "participant_count_mean": float(np.mean(participant_counts)),
        }
