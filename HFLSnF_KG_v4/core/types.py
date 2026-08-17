"""分层联邦任务更新和可合并聚合统计的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

import torch


def clone_state_dict(
    state_dict: Mapping[str, torch.Tensor], to_cpu: bool = True
) -> Dict[str, torch.Tensor]:
    """深拷贝模型状态，并按需把张量移动到CPU以降低显存占用。"""

    cloned = {}
    for name, value in state_dict.items():
        tensor = value.detach().clone()
        if to_cpu:
            tensor = tensor.cpu()
        cloned[str(name)] = tensor
    return cloned


@dataclass
class ClientUpdate:
    """保存一个客户端完成本地训练后提交的通用更新。"""

    client_id: int
    weight: float
    state_dict: Dict[str, torch.Tensor]
    parameter_masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    row_counts: Dict[str, torch.Tensor] = field(default_factory=dict)
    local_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验客户端编号、模型状态及可选逐行聚合信息。"""

        self.client_id = int(self.client_id)
        self.weight = float(self.weight)
        if self.client_id < 0:
            raise ValueError("client_id 不能小于0")
        if self.weight < 0.0:
            raise ValueError("客户端聚合权重不能为负数")
        if not self.state_dict:
            raise ValueError("客户端更新必须包含非空模型状态")
        for collection_name, collection in (
            ("parameter_masks", self.parameter_masks),
            ("row_counts", self.row_counts),
        ):
            unknown_names = set(collection).difference(self.state_dict)
            if unknown_names:
                raise ValueError(
                    "{}包含未知参数{}".format(
                        collection_name, sorted(unknown_names)
                    )
                )
            for name, values in collection.items():
                parameter = self.state_dict[name]
                if parameter.ndim <= 0:
                    raise ValueError(
                        "{}对应参数{}必须至少是一维张量".format(
                            collection_name, name
                        )
                    )
                if values.ndim != 1:
                    raise ValueError(
                        "{}中的{}必须是一维张量".format(
                            collection_name, name
                        )
                    )
                if int(values.shape[0]) != int(parameter.shape[0]):
                    raise ValueError(
                        "{}中的{}长度与参数首维不一致".format(
                            collection_name, name
                        )
                    )
        for name, mask in self.parameter_masks.items():
            if mask.dtype != torch.bool:
                raise TypeError(
                    "parameter_masks中的{}必须是bool张量".format(name)
                )
        for name, counts in self.row_counts.items():
            if not counts.is_floating_point():
                raise TypeError(
                    "row_counts中的{}必须是浮点张量".format(name)
                )
            if bool(torch.any(counts < 0)):
                raise ValueError(
                    "row_counts中的{}不能包含负数".format(name)
                )


@dataclass
class AggregateStats:
    """保存可在边缘和云端继续合并的加权分子、分母与常量缓冲区。"""

    weighted_sums: Dict[str, torch.Tensor]
    constant_tensors: Dict[str, torch.Tensor]
    total_weight: float
    contributor_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        """校验聚合统计必须具有正权重和至少一个贡献客户端。"""

        self.total_weight = float(self.total_weight)
        self.contributor_ids = tuple(int(value) for value in self.contributor_ids)
        if self.total_weight <= 0.0:
            raise ValueError("聚合统计的总权重必须大于0")
        if not self.contributor_ids:
            raise ValueError("聚合统计必须包含至少一个贡献客户端")


@dataclass
class RowMaskedAggregateStats:
    """保存可跨边缘合并的逐参数行分子、分母和贡献客户端。"""

    row_sums: Dict[str, torch.Tensor]
    row_denominators: Dict[str, torch.Tensor]
    constant_tensors: Dict[str, torch.Tensor]
    contributor_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        """校验逐行统计字段、形状和贡献客户端列表。"""

        self.contributor_ids = tuple(
            int(value) for value in self.contributor_ids
        )
        if not self.contributor_ids:
            raise ValueError("逐行聚合统计必须包含至少一个贡献客户端")
        if set(self.row_sums) != set(self.row_denominators):
            raise ValueError("逐行聚合分子和分母的参数键不一致")
        for name, numerator in self.row_sums.items():
            denominator = self.row_denominators[name]
            if numerator.ndim <= 0:
                raise ValueError("逐行聚合参数{}必须至少是一维".format(name))
            if denominator.ndim != 1:
                raise ValueError("参数{}的逐行分母必须是一维".format(name))
            if int(numerator.shape[0]) != int(denominator.shape[0]):
                raise ValueError("参数{}的逐行分子和分母长度不一致".format(name))


@dataclass
class RowCountWeightedAggregateStats:
    """保存可跨边缘无损合并的逐行计数加权统计。"""

    row_sums: Dict[str, torch.Tensor]
    row_denominators: Dict[str, torch.Tensor]
    row_contributor_counts: Dict[str, torch.Tensor]
    constant_tensors: Dict[str, torch.Tensor]
    contributor_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        """校验逐行分子、出现次数分母和贡献客户端计数。"""

        self.contributor_ids = tuple(
            int(value) for value in self.contributor_ids
        )
        if not self.contributor_ids:
            raise ValueError("逐行计数聚合必须包含至少一个贡献客户端")
        expected_keys = set(self.row_sums)
        if expected_keys != set(self.row_denominators):
            raise ValueError("逐行计数聚合分子和分母的参数键不一致")
        if expected_keys != set(self.row_contributor_counts):
            raise ValueError("逐行计数聚合分子和客户端计数的参数键不一致")
        for name, numerator in self.row_sums.items():
            denominator = self.row_denominators[name]
            contributors = self.row_contributor_counts[name]
            if numerator.ndim <= 0:
                raise ValueError(
                    "逐行计数聚合参数{}必须至少是一维".format(name)
                )
            for label, values in (
                ("出现次数分母", denominator),
                ("贡献客户端计数", contributors),
            ):
                if values.ndim != 1:
                    raise ValueError(
                        "参数{}的{}必须是一维".format(name, label)
                    )
                if int(values.shape[0]) != int(numerator.shape[0]):
                    raise ValueError(
                        "参数{}的{}长度与分子不一致".format(
                            name, label
                        )
                    )
                if not values.is_floating_point():
                    raise TypeError(
                        "参数{}的{}必须是浮点张量".format(
                            name, label
                        )
                    )
                if bool(torch.any(values < 0)):
                    raise ValueError(
                        "参数{}的{}不能包含负数".format(name, label)
                    )
