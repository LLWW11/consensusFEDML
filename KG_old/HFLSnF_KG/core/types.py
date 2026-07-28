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
        """校验客户端编号、聚合权重和模型状态的基本合法性。"""

        self.client_id = int(self.client_id)
        self.weight = float(self.weight)
        if self.client_id < 0:
            raise ValueError("client_id 不能小于0")
        if self.weight < 0.0:
            raise ValueError("客户端聚合权重不能为负数")
        if not self.state_dict:
            raise ValueError("客户端更新必须包含非空模型状态")


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
