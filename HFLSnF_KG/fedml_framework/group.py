"""从HFLSnF边缘组结构适配得到的GCN边缘训练与聚合单元。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch

from ..core.aggregation import DenseFedAvgAggregator
from ..core.types import AggregateStats, ClientUpdate
from .client import FedMLGCNClient


class FedMLHierarchicalGroup:
    """调度一个边缘组内的FedML客户端并形成可合并边缘统计。"""

    def __init__(
        self,
        group_id: int,
        client_registry: Dict[int, FedMLGCNClient],
        aggregator: DenseFedAvgAggregator,
    ):
        """保存边缘组编号、全局客户端注册表和聚合器。"""

        self.group_id = int(group_id)
        self.client_registry = client_registry
        self.aggregator = aggregator

    def train_clients(
        self,
        client_ids: Iterable[int],
        global_state: Dict[str, torch.Tensor],
        round_index: int,
    ) -> List[ClientUpdate]:
        """让指定客户端通过FedML生命周期训练并收集有效更新。"""

        updates = []
        for raw_client_id in client_ids:
            client_id = int(raw_client_id)
            if client_id not in self.client_registry:
                raise KeyError(
                    "边缘组{}找不到客户端{}".format(
                        self.group_id, client_id
                    )
                )
            update = self.client_registry[client_id].train_from_global(
                global_state, round_index
            )
            if update is not None and update.weight > 0.0:
                updates.append(update)
        return updates

    def train_and_accumulate(
        self,
        client_ids: Iterable[int],
        global_state: Dict[str, torch.Tensor],
        round_index: int,
    ) -> Tuple[List[ClientUpdate], Optional[AggregateStats]]:
        """完成组内客户端训练并返回边缘可合并统计。"""

        updates = self.train_clients(
            client_ids, global_state, round_index
        )
        if not updates:
            return updates, None
        return updates, self.aggregator.accumulate(updates)
