"""从HFLSnF客户端结构适配得到的FedML GCN客户端。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from fedml.simulation.sp.fedavg.client import Client as FedMLClient

from FedGCN_fedml.data import LocalGraphPartition

from ..core.types import ClientUpdate, clone_state_dict
from .model_trainer import FedMLGCNModelTrainer


class FedMLGCNClient(FedMLClient):
    """使用FedML FedAvg客户端生命周期训练局部诱导子图。"""

    def __init__(
        self,
        partition: LocalGraphPartition,
        args,
        device: torch.device,
        model_trainer: FedMLGCNModelTrainer,
    ):
        """把局部图分区绑定到FedML客户端基类。"""

        super().__init__(
            client_idx=int(partition.client_id),
            local_training_data=partition,
            local_test_data=partition,
            local_sample_number=int(partition.train_node_count),
            args=args,
            device=torch.device(device),
            model_trainer=model_trainer,
        )
        self.partition = partition

    def train_from_global(
        self,
        global_state: Dict[str, torch.Tensor],
        round_index: int,
    ) -> Optional[ClientUpdate]:
        """调用FedML客户端基类从服务器参数开始完成本地训练。"""

        if self.partition.train_node_count <= 0:
            return None
        self.model_trainer.set_id(int(self.client_idx))
        self.args.round_idx = int(round_index)
        # FedML Client.train内部依次完成参数下发、ClientTrainer.train和参数回收。
        local_state = super().train(clone_state_dict(global_state))
        return ClientUpdate(
            client_id=int(self.client_idx),
            weight=float(self.get_sample_number()),
            state_dict=clone_state_dict(local_state),
            local_metrics=dict(self.model_trainer.last_train_metrics),
        )
