"""使用FedML FedAvg客户端生命周期的知识图谱客户端。"""

from __future__ import annotations

from typing import Dict

import torch
from fedml.simulation.sp.fedavg.client import Client as FedMLClient

from ..core.types import ClientUpdate, clone_state_dict
from ..tasks.kge.federated_data import KnowledgeGraphClientPartition
from .model_trainer import FedMLTransEModelTrainer


class FedMLTransEClient(FedMLClient):
    """从同一云端参数开始训练一个知识客户端的完整TransE。"""

    def __init__(
        self,
        partition: KnowledgeGraphClientPartition,
        args,
        device: torch.device,
        model_trainer: FedMLTransEModelTrainer,
    ):
        """把客户端三元组分区绑定到FedML客户端基类。"""

        super().__init__(
            client_idx=int(partition.client_id),
            local_training_data=partition,
            local_test_data=partition,
            local_sample_number=int(partition.triple_count),
            args=args,
            device=torch.device(device),
            model_trainer=model_trainer,
        )
        self.partition = partition

    def train_from_global(
        self,
        global_state: Dict[str, torch.Tensor],
        round_index: int,
    ) -> ClientUpdate:
        """调用FedML基类完成参数下发、本地训练和参数回收。"""

        self.model_trainer.set_id(int(self.client_idx))
        self.args.round_idx = int(round_index)
        local_state = super().train(clone_state_dict(global_state))
        return ClientUpdate(
            client_id=int(self.client_idx),
            weight=float(self.get_sample_number()),
            state_dict=clone_state_dict(local_state),
            local_metrics=dict(self.model_trainer.last_train_metrics),
        )
