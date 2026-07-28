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
        self._parameter_masks, self._row_counts = (
            self._build_row_ownership()
        )

    def _build_row_ownership(
        self,
    ) -> tuple:
        """根据本地正三元组建立实体及关系行掩码和出现次数。"""

        model = self.model_trainer.model
        num_entities = int(model.num_entities)
        num_relations = int(model.num_relations)
        triples = self.partition.train_triples.detach().cpu()

        # FedE存在向量只由正训练事实定义，不让随机负样本取得行所有权。
        entity_values = torch.cat(
            [triples[:, 0], triples[:, 2]], dim=0
        )
        relation_values = triples[:, 1]
        entity_counts = torch.bincount(
            entity_values, minlength=num_entities
        ).to(dtype=torch.float32)
        relation_counts = torch.bincount(
            relation_values, minlength=num_relations
        ).to(dtype=torch.float32)
        masks = {
            "entity_embeddings.weight": entity_counts > 0,
            "relation_embeddings.weight": relation_counts > 0,
        }
        row_counts = {
            "entity_embeddings.weight": entity_counts,
            "relation_embeddings.weight": relation_counts,
        }
        return masks, row_counts

    def get_parameter_masks(self) -> Dict[str, torch.Tensor]:
        """返回本客户端由正训练事实确定的参数行所有权副本。"""

        return {
            name: values.detach().clone()
            for name, values in self._parameter_masks.items()
        }

    def get_row_counts(self) -> Dict[str, torch.Tensor]:
        """返回实体及关系在本地正训练事实中的出现次数副本。"""

        return {
            name: values.detach().clone()
            for name, values in self._row_counts.items()
        }

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
            parameter_masks=self.get_parameter_masks(),
            row_counts=self.get_row_counts(),
            local_metrics=dict(self.model_trainer.last_train_metrics),
        )
