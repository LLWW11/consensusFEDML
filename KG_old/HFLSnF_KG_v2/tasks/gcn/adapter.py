"""把已经验证的FedGCN节点分类任务接入任务无关分层模拟器。"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from FedGCN_fedml.data import FederatedGraphData, LocalGraphPartition
from FedGCN_fedml.model import GCN

from ...core.types import ClientUpdate, clone_state_dict
from ..base import FederatedTask


def classification_accuracy(
    output: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor
) -> float:
    """计算指定节点索引上的分类准确率。"""

    if int(indices.numel()) <= 0:
        raise ValueError("评估节点索引不能为空")
    predictions = output.index_select(0, indices).argmax(dim=1)
    targets = labels.index_select(0, indices)
    correct = int((predictions == targets).sum().item())
    return float(correct) / float(indices.numel())


class CoraGCNTask(FederatedTask):
    """在局部诱导子图上训练两层GCN的联邦任务适配器。"""

    def __init__(
        self,
        dataset: FederatedGraphData,
        device: torch.device,
        hidden_dim: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        seed: int,
    ):
        """保存图数据、模型超参数和设备，并创建初始全局GCN。"""

        self.dataset = dataset
        self.device = torch.device(device)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim 必须大于0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout 必须位于[0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须大于0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay 不能小于0")

        self._partitions = {
            int(partition.client_id): partition
            for partition in dataset.partitions
        }
        if len(self._partitions) != len(dataset.partitions):
            raise ValueError("GCN客户端分区编号存在重复")
        self._model = self._create_model().to(self.device)

    @property
    def task_name(self) -> str:
        """返回包含实际数据集名称的任务标识。"""

        return "gcn_node_classification_{}".format(self.dataset.dataset_name)

    @property
    def client_ids(self) -> Sequence[int]:
        """返回按编号排序的全部GCN客户端。"""

        return tuple(sorted(self._partitions.keys()))

    def _create_model(self) -> GCN:
        """根据完整图维度创建一个结构一致的两层GCN。"""

        return GCN(
            input_dim=self.dataset.num_features,
            hidden_dim=self.hidden_dim,
            output_dim=self.dataset.num_classes,
            dropout=self.dropout,
        )

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回当前全局GCN参数的CPU深拷贝。"""

        return clone_state_dict(self._model.state_dict(), to_cpu=True)

    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """把聚合后的CPU模型参数加载到全局GCN。"""

        self._model.load_state_dict(state_dict)
        self._model.to(self.device)

    def _copy_partition_to_device(
        self, partition: LocalGraphPartition
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """把一个客户端的局部子图统一迁移到当前运行设备。"""

        return (
            partition.features.to(self.device),
            partition.adjacency.to(self.device),
            partition.labels.to(self.device),
            partition.train_indices.to(self.device),
        )

    def train_client(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        local_epochs: int,
        round_index: int,
    ) -> Optional[ClientUpdate]:
        """从同一全局模型开始，在指定客户端局部诱导子图上完成本地训练。"""

        client_id = int(client_id)
        local_epochs = int(local_epochs)
        if client_id not in self._partitions:
            raise KeyError("找不到GCN客户端{}".format(client_id))
        if local_epochs <= 0:
            raise ValueError("local_epochs 必须大于0")
        partition = self._partitions[client_id]
        if partition.train_node_count <= 0:
            return None

        # 深拷贝不会额外消耗参数初始化随机数，可与现有FedGCN顺序模拟保持一致。
        del round_index
        local_model = copy.deepcopy(self._model)
        local_model.load_state_dict(global_state)
        local_model.to(self.device)
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        features, adjacency, labels, train_indices = (
            self._copy_partition_to_device(partition)
        )

        final_loss = float("nan")
        for _ in range(local_epochs):
            local_model.train()
            optimizer.zero_grad()
            output = local_model(features, adjacency)
            loss = F.nll_loss(
                output.index_select(0, train_indices),
                labels.index_select(0, train_indices),
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

        return ClientUpdate(
            client_id=client_id,
            weight=float(partition.train_node_count),
            state_dict=clone_state_dict(local_model.state_dict(), to_cpu=True),
            local_metrics={"train_loss": final_loss},
        )

    @staticmethod
    def _evaluate_split(
        output: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[float, float]:
        """计算完整图中一个标准数据划分的NLL损失和准确率。"""

        if int(indices.numel()) <= 0:
            raise ValueError("全局评估索引不能为空")
        loss = F.nll_loss(
            output.index_select(0, indices),
            labels.index_select(0, indices),
        )
        accuracy = classification_accuracy(output, labels, indices)
        return float(loss.detach().cpu().item()), float(accuracy)

    def evaluate_global(self) -> Dict[str, float]:
        """在完整图训练、验证和测试节点上评估当前全局GCN。"""

        self._model.eval()
        features = self.dataset.features.to(self.device)
        adjacency = self.dataset.adjacency.to(self.device)
        labels = self.dataset.labels.to(self.device)
        idx_train = self.dataset.idx_train.to(self.device)
        idx_val = self.dataset.idx_val.to(self.device)
        idx_test = self.dataset.idx_test.to(self.device)
        with torch.no_grad():
            output = self._model(features, adjacency)
            train_loss, train_accuracy = self._evaluate_split(
                output, labels, idx_train
            )
            val_loss, val_accuracy = self._evaluate_split(
                output, labels, idx_val
            )
            test_loss, test_accuracy = self._evaluate_split(
                output, labels, idx_test
            )
        return {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

    def partition_summary(self) -> Dict[str, object]:
        """返回完整图规模和每个客户端局部子图的可复核摘要。"""

        return {
            "task": self.task_name,
            "dataset": self.dataset.dataset_name,
            "node_count": self.dataset.num_nodes,
            "feature_count": self.dataset.num_features,
            "class_count": self.dataset.num_classes,
            "client_count": len(self.dataset.partitions),
            "clients": [
                {
                    "client_id": int(partition.client_id),
                    "node_count": int(partition.node_count),
                    "labeled_train_node_count": int(
                        partition.train_node_count
                    ),
                    "node_indices": partition.node_indices.tolist(),
                }
                for partition in self.dataset.partitions
            ],
        }
