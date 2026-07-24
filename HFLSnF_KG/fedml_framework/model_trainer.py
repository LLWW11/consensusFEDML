"""基于FedML ClientTrainer接口实现的两层GCN本地训练器。"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from fedml.core import ClientTrainer
from torch.nn import functional as F

from FedGCN_fedml.data import FederatedGraphData, LocalGraphPartition

from ..core.types import clone_state_dict
from ..tasks.gcn.adapter import classification_accuracy


class FedMLGCNModelTrainer(ClientTrainer):
    """让FedML客户端能够训练需要特征和邻接矩阵双输入的GCN。"""

    def __init__(self, model: torch.nn.Module, args):
        """初始化FedML训练器、共享GCN模型和最近一次本地指标。"""

        super().__init__(model, args)
        self.last_train_metrics: Dict[str, float] = {}

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """返回共享GCN当前参数的CPU深拷贝。"""

        return clone_state_dict(self.model.state_dict(), to_cpu=True)

    def set_model_params(
        self, model_parameters: Dict[str, torch.Tensor]
    ) -> None:
        """把服务器下发参数加载到FedML共享GCN模型。"""

        self.model.load_state_dict(model_parameters)

    @staticmethod
    def _partition_to_device(
        partition: LocalGraphPartition, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """把客户端局部诱导子图的全部训练张量移动到目标设备。"""

        return (
            partition.features.to(device),
            partition.adjacency.to(device),
            partition.labels.to(device),
            partition.train_indices.to(device),
        )

    def train(self, train_data, device, args) -> Dict[str, float]:
        """按FedML ClientTrainer约定在一个局部诱导子图上训练GCN。"""

        if not isinstance(train_data, LocalGraphPartition):
            raise TypeError(
                "FedMLGCNModelTrainer只接受LocalGraphPartition，实际为{}".format(
                    type(train_data).__name__
                )
            )
        if train_data.train_node_count <= 0:
            self.last_train_metrics = {
                "train_loss": float("nan"),
                "train_node_count": 0.0,
            }
            return dict(self.last_train_metrics)

        device = torch.device(device)
        self.model.to(device)
        self.model.train()
        learning_rate = float(
            getattr(args, "learning_rate", getattr(args, "lr", 0.5))
        )
        weight_decay = float(getattr(args, "weight_decay", 5e-4))
        local_epochs = int(getattr(args, "epochs", 1))
        if learning_rate <= 0.0:
            raise ValueError("learning_rate 必须大于0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay 不能小于0")
        if local_epochs <= 0:
            raise ValueError("epochs 必须大于0")

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        features, adjacency, labels, train_indices = (
            self._partition_to_device(train_data, device)
        )
        final_loss = float("nan")
        for _ in range(local_epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(features, adjacency)
            loss = F.nll_loss(
                output.index_select(0, train_indices),
                labels.index_select(0, train_indices),
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

        self.last_train_metrics = {
            "train_loss": final_loss,
            "train_node_count": float(train_data.train_node_count),
        }
        return dict(self.last_train_metrics)

    def test(self, test_data, device, args) -> Dict[str, float]:
        """按FedML ClientTrainer约定评估一个客户端的局部标注节点。"""

        del args
        if not isinstance(test_data, LocalGraphPartition):
            raise TypeError(
                "FedMLGCNModelTrainer只接受LocalGraphPartition，实际为{}".format(
                    type(test_data).__name__
                )
            )
        if test_data.train_node_count <= 0:
            return {
                "test_total": 0,
                "test_correct": 0,
                "test_loss": 0.0,
            }

        device = torch.device(device)
        self.model.to(device)
        self.model.eval()
        features, adjacency, labels, indices = self._partition_to_device(
            test_data, device
        )
        with torch.no_grad():
            output = self.model(features, adjacency)
            loss = F.nll_loss(
                output.index_select(0, indices),
                labels.index_select(0, indices),
                reduction="sum",
            )
            predictions = output.index_select(0, indices).argmax(dim=1)
            targets = labels.index_select(0, indices)
            correct = int((predictions == targets).sum().item())
        return {
            "test_total": int(indices.numel()),
            "test_correct": correct,
            "test_loss": float(loss.detach().cpu().item()),
        }

    @staticmethod
    def _evaluate_split(
        output: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[float, float]:
        """计算完整图一个标准划分上的NLL损失和准确率。"""

        if int(indices.numel()) <= 0:
            raise ValueError("全局评估索引不能为空")
        loss = F.nll_loss(
            output.index_select(0, indices),
            labels.index_select(0, indices),
        )
        accuracy = classification_accuracy(output, labels, indices)
        return float(loss.detach().cpu().item()), float(accuracy)

    def evaluate_full_graph(
        self, dataset: FederatedGraphData, device: torch.device
    ) -> Dict[str, float]:
        """在完整图训练、验证和测试索引上评估共享GCN模型。"""

        device = torch.device(device)
        self.model.to(device)
        self.model.eval()
        features = dataset.features.to(device)
        adjacency = dataset.adjacency.to(device)
        labels = dataset.labels.to(device)
        idx_train = dataset.idx_train.to(device)
        idx_val = dataset.idx_val.to(device)
        idx_test = dataset.idx_test.to(device)
        with torch.no_grad():
            output = self.model(features, adjacency)
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
