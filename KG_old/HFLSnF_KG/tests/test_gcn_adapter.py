"""GCN任务适配器的合成小图训练与评估测试。"""

from __future__ import annotations

import math
import unittest

import torch

from FedGCN_fedml.data import FederatedGraphData, LocalGraphPartition

from HFLSnF_KG.tasks.gcn import CoraGCNTask


def build_synthetic_graph() -> FederatedGraphData:
    """构造两个客户端、六个节点的确定性合成联邦图。"""

    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
            [0.9, 0.0, 0.1],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    adjacency = torch.eye(6, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    first_nodes = torch.tensor([0, 1, 4], dtype=torch.long)
    second_nodes = torch.tensor([2, 3, 5], dtype=torch.long)

    partitions = (
        LocalGraphPartition(
            client_id=0,
            node_indices=first_nodes,
            features=features.index_select(0, first_nodes),
            adjacency=torch.eye(3, dtype=torch.float32),
            labels=labels.index_select(0, first_nodes),
            train_indices=torch.tensor([0], dtype=torch.long),
        ),
        LocalGraphPartition(
            client_id=1,
            node_indices=second_nodes,
            features=features.index_select(0, second_nodes),
            adjacency=torch.eye(3, dtype=torch.float32),
            labels=labels.index_select(0, second_nodes),
            train_indices=torch.tensor([0], dtype=torch.long),
        ),
    )
    return FederatedGraphData(
        dataset_name="synthetic",
        features=features,
        adjacency=adjacency,
        labels=labels,
        idx_train=torch.tensor([0, 2], dtype=torch.long),
        idx_val=torch.tensor([1, 3], dtype=torch.long),
        idx_test=torch.tensor([4, 5], dtype=torch.long),
        partitions=partitions,
        num_classes=2,
    )


class CoraGCNTaskTest(unittest.TestCase):
    """验证GCN适配器能够训练局部子图并评估完整图。"""

    def test_local_update_and_global_metrics_are_finite(self) -> None:
        """验证本地更新形状完整且完整图指标为有限值。"""

        task = CoraGCNTask(
            dataset=build_synthetic_graph(),
            device=torch.device("cpu"),
            hidden_dim=4,
            dropout=0.0,
            learning_rate=0.1,
            weight_decay=0.0,
            seed=7,
        )
        global_state = task.get_global_state()
        update = task.train_client(
            client_id=0,
            global_state=global_state,
            local_epochs=1,
            round_index=0,
        )
        self.assertIsNotNone(update)
        self.assertEqual(set(update.state_dict.keys()), set(global_state.keys()))
        self.assertEqual(update.weight, 1.0)
        self.assertTrue(math.isfinite(update.local_metrics["train_loss"]))

        task.set_global_state(update.state_dict)
        metrics = task.evaluate_global()
        self.assertEqual(
            set(metrics.keys()),
            {
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "test_loss",
                "test_accuracy",
            },
        )
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
