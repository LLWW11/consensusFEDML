"""GCN前向传播、FedAvg和模拟训练测试。"""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from FedGCN_fedml.data import FederatedGraphData, LocalGraphPartition
from FedGCN_fedml.model import GCN
from FedGCN_fedml.simulator import FedGCNSimulator, aggregate_state_dicts


class ModelAndAggregationTest(unittest.TestCase):
    """验证模型输出和参数聚合的核心数学行为。"""

    def test_gcn_forward_outputs_valid_log_probabilities(self):
        """GCN输出应覆盖全部节点和类别，指数化后每行概率和为一。"""

        model = GCN(input_dim=3, hidden_dim=4, output_dim=2, dropout=0.0)
        features = torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
        )
        adjacency = torch.eye(3)
        output = model(features, adjacency)
        self.assertEqual(tuple(output.shape), (3, 2))
        self.assertTrue(torch.allclose(output.exp().sum(dim=1), torch.ones(3)))

    def test_weighted_fedavg_matches_manual_result(self):
        """FedAvg结果必须严格等于按本地标注节点数手工加权的结果。"""

        first = {"weight": torch.tensor([1.0, 3.0])}
        second = {"weight": torch.tensor([5.0, 7.0])}
        averaged = aggregate_state_dicts([(1, first), (3, second)])
        expected = (first["weight"] + 3.0 * second["weight"]) / 4.0
        self.assertTrue(torch.allclose(averaged["weight"], expected))

    def test_empty_update_collection_is_rejected(self):
        """没有任何有效客户端贡献时不得生成伪全局模型。"""

        with self.assertRaises(ValueError):
            aggregate_state_dicts([])


class LightweightSimulatorTest(unittest.TestCase):
    """使用合成小图验证空标注客户端跳过和结果文件落盘。"""

    def _build_dataset(self) -> FederatedGraphData:
        """构造一份包含一个空标注客户端的四节点联邦图。"""

        features = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.8, 0.2, 0.4],
                [0.0, 1.0, 0.5],
                [0.2, 0.8, 0.4],
            ],
            dtype=torch.float32,
        )
        adjacency = torch.eye(4, dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        first_nodes = torch.tensor([0, 1], dtype=torch.long)
        second_nodes = torch.tensor([2, 3], dtype=torch.long)
        partitions = (
            LocalGraphPartition(
                client_id=0,
                node_indices=first_nodes,
                features=features.index_select(0, first_nodes),
                adjacency=torch.eye(2),
                labels=labels.index_select(0, first_nodes),
                train_indices=torch.tensor([0], dtype=torch.long),
            ),
            LocalGraphPartition(
                client_id=1,
                node_indices=second_nodes,
                features=features.index_select(0, second_nodes),
                adjacency=torch.eye(2),
                labels=labels.index_select(0, second_nodes),
                train_indices=torch.empty(0, dtype=torch.long),
            ),
        )
        return FederatedGraphData(
            dataset_name="synthetic",
            features=features,
            adjacency=adjacency,
            labels=labels,
            idx_train=torch.tensor([0], dtype=torch.long),
            idx_val=torch.tensor([1], dtype=torch.long),
            idx_test=torch.tensor([2, 3], dtype=torch.long),
            partitions=partitions,
            num_classes=2,
        )

    def test_single_round_skips_unlabeled_client_and_writes_outputs(self):
        """模拟器应跳过空标注客户端，并生成完整可解析的轻量结果。"""

        args = SimpleNamespace(
            comm_round=1,
            epochs=1,
            learning_rate=0.1,
            weight_decay=0.0,
            aggregation_weight_basis="labeled_train_nodes",
            client_num_in_total=2,
            client_num_per_round=2,
            random_seed=3,
        )
        torch.manual_seed(3)
        simulator = FedGCNSimulator(
            args=args,
            device=torch.device("cpu"),
            dataset=self._build_dataset(),
            model=GCN(input_dim=3, hidden_dim=4, output_dim=2, dropout=0.0),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_dir = Path(temporary_directory)
            summary = simulator.train(result_dir)
            for filename in (
                "metrics.csv",
                "summary.json",
                "config_snapshot.json",
                "partition_summary.json",
                "final_model.pt",
            ):
                self.assertTrue((result_dir / filename).is_file(), filename)
            with (result_dir / "metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["contributor_count"]), 1)
            self.assertTrue(math.isfinite(float(rows[0]["train_loss"])))
            with (result_dir / "summary.json").open("r", encoding="utf-8") as handle:
                saved_summary = json.load(handle)
            self.assertEqual(saved_summary["client_count"], 2)
            self.assertTrue(math.isfinite(summary["final_test_loss"]))


if __name__ == "__main__":
    unittest.main()

