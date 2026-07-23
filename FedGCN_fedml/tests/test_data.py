"""图数据划分和局部诱导子图测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from FedGCN_fedml.data import (
    build_federated_graph_data,
    partition_graph_nodes,
)


class GraphPartitionTest(unittest.TestCase):
    """验证节点划分的覆盖性、确定性和局部图切片。"""

    def test_partition_is_complete_disjoint_and_deterministic(self):
        """同一随机种子应产生完整、无重叠且可复现的划分。"""

        labels = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        first = partition_graph_nodes(labels, client_count=4, iid_fraction=0.5, seed=7)
        second = partition_graph_nodes(labels, client_count=4, iid_fraction=0.5, seed=7)
        self.assertEqual(len(first), 4)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertGreater(len(left), 0)
        flattened = np.concatenate(first)
        np.testing.assert_array_equal(np.sort(flattened), np.arange(len(labels)))
        self.assertEqual(len(np.unique(flattened)), len(labels))

    def test_invalid_partition_arguments_are_rejected(self):
        """非法设备数和IID比例必须立即报错。"""

        with self.assertRaises(ValueError):
            partition_graph_nodes([0, 1], client_count=0, iid_fraction=0.5, seed=0)
        with self.assertRaises(ValueError):
            partition_graph_nodes([0, 1], client_count=2, iid_fraction=1.1, seed=0)

    @mock.patch("FedGCN_fedml.data.load_planetoid_graph")
    def test_local_graph_matches_global_induced_subgraph(self, mocked_loader):
        """局部邻接矩阵必须等于完整邻接矩阵按本地节点切片的结果。"""

        features = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        adjacency = torch.arange(36, dtype=torch.float32).reshape(6, 6)
        labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        idx_train = torch.tensor([0, 2, 4], dtype=torch.long)
        idx_val = torch.tensor([1], dtype=torch.long)
        idx_test = torch.tensor([3, 5], dtype=torch.long)
        mocked_loader.return_value = (
            features,
            adjacency,
            labels,
            idx_train,
            idx_val,
            idx_test,
        )

        dataset = build_federated_graph_data(
            dataset_name="cora",
            data_dir=Path("unused"),
            client_count=2,
            iid_fraction=0.5,
            seed=11,
        )
        all_nodes = []
        for partition in dataset.partitions:
            nodes = partition.node_indices
            expected_adjacency = adjacency.index_select(0, nodes).index_select(1, nodes)
            self.assertTrue(torch.equal(partition.adjacency, expected_adjacency))
            expected_train_global = np.intersect1d(
                nodes.numpy(), idx_train.numpy(), assume_unique=True
            )
            actual_train_global = nodes.index_select(0, partition.train_indices).numpy()
            np.testing.assert_array_equal(actual_train_global, expected_train_global)
            all_nodes.extend(nodes.tolist())
        self.assertEqual(sorted(all_nodes), list(range(6)))


if __name__ == "__main__":
    unittest.main()

