"""稠密FedAvg及边缘—云可合并统计测试。"""

from __future__ import annotations

import unittest

import torch

from HFLSnF_KG_v2.core.aggregation import DenseFedAvgAggregator
from HFLSnF_KG_v2.core.types import ClientUpdate


def make_update(
    client_id: int, weight: float, value: float, counter: int = 1
) -> ClientUpdate:
    """构造包含一个浮点参数和一个整数缓冲区的客户端更新。"""

    return ClientUpdate(
        client_id=client_id,
        weight=weight,
        state_dict={
            "weight": torch.tensor([value], dtype=torch.float32),
            "counter": torch.tensor(counter, dtype=torch.long),
        },
    )


class DenseFedAvgAggregatorTest(unittest.TestCase):
    """验证稠密加权平均及两级合并的数学口径。"""

    def setUp(self) -> None:
        """为每个测试创建独立聚合器。"""

        self.aggregator = DenseFedAvgAggregator()

    def test_weighted_average_matches_manual_result(self) -> None:
        """验证聚合结果与手工标量加权计算一致。"""

        state = self.aggregator.aggregate(
            [
                make_update(0, 1.0, 2.0),
                make_update(1, 3.0, 4.0),
            ]
        )
        self.assertTrue(
            torch.allclose(state["weight"], torch.tensor([3.5]))
        )
        self.assertEqual(int(state["counter"].item()), 1)

    def test_hierarchical_merge_matches_direct_fedavg(self) -> None:
        """验证同参与者和权重下两级FedAvg与直接FedAvg一致。"""

        updates = [
            make_update(0, 1.0, 1.0),
            make_update(1, 2.0, 3.0),
            make_update(2, 4.0, 7.0),
        ]
        direct_state = self.aggregator.aggregate(updates)
        edge_left = self.aggregator.accumulate(updates[:2])
        edge_right = self.aggregator.accumulate(updates[2:])
        cloud_stats = self.aggregator.merge([edge_left, edge_right])
        hierarchical_state = self.aggregator.finalize(cloud_stats)
        self.assertTrue(
            torch.allclose(
                direct_state["weight"], hierarchical_state["weight"]
            )
        )
        self.assertEqual(
            int(direct_state["counter"].item()),
            int(hierarchical_state["counter"].item()),
        )

    def test_duplicate_client_across_edges_is_rejected(self) -> None:
        """验证同一客户端不能通过两个边缘组重复贡献。"""

        left = self.aggregator.accumulate([make_update(0, 1.0, 1.0)])
        right = self.aggregator.accumulate([make_update(0, 2.0, 2.0)])
        with self.assertRaisesRegex(ValueError, "重复贡献"):
            self.aggregator.merge([left, right])

    def test_inconsistent_integer_buffer_is_rejected(self) -> None:
        """验证无法加权的整数缓冲区不一致时快速报错。"""

        with self.assertRaisesRegex(ValueError, "非浮点缓冲区"):
            self.aggregator.aggregate(
                [
                    make_update(0, 1.0, 1.0, counter=1),
                    make_update(1, 1.0, 2.0, counter=2),
                ]
            )


if __name__ == "__main__":
    unittest.main()
