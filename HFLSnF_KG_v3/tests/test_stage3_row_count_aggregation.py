"""验证阶段3逐行计数加权聚合的黄金结果和两级等价性。"""

from __future__ import annotations

import unittest

import torch

from HFLSnF_KG_v3.core.aggregation import (
    RowCountWeightedFedAvgAggregator,
)
from HFLSnF_KG_v3.core.types import ClientUpdate


def _state(entity_rows, relation_rows):
    """构造只包含两张TransE嵌入表的测试模型状态。"""

    return {
        "entity_embeddings.weight": torch.tensor(
            entity_rows, dtype=torch.float32
        ),
        "relation_embeddings.weight": torch.tensor(
            relation_rows, dtype=torch.float32
        ),
    }


def _update(
    client_id: int,
    state,
    entity_counts,
    relation_counts,
) -> ClientUpdate:
    """构造携带实体行和关系行正事实出现次数的客户端更新。"""

    return ClientUpdate(
        client_id=client_id,
        weight=1.0,
        state_dict=state,
        row_counts={
            "entity_embeddings.weight": torch.tensor(
                entity_counts, dtype=torch.float32
            ),
            "relation_embeddings.weight": torch.tensor(
                relation_counts, dtype=torch.float32
            ),
        },
    )


class RowCountWeightedAggregationTest(unittest.TestCase):
    """检查计数分子分母、无人行回退和边缘—云无损合并。"""

    def test_manual_count_weighted_rows_and_fallback(self) -> None:
        """逐元素核对手工计数加权结果和无人更新行回退。"""

        aggregator = RowCountWeightedFedAvgAggregator()
        global_state = _state(
            [[10.0], [20.0], [30.0]],
            [[100.0], [200.0]],
        )
        update_zero = _update(
            0,
            _state(
                [[2.0], [4.0], [999.0]],
                [[10.0], [999.0]],
            ),
            [1.0, 3.0, 0.0],
            [4.0, 0.0],
        )
        update_one = _update(
            1,
            _state(
                [[10.0], [20.0], [888.0]],
                [[30.0], [40.0]],
            ),
            [3.0, 1.0, 0.0],
            [2.0, 2.0],
        )

        statistics = aggregator.accumulate(
            [update_zero, update_one]
        )
        result = aggregator.finalize(statistics, global_state)
        summary = aggregator.summarize(statistics)

        torch.testing.assert_close(
            result["entity_embeddings.weight"],
            torch.tensor([[8.0], [8.0], [30.0]]),
        )
        torch.testing.assert_close(
            result["relation_embeddings.weight"],
            torch.tensor([[100.0 / 6.0], [40.0]]),
        )
        entity_summary = summary["entity_embeddings.weight"]
        relation_summary = summary["relation_embeddings.weight"]
        self.assertEqual(entity_summary["updated_row_count"], 2)
        self.assertEqual(entity_summary["fallback_row_count"], 1)
        self.assertEqual(entity_summary["total_row_occurrences"], 8.0)
        self.assertEqual(entity_summary["mean_row_contributors"], 2.0)
        self.assertEqual(relation_summary["updated_row_count"], 2)
        self.assertEqual(relation_summary["total_row_occurrences"], 8.0)
        self.assertEqual(
            relation_summary["mean_row_contributors"], 1.5
        )

    def test_hierarchical_merge_equals_direct_count_weighting(self) -> None:
        """确认边缘统计合并与全部客户端一次性计数加权完全等价。"""

        aggregator = RowCountWeightedFedAvgAggregator()
        global_state = _state(
            [[10.0], [20.0], [30.0]],
            [[100.0], [200.0]],
        )
        updates = [
            _update(
                0,
                _state(
                    [[1.0], [2.0], [3.0]],
                    [[11.0], [12.0]],
                ),
                [1.0, 2.0, 0.0],
                [3.0, 0.0],
            ),
            _update(
                1,
                _state(
                    [[5.0], [6.0], [7.0]],
                    [[21.0], [22.0]],
                ),
                [0.0, 1.0, 4.0],
                [1.0, 2.0],
            ),
            _update(
                2,
                _state(
                    [[9.0], [10.0], [11.0]],
                    [[31.0], [32.0]],
                ),
                [2.0, 0.0, 1.0],
                [0.0, 5.0],
            ),
        ]
        direct_statistics = aggregator.accumulate(updates)
        edge_statistics = [
            aggregator.accumulate(updates[:2]),
            aggregator.accumulate(updates[2:]),
        ]
        hierarchical_statistics = aggregator.merge(edge_statistics)
        direct = aggregator.finalize(
            direct_statistics, global_state
        )
        hierarchical = aggregator.finalize(
            hierarchical_statistics, global_state
        )

        self.assertEqual(
            set(hierarchical_statistics.contributor_ids),
            {0, 1, 2},
        )
        for name in direct:
            torch.testing.assert_close(
                direct[name],
                hierarchical[name],
                rtol=0.0,
                atol=0.0,
            )
        for name in direct_statistics.row_denominators:
            torch.testing.assert_close(
                direct_statistics.row_denominators[name],
                hierarchical_statistics.row_denominators[name],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                direct_statistics.row_contributor_counts[name],
                hierarchical_statistics.row_contributor_counts[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_missing_row_counts_fail_fast(self) -> None:
        """确认客户端漏传行出现次数时聚合在训练前明确失败。"""

        update = ClientUpdate(
            client_id=0,
            weight=1.0,
            state_dict=_state([[1.0]], [[2.0]]),
        )
        with self.assertRaisesRegex(
            ValueError, "没有为参数.*提供逐行出现次数"
        ):
            RowCountWeightedFedAvgAggregator().accumulate([update])


if __name__ == "__main__":
    unittest.main()
