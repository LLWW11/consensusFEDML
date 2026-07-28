"""静态和逐轮拓扑提供器测试。"""

from __future__ import annotations

import unittest

from HFLSnF_KG.core.topology import (
    SequenceTopologyProvider,
    StaticTopologyProvider,
)


class TopologyProviderTest(unittest.TestCase):
    """验证客户端分组、空轮和拓扑范围检查。"""

    def test_round_robin_assigns_every_client_once(self) -> None:
        """验证轮转分组覆盖全部客户端且不存在重复。"""

        provider = StaticTopologyProvider.round_robin(
            client_ids=[0, 1, 2, 3, 4], group_count=2
        )
        topology = provider.get_round(3)
        self.assertEqual(
            topology.group_to_client_indexes,
            {0: (0, 2, 4), 1: (1, 3)},
        )
        self.assertEqual(topology.active_client_indexes, (0, 1, 2, 3, 4))
        self.assertEqual(topology.source_round_index, 3)

    def test_duplicate_client_across_groups_is_rejected(self) -> None:
        """验证同一客户端不能同时属于两个边缘组。"""

        with self.assertRaisesRegex(ValueError, "多个边缘组"):
            StaticTopologyProvider({0: [0, 1], 1: [1, 2]})

    def test_sequence_provider_supports_empty_round(self) -> None:
        """验证逐轮拓扑允许使用空分组表示零参与通信轮。"""

        provider = SequenceTopologyProvider([{0: [0]}, {}])
        empty_round = provider.get_round(1)
        self.assertEqual(empty_round.participant_count, 0)
        self.assertEqual(empty_round.active_client_indexes, ())
        self.assertEqual(empty_round.group_to_client_indexes, {})

    def test_sequence_provider_rejects_out_of_range_round(self) -> None:
        """验证访问不存在的通信轮时给出明确错误。"""

        provider = SequenceTopologyProvider([{0: [0]}])
        with self.assertRaises(IndexError):
            provider.get_round(1)


if __name__ == "__main__":
    unittest.main()
