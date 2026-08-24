"""验证HFLKGE客户端人数单因素消融的配置与调度。"""

from __future__ import annotations

import unittest
from collections import Counter

from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import (
    _schedule_hash,
)
from HFLSnF_KG_v5.tasks.kge.hflkge_client_count_ablation import (
    SCENARIOS,
    build_provider,
    validate_configs,
)


class HFLKGEClientCountAblationTest(unittest.TestCase):
    """检查六组实验只改变人数且每轮随机选择参与客户端。"""

    def test_configs_only_change_count_and_identity_fields(self) -> None:
        """确认行为配置完全一致且每份调度哈希正确。"""

        report = validate_configs()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["behavior_configs_equal"])
        self.assertEqual(len(report["formal_configs"]), 6)

    def test_every_round_uses_exact_count_and_six_groups(self) -> None:
        """确认前150轮严格使用五档人数并平均分入六个HFL组。"""

        for scenario in SCENARIOS:
            provider = build_provider(scenario)
            topologies = [
                provider.get_round(round_index)
                for round_index in range(150)
            ]
            self.assertTrue(
                all(
                    topology.participant_count
                    == scenario.participant_count
                    for topology in topologies
                )
            )
            self.assertTrue(
                all(
                    len(topology.group_to_client_indexes) == 6
                    for topology in topologies
                )
            )
            expected_group_size = scenario.participant_count // 6
            self.assertTrue(
                all(
                    all(
                        len(client_ids) == expected_group_size
                        for client_ids in (
                            topology.group_to_client_indexes.values()
                        )
                    )
                    for topology in topologies
                )
            )
            self.assertEqual(
                _schedule_hash(provider, 150),
                scenario.formal_hash,
            )

    def test_random_participants_change_and_are_reproducible(self) -> None:
        """确认参与集合逐轮变化、覆盖全部客户端且同一种子可复现。"""

        for scenario in SCENARIOS:
            first_provider = build_provider(scenario)
            second_provider = build_provider(scenario)
            first = [
                first_provider.get_round(index).active_client_indexes
                for index in range(150)
            ]
            second = [
                second_provider.get_round(index).active_client_indexes
                for index in range(150)
            ]
            self.assertEqual(first, second)
            self.assertGreater(len(set(first)), 1)
            selection_counts = Counter(
                client_id
                for participants in first
                for client_id in participants
            )
            self.assertEqual(set(selection_counts), set(range(37)))

    def test_random_mode_does_not_read_mat_topology(self) -> None:
        """确认六组随机人数实验不依赖SnF或MAT客户端选择结果。"""

        for scenario in SCENARIOS:
            provider = build_provider(scenario)
            metadata = provider.describe()
            self.assertFalse(metadata["snf_enabled"])
            self.assertEqual(
                metadata["fixed_count_selection_mode"],
                "seeded_random",
            )
            self.assertIsNone(metadata["source_topology"])


if __name__ == "__main__":
    unittest.main()
