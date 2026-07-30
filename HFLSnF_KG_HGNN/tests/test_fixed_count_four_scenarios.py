"""验证固定人数调度、四份正式YAML和唯一烟雾配置。"""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from HFLSnF_KG_v3.tasks.kge.fixed_count_four_scenarios import (
    DYNAMIC_SCENARIOS,
    SCENARIOS,
    _dynamic_schedule_statistics,
    build_dynamic_mat_provider,
    build_scenario_provider,
    load_flat_config,
    validate_four_scenario_configs,
)


class FixedCountTopologyAndConfigTest(unittest.TestCase):
    """检查四个实验臂的人数、分组、确定性和YAML字段。"""

    def test_every_round_uses_exact_yaml_participant_count(self) -> None:
        """确认四个场景前150轮人数和组数都严格固定。"""

        for scenario in SCENARIOS:
            provider = build_scenario_provider(scenario)
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
                    len(topology.group_to_client_indexes)
                    == scenario.group_count
                    for topology in topologies
                )
            )

    def test_no_snf_round_robin_is_deterministic_and_balanced(self) -> None:
        """确认noSnF不读MAT且150轮长期参与频次近似均衡。"""

        for scenario in SCENARIOS:
            if scenario.snf_enabled:
                continue
            first_provider = build_scenario_provider(scenario)
            second_provider = build_scenario_provider(scenario)
            first = [
                first_provider.get_round(index).active_client_indexes
                for index in range(150)
            ]
            second = [
                second_provider.get_round(index).active_client_indexes
                for index in range(150)
            ]
            self.assertEqual(first, second)
            counts = Counter(
                client_id
                for participants in first
                for client_id in participants
            )
            self.assertEqual(len(counts), 37)
            self.assertLessEqual(
                max(counts.values()) - min(counts.values()),
                1,
            )

    def test_snf_projection_retains_source_signal(self) -> None:
        """确认SnF在源人数不超目标时完整保留MAT选中客户端。"""

        for scenario in SCENARIOS:
            if not scenario.snf_enabled:
                continue
            provider = build_scenario_provider(scenario)
            source = provider._source_provider
            self.assertIsNotNone(source)
            checked = False
            for round_index in range(150):
                source_topology = source.get_round(round_index)
                projected = provider.get_round(round_index)
                if (
                    source_topology.participant_count
                    <= scenario.participant_count
                ):
                    self.assertTrue(
                        set(
                            source_topology.active_client_indexes
                        ).issubset(
                            projected.active_client_indexes
                        )
                    )
                    checked = True
            self.assertTrue(checked)

    def test_four_formal_configs_and_smoke_config_pass(self) -> None:
        """确认固定、动态正式YAML和唯一烟雾配置通过校验。"""

        report = validate_four_scenario_configs()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["fixed_formal_configs"]), 4)
        self.assertEqual(len(report["dynamic_formal_configs"]), 4)
        self.assertEqual(report["smoke_config"]["status"], "passed")
        for scenario in SCENARIOS:
            formal = load_flat_config(
                scenario_config_path(scenario.formal_config)
            )
            self.assertEqual(
                formal["client_num_per_round"],
                scenario.participant_count,
            )

    def test_dynamic_mat_replay_preserves_raw_schedule(self) -> None:
        """确认四组直接回放MAT原始人数、分组和调度哈希。"""

        for scenario in DYNAMIC_SCENARIOS:
            formal = load_flat_config(
                scenario_config_path(scenario.formal_config)
            )
            provider = build_dynamic_mat_provider(scenario)
            statistics = _dynamic_schedule_statistics(scenario, 150)
            self.assertEqual(formal["epochs"], 3)
            self.assertEqual(formal["eval_every"], 1)
            self.assertEqual(provider.describe()["provider_type"], "matlab_adapter")
            self.assertEqual(
                statistics["schedule_hash"],
                scenario.formal_hash,
            )
            self.assertEqual(
                statistics["participant_count_min"],
                scenario.participant_min,
            )
            self.assertEqual(
                statistics["participant_count_max"],
                scenario.participant_max,
            )
            self.assertEqual(
                statistics["group_count_min"],
                scenario.group_min,
            )
            self.assertEqual(
                statistics["group_count_max"],
                scenario.group_max,
            )
            self.assertEqual(
                statistics["unique_participant_set_count"],
                scenario.unique_participant_sets,
            )
            self.assertEqual(
                statistics["unique_topology_count"],
                scenario.unique_topologies,
            )


def scenario_config_path(config_name: str) -> Path:
    """返回测试文件对应工程中的正式配置绝对路径。"""

    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / config_name
    )


if __name__ == "__main__":
    unittest.main()
