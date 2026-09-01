"""验证V5图语义HFLnoSnF与FLnoSnF六实验合同。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_topology_extension as runner
from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import (
    load_flat_config,
)
from HFLSnF_KG_v5.tasks.kge.graph_semantic_topology_extension import (
    ARM_ORDER,
    RESULT_ROOT,
    SEEDS,
    expected_flat_config,
    scenarios_from_contract,
)


class GraphSemanticTopologyExtensionTest(unittest.TestCase):
    """检查六场景配置配对关系、顺序和失败停止行为。"""

    def test_six_scenarios_follow_seed_first_order(self) -> None:
        """六场景必须按种子优先且每种子包含两个拓扑臂。"""

        scenarios = scenarios_from_contract()
        self.assertEqual(len(scenarios), 6)
        self.assertEqual(
            [(item.seed, item.arm) for item in scenarios],
            [(seed, arm) for seed in SEEDS for arm in ARM_ORDER],
        )

    def test_configs_are_full_paired_derivatives(self) -> None:
        """每份实际YAML必须等于同种子HFLSnF配置的受控派生。"""

        by_seed = {}
        for scenario in scenarios_from_contract():
            actual = load_flat_config(scenario.config_path)
            self.assertEqual(actual, expected_flat_config(scenario))
            self.assertTrue(actual["require_cuda"])
            self.assertEqual(actual["result_root"], RESULT_ROOT)
            self.assertEqual(
                actual["expected_partition_hash"],
                scenario.partition_hash,
            )
            by_seed.setdefault(scenario.seed, set()).add(
                actual["expected_partition_hash"]
            )
        self.assertTrue(all(len(values) == 1 for values in by_seed.values()))

    def test_arm_topology_fields_are_distinct(self) -> None:
        """HFLnoSnF保留边缘层，FLnoSnF必须移除边缘层。"""

        configs = {
            scenario.arm: expected_flat_config(scenario)
            for scenario in scenarios_from_contract()
            if scenario.seed == 42
        }
        self.assertEqual(configs["hflnosnf"]["topology_architecture"], "hfl")
        self.assertFalse(configs["hflnosnf"]["topology_snf"])
        self.assertEqual(configs["hflnosnf"]["topology_edge_mode"], "fixed")
        self.assertEqual(configs["hflnosnf"]["edge_num"], 6)
        self.assertEqual(configs["flnosnf"]["topology_architecture"], "fl")
        self.assertFalse(configs["flnosnf"]["topology_snf"])
        self.assertEqual(configs["flnosnf"]["topology_edge_mode"], "none")
        self.assertEqual(configs["flnosnf"]["edge_num"], 1)

    def test_first_failure_stops_remaining_runs(self) -> None:
        """首个训练失败后必须保留恢复状态并停止其余五项。"""

        payload = runner._empty_batch_payload()
        failure = runner.TopologyExtensionRunError("模拟训练失败")
        with mock.patch.object(
            runner,
            "_load_batch_manifest",
            return_value=payload,
        ), mock.patch.object(
            runner,
            "_save_batch_manifest",
        ), mock.patch.object(
            runner,
            "_run_scenario",
            side_effect=failure,
        ):
            status = runner.run_training_batch(Path("unused.json"))
        self.assertEqual(status, 1)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["entries"][0]["status"], "failed")
        self.assertTrue(
            all(
                item["status"] == "pending"
                for item in payload["entries"][1:]
            )
        )


if __name__ == "__main__":
    unittest.main()
