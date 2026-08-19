"""验证最终随机拓扑九组配置、调度和批次恢复合同。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from HFLSnF_KG_v4.run_final_stochastic_fedadam import (
    _load_manifest,
    create_batch_manifest,
    run_batch,
)
from HFLSnF_KG_v4.tasks.kge.final_stochastic_fedadam import (
    ARM_CONTRACTS,
    ROUND_COUNT,
    SCENARIOS,
    build_provider,
    schedule_statistics,
    validate_configs,
)


class FinalStochasticTopologyTest(unittest.TestCase):
    """检查固定随机预算、随机均衡分组和配置合同。"""

    def test_every_round_uses_exact_budget_and_balanced_groups(self) -> None:
        """确认150轮人数、组数、互斥覆盖和组规模均符合合同。"""

        for scenario in SCENARIOS:
            provider = build_provider(scenario)
            for round_index in range(ROUND_COUNT):
                topology = provider.get_round(round_index)
                groups = topology.group_to_client_indexes
                flattened = [
                    client_id
                    for client_ids in groups.values()
                    for client_id in client_ids
                ]
                self.assertEqual(
                    topology.participant_count,
                    scenario.contract.participant_count,
                )
                self.assertEqual(len(groups), scenario.contract.group_count)
                self.assertEqual(len(flattened), len(set(flattened)))
                self.assertEqual(set(flattened), set(topology.active_client_indexes))
                sizes = [len(client_ids) for client_ids in groups.values()]
                self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_provider_is_reproducible_and_has_no_mat_source(self) -> None:
        """确认相同种子可复现、不同种子不同且元数据没有MAT依赖。"""

        for scenario in SCENARIOS:
            first = build_provider(scenario)
            second = build_provider(scenario)
            first_rounds = [first.get_round(index) for index in range(ROUND_COUNT)]
            second_rounds = [second.get_round(index) for index in range(ROUND_COUNT)]
            self.assertEqual(first_rounds, second_rounds)
            metadata = first.describe()
            self.assertEqual(metadata["provider_type"], "fixed_count")
            self.assertFalse(metadata["snf_enabled"])
            self.assertIsNone(metadata["source_topology"])
            self.assertIsNone(metadata["mat_file"])
            self.assertIsNone(metadata["topology_util"])
            self.assertEqual(
                metadata["fixed_count_grouping_mode"],
                "seeded_random_balanced",
            )
        for arm in ARM_CONTRACTS:
            schedules = [
                schedule_statistics(scenario)["schedule_hash"]
                for scenario in SCENARIOS
                if scenario.arm == arm
            ]
            self.assertEqual(len(schedules), 3)
            self.assertEqual(len(set(schedules)), 3)

    def test_nine_configs_pass_without_constructing_mat_provider(self) -> None:
        """确认配置合同不需要构造MAT拓扑提供器。"""

        with mock.patch(
            "HFLSnF_KG_v4.core.topology.MatlabTopologyProvider.__init__",
            side_effect=AssertionError("随机实验禁止读取MAT"),
        ):
            report = validate_configs()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["formal_configs"]), 9)


class FinalStochasticBatchTest(unittest.TestCase):
    """检查批次清单创建、恢复跳过和完成状态。"""

    def test_resume_skips_passed_entry_and_finishes_remaining_entries(self) -> None:
        """确认恢复时跳过已通过项目并继续完成其余八项。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            result_root = Path(temporary_directory).resolve()
            manifest_path = create_batch_manifest(result_root=result_root)
            payload = _load_manifest(manifest_path, result_root=result_root)
            first_entry = payload["entries"][0]
            first_entry["status"] = "passed"
            payload["seed_contracts"]["42"]["initial_model_hash"] = "a" * 64
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            fake_result = result_root / "fake_result"
            fake_result.mkdir()

            def fake_validate_result(result_dir, scenario, expected_initial_model_hash=""):
                """返回满足批次初始模型锁定字段的模拟通过报告。"""

                del result_dir, scenario, expected_initial_model_hash
                return {"status": "passed", "initial_model_hash": "a" * 64}

            with mock.patch(
                "HFLSnF_KG_v4.run_final_stochastic_fedadam._run_config",
                return_value=fake_result,
            ) as run_mock, mock.patch(
                "HFLSnF_KG_v4.run_final_stochastic_fedadam.validate_result",
                side_effect=fake_validate_result,
            ):
                exit_code = run_batch(manifest_path, result_root=result_root)
            self.assertEqual(exit_code, 0)
            self.assertEqual(run_mock.call_count, 8)
            completed = _load_manifest(manifest_path, result_root=result_root)
            self.assertEqual(completed["status"], "passed")
            self.assertTrue(
                all(entry["status"] == "passed" for entry in completed["entries"])
            )


if __name__ == "__main__":
    unittest.main()
