"""验证FedAdam阶段一配置、MAT调度和可恢复批量执行。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

from HFLSnF_KG_v4.run_fedadam_stage1 import (
    CONTRACT_FILE_NAME,
    Stage1RunError,
    create_batch_manifest,
    run_batch,
)
from HFLSnF_KG_v4.tasks.kge.fedadam_stage1 import (
    SCENARIOS,
    build_provider,
    schedule_statistics,
    validate_configs,
)


class FedAdamStage1Test(unittest.TestCase):
    """检查八组合同、固定执行顺序以及失败恢复语义。"""

    def test_eight_configs_only_change_expected_fields(self) -> None:
        """确认八份YAML满足单因素合同及参数组优先的执行顺序。"""

        report = validate_configs()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["behavior_configs_equal"])
        self.assertEqual(len(report["formal_configs"]), 8)
        self.assertEqual(
            [scenario.arm for scenario in SCENARIOS],
            ["hflsnf", "hflnosnf"] * 4,
        )
        self.assertEqual(
            [
                (
                    scenario.profile.learning_rate,
                    scenario.profile.tau,
                )
                for scenario in SCENARIOS[::2]
            ],
            [
                (0.1, 0.001),
                (0.05, 0.001),
                (0.03, 0.001),
                (0.05, 0.01),
            ],
        )

    def test_first_40_rounds_are_nonempty_and_match_hashes(self) -> None:
        """确认两种实验臂前40轮无空参与且哈希与统计合同一致。"""

        # 每种实验臂的MAT调度不随FedAdam参数变化，因此各检查一次。
        for scenario in SCENARIOS[:2]:
            provider = build_provider(scenario)
            topologies = [
                provider.get_round(round_index)
                for round_index in range(40)
            ]
            self.assertTrue(
                all(item.participant_count > 0 for item in topologies)
            )
            statistics = schedule_statistics(scenario)
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

    def test_batch_all_success_preserves_order_and_contracts(self) -> None:
        """模拟八组全成功并确认批次状态、顺序和结果合同文件。"""

        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary).resolve()
            manifest_path = create_batch_manifest(result_root)
            called: List[str] = []

            def fake_run(scenario: object) -> Path:
                """为当前模拟场景创建唯一结果目录并记录调用顺序。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                called.append(scenario_id)
                result_dir = result_root / "result_{}".format(scenario_id)
                result_dir.mkdir(parents=True)
                return result_dir

            def fake_validate(
                result_dir: Path,
                scenario: object,
            ) -> Dict[str, object]:
                """返回最小通过报告以隔离批次编排行为。"""

                return {
                    "status": "passed",
                    "result_dir": str(result_dir),
                    "scenario_id": str(
                        getattr(scenario, "scenario_id")
                    ),
                }

            with patch(
                "HFLSnF_KG_v4.run_fedadam_stage1._run_config",
                side_effect=fake_run,
            ), patch(
                "HFLSnF_KG_v4.run_fedadam_stage1.validate_result",
                side_effect=fake_validate,
            ):
                exit_code = run_batch(
                    manifest_path,
                    result_root=result_root,
                )

            payload = self._read_manifest(manifest_path)
            expected_ids = [item.scenario_id for item in SCENARIOS]
            self.assertEqual(exit_code, 0)
            self.assertEqual(called, expected_ids)
            self.assertEqual(payload["status"], "passed")
            self.assertTrue(
                all(
                    entry["status"] == "passed"
                    and len(entry["attempts"]) == 1
                    for entry in payload["entries"]
                )
            )
            for entry in payload["entries"]:
                contract_path = Path(str(entry["contract_file"]))
                self.assertEqual(contract_path.name, CONTRACT_FILE_NAME)
                self.assertTrue(contract_path.is_file())

    def test_failure_stops_and_resume_skips_passed_entry(self) -> None:
        """模拟第二组失败，再确认恢复时不重复首组并新增失败组尝试。"""

        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary).resolve()
            manifest_path = create_batch_manifest(result_root)
            initial_calls: List[str] = []

            def fail_second(scenario: object) -> Path:
                """让第二个场景失败，用于验证立即停止和进度保存。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                initial_calls.append(scenario_id)
                if len(initial_calls) == 2:
                    raise Stage1RunError("模拟训练失败")
                result_dir = result_root / "initial_{}".format(
                    scenario_id
                )
                result_dir.mkdir(parents=True)
                return result_dir

            with patch(
                "HFLSnF_KG_v4.run_fedadam_stage1._run_config",
                side_effect=fail_second,
            ), patch(
                "HFLSnF_KG_v4.run_fedadam_stage1.validate_result",
                return_value={"status": "passed"},
            ):
                first_exit = run_batch(
                    manifest_path,
                    result_root=result_root,
                )

            failed_payload = self._read_manifest(manifest_path)
            self.assertEqual(first_exit, 2)
            self.assertEqual(
                initial_calls,
                [SCENARIOS[0].scenario_id, SCENARIOS[1].scenario_id],
            )
            self.assertEqual(failed_payload["status"], "failed")
            self.assertEqual(
                [entry["status"] for entry in failed_payload["entries"]],
                ["passed", "failed"] + ["pending"] * 6,
            )

            resume_calls: List[str] = []

            def resume_run(scenario: object) -> Path:
                """为恢复阶段的每个未通过场景创建新的模拟结果目录。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                resume_calls.append(scenario_id)
                result_dir = result_root / "resume_{}".format(scenario_id)
                result_dir.mkdir(parents=True)
                return result_dir

            with patch(
                "HFLSnF_KG_v4.run_fedadam_stage1._run_config",
                side_effect=resume_run,
            ), patch(
                "HFLSnF_KG_v4.run_fedadam_stage1.validate_result",
                return_value={"status": "passed"},
            ):
                resume_exit = run_batch(
                    manifest_path,
                    result_root=result_root,
                )

            resumed_payload = self._read_manifest(manifest_path)
            self.assertEqual(resume_exit, 0)
            self.assertEqual(
                resume_calls,
                [item.scenario_id for item in SCENARIOS[1:]],
            )
            self.assertEqual(resumed_payload["status"], "passed")
            self.assertEqual(
                len(resumed_payload["entries"][0]["attempts"]),
                1,
            )
            self.assertEqual(
                len(resumed_payload["entries"][1]["attempts"]),
                2,
            )
            self.assertTrue(
                all(
                    entry["status"] == "passed"
                    for entry in resumed_payload["entries"]
                )
            )

    def test_resume_rejects_manifest_bound_to_pre_archive_path(self) -> None:
        """确认归档前旧配置路径不会被静默替换后继续恢复。"""

        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary).resolve()
            manifest_path = create_batch_manifest(result_root)
            payload = self._read_manifest(manifest_path)
            payload["entries"][0]["config"] = Path(
                str(payload["entries"][0]["config"])
            ).name
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

            with self.assertRaisesRegex(ValueError, "归档前批次"):
                run_batch(manifest_path, result_root=result_root)

    @staticmethod
    def _read_manifest(path: Path) -> Dict[str, object]:
        """从磁盘读取测试批次清单，避免断言依赖内存对象。"""

        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)


if __name__ == "__main__":
    unittest.main()
