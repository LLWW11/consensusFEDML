"""验证最终动态FedAdam九组配置、顺序执行与恢复语义。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

from HFLSnF_KG_v3.run_final_dynamic_fedadam import (
    CONTRACT_FILE_NAME,
    FinalDynamicRunError,
    create_batch_manifest,
    run_batch,
)
from HFLSnF_KG_v3.tasks.kge.final_dynamic_fedadam import (
    ARMS,
    SCENARIOS,
    SEEDS,
    build_provider,
    schedule_statistics,
    validate_configs,
)


class FinalDynamicFedAdamTest(unittest.TestCase):
    """检查九组配置合同、固定顺序、失败停止和恢复行为。"""

    def test_nine_configs_follow_seed_first_order(self) -> None:
        """确认三seed乘三实验臂的配置合同与运行顺序。"""

        report = validate_configs()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["behavior_configs_equal"])
        self.assertEqual(len(report["formal_configs"]), 9)
        self.assertEqual(
            [(item.seed, item.arm) for item in SCENARIOS],
            [(seed, arm) for seed in SEEDS for arm in ARMS],
        )

    def test_schedules_are_nonempty_and_flnosnf_coverage_is_explicit(self) -> None:
        """确认三臂无空轮并锁定FLnoSnF的18个永久缺席客户端。"""

        for scenario in SCENARIOS[:3]:
            provider = build_provider(scenario)
            self.assertTrue(
                all(provider.get_round(index).participant_count > 0 for index in range(150))
            )
            stats = schedule_statistics(scenario)
            self.assertEqual(stats["schedule_hash"], scenario.contract.schedule_hash)
            self.assertEqual(
                stats["zero_participation_clients"],
                scenario.contract.zero_participation_clients,
            )
        fl_stats = schedule_statistics(SCENARIOS[2])
        self.assertEqual(fl_stats["zero_participation_clients"], 18)
        self.assertEqual(
            fl_stats["zero_participation_client_ids"],
            [3, 12, 13, 14, 19, 20, 21, 22, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        )

    def test_batch_all_success_preserves_order_and_contracts(self) -> None:
        """模拟九组全成功并确认seed优先顺序和合同文件。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = create_batch_manifest(root)
            called: List[str] = []

            def fake_run(scenario: object) -> Path:
                """创建模拟结果目录并记录实际调用顺序。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                called.append(scenario_id)
                result = root / "result_{}".format(scenario_id)
                result.mkdir(parents=True)
                return result

            def fake_validate(
                result_dir: Path,
                scenario: object,
                expected_initial_model_hash: str = "",
            ) -> Dict[str, object]:
                """返回含同seed初始模型哈希的最小通过报告。"""

                seed = int(getattr(scenario, "seed"))
                return {
                    "status": "passed",
                    "result_dir": str(result_dir),
                    "partition_hash": "p{}".format(seed),
                    "initial_model_hash": "i{}".format(seed),
                }

            with patch(
                "HFLSnF_KG_v3.run_final_dynamic_fedadam._run_config",
                side_effect=fake_run,
            ), patch(
                "HFLSnF_KG_v3.run_final_dynamic_fedadam.validate_result",
                side_effect=fake_validate,
            ):
                exit_code = run_batch(manifest, result_root=root)

            payload = self._read_manifest(manifest)
            self.assertEqual(exit_code, 0)
            self.assertEqual(called, [item.scenario_id for item in SCENARIOS])
            self.assertEqual(payload["status"], "passed")
            for entry in payload["entries"]:
                self.assertEqual(entry["status"], "passed")
                self.assertEqual(len(entry["attempts"]), 1)
                self.assertEqual(Path(entry["contract_file"]).name, CONTRACT_FILE_NAME)
                self.assertTrue(Path(entry["contract_file"]).is_file())

    def test_failure_stops_and_resume_skips_passed(self) -> None:
        """模拟第二组失败并确认恢复跳过首组且重试失败项目。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = create_batch_manifest(root)
            first_calls: List[str] = []

            def fail_second(scenario: object) -> Path:
                """让第二个场景失败以验证立即停止。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                first_calls.append(scenario_id)
                if len(first_calls) == 2:
                    raise FinalDynamicRunError("模拟训练失败")
                result = root / "first_{}".format(scenario_id)
                result.mkdir(parents=True)
                return result

            with patch(
                "HFLSnF_KG_v3.run_final_dynamic_fedadam._run_config",
                side_effect=fail_second,
            ), patch(
                "HFLSnF_KG_v3.run_final_dynamic_fedadam.validate_result",
                return_value={
                    "status": "passed", "partition_hash": "p42",
                    "initial_model_hash": "i42",
                },
            ):
                first_exit = run_batch(manifest, result_root=root)

            failed = self._read_manifest(manifest)
            self.assertEqual(first_exit, 1)
            self.assertEqual(first_calls, [item.scenario_id for item in SCENARIOS[:2]])
            self.assertEqual(
                [entry["status"] for entry in failed["entries"]],
                ["passed", "failed"] + ["pending"] * 7,
            )

            resumed_calls: List[str] = []

            def resume_run(scenario: object) -> Path:
                """为恢复阶段的每个未通过场景创建新目录。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                resumed_calls.append(scenario_id)
                result = root / "resume_{}".format(scenario_id)
                result.mkdir(parents=True)
                return result

            def resume_validate(
                result_dir: Path,
                scenario: object,
                expected_initial_model_hash: str = "",
            ) -> Dict[str, object]:
                """按seed返回稳定哈希以模拟三臂可比结果。"""

                seed = int(getattr(scenario, "seed"))
                return {
                    "status": "passed", "result_dir": str(result_dir),
                    "partition_hash": "p{}".format(seed),
                    "initial_model_hash": "i{}".format(seed),
                }

            with patch(
                "HFLSnF_KG_v3.run_final_dynamic_fedadam._run_config",
                side_effect=resume_run,
            ), patch(
                "HFLSnF_KG_v3.run_final_dynamic_fedadam.validate_result",
                side_effect=resume_validate,
            ):
                resume_exit = run_batch(manifest, result_root=root)

            resumed = self._read_manifest(manifest)
            self.assertEqual(resume_exit, 0)
            self.assertEqual(resumed_calls, [item.scenario_id for item in SCENARIOS[1:]])
            self.assertEqual(len(resumed["entries"][0]["attempts"]), 1)
            self.assertEqual(len(resumed["entries"][1]["attempts"]), 2)
            self.assertTrue(all(entry["status"] == "passed" for entry in resumed["entries"]))

    @staticmethod
    def _read_manifest(path: Path) -> Dict[str, object]:
        """从磁盘读取批次清单供测试断言。"""

        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)


if __name__ == "__main__":
    unittest.main()
