"""V2同MAT三臂消融公平合同、结果审计和入口测试。"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from HFLSnF_KG_v2 import run_three_arm_ablation
from HFLSnF_KG_v2.tasks.kge.ablation import (
    ABLATION_SUITE_NAME,
    EXPECTED_PARTITION_HASH,
    EXPECTED_SCHEDULE_HASH,
    THREE_ARM_SPECS,
    compare_three_arm_results,
    load_fedml_yaml,
    validate_three_arm_configs,
    write_comparison_outputs,
)


class ThreeArmConfigContractTest(unittest.TestCase):
    """验证三份正式配置只保留约定的实验变量差异。"""

    @staticmethod
    def _package_dir() -> Path:
        """返回测试对应的V2包目录。"""

        return Path(__file__).resolve().parents[1]

    def test_repository_configs_share_one_contract(self) -> None:
        """验证仓库内A、B、C配置通过数据、MAT和训练预算审计。"""

        contract = validate_three_arm_configs(self._package_dir())
        self.assertEqual(contract["status"], "valid")
        self.assertEqual(contract["suite"], ABLATION_SUITE_NAME)
        self.assertEqual(len(contract["arms"]), 3)
        self.assertEqual(
            contract["shared_contract"]["expected_partition_hash"],
            EXPECTED_PARTITION_HASH,
        )
        self.assertEqual(
            contract["shared_contract"][
                "expected_topology_schedule_hash"
            ],
            EXPECTED_SCHEDULE_HASH,
        )

    def test_changed_shared_budget_is_rejected(self) -> None:
        """验证任意实验臂私自更改批次大小都会被公平合同拒绝。"""

        package_dir = self._package_dir()
        with tempfile.TemporaryDirectory() as temp_dir:
            copied_paths = {}
            for spec in THREE_ARM_SPECS:
                source = (
                    package_dir / "configs" / spec.config_filename
                )
                target = Path(temp_dir) / spec.config_filename
                shutil.copy2(str(source), str(target))
                copied_paths[spec.arm] = target

            changed_path = copied_paths["masked_margin"]
            payload = yaml.safe_load(
                changed_path.read_text(encoding="utf-8")
            )
            payload["train_args"]["batch_size"] = 2048
            changed_path.write_text(
                yaml.safe_dump(
                    payload,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "公平合同"):
                validate_three_arm_configs(
                    package_dir,
                    config_paths=copied_paths,
                )


class ThreeArmResultAuditTest(unittest.TestCase):
    """验证结果汇总只接受真正同口径的三次正式运行。"""

    @staticmethod
    def _package_dir() -> Path:
        """返回测试对应的V2包目录。"""

        return Path(__file__).resolve().parents[1]

    @staticmethod
    def _summary(
        arm: str,
        aggregation_mode: str,
        local_objective: str,
        test_mrr: float,
        hits_at_3: float,
    ) -> dict:
        """构造包含全部可比性指纹的微型正式结果摘要。"""

        return {
            "ablation_suite": ABLATION_SUITE_NAME,
            "ablation_arm": arm,
            "aggregation_mode": aggregation_mode,
            "local_objective": local_objective,
            "partition_hash": EXPECTED_PARTITION_HASH,
            "topology_schedule_hash": EXPECTED_SCHEDULE_HASH,
            "initial_model_hash": "same-initial-model",
            "client_count": 37,
            "client_num_in_total": 37,
            "client_num_per_round": "from_matlab",
            "participant_count_min": 11,
            "participant_count_max": 37,
            "participant_count_mean": 35.7,
            "group_count_min": 2,
            "group_count_max": 12,
            "group_count_mean": 6.0,
            "comm_round": 200,
            "local_epochs": 1,
            "effective_global_passes": 193.0,
            "best_round": 180,
            "final_validation_metrics": {
                "mrr": test_mrr - 0.001,
            },
            "final_test_metrics": {
                "mrr": test_mrr,
                "hits_at_3": hits_at_3,
                "hits_at_10": hits_at_3 + 0.1,
            },
        }

    @staticmethod
    def _write_result(
        result_dir: Path,
        spec,
        test_mrr: float,
        hits_at_3: float,
    ) -> None:
        """写出结果审计所需的摘要、配置快照和逐轮耗时。"""

        result_dir.mkdir(parents=True, exist_ok=False)
        summary = ThreeArmResultAuditTest._summary(
            spec.arm,
            spec.aggregation_mode,
            spec.local_objective,
            test_mrr,
            hits_at_3,
        )
        (result_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False),
            encoding="utf-8",
        )
        package_dir = Path(__file__).resolve().parents[1]
        config = load_fedml_yaml(
            package_dir / "configs" / spec.config_filename
        )
        (result_dir / "config_snapshot.json").write_text(
            json.dumps(config, ensure_ascii=False),
            encoding="utf-8",
        )
        with (result_dir / "metrics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["round", "round_seconds"]
            )
            writer.writeheader()
            writer.writerow({"round": 1, "round_seconds": 2.5})
            writer.writerow({"round": 2, "round_seconds": 3.5})

    def _build_three_results(self, root: Path) -> dict:
        """构造MRR逐臂上升的三组同口径测试结果。"""

        scores = {
            "dense_margin": (0.150, 0.200),
            "masked_margin": (0.160, 0.210),
            "masked_fede_fair": (0.164, 0.215),
        }
        result_dirs = {}
        for spec in THREE_ARM_SPECS:
            result_dir = root / spec.arm
            mrr, hits_at_3 = scores[spec.arm]
            self._write_result(
                result_dir, spec, mrr, hits_at_3
            )
            result_dirs[spec.arm] = result_dir
        return result_dirs

    def test_comparison_calculates_effects_and_writes_outputs(
        self,
    ) -> None:
        """验证B减A、C减B及中文汇总文件均正确生成。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dirs = self._build_three_results(root)
            comparison = compare_three_arm_results(
                self._package_dir(),
                result_dirs,
                mrr_threshold=0.003,
            )
            outputs = write_comparison_outputs(
                root / "comparison", comparison
            )

            self.assertEqual(comparison["status"], "comparable")
            self.assertAlmostEqual(
                comparison["effects"][
                    "row_mask_effect_B_minus_A"
                ]["mrr_delta"],
                0.010,
            )
            self.assertAlmostEqual(
                comparison["effects"][
                    "fede_objective_bundle_C_minus_B"
                ]["mrr_delta"],
                0.004,
            )
            self.assertTrue(
                comparison["run_dense_fede_fair_next"]
            )
            for path in outputs.values():
                self.assertTrue(Path(path).is_file())
            report = Path(outputs["comparison_report"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("大白话怎么看", report)
            self.assertIn("初始模型参数哈希相同", report)

    def test_different_initial_model_is_rejected(self) -> None:
        """验证不同初始模型的三次结果不能被强行汇总。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dirs = self._build_three_results(root)
            changed_path = (
                result_dirs["masked_fede_fair"] / "summary.json"
            )
            changed = json.loads(
                changed_path.read_text(encoding="utf-8")
            )
            changed["initial_model_hash"] = "different-model"
            changed_path.write_text(
                json.dumps(changed, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "不可直接比较"):
                compare_three_arm_results(
                    self._package_dir(), result_dirs
                )


class ThreeArmEntrypointTest(unittest.TestCase):
    """验证一键入口的安全默认值和无CUDA快速失败行为。"""

    def test_validate_action_does_not_start_training(self) -> None:
        """验证默认动作仅校验合同，不会创建训练子进程。"""

        with mock.patch.object(
            run_three_arm_ablation.subprocess,
            "Popen",
        ) as popen:
            run_three_arm_ablation.main(["--action", "validate"])
        popen.assert_not_called()

    def test_formal_run_without_cuda_fails_before_subprocess(
        self,
    ) -> None:
        """验证无CUDA时正式三臂运行在启动训练子进程前报错。"""

        with mock.patch(
            "torch.cuda.is_available", return_value=False
        ), mock.patch.object(
            run_three_arm_ablation.subprocess,
            "Popen",
        ) as popen:
            with self.assertRaisesRegex(RuntimeError, "检测不到GPU"):
                run_three_arm_ablation.run_action(
                    ["dense_margin"]
                )
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
