"""验证V5双消融正式场景、失败恢复和三臂配对报告。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_mechanism_ablation as runner
from HFLSnF_KG_v5.tasks.kge.graph_semantic_mechanism_ablation import (
    REFERENCE_DIR,
    REFERENCE_FILES,
    SEEDS,
    expected_flat_config,
    scenarios_from_contract,
    validate_frozen_references,
)


class GraphSemanticMechanismAblationTest(unittest.TestCase):
    """检查六场景身份、失败停止、冻结参考和配对报告。"""

    def test_scenarios_and_configs_are_frozen(self) -> None:
        """六个场景必须按A/B交替顺序绑定正式策略和哈希。"""

        scenarios = scenarios_from_contract()
        self.assertEqual(
            [scenario.scenario_id for scenario in scenarios],
            [
                "graph_only_seed42",
                "semantic_only_seed42",
                "graph_only_seed2024",
                "semantic_only_seed2024",
                "graph_only_seed2025",
                "semantic_only_seed2025",
            ],
        )
        self.assertEqual(
            len({scenario.partition_hash for scenario in scenarios}),
            6,
        )
        for scenario in scenarios:
            config = expected_flat_config(scenario)
            self.assertEqual(
                config["partition_strategy"],
                scenario.partition_strategy,
            )
            self.assertEqual(
                config["expected_partition_hash"],
                scenario.partition_hash,
            )
            self.assertTrue(config["require_cuda"])
            self.assertEqual(config["comm_round"], 150)
            self.assertEqual(config["epochs"], 3)

    def test_full_v5_reference_hashes_pass(self) -> None:
        """五份完整V5参考必须位于隔离目录并保持固定哈希。"""

        self.assertEqual(validate_frozen_references()["status"], "passed")
        for name in REFERENCE_FILES:
            path = (REFERENCE_DIR / name).resolve()
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, REFERENCE_DIR)

    def test_training_failure_stops_after_first_entry(self) -> None:
        """A42训练失败时必须写入恢复状态并停止其余五项。"""

        payload = runner._empty_batch_payload()
        saved = []

        def save_payload(path, value):
            """记录测试中的批次状态快照。"""

            saved.append(dict(value))

        failure = runner.MechanismAblationRunError("模拟训练失败")
        with mock.patch.object(
            runner,
            "_load_batch_manifest",
            return_value=payload,
        ), mock.patch.object(
            runner,
            "_save_batch_manifest",
            side_effect=save_payload,
        ), mock.patch.object(
            runner,
            "_run_config",
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
        self.assertTrue(saved)

    @staticmethod
    def _mock_new_units():
        """构造报告测试所需的A/B六个统一指标单元。"""

        values = []
        for seed_index, seed in enumerate(SEEDS):
            for arm_index, arm in enumerate(("graph_only", "semantic_only")):
                values.append(
                    {
                        "scenario_id": "{}_seed{}".format(arm, seed),
                        "condition": arm,
                        "seed": seed,
                        "result_dir": "unused",
                        "entity_normalized_overlap": 0.13,
                        "relation_normalized_overlap": 0.50,
                        "max_relative_load_deviation": 0.05,
                        "semantic_purity": 0.55 + 0.37 * arm_index,
                        "primary_domain_fraction": (
                            None if arm == "graph_only" else 0.92
                        ),
                        "mean_normalized_domain_entropy": 0.4,
                        "mean_relation_js_divergence": 0.3,
                        "local_entity_reuse_ratio": 6.0,
                        "entity_replication_factor": 6.0,
                        "entity_client_count_mean": 6.0,
                        "relation_client_count_mean": 6.0,
                        "mean_client_domain_count": 5.0,
                        "mean_largest_component_entity_fraction": 0.8,
                        "test_mrr": 0.35 + seed_index * 0.001 + arm_index * 0.002,
                        "test_hits_at_1": 0.26,
                        "test_hits_at_3": 0.38,
                        "test_hits_at_10": 0.52,
                        "head_test_mrr": 0.26,
                        "tail_test_mrr": 0.44,
                        "relation_fallback_round_fraction": 0.5,
                        "relation_fallback_row_count_mean": 3.0,
                        "relation_fallback_row_count_max": 10.0,
                        "best_validation_mrr": 0.36,
                        "last20_validation_mrr_mean": 0.35,
                        "last20_validation_mrr_slope": 0.0,
                        "validation_mrr_auc": 45.0,
                        "round_to_95pct_last20_mean": 50.0,
                        "actual_dense_upload_bytes": 1.0,
                        "logical_sparse_activity_bytes": 1.0,
                        "logical_sparse_activity_rows": 1.0,
                    }
                )
        return values

    def test_report_builds_nine_unit_paired_comparison(self) -> None:
        """报告必须合并F/A/B九单元并计算两种MRR边际差。"""

        root = (
            Path(runner.PACKAGE_DIR)
            / "results"
            / "graph_semantic_mechanism_ablation"
        )
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(root)) as temp:
            batch_path = Path(temp) / runner.BATCH_FILE_NAME
            batch_path.write_text("{}", encoding="utf-8")
            official_path = Path(temp) / "official6.json"
            official_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "units": [{} for _ in range(6)],
                    }
                ),
                encoding="utf-8",
            )
            payload = runner._empty_batch_payload()
            payload["status"] = "passed"
            payload["official_evaluation_manifest"] = str(official_path)
            with mock.patch.object(
                runner,
                "_load_batch_manifest",
                return_value=payload,
            ), mock.patch.object(
                runner,
                "_new_unit_analysis",
                side_effect=self._mock_new_units(),
            ), mock.patch.object(runner, "_save_batch_manifest"):
                status = runner.run_report(batch_path)
            self.assertEqual(status, 0)
            report = json.loads(
                (
                    Path(temp)
                    / "analysis"
                    / runner.REPORT_JSON_FILE_NAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(report["primary_experiment_unit_count"], 9)
            self.assertEqual(report["new_training_unit_count"], 6)
            self.assertEqual(len(report["units"]), 9)
            self.assertEqual(len(report["paired_comparison"]), 3)
            self.assertIn(
                "semantic_added_mrr_delta_mean",
                report["paired_aggregate"],
            )
            self.assertIn(
                "graph_locality_added_mrr_delta_mean",
                report["paired_aggregate"],
            )


if __name__ == "__main__":
    unittest.main()
