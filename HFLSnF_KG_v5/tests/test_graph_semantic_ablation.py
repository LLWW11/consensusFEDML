"""验证V5图语义正式场景、恢复停止和十五单元报告。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_ablation as runner
from HFLSnF_KG_v5.tasks.kge.graph_semantic_ablation import (
    SEEDS,
    SUITE_NAME,
    V4_REFERENCE_DIR,
    V4_REFERENCE_FILES,
    expected_flat_config,
    scenarios_from_contract,
    validate_v4_references,
)


class GraphSemanticAblationTest(unittest.TestCase):
    """检查正式三场景身份、失败恢复和报告合并。"""

    def test_scenarios_and_configs_are_frozen(self) -> None:
        """三个场景必须按种子顺序绑定正式策略和哈希。"""

        scenarios = scenarios_from_contract()
        self.assertEqual(
            [scenario.seed for scenario in scenarios],
            list(SEEDS),
        )
        self.assertEqual(
            [scenario.scenario_id for scenario in scenarios],
            [
                "graph_semantic_seed42",
                "graph_semantic_seed2024",
                "graph_semantic_seed2025",
            ],
        )
        self.assertEqual(
            len({scenario.partition_hash for scenario in scenarios}),
            3,
        )
        for scenario in scenarios:
            config = expected_flat_config(scenario)
            self.assertEqual(
                config["partition_strategy"],
                "semantic_domain_graph_local_balanced",
            )
            self.assertEqual(
                config["expected_partition_hash"],
                scenario.partition_hash,
            )
            self.assertTrue(config["require_cuda"])

    def test_v4_reference_hashes_pass(self) -> None:
        """三份V5内置参考必须保持固定哈希且不访问V4目录。"""

        self.assertEqual(validate_v4_references()["status"], "passed")
        expected_dir = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "graph_semantic"
            / "frozen_v4_reference"
        ).resolve()
        self.assertEqual(V4_REFERENCE_DIR, expected_dir)
        for item in V4_REFERENCE_FILES.values():
            reference_path = Path(item["path"]).resolve()
            self.assertEqual(reference_path.parent, expected_dir)
            self.assertNotIn("HFLSnF_KG_v4", str(reference_path))

    def test_training_failure_stops_after_first_entry(self) -> None:
        """首个训练失败时必须写入恢复状态并停止后续种子。"""

        payload = runner._empty_batch_payload()
        saved = []

        def save_payload(path, value):
            """记录测试中的批次状态快照。"""

            saved.append(dict(value))

        failure = runner.GraphSemanticRunError("模拟训练失败")
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
            status = runner.run_training_batch(
                Path("unused_batch_summary.json")
            )
        self.assertEqual(status, 1)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["entries"][0]["status"], "failed")
        self.assertEqual(payload["entries"][1]["status"], "pending")
        self.assertTrue(saved)

    def test_report_merges_three_new_and_twelve_v4_units(self) -> None:
        """报告必须形成6个主比较和15个背景实验单元。"""

        new_units = [
            {
                "scenario_id": "graph_semantic_seed{}".format(seed),
                "condition": "graph_semantic",
                "seed": seed,
                "result_dir": "unused",
                "entity_normalized_overlap": 0.13,
                "relation_normalized_overlap": 0.50,
                "max_relative_load_deviation": 0.05,
                "semantic_purity": 0.92,
                "mean_relation_js_divergence": 0.50,
                "local_entity_reuse_ratio": 6.7,
                "mean_largest_component_entity_fraction": 0.8,
                "test_mrr": 0.35 + index * 0.001,
                "test_hits_at_1": 0.26,
                "test_hits_at_3": 0.38,
                "test_hits_at_10": 0.52,
                "head_test_mrr": 0.26,
                "tail_test_mrr": 0.44,
            }
            for index, seed in enumerate(SEEDS)
        ]
        root = (
            Path(runner.PACKAGE_DIR)
            / "results"
            / "graph_semantic"
        )
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(root)) as temp:
            batch_path = Path(temp) / runner.BATCH_FILE_NAME
            batch_path.write_text("{}", encoding="utf-8")
            official_path = Path(temp) / "official3.json"
            official_path.write_text(
                json.dumps(
                    {
                        "suite": SUITE_NAME,
                        "status": "passed",
                        "units": [
                            {"seed": seed} for seed in SEEDS
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = runner._empty_batch_payload()
            payload["status"] = "passed"
            payload["official_evaluation_manifest"] = str(
                official_path
            )
            with mock.patch.object(
                runner,
                "_load_batch_manifest",
                return_value=payload,
            ), mock.patch.object(
                runner,
                "_new_unit_analysis",
                side_effect=new_units,
            ), mock.patch.object(
                runner,
                "_save_batch_manifest",
            ):
                status = runner.run_report(batch_path)
            self.assertEqual(status, 0)
            report = json.loads(
                (
                    Path(temp)
                    / "analysis"
                    / runner.REPORT_JSON_FILE_NAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["primary_experiment_unit_count"],
                6,
            )
            self.assertEqual(
                report["context_experiment_unit_count"],
                15,
            )
            self.assertEqual(len(report["combined_units"]), 15)


if __name__ == "__main__":
    unittest.main()
