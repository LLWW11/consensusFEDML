"""四臂二乘二合同与无需重训方向诊断的轻量测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from HFLSnF_KG_v2 import run_four_arm_ablation
from HFLSnF_KG_v2.run_directional_diagnostics import resolve_device
from HFLSnF_KG_v2.tasks.kge.ablation import (
    ABLATION_SUITE_NAME,
    EXPECTED_PARTITION_HASH,
    EXPECTED_SCHEDULE_HASH,
    load_fedml_yaml,
)
from HFLSnF_KG_v2.tasks.kge.data import (
    build_synthetic_knowledge_graph,
)
from HFLSnF_KG_v2.tasks.kge.directional_diagnostics import (
    build_pairwise_query_outcomes,
    build_relation_metrics,
    select_test_triples,
)
from HFLSnF_KG_v2.tasks.kge.factorial_ablation import (
    FOUR_ARM_SPECS,
    compare_four_arm_results,
    validate_four_arm_configs,
    write_factorial_outputs,
)


class FourArmFactorialTest(unittest.TestCase):
    """验证D臂公平合同和二乘二差中差计算。"""

    @staticmethod
    def _package_dir() -> Path:
        """返回测试对应的HFLSnF_KG_v2目录。"""

        return Path(__file__).resolve().parents[1]

    @staticmethod
    def _write_fake_result(
        result_dir: Path,
        spec,
        mrr: float,
        hits_at_3: float,
    ) -> None:
        """写出通过四臂可比性审计所需的微型结果文件。"""

        result_dir.mkdir(parents=True, exist_ok=False)
        summary = {
            "ablation_suite": ABLATION_SUITE_NAME,
            "ablation_arm": spec.arm,
            "aggregation_mode": spec.aggregation_mode,
            "local_objective": spec.local_objective,
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
            "best_round": 200,
            "final_validation_metrics": {"mrr": mrr - 0.001},
            "final_test_metrics": {
                "mrr": mrr,
                "hits_at_3": hits_at_3,
                "hits_at_10": hits_at_3 + 0.1,
            },
        }
        (result_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False),
            encoding="utf-8",
        )
        config = load_fedml_yaml(
            FourArmFactorialTest._package_dir()
            / "configs"
            / spec.config_filename
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
            writer.writerow({"round": 1, "round_seconds": 1.0})

    def test_repository_four_arm_contract_is_valid(self) -> None:
        """验证仓库内D臂仅改变约定的两个二乘二因素。"""

        contract = validate_four_arm_configs(self._package_dir())
        self.assertEqual(contract["status"], "valid")
        self.assertEqual(len(contract["arms"]), 4)
        d_arm = {
            item["arm"]: item for item in contract["arms"]
        }["dense_fede_fair"]
        self.assertEqual(
            d_arm["aggregation_mode"], "dense_triple_weighted"
        )
        self.assertEqual(
            d_arm["local_objective"], "fede_self_adversarial"
        )

    def test_factorial_interaction_and_outputs_are_correct(self) -> None:
        """验证四个条件效应、交互项和中文报告能够生成。"""

        scores = {
            "dense_margin": (0.15, 0.20),
            "masked_margin": (0.16, 0.21),
            "masked_fede_fair": (0.20, 0.25),
            "dense_fede_fair": (0.18, 0.23),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dirs = {}
            for spec in FOUR_ARM_SPECS:
                result_dir = root / spec.arm
                mrr, hits = scores[spec.arm]
                self._write_fake_result(
                    result_dir, spec, mrr, hits
                )
                result_dirs[spec.arm] = result_dir
            comparison = compare_four_arm_results(
                self._package_dir(), result_dirs
            )
            self.assertAlmostEqual(
                comparison["interaction"]["mrr_interaction"],
                0.01,
            )
            outputs = write_factorial_outputs(
                root / "comparison", comparison
            )
            for path in outputs.values():
                self.assertTrue(Path(path).is_file())
            report = Path(outputs["factorial_report"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("大白话怎么看交互", report)

    def test_run_d_without_cuda_fails_before_training(self) -> None:
        """验证无CUDA机器不会误启动D臂正式训练子进程。"""

        with mock.patch(
            "torch.cuda.is_available", return_value=False
        ), mock.patch.object(
            run_four_arm_ablation,
            "_run_one_arm",
        ) as runner:
            with self.assertRaisesRegex(RuntimeError, "检测不到GPU"):
                run_four_arm_ablation.run_d_action()
        runner.assert_not_called()


class DirectionalDiagnosticsTest(unittest.TestCase):
    """验证固定抽样、逐关系统计和逐查询胜负。"""

    def test_test_triple_selection_is_reproducible(self) -> None:
        """验证相同种子始终选到同一批官方测试事实。"""

        dataset = build_synthetic_knowledge_graph()
        first = select_test_triples(dataset, maximum=1, seed=42)
        second = select_test_triples(dataset, maximum=1, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)

    def test_relation_metrics_include_head_tail_and_combined(self) -> None:
        """验证每个关系同时生成头、尾和综合三种口径。"""

        records = [
            {
                "arm": "dense_margin",
                "relation_id": 0,
                "relation_name": "knows",
                "direction": "head",
                "rank": 2,
            },
            {
                "arm": "dense_margin",
                "relation_id": 0,
                "relation_name": "knows",
                "direction": "tail",
                "rank": 4,
            },
        ]
        metrics = build_relation_metrics(records)
        directions = {row["direction"] for row in metrics}
        self.assertEqual(
            directions, {"head", "tail", "combined"}
        )
        combined = [
            row for row in metrics if row["direction"] == "combined"
        ][0]
        self.assertAlmostEqual(combined["mrr"], 0.375)

    def test_pairwise_outcomes_count_wins_losses_and_ties(self) -> None:
        """验证相邻实验臂在每个头尾查询上的胜负统计。"""

        triples = ((0, 0, 1),)
        ranks = {
            "dense_margin": {
                "head": np.asarray([4]),
                "tail": np.asarray([4]),
            },
            "masked_margin": {
                "head": np.asarray([2]),
                "tail": np.asarray([4]),
            },
            "dense_fede_fair": {
                "head": np.asarray([3]),
                "tail": np.asarray([2]),
            },
            "masked_fede_fair": {
                "head": np.asarray([1]),
                "tail": np.asarray([1]),
            },
        }
        details, summaries = build_pairwise_query_outcomes(
            triples, ranks
        )
        self.assertEqual(len(details), 8)
        b_minus_a = [
            row
            for row in summaries
            if row["candidate_arm"] == "masked_margin"
        ][0]
        self.assertEqual(b_minus_a["candidate_win_count"], 1)
        self.assertEqual(b_minus_a["tie_count"], 1)

    def test_directional_cuda_requirement_fails_fast(self) -> None:
        """验证正式方向诊断在无CUDA时不会静默回退CPU。"""

        with mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "检测不到CUDA"):
                resolve_device(True, 0, True)


if __name__ == "__main__":
    unittest.main()
