"""验证V4实体重叠率正式消融的合同、顺序、恢复和汇总。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from HFLSnF_KG_v5.run_overlap_ablation import (
    OverlapRunError,
    _aggregate_conditions,
    _empty_batch_payload,
    _official_units,
    run_training_batch,
)
from HFLSnF_KG_v5.tasks.kge.overlap_ablation import (
    BASELINE_INITIAL_MODEL_HASHES,
    LEVELS,
    SEEDS,
    load_calibration_contract,
    scenarios_from_contract,
    validate_configs,
)


class OverlapAblationContractTest(unittest.TestCase):
    """检查正式校准、九组训练编排和十二格汇总合同。"""

    def test_formal_contract_and_scenario_order(self) -> None:
        """确认正式合同已复现且场景按seed优先、档位次序排列。"""

        contract = load_calibration_contract()
        self.assertEqual(
            contract["reproduction_verification"]["status"],
            "passed",
        )
        scenarios = scenarios_from_contract(contract)
        expected = [
            "overlap_{}_seed{}".format(level, seed)
            for seed in SEEDS
            for level in LEVELS
        ]
        self.assertEqual(
            [scenario.scenario_id for scenario in scenarios],
            expected,
        )
        self.assertEqual(len({item.partition_hash for item in scenarios}), 9)

    def test_nine_configs_match_contract_without_recompute(self) -> None:
        """确认九份YAML、基线和拓扑合同通过快速只读校验。"""

        report = validate_configs(recompute_partitions=False)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["scenario_order"]), 9)

    def test_batch_binds_original_hashes_and_pilot_order(self) -> None:
        """确认批次预锁定原始初始化哈希并把seed42排在最前。"""

        payload = _empty_batch_payload()
        entries = payload["entries"]
        self.assertEqual(len(entries), 9)
        self.assertEqual(
            [entry["scenario_id"] for entry in entries[:3]],
            [
                "overlap_low_seed42",
                "overlap_medium_seed42",
                "overlap_high_seed42",
            ],
        )
        baselines = payload["baseline_units"]
        for seed in SEEDS:
            self.assertEqual(
                baselines[str(seed)]["initial_model_hash"],
                BASELINE_INITIAL_MODEL_HASHES[seed],
            )

    def test_failure_stops_and_resume_skips_passed_entry(self) -> None:
        """确认失败立即停止，恢复时从首个未通过seed42场景继续。"""

        payload = _empty_batch_payload()
        failure = OverlapRunError("模拟训练失败")
        with patch(
            "HFLSnF_KG_v5.run_overlap_ablation._load_batch_manifest",
            return_value=payload,
        ), patch(
            "HFLSnF_KG_v5.run_overlap_ablation._save_batch_manifest"
        ), patch(
            "HFLSnF_KG_v5.run_overlap_ablation._run_config",
            side_effect=failure,
        ) as run_mock:
            self.assertEqual(run_training_batch(Path("batch.json")), 1)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(payload["entries"][0]["status"], "failed")
        self.assertEqual(payload["entries"][1]["status"], "pending")

        resumed = _empty_batch_payload()
        resumed["entries"][0]["status"] = "passed"
        with patch(
            "HFLSnF_KG_v5.run_overlap_ablation._load_batch_manifest",
            return_value=resumed,
        ), patch(
            "HFLSnF_KG_v5.run_overlap_ablation._save_batch_manifest"
        ), patch(
            "HFLSnF_KG_v5.run_overlap_ablation._run_config",
            side_effect=failure,
        ) as resume_mock:
            self.assertEqual(run_training_batch(Path("batch.json")), 1)
        resumed_scenario = resume_mock.call_args[0][0]
        self.assertEqual(resumed_scenario.scenario_id, "overlap_medium_seed42")

    def test_official_units_include_three_baselines_and_nine_new_runs(self) -> None:
        """确认完整官方测试清单严格包含十二个实验单元。"""

        payload = _empty_batch_payload()
        payload["status"] = "passed"
        for entry in payload["entries"]:
            entry["status"] = "passed"
            entry["result_dir"] = "result/{}".format(entry["scenario_id"])
        units = _official_units(payload)
        self.assertEqual(len(units), 12)
        self.assertEqual(
            sum(unit["condition"] == "original" for unit in units),
            3,
        )

    def test_summary_uses_sample_std_and_paired_mrr_delta(self) -> None:
        """确认四条件汇总使用样本标准差和同种子MRR配对差。"""

        units = []
        metric_names = (
            "test_hits_at_1",
            "test_hits_at_3",
            "test_hits_at_10",
            "test_mean_rank",
            "head_test_mrr",
            "tail_test_mrr",
            "best_validation_mrr",
            "last20_validation_mrr_mean",
            "last20_validation_mrr_slope",
            "validation_mrr_auc",
            "round_to_95pct_last20_mean",
            "actual_dense_upload_bytes",
            "logical_sparse_activity_bytes",
        )
        offsets = {"original": 0.0, "low": 0.01, "medium": 0.02, "high": 0.03}
        for condition in ("original", *LEVELS):
            for seed_index, seed in enumerate(SEEDS):
                unit = {
                    "condition": condition,
                    "seed": seed,
                    "test_mrr": 0.20 + seed_index * 0.01 + offsets[condition],
                    "entity_normalized_overlap": 0.20 + offsets[condition],
                }
                unit.update({name: 1.0 + seed_index for name in metric_names})
                units.append(unit)
        summary = _aggregate_conditions(units)
        self.assertAlmostEqual(
            summary["low"]["paired_delta_vs_original_mean"],
            0.01,
        )
        self.assertGreater(summary["original"]["test_mrr_sample_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
