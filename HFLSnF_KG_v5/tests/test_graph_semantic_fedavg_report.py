"""验证图语义FedAvg九实验的统计与中文报告。"""

from __future__ import annotations

import unittest

from HFLSnF_KG_v5 import run_graph_semantic_fedavg_comparison as runner


class GraphSemanticFedAvgReportTest(unittest.TestCase):
    """检查三臂聚合、同种子配对差与报告渲染。"""

    @staticmethod
    def _units():
        """构造三个拓扑、三个种子的确定性测试单元。"""

        bases = {"hflsnf": 0.40, "hflnosnf": 0.35, "flnosnf": 0.30}
        return [
            {
                "scenario_id": "{}_{}".format(arm, seed),
                "arm": arm,
                "seed": seed,
                "best_round": 100,
                "best_validation_mrr": base - 0.01,
                "test_mrr": base + offset,
                "test_mean_rank": 200.0,
                "test_hits_at_1": 0.2,
                "test_hits_at_3": 0.4,
                "test_hits_at_10": 0.6,
                "total_round_seconds": 10.0,
                "mean_round_seconds": 1.0,
            }
            for arm, base in bases.items()
            for seed, offset in zip(runner.SEEDS, (0.0, 0.01, -0.01))
        ]

    def test_aggregates_use_three_seed_sample_std(self) -> None:
        """每个拓扑必须按三种子计算均值和样本标准差。"""

        aggregates = runner._arm_aggregates(self._units())
        self.assertAlmostEqual(aggregates["hflsnf"]["test_mrr"]["mean"], 0.4)
        self.assertAlmostEqual(
            aggregates["hflsnf"]["test_mrr"]["sample_std"], 0.01
        )

    def test_paired_differences_follow_declared_direction(self) -> None:
        """三种配对差必须使用左组减右组并保留逐种子结果。"""

        paired = runner._paired_differences(self._units())
        rows = [
            item
            for item in paired
            if item["comparison"] == "HFLSnF-HFLnoSnF"
        ]
        self.assertEqual(len(rows), 4)
        self.assertTrue(
            all(abs(float(item["test_mrr_delta"]) - 0.05) < 1e-12 for item in rows)
        )

    def test_markdown_contains_all_three_topologies(self) -> None:
        """中文报告必须成功渲染三个规范拓扑名称。"""

        units = self._units()
        report = runner._markdown_report(
            units,
            runner._arm_aggregates(units),
            runner._paired_differences(units),
        )
        self.assertIn("HFLSnF", report)
        self.assertIn("HFLnoSnF", report)
        self.assertIn("FLnoSnF", report)
        self.assertIn("本报告只汇总本批9个", report)


if __name__ == "__main__":
    unittest.main()
