"""验证四组实验分析中的共识、平滑、客户端映射和结果审计逻辑。"""

import csv
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from analyze_experiment_suite import (
    SCENARIO_ORDER,
    consensus_components,
    discover_experiment_dirs,
    first_stable_epoch,
    generalized_js_divergence,
    historical_best,
    load_experiment,
    map_client_ids_to_slots,
    normalized_entropy,
    trailing_mean,
    validate_experiment,
)


class ConsensusMetricTest(unittest.TestCase):
    """验证有效共识及其组成指标的边界行为。"""

    def test_normalized_entropy_boundary_values(self):
        """确认均匀分布熵为1，one-hot分布熵为0。"""

        uniform = np.full(10, 0.1, dtype=np.float64)
        one_hot = np.zeros(10, dtype=np.float64)
        one_hot[4] = 1.0

        self.assertAlmostEqual(float(normalized_entropy(uniform)), 1.0, places=12)
        self.assertAlmostEqual(float(normalized_entropy(one_hot)), 0.0, places=12)

    def test_generalized_js_divergence_detects_disagreement(self):
        """确认相同分布的GJSD为0，而冲突one-hot的GJSD大于0。"""

        identical = np.full((3, 10), 0.1, dtype=np.float64)
        conflicting = np.zeros((2, 10), dtype=np.float64)
        conflicting[0, 0] = 1.0
        conflicting[1, 1] = 1.0

        self.assertAlmostEqual(generalized_js_divergence(identical), 0.0, places=12)
        self.assertGreater(generalized_js_divergence(conflicting), 0.0)

    def test_uniform_probabilities_have_low_effective_consensus(self):
        """确认共同均匀输出只有高一致性，而没有确定性和有效共识。"""

        probabilities = np.full((37, 10), 0.1, dtype=np.float64)
        agreement, certainty, effective = consensus_components(probabilities)

        self.assertAlmostEqual(agreement, 1.0, places=12)
        self.assertAlmostEqual(certainty, 0.0, places=12)
        self.assertAlmostEqual(effective, 0.0, places=12)

    def test_identical_one_hot_probabilities_have_full_consensus(self):
        """确认所有客户端输出相同 one-hot 向量时三个指标都为满分。"""

        probabilities = np.zeros((37, 10), dtype=np.float64)
        probabilities[:, 3] = 1.0
        agreement, certainty, effective = consensus_components(probabilities)

        self.assertAlmostEqual(agreement, 1.0, places=12)
        self.assertAlmostEqual(certainty, 1.0, places=12)
        self.assertAlmostEqual(effective, 1.0, places=12)

    def test_conflicting_one_hot_probabilities_are_not_full_consensus(self):
        """确认确定但互相冲突的 one-hot 输出不会被判定为满共识。"""

        probabilities = np.zeros((2, 10), dtype=np.float64)
        probabilities[0, 0] = 1.0
        probabilities[1, 1] = 1.0
        agreement, certainty, effective = consensus_components(probabilities)

        self.assertLess(agreement, 1.0)
        self.assertAlmostEqual(certainty, 1.0, places=12)
        self.assertLess(effective, 1.0)


class TimeSeriesUtilityTest(unittest.TestCase):
    """验证尾随均值、稳定阈值和历史最佳共识的时间语义。"""

    def test_trailing_mean_keeps_leading_values_empty(self):
        """确认尾随窗口未满时为空，且窗口只使用当前及历史值。"""

        result = trailing_mean([1.0, 2.0, 3.0, 4.0], window=3)

        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[2:], [2.0, 3.0])

    def test_historical_best_is_monotonic_after_first_valid_value(self):
        """确认累计最佳值保留前导空值，并在有效区间单调不降。"""

        result = historical_best([np.nan, np.nan, 0.2, 0.1, 0.4, 0.3])

        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_allclose(result[2:], [0.2, 0.2, 0.4, 0.4])
        self.assertTrue(np.all(np.diff(result[2:]) >= 0.0))

    def test_stable_epoch_requires_no_later_fallback(self):
        """确认稳定达标轮次排除曾短暂越线但随后回落的情况。"""

        values = [0.7] * 4 + [0.9] * 5 + [0.6] * 5 + [0.9] * 8
        self.assertEqual(first_stable_epoch(values, threshold=0.8, window=5), 18)


class ClientMappingTest(unittest.TestCase):
    """验证探针列按每轮候选顺序映射到真实客户端编号。"""

    def test_real_client_ids_map_to_candidate_slots(self):
        """确认任意真实客户端顺序均可映射到对应的候选列位置。"""

        record = {"candidate_client_indexes": [10, 20, 30, 40]}
        self.assertEqual(map_client_ids_to_slots(record, [30, 10, 40]), [2, 0, 3])

    def test_non_candidate_client_is_rejected(self):
        """确认不属于候选集合的客户端不会被静默映射。"""

        record = {"candidate_client_indexes": [10, 20, 30, 40]}
        with self.assertRaises(ValueError):
            map_client_ids_to_slots(record, [99])


class CurrentResultIntegrationTest(unittest.TestCase):
    """使用当前四组原始结果验证150轮调度与探针不变量。"""

    @classmethod
    def setUpClass(cls):
        """发现并加载当前 result 目录中的四组目标实验。"""

        cls.result_root = Path(__file__).resolve().parent / "result"
        if not cls.result_root.is_dir():
            raise unittest.SkipTest("当前工作区没有 result 目录")
        try:
            experiment_dirs = discover_experiment_dirs(cls.result_root)
        except ValueError as exc:
            raise unittest.SkipTest("未发现唯一的四组目标结果：{}".format(exc))
        cls.experiments = [load_experiment(path) for path in experiment_dirs]

    def test_all_four_experiments_have_150_rounds(self):
        """确认四种场景齐全，且每种场景都有150轮实际输出。"""

        self.assertEqual(
            [experiment.scenario for experiment in self.experiments],
            SCENARIO_ORDER,
        )
        self.assertEqual([len(experiment.schedule) for experiment in self.experiments], [150] * 4)

    def test_all_runtime_invariants_pass(self):
        """确认候选、分组、聚合、下发和三层探针检查没有关键失败。"""

        for experiment in self.experiments:
            checks = validate_experiment(experiment)
            failures = [
                check for check in checks
                if check["严重级别"] == "关键" and check["状态"] != "通过"
            ]
            self.assertEqual(failures, [], msg=experiment.label)

    def test_candidate_sets_are_paired_across_scenarios(self):
        """确认同一轮四方案使用相同候选集合，支持配对描述性比较。"""

        reference = [
            tuple(record["candidate_client_indexes"])
            for record in self.experiments[0].schedule
        ]
        for experiment in self.experiments[1:]:
            observed = [
                tuple(record["candidate_client_indexes"])
                for record in experiment.schedule
            ]
            self.assertEqual(observed, reference, msg=experiment.label)


class GeneratedArtifactTest(unittest.TestCase):
    """独立读取原始文件，复核已生成分析包的数字、行数和图片规格。"""

    @classmethod
    def setUpClass(cls):
        """定位分析输出，并建立场景到原始实验目录的映射。"""

        cls.workspace = Path(__file__).resolve().parent
        cls.result_root = cls.workspace / "result"
        cls.output_dir = cls.result_root / "analysis_alpha0p2_u0p5_150rounds_20260713"
        if not (cls.output_dir / "实验汇总.csv").is_file():
            raise unittest.SkipTest("分析包尚未生成")
        experiment_dirs = discover_experiment_dirs(cls.result_root)
        cls.source_by_scenario = {}
        for directory in experiment_dirs:
            metadata = json.loads(
                (directory / "topology_metadata.json").read_text(encoding="utf-8")
            )
            cls.source_by_scenario[str(metadata["scenario"])] = directory

    def test_summary_numbers_match_raw_files(self):
        """从原始指标和JSONL独立复算最终值、峰值、后10轮和累计参与量。"""

        with (self.output_dir / "实验汇总.csv").open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        self.assertEqual(len(summary_rows), 4)

        for row in summary_rows:
            source_dir = self.source_by_scenario[row["场景"]]
            test_accuracy = np.asarray(
                [
                    float(line)
                    for line in (source_dir / "test_acc.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ],
                dtype=np.float64,
            )
            schedules = [
                json.loads(line)
                for line in (source_dir / "topology_schedule.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            self.assertEqual(len(test_accuracy), 150)
            self.assertAlmostEqual(float(row["最终测试准确率"]), test_accuracy[-1], places=12)
            self.assertAlmostEqual(float(row["最佳测试准确率"]), float(test_accuracy.max()), places=12)
            self.assertAlmostEqual(
                float(row["后10轮测试准确率均值"]),
                float(test_accuracy[-10:].mean()),
                places=12,
            )
            self.assertEqual(
                int(row["累计活跃客户端次"]),
                sum(int(record["active_client_count"]) for record in schedules),
            )

    def test_generated_tables_and_manifest_are_complete(self):
        """确认逐轮表为600行，清单声明七张主图且可信度边界存在。"""

        with (self.output_dir / "逐轮指标.csv").open("r", encoding="utf-8", newline="") as handle:
            round_rows = list(csv.DictReader(handle))
        with (self.output_dir / "共识准确率相关.csv").open(
                "r", encoding="utf-8", newline=""
        ) as handle:
            correlation_rows = list(csv.DictReader(handle))
        manifest = json.loads(
            (self.output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(len(round_rows), 600)
        self.assertEqual(len(correlation_rows), 88)
        self.assertTrue(
            {
                "边缘到云一致性A",
                "边缘云联合确定性C",
                "边缘云有效共识S",
                "边缘云有效共识尾随均值",
            }.issubset(round_rows[0])
        )
        self.assertEqual(manifest["confidence"], "可分享但附带限制")
        self.assertEqual(len(manifest["chart_map"]), 7)

    def test_all_figures_are_high_resolution_png(self):
        """确认七张主图均可解码、边长充足且PNG元数据约为300 DPI。"""

        figure_paths = sorted((self.output_dir / "figures").glob("*.png"))
        self.assertEqual(len(figure_paths), 7)
        for figure_path in figure_paths:
            with Image.open(str(figure_path)) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 3000)
                self.assertGreaterEqual(image.height, 1500)
                dpi = image.info.get("dpi", (0.0, 0.0))
                self.assertAlmostEqual(float(dpi[0]), 300.0, delta=1.0)
                self.assertAlmostEqual(float(dpi[1]), 300.0, delta=1.0)

    def test_report_contains_required_limitations(self):
        """确认报告明确披露代码漂移、单种子、探针标签和耗时限制。"""

        report = (self.output_dir / "分析报告.md").read_text(encoding="utf-8")
        required_phrases = [
            "trainer_test.py",
            "当前四份YAML配置为 200 轮",
            "topology_metadata.json 的参与均值覆盖MAT全部200行",
            "单随机种子",
            "没有探针标签",
            "不能比较训练耗时",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
