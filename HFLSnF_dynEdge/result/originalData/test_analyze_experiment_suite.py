"""验证四组实验分析中的共识、平滑、客户端映射和结果审计逻辑。"""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from analyze_experiment_suite import (
    SCENARIO_ORDER,
    REQUIRED_RESULT_FILES,
    build_batch_profile,
    build_fixed_candidate_activity_matrix,
    consensus_components,
    discover_experiment_dirs,
    first_stable_epoch,
    generalized_js_divergence,
    historical_best,
    load_experiment,
    map_client_ids_to_slots,
    normalized_entropy,
    parse_args,
    read_probability_csv,
    run_analysis,
    trailing_mean,
    validate_experiment,
    validate_probability_vector,
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


class ProbabilityCsvTest(unittest.TestCase):
    """验证探针CSV解析会保留空边缘槽位并正确读取概率。"""

    def test_empty_edge_slots_keep_their_column_positions(self):
        """确认中间空列不会被删除，非空概率仍保持原始列索引。"""

        vector = json.dumps([0.1] * 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edge.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(["", vector, "", vector, "", ""])
            rows = read_probability_csv(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 6)
        self.assertIsNone(rows[0][0])
        self.assertIsNotNone(rows[0][1])
        self.assertIsNone(rows[0][2])
        self.assertIsNotNone(rows[0][3])

    def test_illegal_probability_vector_is_rejected(self):
        """确认维度错误、负概率和概率和错误不会通过合法性检查。"""

        self.assertFalse(validate_probability_vector(np.asarray([0.5, 0.5]), 10))
        self.assertFalse(validate_probability_vector(np.asarray([-0.1] + [1.1] + [0.0] * 8), 10))
        self.assertFalse(validate_probability_vector(np.full(10, 0.2), 10))


class CommandLineInterfaceTest(unittest.TestCase):
    """验证新增终端参数与旧版兼容参数的解析行为。"""

    def test_input_output_root_and_smooth_window_are_parsed(self):
        """确认批次目录、输出根目录和平滑窗口可同时传入。"""

        args = parse_args([
            "--input-dir", "result/originalData/1",
            "--output-root", "result/1结果和分析",
            "--smooth-window", "12",
        ])
        self.assertEqual(args.input_dir, Path("result/originalData/1"))
        self.assertEqual(args.output_root, Path("result/1结果和分析"))
        self.assertEqual(args.smooth_window, 12)

    def test_legacy_result_root_is_preserved(self):
        """确认旧版result-root参数仍可单独使用。"""

        args = parse_args(["--result-root", "result/originalData/1"])
        self.assertEqual(args.result_root, Path("result/originalData/1"))

    def test_conflicting_input_parameters_fail(self):
        """确认两个输入目录参数不会被静默覆盖。"""

        with self.assertRaises(SystemExit):
            parse_args([
                "--input-dir", "result/originalData/1",
                "--result-root", "result/originalData/1",
            ])

    def test_empty_batch_has_clear_discovery_error(self):
        """确认只有说明文件的空批次会明确报告四个场景缺失。"""

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "自动发现实验失败"):
                discover_experiment_dirs(Path(directory))


class DiscoveryValidationTest(unittest.TestCase):
    """验证不存在、缺文件和重复场景等批次发现失败路径。"""

    @staticmethod
    def _create_stub_experiment(parent: Path, name: str, scenario: str) -> Path:
        """创建仅供目录发现测试使用的最小文件集合。"""

        directory = parent / name
        directory.mkdir()
        # 发现阶段只读取元数据场景并检查文件是否存在，无需伪造具体实验内容。
        for filename in REQUIRED_RESULT_FILES:
            content = json.dumps({"scenario": scenario}) if filename == "topology_metadata.json" else ""
            (directory / filename).write_text(content, encoding="utf-8")
        return directory

    def test_nonexistent_batch_is_rejected(self):
        """确认不存在的批次目录会报告完整绝对路径。"""

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "不存在"
            with self.assertRaisesRegex(FileNotFoundError, "输入批次目录不存在"):
                discover_experiment_dirs(missing)

    def test_recognized_experiment_with_missing_files_is_rejected(self):
        """确认已识别场景缺少结果文件时会列出缺失内容。"""

        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "incomplete"
            experiment.mkdir()
            (experiment / "topology_metadata.json").write_text(
                json.dumps({"scenario": "hfl_snf_fixed"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "缺少"):
                discover_experiment_dirs(Path(directory))

    def test_duplicate_scenario_is_rejected(self):
        """确认同一场景出现两组完整目录时不会任意选择其一。"""

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            self._create_stub_experiment(parent, "first", "hfl_snf_fixed")
            self._create_stub_experiment(parent, "second", "hfl_snf_fixed")
            with self.assertRaisesRegex(ValueError, "匹配到 2 个目录"):
                discover_experiment_dirs(parent)


class CurrentResultIntegrationTest(unittest.TestCase):
    """使用originalData/1四组原始结果验证200轮调度与探针不变量。"""

    @classmethod
    def setUpClass(cls):
        """发现并加载result/originalData/1中的四组目标实验。"""

        cls.result_root = Path(__file__).resolve().parent / "1"
        if not cls.result_root.is_dir():
            raise unittest.SkipTest("当前工作区没有 result/originalData/1 目录")
        try:
            experiment_dirs = discover_experiment_dirs(cls.result_root)
        except ValueError as exc:
            raise unittest.SkipTest("未发现唯一的四组目标结果：{}".format(exc))
        cls.experiments = [load_experiment(path) for path in experiment_dirs]

    def test_all_four_experiments_have_200_rounds(self):
        """确认四种场景齐全，且每种场景都有200轮实际输出。"""

        self.assertEqual(
            [experiment.scenario for experiment in self.experiments],
            SCENARIO_ORDER,
        )
        self.assertEqual([len(experiment.schedule) for experiment in self.experiments], [200] * 4)

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
        """确认四方案全部轮次共享同一组、同一顺序的固定37人。"""

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
        self.assertEqual(len(set(reference)), 1)

    def test_zero_rounds_keep_full_distribution_without_aggregation(self):
        """确认零参与轮不聚合，但仍向0至199号客户端全量下发。"""

        for experiment in self.experiments:
            zero_records = [
                record for record in experiment.schedule
                if int(record["active_client_count"]) == 0
            ]
            self.assertGreater(len(zero_records), 0, msg=experiment.label)
            for record in zero_records:
                self.assertFalse(record["aggregated"])
                self.assertEqual(record["distributed_client_count"], 200)
                self.assertEqual(record["distributed_client_indexes"], list(range(200)))

    def test_probe_widths_and_hfl_edge_slots_are_preserved(self):
        """确认客户端37列、HFL边缘6列、FL边缘1空列和云端1列。"""

        for experiment in self.experiments:
            self.assertTrue(all(len(row) == 37 for row in experiment.client_probe))
            self.assertTrue(all(len(row) == 1 for row in experiment.cloud_probe))
            if experiment.scenario.startswith("hfl_"):
                self.assertTrue(all(len(row) == 6 for row in experiment.edge_probe))
            else:
                self.assertTrue(all(row == [None] for row in experiment.edge_probe))

    def test_fixed_candidate_activity_matrix_matches_jsonl(self):
        """确认37×200活跃热图逐轮人数与JSONL记录完全一致。"""

        for experiment in self.experiments:
            matrix = build_fixed_candidate_activity_matrix(experiment)
            self.assertEqual(matrix.shape, (37, 200))
            expected = [record["active_client_count"] for record in experiment.schedule]
            np.testing.assert_array_equal(matrix.sum(axis=0), expected)

    def test_different_round_counts_are_rejected(self):
        """确认四方案轮数不同时会在生成报告前终止。"""

        shortened = replace(
            self.experiments[-1], schedule=self.experiments[-1].schedule[:-1]
        )
        with self.assertRaisesRegex(ValueError, "关键元数据不一致"):
            build_batch_profile(
                self.result_root,
                list(self.experiments[:-1]) + [shortened],
            )


class GeneratedArtifactTest(unittest.TestCase):
    """独立读取原始文件，复核已生成分析包的数字、行数和图片规格。"""

    @classmethod
    def setUpClass(cls):
        """在临时目录生成分析包，并建立场景到原始实验目录的映射。"""

        cls.workspace = Path(__file__).resolve().parent.parent.parent
        cls.result_root = Path(__file__).resolve().parent / "1"
        if not cls.result_root.is_dir():
            raise unittest.SkipTest("当前工作区没有 result/originalData/1 目录")
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.output_dir = run_analysis(
            cls.result_root,
            Path(cls.temporary_directory.name) / "analysis",
            smooth_window=10,
        )
        experiment_dirs = discover_experiment_dirs(cls.result_root)
        cls.source_by_scenario = {}
        for directory in experiment_dirs:
            metadata = json.loads(
                (directory / "topology_metadata.json").read_text(encoding="utf-8")
            )
            cls.source_by_scenario[str(metadata["scenario"])] = directory

    @classmethod
    def tearDownClass(cls):
        """删除集成测试生成的临时分析包。"""

        if hasattr(cls, "temporary_directory"):
            cls.temporary_directory.cleanup()

    def test_summary_numbers_match_raw_files(self):
        """从原始指标和JSONL独立复算最终值、峰值、后20轮和累计参与量。"""

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

            self.assertEqual(len(test_accuracy), 200)
            self.assertAlmostEqual(float(row["最终测试准确率"]), test_accuracy[-1], places=12)
            self.assertAlmostEqual(float(row["最佳测试准确率"]), float(test_accuracy.max()), places=12)
            self.assertAlmostEqual(
                float(row["后20轮测试准确率均值"]),
                float(test_accuracy[-20:].mean()),
                places=12,
            )
            self.assertAlmostEqual(
                float(row["后20轮测试准确率标准差"]),
                float(test_accuracy[-20:].std()),
                places=12,
            )
            self.assertEqual(
                int(row["累计活跃客户端次"]),
                sum(int(record["active_client_count"]) for record in schedules),
            )

    def test_generated_tables_and_manifest_are_complete(self):
        """确认逐轮表为800行，固定候选表为148行且清单声明八张主图。"""

        with (self.output_dir / "逐轮指标.csv").open("r", encoding="utf-8", newline="") as handle:
            round_rows = list(csv.DictReader(handle))
        with (self.output_dir / "共识准确率相关.csv").open(
                "r", encoding="utf-8", newline=""
        ) as handle:
            correlation_rows = list(csv.DictReader(handle))
        with (self.output_dir / "固定候选参与统计.csv").open(
                "r", encoding="utf-8-sig", newline=""
        ) as handle:
            client_rows = list(csv.DictReader(handle))
        manifest = json.loads(
            (self.output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual(len(round_rows), 800)
        self.assertEqual(len(correlation_rows), 88)
        self.assertEqual(len(client_rows), 148)
        self.assertTrue(
            {
                "边缘到云一致性A",
                "边缘云联合确定性C",
                "边缘云有效共识S",
                "边缘云有效共识尾随均值",
                "MAT活跃候选槽位",
                "MAT分组候选槽位",
            }.issubset(round_rows[0])
        )
        self.assertEqual(manifest["confidence"], "可分享但附带限制")
        self.assertEqual(len(manifest["chart_map"]), 8)
        self.assertIn(
            "figures\\08_四方案候选有效共识S对比.png",
            [item["file"] for item in manifest["chart_map"]],
        )

    def test_candidate_consensus_comparison_matches_round_rows(self):
        """确认四方案候选S后20轮均值和标准差可由逐轮表独立复算。"""

        with (self.output_dir / "实验汇总.csv").open(
                "r", encoding="utf-8", newline=""
        ) as handle:
            summary_rows = list(csv.DictReader(handle))
        with (self.output_dir / "逐轮指标.csv").open(
                "r", encoding="utf-8", newline=""
        ) as handle:
            round_rows = list(csv.DictReader(handle))

        for summary in summary_rows:
            values = np.asarray([
                float(row["候选有效共识S"])
                for row in round_rows
                if row["场景"] == summary["场景"]
            ])
            self.assertEqual(values.size, 200)
            self.assertAlmostEqual(
                float(summary["后20轮有效共识S"]), float(values[-20:].mean()), places=12
            )
            self.assertAlmostEqual(
                float(summary["后20轮有效共识S标准差"]),
                float(values[-20:].std()), places=12,
            )

    def test_all_figures_are_high_resolution_png(self):
        """确认八张主图均可解码、边长充足且PNG元数据约为300 DPI。"""

        figure_paths = sorted((self.output_dir / "figures").glob("*.png"))
        self.assertEqual(len(figure_paths), 8)
        for figure_path in figure_paths:
            with Image.open(str(figure_path)) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreaterEqual(image.width, 3000)
                self.assertGreaterEqual(image.height, 1500)
                dpi = image.info.get("dpi", (0.0, 0.0))
                self.assertAlmostEqual(float(dpi[0]), 300.0, delta=1.0)
                self.assertAlmostEqual(float(dpi[1]), 300.0, delta=1.0)

    def test_report_contains_required_limitations(self):
        """确认报告披露固定候选、单种子、探针标签、MAT路径和耗时限制。"""

        report = (self.output_dir / "分析报告.md").read_text(encoding="utf-8")
        required_phrases = [
            "trainer_test.py",
            "当前工作区四份YAML配置轮数为",
            "固定候选顺序",
            "MAT绝对路径",
            "单随机种子",
            "没有真实标签",
            "四方案候选有效共识S对比",
            "当前排序第一的是",
            "不能用于比较4090与4060训练速度",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, report)


if __name__ == "__main__":
    unittest.main()
