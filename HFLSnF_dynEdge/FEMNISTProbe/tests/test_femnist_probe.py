"""验证 FEMNIST 数据、MAT 循环、样本聚合和流式 HDF5。"""

from __future__ import absolute_import

import ast
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
import yaml

from FEMNISTProbe.analyze_suite import _cycle_summary
from FEMNISTProbe.data import load_femnist_experiment_data
from FEMNISTProbe.metrics import summarize_probe_observation
from FEMNISTProbe.model import FEMNISTChannelsLastCNN
from FEMNISTProbe.streaming_probe import (
    ProbeObservation,
    StreamingProbeH5Writer,
)
from FEMNISTProbe.topology import CyclicMatlabTopology
from FEMNISTProbe.trainer import FastFEMNISTMatTrainer
from topology_schedule import MatlabTopologySchedule


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEMNIST_DIR = PROJECT_ROOT / "dataset" / "FEMNIST"
MAT_PATH = (
    PROJECT_ROOT
    / "matlab"
    / "result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat"
)


def _probabilities(population_count, probe_count, class_count):
    """构造行和为一的确定性测试概率。"""
    values = np.arange(
        1,
        population_count * probe_count * class_count + 1,
        dtype=np.float32,
    ).reshape(population_count, probe_count, class_count)
    return values / np.sum(values, axis=-1, keepdims=True)


def _observation(epoch, candidate_count=3, edge_count=2):
    """构造一个可写入测试 HDF5 的完整探针时间点。"""
    return ProbeObservation(
        global_epoch=int(epoch),
        topology_cycle_index=max(0, int(epoch) // 2),
        mat_topology_index=max(-1, int(epoch) % 2),
        client_probabilities=_probabilities(candidate_count, 4, 3),
        edge_probabilities=_probabilities(edge_count, 4, 3),
        cloud_probabilities=_probabilities(1, 4, 3)[0],
        active_client_mask=np.asarray([True, False, True]),
        edge_active_mask=np.asarray([True, True]),
    )


class FEMNISTDataTests(unittest.TestCase):
    """核对真实 H5 的固定候选、类别覆盖和探针。"""

    @classmethod
    def setUpClass(cls):
        """只加载一次完整 FEMNIST，避免每项测试重复拼接测试集。"""
        cls.bundle = load_femnist_experiment_data(
            FEMNIST_DIR,
            candidate_count=37,
            candidate_seed=0,
            probe_samples_per_class=10,
            probe_seed=0,
        )

    def test_fixed_candidate_and_sample_counts(self):
        """固定候选应有37人，且聚合样本数等于训练标签数。"""
        self.assertEqual(len(self.bundle.candidate_writer_ids), 37)
        self.assertEqual(self.bundle.population_client_count, 3400)
        self.assertEqual(
            self.bundle.population_train_sample_count,
            671585,
        )
        self.assertEqual(
            int(np.sum(self.bundle.candidate_train_sample_counts)),
            8049,
        )
        for labels, sample_count in zip(
                self.bundle.candidate_train_labels,
                self.bundle.candidate_train_sample_counts,
        ):
            self.assertEqual(int(labels.numel()), int(sample_count))
        covered = torch.unique(torch.cat(
            self.bundle.candidate_train_labels
        )).cpu().numpy()
        np.testing.assert_array_equal(covered, np.arange(62))

    def test_full_test_and_balanced_probe(self):
        """完整测试集应有77483张，探针应为每类10张。"""
        self.assertEqual(
            tuple(self.bundle.global_test_inputs.shape),
            (77483, 1, 28, 28),
        )
        self.assertEqual(tuple(self.bundle.probe_inputs.shape), (620, 1, 28, 28))
        counts = np.bincount(self.bundle.probe_labels, minlength=62)
        np.testing.assert_array_equal(counts, np.full(62, 10))
        self.assertEqual(len(self.bundle.candidate_manifest_hash), 64)
        self.assertEqual(len(self.bundle.probe_hash), 64)


class CyclicTopologyTests(unittest.TestCase):
    """核对真实 MAT 的200轮循环边界与四场景读取。"""

    def _topology(self, architecture, snf_enabled, edge_mode):
        """创建一个固定 u=0.5 的真实 MAT 循环拓扑。"""
        schedule = MatlabTopologySchedule(
            mat_path=str(MAT_PATH),
            architecture=architecture,
            snf_enabled=snf_enabled,
            edge_mode=edge_mode,
            util=0.5,
            client_num_in_total=37,
            candidate_client_count=37,
        )
        return CyclicMatlabTopology(schedule, repeat_mode="cycle")

    def test_cycle_boundaries(self):
        """第199、200和4999轮应映射到正确循环及MAT行。"""
        topology = self._topology("hfl", True, "fixed")
        self.assertEqual(topology.round_count, 200)
        checks = {
            199: (0, 199),
            200: (1, 0),
            4999: (24, 199),
        }
        for epoch, expected in checks.items():
            current = topology.get_round(epoch)
            self.assertEqual(
                (current.topology_cycle_index, current.mat_topology_index),
                expected,
            )
            self.assertEqual(
                current.topology.round_index,
                current.mat_topology_index,
            )

    def test_all_four_scenarios_have_valid_slots(self):
        """四场景首轮活跃槽位均应属于固定37槽位。"""
        scenarios = [
            ("hfl", True, "fixed"),
            ("hfl", False, "fixed"),
            ("fl", True, "none"),
            ("fl", False, "none"),
        ]
        names = set()
        for architecture, snf_enabled, edge_mode in scenarios:
            topology = self._topology(
                architecture, snf_enabled, edge_mode
            )
            names.add(topology.schedule.scenario_name)
            slots = topology.get_round(0).topology.active_candidate_slots
            self.assertGreater(len(slots), 0)
            self.assertTrue(all(0 <= int(slot) < 37 for slot in slots))
        self.assertEqual(len(names), 4)


class AggregationAndMetricTests(unittest.TestCase):
    """验证真实样本数加权和覆盖率加权有效共识。"""

    def test_flat_sample_weighted_aggregation(self):
        """扁平聚合结果应严格使用客户端训练样本数。"""
        trainer = FastFEMNISTMatTrainer.__new__(FastFEMNISTMatTrainer)
        trainer.device = torch.device("cpu")
        vectors = torch.tensor([
            [1.0, 3.0],
            [5.0, 7.0],
        ])
        actual = trainer._aggregate_vectors(vectors, [1, 3])
        expected = torch.tensor([4.0, 6.0])
        torch.testing.assert_close(actual, expected)

    def test_channels_last_model_accepts_explicit_channel(self):
        """兼容CNN应接受四维输入并保持旧版参数规模。"""
        model = FEMNISTChannelsLastCNN()
        inputs = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        outputs = model(inputs.contiguous(memory_format=torch.channels_last))
        self.assertEqual(tuple(outputs.shape), (2, 62))
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            1206590,
        )

    def test_q_equals_coverage_times_active_correct_s(self):
        """主指标Q应等于活跃覆盖率乘活跃正确有效共识。"""
        clients = _probabilities(3, 4, 3)
        edges = _probabilities(1, 4, 3)
        cloud = _probabilities(1, 4, 3)[0]
        summary = summarize_probe_observation(
            global_epoch=0,
            topology_cycle_index=0,
            mat_topology_index=0,
            client_probabilities=clients,
            edge_probabilities=edges,
            cloud_probabilities=cloud,
            active_client_mask=np.asarray([True, False, True]),
            edge_active_mask=np.asarray([True]),
            true_labels=np.asarray([0, 1, 2, 0]),
            groups={0: [0, 2]},
        )
        self.assertAlmostEqual(summary["active_coverage"], 2.0 / 3.0)
        self.assertAlmostEqual(
            summary["coverage_weighted_active_correct_effective"],
            summary["active_coverage"]
            * summary["active_correct_effective"],
        )

    def test_cycle_summary_uses_four_evaluation_points(self):
        """一个200轮周期应汇总第50、100、150和200轮。"""
        rows = []
        for global_epoch in [49, 99, 149, 199]:
            rows.append({
                "global_epoch": str(global_epoch),
                "topology_cycle_index": "0",
                "test_accuracy": "0.5",
                "candidate_effective": "0.4",
                "active_coverage": "0.75",
                "active_effective": "0.3",
                "active_correct_effective": "0.2",
                "active_wrong_effective": "0.1",
                "coverage_weighted_active_correct_effective": "0.15",
                "within_edge_effective": "0.25",
                "edge_effective": "0.2",
                "edge_cloud_effective": "0.1",
            })
        summary = _cycle_summary(rows, 0)
        self.assertAlmostEqual(summary["test_accuracy"], 0.5)
        self.assertAlmostEqual(summary["active_coverage"], 0.75)
        self.assertAlmostEqual(summary["q"], 0.15)


class StreamingH5Tests(unittest.TestCase):
    """验证流式写入形状、概率和检查点续写坐标。"""

    def test_create_and_resume_without_duplicate_rows(self):
        """恢复后应从written_count继续写入而不覆盖已提交行。"""
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "probe.h5"
            writer = StreamingProbeH5Writer(
                output_path=output_path,
                observation_count=3,
                candidate_count=3,
                edge_slot_count=2,
                probe_labels=np.asarray([0, 1, 2, 0]),
                probe_indices=np.asarray([10, 11, 12, 13]),
                class_count=3,
                probe_hash="unit-test-probe",
                resume_count=0,
            )
            writer.submit(_observation(-1))
            writer.close()

            resumed = StreamingProbeH5Writer(
                output_path=output_path,
                observation_count=3,
                candidate_count=3,
                edge_slot_count=2,
                probe_labels=np.asarray([0, 1, 2, 0]),
                probe_indices=np.asarray([10, 11, 12, 13]),
                class_count=3,
                probe_hash="unit-test-probe",
                resume_count=1,
            )
            resumed.submit(_observation(49))
            resumed.submit(_observation(99))
            resumed.close()

            with h5py.File(str(output_path), "r") as archive:
                self.assertEqual(int(archive.attrs["written_count"]), 3)
                self.assertEqual(
                    archive["client_probabilities"].shape,
                    (3, 3, 4, 3),
                )
                np.testing.assert_array_equal(
                    archive["global_epochs"][:],
                    np.asarray([-1, 49, 99]),
                )
                probabilities = archive["cloud_probabilities"][:]
                np.testing.assert_allclose(
                    np.sum(probabilities, axis=-1),
                    1.0,
                    atol=1e-6,
                )


class DocumentationTests(unittest.TestCase):
    """审计新增 Python 函数和类方法的说明字符串。"""

    def test_all_functions_and_methods_have_docstrings(self):
        """FEMNISTProbe下每个函数及方法都必须具有说明。"""
        missing = []
        for path in sorted((PROJECT_ROOT / "FEMNISTProbe").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and ast.get_docstring(node) is None:
                    missing.append("{}:{}".format(path.name, node.name))
        self.assertEqual(missing, [])

    def test_yaml_comments_are_english_and_formal_configs_match_plan(self):
        """新增YAML注释应为英文，四个正式配置应共享核心不变量。"""
        config_paths = sorted(
            (PROJECT_ROOT / "FEMNISTProbe" / "configs").glob("*.yaml")
        )
        self.assertEqual(len(config_paths), 4)
        scenarios = set()
        for path in config_paths:
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                comment = line.split("#", 1)[1] if "#" in line else ""
                self.assertTrue(all(ord(character) < 128 for character in comment))
            configuration = yaml.safe_load(text)
            data_args = configuration["data_args"]
            train_args = configuration["train_args"]
            self.assertEqual(data_args["dataset"], "femnist")
            self.assertEqual(data_args["candidate_seed"], 0)
            self.assertEqual(data_args["client_num_in_total"], 37)
            self.assertEqual(train_args["comm_round"], 5000)
            self.assertEqual(train_args["batch_size"], 20)
            self.assertEqual(train_args["topology_repeat_mode"], "cycle")
            self.assertEqual(train_args["topology_util"], 0.5)
            scenarios.add((
                train_args["topology_architecture"],
                bool(train_args["topology_snf"]),
            ))
        self.assertEqual(
            scenarios,
            {
                ("hfl", True),
                ("hfl", False),
                ("fl", True),
                ("fl", False),
            },
        )


if __name__ == "__main__":
    unittest.main()
