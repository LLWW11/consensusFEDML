"""验证 FEMNIST 数据、MAT 循环、样本聚合和流式 HDF5。"""

from __future__ import absolute_import

import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
import torch
import yaml

from FEMNISTProbe.analyze_suite import _cycle_summary
from FEMNISTProbe.data import (
    FIXED_CANDIDATE_CLIENT_IDS,
    _assign_samples_to_clients,
    _build_dirichlet_proportions,
    _hash_partition,
    build_fixed_candidate_client_ids,
    load_femnist_experiment_data,
)
from FEMNISTProbe.metrics import summarize_probe_observation
from FEMNISTProbe.model import FEMNISTChannelsLastCNN
from FEMNISTProbe.run_suite import _run_group
from FEMNISTProbe.streaming_probe import (
    ProbeObservation,
    StreamingProbeH5Writer,
)
from FEMNISTProbe.topology import CyclicMatlabTopology
from FEMNISTProbe.trainer import FastFEMNISTMatTrainer
from topology_schedule import (
    MatlabTopologySchedule,
    build_balanced_candidate_groups,
)


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
            client_count=250,
            candidate_count=37,
            partition_alpha=0.2,
            partition_seed=0,
            probe_samples_per_class=10,
            probe_seed=0,
        )

    def test_fixed_candidate_and_sample_counts(self):
        """固定候选应严格匹配指定37槽位，且逻辑客户端划分完整非空。"""
        self.assertEqual(
            self.bundle.candidate_client_ids,
            list(FIXED_CANDIDATE_CLIENT_IDS),
        )
        self.assertEqual(self.bundle.population_client_count, 250)
        self.assertEqual(self.bundle.source_writer_count, 3400)
        self.assertEqual(
            self.bundle.population_train_sample_count,
            671585,
        )
        self.assertEqual(
            int(np.sum(self.bundle.candidate_train_sample_counts)),
            100127,
        )
        self.assertEqual(
            int(np.sum(self.bundle.client_train_sample_counts)), 671585
        )
        self.assertEqual(
            int(np.sum(self.bundle.client_test_sample_counts)), 77483
        )
        self.assertGreater(int(np.min(self.bundle.client_train_sample_counts)), 0)
        self.assertGreater(int(np.min(self.bundle.client_test_sample_counts)), 0)
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
        self.assertEqual(
            tuple(self.bundle.global_test_client_ids.shape), (77483,)
        )
        np.testing.assert_array_equal(
            np.bincount(
                self.bundle.global_test_client_ids.numpy(), minlength=250
            ),
            self.bundle.client_test_sample_counts,
        )
        counts = np.bincount(self.bundle.probe_labels, minlength=62)
        np.testing.assert_array_equal(counts, np.full(62, 10))
        self.assertEqual(len(self.bundle.candidate_manifest_hash), 64)
        self.assertEqual(len(self.bundle.partition_hash), 64)
        self.assertEqual(len(self.bundle.probe_hash), 64)

    def test_fixed_candidate_builder_rejects_other_shapes(self):
        """固定槽位构造器必须只接受正式250到37映射。"""
        self.assertEqual(
            build_fixed_candidate_client_ids(250, 37),
            list(FIXED_CANDIDATE_CLIENT_IDS),
        )
        with self.assertRaises(ValueError):
            build_fixed_candidate_client_ids(200, 37)

    def test_dirichlet_partition_is_reproducible_and_alpha_is_hashed(self):
        """相同种子应复现比例与归属，修改alpha必须改变划分哈希。"""
        labels = np.repeat(np.arange(62, dtype=np.int64), 20)
        first = _build_dirichlet_proportions(8, 0.2, 0)
        repeated = _build_dirichlet_proportions(8, 0.2, 0)
        changed = _build_dirichlet_proportions(8, 0.5, 0)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, changed))
        first_owners = _assign_samples_to_clients(labels, first, 1)
        repeated_owners = _assign_samples_to_clients(labels, repeated, 1)
        np.testing.assert_array_equal(first_owners, repeated_owners)
        first_hash = _hash_partition(
            first_owners, first_owners, first, 8, 0.2, 0
        )
        changed_hash = _hash_partition(
            first_owners, first_owners, changed, 8, 0.5, 0
        )
        self.assertNotEqual(first_hash, changed_hash)


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
            client_num_in_total=250,
            candidate_client_count=37,
            assignment_mode="balanced_counts",
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
        """四场景首轮均应按 MAT k/n 构造合法平衡槽位。"""
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
            groups = topology.get_round(0).topology.copy_groups()
            self.assertEqual(len(groups), len(set(groups)))
            flattened = [
                slot for group_slots in groups.values() for slot in group_slots
            ]
            self.assertEqual(len(flattened), len(slots))
            self.assertEqual(len(set(flattened)), len(flattened))
        self.assertEqual(len(names), 4)

    def test_balanced_group_examples(self):
        """验证 k=4,n=21 和 k=1,n=6 的连续候选段均分结果。"""
        groups = build_balanced_candidate_groups(37, 4, 21)
        self.assertEqual(
            {group_id: len(slots) for group_id, slots in groups.items()},
            {0: 6, 1: 5, 2: 5, 3: 5},
        )
        self.assertEqual(groups[0], (0, 1, 2, 3, 4, 5))
        self.assertEqual(groups[1], (9, 10, 11, 12, 13))
        self.assertEqual(
            build_balanced_candidate_groups(37, 1, 6),
            {0: (0, 1, 2, 3, 4, 5)},
        )

    def test_all_mat_rounds_match_balanced_k_and_n(self):
        """遍历真实 MAT 四场景200轮并核对平衡分组人数与唯一性。"""
        scenarios = [
            ("hfl", True, "fixed"),
            ("hfl", False, "fixed"),
            ("fl", True, "none"),
            ("fl", False, "none"),
        ]
        for architecture, snf_enabled, edge_mode in scenarios:
            topology = self._topology(
                architecture, snf_enabled, edge_mode
            )
            for round_index in range(topology.round_count):
                current = topology.get_round(round_index).topology
                groups = current.copy_groups()
                flattened = [
                    slot
                    for group_slots in groups.values()
                    for slot in group_slots
                ]
                self.assertEqual(len(flattened), current.participant_count)
                self.assertEqual(len(flattened), len(set(flattened)))
                self.assertTrue(all(0 <= slot < 37 for slot in flattened))
                self.assertEqual(
                    max(len(group) for group in groups.values())
                    - min(len(group) for group in groups.values()),
                    1 if current.participant_count % len(groups) else 0,
                )


class AggregationAndMetricTests(unittest.TestCase):
    """验证真实样本数加权和覆盖率加权有效共识。"""

    def test_round_progress_prints_mat_counts_and_logical_groups(self):
        """每轮进度应沿用旧标题并打印MAT的k/n及逻辑客户端分组。"""
        trainer = object.__new__(FastFEMNISTMatTrainer)
        trainer.comm_round = 5000
        trainer.data = SimpleNamespace(
            candidate_client_ids=[123, 124, 41]
        )
        trainer.topology = SimpleNamespace(
            schedule=SimpleNamespace(scenario_name="hfl_snf_fixed")
        )
        cyclic_round = SimpleNamespace(
            topology_cycle_index=2,
            mat_topology_index=17,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            trainer._print_round_progress(
                global_epoch=417,
                cyclic_round=cyclic_round,
                groups={0: [0, 1], 1: [2]},
                active_slots=[0, 1, 2],
            )
        text = output.getvalue()
        self.assertIn(
            "################Global Communication Round : 417", text
        )
        self.assertIn("epoch=418/5000", text)
        self.assertIn("mat_cycle=2", text)
        self.assertIn("mat_round=17", text)
        self.assertIn("k=2, n=3", text)
        self.assertIn("{0: [123, 124], 1: [41]}", text)

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

    def test_client_partition_test_summary_matches_central_predictions(self):
        """250端分区累计的正确数应等于同一云模型集中预测的正确数。"""
        trainer = FastFEMNISTMatTrainer.__new__(FastFEMNISTMatTrainer)
        trainer.device = torch.device("cpu")
        trainer.population_client_count = 250
        trainer.profile = {"test_batch_size": 17}
        trainer.test_cached_on_device = True
        trainer.cloud_vector = torch.zeros(1)
        trainer.global_test_inputs = torch.linspace(
            -1.0, 1.0, 250
        ).reshape(250, 1)
        trainer.global_test_labels = (
            trainer.global_test_inputs[:, 0] < 0
        ).long()
        trainer.global_test_client_ids = torch.arange(250, dtype=torch.long)
        trainer.data = SimpleNamespace(
            global_test_labels=trainer.global_test_labels,
            client_test_sample_counts=np.ones(250, dtype=np.int64),
        )
        trainer.model = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            trainer.model.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        trainer._copy_vector_to_model = lambda model_vector: None
        trainer._autocast = lambda: contextlib.nullcontext()

        row = trainer._evaluate_cloud(0, 0, 0)
        with torch.inference_mode():
            expected = int(torch.sum(
                torch.argmax(trainer.model(trainer.global_test_inputs), dim=1)
                == trainer.global_test_labels
            ).item())
        self.assertEqual(row["evaluated_client_count"], 250)
        self.assertEqual(row["test_samples"], 250)
        self.assertEqual(row["test_correct"], expected)
        self.assertAlmostEqual(row["test_accuracy"], expected / 250.0)

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

    def test_serial_suite_output_is_tee_written(self):
        """串行套件应把子进程日志同时写入终端捕获和任务日志。"""
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_dir = Path(temporary_directory) / "logs"
            with contextlib.redirect_stdout(output):
                elapsed, return_codes = _run_group(
                    [[
                        sys.executable,
                        "-c",
                        "print('round-live-marker', flush=True)",
                    ]],
                    log_dir,
                    parallelism=1,
                )
            self.assertEqual(return_codes, [0])
            self.assertGreaterEqual(elapsed, 0.0)
            self.assertIn("round-live-marker", output.getvalue())
            self.assertIn(
                "round-live-marker",
                (log_dir / "job_00.log").read_text(encoding="utf-8"),
            )

    def test_one_click_script_uses_formal_configs_and_live_output(self):
        """一键脚本应校验指定MAT并以串行实时输出方式启动正式套件。"""
        script_path = (
            PROJECT_ROOT / "run_femnist_250_four_experiments.ps1"
        )
        text = script_path.read_text(encoding="utf-8")
        self.assertIn(
            "result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat", text
        )
        for config_name in [
                "femnist_hfl_snf_u05_5000.yaml",
                "femnist_hfl_no_snf_u05_5000.yaml",
                "femnist_fl_snf_u05_5000.yaml",
                "femnist_fl_no_snf_u05_5000.yaml",
        ]:
            self.assertIn(config_name, text)
        self.assertIn("--no-capture-output", text)
        self.assertIn("--mode formal", text)
        self.assertIn("--parallel 1", text)

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
            self.assertEqual(data_args["partition_method"], "dirichlet")
            self.assertEqual(data_args["split_policy"], "non-iid")
            self.assertEqual(data_args["partition_alpha"], 0.2)
            self.assertEqual(data_args["partition_seed"], 0)
            self.assertEqual(data_args["client_num_in_total"], 250)
            self.assertEqual(data_args["client_num_per_round"], 37)
            self.assertEqual(train_args["comm_round"], 5000)
            self.assertEqual(train_args["batch_size"], 20)
            self.assertEqual(train_args["topology_repeat_mode"], "cycle")
            self.assertEqual(
                train_args["topology_assignment_mode"], "balanced_counts"
            )
            self.assertEqual(
                train_args["dynamic_group_mat_file"],
                (
                    "matlab/result-U-6fixedge_epoch200_"
                    "varAlpha_0p1_trainable.mat"
                ),
            )
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
