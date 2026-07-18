import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from analyze_consensus import read_true_labels
from client_test import HFLClient
from probe_batch import (
    FixedProbeSet,
    ProbeBatchRecorder,
    calculate_population_probe_metrics,
    select_fixed_balanced_probe,
)
from topology_schedule import MatlabTopologySchedule
from trainer_test import HierarchicalTrainer


class _FakeTrainingClient:
    """记录测试期间客户端是否完成了本地训练和模型同步。"""

    def __init__(self, client_idx, group_comm_round=1, epochs=1):
        """保存客户端编号、展平参数以及调用记录。"""
        self.client_idx = client_idx
        self.group_comm_round = group_comm_round
        self.epochs = epochs
        self.train_calls = []
        self.synced_states = []

    def train_one_epoch(self, global_round_idx, group_round_idx, epoch_idx):
        """记录一次本地训练并返回与真实客户端一致的全局 epoch。"""
        self.train_calls.append((global_round_idx, group_round_idx, epoch_idx))
        global_epoch = (
            global_round_idx * self.group_comm_round * self.epochs
            + group_round_idx * self.epochs
            + epoch_idx
        )
        return global_epoch, {"client_idx": self.client_idx}

    def set_local_model_state(self, model_state):
        """记录一次云模型下发，供全量同步断言使用。"""
        self.synced_states.append(model_state)


class _FakeEvaluationClient:
    """返回预设训练集和测试集指标的本地评估客户端。"""

    def __init__(self, train_metrics, test_metrics):
        """保存两类评估指标并初始化调用记录。"""
        self.local_test_data = object()
        self.train_metrics = dict(train_metrics)
        self.test_metrics = dict(test_metrics)
        self.evaluation_calls = []

    def evaluate_local_model(self, use_test_dataset):
        """记录本地模型评估，并返回对应数据分区的指标。"""
        self.evaluation_calls.append(use_test_dataset)
        if use_test_dataset:
            return dict(self.test_metrics)
        return dict(self.train_metrics)


class _FakeProbeClient:
    """根据客户端编号生成可识别的探针概率向量。"""

    def __init__(self, client_idx):
        """保存用于构造探针结果的客户端编号。"""
        self.client_idx = client_idx

    def predict_proba(self, sample_x, model_state=None):
        """返回首元素为客户端编号的十维测试向量。"""
        del sample_x, model_state
        return [float(self.client_idx)] + [0.0] * 9


class _FakeModel:
    """提供训练主循环初始化所需的最小模型状态接口。"""

    def __init__(self, model_state=None):
        """保存可由主循环读取的初始模型参数。"""
        self.model_state = model_state if model_state is not None else {"weight": "initial"}

    def state_dict(self):
        """返回当前测试模型参数。"""
        return self.model_state


class HierarchicalSamplingTest(unittest.TestCase):
    """验证固定候选、MAT 精确映射、活跃训练、全量下发和本地评估。"""

    @classmethod
    def setUpClass(cls):
        """加载正式 MAT 的 HFL-SnF-fixed、利用率 0.5 拓扑。"""
        mat_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "matlab",
            "result-U-6fixedge_epoch200.mat",
        )
        cls.schedule = MatlabTopologySchedule(
            mat_path,
            architecture="hfl",
            snf_enabled=True,
            edge_mode="fixed",
            util=0.5,
            client_num_in_total=200,
            candidate_client_count=37,
        )

    def _create_sampling_trainer(self, seed=7, distribution_scope="all"):
        """构造只包含本组单元测试所需属性的轻量训练器。"""
        trainer = HierarchicalTrainer.__new__(HierarchicalTrainer)
        trainer.args = SimpleNamespace(
            client_num_in_total=200,
            client_num_per_round=37,
            random_seed=seed,
            group_method="matlab",
            group_comm_round=1,
            epochs=1,
            group_num=self.schedule.group_capacity,
            enable_wandb=False,
        )
        trainer.model_distribution_scope = distribution_scope
        trainer.topology_schedule = self.schedule
        trainer.current_round_topology = None
        trainer.fixed_candidate_client_indexes = None
        return trainer

    def _prepare_train_loop_trainer(self, global_epoch):
        """构造可验证正常或零参与主循环下发行为的训练器。"""
        trainer = self._create_sampling_trainer(seed=17, distribution_scope="all")
        trainer.args.comm_round = 1
        trainer._initialize_fixed_candidate_clients()
        trainer.model = _FakeModel({"weight": "previous"})
        trainer._setup_result_dir = Mock()
        trainer._initialize_metric_output_files = Mock()
        trainer._is_consensus_probe_enabled = Mock(return_value=False)
        trainer._calculate_global_epoch = Mock(return_value=global_epoch)
        trainer._train_active_clients_one_epoch = Mock(return_value=global_epoch)
        trainer.lifecycle_calls = []
        trainer._sync_clients_to_global = Mock(
            side_effect=lambda *_args: trainer.lifecycle_calls.append("sync")
        )
        trainer._evaluate_all_client_local_models = Mock(
            side_effect=lambda *_args: trainer.lifecycle_calls.append("evaluate")
        )
        trainer._append_runtime_topology_record = Mock()
        return trainer

    def test_fixed_candidates_are_initialized_once_and_stable_across_epochs(self):
        """验证固定 37 人只初始化一次，跨多个 MAT epoch 保持顺序不变。"""
        trainer = self._create_sampling_trainer(seed=11)

        with patch.object(
                trainer,
                "_initialize_fixed_candidate_clients",
                wraps=trainer._initialize_fixed_candidate_clients,
        ) as initialize_candidates:
            trainer._build_round_groups(0)
            first_candidates = list(trainer.fixed_candidate_client_indexes)
            trainer._build_round_groups(1)
            trainer._build_round_groups(5)

        self.assertEqual(initialize_candidates.call_count, 1)
        self.assertEqual(first_candidates, trainer.fixed_candidate_client_indexes)
        self.assertEqual(len(first_candidates), 37)
        self.assertEqual(len(set(first_candidates)), 37)
        self.assertTrue(all(0 <= client_idx < 200 for client_idx in first_candidates))

    def test_different_seeds_produce_different_fixed_candidates(self):
        """验证相同种子结果可复现，而不同种子得到不同的固定 37 人。"""
        first = self._create_sampling_trainer(seed=3)._initialize_fixed_candidate_clients()
        repeated = self._create_sampling_trainer(seed=3)._initialize_fixed_candidate_clients()
        different = self._create_sampling_trainer(seed=19)._initialize_fixed_candidate_clients()

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_mat_slots_map_exactly_to_fixed_candidates(self):
        """验证 MAT 的具体槽位和组内顺序被精确映射到固定候选客户端。"""
        trainer = self._create_sampling_trainer()
        candidates = list(range(100, 137))

        for global_epoch in (0, 5, 50):
            with self.subTest(global_epoch=global_epoch):
                topology = self.schedule.get_round(global_epoch)
                groups = trainer._map_mat_slots_to_clients(candidates, topology)
                # 直接按 MAT 槽位索引构造期望值，可识别退化为“仅按人数切片”的错误实现。
                expected_groups = {
                    int(group_idx): [candidates[slot] for slot in candidate_slots]
                    for group_idx, candidate_slots in topology.copy_groups().items()
                    if candidate_slots
                }
                expected_active = sorted(
                    candidates[slot] for slot in topology.active_candidate_slots
                )

                self.assertEqual(groups, expected_groups)
                self.assertEqual(
                    trainer._get_active_client_indexes(groups),
                    expected_active,
                )

    def test_only_active_clients_train_one_local_epoch(self):
        """验证本地训练只调用 MAT 当前行启用的客户端，未启用客户端不训练。"""
        trainer = self._create_sampling_trainer()
        trainer.client_registry = {
            client_idx: _FakeTrainingClient(client_idx)
            for client_idx in range(200)
        }
        active_clients = [3, 9, 21]

        global_epoch = trainer._train_active_clients_one_epoch(
            global_round_idx=2,
            group_round_idx=0,
            epoch_idx=0,
            active_client_indexes=active_clients,
        )

        self.assertEqual(global_epoch, 2)
        for client_idx, client in trainer.client_registry.items():
            expected_calls = [(2, 0, 0)] if client_idx in active_clients else []
            self.assertEqual(client.train_calls, expected_calls)

    def test_normal_and_empty_epochs_both_distribute_to_all_clients(self):
        """验证正常聚合和零参与 epoch 都把当前云模型下发给全部 200 人。"""
        all_clients = list(range(200))

        normal_trainer = self._prepare_train_loop_trainer(global_epoch=0)
        normal_cloud_state = {"weight": "aggregated"}
        normal_trainer._collect_hierarchical_cloud_inputs = Mock(
            return_value=([(37, {"weight": "local"})], {})
        )
        normal_trainer._aggregate = Mock(return_value=normal_cloud_state)
        normal_trainer.train()

        normal_trainer._sync_clients_to_global.assert_called_once_with(
            all_clients, normal_cloud_state
        )
        normal_trainer._evaluate_all_client_local_models.assert_called_once_with(0)
        self.assertEqual(normal_trainer.lifecycle_calls, ["sync", "evaluate"])

        empty_trainer = self._prepare_train_loop_trainer(global_epoch=50)
        empty_trainer._collect_hierarchical_cloud_inputs = Mock(return_value=([], {}))
        empty_trainer._aggregate = Mock()
        empty_trainer.train()

        empty_trainer._aggregate.assert_not_called()
        empty_trainer._sync_clients_to_global.assert_called_once_with(
            all_clients, {"weight": "previous"}
        )
        empty_trainer._evaluate_all_client_local_models.assert_called_once_with(50)
        self.assertEqual(empty_trainer.lifecycle_calls, ["sync", "evaluate"])

    def test_local_model_evaluation_uses_all_clients_and_global_ratio(self):
        """验证 200 个客户端分别本地评估，并按总正确数 7/总样本数 10 汇总。"""
        trainer = self._create_sampling_trainer()
        # 前三个客户端分别提供 1/2、2/3、4/5，其余客户端仍需被逐一调用。
        zero_metrics = {"test_total": 0, "test_correct": 0, "test_loss": 0.0}
        trainer.client_registry = {
            client_idx: _FakeEvaluationClient(zero_metrics, zero_metrics)
            for client_idx in range(200)
        }
        trainer.client_registry[0] = _FakeEvaluationClient(
            {"test_total": 2, "test_correct": 1, "test_loss": 0.5},
            {"test_total": 2, "test_correct": 1, "test_loss": 0.6},
        )
        trainer.client_registry[1] = _FakeEvaluationClient(
            {"test_total": 3, "test_correct": 2, "test_loss": 0.8},
            {"test_total": 3, "test_correct": 2, "test_loss": 0.9},
        )
        trainer.client_registry[2] = _FakeEvaluationClient(
            {"test_total": 5, "test_correct": 4, "test_loss": 1.2},
            {"test_total": 5, "test_correct": 4, "test_loss": 1.3},
        )
        trainer._append_metric_value = Mock()

        with patch("fedavg_test.mlops.log"):
            summary = trainer._evaluate_all_client_local_models(global_epoch=8)

        self.assertAlmostEqual(summary["test_acc"], 7 / 10)
        self.assertAlmostEqual(summary["train_acc"], 7 / 10)
        self.assertTrue(
            all(
                client.evaluation_calls == [False, True]
                for client in trainer.client_registry.values()
            )
        )
        trainer._append_metric_value.assert_any_call("test_acc.txt", 7 / 10)

    def test_client_evaluation_loads_its_persistent_local_state(self):
        """验证真实客户端评估前把自己的持久状态加载到模型和模型训练器。"""
        client = HFLClient.__new__(HFLClient)
        local_state = {"weight": "client-local"}
        client.get_local_model_state = Mock(return_value=local_state)
        client.model = Mock()
        client.model_trainer = Mock()
        client.local_test = Mock(return_value={"test_total": 2, "test_correct": 1})

        metrics = client.evaluate_local_model(use_test_dataset=True)

        self.assertEqual(metrics["test_correct"], 1)
        client.model.load_state_dict.assert_called_once_with(local_state)
        client.model_trainer.set_model_params.assert_called_once_with(local_state)
        client.local_test.assert_called_once_with(True)

    def test_client_probe_csv_has_stable_thirty_seven_columns(self):
        """验证客户端探针 CSV 始终为 37 列，且跨 epoch 的列身份与顺序稳定。"""
        trainer = self._create_sampling_trainer(seed=23)
        fixed_candidates = trainer._initialize_fixed_candidate_clients()
        trainer.client_registry = {
            client_idx: _FakeProbeClient(client_idx)
            for client_idx in fixed_candidates
        }

        trainer._build_round_groups(0)
        first_row = trainer._build_client_probe_row(None, fixed_candidates)
        trainer._build_round_groups(1)
        second_row = trainer._build_client_probe_row(None, fixed_candidates)

        # 经真实 CSV writer/reader 往返，避免 JSON 向量内部逗号被误判为额外列。
        csv_buffer = io.StringIO(newline="")
        writer = csv.writer(csv_buffer)
        writer.writerow(first_row)
        writer.writerow(second_row)
        csv_buffer.seek(0)
        parsed_rows = list(csv.reader(csv_buffer))

        self.assertEqual([len(row) for row in parsed_rows], [37, 37])
        self.assertEqual(parsed_rows[0], parsed_rows[1])
        for column_index, client_idx in enumerate(fixed_candidates):
            probability_vector = json.loads(parsed_rows[0][column_index])
            self.assertEqual(probability_vector[0], float(client_idx))

    def test_probe_metadata_csv_saves_structured_true_label(self):
        """验证每轮探针真实标签、样本索引和训练坐标写入结构化 CSV。"""
        trainer = self._create_sampling_trainer(seed=31)
        trainer.args.probe_source = "test"

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.args.result_dir = temp_dir
            probe_files, probe_writers = trainer._open_probe_outputs()
            try:
                # 模拟一个训练 epoch 完成后的标签元数据写入。
                probe_writers["meta"].writerow(
                    trainer._build_probe_metadata_row(
                        global_epoch=5,
                        global_round_idx=1,
                        group_round_idx=2,
                        local_epoch_idx=3,
                        probe_index=5,
                        probe_label=7,
                    )
                )
                trainer._flush_probe_outputs(probe_files)
            finally:
                trainer._close_probe_outputs(probe_files)

            metadata_path = os.path.join(temp_dir, "probe_meta.csv")
            with open(metadata_path, "r", encoding="utf-8", newline="") as file_obj:
                rows = list(csv.DictReader(file_obj))

            self.assertEqual(
                rows,
                [
                    {
                        "global_epoch": "5",
                        "global_round_idx": "1",
                        "group_round_idx": "2",
                        "local_epoch_idx": "3",
                        "probe_source": "test",
                        "probe_index": "5",
                        "true_label": "7",
                    }
                ],
            )
            self.assertTrue(
                all(
                    os.path.isfile(os.path.join(temp_dir, filename))
                    for filename in (
                        "probe_client_pre.csv",
                        "probe_edge_post.csv",
                        "probe_cloud_post.csv",
                        "probe_meta.csv",
                    )
                )
            )

    def test_true_labels_align_by_global_epoch_instead_of_row_order(self):
        """验证分析端按 global_epoch 对齐结构化标签，而不是依赖文件行顺序。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = os.path.join(temp_dir, "probe_meta.csv")
            with open(metadata_path, "w", encoding="utf-8", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=["global_epoch", "true_label"])
                writer.writeheader()
                # 故意逆序写入，证明显式轮次坐标能够稳定还原标签序列。
                writer.writerow({"global_epoch": 1, "true_label": 2})
                writer.writerow({"global_epoch": 0, "true_label": 7})

            labels = read_true_labels(Path(metadata_path), round_count=2)

        self.assertEqual(labels, [7, 2])

    def test_runtime_schedule_records_fixed_candidates_and_all_distribution(self):
        """验证运行时 JSONL 记录固定候选、MAT 活跃客户端和全量下发范围。"""
        trainer = self._create_sampling_trainer(seed=29, distribution_scope="all")
        candidates = trainer._initialize_fixed_candidate_clients()
        global_epoch = 5
        groups = trainer._build_round_groups(global_epoch)
        active_clients = trainer._get_active_client_indexes(groups)
        distributed_clients = trainer._get_distribution_client_indexes(active_clients)

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.topology_schedule_output_path = os.path.join(
                temp_dir, "topology_schedule.jsonl"
            )
            trainer._append_runtime_topology_record(
                global_round_idx=5,
                group_round_idx=0,
                epoch_idx=0,
                global_epoch=global_epoch,
                candidate_client_indexes=candidates,
                group_to_client_indexes=groups,
                active_client_indexes=active_clients,
                distributed_client_indexes=distributed_clients,
                aggregated=True,
            )
            with open(
                    trainer.topology_schedule_output_path,
                    "r",
                    encoding="utf-8",
            ) as file_obj:
                record = json.loads(file_obj.readline())

        self.assertEqual(record["candidate_client_indexes"], candidates)
        self.assertEqual(record["active_client_indexes"], active_clients)
        self.assertEqual(record["distributed_client_indexes"], list(range(200)))
        self.assertEqual(record["mat_topology_index"], global_epoch)
        self.assertTrue(record["aggregated"])


class FixedProbeBatchTest(unittest.TestCase):
    """验证固定100图探针、逐图共识计算和结构化检查点。"""

    def test_balanced_probe_is_unique_reproducible_and_rng_isolated(self):
        """验证每类10张、索引唯一、哈希可复现且不改变全局NumPy状态。"""
        labels = torch.arange(10, dtype=torch.long).repeat_interleave(12)
        # 图片内容编码全局样本序号，便于哈希同时覆盖索引、标签和真实内容。
        inputs = torch.arange(120 * 4, dtype=torch.float32).reshape(120, 1, 2, 2)
        probe_data = [
            (inputs[:40], labels[:40]),
            (inputs[40:85], labels[40:85]),
            (inputs[85:], labels[85:]),
        ]

        np.random.seed(20260717)
        state_before = np.random.get_state()
        first = select_fixed_balanced_probe(
            probe_data, samples_per_class=10, seed=7, source="test"
        )
        state_after = np.random.get_state()
        repeated = select_fixed_balanced_probe(
            probe_data, samples_per_class=10, seed=7, source="test"
        )
        different = select_fixed_balanced_probe(
            probe_data, samples_per_class=10, seed=19, source="test"
        )

        self.assertEqual(first.sample_count, 100)
        self.assertEqual(len(set(first.indices.tolist())), 100)
        self.assertEqual(
            np.bincount(first.true_labels, minlength=10).tolist(), [10] * 10
        )
        self.assertTrue(np.array_equal(first.indices, repeated.indices))
        self.assertEqual(first.content_hash, repeated.content_hash)
        self.assertNotEqual(first.content_hash, different.content_hash)
        # RandomState由算法内部独立创建，全局MT19937的完整状态必须逐项不变。
        self.assertEqual(state_before[0], state_after[0])
        self.assertTrue(np.array_equal(state_before[1], state_after[1]))
        self.assertEqual(state_before[2:], state_after[2:])

    def test_consensus_boundaries_match_the_metric_definition(self):
        """验证均匀、确定但不同、确定且相同三种人工概率的A/C/S边界。"""
        uniform = np.full((3, 2, 3), 1.0 / 3.0, dtype=np.float64)
        deterministic_different = np.eye(3, dtype=np.float64)[:, None, :]
        deterministic_same = np.repeat(
            np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float64), 3, axis=0
        )

        uniform_metrics = calculate_population_probe_metrics(
            uniform, np.asarray([0, 1], dtype=np.int64)
        )
        different_metrics = calculate_population_probe_metrics(
            deterministic_different, np.asarray([0], dtype=np.int64)
        )
        same_metrics = calculate_population_probe_metrics(
            deterministic_same, np.asarray([0], dtype=np.int64)
        )

        self.assertAlmostEqual(uniform_metrics["agreement_mean"], 1.0, places=12)
        self.assertAlmostEqual(uniform_metrics["certainty_mean"], 0.0, places=12)
        self.assertAlmostEqual(uniform_metrics["effective_mean"], 0.0, places=12)
        self.assertAlmostEqual(different_metrics["agreement_mean"], 0.0, places=12)
        self.assertAlmostEqual(different_metrics["certainty_mean"], 1.0, places=12)
        self.assertAlmostEqual(different_metrics["effective_mean"], 0.0, places=12)
        self.assertAlmostEqual(same_metrics["agreement_mean"], 1.0, places=12)
        self.assertAlmostEqual(same_metrics["certainty_mean"], 1.0, places=12)
        self.assertAlmostEqual(same_metrics["effective_mean"], 1.0, places=12)

    def test_correct_and_wrong_consensus_partition_pure_consensus(self):
        """验证每张图片及其均值都满足正确共识加错误共识等于纯共识。"""
        probabilities = np.asarray(
            [
                [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]],
                [[0.8, 0.2], [0.7, 0.3], [0.2, 0.8]],
                [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7]],
            ],
            dtype=np.float64,
        )
        # 中间图片多数预测为0但真实标签为1，用于同时覆盖正确和错误分支。
        metrics = calculate_population_probe_metrics(
            probabilities, np.asarray([0, 1, 1], dtype=np.int64)
        )

        self.assertTrue(
            np.allclose(
                metrics["correct_effective_by_sample"]
                + metrics["wrong_effective_by_sample"],
                metrics["effective_by_sample"],
                atol=1e-12,
            )
        )
        self.assertAlmostEqual(
            metrics["correct_effective_mean"]
            + metrics["wrong_effective_mean"],
            metrics["effective_mean"],
            places=12,
        )

    def test_zero_or_single_active_client_returns_empty_consensus(self):
        """验证活跃客户端不足2人时所有活跃共识指标均为空值。"""
        labels = np.asarray([0, 1], dtype=np.int64)
        for model_count in (0, 1):
            with self.subTest(model_count=model_count):
                probabilities = np.full(
                    (model_count, 2, 2), 0.5, dtype=np.float64
                )
                metrics = calculate_population_probe_metrics(probabilities, labels)
                self.assertTrue(np.isnan(metrics["agreement_mean"]))
                self.assertTrue(np.isnan(metrics["certainty_mean"]))
                self.assertTrue(np.isnan(metrics["effective_mean"]))
                self.assertTrue(np.isnan(metrics["correct_effective_mean"]))
                self.assertTrue(np.all(np.isnan(metrics["effective_by_sample"])))

    def test_recorder_saves_valid_prefix_masks_and_nan_edge_slots(self):
        """验证第10轮和关闭时保存合法NPZ前缀、活跃掩码及NaN边缘槽位。"""
        probe_set = FixedProbeSet(
            inputs=torch.zeros(4, 1, 2, 2),
            indices=np.asarray([3, 5, 7, 9], dtype=np.int64),
            true_labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
            content_hash="unit-test-probe-hash",
            source="test",
        )
        client_probabilities = np.asarray(
            [
                [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
                [[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]],
                [[0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]],
            ],
            dtype=np.float32,
        )
        edge_probabilities = np.asarray(
            [
                [[0.85, 0.15], [0.15, 0.85], [0.75, 0.25], [0.25, 0.75]],
                [[np.nan, np.nan]] * 4,
            ],
            dtype=np.float32,
        )
        cloud_probabilities = np.asarray(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ProbeBatchRecorder(
                result_dir=temp_dir,
                total_epochs=12,
                candidate_client_ids=[10, 11, 12],
                edge_slot_count=2,
                probe_set=probe_set,
                class_count=2,
                checkpoint_interval=10,
            )
            for global_epoch in range(9):
                recorder.record_epoch(
                    global_epoch,
                    client_probabilities,
                    edge_probabilities,
                    cloud_probabilities,
                    active_client_ids=[10, 12],
                )
            npz_path = os.path.join(temp_dir, "probe_probabilities.npz")
            self.assertFalse(os.path.exists(npz_path))

            recorder.record_epoch(
                9,
                client_probabilities,
                edge_probabilities,
                cloud_probabilities,
                active_client_ids=[10, 12],
            )
            with np.load(npz_path, allow_pickle=False) as checkpoint:
                self.assertEqual(int(checkpoint["completed_epochs"]), 10)
                self.assertEqual(checkpoint["client_probabilities"].shape, (10, 3, 4, 2))
                self.assertEqual(checkpoint["edge_probabilities"].shape, (10, 2, 4, 2))
                self.assertEqual(checkpoint["cloud_probabilities"].shape, (10, 4, 2))
                self.assertEqual(checkpoint["active_client_mask"].shape, (10, 3))
                self.assertTrue(np.all(checkpoint["active_client_mask"][:, [0, 2]]))
                self.assertTrue(np.all(~checkpoint["active_client_mask"][:, 1]))
                self.assertTrue(np.all(np.isnan(checkpoint["edge_probabilities"][:, 1])))
                self.assertTrue(np.all(~checkpoint["edge_active_mask"][:, 1]))

            # 第11轮先保留在内存，close必须覆盖为11轮的合法部分前缀。
            recorder.record_epoch(
                10,
                client_probabilities,
                edge_probabilities,
                cloud_probabilities,
                active_client_ids=[],
            )
            with np.load(npz_path, allow_pickle=False) as checkpoint:
                self.assertEqual(int(checkpoint["completed_epochs"]), 10)
            recorder.close()

            with np.load(npz_path, allow_pickle=False) as completed:
                self.assertEqual(int(completed["completed_epochs"]), 11)
                self.assertEqual(completed["global_epochs"].tolist(), list(range(11)))
                self.assertEqual(completed["probe_set_hash"].item(), "unit-test-probe-hash")
                self.assertTrue(np.all(~completed["active_client_mask"][10]))
                self.assertTrue(np.all(np.isfinite(completed["client_probabilities"])))
                self.assertTrue(np.allclose(
                    np.sum(completed["client_probabilities"], axis=-1), 1.0
                ))
            self.assertFalse(os.path.exists(npz_path + ".tmp"))

            with open(
                    os.path.join(temp_dir, "probe_epoch_summary.csv"),
                    "r",
                    encoding="utf-8",
                    newline="",
            ) as file_obj:
                rows = list(csv.DictReader(file_obj))
            self.assertEqual(len(rows), 11)
            self.assertEqual(rows[-1]["active_count"], "0")
            self.assertEqual(rows[-1]["active_effective_mean"], "")
            self.assertEqual(
                rows[-1]["coverage_weighted_active_correct_effective"], ""
            )

    def test_client_batches_one_hundred_probes_in_one_forward(self):
        """验证100张探针配置只加载一次模型并只执行一次前向传播。"""
        client = HFLClient.__new__(HFLClient)
        client.local_model_state = {"weight": "client-local"}
        client.device = torch.device("cpu")
        logits = torch.tensor([[2.0, 1.0]], dtype=torch.float32).repeat(100, 1)
        client.model = Mock(return_value=logits)
        probe_inputs = torch.zeros(100, 1, 28, 28, dtype=torch.float32)

        probabilities = client.predict_proba_batch(
            probe_inputs, inference_batch_size=100
        )

        client.model.load_state_dict.assert_called_once_with(client.local_model_state)
        client.model.to.assert_called_once_with(client.device)
        client.model.eval.assert_called_once_with()
        self.assertEqual(client.model.call_count, 1)
        self.assertEqual(len(probabilities), 100)
        self.assertTrue(
            np.allclose(np.sum(np.asarray(probabilities), axis=1), 1.0, atol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()
