import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from analyze_consensus import read_true_labels
from client_test import HFLClient
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


if __name__ == "__main__":
    unittest.main()
