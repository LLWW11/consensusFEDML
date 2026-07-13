import json
import os
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from topology_schedule import MatlabTopologySchedule
from trainer_test import HierarchicalTrainer


class _FakeTrainingClient:
    """记录测试期间客户端是否完成了本地训练。"""

    def __init__(self, client_idx):
        """保存客户端编号并初始化训练调用记录。"""
        self.client_idx = client_idx
        self.calls = []

    def train_one_epoch(self, global_round_idx, group_round_idx, epoch_idx):
        """记录一次本地训练并返回与真实客户端一致的二元结果。"""
        self.calls.append((global_round_idx, group_round_idx, epoch_idx))
        return global_round_idx, {"client_idx": self.client_idx}


class _FakeModel:
    """记录云模型参数是否被显式加载。"""

    def __init__(self):
        """初始化最近一次加载的模型参数。"""
        self.loaded_state = None

    def load_state_dict(self, model_state):
        """保存测试传入的云模型参数。"""
        self.loaded_state = model_state

    def state_dict(self):
        """返回当前保存的模型参数。"""
        return self.loaded_state


class HierarchicalSamplingTest(unittest.TestCase):
    """验证 200 客户端训练后的两级采样、下发和运行时记录。"""

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

    def _create_sampling_trainer(self, seed=7, distribution_scope="active"):
        """构造只包含采样和下发测试所需属性的轻量训练器。"""
        trainer = HierarchicalTrainer.__new__(HierarchicalTrainer)
        trainer.args = SimpleNamespace(
            client_num_in_total=200,
            client_num_per_round=37,
            random_seed=seed,
        )
        trainer.model_distribution_scope = distribution_scope
        trainer.topology_schedule = self.schedule
        return trainer

    def test_first_sampling_is_unique_reproducible_and_epoch_specific(self):
        """验证首次采样是 200 人中的 37 个唯一客户端且按 epoch 可复现。"""
        trainer = self._create_sampling_trainer(seed=11)
        first = trainer._sample_epoch_candidates(3)
        repeated = trainer._sample_epoch_candidates(3)
        next_epoch = trainer._sample_epoch_candidates(4)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(len(first), 37)
        self.assertEqual(len(set(first)), 37)
        self.assertTrue(all(0 <= client_idx < 200 for client_idx in first))

    def test_second_sampling_matches_mat_group_counts(self):
        """验证完整、部分和零参与 epoch 都严格服从 MAT 各组人数。"""
        trainer = self._create_sampling_trainer(seed=13)
        for global_epoch in (0, 5, 50):
            with self.subTest(global_epoch=global_epoch):
                topology = self.schedule.get_round(global_epoch)
                candidates = trainer._sample_epoch_candidates(global_epoch)
                groups = trainer._sample_groups_from_candidates(
                    global_epoch, candidates, topology
                )
                active = trainer._get_active_client_indexes(groups)

                self.assertEqual(len(active), topology.participant_count)
                self.assertTrue(set(active).issubset(set(candidates)))
                self.assertEqual(len(active), len(set(active)))
                for group_idx, expected_count in topology.group_client_counts.items():
                    self.assertEqual(len(groups.get(group_idx, [])), expected_count)

    def test_all_two_hundred_clients_train_before_sampling(self):
        """验证单个本地 epoch 会调用全部 200 个客户端的训练方法。"""
        trainer = self._create_sampling_trainer()
        trainer.client_registry = {
            client_idx: _FakeTrainingClient(client_idx)
            for client_idx in range(200)
        }
        global_epoch = trainer._train_one_epoch_all_clients(2, 0, 0)

        self.assertEqual(global_epoch, 2)
        self.assertTrue(
            all(client.calls == [(2, 0, 0)] for client in trainer.client_registry.values())
        )

    def test_distribution_scope_supports_active_and_all(self):
        """验证 active、all 以及零参与三种下发结果。"""
        active_clients = [3, 9, 21]
        active_trainer = self._create_sampling_trainer(distribution_scope="active")
        all_trainer = self._create_sampling_trainer(distribution_scope="all")

        self.assertEqual(
            active_trainer._get_distribution_client_indexes(active_clients),
            active_clients,
        )
        self.assertEqual(
            all_trainer._get_distribution_client_indexes(active_clients),
            list(range(200)),
        )
        self.assertEqual(all_trainer._get_distribution_client_indexes([]), [])

    def test_global_evaluation_explicitly_loads_cloud_model(self):
        """验证每轮准确率测试显式使用最终云模型。"""
        trainer = self._create_sampling_trainer()
        trainer.model = _FakeModel()
        trainer.model_trainer = Mock()
        trainer._local_test_on_all_clients = Mock()
        cloud_state = {"weight": "cloud"}

        trainer._evaluate_global_model_on_all_clients(8, cloud_state)

        self.assertIs(trainer.model.loaded_state, cloud_state)
        trainer.model_trainer.set_model_params.assert_called_once_with(cloud_state)
        trainer._local_test_on_all_clients.assert_called_once_with(8)

    def test_runtime_schedule_records_real_client_sets(self):
        """验证运行时 JSONL 能追踪候选、二次分组、参与者和下发范围。"""
        trainer = self._create_sampling_trainer(distribution_scope="active")
        global_epoch = 5
        trainer.current_round_topology = self.schedule.get_round(global_epoch)
        candidates = trainer._sample_epoch_candidates(global_epoch)
        groups = trainer._sample_groups_from_candidates(
            global_epoch, candidates, trainer.current_round_topology
        )
        active = trainer._get_active_client_indexes(groups)

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
                active_client_indexes=active,
                distributed_client_indexes=active,
                aggregated=True,
            )
            with open(trainer.topology_schedule_output_path, "r", encoding="utf-8") as file_obj:
                record = json.loads(file_obj.readline())

        self.assertEqual(record["candidate_client_indexes"], candidates)
        self.assertEqual(record["active_client_indexes"], active)
        self.assertEqual(record["distributed_client_indexes"], active)
        self.assertEqual(record["mat_topology_index"], global_epoch)
        self.assertTrue(record["aggregated"])

    def test_zero_participation_epoch_skips_aggregation_and_distribution(self):
        """验证 MAT 人数为零时主循环沿用云模型且不聚合、不下发。"""
        trainer = self._create_sampling_trainer(distribution_scope="all")
        trainer.args.comm_round = 1
        trainer.args.group_comm_round = 1
        trainer.args.epochs = 1
        trainer.args.group_method = "matlab"
        previous_cloud_state = {"weight": "previous"}
        trainer.model = _FakeModel()
        trainer.model.loaded_state = previous_cloud_state
        trainer._setup_result_dir = Mock()
        trainer._initialize_metric_output_files = Mock()
        trainer._is_consensus_probe_enabled = Mock(return_value=False)
        # 正式 MAT 的 global_epoch=50 是零参与 epoch。
        trainer._train_one_epoch_all_clients = Mock(return_value=50)
        trainer._collect_hierarchical_cloud_inputs = Mock(return_value=([], {}))
        trainer._aggregate = Mock()
        trainer._sync_clients_to_global = Mock()
        trainer._evaluate_global_model_on_all_clients = Mock()
        trainer._append_runtime_topology_record = Mock()

        trainer.train()

        trainer._aggregate.assert_not_called()
        trainer._sync_clients_to_global.assert_not_called()
        trainer._evaluate_global_model_on_all_clients.assert_called_once_with(
            50, previous_cloud_state
        )
        runtime_call = trainer._append_runtime_topology_record.call_args
        runtime_keywords = runtime_call[1]
        self.assertFalse(runtime_keywords["aggregated"])
        self.assertEqual(runtime_keywords["distributed_client_indexes"], [])


if __name__ == "__main__":
    unittest.main()
