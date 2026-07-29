"""SevereTest 数据划分、聚合和共识指标单元测试。"""

from __future__ import absolute_import

from collections import OrderedDict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from probe_metrics import calculate_population_probe_metrics  # noqa: E402
from SevereTest.data_partition import (  # noqa: E402
    build_label_modulo_partitions,
    validate_label_modulo_partitions,
)
from SevereTest.run_experiment import validate_requested_device  # noqa: E402
from SevereTest.trainer import (  # noqa: E402
    aggregate_hierarchical_model_states,
    aggregate_weighted_model_states,
    validate_hierarchical_groups,
    validate_training_client_ids,
)


class LabelModuloPartitionTest(unittest.TestCase):
    """验证 200 客户端单标签确定性划分。"""

    def setUp(self):
        """构造每类 23 个样本的轻量测试标签。"""
        self.labels = np.repeat(np.arange(10, dtype=np.int64), 23)

    def test_all_samples_are_assigned_once_and_clients_are_nonempty(self):
        """确认样本无遗漏、无重复且 200 个客户端均非空。"""
        partitions = build_label_modulo_partitions(self.labels, seed=5)
        self.assertEqual(len(partitions), 200)
        self.assertTrue(all(len(indexes) > 0 for indexes in partitions))
        assigned = np.concatenate(partitions)
        np.testing.assert_array_equal(
            np.sort(assigned), np.arange(self.labels.size)
        )
        self.assertEqual(np.unique(assigned).size, self.labels.size)

    def test_client_label_matches_client_id_modulo_ten(self):
        """确认客户端编号模 10 严格等于其唯一标签。"""
        partitions = build_label_modulo_partitions(self.labels, seed=5)
        for client_id, indexes in enumerate(partitions):
            self.assertEqual(
                np.unique(self.labels[indexes]).tolist(), [client_id % 10]
            )
        self.assertTrue(
            validate_label_modulo_partitions(self.labels, partitions)
        )

    def test_partition_is_deterministic_and_balanced_per_class(self):
        """确认同一划分种子可复现且同类客户端样本数最多相差 1。"""
        first = build_label_modulo_partitions(self.labels, seed=5)
        second = build_label_modulo_partitions(self.labels, seed=5)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        for label in range(10):
            sizes = [
                len(first[label + 10 * group_index])
                for group_index in range(20)
            ]
            self.assertLessEqual(max(sizes) - min(sizes), 1)


class FixedTrainingAndAggregationTest(unittest.TestCase):
    """验证固定训练名单和样本加权模型聚合。"""

    def test_training_ids_must_be_unique_and_in_range(self):
        """确认合法名单保持不变，重复或越界名单被拒绝。"""
        self.assertEqual(
            validate_training_client_ids(list(range(10)), 200),
            list(range(10)),
        )
        with self.assertRaises(ValueError):
            validate_training_client_ids([0, 0], 200)
        with self.assertRaises(ValueError):
            validate_training_client_ids([0, 200], 200)

    def test_weighted_aggregation_uses_client_sample_counts(self):
        """确认两个模型按 1:3 样本比例得到正确参数。"""
        first_state = OrderedDict([
            ("weight", torch.tensor([0.0, 4.0], dtype=torch.float32)),
            ("counter", torch.tensor(2, dtype=torch.int64)),
        ])
        second_state = OrderedDict([
            ("weight", torch.tensor([4.0, 0.0], dtype=torch.float32)),
            ("counter", torch.tensor(9, dtype=torch.int64)),
        ])
        aggregated = aggregate_weighted_model_states([
            (1, first_state),
            (3, second_state),
        ])
        torch.testing.assert_close(
            aggregated["weight"], torch.tensor([3.0, 1.0])
        )
        self.assertEqual(int(aggregated["counter"]), 2)

    def test_hierarchical_groups_cover_first_thirty_clients_once(self):
        """确认三个边缘组按顺序且无重复地覆盖客户端 0–29。"""
        groups = [
            list(range(0, 10)),
            list(range(10, 20)),
            list(range(20, 30)),
        ]
        self.assertEqual(
            validate_hierarchical_groups(groups, list(range(30)), 200),
            groups,
        )
        with self.assertRaises(ValueError):
            validate_hierarchical_groups(
                [list(range(10)), list(range(9, 20))],
                list(range(20)),
                200,
            )

    def test_hierarchical_aggregation_matches_all_sample_weighting(self):
        """确认边缘再云端聚合等价于对全部客户端按样本数加权。"""
        grouped_states = [
            [
                (1, OrderedDict([("weight", torch.tensor([0.0]))])),
                (3, OrderedDict([("weight", torch.tensor([4.0]))])),
            ],
            [
                (2, OrderedDict([("weight", torch.tensor([2.0]))])),
                (4, OrderedDict([("weight", torch.tensor([6.0]))])),
            ],
            [
                (5, OrderedDict([("weight", torch.tensor([10.0]))])),
            ],
        ]
        edge_states, cloud_state, edge_counts = (
            aggregate_hierarchical_model_states(grouped_states)
        )
        self.assertEqual(edge_counts, [4, 6, 5])
        self.assertEqual(len(edge_states), 3)
        flat_state = aggregate_weighted_model_states([
            item for group in grouped_states for item in group
        ])
        torch.testing.assert_close(
            cloud_state["weight"], flat_state["weight"]
        )


class GpuDeviceValidationTest(unittest.TestCase):
    """验证 HFL 的强制 GPU 检查不会静默退回 CPU。"""

    def test_cpu_mode_does_not_require_cuda(self):
        """确认显式 CPU 配置不触发 CUDA 可用性检查。"""
        validate_requested_device(
            SimpleNamespace(using_gpu=False, gpu_id=0),
            torch.device("cpu"),
        )

    def test_gpu_mode_rejects_missing_cuda(self):
        """确认请求 GPU 但 CUDA 不可用时给出明确错误。"""
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                validate_requested_device(
                    SimpleNamespace(using_gpu=True, gpu_id=0),
                    torch.device("cpu"),
                )

    def test_gpu_mode_accepts_selected_cuda_device(self):
        """确认 CUDA 可用且设备编号合法时允许继续训练。"""
        with mock.patch(
                "torch.cuda.is_available", return_value=True
        ), mock.patch(
                "torch.cuda.device_count", return_value=1
        ), mock.patch(
                "torch.cuda.get_device_name", return_value="测试显卡"
        ):
            validate_requested_device(
                SimpleNamespace(using_gpu=True, gpu_id=0),
                torch.device("cuda:0"),
            )


class EffectiveConsensusMetricTest(unittest.TestCase):
    """验证有效共识 S 的关键边界行为。"""

    def test_uniform_probabilities_have_zero_effective_consensus(self):
        """确认共同均匀输出虽然一致，但确定性和 S 均接近零。"""
        probabilities = np.full((10, 100, 10), 0.1, dtype=np.float64)
        labels = np.tile(np.arange(10), 10)
        metrics = calculate_population_probe_metrics(probabilities, labels)
        self.assertAlmostEqual(metrics["agreement_mean"], 1.0, places=12)
        self.assertAlmostEqual(metrics["certainty_mean"], 0.0, places=12)
        self.assertAlmostEqual(metrics["effective_mean"], 0.0, places=12)

    def test_identical_confident_predictions_have_full_consensus(self):
        """确认完全相同的置信 one-hot 输出得到满分 S。"""
        probabilities = np.zeros((10, 100, 10), dtype=np.float64)
        probabilities[:, :, 3] = 1.0
        labels = np.full(100, 3, dtype=np.int64)
        metrics = calculate_population_probe_metrics(probabilities, labels)
        self.assertAlmostEqual(metrics["agreement_mean"], 1.0, places=12)
        self.assertAlmostEqual(metrics["certainty_mean"], 1.0, places=12)
        self.assertAlmostEqual(metrics["effective_mean"], 1.0, places=12)
        self.assertAlmostEqual(
            metrics["correct_effective_mean"], 1.0, places=12
        )

    def test_conflicting_confident_predictions_reduce_consensus(self):
        """确认客户端分成两个冲突类别时 S 明显低于满分。"""
        probabilities = np.zeros((10, 100, 10), dtype=np.float64)
        probabilities[:5, :, 2] = 1.0
        probabilities[5:, :, 7] = 1.0
        labels = np.full(100, 2, dtype=np.int64)
        metrics = calculate_population_probe_metrics(probabilities, labels)
        self.assertLess(metrics["agreement_mean"], 1.0)
        self.assertLess(metrics["effective_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
