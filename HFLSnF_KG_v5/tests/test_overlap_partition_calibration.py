"""验证目标实体重叠划分、配置入口和无训练校准合同。"""

from __future__ import annotations

import unittest

from HFLSnF_KG_v5.tasks.kge import (
    BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
    build_knowledge_graph_dataset,
    calibrate_entity_overlap_levels,
    partition_train_triples_by_head,
    partition_train_triples_by_overlap_target,
)


def _build_overlap_test_dataset():
    """构造具有重复尾实体且每个头实体只含一行的测试数据。"""

    train = [
        (
            "head_{}".format(index),
            "relation_{}".format(index % 2),
            "tail_{}".format(index % 3),
        )
        for index in range(18)
    ]
    return build_knowledge_graph_dataset(
        dataset_name="overlap-contract",
        train_triples=train,
        valid_triples=[("valid_head", "relation_0", "tail_0")],
        test_triples=[("test_head", "relation_1", "tail_1")],
    )


class OverlapPartitionCalibrationTest(unittest.TestCase):
    """检查重叠率公式、搜索确定性和实验配置接入。"""

    def setUp(self) -> None:
        """为每个测试创建相互隔离的微型知识图谱。"""

        self.dataset = _build_overlap_test_dataset()

    def test_normalized_overlap_matches_replication_formula(self) -> None:
        """确认摘要中的主指标严格满足复制因子归一化公式。"""

        partition = partition_train_triples_by_head(
            self.dataset, client_count=3, seed=42
        )
        summary = partition.summary()
        expected = (
            (float(summary["entity_replication_factor"]) - 1.0)
            / 2.0
        )
        self.assertAlmostEqual(
            float(summary["entity_normalized_overlap"]),
            expected,
            places=12,
        )
        self.assertIn("frequency_weighted_entity_normalized_overlap", summary)
        self.assertIn("max_relative_load_deviation", summary)

    def test_target_partition_is_deterministic_and_complete(self) -> None:
        """确认同一搜索种子可复现且满足三元组与头实体合同。"""

        endpoint = partition_train_triples_by_overlap_target(
            self.dataset,
            client_count=3,
            seed=42,
            target_entity_overlap=1.0,
            overlap_tolerance=1.0,
            load_tolerance=0.20,
            relation_overlap_tolerance=1.0,
            search_restarts=4,
            strict=False,
        )
        target = float(endpoint.summary()["entity_normalized_overlap"])
        arguments = {
            "dataset": self.dataset,
            "client_count": 3,
            "seed": 42,
            "target_entity_overlap": target,
            "overlap_tolerance": 0.001,
            "load_tolerance": 0.20,
            "relation_overlap_tolerance": 1.0,
            "search_restarts": 4,
            "search_seed": 9001,
        }
        first = partition_train_triples_by_overlap_target(**arguments)
        second = partition_train_triples_by_overlap_target(**arguments)
        self.assertEqual(first.partition_hash, second.partition_hash)
        self.assertEqual(
            first.partition_strategy,
            BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
        )
        combined_count = sum(
            partition.triple_count for partition in first.partitions
        )
        self.assertEqual(
            combined_count, int(self.dataset.train_triples.shape[0])
        )
        all_heads = [
            int(head_id)
            for partition in first.partitions
            for head_id in partition.head_entity_ids.tolist()
        ]
        self.assertEqual(len(all_heads), len(set(all_heads)))
        self.assertLessEqual(
            float(first.summary()["entity_overlap_absolute_error"]),
            0.001,
        )

    def test_calibration_returns_three_levels_without_training(self) -> None:
        """确认校准器返回基线、共同区间和低中高三个档位。"""

        report = calibrate_entity_overlap_levels(
            dataset=self.dataset,
            client_count=3,
            seeds=(42, 2024, 2025),
            overlap_tolerance=0.05,
            load_tolerance=0.20,
            relation_overlap_tolerance=1.0,
            search_restarts=2,
            minimum_overlap_span=0.0,
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            set(report["levels"]), {"low", "medium", "high"}
        )
        self.assertEqual(set(report["baselines"]), {"42", "2024", "2025"})


if __name__ == "__main__":
    unittest.main()
