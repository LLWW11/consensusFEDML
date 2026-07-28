"""MAT动态采样、动态分组和FedML TransE训练链路测试。"""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from HFLSnF_KG_v2.core.randomness import seed_everything
from HFLSnF_KG_v2.core.topology import (
    MatlabTopologyProvider,
    SequenceTopologyProvider,
)
from HFLSnF_KG_v2.fedml_kge import (
    FedMLDynamicTopologyTransERunner,
)
from HFLSnF_KG_v2.tasks.kge import (
    TransE,
    build_knowledge_graph_dataset,
    partition_train_triples_by_head,
)


def build_dynamic_args() -> SimpleNamespace:
    """构造两轮CPU动态联邦TransE单元测试参数。"""

    return SimpleNamespace(
        training_type="simulation",
        backend="sp",
        federated_optimizer="DynamicTopologyTransE",
        comparison_scenario="HFLSnF-Dynamic-Test",
        client_num_in_total=37,
        # 动态模式下该值只表示37个候选客户端的容量。
        client_num_per_round=37,
        topology_architecture="hfl",
        topology_snf=True,
        topology_edge_mode="dynamic",
        comm_round=2,
        epochs=1,
        batch_size=2,
        client_optimizer="adam",
        learning_rate=0.01,
        lr=0.01,
        margin=1.0,
        negative_sample_count=1,
        eval_every=2,
        validation_max_triples=0,
        final_validation_max_triples=0,
        test_max_triples=0,
        evaluation_candidate_batch_size=128,
        centralized_reference_mrr=0.2,
        random_seed=42,
    )


def build_37_client_dataset():
    """构造可被严格划分为37个非空知识客户端的微型图谱。"""

    train = [
        (
            "head_{:02d}".format(client_id),
            "linked_to",
            "tail_{:02d}".format(client_id),
        )
        for client_id in range(37)
    ]
    valid = [("head_00", "linked_to", "tail_01")]
    test = [("head_01", "linked_to", "tail_02")]
    return build_knowledge_graph_dataset(
        "synthetic-dynamic-37-client-kg",
        train,
        valid,
        test,
    )


class MatlabDynamicTopologyTest(unittest.TestCase):
    """验证项目MAT文件确实包含逐轮变化的HFLSnF动态拓扑。"""

    @staticmethod
    def _mat_path() -> Path:
        """返回仓库内正式200轮MAT文件的绝对路径。"""

        return (
            Path(__file__).resolve().parents[1]
            / "matlab"
            / (
                "result-U-6fixedge_epoch200_"
                "varAlpha_0p5_trainable.mat"
            )
        )

    def test_real_mat_dynamic_hfl_schedule_is_valid(self) -> None:
        """验证200轮人数、动态组和0至36客户端映射均有效。"""

        provider = MatlabTopologyProvider(
            mat_path=self._mat_path(),
            architecture="hfl",
            snf_enabled=True,
            edge_mode="dynamic",
            util=0.5,
            client_count=37,
        )
        rounds = [
            provider.get_round(round_index)
            for round_index in range(provider.round_count)
        ]
        participant_counts = [
            item.participant_count for item in rounds
        ]
        group_counts = [
            len(item.group_to_client_indexes) for item in rounds
        ]
        participant_sets = {
            item.active_client_indexes for item in rounds
        }
        topology_signatures = {
            tuple(
                (
                    group_id,
                    tuple(client_ids),
                    item.edge_node_ids.get(group_id),
                )
                for group_id, client_ids in (
                    item.group_to_client_indexes.items()
                )
            )
            for item in rounds
        }

        self.assertEqual(provider.round_count, 200)
        self.assertEqual(min(participant_counts), 11)
        self.assertEqual(max(participant_counts), 37)
        self.assertEqual(min(group_counts), 2)
        self.assertEqual(max(group_counts), 12)
        self.assertGreater(len(participant_sets), 1)
        self.assertGreater(len(topology_signatures), 1)
        for topology in rounds:
            self.assertEqual(
                len(topology.active_client_indexes),
                len(set(topology.active_client_indexes)),
            )
            self.assertTrue(
                all(
                    0 <= client_id < 37
                    for client_id in (
                        topology.active_client_indexes
                    )
                )
            )


class DynamicTopologyFedMLRunnerTest(unittest.TestCase):
    """验证逐轮客户端和分组变化会真实驱动FedML本地训练。"""

    def test_wrong_expected_partition_hash_fails_before_training(
        self,
    ) -> None:
        """验证正式配置中的错误划分指纹会在通信轮开始前被拒绝。"""

        seed_everything(42)
        dataset = build_37_client_dataset()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=37, seed=42
        )
        provider = SequenceTopologyProvider(
            [{0: [0, 1], 1: [2]}, {0: [3], 1: [4]}]
        )
        args = build_dynamic_args()
        args.expected_partition_hash = "wrong-partition"
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
        )
        with self.assertRaisesRegex(ValueError, "客户端划分哈希"):
            FedMLDynamicTopologyTransERunner(
                args=args,
                device=torch.device("cpu"),
                federated_data=federated_data,
                model=model,
                topology_provider=provider,
            )

    def test_two_dynamic_rounds_write_complete_outputs(self) -> None:
        """运行两轮微型图谱并核对调度、指标、汇总和终端信息。"""

        seed_everything(42)
        dataset = build_37_client_dataset()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=37, seed=42
        )
        provider = SequenceTopologyProvider(
            [
                {0: [0, 1], 1: [2]},
                {0: [3], 1: [4, 5, 6]},
            ]
        )
        args = build_dynamic_args()
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
        )
        runner = FedMLDynamicTopologyTransERunner(
            args=args,
            device=torch.device("cpu"),
            federated_data=federated_data,
            model=model,
            topology_provider=provider,
        )

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            with redirect_stdout(output):
                summary = runner.run(result_dir)
            schedule_rows = [
                json.loads(line)
                for line in (
                    result_dir
                    / "dynamic_topology_schedule.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            metrics_text = (
                result_dir / "metrics.csv"
            ).read_text(encoding="utf-8-sig")
            participation = json.loads(
                (
                    result_dir
                    / "dynamic_participation_summary.json"
                ).read_text(encoding="utf-8")
            )

        printed = output.getvalue()
        self.assertIn("epoch=1/2 MAT行=0", printed)
        self.assertIn("epoch=2/2 MAT行=1", printed)
        self.assertIn("验证MRR=未评估", printed)
        self.assertEqual(len(schedule_rows), 2)
        self.assertEqual(
            schedule_rows[0]["active_client_indexes"],
            [0, 1, 2],
        )
        self.assertEqual(
            schedule_rows[1]["active_client_indexes"],
            [3, 4, 5, 6],
        )
        self.assertEqual(
            schedule_rows[0]["aggregation"],
            "hierarchical_two_level_dense_fedavg",
        )
        self.assertTrue(summary["dynamic_client_selection"])
        self.assertTrue(summary["dynamic_grouping"])
        self.assertEqual(len(summary["initial_model_hash"]), 64)
        self.assertEqual(summary["participant_count_min"], 3)
        self.assertEqual(summary["participant_count_max"], 4)
        self.assertEqual(
            participation["unique_participant_set_count"], 2
        )
        self.assertIn("source_round_index", metrics_text)
        self.assertIn("selected_train_triple_fraction", metrics_text)
        for value in summary["final_test_metrics"].values():
            self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
