"""FedML普通联邦TransE客户端、聚合和训练链路测试。"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from fedml.core import ClientTrainer
from fedml.simulation.sp.fedavg.client import Client as FedMLClient

from HFLSnF_KG.core.aggregation import DenseFedAvgAggregator
from HFLSnF_KG.core.randomness import seed_everything
from HFLSnF_KG.core.types import ClientUpdate
from HFLSnF_KG.fedml_kge import FedMLTransERunner
from HFLSnF_KG.tasks.kge import (
    TransE,
    build_synthetic_knowledge_graph,
    partition_train_triples_by_head,
)


def build_federated_transe_args() -> SimpleNamespace:
    """构造三客户端两轮普通联邦TransE轻量参数。"""

    return SimpleNamespace(
        training_type="simulation",
        backend="sp",
        federated_optimizer="FedAvgTransE",
        client_num_in_total=3,
        client_num_per_round=3,
        comm_round=2,
        epochs=1,
        batch_size=4,
        client_optimizer="adam",
        learning_rate=0.01,
        lr=0.01,
        margin=1.0,
        negative_sample_count=1,
        eval_every=1,
        validation_max_triples=0,
        final_validation_max_triples=0,
        test_max_triples=0,
        evaluation_candidate_batch_size=16,
        centralized_reference_mrr=0.3,
        random_seed=23,
    )


class FedMLFederatedTransETest(unittest.TestCase):
    """验证阶段四使用FedML官方客户端接口并正确执行普通FedAvg。"""

    @staticmethod
    def _build_runner() -> FedMLTransERunner:
        """构造固定种子的合成图FedML TransE运行器。"""

        seed_everything(23)
        dataset = build_synthetic_knowledge_graph()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=3, seed=23
        )
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
        )
        return FedMLTransERunner(
            args=build_federated_transe_args(),
            device=torch.device("cpu"),
            federated_data=federated_data,
            model=model,
        )

    def test_clients_use_fedml_bases_and_local_weight(self) -> None:
        """验证客户端和训练器继承FedML基类且更新权重等于三元组数。"""

        runner = self._build_runner()
        trainer = runner.trainer
        self.assertIsInstance(trainer.model_trainer, ClientTrainer)
        self.assertTrue(
            all(
                isinstance(client, FedMLClient)
                for client in trainer.client_registry.values()
            )
        )
        client = trainer.client_registry[0]
        update = client.train_from_global(
            runner.get_global_state(), round_index=0
        )
        self.assertEqual(update.weight, float(client.partition.triple_count))
        self.assertTrue(
            math.isfinite(float(update.local_metrics["train_loss"]))
        )

    def test_dense_fedavg_matches_manual_and_order_is_stable(self) -> None:
        """验证TransE完整参数按三元组数加权且交换客户端顺序结果一致。"""

        first_state = {
            "entity_embeddings.weight": torch.tensor(
                [[1.0, 3.0], [5.0, 7.0]]
            ),
            "relation_embeddings.weight": torch.tensor([[2.0, 4.0]]),
        }
        second_state = {
            "entity_embeddings.weight": torch.tensor(
                [[5.0, 7.0], [9.0, 11.0]]
            ),
            "relation_embeddings.weight": torch.tensor([[6.0, 8.0]]),
        }
        updates = [
            ClientUpdate(0, 1.0, first_state),
            ClientUpdate(1, 3.0, second_state),
        ]
        aggregator = DenseFedAvgAggregator()
        forward = aggregator.aggregate(updates)
        reverse = aggregator.aggregate(list(reversed(updates)))
        expected_entities = (
            first_state["entity_embeddings.weight"]
            + 3.0 * second_state["entity_embeddings.weight"]
        ) / 4.0
        self.assertTrue(
            torch.allclose(
                forward["entity_embeddings.weight"], expected_entities
            )
        )
        for name in forward:
            self.assertTrue(
                torch.allclose(forward[name], reverse[name], atol=1e-7)
            )

    def test_two_round_training_restores_best_and_writes_outputs(self) -> None:
        """验证两轮训练、最佳模型恢复、filtered指标和结果文件完整。"""

        runner = self._build_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            summary = runner.run(result_dir)
            self.assertTrue((result_dir / "metrics.csv").is_file())
            self.assertTrue(
                (result_dir / "participation_schedule.jsonl").is_file()
            )
            self.assertTrue(
                (result_dir / "client_partition_summary.json").is_file()
            )
        self.assertEqual(summary["runtime"], "fedml_fedavg_transe")
        self.assertEqual(summary["comm_round"], 2)
        self.assertGreaterEqual(summary["best_round"], 1)
        for value in summary["final_test_metrics"].values():
            self.assertTrue(math.isfinite(float(value)))
        entity_state = runner.get_global_state()[
            "entity_embeddings.weight"
        ]
        entity_norms = torch.linalg.vector_norm(
            entity_state, ord=2, dim=1
        )
        self.assertTrue(
            torch.allclose(
                entity_norms,
                torch.ones_like(entity_norms),
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
