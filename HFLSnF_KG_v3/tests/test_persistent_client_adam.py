"""验证客户端Adam状态按客户端隔离持久化且重置基线保持不变。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from HFLSnF_KG_v3.fedml_kge import FedMLTransEModelTrainer
from HFLSnF_KG_v3.tasks.kge import (
    TransE,
    build_synthetic_knowledge_graph,
    partition_train_triples_by_head,
)


class PersistentClientAdamTest(unittest.TestCase):
    """检查Adam状态模式、步数续接和客户端间状态隔离。"""

    @staticmethod
    def _build_args(state_mode: str) -> SimpleNamespace:
        """构造能够执行合成知识图谱本地训练的最小参数。"""

        return SimpleNamespace(
            random_seed=42,
            round_idx=0,
            epochs=2,
            batch_size=4,
            client_optimizer="adam",
            client_optimizer_state_mode=state_mode,
            learning_rate=0.01,
            lr=0.01,
            margin=1.0,
            negative_sample_count=4,
            local_objective="bidirectional_self_adversarial",
            fede_gamma=6.0,
            adversarial_temperature=1.0,
            profile_training_timing=False,
        )

    @staticmethod
    def _build_trainer(state_mode: str):
        """创建合成数据、两个客户端分区及指定状态模式的训练器。"""

        dataset = build_synthetic_knowledge_graph()
        federated_data = partition_train_triples_by_head(
            dataset,
            client_count=2,
            seed=42,
        )
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
            distance_norm=1,
        )
        args = PersistentClientAdamTest._build_args(state_mode)
        trainer = FedMLTransEModelTrainer(model, args, dataset)
        return trainer, args, federated_data

    @staticmethod
    def _train_partition(
        trainer,
        args,
        partition,
        global_state,
        round_index: int,
    ):
        """从指定全局参数训练一个客户端并返回本地审计指标。"""

        trainer.set_id(int(partition.client_id))
        trainer.set_model_params(global_state)
        args.round_idx = int(round_index)
        return trainer.train(partition, torch.device("cpu"), args)

    def test_persistent_mode_reuses_each_client_own_adam_state(self) -> None:
        """确认同一客户端续接步数而新客户端从零建立独立状态。"""

        trainer, args, federated_data = self._build_trainer(
            "persistent_per_client"
        )
        global_state = {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
        }
        first_client = federated_data.partitions[0]
        second_client = federated_data.partitions[1]

        first = self._train_partition(
            trainer,
            args,
            first_client,
            global_state,
            round_index=0,
        )
        second = self._train_partition(
            trainer,
            args,
            first_client,
            global_state,
            round_index=1,
        )
        other_client = self._train_partition(
            trainer,
            args,
            second_client,
            global_state,
            round_index=1,
        )

        self.assertEqual(first["optimizer_state_reused"], 0.0)
        self.assertEqual(first["optimizer_step_before"], 0.0)
        self.assertGreater(first["optimizer_step_after"], 0.0)
        self.assertEqual(second["optimizer_state_reused"], 1.0)
        self.assertEqual(
            second["optimizer_step_before"],
            first["optimizer_step_after"],
        )
        self.assertGreater(
            second["optimizer_step_after"],
            second["optimizer_step_before"],
        )
        self.assertEqual(other_client["optimizer_state_reused"], 0.0)
        self.assertEqual(other_client["optimizer_step_before"], 0.0)
        self.assertEqual(trainer.optimizer_state_cache_size, 2)

    def test_reset_mode_rebuilds_adam_for_every_call(self) -> None:
        """确认原重置模式每次调用的Adam步数仍从零开始。"""

        trainer, args, federated_data = self._build_trainer("reset")
        global_state = {
            name: tensor.detach().clone()
            for name, tensor in trainer.model.state_dict().items()
        }
        partition = federated_data.partitions[0]
        first = self._train_partition(
            trainer,
            args,
            partition,
            global_state,
            round_index=0,
        )
        second = self._train_partition(
            trainer,
            args,
            partition,
            global_state,
            round_index=1,
        )

        self.assertEqual(first["optimizer_state_reused"], 0.0)
        self.assertEqual(second["optimizer_state_reused"], 0.0)
        self.assertEqual(first["optimizer_step_before"], 0.0)
        self.assertEqual(second["optimizer_step_before"], 0.0)
        self.assertEqual(trainer.optimizer_state_cache_size, 0)

    def test_invalid_optimizer_state_mode_fails_fast(self) -> None:
        """确认未知Adam状态模式在训练前给出明确错误。"""

        with self.assertRaises(ValueError):
            self._build_trainer("unknown")


if __name__ == "__main__":
    unittest.main()
