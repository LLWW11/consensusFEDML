"""验证阶段2双向自对抗采样、频率权重和集中式训练。"""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from HFLSnF_KG_v3.tasks.kge import (
    CentralizedTransETrainer,
    FilteredRankingEvaluator,
    TransE,
    TripleFrequencySubsampler,
    VectorizedFilteredNegativeSampler,
    build_synthetic_knowledge_graph,
    self_adversarial_loss,
)


class Stage2StrongTransETest(unittest.TestCase):
    """检查强TransE训练配方的关键数值和方向语义。"""

    def test_vectorized_sampler_is_filtered_and_directional(self) -> None:
        """确认批量负采样形状正确且只破坏指定实体方向。"""

        positives = torch.tensor(
            [[0, 0, 1], [2, 1, 3]],
            dtype=torch.long,
        )
        true_triples = {(0, 0, 1), (2, 1, 3), (4, 0, 1)}
        sampler = VectorizedFilteredNegativeSampler(
            6,
            true_triples,
            seed=7,
        )
        head_negatives = sampler.sample(
            positives,
            negative_sample_count=16,
            corruption_mode="head",
        ).reshape(2, 16, 3)
        tail_negatives = sampler.sample(
            positives,
            negative_sample_count=16,
            corruption_mode="tail",
        ).reshape(2, 16, 3)

        self.assertTrue(
            torch.equal(
                head_negatives[:, :, 1:],
                positives[:, 1:].unsqueeze(1).expand(-1, 16, -1),
            )
        )
        self.assertTrue(
            torch.equal(
                tail_negatives[:, :, :2],
                positives[:, :2].unsqueeze(1).expand(-1, 16, -1),
            )
        )
        for row in torch.cat(
            [
                head_negatives.reshape(-1, 3),
                tail_negatives.reshape(-1, 3),
            ],
            dim=0,
        ).tolist():
            self.assertNotIn(tuple(row), true_triples)
        self.assertEqual(
            sampler.backend, "torch_device_searchsorted"
        )

    def test_device_sampler_reset_seed_is_reproducible(self) -> None:
        """确认设备端批量采样器重置种子后完全复现负样本序列。"""

        positives = torch.tensor(
            [[0, 0, 1], [2, 1, 3]],
            dtype=torch.long,
        )
        sampler = VectorizedFilteredNegativeSampler(
            6,
            {(0, 0, 1), (2, 1, 3), (4, 0, 1)},
            seed=19,
            num_relations=2,
        )
        first = sampler.sample(
            positives, 32, corruption_mode="head_tail"
        )
        sampler.reset_seed(19)
        second = sampler.sample(
            positives, 32, corruption_mode="head_tail"
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.device, positives.device)

    def test_subsampling_downweights_frequent_patterns(self) -> None:
        """确认高频头关系和尾关系模式获得更小权重。"""

        triples = torch.tensor(
            [
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [4, 1, 5],
            ],
            dtype=torch.long,
        )
        subsampler = TripleFrequencySubsampler(triples)
        weights = subsampler.weights(
            torch.tensor(
                [[0, 0, 1], [4, 1, 5]],
                dtype=torch.long,
            )
        )
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertTrue(
            torch.allclose(
                subsampler.precomputed_weights,
                subsampler.weights(triples),
            )
        )

    def test_batched_evaluator_matches_serial_exactly(self) -> None:
        """确认批量头尾filtered评估与逐查询精确评估完全一致。"""

        dataset = build_synthetic_knowledge_graph()
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
            distance_norm=1,
        )
        evaluator = FilteredRankingEvaluator(dataset)
        serial = evaluator.evaluate(
            model,
            dataset.test_triples,
            torch.device("cpu"),
            candidate_batch_size=3,
            query_batch_size=1,
        )
        batched = evaluator.evaluate(
            model,
            dataset.test_triples,
            torch.device("cpu"),
            candidate_batch_size=3,
            query_batch_size=2,
        )
        for field in (
            "mrr",
            "mean_rank",
            "hits_at_1",
            "hits_at_3",
            "hits_at_10",
            "evaluated_triple_count",
            "evaluated_query_count",
        ):
            self.assertAlmostEqual(serial[field], batched[field])

    def test_self_adversarial_loss_is_finite_and_differentiable(self) -> None:
        """确认自对抗损失能够对正负距离产生有限梯度。"""

        positive = torch.tensor(
            [1.0, 2.0],
            dtype=torch.float32,
            requires_grad=True,
        )
        negative = torch.tensor(
            [[3.0, 4.0], [2.5, 5.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss = self_adversarial_loss(
            positive,
            negative,
            torch.tensor([1.0, 0.5]),
            gamma=6.0,
            temperature=1.0,
        )
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertIsNotNone(positive.grad)
        self.assertIsNotNone(negative.grad)

    def test_relation_stratified_validation_covers_relations(self) -> None:
        """确认关系分层选择至少保留每个验证关系一条事实。"""

        triples = torch.tensor(
            [
                [0, 0, 1],
                [1, 0, 2],
                [2, 1, 3],
                [3, 1, 4],
                [4, 2, 5],
                [5, 2, 0],
            ],
            dtype=torch.long,
        )
        selected = FilteredRankingEvaluator._select_triples(
            triples,
            max_triples=3,
            seed=42,
            relation_stratified=True,
        )
        self.assertEqual(set(selected[:, 1].tolist()), {0, 1, 2})

    def test_centralized_trainer_uses_both_directions(self) -> None:
        """确认CPU冒烟训练实际消费头、尾两个方向并生成指标。"""

        dataset = build_synthetic_knowledge_graph()
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=16,
            distance_norm=1,
        )
        args = SimpleNamespace(
            epochs=3,
            batch_size=2,
            learning_rate=0.01,
            lr=0.01,
            margin=1.0,
            negative_sample_count=4,
            local_objective="bidirectional_self_adversarial",
            fede_gamma=6.0,
            adversarial_temperature=1.0,
            eval_every=1,
            validation_max_triples=4,
            validation_selection="random",
            monitor_every_epoch=False,
            monitor_validation_max_triples=4,
            final_validation_max_triples=0,
            test_max_triples=0,
            evaluation_candidate_batch_size=32,
            evaluation_query_batch_size=2,
            profile_training_timing=True,
            early_stopping_patience=0,
            random_seed=42,
        )
        trainer = CentralizedTransETrainer(
            args,
            dataset,
            model,
            torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            summary = trainer.train(Path(temporary))
            with (Path(temporary) / "metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(summary["epochs_ran"], 3)
        self.assertEqual(
            summary["local_objective"],
            "bidirectional_self_adversarial",
        )
        self.assertEqual(
            summary["negative_sampling_backend"],
            "torch_device_searchsorted",
        )
        self.assertTrue(summary["subsampling_weights_precomputed"])
        self.assertTrue(rows)
        for field in (
            "sampling_seconds",
            "transfer_seconds",
            "forward_backward_seconds",
        ):
            self.assertTrue(
                all(math.isfinite(float(row[field])) for row in rows)
            )
        total_head = sum(
            float(row["head_positive_count"]) for row in rows
        )
        total_tail = sum(
            float(row["tail_positive_count"]) for row in rows
        )
        self.assertGreater(total_head, 0.0)
        self.assertGreater(total_tail, 0.0)
        self.assertTrue(
            math.isfinite(
                float(summary["final_test_metrics"]["mrr"])
            )
        )


if __name__ == "__main__":
    unittest.main()
