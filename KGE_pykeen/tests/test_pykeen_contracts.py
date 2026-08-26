"""验证PyKEEN模型、损失、采样、评估和检查点数值合同。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from KGE_pykeen.data import build_synthetic_knowledge_graph
from KGE_pykeen.directional import load_project_checkpoint
from KGE_pykeen.evaluator import FilteredRankingEvaluator
from KGE_pykeen.experiment import _checkpoint_payload
from KGE_pykeen.model import TransE
from KGE_pykeen.negative_sampling import (
    VectorizedFilteredNegativeSampler,
)
from KGE_pykeen.objectives import (
    WeightedNSSALossAdapter,
    self_adversarial_loss,
)
from KGE_pykeen.pykeen_bridge import (
    build_triples_factories,
    flatten_metric_results,
)


class PyKEENContractTest(unittest.TestCase):
    """检查严格配方与PyKEEN 1.10.1之间的张量级合同。"""

    def setUp(self) -> None:
        """构造共享编号的合成知识图谱与PyKEEN模型。"""

        torch.manual_seed(42)
        self.dataset = build_synthetic_knowledge_graph()
        self.training_factory, _, _ = build_triples_factories(
            self.dataset
        )
        self.model = TransE(
            triples_factory=self.training_factory,
            embedding_dim=8,
            distance_norm=1,
            random_seed=42,
        )

    def test_shared_triples_factory_preserves_ids(self) -> None:
        """确认PyKEEN工厂不改变映射、编号和逆关系设置。"""

        self.assertEqual(
            dict(self.training_factory.entity_to_id),
            self.dataset.entity_to_id,
        )
        self.assertEqual(
            dict(self.training_factory.relation_to_id),
            self.dataset.relation_to_id,
        )
        self.assertFalse(self.training_factory.create_inverse_triples)
        self.assertTrue(
            torch.equal(
                self.training_factory.mapped_triples,
                self.dataset.train_triples,
            )
        )

    def test_native_sampler_accepts_full_corruption_names(self) -> None:
        """确认PyKEEN 1.10.1原生采样器接受完整头尾名称。"""

        from pykeen.sampling import BasicNegativeSampler

        sampler = BasicNegativeSampler(
            mapped_triples=self.dataset.train_triples,
            num_negs_per_pos=2,
            corruption_scheme=("head", "tail"),
            filtered=True,
        )
        self.assertEqual(
            tuple(sampler.corruption_scheme),
            ("head", "tail"),
        )
    def test_distance_is_negative_pykeen_score(self) -> None:
        """确认适配距离等于PyKEEN TransE分数的相反数。"""

        triples = self.dataset.train_triples[:4]
        distance = self.model.score_triples(triples)
        pykeen_score = self.model.pykeen_model.score_hrt(
            triples
        ).reshape(-1)
        torch.testing.assert_close(
            distance,
            -pykeen_score,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    def test_weighted_nssa_matches_native_for_uniform_weights(self) -> None:
        """确认均匀权重时适配损失及梯度等于PyKEEN原生NSSA。"""

        positive_a = torch.tensor(
            [1.0, 2.0, 3.0], requires_grad=True
        )
        negative_a = torch.tensor(
            [
                [4.0, 5.0],
                [3.0, 6.0],
                [2.0, 7.0],
            ],
            requires_grad=True,
        )
        weights = torch.ones(3)
        adapter = WeightedNSSALossAdapter(
            gamma=9.0,
            temperature=1.0,
        )
        adapted = adapter(positive_a, negative_a, weights)
        adapted_gradients = torch.autograd.grad(
            adapted,
            (positive_a, negative_a),
        )

        positive_b = positive_a.detach().clone().requires_grad_(True)
        negative_b = negative_a.detach().clone().requires_grad_(True)
        native = adapter.pykeen_loss.process_slcwa_scores(
            positive_scores=-positive_b.reshape(-1, 1),
            negative_scores=-negative_b,
        )
        native_gradients = torch.autograd.grad(
            native,
            (positive_b, negative_b),
        )
        torch.testing.assert_close(
            adapted,
            native,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        for adapted_gradient, native_gradient in zip(
            adapted_gradients,
            native_gradients,
        ):
            torch.testing.assert_close(
                adapted_gradient,
                native_gradient,
                rtol=1.0e-6,
                atol=1.0e-7,
            )

    def test_weighted_nssa_matches_reference_with_frequency_weights(self) -> None:
        """确认非均匀频率权重下损失及梯度与原公式一致。"""

        positive_a = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
        negative_a = torch.tensor(
            [[2.0, 3.0], [1.0, 4.0], [3.0, 5.0]],
            requires_grad=True,
        )
        weights = torch.tensor([0.2, 0.3, 0.5])
        adapter = WeightedNSSALossAdapter(gamma=9.0, temperature=1.0)
        adapted = adapter(positive_a, negative_a, weights)
        adapted_gradients = torch.autograd.grad(
            adapted, (positive_a, negative_a)
        )

        positive_b = positive_a.detach().clone().requires_grad_(True)
        negative_b = negative_a.detach().clone().requires_grad_(True)
        reference = self_adversarial_loss(
            positive_b,
            negative_b,
            weights,
            gamma=9.0,
            temperature=1.0,
        )
        reference_gradients = torch.autograd.grad(
            reference, (positive_b, negative_b)
        )
        torch.testing.assert_close(
            adapted, reference, rtol=1.0e-6, atol=1.0e-7
        )
        for adapted_gradient, reference_gradient in zip(
            adapted_gradients, reference_gradients
        ):
            torch.testing.assert_close(
                adapted_gradient,
                reference_gradient,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
    def test_strict_sampler_never_returns_known_true_fact(self) -> None:
        """确认matched负采样结果不包含任一已知真事实。"""

        sampler = VectorizedFilteredNegativeSampler(
            num_entities=self.dataset.num_entities,
            true_triples=self.dataset.all_true_triples,
            seed=1051,
            num_relations=self.dataset.num_relations,
        )
        negatives = sampler.sample(
            self.dataset.train_triples[:4],
            negative_sample_count=8,
            corruption_mode="head",
        )
        self.assertEqual(tuple(negatives.shape), (32, 3))
        known = set(self.dataset.all_true_triples)
        self.assertTrue(
            all(tuple(row) not in known for row in negatives.tolist())
        )

    def test_strict_sampler_preserves_direction_and_seed_sequence(self) -> None:
        """确认负样本数量、破坏方向和固定种子序列保持一致。"""

        positives = self.dataset.train_triples[:4]
        sampler_a = VectorizedFilteredNegativeSampler(
            self.dataset.num_entities,
            self.dataset.all_true_triples,
            seed=1051,
            num_relations=self.dataset.num_relations,
        )
        sampler_b = VectorizedFilteredNegativeSampler(
            self.dataset.num_entities,
            self.dataset.all_true_triples,
            seed=1051,
            num_relations=self.dataset.num_relations,
        )
        head_a = sampler_a.sample(positives, 8, "head").reshape(4, 8, 3)
        head_b = sampler_b.sample(positives, 8, "head").reshape(4, 8, 3)
        tail_a = sampler_a.sample(positives, 8, "tail").reshape(4, 8, 3)
        tail_b = sampler_b.sample(positives, 8, "tail").reshape(4, 8, 3)
        self.assertTrue(torch.equal(head_a, head_b))
        self.assertTrue(torch.equal(tail_a, tail_b))
        self.assertTrue(torch.equal(head_a[..., 1], positives[:, None, 1].expand(4, 8)))
        self.assertTrue(torch.equal(head_a[..., 2], positives[:, None, 2].expand(4, 8)))
        self.assertTrue(torch.equal(tail_a[..., 0], positives[:, None, 0].expand(4, 8)))
        self.assertTrue(torch.equal(tail_a[..., 1], positives[:, None, 1].expand(4, 8)))
    def test_canonical_and_pykeen_optimistic_mrr_match(self) -> None:
        """确认两个评估器的完整乐观filtered MRR一致。"""

        from pykeen.evaluation import RankBasedEvaluator

        canonical = FilteredRankingEvaluator(
            self.dataset
        ).evaluate(
            self.model,
            self.dataset.test_triples,
            torch.device("cpu"),
            query_batch_size=1,
            candidate_batch_size=16,
        )
        metric_results = RankBasedEvaluator(
            filtered=True
        ).evaluate(
            model=self.model.pykeen_model,
            mapped_triples=self.dataset.test_triples,
            additional_filter_triples=[
                self.dataset.train_triples,
                self.dataset.valid_triples,
            ],
            batch_size=1,
            slice_size=16,
            use_tqdm=False,
        )
        flattened = flatten_metric_results(metric_results)
        metric_suffixes = {
            "mrr": "inverse_harmonic_mean_rank",
            "mean_rank": "arithmetic_mean_rank",
            "hits_at_1": "hits_at_1",
            "hits_at_3": "hits_at_3",
            "hits_at_10": "hits_at_10",
        }
        for canonical_key, suffix in metric_suffixes.items():
            candidates = [
                value
                for key, value in flattened.items()
                if key.endswith(
                    "both.optimistic.{}".format(suffix)
                )
            ]
            self.assertEqual(len(candidates), 1, msg=suffix)
            self.assertAlmostEqual(
                canonical[canonical_key], candidates[0], places=7
            )

    def test_checkpoint_uses_canonical_embedding_keys(self) -> None:
        """确认规范化状态继续公开旧评估链要求的两个键。"""

        state = self.model.state_dict()
        self.assertEqual(
            set(state),
            {
                "entity_embeddings.weight",
                "relation_embeddings.weight",
            },
        )
        clone = {
            name: tensor.detach().clone()
            for name, tensor in state.items()
        }
        self.model.load_state_dict(clone)
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, clone[name])

    def test_checkpoint_reloads_through_directional_evaluator(self) -> None:
        """确认规范化检查点的权重形状、数值和映射均可重载。"""

        payload = _checkpoint_payload(self.model, self.dataset, {})
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_best.pt"
            torch.save(payload, checkpoint_path)
            bundle = load_project_checkpoint(
                checkpoint_path,
                self.dataset,
                distance_norm_override=1,
            )
        state = self.model.state_dict()
        torch.testing.assert_close(
            bundle.entity_embeddings,
            state["entity_embeddings.weight"],
        )
        torch.testing.assert_close(
            bundle.relation_embeddings,
            state["relation_embeddings.weight"],
        )
        self.assertEqual(
            tuple(bundle.entity_embeddings.shape),
            (self.dataset.num_entities, 8),
        )
        self.assertEqual(
            tuple(bundle.relation_embeddings.shape),
            (self.dataset.num_relations, 8),
        )

if __name__ == "__main__":
    unittest.main()
