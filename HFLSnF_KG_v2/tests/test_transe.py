"""TransE、filtered负采样和filtered排名测试。"""

from __future__ import annotations

import unittest

import torch

from HFLSnF_KG_v2.tasks.kge import (
    FilteredNegativeSampler,
    FilteredRankingEvaluator,
    TransE,
    build_knowledge_graph_dataset,
    build_synthetic_knowledge_graph,
)


class TransECoreTest(unittest.TestCase):
    """验证TransE核心数学接口和知识图谱评估口径。"""

    def test_forward_shape_and_entity_normalization(self) -> None:
        """验证TransE输出形状正确且初始化实体向量为单位范数。"""

        model = TransE(5, 2, 4, distance_norm=1)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
        scores = model(triples)
        norms = torch.linalg.vector_norm(
            model.entity_embeddings.weight.detach(), ord=2, dim=1
        )
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))

    def test_negative_sampling_is_filtered_and_reproducible(self) -> None:
        """验证负样本不属于真三元组集合且固定种子可复现。"""

        dataset = build_synthetic_knowledge_graph()
        first = FilteredNegativeSampler(
            dataset.num_entities, dataset.all_true_triples, seed=7
        )
        second = FilteredNegativeSampler(
            dataset.num_entities, dataset.all_true_triples, seed=7
        )
        positives = dataset.train_triples[:3]
        negatives_a = first.sample(positives, negative_sample_count=2)
        negatives_b = second.sample(positives, negative_sample_count=2)
        self.assertTrue(torch.equal(negatives_a, negatives_b))
        self.assertEqual(tuple(negatives_a.shape), (6, 3))
        for triple in negatives_a.tolist():
            self.assertNotIn(tuple(triple), dataset.all_true_triples)

    def test_filtered_rank_removes_other_true_entity(self) -> None:
        """验证比目标更优的其他真实体会从排名候选中删除。"""

        dataset = build_knowledge_graph_dataset(
            "rank-test",
            train_triples=[("a", "r", "c")],
            valid_triples=[("d", "r", "b")],
            test_triples=[("a", "r", "b")],
        )
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=1,
            distance_norm=1,
        )
        with torch.no_grad():
            # 编号顺序为a、c、d、b；c比目标b距离更小，但它也是已知真尾实体。
            model.entity_embeddings.weight.copy_(
                torch.tensor([[0.0], [1.0], [2.0], [1.2]])
            )
            model.relation_embeddings.weight.copy_(torch.tensor([[1.0]]))
        evaluator = FilteredRankingEvaluator(dataset)
        rank = evaluator.rank_query(
            model,
            tuple(dataset.test_triples[0].tolist()),
            predict_head=False,
            device=torch.device("cpu"),
            candidate_batch_size=2,
        )
        self.assertEqual(rank, 1)

    def test_full_filtered_metrics_are_finite(self) -> None:
        """验证头尾双向filtered评估输出有限指标和正确查询数。"""

        dataset = build_synthetic_knowledge_graph()
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=4,
        )
        metrics = FilteredRankingEvaluator(dataset).evaluate(
            model,
            dataset.test_triples,
            torch.device("cpu"),
            candidate_batch_size=4,
        )
        for metric_name in (
            "mrr",
            "mean_rank",
            "hits_at_1",
            "hits_at_3",
            "hits_at_10",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[metric_name])))
        self.assertEqual(
            int(metrics["evaluated_query_count"]),
            2 * int(dataset.test_triples.shape[0]),
        )


if __name__ == "__main__":
    unittest.main()
