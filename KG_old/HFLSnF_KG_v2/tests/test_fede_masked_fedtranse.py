"""FedE式行级聚合、本地目标和动态MAT训练链路测试。"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from HFLSnF_KG_v2.core.aggregation import RowMaskedFedAvgAggregator
from HFLSnF_KG_v2.core.randomness import seed_everything
from HFLSnF_KG_v2.core.topology import SequenceTopologyProvider
from HFLSnF_KG_v2.core.types import ClientUpdate
from HFLSnF_KG_v2.fedml_kge import (
    FedMLDynamicTopologyTransERunner,
    FedMLTransEClient,
    FedMLTransEModelTrainer,
)
from HFLSnF_KG_v2.tasks.kge import (
    FilteredNegativeSampler,
    TransE,
    build_knowledge_graph_dataset,
    partition_train_triples_by_head,
)


def _state(
    entity_rows,
    relation_rows,
) -> dict:
    """用短列表构造便于手工核对的TransE完整参数表。"""

    return {
        "entity_embeddings.weight": torch.tensor(
            entity_rows, dtype=torch.float32
        ),
        "relation_embeddings.weight": torch.tensor(
            relation_rows, dtype=torch.float32
        ),
    }


def _update(
    client_id: int,
    state_dict: dict,
    entity_mask,
    relation_mask,
) -> ClientUpdate:
    """构造一个带实体及关系正知识掩码的客户端更新。"""

    masks = {
        "entity_embeddings.weight": torch.tensor(
            entity_mask, dtype=torch.bool
        ),
        "relation_embeddings.weight": torch.tensor(
            relation_mask, dtype=torch.bool
        ),
    }
    return ClientUpdate(
        client_id=client_id,
        weight=1.0,
        state_dict=state_dict,
        parameter_masks=masks,
    )


def _build_37_client_dataset():
    """构造可被头实体策略严格分成37个非空客户端的微型图谱。"""

    train = [
        (
            "head_{:02d}".format(client_id),
            "linked_to",
            "tail_{:02d}".format(client_id),
        )
        for client_id in range(37)
    ]
    return build_knowledge_graph_dataset(
        "synthetic-fede-masked-37-client-kg",
        train,
        [("head_00", "linked_to", "tail_01")],
        [("head_01", "linked_to", "tail_02")],
    )


def _build_dynamic_args(
    local_objective: str,
    local_epochs: int,
    negative_sample_count: int,
) -> SimpleNamespace:
    """构造两轮CPU动态拓扑冒烟所需的完整FedML参数。"""

    return SimpleNamespace(
        training_type="simulation",
        backend="sp",
        federated_optimizer="DynamicTopologyTransE",
        comparison_scenario="HFLSnF-V2-Masked-Test",
        client_num_in_total=37,
        client_num_per_round=37,
        topology_architecture="hfl",
        topology_snf=True,
        topology_edge_mode="dynamic",
        aggregation_mode="row_mask_presence",
        local_objective=local_objective,
        fede_gamma=10.0,
        adversarial_temperature=1.0,
        comm_round=2,
        epochs=local_epochs,
        batch_size=2,
        client_optimizer="adam",
        learning_rate=0.01,
        lr=0.01,
        margin=1.0,
        negative_sample_count=negative_sample_count,
        eval_every=2,
        validation_max_triples=0,
        final_validation_max_triples=0,
        test_max_triples=0,
        evaluation_candidate_batch_size=128,
        centralized_reference_mrr=0.2,
        random_seed=42,
    )


class RowMaskedFedAvgTest(unittest.TestCase):
    """验证实体、关系和回退行的FedE式行级聚合语义。"""

    def test_manual_overlap_and_fallback_rows(self) -> None:
        """手工核对重叠行取均值、独占行保留和无人行回退。"""

        aggregator = RowMaskedFedAvgAggregator()
        global_state = _state(
            [[10.0], [20.0], [30.0], [40.0]],
            [[100.0], [200.0], [300.0]],
        )
        update_zero = _update(
            0,
            _state(
                [[1.0], [2.0], [999.0], [999.0]],
                [[11.0], [999.0], [999.0]],
            ),
            [True, True, False, False],
            [True, False, False],
        )
        update_one = _update(
            1,
            _state(
                [[888.0], [6.0], [7.0], [888.0]],
                [[888.0], [22.0], [888.0]],
            ),
            [False, True, True, False],
            [False, True, False],
        )

        statistics = aggregator.accumulate(
            [update_zero, update_one]
        )
        result = aggregator.finalize(statistics, global_state)
        summary = aggregator.summarize(statistics)

        torch.testing.assert_close(
            result["entity_embeddings.weight"],
            torch.tensor([[1.0], [4.0], [7.0], [40.0]]),
        )
        torch.testing.assert_close(
            result["relation_embeddings.weight"],
            torch.tensor([[11.0], [22.0], [300.0]]),
        )
        self.assertEqual(
            summary["entity_embeddings.weight"]["updated_row_count"],
            3,
        )
        self.assertEqual(
            summary["entity_embeddings.weight"]["fallback_row_count"],
            1,
        )
        self.assertEqual(
            summary["relation_embeddings.weight"]["fallback_row_count"],
            1,
        )

    def test_hierarchical_merge_equals_direct_row_aggregation(self) -> None:
        """验证边缘分子分母合并与同参与集合直接聚合完全等价。"""

        aggregator = RowMaskedFedAvgAggregator()
        global_state = _state(
            [[10.0], [20.0], [30.0], [40.0]],
            [[100.0], [200.0]],
        )
        updates = [
            _update(
                0,
                _state(
                    [[1.0], [2.0], [3.0], [4.0]],
                    [[11.0], [12.0]],
                ),
                [True, True, False, False],
                [True, False],
            ),
            _update(
                1,
                _state(
                    [[5.0], [6.0], [7.0], [8.0]],
                    [[21.0], [22.0]],
                ),
                [False, True, True, False],
                [True, True],
            ),
            _update(
                2,
                _state(
                    [[9.0], [10.0], [11.0], [12.0]],
                    [[31.0], [32.0]],
                ),
                [True, False, True, False],
                [False, True],
            ),
        ]
        direct_statistics = aggregator.accumulate(updates)
        edge_statistics = [
            aggregator.accumulate(updates[:2]),
            aggregator.accumulate(updates[2:]),
        ]
        hierarchical_statistics = aggregator.merge(edge_statistics)
        direct = aggregator.finalize(
            direct_statistics, global_state
        )
        hierarchical = aggregator.finalize(
            hierarchical_statistics, global_state
        )

        self.assertEqual(
            set(direct_statistics.contributor_ids), {0, 1, 2}
        )
        for name in direct:
            torch.testing.assert_close(
                direct[name], hierarchical[name]
            )


class FedEClientAndObjectiveTest(unittest.TestCase):
    """验证正知识掩码、尾负采样和自对抗目标。"""

    def test_client_masks_only_follow_positive_triples(self) -> None:
        """验证客户端行所有权严格匹配本地正三元组知识范围。"""

        dataset = _build_37_client_dataset()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=37, seed=42
        )
        args = _build_dynamic_args(
            "margin_ranking", 1, 1
        )
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
        )
        trainer = FedMLTransEModelTrainer(model, args, dataset)
        partition = federated_data.partitions[0]
        client = FedMLTransEClient(
            partition,
            args,
            torch.device("cpu"),
            trainer,
        )

        masks = client.get_parameter_masks()
        counts = client.get_row_counts()
        expected_entities = set(
            int(value) for value in partition.entity_ids.tolist()
        )
        expected_relations = set(
            int(value) for value in partition.relation_ids.tolist()
        )
        actual_entities = set(
            torch.nonzero(
                masks["entity_embeddings.weight"],
                as_tuple=False,
            )
            .reshape(-1)
            .tolist()
        )
        actual_relations = set(
            torch.nonzero(
                masks["relation_embeddings.weight"],
                as_tuple=False,
            )
            .reshape(-1)
            .tolist()
        )
        self.assertEqual(actual_entities, expected_entities)
        self.assertEqual(actual_relations, expected_relations)
        self.assertEqual(
            float(counts["entity_embeddings.weight"].sum().item()),
            float(2 * partition.triple_count),
        )
        self.assertEqual(
            float(counts["relation_embeddings.weight"].sum().item()),
            float(partition.triple_count),
        )

    def test_tail_negative_sampling_is_global_filtered_and_reproducible(
        self,
    ) -> None:
        """验证FedE尾负样本不会命中全局真事实且固定种子可复现。"""

        true_triples = {(0, 0, 1), (0, 0, 2), (3, 0, 1)}
        positives = torch.tensor(
            [[0, 0, 1], [3, 0, 1]], dtype=torch.long
        )
        first = FilteredNegativeSampler(5, true_triples, seed=9)
        second = FilteredNegativeSampler(5, true_triples, seed=9)
        first_values = first.sample(
            positives, 4, corruption_mode="tail"
        )
        second_values = second.sample(
            positives, 4, corruption_mode="tail"
        )

        torch.testing.assert_close(first_values, second_values)
        repeated_positives = positives.repeat_interleave(4, dim=0)
        torch.testing.assert_close(
            first_values[:, :2], repeated_positives[:, :2]
        )
        for row in first_values.tolist():
            self.assertNotIn(tuple(int(value) for value in row), true_triples)

    def test_self_adversarial_loss_is_finite_and_deterministic(self) -> None:
        """验证FedE自对抗逻辑损失在相同输入下有限且确定。"""

        args = SimpleNamespace(
            fede_gamma=10.0,
            adversarial_temperature=1.0,
        )
        positive_scores = torch.tensor(
            [0.4, 0.4, 0.9, 0.9], dtype=torch.float32
        )
        negative_scores = torch.tensor(
            [2.0, 3.0, 1.5, 2.5], dtype=torch.float32
        )
        first = FedMLTransEModelTrainer._fede_self_adversarial_loss(
            positive_scores,
            negative_scores,
            positive_batch_size=2,
            negative_sample_count=2,
            args=args,
        )
        second = FedMLTransEModelTrainer._fede_self_adversarial_loss(
            positive_scores,
            negative_scores,
            positive_batch_size=2,
            negative_sample_count=2,
            args=args,
        )
        self.assertTrue(math.isfinite(float(first.item())))
        torch.testing.assert_close(first, second)


class DynamicMaskedFedEIntegrationTest(unittest.TestCase):
    """验证三种V2消融均能完成两轮动态拓扑CPU训练。"""

    def test_three_masked_variants_finish_two_rounds(self) -> None:
        """以轻量参数覆盖当前目标、公平FedE和论文参数形态。"""

        variants = (
            ("masked", "margin_ranking", 1, 1),
            ("fede_fair", "fede_self_adversarial", 1, 2),
            ("fede_paper_shape", "fede_self_adversarial", 3, 2),
        )
        provider = SequenceTopologyProvider(
            [
                {0: [0, 1], 1: [2]},
                {0: [3], 1: [4, 5, 6]},
            ]
        )
        dataset = _build_37_client_dataset()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=37, seed=42
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            for (
                variant_name,
                local_objective,
                local_epochs,
                negative_count,
            ) in variants:
                seed_everything(42)
                args = _build_dynamic_args(
                    local_objective,
                    local_epochs,
                    negative_count,
                )
                model = TransE(
                    dataset.num_entities,
                    dataset.num_relations,
                    embedding_dim=8,
                    distance_norm=(
                        1
                        if local_objective
                        == "fede_self_adversarial"
                        else 2
                    ),
                )
                runner = FedMLDynamicTopologyTransERunner(
                    args=args,
                    device=torch.device("cpu"),
                    federated_data=federated_data,
                    model=model,
                    topology_provider=provider,
                )
                result_dir = Path(temp_dir) / variant_name
                summary = runner.run(result_dir)
                schedules = [
                    json.loads(line)
                    for line in (
                        result_dir
                        / "dynamic_topology_schedule.jsonl"
                    )
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]

                self.assertEqual(len(schedules), 2)
                self.assertEqual(
                    schedules[0]["aggregation"],
                    "hierarchical_two_level_row_mask_presence",
                )
                self.assertEqual(
                    schedules[0]["active_client_indexes"],
                    schedules[0]["contributing_client_indexes"],
                )
                self.assertEqual(
                    set(
                        schedules[0][
                            "group_contributing_client_indexes"
                        ]["0"]
                    ),
                    {0, 1},
                )
                self.assertIn(
                    "group_parameter_row_statistics",
                    schedules[0],
                )
                self.assertGreater(
                    schedules[0]["parameter_row_statistics"][
                        "entity_embeddings.weight"
                    ]["updated_row_count"],
                    0,
                )
                self.assertEqual(
                    summary["aggregation_mode"],
                    "row_mask_presence",
                )
                self.assertEqual(
                    summary["local_objective"], local_objective
                )
                for value in summary["final_test_metrics"].values():
                    self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
