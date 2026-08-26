"""验证V5语义主域与实体图局部性的两个删除式消融划分。"""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

from HFLSnF_KG_v5.experiment import build_federated_data
from HFLSnF_KG_v5.tasks.kge import (
    DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
    SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
    partition_train_triples_by_graph_local_no_primary,
    partition_train_triples_by_semantic_domain_graph_local,
    partition_train_triples_by_semantic_domain_no_graph_local,
    relation_domains,
)
from HFLSnF_KG_v5.tasks.kge.graph_semantic_mechanism_partition import (
    _assign_graph_only_once,
    _assign_semantic_only_once,
)
from HFLSnF_KG_v5.tasks.kge.graph_semantic_partition import (
    _build_semantic_packets,
)
from HFLSnF_KG_v5.tests.test_graph_semantic_partition import _build_dataset


class GraphSemanticMechanismPartitionTest(unittest.TestCase):
    """检查两个消融臂的确定性、机制隔离与统一入口。"""

    def setUp(self) -> None:
        """为每项测试建立同一份跨域共享实体合成图。"""

        self.dataset = _build_dataset()
        self.packets = _build_semantic_packets(
            self.dataset,
            relation_domains(self.dataset),
        )

    def test_ablation_partitions_are_deterministic_and_complete(self) -> None:
        """A/B必须可复现、完整、非空且保持域-头实体包互斥。"""

        builders = (
            (
                partition_train_triples_by_graph_local_no_primary,
                DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
            ),
            (
                partition_train_triples_by_semantic_domain_no_graph_local,
                SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
            ),
        )
        for builder, strategy in builders:
            first = builder(
                self.dataset,
                client_count=3,
                seed=42,
                load_tolerance=0.30,
                search_restarts=2,
            )
            second = builder(
                self.dataset,
                client_count=3,
                seed=42,
                load_tolerance=0.30,
                search_restarts=2,
            )
            self.assertEqual(first.partition_hash, second.partition_hash)
            self.assertEqual(first.partition_strategy, strategy)
            self.assertEqual(
                sum(item.triple_count for item in first.partitions),
                int(self.dataset.train_triples.shape[0]),
            )
            self.assertTrue(
                all(item.triple_count > 0 for item in first.partitions)
            )
            owners = {}
            domain_by_relation = relation_domains(self.dataset)
            for partition in first.partitions:
                for head_id, relation_id, _ in (
                    partition.train_triples.tolist()
                ):
                    key = (
                        domain_by_relation[int(relation_id)],
                        int(head_id),
                    )
                    owners.setdefault(key, set()).add(partition.client_id)
            self.assertTrue(all(len(value) == 1 for value in owners.values()))

    def test_graph_only_assignment_does_not_read_domain_labels(self) -> None:
        """固定包结构后，改变域标签不得改变消融A的包归属。"""

        renamed = tuple(
            replace(packet, domain="renamed_{}".format(index))
            for index, packet in enumerate(self.packets)
        )
        expected = _assign_graph_only_once(
            self.packets,
            client_count=3,
            load_tolerance=0.30,
            restart_seed=42,
        )
        actual = _assign_graph_only_once(
            renamed,
            client_count=3,
            load_tolerance=0.30,
            restart_seed=42,
        )
        self.assertEqual(expected, actual)

    def test_semantic_only_assignment_does_not_read_entity_sets(self) -> None:
        """固定域和行号后，改变实体集合不得改变消融B的包归属。"""

        primary_domains = ("film", "sports", "location")
        rewritten = tuple(
            replace(packet, entity_ids=frozenset({100000 + index}))
            for index, packet in enumerate(self.packets)
        )
        expected = _assign_semantic_only_once(
            self.packets,
            client_count=3,
            load_tolerance=0.30,
            restart_seed=42,
            primary_domains=primary_domains,
        )
        actual = _assign_semantic_only_once(
            rewritten,
            client_count=3,
            load_tolerance=0.30,
            restart_seed=42,
            primary_domains=primary_domains,
        )
        self.assertEqual(expected, actual)

    def test_metadata_and_primary_domain_quota_contract(self) -> None:
        """A必须无主域，B的主域名额必须与完整V5一致。"""

        graph_only = partition_train_triples_by_graph_local_no_primary(
            self.dataset,
            client_count=3,
            seed=2024,
            load_tolerance=0.30,
            search_restarts=2,
        )
        semantic_only = (
            partition_train_triples_by_semantic_domain_no_graph_local(
                self.dataset,
                client_count=3,
                seed=2024,
                load_tolerance=0.30,
                search_restarts=2,
            )
        )
        full = partition_train_triples_by_semantic_domain_graph_local(
            self.dataset,
            client_count=3,
            seed=2024,
            load_tolerance=0.30,
            search_restarts=2,
        )
        graph_summary = graph_only.summary()
        semantic_summary = semantic_only.summary()
        self.assertFalse(graph_summary["has_primary_domain"])
        self.assertIsNone(graph_summary["client_primary_domains"])
        self.assertIsNone(
            graph_summary["triple_weighted_primary_domain_fraction"]
        )
        self.assertTrue(graph_summary["uses_entity_locality_objective"])
        self.assertTrue(semantic_summary["has_primary_domain"])
        self.assertFalse(
            semantic_summary["uses_entity_locality_objective"]
        )
        self.assertEqual(
            Counter(semantic_summary["client_primary_domains"]),
            Counter(full.summary()["client_primary_domains"]),
        )

    def test_experiment_dispatches_both_ablation_strategies(self) -> None:
        """统一实验入口必须按配置构造A/B两个划分。"""

        for strategy in (
            DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
            SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
        ):
            args = SimpleNamespace(
                partition_strategy=strategy,
                partition_domain_extractor="freebase_top_level",
                partition_load_tolerance=0.30,
                partition_search_restarts=2,
                client_num_in_total=3,
                random_seed=42,
            )
            partition = build_federated_data(args, self.dataset)
            self.assertEqual(partition.partition_strategy, strategy)

    def test_metadata_rejects_objective_order_drift(self) -> None:
        """公共数据合同必须拒绝A的分配目标顺序被替换。"""

        partition = partition_train_triples_by_graph_local_no_primary(
            self.dataset,
            client_count=3,
            seed=42,
            load_tolerance=0.30,
            search_restarts=2,
        )
        invalid_metadata = dict(partition.partition_metadata)
        invalid_metadata["assignment_objective_order"] = [
            "current_load",
            "new_entity_count",
            "seeded_tie_rank",
        ]
        with self.assertRaisesRegex(ValueError, "消融A包分配目标顺序不正确"):
            replace(partition, partition_metadata=invalid_metadata)


if __name__ == "__main__":
    unittest.main()
