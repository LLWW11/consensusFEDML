"""验证V5图语义划分的域解析、合同和配置接入。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from HFLSnF_KG_v5.experiment import build_federated_data
from HFLSnF_KG_v5.tasks.kge import (
    SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
    build_knowledge_graph_dataset,
    freebase_relation_domain,
    partition_train_triples_by_semantic_domain_graph_local,
    relation_domains,
)


def _build_dataset():
    """构造同时包含跨域头实体和共享尾实体的测试图。"""

    train = []
    domains = ("film", "sports", "location")
    for head_index in range(12):
        head = "head_{}".format(head_index)
        for domain in domains:
            for offset in range(2):
                train.append(
                    (
                        head,
                        "/{}/relation_{}".format(domain, offset),
                        "tail_{}_{}".format(domain, (head_index + offset) % 5),
                    )
                )
    train.extend(
        [
            ("shared", "/film/works", "tail_film"),
            ("shared", "/sports/plays", "tail_sports"),
            ("shared", "/user/example/custom", "tail_user"),
        ]
    )
    return build_knowledge_graph_dataset(
        "graph-semantic-test",
        train,
        [("valid", "/film/works", "tail_film")],
        [("test", "/sports/plays", "tail_sports")],
    )


class GraphSemanticPartitionTest(unittest.TestCase):
    """检查图语义策略的可复现性和不可拆分数据包。"""

    def setUp(self) -> None:
        """为每项测试创建相同的微型知识图谱。"""

        self.dataset = _build_dataset()

    def test_freebase_domain_extractor(self) -> None:
        """顶级域和user归并规则必须稳定。"""

        self.assertEqual(
            freebase_relation_domain("/film/film/genre"),
            "film",
        )
        self.assertEqual(
            freebase_relation_domain("/user/alice/default_domain/value"),
            "user",
        )
        with self.assertRaises(ValueError):
            freebase_relation_domain("")

    def test_partition_is_complete_deterministic_and_packet_exclusive(
        self,
    ) -> None:
        """分区必须完整、可复现且不拆分域-头实体包。"""

        first = partition_train_triples_by_semantic_domain_graph_local(
            self.dataset,
            client_count=3,
            seed=42,
            load_tolerance=0.30,
            search_restarts=2,
        )
        second = partition_train_triples_by_semantic_domain_graph_local(
            self.dataset,
            client_count=3,
            seed=42,
            load_tolerance=0.30,
            search_restarts=2,
        )
        self.assertEqual(first.partition_hash, second.partition_hash)
        self.assertEqual(
            first.partition_strategy,
            SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
        )
        self.assertEqual(
            sum(item.triple_count for item in first.partitions),
            int(self.dataset.train_triples.shape[0]),
        )
        domain_by_relation = relation_domains(self.dataset)
        owners = {}
        head_owners = {}
        for partition in first.partitions:
            for head_id, relation_id, _ in partition.train_triples.tolist():
                key = (
                    domain_by_relation[int(relation_id)],
                    int(head_id),
                )
                owners.setdefault(key, set()).add(partition.client_id)
                head_owners.setdefault(int(head_id), set()).add(
                    partition.client_id
                )
        self.assertTrue(all(len(value) == 1 for value in owners.values()))
        # 图语义策略不施加全局头实体互斥合同。
        self.assertTrue(any(len(value) > 1 for value in head_owners.values()))
        summary = first.summary()
        self.assertIn("triple_weighted_dominant_domain_purity", summary)
        self.assertIn("mean_relation_js_divergence", summary)
        self.assertIn("local_entity_reuse_ratio", summary)
        self.assertIn("mean_largest_component_entity_fraction", summary)

    def test_experiment_builds_configured_partition(self) -> None:
        """统一实验入口必须识别新策略及其公开配置字段。"""

        args = SimpleNamespace(
            partition_strategy=SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
            partition_domain_extractor="freebase_top_level",
            partition_load_tolerance=0.30,
            partition_search_restarts=2,
            client_num_in_total=3,
            random_seed=2024,
        )
        partition = build_federated_data(args, self.dataset)
        self.assertEqual(
            partition.partition_strategy,
            SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
        )
        bad_args = SimpleNamespace(**vars(args))
        bad_args.partition_domain_extractor = "unknown"
        with self.assertRaises(ValueError):
            build_federated_data(bad_args, self.dataset)


if __name__ == "__main__":
    unittest.main()
