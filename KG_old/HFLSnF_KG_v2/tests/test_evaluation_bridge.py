"""无需重训评估桥接的数据、模型和排名协议测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from HFLSnF_KG_v2.run_evaluation_bridge import (
    EvaluationBridgeSettings,
    resolve_bridge_device,
)
from HFLSnF_KG_v2.tasks.kge.data import (
    KnowledgeGraphDataset,
    build_knowledge_graph_dataset,
    load_fb15k237,
)
from HFLSnF_KG_v2.tasks.kge.evaluation_bridge import (
    BatchedFilteredTransEEvaluator,
    TransEEmbeddingBundle,
    bootstrap_mrr_interval,
    bootstrap_paired_delta_interval,
    build_common_holdout,
    build_fede_data_bundle,
    evaluate_fede_original_protocol,
    evaluate_global_protocol,
    hash_triples,
    load_fede_data_bundle,
    load_fede_embedding_bundle,
    load_project_embedding_bundle,
)


def _split_payload(
    triples,
    global_to_local_relation,
) -> dict:
    """把全局编号三元组转换成FedE pickle使用的split字段。"""

    rows = list(triples)
    if rows:
        edge_index = np.asarray(
            [[row[0] for row in rows], [row[2] for row in rows]],
            dtype=np.int64,
        )
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    global_relations = np.asarray(
        [row[1] for row in rows], dtype=np.int64
    )
    local_relations = np.asarray(
        [global_to_local_relation[int(row[1])] for row in rows],
        dtype=np.int64,
    )
    return {
        "edge_index": edge_index.copy(),
        "edge_index_ori": edge_index,
        "edge_type": local_relations,
        "edge_type_ori": global_relations,
    }


def _encode_named_triples(
    dataset: KnowledgeGraphDataset,
    triples,
):
    """使用测试数据集的既有映射编码命名三元组。"""

    return [
        (
            dataset.entity_to_id[head],
            dataset.relation_to_id[relation],
            dataset.entity_to_id[tail],
        )
        for head, relation, tail in triples
    ]


def _build_synthetic_fixture():
    """构造包含跨split泄漏和关系私有映射的两客户端小图。"""

    standard_train = [
        ("a", "r0", "b"),
        ("c", "r1", "d"),
        ("e", "r0", "f"),
        ("g", "r1", "h"),
    ]
    standard_valid = [
        ("a", "r0", "c"),
        ("e", "r0", "b"),
        ("c", "r1", "a"),
    ]
    standard_test = [
        ("c", "r1", "b"),
        ("e", "r0", "d"),
    ]
    dataset = build_knowledge_graph_dataset(
        "bridge-synthetic",
        standard_train,
        standard_valid,
        standard_test,
    )
    client_zero = {
        "train": _split_payload(
            _encode_named_triples(
                dataset,
                [
                    ("a", "r0", "b"),
                    ("e", "r0", "f"),
                    ("e", "r0", "b"),
                    ("e", "r0", "d"),
                ],
            ),
            {dataset.relation_to_id["r0"]: 0},
        ),
        "valid": _split_payload(
            _encode_named_triples(
                dataset,
                [("a", "r0", "c"), ("e", "r0", "b")],
            ),
            {dataset.relation_to_id["r0"]: 0},
        ),
        "test": _split_payload(
            _encode_named_triples(dataset, [("e", "r0", "d")]),
            {dataset.relation_to_id["r0"]: 0},
        ),
    }
    client_one = {
        "train": _split_payload(
            _encode_named_triples(
                dataset,
                [
                    ("c", "r1", "d"),
                    ("g", "r1", "h"),
                    ("c", "r1", "a"),
                ],
            ),
            {dataset.relation_to_id["r1"]: 0},
        ),
        "valid": _split_payload(
            [],
            {dataset.relation_to_id["r1"]: 0},
        ),
        "test": _split_payload(
            _encode_named_triples(dataset, [("c", "r1", "b")]),
            {dataset.relation_to_id["r1"]: 0},
        ),
    }
    fede_data = build_fede_data_bundle(
        [client_zero, client_one],
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
    )
    return dataset, fede_data


def _build_one_dimensional_model(
    dataset: KnowledgeGraphDataset,
) -> TransEEmbeddingBundle:
    """按固定一维坐标构造具有人工黄金排名的TransE模型。"""

    values = {
        "a": 0.0,
        "b": 0.5,
        "c": 4.0,
        "d": 6.0,
        "e": 2.0,
        "f": 5.0,
        "g": 1.0,
        "h": 3.0,
    }
    entity_embeddings = torch.empty(
        (dataset.num_entities, 1), dtype=torch.float32
    )
    for name, entity_id in dataset.entity_to_id.items():
        entity_embeddings[int(entity_id), 0] = float(values[name])
    relation_embeddings = torch.zeros(
        (dataset.num_relations, 1), dtype=torch.float32
    )
    return TransEEmbeddingBundle(
        name="synthetic",
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        distance_norm=1,
        checkpoint_path=Path("<memory>"),
        checkpoint_sha256="synthetic",
        metadata={},
    )


class EvaluationBridgeCoreTest(unittest.TestCase):
    """验证公共留出集、检查点拼接和四种排名口径。"""

    def test_common_holdout_strictly_removes_cross_split_leakage(
        self,
    ) -> None:
        """验证严格公共验证和测试集只保留双方同名split事实。"""

        dataset, fede_data = _build_synthetic_fixture()
        holdout = build_common_holdout(dataset, fede_data)
        expected_valid = _encode_named_triples(
            dataset, [("a", "r0", "c")]
        )
        expected_test = _encode_named_triples(
            dataset, [("c", "r1", "b")]
        )
        self.assertEqual(list(holdout.valid_triples), expected_valid)
        self.assertEqual(list(holdout.test_triples), expected_test)
        self.assertEqual(len(holdout.valid_assignments), 1)
        self.assertEqual(len(holdout.test_assignments), 1)

    def test_relation_mapping_rebuilds_private_tables_without_average(
        self,
    ) -> None:
        """验证两个客户端的局部关系0被放回不同全局关系行。"""

        dataset, fede_data = _build_synthetic_fixture()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fede.best"
            torch.save(
                {
                    "ent_embed": torch.zeros(
                        dataset.num_entities, 1, requires_grad=True
                    ),
                    "rel_embed": [
                        torch.tensor([[2.0]], requires_grad=True),
                        torch.tensor([[7.0]], requires_grad=True),
                    ],
                },
                checkpoint,
            )
            bundle = load_fede_embedding_bundle(
                checkpoint, fede_data
            )
        r0 = dataset.relation_to_id["r0"]
        r1 = dataset.relation_to_id["r1"]
        self.assertEqual(float(bundle.relation_embeddings[r0, 0]), 2.0)
        self.assertEqual(float(bundle.relation_embeddings[r1, 0]), 7.0)
        self.assertFalse(bundle.entity_embeddings.requires_grad)

    def test_local_and_global_protocols_match_golden_ranks(self) -> None:
        """验证局部尾预测与全局头尾预测得到人工黄金排名。"""

        dataset, fede_data = _build_synthetic_fixture()
        model = _build_one_dimensional_model(dataset)
        target = _encode_named_triples(
            dataset, [("a", "r0", "c")]
        )[0]
        client = fede_data.clients[0]
        local_evaluator = BatchedFilteredTransEEvaluator(
            dataset.num_entities, client.all_true_triples
        )
        local_rank = local_evaluator.evaluate_legacy_tail(
            model,
            [target],
            allowed_entity_ids=client.train_entity_ids,
            query_batch_size=1,
            candidate_batch_size=3,
        )
        global_evaluator = BatchedFilteredTransEEvaluator(
            dataset.num_entities, fede_data.all_true_triples
        )
        global_tail = global_evaluator.evaluate_direction(
            model,
            [target],
            predict_head=False,
            query_batch_size=1,
            candidate_batch_size=3,
        )
        global_head = global_evaluator.evaluate_direction(
            model,
            [target],
            predict_head=True,
            query_batch_size=1,
            candidate_batch_size=2,
        )
        self.assertEqual(int(local_rank[0]), 3)
        self.assertEqual(int(global_tail[0]), 5)
        self.assertEqual(int(global_head[0]), 8)
        self.assertAlmostEqual(
            (1.0 / global_tail[0] + 1.0 / global_head[0]) / 2.0,
            0.1625,
        )

    def test_candidate_chunk_size_does_not_change_global_rank(
        self,
    ) -> None:
        """验证候选分块大小只影响速度和显存，不改变排名。"""

        dataset, fede_data = _build_synthetic_fixture()
        model = _build_one_dimensional_model(dataset)
        triples = list(dataset.valid_triples.tolist())
        evaluator = BatchedFilteredTransEEvaluator(
            dataset.num_entities, fede_data.all_true_triples
        )
        first = evaluator.evaluate_direction(
            model,
            triples,
            predict_head=False,
            query_batch_size=2,
            candidate_batch_size=1,
        )
        second = evaluator.evaluate_direction(
            model,
            triples,
            predict_head=False,
            query_batch_size=1,
            candidate_batch_size=5,
        )
        self.assertTrue(np.array_equal(first, second))

    def test_global_protocol_micro_averages_head_and_tail_queries(
        self,
    ) -> None:
        """验证双向MRR来自全部头尾查询的微平均。"""

        dataset, fede_data = _build_synthetic_fixture()
        model = _build_one_dimensional_model(dataset)
        triples = [
            tuple(int(value) for value in dataset.test_triples[0].tolist())
        ]
        summary, records, per_triple = evaluate_global_protocol(
            "E2",
            model,
            triples,
            fede_data.all_true_triples,
            device=torch.device("cpu"),
            query_batch_size=1,
            candidate_batch_size=2,
        )
        record_mrr = np.mean(
            [float(record["reciprocal_rank"]) for record in records]
        )
        self.assertAlmostEqual(
            summary["combined_metrics"]["mrr"], record_mrr
        )
        self.assertAlmostEqual(float(per_triple[0]), record_mrr)
        self.assertEqual(
            summary["combined_metrics"]["evaluated_query_count"], 2
        )

    def test_original_protocol_restores_target_outside_train_candidates(
        self,
    ) -> None:
        """验证目标实体即使未在本地训练出现也不会被候选掩码删除。"""

        dataset, fede_data = _build_synthetic_fixture()
        model = _build_one_dimensional_model(dataset)
        # 人工把common-valid事实作为该客户端的测试查询。
        target = _encode_named_triples(
            dataset, [("a", "r0", "c")]
        )[0]
        modified_test = ((target[0], target[1], target[2], 0, 0),)
        modified = type(fede_data)(
            source_path=fede_data.source_path,
            source_sha256=fede_data.source_sha256,
            num_entities=fede_data.num_entities,
            num_relations=fede_data.num_relations,
            clients=fede_data.clients,
            train_assignments=fede_data.train_assignments,
            valid_assignments=fede_data.valid_assignments,
            test_assignments=modified_test,
            all_true_triples=fede_data.all_true_triples,
            relation_mapping_hash=fede_data.relation_mapping_hash,
        )
        summary, records = evaluate_fede_original_protocol(
            model,
            modified,
            device=torch.device("cpu"),
            max_triples=0,
            seed=42,
            query_batch_size=1,
            candidate_batch_size=2,
        )
        self.assertEqual(records[0]["rank"], 3)
        self.assertTrue(np.isfinite(summary["metrics"]["mrr"]))

    def test_bootstrap_is_reproducible_and_paired(self) -> None:
        """验证固定种子的普通和配对bootstrap输出可复现。"""

        values = [0.1, 0.2, 0.3, 0.4]
        first = bootstrap_mrr_interval(values, 100, 7)
        second = bootstrap_mrr_interval(values, 100, 7)
        paired = bootstrap_paired_delta_interval(
            [0.2, 0.3, 0.4, 0.5], values, 100, 7
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(paired["mean_delta_mrr"], 0.1)

    def test_project_checkpoint_requires_distance_norm_metadata(
        self,
    ) -> None:
        """验证旧检查点缺少距离范数时不会静默猜测L1或L2。"""

        dataset, _ = _build_synthetic_fixture()
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            checkpoint = result_dir / "model_best.pt"
            state = {
                "model_state_dict": {
                    "entity_embeddings.weight": torch.zeros(
                        dataset.num_entities, 2
                    ),
                    "relation_embeddings.weight": torch.zeros(
                        dataset.num_relations, 2
                    ),
                },
                "entity_to_id": dict(dataset.entity_to_id),
                "relation_to_id": dict(dataset.relation_to_id),
            }
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "distance_norm"):
                load_project_embedding_bundle(
                    "missing-norm", checkpoint, dataset
                )
            with (result_dir / "config_snapshot.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "model": "transe",
                        "embedding_dim": 2,
                        "distance_norm": 2,
                    },
                    handle,
                )
            loaded = load_project_embedding_bundle(
                "with-norm", checkpoint, dataset
            )
            self.assertEqual(loaded.distance_norm, 2)


class EvaluationBridgeRepositoryTest(unittest.TestCase):
    """使用仓库真实Fed3数据验证公共集和黄金哈希。"""

    @classmethod
    def setUpClass(cls) -> None:
        """只加载一次22MB标准数据和Fed3 pickle，降低测试耗时。"""

        root = Path(__file__).resolve().parents[2]
        cls.dataset = load_fb15k237(
            root / "HFLSnF_KG_v2" / "data" / "FB15k-237"
        )
        cls.fede_data = load_fede_data_bundle(
            root
            / "1paperAbout"
            / "FedE-master"
            / "data"
            / "FB15k237-Fed3.pkl",
            cls.dataset.num_entities,
            cls.dataset.num_relations,
        )
        cls.holdout = build_common_holdout(
            cls.dataset, cls.fede_data
        )

    def test_real_universe_and_common_holdout_goldens(self) -> None:
        """验证Fed3全集和公共验证测试集数量及哈希未漂移。"""

        summary = self.holdout.summary()
        self.assertEqual(
            hash_triples(self.fede_data.all_true_triples),
            "31400f896391744700e4a78f93b7836f"
            "f6bfe8b3d142e7ed74eca48e5a127c59",
        )
        self.assertEqual(summary["common_valid_count"], 1743)
        self.assertEqual(
            summary["common_valid_hash"],
            "bb731da0f880fbddfec22dddc158515f"
            "f53c606a7979def90ec74cd207cc8a8c",
        )
        self.assertEqual(summary["common_test_count"], 2048)
        self.assertEqual(
            summary["common_test_hash"],
            "72359a1db962f6a2ec724c5237a58654"
            "123d7e8541eb7aecb542a5247f07e422",
        )
        self.assertEqual(
            self.fede_data.relation_mapping_hash,
            "a2f23c4544b8349cf818d1ca158bd721"
            "513a8d0e53f7792d85c53d2aceffad81",
        )

    def test_full_cuda_config_fails_before_data_on_cpu(self) -> None:
        """验证正式配置在无CUDA时快速报错而不静默回退CPU。"""

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "evaluation_bridge_full_cuda.yaml"
        )
        settings = EvaluationBridgeSettings.from_yaml(config_path)
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "没有检测到CUDA"):
                resolve_bridge_device(settings)


if __name__ == "__main__":
    unittest.main()
