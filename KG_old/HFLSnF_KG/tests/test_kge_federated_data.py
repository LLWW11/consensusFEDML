"""知识图谱头实体均衡联邦划分测试。"""

from __future__ import annotations

import unittest

import torch

from HFLSnF_KG.tasks.kge import (
    build_synthetic_knowledge_graph,
    partition_train_triples_by_head,
)


class FederatedKnowledgeGraphDataTest(unittest.TestCase):
    """验证客户端分区覆盖性、头实体归属和可复现性。"""

    def test_partition_has_no_duplicates_or_omissions(self) -> None:
        """验证所有训练三元组被互斥且完整地分配给客户端。"""

        dataset = build_synthetic_knowledge_graph()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=3, seed=42
        )
        combined = torch.cat(
            [
                partition.train_triples
                for partition in federated_data.partitions
            ],
            dim=0,
        )
        source_rows = sorted(map(tuple, dataset.train_triples.tolist()))
        combined_rows = list(map(tuple, combined.tolist()))
        self.assertEqual(sorted(combined_rows), source_rows)
        self.assertEqual(len(combined_rows), len(set(combined_rows)))
        self.assertTrue(
            all(
                partition.triple_count > 0
                for partition in federated_data.partitions
            )
        )

    def test_each_head_belongs_to_exactly_one_client(self) -> None:
        """验证同一头实体的全部事实不会跨知识客户端。"""

        dataset = build_synthetic_knowledge_graph()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=3, seed=42
        )
        head_owners = {}
        for partition in federated_data.partitions:
            for head_id in partition.head_entity_ids.tolist():
                self.assertNotIn(head_id, head_owners)
                head_owners[int(head_id)] = int(partition.client_id)
            for row in partition.train_triples.tolist():
                self.assertEqual(
                    head_owners[int(row[0])], int(partition.client_id)
                )

    def test_partition_hash_is_reproducible_and_load_is_balanced(self) -> None:
        """验证固定种子划分指纹稳定且贪心负载差受最大头实体组约束。"""

        dataset = build_synthetic_knowledge_graph()
        first = partition_train_triples_by_head(
            dataset, client_count=3, seed=19
        )
        second = partition_train_triples_by_head(
            dataset, client_count=3, seed=19
        )
        self.assertEqual(first.partition_hash, second.partition_hash)
        first_states = [
            partition.train_triples
            for partition in first.partitions
        ]
        second_states = [
            partition.train_triples
            for partition in second.partitions
        ]
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(first_states, second_states)
            )
        )
        head_counts = {}
        for head_id in dataset.train_triples[:, 0].tolist():
            head_counts[int(head_id)] = head_counts.get(int(head_id), 0) + 1
        loads = [
            partition.triple_count for partition in first.partitions
        ]
        self.assertLessEqual(
            max(loads) - min(loads), max(head_counts.values())
        )


if __name__ == "__main__":
    unittest.main()
