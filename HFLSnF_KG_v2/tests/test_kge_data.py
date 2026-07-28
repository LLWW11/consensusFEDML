"""FB15k-237文本读取与统一编号测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from HFLSnF_KG_v2.tasks.kge import (
    build_synthetic_knowledge_graph,
    load_fb15k237,
)


class KnowledgeGraphDataTest(unittest.TestCase):
    """验证知识图谱数据读取、去重和全局映射。"""

    @staticmethod
    def _write_split(path: Path, lines) -> None:
        """向临时数据集写入一个UTF-8三元组划分。"""

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_load_fb15k237_builds_shared_global_ids(self) -> None:
        """验证三个划分共享实体关系映射且真三元组完整。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_split(
                root / "train.txt",
                ["a\tr\tb", "b\tr\tc"],
            )
            self._write_split(root / "valid.txt", ["c\ts\ta"])
            self._write_split(root / "test.txt", ["d\tr\ta"])
            dataset = load_fb15k237(root)

        self.assertEqual(dataset.num_entities, 4)
        self.assertEqual(dataset.num_relations, 2)
        self.assertEqual(len(dataset.all_true_triples), 4)
        self.assertEqual(dataset.entity_to_id["a"], 0)
        self.assertEqual(dataset.entity_to_id["d"], 3)

    def test_duplicate_triple_is_rejected(self) -> None:
        """验证同一文件中的重复三元组会被明确拒绝。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_split(
                root / "train.txt",
                ["a r b", "a r b"],
            )
            self._write_split(root / "valid.txt", ["b r c"])
            self._write_split(root / "test.txt", ["c r a"])
            with self.assertRaisesRegex(ValueError, "重复三元组"):
                load_fb15k237(root)

    def test_synthetic_dataset_is_self_contained(self) -> None:
        """验证内置合成图可以独立提供训练、验证和测试数据。"""

        dataset = build_synthetic_knowledge_graph()
        self.assertGreater(dataset.train_triples.shape[0], 0)
        self.assertGreater(dataset.valid_triples.shape[0], 0)
        self.assertGreater(dataset.test_triples.shape[0], 0)
        self.assertEqual(
            len(dataset.all_true_triples),
            int(
                dataset.train_triples.shape[0]
                + dataset.valid_triples.shape[0]
                + dataset.test_triples.shape[0]
            ),
        )


if __name__ == "__main__":
    unittest.main()
