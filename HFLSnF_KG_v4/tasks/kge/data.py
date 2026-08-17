"""FB15k-237三元组读取、全局编号和数据完整性校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch


TextTriple = Tuple[str, str, str]
IdTriple = Tuple[int, int, int]


@dataclass(frozen=True)
class KnowledgeGraphDataset:
    """保存统一实体关系编号下的训练、验证和测试三元组。"""

    dataset_name: str
    train_triples: torch.Tensor
    valid_triples: torch.Tensor
    test_triples: torch.Tensor
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    all_true_triples: Set[IdTriple]

    def __post_init__(self) -> None:
        """校验三元组张量形状、编号范围和训练集非空约束。"""

        if not str(self.dataset_name).strip():
            raise ValueError("知识图谱数据集名称不能为空")
        for split_name, triples in (
            ("train", self.train_triples),
            ("valid", self.valid_triples),
            ("test", self.test_triples),
        ):
            if triples.dtype != torch.long:
                raise TypeError("{}三元组必须使用torch.long".format(split_name))
            if triples.ndim != 2 or triples.shape[1] != 3:
                raise ValueError(
                    "{}三元组形状必须为[N, 3]，实际为{}".format(
                        split_name, tuple(triples.shape)
                    )
                )
        if int(self.train_triples.shape[0]) <= 0:
            raise ValueError("训练三元组不能为空")
        if len(self.entity_to_id) <= 1:
            raise ValueError("知识图谱至少需要两个实体")
        if len(self.relation_to_id) <= 0:
            raise ValueError("知识图谱至少需要一个关系")
        for triples in (
            self.train_triples,
            self.valid_triples,
            self.test_triples,
        ):
            if int(triples.numel()) <= 0:
                continue
            if int(triples[:, (0, 2)].min().item()) < 0:
                raise ValueError("实体编号不能为负数")
            if int(triples[:, (0, 2)].max().item()) >= self.num_entities:
                raise ValueError("三元组包含越界实体编号")
            if int(triples[:, 1].min().item()) < 0:
                raise ValueError("关系编号不能为负数")
            if int(triples[:, 1].max().item()) >= self.num_relations:
                raise ValueError("三元组包含越界关系编号")

    @property
    def num_entities(self) -> int:
        """返回全局实体数量。"""

        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        """返回全局关系数量。"""

        return len(self.relation_to_id)

    def summary(self) -> Dict[str, object]:
        """返回适合写入JSON的数据集摘要。"""

        return {
            "dataset": self.dataset_name,
            "entity_count": self.num_entities,
            "relation_count": self.num_relations,
            "train_triple_count": int(self.train_triples.shape[0]),
            "valid_triple_count": int(self.valid_triples.shape[0]),
            "test_triple_count": int(self.test_triples.shape[0]),
            "all_true_triple_count": len(self.all_true_triples),
        }


def _read_text_triples(path: Path) -> List[TextTriple]:
    """读取以制表符或空白分隔的三列知识图谱文本文件。"""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("找不到知识图谱数据文件：{}".format(path))
    triples: List[TextTriple] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split()
            if len(columns) != 3:
                raise ValueError(
                    "{}第{}行必须包含头实体、关系和尾实体三列".format(
                        path, line_number
                    )
                )
            triple = (columns[0], columns[1], columns[2])
            if triple in seen:
                raise ValueError(
                    "{}包含重复三元组：{}".format(path, triple)
                )
            seen.add(triple)
            triples.append(triple)
    if not triples:
        raise ValueError("知识图谱数据文件不能为空：{}".format(path))
    return triples


def _assign_id(mapping: Dict[str, int], value: str) -> int:
    """按首次出现顺序为实体或关系分配稳定的全局编号。"""

    if value not in mapping:
        mapping[value] = len(mapping)
    return mapping[value]


def _encode_triples(
    triples: Sequence[TextTriple],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
) -> torch.Tensor:
    """使用统一映射将文本三元组编码为长整型张量。"""

    encoded = [
        (
            _assign_id(entity_to_id, head),
            _assign_id(relation_to_id, relation),
            _assign_id(entity_to_id, tail),
        )
        for head, relation, tail in triples
    ]
    return torch.tensor(encoded, dtype=torch.long)


def build_knowledge_graph_dataset(
    dataset_name: str,
    train_triples: Sequence[TextTriple],
    valid_triples: Sequence[TextTriple],
    test_triples: Sequence[TextTriple],
) -> KnowledgeGraphDataset:
    """从三个文本三元组序列构造统一编号的知识图谱数据集。"""

    if not train_triples:
        raise ValueError("训练三元组不能为空")
    entity_to_id: Dict[str, int] = {}
    relation_to_id: Dict[str, int] = {}
    encoded_train = _encode_triples(
        train_triples, entity_to_id, relation_to_id
    )
    encoded_valid = _encode_triples(
        valid_triples, entity_to_id, relation_to_id
    )
    encoded_test = _encode_triples(
        test_triples, entity_to_id, relation_to_id
    )
    split_sets = [
        set(map(tuple, tensor.tolist()))
        for tensor in (encoded_train, encoded_valid, encoded_test)
    ]
    if split_sets[0].intersection(split_sets[1]):
        raise ValueError("训练集与验证集包含重复三元组")
    if split_sets[0].intersection(split_sets[2]):
        raise ValueError("训练集与测试集包含重复三元组")
    if split_sets[1].intersection(split_sets[2]):
        raise ValueError("验证集与测试集包含重复三元组")
    all_true = set().union(*split_sets)
    return KnowledgeGraphDataset(
        dataset_name=str(dataset_name).strip(),
        train_triples=encoded_train,
        valid_triples=encoded_valid,
        test_triples=encoded_test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        all_true_triples=all_true,
    )


def load_fb15k237(data_dir: Path) -> KnowledgeGraphDataset:
    """从标准train.txt、valid.txt和test.txt目录读取FB15k-237。"""

    data_dir = Path(data_dir).expanduser().resolve()
    train_triples = _read_text_triples(data_dir / "train.txt")
    valid_triples = _read_text_triples(data_dir / "valid.txt")
    test_triples = _read_text_triples(data_dir / "test.txt")
    return build_knowledge_graph_dataset(
        "fb15k-237", train_triples, valid_triples, test_triples
    )


def build_synthetic_knowledge_graph() -> KnowledgeGraphDataset:
    """构造无需下载数据的微型知识图谱，用于CPU冒烟和单元测试。"""

    train = [
        ("agent_a", "knows", "agent_b"),
        ("agent_b", "knows", "agent_c"),
        ("agent_c", "knows", "agent_d"),
        ("agent_d", "knows", "agent_e"),
        ("agent_a", "located_in", "region_x"),
        ("agent_b", "located_in", "region_x"),
        ("agent_c", "located_in", "region_y"),
        ("agent_d", "located_in", "region_y"),
    ]
    valid = [
        ("agent_e", "knows", "agent_a"),
        ("agent_e", "located_in", "region_x"),
    ]
    test = [
        ("agent_b", "knows", "agent_d"),
        ("agent_a", "knows", "agent_c"),
    ]
    return build_knowledge_graph_dataset(
        "synthetic-kg", train, valid, test
    )
