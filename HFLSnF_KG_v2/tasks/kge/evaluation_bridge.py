"""FedE与当前TransE检查点之间的无需重训练评估桥接。"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import torch

from .data import IdTriple, KnowledgeGraphDataset


FedEAssignment = Tuple[int, int, int, int, int]


def sha256_file(path: Path) -> str:
    """分块计算本地文件的SHA-256，避免一次性把大文件读入内存。"""

    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_triples(triples: Iterable[IdTriple]) -> str:
    """按编号字典序计算三元组集合的稳定SHA-256。"""

    digest = hashlib.sha256()
    for head, relation, tail in sorted(
        {tuple(int(value) for value in row) for row in triples}
    ):
        digest.update(
            "{}\t{}\t{}\n".format(head, relation, tail).encode("utf-8")
        )
    return digest.hexdigest()


def hash_assignments(assignments: Iterable[FedEAssignment]) -> str:
    """计算带客户端及局部关系编号的稳定分配哈希。"""

    digest = hashlib.sha256()
    for row in sorted(
        {tuple(int(value) for value in item) for item in assignments}
    ):
        digest.update(
            "{}\t{}\t{}\t{}\t{}\n".format(*row).encode("utf-8")
        )
    return digest.hexdigest()


def hash_json(payload: object) -> str:
    """把JSON兼容对象规范化后计算稳定SHA-256。"""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _safe_torch_load(path: Path):
    """兼容PyTorch 1.13与新版weights_only参数加载可信本地检查点。"""

    path = Path(path).expanduser().resolve()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch 1.13尚未提供weights_only参数。
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # 旧检查点包含普通Python元数据；这里只允许加载用户指定的可信文件。
        return torch.load(path, map_location="cpu", weights_only=False)


def _triples_to_tensor(triples: Sequence[IdTriple]) -> torch.Tensor:
    """把编号三元组序列转换成形状为[N,3]的长整型张量。"""

    if not triples:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.tensor(list(triples), dtype=torch.long)


def _select_rows(
    rows: Sequence[Tuple[int, ...]], maximum: int, seed: int
) -> Tuple[Tuple[int, ...], ...]:
    """按固定种子选择有限行；maximum为0时保留全部行。"""

    canonical = tuple(sorted(tuple(int(value) for value in row) for row in rows))
    if int(maximum) <= 0 or int(maximum) >= len(canonical):
        return canonical
    rng = np.random.RandomState(int(seed))
    indices = np.sort(
        rng.choice(len(canonical), size=int(maximum), replace=False)
    )
    return tuple(canonical[int(index)] for index in indices)


@dataclass(frozen=True)
class FedEClientPartition:
    """保存一个FedE客户端的全局三元组和关系编号映射。"""

    client_id: int
    train_triples: Tuple[IdTriple, ...]
    valid_triples: Tuple[IdTriple, ...]
    test_triples: Tuple[IdTriple, ...]
    train_entity_ids: Tuple[int, ...]
    local_to_global_relations: Tuple[int, ...]

    @property
    def all_true_triples(self) -> Set[IdTriple]:
        """返回该客户端训练、验证和测试事实的去重并集。"""

        return set(self.train_triples).union(
            self.valid_triples, self.test_triples
        )

    def summary(self) -> Dict[str, object]:
        """返回便于写入JSON的客户端数据摘要。"""

        return {
            "client_id": int(self.client_id),
            "train_triple_count": len(self.train_triples),
            "valid_triple_count": len(self.valid_triples),
            "test_triple_count": len(self.test_triples),
            "train_candidate_entity_count": len(self.train_entity_ids),
            "local_relation_count": len(self.local_to_global_relations),
        }


@dataclass(frozen=True)
class FedEDataBundle:
    """保存FedE pickle经严格校验后的全局编号数据。"""

    source_path: Path
    source_sha256: str
    num_entities: int
    num_relations: int
    clients: Tuple[FedEClientPartition, ...]
    train_assignments: Tuple[FedEAssignment, ...]
    valid_assignments: Tuple[FedEAssignment, ...]
    test_assignments: Tuple[FedEAssignment, ...]
    all_true_triples: Set[IdTriple]
    relation_mapping_hash: str

    def assignments(self, split: str) -> Tuple[FedEAssignment, ...]:
        """按split名称返回带客户端归属的三元组。"""

        normalized = str(split).strip().lower()
        if normalized == "train":
            return self.train_assignments
        if normalized == "valid":
            return self.valid_assignments
        if normalized == "test":
            return self.test_assignments
        raise ValueError("FedE split必须是train、valid或test")

    def triples(self, split: str) -> Tuple[IdTriple, ...]:
        """按split名称返回去掉客户端信息的全局编号三元组。"""

        return tuple(
            (int(row[0]), int(row[1]), int(row[2]))
            for row in self.assignments(split)
        )

    def summary(self) -> Dict[str, object]:
        """返回FedE数据文件、分区和哈希摘要。"""

        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "client_count": len(self.clients),
            "entity_count": int(self.num_entities),
            "relation_count": int(self.num_relations),
            "train_assignment_count": len(self.train_assignments),
            "valid_assignment_count": len(self.valid_assignments),
            "test_assignment_count": len(self.test_assignments),
            "unique_universe_count": len(self.all_true_triples),
            "universe_hash": hash_triples(self.all_true_triples),
            "relation_mapping_hash": self.relation_mapping_hash,
            "clients": [client.summary() for client in self.clients],
        }


def _extract_fede_split(
    split_data: Mapping[str, object],
    split_name: str,
    client_id: int,
    num_entities: int,
    num_relations: int,
) -> Tuple[List[FedEAssignment], Dict[int, int]]:
    """从一个FedE客户端split提取全局事实和局部到全局关系映射。"""

    required = {
        "edge_index_ori",
        "edge_type",
        "edge_type_ori",
    }
    missing = required.difference(split_data)
    if missing:
        raise ValueError(
            "FedE客户端{}的{}缺少字段{}".format(
                client_id, split_name, sorted(missing)
            )
        )
    edge_index = np.asarray(split_data["edge_index_ori"])
    local_relations = np.asarray(split_data["edge_type"]).reshape(-1)
    global_relations = np.asarray(split_data["edge_type_ori"]).reshape(-1)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "FedE客户端{}的{} edge_index_ori形状必须为[2,N]".format(
                client_id, split_name
            )
        )
    count = int(edge_index.shape[1])
    if local_relations.size != count or global_relations.size != count:
        raise ValueError(
            "FedE客户端{}的{}实体边和关系数量不一致".format(
                client_id, split_name
            )
        )

    assignments: List[FedEAssignment] = []
    relation_mapping: Dict[int, int] = {}
    for index in range(count):
        head = int(edge_index[0, index])
        tail = int(edge_index[1, index])
        local_relation = int(local_relations[index])
        global_relation = int(global_relations[index])
        if min(head, tail) < 0 or max(head, tail) >= int(num_entities):
            raise ValueError("FedE三元组包含越界实体编号")
        if global_relation < 0 or global_relation >= int(num_relations):
            raise ValueError("FedE三元组包含越界全局关系编号")
        if local_relation < 0:
            raise ValueError("FedE局部关系编号不能为负数")
        previous = relation_mapping.setdefault(
            local_relation, global_relation
        )
        if previous != global_relation:
            raise ValueError(
                "FedE客户端{}的局部关系{}映射到多个全局关系".format(
                    client_id, local_relation
                )
            )
        assignments.append(
            (
                head,
                global_relation,
                tail,
                int(client_id),
                local_relation,
            )
        )
    return assignments, relation_mapping


def build_fede_data_bundle(
    raw_clients: Sequence[Mapping[str, object]],
    num_entities: int,
    num_relations: int,
    source_path: Optional[Path] = None,
    source_sha256: str = "",
) -> FedEDataBundle:
    """从已反序列化的FedE客户端列表构造严格全局数据对象。"""

    if not raw_clients:
        raise ValueError("FedE客户端列表不能为空")
    split_rows: Dict[str, List[FedEAssignment]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    clients: List[FedEClientPartition] = []
    global_relation_owner: Dict[int, Tuple[int, int]] = {}

    for client_id, client_data in enumerate(raw_clients):
        split_mappings: List[Dict[int, int]] = []
        client_rows: Dict[str, List[FedEAssignment]] = {}
        for split_name in ("train", "valid", "test"):
            if split_name not in client_data:
                raise ValueError(
                    "FedE客户端{}缺少{} split".format(
                        client_id, split_name
                    )
                )
            rows, relation_mapping = _extract_fede_split(
                client_data[split_name],
                split_name,
                client_id,
                num_entities,
                num_relations,
            )
            client_rows[split_name] = rows
            split_rows[split_name].extend(rows)
            split_mappings.append(relation_mapping)

        merged_mapping: Dict[int, int] = {}
        for mapping in split_mappings:
            for local_relation, global_relation in mapping.items():
                previous = merged_mapping.setdefault(
                    local_relation, global_relation
                )
                if previous != global_relation:
                    raise ValueError(
                        "FedE客户端{}跨split关系映射不一致".format(
                            client_id
                        )
                    )
        expected_local_ids = set(range(len(merged_mapping)))
        if set(merged_mapping) != expected_local_ids:
            raise ValueError(
                "FedE客户端{}局部关系编号必须从0连续开始".format(client_id)
            )
        local_to_global = tuple(
            int(merged_mapping[index])
            for index in range(len(merged_mapping))
        )
        for local_relation, global_relation in enumerate(local_to_global):
            owner = global_relation_owner.setdefault(
                global_relation, (int(client_id), int(local_relation))
            )
            if owner != (int(client_id), int(local_relation)):
                raise ValueError(
                    "FedE全局关系{}被多个客户端拥有".format(
                        global_relation
                    )
                )

        train_triples = tuple(
            (row[0], row[1], row[2]) for row in client_rows["train"]
        )
        valid_triples = tuple(
            (row[0], row[1], row[2]) for row in client_rows["valid"]
        )
        test_triples = tuple(
            (row[0], row[1], row[2]) for row in client_rows["test"]
        )
        train_entities = tuple(
            sorted(
                {
                    int(entity)
                    for head, _, tail in train_triples
                    for entity in (head, tail)
                }
            )
        )
        clients.append(
            FedEClientPartition(
                client_id=int(client_id),
                train_triples=train_triples,
                valid_triples=valid_triples,
                test_triples=test_triples,
                train_entity_ids=train_entities,
                local_to_global_relations=local_to_global,
            )
        )

    if set(global_relation_owner) != set(range(int(num_relations))):
        missing_relations = sorted(
            set(range(int(num_relations))).difference(global_relation_owner)
        )
        raise ValueError(
            "FedE全局关系没有完整覆盖，缺失{}".format(missing_relations)
        )

    # 同一个split中的事实必须唯一归属一个客户端，跨split旧数据允许重复。
    for split_name, rows in split_rows.items():
        triples = [(row[0], row[1], row[2]) for row in rows]
        if len(triples) != len(set(triples)):
            raise ValueError(
                "FedE {} split包含重复事实或跨客户端重复归属".format(
                    split_name
                )
            )

    canonical_rows = {
        split_name: tuple(sorted(rows))
        for split_name, rows in split_rows.items()
    }
    all_true = set()
    for rows in canonical_rows.values():
        all_true.update((row[0], row[1], row[2]) for row in rows)
    # 只保留按客户端顺序排列的映射数组，避免字段命名改变哈希。
    mapping_payload = [
        list(client.local_to_global_relations) for client in clients
    ]
    resolved_source = (
        Path(source_path).expanduser().resolve()
        if source_path is not None
        else Path("<memory>")
    )
    return FedEDataBundle(
        source_path=resolved_source,
        source_sha256=str(source_sha256),
        num_entities=int(num_entities),
        num_relations=int(num_relations),
        clients=tuple(clients),
        train_assignments=canonical_rows["train"],
        valid_assignments=canonical_rows["valid"],
        test_assignments=canonical_rows["test"],
        all_true_triples=all_true,
        relation_mapping_hash=hash_json(mapping_payload),
    )


def load_fede_data_bundle(
    path: Path, num_entities: int, num_relations: int
) -> FedEDataBundle:
    """从可信本地pickle读取FedE数据并转换为全局编号对象。"""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("找不到FedE数据文件：{}".format(path))
    with path.open("rb") as handle:
        raw_clients = pickle.load(handle)
    if not isinstance(raw_clients, list):
        raise TypeError("FedE数据文件顶层必须是客户端列表")
    return build_fede_data_bundle(
        raw_clients,
        num_entities=num_entities,
        num_relations=num_relations,
        source_path=path,
        source_sha256=sha256_file(path),
    )


@dataclass(frozen=True)
class CommonHoldout:
    """保存两种数据划分共同保留的验证和测试事实。"""

    valid_assignments: Tuple[FedEAssignment, ...]
    test_assignments: Tuple[FedEAssignment, ...]
    universe_hash: str

    @property
    def valid_triples(self) -> Tuple[IdTriple, ...]:
        """返回严格公共验证三元组。"""

        return tuple((row[0], row[1], row[2]) for row in self.valid_assignments)

    @property
    def test_triples(self) -> Tuple[IdTriple, ...]:
        """返回严格公共测试三元组。"""

        return tuple((row[0], row[1], row[2]) for row in self.test_assignments)

    def summary(self) -> Dict[str, object]:
        """返回公共留出集数量、哈希和客户端分布。"""

        valid_clients: DefaultDict[int, int] = defaultdict(int)
        test_clients: DefaultDict[int, int] = defaultdict(int)
        for row in self.valid_assignments:
            valid_clients[int(row[3])] += 1
        for row in self.test_assignments:
            test_clients[int(row[3])] += 1
        return {
            "universe_hash": self.universe_hash,
            "common_valid_count": len(self.valid_assignments),
            "common_valid_hash": hash_triples(self.valid_triples),
            "common_valid_assignment_hash": hash_assignments(
                self.valid_assignments
            ),
            "common_valid_count_by_client": {
                str(key): value
                for key, value in sorted(valid_clients.items())
            },
            "common_test_count": len(self.test_assignments),
            "common_test_hash": hash_triples(self.test_triples),
            "common_test_assignment_hash": hash_assignments(
                self.test_assignments
            ),
            "common_test_count_by_client": {
                str(key): value
                for key, value in sorted(test_clients.items())
            },
            "common_test_query_count": 2 * len(self.test_assignments),
        }


def build_common_holdout(
    dataset: KnowledgeGraphDataset, fede_data: FedEDataBundle
) -> CommonHoldout:
    """构造双方都未用于训练或验证的严格公共验证和测试集合。"""

    standard_train = set(map(tuple, dataset.train_triples.tolist()))
    standard_valid = set(map(tuple, dataset.valid_triples.tolist()))
    standard_test = set(map(tuple, dataset.test_triples.tolist()))
    standard_universe = set(dataset.all_true_triples)
    if standard_universe != fede_data.all_true_triples:
        missing_from_fede = len(
            standard_universe.difference(fede_data.all_true_triples)
        )
        missing_from_standard = len(
            fede_data.all_true_triples.difference(standard_universe)
        )
        raise ValueError(
            "当前数据与FedE三元组全集不一致：FedE缺{}条，当前数据缺{}条".format(
                missing_from_fede, missing_from_standard
            )
        )

    fede_train = set(fede_data.triples("train"))
    fede_valid = set(fede_data.triples("valid"))
    fede_test = set(fede_data.triples("test"))
    common_valid = (
        standard_valid.intersection(fede_valid)
        .difference(standard_train)
        .difference(standard_test)
        .difference(fede_train)
        .difference(fede_test)
    )
    common_test = (
        standard_test.intersection(fede_test)
        .difference(standard_train)
        .difference(standard_valid)
        .difference(fede_train)
        .difference(fede_valid)
    )
    valid_lookup = {
        (row[0], row[1], row[2]): row
        for row in fede_data.valid_assignments
    }
    test_lookup = {
        (row[0], row[1], row[2]): row
        for row in fede_data.test_assignments
    }
    valid_assignments = tuple(
        sorted(valid_lookup[triple] for triple in common_valid)
    )
    test_assignments = tuple(
        sorted(test_lookup[triple] for triple in common_test)
    )
    if not valid_assignments or not test_assignments:
        raise ValueError("严格公共验证集或测试集为空")
    return CommonHoldout(
        valid_assignments=valid_assignments,
        test_assignments=test_assignments,
        universe_hash=hash_triples(standard_universe),
    )


@dataclass
class TransEEmbeddingBundle:
    """保存无需训练即可评分的TransE实体与关系表。"""

    name: str
    entity_embeddings: torch.Tensor
    relation_embeddings: torch.Tensor
    distance_norm: int
    checkpoint_path: Path
    checkpoint_sha256: str
    metadata: Dict[str, object]

    def __post_init__(self) -> None:
        """校验两张嵌入表的形状、维数和距离范数。"""

        if self.entity_embeddings.ndim != 2:
            raise ValueError("实体嵌入必须是二维张量")
        if self.relation_embeddings.ndim != 2:
            raise ValueError("关系嵌入必须是二维张量")
        if self.entity_embeddings.shape[1] != self.relation_embeddings.shape[1]:
            raise ValueError("实体和关系嵌入维数必须一致")
        if int(self.distance_norm) not in {1, 2}:
            raise ValueError("TransE距离范数必须是1或2")

    def to(self, device: torch.device) -> "TransEEmbeddingBundle":
        """返回把推理张量移动到指定设备后的独立对象。"""

        device = torch.device(device)
        return TransEEmbeddingBundle(
            name=self.name,
            entity_embeddings=self.entity_embeddings.detach().to(device),
            relation_embeddings=self.relation_embeddings.detach().to(device),
            distance_norm=int(self.distance_norm),
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
            metadata=dict(self.metadata),
        )

    def summary(self) -> Dict[str, object]:
        """返回模型名称、嵌入形状、距离和检查点哈希。"""

        return {
            "name": self.name,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "entity_shape": list(self.entity_embeddings.shape),
            "relation_shape": list(self.relation_embeddings.shape),
            "embedding_dim": int(self.entity_embeddings.shape[1]),
            "distance_norm": int(self.distance_norm),
            "metadata": dict(self.metadata),
        }


def load_fede_embedding_bundle(
    checkpoint_path: Path,
    fede_data: FedEDataBundle,
    name: str = "fede",
) -> TransEEmbeddingBundle:
    """加载FedE检查点并按局部到全局映射拼接完整关系表。"""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "找不到FedE检查点：{}".format(checkpoint_path)
        )
    state = _safe_torch_load(checkpoint_path)
    if not isinstance(state, dict) or set(state) != {
        "ent_embed",
        "rel_embed",
    }:
        raise ValueError("FedE检查点必须且只能包含ent_embed和rel_embed")
    entity_embeddings = state["ent_embed"]
    local_relation_tables = state["rel_embed"]
    if not torch.is_tensor(entity_embeddings):
        raise TypeError("FedE ent_embed必须是张量")
    if not isinstance(local_relation_tables, list):
        raise TypeError("FedE rel_embed必须是客户端关系表列表")
    if len(local_relation_tables) != len(fede_data.clients):
        raise ValueError("FedE关系表数量与客户端数量不一致")
    entity_embeddings = entity_embeddings.detach().cpu().clone()
    if int(entity_embeddings.shape[0]) != int(fede_data.num_entities):
        raise ValueError("FedE实体表行数与数据实体数不一致")

    embedding_dim = int(entity_embeddings.shape[1])
    global_relations = torch.empty(
        (fede_data.num_relations, embedding_dim),
        dtype=entity_embeddings.dtype,
    )
    filled = torch.zeros(fede_data.num_relations, dtype=torch.bool)
    for client, relation_table in zip(
        fede_data.clients, local_relation_tables
    ):
        if not torch.is_tensor(relation_table):
            raise TypeError("FedE客户端关系表必须是张量")
        relation_table = relation_table.detach().cpu()
        if tuple(relation_table.shape) != (
            len(client.local_to_global_relations),
            embedding_dim,
        ):
            raise ValueError(
                "FedE客户端{}关系表形状与映射不一致".format(
                    client.client_id
                )
            )
        for local_id, global_id in enumerate(
            client.local_to_global_relations
        ):
            if bool(filled[int(global_id)]):
                raise ValueError("FedE全局关系行被重复写入")
            global_relations[int(global_id)].copy_(
                relation_table[int(local_id)]
            )
            filled[int(global_id)] = True
    if not bool(filled.all()):
        raise ValueError("FedE全局关系表存在未填充行")

    # 原FedE从不执行单位球投影，因此这里只复制，绝不额外归一化。
    return TransEEmbeddingBundle(
        name=str(name),
        entity_embeddings=entity_embeddings,
        relation_embeddings=global_relations,
        distance_norm=1,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        metadata={
            "format": "fede_private_relation_tables",
            "relation_mapping_hash": fede_data.relation_mapping_hash,
        },
    )


def resolve_project_checkpoint(path_or_directory: Path) -> Path:
    """把结果目录或模型文件统一解析成model_best.pt路径。"""

    resolved = Path(path_or_directory).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "model_best.pt"
    if not resolved.is_file():
        raise FileNotFoundError("找不到TransE检查点：{}".format(resolved))
    return resolved


def _load_json_file(path: Path) -> Dict[str, object]:
    """读取UTF-8 JSON对象并校验顶层类型。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("{}顶层必须是JSON对象".format(path))
    return payload


def load_project_embedding_bundle(
    name: str,
    path_or_directory: Path,
    dataset: KnowledgeGraphDataset,
    distance_norm_override: Optional[int] = None,
) -> TransEEmbeddingBundle:
    """加载V1或V2标准TransE检查点并严格核对映射与配置。"""

    checkpoint_path = resolve_project_checkpoint(path_or_directory)
    state = _safe_torch_load(checkpoint_path)
    if not isinstance(state, dict):
        raise TypeError("当前项目TransE检查点顶层必须是字典")
    required = {
        "model_state_dict",
        "entity_to_id",
        "relation_to_id",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError("TransE检查点缺少字段{}".format(sorted(missing)))
    if dict(state["entity_to_id"]) != dict(dataset.entity_to_id):
        raise ValueError("{}实体编号映射与公共数据不一致".format(name))
    if dict(state["relation_to_id"]) != dict(dataset.relation_to_id):
        raise ValueError("{}关系编号映射与公共数据不一致".format(name))

    model_state = state["model_state_dict"]
    expected_keys = {
        "entity_embeddings.weight",
        "relation_embeddings.weight",
    }
    if not isinstance(model_state, dict) or set(model_state) != expected_keys:
        raise ValueError("{}不是标准TransE模型状态".format(name))
    entity_embeddings = model_state[
        "entity_embeddings.weight"
    ].detach().cpu().clone()
    relation_embeddings = model_state[
        "relation_embeddings.weight"
    ].detach().cpu().clone()
    if int(entity_embeddings.shape[0]) != dataset.num_entities:
        raise ValueError("{}实体表行数与公共数据不一致".format(name))
    if int(relation_embeddings.shape[0]) != dataset.num_relations:
        raise ValueError("{}关系表行数与公共数据不一致".format(name))

    config_path = checkpoint_path.parent / "config_snapshot.json"
    config: Dict[str, object] = {}
    if config_path.is_file():
        config = _load_json_file(config_path)
    if distance_norm_override is None:
        if "distance_norm" not in config:
            raise ValueError(
                "{}检查点没有保存distance_norm，且同目录配置也缺失；"
                "请显式提供distance_norm".format(name)
            )
        distance_norm = int(config["distance_norm"])
    else:
        distance_norm = int(distance_norm_override)
    if "embedding_dim" in config and int(config["embedding_dim"]) != int(
        entity_embeddings.shape[1]
    ):
        raise ValueError("{}配置维数与检查点张量不一致".format(name))
    if "model" in config and str(config["model"]).strip().lower() != "transe":
        raise ValueError("{}配置声明的模型不是TransE".format(name))

    for filename, expected_mapping in (
        ("entity2id.json", dataset.entity_to_id),
        ("relation2id.json", dataset.relation_to_id),
    ):
        mapping_path = checkpoint_path.parent / filename
        if mapping_path.is_file():
            actual_mapping = _load_json_file(mapping_path)
            if actual_mapping != dict(expected_mapping):
                raise ValueError(
                    "{}中的{}与公共数据映射不一致".format(name, filename)
                )
    return TransEEmbeddingBundle(
        name=str(name),
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        distance_norm=distance_norm,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        metadata={
            "format": "hflsnf_transe_state_dict",
            "config_path": str(config_path) if config_path.is_file() else None,
            "run_name": config.get("run_name"),
        },
    )


def metrics_from_ranks(ranks: Sequence[int]) -> Dict[str, float]:
    """从一维正整数排名计算MRR、MR和Hits指标。"""

    rank_array = np.asarray(ranks, dtype=np.float64)
    if rank_array.ndim != 1 or rank_array.size <= 0:
        raise ValueError("排名数组必须是一维非空序列")
    if not np.isfinite(rank_array).all() or np.any(rank_array < 1):
        raise ValueError("排名必须是有限正数")
    return {
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(np.mean(rank_array)),
        "hits_at_1": float(np.mean(rank_array <= 1.0)),
        "hits_at_3": float(np.mean(rank_array <= 3.0)),
        "hits_at_5": float(np.mean(rank_array <= 5.0)),
        "hits_at_10": float(np.mean(rank_array <= 10.0)),
        "evaluated_query_count": int(rank_array.size),
    }


class BatchedFilteredTransEEvaluator:
    """按查询批次和候选分块执行全局或局部TransE排名。"""

    def __init__(
        self,
        num_entities: int,
        all_true_triples: Iterable[IdTriple],
    ):
        """建立头尾filtered索引并记录候选实体总数。"""

        if int(num_entities) <= 1:
            raise ValueError("评估至少需要两个候选实体")
        self.num_entities = int(num_entities)
        self.true_tails: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(
            set
        )
        self.true_heads: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(
            set
        )
        for head, relation, tail in all_true_triples:
            self.true_tails[(int(head), int(relation))].add(int(tail))
            self.true_heads[(int(relation), int(tail))].add(int(head))

    @staticmethod
    def _target_distances(
        bundle: TransEEmbeddingBundle, triples: torch.Tensor
    ) -> torch.Tensor:
        """计算一批目标三元组自身的TransE距离。"""

        heads = bundle.entity_embeddings.index_select(0, triples[:, 0])
        relations = bundle.relation_embeddings.index_select(
            0, triples[:, 1]
        )
        tails = bundle.entity_embeddings.index_select(0, triples[:, 2])
        return torch.linalg.vector_norm(
            heads + relations - tails,
            ord=int(bundle.distance_norm),
            dim=1,
        )

    @staticmethod
    def _candidate_distances(
        bundle: TransEEmbeddingBundle,
        triples: torch.Tensor,
        candidate_ids: torch.Tensor,
        predict_head: bool,
    ) -> torch.Tensor:
        """计算一个查询批次相对一块候选实体的距离矩阵。"""

        candidates = bundle.entity_embeddings.index_select(0, candidate_ids)
        relations = bundle.relation_embeddings.index_select(
            0, triples[:, 1]
        )
        if predict_head:
            tails = bundle.entity_embeddings.index_select(0, triples[:, 2])
            residual = (
                candidates.unsqueeze(0)
                + relations.unsqueeze(1)
                - tails.unsqueeze(1)
            )
        else:
            heads = bundle.entity_embeddings.index_select(0, triples[:, 0])
            residual = (
                heads.unsqueeze(1)
                + relations.unsqueeze(1)
                - candidates.unsqueeze(0)
            )
        return torch.linalg.vector_norm(
            residual,
            ord=int(bundle.distance_norm),
            dim=2,
        )

    def _invalid_candidate_mask(
        self,
        triples: torch.Tensor,
        predict_head: bool,
        start: int,
        stop: int,
        allowed_entities: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """为一块候选实体生成局部候选和filtered联合屏蔽矩阵。"""

        device = triples.device
        batch_size = int(triples.shape[0])
        width = int(stop - start)
        invalid = torch.zeros(
            (batch_size, width), dtype=torch.bool, device=device
        )
        if allowed_entities is not None:
            invalid |= ~allowed_entities[start:stop].view(1, -1)
        rows = triples.detach().cpu().tolist()
        for row_index, row in enumerate(rows):
            head, relation, tail = (int(value) for value in row)
            target = head if predict_head else tail
            filtered = (
                self.true_heads[(relation, tail)]
                if predict_head
                else self.true_tails[(head, relation)]
            )
            local_indices = [
                int(entity_id) - int(start)
                for entity_id in filtered
                if int(start) <= int(entity_id) < int(stop)
                and int(entity_id) != int(target)
            ]
            if local_indices:
                invalid[row_index, local_indices] = True
            # 即使目标不属于本地训练实体，也必须恢复目标候选。
            if int(start) <= int(target) < int(stop):
                invalid[row_index, int(target) - int(start)] = False
        return invalid

    def evaluate_direction(
        self,
        bundle: TransEEmbeddingBundle,
        triples: Sequence[IdTriple],
        predict_head: bool,
        query_batch_size: int,
        candidate_batch_size: int,
        allowed_entity_ids: Optional[Sequence[int]] = None,
        progress_label: str = "",
        progress_every: int = 0,
    ) -> np.ndarray:
        """使用乐观并列规则计算一个预测方向的filtered排名。"""

        if int(query_batch_size) <= 0 or int(candidate_batch_size) <= 0:
            raise ValueError("查询批次和候选批次必须大于0")
        if not triples:
            raise ValueError("评估三元组不能为空")
        device = bundle.entity_embeddings.device
        triple_tensor = _triples_to_tensor(triples).to(device)
        allowed_entities = None
        if allowed_entity_ids is not None:
            allowed_entities = torch.zeros(
                self.num_entities, dtype=torch.bool, device=device
            )
            allowed_entities[
                torch.tensor(
                    list(allowed_entity_ids), dtype=torch.long, device=device
                )
            ] = True

        rank_chunks: List[torch.Tensor] = []
        with torch.inference_mode():
            for query_start in range(
                0, int(triple_tensor.shape[0]), int(query_batch_size)
            ):
                query_stop = min(
                    query_start + int(query_batch_size),
                    int(triple_tensor.shape[0]),
                )
                query = triple_tensor[query_start:query_stop]
                target_distances = self._target_distances(bundle, query)
                ranks = torch.ones(
                    int(query.shape[0]), dtype=torch.long, device=device
                )
                for candidate_start in range(
                    0, self.num_entities, int(candidate_batch_size)
                ):
                    candidate_stop = min(
                        candidate_start + int(candidate_batch_size),
                        self.num_entities,
                    )
                    candidate_ids = torch.arange(
                        candidate_start,
                        candidate_stop,
                        dtype=torch.long,
                        device=device,
                    )
                    distances = self._candidate_distances(
                        bundle, query, candidate_ids, bool(predict_head)
                    )
                    invalid = self._invalid_candidate_mask(
                        query,
                        bool(predict_head),
                        candidate_start,
                        candidate_stop,
                        allowed_entities,
                    )
                    better = distances < target_distances.view(-1, 1)
                    better &= ~invalid
                    ranks += better.sum(dim=1)
                rank_chunks.append(ranks.detach().cpu())
                if (
                    int(progress_every) > 0
                    and (
                        query_start // int(progress_every)
                        != (query_stop - 1) // int(progress_every)
                        or query_stop == int(triple_tensor.shape[0])
                    )
                ):
                    print(
                        "{}：已评估 {}/{} 条三元组".format(
                            progress_label or "排名评估",
                            query_stop,
                            int(triple_tensor.shape[0]),
                        ),
                        flush=True,
                    )
        return torch.cat(rank_chunks).numpy().astype(np.int64)

    def evaluate_legacy_tail(
        self,
        bundle: TransEEmbeddingBundle,
        triples: Sequence[IdTriple],
        allowed_entity_ids: Sequence[int],
        query_batch_size: int,
        candidate_batch_size: int,
        progress_label: str = "",
        progress_every: int = 0,
    ) -> np.ndarray:
        """按原FedE双重argsort规则计算局部候选尾预测排名。"""

        if int(query_batch_size) <= 0 or int(candidate_batch_size) <= 0:
            raise ValueError("查询批次和候选批次必须大于0")
        if not triples:
            raise ValueError("FedE E0评估三元组不能为空")
        device = bundle.entity_embeddings.device
        triple_tensor = _triples_to_tensor(triples).to(device)
        allowed_entities = torch.zeros(
            self.num_entities, dtype=torch.bool, device=device
        )
        allowed_entities[
            torch.tensor(
                list(allowed_entity_ids), dtype=torch.long, device=device
            )
        ] = True
        rank_chunks: List[torch.Tensor] = []

        with torch.inference_mode():
            for query_start in range(
                0, int(triple_tensor.shape[0]), int(query_batch_size)
            ):
                query_stop = min(
                    query_start + int(query_batch_size),
                    int(triple_tensor.shape[0]),
                )
                query = triple_tensor[query_start:query_stop]
                score_chunks: List[torch.Tensor] = []
                for candidate_start in range(
                    0, self.num_entities, int(candidate_batch_size)
                ):
                    candidate_stop = min(
                        candidate_start + int(candidate_batch_size),
                        self.num_entities,
                    )
                    candidate_ids = torch.arange(
                        candidate_start,
                        candidate_stop,
                        dtype=torch.long,
                        device=device,
                    )
                    # gamma对同一查询的候选排序是常数，这里直接使用负距离。
                    scores = -self._candidate_distances(
                        bundle, query, candidate_ids, False
                    )
                    invalid = self._invalid_candidate_mask(
                        query,
                        False,
                        candidate_start,
                        candidate_stop,
                        allowed_entities,
                    )
                    scores = scores.masked_fill(invalid, -10000000.0)
                    score_chunks.append(scores)
                all_scores = torch.cat(score_chunks, dim=1)
                target_ids = query[:, 2]
                # 屏蔽后恢复目标分数，与原FedE评估代码保持一致。
                target_distances = self._target_distances(bundle, query)
                row_ids = torch.arange(
                    int(query.shape[0]), dtype=torch.long, device=device
                )
                all_scores[row_ids, target_ids] = -target_distances
                order = torch.argsort(all_scores, dim=1, descending=True)
                ordinal = torch.argsort(order, dim=1, descending=False)
                ranks = 1 + ordinal[row_ids, target_ids]
                rank_chunks.append(ranks.detach().cpu())
                if (
                    int(progress_every) > 0
                    and (
                        query_start // int(progress_every)
                        != (query_stop - 1) // int(progress_every)
                        or query_stop == int(triple_tensor.shape[0])
                    )
                ):
                    print(
                        "{}：已评估 {}/{} 条三元组".format(
                            progress_label or "FedE原口径",
                            query_stop,
                            int(triple_tensor.shape[0]),
                        ),
                        flush=True,
                    )
        return torch.cat(rank_chunks).numpy().astype(np.int64)


def evaluate_fede_original_protocol(
    bundle: TransEEmbeddingBundle,
    fede_data: FedEDataBundle,
    device: torch.device,
    max_triples: int,
    seed: int,
    query_batch_size: int,
    candidate_batch_size: int,
    progress_every: int = 0,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """执行E0局部候选、局部filtered、仅尾预测的FedE原评估。"""

    selected = _select_rows(
        fede_data.test_assignments, int(max_triples), int(seed)
    )
    device_bundle = bundle.to(device)
    all_ranks: List[int] = []
    records: List[Dict[str, object]] = []
    client_metrics: Dict[str, Dict[str, float]] = {}
    for client in fede_data.clients:
        client_rows = [
            row for row in selected if int(row[3]) == int(client.client_id)
        ]
        if not client_rows:
            continue
        triples = [(row[0], row[1], row[2]) for row in client_rows]
        evaluator = BatchedFilteredTransEEvaluator(
            fede_data.num_entities, client.all_true_triples
        )
        ranks = evaluator.evaluate_legacy_tail(
            device_bundle,
            triples,
            client.train_entity_ids,
            query_batch_size=int(query_batch_size),
            candidate_batch_size=int(candidate_batch_size),
            progress_label="E0客户端{}".format(client.client_id),
            progress_every=int(progress_every),
        )
        client_metrics[str(client.client_id)] = metrics_from_ranks(ranks)
        all_ranks.extend(int(value) for value in ranks.tolist())
        for row, rank in zip(client_rows, ranks.tolist()):
            records.append(
                {
                    "stage": "E0",
                    "model": bundle.name,
                    "protocol": "fede_local_candidates_tail_only",
                    "head": int(row[0]),
                    "relation": int(row[1]),
                    "tail": int(row[2]),
                    "client_id": int(row[3]),
                    "direction": "tail",
                    "rank": int(rank),
                    "reciprocal_rank": float(1.0 / float(rank)),
                }
            )
    metrics = metrics_from_ranks(all_ranks)
    return (
        {
            "protocol": "fede_local_candidates_tail_only",
            "tie_policy": "legacy_double_argsort_ordinal",
            "candidate_scope": "client_train_entities_plus_target",
            "filter_scope": "client_train_valid_test",
            "selected_triple_count": len(selected),
            "metrics": metrics,
            "client_metrics": client_metrics,
        },
        records,
    )


def evaluate_global_protocol(
    stage: str,
    bundle: TransEEmbeddingBundle,
    triples: Sequence[IdTriple],
    all_true_triples: Iterable[IdTriple],
    device: torch.device,
    query_batch_size: int,
    candidate_batch_size: int,
    client_ids: Optional[Sequence[int]] = None,
    progress_every: int = 0,
) -> Tuple[Dict[str, object], List[Dict[str, object]], np.ndarray]:
    """执行全局候选、全局filtered的头尾双向乐观排名评估。"""

    if not triples:
        raise ValueError("全局评估三元组不能为空")
    device_bundle = bundle.to(device)
    evaluator = BatchedFilteredTransEEvaluator(
        int(device_bundle.entity_embeddings.shape[0]), all_true_triples
    )
    head_ranks = evaluator.evaluate_direction(
        device_bundle,
        triples,
        predict_head=True,
        query_batch_size=int(query_batch_size),
        candidate_batch_size=int(candidate_batch_size),
        progress_label="{} {}头预测".format(stage, bundle.name),
        progress_every=int(progress_every),
    )
    tail_ranks = evaluator.evaluate_direction(
        device_bundle,
        triples,
        predict_head=False,
        query_batch_size=int(query_batch_size),
        candidate_batch_size=int(candidate_batch_size),
        progress_label="{} {}尾预测".format(stage, bundle.name),
        progress_every=int(progress_every),
    )
    combined = np.concatenate([head_ranks, tail_ranks])
    per_triple_mrr = (
        (1.0 / head_ranks.astype(np.float64))
        + (1.0 / tail_ranks.astype(np.float64))
    ) / 2.0
    records: List[Dict[str, object]] = []
    normalized_clients = (
        [None] * len(triples)
        if client_ids is None
        else [int(value) for value in client_ids]
    )
    for index, (triple, client_id) in enumerate(
        zip(triples, normalized_clients)
    ):
        for direction, rank in (
            ("head", int(head_ranks[index])),
            ("tail", int(tail_ranks[index])),
        ):
            records.append(
                {
                    "stage": str(stage),
                    "model": bundle.name,
                    "protocol": "global_candidates_head_tail_filtered",
                    "head": int(triple[0]),
                    "relation": int(triple[1]),
                    "tail": int(triple[2]),
                    "client_id": client_id,
                    "direction": direction,
                    "rank": rank,
                    "reciprocal_rank": float(1.0 / float(rank)),
                }
            )
    return (
        {
            "protocol": "global_candidates_head_tail_filtered",
            "tie_policy": "optimistic_strictly_better",
            "candidate_scope": "all_entities",
            "filter_scope": "global_true_universe",
            "selected_triple_count": len(triples),
            "head_metrics": metrics_from_ranks(head_ranks),
            "tail_metrics": metrics_from_ranks(tail_ranks),
            "combined_metrics": metrics_from_ranks(combined),
        },
        records,
        per_triple_mrr,
    )


def bootstrap_mrr_interval(
    per_triple_mrr: Sequence[float],
    bootstrap_samples: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    """对每条三元组的头尾平均倒数排名计算bootstrap区间。"""

    sample_count = int(bootstrap_samples)
    if sample_count <= 0:
        return None
    values = np.asarray(per_triple_mrr, dtype=np.float64)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("bootstrap输入必须是一维非空序列")
    rng = np.random.RandomState(int(seed))
    estimates = np.empty(sample_count, dtype=np.float64)
    for index in range(sample_count):
        sampled_indices = rng.randint(0, values.size, size=values.size)
        estimates[index] = float(np.mean(values[sampled_indices]))
    return {
        "bootstrap_samples": sample_count,
        "mean": float(np.mean(values)),
        "ci_2_5": float(np.percentile(estimates, 2.5)),
        "ci_97_5": float(np.percentile(estimates, 97.5)),
    }


def bootstrap_paired_delta_interval(
    candidate_scores: Sequence[float],
    reference_scores: Sequence[float],
    bootstrap_samples: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    """计算候选模型相对参考模型的配对MRR差值区间。"""

    sample_count = int(bootstrap_samples)
    if sample_count <= 0:
        return None
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("配对bootstrap要求两个一维数组形状一致")
    differences = candidate - reference
    rng = np.random.RandomState(int(seed))
    estimates = np.empty(sample_count, dtype=np.float64)
    for index in range(sample_count):
        sampled_indices = rng.randint(
            0, differences.size, size=differences.size
        )
        estimates[index] = float(np.mean(differences[sampled_indices]))
    return {
        "bootstrap_samples": sample_count,
        "mean_delta_mrr": float(np.mean(differences)),
        "ci_2_5": float(np.percentile(estimates, 2.5)),
        "ci_97_5": float(np.percentile(estimates, 97.5)),
    }


def select_fede_test_assignments(
    fede_data: FedEDataBundle, maximum: int, seed: int
) -> Tuple[FedEAssignment, ...]:
    """为E0和E1按同一规则选择FedE测试事实。"""

    return tuple(
        _select_rows(fede_data.test_assignments, int(maximum), int(seed))
    )


def select_common_test_assignments(
    holdout: CommonHoldout, maximum: int, seed: int
) -> Tuple[FedEAssignment, ...]:
    """按固定种子为E2选择严格公共测试事实。"""

    return tuple(
        _select_rows(holdout.test_assignments, int(maximum), int(seed))
    )


__all__ = [
    "BatchedFilteredTransEEvaluator",
    "CommonHoldout",
    "FedEAssignment",
    "FedEClientPartition",
    "FedEDataBundle",
    "TransEEmbeddingBundle",
    "bootstrap_mrr_interval",
    "bootstrap_paired_delta_interval",
    "build_common_holdout",
    "build_fede_data_bundle",
    "evaluate_fede_original_protocol",
    "evaluate_global_protocol",
    "hash_assignments",
    "hash_json",
    "hash_triples",
    "load_fede_data_bundle",
    "load_fede_embedding_bundle",
    "load_project_embedding_bundle",
    "metrics_from_ranks",
    "resolve_project_checkpoint",
    "select_common_test_assignments",
    "select_fede_test_assignments",
    "sha256_file",
]
