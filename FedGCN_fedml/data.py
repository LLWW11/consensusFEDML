"""Planetoid 图数据读取、规范化和联邦设备划分。"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


SUPPORTED_DATASETS = {"cora", "citeseer"}


@dataclass(frozen=True)
class LocalGraphPartition:
    """保存一台 FedGCN 设备持有的局部诱导子图。"""

    client_id: int
    node_indices: torch.Tensor
    features: torch.Tensor
    adjacency: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor

    @property
    def train_node_count(self) -> int:
        """返回该设备拥有的有标签训练节点数量。"""

        return int(self.train_indices.numel())

    @property
    def node_count(self) -> int:
        """返回该设备持有的全部图节点数量。"""

        return int(self.node_indices.numel())


@dataclass(frozen=True)
class FederatedGraphData:
    """保存完整图以及由完整图切分出的联邦局部子图。"""

    dataset_name: str
    features: torch.Tensor
    adjacency: torch.Tensor
    labels: torch.Tensor
    idx_train: torch.Tensor
    idx_val: torch.Tensor
    idx_test: torch.Tensor
    partitions: Tuple[LocalGraphPartition, ...]
    num_classes: int

    @property
    def num_nodes(self) -> int:
        """返回完整图的节点数量。"""

        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        """返回每个图节点的输入特征维度。"""

        return int(self.features.shape[1])


def parse_index_file(path: Path) -> List[int]:
    """读取 Planetoid 数据集的测试节点索引文件。"""

    indices = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                indices.append(int(text))
    return indices


def _load_pickle(path: Path):
    """以兼容 Python 2 生成文件的方式读取 Planetoid pickle。"""

    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def row_normalize_adjacency(adjacency: sp.spmatrix) -> sp.csr_matrix:
    """添加自环并对稀疏邻接矩阵执行按行归一化。"""

    adjacency_with_self_loops = adjacency.tocsr().astype(np.float32)
    adjacency_with_self_loops = adjacency_with_self_loops + sp.eye(
        adjacency_with_self_loops.shape[0], dtype=np.float32, format="csr"
    )
    row_sum = np.asarray(adjacency_with_self_loops.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero_mask = row_sum != 0
    inverse[nonzero_mask] = 1.0 / row_sum[nonzero_mask]
    return sp.diags(inverse).dot(adjacency_with_self_loops).tocsr()


def load_planetoid_graph(
    data_dir: Path, dataset_name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """读取 Cora 或 Citeseer，并返回完整图张量和标准数据划分。"""

    dataset_name = str(dataset_name).strip().lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            "dataset 必须是 cora 或 citeseer，实际为 {}".format(dataset_name)
        )

    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError("找不到图数据目录：{}".format(data_dir))

    object_names = ("x", "y", "tx", "ty", "allx", "ally", "graph")
    objects: Dict[str, object] = {}
    for object_name in object_names:
        path = data_dir / "ind.{}.{}".format(dataset_name, object_name)
        if not path.is_file():
            raise FileNotFoundError("缺少 Planetoid 数据文件：{}".format(path))
        objects[object_name] = _load_pickle(path)

    test_index_path = data_dir / "ind.{}.test.index".format(dataset_name)
    if not test_index_path.is_file():
        raise FileNotFoundError("缺少测试索引文件：{}".format(test_index_path))
    test_index_reorder = np.asarray(parse_index_file(test_index_path), dtype=np.int64)
    test_index_range = np.sort(test_index_reorder)

    x = objects["x"]
    y = objects["y"]
    tx = objects["tx"]
    ty = objects["ty"]
    allx = objects["allx"]
    ally = objects["ally"]
    graph = objects["graph"]

    if dataset_name == "citeseer":
        # Citeseer 测试索引不连续，需要显式补齐孤立节点对应的零特征和零标签。
        full_test_range = np.arange(test_index_range.min(), test_index_range.max() + 1)
        tx_extended = sp.lil_matrix((len(full_test_range), x.shape[1]), dtype=np.float32)
        tx_extended[test_index_range - full_test_range.min(), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(full_test_range), y.shape[1]), dtype=ty.dtype)
        ty_extended[test_index_range - full_test_range.min(), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_index_reorder, :] = features[test_index_range, :]
    labels_one_hot = np.vstack((ally, ty))
    labels_one_hot[test_index_reorder, :] = labels_one_hot[test_index_range, :]

    graph_nx = nx.from_dict_of_lists(graph)
    adjacency_sparse = nx.adjacency_matrix(graph_nx)
    adjacency_normalized = row_normalize_adjacency(adjacency_sparse)

    features_tensor = torch.tensor(features.toarray(), dtype=torch.float32)
    adjacency_tensor = torch.tensor(
        adjacency_normalized.toarray(), dtype=torch.float32
    )
    labels_tensor = torch.tensor(
        np.argmax(labels_one_hot, axis=1), dtype=torch.long
    )
    idx_train = torch.arange(len(y), dtype=torch.long)
    validation_end = min(len(y) + 500, labels_tensor.numel())
    idx_val = torch.arange(len(y), validation_end, dtype=torch.long)
    idx_test = torch.tensor(test_index_range, dtype=torch.long)
    return (
        features_tensor,
        adjacency_tensor,
        labels_tensor,
        idx_train,
        idx_val,
        idx_test,
    )


def partition_graph_nodes(
    labels: Sequence[int], client_count: int, iid_fraction: float, seed: int
) -> Tuple[np.ndarray, ...]:
    """按类别偏置与 IID 混合策略把全部节点无重叠地分给设备。"""

    client_count = int(client_count)
    if client_count <= 0:
        raise ValueError("client_count 必须大于 0")
    iid_fraction = float(iid_fraction)
    if iid_fraction < 0.0 or iid_fraction > 1.0:
        raise ValueError("iid_fraction 必须位于 [0, 1]")

    if isinstance(labels, torch.Tensor):
        label_array = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    else:
        label_array = np.asarray(labels, dtype=np.int64)
    label_array = label_array.reshape(-1)
    if label_array.size == 0:
        raise ValueError("labels 不能为空")

    rng = np.random.RandomState(int(seed))
    assignments: List[List[int]] = [[] for _ in range(client_count)]
    assigned_mask = np.zeros(label_array.size, dtype=bool)
    non_iid_fraction = 1.0 - iid_fraction

    # 每一类别的非 IID 部分固定交给其亲和设备，剩余节点进入全局 IID 池。
    for class_id in sorted(int(value) for value in np.unique(label_array)):
        class_nodes = np.flatnonzero(label_array == class_id)
        rng.shuffle(class_nodes)
        biased_count = int(np.floor(len(class_nodes) * non_iid_fraction))
        biased_nodes = class_nodes[:biased_count]
        affinity_client = class_id % client_count
        assignments[affinity_client].extend(int(node) for node in biased_nodes)
        assigned_mask[biased_nodes] = True

    iid_nodes = np.flatnonzero(~assigned_mask)
    rng.shuffle(iid_nodes)
    for offset, node_index in enumerate(iid_nodes):
        assignments[offset % client_count].append(int(node_index))

    # 当设备数超过有效类别或 IID 节点过少时，从最大分区移动节点，避免空子图。
    for client_id in range(client_count):
        if assignments[client_id]:
            continue
        donor_id = max(range(client_count), key=lambda value: len(assignments[value]))
        if len(assignments[donor_id]) <= 1:
            raise ValueError("节点数不足，无法为每台设备构造非空子图")
        assignments[client_id].append(assignments[donor_id].pop())

    partitions = tuple(
        np.asarray(sorted(client_nodes), dtype=np.int64)
        for client_nodes in assignments
    )
    flattened = np.concatenate(partitions)
    if flattened.size != label_array.size:
        raise RuntimeError("节点划分没有覆盖完整图")
    if np.unique(flattened).size != flattened.size:
        raise RuntimeError("节点划分中存在重复节点")
    if not np.array_equal(np.sort(flattened), np.arange(label_array.size)):
        raise RuntimeError("节点划分中存在非法或遗漏节点")
    return partitions


def build_federated_graph_data(
    dataset_name: str,
    data_dir: Path,
    client_count: int,
    iid_fraction: float,
    seed: int,
) -> FederatedGraphData:
    """加载完整图，并为每台设备构建局部诱导子图及本地训练索引。"""

    (
        features,
        adjacency,
        labels,
        idx_train,
        idx_val,
        idx_test,
    ) = load_planetoid_graph(data_dir, dataset_name)
    node_partitions = partition_graph_nodes(
        labels=labels,
        client_count=client_count,
        iid_fraction=iid_fraction,
        seed=seed,
    )
    global_train_indices = idx_train.detach().cpu().numpy()
    local_partitions = []
    for client_id, node_indices_array in enumerate(node_partitions):
        # 节点编号经过排序，因此可用 searchsorted 将全局训练编号转换为局部编号。
        local_train_global = np.intersect1d(
            node_indices_array, global_train_indices, assume_unique=True
        )
        local_train_indices = np.searchsorted(
            node_indices_array, local_train_global
        ).astype(np.int64)
        node_indices = torch.tensor(node_indices_array, dtype=torch.long)
        local_partitions.append(
            LocalGraphPartition(
                client_id=client_id,
                node_indices=node_indices,
                features=features.index_select(0, node_indices).clone(),
                adjacency=adjacency.index_select(0, node_indices)
                .index_select(1, node_indices)
                .clone(),
                labels=labels.index_select(0, node_indices).clone(),
                train_indices=torch.tensor(local_train_indices, dtype=torch.long),
            )
        )

    return FederatedGraphData(
        dataset_name=str(dataset_name).strip().lower(),
        features=features,
        adjacency=adjacency,
        labels=labels,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        partitions=tuple(local_partitions),
        num_classes=int(labels.max().item()) + 1,
    )

