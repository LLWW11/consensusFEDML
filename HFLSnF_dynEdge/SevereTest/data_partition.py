"""MNIST 单标签确定性划分与 FedML 数据集构造。"""

from __future__ import absolute_import

from pathlib import Path

import numpy as np

from MNISTReader import MNISTImageReader, MNISTLabelReader
from fedml.ml.engine import ml_engine_adapter


def build_label_modulo_partitions(
        labels,
        client_num=200,
        class_num=10,
        clients_per_class=20,
        seed=5,
):
    """按“客户端编号模 10 等于标签”规则返回每个客户端的样本索引。

    每个类别内部先使用独立、可复现的随机数生成器打乱，再用
    ``numpy.array_split`` 尽量均匀地分给对应的 20 个客户端。
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    client_num = int(client_num)
    class_num = int(class_num)
    clients_per_class = int(clients_per_class)
    if client_num != class_num * clients_per_class:
        raise ValueError(
            "client_num 必须等于 class_num * clients_per_class，实际为 {} != {} * {}。"
            .format(client_num, class_num, clients_per_class)
        )
    if labels.size == 0:
        raise ValueError("标签数组不能为空。")
    if np.any(labels < 0) or np.any(labels >= class_num):
        raise ValueError("标签必须位于 0 到 {}。".format(class_num - 1))

    partitions = [np.empty(0, dtype=np.int64) for _ in range(client_num)]
    for class_id in range(class_num):
        class_indexes = np.flatnonzero(labels == class_id)
        if class_indexes.size < clients_per_class:
            raise ValueError(
                "类别 {} 只有 {} 个样本，无法保证 {} 个客户端均非空。".format(
                    class_id, class_indexes.size, clients_per_class
                )
            )
        # 每个类别使用独立随机流，避免某个类别数量变化影响其他类别的划分。
        class_rng = np.random.RandomState(int(seed) + class_id * 1009)
        shuffled_indexes = class_indexes[class_rng.permutation(class_indexes.size)]
        client_ids = [
            class_id + class_num * group_index
            for group_index in range(clients_per_class)
        ]
        for client_id, shard in zip(
                client_ids, np.array_split(shuffled_indexes, clients_per_class)
        ):
            partitions[client_id] = np.asarray(shard, dtype=np.int64)

    validate_label_modulo_partitions(
        labels=labels,
        partitions=partitions,
        client_num=client_num,
        class_num=class_num,
        clients_per_class=clients_per_class,
    )
    return partitions


def validate_label_modulo_partitions(
        labels,
        partitions,
        client_num=200,
        class_num=10,
        clients_per_class=20,
):
    """校验所有样本只出现一次、客户端非空且每个客户端仅有目标标签。"""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(partitions) != int(client_num):
        raise ValueError("分区数量必须等于客户端总数。")

    assigned_parts = []
    per_class_sizes = {class_id: [] for class_id in range(int(class_num))}
    for client_id, raw_indexes in enumerate(partitions):
        indexes = np.asarray(raw_indexes, dtype=np.int64).reshape(-1)
        if indexes.size == 0:
            raise ValueError("客户端 {} 没有数据。".format(client_id))
        if np.any(indexes < 0) or np.any(indexes >= labels.size):
            raise ValueError("客户端 {} 含有越界样本索引。".format(client_id))
        expected_label = client_id % int(class_num)
        actual_labels = np.unique(labels[indexes])
        if actual_labels.tolist() != [expected_label]:
            raise ValueError(
                "客户端 {} 应仅包含标签 {}，实际为 {}。".format(
                    client_id, expected_label, actual_labels.tolist()
                )
            )
        assigned_parts.append(indexes)
        per_class_sizes[expected_label].append(int(indexes.size))

    assigned = np.concatenate(assigned_parts)
    if assigned.size != labels.size:
        raise ValueError("已分配样本总数与原始标签数不一致。")
    if np.unique(assigned).size != labels.size:
        raise ValueError("样本存在重复分配或遗漏。")
    if not np.array_equal(np.sort(assigned), np.arange(labels.size)):
        raise ValueError("样本索引没有完整覆盖原始数据集。")

    for class_id, sizes in per_class_sizes.items():
        if len(sizes) != int(clients_per_class):
            raise ValueError("类别 {} 对应的客户端数量不正确。".format(class_id))
        if max(sizes) - min(sizes) > 1:
            raise ValueError("类别 {} 的客户端样本数差异超过 1。".format(class_id))
    return True


def _read_idx_pair(image_path, label_path, sample_count):
    """读取一组 IDX 图片和标签，并返回 NumPy 数组。"""
    image_reader = MNISTImageReader(str(image_path))
    label_reader = MNISTLabelReader(str(label_path))
    image_reader.open()
    label_reader.open()
    try:
        _, images = image_reader.read(int(sample_count))
        _, labels = label_reader.read(int(sample_count))
    finally:
        image_reader.close()
        label_reader.close()
    return np.asarray(images), np.asarray(labels, dtype=np.int64)


def load_mnist_arrays(data_dir):
    """从指定目录读取完整 MNIST 训练集和测试集。"""
    data_dir = Path(data_dir)
    train_images, train_labels = _read_idx_pair(
        data_dir / "train-images.idx3-ubyte",
        data_dir / "train-labels.idx1-ubyte",
        60000,
    )
    test_images, test_labels = _read_idx_pair(
        data_dir / "t10k-images.idx3-ubyte",
        data_dir / "t10k-labels.idx1-ubyte",
        10000,
    )
    return train_images, train_labels, test_images, test_labels


def _build_client_batches(args, images, labels, indexes, batch_size, shuffle_seed):
    """把一个客户端的图片和标签转换为 FedML 可消费的批次列表。"""
    indexes = np.asarray(indexes, dtype=np.int64)
    local_rng = np.random.RandomState(int(shuffle_seed))
    shuffled_indexes = indexes[local_rng.permutation(indexes.size)]
    local_images = images[shuffled_indexes].astype(np.float32) / 255.0
    local_labels = labels[shuffled_indexes].astype(np.int64)

    batches = []
    for start_index in range(0, local_labels.size, int(batch_size)):
        end_index = min(start_index + int(batch_size), local_labels.size)
        batch_x, batch_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(
            args,
            local_images[start_index:end_index],
            local_labels[start_index:end_index],
        )
        batches.append((batch_x, batch_y))
    return batches


def load_severe_mnist_data(args):
    """构造 200 个单标签客户端的 FedML 数据集、类别数和划分清单。"""
    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = Path(str(getattr(args, "data_cache_dir", "MNIST")))
    data_dir = raw_data_dir if raw_data_dir.is_absolute() else project_root / raw_data_dir
    train_images, train_labels, test_images, test_labels = load_mnist_arrays(data_dir)

    client_num = int(getattr(args, "client_num_in_total", 200))
    class_num = int(getattr(args, "class_num", 10))
    clients_per_class = int(getattr(args, "clients_per_class", 20))
    partition_seed = int(getattr(args, "partition_seed", 5))
    train_partitions = build_label_modulo_partitions(
        train_labels, client_num, class_num, clients_per_class, partition_seed
    )
    test_partitions = build_label_modulo_partitions(
        test_labels,
        client_num,
        class_num,
        clients_per_class,
        partition_seed + 100000,
    )

    train_data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_global = []
    test_data_global = []
    partition_manifest = []
    for client_id in range(client_num):
        train_batches = _build_client_batches(
            args,
            train_images,
            train_labels,
            train_partitions[client_id],
            args.batch_size,
            partition_seed + client_id,
        )
        test_batches = _build_client_batches(
            args,
            test_images,
            test_labels,
            test_partitions[client_id],
            args.batch_size,
            partition_seed + 200000 + client_id,
        )
        train_count = int(train_partitions[client_id].size)
        test_count = int(test_partitions[client_id].size)
        train_data_local_num_dict[client_id] = train_count
        train_data_local_dict[client_id] = train_batches
        test_data_local_dict[client_id] = test_batches
        train_data_global.extend(train_batches)
        test_data_global.extend(test_batches)
        partition_manifest.append({
            "client_id": client_id,
            "label": client_id % class_num,
            "train_count": train_count,
            "test_count": test_count,
        })

    dataset = [
        int(train_labels.size),
        int(test_labels.size),
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num, partition_manifest

