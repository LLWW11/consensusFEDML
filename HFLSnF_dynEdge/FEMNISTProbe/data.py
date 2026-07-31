"""忽略书写者身份划分 FEMNIST，并构造共享的逻辑客户端与固定探针。"""

from __future__ import absolute_import

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


FEMNIST_CLASS_COUNT = 62
FEMNIST_SOURCE_WRITER_COUNT = 3400
FEMNIST_TRAIN_SAMPLE_COUNT = 671585
FEMNIST_TEST_SAMPLE_COUNT = 77483
FIXED_CANDIDATE_CLIENT_IDS = (
    123, 124, 125, 126, 127, 128,
    41, 42, 43, 44, 45, 46,
    0, 1, 2, 3, 4, 5,
    164, 165, 166, 167, 168, 169,
    82, 83, 84, 85, 86, 87,
    205, 206, 207, 208, 209, 210,
    129,
)


@dataclass
class FEMNISTExperimentData:
    """保存 250 个逻辑分区、37 个固定候选以及完整测试集和固定探针。"""

    candidate_client_ids: list
    candidate_train_inputs: list
    candidate_train_labels: list
    candidate_train_sample_counts: np.ndarray
    client_train_sample_counts: np.ndarray
    client_test_sample_counts: np.ndarray
    global_test_inputs: torch.Tensor
    global_test_labels: torch.Tensor
    global_test_client_ids: torch.Tensor
    probe_inputs: torch.Tensor
    probe_labels: np.ndarray
    probe_indices: np.ndarray
    candidate_manifest_hash: str
    partition_hash: str
    probe_hash: str
    class_count: int
    population_client_count: int
    source_writer_count: int
    population_train_sample_count: int
    population_test_sample_count: int
    partition_alpha: float
    partition_seed: int

    def manifest(self):
        """返回可写入 JSON 的逻辑客户端划分、候选槽位和探针公共清单。"""
        return {
            "schema_version": "femnist_probe_manifest_v2",
            "partition_method": "dirichlet",
            "partition_alpha": float(self.partition_alpha),
            "partition_seed": int(self.partition_seed),
            "partition_hash": self.partition_hash,
            "source_writer_count": int(self.source_writer_count),
            "population_client_count": int(self.population_client_count),
            "population_train_sample_count": int(
                self.population_train_sample_count
            ),
            "population_test_sample_count": int(
                self.population_test_sample_count
            ),
            "client_train_sample_counts": [
                int(value) for value in self.client_train_sample_counts
            ],
            "client_test_sample_counts": [
                int(value) for value in self.client_test_sample_counts
            ],
            "candidate_client_count": len(self.candidate_client_ids),
            "candidate_client_ids": [
                int(value) for value in self.candidate_client_ids
            ],
            "candidate_train_sample_counts": [
                int(value) for value in self.candidate_train_sample_counts
            ],
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "class_count": int(self.class_count),
            "probe_count": int(self.probe_labels.shape[0]),
            "probe_indices": [int(value) for value in self.probe_indices],
            "probe_true_labels": [int(value) for value in self.probe_labels],
            "probe_hash": self.probe_hash,
        }

    def write_manifest(self, output_path):
        """把公共划分与探针清单写入 UTF-8 JSON 文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_fixed_candidate_client_ids(client_count=250, candidate_count=37):
    """返回用户指定的 250 客户端到 37 固定槽位映射并校验其边界。"""
    client_count = int(client_count)
    candidate_count = int(candidate_count)
    if client_count != 250 or candidate_count != 37:
        raise ValueError(
            "固定槽位方案要求 client_count=250 且 candidate_count=37。"
        )
    client_ids = list(FIXED_CANDIDATE_CLIENT_IDS)
    if len(client_ids) != candidate_count or len(set(client_ids)) != candidate_count:
        raise ValueError("固定候选客户端列表长度错误或包含重复编号。")
    if any(client_id < 0 or client_id >= client_count for client_id in client_ids):
        raise ValueError("固定候选客户端编号超出 0..249 范围。")
    return client_ids


def _normalize_image_shape(images):
    """把 FEMNIST 图片统一转换为 `[样本, 1, 28, 28]` 的 float32 数组。"""
    values = np.asarray(images, dtype=np.float32)
    if values.ndim == 3 and values.shape[1:] == (28, 28):
        values = values[:, None, :, :]
    if values.ndim != 4 or values.shape[1:] != (1, 28, 28):
        raise ValueError("FEMNIST 图片形状必须为 [N,28,28] 或 [N,1,28,28]。")
    if not np.all(np.isfinite(values)):
        raise ValueError("FEMNIST 图片包含 NaN 或无穷值。")
    if values.size and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
        raise ValueError("FEMNIST 图片像素必须位于 [0,1]。")
    return np.ascontiguousarray(values)


def _hash_candidate_manifest(client_ids, sample_counts, partition_hash):
    """计算候选客户端顺序、样本数和数据划分的稳定 SHA-256。"""
    digest = hashlib.sha256()
    digest.update(b"femnist-candidate-manifest-v2\0")
    digest.update(str(partition_hash).encode("ascii"))
    digest.update(b"\0")
    for client_id, sample_count in zip(client_ids, sample_counts):
        digest.update(np.asarray(
            [client_id, sample_count], dtype=np.int64
        ).tobytes())
    return digest.hexdigest()


def _hash_partition(
        train_client_ids,
        test_client_ids,
        proportions,
        client_count,
        alpha,
        seed,
):
    """计算狄利克雷参数、比例矩阵及训练测试归属的稳定 SHA-256。"""
    digest = hashlib.sha256()
    digest.update(b"femnist-dirichlet-partition-v2\0")
    digest.update(np.asarray([client_count, seed], dtype=np.int64).tobytes())
    digest.update(np.asarray([alpha], dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(proportions, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(train_client_ids, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(test_client_ids, dtype=np.int32).tobytes())
    return digest.hexdigest()


def _hash_probe(inputs, labels, indices):
    """计算固定探针索引、标签和图片内容的稳定 SHA-256。"""
    digest = hashlib.sha256()
    digest.update(b"femnist-fixed-balanced-probe-v1\0")
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(labels, dtype=np.int64).tobytes())
    image_values = np.ascontiguousarray(inputs.numpy(), dtype=np.float32)
    digest.update(np.asarray(image_values.shape, dtype=np.int64).tobytes())
    digest.update(image_values.tobytes())
    return digest.hexdigest()


def _select_balanced_probe(test_inputs, test_labels, samples_per_class, seed):
    """从完整测试集中为每类无放回选择固定数量的探针图片。"""
    labels = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    samples_per_class = int(samples_per_class)
    if samples_per_class <= 0:
        raise ValueError("probe_samples_per_class 必须大于 0。")
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.arange(FEMNIST_CLASS_COUNT)):
        raise ValueError("完整 FEMNIST 测试集没有覆盖 0 到 61 的全部类别。")

    rng = np.random.RandomState(int(seed))
    selected = []
    for class_id in range(FEMNIST_CLASS_COUNT):
        class_indexes = np.flatnonzero(labels == class_id)
        if class_indexes.size < samples_per_class:
            raise ValueError(
                "类别 {} 只有 {} 个测试样本，少于探针要求 {}。".format(
                    class_id, class_indexes.size, samples_per_class
                )
            )
        chosen = rng.choice(
            class_indexes, size=samples_per_class, replace=False
        )
        selected.extend(sorted(int(value) for value in chosen))
    selected = np.asarray(selected, dtype=np.int64)
    probe_inputs = test_inputs.index_select(
        0, torch.from_numpy(selected).long()
    ).contiguous()
    probe_labels = labels[selected]
    return probe_inputs, probe_labels, selected


def _read_split_labels(h5_path, expected_sample_count):
    """按 H5 原始顺序读取一个 FEMNIST 分片的全部标签和书写者区间。"""
    label_parts = []
    writer_ranges = []
    offset = 0
    with h5py.File(str(h5_path), "r") as archive:
        writer_ids = list(archive["examples"].keys())
        if len(writer_ids) != FEMNIST_SOURCE_WRITER_COUNT:
            raise ValueError(
                "FEMNIST 源书写者数量应为 {}，实际为 {}。".format(
                    FEMNIST_SOURCE_WRITER_COUNT, len(writer_ids)
                )
            )
        for writer_id in writer_ids:
            labels = np.asarray(
                archive["examples"][writer_id]["label"][()],
                dtype=np.int64,
            ).reshape(-1)
            if labels.size == 0:
                raise ValueError("源书写者 {} 没有样本。".format(writer_id))
            label_parts.append(labels)
            writer_ranges.append((writer_id, offset, offset + labels.size))
            offset += int(labels.size)
    all_labels = np.concatenate(label_parts)
    if all_labels.size != int(expected_sample_count):
        raise ValueError(
            "FEMNIST 样本数应为 {}，实际为 {}。".format(
                expected_sample_count, all_labels.size
            )
        )
    if not np.array_equal(
            np.unique(all_labels), np.arange(FEMNIST_CLASS_COUNT)
    ):
        raise ValueError("FEMNIST 分片没有覆盖全部 62 类。")
    return writer_ids, all_labels, writer_ranges


def _build_dirichlet_proportions(client_count, alpha, seed):
    """生成四方案共享的 62 类到逻辑客户端狄利克雷比例矩阵。"""
    client_count = int(client_count)
    alpha = float(alpha)
    if client_count <= 0:
        raise ValueError("client_num_in_total 必须大于 0。")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("partition_alpha 必须是大于 0 的有限数值。")
    rng = np.random.RandomState(int(seed))
    return np.stack([
        rng.dirichlet(np.full(client_count, alpha, dtype=np.float64))
        for _ in range(FEMNIST_CLASS_COUNT)
    ], axis=0)


def _assign_samples_to_clients(labels, proportions, shuffle_seed):
    """按共享类别比例矩阵分配一个数据分片并返回逐样本客户端编号。"""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    proportions = np.asarray(proportions, dtype=np.float64)
    client_count = int(proportions.shape[1])
    if proportions.shape != (FEMNIST_CLASS_COUNT, client_count):
        raise ValueError("狄利克雷比例矩阵形状错误。")
    rng = np.random.RandomState(int(shuffle_seed))
    owners = np.full(labels.shape[0], -1, dtype=np.int32)
    for class_id in range(FEMNIST_CLASS_COUNT):
        class_indexes = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indexes)
        split_points = (
            np.cumsum(proportions[class_id])[:-1] * class_indexes.size
        ).astype(np.int64)
        for client_id, client_indexes in enumerate(
                np.split(class_indexes, split_points)
        ):
            owners[client_indexes] = int(client_id)
    if np.any(owners < 0):
        raise RuntimeError("狄利克雷划分后仍有未分配样本。")
    return owners


def _validate_client_partition(client_ids, client_count, split_name):
    """校验一个逻辑客户端分片完整、编号合法且 250 个客户端均非空。"""
    client_ids = np.asarray(client_ids, dtype=np.int32).reshape(-1)
    client_count = int(client_count)
    if np.any(client_ids < 0) or np.any(client_ids >= client_count):
        raise ValueError("{}划分包含非法客户端编号。".format(split_name))
    counts = np.bincount(client_ids, minlength=client_count).astype(np.int64)
    if counts.shape[0] != client_count or int(np.sum(counts)) != client_ids.size:
        raise ValueError("{}划分的样本数统计不完整。".format(split_name))
    empty_clients = np.flatnonzero(counts == 0)
    if empty_clients.size:
        raise ValueError(
            "{}狄利克雷划分产生空客户端：{}。".format(
                split_name, empty_clients.tolist()
            )
        )
    return counts


def _materialize_candidate_training_data(
        train_path,
        writer_ranges,
        train_labels,
        train_client_ids,
        candidate_client_ids,
):
    """只从源 H5 物化 37 个固定候选客户端需要的训练图片和标签。"""
    candidate_parts = {
        int(client_id): {"images": [], "labels": []}
        for client_id in candidate_client_ids
    }
    candidate_values = np.asarray(candidate_client_ids, dtype=np.int32)
    with h5py.File(str(train_path), "r") as archive:
        for writer_id, start, end in writer_ranges:
            writer_owners = train_client_ids[start:end]
            selected_mask = np.isin(
                writer_owners, candidate_values, assume_unique=False
            )
            if not np.any(selected_mask):
                continue
            writer_group = archive["examples"][writer_id]
            images = _normalize_image_shape(writer_group["pixels"][()])
            labels = train_labels[start:end]
            if images.shape[0] != labels.shape[0]:
                raise ValueError("源书写者 {} 的图片与标签数量不一致。".format(writer_id))
            for client_id in np.unique(writer_owners[selected_mask]):
                local_mask = writer_owners == int(client_id)
                candidate_parts[int(client_id)]["images"].append(images[local_mask])
                candidate_parts[int(client_id)]["labels"].append(labels[local_mask])

    candidate_inputs = []
    candidate_labels = []
    candidate_counts = []
    for client_id in candidate_client_ids:
        parts = candidate_parts[int(client_id)]
        if not parts["labels"]:
            raise ValueError("固定候选客户端 {} 没有训练样本。".format(client_id))
        images = np.concatenate(parts["images"], axis=0)
        labels = np.concatenate(parts["labels"], axis=0)
        candidate_inputs.append(torch.from_numpy(np.ascontiguousarray(images)))
        candidate_labels.append(torch.from_numpy(labels.astype(np.int64)).long())
        candidate_counts.append(int(labels.size))
    return (
        candidate_inputs,
        candidate_labels,
        np.asarray(candidate_counts, dtype=np.int64),
    )


def _materialize_full_test_data(test_path, writer_ranges, test_labels):
    """按标签读取顺序拼接完整 FEMNIST 测试图片。"""
    image_parts = []
    with h5py.File(str(test_path), "r") as archive:
        for writer_id, start, end in writer_ranges:
            images = _normalize_image_shape(
                archive["examples"][writer_id]["pixels"][()]
            )
            if images.shape[0] != end - start:
                raise ValueError("源书写者 {} 的测试图片数量错误。".format(writer_id))
            image_parts.append(images)
    test_images = np.concatenate(image_parts, axis=0)
    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("完整测试图片与标签数量不一致。")
    return torch.from_numpy(np.ascontiguousarray(test_images))


def load_femnist_experiment_data(
        data_dir,
        client_count=250,
        candidate_count=37,
        partition_alpha=0.2,
        partition_seed=0,
        probe_samples_per_class=10,
        probe_seed=0,
):
    """读取完整 FEMNIST 并构造 250 个狄利克雷逻辑客户端。

    训练和测试分片复用同一个类别比例矩阵，但分别确定性打乱类内样本。
    训练图片只物化固定 37 个候选客户端，完整测试集保留逐样本客户端归属。
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "fed_emnist_train.h5"
    test_path = data_dir / "fed_emnist_test.h5"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "FEMNIST 目录必须包含 fed_emnist_train.h5 和 fed_emnist_test.h5。"
        )

    client_count = int(client_count)
    candidate_client_ids = build_fixed_candidate_client_ids(
        client_count=client_count,
        candidate_count=candidate_count,
    )
    train_writer_ids, train_labels, train_writer_ranges = _read_split_labels(
        train_path, FEMNIST_TRAIN_SAMPLE_COUNT
    )
    test_writer_ids, test_labels, test_writer_ranges = _read_split_labels(
        test_path, FEMNIST_TEST_SAMPLE_COUNT
    )
    if test_writer_ids != train_writer_ids:
        raise ValueError("FEMNIST 训练与测试 H5 的源书写者顺序不一致。")

    proportions = _build_dirichlet_proportions(
        client_count, partition_alpha, partition_seed
    )
    train_client_ids = _assign_samples_to_clients(
        train_labels, proportions, int(partition_seed) + 1
    )
    test_client_ids = _assign_samples_to_clients(
        test_labels, proportions, int(partition_seed) + 2
    )
    train_counts = _validate_client_partition(
        train_client_ids, client_count, "训练集"
    )
    test_counts = _validate_client_partition(
        test_client_ids, client_count, "测试集"
    )
    partition_hash = _hash_partition(
        train_client_ids,
        test_client_ids,
        proportions,
        client_count,
        partition_alpha,
        partition_seed,
    )

    candidate_inputs, candidate_labels, candidate_counts = (
        _materialize_candidate_training_data(
            train_path,
            train_writer_ranges,
            train_labels,
            train_client_ids,
            candidate_client_ids,
        )
    )
    expected_candidate_counts = train_counts[
        np.asarray(candidate_client_ids, dtype=np.int64)
    ]
    if not np.array_equal(candidate_counts, expected_candidate_counts):
        raise ValueError("固定候选训练张量数量与狄利克雷划分统计不一致。")

    global_test_inputs = _materialize_full_test_data(
        test_path, test_writer_ranges, test_labels
    )
    global_test_labels = torch.from_numpy(test_labels.astype(np.int64)).long()
    global_test_client_ids = torch.from_numpy(
        test_client_ids.astype(np.int64)
    ).long()
    probe_inputs, probe_labels, probe_indices = _select_balanced_probe(
        global_test_inputs,
        test_labels,
        probe_samples_per_class,
        probe_seed,
    )
    return FEMNISTExperimentData(
        candidate_client_ids=candidate_client_ids,
        candidate_train_inputs=candidate_inputs,
        candidate_train_labels=candidate_labels,
        candidate_train_sample_counts=candidate_counts,
        client_train_sample_counts=train_counts,
        client_test_sample_counts=test_counts,
        global_test_inputs=global_test_inputs,
        global_test_labels=global_test_labels,
        global_test_client_ids=global_test_client_ids,
        probe_inputs=probe_inputs,
        probe_labels=np.asarray(probe_labels, dtype=np.int64),
        probe_indices=np.asarray(probe_indices, dtype=np.int64),
        candidate_manifest_hash=_hash_candidate_manifest(
            candidate_client_ids, candidate_counts, partition_hash
        ),
        partition_hash=partition_hash,
        probe_hash=_hash_probe(
            probe_inputs, probe_labels, probe_indices
        ),
        class_count=FEMNIST_CLASS_COUNT,
        population_client_count=client_count,
        source_writer_count=len(train_writer_ids),
        population_train_sample_count=int(train_labels.size),
        population_test_sample_count=int(test_labels.size),
        partition_alpha=float(partition_alpha),
        partition_seed=int(partition_seed),
    )
