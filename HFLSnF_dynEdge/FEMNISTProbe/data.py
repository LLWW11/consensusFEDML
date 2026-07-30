"""按书写者读取 FEMNIST，并构造四方案共享的候选客户端与固定探针。"""

from __future__ import absolute_import

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


FEMNIST_CLASS_COUNT = 62
FEMNIST_TRAIN_CLIENT_COUNT = 3400
FEMNIST_TRAIN_SAMPLE_COUNT = 671585
FEMNIST_TEST_SAMPLE_COUNT = 77483


@dataclass
class FEMNISTExperimentData:
    """保存固定候选书写者、完整测试集和固定类别均衡探针。"""

    candidate_writer_ids: list
    candidate_train_inputs: list
    candidate_train_labels: list
    candidate_train_sample_counts: np.ndarray
    global_test_inputs: torch.Tensor
    global_test_labels: torch.Tensor
    probe_inputs: torch.Tensor
    probe_labels: np.ndarray
    probe_indices: np.ndarray
    candidate_manifest_hash: str
    probe_hash: str
    class_count: int
    population_client_count: int
    population_train_sample_count: int

    def manifest(self):
        """返回可写入 JSON 的候选客户端与探针公共清单。"""
        return {
            "schema_version": "femnist_probe_manifest_v1",
            "population_client_count": int(self.population_client_count),
            "population_train_sample_count": int(
                self.population_train_sample_count
            ),
            "candidate_client_count": len(self.candidate_writer_ids),
            "candidate_writer_ids": list(self.candidate_writer_ids),
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
        """把公共候选与探针清单写入 UTF-8 JSON 文件。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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


def _hash_candidate_manifest(writer_ids, sample_counts):
    """计算候选书写者顺序和训练样本数的稳定 SHA-256。"""
    digest = hashlib.sha256()
    digest.update(b"femnist-candidate-manifest-v1\0")
    for writer_id, sample_count in zip(writer_ids, sample_counts):
        digest.update(str(writer_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray([sample_count], dtype=np.int64).tobytes())
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


def _select_candidate_indexes(writer_count, candidate_count, seed):
    """按固定随机种子从书写者池中无放回选择候选槽位。"""
    writer_count = int(writer_count)
    candidate_count = int(candidate_count)
    if candidate_count <= 0 or candidate_count > writer_count:
        raise ValueError("candidate_count 必须位于 1 到书写者总数之间。")
    rng = np.random.RandomState(int(seed))
    return rng.choice(
        writer_count, size=candidate_count, replace=False
    ).astype(np.int64)


def _select_balanced_probe(test_inputs, test_labels, samples_per_class, seed):
    """从完整测试集为每类无放回选择固定数量的探针图片。"""
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


def load_femnist_experiment_data(
        data_dir,
        candidate_count=37,
        candidate_seed=0,
        probe_samples_per_class=10,
        probe_seed=0,
):
    """读取完整 FEMNIST，并只构造固定候选书写者的本地训练张量。

    完整测试集只拼接一次，供云模型评估和类别均衡探针选择使用。客户端聚合
    样本数严格取训练标签数量，不使用 DataLoader 批次数或测试样本数。
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "fed_emnist_train.h5"
    test_path = data_dir / "fed_emnist_test.h5"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "FEMNIST 目录必须包含 fed_emnist_train.h5 和 fed_emnist_test.h5。"
        )

    with h5py.File(str(train_path), "r") as train_h5:
        writer_ids = list(train_h5["examples"].keys())
        if len(writer_ids) != FEMNIST_TRAIN_CLIENT_COUNT:
            raise ValueError(
                "FEMNIST 训练书写者数量应为 {}，实际为 {}。".format(
                    FEMNIST_TRAIN_CLIENT_COUNT, len(writer_ids)
                )
            )
        population_train_sample_count = int(sum(
            train_h5["examples"][writer_id]["label"].shape[0]
            for writer_id in writer_ids
        ))
        if population_train_sample_count != FEMNIST_TRAIN_SAMPLE_COUNT:
            raise ValueError(
                "FEMNIST训练样本数应为{}，实际为{}。".format(
                    FEMNIST_TRAIN_SAMPLE_COUNT,
                    population_train_sample_count,
                )
            )
        candidate_indexes = _select_candidate_indexes(
            len(writer_ids), candidate_count, candidate_seed
        )
        candidate_writer_ids = [
            str(writer_ids[int(index)]) for index in candidate_indexes
        ]
        candidate_train_inputs = []
        candidate_train_labels = []
        candidate_sample_counts = []
        candidate_label_parts = []
        for writer_id in candidate_writer_ids:
            writer_group = train_h5["examples"][writer_id]
            images = _normalize_image_shape(writer_group["pixels"][()])
            labels = np.asarray(
                writer_group["label"][()], dtype=np.int64
            ).reshape(-1)
            if images.shape[0] != labels.shape[0] or labels.size == 0:
                raise ValueError("候选书写者 {} 的图片与标签数量错误。".format(writer_id))
            candidate_train_inputs.append(torch.from_numpy(images))
            candidate_train_labels.append(torch.from_numpy(labels).long())
            candidate_sample_counts.append(int(labels.size))
            candidate_label_parts.append(labels)

    covered_labels = np.unique(np.concatenate(candidate_label_parts))
    if not np.array_equal(covered_labels, np.arange(FEMNIST_CLASS_COUNT)):
        raise ValueError(
            "固定候选书写者训练数据必须覆盖全部 62 类，实际缺少 {}。".format(
                sorted(set(range(FEMNIST_CLASS_COUNT)) - set(covered_labels.tolist()))
            )
        )

    test_image_parts = []
    test_label_parts = []
    with h5py.File(str(test_path), "r") as test_h5:
        test_writer_ids = list(test_h5["examples"].keys())
        if test_writer_ids != writer_ids:
            raise ValueError("FEMNIST 训练与测试 H5 的书写者顺序不一致。")
        for writer_id in test_writer_ids:
            writer_group = test_h5["examples"][writer_id]
            test_image_parts.append(_normalize_image_shape(
                writer_group["pixels"][()]
            ))
            test_label_parts.append(np.asarray(
                writer_group["label"][()], dtype=np.int64
            ).reshape(-1))

    test_images = np.concatenate(test_image_parts, axis=0)
    test_labels = np.concatenate(test_label_parts, axis=0)
    if test_labels.size != FEMNIST_TEST_SAMPLE_COUNT:
        raise ValueError(
            "FEMNIST 测试样本数应为 {}，实际为 {}。".format(
                FEMNIST_TEST_SAMPLE_COUNT, test_labels.size
            )
        )
    global_test_inputs = torch.from_numpy(test_images)
    global_test_labels = torch.from_numpy(test_labels).long()
    probe_inputs, probe_labels, probe_indices = _select_balanced_probe(
        global_test_inputs,
        test_labels,
        probe_samples_per_class,
        probe_seed,
    )
    sample_counts = np.asarray(candidate_sample_counts, dtype=np.int64)
    return FEMNISTExperimentData(
        candidate_writer_ids=candidate_writer_ids,
        candidate_train_inputs=candidate_train_inputs,
        candidate_train_labels=candidate_train_labels,
        candidate_train_sample_counts=sample_counts,
        global_test_inputs=global_test_inputs,
        global_test_labels=global_test_labels,
        probe_inputs=probe_inputs,
        probe_labels=np.asarray(probe_labels, dtype=np.int64),
        probe_indices=np.asarray(probe_indices, dtype=np.int64),
        candidate_manifest_hash=_hash_candidate_manifest(
            candidate_writer_ids, sample_counts
        ),
        probe_hash=_hash_probe(
            probe_inputs, probe_labels, probe_indices
        ),
        class_count=FEMNIST_CLASS_COUNT,
        population_client_count=len(writer_ids),
        population_train_sample_count=population_train_sample_count,
    )
