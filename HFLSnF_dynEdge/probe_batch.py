"""固定类别均衡探针的选择、共识计算与结构化保存。"""

import csv
import hashlib
import json
import logging
import os
import random

import numpy as np
import torch

from probe_metrics import calculate_population_probe_metrics, validate_probability_tensor


class FixedProbeSet(object):
    """保存一次实验中固定不变的探针输入、索引、真实标签与内容哈希。"""

    def __init__(self, inputs, indices, true_labels, content_hash, source):
        """初始化固定探针集合，并保留后续批量推理所需的张量。"""
        self.inputs = inputs
        self.indices = np.asarray(indices, dtype=np.int64)
        self.true_labels = np.asarray(true_labels, dtype=np.int64)
        self.content_hash = str(content_hash)
        self.source = str(source)

    @property
    def sample_count(self):
        """返回固定探针图片总数。"""
        return int(self.indices.shape[0])

    @property
    def class_count(self):
        """返回探针真实标签覆盖的类别数。"""
        return int(np.unique(self.true_labels).shape[0])


def _as_cpu_tensor(value):
    """把数组或张量转换为脱离计算图的 CPU 张量。"""
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _hash_probe_content(inputs, indices, true_labels):
    """按索引、标签和标准化图片内容计算可跨实验比较的 SHA-256。"""
    digest = hashlib.sha256()
    digest.update(b"fixed-balanced-probe-v1\0")
    digest.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(true_labels, dtype=np.int64).tobytes())
    # 统一用 float32 哈希，避免相同输入仅因张量存储精度表示不同而误报。
    image_array = np.ascontiguousarray(inputs.detach().cpu().numpy(), dtype=np.float32)
    digest.update(np.asarray(image_array.shape, dtype=np.int64).tobytes())
    digest.update(image_array.tobytes())
    return digest.hexdigest()


def select_fixed_balanced_probe(probe_data, samples_per_class, seed, source="test"):
    """使用局部随机数生成器从数据集中为每个类别无放回选择固定数量样本。

    该函数不会调用全局 NumPy、Python 或 PyTorch 随机接口，因此不会改变训练
    随机状态。返回顺序固定为“类别升序、类别内索引升序”。
    """
    samples_per_class = int(samples_per_class)
    if samples_per_class <= 0:
        raise ValueError("probe_samples_per_class 必须大于 0。")

    # 即便未来把probe_data换成会在迭代时采样的DataLoader，也恢复三类全局CPU随机状态。
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()
    try:
        batches = list(probe_data)
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)
        torch.set_rng_state(torch_random_state)
    if not batches:
        raise ValueError("探针来源数据集为空，无法选择固定探针。")

    label_chunks = []
    for _, labels in batches:
        label_tensor = _as_cpu_tensor(labels).reshape(-1)
        label_chunks.append(label_tensor.numpy().astype(np.int64, copy=False))
    all_labels = np.concatenate(label_chunks, axis=0)
    if all_labels.size == 0:
        raise ValueError("探针来源数据集不包含任何标签。")
    if np.any(all_labels < 0):
        raise ValueError("探针真实标签不能为负数。")

    local_rng = np.random.RandomState(int(seed))
    selected_indices = []
    for class_id in sorted(int(value) for value in np.unique(all_labels)):
        class_indices = np.flatnonzero(all_labels == class_id)
        if class_indices.shape[0] < samples_per_class:
            raise ValueError(
                "类别 {} 只有 {} 张图片，少于 probe_samples_per_class={}。".format(
                    class_id, class_indices.shape[0], samples_per_class
                )
            )
        chosen = local_rng.choice(class_indices, size=samples_per_class, replace=False)
        selected_indices.extend(sorted(int(value) for value in chosen))

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selected_labels = all_labels[selected_indices]
    selected_positions = {
        int(global_index): output_position
        for output_position, global_index in enumerate(selected_indices)
    }
    selected_samples = [None for _ in range(selected_indices.shape[0])]

    global_offset = 0
    for inputs, labels in batches:
        input_tensor = _as_cpu_tensor(inputs)
        label_count = int(_as_cpu_tensor(labels).reshape(-1).shape[0])
        for local_index in range(label_count):
            global_index = global_offset + local_index
            output_position = selected_positions.get(global_index)
            if output_position is not None:
                # 保留单样本原始维度，最终仅在这里统一堆叠为一个推理批次。
                selected_samples[output_position] = input_tensor[local_index]
        global_offset += label_count

    if any(sample is None for sample in selected_samples):
        raise RuntimeError("固定探针索引与数据批次无法完整对齐。")
    probe_inputs = torch.stack(selected_samples, dim=0)
    content_hash = _hash_probe_content(
        probe_inputs, selected_indices, selected_labels
    )
    return FixedProbeSet(
        inputs=probe_inputs,
        indices=selected_indices,
        true_labels=selected_labels,
        content_hash=content_hash,
        source=source,
    )


def _format_csv_value(value):
    """把摘要数值转换为稳定的 CSV 文本，空指标写为空单元格。"""
    if value is None:
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(numeric_value):
        return ""
    return "{:.12g}".format(numeric_value)


class ProbeBatchRecorder(object):
    """逐 epoch 保存固定探针概率、活跃掩码和汇总指标。"""

    SUMMARY_COLUMNS = [
        "global_epoch",
        "probe_count",
        "candidate_count",
        "active_count",
        "active_coverage",
        "candidate_agreement_mean",
        "candidate_certainty_mean",
        "candidate_effective_mean",
        "candidate_correct_effective_mean",
        "candidate_wrong_effective_mean",
        "candidate_effective_q25",
        "candidate_effective_q50",
        "candidate_effective_q75",
        "active_agreement_mean",
        "active_certainty_mean",
        "active_effective_mean",
        "active_correct_effective_mean",
        "active_wrong_effective_mean",
        "active_effective_q25",
        "active_effective_q50",
        "active_effective_q75",
        "coverage_weighted_active_correct_effective",
        "edge_effective_mean",
        "edge_correct_effective_mean",
        "cloud_probe_accuracy",
        "cloud_true_class_probability_mean",
    ]

    def __init__(
            self,
            result_dir,
            total_epochs,
            candidate_client_ids,
            edge_slot_count,
            probe_set,
            class_count,
            npz_filename="probe_probabilities.npz",
            summary_filename="probe_epoch_summary.csv",
            checkpoint_interval=10,
    ):
        """初始化内存缓冲区，并创建带表头的逐 epoch 摘要文件。"""
        self.result_dir = os.path.abspath(result_dir)
        self.total_epochs = int(total_epochs)
        self.candidate_client_ids = np.asarray(candidate_client_ids, dtype=np.int64)
        self.edge_slot_count = int(edge_slot_count)
        self.probe_set = probe_set
        self.class_count = int(class_count)
        self.checkpoint_interval = int(checkpoint_interval)
        if self.total_epochs <= 0:
            raise ValueError("总 epoch 数必须大于 0。")
        if self.candidate_client_ids.size == 0:
            raise ValueError("固定候选客户端列表不能为空。")
        if np.unique(self.candidate_client_ids).size != self.candidate_client_ids.size:
            raise ValueError("固定候选客户端编号不能重复。")
        if self.edge_slot_count < 0:
            raise ValueError("边缘槽位数不能为负数。")
        if self.class_count < 2:
            raise ValueError("概率类别数必须至少为 2。")
        if self.checkpoint_interval <= 0:
            raise ValueError("probe_checkpoint_interval 必须大于 0。")

        os.makedirs(self.result_dir, exist_ok=True)
        self.npz_path = os.path.join(self.result_dir, str(npz_filename))
        self.summary_path = os.path.join(self.result_dir, str(summary_filename))
        probe_count = self.probe_set.sample_count
        candidate_count = int(self.candidate_client_ids.shape[0])
        self.client_probabilities = np.full(
            (self.total_epochs, candidate_count, probe_count, self.class_count),
            np.nan,
            dtype=np.float32,
        )
        self.edge_probabilities = np.full(
            (self.total_epochs, self.edge_slot_count, probe_count, self.class_count),
            np.nan,
            dtype=np.float32,
        )
        self.cloud_probabilities = np.full(
            (self.total_epochs, probe_count, self.class_count),
            np.nan,
            dtype=np.float32,
        )
        self.active_client_mask = np.zeros(
            (self.total_epochs, candidate_count), dtype=np.bool_
        )
        self.edge_active_mask = np.zeros(
            (self.total_epochs, self.edge_slot_count), dtype=np.bool_
        )
        self.global_epochs = np.full(self.total_epochs, -1, dtype=np.int64)
        self.completed_epochs = 0
        self._closed = False
        self._summary_file = open(
            self.summary_path, "w", newline="", encoding="utf-8"
        )
        self._summary_writer = csv.DictWriter(
            self._summary_file, fieldnames=self.SUMMARY_COLUMNS
        )
        self._summary_writer.writeheader()
        self._summary_file.flush()

    def _validate_epoch_probabilities(
            self, client_probabilities, edge_probabilities, cloud_probabilities
    ):
        """校验本 epoch 三层概率形状，并允许未启用边缘槽位整块为 NaN。"""
        clients = validate_probability_tensor(
            client_probabilities, "客户端探针概率"
        )
        expected_client_shape = self.client_probabilities.shape[1:]
        if clients.shape != expected_client_shape:
            raise ValueError(
                "客户端探针概率形状 {}，期望 {}。".format(
                    clients.shape, expected_client_shape
                )
            )

        edges = np.asarray(edge_probabilities, dtype=np.float64)
        expected_edge_shape = self.edge_probabilities.shape[1:]
        if edges.shape != expected_edge_shape:
            raise ValueError(
                "边缘探针概率形状 {}，期望 {}。".format(edges.shape, expected_edge_shape)
            )
        for edge_slot in range(edges.shape[0]):
            edge_values = edges[edge_slot]
            if np.all(np.isnan(edge_values)):
                continue
            if np.any(np.isnan(edge_values)):
                raise ValueError("边缘槽位 {} 只能整块为空，不能部分为 NaN。".format(edge_slot))
            validate_probability_tensor(edge_values[None, :, :], "边缘探针概率")

        cloud = np.asarray(cloud_probabilities, dtype=np.float64)
        expected_cloud_shape = self.cloud_probabilities.shape[1:]
        if cloud.shape != expected_cloud_shape:
            raise ValueError(
                "云端探针概率形状 {}，期望 {}。".format(cloud.shape, expected_cloud_shape)
            )
        validate_probability_tensor(cloud[None, :, :], "云端探针概率")
        return clients, edges, cloud

    def _build_summary_row(self, global_epoch, clients, edges, cloud, active_mask):
        """按照候选、活跃、边缘和云端四个口径构造一行摘要。"""
        labels = self.probe_set.true_labels
        candidate_metrics = calculate_population_probe_metrics(clients, labels)
        active_metrics = calculate_population_probe_metrics(clients[active_mask], labels)
        valid_edge_mask = np.all(np.isfinite(edges), axis=(1, 2))
        edge_metrics = calculate_population_probe_metrics(edges[valid_edge_mask], labels)

        active_count = int(np.sum(active_mask))
        candidate_count = int(active_mask.shape[0])
        active_coverage = float(active_count) / float(candidate_count)
        active_correct = active_metrics["correct_effective_mean"]
        if np.isfinite(active_correct):
            coverage_weighted = active_coverage * float(active_correct)
        else:
            coverage_weighted = np.nan

        cloud_labels = np.argmax(cloud, axis=1)
        cloud_accuracy = float(np.mean(cloud_labels == labels))
        true_class_probabilities = cloud[np.arange(labels.shape[0]), labels]
        row = {
            "global_epoch": int(global_epoch),
            "probe_count": int(labels.shape[0]),
            "candidate_count": candidate_count,
            "active_count": active_count,
            "active_coverage": active_coverage,
            "candidate_agreement_mean": candidate_metrics["agreement_mean"],
            "candidate_certainty_mean": candidate_metrics["certainty_mean"],
            "candidate_effective_mean": candidate_metrics["effective_mean"],
            "candidate_correct_effective_mean": candidate_metrics["correct_effective_mean"],
            "candidate_wrong_effective_mean": candidate_metrics["wrong_effective_mean"],
            "candidate_effective_q25": candidate_metrics["effective_q25"],
            "candidate_effective_q50": candidate_metrics["effective_q50"],
            "candidate_effective_q75": candidate_metrics["effective_q75"],
            "active_agreement_mean": active_metrics["agreement_mean"],
            "active_certainty_mean": active_metrics["certainty_mean"],
            "active_effective_mean": active_metrics["effective_mean"],
            "active_correct_effective_mean": active_metrics["correct_effective_mean"],
            "active_wrong_effective_mean": active_metrics["wrong_effective_mean"],
            "active_effective_q25": active_metrics["effective_q25"],
            "active_effective_q50": active_metrics["effective_q50"],
            "active_effective_q75": active_metrics["effective_q75"],
            "coverage_weighted_active_correct_effective": coverage_weighted,
            "edge_effective_mean": edge_metrics["effective_mean"],
            "edge_correct_effective_mean": edge_metrics["correct_effective_mean"],
            "cloud_probe_accuracy": cloud_accuracy,
            "cloud_true_class_probability_mean": float(np.mean(true_class_probabilities)),
        }
        return row

    def record_epoch(
            self,
            global_epoch,
            client_probabilities,
            edge_probabilities,
            cloud_probabilities,
            active_client_ids,
    ):
        """记录一个已完成 epoch，并按配置间隔原子更新压缩 NPZ。"""
        if self._closed:
            raise RuntimeError("探针记录器已经关闭。")
        row_index = self.completed_epochs
        if row_index >= self.total_epochs:
            raise ValueError("记录的 epoch 数超过配置中的总 epoch 数。")
        if int(global_epoch) != row_index:
            raise ValueError(
                "探针 epoch 必须从 0 连续记录；当前应为 {}，实际为 {}。".format(
                    row_index, global_epoch
                )
            )
        clients, edges, cloud = self._validate_epoch_probabilities(
            client_probabilities, edge_probabilities, cloud_probabilities
        )

        active_id_set = {int(value) for value in active_client_ids}
        active_mask = np.asarray(
            [int(client_id) in active_id_set for client_id in self.candidate_client_ids],
            dtype=np.bool_,
        )
        unknown_active_ids = active_id_set.difference(
            int(value) for value in self.candidate_client_ids
        )
        if unknown_active_ids:
            raise ValueError(
                "活跃客户端不在固定候选集合中：{}。".format(
                    sorted(unknown_active_ids)
                )
            )

        self.client_probabilities[row_index] = clients.astype(np.float32)
        self.edge_probabilities[row_index] = edges.astype(np.float32)
        self.cloud_probabilities[row_index] = cloud.astype(np.float32)
        self.active_client_mask[row_index] = active_mask
        self.edge_active_mask[row_index] = np.all(
            np.isfinite(edges), axis=(1, 2)
        )
        self.global_epochs[row_index] = int(global_epoch)
        summary_row = self._build_summary_row(
            global_epoch, clients, edges, cloud, active_mask
        )
        self._summary_writer.writerow({
            column: _format_csv_value(summary_row.get(column))
            for column in self.SUMMARY_COLUMNS
        })
        self._summary_file.flush()
        self.completed_epochs += 1
        if self.completed_epochs % self.checkpoint_interval == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        """把已完成前缀写入临时文件，再原子替换正式压缩 NPZ。"""
        completed = int(self.completed_epochs)
        temporary_path = self.npz_path + ".tmp"
        payload = {
            "schema_version": np.asarray("fixed_probe_v1"),
            "client_probabilities": self.client_probabilities[:completed],
            "edge_probabilities": self.edge_probabilities[:completed],
            "cloud_probabilities": self.cloud_probabilities[:completed],
            "active_client_mask": self.active_client_mask[:completed],
            "edge_active_mask": self.edge_active_mask[:completed],
            "client_ids": self.candidate_client_ids,
            "probe_indices": self.probe_set.indices,
            "true_labels": self.probe_set.true_labels,
            "global_epochs": self.global_epochs[:completed],
            "completed_epochs": np.asarray(completed, dtype=np.int64),
            "probe_set_hash": np.asarray(self.probe_set.content_hash),
            "probe_source": np.asarray(self.probe_set.source),
        }
        try:
            with open(temporary_path, "wb") as file_obj:
                np.savez_compressed(file_obj, **payload)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary_path, self.npz_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        logging.info(
            "固定探针检查点已保存：%s（已完成 %s 个 epoch）",
            self.npz_path,
            completed,
        )

    def close(self):
        """在正常结束或异常退出时保存已完成前缀并关闭摘要文件。"""
        if self._closed:
            return
        try:
            self.save_checkpoint()
        finally:
            self._summary_file.flush()
            self._summary_file.close()
            self._closed = True

    def metadata(self):
        """返回可写入实验元数据文件的固定探针配置与形状信息。"""
        return {
            "probe_output_format": "npz",
            "probe_source": self.probe_set.source,
            "probe_sample_count": self.probe_set.sample_count,
            "probe_samples_per_class": int(
                self.probe_set.sample_count // self.probe_set.class_count
            ),
            "probe_class_count": self.class_count,
            "probe_set_hash": self.probe_set.content_hash,
            "probe_npz_file": os.path.basename(self.npz_path),
            "probe_summary_file": os.path.basename(self.summary_path),
            "probe_npz_client_shape": [
                self.total_epochs,
                int(self.candidate_client_ids.shape[0]),
                self.probe_set.sample_count,
                self.class_count,
            ],
            "probe_npz_edge_shape": [
                self.total_epochs,
                self.edge_slot_count,
                self.probe_set.sample_count,
                self.class_count,
            ],
            "probe_npz_cloud_shape": [
                self.total_epochs,
                self.probe_set.sample_count,
                self.class_count,
            ],
        }
