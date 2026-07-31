"""面向 FEMNIST 固定37人和MAT拓扑的GPU快速分层联邦训练器。"""

from __future__ import absolute_import

from collections import OrderedDict
import contextlib
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn

from FEMNISTProbe.metrics import SUMMARY_COLUMNS, summarize_probe_observation
from FEMNISTProbe.monitor import GPUMonitor
from FEMNISTProbe.streaming_probe import (
    ProbeObservation,
    StreamingProbeH5Writer,
)


TEST_COLUMNS = [
    "global_epoch",
    "topology_cycle_index",
    "mat_topology_index",
    "evaluated_client_count",
    "test_samples",
    "test_correct",
    "test_accuracy",
    "test_loss",
]

TIMING_COLUMNS = [
    "global_epoch",
    "train_seconds",
    "aggregate_seconds",
    "probe_seconds",
    "test_seconds",
    "io_seconds",
    "checkpoint_seconds",
    "elapsed_seconds",
]


def _format_csv_value(value):
    """把数值转换为稳定CSV文本，非有限浮点数写为空单元格。"""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return ""
    return "{:.12g}".format(number)


def _hash_file(path):
    """以分块方式计算文件SHA-256。"""
    digest = hashlib.sha256()
    with open(str(path), "rb") as file_obj:
        while True:
            block = file_obj.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hash_state_dict(state_dict):
    """按字段顺序计算模型状态的稳定SHA-256。"""
    digest = hashlib.sha256()
    digest.update(b"pytorch-state-dict-v1\0")
    for name, value in state_dict.items():
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class FastFEMNISTMatTrainer:
    """执行固定候选、循环MAT拓扑、稀疏探针和可恢复GPU训练。"""

    def __init__(self, args, device, data_bundle, model, cyclic_topology):
        """初始化模型、GPU缓存、输出目录和可恢复训练状态。"""
        self.args = args
        self.device = torch.device(device)
        self.data = data_bundle
        self.model = model
        self.topology = cyclic_topology
        self.candidate_count = len(self.data.candidate_client_ids)
        self.population_client_count = int(self.data.population_client_count)
        self.class_count = int(self.data.class_count)
        self.comm_round = int(self.args.comm_round)
        self.eval_interval = int(getattr(self.args, "eval_interval", 50))
        self.checkpoint_interval = int(
            getattr(self.args, "checkpoint_interval", self.eval_interval)
        )
        self.local_batch_size = int(getattr(self.args, "batch_size", 20))
        self.learning_rate = float(getattr(self.args, "lr", 0.001))
        self.training_seed = int(getattr(self.args, "random_seed", 0))
        self.amp_requested = bool(getattr(self.args, "amp_enabled", True))
        self.amp_enabled = self.amp_requested and self.device.type == "cuda"
        self.reference_baseline = bool(
            getattr(self.args, "reference_baseline", False)
        )
        self.architecture = self.topology.schedule.architecture
        self.edge_slot_count = max(1, int(self.topology.schedule.group_capacity))
        self._validate_configuration()
        self.result_dir, self.resume_checkpoint = self._resolve_result_directory()
        self._configure_device()
        self._prepare_model()
        self._prepare_cached_data()
        self.observation_epochs = self._build_observation_epochs()
        self.stage_seconds = {
            "train": 0.0,
            "aggregate": 0.0,
            "probe": 0.0,
            "test": 0.0,
            "io": 0.0,
            "checkpoint": 0.0,
        }
        self.amp_scale_backoff_count = 0
        self.amp_consecutive_backoffs = 0
        self.amp_max_consecutive_backoffs = 0
        self.start_epoch = 0
        self.resume_observation_count = 0
        if self.resume_checkpoint is not None:
            self._load_checkpoint(self.resume_checkpoint)
        self._write_metadata(status="initialized")

    def _validate_configuration(self):
        """校验正式实验不变量，避免快速路径静默改变算法。"""
        if self.candidate_count != 37:
            raise ValueError("FEMNIST MAT实验固定要求37个候选客户端。")
        if self.population_client_count != 250:
            raise ValueError("FEMNIST MAT实验固定要求250个逻辑客户端。")
        if self.topology.schedule.assignment_mode != "balanced_counts":
            raise ValueError("FEMNIST 250客户端实验必须使用balanced_counts拓扑模式。")
        if self.local_batch_size != 20:
            raise ValueError("正式快速路径固定要求本地batch_size=20。")
        if str(getattr(self.args, "client_optimizer", "sgd")).lower() != "sgd":
            raise ValueError("GPU快速路径只支持无动量SGD。")
        if self.comm_round <= 0 or self.eval_interval <= 0:
            raise ValueError("comm_round和eval_interval必须大于0。")
        if self.checkpoint_interval != self.eval_interval:
            raise ValueError("检查点间隔必须与评估间隔一致，保证原子恢复。")
        if self.topology.repeat_mode != "cycle" and (
                self.comm_round > self.topology.round_count
        ):
            raise ValueError("5000轮实验必须显式启用topology_repeat_mode=cycle。")

    def _resolve_result_directory(self):
        """创建新结果目录，或从检查点父目录继续训练。"""
        resume_value = str(getattr(self.args, "resume_checkpoint", "")).strip()
        if resume_value:
            checkpoint_path = Path(resume_value).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError("找不到恢复检查点：{}".format(checkpoint_path))
            return checkpoint_path.parent, checkpoint_path

        project_root = Path(__file__).resolve().parents[1]
        configured_root = Path(str(
            getattr(self.args, "result_root", "result/FEMNISTProbe")
        ))
        result_root = (
            configured_root
            if configured_root.is_absolute()
            else project_root / configured_root
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = self.topology.schedule.scenario_name
        experiment_tag = str(getattr(self.args, "experiment_tag", "formal5000"))
        result_dir = result_root / "{}_{}_seed{}_{}".format(
            scenario_name, experiment_tag, self.training_seed, timestamp
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        return result_dir, None

    def _configure_device(self):
        """启用PyTorch 1.13可用的CUDA、TF32和cuDNN优化。"""
        if bool(getattr(self.args, "using_gpu", True)):
            if not torch.cuda.is_available():
                raise RuntimeError("配置要求GPU，但当前PyTorch无法访问CUDA。")
            if self.device.type != "cuda":
                raise RuntimeError("配置要求GPU，但实际设备为{}。".format(self.device))
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            # 与项目既有设置一致，仅从确定性算法集合中选择固定形状最快实现。
            torch.backends.cudnn.deterministic = True

    def _prepare_model(self):
        """把模型放入目标设备并建立扁平参数坐标和复用优化器。"""
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.to(memory_format=torch.channels_last)
        self.parameter_items = list(self.model.named_parameters())
        if not self.parameter_items:
            raise ValueError("模型没有可训练参数。")
        self.parameter_offsets = []
        next_offset = 0
        for name, parameter in self.parameter_items:
            parameter_count = int(parameter.numel())
            self.parameter_offsets.append(
                (name, parameter, next_offset, next_offset + parameter_count)
            )
            next_offset += parameter_count
        self.parameter_count = next_offset
        nonempty_buffers = [
            name for name, buffer in self.model.named_buffers()
            if int(buffer.numel()) > 0
        ]
        if nonempty_buffers:
            raise ValueError(
                "当前扁平快速路径要求模型没有持久缓冲区，实际为{}。".format(
                    nonempty_buffers
                )
            )
        self.cloud_vector = torch.empty(
            self.parameter_count, dtype=torch.float32, device=self.device
        )
        self._copy_model_to_vector(self.cloud_vector)
        self.initial_model_hash = _hash_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _copy_vector_to_model(self, vector):
        """把扁平FP32向量复制到共享模型参数，不建立共享存储视图。"""
        with torch.no_grad():
            for _, parameter, start, end in self.parameter_offsets:
                parameter.copy_(vector[start:end].view_as(parameter))

    def _copy_model_to_vector(self, output_vector):
        """把共享模型参数复制到预分配扁平FP32向量。"""
        with torch.no_grad():
            for _, parameter, start, end in self.parameter_offsets:
                output_vector[start:end].copy_(
                    parameter.detach().reshape(-1).float()
                )

    def _gpu_profile(self):
        """根据显存返回4090D级或4060 Laptop级推理配置。"""
        if self.device.type != "cuda":
            return {
                "name": "cpu",
                "probe_batch_size": int(getattr(
                    self.args, "probe_inference_batch_size", 256
                )),
                "test_batch_size": int(getattr(
                    self.args, "test_inference_batch_size", 1024
                )),
            }
        total_memory = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        large_memory = total_memory >= 16 * 1024 ** 3
        return {
            "name": "large_24gb" if large_memory else "compact_8gb",
            "probe_batch_size": 620 if large_memory else 256,
            "test_batch_size": 4096 if large_memory else 1024,
        }

    def _prepare_cached_data(self):
        """把候选训练数据、探针和可容纳的完整测试集缓存到目标设备。"""
        self.profile = self._gpu_profile()
        self.candidate_inputs = []
        self.candidate_labels = []
        self.candidate_cached_on_device = not self.reference_baseline
        for inputs, labels in zip(
                self.data.candidate_train_inputs,
                self.data.candidate_train_labels,
        ):
            if self.reference_baseline and self.device.type == "cuda":
                # 受控参考路径模拟旧式逐批CPU到GPU复制，仅用于100轮性能对照。
                self.candidate_inputs.append(inputs.pin_memory())
                self.candidate_labels.append(labels.pin_memory())
            else:
                device_inputs = inputs.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                if self.device.type == "cuda":
                    device_inputs = device_inputs.contiguous(
                        memory_format=torch.channels_last
                    )
                self.candidate_inputs.append(device_inputs)
                self.candidate_labels.append(labels.to(
                    self.device, non_blocking=self.device.type == "cuda"
                ))
        self.probe_inputs = self.data.probe_inputs.to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        if self.device.type == "cuda":
            self.probe_inputs = self.probe_inputs.contiguous(
                memory_format=torch.channels_last
            )

        self.test_cached_on_device = True
        if self.device.type == "cuda":
            free_memory, _ = torch.cuda.mem_get_info(self.device)
            required = (
                self.data.global_test_inputs.numel()
                * self.data.global_test_inputs.element_size()
                + self.data.global_test_labels.numel()
                * self.data.global_test_labels.element_size()
                + self.data.global_test_client_ids.numel()
                * self.data.global_test_client_ids.element_size()
            )
            reserve = int(getattr(
                self.args, "gpu_memory_reserve_mb", 2048
            )) * 1024 ** 2
            self.test_cached_on_device = free_memory - required >= reserve
        if self.test_cached_on_device:
            self.global_test_inputs = self.data.global_test_inputs.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            if self.device.type == "cuda":
                self.global_test_inputs = self.global_test_inputs.contiguous(
                    memory_format=torch.channels_last
                )
            self.global_test_labels = self.data.global_test_labels.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            self.global_test_client_ids = (
                self.data.global_test_client_ids.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
            )
        else:
            self.global_test_inputs = self.data.global_test_inputs.pin_memory()
            self.global_test_labels = self.data.global_test_labels.pin_memory()
            self.global_test_client_ids = (
                self.data.global_test_client_ids.pin_memory()
            )

    def _autocast(self):
        """返回兼容PyTorch 1.13的CUDA AMP上下文。"""
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=self.amp_enabled)
        return contextlib.nullcontext()

    def _build_observation_epochs(self):
        """构造训练前基线与固定间隔评估时间点。"""
        epochs = [-1]
        epochs.extend(
            epoch - 1
            for epoch in range(
                self.eval_interval, self.comm_round + 1, self.eval_interval
            )
        )
        if epochs[-1] != self.comm_round - 1:
            epochs.append(self.comm_round - 1)
        return tuple(sorted(set(epochs)))

    def _seed_client_epoch(self, client_slot, global_epoch):
        """为客户端和epoch生成与场景无关的训练及dropout随机种子。"""
        seed = (
            self.training_seed * 1000003
            + int(global_epoch) * 10007
            + int(client_slot) * 101
        ) % (2 ** 31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return seed

    def _train_active_clients(self, active_slots, global_epoch):
        """从同一云模型顺序训练活跃客户端并返回扁平本地参数矩阵。"""
        active_slots = [int(value) for value in active_slots]
        local_vectors = torch.empty(
            (len(active_slots), self.parameter_count),
            dtype=torch.float32,
            device=self.device,
        )
        for row_index, client_slot in enumerate(active_slots):
            self._copy_vector_to_model(self.cloud_vector)
            self.model.train()
            optimizer = self.optimizer
            if self.reference_baseline:
                # 参考路径重复创建优化器，用来量化正式复用优化器的收益。
                optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.learning_rate
                )
            seed = self._seed_client_epoch(client_slot, global_epoch)
            inputs = self.candidate_inputs[client_slot]
            labels = self.candidate_labels[client_slot]
            permutation = np.random.RandomState(seed).permutation(
                int(labels.shape[0])
            )
            permutation = torch.from_numpy(
                permutation.astype(np.int64)
            )
            if not self.reference_baseline or self.device.type != "cuda":
                permutation = permutation.to(self.device)
            for start in range(0, int(labels.shape[0]), self.local_batch_size):
                batch_indexes = permutation[start:start + self.local_batch_size]
                batch_inputs = inputs.index_select(0, batch_indexes)
                batch_labels = labels.index_select(0, batch_indexes)
                if self.reference_baseline and self.device.type == "cuda":
                    batch_inputs = batch_inputs.to(
                        self.device, non_blocking=True
                    ).contiguous(memory_format=torch.channels_last)
                    batch_labels = batch_labels.to(
                        self.device, non_blocking=True
                    )
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits = self.model(batch_inputs)
                    loss = self.criterion(logits, batch_labels)
                if self.amp_enabled:
                    previous_scale = float(self.scaler.get_scale())
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    current_scale = float(self.scaler.get_scale())
                    if current_scale < previous_scale:
                        self.amp_scale_backoff_count += 1
                        self.amp_consecutive_backoffs += 1
                        self.amp_max_consecutive_backoffs = max(
                            self.amp_max_consecutive_backoffs,
                            self.amp_consecutive_backoffs,
                        )
                    else:
                        self.amp_consecutive_backoffs = 0
                else:
                    loss.backward()
                    optimizer.step()
            if self.reference_baseline:
                # 深复制CPU状态模拟旧实现的state_dict复制和设备往返。
                copied_state = copy.deepcopy({
                    name: value.detach().cpu()
                    for name, value in self.model.state_dict().items()
                })
                flat_cpu = torch.cat([
                    copied_state[name].reshape(-1).float()
                    for name, _, _, _ in self.parameter_offsets
                ])
                local_vectors[row_index].copy_(
                    flat_cpu.to(self.device, non_blocking=False)
                )
            else:
                self._copy_model_to_vector(local_vectors[row_index])
        if not torch.isfinite(local_vectors).all():
            raise FloatingPointError("本地训练参数包含NaN或无穷值。")
        return local_vectors

    def _aggregate_vectors(self, vectors, sample_counts):
        """在GPU上按真实训练样本数加权聚合扁平参数。"""
        counts = torch.as_tensor(
            sample_counts, dtype=torch.float32, device=self.device
        )
        if vectors.ndim != 2 or vectors.shape[0] != counts.shape[0]:
            raise ValueError("聚合参数矩阵与样本数数量不一致。")
        if vectors.shape[0] == 0 or torch.any(counts <= 0):
            raise ValueError("聚合至少需要一个正样本数客户端。")
        weights = counts / torch.sum(counts)
        return torch.sum(vectors * weights[:, None], dim=0)

    def _aggregate_round(self, active_slots, local_vectors, groups):
        """按HFL两级或FL直接模式聚合，并返回边缘与云参数。"""
        slot_to_row = {
            int(slot): row for row, slot in enumerate(active_slots)
        }
        sample_counts = self.data.candidate_train_sample_counts
        if self.architecture == "fl":
            rows = [slot_to_row[int(slot)] for slot in active_slots]
            cloud = self._aggregate_vectors(
                local_vectors[rows],
                [sample_counts[int(slot)] for slot in active_slots],
            )
            return {}, cloud

        edge_vectors = {}
        edge_sample_counts = []
        ordered_edges = []
        for group_index in sorted(groups):
            slots = [int(slot) for slot in groups[group_index]]
            rows = [slot_to_row[slot] for slot in slots]
            counts = [sample_counts[slot] for slot in slots]
            edge_vector = self._aggregate_vectors(local_vectors[rows], counts)
            edge_vectors[int(group_index)] = edge_vector
            ordered_edges.append(edge_vector)
            edge_sample_counts.append(int(np.sum(counts)))
        if not ordered_edges:
            raise ValueError("HFL轮次没有可聚合的边缘模型。")
        cloud = self._aggregate_vectors(
            torch.stack(ordered_edges, dim=0), edge_sample_counts
        )
        return edge_vectors, cloud

    def _predict_vector(self, model_vector, inputs, batch_size):
        """加载扁平模型并分块返回float32概率NumPy数组。"""
        self._copy_vector_to_model(model_vector)
        self.model.eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0, int(inputs.shape[0]), int(batch_size)):
                batch = inputs[start:start + int(batch_size)]
                with self._autocast():
                    logits = self.model(batch)
                probabilities.append(
                    torch.softmax(logits.float(), dim=1).cpu().numpy()
                )
        result = np.concatenate(probabilities, axis=0).astype(
            np.float32, copy=False
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("探针概率包含NaN或无穷值。")
        if not np.allclose(np.sum(result, axis=1), 1.0, atol=1e-5):
            raise FloatingPointError("探针概率行和不为1。")
        return result

    def _build_probe_observation(
            self,
            global_epoch,
            cyclic_round,
            active_slots,
            local_vectors,
            edge_vectors,
            cloud_before,
            cloud_after,
    ):
        """构造客户端、边缘、云概率和活跃掩码。"""
        probe_batch = int(self.profile["probe_batch_size"])
        active_slots = [int(value) for value in active_slots]
        slot_to_row = {
            slot: row for row, slot in enumerate(active_slots)
        }
        client_probabilities = []
        for client_slot in range(self.candidate_count):
            vector = (
                local_vectors[slot_to_row[client_slot]]
                if client_slot in slot_to_row
                else cloud_before
            )
            client_probabilities.append(self._predict_vector(
                vector, self.probe_inputs, probe_batch
            ))
        client_probabilities = np.stack(client_probabilities, axis=0)

        probe_count = int(self.probe_inputs.shape[0])
        edge_probabilities = np.full(
            (
                self.edge_slot_count,
                probe_count,
                self.class_count,
            ),
            np.nan,
            dtype=np.float32,
        )
        edge_mask = np.zeros(self.edge_slot_count, dtype=np.bool_)
        for group_index, vector in edge_vectors.items():
            edge_probabilities[int(group_index)] = self._predict_vector(
                vector, self.probe_inputs, probe_batch
            )
            edge_mask[int(group_index)] = True
        cloud_probabilities = self._predict_vector(
            cloud_after, self.probe_inputs, probe_batch
        )
        active_mask = np.zeros(self.candidate_count, dtype=np.bool_)
        active_mask[np.asarray(active_slots, dtype=np.int64)] = True
        return ProbeObservation(
            global_epoch=int(global_epoch),
            topology_cycle_index=int(
                cyclic_round.topology_cycle_index
            ),
            mat_topology_index=int(cyclic_round.mat_topology_index),
            client_probabilities=client_probabilities,
            edge_probabilities=edge_probabilities,
            cloud_probabilities=cloud_probabilities,
            active_client_mask=active_mask,
            edge_active_mask=edge_mask,
        )

    def _build_baseline_observation(self):
        """构造训练前全部候选共享初始云模型的基线探针。"""
        probe_batch = int(self.profile["probe_batch_size"])
        cloud = self._predict_vector(
            self.cloud_vector, self.probe_inputs, probe_batch
        )
        clients = np.repeat(
            cloud[None, :, :], self.candidate_count, axis=0
        )
        edges = np.full(
            (
                self.edge_slot_count,
                cloud.shape[0],
                cloud.shape[1],
            ),
            np.nan,
            dtype=np.float32,
        )
        return ProbeObservation(
            global_epoch=-1,
            topology_cycle_index=-1,
            mat_topology_index=-1,
            client_probabilities=clients,
            edge_probabilities=edges,
            cloud_probabilities=cloud,
            active_client_mask=np.zeros(
                self.candidate_count, dtype=np.bool_
            ),
            edge_active_mask=np.zeros(
                self.edge_slot_count, dtype=np.bool_
            ),
        )

    def _evaluate_cloud(self, global_epoch, cycle_index, mat_index):
        """按 250 个本地测试分区累计正确数并评估当前共享云模型。"""
        self._copy_vector_to_model(self.cloud_vector)
        self.model.eval()
        batch_size = int(self.profile["test_batch_size"])
        loss_sum = 0.0
        total = int(self.data.global_test_labels.shape[0])
        client_correct = torch.zeros(
            self.population_client_count,
            dtype=torch.int64,
            device=self.device,
        )
        client_seen = torch.zeros_like(client_correct)
        with torch.inference_mode():
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                inputs = self.global_test_inputs[start:end]
                labels = self.global_test_labels[start:end]
                client_ids = self.global_test_client_ids[start:end]
                if not self.test_cached_on_device:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    client_ids = client_ids.to(self.device, non_blocking=True)
                if self.device.type == "cuda":
                    inputs = inputs.contiguous(
                        memory_format=torch.channels_last
                    )
                with self._autocast():
                    logits = self.model(inputs)
                logits_float = logits.float()
                loss_sum += float(nn.functional.cross_entropy(
                    logits_float, labels, reduction="sum"
                ).item())
                correct_mask = torch.argmax(logits_float, dim=1) == labels
                client_seen += torch.bincount(
                    client_ids,
                    minlength=self.population_client_count,
                )
                client_correct += torch.bincount(
                    client_ids[correct_mask],
                    minlength=self.population_client_count,
                )
        actual_client_counts = client_seen.detach().cpu().numpy()
        expected_client_counts = np.asarray(
            self.data.client_test_sample_counts, dtype=np.int64
        )
        if not np.array_equal(actual_client_counts, expected_client_counts):
            raise RuntimeError("250个本地测试分区的评估样本数与数据清单不一致。")
        correct = int(torch.sum(client_correct).item())
        if int(torch.sum(client_seen).item()) != total:
            raise RuntimeError("本地测试分区汇总没有覆盖完整测试集。")
        return {
            "global_epoch": int(global_epoch),
            "topology_cycle_index": int(cycle_index),
            "mat_topology_index": int(mat_index),
            "evaluated_client_count": self.population_client_count,
            "test_samples": total,
            "test_correct": correct,
            "test_accuracy": float(correct) / float(total),
            "test_loss": float(loss_sum) / float(total),
        }

    def _write_csv_header(self, path, columns):
        """为新CSV创建表头；恢复运行保留既有文件。"""
        if path.is_file():
            return
        with path.open("w", encoding="utf-8", newline="") as file_obj:
            csv.DictWriter(file_obj, fieldnames=columns).writeheader()

    def _append_csv_row(self, path, columns, row):
        """以稳定文本格式追加一行CSV并立即刷新。"""
        with path.open("a", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=columns)
            writer.writerow({
                key: _format_csv_value(row.get(key))
                for key in columns
            })
            file_obj.flush()

    def _truncate_csv(self, path, keep_data_rows):
        """恢复时把CSV截断到检查点已提交的数据行。"""
        if not path.is_file():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        kept = lines[:1 + int(keep_data_rows)]
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def _truncate_schedule(self, keep_epochs):
        """恢复时只保留检查点之前完整提交的调度记录。"""
        path = self.result_dir / "topology_schedule.jsonl"
        if not path.is_file():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join(lines[:int(keep_epochs)])
            + ("\n" if int(keep_epochs) > 0 else ""),
            encoding="utf-8",
        )

    def _checkpoint_payload(self, next_epoch, observation_count):
        """构造可在epoch边界精确恢复的检查点字典。"""
        next_cycle_index = int(next_epoch) // self.topology.round_count
        next_mat_index = int(next_epoch) % self.topology.round_count
        payload = {
            "schema_version": "femnist_probe_checkpoint_v2",
            "next_epoch": int(next_epoch),
            "next_topology_cycle_index": next_cycle_index,
            "next_mat_topology_index": next_mat_index,
            "observation_count": int(observation_count),
            "cloud_vector": self.cloud_vector.detach().cpu(),
            "scaler_state": copy.deepcopy(self.scaler.state_dict()),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "candidate_manifest_hash": self.data.candidate_manifest_hash,
            "partition_hash": self.data.partition_hash,
            "probe_hash": self.data.probe_hash,
            "initial_model_hash": self.initial_model_hash,
            "mat_file_hash": self.mat_file_hash,
            "scenario": self.topology.schedule.scenario_name,
            "amp_enabled": bool(self.amp_enabled),
            "reference_baseline": bool(self.reference_baseline),
            "amp_scale_backoff_count": int(
                self.amp_scale_backoff_count
            ),
            "amp_consecutive_backoffs": int(
                self.amp_consecutive_backoffs
            ),
            "amp_max_consecutive_backoffs": int(
                self.amp_max_consecutive_backoffs
            ),
        }
        if self.device.type == "cuda":
            payload["cuda_random_states"] = torch.cuda.get_rng_state_all()
        return payload

    def _save_checkpoint(self, next_epoch, observation_count):
        """原子保存最新检查点，并在成功后替换旧文件。"""
        output_path = self.result_dir / "checkpoint_latest.pt"
        temporary_path = self.result_dir / "checkpoint_latest.pt.tmp"
        torch.save(
            self._checkpoint_payload(next_epoch, observation_count),
            str(temporary_path),
        )
        os.replace(str(temporary_path), str(output_path))

    def _load_checkpoint(self, checkpoint_path):
        """加载并校验公共哈希、场景、精度和恢复坐标。"""
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        if payload.get("schema_version") != "femnist_probe_checkpoint_v2":
            raise ValueError("旧版37书写者检查点不能用于250客户端狄利克雷实验。")
        expected = {
            "candidate_manifest_hash": self.data.candidate_manifest_hash,
            "partition_hash": self.data.partition_hash,
            "probe_hash": self.data.probe_hash,
            "initial_model_hash": self.initial_model_hash,
            "mat_file_hash": self.mat_file_hash,
            "scenario": self.topology.schedule.scenario_name,
            "amp_enabled": bool(self.amp_enabled),
            "reference_baseline": bool(self.reference_baseline),
        }
        for key, expected_value in expected.items():
            if payload.get(key) != expected_value:
                raise ValueError("检查点字段{}与当前实验不一致。".format(key))
        self.cloud_vector.copy_(
            payload["cloud_vector"].to(self.device).float()
        )
        self.scaler.load_state_dict(payload.get("scaler_state", {}))
        self.amp_scale_backoff_count = int(
            payload.get("amp_scale_backoff_count", 0)
        )
        self.amp_consecutive_backoffs = int(
            payload.get("amp_consecutive_backoffs", 0)
        )
        self.amp_max_consecutive_backoffs = int(
            payload.get("amp_max_consecutive_backoffs", 0)
        )
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        if self.device.type == "cuda" and "cuda_random_states" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_random_states"])
        self.start_epoch = int(payload["next_epoch"])
        expected_cycle_index = (
            self.start_epoch // self.topology.round_count
        )
        expected_mat_index = (
            self.start_epoch % self.topology.round_count
        )
        if int(payload.get(
                "next_topology_cycle_index", -1
        )) != expected_cycle_index:
            raise ValueError("检查点MAT循环编号与next_epoch不一致。")
        if int(payload.get(
                "next_mat_topology_index", -1
        )) != expected_mat_index:
            raise ValueError("检查点MAT行索引与next_epoch不一致。")
        self.resume_observation_count = int(payload["observation_count"])
        self._truncate_schedule(self.start_epoch)
        self._truncate_csv(
            self.result_dir / "probe_epoch_summary.csv",
            self.resume_observation_count,
        )
        self._truncate_csv(
            self.result_dir / "test_metrics.csv",
            self.resume_observation_count,
        )
        self._truncate_csv(
            self.result_dir / "stage_timing.csv",
            self.resume_observation_count,
        )

    @property
    def mat_file_hash(self):
        """返回源MAT文件SHA-256并缓存结果。"""
        if not hasattr(self, "_mat_file_hash"):
            self._mat_file_hash = _hash_file(
                self.topology.schedule.mat_path
            )
        return self._mat_file_hash

    def _write_metadata(self, status):
        """原子更新实验元数据和当前完成位置。"""
        metadata = self.topology.to_metadata()
        metadata.update({
            "schema_version": "femnist_mat_probe_fast_v2",
            "status": str(status),
            "dataset": "femnist",
            "class_count": self.class_count,
            "partition_method": "dirichlet",
            "partition_alpha": float(self.data.partition_alpha),
            "partition_seed": int(self.data.partition_seed),
            "partition_hash": self.data.partition_hash,
            "population_client_count": self.data.population_client_count,
            "source_writer_count": self.data.source_writer_count,
            "population_train_sample_count": int(
                self.data.population_train_sample_count
            ),
            "population_test_sample_count": int(
                self.data.population_test_sample_count
            ),
            "candidate_client_count": self.candidate_count,
            "candidate_client_ids": self.data.candidate_client_ids,
            "candidate_train_sample_counts": [
                int(value)
                for value in self.data.candidate_train_sample_counts
            ],
            "candidate_manifest_hash": self.data.candidate_manifest_hash,
            "probe_hash": self.data.probe_hash,
            "initial_model_hash": self.initial_model_hash,
            "mat_file_hash": self.mat_file_hash,
            "comm_round": self.comm_round,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "local_batch_size": self.local_batch_size,
            "learning_rate": self.learning_rate,
            "random_seed": self.training_seed,
            "amp_requested": self.amp_requested,
            "amp_enabled": self.amp_enabled,
            "amp_scale": float(self.scaler.get_scale()),
            "amp_scale_backoff_count": int(
                self.amp_scale_backoff_count
            ),
            "amp_max_consecutive_backoffs": int(
                self.amp_max_consecutive_backoffs
            ),
            "reference_baseline": self.reference_baseline,
            "device": str(self.device),
            "gpu_profile": self.profile,
            "candidate_cached_on_device": self.candidate_cached_on_device,
            "test_cached_on_device": self.test_cached_on_device,
            "observation_epochs": list(self.observation_epochs),
            "single_seed_descriptive_only": True,
        })
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            metadata.update({
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_compute_capability": [
                    int(properties.major), int(properties.minor)
                ],
                "torch_version": torch.__version__,
                "cuda_build_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            })
        output_path = self.result_dir / "experiment_metadata.json"
        temporary_path = self.result_dir / "experiment_metadata.json.tmp"
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(str(temporary_path), str(output_path))

    def _append_schedule(self, file_obj, cyclic_round, groups, active_slots):
        """写入 MAT k/n、固定槽位和 250 端全量同步的审计记录。"""
        row = {
            "global_epoch": int(cyclic_round.global_epoch),
            "topology_cycle_index": int(
                cyclic_round.topology_cycle_index
            ),
            "mat_topology_index": int(cyclic_round.mat_topology_index),
            "scenario": self.topology.schedule.scenario_name,
            "candidate_manifest_hash": self.data.candidate_manifest_hash,
            "partition_hash": self.data.partition_hash,
            "active_candidate_slots": [int(value) for value in active_slots],
            "active_client_ids": [
                self.data.candidate_client_ids[int(value)]
                for value in active_slots
            ],
            "mat_group_count": len(groups),
            "mat_participant_count": len(active_slots),
            "group_to_candidate_slots": {
                str(group_index): [int(value) for value in slots]
                for group_index, slots in groups.items()
            },
            "group_to_client_ids": {
                str(group_index): [
                    self.data.candidate_client_ids[int(value)]
                    for value in slots
                ]
                for group_index, slots in groups.items()
            },
            "synchronized_client_ids": list(
                range(self.population_client_count)
            ),
            "synchronized_client_count": self.population_client_count,
        }
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
        file_obj.flush()

    def _print_round_progress(
            self, global_epoch, cyclic_round, groups, active_slots
    ):
        """按旧训练器样式实时打印当前全局轮次及MAT调度详情。"""
        group_client_ids = {
            int(group_index): [
                int(self.data.candidate_client_ids[int(slot)])
                for slot in slots
            ]
            for group_index, slots in groups.items()
        }
        # 保留旧版运行日志的全局通信轮次标题，便于直接对照既有实验。
        print(
            "################Global Communication Round : {}".format(
                int(global_epoch)
            ),
            flush=True,
        )
        # 紧随标题打印本轮真实使用的MAT行、k/n和逻辑客户端分组。
        print(
            (
                "epoch={}/{}, scenario={}, mat_cycle={}, mat_round={}, "
                "k={}, n={}, groups={}"
            ).format(
                int(global_epoch) + 1,
                self.comm_round,
                self.topology.schedule.scenario_name,
                int(cyclic_round.topology_cycle_index),
                int(cyclic_round.mat_topology_index),
                len(groups),
                len(active_slots),
                group_client_ids,
            ),
            flush=True,
        )

    def _write_observation_rows(
            self, observation, groups, summary_path, test_path
    ):
        """计算并追加探针摘要和完整测试指标。"""
        summary = summarize_probe_observation(
            global_epoch=observation.global_epoch,
            topology_cycle_index=observation.topology_cycle_index,
            mat_topology_index=observation.mat_topology_index,
            client_probabilities=observation.client_probabilities,
            edge_probabilities=observation.edge_probabilities,
            cloud_probabilities=observation.cloud_probabilities,
            active_client_mask=observation.active_client_mask,
            edge_active_mask=observation.edge_active_mask,
            true_labels=self.data.probe_labels,
            groups=groups,
        )
        self._append_csv_row(summary_path, SUMMARY_COLUMNS, summary)
        test_row = self._evaluate_cloud(
            observation.global_epoch,
            observation.topology_cycle_index,
            observation.mat_topology_index,
        )
        self._append_csv_row(test_path, TEST_COLUMNS, test_row)
        return summary, test_row

    def train(self):
        """运行可恢复的5000轮训练，并返回本次结果目录。"""
        self.data.write_manifest(self.result_dir / "shared_manifest.json")
        summary_path = self.result_dir / "probe_epoch_summary.csv"
        test_path = self.result_dir / "test_metrics.csv"
        timing_path = self.result_dir / "stage_timing.csv"
        schedule_path = self.result_dir / "topology_schedule.jsonl"
        self._write_csv_header(summary_path, SUMMARY_COLUMNS)
        self._write_csv_header(test_path, TEST_COLUMNS)
        self._write_csv_header(timing_path, TIMING_COLUMNS)
        if not schedule_path.is_file():
            schedule_path.write_text("", encoding="utf-8")

        writer = StreamingProbeH5Writer(
            output_path=self.result_dir / "probe_probabilities.h5",
            observation_count=len(self.observation_epochs),
            candidate_count=self.candidate_count,
            edge_slot_count=self.edge_slot_count,
            probe_labels=self.data.probe_labels,
            probe_indices=self.data.probe_indices,
            class_count=self.class_count,
            probe_hash=self.data.probe_hash,
            resume_count=self.resume_observation_count,
            compression_level=int(
                getattr(self.args, "probe_compression_level", 4)
            ),
        )
        monitor = GPUMonitor(
            self.result_dir / "gpu_monitor.csv",
            gpu_index=int(getattr(self.args, "gpu_id", 0)),
            interval_seconds=int(
                getattr(self.args, "gpu_monitor_interval", 30)
            ),
        )
        start_time = time.perf_counter()
        observation_count = self.resume_observation_count
        self._write_metadata(status="running")
        monitor.start()
        try:
            with schedule_path.open(
                    "a", encoding="utf-8"
            ) as schedule_file:
                if self.start_epoch == 0 and observation_count == 0:
                    probe_start = time.perf_counter()
                    baseline = self._build_baseline_observation()
                    writer.submit(baseline)
                    self.stage_seconds["probe"] += (
                        time.perf_counter() - probe_start
                    )
                    test_start = time.perf_counter()
                    self._write_observation_rows(
                        baseline, {}, summary_path, test_path
                    )
                    self.stage_seconds["test"] += (
                        time.perf_counter() - test_start
                    )
                    observation_count += 1
                    io_start = time.perf_counter()
                    writer.flush()
                    self.stage_seconds["io"] += (
                        time.perf_counter() - io_start
                    )
                    checkpoint_start = time.perf_counter()
                    self._save_checkpoint(0, observation_count)
                    self.stage_seconds["checkpoint"] += (
                        time.perf_counter() - checkpoint_start
                    )
                    self._append_csv_row(
                        timing_path,
                        TIMING_COLUMNS,
                        {
                            "global_epoch": -1,
                            "train_seconds": self.stage_seconds["train"],
                            "aggregate_seconds": self.stage_seconds["aggregate"],
                            "probe_seconds": self.stage_seconds["probe"],
                            "test_seconds": self.stage_seconds["test"],
                            "io_seconds": self.stage_seconds["io"],
                            "checkpoint_seconds": self.stage_seconds["checkpoint"],
                            "elapsed_seconds": time.perf_counter() - start_time,
                        },
                    )

                for global_epoch in range(self.start_epoch, self.comm_round):
                    cyclic_round = self.topology.get_round(global_epoch)
                    round_topology = cyclic_round.topology
                    groups = round_topology.copy_groups()
                    active_slots = list(
                        round_topology.active_candidate_slots
                    )
                    self._print_round_progress(
                        global_epoch, cyclic_round, groups, active_slots
                    )
                    cloud_before = self.cloud_vector.detach().clone()

                    train_start = time.perf_counter()
                    local_vectors = self._train_active_clients(
                        active_slots, global_epoch
                    )
                    self.stage_seconds["train"] += (
                        time.perf_counter() - train_start
                    )

                    aggregate_start = time.perf_counter()
                    edge_vectors, cloud_after = self._aggregate_round(
                        active_slots, local_vectors, groups
                    )
                    self.cloud_vector.copy_(cloud_after)
                    self.stage_seconds["aggregate"] += (
                        time.perf_counter() - aggregate_start
                    )
                    self._append_schedule(
                        schedule_file, cyclic_round, groups, active_slots
                    )

                    if global_epoch in self.observation_epochs:
                        probe_start = time.perf_counter()
                        observation = self._build_probe_observation(
                            global_epoch,
                            cyclic_round,
                            active_slots,
                            local_vectors,
                            edge_vectors,
                            cloud_before,
                            self.cloud_vector,
                        )
                        writer.submit(observation)
                        self.stage_seconds["probe"] += (
                            time.perf_counter() - probe_start
                        )

                        test_start = time.perf_counter()
                        summary, test_row = self._write_observation_rows(
                            observation, groups, summary_path, test_path
                        )
                        self.stage_seconds["test"] += (
                            time.perf_counter() - test_start
                        )
                        observation_count += 1
                        io_start = time.perf_counter()
                        writer.flush()
                        self.stage_seconds["io"] += (
                            time.perf_counter() - io_start
                        )

                        checkpoint_start = time.perf_counter()
                        self._save_checkpoint(
                            global_epoch + 1, observation_count
                        )
                        self.stage_seconds["checkpoint"] += (
                            time.perf_counter() - checkpoint_start
                        )
                        self._append_csv_row(
                            timing_path,
                            TIMING_COLUMNS,
                            {
                                "global_epoch": global_epoch,
                                "train_seconds": self.stage_seconds["train"],
                                "aggregate_seconds": self.stage_seconds["aggregate"],
                                "probe_seconds": self.stage_seconds["probe"],
                                "test_seconds": self.stage_seconds["test"],
                                "io_seconds": self.stage_seconds["io"],
                                "checkpoint_seconds": self.stage_seconds["checkpoint"],
                                "elapsed_seconds": time.perf_counter() - start_time,
                            },
                        )
                        print(
                            "epoch={}/{}, test_acc={:.6f}, Q={:.6f}, S={:.6f}".format(
                                global_epoch + 1,
                                self.comm_round,
                                test_row["test_accuracy"],
                                summary[
                                    "coverage_weighted_active_correct_effective"
                                ],
                                summary["candidate_effective"],
                            ),
                            flush=True,
                        )
                        self._write_metadata(status="running")
        finally:
            writer.close()
            monitor.close()

        if observation_count != len(self.observation_epochs):
            raise RuntimeError(
                "完成探针时间点数{}与预期{}不一致。".format(
                    observation_count, len(self.observation_epochs)
                )
            )
        self._write_metadata(status="complete")
        return self.result_dir
