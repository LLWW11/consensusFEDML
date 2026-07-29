"""SevereTest 普通 FedAvg 与三边缘组分层 FedAvg 训练器。"""

from __future__ import absolute_import

import copy
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from client_test import HFLClient
from fedavg_test import FedAvgAPI
from probe_batch import ProbeBatchRecorder, select_fixed_balanced_probe


def validate_training_client_ids(training_client_ids, client_num_in_total):
    """校验固定训练客户端编号非空、唯一、升序且位于客户端池中。"""
    client_ids = [int(value) for value in training_client_ids]
    if not client_ids:
        raise ValueError("training_client_ids 不能为空。")
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("training_client_ids 不能包含重复编号。")
    if client_ids != sorted(client_ids):
        raise ValueError("training_client_ids 必须按升序配置。")
    if client_ids[0] < 0 or client_ids[-1] >= int(client_num_in_total):
        raise ValueError("training_client_ids 含有超出客户端池的编号。")
    return client_ids


def validate_hierarchical_groups(
        edge_client_groups, training_client_ids, client_num_in_total
):
    """校验边缘分组非空、组内升序、互不重叠并完整覆盖训练客户端。"""
    if isinstance(edge_client_groups, str):
        try:
            edge_client_groups = json.loads(edge_client_groups)
        except json.JSONDecodeError as exc:
            raise ValueError("edge_client_groups 不是合法 JSON 列表。") from exc
    groups = [
        validate_training_client_ids(group, client_num_in_total)
        for group in edge_client_groups
    ]
    if not groups:
        raise ValueError("分层联邦学习至少需要一个边缘组。")
    flattened = [client_id for group in groups for client_id in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("不同边缘组之间不能包含重复客户端。")
    if flattened != list(training_client_ids):
        raise ValueError("边缘分组必须按顺序完整覆盖 training_client_ids。")
    return groups


def aggregate_weighted_model_states(weighted_states):
    """按样本数加权聚合多个 PyTorch 模型状态并返回独立副本。"""
    if not weighted_states:
        raise ValueError("至少需要一个本地模型才能聚合。")
    sample_counts = [int(sample_count) for sample_count, _ in weighted_states]
    if any(sample_count <= 0 for sample_count in sample_counts):
        raise ValueError("参与聚合的客户端样本数必须大于 0。")
    total_samples = float(sum(sample_counts))
    reference_keys = list(weighted_states[0][1].keys())
    for _, model_state in weighted_states:
        if list(model_state.keys()) != reference_keys:
            raise ValueError("参与聚合的模型状态字段不一致。")

    aggregated_state = copy.deepcopy(weighted_states[0][1])
    for parameter_name in reference_keys:
        reference_value = weighted_states[0][1][parameter_name]
        if torch.is_floating_point(reference_value):
            accumulator = torch.zeros_like(reference_value)
            for sample_count, model_state in weighted_states:
                accumulator = accumulator + (
                    model_state[parameter_name] * (float(sample_count) / total_samples)
                )
            aggregated_state[parameter_name] = accumulator
        else:
            # 整型计数器不适合线性加权；同步 FedAvg 中保留首个客户端值。
            aggregated_state[parameter_name] = reference_value.clone()
    return aggregated_state


def aggregate_hierarchical_model_states(grouped_weighted_states):
    """先在每个边缘组内按样本数聚合，再按组样本数执行云端聚合。"""
    if not grouped_weighted_states:
        raise ValueError("至少需要一个边缘组才能执行分层聚合。")
    edge_states = []
    edge_sample_counts = []
    for group_index, weighted_states in enumerate(grouped_weighted_states):
        if not weighted_states:
            raise ValueError("边缘组 {} 不能为空。".format(group_index))
        edge_sample_count = sum(
            int(sample_count) for sample_count, _ in weighted_states
        )
        edge_states.append(aggregate_weighted_model_states(weighted_states))
        edge_sample_counts.append(edge_sample_count)

    # 云端使用每个边缘组覆盖的训练样本数作为二级 FedAvg 权重。
    cloud_state = aggregate_weighted_model_states(list(zip(
        edge_sample_counts, edge_states
    )))
    return edge_states, cloud_state, edge_sample_counts


class SevereFixedFedAvg(FedAvgAPI):
    """执行固定 10 人训练、200 人全量下发和逐轮探针记录。"""

    EXPECTED_TRAINING_CLIENT_IDS = tuple(range(10))
    SCHEMA_VERSION = "severe_single_label_v1"
    AGGREGATION_NAME = "sample_weighted_fedavg"

    TEST_METRIC_COLUMNS = [
        "global_epoch", "test_samples", "test_correct", "test_acc", "test_loss"
    ]
    CLASS_METRIC_COLUMNS = [
        "global_epoch", "label", "test_samples", "test_correct", "test_acc", "test_loss"
    ]

    def __init__(self, args, device, dataset, model, partition_manifest):
        """初始化配置、200 个持久客户端和划分清单。"""
        self.partition_manifest = list(partition_manifest)
        self.training_client_ids = validate_training_client_ids(
            getattr(args, "training_client_ids", self.EXPECTED_TRAINING_CLIENT_IDS),
            args.client_num_in_total,
        )
        expected_training_ids = list(self.EXPECTED_TRAINING_CLIENT_IDS)
        if self.training_client_ids != expected_training_ids:
            raise ValueError(
                "本实验要求 training_client_ids 严格等于 {}。".format(
                    expected_training_ids
                )
            )
        if int(args.client_num_per_round) != len(self.training_client_ids):
            raise ValueError(
                "client_num_per_round 必须等于固定训练客户端数量 {}。".format(
                    len(self.training_client_ids)
                )
            )
        if int(args.epochs) != 1 or int(args.group_comm_round) != 1:
            raise ValueError("SevereTest 固定要求 epochs=1 且 group_comm_round=1。")
        self.project_root = Path(__file__).resolve().parents[1]
        self.result_dir = None
        self.completed_epochs = 0
        super().__init__(args, device, dataset, model)

    def _setup_clients(
            self,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
    ):
        """为全部 200 个真实客户端创建唯一且持久的本地模型状态。"""
        self.client_registry = {}
        for client_id in range(int(self.args.client_num_in_total)):
            self.client_registry[client_id] = HFLClient(
                client_id,
                train_data_local_dict[client_id],
                test_data_local_dict[client_id],
                train_data_local_num_dict[client_id],
                self.args,
                self.device,
                self.model,
                model_trainer,
            )
        self.client_list = [
            self.client_registry[client_id]
            for client_id in range(int(self.args.client_num_in_total))
        ]

    def _create_result_dir(self):
        """创建不会覆盖历史实验的时间戳结果目录。"""
        configured_root = Path(
            str(getattr(self.args, "severe_result_root", "result/SevereTest"))
        )
        result_root = (
            configured_root
            if configured_root.is_absolute()
            else self.project_root / configured_root
        )
        experiment_name = str(
            getattr(self.args, "experiment_name", "mnist_single_label_first10")
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = result_root / "{}_{}".format(experiment_name, timestamp)
        result_dir.mkdir(parents=True, exist_ok=False)
        self.result_dir = result_dir
        self.args.result_dir = str(result_dir)
        return result_dir

    def _write_partition_manifest(self):
        """把 200 个客户端的标签和样本数写入 CSV。"""
        output_path = self.result_dir / "partition_manifest.csv"
        with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=["client_id", "label", "train_count", "test_count"],
            )
            writer.writeheader()
            writer.writerows(self.partition_manifest)

    def _build_probe_probabilities(self, probe_inputs, client_ids):
        """按固定客户端顺序构造三维探针概率张量。"""
        inference_batch_size = int(
            getattr(self.args, "probe_inference_batch_size", probe_inputs.shape[0])
        )
        probability_batches = []
        for client_id in client_ids:
            probability_batches.append(np.asarray(
                self.client_registry[client_id].predict_proba_batch(
                    probe_inputs,
                    inference_batch_size=inference_batch_size,
                ),
                dtype=np.float32,
            ))
        return np.stack(probability_batches, axis=0)

    def _build_cloud_probabilities(self, probe_inputs, cloud_state):
        """使用聚合后的云模型计算固定探针概率。"""
        return self._build_model_probabilities(probe_inputs, cloud_state)

    def _build_model_probabilities(self, probe_inputs, model_state):
        """使用给定模型状态计算固定探针概率，供边缘与云端共同复用。"""
        inference_batch_size = int(
            getattr(self.args, "probe_inference_batch_size", probe_inputs.shape[0])
        )
        return np.asarray(
            self.client_registry[0].predict_proba_batch(
                probe_inputs,
                model_state=model_state,
                inference_batch_size=inference_batch_size,
            ),
            dtype=np.float32,
        )

    def _distribute_cloud_model(self, cloud_state):
        """把云模型显式下发到全部 200 个客户端。"""
        distributed_client_ids = list(range(int(self.args.client_num_in_total)))
        for client_id in distributed_client_ids:
            self.client_registry[client_id].set_local_model_state(cloud_state)
        return distributed_client_ids

    def _evaluate_all_clients(self, global_epoch, test_writer, class_writer):
        """评估全部客户端测试分区，并同时写入总体与逐类指标。"""
        class_totals = {
            label: {"samples": 0, "correct": 0, "loss": 0.0}
            for label in range(10)
        }
        for client_id in range(int(self.args.client_num_in_total)):
            metrics = self.client_registry[client_id].evaluate_local_model(True)
            label = client_id % 10
            class_totals[label]["samples"] += int(metrics["test_total"])
            class_totals[label]["correct"] += int(metrics["test_correct"])
            class_totals[label]["loss"] += float(metrics["test_loss"])

        total_samples = sum(item["samples"] for item in class_totals.values())
        total_correct = sum(item["correct"] for item in class_totals.values())
        total_loss = sum(item["loss"] for item in class_totals.values())
        if total_samples <= 0:
            raise ValueError("全部客户端测试样本总数必须大于 0。")
        test_row = {
            "global_epoch": int(global_epoch),
            "test_samples": int(total_samples),
            "test_correct": int(total_correct),
            "test_acc": float(total_correct) / float(total_samples),
            "test_loss": float(total_loss) / float(total_samples),
        }
        test_writer.writerow(test_row)

        for label in range(10):
            label_values = class_totals[label]
            label_samples = int(label_values["samples"])
            class_writer.writerow({
                "global_epoch": int(global_epoch),
                "label": label,
                "test_samples": label_samples,
                "test_correct": int(label_values["correct"]),
                "test_acc": (
                    float(label_values["correct"]) / float(label_samples)
                    if label_samples else np.nan
                ),
                "test_loss": (
                    float(label_values["loss"]) / float(label_samples)
                    if label_samples else np.nan
                ),
            })

        with (self.result_dir / "test_acc.txt").open(
                "a", encoding="utf-8"
        ) as file_obj:
            file_obj.write("{}\n".format(test_row["test_acc"]))
        with (self.result_dir / "test_loss.txt").open(
                "a", encoding="utf-8"
        ) as file_obj:
            file_obj.write("{}\n".format(test_row["test_loss"]))
        return test_row

    def _write_schedule_row(
            self, schedule_file, global_epoch, distributed_client_ids
    ):
        """写入一轮固定训练和全量下发的运行时记录。"""
        row = {
            "global_epoch": int(global_epoch),
            "candidate_client_ids": list(self.training_client_ids),
            "active_client_ids": list(self.training_client_ids),
            "active_client_count": len(self.training_client_ids),
            "distributed_client_ids": list(distributed_client_ids),
            "distributed_client_count": len(distributed_client_ids),
            "aggregation": self.AGGREGATION_NAME,
        }
        edge_groups = self._get_edge_client_groups()
        if edge_groups:
            row.update({
                "edge_client_groups": edge_groups,
                "edge_active_group_ids": list(range(len(edge_groups))),
                "edge_aggregation": "sample_weighted_fedavg",
                "cloud_aggregation": "sample_weighted_by_edge_train_samples",
            })
        schedule_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        schedule_file.flush()

    def _get_edge_client_groups(self):
        """返回当前训练器的边缘客户端分组；普通 FedAvg 返回空列表。"""
        return []

    def _write_metadata(self, probe_recorder, status):
        """原子更新实验元数据和当前完成轮数。"""
        active_train_samples = sum(
            int(self.train_data_local_num_dict[client_id])
            for client_id in self.training_client_ids
        )
        metadata = {
            "schema_version": self.SCHEMA_VERSION,
            "status": str(status),
            "completed_epochs": int(self.completed_epochs),
            "dataset": "mnist",
            "partition_strategy": str(self.args.partition_strategy),
            "partition_alpha": float(self.args.partition_alpha),
            "partition_alpha_used": False,
            "partition_seed": int(self.args.partition_seed),
            "client_num_in_total": int(self.args.client_num_in_total),
            "clients_per_class": int(self.args.clients_per_class),
            "training_client_ids": list(self.training_client_ids),
            "aggregation": self.AGGREGATION_NAME,
            "edge_client_groups": self._get_edge_client_groups(),
            "edge_group_count": len(self._get_edge_client_groups()),
            "client_num_per_round": int(self.args.client_num_per_round),
            "comm_round": int(self.args.comm_round),
            "epochs": int(self.args.epochs),
            "batch_size": int(self.args.batch_size),
            "client_optimizer": str(self.args.client_optimizer),
            "learning_rate": float(self.args.lr),
            "weight_decay": float(self.args.wd),
            "model_distribution_scope": "all",
            "distributed_client_count": int(self.args.client_num_in_total),
            "active_train_samples": int(active_train_samples),
            "active_train_fraction": (
                float(active_train_samples) / float(self.train_data_num_in_total)
            ),
            "random_seed": int(self.args.random_seed),
            "using_gpu": bool(self.args.using_gpu),
            "probe_formula": {
                "agreement": "A=1-normalized_generalized_JSD",
                "certainty": "C=1-mean_normalized_entropy",
                "effective_consensus": "S=A*C",
                "aggregation": "mean_over_100_probe_images",
            },
            "config_file": str(getattr(self.args, "yaml_config_file", "")),
        }
        metadata.update(probe_recorder.metadata())
        output_path = self.result_dir / "experiment_metadata.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(str(temporary_path), str(output_path))

    def train(self):
        """运行 200 轮固定客户端训练、探针、聚合、下发和全量评估。"""
        self._create_result_dir()
        self._write_partition_manifest()
        fixed_probe = select_fixed_balanced_probe(
            probe_data=self.test_global,
            samples_per_class=int(self.args.probe_samples_per_class),
            seed=int(self.args.probe_seed),
            source=str(self.args.probe_source),
        )
        probe_recorder = ProbeBatchRecorder(
            result_dir=str(self.result_dir),
            total_epochs=int(self.args.comm_round),
            candidate_client_ids=self.training_client_ids,
            edge_slot_count=0,
            probe_set=fixed_probe,
            class_count=10,
            npz_filename=str(self.args.probe_npz_file),
            summary_filename=str(self.args.probe_summary_file),
            checkpoint_interval=int(self.args.probe_checkpoint_interval),
        )
        cloud_state = copy.deepcopy(self.model_trainer.get_model_params())
        self._write_metadata(probe_recorder, status="running")

        test_path = self.result_dir / "test_metrics.csv"
        class_path = self.result_dir / "class_test_metrics.csv"
        schedule_path = self.result_dir / "training_schedule.jsonl"
        try:
            with test_path.open(
                    "w", encoding="utf-8", newline=""
            ) as test_file, class_path.open(
                    "w", encoding="utf-8", newline=""
            ) as class_file, schedule_path.open(
                    "w", encoding="utf-8"
            ) as schedule_file:
                test_writer = csv.DictWriter(
                    test_file, fieldnames=self.TEST_METRIC_COLUMNS
                )
                class_writer = csv.DictWriter(
                    class_file, fieldnames=self.CLASS_METRIC_COLUMNS
                )
                test_writer.writeheader()
                class_writer.writeheader()

                for global_epoch in range(int(self.args.comm_round)):
                    weighted_states = []
                    for client_id in self.training_client_ids:
                        _, local_state = self.client_registry[
                            client_id
                        ].train_one_epoch(global_epoch, 0, 0, w=cloud_state)
                        weighted_states.append((
                            self.client_registry[client_id].get_sample_number(),
                            local_state,
                        ))

                    # 客户端探针位于本地训练后、云端聚合前。
                    client_probabilities = self._build_probe_probabilities(
                        fixed_probe.inputs, self.training_client_ids
                    )
                    cloud_state = aggregate_weighted_model_states(weighted_states)
                    self.model_trainer.set_model_params(cloud_state)
                    cloud_probabilities = self._build_cloud_probabilities(
                        fixed_probe.inputs, cloud_state
                    )
                    empty_edge_probabilities = np.empty(
                        (0, fixed_probe.sample_count, 10), dtype=np.float32
                    )
                    probe_recorder.record_epoch(
                        global_epoch=global_epoch,
                        client_probabilities=client_probabilities,
                        edge_probabilities=empty_edge_probabilities,
                        cloud_probabilities=cloud_probabilities,
                        active_client_ids=self.training_client_ids,
                    )

                    distributed_client_ids = self._distribute_cloud_model(cloud_state)
                    test_row = self._evaluate_all_clients(
                        global_epoch, test_writer, class_writer
                    )
                    self._write_schedule_row(
                        schedule_file, global_epoch, distributed_client_ids
                    )
                    test_file.flush()
                    class_file.flush()
                    self.completed_epochs = global_epoch + 1
                    self._write_metadata(probe_recorder, status="running")
                    logging.info(
                        "SevereTest epoch %s/%s, test_acc=%.6f",
                        self.completed_epochs,
                        self.args.comm_round,
                        test_row["test_acc"],
                    )
        finally:
            probe_recorder.close()

        self._write_metadata(probe_recorder, status="complete")
        return self.result_dir


class SevereHierarchicalFedAvg(SevereFixedFedAvg):
    """执行 30 个客户端、三个边缘组和一个云端的分层 FedAvg。"""

    EXPECTED_TRAINING_CLIENT_IDS = tuple(range(30))
    EXPECTED_EDGE_CLIENT_GROUPS = (
        tuple(range(0, 10)),
        tuple(range(10, 20)),
        tuple(range(20, 30)),
    )
    SCHEMA_VERSION = "severe_single_label_hfl_v1"
    AGGREGATION_NAME = "hierarchical_sample_weighted_fedavg"

    def __init__(self, args, device, dataset, model, partition_manifest):
        """初始化 30 个固定训练客户端并校验三个十客户端边缘组。"""
        super().__init__(args, device, dataset, model, partition_manifest)
        configured_groups = getattr(
            args,
            "edge_client_groups",
            [list(group) for group in self.EXPECTED_EDGE_CLIENT_GROUPS],
        )
        self.edge_client_groups = validate_hierarchical_groups(
            configured_groups,
            self.training_client_ids,
            args.client_num_in_total,
        )
        expected_groups = [
            list(group) for group in self.EXPECTED_EDGE_CLIENT_GROUPS
        ]
        if self.edge_client_groups != expected_groups:
            raise ValueError(
                "分层实验要求边缘组严格为 0–9、10–19 和 20–29。"
            )

    def _get_edge_client_groups(self):
        """返回可安全写入 JSON 的三个固定边缘客户端组。"""
        return [list(group) for group in self.edge_client_groups]

    def _build_edge_probabilities(self, probe_inputs, edge_states):
        """按边缘组编号顺序计算三个边缘聚合模型的探针概率。"""
        return np.stack([
            self._build_model_probabilities(probe_inputs, edge_state)
            for edge_state in edge_states
        ], axis=0).astype(np.float32, copy=False)

    def train(self):
        """运行客户端、边缘、云端三级训练、探针、下发和全量评估。"""
        self._create_result_dir()
        self._write_partition_manifest()
        fixed_probe = select_fixed_balanced_probe(
            probe_data=self.test_global,
            samples_per_class=int(self.args.probe_samples_per_class),
            seed=int(self.args.probe_seed),
            source=str(self.args.probe_source),
        )
        probe_recorder = ProbeBatchRecorder(
            result_dir=str(self.result_dir),
            total_epochs=int(self.args.comm_round),
            candidate_client_ids=self.training_client_ids,
            edge_slot_count=len(self.edge_client_groups),
            probe_set=fixed_probe,
            class_count=10,
            npz_filename=str(self.args.probe_npz_file),
            summary_filename=str(self.args.probe_summary_file),
            checkpoint_interval=int(self.args.probe_checkpoint_interval),
        )
        cloud_state = copy.deepcopy(self.model_trainer.get_model_params())
        self._write_metadata(probe_recorder, status="running")

        test_path = self.result_dir / "test_metrics.csv"
        class_path = self.result_dir / "class_test_metrics.csv"
        schedule_path = self.result_dir / "training_schedule.jsonl"
        try:
            with test_path.open(
                    "w", encoding="utf-8", newline=""
            ) as test_file, class_path.open(
                    "w", encoding="utf-8", newline=""
            ) as class_file, schedule_path.open(
                    "w", encoding="utf-8"
            ) as schedule_file:
                test_writer = csv.DictWriter(
                    test_file, fieldnames=self.TEST_METRIC_COLUMNS
                )
                class_writer = csv.DictWriter(
                    class_file, fieldnames=self.CLASS_METRIC_COLUMNS
                )
                test_writer.writeheader()
                class_writer.writeheader()

                for global_epoch in range(int(self.args.comm_round)):
                    grouped_weighted_states = []
                    for edge_id, client_group in enumerate(
                            self.edge_client_groups
                    ):
                        weighted_states = []
                        for client_id in client_group:
                            _, local_state = self.client_registry[
                                client_id
                            ].train_one_epoch(
                                global_epoch,
                                edge_id,
                                edge_id,
                                w=cloud_state,
                            )
                            weighted_states.append((
                                self.client_registry[
                                    client_id
                                ].get_sample_number(),
                                local_state,
                            ))
                        grouped_weighted_states.append(weighted_states)

                    # 先保存 30 个本地模型输出，再执行边缘和云端两级聚合。
                    client_probabilities = self._build_probe_probabilities(
                        fixed_probe.inputs, self.training_client_ids
                    )
                    edge_states, cloud_state, edge_sample_counts = (
                        aggregate_hierarchical_model_states(
                            grouped_weighted_states
                        )
                    )
                    edge_probabilities = self._build_edge_probabilities(
                        fixed_probe.inputs, edge_states
                    )
                    self.model_trainer.set_model_params(cloud_state)
                    cloud_probabilities = self._build_cloud_probabilities(
                        fixed_probe.inputs, cloud_state
                    )
                    probe_recorder.record_epoch(
                        global_epoch=global_epoch,
                        client_probabilities=client_probabilities,
                        edge_probabilities=edge_probabilities,
                        cloud_probabilities=cloud_probabilities,
                        active_client_ids=self.training_client_ids,
                    )

                    distributed_client_ids = self._distribute_cloud_model(
                        cloud_state
                    )
                    test_row = self._evaluate_all_clients(
                        global_epoch, test_writer, class_writer
                    )
                    self._write_schedule_row(
                        schedule_file, global_epoch, distributed_client_ids
                    )
                    test_file.flush()
                    class_file.flush()
                    self.completed_epochs = global_epoch + 1
                    self._write_metadata(probe_recorder, status="running")
                    logging.info(
                        "SevereTest HFL epoch %s/%s, edge_samples=%s, "
                        "test_acc=%.6f",
                        self.completed_epochs,
                        self.args.comm_round,
                        edge_sample_counts,
                        test_row["test_acc"],
                    )
        finally:
            probe_recorder.close()

        self._write_metadata(probe_recorder, status="complete")
        return self.result_dir
