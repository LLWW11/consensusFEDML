"""基于逐轮MATLAB拓扑的动态采样与分组联邦TransE训练。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.aggregation import (
    RowCountWeightedFedAvgAggregator,
    RowMaskedFedAvgAggregator,
)
from ..core.device import as_bool
from ..core.result_writer import ExperimentResultWriter
from ..core.server_optimization import RowWiseFedAdamOptimizer
from ..core.topology import RoundTopology, TopologyProvider
from ..core.types import ClientUpdate, clone_state_dict
from .trainer import FedMLFederatedTransETrainer


def _state_dict_sha256(
    state_dict: Dict[str, torch.Tensor],
) -> str:
    """计算模型状态字典的稳定SHA-256，用来核对三臂初始模型是否一致。"""

    digest = hashlib.sha256()
    for parameter_name in sorted(state_dict):
        tensor = (
            state_dict[parameter_name]
            .detach()
            .cpu()
            .contiguous()
        )
        # 名称、类型、形状和原始数值都进入摘要，避免不同参数误判为相同。
        digest.update(parameter_name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class FedMLDynamicTopologyTransETrainer(FedMLFederatedTransETrainer):
    """按MAT逐轮选择客户端和分组，并执行稠密或行级聚合。"""

    def __init__(
        self,
        args,
        device: torch.device,
        federated_data,
        model_trainer,
        topology_provider: TopologyProvider,
    ):
        """初始化FedML客户端、排名评估器和经过预检的动态拓扑序列。"""

        super().__init__(
            args=args,
            device=device,
            federated_data=federated_data,
            model_trainer=model_trainer,
            fixed_topology=None,
        )
        self.topology_provider = topology_provider
        self.topology_metadata = dict(topology_provider.describe())
        self.architecture = str(
            self.topology_metadata.get(
                "architecture",
                getattr(args, "topology_architecture", "hfl"),
            )
        ).strip().lower()
        self.snf_enabled = bool(
            self.topology_metadata.get(
                "snf_enabled",
                getattr(args, "topology_snf", True),
            )
        )
        self.edge_mode = str(
            self.topology_metadata.get(
                "edge_mode",
                getattr(args, "topology_edge_mode", "dynamic"),
            )
        ).strip().lower()
        self.scenario_name = str(
            getattr(
                args,
                "comparison_scenario",
                self.topology_metadata.get(
                    "scenario", "DynamicMatFedTransE"
                ),
            )
        ).strip()
        self.aggregation_mode = str(
            getattr(
                args,
                "aggregation_mode",
                "dense_triple_weighted",
            )
        ).strip().lower()
        self.local_objective = str(
            getattr(args, "local_objective", "margin_ranking")
        ).strip().lower()
        if self.aggregation_mode not in {
            "dense_triple_weighted",
            "row_mask_presence",
            "row_count_weighted",
        }:
            raise ValueError(
                "aggregation_mode必须是dense_triple_weighted"
                "、row_mask_presence或row_count_weighted"
            )
        if self.local_objective not in {
            "margin_ranking",
            "fede_self_adversarial",
            "bidirectional_self_adversarial",
        }:
            raise ValueError(
                "local_objective必须是margin_ranking或"
                "fede_self_adversarial或"
                "bidirectional_self_adversarial"
            )
        self.row_masked_aggregator = RowMaskedFedAvgAggregator()
        self.row_count_weighted_aggregator = (
            RowCountWeightedFedAvgAggregator()
        )
        self.server_optimizer_name = str(
            getattr(args, "server_optimizer", "fedavg")
        ).strip().lower()
        if self.server_optimizer_name not in {"fedavg", "fedadam"}:
            raise ValueError(
                "server_optimizer必须是fedavg或fedadam"
            )
        if (
            self.server_optimizer_name == "fedadam"
            and self.aggregation_mode != "row_count_weighted"
        ):
            raise ValueError(
                "服务器端FedAdam当前只支持row_count_weighted聚合"
            )
        self.server_fedadam_optimizer: Optional[
            RowWiseFedAdamOptimizer
        ] = None
        if self.server_optimizer_name == "fedadam":
            self.server_fedadam_optimizer = RowWiseFedAdamOptimizer(
                learning_rate=float(
                    getattr(args, "server_learning_rate", 0.1)
                ),
                beta1=float(getattr(args, "server_beta1", 0.9)),
                beta2=float(getattr(args, "server_beta2", 0.99)),
                tau=float(getattr(args, "server_tau", 0.001)),
                bias_correction=as_bool(
                    getattr(
                        args,
                        "server_bias_correction",
                        False,
                    )
                ),
            )
        self._round_topologies = tuple(
            topology_provider.get_round(round_index)
            for round_index in range(self.comm_round)
        )
        self._validate_dynamic_topologies()
        self.participation_summary = (
            self._build_participation_summary()
        )
        self.initial_model_hash = _state_dict_sha256(
            self.get_global_state()
        )
        self._validate_experiment_fingerprints()

    def _validate_experiment_fingerprints(self) -> None:
        """在正式训练前核对客户端划分和MAT调度，防止三臂口径悄悄变化。"""

        expected_partition_hash = str(
            getattr(self.args, "expected_partition_hash", "")
        ).strip()
        if (
            expected_partition_hash
            and expected_partition_hash
            != self.federated_data.partition_hash
        ):
            raise ValueError(
                "客户端划分哈希与配置不一致：期望{}，实际{}".format(
                    expected_partition_hash,
                    self.federated_data.partition_hash,
                )
            )

        actual_schedule_hash = str(
            self.participation_summary["schedule_hash"]
        )
        expected_schedule_hash = str(
            getattr(
                self.args,
                "expected_topology_schedule_hash",
                "",
            )
        ).strip()
        if (
            expected_schedule_hash
            and expected_schedule_hash != actual_schedule_hash
        ):
            raise ValueError(
                "MAT调度哈希与配置不一致：期望{}，实际{}".format(
                    expected_schedule_hash,
                    actual_schedule_hash,
                )
            )

    def _validate_dynamic_topologies(self) -> None:
        """校验每轮MAT客户端编号、互斥分组和训练轮数均可执行。"""

        if self.architecture not in {"fl", "hfl"}:
            raise ValueError("动态TransE拓扑结构必须是fl或hfl")
        available_rounds = self.topology_metadata.get("round_count")
        schedule_policy = str(
            self.topology_metadata.get(
                "topology_schedule_policy", "strict"
            )
        ).strip().lower()
        if (
            available_rounds is not None
            and self.comm_round > int(available_rounds)
            and schedule_policy != "cycle"
        ):
            raise ValueError(
                "comm_round={}超过MAT可用轮数{}".format(
                    self.comm_round, int(available_rounds)
                )
            )

        known_clients = set(self.client_registry.keys())
        for round_index, topology in enumerate(
            self._round_topologies
        ):
            active_clients = set(topology.active_client_indexes)
            if not active_clients:
                raise ValueError(
                    "MAT第{}行没有参与客户端".format(round_index)
                )
            unknown_clients = sorted(active_clients - known_clients)
            if unknown_clients:
                raise ValueError(
                    "MAT第{}行包含数据分区中不存在的客户端{}".format(
                        round_index, unknown_clients
                    )
                )
            grouped_clients = [
                int(client_id)
                for client_ids in (
                    topology.group_to_client_indexes.values()
                )
                for client_id in client_ids
            ]
            if len(grouped_clients) != len(set(grouped_clients)):
                raise ValueError(
                    "MAT第{}行的动态分组包含重复客户端".format(
                        round_index
                    )
                )
            if set(grouped_clients) != active_clients:
                raise ValueError(
                    "MAT第{}行的分组没有完整覆盖参与客户端".format(
                        round_index
                    )
                )
            if self.architecture == "fl" and len(
                topology.group_to_client_indexes
            ) != 1:
                raise ValueError("FL动态拓扑每轮必须只有一个直连组")

    def _train_dynamic_round(
        self,
        round_index: int,
        topology: RoundTopology,
    ) -> Tuple[Sequence[ClientUpdate], Dict[str, object]]:
        """训练MAT当前行选中的客户端并按当前动态组完成聚合。"""

        global_state = self.get_global_state()
        updates: List[ClientUpdate] = []
        for client_id in topology.active_client_indexes:
            # 所有活跃客户端必须从同一份轮初全局参数开始。
            updates.append(
                self.client_registry[int(client_id)].train_from_global(
                    global_state, round_index
                )
            )

        if self.aggregation_mode == "row_count_weighted":
            new_global_state, aggregation_details = (
                self._aggregate_row_count_weighted_updates(
                    updates=updates,
                    topology=topology,
                    global_state=global_state,
                )
            )
        elif self.aggregation_mode == "row_mask_presence":
            new_global_state, aggregation_details = (
                self._aggregate_row_masked_updates(
                    updates=updates,
                    topology=topology,
                    global_state=global_state,
                )
            )
        elif self.architecture == "fl":
            new_global_state = self.aggregator.aggregate(updates)
            aggregation_details = {
                "aggregation": "direct_dense_fedavg",
                "edge_group_count": 0,
                "edge_node_ids": {},
                "group_aggregation_weights": {},
                "group_contributing_client_indexes": {},
            }
        else:
            update_by_client = {
                int(update.client_id): update for update in updates
            }
            edge_statistics = []
            group_weights: Dict[str, float] = {}
            group_contributors: Dict[str, List[int]] = {}
            for group_id, client_ids in (
                topology.group_to_client_indexes.items()
            ):
                group_updates = [
                    update_by_client[int(client_id)]
                    for client_id in client_ids
                ]
                statistics = self.aggregator.accumulate(group_updates)
                edge_statistics.append(statistics)
                group_weights[str(group_id)] = float(
                    statistics.total_weight
                )
                group_contributors[str(group_id)] = [
                    int(value)
                    for value in statistics.contributor_ids
                ]
            cloud_statistics = self.aggregator.merge(edge_statistics)
            new_global_state = self.aggregator.finalize(
                cloud_statistics
            )
            aggregation_details = {
                "aggregation": (
                    "hierarchical_two_level_dense_fedavg"
                ),
                "edge_group_count": len(edge_statistics),
                "edge_node_ids": {
                    str(group_id): int(edge_id)
                    for group_id, edge_id in (
                        topology.edge_node_ids.items()
                    )
                },
                "group_aggregation_weights": group_weights,
                "group_contributing_client_indexes": (
                    group_contributors
                ),
            }

        self.model_trainer.set_model_params(new_global_state)
        # 所有聚合方式都会改变实体行范数，因此在云端统一投影。
        self.model_trainer.model.to(self.device)
        self.model_trainer.model.normalize_entity_embeddings()
        return tuple(updates), aggregation_details

    def _aggregate_row_masked_updates(
        self,
        updates: Sequence[ClientUpdate],
        topology: RoundTopology,
        global_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """按当前MAT动态组执行FedE式逐行边缘聚合和云端合并。"""

        if self.architecture == "fl":
            cloud_statistics = self.row_masked_aggregator.accumulate(
                updates
            )
            group_statistics = {"0": cloud_statistics}
            group_contributors = {
                "0": [
                    int(update.client_id) for update in updates
                ]
            }
        else:
            update_by_client = {
                int(update.client_id): update for update in updates
            }
            edge_statistics = []
            group_statistics = {}
            group_contributors = {}
            for group_id, client_ids in (
                topology.group_to_client_indexes.items()
            ):
                group_updates = [
                    update_by_client[int(client_id)]
                    for client_id in client_ids
                ]
                statistics = self.row_masked_aggregator.accumulate(
                    group_updates
                )
                edge_statistics.append(statistics)
                group_statistics[str(group_id)] = statistics
                group_contributors[str(group_id)] = [
                    int(value) for value in client_ids
                ]
            cloud_statistics = self.row_masked_aggregator.merge(
                edge_statistics
            )

        new_global_state = self.row_masked_aggregator.finalize(
            cloud_statistics, global_state
        )
        parameter_statistics = (
            self.row_masked_aggregator.summarize(cloud_statistics)
        )
        group_row_statistics = {
            str(group_id): self.row_masked_aggregator.summarize(
                statistics
            )
            for group_id, statistics in group_statistics.items()
        }
        return new_global_state, {
            "aggregation": (
                "hierarchical_two_level_row_mask_presence"
                if self.architecture == "hfl"
                else "direct_row_mask_presence"
            ),
            "edge_group_count": (
                len(topology.group_to_client_indexes)
                if self.architecture == "hfl"
                else 0
            ),
            "edge_node_ids": {
                str(group_id): int(edge_id)
                for group_id, edge_id in (
                    topology.edge_node_ids.items()
                )
            },
            "group_aggregation_weights": {},
            "group_contributing_client_indexes": (
                group_contributors
            ),
            "parameter_row_statistics": parameter_statistics,
            "group_parameter_row_statistics": group_row_statistics,
        }

    def _aggregate_row_count_weighted_updates(
        self,
        updates: Sequence[ClientUpdate],
        topology: RoundTopology,
        global_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """按正事实行出现次数执行边缘聚合，并在云端无损合并统计。"""

        aggregator = self.row_count_weighted_aggregator
        if self.architecture == "fl":
            cloud_statistics = aggregator.accumulate(updates)
            group_statistics = {"0": cloud_statistics}
            group_contributors = {
                "0": [
                    int(update.client_id) for update in updates
                ]
            }
        else:
            update_by_client = {
                int(update.client_id): update for update in updates
            }
            edge_statistics = []
            group_statistics = {}
            group_contributors = {}
            for group_id, client_ids in (
                topology.group_to_client_indexes.items()
            ):
                group_updates = [
                    update_by_client[int(client_id)]
                    for client_id in client_ids
                ]
                statistics = aggregator.accumulate(group_updates)
                edge_statistics.append(statistics)
                group_statistics[str(group_id)] = statistics
                group_contributors[str(group_id)] = [
                    int(value) for value in client_ids
                ]
            cloud_statistics = aggregator.merge(edge_statistics)

        fedavg_candidate_state = aggregator.finalize(
            cloud_statistics, global_state
        )
        active_row_masks = {
            str(name): denominator > 0
            for name, denominator in (
                cloud_statistics.row_denominators.items()
            )
        }
        if self.server_fedadam_optimizer is not None:
            new_global_state, server_optimizer_details = (
                self.server_fedadam_optimizer.step(
                    global_state=global_state,
                    candidate_state=fedavg_candidate_state,
                    active_row_masks=active_row_masks,
                )
            )
            server_optimizer_details[
                "server_optimizer_state_hash"
            ] = _state_dict_sha256(
                self.server_fedadam_optimizer.state_dict()
            )
        else:
            new_global_state = fedavg_candidate_state
            server_optimizer_details = {
                "server_optimizer": "fedavg",
                "server_optimizer_step": 0,
                "server_learning_rate": 1.0,
                "server_beta1": 0.0,
                "server_beta2": 0.0,
                "server_tau": 0.0,
                "server_bias_correction": False,
                "server_active_row_count": int(
                    sum(
                        int(mask.sum().item())
                        for mask in active_row_masks.values()
                    )
                ),
                "server_model_delta_l2": 0.0,
                "server_update_l2": 0.0,
                "server_update_max_abs": 0.0,
                "server_parameter_statistics": {},
                "server_optimizer_state_hash": "",
            }
        parameter_statistics = aggregator.summarize(
            cloud_statistics
        )
        group_row_statistics = {
            str(group_id): aggregator.summarize(statistics)
            for group_id, statistics in group_statistics.items()
        }
        group_state_hashes = {
            str(group_id): _state_dict_sha256(
                aggregator.finalize(statistics, global_state)
            )
            for group_id, statistics in group_statistics.items()
        }
        group_row_occurrence_weights = {
            str(group_id): {
                str(name): float(denominator.sum().item())
                for name, denominator in (
                    statistics.row_denominators.items()
                )
            }
            for group_id, statistics in group_statistics.items()
        }
        return new_global_state, {
            "aggregation": (
                "hierarchical_two_level_row_count_weighted"
                if self.architecture == "hfl"
                else "direct_row_count_weighted"
            ),
            "aggregation_weight_basis": (
                "local_positive_triple_row_occurrences"
            ),
            "row_count_source": "local_positive_train_triples",
            "edge_group_count": (
                len(topology.group_to_client_indexes)
                if self.architecture == "hfl"
                else 0
            ),
            "edge_node_ids": {
                str(group_id): int(edge_id)
                for group_id, edge_id in (
                    topology.edge_node_ids.items()
                )
            },
            "group_aggregation_weights": (
                group_row_occurrence_weights
            ),
            "group_contributing_client_indexes": (
                group_contributors
            ),
            "parameter_row_statistics": parameter_statistics,
            "group_parameter_row_statistics": group_row_statistics,
            "group_parameter_state_hashes": group_state_hashes,
            "fedavg_candidate_state_hash": _state_dict_sha256(
                fedavg_candidate_state
            ),
            "cloud_parameter_state_hash": _state_dict_sha256(
                new_global_state
            ),
            "parameter_hash_stage": (
                "after_server_optimizer_before_entity_normalization"
            ),
            **server_optimizer_details,
        }

    @staticmethod
    def _row_statistic_value(
        aggregation_details: Dict[str, object],
        parameter_name: str,
        statistic_name: str,
    ) -> float:
        """安全读取逐行聚合统计，稠密模式返回零以保持CSV字段稳定。"""

        parameter_statistics = aggregation_details.get(
            "parameter_row_statistics", {}
        )
        if not isinstance(parameter_statistics, dict):
            return 0.0
        values = parameter_statistics.get(parameter_name, {})
        if not isinstance(values, dict):
            return 0.0
        return float(values.get(statistic_name, 0.0))

    @staticmethod
    def _server_parameter_statistic_value(
        aggregation_details: Dict[str, object],
        parameter_name: str,
        statistic_name: str,
    ) -> float:
        """安全读取服务器优化器的逐参数统计，FedAvg模式返回零。"""

        parameter_statistics = aggregation_details.get(
            "server_parameter_statistics", {}
        )
        if not isinstance(parameter_statistics, dict):
            return 0.0
        values = parameter_statistics.get(parameter_name, {})
        if not isinstance(values, dict):
            return 0.0
        return float(values.get(statistic_name, 0.0))

    def _build_participation_summary(self) -> Dict[str, object]:
        """汇总逐轮参与人数、动态组数、客户端频次和数据暴露预算。"""

        participant_counts = [
            int(topology.participant_count)
            for topology in self._round_topologies
        ]
        group_counts = [
            len(topology.group_to_client_indexes)
            for topology in self._round_topologies
        ]
        client_selection_counts = {
            int(client_id): 0 for client_id in self.client_registry
        }
        partition_by_id = {
            int(partition.client_id): partition
            for partition in self.federated_data.partitions
        }
        scheduled_triple_count = 0
        participant_signatures = set()
        topology_signatures = set()
        digest = hashlib.sha256()
        for topology in self._round_topologies:
            participant_signatures.add(
                tuple(topology.active_client_indexes)
            )
            canonical_groups = tuple(
                (
                    int(group_id),
                    tuple(int(value) for value in client_ids),
                    int(topology.edge_node_ids.get(group_id, -1)),
                )
                for group_id, client_ids in (
                    topology.group_to_client_indexes.items()
                )
            )
            topology_signatures.add(canonical_groups)
            canonical_record = {
                "source_round_index": int(
                    topology.source_round_index
                ),
                "groups": topology.copy_groups(),
                "edge_node_ids": {
                    str(group_id): int(edge_id)
                    for group_id, edge_id in (
                        topology.edge_node_ids.items()
                    )
                },
            }
            digest.update(
                json.dumps(
                    canonical_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for client_id in topology.active_client_indexes:
                client_id = int(client_id)
                client_selection_counts[client_id] += 1
                scheduled_triple_count += int(
                    partition_by_id[client_id].triple_count
                )

        total_train_triples = int(
            self.dataset.train_triples.shape[0]
        )
        local_epochs = int(getattr(self.args, "epochs", 1))
        raw_source_round_count = self.topology_metadata.get(
            "source_round_count",
            self.topology_metadata.get("round_count"),
        )
        # noSnF固定人数调度可按规则无限生成，不存在有限MAT源轮数。
        source_round_count = (
            int(raw_source_round_count)
            if raw_source_round_count is not None
            else int(self.comm_round)
        )
        schedule_policy = str(
            self.topology_metadata.get(
                "topology_schedule_policy", "strict"
            )
        )
        return {
            "topology_metadata": self.topology_metadata,
            "used_round_count": self.comm_round,
            "topology_schedule_policy": schedule_policy,
            "source_round_count": source_round_count,
            # 循环策略下明确记录超出源MAT首轮覆盖范围的复用轮数。
            "cycled_round_count": max(
                0, self.comm_round - source_round_count
            ),
            "schedule_hash": digest.hexdigest(),
            "participant_count_min": min(participant_counts),
            "participant_count_max": max(participant_counts),
            "participant_count_mean": float(
                np.mean(participant_counts)
            ),
            "group_count_min": min(group_counts),
            "group_count_max": max(group_counts),
            "group_count_mean": float(np.mean(group_counts)),
            "unique_participant_set_count": len(
                participant_signatures
            ),
            "unique_topology_count": len(topology_signatures),
            "dynamic_client_selection": (
                len(participant_signatures) > 1
            ),
            "dynamic_grouping": len(topology_signatures) > 1,
            "client_selection_counts": {
                str(client_id): int(count)
                for client_id, count in client_selection_counts.items()
            },
            "client_selection_fractions": {
                str(client_id): float(count) / float(self.comm_round)
                for client_id, count in client_selection_counts.items()
            },
            "scheduled_positive_triple_count_per_local_epoch": int(
                scheduled_triple_count
            ),
            "total_positive_triple_exposures": int(
                scheduled_triple_count * local_epochs
            ),
            "effective_full_data_passes": (
                float(scheduled_triple_count * local_epochs)
                / float(total_train_triples)
            ),
        }

    @staticmethod
    def _print_dynamic_round_progress(
        round_number: int,
        total_rounds: int,
        scenario_name: str,
        source_round_index: int,
        active_client_count: int,
        active_group_count: int,
        weighted_loss: float,
        aggregation_weight: float,
        round_seconds: float,
        validation_metrics: Optional[Dict[str, float]],
    ) -> None:
        """逐epoch打印MAT行、动态规模、损失、耗时及选模指标。"""

        if validation_metrics is None:
            mrr_text = "未评估"
            hits_at_3_text = "未评估"
        else:
            mrr_text = "{:.6f}".format(
                float(validation_metrics["mrr"])
            )
            hits_at_3_text = "{:.6f}".format(
                float(validation_metrics["hits_at_3"])
            )
        print(
            (
                "[动态联邦TransE] 方案={} epoch={}/{} MAT行={} "
                "参与客户端={} 动态分组={} 加权损失={:.6f} "
                "聚合权重={:.0f} 耗时={:.2f}s "
                "验证MRR={} 验证Hits@3={}"
            ).format(
                scenario_name,
                round_number,
                total_rounds,
                source_round_index,
                active_client_count,
                active_group_count,
                weighted_loss,
                aggregation_weight,
                round_seconds,
                mrr_text,
                hits_at_3_text,
            ),
            flush=True,
        )

    def _reset_cuda_peak_memory(self) -> None:
        """在CUDA训练轮开始前重置设备峰值显存统计。"""

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_peak_memory_megabytes(self) -> Tuple[float, float]:
        """返回当前轮CUDA峰值已分配和保留显存，CPU返回零。"""

        if self.device.type != "cuda":
            return 0.0, 0.0
        megabyte = float(1024 ** 2)
        allocated = float(
            torch.cuda.max_memory_allocated(self.device)
        ) / megabyte
        reserved = float(
            torch.cuda.max_memory_reserved(self.device)
        ) / megabyte
        return allocated, reserved

    def train(self, result_dir: Path) -> Dict[str, object]:
        """执行动态采样联邦TransE、恢复最佳模型并完成完整评估。"""

        best_state = clone_state_dict(
            self.get_global_state(), to_cpu=True
        )
        best_round = 0
        best_validation_mrr = -math.inf
        total_train_triples = int(
            self.dataset.train_triples.shape[0]
        )
        round_cuda_allocated_mb: List[float] = []
        round_cuda_reserved_mb: List[float] = []

        with ExperimentResultWriter(
            result_dir,
            schedule_filename="dynamic_topology_schedule.jsonl",
        ) as writer:
            writer.write_json(
                "client_partition_summary.json",
                self.federated_data.summary(),
            )
            writer.write_json(
                "topology_metadata.json", self.topology_metadata
            )
            writer.write_json(
                "dynamic_participation_summary.json",
                self.participation_summary,
            )
            for round_index, topology in enumerate(
                self._round_topologies
            ):
                self._reset_cuda_peak_memory()
                round_started_at = time.perf_counter()
                updates, aggregation_details = (
                    self._train_dynamic_round(
                        round_index, topology
                    )
                )
                round_number = round_index + 1
                should_evaluate = (
                    round_number % self.eval_every == 0
                    or round_number == self.comm_round
                )
                validation_metrics = None
                if should_evaluate:
                    validation_metrics = self._evaluate_validation(
                        self.validation_max_triples
                    )
                    current_mrr = float(validation_metrics["mrr"])
                    if current_mrr > best_validation_mrr:
                        best_validation_mrr = current_mrr
                        best_round = round_number
                        best_state = clone_state_dict(
                            self.get_global_state(), to_cpu=True
                        )

                aggregation_weight = float(
                    sum(update.weight for update in updates)
                )
                weighted_local_loss = self._weighted_local_loss(
                    updates
                )
                optimizer_step_before = (
                    self._local_metric_statistics(
                        updates, "optimizer_step_before"
                    )
                )
                optimizer_step_after = (
                    self._local_metric_statistics(
                        updates, "optimizer_step_after"
                    )
                )
                round_seconds = (
                    time.perf_counter() - round_started_at
                )
                (
                    cuda_peak_allocated_mb,
                    cuda_peak_reserved_mb,
                ) = self._cuda_peak_memory_megabytes()
                round_cuda_allocated_mb.append(
                    cuda_peak_allocated_mb
                )
                round_cuda_reserved_mb.append(cuda_peak_reserved_mb)
                metric_row: Dict[str, object] = {
                    "round": round_number,
                    "ablation_suite": str(
                        getattr(self.args, "ablation_suite", "")
                    ).strip(),
                    "ablation_arm": str(
                        getattr(self.args, "ablation_arm", "")
                    ).strip(),
                    "source_round_index": int(
                        topology.source_round_index
                    ),
                    "scenario": self.scenario_name,
                    "architecture": self.architecture,
                    "snf_enabled": bool(self.snf_enabled),
                    "edge_mode": self.edge_mode,
                    "active_client_count": int(
                        topology.participant_count
                    ),
                    "active_group_count": len(
                        topology.group_to_client_indexes
                    ),
                    "contributing_client_count": len(updates),
                    "aggregation_weight": aggregation_weight,
                    "selected_train_triple_fraction": (
                        aggregation_weight
                        / float(total_train_triples)
                    ),
                    "mean_client_train_loss": weighted_local_loss,
                    "client_sampling_seconds": (
                        self._summed_local_metric(
                            updates, "sampling_seconds"
                        )
                    ),
                    "client_transfer_seconds": (
                        self._summed_local_metric(
                            updates, "transfer_seconds"
                        )
                    ),
                    "client_forward_backward_seconds": (
                        self._summed_local_metric(
                            updates, "forward_backward_seconds"
                        )
                    ),
                    "client_optimizer_state_mode": str(
                        getattr(
                            self.args,
                            "client_optimizer_state_mode",
                            "reset",
                        )
                    ),
                    "optimizer_state_reused_client_count": (
                        self._summed_local_metric(
                            updates, "optimizer_state_reused"
                        )
                    ),
                    "optimizer_step_before_min": (
                        optimizer_step_before["min"]
                    ),
                    "optimizer_step_before_mean": (
                        optimizer_step_before["mean"]
                    ),
                    "optimizer_step_before_max": (
                        optimizer_step_before["max"]
                    ),
                    "optimizer_step_after_min": (
                        optimizer_step_after["min"]
                    ),
                    "optimizer_step_after_mean": (
                        optimizer_step_after["mean"]
                    ),
                    "optimizer_step_after_max": (
                        optimizer_step_after["max"]
                    ),
                    "aggregation_mode": self.aggregation_mode,
                    "server_optimizer": self.server_optimizer_name,
                    "server_optimizer_step": int(
                        aggregation_details.get(
                            "server_optimizer_step", 0
                        )
                    ),
                    "server_learning_rate": float(
                        aggregation_details.get(
                            "server_learning_rate", 1.0
                        )
                    ),
                    "server_beta1": float(
                        aggregation_details.get("server_beta1", 0.0)
                    ),
                    "server_beta2": float(
                        aggregation_details.get("server_beta2", 0.0)
                    ),
                    "server_tau": float(
                        aggregation_details.get("server_tau", 0.0)
                    ),
                    "server_bias_correction": bool(
                        aggregation_details.get(
                            "server_bias_correction", False
                        )
                    ),
                    "server_active_row_count": int(
                        aggregation_details.get(
                            "server_active_row_count", 0
                        )
                    ),
                    "server_model_delta_l2": float(
                        aggregation_details.get(
                            "server_model_delta_l2", 0.0
                        )
                    ),
                    "server_update_l2": float(
                        aggregation_details.get(
                            "server_update_l2", 0.0
                        )
                    ),
                    "server_update_max_abs": float(
                        aggregation_details.get(
                            "server_update_max_abs", 0.0
                        )
                    ),
                    "server_optimizer_state_hash": str(
                        aggregation_details.get(
                            "server_optimizer_state_hash", ""
                        )
                    ),
                    "entity_server_active_row_count": (
                        self._server_parameter_statistic_value(
                            aggregation_details,
                            "entity_embeddings.weight",
                            "active_row_count",
                        )
                    ),
                    "relation_server_active_row_count": (
                        self._server_parameter_statistic_value(
                            aggregation_details,
                            "relation_embeddings.weight",
                            "active_row_count",
                        )
                    ),
                    "local_objective": self.local_objective,
                    "entity_updated_row_count": (
                        self._row_statistic_value(
                            aggregation_details,
                            "entity_embeddings.weight",
                            "updated_row_count",
                        )
                    ),
                    "entity_fallback_row_count": (
                        self._row_statistic_value(
                            aggregation_details,
                            "entity_embeddings.weight",
                            "fallback_row_count",
                        )
                    ),
                    "relation_updated_row_count": (
                        self._row_statistic_value(
                            aggregation_details,
                            "relation_embeddings.weight",
                            "updated_row_count",
                        )
                    ),
                    "relation_fallback_row_count": (
                        self._row_statistic_value(
                            aggregation_details,
                            "relation_embeddings.weight",
                            "fallback_row_count",
                        )
                    ),
                    "round_seconds": round_seconds,
                    "cuda_peak_memory_allocated_mb": (
                        cuda_peak_allocated_mb
                    ),
                    "cuda_peak_memory_reserved_mb": (
                        cuda_peak_reserved_mb
                    ),
                }
                for prefix, parameter_name in (
                    ("entity", "entity_embeddings.weight"),
                    ("relation", "relation_embeddings.weight"),
                ):
                    for statistic_name in (
                        "min_row_contributors",
                        "mean_row_contributors",
                        "max_row_contributors",
                        "min_row_occurrences",
                        "mean_row_occurrences",
                        "max_row_occurrences",
                        "total_row_occurrences",
                    ):
                        metric_row[
                            "{}_{}".format(prefix, statistic_name)
                        ] = self._row_statistic_value(
                            aggregation_details,
                            parameter_name,
                            statistic_name,
                        )
                metric_row.update(
                    self._validation_fields(validation_metrics)
                )
                writer.write_metrics(metric_row)
                writer.write_topology(
                    {
                        "round": round_number,
                        "ablation_suite": str(
                            getattr(self.args, "ablation_suite", "")
                        ).strip(),
                        "ablation_arm": str(
                            getattr(self.args, "ablation_arm", "")
                        ).strip(),
                        "source_round_index": int(
                            topology.source_round_index
                        ),
                        "source_round_number": int(
                            topology.source_round_index + 1
                        ),
                        "scenario": self.scenario_name,
                        "dynamic_client_selection": bool(
                            self.participation_summary[
                                "dynamic_client_selection"
                            ]
                        ),
                        "dynamic_grouping": bool(
                            self.participation_summary[
                                "dynamic_grouping"
                            ]
                        ),
                        "fedml_client_lifecycle": True,
                        "architecture": self.architecture,
                        "snf_enabled": bool(self.snf_enabled),
                        "edge_mode": self.edge_mode,
                        "aggregation_mode": self.aggregation_mode,
                        "server_optimizer": (
                            self.server_optimizer_name
                        ),
                        "local_objective": self.local_objective,
                        "client_optimizer_state_mode": str(
                            getattr(
                                self.args,
                                "client_optimizer_state_mode",
                                "reset",
                            )
                        ),
                        "optimizer_state_reused_client_count": (
                            self._summed_local_metric(
                                updates,
                                "optimizer_state_reused",
                            )
                        ),
                        "optimizer_step_before": (
                            optimizer_step_before
                        ),
                        "optimizer_step_after": (
                            optimizer_step_after
                        ),
                        **aggregation_details,
                        "active_client_indexes": [
                            int(value)
                            for value in (
                                topology.active_client_indexes
                            )
                        ],
                        "contributing_client_indexes": [
                            int(update.client_id)
                            for update in updates
                        ],
                        "group_to_client_indexes": {
                            str(group_id): [
                                int(value) for value in client_ids
                            ]
                            for group_id, client_ids in (
                                topology.group_to_client_indexes.items()
                            )
                        },
                        "client_weights": {
                            str(update.client_id): float(update.weight)
                            for update in updates
                        },
                        "aggregation_weight": aggregation_weight,
                    }
                )
                self._print_dynamic_round_progress(
                    round_number=round_number,
                    total_rounds=self.comm_round,
                    scenario_name=self.scenario_name,
                    source_round_index=int(
                        topology.source_round_index
                    ),
                    active_client_count=int(
                        topology.participant_count
                    ),
                    active_group_count=len(
                        topology.group_to_client_indexes
                    ),
                    weighted_loss=weighted_local_loss,
                    aggregation_weight=aggregation_weight,
                    round_seconds=round_seconds,
                    validation_metrics=validation_metrics,
                )

            self.model_trainer.set_model_params(best_state)
            final_validation = self._evaluate_validation(
                self.final_validation_max_triples
            )
            final_test = self.evaluator.evaluate(
                self.model_trainer.model,
                self.dataset.test_triples,
                self.device,
                max_triples=self.test_max_triples,
                seed=self.seed + 29,
                candidate_batch_size=self.candidate_batch_size,
                query_batch_size=self.query_batch_size,
            )
            final_test_mrr = float(final_test["mrr"])
            mrr_delta = (
                final_test_mrr - self.centralized_reference_mrr
                if math.isfinite(self.centralized_reference_mrr)
                else float("nan")
            )
            is_matlab_direct = (
                self.topology_metadata.get("provider_type")
                == "matlab_adapter"
            )
            configured_per_round = int(
                getattr(
                    self.args,
                    "client_num_per_round",
                    self.participation_summary[
                        "participant_count_max"
                    ],
                )
            )
            summary: Dict[str, object] = {
                "task": (
                    "dynamic_mat_federated_knowledge_graph_completion"
                ),
                "runtime": "fedml_dynamic_mat_hierarchical_transe",
                "dataset": self.dataset.dataset_name,
                "ablation_suite": str(
                    getattr(self.args, "ablation_suite", "")
                ).strip(),
                "ablation_arm": str(
                    getattr(self.args, "ablation_arm", "")
                ).strip(),
                "scenario": self.scenario_name,
                "architecture": self.architecture,
                "snf_enabled": bool(self.snf_enabled),
                "edge_mode": self.edge_mode,
                "aggregation_mode": self.aggregation_mode,
                "server_optimizer": self.server_optimizer_name,
                "server_learning_rate": (
                    float(
                        self.server_fedadam_optimizer.learning_rate
                    )
                    if self.server_fedadam_optimizer is not None
                    else 1.0
                ),
                "server_beta1": (
                    float(self.server_fedadam_optimizer.beta1)
                    if self.server_fedadam_optimizer is not None
                    else 0.0
                ),
                "server_beta2": (
                    float(self.server_fedadam_optimizer.beta2)
                    if self.server_fedadam_optimizer is not None
                    else 0.0
                ),
                "server_tau": (
                    float(self.server_fedadam_optimizer.tau)
                    if self.server_fedadam_optimizer is not None
                    else 0.0
                ),
                "server_bias_correction": (
                    bool(
                        self.server_fedadam_optimizer.bias_correction
                    )
                    if self.server_fedadam_optimizer is not None
                    else False
                ),
                "server_optimizer_step_count": (
                    int(self.server_fedadam_optimizer.step_count)
                    if self.server_fedadam_optimizer is not None
                    else 0
                ),
                "server_optimizer_state_hash": (
                    _state_dict_sha256(
                        self.server_fedadam_optimizer.state_dict()
                    )
                    if self.server_fedadam_optimizer is not None
                    else ""
                ),
                "local_objective": self.local_objective,
                "dynamic_client_selection": bool(
                    self.participation_summary[
                        "dynamic_client_selection"
                    ]
                ),
                "dynamic_grouping": bool(
                    self.participation_summary["dynamic_grouping"]
                ),
                "client_count": self.federated_data.client_count,
                "client_num_in_total": (
                    self.federated_data.client_count
                ),
                "client_num_per_round": (
                    self.participation_summary[
                        "participant_count_max"
                    ]
                    if is_matlab_direct
                    else configured_per_round
                ),
                "client_num_per_round_config": configured_per_round,
                "client_num_per_round_source": (
                    "matlab"
                    if is_matlab_direct
                    else "yaml_fixed_count"
                ),
                "participant_count_min": (
                    self.participation_summary[
                        "participant_count_min"
                    ]
                ),
                "participant_count_max": (
                    self.participation_summary[
                        "participant_count_max"
                    ]
                ),
                "participant_count_mean": (
                    self.participation_summary[
                        "participant_count_mean"
                    ]
                ),
                "group_count_min": self.participation_summary[
                    "group_count_min"
                ],
                "group_count_max": self.participation_summary[
                    "group_count_max"
                ],
                "group_count_mean": self.participation_summary[
                    "group_count_mean"
                ],
                "unique_participant_set_count": (
                    self.participation_summary[
                        "unique_participant_set_count"
                    ]
                ),
                "unique_topology_count": (
                    self.participation_summary[
                        "unique_topology_count"
                    ]
                ),
                "comm_round": self.comm_round,
                "topology_schedule_policy": (
                    self.participation_summary[
                        "topology_schedule_policy"
                    ]
                ),
                "source_topology_round_count": (
                    self.participation_summary[
                        "source_round_count"
                    ]
                ),
                "cycled_topology_round_count": (
                    self.participation_summary[
                        "cycled_round_count"
                    ]
                ),
                "local_epochs": int(
                    getattr(self.args, "epochs", 1)
                ),
                "client_optimizer_state_mode": str(
                    getattr(
                        self.args,
                        "client_optimizer_state_mode",
                        "reset",
                    )
                ),
                "optimizer_state_cached_client_count": int(
                    self.model_trainer.optimizer_state_cache_size
                ),
                "embedding_dim": int(
                    self.model_trainer.model.embedding_dim
                ),
                "distance_norm": int(
                    self.model_trainer.model.distance_norm
                ),
                "batch_size": int(
                    getattr(self.args, "batch_size", 0)
                ),
                "negative_sample_count": int(
                    getattr(self.args, "negative_sample_count", 0)
                ),
                "learning_rate": float(
                    getattr(
                        self.args,
                        "learning_rate",
                        getattr(self.args, "lr", 0.0),
                    )
                ),
                "effective_global_passes": (
                    self.participation_summary[
                        "effective_full_data_passes"
                    ]
                ),
                "aggregation": (
                    (
                        "hierarchical_two_level_row_count_weighted"
                        if self.architecture == "hfl"
                        else "direct_row_count_weighted"
                    )
                    if self.aggregation_mode == "row_count_weighted"
                    else (
                        "hierarchical_two_level_row_mask_presence"
                        if self.architecture == "hfl"
                        else "direct_row_mask_presence"
                    )
                    if self.aggregation_mode == "row_mask_presence"
                    else (
                        "hierarchical_two_level_dense_fedavg"
                        if self.architecture == "hfl"
                        else "direct_dense_fedavg"
                    )
                ),
                "aggregation_weight_basis": (
                    "local_positive_triple_row_occurrences"
                    if self.aggregation_mode == "row_count_weighted"
                    else (
                        "parameter_row_presence_equal"
                        if self.aggregation_mode
                        == "row_mask_presence"
                        else "local_positive_triples"
                    )
                ),
                "row_count_source": (
                    "local_positive_train_triples"
                    if self.aggregation_mode == "row_count_weighted"
                    else None
                ),
                "negative_sampling_backend": (
                    "torch_device_searchsorted"
                    if self.local_objective
                    == "bidirectional_self_adversarial"
                    else "legacy_numpy_per_triple"
                ),
                "subsampling_weights_cached_per_client": (
                    self.local_objective
                    == "bidirectional_self_adversarial"
                ),
                "profile_training_timing": as_bool(
                    getattr(
                        self.args,
                        "profile_training_timing",
                        False,
                    )
                ),
                "cuda_peak_memory_allocated_mb_max": float(
                    max(round_cuda_allocated_mb)
                    if round_cuda_allocated_mb
                    else 0.0
                ),
                "cuda_peak_memory_reserved_mb_max": float(
                    max(round_cuda_reserved_mb)
                    if round_cuda_reserved_mb
                    else 0.0
                ),
                "evaluation_query_batch_size": self.query_batch_size,
                "evaluation_candidate_batch_size": (
                    self.candidate_batch_size
                ),
                "best_round": best_round,
                "best_validation_mrr_during_training": (
                    best_validation_mrr
                ),
                "final_validation_metrics": final_validation,
                "final_test_metrics": final_test,
                "centralized_reference_test_mrr": (
                    self.centralized_reference_mrr
                ),
                "test_mrr_delta_vs_centralized": mrr_delta,
                "partition_hash": (
                    self.federated_data.partition_hash
                ),
                "topology_schedule_hash": (
                    self.participation_summary["schedule_hash"]
                ),
                "initial_model_hash": self.initial_model_hash,
                "mat_file": self.topology_metadata.get("mat_file"),
                "topology_util": self.topology_metadata.get(
                    "topology_util"
                ),
                "metrics_file": str(writer.metrics_path),
                "topology_schedule_file": str(
                    writer.topology_path
                ),
                "aggregation_audit_file": str(
                    writer.topology_path
                ),
                "topology_metadata_file": str(
                    Path(result_dir) / "topology_metadata.json"
                ),
                "dynamic_participation_summary_file": str(
                    Path(result_dir)
                    / "dynamic_participation_summary.json"
                ),
            }
            writer.write_json("summary.json", summary)
        return summary
