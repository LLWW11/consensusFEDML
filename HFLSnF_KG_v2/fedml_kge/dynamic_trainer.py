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

from ..core.aggregation import RowMaskedFedAvgAggregator
from ..core.result_writer import ExperimentResultWriter
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
        }:
            raise ValueError(
                "aggregation_mode必须是dense_triple_weighted"
                "或row_mask_presence"
            )
        if self.local_objective not in {
            "margin_ranking",
            "fede_self_adversarial",
        }:
            raise ValueError(
                "local_objective必须是margin_ranking或"
                "fede_self_adversarial"
            )
        self.row_masked_aggregator = RowMaskedFedAvgAggregator()
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
        if (
            available_rounds is not None
            and self.comm_round > int(available_rounds)
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

        if self.aggregation_mode == "row_mask_presence":
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
        # 两种聚合都会改变实体行范数，因此在云端统一投影。
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
        return {
            "topology_metadata": self.topology_metadata,
            "used_round_count": self.comm_round,
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
                round_seconds = (
                    time.perf_counter() - round_started_at
                )
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
                    "aggregation_mode": self.aggregation_mode,
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
                }
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
                        "dynamic_client_selection": True,
                        "dynamic_grouping": True,
                        "fedml_client_lifecycle": True,
                        "architecture": self.architecture,
                        "snf_enabled": bool(self.snf_enabled),
                        "edge_mode": self.edge_mode,
                        "aggregation_mode": self.aggregation_mode,
                        "local_objective": self.local_objective,
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
            )
            final_test_mrr = float(final_test["mrr"])
            mrr_delta = (
                final_test_mrr - self.centralized_reference_mrr
                if math.isfinite(self.centralized_reference_mrr)
                else float("nan")
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
                "client_num_per_round": "from_matlab",
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
                "local_epochs": int(
                    getattr(self.args, "epochs", 1)
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
                    "parameter_row_presence_equal"
                    if self.aggregation_mode == "row_mask_presence"
                    else "local_positive_triples"
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
