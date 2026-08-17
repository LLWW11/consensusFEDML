"""FedML普通单层FedAvg TransE训练、选模和全局评估。"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..core.aggregation import DenseFedAvgAggregator
from ..core.device import as_bool
from ..core.result_writer import ExperimentResultWriter
from ..core.types import ClientUpdate, clone_state_dict
from ..tasks.kge.evaluator import FilteredRankingEvaluator
from ..tasks.kge.federated_data import FederatedKnowledgeGraphData
from ..tasks.kge.fixed_topology import FixedParticipantTopology
from .client import FedMLTransEClient
from .model_trainer import FedMLTransEModelTrainer


class FedMLFederatedTransETrainer:
    """执行全客户端本地训练、直接云FedAvg和filtered全局评估。"""

    def __init__(
        self,
        args,
        device: torch.device,
        federated_data: FederatedKnowledgeGraphData,
        model_trainer: FedMLTransEModelTrainer,
        fixed_topology: Optional[FixedParticipantTopology] = None,
    ):
        """初始化客户端、固定拓扑、稠密FedAvg和全局排名评估器。"""

        self.args = args
        self.device = torch.device(device)
        self.federated_data = federated_data
        self.dataset = federated_data.dataset
        self.model_trainer = model_trainer
        self.fixed_topology = fixed_topology
        self.aggregator = DenseFedAvgAggregator()
        self.evaluator = FilteredRankingEvaluator(self.dataset)
        self.comm_round = int(getattr(args, "comm_round", 1))
        self.eval_every = int(getattr(args, "eval_every", 1))
        self.validation_max_triples = int(
            getattr(args, "validation_max_triples", 0)
        )
        self.validation_selection = str(
            getattr(args, "validation_selection", "random")
        ).strip().lower()
        self.final_validation_max_triples = int(
            getattr(args, "final_validation_max_triples", 0)
        )
        self.test_max_triples = int(
            getattr(args, "test_max_triples", 0)
        )
        self.candidate_batch_size = int(
            getattr(args, "evaluation_candidate_batch_size", 4096)
        )
        self.query_batch_size = int(
            getattr(args, "evaluation_query_batch_size", 1)
        )
        self.seed = int(getattr(args, "random_seed", 0))
        self.centralized_reference_mrr = float(
            getattr(args, "centralized_reference_mrr", float("nan"))
        )
        self._validate_configuration()
        self.client_registry = self._build_client_registry()

    def _validate_configuration(self) -> None:
        """校验普通全参与、固定拓扑或动态拓扑的客户端人数合同。"""

        if self.comm_round <= 0:
            raise ValueError("comm_round必须大于0")
        if self.eval_every <= 0:
            raise ValueError("eval_every必须大于0")
        if self.candidate_batch_size <= 0:
            raise ValueError("evaluation_candidate_batch_size必须大于0")
        if self.query_batch_size <= 0:
            raise ValueError("evaluation_query_batch_size必须大于0")
        if self.validation_selection not in {
            "random",
            "relation_stratified",
        }:
            raise ValueError(
                "validation_selection必须是random或"
                "relation_stratified"
            )
        expected_client_count = self.federated_data.client_count
        configured_total = int(
            getattr(self.args, "client_num_in_total", expected_client_count)
        )
        configured_per_round = int(
            getattr(self.args, "client_num_per_round", expected_client_count)
        )
        if configured_total != expected_client_count:
            raise ValueError(
                "client_num_in_total={}与数据分区数{}不一致".format(
                    configured_total, expected_client_count
                )
            )
        optimizer_name = str(
            getattr(self.args, "federated_optimizer", "")
        ).strip().lower()
        uses_dynamic_topology = (
            optimizer_name == "dynamictopologytranse"
        )
        if uses_dynamic_topology and not (
            0 < configured_per_round <= expected_client_count
        ):
            raise ValueError(
                "动态拓扑client_num_per_round必须位于1和{}之间，"
                "实际为{}".format(
                    expected_client_count,
                    configured_per_round,
                )
            )
        if (
            self.fixed_topology is None
            and not uses_dynamic_topology
            and (
                configured_per_round != expected_client_count
            )
        ):
            raise ValueError(
                "阶段四要求每轮全部{}个客户端参与，实际为{}".format(
                    expected_client_count, configured_per_round
                )
            )
        if self.fixed_topology is not None:
            if (
                self.fixed_topology.client_num_in_total
                != expected_client_count
            ):
                raise ValueError("固定拓扑的客户端总数与数据分区数不一致")
            if (
                configured_per_round
                != self.fixed_topology.client_num_per_round
            ):
                raise ValueError(
                    "client_num_per_round与固定拓扑参与人数不一致"
                )

    def _build_client_registry(self) -> Dict[int, FedMLTransEClient]:
        """为每个三元组分区创建共享ClientTrainer的FedML客户端。"""

        return {
            int(partition.client_id): FedMLTransEClient(
                partition=partition,
                args=self.args,
                device=self.device,
                model_trainer=self.model_trainer,
            )
            for partition in self.federated_data.partitions
        }

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回当前全局TransE模型参数的CPU深拷贝。"""

        return self.model_trainer.get_model_params()

    def _active_client_ids(self) -> Tuple[int, ...]:
        """返回普通全参与或固定拓扑方案的活跃客户端编号。"""

        if self.fixed_topology is None:
            return tuple(sorted(self.client_registry.keys()))
        return self.fixed_topology.active_client_ids

    def _aggregate_updates(
        self,
        updates: Sequence[ClientUpdate],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """按FL直连或HFL组内到云端两级方式聚合客户端更新。"""

        if (
            self.fixed_topology is None
            or self.fixed_topology.architecture == "fl"
        ):
            return self.aggregator.aggregate(updates), {
                "aggregation": "direct_dense_fedavg",
                "edge_group_count": 0,
                "group_aggregation_weights": {},
                "group_contributing_client_indexes": {},
            }

        update_by_client = {
            int(update.client_id): update for update in updates
        }
        edge_statistics = []
        group_weights: Dict[str, float] = {}
        group_contributors: Dict[str, List[int]] = {}
        for group_id, client_ids in (
            self.fixed_topology.group_mapping().items()
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
                int(value) for value in statistics.contributor_ids
            ]
        cloud_statistics = self.aggregator.merge(edge_statistics)
        return self.aggregator.finalize(cloud_statistics), {
            "aggregation": "hierarchical_two_level_dense_fedavg",
            "edge_group_count": len(edge_statistics),
            "group_aggregation_weights": group_weights,
            "group_contributing_client_indexes": group_contributors,
        }

    def _train_round(
        self, round_index: int
    ) -> Tuple[Sequence[ClientUpdate], Dict[str, object]]:
        """让固定活跃客户端从同一全局参数开始完成一轮本地训练。"""

        global_state = self.get_global_state()
        updates: List[ClientUpdate] = []
        for client_id in self._active_client_ids():
            updates.append(
                self.client_registry[client_id].train_from_global(
                    global_state, round_index
                )
            )
        new_global_state, aggregation_details = self._aggregate_updates(
            updates
        )
        self.model_trainer.set_model_params(new_global_state)
        # FedAvg后的单位向量平均值通常不再是单位范数，重新执行TransE约束投影。
        self.model_trainer.model.to(self.device)
        self.model_trainer.model.normalize_entity_embeddings()
        return tuple(updates), aggregation_details

    @staticmethod
    def _weighted_local_loss(
        updates: Sequence[ClientUpdate],
    ) -> float:
        """按本地正三元组数计算客户端训练损失加权均值。"""

        numerator = 0.0
        denominator = 0.0
        for update in updates:
            loss = float(update.local_metrics["train_loss"])
            if not math.isfinite(loss):
                continue
            numerator += float(update.weight) * loss
            denominator += float(update.weight)
        if denominator <= 0.0:
            raise RuntimeError("没有有限的客户端训练损失")
        return numerator / denominator

    @staticmethod
    def _summed_local_metric(
        updates: Sequence[ClientUpdate], field: str
    ) -> float:
        """汇总顺序执行客户端的有限分段耗时，未剖析时返回缺失值。"""

        values = [
            float(update.local_metrics.get(field, float("nan")))
            for update in updates
        ]
        if not values or not all(math.isfinite(value) for value in values):
            return float("nan")
        return float(sum(values))

    @staticmethod
    def _local_metric_statistics(
        updates: Sequence[ClientUpdate],
        field: str,
    ) -> Dict[str, float]:
        """返回一个有限客户端指标的最小值、均值和最大值。"""

        values = [
            float(update.local_metrics.get(field, float("nan")))
            for update in updates
        ]
        if not values or not all(
            math.isfinite(value) for value in values
        ):
            return {
                "min": float("nan"),
                "mean": float("nan"),
                "max": float("nan"),
            }
        return {
            "min": float(min(values)),
            "mean": float(sum(values) / len(values)),
            "max": float(max(values)),
        }

    def _evaluate_validation(self, max_triples: int) -> Dict[str, float]:
        """在全局验证集上计算filtered排名指标。"""

        return self.evaluator.evaluate(
            self.model_trainer.model,
            self.dataset.valid_triples,
            self.device,
            max_triples=max_triples,
            seed=self.seed + 17,
            candidate_batch_size=self.candidate_batch_size,
            query_batch_size=self.query_batch_size,
            relation_stratified=(
                self.validation_selection
                == "relation_stratified"
            ),
        )

    @staticmethod
    def _validation_fields(
        metrics: Dict[str, float] = None,
    ) -> Dict[str, float]:
        """返回字段固定的逐轮验证指标。"""

        metrics = metrics or {}
        return {
            "val_mrr": float(metrics.get("mrr", float("nan"))),
            "val_mean_rank": float(
                metrics.get("mean_rank", float("nan"))
            ),
            "val_hits_at_1": float(
                metrics.get("hits_at_1", float("nan"))
            ),
            "val_hits_at_3": float(
                metrics.get("hits_at_3", float("nan"))
            ),
            "val_hits_at_10": float(
                metrics.get("hits_at_10", float("nan"))
            ),
            "val_evaluated_triple_count": float(
                metrics.get(
                    "evaluated_triple_count", float("nan")
                )
            ),
        }

    @staticmethod
    def _print_round_progress(
        round_number: int,
        total_rounds: int,
        scenario_name: str,
        architecture: str,
        active_client_count: int,
        active_group_count: int,
        weighted_loss: float,
        aggregation_weight: float,
        round_seconds: float,
        validation_metrics: Optional[Dict[str, float]],
    ) -> None:
        """打印一轮联邦训练的进度、损失、耗时和可用验证指标。"""

        # 非选模轮仍打印训练信息，但不为终端输出额外执行昂贵的排名评估。
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
                "[联邦TransE] 方案={} epoch={}/{} "
                "结构={} 参与客户端={} 分组={} "
                "加权损失={:.6f} 聚合权重={:.0f} "
                "耗时={:.2f}s 验证MRR={} 验证Hits@3={}"
            ).format(
                scenario_name,
                round_number,
                total_rounds,
                architecture.upper(),
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

    def _fixed_participation_payload(self) -> Dict[str, object]:
        """补充固定参与客户端及六组本地知识规模统计。"""

        if self.fixed_topology is None:
            raise RuntimeError("普通全参与阶段没有固定对照拓扑")
        partition_by_id = {
            int(partition.client_id): partition
            for partition in self.federated_data.partitions
        }
        payload = self.fixed_topology.summary()
        selected_partitions = [
            partition_by_id[client_id]
            for client_id in self.fixed_topology.active_client_ids
        ]
        selected_triple_count = sum(
            partition.triple_count for partition in selected_partitions
        )
        total_triple_count = int(
            self.dataset.train_triples.shape[0]
        )
        payload["selected_train_triple_count"] = int(
            selected_triple_count
        )
        payload["selected_train_triple_fraction"] = (
            float(selected_triple_count) / float(total_triple_count)
        )
        payload["selected_client_data"] = [
            partition.summary() for partition in selected_partitions
        ]
        group_data: Dict[str, object] = {}
        for group_id, client_ids in (
            self.fixed_topology.group_mapping().items()
        ):
            group_partitions = [
                partition_by_id[int(client_id)]
                for client_id in client_ids
            ]
            group_entities = set()
            group_relations = set()
            for partition in group_partitions:
                group_entities.update(
                    int(value) for value in partition.entity_ids.tolist()
                )
                group_relations.update(
                    int(value) for value in partition.relation_ids.tolist()
                )
            group_data[str(group_id)] = {
                "client_ids": [
                    int(value) for value in client_ids
                ],
                "client_count": len(client_ids),
                "train_triple_count": int(
                    sum(
                        partition.triple_count
                        for partition in group_partitions
                    )
                ),
                "entity_count": len(group_entities),
                "relation_count": len(group_relations),
            }
        payload["group_data"] = group_data
        return payload

    def train(self, result_dir: Path) -> Dict[str, object]:
        """执行全参与或固定拓扑联邦TransE训练与最终完整评估。"""

        best_state = clone_state_dict(
            self.get_global_state(), to_cpu=True
        )
        best_round = 0
        best_validation_mrr = -math.inf
        active_client_ids = list(self._active_client_ids())
        if self.fixed_topology is None:
            scenario_name = "FedAvgTransE"
            architecture = "fl"
            snf_enabled = False
            topology_summary = None
            group_mapping = {0: tuple(active_client_ids)}
        else:
            scenario_name = self.fixed_topology.scenario_name
            architecture = self.fixed_topology.architecture
            snf_enabled = self.fixed_topology.snf_enabled
            topology_summary = self._fixed_participation_payload()
            group_mapping = self.fixed_topology.group_mapping()

        with ExperimentResultWriter(
            result_dir,
            schedule_filename="participation_schedule.jsonl",
        ) as writer:
            writer.write_json(
                "client_partition_summary.json",
                self.federated_data.summary(),
            )
            if topology_summary is not None:
                writer.write_json(
                    "fixed_participation.json",
                    topology_summary,
                )
            for round_index in range(self.comm_round):
                round_started_at = time.perf_counter()
                updates, aggregation_details = self._train_round(
                    round_index
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
                weighted_local_loss = self._weighted_local_loss(updates)
                round_seconds = time.perf_counter() - round_started_at
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
                metric_row: Dict[str, object] = {
                    "round": round_number,
                    "scenario": scenario_name,
                    "architecture": architecture,
                    "snf_enabled": bool(snf_enabled),
                    "active_client_count": len(active_client_ids),
                    "active_group_count": (
                        len(group_mapping)
                        if architecture == "hfl"
                        else 0
                    ),
                    "contributing_client_count": len(updates),
                    "aggregation_weight": aggregation_weight,
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
                    "round_seconds": round_seconds,
                }
                metric_row.update(
                    self._validation_fields(validation_metrics)
                )
                writer.write_metrics(metric_row)
                writer.write_topology(
                    {
                        "round": round_number,
                        "scenario": scenario_name,
                        "dynamic_client_selection": False,
                        "fedml_client_lifecycle": True,
                        "architecture": architecture,
                        "snf_enabled": bool(snf_enabled),
                        **aggregation_details,
                        "active_client_indexes": active_client_ids,
                        "contributing_client_indexes": [
                            int(update.client_id) for update in updates
                        ],
                        "group_to_client_indexes": {
                            str(group_id): [
                                int(value) for value in client_ids
                            ]
                            for group_id, client_ids in (
                                group_mapping.items()
                            )
                        },
                        "client_weights": {
                            str(update.client_id): float(update.weight)
                            for update in updates
                        },
                        "aggregation_weight": aggregation_weight,
                    }
                )
                self._print_round_progress(
                    round_number=round_number,
                    total_rounds=self.comm_round,
                    scenario_name=scenario_name,
                    architecture=architecture,
                    active_client_count=len(active_client_ids),
                    active_group_count=(
                        len(group_mapping)
                        if architecture == "hfl"
                        else 1
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
            summary: Dict[str, object] = {
                "task": (
                    "fixed_topology_federated_knowledge_graph_completion"
                    if self.fixed_topology is not None
                    else "federated_knowledge_graph_completion"
                ),
                "runtime": (
                    "fedml_fixed_topology_transe"
                    if self.fixed_topology is not None
                    else "fedml_fedavg_transe"
                ),
                "dataset": self.dataset.dataset_name,
                "scenario": scenario_name,
                "architecture": architecture,
                "snf_enabled": bool(snf_enabled),
                "dynamic_client_selection": False,
                "client_count": self.federated_data.client_count,
                "client_num_in_total": self.federated_data.client_count,
                "client_num_per_round": len(active_client_ids),
                "active_client_ids": active_client_ids,
                "group_num": (
                    len(group_mapping)
                    if architecture == "hfl"
                    else 1
                ),
                "group_to_client_indexes": {
                    str(group_id): [
                        int(value) for value in client_ids
                    ]
                    for group_id, client_ids in group_mapping.items()
                },
                "comm_round": self.comm_round,
                "local_epochs": int(getattr(self.args, "epochs", 1)),
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
                "effective_global_passes": (
                    self.comm_round * int(getattr(self.args, "epochs", 1))
                ),
                "aggregation": (
                    "hierarchical_two_level_dense_fedavg"
                    if architecture == "hfl"
                    else "direct_dense_fedavg"
                ),
                "aggregation_weight_basis": "local_positive_triples",
                "negative_sampling_backend": (
                    "torch_device_searchsorted"
                    if str(
                        getattr(
                            self.args,
                            "local_objective",
                            "margin_ranking",
                        )
                    ).strip().lower()
                    == "bidirectional_self_adversarial"
                    else "legacy_numpy_per_triple"
                ),
                "subsampling_weights_cached_per_client": (
                    str(
                        getattr(
                            self.args,
                            "local_objective",
                            "margin_ranking",
                        )
                    ).strip().lower()
                    == "bidirectional_self_adversarial"
                ),
                "profile_training_timing": as_bool(
                    getattr(
                        self.args,
                        "profile_training_timing",
                        False,
                    )
                ),
                "best_round": best_round,
                "best_validation_mrr_during_training": best_validation_mrr,
                "final_validation_metrics": final_validation,
                "final_test_metrics": final_test,
                "centralized_reference_test_mrr": (
                    self.centralized_reference_mrr
                ),
                "test_mrr_delta_vs_centralized": mrr_delta,
                "partition_hash": self.federated_data.partition_hash,
                "participant_set_hash": (
                    self.fixed_topology.participant_set_hash
                    if self.fixed_topology is not None
                    else None
                ),
                "topology_hash": (
                    self.fixed_topology.topology_hash
                    if self.fixed_topology is not None
                    else None
                ),
                "metrics_file": str(writer.metrics_path),
                "evaluation_query_batch_size": self.query_batch_size,
                "evaluation_candidate_batch_size": (
                    self.candidate_batch_size
                ),
                "participation_file": str(writer.topology_path),
                "fixed_participation_file": (
                    str(Path(result_dir) / "fixed_participation.json")
                    if self.fixed_topology is not None
                    else None
                ),
            }
            writer.write_json("summary.json", summary)
        return summary
