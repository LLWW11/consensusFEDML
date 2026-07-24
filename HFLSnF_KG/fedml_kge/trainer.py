"""FedML普通单层FedAvg TransE训练、选模和全局评估。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from ..core.aggregation import DenseFedAvgAggregator
from ..core.result_writer import ExperimentResultWriter
from ..core.types import ClientUpdate, clone_state_dict
from ..tasks.kge.evaluator import FilteredRankingEvaluator
from ..tasks.kge.federated_data import FederatedKnowledgeGraphData
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
    ):
        """初始化客户端注册表、稠密FedAvg和全局排名评估器。"""

        self.args = args
        self.device = torch.device(device)
        self.federated_data = federated_data
        self.dataset = federated_data.dataset
        self.model_trainer = model_trainer
        self.aggregator = DenseFedAvgAggregator()
        self.evaluator = FilteredRankingEvaluator(self.dataset)
        self.comm_round = int(getattr(args, "comm_round", 1))
        self.eval_every = int(getattr(args, "eval_every", 1))
        self.validation_max_triples = int(
            getattr(args, "validation_max_triples", 0)
        )
        self.final_validation_max_triples = int(
            getattr(args, "final_validation_max_triples", 0)
        )
        self.test_max_triples = int(
            getattr(args, "test_max_triples", 0)
        )
        self.candidate_batch_size = int(
            getattr(args, "evaluation_candidate_batch_size", 4096)
        )
        self.seed = int(getattr(args, "random_seed", 0))
        self.centralized_reference_mrr = float(
            getattr(args, "centralized_reference_mrr", float("nan"))
        )
        self._validate_configuration()
        self.client_registry = self._build_client_registry()

    def _validate_configuration(self) -> None:
        """校验阶段四为全客户端参与的单层普通联邦实验。"""

        if self.comm_round <= 0:
            raise ValueError("comm_round必须大于0")
        if self.eval_every <= 0:
            raise ValueError("eval_every必须大于0")
        if self.candidate_batch_size <= 0:
            raise ValueError("evaluation_candidate_batch_size必须大于0")
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
        if configured_per_round != expected_client_count:
            raise ValueError(
                "阶段四要求每轮全部{}个客户端参与，实际为{}".format(
                    expected_client_count, configured_per_round
                )
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

    def _train_round(
        self, round_index: int
    ) -> Sequence[ClientUpdate]:
        """让全部知识客户端从同一全局参数开始完成一轮本地训练。"""

        global_state = self.get_global_state()
        updates: List[ClientUpdate] = []
        for client_id in sorted(self.client_registry.keys()):
            updates.append(
                self.client_registry[client_id].train_from_global(
                    global_state, round_index
                )
            )
        new_global_state = self.aggregator.aggregate(updates)
        self.model_trainer.set_model_params(new_global_state)
        # FedAvg后的单位向量平均值通常不再是单位范数，重新执行TransE约束投影。
        self.model_trainer.model.to(self.device)
        self.model_trainer.model.normalize_entity_embeddings()
        return tuple(updates)

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

    def _evaluate_validation(self, max_triples: int) -> Dict[str, float]:
        """在全局验证集上计算filtered排名指标。"""

        return self.evaluator.evaluate(
            self.model_trainer.model,
            self.dataset.valid_triples,
            self.device,
            max_triples=max_triples,
            seed=self.seed + 17,
            candidate_batch_size=self.candidate_batch_size,
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

    def train(self, result_dir: Path) -> Dict[str, object]:
        """执行普通联邦TransE训练、最佳模型恢复和最终完整评估。"""

        best_state = clone_state_dict(
            self.get_global_state(), to_cpu=True
        )
        best_round = 0
        best_validation_mrr = -math.inf
        all_client_ids = sorted(self.client_registry.keys())

        with ExperimentResultWriter(
            result_dir,
            schedule_filename="participation_schedule.jsonl",
        ) as writer:
            writer.write_json(
                "client_partition_summary.json",
                self.federated_data.summary(),
            )
            for round_index in range(self.comm_round):
                updates = self._train_round(round_index)
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
                metric_row: Dict[str, object] = {
                    "round": round_number,
                    "active_client_count": len(all_client_ids),
                    "contributing_client_count": len(updates),
                    "aggregation_weight": aggregation_weight,
                    "mean_client_train_loss": self._weighted_local_loss(
                        updates
                    ),
                }
                metric_row.update(
                    self._validation_fields(validation_metrics)
                )
                writer.write_metrics(metric_row)
                writer.write_topology(
                    {
                        "round": round_number,
                        "fedml_client_lifecycle": True,
                        "aggregation": "direct_dense_fedavg",
                        "active_client_indexes": all_client_ids,
                        "contributing_client_indexes": [
                            int(update.client_id) for update in updates
                        ],
                        "client_weights": {
                            str(update.client_id): float(update.weight)
                            for update in updates
                        },
                        "aggregation_weight": aggregation_weight,
                    }
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
                "task": "federated_knowledge_graph_completion",
                "runtime": "fedml_fedavg_transe",
                "dataset": self.dataset.dataset_name,
                "client_count": self.federated_data.client_count,
                "comm_round": self.comm_round,
                "local_epochs": int(getattr(self.args, "epochs", 1)),
                "effective_global_passes": (
                    self.comm_round * int(getattr(self.args, "epochs", 1))
                ),
                "aggregation": "direct_dense_fedavg",
                "aggregation_weight_basis": "local_positive_triples",
                "best_round": best_round,
                "best_validation_mrr_during_training": best_validation_mrr,
                "final_validation_metrics": final_validation,
                "final_test_metrics": final_test,
                "centralized_reference_test_mrr": (
                    self.centralized_reference_mrr
                ),
                "test_mrr_delta_vs_centralized": mrr_delta,
                "partition_hash": self.federated_data.partition_hash,
                "metrics_file": str(writer.metrics_path),
                "participation_file": str(writer.topology_path),
            }
            writer.write_json("summary.json", summary)
        return summary
