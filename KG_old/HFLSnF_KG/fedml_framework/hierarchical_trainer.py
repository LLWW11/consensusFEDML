"""从HFLSnF层次训练循环迁移并适配GCN任务的FedML训练器。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from FedGCN_fedml.data import FederatedGraphData

from ..core.aggregation import DenseFedAvgAggregator
from ..core.result_writer import ExperimentResultWriter
from ..core.topology import RoundTopology, TopologyProvider
from ..core.types import AggregateStats, ClientUpdate
from .client import FedMLGCNClient
from .group import FedMLHierarchicalGroup
from .model_trainer import FedMLGCNModelTrainer


class FedMLHierarchicalGCNTrainer:
    """执行FedML客户端训练、边缘聚合、云聚合和完整图评估。"""

    SUPPORTED_AGGREGATION_MODES = {"hierarchical", "direct"}

    def __init__(
        self,
        args,
        device: torch.device,
        dataset: FederatedGraphData,
        model_trainer: FedMLGCNModelTrainer,
        topology_provider: TopologyProvider,
    ):
        """初始化FedML客户端注册表、聚合器和拓扑提供器。"""

        self.args = args
        self.device = torch.device(device)
        self.dataset = dataset
        self.model_trainer = model_trainer
        self.topology_provider = topology_provider
        self.aggregator = DenseFedAvgAggregator()
        self.aggregation_mode = str(
            getattr(args, "aggregation_mode", "hierarchical")
        ).strip().lower()
        if self.aggregation_mode not in self.SUPPORTED_AGGREGATION_MODES:
            raise ValueError(
                "aggregation_mode 必须是{}，实际为{}".format(
                    sorted(self.SUPPORTED_AGGREGATION_MODES),
                    self.aggregation_mode,
                )
            )
        self.client_registry = self._build_client_registry()
        self._valid_client_ids = set(self.client_registry.keys())

    def _build_client_registry(self) -> Dict[int, FedMLGCNClient]:
        """为每个图分区创建一个复用共享ClientTrainer的FedML客户端。"""

        registry = {}
        for partition in self.dataset.partitions:
            client_id = int(partition.client_id)
            if client_id in registry:
                raise ValueError("GCN客户端分区编号{}重复".format(client_id))
            registry[client_id] = FedMLGCNClient(
                partition=partition,
                args=self.args,
                device=self.device,
                model_trainer=self.model_trainer,
            )
        if not registry:
            raise ValueError("至少需要一个GCN客户端")
        return registry

    def _validate_topology(self, topology: RoundTopology) -> None:
        """校验拓扑活跃客户端与FedML客户端注册表一致。"""

        active_ids = tuple(int(value) for value in topology.active_client_indexes)
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("当前拓扑包含重复活跃客户端")
        unknown_ids = sorted(set(active_ids).difference(self._valid_client_ids))
        if unknown_ids:
            raise ValueError(
                "当前拓扑包含没有数据分区的客户端：{}".format(unknown_ids)
            )
        flattened = sorted(
            int(client_id)
            for client_ids in topology.group_to_client_indexes.values()
            for client_id in client_ids
        )
        if flattened != sorted(active_ids):
            raise ValueError("拓扑活跃客户端列表与边缘分组不一致")

    def _train_round(
        self,
        topology: RoundTopology,
        global_state: Dict[str, torch.Tensor],
        round_index: int,
    ) -> Tuple[
        List[ClientUpdate],
        Sequence[AggregateStats],
        Dict[str, torch.Tensor],
        float,
    ]:
        """通过FedML客户端和边缘组完成一轮训练及聚合。"""

        all_updates: List[ClientUpdate] = []
        edge_stats: List[AggregateStats] = []
        for group_id in sorted(topology.group_to_client_indexes.keys()):
            group = FedMLHierarchicalGroup(
                group_id=group_id,
                client_registry=self.client_registry,
                aggregator=self.aggregator,
            )
            updates, statistics = group.train_and_accumulate(
                topology.group_to_client_indexes[group_id],
                global_state,
                round_index,
            )
            all_updates.extend(updates)
            if statistics is not None:
                edge_stats.append(statistics)

        if not all_updates:
            return all_updates, (), global_state, 0.0
        if self.aggregation_mode == "direct":
            cloud_stats = self.aggregator.accumulate(all_updates)
            return (
                all_updates,
                (),
                self.aggregator.finalize(cloud_stats),
                float(cloud_stats.total_weight),
            )

        cloud_stats = self.aggregator.merge(edge_stats)
        return (
            all_updates,
            tuple(edge_stats),
            self.aggregator.finalize(cloud_stats),
            float(cloud_stats.total_weight),
        )

    @staticmethod
    def _weighted_local_loss(updates: Sequence[ClientUpdate]) -> float:
        """按本地标注训练节点数计算客户端训练损失加权均值。"""

        numerator = 0.0
        denominator = 0.0
        for update in updates:
            loss = float(update.local_metrics.get("train_loss", float("nan")))
            if not math.isfinite(loss):
                continue
            numerator += float(update.weight) * loss
            denominator += float(update.weight)
        if denominator <= 0.0:
            return float("nan")
        return numerator / denominator

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回FedML共享模型当前参数的CPU深拷贝。"""

        return self.model_trainer.get_model_params()

    def train(self, result_dir: Path) -> Dict[str, object]:
        """执行完整FedML分层GCN训练并写出逐轮结果。"""

        comm_round = int(self.args.comm_round)
        if comm_round <= 0:
            raise ValueError("comm_round 必须大于0")
        final_metrics: Dict[str, float] = {}
        aggregated_rounds = 0

        with ExperimentResultWriter(result_dir) as writer:
            writer.write_json(
                "topology_metadata.json", self.topology_provider.describe()
            )
            writer.write_json(
                "partition_summary.json",
                {
                    "task": "gcn_node_classification_{}".format(
                        self.dataset.dataset_name
                    ),
                    "dataset": self.dataset.dataset_name,
                    "node_count": self.dataset.num_nodes,
                    "feature_count": self.dataset.num_features,
                    "class_count": self.dataset.num_classes,
                    "client_count": len(self.dataset.partitions),
                    "clients": [
                        {
                            "client_id": int(partition.client_id),
                            "node_count": int(partition.node_count),
                            "labeled_train_node_count": int(
                                partition.train_node_count
                            ),
                            "node_indices": partition.node_indices.tolist(),
                        }
                        for partition in self.dataset.partitions
                    ],
                },
            )

            for round_index in range(comm_round):
                topology = self.topology_provider.get_round(round_index)
                self._validate_topology(topology)
                global_state = self.get_global_state()
                (
                    all_updates,
                    edge_stats,
                    new_global_state,
                    total_weight,
                ) = self._train_round(
                    topology, global_state, round_index
                )
                aggregated = bool(all_updates)
                if aggregated:
                    self.model_trainer.set_model_params(new_global_state)
                    aggregated_rounds += 1

                final_metrics = self.model_trainer.evaluate_full_graph(
                    self.dataset, self.device
                )
                metric_row: Dict[str, object] = {
                    "round": int(round_index),
                    "active_client_count": int(topology.participant_count),
                    "contributing_client_count": len(all_updates),
                    "edge_group_count": len(
                        topology.group_to_client_indexes
                    ),
                    "contributing_edge_count": len(edge_stats),
                    "aggregated": int(aggregated),
                    "aggregation_weight": float(total_weight),
                    "mean_client_train_loss": self._weighted_local_loss(
                        all_updates
                    ),
                }
                for metric_name in sorted(final_metrics.keys()):
                    metric_row[metric_name] = float(final_metrics[metric_name])
                writer.write_metrics(metric_row)
                writer.write_topology(
                    {
                        "round": int(round_index),
                        "source_round_index": int(
                            topology.source_round_index
                        ),
                        "fedml_runner": True,
                        "aggregation_mode": self.aggregation_mode,
                        "group_to_client_indexes": {
                            str(group_id): list(client_ids)
                            for group_id, client_ids
                            in topology.group_to_client_indexes.items()
                        },
                        "active_client_indexes": list(
                            topology.active_client_indexes
                        ),
                        "contributing_client_indexes": [
                            int(update.client_id) for update in all_updates
                        ],
                        "edge_node_ids": {
                            str(group_id): int(edge_id)
                            for group_id, edge_id
                            in topology.edge_node_ids.items()
                        },
                        "aggregated": bool(aggregated),
                        "aggregation_weight": float(total_weight),
                    }
                )

            summary: Dict[str, object] = {
                "task": "gcn_node_classification_{}".format(
                    self.dataset.dataset_name
                ),
                "runtime": "fedml_framework",
                "comm_round": comm_round,
                "local_epochs": int(self.args.epochs),
                "aggregation_mode": self.aggregation_mode,
                "aggregated_rounds": aggregated_rounds,
                "final_metrics": final_metrics,
                "metrics_file": str(writer.metrics_path),
                "topology_file": str(writer.topology_path),
            }
            writer.write_json("summary.json", summary)
        return summary
