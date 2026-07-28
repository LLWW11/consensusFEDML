"""任务无关的单进程分层联邦学习模拟器。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .aggregation import DenseFedAvgAggregator
from .result_writer import ExperimentResultWriter
from .topology import RoundTopology, TopologyProvider
from .types import AggregateStats, ClientUpdate
from ..tasks.base import FederatedTask


class HierarchicalSimulator:
    """顺序执行客户端本地训练、边缘聚合、云聚合和全局评估。"""

    SUPPORTED_AGGREGATION_MODES = {"hierarchical", "direct"}

    def __init__(
        self,
        task: FederatedTask,
        topology_provider: TopologyProvider,
        aggregator: DenseFedAvgAggregator,
        comm_round: int,
        local_epochs: int,
        aggregation_mode: str,
    ):
        """保存任务、拓扑和训练轮次，并校验模拟器配置。"""

        self.task = task
        self.topology_provider = topology_provider
        self.aggregator = aggregator
        self.comm_round = int(comm_round)
        self.local_epochs = int(local_epochs)
        self.aggregation_mode = str(aggregation_mode).strip().lower()
        if self.comm_round <= 0:
            raise ValueError("comm_round 必须大于0")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs 必须大于0")
        if self.aggregation_mode not in self.SUPPORTED_AGGREGATION_MODES:
            raise ValueError(
                "aggregation_mode 必须是{}，实际为{}".format(
                    sorted(self.SUPPORTED_AGGREGATION_MODES),
                    self.aggregation_mode,
                )
            )
        self._valid_client_ids = set(int(value) for value in task.client_ids)
        if not self._valid_client_ids:
            raise ValueError("联邦任务必须至少包含一个客户端")

    def _validate_topology(self, topology: RoundTopology) -> None:
        """校验当前拓扑中的活跃客户端均属于任务数据分区。"""

        active_ids = list(int(value) for value in topology.active_client_indexes)
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("当前通信轮包含重复活跃客户端")
        unknown_ids = sorted(set(active_ids).difference(self._valid_client_ids))
        if unknown_ids:
            raise ValueError(
                "当前拓扑包含没有数据分区的客户端：{}".format(unknown_ids)
            )
        flattened_ids = sorted(
            client_id
            for client_ids in topology.group_to_client_indexes.values()
            for client_id in client_ids
        )
        if flattened_ids != sorted(active_ids):
            raise ValueError("拓扑活跃客户端列表与边缘分组不一致")

    def _train_round_clients(
        self,
        topology: RoundTopology,
        global_state: Dict[str, torch.Tensor],
        round_index: int,
    ) -> Tuple[Dict[int, List[ClientUpdate]], List[ClientUpdate]]:
        """按边缘组训练当前活跃客户端，并同时返回分组和扁平更新列表。"""

        group_updates: Dict[int, List[ClientUpdate]] = {}
        all_updates: List[ClientUpdate] = []
        for group_id in sorted(topology.group_to_client_indexes.keys()):
            updates = []
            for client_id in topology.group_to_client_indexes[group_id]:
                update = self.task.train_client(
                    client_id=int(client_id),
                    global_state=global_state,
                    local_epochs=self.local_epochs,
                    round_index=int(round_index),
                )
                if update is None or update.weight <= 0.0:
                    continue
                updates.append(update)
                all_updates.append(update)
            if updates:
                group_updates[int(group_id)] = updates
        return group_updates, all_updates

    def _aggregate_round(
        self,
        group_updates: Dict[int, List[ClientUpdate]],
        all_updates: Sequence[ClientUpdate],
    ) -> Tuple[Dict[str, torch.Tensor], Sequence[AggregateStats], float]:
        """按配置执行直接云聚合或边缘—云两级聚合。"""

        if not all_updates:
            raise ValueError("没有客户端更新时不应调用聚合函数")

        if self.aggregation_mode == "direct":
            cloud_stats = self.aggregator.accumulate(all_updates)
            return (
                self.aggregator.finalize(cloud_stats),
                (),
                float(cloud_stats.total_weight),
            )

        edge_stats = []
        for group_id in sorted(group_updates.keys()):
            edge_stats.append(
                self.aggregator.accumulate(group_updates[group_id])
            )
        cloud_stats = self.aggregator.merge(edge_stats)
        return (
            self.aggregator.finalize(cloud_stats),
            tuple(edge_stats),
            float(cloud_stats.total_weight),
        )

    @staticmethod
    def _weighted_local_loss(updates: Sequence[ClientUpdate]) -> float:
        """按客户端聚合权重计算本轮本地训练损失均值。"""

        numerator = 0.0
        denominator = 0.0
        for update in updates:
            if "train_loss" not in update.local_metrics:
                continue
            loss = float(update.local_metrics["train_loss"])
            if not math.isfinite(loss):
                continue
            numerator += float(update.weight) * loss
            denominator += float(update.weight)
        if denominator <= 0.0:
            return float("nan")
        return numerator / denominator

    def run(self, result_dir: Path) -> Dict[str, object]:
        """完成全部通信轮并写出逐轮指标、拓扑记录和最终摘要。"""

        final_metrics: Dict[str, float] = {}
        aggregated_rounds = 0
        with ExperimentResultWriter(result_dir) as writer:
            writer.write_json("topology_metadata.json", self.topology_provider.describe())
            writer.write_json("partition_summary.json", self.task.partition_summary())

            for round_index in range(self.comm_round):
                topology = self.topology_provider.get_round(round_index)
                self._validate_topology(topology)
                global_state = self.task.get_global_state()
                group_updates, all_updates = self._train_round_clients(
                    topology, global_state, round_index
                )

                aggregated = bool(all_updates)
                edge_stats: Sequence[AggregateStats] = ()
                total_weight = 0.0
                if aggregated:
                    new_state, edge_stats, total_weight = self._aggregate_round(
                        group_updates, all_updates
                    )
                    self.task.set_global_state(new_state)
                    aggregated_rounds += 1

                final_metrics = self.task.evaluate_global()
                metric_row: Dict[str, object] = {
                    "round": int(round_index),
                    "active_client_count": int(topology.participant_count),
                    "contributing_client_count": len(all_updates),
                    "edge_group_count": len(topology.group_to_client_indexes),
                    "contributing_edge_count": len(edge_stats),
                    "aggregated": int(aggregated),
                    "aggregation_weight": float(total_weight),
                    "mean_client_train_loss": self._weighted_local_loss(all_updates),
                }
                for metric_name in sorted(final_metrics.keys()):
                    metric_row[metric_name] = float(final_metrics[metric_name])
                writer.write_metrics(metric_row)
                writer.write_topology(
                    {
                        "round": int(round_index),
                        "source_round_index": int(topology.source_round_index),
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
                "task": self.task.task_name,
                "comm_round": self.comm_round,
                "local_epochs": self.local_epochs,
                "aggregation_mode": self.aggregation_mode,
                "aggregated_rounds": aggregated_rounds,
                "final_metrics": final_metrics,
                "metrics_file": str(writer.metrics_path),
                "topology_file": str(writer.topology_path),
            }
            writer.write_json("summary.json", summary)
        return summary
