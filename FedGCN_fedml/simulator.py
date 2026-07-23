"""适配图节点分类任务的单进程FedGCN模拟器。"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import torch
from torch.nn import functional as F

from .data import FederatedGraphData, LocalGraphPartition


@dataclass(frozen=True)
class ClientUpdate:
    """保存一台设备完成本地训练后的参数和聚合权重。"""

    client_id: int
    weight: int
    state_dict: Mapping[str, torch.Tensor]
    final_loss: float


def aggregate_state_dicts(
    weighted_states: Iterable[Tuple[float, Mapping[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """按给定正权重对多个模型状态执行FedAvg。"""

    weighted_states = list(weighted_states)
    if not weighted_states:
        raise ValueError("FedAvg至少需要一个客户端模型")
    total_weight = float(sum(float(weight) for weight, _ in weighted_states))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("FedAvg权重之和必须是有限正数")

    reference_keys = tuple(weighted_states[0][1].keys())
    for _, state_dict in weighted_states:
        if tuple(state_dict.keys()) != reference_keys:
            raise ValueError("参与FedAvg的模型参数键不一致")

    averaged_state: Dict[str, torch.Tensor] = {}
    for key in reference_keys:
        reference_value = weighted_states[0][1][key]
        if torch.is_floating_point(reference_value):
            accumulator = torch.zeros_like(reference_value)
            for weight, state_dict in weighted_states:
                accumulator.add_(state_dict[key], alpha=float(weight) / total_weight)
            averaged_state[key] = accumulator
        else:
            # GCN没有整数缓冲区；保留此分支可防止未来扩展模型时错误平均计数器。
            averaged_state[key] = reference_value.clone()
    return averaged_state


def classification_accuracy(
    output: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor
) -> float:
    """计算指定节点集合上的分类准确率。"""

    if indices.numel() == 0:
        raise ValueError("评估节点索引不能为空")
    predictions = output.index_select(0, indices).argmax(dim=1)
    expected = labels.index_select(0, indices)
    return float(predictions.eq(expected).float().mean().item())


class FedGCNSimulator:
    """在单进程中顺序模拟多设备本地GCN训练和参数聚合。"""

    def __init__(
        self,
        args,
        device: torch.device,
        dataset: FederatedGraphData,
        model: torch.nn.Module,
    ):
        """保存配置、完整图、全局模型和实际运行设备。"""

        self.args = args
        self.device = torch.device(device)
        self.dataset = dataset
        self.model = model.to(self.device)
        self.comm_round = int(getattr(args, "comm_round"))
        self.local_epochs = int(getattr(args, "epochs"))
        self.learning_rate = float(
            getattr(args, "learning_rate", getattr(args, "lr", 0.5))
        )
        self.weight_decay = float(getattr(args, "weight_decay", 5e-4))
        self.aggregation_weight_basis = str(
            getattr(args, "aggregation_weight_basis", "labeled_train_nodes")
        ).strip().lower()
        if self.comm_round <= 0:
            raise ValueError("comm_round 必须大于 0")
        if self.local_epochs <= 0:
            raise ValueError("epochs 必须大于 0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须大于 0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay 不能小于 0")
        if self.aggregation_weight_basis != "labeled_train_nodes":
            raise ValueError(
                "当前核心复现仅支持 aggregation_weight_basis=labeled_train_nodes"
            )
        expected_clients = int(getattr(args, "client_num_in_total"))
        if expected_clients != len(dataset.partitions):
            raise ValueError(
                "配置客户端数 {} 与数据分区数 {} 不一致".format(
                    expected_clients, len(dataset.partitions)
                )
            )
        clients_per_round = int(getattr(args, "client_num_per_round", expected_clients))
        if clients_per_round != expected_clients:
            raise ValueError("FedGCN核心复现要求每轮全部设备参与")

    def _copy_partition_to_device(
        self, partition: LocalGraphPartition
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """将一台设备的局部子图张量统一迁移到当前torch.device。"""

        return (
            partition.features.to(self.device),
            partition.adjacency.to(self.device),
            partition.labels.to(self.device),
            partition.train_indices.to(self.device),
        )

    def _train_one_client(
        self,
        partition: LocalGraphPartition,
        global_state: Mapping[str, torch.Tensor],
    ) -> ClientUpdate:
        """从全局参数开始，在一台设备的局部诱导子图上完成本地训练。"""

        if partition.train_node_count <= 0:
            raise ValueError("没有标注训练节点的设备不应进入本地训练函数")
        local_model = copy.deepcopy(self.model)
        local_model.load_state_dict(global_state)
        local_model.to(self.device)
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        features, adjacency, labels, train_indices = self._copy_partition_to_device(
            partition
        )

        final_loss = float("nan")
        for _ in range(self.local_epochs):
            local_model.train()
            optimizer.zero_grad()
            output = local_model(features, adjacency)
            loss = F.nll_loss(
                output.index_select(0, train_indices),
                labels.index_select(0, train_indices),
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

        local_state = {
            key: value.detach().clone()
            for key, value in local_model.state_dict().items()
        }
        return ClientUpdate(
            client_id=partition.client_id,
            weight=partition.train_node_count,
            state_dict=local_state,
            final_loss=final_loss,
        )

    def _evaluate_split(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[float, float]:
        """计算一个完整图数据划分上的NLL损失和准确率。"""

        loss = F.nll_loss(
            output.index_select(0, indices), labels.index_select(0, indices)
        )
        accuracy = classification_accuracy(output, labels, indices)
        return float(loss.detach().cpu().item()), accuracy

    def evaluate_global_model(self) -> Dict[str, float]:
        """在完整图的训练、验证和测试索引上评估当前全局模型。"""

        self.model.eval()
        features = self.dataset.features.to(self.device)
        adjacency = self.dataset.adjacency.to(self.device)
        labels = self.dataset.labels.to(self.device)
        idx_train = self.dataset.idx_train.to(self.device)
        idx_val = self.dataset.idx_val.to(self.device)
        idx_test = self.dataset.idx_test.to(self.device)
        with torch.no_grad():
            output = self.model(features, adjacency)
            train_loss, train_accuracy = self._evaluate_split(
                output, labels, idx_train
            )
            val_loss, val_accuracy = self._evaluate_split(output, labels, idx_val)
            test_loss, test_accuracy = self._evaluate_split(
                output, labels, idx_test
            )
        return {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }

    def _write_metadata(self, result_dir: Path) -> None:
        """保存可复核的运行配置和客户端数据划分摘要。"""

        config_snapshot = {
            key: value for key, value in sorted(vars(self.args).items())
        }
        with (result_dir / "config_snapshot.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                config_snapshot,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        partition_summary = {
            "dataset": self.dataset.dataset_name,
            "num_nodes": self.dataset.num_nodes,
            "num_features": self.dataset.num_features,
            "num_classes": self.dataset.num_classes,
            "aggregation_weight_basis": self.aggregation_weight_basis,
            "clients": [
                {
                    "client_id": partition.client_id,
                    "node_count": partition.node_count,
                    "labeled_train_node_count": partition.train_node_count,
                    "node_indices": partition.node_indices.tolist(),
                }
                for partition in self.dataset.partitions
            ],
        }
        with (result_dir / "partition_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(partition_summary, handle, ensure_ascii=False, indent=2)

    def train(self, result_dir: Path) -> Dict[str, object]:
        """执行全部通信轮次并保存指标、最终模型和汇总信息。"""

        result_dir = Path(result_dir).resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        self._write_metadata(result_dir)
        metrics_path = result_dir / "metrics.csv"
        metric_rows: List[Dict[str, object]] = []

        with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = [
                "round",
                "contributor_count",
                "aggregation_weight_total",
                "mean_local_loss",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for round_index in range(self.comm_round):
                global_state = {
                    key: value.detach().clone()
                    for key, value in self.model.state_dict().items()
                }
                client_updates = []
                for partition in self.dataset.partitions:
                    if partition.train_node_count == 0:
                        # 原notebook也跳过无本地标注节点的设备。
                        continue
                    client_updates.append(
                        self._train_one_client(partition, global_state)
                    )

                if not client_updates:
                    raise RuntimeError("本轮没有任何拥有标注训练节点的设备")
                aggregated_state = aggregate_state_dicts(
                    (update.weight, update.state_dict)
                    for update in client_updates
                )
                self.model.load_state_dict(aggregated_state)
                global_metrics = self.evaluate_global_model()
                metric_row = {
                    "round": round_index,
                    "contributor_count": len(client_updates),
                    "aggregation_weight_total": sum(
                        update.weight for update in client_updates
                    ),
                    "mean_local_loss": sum(
                        update.final_loss for update in client_updates
                    )
                    / len(client_updates),
                    "train_loss": global_metrics["train_loss"],
                    "train_accuracy": global_metrics["train_accuracy"],
                    "val_loss": global_metrics["val_loss"],
                    "val_accuracy": global_metrics["val_accuracy"],
                }
                writer.writerow(metric_row)
                handle.flush()
                metric_rows.append(metric_row)

        final_metrics = self.evaluate_global_model()
        summary: Dict[str, object] = {
            "dataset": self.dataset.dataset_name,
            "device": str(self.device),
            "rounds": self.comm_round,
            "local_epochs": self.local_epochs,
            "client_count": len(self.dataset.partitions),
            "final_train_loss": final_metrics["train_loss"],
            "final_train_accuracy": final_metrics["train_accuracy"],
            "final_val_loss": final_metrics["val_loss"],
            "final_val_accuracy": final_metrics["val_accuracy"],
            "final_test_loss": final_metrics["test_loss"],
            "final_test_accuracy": final_metrics["test_accuracy"],
            "metrics_file": str(metrics_path),
        }
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "summary": summary,
            },
            result_dir / "final_model.pt",
        )
        with (result_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        return summary

