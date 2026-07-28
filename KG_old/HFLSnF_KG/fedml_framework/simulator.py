"""从HFLSnF SimulatorSingleProcess迁移得到的GCN模拟器路由。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from FedGCN_fedml.data import FederatedGraphData

from ..core.topology import TopologyProvider
from .hierarchical_trainer import FedMLHierarchicalGCNTrainer
from .model_trainer import FedMLGCNModelTrainer


class SimulatorSingleProcess:
    """在FedML单进程后端中选择并运行分层GCN训练器。"""

    def __init__(
        self,
        args,
        device: torch.device,
        dataset: FederatedGraphData,
        model: torch.nn.Module,
        topology_provider: TopologyProvider,
        client_trainer: FedMLGCNModelTrainer = None,
        server_aggregator=None,
    ):
        """根据自定义联邦优化器名称构建分层GCN训练器。"""

        del server_aggregator
        optimizer_name = str(
            getattr(args, "federated_optimizer", "")
        ).strip().lower()
        if optimizer_name not in {
            "hierarchicalgcn",
            "hierarchicalfl",
        }:
            raise ValueError(
                "FedML框架版只支持HierarchicalGCN，实际为{}".format(
                    getattr(args, "federated_optimizer", None)
                )
            )
        if client_trainer is None:
            client_trainer = FedMLGCNModelTrainer(model, args)
        self.fl_trainer = FedMLHierarchicalGCNTrainer(
            args=args,
            device=device,
            dataset=dataset,
            model_trainer=client_trainer,
            topology_provider=topology_provider,
        )

    def run(self, result_dir: Path) -> Dict[str, object]:
        """调用分层GCN训练器并返回最终汇总。"""

        return self.fl_trainer.train(result_dir)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回模拟器内部FedML共享模型的最终状态。"""

        return self.fl_trainer.get_global_state()
