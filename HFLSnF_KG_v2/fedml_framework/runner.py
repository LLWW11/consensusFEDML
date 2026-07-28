"""从HFLSnF FedMLRunner迁移得到的GCN运行器。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from fedml.constants import (
    FEDML_SIMULATION_TYPE_SP,
    FEDML_TRAINING_PLATFORM_SIMULATION,
)
from fedml.core import ClientTrainer, ServerAggregator

from FedGCN_fedml.data import FederatedGraphData

from ..core.topology import TopologyProvider
from .simulator import SimulatorSingleProcess


class FedMLRunner:
    """按照FedML训练类型和通信后端创建单进程分层GCN模拟器。"""

    def __init__(
        self,
        args,
        device: torch.device,
        dataset: FederatedGraphData,
        model: torch.nn.Module,
        topology_provider: TopologyProvider,
        client_trainer: ClientTrainer = None,
        server_aggregator: ServerAggregator = None,
    ):
        """校验FedML模拟平台并初始化单进程运行器。"""

        training_type = str(getattr(args, "training_type", "")).strip()
        backend = str(getattr(args, "backend", "")).strip()
        if training_type != FEDML_TRAINING_PLATFORM_SIMULATION:
            raise ValueError(
                "当前迁移阶段只支持FedML simulation，实际为{}".format(
                    training_type
                )
            )
        if backend != FEDML_SIMULATION_TYPE_SP:
            raise ValueError(
                "当前迁移阶段只支持FedML sp后端，实际为{}".format(
                    backend
                )
            )
        self.runner = SimulatorSingleProcess(
            args=args,
            device=device,
            dataset=dataset,
            model=model,
            topology_provider=topology_provider,
            client_trainer=client_trainer,
            server_aggregator=server_aggregator,
        )

    def run(self, result_dir: Path) -> Dict[str, object]:
        """启动FedML单进程模拟并返回训练汇总。"""

        return self.runner.run(result_dir)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回FedML运行器内部最终全局模型状态。"""

        return self.runner.get_global_state()
