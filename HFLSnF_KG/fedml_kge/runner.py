"""普通联邦TransE的FedML单进程运行器。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from fedml.constants import (
    FEDML_SIMULATION_TYPE_SP,
    FEDML_TRAINING_PLATFORM_SIMULATION,
)

from ..tasks.kge.federated_data import FederatedKnowledgeGraphData
from .model_trainer import FedMLTransEModelTrainer
from .trainer import FedMLFederatedTransETrainer


class FedMLTransERunner:
    """校验FedML simulation/sp环境并创建普通联邦TransE训练器。"""

    def __init__(
        self,
        args,
        device: torch.device,
        federated_data: FederatedKnowledgeGraphData,
        model: torch.nn.Module,
        client_trainer: FedMLTransEModelTrainer = None,
    ):
        """根据FedML配置初始化TransE ClientTrainer和联邦训练器。"""

        training_type = str(getattr(args, "training_type", "")).strip()
        backend = str(getattr(args, "backend", "")).strip()
        optimizer_name = str(
            getattr(args, "federated_optimizer", "")
        ).strip().lower()
        if training_type != FEDML_TRAINING_PLATFORM_SIMULATION:
            raise ValueError(
                "阶段四只支持FedML simulation，实际为{}".format(
                    training_type
                )
            )
        if backend != FEDML_SIMULATION_TYPE_SP:
            raise ValueError(
                "阶段四只支持FedML sp后端，实际为{}".format(backend)
            )
        if optimizer_name != "fedavgtranse":
            raise ValueError(
                "阶段四federated_optimizer必须是FedAvgTransE"
            )
        if client_trainer is None:
            client_trainer = FedMLTransEModelTrainer(
                model, args, federated_data.dataset
            )
        self.trainer = FedMLFederatedTransETrainer(
            args=args,
            device=device,
            federated_data=federated_data,
            model_trainer=client_trainer,
        )

    def run(self, result_dir: Path) -> Dict[str, object]:
        """启动FedML普通联邦TransE训练并返回汇总。"""

        return self.trainer.train(result_dir)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回已恢复最佳轮次的全局TransE参数。"""

        return self.trainer.get_global_state()
