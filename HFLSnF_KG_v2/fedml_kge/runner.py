"""普通联邦TransE的FedML单进程运行器。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from fedml.constants import (
    FEDML_SIMULATION_TYPE_SP,
    FEDML_TRAINING_PLATFORM_SIMULATION,
)

from ..core.topology import TopologyProvider
from ..tasks.kge.federated_data import FederatedKnowledgeGraphData
from ..tasks.kge.fixed_topology import (
    FixedParticipantTopology,
    build_fixed_participant_topology,
)
from .model_trainer import FedMLTransEModelTrainer
from .dynamic_trainer import FedMLDynamicTopologyTransETrainer
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


class FedMLFixedTopologyTransERunner:
    """创建四种固定参与集合的FL/HFL TransE对照训练器。"""

    def __init__(
        self,
        args,
        device: torch.device,
        federated_data: FederatedKnowledgeGraphData,
        model: torch.nn.Module,
        client_trainer: FedMLTransEModelTrainer = None,
    ):
        """校验FedML环境，构造固定客户端拓扑并初始化训练器。"""

        training_type = str(getattr(args, "training_type", "")).strip()
        backend = str(getattr(args, "backend", "")).strip()
        optimizer_name = str(
            getattr(args, "federated_optimizer", "")
        ).strip().lower()
        if training_type != FEDML_TRAINING_PLATFORM_SIMULATION:
            raise ValueError("固定拓扑TransE只支持FedML simulation")
        if backend != FEDML_SIMULATION_TYPE_SP:
            raise ValueError("固定拓扑TransE只支持FedML sp后端")
        if optimizer_name != "fixedtopologytranse":
            raise ValueError(
                "federated_optimizer必须是FixedTopologyTransE"
            )
        self.fixed_topology: FixedParticipantTopology = (
            build_fixed_participant_topology(
                args,
                actual_client_count=federated_data.client_count,
            )
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
            fixed_topology=self.fixed_topology,
        )

    def run(self, result_dir: Path) -> Dict[str, object]:
        """启动固定FL/HFL TransE训练并返回汇总。"""

        return self.trainer.train(result_dir)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回已经恢复最佳轮次的全局TransE参数。"""

        return self.trainer.get_global_state()

    def get_fixed_topology_summary(self) -> Dict[str, object]:
        """返回运行开始后整次实验保持不变的参与拓扑。"""

        return self.fixed_topology.summary()


class FedMLDynamicTopologyTransERunner:
    """创建由MAT逐轮驱动客户端采样和边缘分组的TransE训练器。"""

    def __init__(
        self,
        args,
        device: torch.device,
        federated_data: FederatedKnowledgeGraphData,
        model: torch.nn.Module,
        topology_provider: TopologyProvider,
        client_trainer: FedMLTransEModelTrainer = None,
    ):
        """校验FedML环境并连接MAT拓扑提供器和知识客户端。"""

        training_type = str(getattr(args, "training_type", "")).strip()
        backend = str(getattr(args, "backend", "")).strip()
        optimizer_name = str(
            getattr(args, "federated_optimizer", "")
        ).strip().lower()
        if training_type != FEDML_TRAINING_PLATFORM_SIMULATION:
            raise ValueError("动态MAT TransE只支持FedML simulation")
        if backend != FEDML_SIMULATION_TYPE_SP:
            raise ValueError("动态MAT TransE只支持FedML sp后端")
        if optimizer_name != "dynamictopologytranse":
            raise ValueError(
                "federated_optimizer必须是DynamicTopologyTransE"
            )
        if client_trainer is None:
            client_trainer = FedMLTransEModelTrainer(
                model, args, federated_data.dataset
            )
        self.trainer = FedMLDynamicTopologyTransETrainer(
            args=args,
            device=device,
            federated_data=federated_data,
            model_trainer=client_trainer,
            topology_provider=topology_provider,
        )

    def run(self, result_dir: Path) -> Dict[str, object]:
        """启动MAT动态采样与分组联邦TransE训练并返回汇总。"""

        return self.trainer.train(result_dir)

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回已恢复最佳验证轮次的全局TransE参数。"""

        return self.trainer.get_global_state()

    def get_dynamic_participation_summary(
        self,
    ) -> Dict[str, object]:
        """返回训练使用的逐轮动态参与预算和调度指纹。"""

        return dict(self.trainer.participation_summary)
