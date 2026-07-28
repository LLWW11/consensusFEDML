"""FedML Runner、ClientTrainer和Client完整链路测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from fedml.core import ClientTrainer
from fedml.simulation.sp.fedavg.client import Client as FedMLClient

from FedGCN_fedml.model import GCN

from HFLSnF_KG_v2.core.aggregation import DenseFedAvgAggregator
from HFLSnF_KG_v2.core.randomness import seed_everything
from HFLSnF_KG_v2.core.simulator import HierarchicalSimulator
from HFLSnF_KG_v2.core.topology import StaticTopologyProvider
from HFLSnF_KG_v2.fedml_framework import FedMLRunner
from HFLSnF_KG_v2.tasks.gcn import CoraGCNTask
from test_gcn_adapter import build_synthetic_graph


def build_framework_args() -> SimpleNamespace:
    """构造FedML框架轻量测试所需的最小参数对象。"""

    return SimpleNamespace(
        training_type="simulation",
        backend="sp",
        federated_optimizer="HierarchicalGCN",
        aggregation_mode="hierarchical",
        comm_round=2,
        epochs=1,
        learning_rate=0.1,
        lr=0.1,
        weight_decay=0.0,
    )


class FedMLFrameworkTest(unittest.TestCase):
    """验证新默认路径确实使用FedML客户端与训练器接口。"""

    def test_runner_uses_fedml_client_and_client_trainer(self) -> None:
        """验证Runner链路内部对象继承FedML官方基类并能完成训练。"""

        dataset = build_synthetic_graph()
        seed_everything(11)
        model = GCN(
            input_dim=dataset.num_features,
            hidden_dim=4,
            output_dim=dataset.num_classes,
            dropout=0.0,
        )
        runner = FedMLRunner(
            args=build_framework_args(),
            device=torch.device("cpu"),
            dataset=dataset,
            model=model,
            topology_provider=StaticTopologyProvider({0: [0], 1: [1]}),
        )
        trainer = runner.runner.fl_trainer
        self.assertIsInstance(trainer.model_trainer, ClientTrainer)
        self.assertTrue(
            all(
                isinstance(client, FedMLClient)
                for client in trainer.client_registry.values()
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = runner.run(Path(temp_dir))
            self.assertEqual(summary["runtime"], "fedml_framework")
            self.assertTrue((Path(temp_dir) / "metrics.csv").is_file())
            self.assertTrue(
                (Path(temp_dir) / "topology_schedule.jsonl").is_file()
            )

    def test_fedml_framework_matches_lightweight_reference(self) -> None:
        """验证FedML链路与轻量参考核心在相同更新下得到相同模型。"""

        dataset = build_synthetic_graph()
        topology = StaticTopologyProvider({0: [0], 1: [1]})

        seed_everything(23)
        framework_model = GCN(
            input_dim=dataset.num_features,
            hidden_dim=4,
            output_dim=dataset.num_classes,
            dropout=0.0,
        )
        framework_runner = FedMLRunner(
            args=build_framework_args(),
            device=torch.device("cpu"),
            dataset=dataset,
            model=framework_model,
            topology_provider=topology,
        )

        seed_everything(23)
        reference_task = CoraGCNTask(
            dataset=dataset,
            device=torch.device("cpu"),
            hidden_dim=4,
            dropout=0.0,
            learning_rate=0.1,
            weight_decay=0.0,
            seed=23,
        )
        reference_simulator = HierarchicalSimulator(
            task=reference_task,
            topology_provider=topology,
            aggregator=DenseFedAvgAggregator(),
            comm_round=2,
            local_epochs=1,
            aggregation_mode="hierarchical",
        )

        with tempfile.TemporaryDirectory() as framework_dir:
            framework_runner.run(Path(framework_dir))
        with tempfile.TemporaryDirectory() as reference_dir:
            reference_simulator.run(Path(reference_dir))

        framework_state = framework_runner.get_global_state()
        reference_state = reference_task.get_global_state()
        self.assertEqual(set(framework_state.keys()), set(reference_state.keys()))
        for name in framework_state.keys():
            self.assertTrue(
                torch.allclose(
                    framework_state[name],
                    reference_state[name],
                    atol=1e-7,
                    rtol=1e-6,
                ),
                msg="参数{}不一致".format(name),
            )


if __name__ == "__main__":
    unittest.main()
