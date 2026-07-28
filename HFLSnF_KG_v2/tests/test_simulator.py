"""任务无关分层模拟器的活跃客户端与空轮测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from HFLSnF_KG_v2.core.aggregation import DenseFedAvgAggregator
from HFLSnF_KG_v2.core.simulator import HierarchicalSimulator
from HFLSnF_KG_v2.core.topology import (
    SequenceTopologyProvider,
    StaticTopologyProvider,
)
from HFLSnF_KG_v2.core.types import ClientUpdate, clone_state_dict
from HFLSnF_KG_v2.tasks.base import FederatedTask


class ScalarFederatedTask(FederatedTask):
    """使用单个标量参数模拟客户端训练的最小联邦任务。"""

    def __init__(self, client_ids: Sequence[int]):
        """初始化合法客户端、全局标量和训练调用记录。"""

        self._client_ids = tuple(int(value) for value in client_ids)
        self._state = {"weight": torch.tensor([0.0])}
        self.trained_clients = []

    @property
    def task_name(self) -> str:
        """返回合成标量任务名称。"""

        return "scalar_test"

    @property
    def client_ids(self) -> Sequence[int]:
        """返回合成任务的全部客户端编号。"""

        return self._client_ids

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回当前全局标量的深拷贝。"""

        return clone_state_dict(self._state)

    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """用聚合状态替换当前全局标量。"""

        self._state = clone_state_dict(state_dict)

    def train_client(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        local_epochs: int,
        round_index: int,
    ) -> Optional[ClientUpdate]:
        """让客户端把自己的编号偏移量加到全局标量上。"""

        del round_index
        self.trained_clients.append(int(client_id))
        value = global_state["weight"] + float(client_id + 1) * int(
            local_epochs
        )
        return ClientUpdate(
            client_id=client_id,
            weight=float(client_id + 1),
            state_dict={"weight": value.clone()},
            local_metrics={"train_loss": float(client_id)},
        )

    def evaluate_global(self) -> Dict[str, float]:
        """把当前标量作为唯一全局评估指标。"""

        return {"value": float(self._state["weight"].item())}

    def partition_summary(self) -> Dict[str, object]:
        """返回合成客户端列表。"""

        return {"clients": list(self._client_ids)}


class HierarchicalSimulatorTest(unittest.TestCase):
    """验证模拟器只训练活跃客户端并安全处理零参与轮。"""

    def test_direct_and_hierarchical_modes_match(self) -> None:
        """验证模拟器两种聚合路径在相同更新下产生相同全局状态。"""

        topology = StaticTopologyProvider({0: [0, 1], 1: [2]})
        hierarchical_task = ScalarFederatedTask([0, 1, 2])
        direct_task = ScalarFederatedTask([0, 1, 2])
        hierarchical = HierarchicalSimulator(
            task=hierarchical_task,
            topology_provider=topology,
            aggregator=DenseFedAvgAggregator(),
            comm_round=1,
            local_epochs=1,
            aggregation_mode="hierarchical",
        )
        direct = HierarchicalSimulator(
            task=direct_task,
            topology_provider=topology,
            aggregator=DenseFedAvgAggregator(),
            comm_round=1,
            local_epochs=1,
            aggregation_mode="direct",
        )

        with tempfile.TemporaryDirectory() as first_dir:
            hierarchical.run(Path(first_dir))
        with tempfile.TemporaryDirectory() as second_dir:
            direct.run(Path(second_dir))
            with (Path(second_dir) / "metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                direct_rows = list(csv.DictReader(handle))

        self.assertTrue(
            torch.allclose(
                hierarchical_task.get_global_state()["weight"],
                direct_task.get_global_state()["weight"],
            )
        )
        self.assertEqual(direct_rows[0]["contributing_edge_count"], "0")

    def test_active_clients_and_empty_round(self) -> None:
        """验证第一轮聚合且第二个空轮沿用已有全局状态。"""

        task = ScalarFederatedTask([0, 1, 2])
        topology = SequenceTopologyProvider([{0: [0], 1: [1]}, {}])
        simulator = HierarchicalSimulator(
            task=task,
            topology_provider=topology,
            aggregator=DenseFedAvgAggregator(),
            comm_round=2,
            local_epochs=1,
            aggregation_mode="hierarchical",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            summary = simulator.run(result_dir)
            expected = (1.0 * 1.0 + 2.0 * 2.0) / 3.0
            self.assertAlmostEqual(
                float(task.get_global_state()["weight"].item()), expected
            )
            self.assertEqual(task.trained_clients, [0, 1])
            self.assertEqual(summary["aggregated_rounds"], 1)

            with (result_dir / "metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["aggregated"], "0")

            with (result_dir / "topology_schedule.jsonl").open(
                "r", encoding="utf-8"
            ) as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(records[1]["active_client_indexes"], [])
            self.assertFalse(records[1]["aggregated"])


if __name__ == "__main__":
    unittest.main()
