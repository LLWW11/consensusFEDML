"""固定FL/HFL TransE四方案的参与拓扑与训练链路测试。"""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from HFLSnF_KG.core.aggregation import DenseFedAvgAggregator
from HFLSnF_KG.core.randomness import seed_everything
from HFLSnF_KG.core.types import ClientUpdate
from HFLSnF_KG.fedml_kge import FedMLFixedTopologyTransERunner
from HFLSnF_KG.tasks.kge import (
    TransE,
    build_fixed_participant_topology,
    build_knowledge_graph_dataset,
    partition_train_triples_by_head,
)


SCENARIO_SETTINGS = {
    "FLnoSnF": ("fl", False, 5, 1),
    "FLSnF": ("fl", True, 25, 1),
    "HFLnoSnF": ("hfl", False, 15, 6),
    "HFLSnF": ("hfl", True, 35, 6),
}


def build_fixed_args(scenario: str) -> SimpleNamespace:
    """构造一种四方案固定参与拓扑的轻量测试参数。"""

    architecture, snf_enabled, participant_count, group_num = (
        SCENARIO_SETTINGS[scenario]
    )
    return SimpleNamespace(
        training_type="simulation",
        backend="sp",
        federated_optimizer="FixedTopologyTransE",
        comparison_scenario=scenario,
        client_num_in_total=37,
        client_num_per_round=participant_count,
        participant_selection="fixed_once",
        fixed_client_seed=42,
        enforce_comparison_budget=True,
        topology_architecture=architecture,
        topology_snf=snf_enabled,
        topology_edge_mode="fixed",
        group_num=group_num,
        comm_round=1,
        epochs=1,
        batch_size=2,
        client_optimizer="adam",
        learning_rate=0.01,
        lr=0.01,
        margin=1.0,
        negative_sample_count=1,
        eval_every=1,
        validation_max_triples=0,
        final_validation_max_triples=0,
        test_max_triples=0,
        evaluation_candidate_batch_size=128,
        centralized_reference_mrr=0.2,
        random_seed=42,
    )


def build_37_client_dataset():
    """构造每个头实体可独占一个客户端的37客户端微型图谱。"""

    train = [
        (
            "head_{:02d}".format(client_id),
            "linked_to",
            "tail_{:02d}".format(client_id),
        )
        for client_id in range(37)
    ]
    valid = [("head_00", "linked_to", "tail_01")]
    test = [("head_01", "linked_to", "tail_02")]
    return build_knowledge_graph_dataset(
        "synthetic-37-client-kg",
        train,
        valid,
        test,
    )


class FixedParticipantTopologyTest(unittest.TestCase):
    """验证四方案客户端集合固定、预算正确且HFL分为六组。"""

    def test_scenario_budgets_are_reproducible_and_nested(self) -> None:
        """验证共同种子产生固定前缀集合及5、25、15、35人预算。"""

        topologies = {
            scenario: build_fixed_participant_topology(
                build_fixed_args(scenario), actual_client_count=37
            )
            for scenario in SCENARIO_SETTINGS
        }
        self.assertEqual(
            len(topologies["FLnoSnF"].active_client_ids), 5
        )
        self.assertEqual(len(topologies["FLSnF"].active_client_ids), 25)
        self.assertEqual(
            len(topologies["HFLnoSnF"].active_client_ids), 15
        )
        self.assertEqual(len(topologies["HFLSnF"].active_client_ids), 35)
        self.assertTrue(
            set(topologies["FLnoSnF"].active_client_ids).issubset(
                topologies["HFLnoSnF"].active_client_ids
            )
        )
        self.assertTrue(
            set(topologies["HFLnoSnF"].active_client_ids).issubset(
                topologies["FLSnF"].active_client_ids
            )
        )
        self.assertTrue(
            set(topologies["FLSnF"].active_client_ids).issubset(
                topologies["HFLSnF"].active_client_ids
            )
        )
        repeated = build_fixed_participant_topology(
            build_fixed_args("HFLSnF"), actual_client_count=37
        )
        self.assertEqual(repeated, topologies["HFLSnF"])

    def test_hierarchical_groups_are_balanced_and_complete(self) -> None:
        """验证15人与35人的固定参与者都被六组互斥完整覆盖。"""

        for scenario, expected_sizes in (
            ("HFLnoSnF", [3, 3, 3, 2, 2, 2]),
            ("HFLSnF", [6, 6, 6, 6, 6, 5]),
        ):
            topology = build_fixed_participant_topology(
                build_fixed_args(scenario), actual_client_count=37
            )
            actual_sizes = [
                len(group)
                for group in topology.group_client_ids
            ]
            self.assertEqual(actual_sizes, expected_sizes)
            grouped = {
                client_id
                for group in topology.group_client_ids
                for client_id in group
            }
            self.assertEqual(grouped, set(topology.active_client_ids))

    def test_invalid_budget_is_rejected(self) -> None:
        """验证四方案名称与约定参与人数不一致时快速报错。"""

        args = build_fixed_args("FLnoSnF")
        args.client_num_per_round = 6
        with self.assertRaisesRegex(ValueError, "要求client_num_per_round"):
            build_fixed_participant_topology(
                args, actual_client_count=37
            )


class FixedTopologyAggregationTest(unittest.TestCase):
    """验证相同参与集合的两级稠密FedAvg与直接FedAvg等价。"""

    def test_two_level_aggregation_matches_direct_fedavg(self) -> None:
        """验证六组先聚合再上云不会改变三元组数加权结果。"""

        topology = build_fixed_participant_topology(
            build_fixed_args("HFLnoSnF"),
            actual_client_count=37,
        )
        updates = []
        for client_id in topology.active_client_ids:
            updates.append(
                ClientUpdate(
                    client_id=client_id,
                    weight=float(client_id + 1),
                    state_dict={
                        "weight": torch.tensor(
                            [float(client_id), float(client_id + 2)]
                        )
                    },
                )
            )
        update_by_id = {
            update.client_id: update for update in updates
        }
        aggregator = DenseFedAvgAggregator()
        direct = aggregator.aggregate(updates)
        edge_statistics = [
            aggregator.accumulate(
                [update_by_id[client_id] for client_id in group]
            )
            for group in topology.group_client_ids
        ]
        hierarchical = aggregator.finalize(
            aggregator.merge(edge_statistics)
        )
        self.assertTrue(
            torch.allclose(
                direct["weight"],
                hierarchical["weight"],
                atol=1e-6,
            )
        )


class FixedTopologyFedMLRunnerTest(unittest.TestCase):
    """验证FedML固定HFL链路写出客户端、分组和逐轮贡献数据。"""

    def test_each_round_prints_progress_information(self) -> None:
        """验证每个通信epoch都打印损失、耗时及MRR和Hits@3状态。"""

        seed_everything(42)
        dataset = build_37_client_dataset()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=37, seed=42
        )
        args = build_fixed_args("FLnoSnF")
        args.comm_round = 2
        args.eval_every = 2
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
        )
        runner = FedMLFixedTopologyTransERunner(
            args=args,
            device=torch.device("cpu"),
            federated_data=federated_data,
            model=model,
        )
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            with redirect_stdout(output):
                runner.run(Path(temp_dir))
            metrics_text = (
                Path(temp_dir) / "metrics.csv"
            ).read_text(encoding="utf-8-sig")
        printed = output.getvalue()
        self.assertIn("方案=FLnoSnF epoch=1/2", printed)
        self.assertIn("方案=FLnoSnF epoch=2/2", printed)
        self.assertIn("加权损失=", printed)
        self.assertIn("耗时=", printed)
        self.assertIn("验证MRR=未评估", printed)
        self.assertIn("验证Hits@3=", printed)
        self.assertIn("round_seconds", metrics_text)

    def test_all_four_scenarios_complete_one_round(self) -> None:
        """验证四种固定方案都能完成一轮FedML客户端训练和评估。"""

        dataset = build_37_client_dataset()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=37, seed=42
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for scenario in SCENARIO_SETTINGS:
                seed_everything(42)
                args = build_fixed_args(scenario)
                model = TransE(
                    dataset.num_entities,
                    dataset.num_relations,
                    embedding_dim=8,
                )
                runner = FedMLFixedTopologyTransERunner(
                    args=args,
                    device=torch.device("cpu"),
                    federated_data=federated_data,
                    model=model,
                )
                result_dir = root / scenario
                summary = runner.run(result_dir)
                self.assertEqual(summary["scenario"], scenario)
                self.assertEqual(
                    summary["client_num_per_round"],
                    SCENARIO_SETTINGS[scenario][2],
                )
                self.assertTrue(
                    (result_dir / "fixed_participation.json").is_file()
                )
                self.assertTrue(
                    (result_dir / "participation_schedule.jsonl").is_file()
                )
                self.assertTrue((result_dir / "metrics.csv").is_file())
                for value in summary["final_test_metrics"].values():
                    self.assertTrue(math.isfinite(float(value)))

    def test_hfl_runner_writes_fixed_topology_outputs(self) -> None:
        """运行一轮37客户端微型图谱并核对固定六组结果文件。"""

        seed_everything(42)
        dataset = build_37_client_dataset()
        federated_data = partition_train_triples_by_head(
            dataset, client_count=37, seed=42
        )
        args = build_fixed_args("HFLnoSnF")
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
        )
        runner = FedMLFixedTopologyTransERunner(
            args=args,
            device=torch.device("cpu"),
            federated_data=federated_data,
            model=model,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            summary = runner.run(result_dir)
            fixed_path = result_dir / "fixed_participation.json"
            schedule_path = result_dir / "participation_schedule.jsonl"
            self.assertTrue(fixed_path.is_file())
            self.assertTrue(schedule_path.is_file())
            fixed_payload = json.loads(
                fixed_path.read_text(encoding="utf-8")
            )
            schedule_rows = [
                json.loads(line)
                for line in schedule_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        self.assertEqual(summary["runtime"], "fedml_fixed_topology_transe")
        self.assertEqual(summary["scenario"], "HFLnoSnF")
        self.assertEqual(summary["client_num_per_round"], 15)
        self.assertEqual(summary["group_num"], 6)
        self.assertEqual(fixed_payload["selected_train_triple_count"], 15)
        self.assertEqual(len(schedule_rows), 1)
        self.assertEqual(
            schedule_rows[0]["aggregation"],
            "hierarchical_two_level_dense_fedavg",
        )
        self.assertEqual(schedule_rows[0]["edge_group_count"], 6)
        self.assertEqual(
            schedule_rows[0]["active_client_indexes"],
            summary["active_client_ids"],
        )
        for value in summary["final_test_metrics"].values():
            self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
