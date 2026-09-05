"""验证V5图语义FedAvg三拓扑九实验合同。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_fedavg_comparison as runner
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_comparison as contract
from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import (
    load_flat_config,
)


class GraphSemanticFedAvgComparisonTest(unittest.TestCase):
    """检查九场景配对、恢复门禁和FedAvg审计行为。"""

    def test_nine_scenarios_follow_seed_first_order(self) -> None:
        """九场景必须按种子优先且每种子包含三个拓扑臂。"""

        scenarios = contract.scenarios_from_contract()
        self.assertEqual(len(scenarios), 9)
        self.assertEqual(
            [(item.seed, item.arm) for item in scenarios],
            [
                (seed, arm)
                for seed in contract.SEEDS
                for arm in contract.ARM_ORDER
            ],
        )

    def test_configs_are_exact_fedavg_derivatives(self) -> None:
        """九份YAML只允许服务器优化器和审计身份发生变化。"""

        partition_hashes = {}
        for scenario in contract.scenarios_from_contract():
            actual = load_flat_config(scenario.config_path)
            self.assertEqual(actual, contract.expected_flat_config(scenario))
            self.assertEqual(actual["server_optimizer"], "fedavg")
            self.assertEqual(actual["client_optimizer"], "adam")
            self.assertEqual(actual["aggregation_mode"], "row_count_weighted")
            self.assertTrue(actual["require_cuda"])
            for field in contract.FEDADAM_ONLY_FIELDS:
                self.assertNotIn(field, actual)
            partition_hashes.setdefault(scenario.seed, set()).add(
                actual["expected_partition_hash"]
            )
        self.assertTrue(
            all(len(values) == 1 for values in partition_hashes.values())
        )

    def test_three_topology_contracts_remain_distinct(self) -> None:
        """HFLSnF、HFLnoSnF和FLnoSnF必须保留冻结拓扑差异。"""

        configs = {
            scenario.arm: contract.expected_flat_config(scenario)
            for scenario in contract.scenarios_from_contract()
            if scenario.seed == 42
        }
        self.assertEqual(configs["hflsnf"]["topology_architecture"], "hfl")
        self.assertTrue(configs["hflsnf"]["topology_snf"])
        self.assertEqual(configs["hflnosnf"]["topology_architecture"], "hfl")
        self.assertFalse(configs["hflnosnf"]["topology_snf"])
        self.assertEqual(configs["flnosnf"]["topology_architecture"], "fl")
        self.assertFalse(configs["flnosnf"]["topology_snf"])
        self.assertEqual(configs["flnosnf"]["topology_edge_mode"], "none")

    def test_seed42_gate_requires_all_three_arms(self) -> None:
        """三个seed42场景未全部通过时不得开放后续种子。"""

        payload = runner._empty_batch_payload()
        for entry in payload["entries"]:
            if entry["seed"] == 42:
                entry["status"] = "passed"
        payload["entries"][2]["status"] = "failed"
        runner._update_seed42_gate(payload)
        self.assertEqual(payload["pilot_seed42_gate"], "pending")
        payload["entries"][2]["status"] = "passed"
        runner._update_seed42_gate(payload)
        self.assertEqual(payload["pilot_seed42_gate"], "passed")

    def test_first_failure_stops_remaining_runs(self) -> None:
        """首个训练失败后必须保留恢复状态并停止其余八项。"""

        payload = runner._empty_batch_payload()
        failure = runner.FedAvgComparisonRunError("模拟训练失败")
        with mock.patch.object(
            runner, "_load_batch_manifest", return_value=payload
        ), mock.patch.object(runner, "_save_batch_manifest"), mock.patch.object(
            runner, "_run_scenario", side_effect=failure
        ):
            status = runner.run_training_batch(Path("unused.json"))
        self.assertEqual(status, 1)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["entries"][0]["status"], "failed")
        self.assertTrue(
            all(
                item["status"] == "pending"
                for item in payload["entries"][1:]
            )
        )

    def test_duplicate_new_result_directories_are_rejected(self) -> None:
        """一次训练若产生两个新目录必须拒绝绑定任意一个。"""

        scenario = contract.scenarios_from_contract()[0]
        first = Path("first")
        second = Path("second")
        with mock.patch.object(
            runner,
            "_result_directories",
            side_effect=[tuple(), (first, second)],
        ), mock.patch.object(runner.subprocess, "run"):
            with self.assertRaises(runner.FedAvgComparisonRunError):
                runner._run_scenario(scenario)

    def test_round_audit_accepts_direct_fedavg(self) -> None:
        """合法审计必须证明候选状态由云端直接采用。"""

        scenario = contract.scenarios_from_contract()[0]
        valid_hash = "a" * 64
        rows = [
            {
                "round": index,
                "active_client_count": 2,
                "contributing_client_count": 2,
                "active_group_count": 1,
                "mean_client_train_loss": 1.0,
                "server_active_row_count": 3,
                "server_model_delta_l2": 0.0,
                "server_update_l2": 0.0,
                "val_mrr": 0.1,
                "round_seconds": 1.0,
                "server_optimizer": "fedavg",
                "server_optimizer_step": 0,
                "server_optimizer_state_hash": "",
            }
            for index in range(1, contract.ROUND_COUNT + 1)
        ]
        audits = [
            {
                "round": index,
                "fedavg_candidate_state_hash": valid_hash,
                "cloud_parameter_state_hash": valid_hash,
                "server_optimizer_state_hash": "",
            }
            for index in range(1, contract.ROUND_COUNT + 1)
        ]
        topology = SimpleNamespace(
            participant_count=2,
            group_to_client_indexes={"0": (0, 1)},
        )
        provider = mock.Mock()
        provider.get_round.return_value = topology
        checks = []
        with mock.patch.object(
            contract, "build_provider", return_value=provider
        ):
            contract._validate_round_metrics(
                checks, rows, audits, scenario
            )
        self.assertTrue(checks)
        self.assertTrue(all(item["passed"] for item in checks))

    def test_round_audit_rejects_fedadam_residue(self) -> None:
        """非零步数、动量哈希或候选状态不等必须导致合同失败。"""

        scenario = contract.scenarios_from_contract()[0]
        rows = [
            {
                "round": index,
                "active_client_count": 2,
                "contributing_client_count": 2,
                "active_group_count": 1,
                "mean_client_train_loss": 1.0,
                "server_active_row_count": 3,
                "server_model_delta_l2": 0.0,
                "server_update_l2": 0.0,
                "val_mrr": 0.1,
                "round_seconds": 1.0,
                "server_optimizer": "fedavg",
                "server_optimizer_step": 0,
                "server_optimizer_state_hash": "",
            }
            for index in range(1, contract.ROUND_COUNT + 1)
        ]
        audits = [
            {
                "round": index,
                "fedavg_candidate_state_hash": "a" * 64,
                "cloud_parameter_state_hash": "a" * 64,
                "server_optimizer_state_hash": "",
            }
            for index in range(1, contract.ROUND_COUNT + 1)
        ]
        rows[0]["server_optimizer_step"] = 1
        rows[0]["server_optimizer_state_hash"] = "b" * 64
        audits[0]["cloud_parameter_state_hash"] = "c" * 64
        topology = SimpleNamespace(
            participant_count=2,
            group_to_client_indexes={"0": (0, 1)},
        )
        provider = mock.Mock()
        provider.get_round.return_value = topology
        checks = []
        with mock.patch.object(
            contract, "build_provider", return_value=provider
        ):
            contract._validate_round_metrics(
                checks, rows, audits, scenario
            )
        failed_names = {
            item["name"] for item in checks if not item["passed"]
        }
        self.assertIn("逐轮服务器步数为零", failed_names)
        self.assertIn("逐轮服务器状态为空", failed_names)
        self.assertIn("FedAvg候选状态由云端直接采用", failed_names)

    def test_result_contract_rejects_wrong_partition_and_topology(self) -> None:
        """错误分区哈希或拓扑哈希必须使正式结果合同失败。"""

        scenario = contract.scenarios_from_contract()[0]
        summary = {
            "device": "cuda:0",
            "ablation_suite": contract.SUITE_NAME,
            "ablation_arm": scenario.scenario_id,
            "architecture": "hfl",
            "snf_enabled": True,
            "edge_mode": "fixed",
            "client_count": 37,
            "client_num_in_total": 37,
            "client_num_per_round": 35,
            "comm_round": 150,
            "local_epochs": 3,
            "aggregation_mode": "row_count_weighted",
            "server_optimizer": "fedavg",
            "server_learning_rate": 1.0,
            "server_beta1": 0.0,
            "server_beta2": 0.0,
            "server_tau": 0.0,
            "server_bias_correction": False,
            "server_optimizer_step_count": 0,
            "server_optimizer_state_hash": "",
            "client_optimizer_state_mode": "reset",
            "topology_schedule_hash": "wrong-topology",
            "partition_hash": "wrong-partition",
            "initial_model_hash": contract.BASELINE_INITIAL_MODEL_HASHES[42],
            "test_evaluation_performed": False,
            "final_test_metrics": None,
        }
        snapshot = contract.expected_flat_config(scenario)
        metadata = {
            "provider_type": "matlab_adapter",
            "architecture": "hfl",
            "snf_enabled": True,
            "edge_mode": "fixed",
            "topology_util": 0.6,
            "topology_schedule_policy": "strict",
        }
        participation = {"schedule_hash": "wrong-topology"}
        partition = {
            "partition_hash": "wrong-partition",
            "max_relative_load_deviation": 0.05,
        }
        payloads = iter(
            [summary, snapshot, metadata, participation, partition]
        )
        with mock.patch.object(
            contract, "_load_json", side_effect=lambda path: next(payloads)
        ), mock.patch.object(
            contract, "_load_metrics", return_value=[]
        ), mock.patch.object(
            contract, "_load_json_lines", return_value=[]
        ), mock.patch.object(
            contract,
            "schedule_statistics",
            return_value={"participant_count_max": 35},
        ), mock.patch.object(contract, "_validate_round_metrics"):
            report = contract.validate_result(Path("unused"), scenario)
        self.assertEqual(report["status"], "failed")
        failed_names = {
            item["name"]
            for item in report["checks"]
            if not item["passed"]
        }
        self.assertIn("topology_schedule_hash", failed_names)
        self.assertIn("训练汇总分区哈希", failed_names)
        self.assertIn("分区摘要哈希", failed_names)

    def test_official_summary_must_bind_best_checkpoint(self) -> None:
        """官方测试必须绑定训练阶段按验证MRR选出的最佳检查点。"""

        root = (Path(runner.PACKAGE_DIR) / runner.RESULT_ROOT).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(root)) as temp:
            result_dir = Path(temp) / "result"
            output_dir = Path(temp) / "official"
            result_dir.mkdir()
            output_dir.mkdir()
            checkpoint_path = result_dir / "model_best.pt"
            checkpoint_path.write_bytes(b"best-checkpoint")
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_round": 17,
                        "best_validation_mrr_during_training": 0.25,
                    }
                ),
                encoding="utf-8",
            )
            summary_path = output_dir / "official_evaluation_summary.json"
            official = {
                "status": "passed",
                "full_official_test": True,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": runner._file_sha256(checkpoint_path),
                "best_round": 17,
                "best_validation_mrr": 0.25,
                "head_metrics": {
                    "mrr": 0.2,
                    "mean_rank": 10.0,
                    "hits_at_1": 0.1,
                    "hits_at_3": 0.2,
                    "hits_at_10": 0.3,
                },
                "tail_metrics": {
                    "mrr": 0.3,
                    "mean_rank": 9.0,
                    "hits_at_1": 0.2,
                    "hits_at_3": 0.3,
                    "hits_at_10": 0.4,
                },
                "combined_metrics": {
                    "mrr": 0.25,
                    "mean_rank": 9.5,
                    "hits_at_1": 0.15,
                    "hits_at_3": 0.25,
                    "hits_at_10": 0.35,
                },
            }
            summary_path.write_text(json.dumps(official), encoding="utf-8")
            unit = {"result_dir": str(result_dir)}
            runner._validate_official_summary(unit, summary_path)
            official["checkpoint_path"] = str(result_dir / "model_last.pt")
            summary_path.write_text(json.dumps(official), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner._validate_official_summary(unit, summary_path)



    def test_official_summary_rejects_non_finite_metrics(self) -> None:
        """官方测试合同必须拒绝NaN和无穷大指标。"""

        root = (Path(runner.PACKAGE_DIR) / runner.RESULT_ROOT).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(root)) as temp:
            result_dir = Path(temp) / "result"
            output_dir = Path(temp) / "official"
            result_dir.mkdir()
            output_dir.mkdir()
            checkpoint_path = result_dir / "model_best.pt"
            checkpoint_path.write_bytes(b"best-checkpoint")
            (result_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_round": 17,
                        "best_validation_mrr_during_training": 0.25,
                    }
                ),
                encoding="utf-8",
            )
            metric_block = {
                "mrr": 0.25,
                "mean_rank": 9.5,
                "hits_at_1": 0.15,
                "hits_at_3": 0.25,
                "hits_at_10": 0.35,
            }
            official = {
                "status": "passed",
                "full_official_test": True,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": runner._file_sha256(checkpoint_path),
                "best_round": 17,
                "best_validation_mrr": 0.25,
                "head_metrics": dict(metric_block),
                "tail_metrics": dict(metric_block),
                "combined_metrics": dict(metric_block),
            }
            official["combined_metrics"]["mrr"] = float("nan")
            summary_path = output_dir / "official_evaluation_summary.json"
            summary_path.write_text(json.dumps(official), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner._validate_official_summary(
                    {"result_dir": str(result_dir)}, summary_path
                )
if __name__ == "__main__":
    unittest.main()
