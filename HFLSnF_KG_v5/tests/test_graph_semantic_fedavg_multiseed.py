"""验证固定5e-4的三种子九实验矩阵、初始化、恢复和跨种子统计。"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_fedavg_multiseed as runner
from HFLSnF_KG_v5 import run_graph_semantic_fedavg_lr_sweep as engine
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_multiseed as contract
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_seed42 as shared_contract
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_lr_sweep as audit_contract
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_comparison as original_contract
from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import load_flat_config
from HFLSnF_KG_v5.tests.test_graph_semantic_fedavg_lr_sweep import metrics_row, write_metrics


def simulated_definition() -> engine.ExperimentDefinition:
    """保留九实验身份，仅替换完整训练合同用于模拟批次调度。"""

    return replace(runner.experiment_definition(),
                   validate_result=mock.Mock(return_value={"status": "passed"}))


def simulated_results(root: Path) -> list:
    """生成具有种子差异和配对差异的九组完整模拟曲线及预算。"""

    directories = []
    for index, item in enumerate(contract.scenarios()):
        seed_index, arm_index = divmod(index, 3)
        score = 0.2 + seed_index * 0.02 + arm_index * (0.03 + seed_index * 0.005)
        directory = root / item.tag
        directory.mkdir()
        (directory / "model_best.pt").write_bytes(b"simulated-checkpoint")
        write_metrics(directory / "metrics.csv",
                      [metrics_row(number, score) for number in range(1, 151)])
        (directory / "summary.json").write_text(json.dumps({
            "participant_count_mean": [34.21, 12.49, 4.74][arm_index],
            "group_count_mean": [5.79, 2.77, 1.0][arm_index],
            "effective_global_passes": [416.06, 151.44, 56.82][arm_index],
            "initial_model_hash": audit_contract.BASELINE_INITIAL_MODEL_HASHES[item.scenario.seed],
        }), encoding="utf-8")
        directories.append(directory)
    return directories


class MultiSeedTopologyTest(unittest.TestCase):
    """检查九实验只改变种子和拓扑，并准确区分各自结果。"""

    def test_nine_configs_have_frozen_recipe_and_matching_seed_partition(self) -> None:
        """矩阵为三个种子乘三个拓扑，各配置只改变来源学习率及身份。"""

        items = contract.scenarios()
        self.assertEqual([(item.scenario.seed, item.scenario.arm) for item in items],
                         [(seed, arm) for seed in (42, 2024, 2025)
                          for arm in ("hflsnf", "hflnosnf", "flnosnf")])
        self.assertEqual(len({item.tag for item in items}), 9)
        self.assertEqual(len({item.scenario.config_path for item in items}), 9)
        excluded = {"learning_rate", "lr", "ablation_suite", "ablation_arm",
                    "comparison_scenario", "run_name", "result_root"}
        common = []
        for item in items:
            actual = load_flat_config(item.scenario.config_path)
            source = load_flat_config(item.scenario.source_config_path)
            self.assertEqual(actual, contract.expected_flat_config(item))
            self.assertEqual(source["learning_rate"], 5e-5)
            self.assertEqual({key: value for key, value in actual.items() if key not in excluded},
                             {key: value for key, value in source.items() if key not in excluded})
            self.assertEqual(actual["random_seed"], item.scenario.seed)
            self.assertEqual(actual["partition_search_seed"], item.scenario.seed)
            self.assertEqual(actual["learning_rate"], actual["lr"])
            self.assertEqual(actual["lr"], 5e-4)
            self.assertTrue(actual["require_cuda"])
            self.assertFalse(actual["evaluate_test_after_training"])
            common.append({key: value for key, value in actual.items()
                           if key not in shared_contract.TOPOLOGY_AND_IDENTITY_FIELDS |
                           {"random_seed", "partition_search_seed", "expected_partition_hash"}})
        self.assertTrue(all(value == common[0] for value in common))
        for seed in (42, 2024, 2025):
            self.assertEqual(len({item.scenario.partition_hash for item in items if item.scenario.seed == seed}), 1)
        self.assertEqual(len({item.scenario.partition_hash for item in items}), 3)

    def test_config_validation_recomputes_each_seed_once(self) -> None:
        """每个种子只复算一次共同分区，不错误复用seed42分区。"""

        partitions = {item.scenario.seed: item.scenario.partition_hash for item in contract.scenarios()}
        with mock.patch.object(shared_contract, "load_fb15k237"), \
                mock.patch.object(shared_contract, "schedule_statistics", side_effect=lambda scenario: {
                    "schedule_hash": scenario.contract.schedule_hash,
                }), mock.patch.object(shared_contract, "partition_train_triples_by_semantic_domain_graph_local",
                                       side_effect=lambda **kwargs: SimpleNamespace(
                                           partition_hash=partitions[kwargs["seed"]])) as partition:
            result = contract.validate_configs(True)
        self.assertEqual(result["status"], "passed")
        self.assertEqual([call[1]["seed"] for call in partition.call_args_list], [42, 2024, 2025])
        self.assertEqual(result["partition_hashes"], {str(seed): value for seed, value in partitions.items()})

    def test_failure_then_resume_skips_completed_seed_and_runs_remaining_five(self) -> None:
        """第五组失败后恢复，仅重跑该组及后续组，保留独立尝试日志。"""

        definition = simulated_definition()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = simulated_results(root)
            manifest = root / "batch_summary.json"
            engine._save_batch_manifest(manifest, engine._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True}, definition))
            with mock.patch.object(engine, "_run_scenario", side_effect=directories[:4] + [
                engine.TrainingRunError("模拟第五组失败"),
            ]) as train:
                self.assertEqual(engine.run_training_batch(manifest, definition), 1)
                self.assertEqual(train.call_count, 5)
            with mock.patch.object(engine, "_run_scenario", side_effect=directories[4:]) as train:
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
                self.assertEqual([call[0][0].tag for call in train.call_args_list],
                                 [item.tag for item in contract.scenarios()[4:]])
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "passed")
            self.assertEqual([len(entry["attempts"]) for entry in saved["entries"]], [1, 1, 1, 1, 2, 1, 1, 1, 1])

    def test_full_report_uses_sample_std_and_same_seed_pairing(self) -> None:
        """九组不得按拓扑名称覆盖，跨种子标准差与配对差必须正确。"""

        definition = simulated_definition()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = simulated_results(root)
            manifest = root / "batch_summary.json"
            engine._save_batch_manifest(manifest, engine._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True}, definition))
            with mock.patch.object(engine, "_run_scenario", side_effect=directories):
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
            result = json.loads((root / "analysis/multiseed_topology_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["units"]), 9)
            self.assertEqual(len(result["paired_differences"]), 9)
            self.assertEqual([pair["seed"] for pair in result["paired_differences"]], [42] * 3 + [2024] * 3 + [2025] * 3)
            aggregate = result["cross_seed_summary"][0]
            self.assertEqual(aggregate["completed_seeds"], 3)
            self.assertAlmostEqual(aggregate["best_validation_mrr_mean"], 0.22)
            self.assertAlmostEqual(aggregate["best_validation_mrr_std"], 0.02)
            contrast = result["cross_seed_paired_summary"][0]
            self.assertAlmostEqual(contrast["best_validation_mrr_mean"], -0.035)
            self.assertAlmostEqual(contrast["best_validation_mrr_std"], 0.005)
            for name in ("units", "paired_differences", "cross_seed_summary", "cross_seed_paired_summary"):
                self.assertTrue((root / ("analysis/multiseed_topology_" + name + ".csv")).is_file())
            report = (root / "analysis/FedAvg三种子三拓扑九实验报告.md").read_text(encoding="utf-8")
            self.assertIn("跨种子统计", report)
            self.assertIn("样本标准差", report)

    def test_unstable_seed_does_not_become_partial_three_seed_average(self) -> None:
        """不稳定组之后仍执行，缺少一种子时不把两种子均值冒充完整统计。"""

        definition = simulated_definition()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = simulated_results(root)
            write_metrics(directories[4] / "metrics.csv", [metrics_row(1), metrics_row(2, float("nan"))])
            manifest = root / "batch_summary.json"
            engine._save_batch_manifest(manifest, engine._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True}, definition))
            outputs = directories[:4] + [engine.NumericalInstability("非有限MRR", [directories[4]])] + directories[5:]
            with mock.patch.object(engine, "_run_scenario", side_effect=outputs) as train:
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
                self.assertEqual(train.call_count, 9)
                train.reset_mock()
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
                train.assert_not_called()
            result = json.loads((root / "analysis/multiseed_topology_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed_with_unstable")
            aggregate = result["cross_seed_summary"][1]
            self.assertEqual(aggregate["completed_seeds"], 2)
            self.assertIsNone(aggregate["best_validation_mrr_mean"])
            self.assertIsNone(result["cross_seed_paired_summary"][0]["last30_mean_mrr_mean"])
            self.assertEqual(result["cross_seed_paired_summary"][1]["completed_seeds"], 3)

    def test_restore_rejects_three_group_batch_and_changed_seed_binding(self) -> None:
        """不能把三组清单扩充为九组恢复，也不能替换某种子的分区绑定。"""

        definition = runner.experiment_definition()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "batch_summary.json"
            for mismatch in ("suite", "count", "partition"):
                payload = engine._empty_batch_payload(
                    {"status": "passed", "recompute_partitions": True}, definition)
                if mismatch == "suite":
                    payload["suite"] = shared_contract.SUITE_NAME
                elif mismatch == "count":
                    payload["entries"] = payload["entries"][:3]
                else:
                    payload["entries"][3]["partition_hash"] = payload["entries"][0]["partition_hash"]
                engine._save_batch_manifest(manifest, payload)
                with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                    engine._load_batch_manifest(manifest, definition)

    def test_cpu_formal9_refuses_before_creating_batch(self) -> None:
        """无CUDA时明确拒绝九组正式训练，禁止CPU回退。"""

        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(contract, "validate_configs") as validate, \
                mock.patch.object(engine, "create_batch_manifest") as create:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runner.main(["formal9"])
        validate.assert_not_called()
        create.assert_not_called()

    def test_formal9_cannot_skip_real_partitions_or_use_invalid_options(self) -> None:
        """正式入口不能跳过三分区复算，参数不得静默错用其他批次命令。"""

        for arguments in (["formal9", "--skip-partition-recompute"], ["formal3"],
                          ["validate", "--resume", "batch.json"], ["report"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as caught:
                runner.main(arguments)
            self.assertEqual(caught.exception.code, 2)

    def test_each_seed_result_requires_its_own_initialization_hash(self) -> None:
        """模拟完整逐轮合同，2024和2025不能错误使用42的初始模型哈希。"""

        for item in contract.scenarios()[::3]:
            config = contract.expected_flat_config(item)
            topology = item.scenario.topology_contract
            summary = dict(config, device="cuda:0", scenario=config["comparison_scenario"],
                           architecture=topology.architecture, snf_enabled=topology.snf_enabled,
                           edge_mode=topology.edge_mode, client_count=37, local_epochs=3,
                           server_learning_rate=1.0, server_beta1=0.0, server_beta2=0.0, server_tau=0.0,
                           server_bias_correction=False, server_optimizer_step_count=0, server_optimizer_state_hash="",
                           initial_model_hash=audit_contract.BASELINE_INITIAL_MODEL_HASHES[item.scenario.seed],
                           partition_hash=item.scenario.partition_hash, topology_schedule_hash=topology.schedule_hash,
                           test_evaluation_performed=False, final_test_metrics=None, best_round=1,
                           best_validation_mrr_during_training=0.2, participant_count_mean=2.0,
                           group_count_mean=1.0, effective_global_passes=100.0)
            rows, audits = [], []
            for number in range(1, 151):
                row = metrics_row(number)
                row.update(active_client_count="2", contributing_client_count="2", active_group_count="1",
                           server_active_row_count="3", server_model_delta_l2="0", server_update_l2="0",
                           server_optimizer="fedavg", server_optimizer_step="0", server_optimizer_state_hash="")
                rows.append(row)
                audits.append({"round": number, "fedavg_candidate_state_hash": "a" * 64,
                               "cloud_parameter_state_hash": "a" * 64, "server_optimizer_state_hash": ""})
            fixtures = {
                "summary.json": summary, "config_snapshot.json": config,
                "client_partition_summary.json": {"partition_hash": item.scenario.partition_hash,
                                                   "max_relative_load_deviation": 0.04},
                "dynamic_participation_summary.json": {"schedule_hash": topology.schedule_hash},
                "topology_metadata.json": {"provider_type": "matlab_adapter", "architecture": topology.architecture,
                                           "snf_enabled": topology.snf_enabled, "edge_mode": topology.edge_mode,
                                           "topology_util": 0.6, "topology_schedule_policy": "strict"},
            }
            provider = mock.Mock()
            provider.get_round.return_value = SimpleNamespace(participant_count=2, group_to_client_indexes={"0": [0, 1]})
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "model_best.pt").write_bytes(b"simulated-checkpoint")
                with mock.patch.object(audit_contract, "_load_json", side_effect=lambda path: fixtures[path.name]), \
                        mock.patch.object(audit_contract, "_load_metrics", return_value=rows), \
                        mock.patch.object(audit_contract, "_load_json_lines", return_value=audits), \
                        mock.patch.object(original_contract, "build_provider", return_value=provider):
                    self.assertEqual(contract.validate_result(directory, item)["status"], "passed")
                    wrong_seed = 2024 if item.scenario.seed == 42 else 42
                    summary["initial_model_hash"] = audit_contract.BASELINE_INITIAL_MODEL_HASHES[wrong_seed]
                    result = contract.validate_result(directory, item)
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("summary.initial_model_hash", [check["name"] for check in result["checks"]
                                                                 if not check["passed"]])


if __name__ == "__main__":
    unittest.main()
