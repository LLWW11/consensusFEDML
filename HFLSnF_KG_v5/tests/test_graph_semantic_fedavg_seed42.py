"""验证固定学习率、种子42三拓扑的隔离、监测、恢复和结果合同。"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_fedavg_seed42 as runner
from HFLSnF_KG_v5 import run_graph_semantic_fedavg_lr_sweep as engine
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_seed42 as contract
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_lr_sweep as audit_contract
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_comparison as original_contract
from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import load_flat_config
from HFLSnF_KG_v5.tests.test_graph_semantic_fedavg_lr_sweep import metrics_row, write_metrics


def simulated_definition() -> engine.ExperimentDefinition:
    """保留真实实验绑定，仅替换昂贵的150轮结果审计用于模拟批次控制。"""

    return replace(runner.experiment_definition(),
                   validate_result=mock.Mock(return_value={"status": "passed"}))


def simulated_results(root: Path) -> list:
    """为三个拓扑生成含检查点、曲线和预算汇总的模拟结果目录。"""

    directories = []
    # 特意让HFLSnF数值较低，证明结果有效性不依赖预期排序。
    for index, item in enumerate(contract.scenarios()):
        directory = root / item.tag
        directory.mkdir()
        (directory / "model_best.pt").write_bytes(b"simulated-checkpoint")
        write_metrics(directory / "metrics.csv",
                      [metrics_row(number, 0.2 + index * 0.05) for number in range(1, 151)])
        (directory / "summary.json").write_text(json.dumps({
            "participant_count_mean": [34.21, 12.49, 4.74][index],
            "group_count_mean": [5.79, 2.77, 1.0][index],
            "effective_global_passes": [416.06, 151.44, 56.82][index],
        }), encoding="utf-8")
        directories.append(directory)
    return directories


class Seed42TopologyTest(unittest.TestCase):
    """检查三拓扑限定范围、共享引擎接线和异常恢复语义。"""

    def test_three_configs_share_recipe_except_topology_and_identity(self) -> None:
        """三组只使用seed42和5e-4，同源配置不被修改且新批次不引用旧结果。"""

        items = contract.scenarios()
        self.assertEqual([item.tag for item in items], ["hflsnf", "hflnosnf", "flnosnf"])
        self.assertEqual([item.scenario.seed for item in items], [42] * 3)
        common = []
        allowed = {"learning_rate", "lr", "ablation_suite", "ablation_arm",
                   "comparison_scenario", "run_name", "result_root"}
        for item in items:
            actual = load_flat_config(item.scenario.config_path)
            source = load_flat_config(item.scenario.source_config_path)
            self.assertEqual(actual, contract.expected_flat_config(item))
            self.assertEqual(source["learning_rate"], 5e-5)
            self.assertEqual({key: value for key, value in actual.items() if key not in allowed},
                             {key: value for key, value in source.items() if key not in allowed})
            self.assertEqual(actual["learning_rate"], 5e-4)
            self.assertEqual(actual["lr"], 5e-4)
            self.assertTrue(actual["require_cuda"])
            self.assertFalse(actual["evaluate_test_after_training"])
            common.append({key: value for key, value in actual.items()
                           if key not in contract.TOPOLOGY_AND_IDENTITY_FIELDS})
        self.assertEqual(common[0], common[1])
        self.assertEqual(common[1], common[2])
        self.assertEqual(len({item.scenario.topology_contract.schedule_hash for item in items}), 3)
        payload = engine._empty_batch_payload({"status": "passed", "recompute_partitions": True},
                                              runner.experiment_definition())
        self.assertEqual(len(payload["entries"]), 3)
        self.assertTrue(all(entry["status"] == "pending" and entry["result_dir"] is None
                            for entry in payload["entries"]))

    def test_each_schedule_is_checked_against_its_own_contract(self) -> None:
        """不能要求不同拓扑调度相同，也不能接受与本臂合同不符的调度。"""

        with mock.patch.object(contract, "schedule_statistics", side_effect=lambda scenario: {
            "schedule_hash": scenario.contract.schedule_hash,
        }):
            self.assertEqual(contract.validate_configs(False)["status"], "passed")
        with mock.patch.object(contract, "schedule_statistics", return_value={"schedule_hash": "0" * 64}):
            result = contract.validate_configs(False)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len([check for check in result["checks"]
                                 if "真实150轮调度" in check["name"] and not check["passed"]]), 3)

    def test_failure_then_resume_only_restarts_unfinished_topologies(self) -> None:
        """第二组失败停止批次，恢复跳过第一组并独立重跑剩余两组。"""

        definition = simulated_definition()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = simulated_results(root)
            manifest = root / "batch_summary.json"
            engine._save_batch_manifest(manifest, engine._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True}, definition))
            with mock.patch.object(engine, "_run_scenario", side_effect=[
                directories[0], engine.TrainingRunError("模拟子进程异常"),
            ]) as train:
                self.assertEqual(engine.run_training_batch(manifest, definition), 1)
                self.assertEqual([call[0][0].tag for call in train.call_args_list],
                                 ["hflsnf", "hflnosnf"])
            with mock.patch.object(engine, "_run_scenario", side_effect=directories[1:]) as train:
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
                self.assertEqual([call[0][0].tag for call in train.call_args_list],
                                 ["hflnosnf", "flnosnf"])
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "passed")
            self.assertEqual([len(entry["attempts"]) for entry in saved["entries"]], [1, 2, 1])
            self.assertNotEqual(saved["entries"][1]["attempts"][0]["log_file"],
                                saved["entries"][1]["attempts"][1]["log_file"])

    def test_unstable_continues_and_is_excluded_from_paired_comparison(self) -> None:
        """非有限候选停止但不影响下一组，有限前缀不能参与完整配对差。"""

        definition = simulated_definition()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = simulated_results(root)
            write_metrics(directories[1] / "metrics.csv", [metrics_row(1), metrics_row(2, float("nan"))])
            manifest = root / "batch_summary.json"
            engine._save_batch_manifest(manifest, engine._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True}, definition))
            with mock.patch.object(engine, "_run_scenario", side_effect=[
                directories[0], engine.NumericalInstability("模拟非有限MRR", [directories[1]]), directories[2],
            ]) as train:
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
                self.assertEqual(train.call_count, 3)
                train.reset_mock()
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
                train.assert_not_called()
            result = json.loads((root / "analysis/seed42_topology_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed_with_unstable")
            self.assertEqual(result["units"][1]["finite_rounds"], 1)
            self.assertIsNone(result["units"][1]["last30_mean_mrr"])
            self.assertIsNone(result["units"][1]["participant_count_mean"])
            self.assertIsNone(result["paired_differences"][0]["best_validation_mrr"])
            self.assertIsNotNone(result["paired_differences"][1]["best_validation_mrr"])

    def test_topology_subprocess_uses_own_config_and_stops_nonfinite_loss(self) -> None:
        """模拟真实子进程接口，验证本臂输出查找和非有限轮终止接线。"""

        item = contract.scenarios()[2]
        definition = runner.experiment_definition()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = mock.Mock()
            process.poll.return_value = None

            def start_child(*args, **kwargs):
                """在启动瞬间生成FLnoSnF对应目录和完整非有限损失记录。"""

                directory = root / definition.result_root / (definition.expected_config(item)["run_name"] + "_mock")
                directory.mkdir(parents=True)
                row = metrics_row(1)
                row["mean_client_train_loss"] = "nan"
                write_metrics(directory / "metrics.csv", [row])
                return process

            with mock.patch.object(engine, "PACKAGE_DIR", root), \
                    mock.patch.object(engine.subprocess, "Popen", side_effect=start_child) as child:
                with self.assertRaises(engine.NumericalInstability):
                    engine._run_scenario(item, root / "training.log", definition)
            command = child.call_args[0][0]
            self.assertEqual(command[2], "HFLSnF_KG_v5.run_federated_transe")
            self.assertEqual(command[-1], str(item.scenario.config_path))
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=10)

    def test_user_interrupt_records_state_and_does_not_run_next_group(self) -> None:
        """中断保留恢复清单，不启动后续组。"""

        definition = simulated_definition()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "batch_summary.json"
            engine._save_batch_manifest(manifest, engine._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True}, definition))
            with mock.patch.object(engine, "_run_scenario", side_effect=KeyboardInterrupt) as train:
                self.assertEqual(engine.run_training_batch(manifest, definition), 130)
                train.assert_called_once()
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual([entry["status"] for entry in saved["entries"]],
                             ["interrupted", "pending", "pending"])

    def test_manifest_rejects_other_suite_reordering_and_topology_tampering(self) -> None:
        """不能恢复学习率扫描清单、交换拓扑顺序或替换冻结调度身份。"""

        definition = runner.experiment_definition()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "batch_summary.json"
            for change in ("suite", "order", "topology"):
                payload = engine._empty_batch_payload(
                    {"status": "passed", "recompute_partitions": True}, definition)
                if change == "suite":
                    payload["suite"] = audit_contract.SUITE_NAME
                elif change == "order":
                    payload["entries"].reverse()
                else:
                    payload["entries"][1]["topology_hash"] = "0" * 64
                engine._save_batch_manifest(manifest, payload)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    engine._load_batch_manifest(manifest, definition)

    def test_report_contains_budget_and_signed_differences_without_order_requirement(self) -> None:
        """允许任何有效MRR排序，报告配对差与预算必须按对应臂计算。"""

        definition = simulated_definition()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = simulated_results(root)
            manifest = root / "batch_summary.json"
            engine._save_batch_manifest(manifest, engine._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True}, definition))
            with mock.patch.object(engine, "_run_scenario", side_effect=directories):
                self.assertEqual(engine.run_training_batch(manifest, definition), 0)
            result = json.loads((root / "analysis/seed42_topology_summary.json").read_text(encoding="utf-8"))
            self.assertEqual([unit["arm"] for unit in result["units"]], ["hflsnf", "hflnosnf", "flnosnf"])
            self.assertEqual([unit["participant_count_mean"] for unit in result["units"]], [34.21, 12.49, 4.74])
            self.assertAlmostEqual(result["paired_differences"][0]["best_validation_mrr"], -0.05)
            self.assertAlmostEqual(result["paired_differences"][1]["last30_mean_mrr"], -0.1)
            report = (root / "analysis/FedAvg种子42三拓扑实验报告.md").read_text(encoding="utf-8")
            self.assertIn("系统级收益", report)
            self.assertTrue((root / "analysis/seed42_topology_units.csv").is_file())

    def test_cpu_formal_refuses_before_validation_or_batch_creation(self) -> None:
        """没有CUDA时不得创建批次、复算分区或启动训练。"""

        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(contract, "validate_configs") as validate, \
                mock.patch.object(engine, "create_batch_manifest") as create:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runner.main(["formal3"])
        validate.assert_not_called()
        create.assert_not_called()

    def test_formal_cli_rejects_skip_recompute_and_other_seeds(self) -> None:
        """正式命令不能跳过真实分区，也不支持静默扩展到其他种子。"""

        for arguments in (["formal3", "--skip-partition-recompute"], ["formal3", "--seed", "2024"],
                          ["formal9"], ["validate", "--resume", "batch.json"], ["report"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as caught:
                runner.main(arguments)
            self.assertEqual(caught.exception.code, 2)

    def test_invalid_config_stops_before_creating_batch(self) -> None:
        """配置合同失败必须阻止批次创建和任何训练子进程。"""

        with mock.patch.object(engine, "check_formal_environment"), \
                mock.patch.object(contract, "validate_configs", return_value={
                    "status": "failed", "recompute_partitions": True, "checks": [],
                }), mock.patch.object(engine, "create_batch_manifest") as create:
            self.assertEqual(runner.main(["formal3"]), 1)
        create.assert_not_called()

    def test_all_topology_result_contracts_bind_rate_partition_and_schedule(self) -> None:
        """模拟完整逐轮审计，三个拓扑都接受正确结果并拒绝错配结果。"""

        for item in contract.scenarios():
            config = contract.expected_flat_config(item)
            topology = item.scenario.topology_contract
            summary = dict(config, device="cuda:0", scenario=config["comparison_scenario"],
                           architecture=topology.architecture, snf_enabled=topology.snf_enabled,
                           edge_mode=topology.edge_mode, client_count=37, local_epochs=3,
                           server_learning_rate=1.0, server_beta1=0.0, server_beta2=0.0, server_tau=0.0,
                           server_bias_correction=False, server_optimizer_step_count=0,
                           server_optimizer_state_hash="", initial_model_hash=audit_contract.BASELINE_INITIAL_MODEL_HASHES[42],
                           partition_hash=item.scenario.partition_hash,
                           topology_schedule_hash=topology.schedule_hash, test_evaluation_performed=False,
                           final_test_metrics=None, best_round=1, best_validation_mrr_during_training=0.2,
                           participant_count_mean=2.0, group_count_mean=1.0, effective_global_passes=100.0)
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
            provider.get_round.return_value = SimpleNamespace(
                participant_count=2, group_to_client_indexes={"0": [0, 1]})
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / "model_best.pt").write_bytes(b"simulated-checkpoint")
                with mock.patch.object(audit_contract, "_load_json", side_effect=lambda path: fixtures[path.name]), \
                        mock.patch.object(audit_contract, "_load_metrics", return_value=rows), \
                        mock.patch.object(audit_contract, "_load_json_lines", return_value=audits), \
                        mock.patch.object(original_contract, "build_provider", return_value=provider):
                    self.assertEqual(contract.validate_result(directory, item)["status"], "passed")
                    for field, value in (("learning_rate", 5e-5), ("partition_hash", "0" * 64),
                                         ("topology_schedule_hash", "0" * 64), ("snf_enabled", not topology.snf_enabled),
                                         ("participant_count_mean", 99.0)):
                        original = summary[field]
                        summary[field] = value
                        with self.subTest(arm=item.tag, field=field):
                            self.assertEqual(contract.validate_result(directory, item)["status"], "failed")
                        summary[field] = original


if __name__ == "__main__":
    unittest.main()
