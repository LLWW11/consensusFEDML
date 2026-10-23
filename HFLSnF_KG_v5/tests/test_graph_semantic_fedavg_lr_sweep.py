"""验证四学习率隔离、子进程监测、恢复及曲线统计。"""

from __future__ import annotations

import csv
import io
import json
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_fedavg_lr_sweep as runner
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_lr_sweep as contract
from HFLSnF_KG_v5.tasks.kge import graph_semantic_fedavg_comparison as original_contract
from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import load_flat_config


def metrics_row(number: int, mrr: float = 0.2) -> dict:
    """提供包含合法占位NaN的模拟逐轮指标。"""

    return {
        "round": str(number), "mean_client_train_loss": "1.0",
        "val_mrr": str(mrr), "val_mean_rank": "100.0", "round_seconds": "2.0",
        "client_sampling_seconds": "nan",
    }


def write_metrics(path: Path, rows: list) -> None:
    """通过CSV写入器生成与真实训练相同的完整记录。"""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LearningRateSweepTest(unittest.TestCase):
    """检查正式条件保持和不会错误终止正常训练的数值监测。"""

    def test_four_configs_change_only_rates_and_identity(self) -> None:
        """四组必须从同一来源派生，并重跑包含5e-5的完整矩阵。"""

        items = contract.scenarios()
        self.assertEqual([item.learning_rate for item in items], [5e-5, 1e-4, 2e-4, 5e-4])
        excluded = {"learning_rate", "lr", "ablation_suite", "ablation_arm",
                    "comparison_scenario", "run_name", "result_root"}
        for item in items:
            actual = load_flat_config(item.scenario.config_path)
            source = load_flat_config(item.scenario.source_config_path)
            self.assertEqual(actual, contract.expected_flat_config(item))
            self.assertEqual({key: value for key, value in actual.items() if key not in excluded},
                             {key: value for key, value in source.items() if key not in excluded})
            self.assertEqual(actual["learning_rate"], actual["lr"])
            self.assertTrue(actual["require_cuda"])
        payload = runner._empty_batch_payload({"status": "passed", "recompute_partitions": True})
        self.assertEqual([entry["status"] for entry in payload["entries"]], ["pending"] * 4)
        self.assertTrue(all(entry["result_dir"] is None for entry in payload["entries"]))

    def test_partial_csv_is_not_a_nonfinite_round(self) -> None:
        """正在写入的非有限末行必须等待完整换行后才能判定。"""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            path.write_text("round,mean_client_train_loss,val_mrr\n1,1,0.2\n2,nan,",
                            encoding="utf-8-sig")
            rows = runner._read_completed_metrics(path)
            self.assertEqual(len(rows), 1)
            runner._check_finite_metrics(rows)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("0.3\n")
            with self.assertRaises(runner.NumericalInstability):
                runner._check_finite_metrics(runner._read_completed_metrics(path))

    def test_monitor_allows_fluctuation_and_unused_timing_nan(self) -> None:
        """验证MRR回落与未开启剖析的NaN不能终止候选。"""

        rows = [metrics_row(1, 0.3), metrics_row(2, 0.1)]
        runner._check_finite_metrics(rows)

    def test_monitor_rejects_each_nonfinite_loss_or_validation_field(self) -> None:
        """损失、MRR或其他验证数值出现NaN或无穷必须识别为不稳定。"""

        for field in ("mean_client_train_loss", "val_mrr", "val_mean_rank"):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(field=field, value=value):
                    row = metrics_row(3)
                    row[field] = value
                    with self.assertRaises(runner.NumericalInstability):
                        runner._check_finite_metrics([row])

    def test_nonfinite_subprocess_is_stopped_and_bound(self) -> None:
        """模拟子进程输出非有限完整轮，必须停止该进程并保留目录。"""

        item = contract.scenarios()[0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "result"
            directory.mkdir()
            write_metrics(directory / "metrics.csv", [metrics_row(1, float("nan"))])
            process = mock.Mock()
            process.poll.return_value = None
            with mock.patch.object(runner, "_result_directories", side_effect=[(), (directory,)]), \
                    mock.patch.object(runner.subprocess, "Popen", return_value=process):
                with self.assertRaises(runner.NumericalInstability) as caught:
                    runner._run_scenario(item, Path(temporary) / "training.log")
            self.assertEqual(caught.exception.created_results, (directory.resolve(),))
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=10)
            process.kill.assert_not_called()

    def test_last_round_written_on_process_exit_is_checked(self) -> None:
        """退出瞬间写出的非有限记录不能漏过最后一次监测。"""

        item = contract.scenarios()[0]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "result"
            directory.mkdir()
            path = directory / "metrics.csv"
            write_metrics(path, [metrics_row(1)])
            process = mock.Mock()

            def exit_with_nonfinite_row() -> int:
                """模拟训练退出前完成新的非有限轮记录。"""

                write_metrics(path, [metrics_row(1), metrics_row(2, float("inf"))])
                return 0

            process.poll.side_effect = exit_with_nonfinite_row
            with mock.patch.object(runner, "_result_directories",
                                   side_effect=[(), (directory,), (directory,)]), \
                    mock.patch.object(runner.subprocess, "Popen", return_value=process):
                with self.assertRaises(runner.NumericalInstability):
                    runner._run_scenario(item, Path(temporary) / "training.log")
            process.terminate.assert_not_called()

    def test_stop_process_kills_only_after_termination_timeout(self) -> None:
        """子进程不响应终止时才强制结束，并等待其退出。"""

        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("training", 10), 0]
        runner._stop_process(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_unstable_continues_and_resume_skips_finished_entries(self) -> None:
        """不稳定组之后继续执行，恢复不能重跑已完成或已判不稳定组。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "batch_summary.json"
            payload = runner._empty_batch_payload({"status": "passed", "recompute_partitions": True})
            directories = []
            for index in range(4):
                directory = root / ("run{}".format(index))
                directory.mkdir()
                (directory / "model_best.pt").write_bytes(b"test-checkpoint")
                directories.append(directory)
            runner._save_batch_manifest(manifest, payload)
            instability = runner.NumericalInstability("模拟非有限损失", [directories[1]])
            with mock.patch.object(runner, "_run_scenario",
                                   side_effect=[directories[0], instability, directories[2], directories[3]]) as run, \
                    mock.patch.object(runner, "validate_result", return_value={"status": "passed"}), \
                    mock.patch.object(runner, "generate_report"):
                self.assertEqual(runner.run_training_batch(manifest), 0)
                self.assertEqual(run.call_count, 4)
                saved = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "completed_with_unstable")
                self.assertEqual([entry["status"] for entry in saved["entries"]],
                                 ["passed", "unstable", "passed", "passed"])
                run.reset_mock()
                self.assertEqual(runner.run_training_batch(manifest), 0)
                run.assert_not_called()

    def test_exit_race_does_not_hide_numerical_failure(self) -> None:
        """进程在检查与终止之间退出时，清理不能覆盖原始不稳定原因。"""

        process = mock.Mock()
        process.poll.return_value = None
        process.terminate.side_effect = ProcessLookupError("进程刚刚退出")
        runner._stop_process(process)
        process.wait.assert_called_once_with()
        process.kill.assert_not_called()

    def test_failed_batch_restarts_only_unfinished_groups(self) -> None:
        """执行失败停止后，恢复从未完成组重跑并保留独立尝试日志。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "batch_summary.json"
            payload = runner._empty_batch_payload({"status": "passed", "recompute_partitions": True})
            runner._save_batch_manifest(manifest, payload)
            directory = root / "completed"
            directory.mkdir()
            (directory / "model_best.pt").write_bytes(b"test-checkpoint")
            with mock.patch.object(runner, "validate_result", return_value={"status": "passed"}), \
                    mock.patch.object(runner, "generate_report"), \
                    mock.patch.object(runner, "_run_scenario", side_effect=[
                        directory, runner.TrainingRunError("模拟CUDA错误"),
                    ]) as run:
                self.assertEqual(runner.run_training_batch(manifest), 1)
                self.assertEqual(run.call_count, 2)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual([entry["status"] for entry in saved["entries"]],
                             ["passed", "failed", "pending", "pending"])
            with mock.patch.object(runner, "validate_result", return_value={"status": "passed"}), \
                    mock.patch.object(runner, "generate_report"), \
                    mock.patch.object(runner, "_run_scenario", return_value=directory) as run:
                self.assertEqual(runner.run_training_batch(manifest), 0)
                self.assertEqual(run.call_count, 3)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["entries"][1]["attempts"]), 2)
            self.assertNotEqual(saved["entries"][1]["attempts"][0]["log_file"],
                                saved["entries"][1]["attempts"][1]["log_file"])

    def test_keyboard_interrupt_records_recoverable_state(self) -> None:
        """用户中断不得继续下一组，清单保留已开始的尝试。"""

        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "batch_summary.json"
            runner._save_batch_manifest(manifest, runner._empty_batch_payload(
                {"status": "passed", "recompute_partitions": True},
            ))
            with mock.patch.object(runner, "_run_scenario", side_effect=KeyboardInterrupt):
                self.assertEqual(runner.run_training_batch(manifest), 130)
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "interrupted")
            self.assertEqual(saved["entries"][0]["status"], "interrupted")
            self.assertEqual(saved["entries"][1]["status"], "pending")

    def test_manifest_rejects_rate_binding_and_checkpoint_changes(self) -> None:
        """恢复必须拒绝学习率身份变化及已完成检查点被替换。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "batch_summary.json"
            payload = runner._empty_batch_payload({"status": "passed", "recompute_partitions": True})
            payload["entries"][0]["learning_rate"] = 0.1
            runner._save_batch_manifest(manifest, payload)
            with self.assertRaisesRegex(ValueError, "恢复绑定"):
                runner._load_batch_manifest(manifest)
            payload = runner._empty_batch_payload({"status": "passed", "recompute_partitions": True})
            (root / "model_best.pt").write_bytes(b"changed-checkpoint")
            payload["entries"][0].update(status="passed", result_dir=str(root),
                                          checkpoint_sha256="0" * 64)
            runner._save_batch_manifest(manifest, payload)
            with mock.patch.object(runner, "validate_result", return_value={"status": "passed"}):
                with self.assertRaisesRegex(ValueError, "检查点内容"):
                    runner._load_batch_manifest(manifest)

    def test_curve_statistics_and_partial_report_have_explicit_meaning(self) -> None:
        """最佳值、150轮值、总体标准差和绝对回撤必须来自实际曲线。"""

        rows = [metrics_row(index, 0.1) for index in range(1, 151)]
        rows[19]["val_mrr"] = "0.5"
        rows[20]["val_mrr"] = "0.2"
        rows[-1]["val_mrr"] = "0.3"
        summary = runner.summarize_metrics(rows)
        self.assertEqual(summary["best_round"], 20)
        self.assertEqual(summary["best_validation_mrr"], 0.5)
        self.assertEqual(summary["round150_mrr"], 0.3)
        self.assertAlmostEqual(summary["max_mrr_drawdown"], 0.4)
        self.assertEqual(summary["total_round_seconds"], 300)
        self.assertAlmostEqual(summary["last30_mean_mrr"], statistics.mean([0.1] * 29 + [0.3]))
        self.assertAlmostEqual(summary["last30_std_mrr"], statistics.pstdev([0.1] * 29 + [0.3]))
        partial = runner.summarize_metrics(rows[:20] + [metrics_row(21, float("nan"))])
        self.assertEqual(partial["finite_rounds"], 20)
        self.assertIsNone(partial["round150_mrr"])
        self.assertIsNone(partial["last30_mean_mrr"])

    def test_cpu_formal_entry_fails_before_creating_manifest(self) -> None:
        """CUDA不可用时不得创建正式批次，也不得进行分区复算或训练。"""

        with mock.patch("torch.cuda.is_available", return_value=False), \
                mock.patch.object(runner, "create_batch_manifest") as create, \
                mock.patch.object(runner, "validate_configs") as validate:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runner.main(["formal4"])
        create.assert_not_called()
        validate.assert_not_called()

    def test_report_exports_complete_and_unstable_candidates(self) -> None:
        """导出报告必须区分150轮模型和不稳定前缀，并保留JSON与CSV数值。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "batch_summary.json"
            payload = runner._empty_batch_payload({"status": "passed", "recompute_partitions": True})
            payload["status"] = "completed_with_unstable"
            for index, entry in enumerate(payload["entries"]):
                directory = root / ("run{}".format(index))
                directory.mkdir()
                rows = [metrics_row(number, 0.1) for number in range(1, 151)]
                if index == 1:
                    rows = [metrics_row(1), metrics_row(2, float("nan"))]
                    entry["status"] = "unstable"
                else:
                    rows[-1]["val_mrr"] = "0.3"
                    entry["status"] = "passed"
                    entry["checkpoint_sha256"] = "test-checkpoint"
                entry["result_dir"] = str(directory)
                write_metrics(directory / "metrics.csv", rows)
            runner._save_batch_manifest(manifest, payload)
            with mock.patch.object(runner, "_verify_passed_entry"):
                report = runner.generate_report(manifest)
            self.assertIn("末30轮", report.read_text(encoding="utf-8"))
            summary = json.loads((root / "analysis" / "lr_sweep_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["units"][0]["round150_mrr"], 0.3)
            self.assertIsNone(summary["units"][1]["round150_mrr"])
            self.assertEqual(summary["units"][1]["finite_rounds"], 1)
            csv_text = (root / "analysis" / "lr_sweep_units.csv").read_text(encoding="utf-8-sig")
            units = list(csv.DictReader(io.StringIO(csv_text)))
            self.assertEqual(len(units), 4)
            self.assertEqual(units[1]["status"], "unstable")
            self.assertEqual(units[1]["round150_mrr"], "")

    def test_formal_cannot_skip_partition_recompute(self) -> None:
        """正式入口必须拒绝仅静态校验的启动方式。"""

        with self.assertRaises(SystemExit) as caught:
            runner.main(["formal4", "--skip-partition-recompute"])
        self.assertEqual(caught.exception.code, 2)

    def test_result_contract_rejects_effective_learning_rate_mismatch(self) -> None:
        """配置正确但实际训练汇总学习率不同仍必须拒绝正式结果。"""

        item = contract.scenarios()[1]
        config = contract.expected_flat_config(item)
        summary = dict(config, device="cuda:0", scenario=config["comparison_scenario"],
                       architecture="hfl", snf_enabled=True, edge_mode="fixed", client_count=37,
                       local_epochs=3, server_learning_rate=1.0, server_beta1=0.0,
                       server_beta2=0.0, server_tau=0.0, server_bias_correction=False,
                       server_optimizer_step_count=0, server_optimizer_state_hash="",
                       initial_model_hash=contract.BASELINE_INITIAL_MODEL_HASHES[42],
                       partition_hash=item.scenario.partition_hash,
                       topology_schedule_hash=item.scenario.topology_contract.schedule_hash,
                       test_evaluation_performed=False, final_test_metrics=None,
                       best_round=1, best_validation_mrr_during_training=0.2)
        rows = []
        audits = []
        for index in range(1, 151):
            row = metrics_row(index)
            row.update(active_client_count="2", contributing_client_count="2", active_group_count="1",
                       server_active_row_count="3", server_model_delta_l2="0", server_update_l2="0",
                       server_optimizer="fedavg", server_optimizer_step="0", server_optimizer_state_hash="")
            rows.append(row)
            audits.append({"round": index, "fedavg_candidate_state_hash": "a" * 64,
                           "cloud_parameter_state_hash": "a" * 64, "server_optimizer_state_hash": ""})
        fixtures = {
            "summary.json": summary, "config_snapshot.json": config,
            "client_partition_summary.json": {"partition_hash": item.scenario.partition_hash,
                                               "max_relative_load_deviation": 0.04},
            "dynamic_participation_summary.json": {"schedule_hash": item.scenario.topology_contract.schedule_hash},
            "topology_metadata.json": {"provider_type": "matlab_adapter", "architecture": "hfl",
                                       "snf_enabled": True, "edge_mode": "fixed", "topology_util": 0.6,
                                       "topology_schedule_policy": "strict"},
        }
        provider = mock.Mock()
        provider.get_round.return_value = SimpleNamespace(participant_count=2, group_to_client_indexes={"0": [0, 1]})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "model_best.pt").write_bytes(b"test-checkpoint")
            with mock.patch.object(contract, "_load_json", side_effect=lambda path: fixtures[path.name]), \
                    mock.patch.object(contract, "_load_metrics", return_value=rows), \
                    mock.patch.object(contract, "_load_json_lines", return_value=audits), \
                    mock.patch.object(original_contract, "build_provider", return_value=provider):
                self.assertEqual(contract.validate_result(directory, item)["status"], "passed")
                summary["learning_rate"] = 5e-5
                result = contract.validate_result(directory, item)
                self.assertEqual(result["status"], "failed")
                self.assertIn("summary.learning_rate", [check["name"] for check in result["checks"]
                                                        if not check["passed"]])


if __name__ == "__main__":
    unittest.main()
