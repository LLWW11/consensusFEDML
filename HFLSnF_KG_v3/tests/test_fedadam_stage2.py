"""验证FedAdam阶段二合同、候选选择和二十二组可恢复编排。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Mapping
from unittest.mock import patch

from HFLSnF_KG_v3.run_fedadam_stage2 import (
    PLANNED_RUN_COUNT,
    Stage2RunError,
    create_batch_manifest,
    run_phase,
)
from HFLSnF_KG_v3.tasks.kge.fedadam_stage2 import (
    BASELINE_SETTING_KEY,
    ROUND_COUNT,
    SCREEN_SCENARIOS,
    select_screen_candidate,
    validate_screen_configs,
)
from HFLSnF_KG_v3.tasks.kge.fixed_count_four_scenarios import (
    INITIAL_MODEL_HASH,
    PARTITION_HASH,
)


class FedAdamStage2Test(unittest.TestCase):
    """检查全因子配置、自动选择、阶段绑定和失败恢复。"""

    def test_screen_configs_and_schedules_match_contracts(self) -> None:
        """确认八份YAML只改变实验臂、util、bias和身份字段。"""

        report = validate_screen_configs()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["behavior_configs_equal"])
        self.assertEqual(len(report["formal_configs"]), 8)
        self.assertEqual(
            [scenario.arm for scenario in SCREEN_SCENARIOS],
            ["hflsnf", "hflnosnf"] * 4,
        )
        for item in report["formal_configs"]:
            stats = item["schedule_statistics"]
            self.assertEqual(stats["rounds"], ROUND_COUNT)
            self.assertEqual(stats["zero_participation_clients"], 0)
            self.assertGreater(stats["participant_count_min"], 0)

    def test_candidate_selector_prefers_qualified_platform_gap(self) -> None:
        """确认达到0.01平台改进的非劣候选被标记为胜出配置。"""

        summaries = self._screen_summaries(
            {
                "u0p5_bctrue": (0.34, 0.32),
                "u0p5_bcfalse": (0.338, 0.317),
                "u0p6_bctrue": (0.337, 0.314),
                "u0p6_bcfalse": (0.345, 0.295),
            }
        )
        selection = select_screen_candidate(summaries)
        self.assertEqual(selection["baseline_setting"], BASELINE_SETTING_KEY)
        self.assertEqual(selection["selected_setting"], "u0p6_bcfalse")
        self.assertEqual(selection["selection_label"], "winner")
        self.assertGreaterEqual(
            next(
                item["platform_gap_improvement"]
                for item in selection["candidates"]
                if item["setting"] == "u0p6_bcfalse"
            ),
            0.01,
        )

    def test_candidate_selector_uses_challenger_when_gate_fails(self) -> None:
        """确认没有候选过门槛时仍固化挑战组但不误称胜出。"""

        summaries = self._screen_summaries(
            {
                "u0p5_bctrue": (0.34, 0.32),
                "u0p5_bcfalse": (0.339, 0.319),
                "u0p6_bctrue": (0.338, 0.317),
                "u0p6_bcfalse": (0.337, 0.316),
            }
        )
        selection = select_screen_candidate(summaries)
        self.assertEqual(selection["selection_label"], "challenger")
        self.assertEqual(
            selection["selection_reason"], "no_candidate_met_gate"
        )
        self.assertNotEqual(
            selection["selected_setting"], BASELINE_SETTING_KEY
        )

    def test_full_twenty_two_run_pipeline_is_bound_and_ordered(self) -> None:
        """模拟三阶段全成功并确认8加8加6组及自动绑定配置。"""

        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary).resolve()
            manifest_path = create_batch_manifest(result_root)
            called: List[str] = []

            with self._successful_pipeline_patches(result_root, called):
                self.assertEqual(
                    run_phase(
                        manifest_path,
                        "screen150",
                        result_root=result_root,
                    ),
                    0,
                )
                screened = self._read_manifest(manifest_path)
                self.assertEqual(screened["phases"]["screen"]["status"], "passed")
                self.assertEqual(screened["selection"]["selected_setting"], "u0p6_bcfalse")
                self.assertEqual(len(screened["phases"]["confirm"]["entries"]), 8)
                self.assertEqual(len(screened["phases"]["controls"]["entries"]), 6)
                self.assertEqual(
                    sum(
                        len(screened["phases"][name]["entries"])
                        for name in ("screen", "confirm", "controls")
                    ),
                    PLANNED_RUN_COUNT,
                )
                confirm_settings = {
                    entry["setting"]
                    for entry in screened["phases"]["confirm"]["entries"]
                }
                self.assertEqual(
                    confirm_settings,
                    {BASELINE_SETTING_KEY, "u0p6_bcfalse"},
                )
                self.assertEqual(
                    {
                        entry["participant_count"]
                        for entry in screened["phases"]["controls"]["entries"]
                    },
                    {34, 12},
                )
                self.assertEqual(
                    run_phase(
                        manifest_path,
                        "confirm150",
                        result_root=result_root,
                    ),
                    0,
                )
                self.assertEqual(
                    run_phase(
                        manifest_path,
                        "controls150",
                        result_root=result_root,
                    ),
                    0,
                )

            final_payload = self._read_manifest(manifest_path)
            self.assertEqual(final_payload["status"], "passed")
            self.assertEqual(len(called), PLANNED_RUN_COUNT)
            self.assertEqual(
                called[:8],
                [scenario.scenario_id for scenario in SCREEN_SCENARIOS],
            )
            for phase_name in ("screen", "confirm", "controls"):
                self.assertTrue(
                    all(
                        entry["status"] == "passed"
                        and Path(str(entry["contract_file"])).is_file()
                        for entry in final_payload["phases"][phase_name]["entries"]
                    )
                )

    def test_screen_failure_stops_and_resume_skips_passed(self) -> None:
        """模拟第二组失败并确认恢复不重跑首组且增加失败组尝试。"""

        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary).resolve()
            manifest_path = create_batch_manifest(result_root)
            initial_calls: List[str] = []

            def fail_second(scenario: object) -> Path:
                """让第二组训练失败以验证立即停止语义。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                initial_calls.append(scenario_id)
                if len(initial_calls) == 2:
                    raise Stage2RunError("模拟训练失败")
                result_dir = result_root / "initial_{}".format(scenario_id)
                result_dir.mkdir(parents=True)
                return result_dir

            with patch(
                "HFLSnF_KG_v3.run_fedadam_stage2._run_config",
                side_effect=fail_second,
            ), patch(
                "HFLSnF_KG_v3.run_fedadam_stage2.validate_result",
                side_effect=self._fake_contract,
            ), patch(
                "HFLSnF_KG_v3.run_fedadam_stage2.summarize_result",
                side_effect=self._fake_summary,
            ), patch(
                "HFLSnF_KG_v3.run_fedadam_stage2.write_phase_artifacts",
                return_value={},
            ):
                first_exit = run_phase(
                    manifest_path,
                    "screen150",
                    result_root=result_root,
                )

            failed = self._read_manifest(manifest_path)
            self.assertEqual(first_exit, 2)
            self.assertEqual(
                [entry["status"] for entry in failed["phases"]["screen"]["entries"]],
                ["passed", "failed"] + ["pending"] * 6,
            )

            resumed_calls: List[str] = []

            def resume_run(scenario: object) -> Path:
                """为恢复阶段未通过场景创建新的模拟结果目录。"""

                scenario_id = str(getattr(scenario, "scenario_id"))
                resumed_calls.append(scenario_id)
                result_dir = result_root / "resume_{}".format(scenario_id)
                result_dir.mkdir(parents=True)
                return result_dir

            with patch(
                "HFLSnF_KG_v3.run_fedadam_stage2._run_config",
                side_effect=resume_run,
            ), patch(
                "HFLSnF_KG_v3.run_fedadam_stage2.validate_result",
                side_effect=self._fake_contract,
            ), patch(
                "HFLSnF_KG_v3.run_fedadam_stage2.summarize_result",
                side_effect=self._fake_summary,
            ), patch(
                "HFLSnF_KG_v3.run_fedadam_stage2.write_phase_artifacts",
                return_value={},
            ):
                resume_exit = run_phase(
                    manifest_path,
                    "screen150",
                    result_root=result_root,
                )

            resumed = self._read_manifest(manifest_path)
            self.assertEqual(resume_exit, 0)
            self.assertEqual(
                resumed_calls,
                [item.scenario_id for item in SCREEN_SCENARIOS[1:]],
            )
            self.assertEqual(
                len(resumed["phases"]["screen"]["entries"][0]["attempts"]),
                1,
            )
            self.assertEqual(
                len(resumed["phases"]["screen"]["entries"][1]["attempts"]),
                2,
            )

    def test_report_writer_creates_all_required_plots(self) -> None:
        """模拟完整结果并确认全程、复验和参与预算图均可生成。"""

        from HFLSnF_KG_v3.reports.gen_fedadam_stage2_report import (
            write_phase_artifacts,
        )

        with tempfile.TemporaryDirectory() as temporary:
            batch_dir = Path(temporary).resolve()
            manifest_path = batch_dir / "batch_summary.json"
            manifest_path.write_text("{}", encoding="utf-8")
            payload = self._report_payload()

            def fake_full_summary(
                entry: Mapping[str, object],
            ) -> Dict[str, object]:
                """按报告项目身份构造完整150轮绘图摘要。"""

                arm = str(entry["arm"])
                setting = str(entry["setting"])
                seed = int(entry["seed"])
                participant = entry.get("participant_count")
                arm_offset = 0.02 if arm == "hflsnf" else 0.0
                seed_offset = (seed % 10) * 0.0001
                value = 0.30 + arm_offset + seed_offset
                if arm == "hflkge":
                    value = 0.25 + int(participant) * 0.002 + seed_offset
                return {
                    "scenario_id": entry["scenario_id"],
                    "arm": arm,
                    "setting": setting,
                    "seed": seed,
                    "participant_count": participant,
                    "platform": {"mrr_mean": value},
                    "client_participation_min": 1,
                    "client_participation_median": 10,
                    "client_participation_max": 150,
                    "rounds": [
                        {
                            "round": index,
                            "val_mrr": value - 0.1 / index,
                            "server_update_l2": 10.0 / index,
                        }
                        for index in range(1, ROUND_COUNT + 1)
                    ],
                }

            with patch(
                "HFLSnF_KG_v3.reports.gen_fedadam_stage2_report._full_summary",
                side_effect=fake_full_summary,
            ):
                for phase_name in ("screen", "confirm", "controls"):
                    artifacts = write_phase_artifacts(
                        manifest_path,
                        payload,
                        phase_name,
                    )
                    self.assertTrue(Path(artifacts["analysis_json"]).is_file())
                    self.assertTrue(Path(artifacts["analysis_markdown"]).is_file())
                    self.assertTrue(
                        all(Path(path).is_file() for path in artifacts["plots"])
                    )

    def _successful_pipeline_patches(
        self,
        result_root: Path,
        called: List[str],
    ) -> object:
        """返回三阶段全成功测试使用的组合补丁上下文。"""

        test_case = self

        class PatchStack:
            """管理多个unittest.mock补丁的进入与退出。"""

            def __enter__(self) -> "PatchStack":
                """启动训练、合同、分析和报告四个模拟补丁。"""

                def fake_run(scenario: object) -> Path:
                    """创建唯一模拟结果目录并记录实际编排顺序。"""

                    scenario_id = str(getattr(scenario, "scenario_id"))
                    called.append(scenario_id)
                    result_dir = result_root / "result_{}_{}".format(
                        len(called), scenario_id
                    )
                    result_dir.mkdir(parents=True)
                    return result_dir

                self.patchers = [
                    patch(
                        "HFLSnF_KG_v3.run_fedadam_stage2._run_config",
                        side_effect=fake_run,
                    ),
                    patch(
                        "HFLSnF_KG_v3.run_fedadam_stage2.validate_result",
                        side_effect=test_case._fake_contract,
                    ),
                    patch(
                        "HFLSnF_KG_v3.run_fedadam_stage2.summarize_result",
                        side_effect=test_case._fake_summary,
                    ),
                    patch(
                        "HFLSnF_KG_v3.run_fedadam_stage2.write_phase_artifacts",
                        return_value={},
                    ),
                ]
                for patcher in self.patchers:
                    patcher.start()
                return self

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                """按逆序停止全部模拟补丁。"""

                for patcher in reversed(self.patchers):
                    patcher.stop()

        return PatchStack()

    @staticmethod
    def _report_payload() -> Dict[str, object]:
        """构造报告绘图测试使用的8加8加6组项目清单。"""

        settings = (
            "u0p5_bctrue",
            "u0p5_bcfalse",
            "u0p6_bctrue",
            "u0p6_bcfalse",
        )
        screen = [
            {
                "scenario_id": "{}_{}_seed42".format(arm, setting),
                "status": "passed",
                "arm": arm,
                "setting": setting,
                "seed": 42,
                "participant_count": None,
                "result_dir": "unused",
            }
            for setting in settings
            for arm in ("hflsnf", "hflnosnf")
        ]
        confirm = [
            {
                "scenario_id": "{}_{}_seed{}".format(arm, setting, seed),
                "status": "passed",
                "arm": arm,
                "setting": setting,
                "seed": seed,
                "participant_count": None,
                "result_dir": "unused",
            }
            for seed in (2024, 2025)
            for setting in ("u0p5_bctrue", "u0p6_bcfalse")
            for arm in ("hflsnf", "hflnosnf")
        ]
        controls = [
            {
                "scenario_id": "hflkge_k{}_u0p6_bcfalse_seed{}".format(
                    participant_count, seed
                ),
                "status": "passed",
                "arm": "hflkge",
                "setting": "u0p6_bcfalse",
                "seed": seed,
                "participant_count": participant_count,
                "result_dir": "unused",
            }
            for participant_count in (34, 12)
            for seed in (42, 2024, 2025)
        ]
        return {
            "selection": {
                "baseline_setting": "u0p5_bctrue",
                "selected_setting": "u0p6_bcfalse",
                "selection_label": "winner",
                "pairs": {},
            },
            "phases": {
                "screen": {"entries": screen},
                "confirm": {"entries": confirm},
                "controls": {"entries": controls},
            },
        }

    @staticmethod
    def _fake_contract(
        result_dir: Path,
        scenario: object,
        expected_partition_hash: object = None,
        expected_initial_model_hash: object = None,
    ) -> Dict[str, object]:
        """返回按seed稳定的最小通过合同供批次测试使用。"""

        seed = int(getattr(scenario, "seed"))
        if seed == 42:
            partition_hash = PARTITION_HASH
            initial_hash = INITIAL_MODEL_HASH
        else:
            partition_hash = ("{:064x}".format(seed))[-64:]
            initial_hash = ("{:064x}".format(seed + 10000))[-64:]
        if expected_partition_hash is not None:
            if partition_hash != expected_partition_hash:
                return {"status": "failed"}
        if expected_initial_model_hash is not None:
            if initial_hash != expected_initial_model_hash:
                return {"status": "failed"}
        return {
            "status": "passed",
            "scenario_id": str(getattr(scenario, "scenario_id")),
            "result_dir": str(result_dir),
            "partition_hash": partition_hash,
            "initial_model_hash": initial_hash,
        }

    @staticmethod
    def _fake_summary(
        result_dir: Path,
        scenario: object,
    ) -> Dict[str, object]:
        """按设置生成能稳定选中u0p6_bcfalse的模拟逐轮摘要。"""

        setting = str(getattr(getattr(scenario, "setting"), "key"))
        arm = str(getattr(scenario, "arm"))
        platform_map: Mapping[str, tuple] = {
            "u0p5_bctrue": (0.34, 0.32),
            "u0p5_bcfalse": (0.338, 0.317),
            "u0p6_bctrue": (0.337, 0.314),
            "u0p6_bcfalse": (0.345, 0.295),
        }
        if arm == "hflkge":
            participant_count = int(getattr(scenario, "participant_count"))
            value = 0.25 + participant_count * 0.002
        else:
            value = platform_map[setting][0 if arm == "hflsnf" else 1]
            participant_count = getattr(scenario, "participant_count")
        return {
            "scenario_id": str(getattr(scenario, "scenario_id")),
            "phase": str(getattr(scenario, "phase")),
            "arm": arm,
            "setting": setting,
            "seed": int(getattr(scenario, "seed")),
            "participant_count": participant_count,
            "platform": {
                "mrr_mean": value,
                "mrr_std": 0.0,
                "mrr_slope": 0.0,
                "label": "platform",
            },
            "cold_start": {"drawdown": 0.0, "recovery_round": 1},
            "zero_participation_clients": 0,
            "rounds": [
                {
                    "round": index,
                    "val_mrr": value,
                    "server_update_l2": 1.0,
                }
                for index in range(1, ROUND_COUNT + 1)
            ],
        }

    @classmethod
    def _screen_summaries(
        cls,
        platform_values: Mapping[str, tuple],
    ) -> List[Dict[str, object]]:
        """按四个设置的平台值构造选择器单元测试输入。"""

        summaries: List[Dict[str, object]] = []
        for setting, pair in platform_values.items():
            for arm, value in zip(("hflsnf", "hflnosnf"), pair):
                summaries.append(
                    {
                        "setting": setting,
                        "arm": arm,
                        "platform": {"mrr_mean": float(value)},
                        "cold_start": {"drawdown": 0.0},
                        "zero_participation_clients": 0,
                        "rounds": [
                            {"round": index, "val_mrr": float(value)}
                            for index in range(1, ROUND_COUNT + 1)
                        ],
                    }
                )
        return summaries

    @staticmethod
    def _read_manifest(path: Path) -> Dict[str, object]:
        """从磁盘读取批次清单以避免断言依赖内存对象。"""

        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)


if __name__ == "__main__":
    unittest.main()
