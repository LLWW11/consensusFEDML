"""校验、运行和恢复FedAdam阶段二最多二十二组正式实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .reports.gen_fedadam_stage2_report import write_phase_artifacts
from .tasks.kge.fedadam_stage2 import (
    ROUND_COUNT,
    SCREEN_SCENARIOS,
    SUITE_NAME,
    Stage2Scenario,
    build_followup_scenarios,
    expected_flat_config,
    scenario_from_manifest_entry,
    scenario_to_manifest_entry,
    select_screen_candidate,
    stable_payload_hash,
    summarize_result,
    validate_result,
    validate_screen_configs,
)
from .tasks.kge.fixed_count_four_scenarios import (
    INITIAL_MODEL_HASH,
    PACKAGE_DIR,
    PARTITION_HASH,
    load_flat_config,
    write_json_report,
)


BATCH_FILE_NAME = "batch_summary.json"
PLANNED_RUN_COUNT = 22
PHASE_ACTIONS = ("screen150", "confirm150", "controls150")


class Stage2RunError(RuntimeError):
    """表示一次阶段二子进程或结果目录发现失败。"""

    def __init__(
        self,
        message: str,
        created_results: Sequence[Path] = (),
    ) -> None:
        """保存错误文本以及本次尝试可能创建的结果目录。"""

        super().__init__(message)
        self.created_results = tuple(
            Path(path).resolve() for path in created_results
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """创建阶段二校验、三阶段运行和恢复参数解析器。"""

    parser = argparse.ArgumentParser(
        description="校验或顺序运行FedAdam阶段二全因子实验"
    )
    parser.add_argument(
        "action",
        choices=("validate",) + PHASE_ACTIONS,
        help="校验配置，或运行筛选、复验、人数对照阶段",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="恢复既有batch_summary.json；复验和对照阶段必须提供",
    )
    return parser


def _timestamp() -> str:
    """返回适合批次清单审计字段的本地时间文本。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    """返回配置文件的SHA-256以锁定恢复批次参数。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_binding_fields(
    entry: Mapping[str, object],
) -> Dict[str, object]:
    """提取恢复期间禁止变化的场景和配置身份字段。"""

    keys = (
        "scenario_id",
        "phase",
        "arm",
        "seed",
        "setting",
        "topology_util",
        "server_bias_correction",
        "topology_type",
        "snf_enabled",
        "participant_count",
        "config",
        "schedule_hash",
        "config_sha256",
    )
    return {key: entry.get(key) for key in keys}


def _bound_entry(scenario: Stage2Scenario) -> Dict[str, object]:
    """构造含配置指纹和场景绑定指纹的批次项目。"""

    entry = scenario_to_manifest_entry(scenario)
    entry["config_sha256"] = _file_sha256(scenario.config_path)
    entry["binding_hash"] = stable_payload_hash(
        _entry_binding_fields(entry)
    )
    return entry


def _empty_batch_payload() -> Dict[str, object]:
    """构造只包含八组筛选实验的阶段二初始清单。"""

    now = _timestamp()
    return {
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "planned_run_count": PLANNED_RUN_COUNT,
        "status": "pending",
        "current_phase": None,
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "selection": None,
        "seed_contracts": {
            "42": {
                "partition_hash": PARTITION_HASH,
                "initial_model_hash": INITIAL_MODEL_HASH,
            }
        },
        "phases": {
            "screen": {
                "status": "pending",
                "completed_at": None,
                "entries": [
                    _bound_entry(scenario)
                    for scenario in SCREEN_SCENARIOS
                ],
            },
            "confirm": {
                "status": "locked",
                "completed_at": None,
                "entries": [],
            },
            "controls": {
                "status": "locked",
                "completed_at": None,
                "entries": [],
            },
        },
    }


def create_batch_manifest(
    result_root: Optional[Path] = None,
) -> Path:
    """创建不会覆盖旧结果的阶段二批次目录和初始清单。"""

    resolved_root = (
        Path(result_root).resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results").resolve()
    )
    batch_name = "fedadam_stage2_batch_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    manifest_path = resolved_root / batch_name / BATCH_FILE_NAME
    write_json_report(_empty_batch_payload(), manifest_path)
    return manifest_path


def _validate_entry_binding(entry: Mapping[str, object]) -> None:
    """拒绝配置文件或批次场景字段在恢复前被替换。"""

    config_path = Path(str(entry.get("config", ""))).resolve()
    if not config_path.is_file():
        raise FileNotFoundError("批次配置不存在：{}".format(config_path))
    actual_config_hash = _file_sha256(config_path)
    if actual_config_hash != entry.get("config_sha256"):
        raise ValueError("批次配置文件已被修改：{}".format(config_path))
    actual_binding = stable_payload_hash(_entry_binding_fields(entry))
    if actual_binding != entry.get("binding_hash"):
        raise ValueError(
            "批次项目绑定字段已被修改：{}".format(
                entry.get("scenario_id")
            )
        )
    scenario = scenario_from_manifest_entry(entry)
    config = load_flat_config(scenario.config_path)
    for field, expected in expected_flat_config(scenario).items():
        if config.get(field) != expected:
            raise ValueError(
                "{}配置字段{}已偏离批次合同".format(
                    scenario.scenario_id, field
                )
            )


def _load_batch_manifest(
    path: Path,
    result_root: Optional[Path] = None,
) -> Dict[str, object]:
    """读取并校验阶段二清单路径、套件、阶段和全部绑定字段。"""

    resolved_path = Path(path).expanduser().resolve()
    resolved_root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results").resolve()
    )
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("恢复清单必须位于指定results目录内") from error
    if resolved_path.name != BATCH_FILE_NAME or not resolved_path.is_file():
        raise FileNotFoundError(
            "找不到合法的阶段二批次清单：{}".format(resolved_path)
        )
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("恢复清单的实验套件与阶段二不一致")
    if int(payload.get("round_count", -1)) != ROUND_COUNT:
        raise ValueError("恢复清单的通信轮数不是150")
    if int(payload.get("planned_run_count", -1)) != PLANNED_RUN_COUNT:
        raise ValueError("恢复清单的计划实验数量不是22")
    phases = payload.get("phases")
    if not isinstance(phases, dict):
        raise TypeError("恢复清单phases必须是对象")
    for phase_name in ("screen", "confirm", "controls"):
        phase = phases.get(phase_name)
        if not isinstance(phase, dict) or not isinstance(
            phase.get("entries"), list
        ):
            raise TypeError("恢复清单阶段{}结构无效".format(phase_name))
        for entry in phase["entries"]:
            _validate_entry_binding(entry)
    expected_screen = [item.scenario_id for item in SCREEN_SCENARIOS]
    actual_screen = [
        str(item.get("scenario_id"))
        for item in phases["screen"]["entries"]
    ]
    if actual_screen != expected_screen:
        raise ValueError("筛选阶段八组场景集合或顺序已改变")
    return payload


def _save_batch_manifest(
    manifest_path: Path,
    payload: Mapping[str, object],
) -> None:
    """更新时间并把批次状态持久化到固定清单路径。"""

    mutable = dict(payload)
    mutable["updated_at"] = _timestamp()
    write_json_report(mutable, manifest_path)


def _result_directories(run_name: str) -> Tuple[Path, ...]:
    """返回项目结果目录中匹配运行名前缀的全部目录。"""

    return tuple(
        sorted(
            (
                path.resolve()
                for path in (PACKAGE_DIR / "results").glob(
                    "{}_*".format(run_name)
                )
                if path.is_dir()
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
    )


def _new_result_directories(
    run_name: str,
    before: Sequence[Path],
) -> Tuple[Path, ...]:
    """通过运行前后集合差找出本次尝试创建的结果目录。"""

    existing = {Path(path).resolve() for path in before}
    return tuple(
        path for path in _result_directories(run_name) if path not in existing
    )


def _run_config(scenario: Stage2Scenario) -> Path:
    """调用统一训练入口并返回本次唯一新建的结果目录。"""

    config = load_flat_config(scenario.config_path)
    run_name = str(config["run_name"])
    before = _result_directories(run_name)
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v3.run_federated_transe",
        "--cf",
        str(scenario.config_path),
    ]
    try:
        # 子进程继承当前Python和CUDA环境，训练日志直接输出到终端。
        subprocess.run(
            command,
            cwd=str(PACKAGE_DIR.parent),
            check=True,
        )
    except subprocess.CalledProcessError as error:
        created = _new_result_directories(run_name, before)
        raise Stage2RunError(
            "{}训练子进程失败，退出码{}".format(
                scenario.scenario_id, error.returncode
            ),
            created_results=created,
        ) from error
    created = _new_result_directories(run_name, before)
    if len(created) != 1:
        raise Stage2RunError(
            "{}期望新建1个结果目录，实际为{}：{}".format(
                scenario.scenario_id, len(created), created
            ),
            created_results=created,
        )
    return created[0]


def _phase_name(action: str) -> str:
    """把命令行动作映射为批次清单内部阶段键。"""

    mapping = {
        "screen150": "screen",
        "confirm150": "confirm",
        "controls150": "controls",
    }
    return mapping[str(action)]


def _contract_file_name(phase: str) -> str:
    """返回一个阶段写入每个结果目录的结果合同文件名。"""

    return "fedadam_stage2_{}_formal150_contract.json".format(phase)


def _compact_analysis(
    analysis: Mapping[str, object],
) -> Dict[str, object]:
    """移除逐轮数组后保存适合批次清单的单次结果摘要。"""

    return {
        key: value for key, value in analysis.items() if key != "rounds"
    }


def _seed_contract(
    payload: Mapping[str, object],
    seed: int,
) -> Tuple[Optional[str], Optional[str]]:
    """读取当前批次已经锁定的同seed分区和初始模型哈希。"""

    contracts = payload.get("seed_contracts", {})
    item = contracts.get(str(int(seed))) if isinstance(contracts, dict) else None
    if not isinstance(item, dict):
        return None, None
    return (
        str(item.get("partition_hash", "")) or None,
        str(item.get("initial_model_hash", "")) or None,
    )


def _lock_seed_contract(
    payload: Dict[str, object],
    scenario: Stage2Scenario,
    report: Mapping[str, object],
) -> None:
    """用同seed首个通过结果锁定后续各实验臂的可比性哈希。"""

    contracts = payload.setdefault("seed_contracts", {})
    if not isinstance(contracts, dict):
        raise TypeError("批次清单seed_contracts必须是对象")
    key = str(scenario.seed)
    actual = {
        "partition_hash": str(report["partition_hash"]),
        "initial_model_hash": str(report["initial_model_hash"]),
    }
    existing = contracts.get(key)
    if existing is None:
        contracts[key] = actual
    elif existing != actual:
        raise ValueError("同seed分区或初始模型哈希不一致：{}".format(key))


def _record_failure(
    manifest_path: Path,
    payload: Dict[str, object],
    phase_name: str,
    entry: Dict[str, object],
    attempt: Dict[str, object],
    error: BaseException,
    created_results: Sequence[Path] = (),
) -> int:
    """记录失败尝试、停止当前阶段并给出对应恢复命令。"""

    created = [str(Path(path).resolve()) for path in created_results]
    attempt.update(
        {
            "status": "failed",
            "completed_at": _timestamp(),
            "created_results": created,
            "error": "{}: {}".format(type(error).__name__, error),
        }
    )
    entry.update(
        {
            "status": "failed",
            "error": attempt["error"],
            "result_dir": created[0] if len(created) == 1 else None,
        }
    )
    phase = payload["phases"][phase_name]
    phase["status"] = "failed"
    payload["status"] = "failed"
    payload["current_phase"] = phase_name
    _save_batch_manifest(manifest_path, payload)
    action = {
        "screen": "screen150",
        "confirm": "confirm150",
        "controls": "controls150",
    }[phase_name]
    print("FedAdam阶段二{}已停止：{}".format(phase_name, attempt["error"]))
    print("恢复命令：")
    print(
        "python -m HFLSnF_KG_v3.run_fedadam_stage2 "
        "{} --resume {}".format(action, manifest_path)
    )
    return 2


def _populate_followups(
    manifest_path: Path,
    payload: Dict[str, object],
) -> None:
    """根据八组筛选摘要选择候选并固化14组后续配置。"""

    screen_entries = payload["phases"]["screen"]["entries"]
    summaries = [entry.get("analysis") for entry in screen_entries]
    if not all(isinstance(item, dict) for item in summaries):
        raise ValueError("筛选阶段缺少完整单次分析摘要")
    # 候选选择需要逐轮曲线，因此从已通过结果重新读取，不依赖清单压缩数据。
    full_summaries = [
        summarize_result(
            Path(str(entry["result_dir"])),
            scenario_from_manifest_entry(entry),
        )
        for entry in screen_entries
    ]
    selection = select_screen_candidate(full_summaries)
    selected_key = str(selection["selected_setting"])
    confirm, controls = build_followup_scenarios(
        manifest_path.parent,
        selected_key,
    )
    payload["selection"] = selection
    payload["phases"]["confirm"] = {
        "status": "pending",
        "completed_at": None,
        "entries": [_bound_entry(item) for item in confirm],
    }
    payload["phases"]["controls"] = {
        "status": "pending",
        "completed_at": None,
        "entries": [_bound_entry(item) for item in controls],
    }
    actual_total = sum(
        len(payload["phases"][name]["entries"])
        for name in ("screen", "confirm", "controls")
    )
    if actual_total != PLANNED_RUN_COUNT:
        raise RuntimeError(
            "阶段二固化后应为22组，实际为{}".format(actual_total)
        )


def _finalize_phase(
    manifest_path: Path,
    payload: Dict[str, object],
    phase_name: str,
) -> None:
    """完成候选固化、分析报告和批次阶段状态更新。"""

    if phase_name == "screen":
        _populate_followups(manifest_path, payload)
    write_phase_artifacts(manifest_path, payload, phase_name)
    phase = payload["phases"][phase_name]
    phase["status"] = "passed"
    phase["completed_at"] = _timestamp()
    payload["current_phase"] = None
    if phase_name == "screen":
        payload["status"] = "screen_passed"
    elif phase_name == "confirm":
        payload["status"] = "confirm_passed"
    else:
        payload["status"] = "passed"
        payload["completed_at"] = _timestamp()
    _save_batch_manifest(manifest_path, payload)


def _check_phase_prerequisites(
    payload: Mapping[str, object],
    phase_name: str,
) -> None:
    """保证复验和人数对照只能沿用已完成的筛选批次。"""

    phases = payload["phases"]
    if phase_name == "confirm" and phases["screen"]["status"] != "passed":
        raise ValueError("confirm150要求筛选阶段已经通过")
    if phase_name == "controls":
        if phases["screen"]["status"] != "passed":
            raise ValueError("controls150要求筛选阶段已经通过")
        if phases["confirm"]["status"] != "passed":
            raise ValueError("controls150要求复验阶段已经通过")
    if phase_name != "screen" and not isinstance(payload.get("selection"), dict):
        raise ValueError("后续阶段必须绑定筛选阶段自动选择结果")


def run_phase(
    manifest_path: Path,
    action: str,
    result_root: Optional[Path] = None,
) -> int:
    """跳过已通过项目，顺序运行指定阶段并逐项保存进度。"""

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved_manifest, result_root=result_root)
    phase_name = _phase_name(action)
    _check_phase_prerequisites(payload, phase_name)
    phase = payload["phases"][phase_name]
    if phase.get("status") == "passed":
        # 已完成阶段只做只读确认，避免误操作重建后续配置或覆盖状态。
        print("FedAdam阶段二{}已经完成，无需重复运行".format(phase_name))
        print("批次清单：{}".format(resolved_manifest))
        return 0
    payload["status"] = "running"
    payload["current_phase"] = phase_name
    phase["status"] = "running"
    _save_batch_manifest(resolved_manifest, payload)
    for entry in phase["entries"]:
        if entry.get("status") == "passed":
            print("跳过已通过实验：{}".format(entry["scenario_id"]))
            continue
        scenario = scenario_from_manifest_entry(entry)
        attempts = entry.get("attempts")
        if not isinstance(attempts, list):
            raise TypeError("批次清单attempts必须是列表")
        attempt: Dict[str, object] = {
            "attempt_number": len(attempts) + 1,
            "status": "running",
            "started_at": _timestamp(),
            "completed_at": None,
            "result_dir": None,
            "contract_file": None,
            "created_results": [],
            "error": None,
        }
        attempts.append(attempt)
        entry.update({"status": "running", "error": None})
        _save_batch_manifest(resolved_manifest, payload)
        print("开始阶段二{}实验：{}".format(phase_name, scenario.scenario_id))
        try:
            result_dir = _run_config(scenario)
            attempt["result_dir"] = str(result_dir)
            attempt["created_results"] = [str(result_dir)]
            expected_partition, expected_initial = _seed_contract(
                payload, scenario.seed
            )
            report = validate_result(
                result_dir,
                scenario,
                expected_partition_hash=expected_partition,
                expected_initial_model_hash=expected_initial,
            )
            contract_path = result_dir / _contract_file_name(phase_name)
            write_json_report(report, contract_path)
            attempt["contract_file"] = str(contract_path)
            if report["status"] != "passed":
                raise RuntimeError("结果合同校验未通过")
            _lock_seed_contract(payload, scenario, report)
            analysis = summarize_result(result_dir, scenario)
        except KeyboardInterrupt as error:
            return _record_failure(
                resolved_manifest,
                payload,
                phase_name,
                entry,
                attempt,
                error,
                created_results=tuple(
                    Path(path) for path in attempt["created_results"]
                ),
            )
        except Exception as error:
            created_results = (
                error.created_results
                if isinstance(error, Stage2RunError)
                else tuple(
                    Path(path) for path in attempt["created_results"]
                )
            )
            return _record_failure(
                resolved_manifest,
                payload,
                phase_name,
                entry,
                attempt,
                error,
                created_results=created_results,
            )
        attempt.update(
            {"status": "passed", "completed_at": _timestamp()}
        )
        entry.update(
            {
                "status": "passed",
                "result_dir": attempt["result_dir"],
                "contract_file": attempt["contract_file"],
                "analysis": _compact_analysis(analysis),
                "error": None,
            }
        )
        _save_batch_manifest(resolved_manifest, payload)
        print("{}结果合同：passed".format(scenario.scenario_id))
    try:
        _finalize_phase(resolved_manifest, payload, phase_name)
    except Exception as error:
        # 阶段级分析失败时保留全部已通过训练，恢复不会重复训练。
        phase["status"] = "failed"
        phase["finalization_error"] = "{}: {}".format(
            type(error).__name__, error
        )
        payload["status"] = "failed"
        _save_batch_manifest(resolved_manifest, payload)
        print("阶段二{}汇总失败：{}".format(phase_name, error))
        return 2
    print("FedAdam阶段二{}全部完成".format(phase_name))
    print("批次清单：{}".format(resolved_manifest))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """先校验筛选合同，再创建、运行或恢复指定阶段。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config_report = validate_screen_configs()
    print("FedAdam阶段二筛选配置合同：{}".format(config_report["status"]))
    if config_report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        if args.resume is not None:
            parser.error("validate操作不能使用--resume")
        return
    if args.action in ("confirm150", "controls150") and args.resume is None:
        parser.error("{}必须通过--resume绑定筛选批次".format(args.action))
    manifest_path = (
        Path(args.resume).expanduser().resolve()
        if args.resume is not None
        else create_batch_manifest()
    )
    print("批次清单：{}".format(manifest_path))
    exit_code = run_phase(manifest_path, args.action)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
