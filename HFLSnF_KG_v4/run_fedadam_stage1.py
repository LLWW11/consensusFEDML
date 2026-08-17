"""校验、批量运行和恢复FedAdam阶段一八组实验。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .tasks.kge.fedadam_stage1 import (
    ROUND_COUNT,
    SCENARIOS,
    SUITE_NAME,
    Stage1Scenario,
    scenario_by_id,
    validate_configs,
    validate_result,
)
from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    load_flat_config,
    write_json_report,
)


CONTRACT_FILE_NAME = "fedadam_stage1_formal40_contract.json"
BATCH_FILE_NAME = "batch_summary.json"


class Stage1RunError(RuntimeError):
    """表示一次子进程运行或结果目录发现失败。"""

    def __init__(
        self,
        message: str,
        created_results: Sequence[Path] = (),
    ) -> None:
        """保存错误文本以及失败尝试可能创建的结果目录。"""

        super().__init__(message)
        self.created_results = tuple(
            Path(path).resolve() for path in created_results
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """创建阶段一配置校验、批量运行和恢复参数解析器。"""

    parser = argparse.ArgumentParser(
        description="校验或顺序运行FedAdam阶段一八组40轮实验"
    )
    parser.add_argument(
        "action",
        choices=("validate", "formal40"),
        help="只校验全部配置，或运行完整40轮批次",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从既有batch_summary.json恢复未通过的实验",
    )
    return parser


def _timestamp() -> str:
    """返回适合结果目录和审计字段使用的本地时间文本。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _result_directories(run_name: str) -> Tuple[Path, ...]:
    """返回当前结果根目录中匹配运行名的所有目录。"""

    result_root = PACKAGE_DIR / "results"
    return tuple(
        sorted(
            (
                path.resolve()
                for path in result_root.glob(
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
    """通过运行前后的集合差找出本次尝试创建的结果目录。"""

    before_set = {Path(path).resolve() for path in before}
    return tuple(
        path
        for path in _result_directories(run_name)
        if path not in before_set
    )


def _run_config(scenario: Stage1Scenario) -> Path:
    """使用统一训练入口运行一个场景并返回唯一的新结果目录。"""

    config_path = PACKAGE_DIR / "configs" / scenario.formal_config
    config = load_flat_config(config_path)
    run_name = str(config["run_name"])
    before = _result_directories(run_name)
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v3.run_federated_transe",
        "--cf",
        str(config_path),
    ]
    try:
        # 子进程继承当前Python与CUDA环境，并把训练日志直接输出到终端。
        subprocess.run(
            command,
            cwd=str(PACKAGE_DIR.parent),
            check=True,
        )
    except subprocess.CalledProcessError as error:
        created = _new_result_directories(run_name, before)
        raise Stage1RunError(
            "{}训练子进程失败，退出码{}".format(
                scenario.scenario_id,
                error.returncode,
            ),
            created_results=created,
        ) from error
    created = _new_result_directories(run_name, before)
    if len(created) != 1:
        raise Stage1RunError(
            "{}期望新建1个结果目录，实际为{}：{}".format(
                scenario.scenario_id,
                len(created),
                created,
            ),
            created_results=created,
        )
    return created[0]


def _empty_batch_payload() -> Dict[str, object]:
    """构造一个尚未开始训练的完整八组批次清单。"""

    now = _timestamp()
    entries: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        entries.append(
            {
                "scenario_id": scenario.scenario_id,
                "profile": scenario.profile.key,
                "arm": scenario.arm,
                "config": scenario.formal_config,
                "server_learning_rate": (
                    scenario.profile.learning_rate
                ),
                "server_tau": scenario.profile.tau,
                "server_bias_correction": True,
                "status": "pending",
                "result_dir": None,
                "contract_file": None,
                "error": None,
                "attempts": [],
            }
        )
    return {
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "status": "pending",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "entries": entries,
    }


def create_batch_manifest(
    result_root: Optional[Path] = None,
) -> Path:
    """创建不会覆盖旧批次的目录和初始批次清单。"""

    resolved_root = (
        Path(result_root).resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results").resolve()
    )
    directory_name = "fedadam_stage1_batch_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    manifest_path = resolved_root / directory_name / BATCH_FILE_NAME
    write_json_report(_empty_batch_payload(), manifest_path)
    return manifest_path


def _load_batch_manifest(
    path: Path,
    result_root: Optional[Path] = None,
) -> Dict[str, object]:
    """读取并校验恢复清单属于指定结果根目录、当前套件和固定顺序。"""

    resolved_path = Path(path).expanduser().resolve()
    resolved_root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results").resolve()
    )
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("恢复清单必须位于项目results目录内") from error
    if resolved_path.name != BATCH_FILE_NAME or not resolved_path.is_file():
        raise FileNotFoundError(
            "找不到合法的阶段一批次清单：{}".format(resolved_path)
        )
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("恢复清单的实验套件与当前阶段一不一致")
    if int(payload.get("round_count", -1)) != ROUND_COUNT:
        raise ValueError("恢复清单的通信轮数不是40")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise TypeError("恢复清单entries必须是列表")
    expected_ids = [item.scenario_id for item in SCENARIOS]
    actual_ids = [str(item.get("scenario_id")) for item in entries]
    if actual_ids != expected_ids:
        raise ValueError("恢复清单的八组场景集合或顺序已改变")
    expected_configs = [item.formal_config for item in SCENARIOS]
    actual_configs = [str(item.get("config")) for item in entries]
    # 配置归档会改变路径绑定；拒绝静默地用新路径恢复旧批次。
    if actual_configs != expected_configs:
        raise ValueError(
            "恢复清单的配置路径与当前阶段一归档入口不一致；"
            "归档前批次仅支持只读审计"
        )
    return payload


def _save_batch_manifest(
    manifest_path: Path,
    payload: Mapping[str, object],
) -> None:
    """更新时间并把批次状态持久化到固定清单路径。"""

    mutable_payload = dict(payload)
    mutable_payload["updated_at"] = _timestamp()
    write_json_report(mutable_payload, manifest_path)


def _record_failure(
    manifest_path: Path,
    payload: Dict[str, object],
    entry: Dict[str, object],
    attempt: Dict[str, object],
    error: BaseException,
    created_results: Sequence[Path] = (),
) -> int:
    """记录当前尝试失败、停止批次并返回统一失败退出码。"""

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
    payload["status"] = "failed"
    payload["completed_at"] = _timestamp()
    _save_batch_manifest(manifest_path, payload)
    print("阶段一批次已停止：{}".format(attempt["error"]))
    print("恢复命令：")
    print(
        "python -m HFLSnF_KG_v3.run_fedadam_stage1 "
        "formal40 --resume {}".format(manifest_path)
    )
    return 2


def run_batch(
    manifest_path: Path,
    result_root: Optional[Path] = None,
) -> int:
    """跳过已通过项目，顺序执行其余实验并逐项写入审计状态。"""

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved_manifest, result_root=result_root)
    payload["status"] = "running"
    payload["completed_at"] = None
    _save_batch_manifest(resolved_manifest, payload)
    entries = payload["entries"]
    for entry in entries:
        if entry.get("status") == "passed":
            print("跳过已通过实验：{}".format(entry["scenario_id"]))
            continue
        scenario = scenario_by_id(str(entry["scenario_id"]))
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
        entry.update(
            {
                "status": "running",
                "error": None,
            }
        )
        _save_batch_manifest(resolved_manifest, payload)
        print("开始阶段一实验：{}".format(scenario.scenario_id))
        try:
            result_dir = _run_config(scenario)
            attempt["result_dir"] = str(result_dir)
            attempt["created_results"] = [str(result_dir)]
            report = validate_result(result_dir, scenario)
            contract_path = result_dir / CONTRACT_FILE_NAME
            write_json_report(report, contract_path)
            attempt["contract_file"] = str(contract_path)
            if report["status"] != "passed":
                raise RuntimeError("结果合同校验未通过")
        except KeyboardInterrupt as error:
            return _record_failure(
                resolved_manifest,
                payload,
                entry,
                attempt,
                error,
                created_results=(
                    tuple(
                        Path(path)
                        for path in attempt["created_results"]
                    )
                ),
            )
        except Exception as error:
            created_results = (
                error.created_results
                if isinstance(error, Stage1RunError)
                else tuple(
                    Path(path)
                    for path in attempt["created_results"]
                )
            )
            return _record_failure(
                resolved_manifest,
                payload,
                entry,
                attempt,
                error,
                created_results=created_results,
            )
        attempt.update(
            {
                "status": "passed",
                "completed_at": _timestamp(),
            }
        )
        entry.update(
            {
                "status": "passed",
                "result_dir": attempt["result_dir"],
                "contract_file": attempt["contract_file"],
                "error": None,
            }
        )
        _save_batch_manifest(resolved_manifest, payload)
        print("{}结果合同：passed".format(scenario.scenario_id))
    payload["status"] = "passed"
    payload["completed_at"] = _timestamp()
    _save_batch_manifest(resolved_manifest, payload)
    print("FedAdam阶段一八组实验全部完成")
    print("批次清单：{}".format(resolved_manifest))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """先校验全部配置，再执行新批次或恢复既有批次。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config_report = validate_configs()
    print("FedAdam阶段一配置合同：{}".format(config_report["status"]))
    if config_report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        if args.resume is not None:
            parser.error("validate操作不能使用--resume")
        return
    manifest_path = (
        Path(args.resume).expanduser().resolve()
        if args.resume is not None
        else create_batch_manifest()
    )
    print("批次清单：{}".format(manifest_path))
    exit_code = run_batch(manifest_path)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
