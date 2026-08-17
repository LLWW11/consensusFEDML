"""校验、顺序运行并恢复最终随机拓扑FedAdam九组实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .tasks.kge.final_stochastic_fedadam import (
    PARTITION_HASHES,
    ROUND_COUNT,
    SCENARIOS,
    SUITE_NAME,
    FinalStochasticScenario,
    expected_flat_config,
    scenario_by_id,
    validate_configs,
    validate_result,
)
from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    load_flat_config,
    write_json_report,
)


BATCH_FILE_NAME = "batch_summary.json"
CONTRACT_FILE_NAME = "final_stochastic_fedadam_formal150_contract.json"


class FinalStochasticRunError(RuntimeError):
    """表示训练失败或没有产生唯一结果目录。"""

    def __init__(
        self,
        message: str,
        created_results: Sequence[Path] = (),
    ) -> None:
        """保存错误文本和本次尝试可能创建的结果目录。"""

        super().__init__(message)
        self.created_results = tuple(Path(path).resolve() for path in created_results)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建随机九组实验的命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="校验或运行最终随机拓扑FedAdam九组实验"
    )
    parser.add_argument(
        "action",
        choices=("validate", "formal150"),
        help="只校验配置，或顺序运行九组150轮实验",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从既有batch_summary.json跳过已通过项目并继续",
    )
    return parser


def _timestamp() -> str:
    """返回带时区的本地时间文本。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    """计算文件SHA-256以锁定批次输入。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_directories(run_name: str) -> Tuple[Path, ...]:
    """列出与运行名前缀匹配的既有结果目录。"""

    return tuple(sorted(
        (
            path.resolve()
            for path in (PACKAGE_DIR / "results").glob("{}_*".format(run_name))
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime_ns,
    ))


def _new_result_directories(
    run_name: str,
    before: Sequence[Path],
) -> Tuple[Path, ...]:
    """返回本次子进程新增的结果目录集合。"""

    existing = {Path(path).resolve() for path in before}
    return tuple(path for path in _result_directories(run_name) if path not in existing)


def _manifest_entry(
    scenario: FinalStochasticScenario,
    order_index: int,
) -> Dict[str, object]:
    """构造一条绑定配置指纹的待运行批次项目。"""

    return {
        "scenario_id": scenario.scenario_id,
        "arm": scenario.arm,
        "seed": scenario.seed,
        "config": str(scenario.config_path),
        "schedule_hash": scenario.schedule_hash,
        "config_sha256": _file_sha256(scenario.config_path),
        "order_index": order_index,
        "status": "pending",
        "attempt_count": 0,
        "attempts": [],
        "result_dir": None,
        "contract_file": None,
        "error": None,
    }


def _empty_manifest() -> Dict[str, object]:
    """构造按种子优先顺序排列的九组初始清单。"""

    now = _timestamp()
    return {
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "planned_run_count": len(SCENARIOS),
        "status": "pending",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "run_order": [scenario.scenario_id for scenario in SCENARIOS],
        "seed_contracts": {
            str(seed): {
                "partition_hash": partition_hash,
                "initial_model_hash": None,
            }
            for seed, partition_hash in PARTITION_HASHES.items()
        },
        "entries": [
            _manifest_entry(scenario, index)
            for index, scenario in enumerate(SCENARIOS, start=1)
        ],
    }


def _save_manifest(path: Path, payload: Mapping[str, object]) -> None:
    """更新时间并写回批次清单。"""

    mutable = dict(payload)
    mutable["updated_at"] = _timestamp()
    write_json_report(mutable, path)


def create_batch_manifest(result_root: Optional[Path] = None) -> Path:
    """创建不会覆盖旧结果的新批次目录和清单。"""

    root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results").resolve()
    )
    batch_name = "final_stochastic_fedadam_batch_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    manifest_path = root / batch_name / BATCH_FILE_NAME
    write_json_report(_empty_manifest(), manifest_path)
    return manifest_path


def _validate_manifest_entry(entry: Mapping[str, object]) -> None:
    """拒绝配置路径、内容或场景身份已经变化的恢复项目。"""

    scenario = scenario_by_id(str(entry.get("scenario_id", "")))
    config_path = Path(str(entry.get("config", ""))).expanduser().resolve()
    if config_path != scenario.config_path or not config_path.is_file():
        raise FileNotFoundError("批次配置不存在或路径已变化：{}".format(config_path))
    if _file_sha256(config_path) != entry.get("config_sha256"):
        raise ValueError("批次配置已被修改：{}".format(config_path))
    for field, expected in expected_flat_config(scenario).items():
        actual = load_flat_config(config_path).get(field)
        if actual != expected:
            raise ValueError("{}配置字段{}已偏离合同".format(scenario.scenario_id, field))
    for field, expected in (
        ("arm", scenario.arm),
        ("seed", scenario.seed),
        ("schedule_hash", scenario.schedule_hash),
    ):
        if entry.get(field) != expected:
            raise ValueError("批次字段{}已偏离场景合同".format(field))


def _load_manifest(
    path: Path,
    result_root: Optional[Path] = None,
) -> Dict[str, object]:
    """读取并校验批次位置、顺序和全部配置指纹。"""

    resolved = Path(path).expanduser().resolve()
    root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results").resolve()
    )
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("恢复清单必须位于指定results目录内") from error
    if resolved.name != BATCH_FILE_NAME or not resolved.is_file():
        raise FileNotFoundError("找不到合法批次清单：{}".format(resolved))
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("批次清单顶层必须是对象")
    expected_order = [scenario.scenario_id for scenario in SCENARIOS]
    entries = payload.get("entries")
    if (
        payload.get("suite") != SUITE_NAME
        or int(payload.get("round_count", -1)) != ROUND_COUNT
        or not isinstance(entries, list)
        or [str(item.get("scenario_id")) for item in entries] != expected_order
        or payload.get("run_order") != expected_order
    ):
        raise ValueError("批次套件、轮数或九组顺序不符合合同")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("批次项目必须是对象")
        _validate_manifest_entry(entry)
    return payload


def _run_config(scenario: FinalStochasticScenario) -> Path:
    """调用统一训练入口并返回本次唯一的新结果目录。"""

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
        subprocess.run(command, cwd=str(PACKAGE_DIR.parent), check=True)
    except subprocess.CalledProcessError as error:
        raise FinalStochasticRunError(
            "{}训练子进程失败，退出码{}".format(
                scenario.scenario_id, error.returncode
            ),
            created_results=_new_result_directories(run_name, before),
        ) from error
    created = _new_result_directories(run_name, before)
    if len(created) != 1:
        raise FinalStochasticRunError(
            "{}期望新建1个结果目录，实际为{}".format(
                scenario.scenario_id, len(created)
            ),
            created_results=created,
        )
    return created[0]


def _expected_initial_hash(
    payload: Mapping[str, object],
    seed: int,
) -> str:
    """读取同种子首个通过实验已经锁定的初始模型哈希。"""

    contracts = payload.get("seed_contracts")
    if not isinstance(contracts, dict) or not isinstance(contracts.get(str(seed)), dict):
        raise KeyError("批次缺少seed{}合同".format(seed))
    return str(contracts[str(seed)].get("initial_model_hash") or "")


def _lock_initial_hash(
    payload: Dict[str, object],
    scenario: FinalStochasticScenario,
    report: Mapping[str, object],
) -> None:
    """用同种子首个结果锁定后三个预算档位共享的初始模型。"""

    contracts = payload["seed_contracts"]
    assert isinstance(contracts, dict)
    contract = contracts[str(scenario.seed)]
    assert isinstance(contract, dict)
    actual = str(report["initial_model_hash"])
    existing = str(contract.get("initial_model_hash") or "")
    if existing and existing != actual:
        raise ValueError("seed{}三个档位的初始模型哈希不一致".format(scenario.seed))
    contract["initial_model_hash"] = actual


def run_batch(
    manifest_path: Path,
    result_root: Optional[Path] = None,
) -> int:
    """顺序运行未通过项目，失败即停并保存可恢复状态。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_manifest(resolved, result_root=result_root)
    payload["status"] = "running"
    _save_manifest(resolved, payload)
    entries = payload["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry.get("status") == "passed":
            print("跳过已通过实验：{}".format(entry["scenario_id"]))
            continue
        scenario = scenario_by_id(str(entry["scenario_id"]))
        attempts = entry.get("attempts")
        if not isinstance(attempts, list):
            raise TypeError("批次项目attempts必须是列表")
        attempt: Dict[str, object] = {
            "attempt": len(attempts) + 1,
            "status": "running",
            "started_at": _timestamp(),
            "completed_at": None,
            "result_dir": None,
            "contract_file": None,
            "created_results": [],
            "error": None,
        }
        attempts.append(attempt)
        entry.update({"status": "running", "attempt_count": len(attempts), "error": None})
        _save_manifest(resolved, payload)
        print("开始最终随机实验：{}".format(scenario.scenario_id))
        try:
            result_dir = _run_config(scenario)
            attempt["result_dir"] = str(result_dir)
            attempt["created_results"] = [str(result_dir)]
            report = validate_result(
                result_dir,
                scenario,
                expected_initial_model_hash=_expected_initial_hash(
                    payload, scenario.seed
                ),
            )
            contract_path = result_dir / CONTRACT_FILE_NAME
            write_json_report(report, contract_path)
            attempt["contract_file"] = str(contract_path)
            if report["status"] != "passed":
                raise RuntimeError("结果合同校验未通过")
            _lock_initial_hash(payload, scenario, report)
        except (Exception, KeyboardInterrupt) as error:
            created = (
                error.created_results
                if isinstance(error, FinalStochasticRunError)
                else tuple(Path(path) for path in attempt["created_results"])
            )
            attempt.update({
                "status": "failed",
                "completed_at": _timestamp(),
                "error": "{}: {}".format(type(error).__name__, error),
                "created_results": [str(path) for path in created],
            })
            entry.update({"status": "failed", "error": attempt["error"]})
            payload["status"] = "failed"
            _save_manifest(resolved, payload)
            print("实验失败并已停止：{}".format(scenario.scenario_id))
            print(
                "恢复命令：python -m HFLSnF_KG_v3.run_final_stochastic_fedadam "
                "formal150 --resume \"{}\"".format(resolved)
            )
            return 1
        attempt.update({"status": "passed", "completed_at": _timestamp()})
        entry.update({
            "status": "passed",
            "result_dir": attempt["result_dir"],
            "contract_file": attempt["contract_file"],
            "error": None,
        })
        _save_manifest(resolved, payload)
    payload["status"] = "passed"
    payload["completed_at"] = _timestamp()
    _save_manifest(resolved, payload)
    print("最终随机拓扑FedAdam九组实验全部完成")
    print("批次清单：{}".format(resolved))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """先校验九份配置，再执行新批次或恢复既有批次。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    report = validate_configs()
    print("最终随机FedAdam九组配置合同：{}".format(report["status"]))
    if report["status"] != "passed":
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
