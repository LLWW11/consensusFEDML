"""校验并顺序运行V5图语义拓扑对照的六次正式训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    load_flat_config,
    write_json_report,
)
from .tasks.kge.graph_semantic_topology_extension import (
    ARM_ORDER,
    RESULT_ROOT,
    SUITE_NAME,
    GraphSemanticTopologyScenario,
    scenario_by_id,
    scenarios_from_contract,
    validate_configs,
    validate_result,
)


BATCH_FILE_NAME = "batch_summary.json"
RESULT_CONTRACT_FILE_NAME = "topology_extension_formal150_contract.json"


class TopologyExtensionRunError(RuntimeError):
    """表示训练失败或一次运行产生了非唯一结果目录。"""

    def __init__(
        self,
        message: str,
        created_results: Sequence[Path] = (),
    ) -> None:
        """保存错误文本和本次尝试创建的结果目录。"""

        super().__init__(message)
        self.created_results = tuple(
            Path(path).resolve() for path in created_results
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """创建六实验配置校验与正式训练参数。"""

    parser = argparse.ArgumentParser(
        description="运行V5图语义HFLnoSnF与FLnoSnF六实验"
    )
    parser.add_argument(
        "action",
        choices=("validate", "formal6"),
        help="选择配置校验或六次CUDA正式训练",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="formal6恢复使用的batch_summary.json",
    )
    parser.add_argument(
        "--skip-partition-recompute",
        action="store_true",
        help="仅供快速静态校验；formal6禁止使用",
    )
    return parser


def _timestamp() -> str:
    """返回包含本地时区的秒级时间。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    """分块计算一个文件的SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding_fields(entry: Mapping[str, object]) -> Dict[str, object]:
    """提取恢复时不得修改的批次身份字段。"""

    return {
        key: entry.get(key)
        for key in (
            "scenario_id",
            "arm",
            "seed",
            "partition_hash",
            "config",
            "config_sha256",
            "order_index",
        )
    }


def _stable_hash(payload: Mapping[str, object]) -> str:
    """计算排序JSON对象的稳定绑定哈希。"""

    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bound_entry(
    scenario: GraphSemanticTopologyScenario,
    order_index: int,
) -> Dict[str, object]:
    """构造一个与场景配置和分区哈希绑定的批次项目。"""

    entry: Dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "arm": scenario.arm,
        "seed": scenario.seed,
        "partition_hash": scenario.partition_hash,
        "config": str(scenario.config_path),
        "config_sha256": _file_sha256(scenario.config_path),
        "order_index": int(order_index),
        "status": "pending",
        "attempt_count": 0,
        "attempts": [],
        "result_dir": None,
        "contract_file": None,
        "error": None,
    }
    entry["binding_hash"] = _stable_hash(_binding_fields(entry))
    return entry


def _empty_batch_payload() -> Dict[str, object]:
    """创建包含六个新增训练项目的初始批次对象。"""

    scenarios = scenarios_from_contract()
    now = _timestamp()
    return {
        "suite": SUITE_NAME,
        "planned_new_run_count": 6,
        "status": "pending",
        "pilot_gates": {arm: "pending" for arm in ARM_ORDER},
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "run_order": [item.scenario_id for item in scenarios],
        "entries": [
            _bound_entry(item, index)
            for index, item in enumerate(scenarios, start=1)
        ],
    }


def create_batch_manifest(result_root: Optional[Path] = None) -> Path:
    """在隔离结果目录创建一个不会覆盖旧结果的批次清单。"""

    root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / RESULT_ROOT).resolve()
    )
    name = "graph_semantic_topology_extension_batch_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    path = root / name / BATCH_FILE_NAME
    write_json_report(_empty_batch_payload(), path)
    return path


def _validate_entry_binding(entry: Mapping[str, object]) -> None:
    """校验恢复项目的场景身份、配置内容和稳定绑定。"""

    scenario = scenario_by_id(str(entry.get("scenario_id", "")))
    config_path = Path(str(entry.get("config", ""))).resolve()
    if config_path != scenario.config_path or not config_path.is_file():
        raise ValueError("批次配置路径与正式场景不一致")
    if _file_sha256(config_path) != str(entry.get("config_sha256", "")):
        raise ValueError("批次创建后正式配置已被修改")
    if _stable_hash(_binding_fields(entry)) != entry.get("binding_hash"):
        raise ValueError("批次场景绑定字段已被修改")


def _load_batch_manifest(path: Path) -> Dict[str, object]:
    """读取并校验六实验批次清单的位置、顺序和绑定。"""

    resolved = Path(path).expanduser().resolve()
    root = (PACKAGE_DIR / RESULT_ROOT).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("批次必须位于拓扑扩展结果目录") from error
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = scenarios_from_contract()
    expected_order = [item.scenario_id for item in scenarios]
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("批次套件身份不一致")
    if payload.get("run_order") != expected_order:
        raise ValueError("六次运行顺序已被修改")
    if int(payload.get("planned_new_run_count", -1)) != 6:
        raise ValueError("批次新增训练数量必须为6")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError("批次必须包含六个训练项目")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("批次项目必须是对象")
        _validate_entry_binding(entry)
    return payload


def _save_batch_manifest(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    """更新时间并写回批次清单。"""

    mutable = dict(payload)
    mutable["updated_at"] = _timestamp()
    write_json_report(mutable, path)


def _result_directories(
    scenario: GraphSemanticTopologyScenario,
) -> Tuple[Path, ...]:
    """返回与正式运行名前缀匹配的既有结果目录。"""

    config = load_flat_config(scenario.config_path)
    root = (PACKAGE_DIR / str(config["result_root"])).resolve()
    run_name = str(config["run_name"])
    return tuple(
        sorted(
            (path.resolve() for path in root.glob(run_name + "_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
        )
    )


def _new_result_directories(
    scenario: GraphSemanticTopologyScenario,
    before: Sequence[Path],
) -> Tuple[Path, ...]:
    """返回一次训练后相对运行前唯一新增的结果目录集合。"""

    existing = {Path(path).resolve() for path in before}
    return tuple(
        path for path in _result_directories(scenario) if path not in existing
    )


def _run_scenario(scenario: GraphSemanticTopologyScenario) -> Path:
    """调用统一训练入口并返回本次唯一新增结果目录。"""

    before = _result_directories(scenario)
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v5.run_federated_transe",
        "--cf",
        str(scenario.config_path),
    ]
    try:
        subprocess.run(command, cwd=str(PACKAGE_DIR.parent), check=True)
    except subprocess.CalledProcessError as error:
        raise TopologyExtensionRunError(
            "{}训练失败，退出码{}".format(
                scenario.scenario_id, error.returncode
            ),
            created_results=_new_result_directories(scenario, before),
        ) from error
    created = _new_result_directories(scenario, before)
    if len(created) != 1:
        raise TopologyExtensionRunError(
            "{}期望新增1个结果目录，实际为{}".format(
                scenario.scenario_id, len(created)
            ),
            created_results=created,
        )
    return created[0]


def _record_failure(
    manifest_path: Path,
    payload: Dict[str, object],
    entry: Dict[str, object],
    attempt: Dict[str, object],
    error: BaseException,
    created_results: Sequence[Path] = (),
) -> int:
    """记录失败状态、错误和可能产生的目录后停止批次。"""

    attempt.update(
        {
            "status": "failed",
            "completed_at": _timestamp(),
            "error": "{}: {}".format(type(error).__name__, error),
            "created_results": [str(path) for path in created_results],
        }
    )
    entry.update({"status": "failed", "error": attempt["error"]})
    payload["status"] = "failed"
    _save_batch_manifest(manifest_path, payload)
    print("实验失败并停止：{}".format(entry["scenario_id"]))
    print(
        "恢复命令：python -m "
        "HFLSnF_KG_v5.run_graph_semantic_topology_extension "
        'formal6 --resume "{}"'.format(manifest_path)
    )
    return 1


def run_training_batch(manifest_path: Path) -> int:
    """按固定顺序训练六场景，并对每个组执行seed42先行门禁。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    payload["status"] = "running"
    _save_batch_manifest(resolved, payload)
    for entry in payload["entries"]:
        if entry.get("status") == "passed":
            print("跳过已通过实验：{}".format(entry["scenario_id"]))
            continue
        scenario = scenario_by_id(str(entry["scenario_id"]))
        if scenario.seed != 42 and payload["pilot_gates"].get(
            scenario.arm
        ) != "passed":
            raise RuntimeError(
                "{}的seed42结果合同未通过".format(scenario.arm)
            )
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
        entry.update(
            {
                "status": "running",
                "attempt_count": len(attempts),
                "error": None,
            }
        )
        _save_batch_manifest(resolved, payload)
        print("开始V5图语义拓扑实验：{}".format(scenario.scenario_id))
        try:
            result_dir = _run_scenario(scenario)
            attempt["result_dir"] = str(result_dir)
            attempt["created_results"] = [str(result_dir)]
            report = validate_result(result_dir, scenario)
            contract_path = result_dir / RESULT_CONTRACT_FILE_NAME
            write_json_report(report, contract_path)
            attempt["contract_file"] = str(contract_path)
            if report["status"] != "passed":
                raise RuntimeError("结果合同校验未通过")
        except KeyboardInterrupt as error:
            return _record_failure(
                resolved, payload, entry, attempt, error
            )
        except Exception as error:
            created = (
                error.created_results
                if isinstance(error, TopologyExtensionRunError)
                else tuple(Path(path) for path in attempt["created_results"])
            )
            return _record_failure(
                resolved, payload, entry, attempt, error, created
            )
        attempt.update({"status": "passed", "completed_at": _timestamp()})
        entry.update(
            {
                "status": "passed",
                "result_dir": attempt["result_dir"],
                "contract_file": attempt["contract_file"],
                "error": None,
            }
        )
        if scenario.seed == 42:
            payload["pilot_gates"][scenario.arm] = "passed"
        _save_batch_manifest(resolved, payload)
        print("{}结果合同：passed".format(scenario.scenario_id))
    payload["status"] = "passed"
    payload["completed_at"] = _timestamp()
    _save_batch_manifest(resolved, payload)
    print("V5图语义拓扑六次训练全部完成")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """执行配置校验或可恢复的六次正式训练。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.action == "formal6" and args.skip_partition_recompute:
        parser.error("formal6禁止--skip-partition-recompute")
    recompute = not bool(args.skip_partition_recompute)
    report = validate_configs(recompute_partitions=recompute)
    print("V5图语义拓扑扩展配置合同：{}".format(report["status"]))
    if report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        if args.resume is not None:
            parser.error("validate不能使用--resume")
        return
    manifest = (
        Path(args.resume).expanduser().resolve()
        if args.resume is not None
        else create_batch_manifest()
    )
    print("批次清单：{}".format(manifest))
    if run_training_batch(manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
