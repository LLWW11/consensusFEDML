"""校验并运行V5图语义FedAvg三拓扑九实验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    _load_json,
    _load_metrics,
    load_flat_config,
    write_json_report,
)
from .tasks.kge.graph_semantic_fedavg_comparison import (
    ARM_ORDER,
    RESULT_ROOT,
    SEEDS,
    SUITE_NAME,
    GraphSemanticFedAvgScenario,
    scenario_by_id,
    scenarios_from_contract,
    validate_configs,
    validate_result,
)


BATCH_FILE_NAME = "batch_summary.json"
RESULT_CONTRACT_FILE_NAME = "fedavg_formal150_contract.json"
OFFICIAL_MANIFEST_FILE_NAME = "official9_summary.json"
REPORT_JSON_FILE_NAME = "fedavg_three_topology_summary.json"
REPORT_CSV_FILE_NAME = "fedavg_three_topology_units.csv"
REPORT_MARKDOWN_FILE_NAME = "图语义FedAvg三拓扑九实验报告.md"


class FedAvgComparisonRunError(RuntimeError):
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
    """创建九实验校验、训练、官方测试与报告参数。"""

    parser = argparse.ArgumentParser(
        description="运行V5图语义FedAvg三拓扑九实验"
    )
    parser.add_argument(
        "action",
        choices=("validate", "formal9", "official9", "report"),
        help="选择配置校验、九次训练、九次官方测试或报告",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="formal9恢复使用的batch_summary.json",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="official9和report使用的batch_summary.json",
    )
    parser.add_argument(
        "--skip-partition-recompute",
        action="store_true",
        help="仅供快速静态校验；formal9禁止使用",
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


def _stable_hash(payload: Mapping[str, object]) -> str:
    """计算排序JSON对象的稳定绑定哈希。"""

    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding_fields(entry: Mapping[str, object]) -> Dict[str, object]:
    """提取恢复时不得改变的场景与配置身份字段。"""

    return {
        key: entry.get(key)
        for key in (
            "scenario_id",
            "arm",
            "seed",
            "partition_hash",
            "topology_hash",
            "config",
            "config_sha256",
            "source_config",
            "source_config_sha256",
            "order_index",
        )
    }


def _bound_entry(
    scenario: GraphSemanticFedAvgScenario,
    order_index: int,
) -> Dict[str, object]:
    """构造一个与配置、分区及拓扑绑定的批次项目。"""

    entry: Dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "arm": scenario.arm,
        "seed": scenario.seed,
        "partition_hash": scenario.partition_hash,
        "topology_hash": scenario.topology_contract.schedule_hash,
        "config": str(scenario.config_path),
        "config_sha256": _file_sha256(scenario.config_path),
        "source_config": str(scenario.source_config_path),
        "source_config_sha256": _file_sha256(scenario.source_config_path),
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
    """创建包含九个正式训练项目的初始批次对象。"""

    scenarios = scenarios_from_contract()
    now = _timestamp()
    return {
        "suite": SUITE_NAME,
        "planned_new_run_count": 9,
        "status": "pending",
        "pilot_seed42_gate": "pending",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "run_order": [scenario.scenario_id for scenario in scenarios],
        "entries": [
            _bound_entry(scenario, index)
            for index, scenario in enumerate(scenarios, start=1)
        ],
        "official_evaluation_manifest": None,
        "report_json": None,
        "report_csv": None,
        "report_markdown": None,
    }


def create_batch_manifest(result_root: Optional[Path] = None) -> Path:
    """在隔离结果目录创建一个不会覆盖旧结果的批次清单。"""

    root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / RESULT_ROOT).resolve()
    )
    name = "graph_semantic_fedavg_batch_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    path = root / name / BATCH_FILE_NAME
    write_json_report(_empty_batch_payload(), path)
    return path


def _validate_entry_binding(entry: Mapping[str, object]) -> None:
    """校验恢复项目的配置、来源配置与稳定绑定。"""

    scenario = scenario_by_id(str(entry.get("scenario_id", "")))
    config_path = Path(str(entry.get("config", ""))).resolve()
    source_path = Path(str(entry.get("source_config", ""))).resolve()
    if config_path != scenario.config_path or not config_path.is_file():
        raise ValueError("批次配置路径与正式场景不一致")
    if source_path != scenario.source_config_path or not source_path.is_file():
        raise ValueError("批次来源配置路径与正式场景不一致")
    if _file_sha256(config_path) != str(entry.get("config_sha256", "")):
        raise ValueError("批次创建后FedAvg正式配置已被修改")
    if _file_sha256(source_path) != str(
        entry.get("source_config_sha256", "")
    ):
        raise ValueError("批次创建后FedAdam来源配置已被修改")
    if entry.get("partition_hash") != scenario.partition_hash:
        raise ValueError("批次分区哈希与正式场景不一致")
    if entry.get("topology_hash") != scenario.topology_contract.schedule_hash:
        raise ValueError("批次拓扑哈希与正式场景不一致")
    if _stable_hash(_binding_fields(entry)) != entry.get("binding_hash"):
        raise ValueError("批次场景绑定字段已被修改")


def _load_batch_manifest(path: Path) -> Dict[str, object]:
    """读取并校验九实验批次清单的位置、顺序和绑定。"""

    resolved = Path(path).expanduser().resolve()
    root = (PACKAGE_DIR / RESULT_ROOT).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("批次必须位于FedAvg九实验结果目录") from error
    payload = _load_json(resolved)
    scenarios = scenarios_from_contract()
    expected_order = [scenario.scenario_id for scenario in scenarios]
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("批次套件身份不一致")
    if payload.get("run_order") != expected_order:
        raise ValueError("九次运行顺序已被修改")
    if int(payload.get("planned_new_run_count", -1)) != 9:
        raise ValueError("批次训练数量必须为9")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 9:
        raise ValueError("批次必须包含九个训练项目")
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
    scenario: GraphSemanticFedAvgScenario,
) -> Tuple[Path, ...]:
    """返回与正式运行名前缀匹配的既有结果目录。"""

    config = load_flat_config(scenario.config_path)
    root = (PACKAGE_DIR / str(config["result_root"])).resolve()
    run_name = str(config["run_name"])
    return tuple(
        sorted(
            (
                path.resolve()
                for path in root.glob(run_name + "_*")
                if path.is_dir()
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
    )


def _new_result_directories(
    scenario: GraphSemanticFedAvgScenario,
    before: Sequence[Path],
) -> Tuple[Path, ...]:
    """返回一次训练后相对运行前唯一新增的结果目录集合。"""

    existing = {Path(path).resolve() for path in before}
    return tuple(
        path for path in _result_directories(scenario) if path not in existing
    )


def _run_scenario(scenario: GraphSemanticFedAvgScenario) -> Path:
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
        raise FedAvgComparisonRunError(
            "{}训练失败，退出码{}".format(
                scenario.scenario_id, error.returncode
            ),
            created_results=_new_result_directories(scenario, before),
        ) from error
    created = _new_result_directories(scenario, before)
    if len(created) != 1:
        raise FedAvgComparisonRunError(
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
        "HFLSnF_KG_v5.run_graph_semantic_fedavg_comparison "
        'formal9 --resume "{}"'.format(manifest_path)
    )
    return 1


def _update_seed42_gate(payload: Dict[str, object]) -> None:
    """在三个seed42场景全部通过后开启后续种子门禁。"""

    seed42_entries = [
        entry for entry in payload["entries"] if int(entry["seed"]) == 42
    ]
    if len(seed42_entries) == len(ARM_ORDER) and all(
        entry.get("status") == "passed" for entry in seed42_entries
    ):
        payload["pilot_seed42_gate"] = "passed"


def run_training_batch(manifest_path: Path) -> int:
    """按固定九格顺序训练，并执行三拓扑seed42先行门禁。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    payload["status"] = "running"
    _update_seed42_gate(payload)
    _save_batch_manifest(resolved, payload)
    for entry in payload["entries"]:
        if entry.get("status") == "passed":
            print("跳过已通过实验：{}".format(entry["scenario_id"]))
            continue
        scenario = scenario_by_id(str(entry["scenario_id"]))
        if scenario.seed != 42 and payload.get(
            "pilot_seed42_gate"
        ) != "passed":
            raise RuntimeError("三个seed42结果合同未全部通过")
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
        print("开始图语义FedAvg实验：{}".format(scenario.scenario_id))
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
            return _record_failure(resolved, payload, entry, attempt, error)
        except Exception as error:
            created = (
                error.created_results
                if isinstance(error, FedAvgComparisonRunError)
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
        _update_seed42_gate(payload)
        _save_batch_manifest(resolved, payload)
        print("{}结果合同：passed".format(scenario.scenario_id))
    payload["status"] = "passed"
    payload["completed_at"] = _timestamp()
    _save_batch_manifest(resolved, payload)
    print("图语义FedAvg九次训练全部完成")
    return 0


def _official_units(
    payload: Mapping[str, object],
) -> List[Dict[str, object]]:
    """从已通过批次构造九个最佳检查点官方测试单元。"""

    if payload.get("status") != "passed":
        raise ValueError("九组训练未全部通过，禁止官方测试")
    units = []
    for entry in payload["entries"]:
        if entry.get("status") != "passed":
            raise ValueError("存在未通过的训练项目")
        units.append(
            {
                "scenario_id": str(entry["scenario_id"]),
                "arm": str(entry["arm"]),
                "seed": int(entry["seed"]),
                "result_dir": str(entry["result_dir"]),
            }
        )
    return units


def _validate_official_summary(
    unit: Mapping[str, object],
    summary_path: Path,
) -> None:
    """校验官方测试完整性及其最佳验证检查点绑定。"""

    official = _load_json(summary_path)
    if official.get("status") != "passed" or not official.get(
        "full_official_test"
    ):
        raise RuntimeError("完整官方测试合同未通过")
    result_dir = Path(str(unit["result_dir"])).resolve()
    training = _load_json(result_dir / "summary.json")
    checkpoint = Path(str(official.get("checkpoint_path", ""))).resolve()
    if checkpoint != (result_dir / "model_best.pt").resolve():
        raise RuntimeError("官方测试未绑定验证最佳检查点")
    if not checkpoint.is_file():
        raise FileNotFoundError("官方测试绑定的验证最佳检查点不存在")
    if str(official.get("checkpoint_sha256", "")) != _file_sha256(
        checkpoint
    ):
        raise RuntimeError("官方测试最佳检查点哈希不一致")
    if int(official.get("best_round", -1)) != int(
        training.get("best_round", -2)
    ):
        raise RuntimeError("官方测试最佳轮次与训练汇总不一致")
    expected_mrr = float(training["best_validation_mrr_during_training"])
    actual_mrr = float(
        official.get("best_validation_mrr", float("nan"))
    )
    if not math.isfinite(actual_mrr) or not math.isfinite(expected_mrr):
        raise RuntimeError("官方测试最佳验证MRR不是有限数")
    if abs(actual_mrr - expected_mrr) > 1e-12:
        raise RuntimeError("官方测试最佳验证MRR与训练汇总不一致")
    for block_name in ("head_metrics", "tail_metrics", "combined_metrics"):
        block = official.get(block_name)
        if not isinstance(block, dict):
            raise RuntimeError("官方测试缺少{}".format(block_name))
        for field in (
            "mrr", "mean_rank", "hits_at_1", "hits_at_3", "hits_at_10"
        ):
            value = float(block.get(field, float("nan")))
            if not math.isfinite(value):
                raise RuntimeError(
                    "官方测试{}.{}不是有限数".format(block_name, field)
                )


def run_official9(manifest_path: Path) -> int:
    """顺序执行九个验证最佳检查点的完整官方测试。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    units = _official_units(payload)
    existing = payload.get("official_evaluation_manifest")
    if existing:
        official_path = Path(str(existing)).resolve()
        official = _load_json(official_path)
    else:
        root = resolved.parent / "official9"
        official_path = root / OFFICIAL_MANIFEST_FILE_NAME
        official = {
            "suite": SUITE_NAME,
            "status": "pending",
            "started_at": _timestamp(),
            "updated_at": _timestamp(),
            "completed_at": None,
            "units": [
                {
                    **unit,
                    "status": "pending",
                    "output_dir": str(root / str(unit["scenario_id"])),
                    "summary_file": None,
                    "error": None,
                }
                for unit in units
            ],
        }
        write_json_report(official, official_path)
        payload["official_evaluation_manifest"] = str(official_path)
        _save_batch_manifest(resolved, payload)
    actual_units = official.get("units")
    if not isinstance(actual_units, list) or len(actual_units) != 9:
        raise ValueError("官方测试清单必须包含九个单元")
    identity_fields = ("scenario_id", "arm", "seed", "result_dir")
    actual_identity = [
        {field: unit.get(field) for field in identity_fields}
        for unit in actual_units
    ]
    expected_identity = [
        {field: unit.get(field) for field in identity_fields} for unit in units
    ]
    if actual_identity != expected_identity:
        raise ValueError("官方测试清单身份已被修改")
    official["status"] = "running"
    for unit in actual_units:
        if unit.get("status") == "passed":
            summary_file = Path(str(unit.get("summary_file", ""))).resolve()
            _validate_official_summary(unit, summary_file)
            print("跳过已通过官方测试：{}".format(unit["scenario_id"]))
            continue
        output_dir = Path(str(unit["output_dir"])).resolve()
        command = [
            sys.executable,
            "-m",
            "HFLSnF_KG_v5.run_best_checkpoint_official_evaluation",
            "--result-dir",
            str(unit["result_dir"]),
            "--output-dir",
            str(output_dir),
            "--using-gpu",
            "--require-cuda",
            "--gpu-id",
            "0",
            "--query-batch-size",
            "64",
            "--candidate-batch-size",
            "8192",
        ]
        try:
            subprocess.run(command, cwd=str(PACKAGE_DIR.parent), check=True)
            summary_path = output_dir / "official_evaluation_summary.json"
            _validate_official_summary(unit, summary_path)
            unit.update(
                {
                    "status": "passed",
                    "summary_file": str(summary_path),
                    "error": None,
                }
            )
        except Exception as error:
            unit.update(
                {
                    "status": "failed",
                    "error": "{}: {}".format(type(error).__name__, error),
                }
            )
            official["status"] = "failed"
            official["updated_at"] = _timestamp()
            write_json_report(official, official_path)
            return 1
        official["updated_at"] = _timestamp()
        write_json_report(official, official_path)
    official.update(
        {
            "status": "passed",
            "completed_at": _timestamp(),
            "updated_at": _timestamp(),
        }
    )
    write_json_report(official, official_path)
    return 0


def _unit_analysis(unit: Mapping[str, object]) -> Dict[str, object]:
    """汇总一个FedAvg场景的训练、官方测试与审计标识。"""

    result_dir = Path(str(unit["result_dir"])).resolve()
    training = _load_json(result_dir / "summary.json")
    official = _load_json(Path(str(unit["summary_file"])).resolve())
    rows = _load_metrics(result_dir / "metrics.csv")
    combined = official["combined_metrics"]
    return {
        "scenario_id": str(unit["scenario_id"]),
        "arm": str(unit["arm"]),
        "seed": int(unit["seed"]),
        "best_round": int(official["best_round"]),
        "best_validation_mrr": float(official["best_validation_mrr"]),
        "test_mrr": float(combined["mrr"]),
        "test_mean_rank": float(combined["mean_rank"]),
        "test_hits_at_1": float(combined["hits_at_1"]),
        "test_hits_at_3": float(combined["hits_at_3"]),
        "test_hits_at_10": float(combined["hits_at_10"]),
        "total_round_seconds": float(
            sum(float(row["round_seconds"]) for row in rows)
        ),
        "mean_round_seconds": float(
            statistics.mean(float(row["round_seconds"]) for row in rows)
        ),
        "partition_hash": str(training["partition_hash"]),
        "topology_schedule_hash": str(training["topology_schedule_hash"]),
        "initial_model_hash": str(training["initial_model_hash"]),
        "checkpoint_sha256": str(official["checkpoint_sha256"]),
        "result_dir": str(result_dir),
    }


def _mean_and_sample_std(values: Sequence[float]) -> Dict[str, float]:
    """返回非空序列的均值与样本标准差。"""

    if not values:
        raise ValueError("统计输入不能为空")
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def _arm_aggregates(
    units: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """按拓扑臂汇总三种子的均值与样本标准差。"""

    metric_names = (
        "best_validation_mrr",
        "test_mrr",
        "test_mean_rank",
        "test_hits_at_1",
        "test_hits_at_3",
        "test_hits_at_10",
        "total_round_seconds",
        "mean_round_seconds",
    )
    aggregates: Dict[str, object] = {}
    for arm in ARM_ORDER:
        arm_units = [unit for unit in units if unit["arm"] == arm]
        if [int(unit["seed"]) for unit in arm_units] != list(SEEDS):
            raise ValueError("{}缺少固定三种子结果".format(arm))
        aggregates[arm] = {
            metric: _mean_and_sample_std(
                [float(unit[metric]) for unit in arm_units]
            )
            for metric in metric_names
        }
    return aggregates


def _paired_differences(
    units: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """计算三组拓扑在同种子下的官方测试MRR配对差。"""

    by_key = {
        (str(unit["arm"]), int(unit["seed"])): unit for unit in units
    }
    pairs = (
        ("hflsnf", "hflnosnf", "HFLSnF-HFLnoSnF"),
        ("hflsnf", "flnosnf", "HFLSnF-FLnoSnF"),
        ("hflnosnf", "flnosnf", "HFLnoSnF-FLnoSnF"),
    )
    rows: List[Dict[str, object]] = []
    for left, right, label in pairs:
        deltas = []
        for seed in SEEDS:
            left_mrr = float(by_key[(left, seed)]["test_mrr"])
            right_mrr = float(by_key[(right, seed)]["test_mrr"])
            delta = left_mrr - right_mrr
            deltas.append(delta)
            rows.append(
                {
                    "comparison": label,
                    "seed": seed,
                    "left_test_mrr": left_mrr,
                    "right_test_mrr": right_mrr,
                    "test_mrr_delta": delta,
                }
            )
        summary = _mean_and_sample_std(deltas)
        rows.append(
            {
                "comparison": label,
                "seed": "aggregate",
                "left_test_mrr": None,
                "right_test_mrr": None,
                "test_mrr_delta": summary["mean"],
                "test_mrr_delta_sample_std": summary["sample_std"],
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """以UTF-8 BOM写入字段并集稳定排序的CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(set(row.keys()) for row in rows)))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(
    units: Sequence[Mapping[str, object]],
    aggregates: Mapping[str, object],
    paired: Sequence[Mapping[str, object]],
) -> str:
    """构造只包含本批九组FedAvg结果的简体中文报告。"""

    labels = {
        "hflsnf": "HFLSnF",
        "hflnosnf": "HFLnoSnF",
        "flnosnf": "FLnoSnF",
    }
    lines = [
        "# V5图语义FedAvg三拓扑九实验报告",
        "",
        "## 九组官方结果",
        "",
        "| 组别 | 种子 | 最佳轮次 | 验证MRR | 测试MRR | MR | Hits@1 | Hits@3 | Hits@10 | 总训练秒数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for unit in units:
        lines.append(
            "| {arm_label} | {seed} | {best_round} | {best_validation_mrr:.6f} | "
            "{test_mrr:.6f} | {test_mean_rank:.3f} | {test_hits_at_1:.6f} | "
            "{test_hits_at_3:.6f} | {test_hits_at_10:.6f} | "
            "{total_round_seconds:.2f} |".format(
                arm_label=labels[str(unit["arm"])], **unit
            )
        )
    lines.extend(
        [
            "",
            "## 三种子汇总",
            "",
            "| 组别 | 测试MRR均值 | 测试MRR样本标准差 | Hits@10均值 | 总训练秒数均值 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in ARM_ORDER:
        item = aggregates[arm]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.2f} |".format(
                labels[arm],
                item["test_mrr"]["mean"],
                item["test_mrr"]["sample_std"],
                item["test_hits_at_10"]["mean"],
                item["total_round_seconds"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## 同种子MRR配对差",
            "",
            "| 比较 | 种子 | MRR差值 | 样本标准差 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for item in paired:
        std = item.get("test_mrr_delta_sample_std")
        lines.append(
            "| {} | {} | {:.6f} | {} |".format(
                item["comparison"],
                item["seed"],
                item["test_mrr_delta"],
                "{:.6f}".format(std) if std is not None else "—",
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本报告只汇总本批9个服务器端FedAvg正式实验。现有FedAdam结果没有重新训练、改写或混入本批统计。",
            "",
        ]
    )
    return "\n".join(lines)


def run_report(manifest_path: Path) -> int:
    """生成九组明细、三臂汇总和同种子配对报告。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    official_path = payload.get("official_evaluation_manifest")
    if not official_path:
        raise ValueError("尚未执行official9")
    official = _load_json(Path(str(official_path)).resolve())
    if official.get("status") != "passed":
        raise ValueError("official9尚未全部通过")
    units = [_unit_analysis(unit) for unit in official["units"]]
    aggregates = _arm_aggregates(units)
    paired = _paired_differences(units)
    report = {
        "suite": SUITE_NAME,
        "status": "passed",
        "experiment_unit_count": len(units),
        "units": units,
        "arm_aggregates": aggregates,
        "paired_test_mrr_differences": paired,
        "interpretation_boundary": (
            "本报告只汇总服务器端FedAvg九组，不混入冻结FedAdam结果。"
        ),
    }
    report_dir = resolved.parent / "analysis"
    json_path = report_dir / REPORT_JSON_FILE_NAME
    csv_path = report_dir / REPORT_CSV_FILE_NAME
    markdown_path = report_dir / REPORT_MARKDOWN_FILE_NAME
    write_json_report(report, json_path)
    _write_csv(csv_path, units)
    markdown_path.write_text(
        _markdown_report(units, aggregates, paired), encoding="utf-8"
    )
    payload["report_json"] = str(json_path)
    payload["report_csv"] = str(csv_path)
    payload["report_markdown"] = str(markdown_path)
    _save_batch_manifest(resolved, payload)
    print("FedAvg九实验报告：{}".format(markdown_path))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """执行指定阶段，并在合同失败时返回非零退出码。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.action == "formal9" and args.skip_partition_recompute:
        parser.error("formal9禁止--skip-partition-recompute")
    recompute = not bool(args.skip_partition_recompute)
    config_report = validate_configs(recompute_partitions=recompute)
    print("V5图语义FedAvg九配置合同：{}".format(config_report["status"]))
    if config_report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        if args.resume is not None or args.batch is not None:
            parser.error("validate不能使用--resume或--batch")
        return
    if args.action == "formal9":
        if args.batch is not None:
            parser.error("formal9不能使用--batch")
        manifest = (
            Path(args.resume).expanduser().resolve()
            if args.resume is not None
            else create_batch_manifest()
        )
        print("批次清单：{}".format(manifest))
        if run_training_batch(manifest):
            raise SystemExit(1)
        return
    if args.resume is not None or args.batch is None:
        parser.error("official9和report必须使用--batch且不能使用--resume")
    if args.action == "official9":
        if run_official9(args.batch):
            raise SystemExit(1)
        return
    if run_report(args.batch):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
