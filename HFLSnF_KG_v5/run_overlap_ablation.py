"""校验、训练、官方测试并汇总V4实体重叠率正式消融实验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from .tasks.kge.overlap_ablation import (
    BASELINE_INITIAL_MODEL_HASHES,
    BASELINE_RESULT_DIRS,
    CALIBRATION_CONTRACT_PATH,
    LEVELS,
    ROUND_COUNT,
    SEEDS,
    SUITE_NAME,
    OverlapScenario,
    communication_statistics,
    convergence_statistics,
    scenario_by_id,
    scenarios_from_contract,
    validate_configs,
    validate_result,
)
from .tasks.kge import (
    load_fb15k237,
    partition_train_triples_by_head,
)


BATCH_FILE_NAME = "batch_summary.json"
RESULT_CONTRACT_FILE_NAME = "overlap_ablation_formal150_contract.json"
OFFICIAL_MANIFEST_FILE_NAME = "official12_summary.json"
REPORT_JSON_FILE_NAME = "overlap_ablation_summary.json"
REPORT_MARKDOWN_FILE_NAME = "重叠率消融实验报告.md"


class OverlapRunError(RuntimeError):
    """表示训练子进程失败或没有产生唯一结果目录。"""

    def __init__(
        self,
        message: str,
        created_results: Sequence[Path] = (),
    ) -> None:
        """保存错误信息和当前尝试可能创建的结果目录。"""

        super().__init__(message)
        self.created_results = tuple(
            Path(path).resolve() for path in created_results
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """创建配置校验、九组训练、十二格测试和报告入口。"""

    parser = argparse.ArgumentParser(
        description="运行HFLSnF_KG_v5实体重叠率正式消融实验"
    )
    parser.add_argument(
        "action",
        choices=("validate", "formal150", "official12", "report"),
        help="选择只读校验、九组训练、十二格官方测试或报告生成",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="formal150使用的既有batch_summary.json",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="official12和report必须指定的batch_summary.json",
    )
    parser.add_argument(
        "--skip-partition-recompute",
        action="store_true",
        help="仅供快速开发检查；formal150禁止跳过九个正式分区复算",
    )
    return parser


def _timestamp() -> str:
    """返回包含时区的秒级审计时间。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    """分块计算配置或合同文件的SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_payload_hash(payload: Mapping[str, object]) -> str:
    """计算排序JSON对象的稳定SHA-256绑定值。"""

    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _entry_binding_fields(entry: Mapping[str, object]) -> Dict[str, object]:
    """提取恢复时不允许变化的训练项目字段。"""

    keys = (
        "scenario_id",
        "level",
        "seed",
        "target_entity_overlap",
        "partition_hash",
        "config",
        "config_sha256",
        "order_index",
    )
    return {key: entry.get(key) for key in keys}


def _bound_entry(
    scenario: OverlapScenario,
    order_index: int,
) -> Dict[str, object]:
    """构造一个与正式配置和分区哈希绑定的待运行项目。"""

    entry: Dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "level": scenario.level,
        "seed": scenario.seed,
        "target_entity_overlap": scenario.target_entity_overlap,
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
    entry["binding_hash"] = _stable_payload_hash(
        _entry_binding_fields(entry)
    )
    return entry


def _empty_batch_payload() -> Dict[str, object]:
    """构造含三项原始对照和九项新训练的批次清单。"""

    scenarios = scenarios_from_contract()
    now = _timestamp()
    return {
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "planned_new_run_count": len(scenarios),
        "total_experiment_unit_count": 12,
        "status": "pending",
        "pilot_seed42_gate": "pending",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "calibration_contract": str(CALIBRATION_CONTRACT_PATH),
        "calibration_contract_sha256": _file_sha256(
            CALIBRATION_CONTRACT_PATH
        ),
        "baseline_units": {
            str(seed): {
                "scenario_id": "original_seed{}".format(seed),
                "result_dir": str(BASELINE_RESULT_DIRS[seed]),
                "initial_model_hash": BASELINE_INITIAL_MODEL_HASHES[seed],
            }
            for seed in SEEDS
        },
        "run_order": [item.scenario_id for item in scenarios],
        "entries": [
            _bound_entry(scenario, index)
            for index, scenario in enumerate(scenarios, start=1)
        ],
        "official_evaluation_manifest": None,
        "report_json": None,
        "report_markdown": None,
    }


def create_batch_manifest(result_root: Optional[Path] = None) -> Path:
    """创建不会覆盖旧结果的V4重叠率训练批次。"""

    root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results").resolve()
    )
    batch_name = "overlap_ablation_batch_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    manifest_path = root / batch_name / BATCH_FILE_NAME
    write_json_report(_empty_batch_payload(), manifest_path)
    return manifest_path


def _validate_entry_binding(entry: Mapping[str, object]) -> None:
    """校验恢复清单中的场景身份、配置内容和稳定绑定哈希。"""

    scenario = scenario_by_id(str(entry.get("scenario_id", "")))
    config_path = Path(str(entry.get("config", ""))).resolve()
    if config_path != scenario.config_path or not config_path.is_file():
        raise ValueError("批次配置路径与正式场景不一致")
    if _file_sha256(config_path) != str(entry.get("config_sha256", "")):
        raise ValueError("批次配置已被修改：{}".format(config_path))
    if _stable_payload_hash(_entry_binding_fields(entry)) != entry.get(
        "binding_hash"
    ):
        raise ValueError("批次场景绑定字段已被修改")


def _load_batch_manifest(path: Path) -> Dict[str, object]:
    """读取并完整校验批次位置、合同、顺序和配置绑定。"""

    resolved = Path(path).expanduser().resolve()
    results_root = (PACKAGE_DIR / "results").resolve()
    try:
        resolved.relative_to(results_root)
    except ValueError as error:
        raise ValueError("批次清单必须位于V4 results目录") from error
    payload = _load_json(resolved)
    scenarios = scenarios_from_contract()
    expected_order = [item.scenario_id for item in scenarios]
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("批次套件与V4重叠率实验不一致")
    if payload.get("run_order") != expected_order:
        raise ValueError("批次九组运行顺序已被修改")
    if int(payload.get("planned_new_run_count", -1)) != 9 or int(
        payload.get("total_experiment_unit_count", -1)
    ) != 12:
        raise ValueError("批次实验单元数量合同已被修改")
    if _file_sha256(CALIBRATION_CONTRACT_PATH) != payload.get(
        "calibration_contract_sha256"
    ):
        raise ValueError("正式校准合同已被修改")
    expected_baselines = {
        str(seed): {
            "scenario_id": "original_seed{}".format(seed),
            "result_dir": str(BASELINE_RESULT_DIRS[seed]),
            "initial_model_hash": BASELINE_INITIAL_MODEL_HASHES[seed],
        }
        for seed in SEEDS
    }
    if payload.get("baseline_units") != expected_baselines:
        raise ValueError("批次原始对照路径或初始化哈希已被修改")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 9:
        raise ValueError("批次必须包含九个新训练项目")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("批次项目必须是对象")
        _validate_entry_binding(entry)
    return payload


def _save_batch_manifest(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    """更新时间并原子写回批次清单。"""

    mutable = dict(payload)
    mutable["updated_at"] = _timestamp()
    write_json_report(mutable, path)


def _result_directories(run_name: str) -> Tuple[Path, ...]:
    """返回当前与运行名前缀匹配的结果目录。"""

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
    """通过运行前后目录集合差找到本次新建结果。"""

    existing = {Path(path).resolve() for path in before}
    return tuple(
        path for path in _result_directories(run_name) if path not in existing
    )


def _run_config(scenario: OverlapScenario) -> Path:
    """调用V4统一训练入口并返回唯一新建结果目录。"""

    config = load_flat_config(scenario.config_path)
    run_name = str(config["run_name"])
    before = _result_directories(run_name)
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v5.run_federated_transe",
        "--cf",
        str(scenario.config_path),
    ]
    try:
        # 子进程继承当前CUDA环境，训练日志直接显示在调用终端。
        subprocess.run(command, cwd=str(PACKAGE_DIR.parent), check=True)
    except subprocess.CalledProcessError as error:
        created = _new_result_directories(run_name, before)
        raise OverlapRunError(
            "{}训练失败，退出码{}".format(
                scenario.scenario_id, error.returncode
            ),
            created_results=created,
        ) from error
    created = _new_result_directories(run_name, before)
    if len(created) != 1:
        raise OverlapRunError(
            "{}期望新建1个结果目录，实际为{}".format(
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
    """记录失败尝试、停止批次并输出恢复命令。"""

    attempt.update(
        {
            "status": "failed",
            "completed_at": _timestamp(),
            "error": "{}: {}".format(type(error).__name__, error),
            "created_results": [
                str(Path(path).resolve()) for path in created_results
            ],
        }
    )
    entry.update({"status": "failed", "error": attempt["error"]})
    payload["status"] = "failed"
    _save_batch_manifest(manifest_path, payload)
    print("实验失败并停止：{}".format(entry["scenario_id"]))
    print("恢复命令：python -m HFLSnF_KG_v5.run_overlap_ablation "
          "formal150 --resume \"{}\"".format(manifest_path))
    return 1


def run_training_batch(manifest_path: Path) -> int:
    """按seed优先顺序运行九组训练，失败即停并支持恢复。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    payload["status"] = "running"
    _save_batch_manifest(resolved, payload)
    entries = payload["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry.get("status") == "passed":
            print("跳过已通过实验：{}".format(entry["scenario_id"]))
            continue
        scenario = scenario_by_id(str(entry["scenario_id"]))
        if scenario.seed != 42 and payload.get("pilot_seed42_gate") != "passed":
            raise RuntimeError("seed42低中高三组尚未全部通过，禁止运行后续种子")
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
        print("开始V4重叠率实验：{}".format(scenario.scenario_id))
        try:
            result_dir = _run_config(scenario)
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
                resolved,
                payload,
                entry,
                attempt,
                error,
                created_results=tuple(
                    Path(path) for path in attempt["created_results"]
                ),
            )
        except Exception as error:
            created = (
                error.created_results
                if isinstance(error, OverlapRunError)
                else tuple(Path(path) for path in attempt["created_results"])
            )
            return _record_failure(
                resolved,
                payload,
                entry,
                attempt,
                error,
                created_results=created,
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
        seed42_entries = [
            item for item in entries if int(item["seed"]) == 42
        ]
        if all(item.get("status") == "passed" for item in seed42_entries):
            payload["pilot_seed42_gate"] = "passed"
        _save_batch_manifest(resolved, payload)
        print("{}结果合同：passed".format(scenario.scenario_id))
    payload["status"] = "passed"
    payload["completed_at"] = _timestamp()
    _save_batch_manifest(resolved, payload)
    print("V4实体重叠率九组训练全部完成")
    return 0


def _official_units(payload: Mapping[str, object]) -> List[Dict[str, object]]:
    """从基线和已通过训练项目构造十二格官方测试清单。"""

    if payload.get("status") != "passed":
        raise ValueError("九组训练批次未全部通过，禁止官方测试")
    units: List[Dict[str, object]] = []
    for seed in SEEDS:
        units.append(
            {
                "scenario_id": "original_seed{}".format(seed),
                "condition": "original",
                "seed": seed,
                "result_dir": str(BASELINE_RESULT_DIRS[seed]),
            }
        )
    entries = payload.get("entries")
    assert isinstance(entries, list)
    for entry in entries:
        if entry.get("status") != "passed":
            raise ValueError("存在未通过的新训练项目")
        units.append(
            {
                "scenario_id": str(entry["scenario_id"]),
                "condition": str(entry["level"]),
                "seed": int(entry["seed"]),
                "result_dir": str(entry["result_dir"]),
            }
        )
    return units


def run_official12(manifest_path: Path) -> int:
    """在集中输出目录顺序评估十二个最佳验证检查点。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    units = _official_units(payload)
    existing_manifest = payload.get("official_evaluation_manifest")
    if existing_manifest:
        official_path = Path(str(existing_manifest)).resolve()
        official = _load_json(official_path)
    else:
        output_root = resolved.parent / "official12"
        official_path = output_root / OFFICIAL_MANIFEST_FILE_NAME
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
                    "output_dir": str(
                        output_root / str(unit["scenario_id"])
                    ),
                    "summary_file": None,
                    "error": None,
                }
                for unit in units
            ],
        }
        write_json_report(official, official_path)
        payload["official_evaluation_manifest"] = str(official_path)
        _save_batch_manifest(resolved, payload)
    official_units = official.get("units")
    if not isinstance(official_units, list) or len(official_units) != 12:
        raise ValueError("官方测试清单必须包含十二个单元")
    identity_fields = ("scenario_id", "condition", "seed", "result_dir")
    actual_identities = [
        {field: unit.get(field) for field in identity_fields}
        for unit in official_units
    ]
    expected_identities = [
        {field: unit.get(field) for field in identity_fields}
        for unit in units
    ]
    if actual_identities != expected_identities:
        raise ValueError("官方测试清单的十二格身份或结果路径已被修改")
    official["status"] = "running"
    for unit in official_units:
        if unit.get("status") == "passed":
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
            summary = _load_json(summary_path)
            if summary.get("status") != "passed" or not summary.get(
                "full_official_test"
            ):
                raise RuntimeError("完整官方测试合同未通过")
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
    official["status"] = "passed"
    official["completed_at"] = _timestamp()
    official["updated_at"] = _timestamp()
    write_json_report(official, official_path)
    return 0


def _sample_std(values: Sequence[float]) -> float:
    """对三个种子结果计算样本标准差，单值时返回零。"""

    return statistics.stdev(values) if len(values) > 1 else 0.0


def _unit_analysis(unit: Mapping[str, object]) -> Dict[str, object]:
    """汇总一个实验单元的测试、收敛、数据和通信指标。"""

    result_dir = Path(str(unit["result_dir"])).resolve()
    official = _load_json(Path(str(unit["summary_file"])).resolve())
    partition = _load_json(result_dir / "client_partition_summary.json")
    if "entity_replication_factor" not in partition:
        # 历史原始对照缺少V4新增统计，按同种子原始策略只读重算摘要。
        dataset = load_fb15k237(PACKAGE_DIR / "data" / "FB15k-237")
        reconstructed = partition_train_triples_by_head(
            dataset=dataset,
            client_count=37,
            seed=int(unit["seed"]),
        )
        if reconstructed.partition_hash != partition.get("partition_hash"):
            raise RuntimeError("原始对照分区重算哈希不一致")
        partition = reconstructed.summary()
    rows = _load_metrics(result_dir / "metrics.csv")
    combined = official["combined_metrics"]
    head = official["head_metrics"]
    tail = official["tail_metrics"]
    assert isinstance(combined, dict)
    assert isinstance(head, dict)
    assert isinstance(tail, dict)
    return {
        "scenario_id": unit["scenario_id"],
        "condition": unit["condition"],
        "seed": int(unit["seed"]),
        "result_dir": str(result_dir),
        "entity_normalized_overlap": float(
            partition["entity_normalized_overlap"]
        ),
        "entity_replication_factor": float(
            partition["entity_replication_factor"]
        ),
        "entity_jaccard_mean": float(partition["entity_jaccard_mean"]),
        "entity_client_count_median": float(
            partition["entity_client_count_median"]
        ),
        "entity_client_count_p90": float(
            partition["entity_client_count_p90"]
        ),
        "shared_entity_fraction": float(partition["shared_entity_fraction"]),
        "relation_normalized_overlap": float(
            partition["relation_normalized_overlap"]
        ),
        "max_relative_load_deviation": float(
            partition["max_relative_load_deviation"]
        ),
        "test_mrr": float(combined["mrr"]),
        "test_hits_at_1": float(combined["hits_at_1"]),
        "test_hits_at_3": float(combined["hits_at_3"]),
        "test_hits_at_10": float(combined["hits_at_10"]),
        "test_mean_rank": float(combined["mean_rank"]),
        "head_test_mrr": float(head["mrr"]),
        "tail_test_mrr": float(tail["mrr"]),
        **convergence_statistics(rows),
        **communication_statistics(rows),
    }


def _aggregate_conditions(
    units: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """按四种数据条件计算三种子均值、样本标准差和配对差。"""

    aggregates: Dict[str, object] = {}
    originals = {
        int(unit["seed"]): float(unit["test_mrr"])
        for unit in units
        if unit["condition"] == "original"
    }
    aggregate_metrics = (
        "test_mrr",
        "test_hits_at_1",
        "test_hits_at_3",
        "test_hits_at_10",
        "test_mean_rank",
        "head_test_mrr",
        "tail_test_mrr",
        "best_validation_mrr",
        "last20_validation_mrr_mean",
        "last20_validation_mrr_slope",
        "validation_mrr_auc",
        "round_to_95pct_last20_mean",
        "actual_dense_upload_bytes",
        "logical_sparse_activity_bytes",
    )
    for condition in ("original", *LEVELS):
        selected = [unit for unit in units if unit["condition"] == condition]
        paired = [
            float(unit["test_mrr"]) - originals[int(unit["seed"])]
            for unit in selected
        ]
        summary: Dict[str, object] = {
            "seed_count": len(selected),
            "paired_delta_vs_original": paired,
            "paired_delta_vs_original_mean": statistics.mean(paired),
            "entity_normalized_overlap_mean": statistics.mean(
                float(unit["entity_normalized_overlap"])
                for unit in selected
            ),
        }
        for metric_name in aggregate_metrics:
            metric_values = [
                float(unit[metric_name]) for unit in selected
            ]
            summary["{}_mean".format(metric_name)] = statistics.mean(
                metric_values
            )
            summary["{}_sample_std".format(metric_name)] = _sample_std(
                metric_values
            )
        aggregates[condition] = summary
    return aggregates


def _write_markdown_report(
    report: Mapping[str, object],
    path: Path,
) -> None:
    """用简体中文写出结论克制的四条件三种子报告。"""

    aggregates = report["condition_aggregates"]
    assert isinstance(aggregates, dict)
    lines = [
        "# HFLSnF_KG_v5实体重叠率消融实验报告",
        "",
        "本报告比较原始划分与低、中、高实体重叠划分。三种子仅用于描述均值、样本标准差和配对效应，不进行小样本显著性检验。",
        "",
        "| 数据条件 | 平均实体重叠率 | 测试MRR均值 | MRR样本标准差 | 相对原始配对差均值 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {"original": "原始", "low": "低", "medium": "中", "high": "高"}
    for condition in ("original", *LEVELS):
        item = aggregates[condition]
        assert isinstance(item, dict)
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                labels[condition],
                float(item["entity_normalized_overlap_mean"]),
                float(item["test_mrr_mean"]),
                float(item["test_mrr_sample_std"]),
                float(item["paired_delta_vs_original_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## 次要测试指标",
            "",
            "| 数据条件 | Hits@1 | Hits@3 | Hits@10 | 头预测MRR | 尾预测MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in ("original", *LEVELS):
        item = aggregates[condition]
        assert isinstance(item, dict)
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                labels[condition],
                float(item["test_hits_at_1_mean"]),
                float(item["test_hits_at_3_mean"]),
                float(item["test_hits_at_10_mean"]),
                float(item["head_test_mrr_mean"]),
                float(item["tail_test_mrr_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## 收敛与通信",
            "",
            "| 数据条件 | 后20轮MRR均值 | 后20轮斜率 | 95%到达轮次 | 实际密集上传字节 | 逻辑稀疏活动字节 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in ("original", *LEVELS):
        item = aggregates[condition]
        assert isinstance(item, dict)
        lines.append(
            "| {} | {:.6f} | {:.8f} | {:.2f} | {:.0f} | {:.0f} |".format(
                labels[condition],
                float(item["last20_validation_mrr_mean_mean"]),
                float(item["last20_validation_mrr_slope_mean"]),
                float(item["round_to_95pct_last20_mean_mean"]),
                float(item["actual_dense_upload_bytes_mean"]),
                float(item["logical_sparse_activity_bytes_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## 图表",
            "",
            "![实体重叠率与测试MRR](overlap_vs_test_mrr.png)",
            "",
            "![十二格验证MRR曲线](validation_mrr_curves.png)",
            "",
            "## 解释边界",
            "",
            "- 本实验只改变客户端训练三元组分区，模型、拓扑、初始化和优化参数保持不变。",
            "- 当前实现上传完整嵌入表；逻辑稀疏活动字节数不是已经观测到的网络节省。",
            "- 三个随机种子的结果用于一致性和效应大小描述，不据此作强显著性或因果结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_report(manifest_path: Path) -> int:
    """汇总十二格结果并写出JSON、CSV、图和简体中文报告。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    official_path = payload.get("official_evaluation_manifest")
    if not official_path:
        raise ValueError("尚未创建十二格官方测试清单")
    official = _load_json(Path(str(official_path)).resolve())
    if official.get("status") != "passed":
        raise ValueError("十二格官方测试尚未全部通过")
    official_units = official.get("units")
    assert isinstance(official_units, list)
    units = [_unit_analysis(unit) for unit in official_units]
    report = {
        "suite": SUITE_NAME,
        "status": "passed",
        "experiment_unit_count": len(units),
        "inference_policy": (
            "仅报告三种子均值、样本标准差和配对效应，不执行小样本显著性检验"
        ),
        "units": units,
        "condition_aggregates": _aggregate_conditions(units),
    }
    report_dir = resolved.parent / "analysis"
    json_path = report_dir / REPORT_JSON_FILE_NAME
    markdown_path = report_dir / REPORT_MARKDOWN_FILE_NAME
    csv_path = report_dir / "overlap_ablation_units.csv"
    write_json_report(report, json_path)
    _write_markdown_report(report, markdown_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(units[0].keys()))
        writer.writeheader()
        writer.writerows(units)
    _write_figures(units, report_dir)
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(markdown_path)
    _save_batch_manifest(resolved, payload)
    return 0


def _write_figures(
    units: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """生成重叠率—测试MRR散点图和四条件验证曲线。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "original": "#4C78A8",
        "low": "#59A14F",
        "medium": "#F28E2B",
        "high": "#E15759",
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    labels = {
        "original": "Original",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
    }
    for condition in ("original", *LEVELS):
        selected = [
            unit for unit in units if unit["condition"] == condition
        ]
        axis.scatter(
            [float(unit["entity_normalized_overlap"]) for unit in selected],
            [float(unit["test_mrr"]) for unit in selected],
            color=colors[condition],
            s=42,
            label=labels[condition],
        )
    axis.set_xlabel("Normalized entity overlap")
    axis.set_ylabel("Official test MRR")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "overlap_vs_test_mrr.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    labeled_conditions = set()
    for unit in units:
        rows = _load_metrics(Path(str(unit["result_dir"])) / "metrics.csv")
        values = [float(row["val_mrr"]) for row in rows]
        condition = str(unit["condition"])
        axis.plot(
            range(1, len(values) + 1),
            values,
            color=colors[condition],
            alpha=0.45,
            linewidth=1.0,
            label=(
                labels[condition]
                if condition not in labeled_conditions
                else None
            ),
        )
        labeled_conditions.add(condition)
    axis.set_xlabel("Communication round")
    axis.set_ylabel("Validation MRR")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "validation_mrr_curves.png", dpi=180)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """执行所选阶段，并在任何合同失败时返回非零退出码。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.action == "formal150" and args.skip_partition_recompute:
        parser.error("formal150禁止--skip-partition-recompute")
    recompute_partitions = (
        args.action in {"validate", "formal150"}
        and not bool(args.skip_partition_recompute)
    )
    config_report = validate_configs(
        recompute_partitions=recompute_partitions
    )
    print("V4重叠率配置合同：{}".format(config_report["status"]))
    if config_report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        if args.resume is not None or args.batch is not None:
            parser.error("validate不能使用--resume或--batch")
        return
    if args.action == "formal150":
        if args.batch is not None:
            parser.error("formal150使用--resume而不是--batch")
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
        parser.error("official12和report必须使用--batch且不能使用--resume")
    if args.action == "official12":
        if run_official12(args.batch):
            raise SystemExit(1)
        return
    if run_report(args.batch):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
