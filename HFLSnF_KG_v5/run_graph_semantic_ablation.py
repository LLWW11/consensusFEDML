"""校验、训练、官方测试并汇总V5图语义三种子实验。"""

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
from .tasks.kge.graph_semantic_ablation import (
    BASELINE_INITIAL_MODEL_HASHES,
    BASELINE_RESULT_DIRS,
    CALIBRATION_CONTRACT_PATH,
    ROUND_COUNT,
    SEEDS,
    SUITE_NAME,
    V4_REFERENCE_FILES,
    GraphSemanticScenario,
    communication_statistics,
    convergence_statistics,
    scenario_by_id,
    scenarios_from_contract,
    validate_configs,
    validate_result,
    validate_v4_references,
)


BATCH_FILE_NAME = "batch_summary.json"
RESULT_CONTRACT_FILE_NAME = "graph_semantic_formal150_contract.json"
OFFICIAL_MANIFEST_FILE_NAME = "official3_summary.json"
REPORT_JSON_FILE_NAME = "graph_semantic_summary.json"
REPORT_CSV_FILE_NAME = "graph_semantic_units.csv"
REPORT_MARKDOWN_FILE_NAME = "图语义划分实验报告.md"


class GraphSemanticRunError(RuntimeError):
    """表示训练子进程失败或结果目录数量不符合合同。"""

    def __init__(
        self,
        message: str,
        created_results: Sequence[Path] = (),
    ) -> None:
        """保存错误文本和当前尝试产生的结果目录。"""

        super().__init__(message)
        self.created_results = tuple(
            Path(path).resolve() for path in created_results
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """创建图语义四阶段命令行参数。"""

    parser = argparse.ArgumentParser(
        description="运行HFLSnF_KG_v5图语义三种子正式实验"
    )
    parser.add_argument(
        "action",
        choices=("validate", "formal150", "official3", "report"),
        help="选择配置校验、三组训练、三组官方测试或报告",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="formal150恢复使用的batch_summary.json",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="official3和report必须指定的批次清单",
    )
    parser.add_argument(
        "--skip-partition-recompute",
        action="store_true",
        help="仅供快速开发校验；formal150禁止使用",
    )
    return parser


def _timestamp() -> str:
    """返回包含本地时区的秒级时间。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    """分块计算文件SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_payload_hash(payload: Mapping[str, object]) -> str:
    """计算排序JSON对象的稳定绑定哈希。"""

    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _entry_binding_fields(entry: Mapping[str, object]) -> Dict[str, object]:
    """提取恢复时禁止修改的场景字段。"""

    keys = (
        "scenario_id",
        "seed",
        "partition_hash",
        "config",
        "config_sha256",
        "order_index",
    )
    return {key: entry.get(key) for key in keys}


def _bound_entry(
    scenario: GraphSemanticScenario,
    order_index: int,
) -> Dict[str, object]:
    """构造与正式配置和分区哈希绑定的训练项目。"""

    entry: Dict[str, object] = {
        "scenario_id": scenario.scenario_id,
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
    entry["binding_hash"] = _stable_payload_hash(
        _entry_binding_fields(entry)
    )
    return entry


def _empty_batch_payload() -> Dict[str, object]:
    """构造三项原始对照和三项新训练的批次清单。"""

    scenarios = scenarios_from_contract()
    now = _timestamp()
    return {
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "planned_new_run_count": 3,
        "primary_experiment_unit_count": 6,
        "context_experiment_unit_count": 15,
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
                "initial_model_hash": (
                    BASELINE_INITIAL_MODEL_HASHES[seed]
                ),
            }
            for seed in SEEDS
        },
        "run_order": [
            scenario.scenario_id for scenario in scenarios
        ],
        "entries": [
            _bound_entry(scenario, index)
            for index, scenario in enumerate(scenarios, start=1)
        ],
        "official_evaluation_manifest": None,
        "report_json": None,
        "report_markdown": None,
    }


def create_batch_manifest(
    result_root: Optional[Path] = None,
) -> Path:
    """创建不会覆盖旧结果的图语义批次清单。"""

    root = (
        Path(result_root).expanduser().resolve()
        if result_root is not None
        else (PACKAGE_DIR / "results" / "graph_semantic").resolve()
    )
    name = "graph_semantic_batch_{}".format(
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    path = root / name / BATCH_FILE_NAME
    write_json_report(_empty_batch_payload(), path)
    return path


def _validate_entry_binding(entry: Mapping[str, object]) -> None:
    """校验恢复项目的身份、配置和稳定绑定。"""

    scenario = scenario_by_id(str(entry.get("scenario_id", "")))
    config_path = Path(str(entry.get("config", ""))).resolve()
    if config_path != scenario.config_path or not config_path.is_file():
        raise ValueError("批次配置路径与正式场景不一致")
    if _file_sha256(config_path) != str(
        entry.get("config_sha256", "")
    ):
        raise ValueError("批次配置已被修改：{}".format(config_path))
    if _stable_payload_hash(
        _entry_binding_fields(entry)
    ) != entry.get("binding_hash"):
        raise ValueError("批次场景绑定字段已被修改")


def _load_batch_manifest(path: Path) -> Dict[str, object]:
    """读取并校验批次位置、合同、顺序和配置绑定。"""

    resolved = Path(path).expanduser().resolve()
    root = (PACKAGE_DIR / "results" / "graph_semantic").resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("批次必须位于V5图语义结果目录") from error
    payload = _load_json(resolved)
    scenarios = scenarios_from_contract()
    expected_order = [
        scenario.scenario_id for scenario in scenarios
    ]
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("批次套件身份不一致")
    if payload.get("run_order") != expected_order:
        raise ValueError("三组运行顺序已被修改")
    if int(payload.get("planned_new_run_count", -1)) != 3:
        raise ValueError("批次新训练数量必须为3")
    if _file_sha256(CALIBRATION_CONTRACT_PATH) != payload.get(
        "calibration_contract_sha256"
    ):
        raise ValueError("图语义校准合同已被修改")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("批次必须包含三个训练项目")
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


def _result_root_for_scenario(
    scenario: GraphSemanticScenario,
) -> Path:
    """按正式配置解析训练结果根目录。"""

    config = load_flat_config(scenario.config_path)
    return (
        PACKAGE_DIR / str(config.get("result_root", "results"))
    ).resolve()


def _result_directories(
    scenario: GraphSemanticScenario,
) -> Tuple[Path, ...]:
    """返回与场景运行名前缀匹配的既有结果目录。"""

    config = load_flat_config(scenario.config_path)
    run_name = str(config["run_name"])
    root = _result_root_for_scenario(scenario)
    return tuple(
        sorted(
            (
                path.resolve()
                for path in root.glob("{}_*".format(run_name))
                if path.is_dir()
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
    )


def _new_result_directories(
    scenario: GraphSemanticScenario,
    before: Sequence[Path],
) -> Tuple[Path, ...]:
    """返回一次训练后唯一新增的结果目录集合。"""

    existing = {Path(path).resolve() for path in before}
    return tuple(
        path
        for path in _result_directories(scenario)
        if path not in existing
    )


def _run_config(scenario: GraphSemanticScenario) -> Path:
    """调用V5统一训练入口并返回唯一新增结果目录。"""

    before = _result_directories(scenario)
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v5.run_federated_transe",
        "--cf",
        str(scenario.config_path),
    ]
    try:
        subprocess.run(
            command,
            cwd=str(PACKAGE_DIR.parent),
            check=True,
        )
    except subprocess.CalledProcessError as error:
        created = _new_result_directories(scenario, before)
        raise GraphSemanticRunError(
            "{}训练失败，退出码{}".format(
                scenario.scenario_id,
                error.returncode,
            ),
            created_results=created,
        ) from error
    created = _new_result_directories(scenario, before)
    if len(created) != 1:
        raise GraphSemanticRunError(
            "{}期望新增1个结果目录，实际为{}".format(
                scenario.scenario_id,
                len(created),
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
    """记录失败并停止批次，保留恢复信息。"""

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
    print(
        "恢复命令：python -m "
        "HFLSnF_KG_v5.run_graph_semantic_ablation "
        'formal150 --resume "{}"'.format(manifest_path)
    )
    return 1


def run_training_batch(manifest_path: Path) -> int:
    """按seed42、2024、2025顺序训练并失败即停。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    payload["status"] = "running"
    _save_batch_manifest(resolved, payload)
    entries = payload["entries"]
    for entry in entries:
        if entry.get("status") == "passed":
            print("跳过已通过实验：{}".format(entry["scenario_id"]))
            continue
        scenario = scenario_by_id(str(entry["scenario_id"]))
        if scenario.seed != 42 and payload.get(
            "pilot_seed42_gate"
        ) != "passed":
            raise RuntimeError(
                "seed42结果合同未通过，禁止运行后续种子"
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
        print("开始V5图语义实验：{}".format(scenario.scenario_id))
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
                tuple(Path(path) for path in attempt["created_results"]),
            )
        except Exception as error:
            created = (
                error.created_results
                if isinstance(error, GraphSemanticRunError)
                else tuple(
                    Path(path)
                    for path in attempt["created_results"]
                )
            )
            return _record_failure(
                resolved,
                payload,
                entry,
                attempt,
                error,
                created,
            )
        attempt.update(
            {"status": "passed", "completed_at": _timestamp()}
        )
        entry.update(
            {
                "status": "passed",
                "result_dir": attempt["result_dir"],
                "contract_file": attempt["contract_file"],
                "error": None,
            }
        )
        if scenario.seed == 42:
            payload["pilot_seed42_gate"] = "passed"
        _save_batch_manifest(resolved, payload)
        print("{}结果合同：passed".format(scenario.scenario_id))
    payload["status"] = "passed"
    payload["completed_at"] = _timestamp()
    _save_batch_manifest(resolved, payload)
    print("V5图语义三组训练全部完成")
    return 0


def _official_units(
    payload: Mapping[str, object],
) -> List[Dict[str, object]]:
    """从已通过批次构造三个新场景的官方测试清单。"""

    if payload.get("status") != "passed":
        raise ValueError("三组训练未全部通过，禁止官方测试")
    units = []
    for entry in payload["entries"]:
        if entry.get("status") != "passed":
            raise ValueError("存在未通过的新训练项目")
        units.append(
            {
                "scenario_id": str(entry["scenario_id"]),
                "condition": "graph_semantic",
                "seed": int(entry["seed"]),
                "result_dir": str(entry["result_dir"]),
            }
        )
    return units


def run_official3(manifest_path: Path) -> int:
    """顺序执行三个图语义最佳检查点的完整官方测试。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    units = _official_units(payload)
    existing = payload.get("official_evaluation_manifest")
    if existing:
        official_path = Path(str(existing)).resolve()
        official = _load_json(official_path)
    else:
        root = resolved.parent / "official3"
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
                    "output_dir": str(
                        root / str(unit["scenario_id"])
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
    actual_units = official.get("units")
    if not isinstance(actual_units, list) or len(actual_units) != 3:
        raise ValueError("官方测试清单必须包含三个单元")
    identity_fields = ("scenario_id", "condition", "seed", "result_dir")
    if [
        {field: unit.get(field) for field in identity_fields}
        for unit in actual_units
    ] != [
        {field: unit.get(field) for field in identity_fields}
        for unit in units
    ]:
        raise ValueError("官方测试清单身份已被修改")

    official["status"] = "running"
    for unit in actual_units:
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
            subprocess.run(
                command,
                cwd=str(PACKAGE_DIR.parent),
                check=True,
            )
            summary_path = (
                output_dir / "official_evaluation_summary.json"
            )
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
                    "error": "{}: {}".format(
                        type(error).__name__,
                        error,
                    ),
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


def _new_unit_analysis(
    unit: Mapping[str, object],
) -> Dict[str, object]:
    """汇总一个图语义单元的测试、数据和收敛指标。"""

    result_dir = Path(str(unit["result_dir"])).resolve()
    official = _load_json(Path(str(unit["summary_file"])).resolve())
    partition = _load_json(
        result_dir / "client_partition_summary.json"
    )
    rows = _load_metrics(result_dir / "metrics.csv")
    combined = official["combined_metrics"]
    head = official["head_metrics"]
    tail = official["tail_metrics"]
    return {
        "scenario_id": str(unit["scenario_id"]),
        "condition": "graph_semantic",
        "seed": int(unit["seed"]),
        "result_dir": str(result_dir),
        "entity_normalized_overlap": float(
            partition["entity_normalized_overlap"]
        ),
        "relation_normalized_overlap": float(
            partition["relation_normalized_overlap"]
        ),
        "max_relative_load_deviation": float(
            partition["max_relative_load_deviation"]
        ),
        "semantic_purity": float(
            partition[
                "triple_weighted_dominant_domain_purity"
            ]
        ),
        "mean_relation_js_divergence": float(
            partition["mean_relation_js_divergence"]
        ),
        "local_entity_reuse_ratio": float(
            partition["local_entity_reuse_ratio"]
        ),
        "mean_largest_component_entity_fraction": float(
            partition[
                "mean_largest_component_entity_fraction"
            ]
        ),
        "test_mrr": float(combined["mrr"]),
        "test_hits_at_1": float(combined["hits_at_1"]),
        "test_hits_at_3": float(combined["hits_at_3"]),
        "test_hits_at_10": float(combined["hits_at_10"]),
        "head_test_mrr": float(head["mrr"]),
        "tail_test_mrr": float(tail["mrr"]),
        **convergence_statistics(rows),
        **communication_statistics(rows),
    }


def _mean(values: Sequence[float]) -> float:
    """返回非空数值序列均值。"""

    if not values:
        raise ValueError("均值输入不能为空")
    return float(statistics.mean(values))


def run_report(manifest_path: Path) -> int:
    """生成三组主比较和十五单元背景合并报告。"""

    resolved = Path(manifest_path).expanduser().resolve()
    payload = _load_batch_manifest(resolved)
    official_path = payload.get("official_evaluation_manifest")
    if not official_path:
        raise ValueError("尚未执行official3")
    official = _load_json(Path(str(official_path)).resolve())
    if official.get("status") != "passed":
        raise ValueError("official3尚未全部通过")
    reference = validate_v4_references()
    if reference["status"] != "passed":
        raise ValueError("V4十二格参考合同失败")
    new_units = [
        _new_unit_analysis(unit) for unit in official["units"]
    ]
    v4_summary = _load_json(
        Path(
            V4_REFERENCE_FILES["analysis_summary"]["path"]
        ).resolve()
    )
    v4_units = list(v4_summary["units"])
    originals = {
        int(unit["seed"]): unit
        for unit in v4_units
        if unit["condition"] == "original"
    }
    paired = [
        {
            "seed": int(unit["seed"]),
            "graph_semantic_test_mrr": float(unit["test_mrr"]),
            "original_test_mrr": float(
                originals[int(unit["seed"])]["test_mrr"]
            ),
            "paired_mrr_delta": (
                float(unit["test_mrr"])
                - float(originals[int(unit["seed"])]["test_mrr"])
            ),
        }
        for unit in new_units
    ]
    aggregate = {
        "test_mrr_mean": _mean(
            [float(unit["test_mrr"]) for unit in new_units]
        ),
        "test_mrr_sample_std": (
            float(
                statistics.stdev(
                    [float(unit["test_mrr"]) for unit in new_units]
                )
            )
            if len(new_units) > 1
            else 0.0
        ),
        "paired_mrr_delta_mean": _mean(
            [float(item["paired_mrr_delta"]) for item in paired]
        ),
        "semantic_purity_mean": _mean(
            [float(unit["semantic_purity"]) for unit in new_units]
        ),
        "relation_js_divergence_mean": _mean(
            [
                float(unit["mean_relation_js_divergence"])
                for unit in new_units
            ]
        ),
        "local_entity_reuse_ratio_mean": _mean(
            [
                float(unit["local_entity_reuse_ratio"])
                for unit in new_units
            ]
        ),
    }
    report = {
        "suite": SUITE_NAME,
        "status": "passed",
        "primary_experiment_unit_count": 6,
        "context_experiment_unit_count": 15,
        "new_units": new_units,
        "paired_comparison": paired,
        "graph_semantic_aggregate": aggregate,
        "v4_reference": {
            "suite": v4_summary["suite"],
            "unit_count": len(v4_units),
            "condition_aggregates": v4_summary[
                "condition_aggregates"
            ],
            "files": {
                name: {
                    "path": str(item["path"]),
                    "sha256": str(item["sha256"]),
                }
                for name, item in V4_REFERENCE_FILES.items()
            },
        },
        "combined_units": v4_units + new_units,
        "interpretation_boundary": (
            "图语义划分同时改变语义异构、图局部性和实体重叠，"
            "结果只能解释为现实代理场景的联合变化，不能归因于"
            "实体重叠率单因素。"
        ),
    }
    report_dir = resolved.parent / "analysis"
    json_path = report_dir / REPORT_JSON_FILE_NAME
    csv_path = report_dir / REPORT_CSV_FILE_NAME
    markdown_path = report_dir / REPORT_MARKDOWN_FILE_NAME
    write_json_report(report, json_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted(
        set().union(
            *(set(unit.keys()) for unit in report["combined_units"])
        )
    )
    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["combined_units"])
    lines = [
        "# V5图语义感知划分实验报告",
        "",
        "## 主要结果",
        "",
        "| 种子 | 图语义测试MRR | 原始测试MRR | 配对差 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for item in paired:
        lines.append(
            "| {seed} | {graph_semantic_test_mrr:.6f} | "
            "{original_test_mrr:.6f} | {paired_mrr_delta:.6f} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "- 图语义测试MRR均值：{:.6f}".format(
                aggregate["test_mrr_mean"]
            ),
            "- 相对原始划分配对差均值：{:.6f}".format(
                aggregate["paired_mrr_delta_mean"]
            ),
            "- 平均语义纯度：{:.6f}".format(
                aggregate["semantic_purity_mean"]
            ),
            "- 平均关系JS散度：{:.6f}".format(
                aggregate["relation_js_divergence_mean"]
            ),
            "- 平均局部实体复用率：{:.6f}".format(
                aggregate["local_entity_reuse_ratio_mean"]
            ),
            "",
            "## 解释边界",
            "",
            report["interpretation_boundary"],
            "",
            "V4十二格结果仅作为固定哈希的背景参照，本报告没有"
            "重新训练或改写V4产物。",
        ]
    )
    markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(markdown_path)
    _save_batch_manifest(resolved, payload)
    print("图语义报告：{}".format(markdown_path))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """执行指定阶段，并在合同失败时返回非零退出码。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.action == "formal150" and args.skip_partition_recompute:
        parser.error("formal150禁止--skip-partition-recompute")
    recompute = (
        args.action in {"validate", "formal150"}
        and not bool(args.skip_partition_recompute)
    )
    config_report = validate_configs(recompute_partitions=recompute)
    print("V5图语义配置合同：{}".format(config_report["status"]))
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
        parser.error(
            "official3和report必须使用--batch且不能使用--resume"
        )
    if args.action == "official3":
        if run_official3(args.batch):
            raise SystemExit(1)
        return
    if run_report(args.batch):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
