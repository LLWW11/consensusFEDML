"""V2同MAT三臂TransE消融的配置合同、结果审计和汇总工具。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import yaml


ABLATION_SUITE_NAME = "v2_same_mat_three_arm_v1"
EXPECTED_PARTITION_HASH = (
    "8bcac64b705ec2db8721de6a36130625a460c11e0da46e2c22bd852ff015fb19"
)
EXPECTED_SCHEDULE_HASH = (
    "811b2940bada978fea233b968dbfa873fcef2e7664842945c3d3fb60e5836066"
)


@dataclass(frozen=True)
class ThreeArmSpec:
    """描述一个三臂消融方案允许改变的训练语义。"""

    arm: str
    label: str
    config_filename: str
    aggregation_mode: str
    local_objective: str
    distance_norm: int

    def summary(self) -> Dict[str, object]:
        """返回适合写入JSON合同的方案摘要。"""

        return {
            "arm": self.arm,
            "label": self.label,
            "config_filename": self.config_filename,
            "aggregation_mode": self.aggregation_mode,
            "local_objective": self.local_objective,
            "distance_norm": int(self.distance_norm),
        }


THREE_ARM_SPECS: Tuple[ThreeArmSpec, ...] = (
    ThreeArmSpec(
        arm="dense_margin",
        label="A：dense+margin",
        config_filename=(
            "server_fb15k237_hflsnf_dynamic_mat_cuda.yaml"
        ),
        aggregation_mode="dense_triple_weighted",
        local_objective="margin_ranking",
        distance_norm=2,
    ),
    ThreeArmSpec(
        arm="masked_margin",
        label="B：masked+margin",
        config_filename=(
            "server_fb15k237_hflsnf_dynamic_mat_masked_cuda.yaml"
        ),
        aggregation_mode="row_mask_presence",
        local_objective="margin_ranking",
        distance_norm=2,
    ),
    ThreeArmSpec(
        arm="masked_fede_fair",
        label="C：masked+FedE-fair",
        config_filename=(
            "server_fb15k237_hflsnf_dynamic_mat_masked_fede_fair_cuda.yaml"
        ),
        aggregation_mode="row_mask_presence",
        local_objective="fede_self_adversarial",
        distance_norm=1,
    ),
)


SHARED_CONTRACT_FIELDS: Tuple[str, ...] = (
    "ablation_suite",
    "random_seed",
    "dataset",
    "data_dir",
    "partition_strategy",
    "model",
    "embedding_dim",
    "client_num_in_total",
    "client_num_per_round",
    "topology_type",
    "topology_architecture",
    "topology_snf",
    "topology_edge_mode",
    "topology_util",
    "dynamic_group_mat_file",
    "comm_round",
    "epochs",
    "batch_size",
    "client_optimizer",
    "learning_rate",
    "lr",
    "margin",
    "negative_sample_count",
    "eval_every",
    "validation_max_triples",
    "final_validation_max_triples",
    "test_max_triples",
    "evaluation_candidate_batch_size",
    "centralized_reference_mrr",
    "using_gpu",
    "gpu_id",
    "require_cuda",
    "backend",
    "result_root",
    "expected_partition_hash",
    "expected_topology_schedule_hash",
)


RESULT_EQUALITY_FIELDS: Tuple[str, ...] = (
    "ablation_suite",
    "partition_hash",
    "topology_schedule_hash",
    "initial_model_hash",
    "client_count",
    "client_num_in_total",
    "client_num_per_round",
    "participant_count_min",
    "participant_count_max",
    "participant_count_mean",
    "group_count_min",
    "group_count_max",
    "group_count_mean",
    "comm_round",
    "local_epochs",
    "effective_global_passes",
)


def _canonical_hash(payload: object) -> str:
    """对排序后的紧凑JSON计算稳定SHA-256。"""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """分块计算文件SHA-256，避免一次读取大文件。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_fedml_yaml(path: Path) -> Dict[str, object]:
    """读取FedML分组YAML并展开成单层运行参数字典。"""

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("FedML配置顶层必须是字典：{}".format(path))

    flattened: Dict[str, object] = {}
    for section_name, section in raw.items():
        if not isinstance(section, dict):
            raise ValueError(
                "配置节{}必须是字典：{}".format(section_name, path)
            )
        for key, value in section.items():
            if key in flattened and flattened[key] != value:
                raise ValueError(
                    "配置字段{}在多个节中取值冲突".format(key)
                )
            flattened[str(key)] = value
    return flattened


def _require_fields(
    config: Mapping[str, object],
    fields: Sequence[str],
    arm: str,
) -> None:
    """确认一个实验臂包含公平合同要求的全部字段。"""

    missing = [field for field in fields if field not in config]
    if missing:
        raise ValueError(
            "实验臂{}缺少公平合同字段：{}".format(
                arm, ", ".join(missing)
            )
        )


def _resolve_package_path(package_dir: Path, value: object) -> Path:
    """把配置中的V2相对路径解析为绝对路径。"""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(package_dir) / path
    return path.resolve()


def _validate_arm_semantics(
    spec: ThreeArmSpec,
    config: Mapping[str, object],
) -> None:
    """校验A、B、C各自只采用约定的聚合和本地目标。"""

    actual_arm = str(config.get("ablation_arm", "")).strip()
    if actual_arm != spec.arm:
        raise ValueError(
            "{}的ablation_arm应为{}，实际为{}".format(
                spec.label, spec.arm, actual_arm
            )
        )
    actual_aggregation = str(
        config.get("aggregation_mode", "")
    ).strip()
    if actual_aggregation != spec.aggregation_mode:
        raise ValueError(
            "{}的aggregation_mode应为{}，实际为{}".format(
                spec.label,
                spec.aggregation_mode,
                actual_aggregation,
            )
        )
    actual_objective = str(
        config.get("local_objective", "")
    ).strip()
    if actual_objective != spec.local_objective:
        raise ValueError(
            "{}的local_objective应为{}，实际为{}".format(
                spec.label,
                spec.local_objective,
                actual_objective,
            )
        )
    actual_norm = int(config.get("distance_norm", 0))
    if actual_norm != spec.distance_norm:
        raise ValueError(
            "{}的distance_norm应为{}，实际为{}".format(
                spec.label, spec.distance_norm, actual_norm
            )
        )

    if spec.local_objective == "fede_self_adversarial":
        if float(config.get("fede_gamma", 0.0)) != 10.0:
            raise ValueError("{}的fede_gamma必须为10.0".format(spec.label))
        if float(config.get("adversarial_temperature", 0.0)) != 1.0:
            raise ValueError(
                "{}的adversarial_temperature必须为1.0".format(
                    spec.label
                )
            )


def _default_config_paths(package_dir: Path) -> Dict[str, Path]:
    """返回仓库内A、B、C三份正式配置路径。"""

    config_dir = Path(package_dir) / "configs"
    return {
        spec.arm: (config_dir / spec.config_filename).resolve()
        for spec in THREE_ARM_SPECS
    }


def validate_three_arm_configs(
    package_dir: Path,
    config_paths: Optional[Mapping[str, Path]] = None,
) -> Dict[str, object]:
    """验证三份YAML只存在预先声明的实验变量差异。"""

    package_dir = Path(package_dir).expanduser().resolve()
    resolved_paths = (
        _default_config_paths(package_dir)
        if config_paths is None
        else {
            str(arm): Path(path).expanduser().resolve()
            for arm, path in config_paths.items()
        }
    )
    expected_arms = {spec.arm for spec in THREE_ARM_SPECS}
    if set(resolved_paths) != expected_arms:
        raise ValueError(
            "三臂配置必须且只能包含{}".format(
                ", ".join(sorted(expected_arms))
            )
        )

    configs: Dict[str, Dict[str, object]] = {}
    for spec in THREE_ARM_SPECS:
        path = resolved_paths[spec.arm]
        if not path.is_file():
            raise FileNotFoundError(
                "找不到{}配置：{}".format(spec.label, path)
            )
        config = load_fedml_yaml(path)
        _require_fields(
            config,
            SHARED_CONTRACT_FIELDS
            + (
                "ablation_arm",
                "aggregation_mode",
                "local_objective",
                "distance_norm",
                "run_name",
            ),
            spec.arm,
        )
        _validate_arm_semantics(spec, config)
        configs[spec.arm] = config

    reference = configs[THREE_ARM_SPECS[0].arm]
    shared_contract = {
        field: reference[field] for field in SHARED_CONTRACT_FIELDS
    }
    for spec in THREE_ARM_SPECS[1:]:
        config = configs[spec.arm]
        differences = {
            field: {
                "reference": reference[field],
                "actual": config[field],
            }
            for field in SHARED_CONTRACT_FIELDS
            if config[field] != reference[field]
        }
        if differences:
            raise ValueError(
                "{}破坏三臂公平合同：{}".format(
                    spec.label,
                    json.dumps(
                        differences,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )

    if shared_contract["ablation_suite"] != ABLATION_SUITE_NAME:
        raise ValueError(
            "ablation_suite必须为{}".format(ABLATION_SUITE_NAME)
        )
    if (
        str(shared_contract["expected_partition_hash"])
        != EXPECTED_PARTITION_HASH
    ):
        raise ValueError("三臂配置中的预期客户端划分哈希不正确")
    if (
        str(shared_contract["expected_topology_schedule_hash"])
        != EXPECTED_SCHEDULE_HASH
    ):
        raise ValueError("三臂配置中的预期MAT调度哈希不正确")

    data_dir = _resolve_package_path(
        package_dir, shared_contract["data_dir"]
    )
    data_files = {}
    for split_name in ("train", "valid", "test"):
        path = data_dir / "{}.txt".format(split_name)
        if not path.is_file():
            raise FileNotFoundError("找不到数据文件：{}".format(path))
        data_files[split_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    mat_path = _resolve_package_path(
        package_dir, shared_contract["dynamic_group_mat_file"]
    )
    if not mat_path.is_file():
        raise FileNotFoundError("找不到MAT调度文件：{}".format(mat_path))
    mat_sha256 = sha256_file(mat_path)

    contract_material = {
        "suite": ABLATION_SUITE_NAME,
        "shared_contract": shared_contract,
        "data_file_hashes": {
            name: values["sha256"]
            for name, values in data_files.items()
        },
        "mat_sha256": mat_sha256,
    }
    arms = []
    for spec in THREE_ARM_SPECS:
        path = resolved_paths[spec.arm]
        config = configs[spec.arm]
        arms.append(
            {
                **spec.summary(),
                "config_path": str(path),
                "config_sha256": sha256_file(path),
                "run_name": str(config["run_name"]),
            }
        )

    return {
        "status": "valid",
        "schema_version": 1,
        "suite": ABLATION_SUITE_NAME,
        "contract_hash": _canonical_hash(contract_material),
        "shared_contract": shared_contract,
        "data_files": data_files,
        "mat_file": {
            "path": str(mat_path),
            "sha256": mat_sha256,
        },
        "arms": arms,
        "allowed_differences": {
            "A_vs_B": [
                "aggregation_mode：稠密三元组数加权改为逐行所有权等权",
            ],
            "B_vs_C": [
                "distance_norm：L2改为L1",
                "negative_sampling：头尾替换改为只替换尾实体",
                "local_objective：间隔排序改为FedE自对抗逻辑损失",
            ],
        },
    }


def _read_json(path: Path) -> Dict[str, object]:
    """读取UTF-8 JSON并确认顶层是字典。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON顶层必须是字典：{}".format(path))
    return payload


def _metrics_runtime_seconds(path: Path) -> float:
    """汇总一个实验结果中全部通信轮的训练与评估耗时。"""

    total = 0.0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row.get("round_seconds", "nan"))
            if math.isfinite(value):
                total += value
    return total


def _finite_metric(
    metrics: Mapping[str, object],
    name: str,
    arm: str,
) -> float:
    """读取一个必须存在且有限的最终评估指标。"""

    if name not in metrics:
        raise ValueError("{}缺少最终指标{}".format(arm, name))
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError("{}的{}不是有限值".format(arm, name))
    return value


def _load_result_arm(
    spec: ThreeArmSpec,
    result_dir: Path,
) -> Dict[str, object]:
    """读取并校验一个三臂训练结果目录。"""

    result_dir = Path(result_dir).expanduser().resolve()
    summary_path = result_dir / "summary.json"
    config_path = result_dir / "config_snapshot.json"
    metrics_path = result_dir / "metrics.csv"
    for path in (summary_path, config_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(
                "{}缺少结果文件：{}".format(spec.label, path)
            )

    summary = _read_json(summary_path)
    config = _read_json(config_path)
    if str(summary.get("ablation_arm", "")) != spec.arm:
        raise ValueError(
            "{}结果中的ablation_arm不正确".format(spec.label)
        )
    if str(summary.get("aggregation_mode", "")) != spec.aggregation_mode:
        raise ValueError("{}结果聚合模式不正确".format(spec.label))
    if str(summary.get("local_objective", "")) != spec.local_objective:
        raise ValueError("{}结果本地目标不正确".format(spec.label))
    if int(config.get("distance_norm", 0)) != spec.distance_norm:
        raise ValueError("{}结果距离范数不正确".format(spec.label))

    final_test = summary.get("final_test_metrics")
    if not isinstance(final_test, dict):
        raise ValueError("{}缺少final_test_metrics".format(spec.label))
    final_validation = summary.get("final_validation_metrics")
    if not isinstance(final_validation, dict):
        raise ValueError(
            "{}缺少final_validation_metrics".format(spec.label)
        )
    return {
        "arm": spec.arm,
        "label": spec.label,
        "result_dir": str(result_dir),
        "aggregation_mode": spec.aggregation_mode,
        "local_objective": spec.local_objective,
        "distance_norm": spec.distance_norm,
        "partition_hash": str(summary.get("partition_hash", "")),
        "topology_schedule_hash": str(
            summary.get("topology_schedule_hash", "")
        ),
        "initial_model_hash": str(
            summary.get("initial_model_hash", "")
        ),
        "summary": summary,
        "config": config,
        "runtime_seconds": _metrics_runtime_seconds(metrics_path),
        "test_mrr": _finite_metric(final_test, "mrr", spec.arm),
        "test_hits_at_3": _finite_metric(
            final_test, "hits_at_3", spec.arm
        ),
        "test_hits_at_10": _finite_metric(
            final_test, "hits_at_10", spec.arm
        ),
        "validation_mrr": _finite_metric(
            final_validation, "mrr", spec.arm
        ),
        "best_round": int(summary.get("best_round", 0)),
    }


def _validate_result_comparability(
    records: Sequence[Mapping[str, object]],
    contract: Mapping[str, object],
    expected_specs: Sequence[ThreeArmSpec] = THREE_ARM_SPECS,
) -> None:
    """确认指定实验臂确实满足同数据、同MAT和同训练预算。"""

    if len(records) != len(expected_specs):
        raise ValueError(
            "结果汇总必须同时提供{}个约定实验臂".format(
                len(expected_specs)
            )
        )
    reference_summary = records[0]["summary"]
    if not isinstance(reference_summary, dict):
        raise TypeError("结果summary必须是字典")
    mismatches: Dict[str, object] = {}
    for field in RESULT_EQUALITY_FIELDS:
        reference_value = reference_summary.get(field)
        values = {
            str(record["arm"]): record["summary"].get(field)
            for record in records
        }
        if any(value != reference_value for value in values.values()):
            mismatches[field] = values
    if mismatches:
        raise ValueError(
            "三臂结果不可直接比较：{}".format(
                json.dumps(
                    mismatches,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )

    shared = contract.get("shared_contract", {})
    if not isinstance(shared, dict):
        raise TypeError("公平合同shared_contract必须是字典")
    config_mismatches: Dict[str, object] = {}
    for record in records:
        arm = str(record["arm"])
        config = record.get("config", {})
        if not isinstance(config, dict):
            raise TypeError("{}配置快照必须是字典".format(arm))
        differences = {
            field: {
                "expected": expected_value,
                "actual": config.get(field),
            }
            for field, expected_value in shared.items()
            if config.get(field) != expected_value
        }
        if differences:
            config_mismatches[arm] = differences
    if config_mismatches:
        raise ValueError(
            "结果配置快照不符合三臂公平合同：{}".format(
                json.dumps(
                    config_mismatches,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )

    expected_partition = str(shared["expected_partition_hash"])
    expected_schedule = str(
        shared["expected_topology_schedule_hash"]
    )
    if str(reference_summary.get("partition_hash")) != expected_partition:
        raise ValueError("正式结果客户端划分哈希与公平合同不一致")
    if (
        str(reference_summary.get("topology_schedule_hash"))
        != expected_schedule
    ):
        raise ValueError("正式结果MAT调度哈希与公平合同不一致")
    if not str(reference_summary.get("initial_model_hash", "")).strip():
        raise ValueError("正式结果缺少初始模型哈希")


def _effect_record(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    threshold: float,
) -> Dict[str, object]:
    """计算相邻实验臂的MRR和Hits@3差值及初筛结论。"""

    mrr_delta = float(candidate["test_mrr"]) - float(
        baseline["test_mrr"]
    )
    hits_delta = float(candidate["test_hits_at_3"]) - float(
        baseline["test_hits_at_3"]
    )
    return {
        "baseline_arm": str(baseline["arm"]),
        "candidate_arm": str(candidate["arm"]),
        "mrr_delta": mrr_delta,
        "hits_at_3_delta": hits_delta,
        "mrr_threshold": float(threshold),
        "hits_at_3_non_degraded": hits_delta >= 0.0,
        "passes_screening": (
            mrr_delta >= float(threshold) and hits_delta >= 0.0
        ),
    }


def compare_three_arm_results(
    package_dir: Path,
    result_dirs: Mapping[str, Path],
    mrr_threshold: float = 0.003,
) -> Dict[str, object]:
    """审计A、B、C结果可比性并生成相邻实验效应汇总。"""

    if float(mrr_threshold) < 0.0:
        raise ValueError("mrr_threshold不能小于0")
    contract = validate_three_arm_configs(package_dir)
    records = []
    for spec in THREE_ARM_SPECS:
        if spec.arm not in result_dirs:
            raise ValueError("缺少{}结果目录".format(spec.label))
        records.append(
            _load_result_arm(spec, Path(result_dirs[spec.arm]))
        )
    _validate_result_comparability(records, contract)

    effect_a_to_b = _effect_record(
        records[0], records[1], float(mrr_threshold)
    )
    effect_b_to_c = _effect_record(
        records[1], records[2], float(mrr_threshold)
    )
    winner = max(records, key=lambda item: float(item["test_mrr"]))
    should_run_dense_fede = bool(
        effect_b_to_c["passes_screening"]
    )
    return {
        "status": "comparable",
        "schema_version": 1,
        "suite": ABLATION_SUITE_NAME,
        "contract_hash": contract["contract_hash"],
        "comparison_basis": {
            "same_partition": True,
            "same_mat_schedule": True,
            "same_initial_model": True,
            "same_training_budget": True,
            "global_head_tail_filtered_evaluation": True,
            "single_training_seed": int(
                contract["shared_contract"]["random_seed"]
            ),
        },
        "arms": [
            {
                key: value
                for key, value in record.items()
                if key not in {"summary", "config"}
            }
            for record in records
        ],
        "effects": {
            "row_mask_effect_B_minus_A": effect_a_to_b,
            "fede_objective_bundle_C_minus_B": effect_b_to_c,
        },
        "winner_by_single_seed_test_mrr": str(winner["arm"]),
        "run_dense_fede_fair_next": should_run_dense_fede,
        "decision_note": (
            "C相对B通过MRR与Hits@3初筛，应补充D：dense+FedE-fair"
            if should_run_dense_fede
            else "C相对B未通过初筛，暂不补充高优先级D臂"
        ),
        "limitations": [
            "当前只有一个训练随机种子，不能替代三种子均值和置信区间。",
            "B到C同时改变L1距离、尾负采样和自对抗损失，不能拆成单因素结论。",
        ],
    }


def render_comparison_markdown(
    comparison: Mapping[str, object],
) -> str:
    """把三臂比较结果写成简体中文大白话Markdown。"""

    records = {
        str(item["arm"]): item
        for item in comparison.get("arms", [])
    }
    effects = comparison.get("effects", {})
    row_effect = effects.get("row_mask_effect_B_minus_A", {})
    objective_effect = effects.get(
        "fede_objective_bundle_C_minus_B", {}
    )
    lines = [
        "# V2同MAT三臂消融结果",
        "",
        "## 一句话结论",
        "",
        str(comparison.get("decision_note", "")) + "。",
        "",
        "## 三组最终成绩",
        "",
        "| 实验臂 | 测试MRR | Hits@3 | Hits@10 | 最佳轮次 | 累计耗时（秒） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in THREE_ARM_SPECS:
        item = records[spec.arm]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {} | {:.2f} |".format(
                spec.label,
                float(item["test_mrr"]),
                float(item["test_hits_at_3"]),
                float(item["test_hits_at_10"]),
                int(item["best_round"]),
                float(item["runtime_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "## 大白话怎么看",
            "",
            "- B减A只改变聚合方式，用来回答“按行聚合有没有缓解完整参数平均造成的更新稀释”。",
            "- C减B替换整套FedE本地目标，用来回答“L1、尾负采样和自对抗损失这一整包是否有帮助”。",
            "",
            "B相对A的MRR变化为`{:+.6f}`，Hits@3变化为`{:+.6f}`。".format(
                float(row_effect.get("mrr_delta", float("nan"))),
                float(
                    row_effect.get(
                        "hits_at_3_delta", float("nan")
                    )
                ),
            ),
            "",
            "C相对B的MRR变化为`{:+.6f}`，Hits@3变化为`{:+.6f}`。".format(
                float(
                    objective_effect.get(
                        "mrr_delta", float("nan")
                    )
                ),
                float(
                    objective_effect.get(
                        "hits_at_3_delta", float("nan")
                    )
                ),
            ),
            "",
            "## 为什么这三组能比较",
            "",
            "- 客户端划分哈希相同；",
            "- 200行MAT调度哈希相同；",
            "- 初始模型参数哈希相同；",
            "- 通信轮数、本地epoch、批次、负样本数量和评估方式相同。",
            "",
            "## 限制",
            "",
            "- 当前只代表种子42，胜出方案仍需运行三个随机种子；",
            "- C相对B同时改变三个本地训练因素，不能说提升只由其中某一个因素造成。",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_outputs(
    output_dir: Path,
    comparison: Mapping[str, object],
) -> Dict[str, str]:
    """写出三臂JSON、CSV和大白话Markdown结果。"""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_summary.json"
    csv_path = output_dir / "comparison_metrics.csv"
    markdown_path = output_dir / "comparison_report.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            comparison,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "arm",
            "label",
            "aggregation_mode",
            "local_objective",
            "distance_norm",
            "test_mrr",
            "test_hits_at_3",
            "test_hits_at_10",
            "validation_mrr",
            "best_round",
            "runtime_seconds",
            "result_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in comparison.get("arms", []):
            writer.writerow({field: record.get(field) for field in fields})
    markdown_path.write_text(
        render_comparison_markdown(comparison),
        encoding="utf-8",
    )
    return {
        "comparison_summary": str(json_path),
        "comparison_metrics": str(csv_path),
        "comparison_report": str(markdown_path),
    }
