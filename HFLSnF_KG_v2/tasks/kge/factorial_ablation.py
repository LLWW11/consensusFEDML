"""V2同MAT四臂二乘二消融的配置合同、结果审计和汇总工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .ablation import (
    ABLATION_SUITE_NAME,
    SHARED_CONTRACT_FIELDS,
    THREE_ARM_SPECS,
    ThreeArmSpec,
    _canonical_hash,
    _effect_record,
    _load_result_arm,
    _require_fields,
    _validate_arm_semantics,
    _validate_result_comparability,
    load_fedml_yaml,
    sha256_file,
    validate_three_arm_configs,
)


D_ARM_SPEC = ThreeArmSpec(
    arm="dense_fede_fair",
    label="D：dense+FedE-fair",
    config_filename=(
        "server_fb15k237_hflsnf_dynamic_mat_dense_fede_fair_cuda.yaml"
    ),
    aggregation_mode="dense_triple_weighted",
    local_objective="fede_self_adversarial",
    distance_norm=1,
)

# 保留A、B、C的历史顺序，并把D放在最后，避免破坏已有三臂入口。
FOUR_ARM_SPECS: Tuple[ThreeArmSpec, ...] = THREE_ARM_SPECS + (D_ARM_SPEC,)


def _validate_d_fede_parameters(config: Mapping[str, object]) -> None:
    """校验D臂使用与C臂完全一致的FedE公平预算目标参数。"""

    if float(config.get("fede_gamma", 0.0)) != 10.0:
        raise ValueError("D臂fede_gamma必须为10.0")
    if float(config.get("adversarial_temperature", 0.0)) != 1.0:
        raise ValueError("D臂adversarial_temperature必须为1.0")


def validate_four_arm_configs(
    package_dir: Path,
    d_config_path: Optional[Path] = None,
) -> Dict[str, object]:
    """在已有三臂公平合同上追加D臂并校验全部共享字段。"""

    package_dir = Path(package_dir).expanduser().resolve()
    three_contract = validate_three_arm_configs(package_dir)
    d_path = (
        Path(d_config_path).expanduser().resolve()
        if d_config_path is not None
        else (
            package_dir / "configs" / D_ARM_SPEC.config_filename
        ).resolve()
    )
    if not d_path.is_file():
        raise FileNotFoundError("找不到D臂配置：{}".format(d_path))

    d_config = load_fedml_yaml(d_path)
    _require_fields(
        d_config,
        SHARED_CONTRACT_FIELDS
        + (
            "ablation_arm",
            "aggregation_mode",
            "local_objective",
            "distance_norm",
            "run_name",
            "fede_gamma",
            "adversarial_temperature",
        ),
        D_ARM_SPEC.arm,
    )
    _validate_arm_semantics(D_ARM_SPEC, d_config)
    _validate_d_fede_parameters(d_config)

    shared = three_contract["shared_contract"]
    if not isinstance(shared, dict):
        raise TypeError("三臂合同中的shared_contract必须是字典")
    differences = {
        field: {
            "expected": shared[field],
            "actual": d_config[field],
        }
        for field in SHARED_CONTRACT_FIELDS
        if d_config[field] != shared[field]
    }
    if differences:
        raise ValueError(
            "D臂破坏四臂公平合同：{}".format(
                json.dumps(
                    differences,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )

    arms = list(three_contract["arms"])
    arms.append(
        {
            **D_ARM_SPEC.summary(),
            "config_path": str(d_path),
            "config_sha256": sha256_file(d_path),
            "run_name": str(d_config["run_name"]),
        }
    )
    contract_material = {
        "suite": ABLATION_SUITE_NAME,
        "shared_contract": shared,
        "data_file_hashes": {
            name: values["sha256"]
            for name, values in three_contract["data_files"].items()
        },
        "mat_sha256": three_contract["mat_file"]["sha256"],
        "arm_config_hashes": {
            str(arm["arm"]): str(arm["config_sha256"])
            for arm in arms
        },
    }
    return {
        "status": "valid",
        "schema_version": 1,
        "suite": ABLATION_SUITE_NAME,
        "design": "2x2_aggregation_by_local_objective",
        "contract_hash": _canonical_hash(contract_material),
        "shared_contract": dict(shared),
        "data_files": dict(three_contract["data_files"]),
        "mat_file": dict(three_contract["mat_file"]),
        "arms": arms,
        "allowed_differences": {
            "aggregation_factor": (
                "dense_triple_weighted或row_mask_presence"
            ),
            "local_objective_factor": (
                "margin_ranking或fede_self_adversarial"
            ),
        },
    }


def _interaction_record(
    a: Mapping[str, object],
    b: Mapping[str, object],
    c: Mapping[str, object],
    d: Mapping[str, object],
) -> Dict[str, object]:
    """计算二乘二中聚合方式与本地目标的差中差交互项。"""

    mrr_interaction = (
        (float(c["test_mrr"]) - float(b["test_mrr"]))
        - (float(d["test_mrr"]) - float(a["test_mrr"]))
    )
    hits_interaction = (
        (
            float(c["test_hits_at_3"])
            - float(b["test_hits_at_3"])
        )
        - (
            float(d["test_hits_at_3"])
            - float(a["test_hits_at_3"])
        )
    )
    if mrr_interaction > 0.003:
        interpretation = "正交互：行级聚合与FedE目标组合后有额外收益"
    elif mrr_interaction < -0.003:
        interpretation = "负交互：两项改进存在部分重叠或相互抵消"
    else:
        interpretation = "交互接近零：两项改进在当前种子下近似可加"
    return {
        "formula": "(C-B)-(D-A)，等价于(C-D)-(B-A)",
        "mrr_interaction": mrr_interaction,
        "hits_at_3_interaction": hits_interaction,
        "descriptive_threshold": 0.003,
        "interpretation": interpretation,
    }


def compare_four_arm_results(
    package_dir: Path,
    result_dirs: Mapping[str, Path],
    mrr_threshold: float = 0.003,
) -> Dict[str, object]:
    """审计A、B、C、D结果并计算主效应、条件效应和交互项。"""

    if float(mrr_threshold) < 0.0:
        raise ValueError("mrr_threshold不能小于0")
    contract = validate_four_arm_configs(package_dir)
    expected = {spec.arm for spec in FOUR_ARM_SPECS}
    if set(result_dirs) != expected:
        raise ValueError(
            "四臂汇总必须且只能提供{}".format(
                "、".join(sorted(expected))
            )
        )

    records = [
        _load_result_arm(spec, Path(result_dirs[spec.arm]))
        for spec in FOUR_ARM_SPECS
    ]
    _validate_result_comparability(
        records,
        contract,
        expected_specs=FOUR_ARM_SPECS,
    )
    by_arm = {
        str(record["arm"]): record
        for record in records
    }
    a = by_arm["dense_margin"]
    b = by_arm["masked_margin"]
    c = by_arm["masked_fede_fair"]
    d = by_arm["dense_fede_fair"]
    effects = {
        "row_mask_under_margin_B_minus_A": _effect_record(
            a, b, float(mrr_threshold)
        ),
        "fede_under_dense_D_minus_A": _effect_record(
            a, d, float(mrr_threshold)
        ),
        "fede_under_mask_C_minus_B": _effect_record(
            b, c, float(mrr_threshold)
        ),
        "row_mask_under_fede_C_minus_D": _effect_record(
            d, c, float(mrr_threshold)
        ),
    }
    winner = max(records, key=lambda item: float(item["test_mrr"]))
    return {
        "status": "comparable",
        "schema_version": 1,
        "suite": ABLATION_SUITE_NAME,
        "design": "2x2_aggregation_by_local_objective",
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
        "effects": effects,
        "interaction": _interaction_record(a, b, c, d),
        "winner_by_single_seed_test_mrr": str(winner["arm"]),
        "limitations": [
            "当前差值只来自一个训练随机种子，不能当成显著性检验。",
            "FedE目标因子是一整套L1、尾负采样和自对抗损失，不能继续拆成单因素结论。",
        ],
    }


def render_factorial_markdown(
    comparison: Mapping[str, object],
) -> str:
    """把四臂二乘二结果渲染成简体中文大白话报告。"""

    records = {
        str(item["arm"]): item
        for item in comparison.get("arms", [])
    }
    effects = comparison.get("effects", {})
    interaction = comparison.get("interaction", {})
    lines = [
        "# V2同MAT四臂二乘二结果",
        "",
        "## 一句话结论",
        "",
        "当前单种子下，测试MRR最高的是`{}`。{}".format(
            comparison.get("winner_by_single_seed_test_mrr", ""),
            interaction.get("interpretation", ""),
        ),
        "",
        "## 四组最终成绩",
        "",
        "| 实验臂 | 聚合 | 本地目标 | 测试MRR | Hits@3 | Hits@10 | 最佳轮次 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for spec in FOUR_ARM_SPECS:
        item = records[spec.arm]
        lines.append(
            "| {} | `{}` | `{}` | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                spec.label,
                item["aggregation_mode"],
                item["local_objective"],
                float(item["test_mrr"]),
                float(item["test_hits_at_3"]),
                float(item["test_hits_at_10"]),
                int(item["best_round"]),
            )
        )

    effect_labels = (
        ("row_mask_under_margin_B_minus_A", "B-A：margin下换成行级聚合"),
        ("fede_under_dense_D_minus_A", "D-A：dense下换成FedE目标"),
        ("fede_under_mask_C_minus_B", "C-B：行级聚合下换成FedE目标"),
        ("row_mask_under_fede_C_minus_D", "C-D：FedE目标下换成行级聚合"),
    )
    lines.extend(
        [
            "",
            "## 四个最关键的差值",
            "",
            "| 问题 | MRR变化 | Hits@3变化 |",
            "|---|---:|---:|",
        ]
    )
    for key, label in effect_labels:
        effect = effects.get(key, {})
        lines.append(
            "| {} | {:+.6f} | {:+.6f} |".format(
                label,
                float(effect.get("mrr_delta", float("nan"))),
                float(
                    effect.get("hits_at_3_delta", float("nan"))
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 大白话怎么看交互",
            "",
            "交互项采用`(C-B)-(D-A)`，MRR交互为`{:+.6f}`，"
            "Hits@3交互为`{:+.6f}`。".format(
                float(
                    interaction.get(
                        "mrr_interaction", float("nan")
                    )
                ),
                float(
                    interaction.get(
                        "hits_at_3_interaction", float("nan")
                    )
                ),
            ),
            "",
            str(interaction.get("interpretation", "")) + "。",
            "",
            "它回答的是：FedE目标放到行级聚合上以后，收益是否超过"
            "“FedE目标自身收益 + 行级聚合自身收益”的简单相加。",
            "",
            "## 限制",
            "",
            "- 当前只有种子42，交互项是描述性结果，不是统计显著性结论。",
            "- FedE目标仍是一整套训练改动，不能把收益只归因于L1或自对抗损失。",
            "",
        ]
    )
    return "\n".join(lines)


def write_factorial_outputs(
    output_dir: Path,
    comparison: Mapping[str, object],
) -> Dict[str, str]:
    """写出四臂JSON、CSV和简体中文Markdown报告。"""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "factorial_summary.json"
    csv_path = output_dir / "factorial_metrics.csv"
    markdown_path = output_dir / "factorial_report.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            comparison,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    fields = (
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
    )
    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in comparison.get("arms", []):
            writer.writerow(
                {field: record.get(field) for field in fields}
            )
    markdown_path.write_text(
        render_factorial_markdown(comparison),
        encoding="utf-8",
    )
    return {
        "factorial_summary": str(json_path),
        "factorial_metrics": str(csv_path),
        "factorial_report": str(markdown_path),
    }
