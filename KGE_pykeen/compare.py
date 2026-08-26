"""集中式自研、严格PyKEEN和原生PyKEEN结果对比入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


COMMON_CONFIG_FIELDS = (
    "random_seed",
    "dataset",
    "embedding_dim",
    "distance_norm",
    "epochs",
    "batch_size",
    "learning_rate",
    "negative_sample_count",
    "fede_gamma",
    "adversarial_temperature",
)


def _read_json(path: Path) -> Dict[str, object]:
    """读取一个UTF-8 JSON对象。"""

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON顶层必须是对象：{}".format(path))
    return payload


def _resolve_result(path: Path) -> Path:
    """把结果目录或summary.json路径统一为结果目录。"""

    resolved = Path(path).expanduser().resolve()
    if resolved.is_file() and resolved.name == "summary.json":
        resolved = resolved.parent
    if not (resolved / "summary.json").is_file():
        raise FileNotFoundError("结果目录缺少summary.json：{}".format(resolved))
    return resolved


def _file_sha256(path: Path) -> str:
    """计算文件字节的SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_result(path: Path) -> Dict[str, object]:
    """读取对比所需的结果摘要、配置、映射和合同。"""

    root = _resolve_result(path)
    contract_path = root / "comparison_contract.json"
    return {
        "root": str(root),
        "summary": _read_json(root / "summary.json"),
        "dataset_summary": _read_json(
            root / "dataset_summary.json"
        ),
        "config": _read_json(root / "config_snapshot.json"),
        "entity_mapping_sha256": _file_sha256(root / "entity2id.json"),
        "relation_mapping_sha256": _file_sha256(root / "relation2id.json"),
        "contract": (
            _read_json(contract_path) if contract_path.is_file() else {}
        ),
    }


def _validate_contracts(
    central: Mapping[str, object],
    matched: Mapping[str, object],
    native: Mapping[str, object],
) -> List[str]:
    """检查三个结果的共享配置、数据规模和映射是否一致。"""

    errors: List[str] = []
    arms = {
        "matched": matched,
        "native": native,
    }
    central_config = central["config"]
    central_summary = central["summary"]
    expected_modes = {
        "matched": "matched_recipe",
        "native": "pykeen_native",
    }
    for name, arm in arms.items():
        config = arm["config"]
        actual_mode = str(config.get("comparison_mode", ""))
        if actual_mode != expected_modes[name]:
            errors.append(
                "{}的comparison_mode应为{}，实际为{}".format(
                    name, expected_modes[name], actual_mode
                )
            )
        if central["dataset_summary"] != arm["dataset_summary"]:
            errors.append("{}的数据摘要不一致".format(name))
        for field in COMMON_CONFIG_FIELDS:
            if central_config.get(field) != config.get(field):
                errors.append(
                    "{}的配置字段{}不一致：{} != {}".format(
                        name,
                        field,
                        central_config.get(field),
                        config.get(field),
                    )
                )
        for mapping_field in (
            "entity_mapping_sha256",
            "relation_mapping_sha256",
        ):
            if central[mapping_field] != arm[mapping_field]:
                errors.append("{}的{}不一致".format(name, mapping_field))
        for count_field in (
            "dataset",
            "final_test_metrics",
        ):
            if count_field == "dataset":
                if central_summary.get(count_field) != arm[
                    "summary"
                ].get(count_field):
                    errors.append("{}的数据集名称不一致".format(name))
            else:
                central_count = central_summary.get(
                    count_field, {}
                ).get("evaluated_triple_count")
                arm_count = arm["summary"].get(
                    count_field, {}
                ).get("evaluated_triple_count")
                if central_count != arm_count:
                    errors.append("{}的完整测试规模不一致".format(name))
    matched_data = matched.get("contract", {}).get(
        "dataset_contract", {}
    )
    native_data = native.get("contract", {}).get(
        "dataset_contract", {}
    )
    for field in (
        "entity_mapping_sha256",
        "relation_mapping_sha256",
        "train_triples_sha256",
        "valid_triples_sha256",
        "test_triples_sha256",
    ):
        if not matched_data.get(field) or not native_data.get(field):
            errors.append("PyKEEN两臂缺少数据合同字段{}".format(field))
        elif matched_data[field] != native_data[field]:
            errors.append("PyKEEN两臂的数据合同字段{}不一致".format(field))
    return errors


def _metric_row(
    label: str,
    result: Mapping[str, object],
) -> Dict[str, object]:
    """提取一个结果的canonical完整测试指标。"""

    metrics = result["summary"].get("final_test_metrics", {})
    return {
        "label": label,
        "mrr": metrics.get("mrr"),
        "hits_at_1": metrics.get("hits_at_1"),
        "hits_at_3": metrics.get("hits_at_3"),
        "hits_at_10": metrics.get("hits_at_10"),
        "mean_rank": metrics.get("mean_rank"),
        "triple_count": metrics.get("evaluated_triple_count"),
        "query_count": metrics.get("evaluated_query_count"),
    }


def _format_number(value: object) -> str:
    """把报告数值格式化为固定六位小数。"""

    if value is None:
        return "缺失"
    try:
        return "{:.6f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _render_markdown(
    rows: Sequence[Mapping[str, object]],
    errors: Sequence[str],
    native_differences: Sequence[str],
) -> str:
    """渲染简体中文三臂对比报告。"""

    lines = [
        "# KGE自研框架与PyKEEN可信对照报告",
        "",
        "## 合同状态",
        "",
        "- 状态：{}".format("通过" if not errors else "失败"),
    ]
    for error in errors:
        lines.append("- 错误：{}".format(error))
    lines.extend(
        [
            "",
            "数据合同说明：历史central结果未保存三元组划分哈希，因此central与另外两臂校验数据摘要和完整映射；matched与native另外逐项校验训练、验证、测试划分SHA-256。",
            "",
            "## Canonical完整测试指标",
            "",
            "| 实验臂 | MRR | Hits@1 | Hits@3 | Hits@10 | 平均排名 | 三元组数 | 查询数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {mrr} | {h1} | {h3} | {h10} | {mr} | {tc} | {qc} |".format(
                label=row["label"],
                mrr=_format_number(row["mrr"]),
                h1=_format_number(row["hits_at_1"]),
                h3=_format_number(row["hits_at_3"]),
                h10=_format_number(row["hits_at_10"]),
                mr=_format_number(row["mean_rank"]),
                tc=_format_number(row["triple_count"]),
                qc=_format_number(row["query_count"]),
            )
        )
    lines.extend(["", "## PyKEEN原生模式差异", ""])
    for difference in native_differences:
        lines.append("- {}".format(difference))
    lines.extend(
        [
            "",
            "严格模式用于张量级正确性验证；原生模式用于标准库端到端对照，不能表述为严格等价复现。",
            "",
        ]
    )
    return "\n".join(lines)


def compare_results(
    central_path: Path,
    matched_path: Path,
    native_path: Path,
    output_dir: Path,
) -> Tuple[Path, Path, Dict[str, object]]:
    """校验三臂合同并写出JSON和简体中文Markdown报告。"""

    central = _load_result(central_path)
    matched = _load_result(matched_path)
    native = _load_result(native_path)
    errors = _validate_contracts(central, matched, native)
    rows = [
        _metric_row("自研KGE_central", central),
        _metric_row("PyKEEN严格配方", matched),
        _metric_row("PyKEEN原生流水线", native),
    ]
    native_differences = native["contract"].get(
        "native_semantic_differences", []
    )
    payload = {
        "passed": not errors,
        "errors": errors,
        "results": rows,
        "native_semantic_differences": native_differences,
        "data_contract_scope": {
            "central": "dataset_summary_and_mapping_files",
            "matched_native": "mapping_and_all_split_sha256",
        },
        "result_directories": {
            "central": central["root"],
            "matched": matched["root"],
            "native": native["root"],
        },
    }
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison_report.json"
    markdown_path = output_dir / "comparison_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write(
            _render_markdown(
                rows,
                errors,
                native_differences,
            )
        )
    return json_path, markdown_path, payload


def build_argument_parser() -> argparse.ArgumentParser:
    """创建三臂结果对比命令行解析器。"""

    parser = argparse.ArgumentParser(
        description="生成KGE自研与PyKEEN双口径对比报告"
    )
    parser.add_argument("--central", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """解析参数并生成三臂对比报告。"""

    args = build_argument_parser().parse_args(argv)
    json_path, markdown_path, payload = compare_results(
        args.central,
        args.matched,
        args.native,
        args.output_dir,
    )
    print("合同状态：{}".format("通过" if payload["passed"] else "失败"))
    print("JSON报告：{}".format(json_path))
    print("Markdown报告：{}".format(markdown_path))


if __name__ == "__main__":
    main()
