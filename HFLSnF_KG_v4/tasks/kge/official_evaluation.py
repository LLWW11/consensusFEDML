"""最佳TransE检查点的完整官方测试合同与报告。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

from .data import KnowledgeGraphDataset
from .directional import run_directional_diagnostic


def _load_json_object(path: Path) -> Dict[str, object]:
    """读取顶层必须为对象的UTF-8 JSON文件。"""

    normalized_path = Path(path).expanduser().resolve()
    if not normalized_path.is_file():
        raise FileNotFoundError("找不到JSON文件：{}".format(normalized_path))
    with normalized_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON顶层必须是对象：{}".format(normalized_path))
    return payload


def _finite_optional_number(
    payload: Mapping[str, object],
    field_name: str,
) -> float:
    """读取可选有限数值字段，不存在时返回NaN。"""

    value = payload.get(field_name)
    if value is None:
        return float("nan")
    numeric = float(value)
    if math.isnan(numeric):
        return float("nan")
    if not math.isfinite(numeric):
        raise ValueError("字段{}不是有限数".format(field_name))
    return numeric


def _file_sha256(path: Path) -> str:
    """流式计算文件SHA-256，用于确认缓存属于当前检查点。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metric_block(
    payload: Mapping[str, object],
    block_name: str,
    expected_query_count: int,
) -> None:
    """校验一个方向评估指标块的有限性、范围和查询数量。"""

    block = payload.get(block_name)
    if not isinstance(block, dict):
        raise ValueError("方向评估缓存缺少{}指标块".format(block_name))
    for field_name in ("mrr", "hits_at_1", "hits_at_3", "hits_at_10"):
        value = float(block[field_name])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(
                "方向评估缓存{}.{}不是有效比例".format(
                    block_name,
                    field_name,
                )
            )
    mean_rank = float(block["mean_rank"])
    if not math.isfinite(mean_rank) or mean_rank < 1.0:
        raise ValueError(
            "方向评估缓存{}.mean_rank不是有效排名".format(block_name)
        )
    if int(block["evaluated_query_count"]) != int(expected_query_count):
        raise ValueError(
            "方向评估缓存{}查询数不一致".format(block_name)
        )


def _load_reusable_directional_summary(
    dataset: KnowledgeGraphDataset,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    """读取并严格校验可安全复用的完整方向评估摘要。"""

    summary_path = output_dir / "directional_summary.json"
    if not summary_path.is_file():
        return None
    required_artifacts = (
        summary_path,
        output_dir / "query_ranks.csv",
        output_dir / "relation_metrics.csv",
        output_dir / "方向诊断报告.md",
    )
    missing = [str(path) for path in required_artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "方向评估缓存不完整，缺少：{}".format(missing)
        )

    summary = _load_json_object(summary_path)
    if summary.get("status") != "completed":
        raise ValueError("方向评估缓存状态不是completed")
    if bool(summary.get("training_performed", True)):
        raise ValueError("方向评估缓存错误地记录了训练行为")
    if str(summary.get("device", "")) != str(torch.device(device)):
        raise ValueError("方向评估缓存设备与当前正式评估设备不一致")

    expected_triple_count = int(dataset.test_triples.shape[0])
    if not bool(summary.get("full_official_test", False)):
        raise ValueError("方向评估缓存不是完整官方测试")
    if int(summary.get("selected_triple_count", -1)) != expected_triple_count:
        raise ValueError("方向评估缓存的实际三元组数不一致")
    if int(summary.get(
        "official_test_triple_count", -1
    )) != expected_triple_count:
        raise ValueError("方向评估缓存的官方三元组数不一致")

    cached_checkpoint = Path(
        str(summary.get("checkpoint_path", ""))
    ).expanduser().resolve()
    if cached_checkpoint != checkpoint_path:
        raise ValueError("方向评估缓存对应的检查点路径不一致")
    if str(summary.get("checkpoint_sha256", "")) != _file_sha256(
        checkpoint_path
    ):
        raise ValueError("方向评估缓存对应的检查点哈希不一致")

    _validate_metric_block(summary, "head_metrics", expected_triple_count)
    _validate_metric_block(summary, "tail_metrics", expected_triple_count)
    _validate_metric_block(
        summary,
        "combined_metrics",
        expected_triple_count * 2,
    )
    return summary


def build_official_evaluation_contract(
    training_summary: Mapping[str, object],
    directional_summary: Mapping[str, object],
    result_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    """合并训练与方向摘要，并严格确认使用完整官方测试集。"""

    selected_count = int(directional_summary["selected_triple_count"])
    official_count = int(directional_summary["official_test_triple_count"])
    full_official_test = bool(directional_summary["full_official_test"])
    if not full_official_test or selected_count != official_count:
        raise RuntimeError(
            "完整官方测试合同失败：实际{}条，官方{}条".format(
                selected_count,
                official_count,
            )
        )
    combined_mrr = float(
        directional_summary["combined_metrics"]["mrr"]
    )
    centralized_reference = _finite_optional_number(
        training_summary,
        "centralized_reference_test_mrr",
    )
    subset_metrics = training_summary.get("final_test_metrics", {})
    subset_mrr = (
        float(subset_metrics["mrr"])
        if isinstance(subset_metrics, dict) and "mrr" in subset_metrics
        else float("nan")
    )
    return {
        "status": "passed",
        "training_performed": False,
        "result_dir": str(Path(result_dir).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "checkpoint_path": directional_summary["checkpoint_path"],
        "checkpoint_sha256": directional_summary["checkpoint_sha256"],
        "dataset": training_summary.get("dataset", ""),
        "best_round": int(training_summary.get("best_round", 0)),
        "best_validation_mrr": float(
            training_summary.get(
                "best_validation_mrr_during_training",
                float("nan"),
            )
        ),
        "screening_test_triple_count": int(
            subset_metrics.get("evaluated_triple_count", 0)
        )
        if isinstance(subset_metrics, dict)
        else 0,
        "screening_test_mrr": subset_mrr,
        "official_test_triple_count": official_count,
        "official_test_query_count": int(
            directional_summary["combined_metrics"][
                "evaluated_query_count"
            ]
        ),
        "head_metrics": directional_summary["head_metrics"],
        "tail_metrics": directional_summary["tail_metrics"],
        "combined_metrics": directional_summary["combined_metrics"],
        "mrr_delta_vs_screening_subset": (
            combined_mrr - subset_mrr
            if math.isfinite(subset_mrr)
            else float("nan")
        ),
        "centralized_reference_test_mrr": centralized_reference,
        "mrr_delta_vs_centralized": (
            combined_mrr - centralized_reference
            if math.isfinite(centralized_reference)
            else float("nan")
        ),
        "full_official_test": True,
    }


def write_official_evaluation_report(
    contract: Mapping[str, object],
    output_dir: Path,
) -> None:
    """写出机器可读合同和简体中文完整官方测试报告。"""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "official_evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(dict(contract), handle, ensure_ascii=False, indent=2)
    report = "\n".join(
        [
            "# 最佳模型完整官方测试报告",
            "",
            "本次测试只读取最佳检查点，没有重新训练或修改模型。",
            "",
            "- 训练最佳轮次：`{}`".format(contract["best_round"]),
            "- 官方测试三元组数：`{}`".format(
                contract["official_test_triple_count"]
            ),
            "- 官方测试查询数：`{}`".format(
                contract["official_test_query_count"]
            ),
            "- 头预测MRR：`{:.6f}`".format(
                float(contract["head_metrics"]["mrr"])
            ),
            "- 尾预测MRR：`{:.6f}`".format(
                float(contract["tail_metrics"]["mrr"])
            ),
            "- 综合MRR：`{:.6f}`".format(
                float(contract["combined_metrics"]["mrr"])
            ),
            "- 综合Hits@1：`{:.6f}`".format(
                float(contract["combined_metrics"]["hits_at_1"])
            ),
            "- 综合Hits@3：`{:.6f}`".format(
                float(contract["combined_metrics"]["hits_at_3"])
            ),
            "- 综合Hits@10：`{:.6f}`".format(
                float(contract["combined_metrics"]["hits_at_10"])
            ),
            "- 综合平均排名：`{:.2f}`".format(
                float(contract["combined_metrics"]["mean_rank"])
            ),
            "",
            "## 合同结论",
            "",
            "已确认使用全部官方测试三元组和双向filtered排名。",
            "",
        ]
    )
    (output_dir / "完整官方测试报告.md").write_text(
        report,
        encoding="utf-8",
    )


def run_best_checkpoint_official_evaluation(
    dataset: KnowledgeGraphDataset,
    result_dir: Path,
    output_dir: Path,
    device: torch.device,
    query_batch_size: int,
    candidate_batch_size: int,
    progress_every: int,
) -> Dict[str, object]:
    """对结果目录最佳检查点执行完整头尾测试并写出合同报告。"""

    result_dir = Path(result_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    training_summary = _load_json_object(result_dir / "summary.json")
    checkpoint_path = result_dir / "model_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "找不到最佳模型检查点：{}".format(checkpoint_path)
        )
    directional_summary = _load_reusable_directional_summary(
        dataset,
        checkpoint_path,
        output_dir,
        device,
    )
    if directional_summary is None:
        directional_summary = run_directional_diagnostic(
            dataset=dataset,
            checkpoint=checkpoint_path,
            output_dir=output_dir,
            device=device,
            max_triples=0,
            selection_seed=42,
            query_batch_size=int(query_batch_size),
            candidate_batch_size=int(candidate_batch_size),
            progress_every=int(progress_every),
            distance_norm_override=0,
        )
    else:
        print(
            "复用已完成的方向评估摘要：{}".format(
                output_dir / "directional_summary.json"
            )
        )
    contract = build_official_evaluation_contract(
        training_summary,
        directional_summary,
        result_dir,
        output_dir,
    )
    write_official_evaluation_report(contract, output_dir)
    return contract
