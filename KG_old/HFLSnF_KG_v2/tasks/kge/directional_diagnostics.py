"""从TransE最佳检查点执行无需重训的头尾方向与逐关系诊断。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .data import IdTriple, KnowledgeGraphDataset
from .evaluation_bridge import (
    BatchedFilteredTransEEvaluator,
    hash_triples,
    load_project_embedding_bundle,
    metrics_from_ranks,
)


DESIGNED_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("dense_margin", "masked_margin", "B-A：margin下行级聚合"),
    ("dense_margin", "dense_fede_fair", "D-A：dense下FedE目标"),
    ("masked_margin", "masked_fede_fair", "C-B：masked下FedE目标"),
    ("dense_fede_fair", "masked_fede_fair", "C-D：FedE下行级聚合"),
)


def select_test_triples(
    dataset: KnowledgeGraphDataset,
    maximum: int,
    seed: int,
) -> Tuple[IdTriple, ...]:
    """按固定种子从官方测试集选择诊断三元组，0表示使用全部。"""

    triples = tuple(
        tuple(int(value) for value in row)
        for row in dataset.test_triples.tolist()
    )
    if int(maximum) <= 0 or int(maximum) >= len(triples):
        return triples
    rng = np.random.RandomState(int(seed))
    indices = np.sort(
        rng.choice(len(triples), size=int(maximum), replace=False)
    )
    return tuple(triples[int(index)] for index in indices)


def _inverse_mapping(mapping: Mapping[str, int]) -> Dict[int, str]:
    """把名称到编号的映射反转，并校验编号不存在重复。"""

    inverse = {int(value): str(key) for key, value in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("名称编号映射包含重复编号")
    return inverse


def _read_json(path: Path) -> Dict[str, object]:
    """读取UTF-8 JSON对象并校验顶层类型。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("{}顶层必须是JSON对象".format(path))
    return payload


def validate_checkpoint_fingerprints(
    result_dirs: Mapping[str, Path],
) -> Dict[str, object]:
    """确认多个训练结果使用相同划分、MAT调度和初始模型。"""

    if len(result_dirs) < 2:
        raise ValueError("方向诊断至少需要两个模型结果")
    fields = (
        "partition_hash",
        "topology_schedule_hash",
        "initial_model_hash",
    )
    summaries: Dict[str, Dict[str, object]] = {}
    for arm, result_dir in result_dirs.items():
        summary_path = Path(result_dir).expanduser().resolve() / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                "{}缺少summary.json：{}".format(arm, summary_path)
            )
        summaries[str(arm)] = _read_json(summary_path)

    reference_arm = next(iter(summaries))
    reference = summaries[reference_arm]
    mismatches: Dict[str, object] = {}
    for field in fields:
        values = {
            arm: summary.get(field)
            for arm, summary in summaries.items()
        }
        if any(value != reference.get(field) for value in values.values()):
            mismatches[field] = values
    if mismatches:
        raise ValueError(
            "检查点不满足同口径诊断条件：{}".format(
                json.dumps(
                    mismatches,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )
    return {
        field: reference.get(field)
        for field in fields
    }


def evaluate_directional_ranks(
    dataset: KnowledgeGraphDataset,
    triples: Sequence[IdTriple],
    result_dirs: Mapping[str, Path],
    device: torch.device,
    query_batch_size: int,
    candidate_batch_size: int,
    progress_every: int,
) -> Tuple[
    Dict[str, Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, Dict[str, np.ndarray]],
]:
    """逐模型计算头尾filtered排名，并返回摘要、明细和排名数组。"""

    evaluator = BatchedFilteredTransEEvaluator(
        dataset.num_entities,
        dataset.all_true_triples,
    )
    entity_names = _inverse_mapping(dataset.entity_to_id)
    relation_names = _inverse_mapping(dataset.relation_to_id)
    summaries: Dict[str, Dict[str, object]] = {}
    records: List[Dict[str, object]] = []
    rank_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    for arm, result_dir in result_dirs.items():
        bundle = load_project_embedding_bundle(
            str(arm),
            Path(result_dir),
            dataset,
        ).to(device)
        print("开始评估{}，设备={}".format(arm, device), flush=True)
        head_ranks = evaluator.evaluate_direction(
            bundle,
            triples,
            predict_head=True,
            query_batch_size=int(query_batch_size),
            candidate_batch_size=int(candidate_batch_size),
            progress_label="{}头预测".format(arm),
            progress_every=int(progress_every),
        )
        tail_ranks = evaluator.evaluate_direction(
            bundle,
            triples,
            predict_head=False,
            query_batch_size=int(query_batch_size),
            candidate_batch_size=int(candidate_batch_size),
            progress_label="{}尾预测".format(arm),
            progress_every=int(progress_every),
        )
        combined_ranks = np.concatenate([head_ranks, tail_ranks])
        summaries[str(arm)] = {
            "head": metrics_from_ranks(head_ranks),
            "tail": metrics_from_ranks(tail_ranks),
            "combined": metrics_from_ranks(combined_ranks),
            "checkpoint_path": str(bundle.checkpoint_path),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "distance_norm": int(bundle.distance_norm),
            "embedding_dim": int(bundle.entity_embeddings.shape[1]),
        }
        rank_arrays[str(arm)] = {
            "head": head_ranks,
            "tail": tail_ranks,
        }
        for triple_index, triple in enumerate(triples):
            head, relation, tail = triple
            for direction, ranks in (
                ("head", head_ranks),
                ("tail", tail_ranks),
            ):
                rank = int(ranks[triple_index])
                records.append(
                    {
                        "arm": str(arm),
                        "triple_index": int(triple_index),
                        "head_id": int(head),
                        "head_name": entity_names[int(head)],
                        "relation_id": int(relation),
                        "relation_name": relation_names[int(relation)],
                        "tail_id": int(tail),
                        "tail_name": entity_names[int(tail)],
                        "direction": direction,
                        "rank": rank,
                        "reciprocal_rank": 1.0 / float(rank),
                        "hit_at_1": int(rank <= 1),
                        "hit_at_3": int(rank <= 3),
                        "hit_at_10": int(rank <= 10),
                    }
                )
        # 释放当前模型的显存，再加载下一份检查点。
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summaries, records, rank_arrays


def build_relation_metrics(
    query_records: Iterable[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """把逐查询排名按模型、关系和头尾方向汇总。"""

    buckets: Dict[Tuple[str, int, str, str], List[int]] = defaultdict(list)
    relation_names: Dict[int, str] = {}
    for record in query_records:
        arm = str(record["arm"])
        relation_id = int(record["relation_id"])
        direction = str(record["direction"])
        rank = int(record["rank"])
        relation_names[relation_id] = str(record["relation_name"])
        buckets[(arm, relation_id, relation_names[relation_id], direction)].append(
            rank
        )
        buckets[(arm, relation_id, relation_names[relation_id], "combined")].append(
            rank
        )

    rows: List[Dict[str, object]] = []
    for key in sorted(buckets):
        arm, relation_id, relation_name, direction = key
        metrics = metrics_from_ranks(buckets[key])
        rows.append(
            {
                "arm": arm,
                "relation_id": relation_id,
                "relation_name": relation_name,
                "direction": direction,
                **metrics,
            }
        )
    return rows


def build_pairwise_query_outcomes(
    triples: Sequence[IdTriple],
    rank_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """计算预设相邻实验臂在每个查询上的胜、负、平和总体汇总。"""

    detail_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    available = set(rank_arrays)
    for baseline, candidate, label in DESIGNED_PAIRS:
        if baseline not in available or candidate not in available:
            continue
        candidate_wins = 0
        baseline_wins = 0
        ties = 0
        reciprocal_deltas: List[float] = []
        for direction in ("head", "tail"):
            baseline_ranks = rank_arrays[baseline][direction]
            candidate_ranks = rank_arrays[candidate][direction]
            for triple_index, triple in enumerate(triples):
                baseline_rank = int(baseline_ranks[triple_index])
                candidate_rank = int(candidate_ranks[triple_index])
                if candidate_rank < baseline_rank:
                    outcome = "candidate_win"
                    candidate_wins += 1
                elif candidate_rank > baseline_rank:
                    outcome = "baseline_win"
                    baseline_wins += 1
                else:
                    outcome = "tie"
                    ties += 1
                rr_delta = (
                    1.0 / float(candidate_rank)
                    - 1.0 / float(baseline_rank)
                )
                reciprocal_deltas.append(rr_delta)
                detail_rows.append(
                    {
                        "comparison": label,
                        "baseline_arm": baseline,
                        "candidate_arm": candidate,
                        "triple_index": int(triple_index),
                        "head_id": int(triple[0]),
                        "relation_id": int(triple[1]),
                        "tail_id": int(triple[2]),
                        "direction": direction,
                        "baseline_rank": baseline_rank,
                        "candidate_rank": candidate_rank,
                        "outcome": outcome,
                        "reciprocal_rank_delta": rr_delta,
                    }
                )
        query_count = len(reciprocal_deltas)
        summary_rows.append(
            {
                "comparison": label,
                "baseline_arm": baseline,
                "candidate_arm": candidate,
                "query_count": query_count,
                "candidate_win_count": candidate_wins,
                "baseline_win_count": baseline_wins,
                "tie_count": ties,
                "candidate_win_rate": (
                    float(candidate_wins) / float(query_count)
                ),
                "baseline_win_rate": (
                    float(baseline_wins) / float(query_count)
                ),
                "tie_rate": float(ties) / float(query_count),
                "mean_reciprocal_rank_delta": float(
                    np.mean(reciprocal_deltas)
                ),
            }
        )
    return detail_rows, summary_rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """把同构字典行写成带BOM的UTF-8 CSV。"""

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0].keys())
    with Path(path).open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_directional_report(
    summary: Mapping[str, object],
) -> str:
    """把方向诊断摘要渲染为简体中文大白话报告。"""

    model_metrics = summary.get("model_metrics", {})
    pairwise = summary.get("pairwise_summary", [])
    lines = [
        "# TransE无需重训方向诊断",
        "",
        "## 先说结论怎么看",
        "",
        "这份报告没有重新训练模型，只把各自最佳检查点放到同一批官方测试事实上，"
        "分别做头预测和尾预测。头、尾差异可以判断收益是否只集中在尾负采样直接优化的方向。",
        "",
        "## 头预测与尾预测",
        "",
        "| 实验臂 | 头MRR | 头Hits@3 | 尾MRR | 尾Hits@3 | 综合MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if isinstance(model_metrics, dict):
        for arm, values in model_metrics.items():
            lines.append(
                "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                    arm,
                    float(values["head"]["mrr"]),
                    float(values["head"]["hits_at_3"]),
                    float(values["tail"]["mrr"]),
                    float(values["tail"]["hits_at_3"]),
                    float(values["combined"]["mrr"]),
                )
            )
    lines.extend(
        [
            "",
            "## 相邻方案逐查询胜负",
            "",
            "| 对比 | 候选胜 | 基线胜 | 平局 | 平均倒数排名变化 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    if isinstance(pairwise, list):
        for row in pairwise:
            lines.append(
                "| {} | {} | {} | {} | {:+.6f} |".format(
                    row["comparison"],
                    int(row["candidate_win_count"]),
                    int(row["baseline_win_count"]),
                    int(row["tie_count"]),
                    float(row["mean_reciprocal_rank_delta"]),
                )
            )
    lines.extend(
        [
            "",
            "## 文件怎么用",
            "",
            "- `query_ranks.csv`：每条三元组的头、尾排名，可定位具体失败查询。",
            "- `relation_metrics.csv`：每种关系分别看头、尾和综合指标。",
            "- `pairwise_query_outcomes.csv`：相邻方案在每个查询上到底谁赢。",
            "- `pairwise_summary.csv`：把逐查询胜负汇总成易读数字。",
            "",
            "逐关系样本数差异很大，小样本关系的单次涨跌不要直接写成普遍结论。",
            "",
        ]
    )
    return "\n".join(lines)


def write_directional_outputs(
    output_dir: Path,
    summary: Mapping[str, object],
    query_records: Sequence[Mapping[str, object]],
    relation_metrics: Sequence[Mapping[str, object]],
    pairwise_details: Sequence[Mapping[str, object]],
    pairwise_summary: Sequence[Mapping[str, object]],
) -> Dict[str, str]:
    """写出方向诊断JSON、四类CSV和简体中文说明。"""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "directional_summary.json"
    query_path = output_dir / "query_ranks.csv"
    relation_path = output_dir / "relation_metrics.csv"
    detail_path = output_dir / "pairwise_query_outcomes.csv"
    pairwise_path = output_dir / "pairwise_summary.csv"
    report_path = output_dir / "directional_report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    _write_csv(query_path, query_records)
    _write_csv(relation_path, relation_metrics)
    _write_csv(detail_path, pairwise_details)
    _write_csv(pairwise_path, pairwise_summary)
    report_path.write_text(
        render_directional_report(summary),
        encoding="utf-8",
    )
    return {
        "summary": str(summary_path),
        "query_ranks": str(query_path),
        "relation_metrics": str(relation_path),
        "pairwise_query_outcomes": str(detail_path),
        "pairwise_summary": str(pairwise_path),
        "report": str(report_path),
    }


def create_directional_result_dir(
    result_root: Path,
    run_name: str,
) -> Path:
    """创建不会覆盖已有诊断结果的时间戳目录。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = (
        Path(result_root).expanduser().resolve()
        / "{}_{}".format(run_name, timestamp)
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def run_directional_diagnostics(
    dataset: KnowledgeGraphDataset,
    result_dirs: Mapping[str, Path],
    output_dir: Path,
    device: torch.device,
    max_triples: int,
    selection_seed: int,
    query_batch_size: int,
    candidate_batch_size: int,
    progress_every: int,
) -> Dict[str, object]:
    """执行完整只读诊断流程并写出所有结果文件。"""

    fingerprints = validate_checkpoint_fingerprints(result_dirs)
    triples = select_test_triples(
        dataset, int(max_triples), int(selection_seed)
    )
    model_metrics, query_records, rank_arrays = evaluate_directional_ranks(
        dataset,
        triples,
        result_dirs,
        device,
        int(query_batch_size),
        int(candidate_batch_size),
        int(progress_every),
    )
    relation_metrics = build_relation_metrics(query_records)
    pairwise_details, pairwise_summary = build_pairwise_query_outcomes(
        triples, rank_arrays
    )
    summary: Dict[str, object] = {
        "status": "completed",
        "protocol": "global_candidates_head_tail_filtered",
        "tie_policy": "optimistic_strictly_better",
        "device": str(device),
        "selected_test_triple_count": len(triples),
        "selected_test_triple_hash": hash_triples(triples),
        "selection_seed": int(selection_seed),
        "max_triples": int(max_triples),
        "checkpoint_fingerprints": fingerprints,
        "model_metrics": model_metrics,
        "pairwise_summary": pairwise_summary,
    }
    written = write_directional_outputs(
        output_dir,
        summary,
        query_records,
        relation_metrics,
        pairwise_details,
        pairwise_summary,
    )
    summary["output_files"] = written
    # 增补文件清单后重写摘要，保证摘要自身也能指向全部输出。
    with (Path(output_dir) / "directional_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return summary
