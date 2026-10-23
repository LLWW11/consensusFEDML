"""只读审计指定WN18RR结果，在独立报告目录保存可复核的诊断证据。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TREND_SQL = "SELECT epoch, train_loss, val_mrr, val_hits_at_1, val_hits_at_3, val_hits_at_10, validation_triples FROM training_metrics WHERE val_mrr IS NOT NULL ORDER BY epoch"
RELATION_SQL = "SELECT q.relation, s.train_triples, COUNT(*) / 2 AS test_triples, COUNT(*) AS queries, SUM(q.blocked) AS self_blocks, AVG(q.blocked) AS self_block_rate, s.relation_l1, s.relation_l2 FROM query_audit AS q JOIN relation_stats AS s ON q.relation = s.relation GROUP BY q.relation, s.train_triples, s.relation_l1, s.relation_l2 ORDER BY self_block_rate DESC, queries DESC"

from KGE_central.data import load_wn18rr
from KGE_central.evaluator import FilteredRankingEvaluator
from KGE_central.experiment import _state_dict_sha256
from KGE_central.model import TransE


def analyze() -> dict:
    """核对检查点、训练趋势、自身候选压制及逐查询与批量评估一致性。"""

    run = ROOT / "KGE_central/results/kge_central_wn18rr_strong_transe_cuda_20261016_143203_994086"
    data = ROOT / "HFLSnF_KG_v5/data/wn18rr"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config_snapshot.json").read_text(encoding="utf-8"))
    dataset = load_wn18rr(data)
    checkpoint = torch.load(str(run / "model_best.pt"), map_location="cpu")
    assert checkpoint["entity_to_id"] == dataset.entity_to_id
    assert checkpoint["relation_to_id"] == dataset.relation_to_id
    model = TransE(dataset.num_entities, dataset.num_relations, config["embedding_dim"], config["distance_norm"])
    model.load_state_dict(checkpoint["model_state_dict"])
    assert _state_dict_sha256(model) == summary["best_model_hash"]

    with (run / "metrics.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    trend = []
    for row in rows:
        if math.isfinite(float(row["val_mrr"])):
            trend.append({
                "epoch": int(row["epoch"]),
                "val_mrr": float(row["val_mrr"]),
                "train_loss": float(row["train_loss"]),
                "val_hits_at_1": float(row["val_hits_at_1"]),
                "val_hits_at_3": float(row["val_hits_at_3"]),
                "val_hits_at_10": float(row["val_hits_at_10"]),
                "validation_triples": int(float(row["val_evaluated_triple_count"])),
            })

    triples = dataset.test_triples
    with torch.no_grad():
        target_distances = model.score_triples(triples)
        head_self = triples.clone()
        head_self[:, 0] = triples[:, 2]
        tail_self = triples.clone()
        tail_self[:, 2] = triples[:, 0]
        head_distances = model.score_triples(head_self)
        tail_distances = model.score_triples(tail_self)
    head_blocks, tail_blocks = [], []
    for index, (head, relation, tail) in enumerate(triples.tolist()):
        # 只统计非目标且不会被标准filtered规则删除的自身候选。
        head_blocks.append(bool(head != tail and (tail, relation, tail) not in dataset.all_true_triples and head_distances[index] < target_distances[index]))
        tail_blocks.append(bool(head != tail and (head, relation, head) not in dataset.all_true_triples and tail_distances[index] < target_distances[index]))
    train_counts = Counter(dataset.train_triples[:, 1].tolist())
    relations = []
    audit_indices = []
    for name, relation in dataset.relation_to_id.items():
        indices = (triples[:, 1] == relation).nonzero().reshape(-1).tolist()
        audit_indices.extend(indices[:2])
        blocked = sum(head_blocks[index] + tail_blocks[index] for index in indices)
        relations.append({
            "relation": name, "train_triples": train_counts[relation],
            "test_triples": len(indices), "queries": 2 * len(indices),
            "self_blocks": blocked, "self_block_rate": blocked / (2 * len(indices)),
            "relation_l1": float(model.relation_embeddings.weight[relation].abs().sum()),
            "relation_l2": float(model.relation_embeddings.weight[relation].norm()),
        })
    evaluator = FilteredRankingEvaluator(dataset)
    audit_triples = triples[audit_indices]
    scalar = evaluator.evaluate(model, audit_triples, torch.device("cpu"), candidate_batch_size=4096, query_batch_size=1)
    batched = evaluator.evaluate(model, audit_triples, torch.device("cpu"), candidate_batch_size=2048, query_batch_size=4)
    assert scalar == batched
    train_entities = set(dataset.train_triples[:, (0, 2)].reshape(-1).tolist())
    unknown = sum(head not in train_entities or tail not in train_entities for head, _, tail in triples.tolist())
    source_paths = [run / name for name in ("summary.json", "config_snapshot.json", "metrics.csv", "model_best.pt")]
    source_paths += [data / (split + ".txt") for split in ("train", "valid", "test")]
    # 原始诊断值落入独立数据库，实际执行SQL聚合，供报告逐项追溯。
    database = sqlite3.connect(str(Path(__file__).with_name("analysis.sqlite")))
    database.row_factory = sqlite3.Row
    with database:
        database.executescript("DROP TABLE IF EXISTS training_metrics; DROP TABLE IF EXISTS query_audit; DROP TABLE IF EXISTS relation_stats; CREATE TABLE training_metrics (epoch INTEGER, train_loss REAL, val_mrr REAL, val_hits_at_1 REAL, val_hits_at_3 REAL, val_hits_at_10 REAL, validation_triples INTEGER); CREATE TABLE query_audit (relation TEXT, direction TEXT, blocked INTEGER); CREATE TABLE relation_stats (relation TEXT, train_triples INTEGER, relation_l1 REAL, relation_l2 REAL);")
        database.executemany("INSERT INTO training_metrics VALUES (:epoch, :train_loss, :val_mrr, :val_hits_at_1, :val_hits_at_3, :val_hits_at_10, :validation_triples)", trend)
        database.executemany("INSERT INTO relation_stats VALUES (:relation, :train_triples, :relation_l1, :relation_l2)", relations)
        names = {value: key for key, value in dataset.relation_to_id.items()}
        database.executemany("INSERT INTO query_audit VALUES (?, ?, ?)", [
            (names[int(triple[1])], direction, int(flags[index]))
            for index, triple in enumerate(triples.tolist())
            for direction, flags in (("head", head_blocks), ("tail", tail_blocks))
        ])
        trend = [dict(row) for row in database.execute(TREND_SQL)]
        relations = [dict(row) for row in database.execute(RELATION_SQL)]
    database.close()
    return {
        "summary": summary, "config": config, "dataset": dataset.summary(),
        "trend": trend, "relations": relations,
        "report_queries": {"trend": TREND_SQL, "relations": RELATION_SQL},
        "self_blocks": sum(head_blocks) + sum(tail_blocks), "queries": 2 * len(triples),
        "unseen_entity_test_triples": unknown,
        "unseen_train_entities": dataset.num_entities - len(train_entities),
        "single_specific_candidate_sampling_probability": 1 - (1 - 1 / dataset.num_entities) ** config["negative_sample_count"],
        "optimizer_steps": math.ceil(len(dataset.train_triples) / config["batch_size"]) * len(rows),
        "epoch_seconds_sum": sum(float(row["epoch_seconds"]) for row in rows),
        "evaluated_checkpoints": len(trend),
        "nonfinite_training_loss_count": sum(not math.isfinite(float(row["train_loss"])) for row in rows),
        "scalar_batched_audit": {"triples": len(audit_indices), "equal": scalar == batched, "metrics": scalar},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "limitations": [
            "仅对二十二条事实、四十四次查询复核逐查询与批量排名；完整测试指标来自指定运行摘要。",
            "自身候选压制统计覆盖全部测试查询，表示自身比正确答案分数更优，不表示自身一定排名第一。",
            "只有一个种子，未执行学习率、训练预算、损失参数或模型替换实验；改进幅度尚未知。",
        ],
    }


if __name__ == "__main__":
    torch.set_num_threads(4)
    evidence = analyze()
    destination = Path(__file__).with_name("evidence.json")
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file": str(destination), "self_blocks": evidence["self_blocks"], "queries": evidence["queries"], "audit_equal": evidence["scalar_batched_audit"]["equal"]}))
