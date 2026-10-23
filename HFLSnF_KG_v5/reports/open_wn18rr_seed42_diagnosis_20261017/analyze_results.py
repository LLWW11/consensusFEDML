"""只读诊断开放网络词汇图谱三方法结果，保存可复核证据与报告输入。"""

import csv
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import torch

from HFLSnF_KG_v5.multi_experiment import file_sha256
from HFLSnF_KG_v5.core.randomness import seed_everything
from HFLSnF_KG_v5.run_multi_topology_dataset import validate_result
from HFLSnF_KG_v5.tasks.kge.data import load_dataset
from HFLSnF_KG_v5.tasks.kge.directional import BatchedDirectionalEvaluator, TransEEmbeddingBundle
from HFLSnF_KG_v5.tasks.kge.evaluator import FilteredRankingEvaluator
from HFLSnF_KG_v5.tasks.kge.model import TransE


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
ROOT = PROJECT / "results/multi_topology_dataset/open_wn18rr_seed42"
LABELS = {"hflsnf": "HFLSnF", "hflnosnf": "HFLnoSnF", "flnosnf": "FLnoSnF"}


def read_json(path):
    """读取现有实验记录，不改变原文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(name, value):
    """保存有限值的独立诊断产物。"""

    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def self_candidate_statistics(dataset, state):
    """逐查询检查未被标准过滤的自身实体是否严格优于目标答案。"""

    entities = state["entity_embeddings.weight"]
    relations = state["relation_embeddings.weight"]
    valid = dataset.valid_triples
    positive = (entities[valid[:, 0]] + relations[valid[:, 1]] - entities[valid[:, 2]]).abs().sum(1)
    own = relations[valid[:, 1]].abs().sum(1)
    # 自身三元组的距离恰好为关系向量L1范数；保留与原评估相同的过滤规则。
    allowed = []
    for head, relation, tail in valid.tolist():
        allowed.append((head != tail and (tail, relation, tail) not in dataset.all_true_triples,
                        head != tail and (head, relation, head) not in dataset.all_true_triples))
    allowed = torch.tensor(allowed, dtype=torch.bool)
    defeats = (positive > own).unsqueeze(1) & allowed
    result = {
        "query_count": len(valid) * 2,
        "self_outranks_count": int(defeats.sum()),
        "self_outranks_fraction": int(defeats.sum()) / float(len(valid) * 2),
        "hits1_upper_bound_from_self": 1.0 - int(defeats.sum()) / float(len(valid) * 2),
        "positive_distance_mean": float(positive.mean()),
        "self_distance_mean": float(own.mean()),
        "relations": [],
    }
    for name, index in dataset.relation_to_id.items():
        mask = valid[:, 1] == index
        result["relations"].append({
            "relation": name, "valid_triples": int(mask.sum()),
            "relation_l1": float(relations[index].abs().sum()),
            "relation_l2": float(relations[index].norm()),
            "self_outranks_fraction": float(defeats[mask].float().mean()),
            "positive_distance_mean": float(positive[mask].mean()),
        })
    return result


def check_rank_implementations(dataset, state):
    """对四条固定验证事实核对串行与批量全候选排名，避免仅凭曲线猜测评估故障。"""

    selected = [tuple(row) for row in dataset.valid_triples[:4].tolist()]
    model = TransE(dataset.num_entities, dataset.num_relations, 256, 1)
    model.load_state_dict(state)
    bundle = TransEEmbeddingBundle(state["entity_embeddings.weight"], state["relation_embeddings.weight"], 1, Path("诊断中模型"), "")
    serial = FilteredRankingEvaluator(dataset)
    batched = BatchedDirectionalEvaluator(dataset.num_entities, dataset.all_true_triples)
    result = {}
    for head in (True, False):
        with torch.no_grad():
            one = [serial.rank_query(model, row, head, torch.device("cpu"), 4096) for row in selected]
            many = batched.evaluate_direction(bundle, selected, head, 4, 4096).tolist()
        if one != many:
            raise ValueError("串行与批量排名不一致：{} {}".format(one, many))
        result["head" if head else "tail"] = one
    return result


def source(identifier, label, path):
    """为报告附上明确的项目内来源路径，便于追溯指标定义。"""

    return {"id": identifier, "label": label, "path": path}


def chart(identifier, title, dataset, x, y, source_id, kind="line", color=None):
    """生成原生报告图表合同，趋势使用完整150轮，不平滑或删点。"""

    encodings = {"x": {"field": x}, "y": {"field": y}}
    if color:
        encodings["color"] = {"field": color}
    return {"id": identifier, "title": title, "type": kind, "dataset": dataset,
            "intent": "trend" if kind == "line" else "comparison", "sourceId": source_id,
            "question": "比较三方法的验证表现及诊断证据", "rationale": "按实际通信轮及训练暴露量比较轨迹；错误候选占比使用类别比较。",
            "encodings": encodings}


def make_report(analysis, trends):
    """将诊断结论组织为中文技术报告，不把验证指标标成测试结果。"""

    title = "OPEN＋WN18RR：三方法结果异常诊断"
    blocks = []

    def markdown(identifier, body):
        """添加一个可独立阅读的报告段落。"""

        blocks.append({"id": identifier, "type": "markdown", "body": body})

    markdown("title", "# " + title)
    markdown("summary", "## 结论：有真实的选模回落，也有模型本身的排名偏置\n\n三组均完成150轮CUDA训练，分区、初始化和检查点一致性核对通过。HFLnoSnF最佳验证MRR为0.2181，高于HFLSnF的0.2046；FLnoSnF在末轮仍增长至0.2035。不能预设更多参与者必然带来更高MRR。\n\n约93.3%～93.5%的验证查询中，自身实体这一未被过滤的错误候选严格优于真实答案，直接限制Hits@1。后期损失继续下降而MRR回落，表明局部训练目标与全局排名出现偏离。模型能力、训练配方和参与集合共同影响结果；现有证据不足以分摊它们的因果贡献。")
    markdown("scope", "## 比较口径：全部验证集，尚无本批次完整测试\n\n每轮使用同一组3,034条验证事实、6,068个头尾查询、40,943个实体候选，采用全候选精确过滤排名。最佳轮按验证MRR选择并恢复。下表的Hits指标均来自各方法最佳MRR轮，末轮MRR另列，不混用。")
    lines = ["| 方法 | 最佳轮 | 最佳验证MRR | 末轮MRR | 最佳轮Hits@1 | 最佳轮Hits@3 | 最佳轮Hits@10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in analysis["methods"]:
        best = item["best_metrics"]
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.2%} | {:.2%} | {:.2%} |".format(item["method"], item["best_round"], best["mrr"], item["last_mrr"], best["hits_at_1"], best["hits_at_3"], best["hits_at_10"]))
    markdown("values", "\n".join(lines))
    markdown("curves", "## HFLSnF较早见顶，FLnoSnF尚未达到平台\n\nHFLSnF从第73轮0.2046降至第150轮0.1843，下降约9.89%；HFLnoSnF从第99轮0.2181降至0.2047，下降约6.13%。两组后期训练损失仍下降，而Hits@10仍增长，因此这也包含前几名排序退化，不能只用整体是否过拟合一词解释。FLnoSnF的最佳轮正好在预算终点，150轮不是三个方法均已收敛的证据。")
    blocks.append({"id": "curve_plot", "type": "chart", "chartId": "mrr_round"})
    markdown("scale", "## 前期损失下降与排名提升并不同步\n\n按相同种子重建初始模型，关系向量平均L1范数约47.99，验证正事实平均距离约50.67，而目标函数gamma为9。初期损失可以主要通过缩小距离而下降，尚不保证实体之间的相对排名改善。这与三条曲线先低位停留、再明显上升的现象相容；由于没有逐轮关系向量快照，不能把转折点完全归因于该机制。\n\n验证集中有210条事实含训练未出现实体，约占6.92%；以预测目标计为211/6,068＝3.48%。它会拉低总体结果，但不足以单独解释约93%的自身候选压过现象。")
    markdown("self", "## 自身候选压过真实答案，解释了低Hits@1\n\nTransE距离为 ‖h＋r－t‖₁。预测尾实体时，错误候选t＝h的距离恒为‖r‖₁；头预测同理。本轮直接对全部验证查询检查该错误候选，计入其他真实答案过滤之后，三方法分别仍有5,676、5,664、5,670个查询被其压过，Hits@1的上限因此分别只有约6.46%、6.66%、6.56%。这只是由一个错误候选给出的上限，并非完整排序结果。\n\n训练集中，`_derivationally_related_form`的93.22%事实存在同关系反向事实，其验证占比35.53%。这类对称结构使TransE容易把关系向量压小、关联实体拉近；HFLnoSnF中该关系向量L1范数约0.808，而对应验证正事实平均距离约7.269。超强自身候选并非只出现在对称关系：该模型的`_hypernym`验证事实中，99.74%也被自身候选压过。\n\n采样器先均匀抽256个候选再作自对抗加权。对单个特定困难候选，单次抽中概率仅约0.623%；自对抗权重无法补救未抽中的候选。该采样不足是合理机制解释，尚需受控训练消融验证。标准评估保留此类错误候选是正常行为，不应删掉它来提高指标。")
    blocks.append({"id": "self_plot", "type": "chart", "chartId": "self_failure"})
    markdown("budget", "## 参与人数和总数据暴露量不是相同的优化预算\n\n三方法平均参与人数分别为17.747、10.567、4.913，对应有效全数据暴露443.65、264.13、122.83遍。本地训练3个周期且每次重置Adam；加入更多客户端会增加被平均的局部更新，不会把这些更新直接相加。因此不能把总样本暴露量等同为同一条优化轨迹上的训练周期。\n\n在约120遍暴露处，HFLSnF约第40轮MRR＝0.1214，HFLnoSnF约第67轮＝0.1935，FLnoSnF约第147轮＝0.2004。这个结果说明差异并非仅由总暴露量解释，通信轮数、客户端选择及局部漂移仍混在一起。下图只比较三方法共同覆盖的暴露区间，保留原始观测点。")
    blocks.append({"id": "exposure_plot", "type": "chart", "chartId": "mrr_exposure"})
    markdown("design", "## 当前实现主要比较参与调度，分组不保证额外学习收益\n\n逐行计数聚合在边缘层保留分子和分母，云端将其相加。对相同客户端更新，分层聚合与直接聚合在数学上相同，仅可能有浮点求和顺序差异。当前各客户端从同一轮全局模型开始，本轮没有额外的边缘独立训练。因而边缘分组数更多或更少，本身不保证提高精度；主要学习差别来自参与者集合及其时序。\n\n关系语义域划分的加权主域纯度为93.58%，使客户端目标高度异构。局部漂移和聚合后的排名损失是合理候选原因，但没有逐客户端梯度与消融，不能声称已证明。OPEN映射本身由受控人数重建，物理可行性未验证，这批结果也不足以作真实网络链路收益结论。")
    central = analysis["centralized"]["best_metrics"]
    markdown("baseline", "## 集中式也出现低Hits@1，应先校准模型预期\n\n已有集中式同一验证集MRR为{:.4f}、Hits@1为{:.2%}。HFLnoSnF的验证MRR比它低约0.0127，而低Hits@1在集中式同样存在。集中式学习率、优化器状态连续性及训练节奏不同，不能直接把这0.0127全部归为联邦分区损失。\n\n[RotatE原论文表5](https://arxiv.org/html/1902.10197)给出的TransE在WN18RR完整测试MRR约0.226、Hits@10约0.501；这只用于模型量级参考，文献测试集与本轮验证集、训练配方均不同，不是可直接比较的验收门槛。".format(central["mrr"], central["hits_at_1"]))
    markdown("next", "## 建议：先检验训练与模型机制，保持评估不变\n\n1. 保留本批结果，继续按最佳验证MRR选模；不要用末轮代替已恢复的最佳检查点。完成三组独立完整测试，并补种子2024、2025后再判断方法排序。\n2. 先做小规模受控配方对照：本地周期3降为1；另做学习率5e-4降至1e-4或2e-4，每次只改一项，三方法共同使用。观察验证MRR、Hits@1及自身候选压过比例，不能预先保证改善。\n3. 保留当前语义划分，对照普通头实体均衡划分，检验关系隔离对局部漂移的影响。另用相同参与者与相同时序检查分层和直接聚合，单独确认分组效应。\n4. 若目标是提高WN18RR本身的表达能力，另建集中式RotatE或ComplEx对照再扩展到联邦；若继续使用TransE，可单独试验加入未成真的自身三元组作为困难负样本，但不能保证克服其对称关系限制。")
    markdown("limits", "## 尚未确定的部分\n\n当前只有一个种子、三份最佳检查点，没有保存各轮关系级全量排名和局部梯度；无法定量分解模型表达能力、局部漂移、采样难度和调度噪声的独立贡献。串行与批量排名仅在固定四条验证事实的头尾查询上交叉核对，通过不等于对全部实现的形式化证明。业务代码、原始结果和配置均未修改，也未启动补训。")
    markdown("logging", "## 部分日志占位数值不代表训练故障\n\n未开启训练计时采样，因此采样、传输等计时列保留NaN。FedAvg分支将服务器更新范数等诊断字段写成0、服务器步长写成1，这是当前记录实现的占位含义，不表示模型没有变化或本地学习率没有生效。本地Adam实际学习率仍为5e-4；损失和验证指标均为有限值。")
    charts = [chart("mrr_round", "三方法逐轮验证MRR", "trends", "round", "mrr", "metrics", color="method"),
              chart("self_failure", "自身错误候选严格优于真实答案的查询占比", "self", "method", "self_fraction", "diagnosis", "bar"),
              chart("mrr_exposure", "共同数据暴露区间内的验证MRR", "common_exposure", "passes", "mrr", "exposure", color="method")]
    sources = [source("metrics", "三方法原始逐轮指标", "HFLSnF_KG_v5/results/multi_topology_dataset/open_wn18rr_seed42"),
               source("diagnosis", "完整验证集自身候选诊断及检查点", "HFLSnF_KG_v5/reports/open_wn18rr_seed42_diagnosis_20261017/diagnosis.json")]
    sources.append(source("exposure", "原始逐轮指标的共同暴露区间", "HFLSnF_KG_v5/reports/open_wn18rr_seed42_diagnosis_20261017/diagnosis.sqlite3"))
    return {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title, "blocks": blocks, "charts": charts, "sources": sources},
            "snapshot": {"version": 1, "status": "ready", "datasets": {
                "trends": trends, "common_exposure": [row for row in trends if row["passes"] <= 122.83016064950769],
                "self": [{"method": item["method"], "self_fraction": item["self_candidate"]["self_outranks_fraction"]} for item in analysis["methods"]]}}, "sources": sources}


def query_report_data(artifact):
    """将已复核的诊断数值写入独立数据库，并实际执行报告图表查询。"""

    queries = {
        "metrics": "SELECT method, round, mrr, passes FROM training_rounds ORDER BY method, round",
        "exposure": "SELECT method, round, mrr, passes FROM training_rounds WHERE passes <= 122.83016064950769 ORDER BY method, round",
        "diagnosis": "SELECT method, self_fraction FROM self_diagnostics ORDER BY method",
    }
    with sqlite3.connect(str(OUTPUT / "diagnosis.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE IF NOT EXISTS training_rounds (method TEXT, round INTEGER, mrr REAL, passes REAL)")
        connection.execute("CREATE TABLE IF NOT EXISTS self_diagnostics (method TEXT, self_fraction REAL)")
        connection.execute("DELETE FROM training_rounds")
        connection.execute("DELETE FROM self_diagnostics")
        connection.executemany("INSERT INTO training_rounds VALUES (:method,:round,:mrr,:passes)", artifact["snapshot"]["datasets"]["trends"])
        connection.executemany("INSERT INTO self_diagnostics VALUES (:method,:self_fraction)", artifact["snapshot"]["datasets"]["self"])
        for identifier, dataset in (("metrics", "trends"), ("exposure", "common_exposure"), ("diagnosis", "self")):
            artifact["snapshot"]["datasets"][dataset] = [dict(row) for row in connection.execute(queries[identifier])]
        for item in artifact["manifest"]["sources"]:
            identifier = item["id"]
            item["path"] = "HFLSnF_KG_v5/reports/open_wn18rr_seed42_diagnosis_20261017/diagnosis.sqlite3"
            item["query"] = {"engine": "sqlite", "language": "sql", "sql": queries[identifier],
                "description": "从已复核的原始指标和完整验证集诊断构建的本地审计表查询，不重新训练模型。",
                "tables_used": ["self_diagnostics" if identifier == "diagnosis" else "training_rounds"]}
    (OUTPUT / "图表查询.sql").write_text(";\n\n".join(queries.values()) + ";\n", encoding="utf-8")


def main():
    """核对输入与结果，生成独立诊断证据和规范报告输入。"""

    torch.set_num_threads(4)
    dataset = load_dataset("wn18rr")
    seed_everything(42)
    initial = TransE(dataset.num_entities, dataset.num_relations, 256, 1)
    with torch.no_grad():
        initial_scale = {"relation_l1_mean": float(initial.relation_embeddings.weight.abs().sum(1).mean()),
                         "positive_distance_mean": float(initial.score_triples(dataset.valid_triples).mean())}
    seen = set(dataset.train_triples[:, [0, 2]].reshape(-1).tolist())
    cold_start = {"affected_valid_triples": sum(h not in seen or t not in seen for h, r, t in dataset.valid_triples.tolist()),
                  "unseen_target_queries": sum((h not in seen) + (t not in seen) for h, r, t in dataset.valid_triples.tolist())}
    manifest = read_json(next(ROOT.glob("batch_*/batch_summary.json")))
    analysis = {"batch_status": manifest["status"], "dataset": dataset.summary(), "initial_scale": initial_scale,
                "cold_start": cold_start, "methods": [], "train_relation_symmetry": []}
    trends = []
    for state, entry in zip(manifest["entries"], manifest["validation"]["entries"]):
        directory = Path(state["result_dir"])
        # 检查实际150轮结果、选模与检查点摘要，不重跑训练或重新分区。
        checked = validate_result(directory, entry)
        assert checked["checkpoint_sha256"] == state["result"]["checkpoint_sha256"]
        assert file_sha256(entry["configuration"]["dynamic_group_mat_file"]) == entry["mat_sha256"]
        for split, expected in entry["data_sha256"].items():
            assert file_sha256(PROJECT / "data/wn18rr" / (split + ".txt")) == expected
        summary = read_json(directory / "summary.json")
        rows = list(csv.DictReader((directory / "metrics.csv").open(encoding="utf-8-sig")))
        checkpoint = torch.load(str(directory / "model_best.pt"), map_location="cpu")
        assert checkpoint["entity_to_id"] == dataset.entity_to_id
        assert checkpoint["relation_to_id"] == dataset.relation_to_id
        own = self_candidate_statistics(dataset, checkpoint["model_state_dict"])
        passes = 0.0
        for row in rows:
            passes += float(row["selected_train_triple_fraction"]) * 3
            trends.append({"method": LABELS[summary["ablation_arm"]], "round": int(row["round"]), "mrr": float(row["val_mrr"]), "loss": float(row["mean_client_train_loss"]), "passes": passes,
                           "hits1": float(row["val_hits_at_1"]), "hits3": float(row["val_hits_at_3"]), "hits10": float(row["val_hits_at_10"]), "participants": int(row["active_client_count"])})
        assert math.isclose(passes, summary["effective_global_passes"], rel_tol=1e-10)
        analysis["methods"].append({"method": LABELS[summary["ablation_arm"]], "result_dir": str(directory), "best_round": summary["best_round"], "best_metrics": summary["final_validation_metrics"], "last_mrr": float(rows[-1]["val_mrr"]),
            "best_to_last_relative_drop": (summary["best_validation_mrr_during_training"] - float(rows[-1]["val_mrr"])) / summary["best_validation_mrr_during_training"],
            "passes": passes, "participant_mean": summary["participant_count_mean"], "participant_variance": float(np.var([int(row["active_client_count"]) for row in rows], ddof=1)),
            "self_candidate": own, "checkpoint_sha256": checked["checkpoint_sha256"], "initial_model_hash": summary["initial_model_hash"], "partition_hash": summary["partition_hash"]})
        if summary["ablation_arm"] == "hflsnf":
            analysis["serial_batched_rank_check"] = check_rank_implementations(dataset, checkpoint["model_state_dict"])
    assert len({item["initial_model_hash"] for item in analysis["methods"]}) == 1
    assert len({item["partition_hash"] for item in analysis["methods"]}) == 1
    train = set(map(tuple, dataset.train_triples.tolist()))
    for name, index in dataset.relation_to_id.items():
        triples = [(h, r, t) for h, r, t in train if r == index]
        analysis["train_relation_symmetry"].append({"relation": name, "train_triples": len(triples), "reverse_fraction": sum((t, r, h) in train for h, r, t in triples) / float(len(triples))})
    central_path = next((PROJECT.parent / "KGE_central/results").glob("*wn18rr*/summary.json"))
    central = read_json(central_path)
    analysis["centralized"] = {"summary_path": str(central_path), "best_metrics": central["final_validation_metrics"], "best_epoch": central["best_epoch"]}
    save_json("diagnosis.json", analysis)
    artifact = make_report(analysis, trends)
    query_report_data(artifact)
    save_json("artifact.json", artifact)
    # 同时保留纯中文正文，便于在编辑器中审阅全部推断与边界。
    (OUTPUT / "结果诊断.md").write_text("\n\n".join(block["body"] for block in artifact["manifest"]["blocks"] if block["type"] == "markdown") + "\n", encoding="utf-8")
    print("结果完整性、四条事实的八次排名交叉核对、全部6068查询自身候选诊断通过；已生成独立证据与报告输入。")


if __name__ == "__main__":
    main()
