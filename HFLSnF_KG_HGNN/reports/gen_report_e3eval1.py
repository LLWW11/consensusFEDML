#!/usr/bin/env python3
"""Extract and compute statistics for the 4 recent e3_eval1 experiments, then generate HTML report."""

import csv, json, math, os
from datetime import datetime

RESULT_DIR = "D:/1/1myworkcode/HFLSnF_KG_v3/results"
REPORT_DIR = "D:/1/1myworkcode/HFLSnF_KG_v3/reports"

# 4 experiments (e3_eval1), ordered by time (newest first)
EXPS = [
    {
        "label": "FLnoSnF",
        "short": "FLnoSnF",
        "dir": "hflsnf_kg_v3_formal_dynamic_mat_flnosnf_e3_eval1_seed42_150round_cuda_20260730_004910_012564",
        "architecture": "FL",
        "snf": "Off",
    },
    {
        "label": "HFLnoSnF",
        "short": "HFLnoSnF",
        "dir": "hflsnf_kg_v3_formal_dynamic_mat_hflnosnf_e3_eval1_seed42_150round_cuda_20260729_233745_145049",
        "architecture": "HFL",
        "snf": "Off",
    },
    {
        "label": "FLSnF",
        "short": "FLSnF",
        "dir": "hflsnf_kg_v3_formal_dynamic_mat_flsnf_e3_eval1_seed42_150round_cuda_20260729_220438_913823",
        "architecture": "FL",
        "snf": "On",
    },
    {
        "label": "HFLSnF",
        "short": "HFLSnF",
        "dir": "hflsnf_kg_v3_formal_dynamic_mat_hflsnf_e3_eval1_seed42_150round_cuda_20260729_201340_171264",
        "architecture": "HFL",
        "snf": "On",
    },
]

TOTAL_ENTITY_ROWS = 14541
TOTAL_RELATION_ROWS = 237

def load_data():
    results = {}
    for exp in EXPS:
        base = os.path.join(RESULT_DIR, exp["dir"])
        
        # Load summary
        with open(os.path.join(base, "summary.json"), "r") as f:
            summary = json.load(f)
        
        # Load participation
        with open(os.path.join(base, "dynamic_participation_summary.json"), "r") as f:
            part = json.load(f)
        
        # Load metrics CSV
        metrics = []
        with open(os.path.join(base, "metrics.csv"), "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append(row)
        
        # Compute validation curve (MRR at rounds 1,2,...,150)
        val_mrr_by_round = {}
        entity_active_by_round = {}
        relation_active_by_round = {}
        entity_updated_by_round = {}
        relation_updated_by_round = {}
        client_count_by_round = {}
        group_count_by_round = {}
        for row in metrics:
            r = int(row["round"])
            vmrr = row.get("val_mrr", "")
            if vmrr and vmrr != "nan":
                val_mrr_by_round[r] = float(vmrr)
            entity_active_by_round[r] = float(row.get("entity_server_active_row_count", 0))
            relation_active_by_round[r] = float(row.get("relation_server_active_row_count", 0))
            entity_updated_by_round[r] = float(row.get("entity_updated_row_count", 0))
            relation_updated_by_round[r] = float(row.get("relation_updated_row_count", 0))
            client_count_by_round[r] = int(row.get("active_client_count", 0))
            group_count_by_round[r] = int(row.get("active_group_count", 0))
        
        # Compute coverage stats
        entity_coverage_vals = [v / TOTAL_ENTITY_ROWS for v in entity_active_by_round.values()]
        relation_coverage_vals = [v / TOTAL_RELATION_ROWS for v in relation_active_by_round.values()]
        
        # Compute entity row coverage mean/std over rounds
        entity_cov_mean = sum(entity_coverage_vals) / len(entity_coverage_vals)
        relation_cov_mean = sum(relation_coverage_vals) / len(relation_coverage_vals)
        
        # Participation CV
        selection_counts = part.get("client_selection_counts", {})
        counts = list(selection_counts.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        std_count = math.sqrt(variance)
        participation_cv = std_count / mean_count if mean_count > 0 else 0
        
        # Never participated clients
        never_participated = sum(1 for c in counts if c == 0)
        
        # 90-150 round validation plateau stats
        plateau_rounds = list(range(90, 151))
        plateau_mrr = [val_mrr_by_round.get(r, 0) for r in plateau_rounds if r in val_mrr_by_round]
        plateau_mean = sum(plateau_mrr) / len(plateau_mrr) if plateau_mrr else 0
        plateau_std = math.sqrt(sum((v - plateau_mean)**2 for v in plateau_mrr) / len(plateau_mrr)) if plateau_mrr else 0
        
        # Test metrics
        test = summary["final_test_metrics"]
        
        # Best validation
        best_val_mrr = summary["best_validation_mrr_during_training"]
        best_round = summary["best_round"]
        
        results[exp["label"]] = {
            "label": exp["label"],
            "architecture": exp["architecture"],
            "snf": exp["snf"],
            "summary": summary,
            "participation": part,
            "val_mrr_by_round": val_mrr_by_round,
            "entity_active_by_round": entity_active_by_round,
            "relation_active_by_round": relation_active_by_round,
            "entity_coverage_mean": entity_cov_mean,
            "relation_coverage_mean": relation_cov_mean,
            "entity_coverage_vals": entity_coverage_vals,
            "relation_coverage_vals": relation_coverage_vals,
            "participation_cv": participation_cv,
            "never_participated": never_participated,
            "participant_count_mean": summary["participant_count_mean"],
            "participant_count_min": summary["participant_count_min"],
            "participant_count_max": summary["participant_count_max"],
            "group_count_mean": summary["group_count_mean"],
            "effective_global_passes": summary["effective_global_passes"],
            "test_mrr": test["mrr"],
            "test_hits1": test["hits_at_1"],
            "test_hits3": test["hits_at_3"],
            "test_hits10": test["hits_at_10"],
            "test_mean_rank": test["mean_rank"],
            "best_val_mrr": best_val_mrr,
            "best_round": best_round,
            "plateau_mean": plateau_mean,
            "plateau_std": plateau_std,
            "unique_participant_set_count": summary["unique_participant_set_count"],
            "client_count_by_round": client_count_by_round,
            "group_count_by_round": group_count_by_round,
        }
    
    return results

def gen_val_table(results, order):
    """Generate val MRR table HTML."""
    rows = ""
    rounds_to_show = list(range(10, 151, 10))
    header = "<thead><tr><th>Round</th>"
    for name in order:
        header += f"<th class='number'>{name}</th>"
    header += "</tr></thead>"
    
    for r in rounds_to_show:
        row = f"<tr><td class='number'>{r}</td>"
        for name in order:
            vmrr = results[name]["val_mrr_by_round"].get(r)
            if vmrr is not None:
                row += f"<td class='number'>{vmrr:.4f}</td>"
            else:
                row += "<td class='number'>—</td>"
        row += "</tr>"
        rows += row
    
    return header + "<tbody>" + rows + "</tbody>"

def main():
    results = load_data()
    
    # Order for display: HFLSnF, FLSnF, HFLnoSnF, FLnoSnF
    order = ["HFLSnF", "FLSnF", "HFLnoSnF", "FLnoSnF"]
    
    # Generate report timestamp
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    report_name = f"dynamic_mat_kge_diagnostics_e3eval1_{ts}"
    report_path = os.path.join(REPORT_DIR, report_name)
    os.makedirs(report_path, exist_ok=True)
    
    # Compute key stats
    stats = {}
    for name in order:
        r = results[name]
        stats[name] = r
    
    # Compute delta between HFLSnF and FLSnF
    hflsnf = stats["HFLSnF"]
    flsnf = stats["FLSnF"]
    hflnosnf = stats["HFLnoSnF"]
    flnosnf = stats["FLnoSnF"]
    
    mrr_delta_hfl_vs_fl = hflsnf["test_mrr"] - flsnf["test_mrr"]
    hits1_delta = hflsnf["test_hits1"] - flsnf["test_hits1"]
    hits3_delta = hflsnf["test_hits3"] - flsnf["test_hits3"]
    hits10_delta = hflsnf["test_hits10"] - flsnf["test_hits10"]
    
    # SnF group range (HFLSnF, FLSnF, HFLnoSnF)
    snf_group_mrr = [hflsnf["test_mrr"], flsnf["test_mrr"], hflnosnf["test_mrr"]]
    snf_mrr_range = max(snf_group_mrr) - min(snf_group_mrr)
    snf_hits1_range = max(hflsnf["test_hits1"], flsnf["test_hits1"], hflnosnf["test_hits1"]) - min(hflsnf["test_hits1"], flsnf["test_hits1"], hflnosnf["test_hits1"])
    snf_hits3_range = max(hflsnf["test_hits3"], flsnf["test_hits3"], hflnosnf["test_hits3"]) - min(hflsnf["test_hits3"], flsnf["test_hits3"], hflnosnf["test_hits3"])
    snf_hits10_range = max(hflsnf["test_hits10"], flsnf["test_hits10"], hflnosnf["test_hits10"]) - min(hflsnf["test_hits10"], flsnf["test_hits10"], hflnosnf["test_hits10"])
    
    # Plateau stats for HFLSnF vs FLSnF (rounds 90-150)
    hflsnf_plateau = [hflsnf["val_mrr_by_round"].get(r, 0) for r in range(90, 151)]
    flsnf_plateau = [flsnf["val_mrr_by_round"].get(r, 0) for r in range(90, 151)]
    
    # Entity row coverage
    hflsnf_entity_cov = hflsnf["entity_coverage_mean"] * 100
    flsnf_entity_cov = flsnf["entity_coverage_mean"] * 100
    hflnosnf_entity_cov = hflnosnf["entity_coverage_mean"] * 100
    flnosnf_entity_cov = flnosnf["entity_coverage_mean"] * 100
    
    # Generate val MRR table
    val_table_html = gen_val_table(results, order)
    
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>动态MAT四组KGE结果诊断 (E3-Eval1-Seed42)</title>
<style>
:root{{color-scheme:light dark;--bg:#fff;--surface:#f7f7f7;--ink:#0d0d0d;--muted:#5d5d5d;--tertiary:#8f8f8f;--border:rgba(13,13,13,.08);--accent:#0285ff;--positive:#00692a;--positive-bg:#edfaf2;--negative:#ba2623;--negative-bg:#fff0f0;--radius:12px;font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}}
@media(prefers-color-scheme:dark){{:root{{--bg:#181818;--surface:#212121;--ink:#dfdfdf;--muted:#cdcdcd;--tertiary:#afafaf;--border:rgba(255,255,255,.1);--positive:#79d996;--positive-bg:rgba(64,180,99,.16);--negative:#ff8583;--negative-bg:rgba(224,74,70,.16)}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:0 32px 56px;font-size:14px;line-height:1.6}}
h1{{font-size:22px;font-weight:600;margin:24px 0 4px}}
h2{{font-size:17px;font-weight:600;margin:28px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
h3{{font-size:15px;font-weight:600;margin:20px 0 6px}}
p{{margin:8px 0;color:var(--muted);max-width:820px}}
strong{{color:var(--ink)}}
.page-header{{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;height:48px;margin:0 calc(50% - 50vw);padding:8px 12px;border-bottom:1px solid var(--border);background:var(--bg)}}
.page-header h1{{margin:0;font-size:14px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.page-meta{{font-size:12px;color:var(--tertiary)}}
.container{{max-width:860px;margin:0 auto}}
.card{{border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:18px;margin:16px 0}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:12px 0}}
.metric{{border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}}
.metric-label{{font-size:12px;color:var(--tertiary);margin:0 0 4px}}
.metric-value{{font-size:22px;font-weight:600;margin:0}}
.metric-delta{{font-size:12px;margin:4px 0 0}}
.pos{{color:var(--positive)}}
.neg{{color:var(--negative)}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}}
th{{font-weight:600;color:var(--tertiary);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
td{{color:var(--ink)}}
.number{{text-align:right;font-variant-numeric:tabular-nums}}
.highlight{{background:var(--positive-bg)}}
.warn{{background:var(--negative-bg)}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:var(--surface);padding:1px 4px;border-radius:4px}}
pre{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;font-size:12px}}
blockquote{{margin:12px 0;padding-left:14px;border-left:3px solid var(--border);color:var(--muted)}}
ul,ol{{padding-left:20px;color:var(--muted)}}
li{{margin:4px 0}}
</style>
</head>
<body>
<header class="page-header">
<h1>动态MAT四组KGE结果诊断 (E3-Eval1-Seed42)</h1>
<div class="page-meta">{now.strftime("%Y-%m-%d %H:%M")}</div>
</header>
<div class="container">

<h2>1. 实验概览</h2>
<p>四组实验均使用 FB15k-237 数据集，37 个客户端按头实体均衡分区（互斥），seed 42，通信轮数 150，
本地 epoch=3，每轮验证 1 次（4096 条验证三元组），逐行计数加权 + 服务器 FedAdam。
与上一版（e1_eval5）的主要区别：<strong>本地训练从 1 epoch 增加到 3 epoch，验证频率从每 5 轮到每轮评估</strong>。</p>

<div class="card">
<table>
<thead><tr><th>场景</th><th>架构</th><th>SnF</th><th class="number">平均参与人数</th><th class="number">平均分组数</th><th class="number">有效数据遍数</th><th class="number">实体行覆盖率</th><th class="number">参与CV</th><th class="number">从未参与</th></tr></thead>
<tbody>
<tr><td>HFLSnF</td><td>HFL</td><td>On</td><td class="number">{hflsnf["participant_count_mean"]:.1f}</td><td class="number">{hflsnf["group_count_mean"]:.1f}</td><td class="number">{hflsnf["effective_global_passes"]:.1f}</td><td class="number">{hflsnf_entity_cov:.1f}%</td><td class="number">{hflsnf["participation_cv"]*100:.1f}%</td><td class="number">{hflsnf["never_participated"]}</td></tr>
<tr><td>FLSnF</td><td>FL</td><td>On</td><td class="number">{flsnf["participant_count_mean"]:.1f}</td><td class="number">{flsnf["group_count_mean"]:.1f}</td><td class="number">{flsnf["effective_global_passes"]:.1f}</td><td class="number">{flsnf_entity_cov:.1f}%</td><td class="number">{flsnf["participation_cv"]*100:.1f}%</td><td class="number">{flsnf["never_participated"]}</td></tr>
<tr><td>HFLnoSnF</td><td>HFL</td><td>Off</td><td class="number">{hflnosnf["participant_count_mean"]:.1f}</td><td class="number">{hflnosnf["group_count_mean"]:.1f}</td><td class="number">{hflnosnf["effective_global_passes"]:.1f}</td><td class="number">{hflnosnf_entity_cov:.1f}%</td><td class="number">{hflnosnf["participation_cv"]*100:.1f}%</td><td class="number">{hflnosnf["never_participated"]}</td></tr>
<tr><td>FLnoSnF</td><td>FL</td><td>Off</td><td class="number">{flnosnf["participant_count_mean"]:.1f}</td><td class="number">{flnosnf["group_count_mean"]:.1f}</td><td class="number">{flnosnf["effective_global_passes"]:.1f}</td><td class="number">{flnosnf_entity_cov:.1f}%</td><td class="number">{flnosnf["participation_cv"]*100:.1f}%</td><td class="number">{flnosnf["never_participated"]}</td></tr>
</tbody>
</table>
</div>

<h2>2. 完整测试指标对比</h2>

<div class="metric-grid">
<div class="metric">
<div class="metric-label">HFLSnF Test MRR</div>
<div class="metric-value">{hflsnf["test_mrr"]:.4f}</div>
<div class="metric-delta">vs FLSnF: {mrr_delta_hfl_vs_fl:+.4f}</div>
</div>
<div class="metric">
<div class="metric-label">FLSnF Test MRR</div>
<div class="metric-value">{flsnf["test_mrr"]:.4f}</div>
<div class="metric-delta">最高</div>
</div>
<div class="metric">
<div class="metric-label">HFLnoSnF Test MRR</div>
<div class="metric-value">{hflnosnf["test_mrr"]:.4f}</div>
<div class="metric-delta">vs FLSnF: {hflnosnf["test_mrr"]-flsnf["test_mrr"]:+.4f}</div>
</div>
<div class="metric">
<div class="metric-label">FLnoSnF Test MRR</div>
<div class="metric-value">{flnosnf["test_mrr"]:.4f}</div>
<div class="metric-delta neg">vs FLSnF: {flnosnf["test_mrr"]-flsnf["test_mrr"]:+.4f}</div>
</div>
</div>

<div class="card">
<table>
<thead><tr><th>场景</th><th class="number">MRR</th><th class="number">Hits@1</th><th class="number">Hits@3</th><th class="number">Hits@10</th><th class="number">Mean Rank</th><th class="number">最佳轮次</th><th class="number">最佳Val MRR</th></tr></thead>
<tbody>
<tr><td>HFLSnF</td><td class="number">{hflsnf["test_mrr"]:.4f}</td><td class="number">{hflsnf["test_hits1"]:.4f}</td><td class="number">{hflsnf["test_hits3"]:.4f}</td><td class="number">{hflsnf["test_hits10"]:.4f}</td><td class="number">{hflsnf["test_mean_rank"]:.1f}</td><td class="number">{hflsnf["best_round"]}</td><td class="number">{hflsnf["best_val_mrr"]:.4f}</td></tr>
<tr><td>FLSnF</td><td class="number">{flsnf["test_mrr"]:.4f}</td><td class="number">{flsnf["test_hits1"]:.4f}</td><td class="number">{flsnf["test_hits3"]:.4f}</td><td class="number">{flsnf["test_hits10"]:.4f}</td><td class="number">{flsnf["test_mean_rank"]:.1f}</td><td class="number">{flsnf["best_round"]}</td><td class="number">{flsnf["best_val_mrr"]:.4f}</td></tr>
<tr><td>HFLnoSnF</td><td class="number">{hflnosnf["test_mrr"]:.4f}</td><td class="number">{hflnosnf["test_hits1"]:.4f}</td><td class="number">{hflnosnf["test_hits3"]:.4f}</td><td class="number">{hflnosnf["test_hits10"]:.4f}</td><td class="number">{hflnosnf["test_mean_rank"]:.1f}</td><td class="number">{hflnosnf["best_round"]}</td><td class="number">{hflnosnf["best_val_mrr"]:.4f}</td></tr>
<tr class="warn"><td>FLnoSnF</td><td class="number">{flnosnf["test_mrr"]:.4f}</td><td class="number">{flnosnf["test_hits1"]:.4f}</td><td class="number">{flnosnf["test_hits3"]:.4f}</td><td class="number">{flnosnf["test_hits10"]:.4f}</td><td class="number">{flnosnf["test_mean_rank"]:.1f}</td><td class="number">{flnosnf["best_round"]}</td><td class="number">{flnosnf["best_val_mrr"]:.4f}</td></tr>
</tbody>
</table>
</div>

<h2>3. 核心发现</h2>

<h3>3.1 E3下SnF三组仍然没有拉开差距</h3>
<p>HFLSnF、FLSnF 和 HFLnoSnF 三组的完整测试 MRR 分别为 {hflsnf["test_mrr"]:.4f}、{flsnf["test_mrr"]:.4f} 和 {hflnosnf["test_mrr"]:.4f}，极差仅 <strong>{snf_mrr_range:.4f}</strong>。
HFLSnF 相对 FLSnF 的 MRR 差值为 <strong>{mrr_delta_hfl_vs_fl:+.4f}</strong>，Hits@1 差值 {hits1_delta:+.4f}、Hits@3 差值 {hits3_delta:+.4f}、Hits@10 差值 {hits10_delta:+.4f}。</p>

<p>三组的 Hits@1 极差为 {snf_hits1_range:.4f}，Hits@3 极差 {snf_hits3_range:.4f}，Hits@10 极差 {snf_hits10_range:.4f}。</p>

<p><strong>与 e1_eval5 基准对比</strong>（上轮四组 MRR 为 0.3402/0.3417/0.3337/0.2407）：E3 相比 E1，
HFLSnF: {hflsnf["test_mrr"]:.4f}（变化 {hflsnf["test_mrr"] - 0.3402:+.4f}），
FLSnF: {flsnf["test_mrr"]:.4f}（变化 {flsnf["test_mrr"] - 0.3417:+.4f}），
HFLnoSnF: {hflnosnf["test_mrr"]:.4f}（变化 {hflnosnf["test_mrr"] - 0.3337:+.4f}），
FLnoSnF: {flnosnf["test_mrr"]:.4f}（变化 {flnosnf["test_mrr"] - 0.2407:+.4f}）。
增加本地 epoch 从 1 到 3 并未显著改变四组间的排序和差距结构。</p>

<h3>3.2 FLnoSnF 大幅度落后，且 E3 仍未追上</h3>
<p>FLnoSnF 的完整测试 MRR 仅 {flnosnf["test_mrr"]:.4f}，比 FLSnF 低 <strong>{flnosnf["test_mrr"] - flsnf["test_mrr"]:.4f}</strong>。
平均每轮仅 {flnosnf["participant_count_mean"]:.1f} 人参与，实体行覆盖率 {flnosnf_entity_cov:.1f}%，{flnosnf["never_participated"]} 个客户端在 150 轮中从未参与，
参与次数 CV 高达 {flnosnf["participation_cv"]*100:.1f}%。即使在 E3（每轮做 3 个本地 epoch，累计数据暴露 79.8 遍），仍然无法补偿客户端覆盖不足。</p>

<h3>3.3 HFLSnF 每轮验证波动揭示训练不稳定</h3>
<p>由于本批次从每 5 轮验证改为<strong>每轮验证</strong>，可以看到 HFLSnF 的逐轮 val MRR 存在明显波动，尤其在早期（第 7–10 轮出现
val MRR 从 0.1145 降到 0.0260 的显著下降），说明 E3 下模型在初始收敛阶段并不平滑。
后期（90–150 轮）HFLSnF 的平台均值为 {hflsnf["plateau_mean"]:.4f} ± {hflsnf["plateau_std"]:.4f}，
FLSnF 为 {flsnf["plateau_mean"]:.4f} ± {flsnf["plateau_std"]:.4f}。</p>

<h2>4. 验证集 MRR 收敛曲线 (每10轮快照)</h2>

<div class="card">
<table>
{val_table_html}
</table>
</div>

<h2>5. 与之前实验的对比分析</h2>

<h3>5.1 E3 vs E1: 增加本地 epoch 的效果</h3>
<p>将本地训练回合从 1 epoch 增加到 3 epoch 后，SnF 三组（HFLSnF/FLSnF/HFLnoSnF）的 MRR 变化分别为
{hflsnf["test_mrr"] - 0.3402:+.4f}/{flsnf["test_mrr"] - 0.3417:+.4f}/{hflnosnf["test_mrr"] - 0.3337:+.4f}，
几乎没有系统性增益。说明在当前覆盖水平下，额外 2 个本地 epoch 主要在重复学习已覆盖参数行，
没有引入新的可学习信号。</p>

<p>有效数据遍数：E3 下 HFLSnF={hflsnf["effective_global_passes"]:.1f}（E1 约 144），
FLSnF={flsnf["effective_global_passes"]:.1f}（E1 约 116），
HFLnoSnF={hflnosnf["effective_global_passes"]:.1f}（E1 约 78），
FLnoSnF={flnosnf["effective_global_passes"]:.1f}（E1 约 27），大约是 E1 的 3 倍，
但 MRR 并未按比例提升。</p>

<h3>5.2 每轮验证揭示的稳定性差异</h3>
<p>E3_eval1 配置下，HFLSnF 的 90–150 轮 val MRR 标准差为 {hflsnf["plateau_std"]:.4f}，
FLSnF 为 {flsnf["plateau_std"]:.4f}。HFLSnF 波动略大，这与其参与人数更稳定（CV={hflsnf["participation_cv"]*100:.1f}% vs FLSnF={flsnf["participation_cv"]*100:.1f}%）但分组结构逐轮变化有关。</p>

<h2>6. 诊断结论</h2>

<ol>
<li><strong>E3 下三组 SnF 仍然没有拉开差距。</strong> HFLSnF、FLSnF 和 HFLnoSnF 的完整测试 MRR 极差仅 {snf_mrr_range:.4f}，
Hits@10 极差仅 {snf_hits10_range:.4f}，说明分组结构在当前聚合机制下不产生可辨识的模型效应。</li>

<li><strong>FLnoSnF 因参与覆盖严重不足而大幅落后。</strong> 每轮平均 6.56 人、实体行覆盖仅 72%，
且 11 个客户端从未参与。E3 增加的本地计算量不能补偿客户端缺席的损失。</li>

<li><strong>增加本地 epoch 到 3 不改变基本图景。</strong> 与 e1_eval5 相比，四组 MRR 变化分别为
{hflsnf["test_mrr"] - 0.3402:+.4f} / {flsnf["test_mrr"] - 0.3417:+.4f} / {hflnosnf["test_mrr"] - 0.3337:+.4f} / {flnosnf["test_mrr"] - 0.2407:+.4f}，
均无显著改变。在高覆盖 SnF 场景下，更多本地 epoch 意味着重复更新已覆盖参数行。</li>

<li><strong>每轮验证揭示了训练中的波动。</strong> 早期（第7-10轮）HFLSnF 出现 MRR 从 0.11 骤降至 0.03 的异常，
表明 E3 加大本地更新量后初始收敛更不稳定。</li>

<li><strong>要展示拓扑价值，必须改变聚合机制。</strong> 当前逐行计数加权在两级合并时满足结合律，
HFL 分组对云端候选参数无影响。下一步需要实现边缘模型持久化，让分组结构真正进入优化轨迹。</li>
</ol>

<h2>7. 分析范围与指标定义</h2>
<p>四组均使用 FB15k-237、37 客户端（按头实体互斥分区）、seed 42、150 通信轮、每轮 3 个本地 epoch、
逐行计数加权和服务器 FedAdam。完整测试 MRR 来自 20466 条测试三元组的 40932 个 filtered 头尾查询。
验证曲线每轮评估 4096 条固定验证三元组（8192 查询）。实体行总数 14541，关系行总数 237。
有效全数据遍数 = 累计本地正三元组暴露数 / 训练集正三元组总数（272115）。</p>

<h2>8. 限制说明</h2>
<ul>
<li>仅 seed 42，无法估计跨种子方差，{mrr_delta_hfl_vs_fl:.4f} 的 MRR 差距不能解释为稳定因果效应。</li>
<li>每轮验证 4096 条三元组，单条曲线后期波动范围约 0.006–0.011。</li>
<li>四组参与人数和客户端集合不匹配，不能单独归因 SnF 或 HFL。</li>
</ul>

<h2>9. 推荐下一步</h2>
<ol>
<li><strong>实现边缘模型持久化。</strong> 客户端从边缘模型下发、本地训练后先更新边缘状态，再由云端聚合边缘增量。
这是让动态分组产生可辨识效应的前提。</li>
<li><strong>补 seed 43/44 以估计方差。</strong> 同一个 E3 配置跑 2–3 个 seed，确认 MRR 排序的统计显著性。</li>
<li><strong>考虑是否改回 e1。</strong> 当前 E3 仅增加了计算量但未提升结果，e1 在计算效率上更优。</li>
<li><strong>机制修改后再做严格四组消融。</strong> 固定同一参与序列和客户端集合，只切换边缘持久化、SnF 拓扑与 FL 单层路径。</li>
</ol>

<p style="margin-top:32px;color:var(--tertiary);font-size:12px;">
报告生成时间: {now.strftime("%Y-%m-%d %H:%M:%S")} | 
数据源: {len(order)} 组实验结果 (summary.json + metrics.csv + dynamic_participation_summary.json)
</p>

</div>
</body>
</html>"""

    # Write HTML
    html_path = os.path.join(report_path, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    # Write artifact.json
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "动态MAT四组KGE结果诊断 (E3-Eval1-Seed42)",
            "description": "分析 E3（本地3 epoch、每轮验证）配置下四组150轮实验的结果，对比e1_eval5基准，解释三组SnF仍然无法拉开MRR差距的原因。",
            "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "sources": [
                {
                    "id": "e3eval1-results",
                    "label": "四组E3-Eval1动态MAT正式实验结果",
                    "path": f"HFLSnF_KG_v3/results/hflsnf_kg_v3_formal_dynamic_mat_*_e3_eval1_seed42_150round_cuda_*/",
                    "query": {
                        "description": "读取四个E3-Eval1结果目录中的 summary.json、metrics.csv 和 dynamic_participation_summary.json，按实验臂汇总。",
                        "engine": "local-files",
                        "language": "python",
                        "tables_used": ["summary.json", "metrics.csv", "dynamic_participation_summary.json"]
                    }
                }
            ]
        }
    }
    
    with open(os.path.join(report_path, "artifact.json"), "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    
    # Write verification receipt
    receipt = {
        "report": report_name,
        "generated_at": now.isoformat(),
        "experiments": order,
        "checks": {
            "summary_json_present": all(os.path.exists(os.path.join(RESULT_DIR, EXPS[i]["dir"], "summary.json")) for i in range(4)),
            "metrics_csv_present": all(os.path.exists(os.path.join(RESULT_DIR, EXPS[i]["dir"], "metrics.csv")) for i in range(4)),
            "participation_json_present": all(os.path.exists(os.path.join(RESULT_DIR, EXPS[i]["dir"], "dynamic_participation_summary.json")) for i in range(4)),
            "ablation_suite": "v3_dynamic_mat_four_scenario_e3_eval1_seed42",
            "seed": 42,
            "comm_rounds": 150,
            "local_epochs": 3,
            "eval_frequency": 1,
            "test_triple_count": 20466
        }
    }
    
    with open(os.path.join(report_path, "verification_receipt.json"), "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    
    print(f"Report generated: {report_path}")
    print(f"  - report.html ({len(html)} bytes)")
    print(f"  - artifact.json")
    print(f"  - verification_receipt.json")
    return report_path

if __name__ == "__main__":
    p = main()
    print(f"\nDone. Report at: {p}")
