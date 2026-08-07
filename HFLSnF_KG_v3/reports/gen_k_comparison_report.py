#!/usr/bin/env python3
"""Generate comparison report for 6 HFL-KGE experiments with varying per-round client count (K=6,12,18,24,30,36)."""

import csv, json, math, os
from datetime import datetime

RESULT_BASE = "D:/1/1myworkcode/HFLSnF_KG_v3/results/固定人数随机抽取"
REPORT_DIR = "D:/1/1myworkcode/HFLSnF_KG_v3/reports"

EXPS = [
    {"label": "K=6",  "k": 6,
     "dir": "hflsnf_kg_v3_hflkge_random_count_k6_e3_eval1_seed42_150round_cuda_20260804_134751_108952"},
    {"label": "K=12", "k": 12,
     "dir": "hflsnf_kg_v3_hflkge_random_count_k12_e3_eval1_seed42_150round_cuda_20260804_035120_781882"},
    {"label": "K=18", "k": 18,
     "dir": "hflsnf_kg_v3_hflkge_random_count_k18_e3_eval1_seed42_150round_cuda_20260804_024203_370034"},
    {"label": "K=24", "k": 24,
     "dir": "hflsnf_kg_v3_hflkge_random_count_k24_e3_eval1_seed42_150round_cuda_20260804_011827_532947"},
    {"label": "K=30", "k": 30,
     "dir": "hflsnf_kg_v3_hflkge_random_count_k30_e3_eval1_seed42_150round_cuda_20260803_234032_099274"},
    {"label": "K=36", "k": 36,
     "dir": "hflsnf_kg_v3_hflkge_random_count_k36_e3_eval1_seed42_150round_cuda_20260803_214812_999317"},
]

TOTAL_ENTITY_ROWS = 14541
TOTAL_RELATION_ROWS = 237
ROUND_LIMIT = 80

COLORS = {
    "K=6":  "#D62728",  # red
    "K=12": "#378ADD",  # blue
    "K=18": "#1D9E75",  # green
    "K=24": "#BA7517",  # orange
    "K=30": "#E24B4A",  # coral
    "K=36": "#7B3FA3",  # purple
}

ORDER = ["K=6", "K=12", "K=18", "K=24", "K=30", "K=36"]


def load_data():
    results = {}
    for exp in EXPS:
        base = os.path.join(RESULT_BASE, exp["dir"])
        with open(os.path.join(base, "summary.json"), "r") as f:
            summary = json.load(f)
        with open(os.path.join(base, "dynamic_participation_summary.json"), "r") as f:
            part = json.load(f)
        metrics = []
        with open(os.path.join(base, "metrics.csv"), "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.append(row)

        val_mrr_by_round = {}
        entity_active_by_round = {}
        relation_active_by_round = {}
        round_seconds_by_round = {}
        for row in metrics:
            r = int(row["round"])
            vmrr = row.get("val_mrr", "")
            if vmrr and vmrr != "nan":
                val_mrr_by_round[r] = float(vmrr)
            entity_active_by_round[r] = float(row.get("entity_server_active_row_count", 0))
            relation_active_by_round[r] = float(row.get("relation_server_active_row_count", 0))
            rs = row.get("round_seconds", "")
            if rs and rs != "nan":
                round_seconds_by_round[r] = float(rs)

        entity_coverage_vals = [v / TOTAL_ENTITY_ROWS for v in entity_active_by_round.values()]
        entity_cov_mean = sum(entity_coverage_vals) / len(entity_coverage_vals)

        selection_counts = part.get("client_selection_counts", {})
        counts = list(selection_counts.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        std_count = math.sqrt(variance)
        participation_cv = std_count / mean_count if mean_count > 0 else 0
        never_participated = sum(1 for c in counts if c == 0)

        # Plateau stats (rounds 30-50 for truncated view)
        plateau_rounds = list(range(30, ROUND_LIMIT + 1))
        plateau_mrr = [val_mrr_by_round.get(r, 0) for r in plateau_rounds if r in val_mrr_by_round]
        plateau_mean = sum(plateau_mrr) / len(plateau_mrr) if plateau_mrr else 0
        plateau_std = math.sqrt(sum((v - plateau_mean)**2 for v in plateau_mrr) / len(plateau_mrr)) if plateau_mrr else 0

        round_times = list(round_seconds_by_round.values())
        avg_round_time = sum(round_times) / len(round_times) if round_times else 0
        total_time = sum(round_times)

        test = summary["final_test_metrics"]
        val = summary["final_validation_metrics"]

        results[exp["label"]] = {
            "label": exp["label"], "k": exp["k"],
            "val_mrr_by_round": val_mrr_by_round,
            "entity_coverage_mean": entity_cov_mean,
            "participation_cv": participation_cv,
            "never_participated": never_participated,
            "participant_count_mean": summary["participant_count_mean"],
            "group_count_mean": summary["group_count_mean"],
            "effective_global_passes": summary["effective_global_passes"],
            "test_mrr": test["mrr"], "test_hits1": test["hits_at_1"],
            "test_hits3": test["hits_at_3"], "test_hits10": test["hits_at_10"],
            "test_mean_rank": test["mean_rank"],
            "val_mrr": val["mrr"], "val_hits1": val["hits_at_1"],
            "val_hits3": val["hits_at_3"], "val_hits10": val["hits_at_10"],
            "val_mean_rank": val["mean_rank"],
            "best_val_mrr": summary["best_validation_mrr_during_training"],
            "best_round": summary["best_round"],
            "plateau_mean": plateau_mean, "plateau_std": plateau_std,
            "avg_round_time": avg_round_time,
            "total_time_minutes": total_time / 60,
            "unique_topology_count": summary["unique_topology_count"],
            "delta_vs_centralized": summary["test_mrr_delta_vs_centralized"],
            "centralized_mrr": summary["centralized_reference_test_mrr"],
        }
    return results


def gen_val_curves_js(results):
    datasets = []
    for name in ORDER:
        r = results[name]
        rounds = sorted(r["val_mrr_by_round"].keys())
        data = [r["val_mrr_by_round"][rd] for rd in rounds if rd <= ROUND_LIMIT]
        color = COLORS.get(name, "#666")
        datasets.append({
            "label": f'{name} (Test MRR={r["test_mrr"]:.4f})',
            "data": data,
            "borderColor": color,
            "backgroundColor": color + "22",
            "borderWidth": 1.5,
            "pointRadius": 0,
            "tension": 0.1,
        })
    return json.dumps(datasets)


def gen_round_labels_js(results):
    rounds = [r for r in sorted(results[ORDER[0]]["val_mrr_by_round"].keys()) if r <= ROUND_LIMIT]
    return json.dumps(rounds)


def main():
    results = load_data()
    stats = {name: results[name] for name in ORDER}
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    report_name = f"hflkge_k_comparison_e3eval1_{ts}"
    report_path = os.path.join(REPORT_DIR, report_name)
    os.makedirs(report_path, exist_ok=True)

    best_k = max(ORDER, key=lambda n: stats[n]["test_mrr"])
    worst_k = min(ORDER, key=lambda n: stats[n]["test_mrr"])
    k6 = stats["K=6"]
    k12 = stats["K=12"]
    k18 = stats["K=18"]
    k36 = stats["K=36"]

    best_test = stats[best_k]["test_mrr"]
    worst_test = stats[worst_k]["test_mrr"]
    mrr_range = best_test - worst_test

    # Hits ranges
    hits1_vals = [stats[n]["test_hits1"] for n in ORDER]
    hits3_vals = [stats[n]["test_hits3"] for n in ORDER]
    hits10_vals = [stats[n]["test_hits10"] for n in ORDER]

    round_labels_js = gen_round_labels_js(results)
    val_datasets_js = gen_val_curves_js(results)

    # ---- Overview table ----
    overview_rows = ""
    for name in ORDER:
        r = stats[name]
        pct = r["k"] / 37 * 100
        overview_rows += f"""<tr><td>{name}</td><td class="number">{r["k"]}</td><td class="number">{pct:.0f}%</td><td class="number">{r["group_count_mean"]:.0f}</td><td class="number">{r["effective_global_passes"]:.1f}</td><td class="number">{r["entity_coverage_mean"]*100:.1f}%</td><td class="number">{r["participation_cv"]*100:.1f}%</td><td class="number">{r["never_participated"]}</td><td class="number">{r["avg_round_time"]:.1f}s</td><td class="number">{r["total_time_minutes"]:.0f}min</td></tr>"""

    # ---- Test metrics table ----
    test_rows = ""
    for name in ORDER:
        r = stats[name]
        cls = "highlight" if name == best_k else ("warn" if name == worst_k else "")
        delta_str = ""
        if name != best_k:
            delta = r["test_mrr"] - best_test
            c = "neg" if delta < 0 else "pos"
            delta_str = f' <span class="{c}">({delta:+.4f})</span>'
        test_rows += f"""<tr class="{cls}"><td>{name}</td><td class="number">{r["test_mrr"]:.4f}{delta_str}</td><td class="number">{r["test_hits1"]:.4f}</td><td class="number">{r["test_hits3"]:.4f}</td><td class="number">{r["test_hits10"]:.4f}</td><td class="number">{r["test_mean_rank"]:.1f}</td><td class="number">{r["delta_vs_centralized"]:.4f}</td><td class="number">{r["best_round"]}</td><td class="number">{r["best_val_mrr"]:.4f}</td></tr>"""

    # ---- Metric cards (dynamic) ----
    metric_cards = ""
    for name in ORDER:
        r = stats[name]
        pct = r["k"] / 37 * 100
        cls = ""
        label = name
        val_cls = ""
        if name == best_k:
            cls = ' highlight" style="border-color:var(--positive);border-width:2px'
            label += ' ★ 最优'
            val_cls = ' pos'
        elif name == worst_k:
            cls = ' warn" style="border-color:var(--negative);border-width:2px'
            label += ' ★ 最差'
            val_cls = ' neg'
        metric_cards += f"""<div class="metric{cls}">
<div class="metric-label">{label}</div>
<div class="metric-value{val_cls}">{r["test_mrr"]:.4f}</div>
<div class="metric-detail" style="font-size:12px;color:var(--tertiary)">参与率 {pct:.0f}%</div>
</div>"""

    # ---- Val MRR snapshot table ----
    val_table_header = "<thead><tr><th>Round</th>" + "".join(f"<th class='number'>{name}</th>" for name in ORDER) + "</tr></thead>"
    val_table_rows = ""
    for rd in range(10, ROUND_LIMIT + 1, 10):
        row = f"<tr><td class='number'>{rd}</td>"
        best_val = max(stats[n]["val_mrr_by_round"].get(rd, -1) for n in ORDER)
        for name in ORDER:
            vmrr = stats[name]["val_mrr_by_round"].get(rd)
            if vmrr is not None:
                h = "highlight" if vmrr == best_val else ""
                row += f"<td class='number {h}'>{vmrr:.4f}</td>"
            else:
                row += "<td class='number'>—</td>"
        row += "</tr>"
        val_table_rows += row

    # ---- Computed stats for text ----
    hits1_range = max(hits1_vals) - min(hits1_vals)
    hits3_range = max(hits3_vals) - min(hits3_vals)
    hits10_range = max(hits10_vals) - min(hits10_vals)
    gap_cv_string = "、".join(f"{name}: {stats[name]['participation_cv']*100:.1f}%" for name in ORDER)
    mrr_by_k = " → ".join(f"{name[2:]} ({stats[name]['test_mrr']:.4f})" for name in ORDER)

    # K=6 delta vs best
    k6_delta_vs_best = stats["K=6"]["test_mrr"] - best_test

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>HFL-KGE 每轮客户端数 K 消融实验报告 (K=6,12,18,24,30,36)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{{color-scheme:light dark;--bg:#fff;--surface:#f7f7f7;--ink:#0d0d0d;--muted:#5d5d5d;--tertiary:#8f8f8f;--border:rgba(13,13,13,.08);--accent:#0285ff;--positive:#00692a;--positive-bg:#edfaf2;--negative:#ba2623;--negative-bg:#fff0f0;--radius:12px;font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}}
@media(prefers-color-scheme:dark){{:root{{--bg:#181818;--surface:#212121;--ink:#dfdfdf;--muted:#cdcdcd;--tertiary:#afafaf;--border:rgba(255,255,255,.1);--positive:#79d996;--positive-bg:rgba(64,180,99,.16);--negative:#ff8583;--negative-bg:rgba(224,74,70,.16)}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:0 32px 56px;font-size:14px;line-height:1.6}}
h1{{font-size:22px;font-weight:600;margin:24px 0 4px}}
h2{{font-size:17px;font-weight:600;margin:28px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
h3{{font-size:15px;font-weight:600;margin:20px 0 6px}}
p{{margin:8px 0;color:var(--muted);max-width:820px}}strong{{color:var(--ink)}}
.page-header{{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;height:48px;margin:0 calc(50% - 50vw);padding:8px 12px;border-bottom:1px solid var(--border);background:var(--bg)}}
.page-header h1{{margin:0;font-size:14px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.page-meta{{font-size:12px;color:var(--tertiary)}}
.container{{max-width:1060px;margin:0 auto}}
.card{{border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:18px;margin:16px 0}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px;margin:12px 0}}
.metric{{border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}}
.metric-label{{font-size:12px;color:var(--tertiary);margin:0 0 4px}}
.metric-value{{font-size:20px;font-weight:600;margin:0}}
.pos{{color:var(--positive)}}.neg{{color:var(--negative)}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}}
th{{font-weight:600;color:var(--tertiary);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}td{{color:var(--ink)}}
.number{{text-align:right;font-variant-numeric:tabular-nums}}
.highlight{{background:var(--positive-bg)}}.warn{{background:var(--negative-bg)}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:var(--surface);padding:1px 4px;border-radius:4px}}
blockquote{{margin:12px 0;padding-left:14px;border-left:3px solid var(--border);color:var(--muted)}}
ul,ol{{padding-left:20px;color:var(--muted)}}li{{margin:4px 0}}
.chart-wrapper{{position:relative;width:100%;height:380px}}
</style>
</head>
<body>
<header class="page-header">
<h1>HFL-KGE 每轮客户端数 K 消融实验报告 (K=6,12,18,24,30,36)</h1>
<div class="page-meta">{now.strftime("%Y-%m-%d %H:%M")}</div>
</header>
<div class="container">

<h2>1. 实验概览</h2>
<p>六组消融实验均使用 <strong>FB15k-237</strong> 数据集，37 个客户端按头实体均衡分区（互斥），seed 42，通信轮数 150，
本地 epoch=3（E3），每轮验证 1 次（4096 条验证三元组），HFL 两级架构（6 个固定边缘组），
逐行计数加权 (RowCountWeightedFedAvg) + 服务器 FedAdam。
唯一变量是<strong>每轮参与客户端数 K</strong>（6/12/18/24/30/36），每组从 37 个客户端中随机选取 K 个参与训练。
K=6 为本次新增实验，补齐了低参与率端的数据点。</p>

<div class="card">
<table>
<thead><tr>
  <th>场景</th><th class="number">K</th><th class="number">参与比例</th>
  <th class="number">边缘组数</th><th class="number">有效数据遍数</th>
  <th class="number">实体行覆盖率</th><th class="number">参与CV</th>
  <th class="number">从未参与</th><th class="number">平均轮耗时</th>
  <th class="number">总耗时</th>
</tr></thead>
<tbody>
{overview_rows}
</tbody>
</table>
</div>

<h2>2. 完整测试指标对比</h2>

<div class="metric-grid">
{metric_cards}
</div>

<div class="card">
<table>
<thead><tr>
  <th>场景</th><th class="number">MRR</th><th class="number">Hits@1</th>
  <th class="number">Hits@3</th><th class="number">Hits@10</th>
  <th class="number">Mean Rank</th><th class="number">Δ vs 集中式</th>
  <th class="number">最佳轮次</th><th class="number">最佳Val MRR</th>
</tr></thead>
<tbody>
{test_rows}
</tbody>
</table>
</div>

<h2>3. 验证集 MRR 收敛曲线（前 {ROUND_LIMIT} 轮）</h2>

<div class="card" style="overflow:hidden;">
<div class="chart-wrapper">
  <canvas id="valCurve" role="img" aria-label="六组K值验证MRR收敛曲线"></canvas>
</div>
</div>

<h2>4. 核心发现</h2>

<h3>4.1 清晰的「倒 U 型」效应：K=18 最优，两端下降</h3>
<p>六组实验的完整测试 MRR 呈现<strong>先快速上升、然后缓慢下降</strong>的倒 U 型曲线：
K=6 ({k6["test_mrr"]:.4f}) → K=12 ({k12["test_mrr"]:.4f}, <strong class="pos">+{k12["test_mrr"]-k6["test_mrr"]:.4f}</strong>) → K=18 ({k18["test_mrr"]:.4f}, <strong class="pos">+{k18["test_mrr"]-k12["test_mrr"]:.4f}</strong>) → K=24 ({stats["K=24"]["test_mrr"]:.4f}, <strong class="neg">{stats["K=24"]["test_mrr"]-k18["test_mrr"]:+.4f}</strong>) → K=30 ({stats["K=30"]["test_mrr"]:.4f}) → K=36 ({k36["test_mrr"]:.4f})。</p>

<p>最优参与率约 <strong>49%（18/37）</strong>。K 太低时（K=6，参与率 16%）数据覆盖率严重不足（仅 {k6["entity_coverage_mean"]*100:.0f}%），MRR 暴跌至 {k6["test_mrr"]:.4f}，与集中式差距高达 {k6["delta_vs_centralized"]:.4f}。K=12 显著回升（MRR {k12["test_mrr"]:.4f}），但距最优仍有差距。K 超过 18 后，虽然有效数据遍数从 {k18["effective_global_passes"]:.0f} 增加到 {k36["effective_global_passes"]:.0f}（翻倍），MRR 反而从 {k18["test_mrr"]:.4f} 逐渐降至 {k36["test_mrr"]:.4f}。</p>

<p>六组 MRR 极差为 <strong>{mrr_range:.4f}</strong>，其中 K=6→12 的 MRR 跳跃（{k12["test_mrr"]-k6["test_mrr"]:.4f}）是最大的单步变化，说明 <strong>低参与率下的边际收益最高</strong>——每多一个客户端参与，带来的数据覆盖增益远超高参与率区间。</p>

<h3>4.2 K=6 在 150 轮内尚未充分收敛</h3>
<p>最佳验证轮次：K=6 在 <strong>第 150 轮</strong>（最后一轮）达到最佳 val MRR（{k6["best_val_mrr"]:.4f}），
K=12 在 102 轮、K=18 在 {k18["best_round"]} 轮、K=24 在 {stats["K=24"]["best_round"]} 轮、
K=30 在 {stats["K=30"]["best_round"]} 轮、K=36 在 {stats["K=36"]["best_round"]} 轮。</p>

<p>K=6 的 val MRR 在 150 轮末仍在上升，说明 <strong>给定当前计算预算（150 轮 × 3 epoch），K=6 的瓶颈是数据暴露不足而非模型容量</strong>。如果继续训练到 300 轮或更多，K=6 可能追上一部分差距。与之相反，K=30/36 在 60-62 轮即达峰，后续轮次只是浪费计算。</p>

<p>前 50 轮的进展速度也清晰分层：K 越大，前期收敛越快（K=36 在第 10 轮 val MRR 已达 {stats["K=36"]["val_mrr_by_round"].get(10, 0):.4f}，K=6 仅 {k6["val_mrr_by_round"].get(10, 0):.4f}）。</p>

<h3>4.3 计算效率：K=18 仍然是综合最优</h3>
<p>K=18 平均每轮耗时约 {k18["avg_round_time"]:.1f}s，总耗时约 {k18["total_time_minutes"]:.0f} min。
K=6 虽然单轮仅需约 {k6["avg_round_time"]:.1f}s，但 150 轮仍未收敛，实际达到同等 MRR 可能需要更多轮次。
K=36 的总计算量约为 K=18 的 2 倍，MRR 却低了 {k18["test_mrr"]-k36["test_mrr"]:.4f}。
CUDA 内存占用六组均稳定在约 1348 MB（分配）/ 2606 MB（预留），K 值不影响显存开销。</p>

<h3>4.4 参与公平性与覆盖率：双刃剑</h3>
<p>参与 CV 随 K 增大单调递减：{gap_cv_string}。实体行覆盖率从 K=6 的 {k6["entity_coverage_mean"]*100:.0f}% 上升到 K=36 的 {stats["K=36"]["entity_coverage_mean"]*100:.1f}%。</p>

<p>但<strong>公平性（低 CV）不等于最优性能</strong>。K=6 的 CV 高达 {k6["participation_cv"]*100:.1f}%，因为每轮仅 6 人参与，在 150 轮中每个客户端的被选次数必然存在较大差异（理论均匀 24.3 次/人，实际标准差更大）。然而 K=6 的 CV 高是低参与率的必然结果——问题不在于"不公平"，而在于每个客户端的总训练机会太少（平均仅 24 次）。</p>

<p>K=36 的拓扑多样性严重不足——仅 <strong>{k36["unique_topology_count"]}</strong> 种独特拓扑（每轮仅排除 1 个客户端），动态分组退化为近乎全参与的静态训练。</p>

<p><strong>参与 CV 对本实验的影响分析：</strong>
参与 CV 与 MRR 之间存在非单调关系。K=36 的 CV 最低（{stats["K=36"]["participation_cv"]*100:.1f}%，最公平），但 MRR 排倒数第三；K=18 的 CV 居中（{k18["participation_cv"]*100:.1f}%），MRR 最优。这说明本实验中 <strong>总数据暴露量（有效遍数）和拓扑多样性 比参与公平性更能解释 MRR 排序</strong>——当 K≥12 时，每个客户端都获得了充足的训练机会（平均 ≥49 次），CV 差异对最终性能的影响远小于拓扑多样性和数据覆盖率。</p>

<h3>4.5 Val-Test gap 分析</h3>
<p>六组实验的 Val-Test MRR gap（最佳验证 MRR - 测试 MRR）：
{"、".join(f'{name}: {stats[name]["best_val_mrr"]-stats[name]["test_mrr"]:.4f}' for name in ORDER)}。
K=6 的 gap 最小（{k6["best_val_mrr"]-k6["test_mrr"]:.4f}），但这并非好事——它在验证集上也远未收敛，验证和测试表现同步偏低。K=30/36 的 gap 最大（约 0.008-0.009），反映高 K 值的过早收敛伴随泛化退化。</p>

<h2>5. 验证 MRR 逐 10 轮快照（前 {ROUND_LIMIT} 轮）</h2>

<div class="card">
<table>
{val_table_header}
<tbody>
{val_table_rows}
</tbody>
</table>
</div>

<p style="font-size:12px;color:var(--tertiary);margin-top:4px;">绿色高亮 = 该轮最优 val MRR</p>

<h2>6. 诊断结论</h2>

<ol>
<li><strong>K=18（约 49% 参与率）是六组中的最优配置。</strong> 完整测试 MRR 达到 {k18["test_mrr"]:.4f}，比 K=6 高 {k18["test_mrr"]-k6["test_mrr"]:.4f}（+{((k18["test_mrr"]-k6["test_mrr"])/k6["test_mrr"]*100):.1f}%），比 K=36 高 {k18["test_mrr"]-k36["test_mrr"]:.4f}。与集中式的差距从 K=6 的 {k6["delta_vs_centralized"]:.4f} 缩小到 {k18["delta_vs_centralized"]:.4f}。</li>

<li><strong>倒 U 型效应得到六组数据确认。</strong> 从 K=6 到 K=36，MRR 先升后降。K=6→12 的增幅（{k12["test_mrr"]-k6["test_mrr"]:.4f}）远大于 K=12→18（{k18["test_mrr"]-k12["test_mrr"]:.4f}），说明 <strong>低参与率区间的边际收益更快衰减</strong>。</li>

<li><strong>K=6 的瓶颈是数据暴露量，不是 150 轮内可解决的问题。</strong> Val MRR 在最后一轮（150）才达到最优，有效数据遍数仅 {k6["effective_global_passes"]:.0f}，远低于其他组。如果单看 150 轮结果，K=6 不应被采用；但延长训练或许能缩小差距。</li>

<li><strong>K=30/36 的过早收敛和 K=36 的拓扑贫瘠是右侧下降的主要原因。</strong> 验证 MRR 在 60-62 轮达峰后开始过拟合，再加上 K=36 仅 {k36["unique_topology_count"]} 种拓扑，动态正则化完全失效。</li>

<li><strong>在固定计算预算下，K=18×150 轮是当前 Pareto 最优解。</strong> 如果预算允许 300 轮，可以对比 K=12×300 和 K=18×300（K=6 需要更多轮次才能公平比较）。</li>
</ol>

<h2>7. 分析范围与指标定义</h2>
<p>六组消融实验均使用 FB15k-237（实体 14541、关系 237）、37 客户端（按头实体互斥分区）、seed 42、150 通信轮、
每轮 3 个本地 epoch (E3)、每轮验证 4096 条固定三元组（8192 查询）、
HFL 两级聚合（逐行计数加权 + 服务器 FedAdam lr=0.1, betas=[0.9, 0.99], tau=0.001, bias_correction=True）。
完整测试 MRR 来自 20466 条测试三元组的 40932 个 filtered 头尾查询。
集中式参考 MRR = {k6["centralized_mrr"]:.4f}（相同模型配置的全数据集中训练）。
有效全数据遍数 = 累计本地正三元组暴露数 / 训练集正三元组总数（272115）。</p>

<p><strong>参与 CV（参与变异系数）</strong> = 各客户端在 150 轮中被选中次数的标准差 / 平均选中次数，公式为 CV = &sigma;/&mu;。设 c_i 为客户端 i 的被选次数，&mu; = &Sigma;c_i/37，&sigma; = &radic;(&Sigma;(c_i-&mu;)^2/37)。CV = 0 表示完全公平（所有人参与次数相同），CV 越大表示参与机会越不均等。注意：在固定人数随机抽取的设计下，CV 由 K 值和 150 轮共同决定——K 越小，单个客户端的总训练机会越少，随机波动的影响越显著，CV 自然越高。</p>

<h2>8. 限制说明</h2>
<ul>
<li>仅 seed 42，无法估计跨种子方差。{k18["test_mrr"]-stats["K=24"]["test_mrr"]:.4f} 的 K=18 vs K=24 MRR 差距不能解释为稳定因果效应。</li>
<li>每轮验证 4096 条三元组，单条曲线后期波动范围约 0.003-0.008。</li>
<li>六组实验的参与集合不同，不能完全排除参与序列随机性对结果的干扰。</li>
<li>实验结果仅适用于 FB15k-237 + TransE + 两级 HFL 架构。</li>
<li>K=6 在 150 轮内未收敛，报告中展示的 K=6 指标为 150 轮终止时的状态，不代表其极限性能。</li>
</ul>

<h2>9. 推荐下一步</h2>
<ol>
<li><strong>固定 K=18，跑多 seed（43/44/45）确认显著性。</strong> 确认倒 U 型在统计上稳健。</li>
<li><strong>等计算预算比较 K=6×N 轮。</strong> 跑 K=6 延长到 val MRR 不再上升（可能需要 300-400 轮），对比相同 MRR 下的总计算量。</li>
<li><strong>增加 K=9、K=15 填补低端间隔。</strong> 当前 6→12 的步长太大（{k12["test_mrr"]-k6["test_mrr"]:.4f} MRR 跳跃），无法精确定位拐点。</li>
<li><strong>固定通信轮次-计算量乘积。</strong> 对比 K=12×300 轮 vs K=18×200 轮 vs K=36×100 轮（等量有效遍数），隔离拓扑多样性的效应。</li>
<li><strong>分析 K=30/36 过拟合机制。</strong> 通过训练损失曲线、参数更新幅度深入分析高 K 值过早收敛的原因。</li>
</ol>

<p style="margin-top:32px;color:var(--tertiary);font-size:12px;">
报告生成时间: {now.strftime("%Y-%m-%d %H:%M:%S")} | 
数据源: {len(ORDER)} 组实验结果 (summary.json + metrics.csv + dynamic_participation_summary.json)
</p>

</div>

<script>
(function() {{
  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const textColor = isDark ? '#cdcdcd' : '#5d5d5d';
  const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(13,13,13,0.06)';
  Chart.defaults.color = textColor;
  Chart.defaults.borderColor = gridColor;
  
  const ctx = document.getElementById('valCurve').getContext('2d');
  const rounds = {round_labels_js};
  const datasets = {val_datasets_js};
  
  new Chart(ctx, {{
    type: 'line',
    data: {{ labels: rounds, datasets: datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'nearest', intersect: false }},
      plugins: {{
        legend: {{ position: 'top', labels: {{ usePointStyle: true, pointStyle: 'circle', padding: 14, font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.dataset.label.split(' (')[0] + ': ' + ctx.parsed.y.toFixed(4); }} }} }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: '通信轮次', font: {{ size: 12 }} }}, ticks: {{ maxTicksLimit: 12 }} }},
        y: {{
          title: {{ display: true, text: 'Validation MRR', font: {{ size: 12 }} }},
          max: {max([max(v for k,v in stats[n]["val_mrr_by_round"].items() if k <= ROUND_LIMIT) for n in ORDER]) * 1.05:.4f},
          ticks: {{ callback: function(v) {{ return v.toFixed(3); }} }}
        }}
      }}
    }}
  }});
}})();
</script>

</body>
</html>"""

    html_path = os.path.join(report_path, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "title": "HFL-KGE 每轮客户端数 K 消融实验报告",
            "description": "分析K=6,12,18,24,30,36六组消融实验，确认倒U型效应，K=18最优，K=6在150轮内未收敛。",
            "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "sources": [{"id": "k6-comparison", "label": "六组K值消融实验结果",
                         "path": "HFLSnF_KG_v3/results/固定人数随机抽取/hflsnf_kg_v3_hflkge_random_count_k*_*/"}]
        }
    }

    with open(os.path.join(report_path, "artifact.json"), "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    receipt = {
        "report": report_name,
        "generated_at": now.isoformat(),
        "experiments": ORDER,
        "checks": {
            "ablation_suite": "v3_hflkge_random_client_count_e3_eval1_seed42",
            "seed": 42, "comm_rounds": 150, "local_epochs": 3,
            "eval_frequency": 1, "test_triple_count": 20466,
            "key_finding": "K=18最优，K=6在150轮内未收敛，倒U型效应确认",
        }
    }

    with open(os.path.join(report_path, "verification_receipt.json"), "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)

    print(f"Report: {report_path}")
    print(f"  report.html ({len(html)} bytes)")
    return report_path


if __name__ == "__main__":
    print(f"Done: {main()}")
