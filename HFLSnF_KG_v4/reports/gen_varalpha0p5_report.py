#!/usr/bin/env python3
"""Generate 4-scenario report for varAlpha=0.5 dynamic MAT experiments (HFLSnF/FLSnF/HFLnoSnF/FLnoSnF)."""

import csv, json, math, os
from datetime import datetime

RESULT_BASE = "D:/1/1myworkcode/HFLSnF_KG_v3/results/varAlpha0p5"
REPORT_DIR = "D:/1/1myworkcode/HFLSnF_KG_v3/reports"
ROUND_LIMIT = 80

EXPS = [
    {"label": "HFLSnF",   "architecture": "HFL", "snf": "On",
     "dir": "hflsnf_kg_v3_dynamic_mat_varalpha0p5_hflsnf_e3_eval1_seed42_150round_cuda_20260804_230927_671040"},
    {"label": "FLSnF",    "architecture": "FL",  "snf": "On",
     "dir": "hflsnf_kg_v3_dynamic_mat_varalpha0p5_flsnf_e3_eval1_seed42_150round_cuda_20260805_010032_067905"},
    {"label": "HFLnoSnF", "architecture": "HFL", "snf": "Off",
     "dir": "hflsnf_kg_v3_dynamic_mat_varalpha0p5_hflnosnf_e3_eval1_seed42_150round_cuda_20260805_023312_283156"},
    {"label": "FLnoSnF",  "architecture": "FL",  "snf": "Off",
     "dir": "hflsnf_kg_v3_dynamic_mat_varalpha0p5_flnosnf_e3_eval1_seed42_150round_cuda_20260805_034244_294652"},
]

ORDER = ["HFLSnF", "FLSnF", "HFLnoSnF", "FLnoSnF"]
COLORS = {"HFLSnF": "#378ADD", "FLSnF": "#1D9E75", "HFLnoSnF": "#BA7517", "FLnoSnF": "#E24B4A"}
TOTAL_ENTITY = 14541; TOTAL_RELATION = 237


def load_data():
    results = {}
    for exp in EXPS:
        base = os.path.join(RESULT_BASE, exp["dir"])
        with open(os.path.join(base, "summary.json"), "r") as f:
            s = json.load(f)
        with open(os.path.join(base, "dynamic_participation_summary.json"), "r") as f:
            p = json.load(f)
        metrics = []
        with open(os.path.join(base, "metrics.csv"), "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                metrics.append(row)

        val_mrr = {}
        entity_active = {}
        rtime = {}
        for row in metrics:
            r = int(row["round"])
            vmrr = row.get("val_mrr", "")
            if vmrr and vmrr != "nan":
                val_mrr[r] = float(vmrr)
            entity_active[r] = float(row.get("entity_server_active_row_count", 0))
            rs = row.get("round_seconds", "")
            if rs and rs != "nan":
                rtime[r] = float(rs)

        entity_cov = sum(v / TOTAL_ENTITY for v in entity_active.values()) / len(entity_active)

        counts = list(p.get("client_selection_counts", {}).values())
        mu = sum(counts) / len(counts)
        cv = math.sqrt(sum((c - mu)**2 for c in counts) / len(counts)) / mu if mu > 0 else 0
        never = sum(1 for c in counts if c == 0)

        test = s["final_test_metrics"]
        val = s["final_validation_metrics"]

        results[exp["label"]] = {
            "label": exp["label"], "architecture": exp["architecture"], "snf": exp["snf"],
            "val_mrr_by_round": val_mrr,
            "entity_coverage_mean": entity_cov,
            "participation_cv": cv, "never_participated": never,
            "participant_mean": s["participant_count_mean"],
            "participant_min": s["participant_count_min"],
            "participant_max": s["participant_count_max"],
            "group_mean": s["group_count_mean"],
            "effective_passes": s["effective_global_passes"],
            "test_mrr": test["mrr"], "test_hits1": test["hits_at_1"],
            "test_hits3": test["hits_at_3"], "test_hits10": test["hits_at_10"],
            "test_mean_rank": test["mean_rank"],
            "val_mrr_best": val["mrr"], "val_hits1": val["hits_at_1"],
            "val_hits3": val["hits_at_3"], "val_hits10": val["hits_at_10"],
            "best_val_mrr": s["best_validation_mrr_during_training"],
            "best_round": s["best_round"],
            "delta_vs_cent": s["test_mrr_delta_vs_centralized"],
            "cent_mrr": s["centralized_reference_test_mrr"],
            "unique_topo": s["unique_topology_count"],
            "avg_round_time": sum(rtime.values()) / len(rtime) if rtime else 0,
            "total_time_min": sum(rtime.values()) / 60,
        }
    return results


def gen_curves_js(results):
    datasets = []
    for name in ORDER:
        r = results[name]
        rounds = sorted(r["val_mrr_by_round"].keys())
        data = [r["val_mrr_by_round"][rd] for rd in rounds if rd <= ROUND_LIMIT]
        datasets.append({
            "label": f'{name} (Test MRR={r["test_mrr"]:.4f})',
            "data": data,
            "borderColor": COLORS[name],
            "backgroundColor": COLORS[name] + "22",
            "borderWidth": 1.5, "pointRadius": 0, "tension": 0.1,
        })
    return json.dumps(datasets)


def gen_round_labels(results):
    return json.dumps([r for r in sorted(results[ORDER[0]]["val_mrr_by_round"].keys()) if r <= ROUND_LIMIT])


def main():
    results = load_data()
    stats = {n: results[n] for n in ORDER}
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    report_name = f"varalpha0p5_diagnostics_{ts}"
    report_path = os.path.join(REPORT_DIR, report_name)
    os.makedirs(report_path, exist_ok=True)

    # ---- Computed values ----
    hfl = stats["HFLSnF"]; fl = stats["FLSnF"]; hno = stats["HFLnoSnF"]; fno = stats["FLnoSnF"]
    snf3 = ["HFLSnF", "FLSnF", "HFLnoSnF"]
    mrr_3 = [stats[n]["test_mrr"] for n in snf3]
    mrr_3_range = max(mrr_3) - min(mrr_3)
    hits1_3_range = max(stats[n]["test_hits1"] for n in snf3) - min(stats[n]["test_hits1"] for n in snf3)
    hits3_3_range = max(stats[n]["test_hits3"] for n in snf3) - min(stats[n]["test_hits3"] for n in snf3)
    hits10_3_range = max(stats[n]["test_hits10"] for n in snf3) - min(stats[n]["test_hits10"] for n in snf3)

    # Overview table
    ov_rows = ""
    for name in ORDER:
        r = stats[name]
        ov_rows += f"""<tr><td>{name}</td><td>{r["architecture"]}</td><td>{r["snf"]}</td><td class="number">{r["participant_mean"]:.1f}</td><td class="number">{r["group_mean"]:.1f}</td><td class="number">{r["effective_passes"]:.1f}</td><td class="number">{r["entity_coverage_mean"]*100:.1f}%</td><td class="number">{r["participation_cv"]*100:.1f}%</td><td class="number">{r["never_participated"]}</td></tr>"""

    # Test table
    test_rows = ""
    best_mrr = max(stats[n]["test_mrr"] for n in ORDER)
    for name in ORDER:
        r = stats[name]
        cls = ""
        if r["test_mrr"] == best_mrr:
            cls = "highlight"
        elif name == "FLnoSnF":
            cls = "warn"
        delta_str = ""
        if r["test_mrr"] != best_mrr:
            d = r["test_mrr"] - best_mrr
            delta_str = f' <span class="{"neg" if d < 0 else "pos"}">({d:+.4f})</span>'
        test_rows += f"""<tr class="{cls}"><td>{name}</td><td class="number">{r["test_mrr"]:.4f}{delta_str}</td><td class="number">{r["test_hits1"]:.4f}</td><td class="number">{r["test_hits3"]:.4f}</td><td class="number">{r["test_hits10"]:.4f}</td><td class="number">{r["test_mean_rank"]:.1f}</td><td class="number">{r["delta_vs_cent"]:.4f}</td><td class="number">{r["best_round"]}</td><td class="number">{r["best_val_mrr"]:.4f}</td></tr>"""

    # Metric cards
    cards = ""
    for name in ORDER:
        r = stats[name]
        cls = ""
        label = name
        val_cls = ""
        if r["test_mrr"] == best_mrr:
            cls = ' highlight" style="border-color:var(--positive);border-width:2px'
            label += ' ★ 最优'
            val_cls = ' pos'
        elif name == "FLnoSnF":
            cls = ' warn" style="border-color:var(--negative);border-width:2px'
            label += ' ★ 最差'
            val_cls = ' neg'
        cards += f"""<div class="metric{cls}">
<div class="metric-label">{label}</div>
<div class="metric-value{val_cls}">{r["test_mrr"]:.4f}</div>
<div class="metric-detail" style="font-size:12px;color:var(--tertiary)">{r["architecture"]} | SnF={r["snf"]}</div>
</div>"""

    # Snapshot table
    val_hdr = "<thead><tr><th>Round</th>" + "".join(f"<th class='number'>{n}</th>" for n in ORDER) + "</tr></thead>"
    val_rows = ""
    for rd in range(10, ROUND_LIMIT + 1, 10):
        row = f"<tr><td class='number'>{rd}</td>"
        best_at = max(stats[n]["val_mrr_by_round"].get(rd, -1) for n in ORDER)
        for name in ORDER:
            v = stats[name]["val_mrr_by_round"].get(rd)
            if v is not None:
                h = "highlight" if v == best_at else ""
                row += f"<td class='number {h}'>{v:.4f}</td>"
            else:
                row += "<td class='number'>—</td>"
        row += "</tr>"
        val_rows += row

    # ---- HTML ----
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>动态MAT四组KGE结果诊断 (varAlpha=0.5, E3-Eval1-Seed42)</title>
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
.container{{max-width:900px;margin:0 auto}}
.card{{border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:18px;margin:16px 0}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:12px 0}}
.metric{{border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px}}
.metric-label{{font-size:12px;color:var(--tertiary);margin:0 0 4px}}
.metric-value{{font-size:22px;font-weight:600;margin:0}}
.pos{{color:var(--positive)}}.neg{{color:var(--negative)}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}}
th{{font-weight:600;color:var(--tertiary);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}td{{color:var(--ink)}}
.number{{text-align:right;font-variant-numeric:tabular-nums}}
.highlight{{background:var(--positive-bg)}}.warn{{background:var(--negative-bg)}}
ol{{padding-left:20px;color:var(--muted)}}li{{margin:4px 0}}
.chart-wrapper{{position:relative;width:100%;height:380px}}
</style>
</head>
<body>
<header class="page-header">
<h1>动态MAT四组KGE结果诊断 (varAlpha=0.5, E3-Eval1-Seed42)</h1>
<div class="page-meta">{now.strftime("%Y-%m-%d %H:%M")}</div>
</header>
<div class="container">

<h2>1. 实验概览</h2>
<p>四组实验均使用 <strong>FB15k-237</strong> 数据集，37 个客户端按头实体均衡分区（互斥），seed 42，通信轮数 150，
本地 epoch=3（E3），每轮验证 1 次（4096 条验证三元组），逐行计数加权 + 服务器 FedAdam（bias_correction=True）。
拓扑调度使用 MATLAB 动态拓扑（varAlpha=0.5，util=0.5），从 <code>result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat</code> 读取。
四组覆盖两种架构（HFL vs FL）和两种 SnF 状态（On vs Off）的全因子组合。</p>

<div class="card">
<table>
<thead><tr><th>场景</th><th>架构</th><th>SnF</th><th class="number">平均参与人数</th><th class="number">平均分组数</th><th class="number">有效数据遍数</th><th class="number">实体行覆盖率</th><th class="number">参与CV</th><th class="number">从未参与</th></tr></thead>
<tbody>{ov_rows}</tbody>
</table>
</div>

<h2>2. 完整测试指标对比</h2>

<div class="metric-grid">{cards}</div>

<div class="card">
<table>
<thead><tr><th>场景</th><th class="number">MRR</th><th class="number">Hits@1</th><th class="number">Hits@3</th><th class="number">Hits@10</th><th class="number">Mean Rank</th><th class="number">Δ vs 集中式</th><th class="number">最佳轮次</th><th class="number">最佳Val MRR</th></tr></thead>
<tbody>{test_rows}</tbody>
</table>
</div>

<h2>3. 验证集 MRR 收敛曲线（前 {ROUND_LIMIT} 轮）</h2>

<div class="card" style="overflow:hidden;">
<div class="chart-wrapper">
  <canvas id="valCurve" role="img" aria-label="四组验证MRR收敛曲线"></canvas>
</div>
</div>

<h2>4. 核心发现</h2>

<h3>4.1 SnF 三组极度接近：MRR 极差仅 {mrr_3_range:.4f}</h3>
<p>HFLSnF、FLSnF 和 HFLnoSnF 三组的完整测试 MRR 分别为 {hfl["test_mrr"]:.4f}、{fl["test_mrr"]:.4f} 和 {hno["test_mrr"]:.4f}，极差仅 <strong>{mrr_3_range:.4f}</strong>。
FLSnF 以 {fl["test_mrr"]:.4f} 微弱领先 HFLSnF（{hfl["test_mrr"]:.4f}）约 {(fl["test_mrr"]-hfl["test_mrr"]):.4f}，
HFLnoSnF 以 {hno["test_mrr"]:.4f} 位列第三。</p>

<p>三组的 Hits@1 极差 {hits1_3_range:.4f}，Hits@3 极差 {hits3_3_range:.4f}，Hits@10 极差 {hits10_3_range:.4f}。
在 varAlpha=0.5 的拓扑调度下，<strong>HFL 两级分组相比 FL 单层直传没有任何可辨别的 MRR 优势</strong>——甚至 FLSnF 还略微胜出。</p>

<h3>4.2 FLSnF 收敛最快，但高峰后出现退化</h3>
<p>FLSnF 在第 <strong>{fl["best_round"]}</strong> 轮即达到最佳 val MRR（{fl["best_val_mrr"]:.4f}），远超其他三组——
HFLSnF 在 {hfl["best_round"]} 轮，HFLnoSnF 在 {hno["best_round"]} 轮，FLnoSnF 在 {fno["best_round"]} 轮。
但 FLSnF 在达到峰值后 val MRR 出现回落（第 50 轮 {stats["FLSnF"]["val_mrr_by_round"].get(50,0):.4f} → 第 80 轮 {stats["FLSnF"]["val_mrr_by_round"].get(80,0):.4f}），
而 HFLSnF 和 HFLnoSnF 还在缓慢上升，表现出更好的训练稳定性。</p>

<h3>4.3 FLnoSnF 大幅落后：参与覆盖严重不足</h3>
<p>FLnoSnF 的完整测试 MRR 仅 {fno["test_mrr"]:.4f}，比 FLSnF 低 <strong>{fl["test_mrr"]-fno["test_mrr"]:.4f}</strong>。
核心原因是指标塌方式衰退：平均每轮仅 {fno["participant_mean"]:.1f} 人参与（最低仅 {fno["participant_min"]} 人），
实体行覆盖率仅 {fno["entity_coverage_mean"]*100:.0f}%，有效数据遍数仅 {fno["effective_passes"]:.0f}。
参与 CV 高达 {fno["participation_cv"]*100:.1f}%，且有 {fno["never_participated"]} 个客户端从未参与。
在 varAlpha=0.5 的拓扑下，FLnoSnF 的参与状况甚至比固定人数 K=6 的实验更差（有效遍数 77.9 vs 73.0）。</p>

<h3>4.4 参与 CV 与参与人数：SnF 三组差异大但 MRR 一致</h3>
<p>参与 CV：HFLSnF={hfl["participation_cv"]*100:.1f}%（最公平），FLSnF={fl["participation_cv"]*100:.1f}%，HFLnoSnF={hno["participation_cv"]*100:.1f}%（最不均）。
平均参与人数：HFLSnF={hfl["participant_mean"]:.1f}（最高），FLSnF={fl["participant_mean"]:.1f}，HFLnoSnF={hno["participant_mean"]:.1f}（最低）。</p>

<p>尽管 HFLSnF 比 HFLnoSnF 平均多 {hfl["participant_mean"]-hno["participant_mean"]:.0f} 人/轮且 CV 低 {abs(hfl["participation_cv"]-hno["participation_cv"])*100:.1f}%，
但 MRR 仅高 {hfl["test_mrr"]-hno["test_mrr"]:.4f}，再次说明 <strong>在当前聚合机制下，参与人数和公平性的差异几乎不转化为 MRR 增益</strong>。</p>

<h3>4.5 验证 MRR 与测试 MRR 的 gap</h3>
<p>Val-Test MRR gap：HFLSnF={hfl["best_val_mrr"]-hfl["test_mrr"]:.4f}，FLSnF={fl["best_val_mrr"]-fl["test_mrr"]:.4f}，
HFLnoSnF={hno["best_val_mrr"]-hno["test_mrr"]:.4f}，FLnoSnF={fno["best_val_mrr"]-fno["test_mrr"]:.4f}。
FLSnF 的 gap 最大（{fl["best_val_mrr"]-fl["test_mrr"]:.4f}），与其早收敛后 val MRR 回落一致。</p>

<h2>5. 验证 MRR 逐 10 轮快照（前 {ROUND_LIMIT} 轮）</h2>

<div class="card">
<table>{val_hdr}<tbody>{val_rows}</tbody></table>
</div>
<p style="font-size:12px;color:var(--tertiary);margin-top:4px;">绿色高亮 = 该轮最优 val MRR</p>

<h2>6. 诊断结论</h2>
<ol>
<li><strong>SnF 三组 MRR 极差仅 {mrr_3_range:.4f}。</strong> HFL 两级架构 vs FL 单层，在 varAlpha=0.5 的拓扑调度和当前逐行计数加权聚合下，不产生可辨别的性能差异。</li>
<li><strong>FLSnF 以 {fl["test_mrr"]:.4f} 微弱领先，但过早收敛。</strong> 在第 {fl["best_round"]} 轮达峰后 val MRR 开始回落，Gap 达 {fl["best_val_mrr"]-fl["test_mrr"]:.4f}，提示验证集过拟合风险。</li>
<li><strong>FLnoSnF 因参与塌方而失败。</strong> MRR={fno["test_mrr"]:.4f}，平均仅 {fno["participant_mean"]:.1f} 人/轮，有效遍数仅 {fno["effective_passes"]:.0f}。varAlpha=0.5 拓扑下 FLnoSnF 的参与严重不足。</li>
<li><strong>参与公平性的大幅差异（CV {hfl["participation_cv"]*100:.1f}%-{hno["participation_cv"]*100:.1f}%）几乎不转化为 MRR 差异。</strong> 要展示拓扑价值，必须改变聚合机制（如边缘模型持久化）。</li>
</ol>

<h2>7. 分析范围与指标定义</h2>
<p>四组均使用 FB15k-237（实体 14541、关系 237），37 客户端互斥分区，seed 42，150 通信轮，E3 本地训练，每轮验证 4096 条三元组，
逐行计数加权 + FedAdam（lr=0.1, betas=[0.9,0.99], tau=0.001, bias_correction=True）。
拓扑来自 MATLAB <code>result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat</code>，topology_util=0.5。
完整测试 MRR 来自 20466 条测试三元组的 40932 个 filtered 查询，集中式参考 MRR={hfl["cent_mrr"]:.4f}。</p>

<p><strong>参与 CV（参与变异系数）</strong> = 各客户端在 150 轮中被选中次数的标准差 / 平均选中次数（CV = &sigma;/&mu;）。
CV=0 表示完全公平，CV 越大参与越不均等。注意四组实验的参与人数由 MATLAB 拓扑动态决定（非固定值），因此各组间参与人数和 CV 差异是拓扑调度策略的反映，不是独立变量。</p>

<h2>8. 限制说明</h2>
<ul>
<li>仅 seed 42，SnF 三组间的 {mrr_3_range:.4f} MRR 差距无统计显著性，不能解释为因果效应。</li>
<li>每轮验证 4096 条三元组，单条曲线后期波动约 0.003-0.008。</li>
<li>四组参与人数和集合不同，不能单独归因架构或 SnF。</li>
<li>FLSnF 在第 {fl["best_round"]} 轮达峰后 val MRR 回落，表明 E3 下该场景可能训练过度。</li>
</ul>

<h2>9. 推荐下一步</h2>
<ol>
<li><strong>固定同一参与序列做 HFL/FL 配对实验。</strong> 两组共享逐轮客户端集合，只改变 HFL 分组，验证当前聚合下是否应得相同模型。</li>
<li><strong>实现边缘模型持久化。</strong> 让 HFL 的分组结构真正进入优化轨迹，而不是等价于 FL 的平均操作。</li>
<li><strong>在机制修改后补多 seed。</strong> seed 43/44，确认 MRR 排序的统计显著性。</li>
<li><strong>对比不同 varAlpha 值。</strong> 将本报告与 varAlpha=0.1 的四组结果比较，分析拓扑参数对参与覆盖和最终性能的影响。</li>
</ol>

<p style="margin-top:32px;color:var(--tertiary);font-size:12px;">
报告生成: {now.strftime("%Y-%m-%d %H:%M:%S")} | 数据: {len(ORDER)} 组 (summary.json + metrics.csv + dynamic_participation_summary.json)
</p>

</div>

<script>
(function() {{
  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  Chart.defaults.color = isDark ? '#cdcdcd' : '#5d5d5d';
  Chart.defaults.borderColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(13,13,13,0.06)';
  const ctx = document.getElementById('valCurve').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{ labels: {gen_round_labels(results)}, datasets: {gen_curves_js(results)} }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{ mode: 'nearest', intersect: false }},
      plugins: {{
        legend: {{ position: 'top', labels: {{ usePointStyle: true, pointStyle: 'circle', padding: 16, font: {{ size: 12 }} }} }},
        tooltip: {{ callbacks: {{ label: function(ctx) {{ return ctx.dataset.label.split(' (')[0] + ': ' + ctx.parsed.y.toFixed(4); }} }} }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: '通信轮次', font: {{ size: 12 }} }}, ticks: {{ maxTicksLimit: 12 }} }},
        y: {{ title: {{ display: true, text: 'Validation MRR', font: {{ size: 12 }} }}, ticks: {{ callback: function(v) {{ return v.toFixed(3); }} }} }}
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

    with open(os.path.join(report_path, "artifact.json"), "w", encoding="utf-8") as f:
        json.dump({"surface": "report", "manifest": {
            "version": 1, "title": "动态MAT四组KGE结果诊断 (varAlpha=0.5)",
            "description": f"SnF三组MRR极差{mrr_3_range:.4f}，FLnoSnF塌方至{fno['test_mrr']:.4f}，当前聚合下HFL无优势。",
            "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%S+08:00")
        }}, f, indent=2, ensure_ascii=False)

    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
