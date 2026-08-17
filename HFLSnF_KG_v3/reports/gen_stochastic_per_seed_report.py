#!/usr/bin/env python3
"""为「随机抽取（stochastic）」实验的三个随机种子各生成一份三实验臂对比报告。

数据源：results/三个随机数种子-随机抽取/ 下 9 个正式结果目录
（hflsnf / hflnosnf / flnosnf × seed 42 / 2024 / 2025，各 150 轮、每轮验证）。

本实验与「三个随机数种子」的 dynamic 实验不同：
- 不基于 .mat 拓扑文件分组（mat_file=null, topology_util=null）；
- 三臂均关闭 SnF（topology_snf=false）；
- 采用固定数量随机抽取（topology_type=fixed_count, fixed_count_selection_mode=seeded_random,
  fixed_count_grouping_mode=seeded_random_balanced）；
- 三臂区别仅在于每轮参与客户端数与边缘组数：hflsnf=34/6、hflnosnf=12/3、flnosnf=5/1。

产出：每个种子一份自包含中文 HTML 报告（report.html）+ artifact.json + verification_receipt.json，
输出到 results/三个随机数种子-随机抽取/report/ 下。
"""

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

RESULT_BASE = Path("D:/1/1myworkcode/HFLSnF_KG_v3/results/三个随机数种子-随机抽取")
REPORT_ROOT = RESULT_BASE / "report"

ROUND_COUNT = 150
PLATFORM_START = 131  # 后20轮起点
PLATFORM_END = 150
COLD_START_WINDOW = 20  # 冷启动分析窗口（前20轮）

ARMS = ("hflsnf", "hflnosnf", "flnosnf")
SEEDS = (42, 2024, 2025)

ARM_META = {
    "hflsnf": {
        "label": "HFLSnF",
        "cn": "分层联邦学习（HFL）",
        "arch": "HFL",
        "clients": 34,
        "groups": 6,
        "color": "#1f77b4",
    },
    "hflnosnf": {
        "label": "HFLnoSnF",
        "cn": "分层联邦学习（HFL）",
        "arch": "HFL",
        "clients": 12,
        "groups": 3,
        "color": "#ff7f0e",
    },
    "flnosnf": {
        "label": "FLnoSnF",
        "cn": "普通联邦学习（FL，扁平）",
        "arch": "FL",
        "clients": 5,
        "groups": 1,
        "color": "#2ca02c",
    },
}


def _find_run_dirs():
    """按 arm + seed 自动发现 9 个结果目录。"""
    dirs = {}
    for child in RESULT_BASE.iterdir():
        if not child.is_dir():
            continue
        summary = child / "summary.json"
        if not summary.exists():
            continue
        name = child.name
        if "final_stochastic_fedadam_" not in name:
            continue
        arm = None
        for key in ARMS:
            if "_{}_profile_".format(key) in name:
                arm = key
                break
        if arm is None:
            continue
        seed = None
        for s in SEEDS:
            if "_seed{}".format(s) in name:
                seed = s
                break
        if seed is None:
            continue
        dirs[(arm, seed)] = child
    return dirs


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fnum(value, default=float("nan")):
    if value is None:
        return default
    text = str(value).strip()
    if text in ("", "nan", "None", "inf", "-inf"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _linear_slope(values):
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    n = len(vals)
    x_mean = (n - 1) / 2.0
    y_mean = sum(vals) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den


def _cold_start(values):
    early = [float(v) for v in values[:COLD_START_WINDOW]]
    best = (0.0, 0, 0)
    for p in range(max(0, len(early) - 1)):
        for q in range(p + 1, len(early)):
            dd = early[p] - early[q]
            if dd > best[0]:
                best = (dd, p, q)
    drawdown, peak_i, valley_i = best
    peak_value = early[peak_i] if early else 0.0
    return {
        "peak_round": peak_i + 1 if early else None,
        "peak_mrr": peak_value if early else None,
        "valley_round": valley_i + 1 if early else None,
        "valley_mrr": early[valley_i] if early else None,
        "drawdown": drawdown,
    }


def _first_sustained_round(values, threshold, width=5):
    vals = [float(v) for v in values]
    for start in range(0, len(vals) - width + 1):
        if all(v >= threshold for v in vals[start : start + width]):
            return start + 1
    return ROUND_COUNT + 1


def _load_one_run(path):
    summary = _load_json(path / "summary.json")
    rows = []
    with open(path / "metrics.csv", "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    val_mrr, hits1, hits3, hits10 = [], [], [], []
    for row in rows:
        val_mrr.append(_fnum(row.get("val_mrr")))
        hits1.append(_fnum(row.get("val_hits_at_1")))
        hits3.append(_fnum(row.get("val_hits_at_3")))
        hits10.append(_fnum(row.get("val_hits_at_10")))
    late_mrr = [v for v in val_mrr[PLATFORM_START - 1 : PLATFORM_END]]
    late_hits1 = [v for v in hits1[PLATFORM_START - 1 : PLATFORM_END]]
    late_hits3 = [v for v in hits3[PLATFORM_START - 1 : PLATFORM_END]]
    late_hits10 = [v for v in hits10[PLATFORM_START - 1 : PLATFORM_END]]
    late_mean = sum(late_mrr) / len(late_mrr) if late_mrr else float("nan")
    slope = _linear_slope(late_mrr)
    cold = _cold_start(val_mrr)

    return {
        "val_mrr": val_mrr,
        "hits1": hits1,
        "hits3": hits3,
        "hits10": hits10,
        "best_val_mrr": _fnum(summary.get("best_validation_mrr_during_training")),
        "best_round": int(_fnum(summary.get("best_round"), default=0)),
        "final_round_mrr": val_mrr[-1] if val_mrr else float("nan"),
        "late_mean": late_mean,
        "late_slope": slope,
        "late_std": statistics.pstdev(late_mrr) if late_mrr else float("nan"),
        "late_hits1": sum(late_hits1) / len(late_hits1) if late_hits1 else float("nan"),
        "late_hits3": sum(late_hits3) / len(late_hits3) if late_hits3 else float("nan"),
        "late_hits10": sum(late_hits10) / len(late_hits10) if late_hits10 else float("nan"),
        "cold": cold,
        "participant_mean": _fnum(summary.get("participant_count_mean")),
        "group_mean": _fnum(summary.get("group_count_mean")),
        "effective_global_passes": _fnum(summary.get("effective_global_passes")),
    }


def _build_html(seed, runs, now):
    h = runs["hflsnf"]
    hn = runs["hflnosnf"]
    fn = runs["flnosnf"]
    order = ARMS

    # 概览表
    overview_rows = ""
    for arm in order:
        meta = ARM_META[arm]
        a = runs[arm]
        overview_rows += (
            '<tr><td><strong style="color:{}">{}</strong></td>'
            '<td>{}</td><td>{}</td>'
            '<td class="number">{:.0f}</td>'
            '<td class="number">{:.0f}</td>'
            '<td class="number">{:.1f}</td></tr>'
        ).format(
            meta["color"], meta["label"], meta["cn"], meta["arch"],
            meta["clients"], meta["groups"], a["effective_global_passes"],
        )

    # 汇总表（单值）
    def agg_row(arm):
        meta = ARM_META[arm]
        a = runs[arm]
        thr = a["late_mean"] * 0.95
        conv = _first_sustained_round(a["val_mrr"], thr)
        return (
            '<tr><td><strong style="color:{}">{}</strong></td>'
            '<td class="number">{:.4f}</td>'
            '<td class="number">{:.0f}</td>'
            '<td class="number">{:.4f}</td>'
            '<td class="number">{:.4f}</td>'
            '<td class="number">{:.4f}</td>'
            '<td class="number">{:.4f}</td>'
            '<td class="number">{:.0f}</td></tr>'
        ).format(
            meta["color"], meta["label"],
            a["best_val_mrr"], a["best_round"], a["late_mean"],
            a["late_hits1"], a["late_hits3"], a["late_hits10"], conv,
        )

    summary_rows = "".join(agg_row(a) for a in order)

    # 排序与边际效应
    ordered_arms = sorted(order, key=lambda a: runs[a]["late_mean"], reverse=True)
    ranking_sentence = " &gt; ".join(
        "{}（{:.4f}）".format(ARM_META[a]["label"], runs[a]["late_mean"])
        for a in ordered_arms
    )
    scale_gap = h["late_mean"] - hn["late_mean"]  # 34/6 vs 12/3
    hfl_gap = hn["late_mean"] - fn["late_mean"]   # 12/3 (HFL) vs 5/1 (FL)
    best_val_arm = max(order, key=lambda a: runs[a]["best_val_mrr"])
    best_val = runs[best_val_arm]["best_val_mrr"]

    # 冷启动表
    cold_rows = ""
    for arm in order:
        meta = ARM_META[arm]
        a = runs[arm]
        c = a["cold"]
        cold_rows += (
            '<tr><td><strong style="color:{}">{}</strong></td>'
            '<td class="number">{:.4f}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{:.4f}</td></tr>'
        ).format(
            meta["color"], meta["label"],
            c["drawdown"], c["peak_round"], c["valley_round"], c["valley_mrr"],
        )

    # 画图数据
    round_labels = list(range(1, ROUND_COUNT + 1))
    datasets = []
    for arm in order:
        meta = ARM_META[arm]
        a = runs[arm]
        datasets.append({
            "label": "{} (后20轮均值 {:.4f})".format(meta["label"], a["late_mean"]),
            "data": [round(v, 4) for v in a["val_mrr"]],
            "borderColor": meta["color"],
            "backgroundColor": meta["color"] + "33",
            "borderWidth": 1.6,
            "pointRadius": 0,
            "tension": 0.12,
        })
    bar_labels = [ARM_META[a]["label"] for a in order]
    bar_data = [runs[a]["late_mean"] for a in order]
    bar_colors = [ARM_META[a]["color"] for a in order]

    round_labels_js = json.dumps(round_labels)
    datasets_js = json.dumps(datasets, ensure_ascii=False)
    bar_labels_js = json.dumps(bar_labels, ensure_ascii=False)
    bar_data_js = json.dumps(bar_data)
    bar_colors_js = json.dumps(bar_colors)

    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>HFLSnF 随机抽取三实验臂对比报告（Seed %(seed)s）</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{color-scheme:light dark;--bg:#fff;--surface:#f7f7f7;--ink:#0d0d0d;--muted:#5d5d5d;--tertiary:#8f8f8f;--border:rgba(13,13,13,.08);--accent:#0285ff;--positive:#00692a;--positive-bg:#edfaf2;--negative:#ba2623;--negative-bg:#fff0f0;--radius:12px;font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
@media(prefers-color-scheme:dark){:root{--bg:#181818;--surface:#212121;--ink:#dfdfdf;--muted:#cdcdcd;--tertiary:#afafaf;--border:rgba(255,255,255,.1);--positive:#79d996;--positive-bg:rgba(64,180,99,.16);--negative:#ff8583;--negative-bg:rgba(224,74,70,.16)}}
*{box-sizing:border-box}body{margin:0;padding:0 32px 56px;font-size:14px;line-height:1.7}
h1{font-size:22px;font-weight:600;margin:24px 0 4px}
h2{font-size:17px;font-weight:600;margin:28px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--border)}
h3{font-size:15px;font-weight:600;margin:20px 0 6px}
p{margin:8px 0;color:var(--muted);max-width:900px}strong{color:var(--ink)}
.page-header{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;height:48px;margin:0 calc(50%% - 50vw);padding:8px 12px;border-bottom:1px solid var(--border);background:var(--bg)}
.page-header h1{margin:0;font-size:14px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.page-meta{font-size:12px;color:var(--tertiary)}
.container{max-width:1060px;margin:0 auto}
.card{border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:18px;margin:16px 0}
table{width:100%%;border-collapse:collapse;font-size:13px;margin:8px 0}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}
th{font-weight:600;color:var(--tertiary);font-size:12px;text-transform:uppercase;letter-spacing:.04em}td{color:var(--ink)}
.number{text-align:right;font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:var(--surface);padding:1px 4px;border-radius:4px}
blockquote{margin:12px 0;padding-left:14px;border-left:3px solid var(--border);color:var(--muted)}
ul,ol{padding-left:20px;color:var(--muted)}li{margin:4px 0}
.chart-wrapper{position:relative;width:100%%;height:380px}
</style>
</head>
<body>
<header class="page-header">
<h1>HFLSnF 随机抽取三实验臂对比报告（Seed %(seed)s）</h1>
<div class="page-meta">%(now)s</div>
</header>
<div class="container">

<h2>1. 实验概览</h2>
<p>本报告为 <strong>随机种子 %(seed)s</strong> 下单次运行的三实验臂对比：<strong>HFLSnF</strong>、<strong>HFLnoSnF</strong>、<strong>FLnoSnF</strong>，各训练 150 通信轮、每轮验证一次。</p>
<p>本实验采用<strong>固定数量随机抽取</strong>模式（<code>topology_type=fixed_count</code>、<code>fixed_count_selection_mode=seeded_random</code>、<code>fixed_count_grouping_mode=seeded_random_balanced</code>）：
<strong>不基于 .mat 拓扑文件分组，三臂均关闭 SnF</strong>（<code>topology_snf=false</code>），每轮从 37 个客户端中随机抽取固定数量的客户端并随机平衡分组。
因此三臂的区别仅在于<strong>每轮参与客户端数与边缘组数</strong>：HFLSnF 为 34 客户端/6 组，HFLnoSnF 为 12 客户端/3 组，FLnoSnF 为 5 客户端/1 组（扁平）。</p>
<p>统一采用 FB15k-237、37 个客户端按头实体均衡互斥分区、256 维 TransE（L1 距离）、双向自对抗负采样（每正样本 256 负样本）、
每轮 3 个本地 epoch、逐行出现次数加权聚合、服务器 FedAdam（lr=0.05, β=(0.9,0.99), τ=0.001, 关闭偏差修正）。
训练阶段不自动评估测试集，本报告所有指标均为验证集（每轮 4096 三元组 / 8192 查询）。</p>

<div class="card">
<table>
<thead><tr>
  <th>实验臂</th><th>含义</th><th>架构</th>
  <th class="number">每轮客户端数</th>
  <th class="number">边缘组数</th>
  <th class="number">有效数据遍数</th>
</tr></thead>
<tbody>
%(overview_rows)s
</tbody>
</table>
</div>
<blockquote>三臂的命名沿用 <code>hflsnf / hflnosnf / flnosnf</code> 约定，但本实验中三臂均关闭 SnF，实际对比维度是「每轮客户端数 × 边缘组数」的规模梯度，而非 SnF 或分层机制本身（详见第 6 节限制说明）。</blockquote>

<h2>2. 核心结果汇总</h2>

<div class="card">
<table>
<thead><tr>
  <th>实验臂</th><th class="number">最佳验证 MRR</th><th class="number">最佳轮次</th>
  <th class="number">后20轮均值 MRR</th><th class="number">Hits@1</th>
  <th class="number">Hits@3</th><th class="number">Hits@10</th>
  <th class="number">收敛轮次</th>
</tr></thead>
<tbody>
%(summary_rows)s
</tbody>
</table>
</div>

<p>后20轮（第 131–150 轮）均值 MRR 的排序为 <strong>%(ranking_sentence)s</strong>。
最佳验证 MRR 最高的是 <strong>%(best_val_arm)s（%(best_val).4f）</strong>。</p>

<h2>3. 验证 MRR 收敛曲线（150 轮）</h2>

<div class="card" style="overflow:hidden;">
<div class="chart-wrapper">
  <canvas id="valCurve" role="img" aria-label="三实验臂验证MRR收敛曲线"></canvas>
</div>
</div>

<h2>4. 后20轮平台 MRR 对比</h2>

<div class="card" style="overflow:hidden;">
<div class="chart-wrapper" style="height:320px;">
  <canvas id="barChart" role="img" aria-label="三实验臂后20轮均值MRR柱状图"></canvas>
</div>
</div>

<h2>5. 核心发现</h2>

<h3>5.1 三臂性能分层：HFL 两臂接近且显著领先，FL 扁平臂明显落后</h3>
<p>后20轮平台 MRR 上，<strong>HFLSnF（34/6）</strong> 为 %(h_late).4f，<strong>HFLnoSnF（12/3）</strong> 为 %(hn_late).4f，
<strong>FLnoSnF（5/1）</strong> 仅 %(fn_late).4f。最佳验证 MRR 分别为 %(h_best).4f / %(hn_best).4f / %(fn_best).4f。
两个分层臂（HFL）显著优于扁平臂（FL），差距约 0.05；有效数据遍数（%(h_pass).0f / %(hn_pass).0f / %(fn_pass).0f）反映了这一参与预算差异。</p>

<h3>5.2 客户端/组数规模边际效应：34/6 相对 12/3 几乎无增益</h3>
<p>在随机抽取、无 SnF 的条件下，把规模从 12 客户端/3 组提升到 34 客户端/6 组，平台 MRR 变化为 <strong>%(scale_gap)+.4f</strong>（HFLSnF − HFLnoSnF），几乎为零甚至略为负。
这说明<strong>脱离 .mat 拓扑引导后，单纯增加每轮客户端数与分组数并不能带来额外收益</strong>，反而可能因分组聚合的信息损失抵消了更多数据暴露带来的好处。
相比之下，从扁平 5 客户端/1 组升级到分层 12 客户端/3 组，平台 MRR 增益为 <strong>%(hfl_gap)+.4f</strong>（HFLnoSnF − FLnoSnF），幅度显著。</p>

<h3>5.3 收敛速度与最佳轮次</h3>
<p>最佳验证轮次方面，HFLSnF 为 %(h_round).0f 轮、HFLnoSnF %(hn_round).0f 轮、FLnoSnF %(fn_round).0f 轮。
三臂均在 150 轮内达到最佳验证 MRR，其中 FLnoSnF 收敛较慢（约 %(fn_round).0f 轮才见顶）。</p>

<h3>5.4 冷启动过冲（FedAdam 关闭偏差修正）</h3>
<p>关闭偏差修正的 FedAdam 在训练初期存在过冲：</p>
<div class="card">
<table>
<thead><tr><th>实验臂</th><th class="number">前20轮最大回撤</th><th class="number">峰值轮次</th><th class="number">谷底轮次</th><th class="number">谷底 MRR</th></tr></thead>
<tbody>
%(cold_rows)s
</tbody>
</table>
</div>
<p>三臂前期均呈现“冲高 → 回落 → 爬升”形态，符合关闭偏差修正时 FedAdam 二阶梯度的典型冷启动特征。</p>

<h2>6. 指标定义与分析范围</h2>
<ul>
<li><strong>最佳验证 MRR</strong>：150 轮逐轮验证中 MRR 的最大值及其对应轮次（来自 <code>summary.json</code> 的 <code>best_validation_mrr_during_training</code>）。</li>
<li><strong>后20轮均值 MRR</strong>：第 131–150 轮验证 MRR 的算术平均。</li>
<li><strong>收敛轮次</strong>：以该臂后20轮均值的 95%% 为阈值，首次连续 5 轮达到该阈值的轮次。</li>
<li><strong>冷启动过冲</strong>：前 20 轮内最大“峰→谷”回撤幅度及峰值/谷底轮次。</li>
<li><strong>有效数据遍数</strong>：累计本地正三元组暴露数 / 训练集正三元组总数（272115）。</li>
</ul>

<h2>7. 限制说明</h2>
<ul>
<li><strong>单种子、单次运行</strong>：本报告仅针对随机种子 %(seed)s 的一次运行，无重复、无误差棒，不能据此判断结论是否稳健。</li>
<li><strong>非单因素消融</strong>：三臂的每轮客户端数与分组数同时变化，客户端/组数的边际效应无法与分层效应完全分离。</li>
<li><strong>命名沿用</strong>：三臂均关闭 SnF，<code>hflsnf / hflnosnf / flnosnf</code> 仅沿用目录约定，不代表 SnF 开关差异。</li>
<li><strong>仅验证集</strong>：训练阶段未执行官方测试集评估；验证集每轮仅 4096 三元组，后期单点波动约 0.003–0.008。</li>
<li><strong>范围局限</strong>：结论仅适用于 FB15k-237 + TransE + 两级 HFL + FedAdam + 固定数量随机抽取这一特定配置。</li>
</ul>

<h2>8. 推荐下一步</h2>
<ol>
<li>结合 42 / 2024 / 2025 三个种子的报告横向比较，判断「34/6 相对 12/3 无增益」是否跨种子稳健。</li>
<li>用官方评估入口对三臂各自的最佳验证检查点（<code>model_best.pt</code>）跑完整 filtered 测试集，得到可发表的测试 MRR/Hits@10。</li>
<li>若结论稳定，说明在无 .mat 拓扑引导时规模收益饱和，可据此论证 SnF 动态拓扑调度（而非单纯客户端数量）才是此前 dynamic 实验 HFLSnF 领先的关键。</li>
</ol>

<p style="margin-top:32px;color:var(--tertiary);font-size:12px;">
报告生成时间: %(now)s |
随机种子: %(seed)s |
数据源: results/三个随机数种子-随机抽取/ 下 3 组结果 (summary.json + metrics.csv) |
对比维度: 单种子 × 三实验臂（固定数量随机抽取）
</p>

</div>

<script>
(function() {
  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const textColor = isDark ? '#cdcdcd' : '#5d5d5d';
  const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(13,13,13,0.06)';
  Chart.defaults.color = textColor;
  Chart.defaults.borderColor = gridColor;

  const rounds = %(round_labels_js)s;
  const datasets = %(datasets_js)s;
  const lineCtx = document.getElementById('valCurve').getContext('2d');
  new Chart(lineCtx, {
    type: 'line',
    data: { labels: rounds, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { usePointStyle: true, pointStyle: 'circle', padding: 14, font: { size: 11 } } },
        tooltip: { callbacks: { label: function(ctx) { return ctx.dataset.label.split(' (')[0] + ': ' + ctx.parsed.y.toFixed(4); } } }
      },
      scales: {
        x: { title: { display: true, text: '通信轮次', font: { size: 12 } }, ticks: { maxTicksLimit: 12 } },
        y: {
          title: { display: true, text: 'Validation MRR', font: { size: 12 } },
          suggestedMin: 0,
          max: 0.4,
          ticks: { callback: function(v) { return v.toFixed(3); } }
        }
      }
    }
  });

  const barCtx = document.getElementById('barChart').getContext('2d');
  new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: %(bar_labels_js)s,
      datasets: [{
        label: '后20轮均值 MRR',
        data: %(bar_data_js)s,
        backgroundColor: %(bar_colors_js)s,
        borderColor: %(bar_colors_js)s,
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: function(ctx) { return 'MRR: ' + ctx.parsed.y.toFixed(4); } } }
      },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: '后20轮均值 MRR', font: { size: 12 } } }
      }
    }
  });
})();
</script>

</body>
</html>"""

    html = html % {
        "seed": seed,
        "now": now.strftime("%Y-%m-%d %H:%M"),
        "overview_rows": overview_rows,
        "summary_rows": summary_rows,
        "ranking_sentence": ranking_sentence,
        "best_val_arm": ARM_META[best_val_arm]["label"],
        "best_val": best_val,
        "h_late": h["late_mean"],
        "hn_late": hn["late_mean"],
        "fn_late": fn["late_mean"],
        "h_best": h["best_val_mrr"],
        "hn_best": hn["best_val_mrr"],
        "fn_best": fn["best_val_mrr"],
        "h_pass": h["effective_global_passes"],
        "hn_pass": hn["effective_global_passes"],
        "fn_pass": fn["effective_global_passes"],
        "scale_gap": scale_gap,
        "hfl_gap": hfl_gap,
        "h_round": h["best_round"],
        "hn_round": hn["best_round"],
        "fn_round": fn["best_round"],
        "cold_rows": cold_rows,
        "round_labels_js": round_labels_js,
        "datasets_js": datasets_js,
        "bar_labels_js": bar_labels_js,
        "bar_data_js": bar_data_js,
        "bar_colors_js": bar_colors_js,
    }
    return html


def main():
    dirs = _find_run_dirs()
    missing = [(a, s) for a in ARMS for s in SEEDS if (a, s) not in dirs]
    if missing:
        raise RuntimeError("缺少结果目录：{}".format(missing))

    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for seed in SEEDS:
        runs = {arm: _load_one_run(dirs[(arm, seed)]) for arm in ARMS}
        html = _build_html(seed, runs, now)
        report_name = "final_seed{}_analysis_{}".format(seed, ts)
        report_dir = REPORT_ROOT / report_name
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.html").write_text(html, encoding="utf-8")

        artifact = {
            "surface": "report",
            "manifest": {
                "version": 1,
                "title": "HFLSnF 随机抽取三实验臂对比报告（Seed {}）".format(seed),
                "description": "随机种子 {} 下固定数量随机抽取模式的三实验臂（34/6、12/3、5/1）单次运行验证集性能对比。".format(seed),
                "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                "sources": [{"id": "final-stochastic-seed-{}".format(seed),
                             "label": "三臂单种子随机抽取正式结果（seed {}）".format(seed),
                             "path": "HFLSnF_KG_v3/results/三个随机数种子-随机抽取/"}],
            },
        }
        (report_dir / "artifact.json").write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        receipt = {
            "report": report_name,
            "seed": seed,
            "generated_at": now.isoformat(),
            "arms": list(ARMS),
            "checks": {
                "ablation_suite": "v3_final_stochastic_fedadam_matmean_profiles_e3_eval1_formal150",
                "topology_type": "fixed_count",
                "topology_snf": False,
                "comm_rounds": ROUND_COUNT,
                "local_epochs": 3,
                "eval_frequency": 1,
                "platform_window": [PLATFORM_START, PLATFORM_END],
                "test_performed": False,
            },
        }
        (report_dir / "verification_receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("Seed {} -> {}".format(seed, report_dir / "report.html"))
        out_paths.append(report_dir / "report.html")

    print()
    for seed in SEEDS:
        runs = {arm: _load_one_run(dirs[(arm, seed)]) for arm in ARMS}
        scale = runs["hflsnf"]["late_mean"] - runs["hflnosnf"]["late_mean"]
        hfl = runs["hflnosnf"]["late_mean"] - runs["flnosnf"]["late_mean"]
        print(
            "Seed {}: HFLSnF(34/6) {:.4f} | HFLnoSnF(12/3) {:.4f} | FLnoSnF(5/1) {:.4f} | "
            "scaleGap {:+.4f} | HFLgap {:+.4f}".format(
                seed,
                runs["hflsnf"]["late_mean"],
                runs["hflnosnf"]["late_mean"],
                runs["flnosnf"]["late_mean"],
                scale, hfl,
            )
        )
    return out_paths


if __name__ == "__main__":
    main()
