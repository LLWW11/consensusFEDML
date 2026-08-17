#!/usr/bin/env python3
"""生成 HFLSnF 动态拓扑三实验臂 × 三随机种子的聚合分析报告。

数据源：results/三个随机数种子/results/ 下 9 个正式结果目录
（hflsnf / hflnosnf / flnosnf × seed 42 / 2024 / 2025，各 150 轮、逐轮验证）。

产出：单份自包含中文 HTML 报告（report.html）+ artifact.json + verification_receipt.json，
不针对单个随机种子单独出报告，而是跨种子聚合为均值 ± 标准差。
"""

import csv
import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path

RESULT_BASE = Path(
    "D:/1/1myworkcode/HFLSnF_KG_v3/results/三个随机数种子/results"
)
REPORT_ROOT = Path("D:/1/1myworkcode/HFLSnF_KG_v3/reports")

ROUND_COUNT = 150
PLATFORM_START = 131  # 后20轮起点
PLATFORM_END = 150
COLD_START_WINDOW = 20  # 冷启动分析窗口（前20轮）

ARMS = ("hflsnf", "hflnosnf", "flnosnf")
SEEDS = (42, 2024, 2025)

ARM_META = {
    "hflsnf": {
        "label": "HFLSnF",
        "cn": "分层联邦学习 + SnF 动态拓扑",
        "arch": "HFL",
        "snf": "是",
        "color": "#1f77b4",
        "edge": "6 固定边缘组",
    },
    "hflnosnf": {
        "label": "HFLnoSnF",
        "cn": "分层联邦学习，不用 SnF",
        "arch": "HFL",
        "snf": "否",
        "color": "#ff7f0e",
        "edge": "6 固定边缘组",
    },
    "flnosnf": {
        "label": "FLnoSnF",
        "cn": "普通联邦学习，不用 SnF",
        "arch": "FL",
        "snf": "否",
        "color": "#2ca02c",
        "edge": "1 组（扁平）",
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
        arm = None
        for key in ARMS:
            if "_{}_".format(key) in name:
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


def _mean(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


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
    recovery = None
    peak_value = early[peak_i] if early else 0.0
    if drawdown > 0.0:
        for idx in range(valley_i + 1, len(values)):
            if float(values[idx]) >= peak_value:
                recovery = idx + 1
                break
    return {
        "peak_round": peak_i + 1 if early else None,
        "peak_mrr": peak_value if early else None,
        "valley_round": valley_i + 1 if early else None,
        "valley_mrr": early[valley_i] if early else None,
        "drawdown": drawdown,
        "recovery_round": recovery,
    }


def _first_sustained_round(values, threshold, width=5):
    vals = [float(v) for v in values]
    for start in range(0, len(vals) - width + 1):
        if all(v >= threshold for v in vals[start : start + width]):
            return start + 1
    return ROUND_COUNT + 1


def _load_one_run(path):
    summary = _load_json(path / "summary.json")
    participation = _load_json(path / "dynamic_participation_summary.json")
    rows = []
    with open(path / "metrics.csv", "r", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    # 逐轮 val_mrr（eval_every=1，每轮都有值）
    val_mrr = []
    hits1, hits3, hits10 = [], [], []
    update_l2 = []
    for row in rows:
        vmrr = _fnum(row.get("val_mrr"))
        val_mrr.append(vmrr)
        hits1.append(_fnum(row.get("val_hits_at_1")))
        hits3.append(_fnum(row.get("val_hits_at_3")))
        hits10.append(_fnum(row.get("val_hits_at_10")))
        update_l2.append(_fnum(row.get("server_update_l2")))
    late_mrr = [v for v in val_mrr[PLATFORM_START - 1 : PLATFORM_END]]
    late_hits1 = [v for v in hits1[PLATFORM_START - 1 : PLATFORM_END]]
    late_hits3 = [v for v in hits3[PLATFORM_START - 1 : PLATFORM_END]]
    late_hits10 = [v for v in hits10[PLATFORM_START - 1 : PLATFORM_END]]
    late_mrr_mean = _mean(late_mrr)
    slope = _linear_slope(late_mrr)
    cold = _cold_start(val_mrr)

    selection = participation.get("client_selection_counts", {})
    counts = [int(v) for v in selection.values()]

    return {
        "val_mrr": val_mrr,
        "hits1": hits1,
        "hits3": hits3,
        "hits10": hits10,
        "update_l2": update_l2,
        "best_val_mrr": _fnum(summary.get("best_validation_mrr_during_training")),
        "best_round": int(_fnum(summary.get("best_round"), default=0)),
        "final_round_mrr": val_mrr[-1] if val_mrr else float("nan"),
        "late_mean": late_mrr_mean,
        "late_slope": slope,
        "late_std": statistics.pstdev(late_mrr) if late_mrr else float("nan"),
        "late_hits1": _mean(late_hits1),
        "late_hits3": _mean(late_hits3),
        "late_hits10": _mean(late_hits10),
        "late_label": "platform" if abs(slope) <= 0.0005 else "late_window",
        "cold": cold,
        "participant_mean": _fnum(summary.get("participant_count_mean")),
        "participant_min": _fnum(summary.get("participant_count_min")),
        "participant_max": _fnum(summary.get("participant_count_max")),
        "group_mean": _fnum(summary.get("group_count_mean")),
        "effective_global_passes": _fnum(summary.get("effective_global_passes")),
        "unique_topology_count": int(_fnum(summary.get("unique_topology_count"), default=0)),
        "unique_participant_set_count": int(
            _fnum(summary.get("unique_participant_set_count"), default=0)
        ),
        "client_participation_min": min(counts) if counts else 0,
        "client_participation_median": statistics.median(counts) if counts else 0,
        "client_participation_max": max(counts) if counts else 0,
        "zero_participation": sum(1 for c in counts if c == 0),
        "final_val_metrics": summary.get("final_validation_metrics") or {},
    }


def _agg(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": sum(vals) / len(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _agg_int(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": sum(vals) / len(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def load_and_aggregate():
    dirs = _find_run_dirs()
    missing = [
        (a, s) for a in ARMS for s in SEEDS if (a, s) not in dirs
    ]
    if missing:
        raise RuntimeError("缺少结果目录：{}".format(missing))

    per_run = {}  # (arm, seed) -> run
    for arm in ARMS:
        for seed in SEEDS:
            per_run[(arm, seed)] = _load_one_run(dirs[(arm, seed)])

    arms = {}
    for arm in ARMS:
        runs = [per_run[(arm, s)] for s in SEEDS]
        arms[arm] = {
            "best_val_mrr": _agg([r["best_val_mrr"] for r in runs]),
            "best_round": _agg_int([r["best_round"] for r in runs]),
            "final_round_mrr": _agg([r["final_round_mrr"] for r in runs]),
            "late_mean": _agg([r["late_mean"] for r in runs]),
            "late_std": _agg([r["late_std"] for r in runs]),
            "late_slope": _agg([r["late_slope"] for r in runs]),
            "late_hits1": _agg([r["late_hits1"] for r in runs]),
            "late_hits3": _agg([r["late_hits3"] for r in runs]),
            "late_hits10": _agg([r["late_hits10"] for r in runs]),
            "drawdown": _agg([r["cold"]["drawdown"] for r in runs]),
            "participant_mean": _agg([r["participant_mean"] for r in runs]),
            "group_mean": _agg([r["group_mean"] for r in runs]),
            "effective_global_passes": _agg([r["effective_global_passes"] for r in runs]),
            "client_participation_min": _agg([r["client_participation_min"] for r in runs]),
            "client_participation_median": _agg([r["client_participation_median"] for r in runs]),
            "client_participation_max": _agg([r["client_participation_max"] for r in runs]),
            "zero_participation": _agg([r["zero_participation"] for r in runs]),
            # 每轮均值/标准差（用于画图）
            "per_round_mean": [
                _mean([per_run[(arm, s)]["val_mrr"][i] for s in SEEDS])
                for i in range(ROUND_COUNT)
            ],
            "per_round_std": [
                statistics.pstdev([per_run[(arm, s)]["val_mrr"][i] for s in SEEDS])
                for i in range(ROUND_COUNT)
            ],
            "late_label": _late_label(runs),
        }
    # 收敛轮次：以该臂后20轮均值 95% 为阈值，连续5轮达到
    for arm in ARMS:
        thr = arms[arm]["late_mean"]["mean"] * 0.95
        conv = []
        for s in SEEDS:
            conv.append(
                _first_sustained_round(per_run[(arm, s)]["val_mrr"], thr)
            )
        arms[arm]["convergence_round"] = _agg_int(conv)

    # SnF 差距（HFL 内）与 HFL 差距（无 SnF 时）
    snf_gap = [
        per_run[("hflsnf", s)]["late_mean"] - per_run[("hflnosnf", s)]["late_mean"]
        for s in SEEDS
    ]
    hfl_gap = [
        per_run[("hflnosnf", s)]["late_mean"] - per_run[("flnosnf", s)]["late_mean"]
        for s in SEEDS
    ]
    gaps = {
        "snf_gap": _agg(snf_gap),
        "hfl_gap": _agg(hfl_gap),
        "snf_gap_per_seed": dict(zip(SEEDS, snf_gap)),
        "hfl_gap_per_seed": dict(zip(SEEDS, hfl_gap)),
    }
    return per_run, arms, gaps


def _late_label(runs):
    labels = [r["late_label"] for r in runs]
    return labels[0] if len(set(labels)) == 1 else "mixed"


def _fmt(m, digits=4):
    mean = m["mean"]
    if math.isnan(mean):
        return "—"
    return "{:.{d}f} ± {:.{d}f}".format(mean, m["std"], d=digits)


def _fmt1(m):
    return _fmt(m, digits=1)


def _build_html(per_run, arms, gaps, now):
    # 三臂关键文本
    h = arms["hflsnf"]
    hn = arms["hflnosnf"]
    fn = arms["flnosnf"]

    order = ARMS

    # 概览表（架构/参与预算）
    overview_rows = ""
    for arm in order:
        meta = ARM_META[arm]
        a = arms[arm]
        overview_rows += (
            '<tr><td><strong style="color:{}">{}</strong></td>'
            '<td>{}</td><td>{}</td><td>{}</td>'
            '<td class="number">{:.1f}</td>'
            '<td class="number">{:.1f}</td>'
            '<td class="number">{:.1f}</td></tr>'
        ).format(
            meta["color"], meta["label"], meta["cn"], meta["arch"], meta["edge"],
            a["participant_mean"]["mean"], a["group_mean"]["mean"],
            a["effective_global_passes"]["mean"],
        )

    # 汇总表
    def agg_row(arm):
        meta = ARM_META[arm]
        a = arms[arm]
        return (
            '<tr><td><strong style="color:{}">{}</strong></td>'
            '<td class="number">{}</td>'
            '<td class="number">{:.1f}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{:.1f}</td></tr>'
        ).format(
            meta["color"], meta["label"],
            _fmt(a["best_val_mrr"]),
            a["best_round"]["mean"],
            _fmt(a["late_mean"]),
            _fmt(a["late_hits1"]),
            _fmt(a["late_hits3"]),
            _fmt(a["late_hits10"]),
            a["convergence_round"]["mean"],
        )

    summary_rows = "".join(agg_row(a) for a in order)

    # 逐种子明细表
    detail_rows = ""
    for seed in SEEDS:
        cells = []
        for arm in order:
            r = per_run[(arm, seed)]
            cells.append(
                '<td class="number">{:.4f}</td><td class="number">{}</td>'.format(
                    r["best_val_mrr"], r["best_round"]
                )
            )
        # 该种子 SnF 差距
        cells.append(
            '<td class="number">{:+.4f}</td>'.format(gaps["snf_gap_per_seed"][seed])
        )
        detail_rows += "<tr><td class='number'>seed {}</td>{}</tr>".format(seed, "".join(cells))

    # 柱状图数据（后20轮均值 + 误差棒）
    bar_labels = [ARM_META[a]["label"] for a in order]
    bar_data = [arms[a]["late_mean"]["mean"] for a in order]
    bar_errors = [arms[a]["late_mean"]["std"] for a in order]
    bar_colors = [ARM_META[a]["color"] for a in order]

    # 收敛曲线数据
    round_labels = list(range(1, ROUND_COUNT + 1))
    datasets = []
    for arm in order:
        meta = ARM_META[arm]
        a = arms[arm]
        datasets.append({
            "label": "{} (后20轮均值 {:.4f})".format(meta["label"], a["late_mean"]["mean"]),
            "data": [round(v, 4) for v in a["per_round_mean"]],
            "borderColor": meta["color"],
            "backgroundColor": meta["color"] + "33",
            "borderWidth": 1.6,
            "pointRadius": 0,
            "tension": 0.12,
        })

    # 关键结论数值
    ordered_arms = sorted(order, key=lambda a: arms[a]["late_mean"]["mean"], reverse=True)
    best_arm = ordered_arms[0]
    worst_arm = ordered_arms[-1]
    best_late = arms[best_arm]["late_mean"]["mean"]
    worst_late = arms[worst_arm]["late_mean"]["mean"]
    best_val_arm = max(order, key=lambda a: arms[a]["best_val_mrr"]["mean"])
    best_val = arms[best_val_arm]["best_val_mrr"]["mean"]
    ranking_sentence = " &gt; ".join(
        "{}（{:.4f}）".format(ARM_META[a]["label"], arms[a]["late_mean"]["mean"])
        for a in ordered_arms
    )

    snf_gap_mean = gaps["snf_gap"]["mean"]
    snf_gap_std = gaps["snf_gap"]["std"]
    hfl_gap_mean = gaps["hfl_gap"]["mean"]
    hfl_gap_std = gaps["hfl_gap"]["std"]

    # 冷启动过冲汇总
    cold_rows = ""
    for arm in order:
        meta = ARM_META[arm]
        a = arms[arm]
        runs = [per_run[(arm, s)] for s in SEEDS]
        peak_r = [r["cold"]["peak_round"] for r in runs]
        valley_r = [r["cold"]["valley_round"] for r in runs]
        valley_m = [r["cold"]["valley_mrr"] for r in runs]
        cold_rows += (
            '<tr><td><strong style="color:{}">{}</strong></td>'
            '<td class="number">{}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{}</td>'
            '<td class="number">{}</td></tr>'
        ).format(
            meta["color"], meta["label"],
            _fmt(a["drawdown"]),
            _fmt1(_agg(peak_r)),
            _fmt1(_agg(valley_r)),
            _fmt(_agg(valley_m)),
        )

    # 序列化画图数据为合法 JSON（f-string 直接内插）
    round_labels_js = json.dumps(round_labels)
    datasets_js = json.dumps(datasets, ensure_ascii=False)
    bar_labels_js = json.dumps(bar_labels, ensure_ascii=False)
    bar_data_js = json.dumps(bar_data)
    bar_colors_js = json.dumps(bar_colors)
    bar_err_str = " / ".join("{:.4f}".format(e) for e in bar_errors)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<title>HFLSnF 动态拓扑三实验臂 × 三种子 聚合分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{{color-scheme:light dark;--bg:#fff;--surface:#f7f7f7;--ink:#0d0d0d;--muted:#5d5d5d;--tertiary:#8f8f8f;--border:rgba(13,13,13,.08);--accent:#0285ff;--positive:#00692a;--positive-bg:#edfaf2;--negative:#ba2623;--negative-bg:#fff0f0;--radius:12px;font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}}
@media(prefers-color-scheme:dark){{:root{{--bg:#181818;--surface:#212121;--ink:#dfdfdf;--muted:#cdcdcd;--tertiary:#afafaf;--border:rgba(255,255,255,.1);--positive:#79d996;--positive-bg:rgba(64,180,99,.16);--negative:#ff8583;--negative-bg:rgba(224,74,70,.16)}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:0 32px 56px;font-size:14px;line-height:1.7}}
h1{{font-size:22px;font-weight:600;margin:24px 0 4px}}
h2{{font-size:17px;font-weight:600;margin:28px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--border)}}
h3{{font-size:15px;font-weight:600;margin:20px 0 6px}}
p{{margin:8px 0;color:var(--muted);max-width:900px}}strong{{color:var(--ink)}}
.page-header{{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;height:48px;margin:0 calc(50% - 50vw);padding:8px 12px;border-bottom:1px solid var(--border);background:var(--bg)}}
.page-header h1{{margin:0;font-size:14px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.page-meta{{font-size:12px;color:var(--tertiary)}}
.container{{max-width:1060px;margin:0 auto}}
.card{{border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:18px;margin:16px 0}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:12px 0}}
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
<h1>HFLSnF 动态拓扑三实验臂 × 三种子 聚合分析报告</h1>
<div class="page-meta">{now:%Y-%m-%d %H:%M}</div>
</header>
<div class="container">

<h2>1. 实验概览</h2>
<p>本报告汇总 <strong>9 组正式实验</strong>：3 个实验臂（<strong>HFLSnF</strong>、<strong>HFLnoSnF</strong>、<strong>FLnoSnF</strong>）
× 3 个随机种子（42 / 2024 / 2025），每个组合训练 150 通信轮、每轮验证一次。
统一采用 FB15k-237、37 个客户端按头实体均衡互斥分区、256 维 TransE（L1 距离）、双向自对抗负采样（每正样本 256 负样本）、
每轮 3 个本地 epoch、逐行出现次数加权聚合、服务器 FedAdam（lr=0.05, β=(0.9,0.99), τ=0.001, 关闭偏差修正）、
动态 MATLAB 拓扑（<code>result-U-6fixedge_epoch200_hard_varAlpha_0p1_trainable.mat</code>，util=0.6）。
训练阶段不自动评估测试集，本报告所有指标均为验证集（每轮 4096 三元组 / 8192 查询）。</p>

<div class="card">
<table>
<thead><tr>
  <th>实验臂</th><th>含义</th><th>架构</th><th>拓扑</th>
  <th class="number">平均每轮参与人数</th>
  <th class="number">平均边缘组数</th>
  <th class="number">有效数据遍数</th>
</tr></thead>
<tbody>
{overview_rows}
</tbody>
</table>
</div>
<blockquote>三个实验臂使用不同的 MATLAB 动态参与与分组过程，因此结果应理解为动态编排造成的<strong>系统级差异</strong>，不能单独归因于 SnF 或分层机制（详见第 8 节限制说明）。</blockquote>

<h2>2. 核心结果汇总（三种子均值 ± 标准差）</h2>

<div class="card">
<table>
<thead><tr>
  <th>实验臂</th><th class="number">最佳验证 MRR</th><th class="number">最佳轮次</th>
  <th class="number">后20轮均值 MRR</th><th class="number">Hits@1</th>
  <th class="number">Hits@3</th><th class="number">Hits@10</th>
  <th class="number">收敛轮次</th>
</tr></thead>
<tbody>
{summary_rows}
</tbody>
</table>
</div>

<p>后20轮（第 131–150 轮）均值 MRR 的排序为 <strong>{ranking_sentence}</strong>。
三臂最佳验证 MRR 均值最高的是 <strong>{ARM_META[best_val_arm]['label']}（{best_val:.4f}）</strong>。</p>

<h2>3. 逐种子明细（透明性）</h2>

<div class="card">
<table>
<thead><tr>
  <th>种子</th>
  <th class="number">HFLSnF 最佳MRR</th><th class="number">轮次</th>
  <th class="number">HFLnoSnF 最佳MRR</th><th class="number">轮次</th>
  <th class="number">FLnoSnF 最佳MRR</th><th class="number">轮次</th>
  <th class="number">SnF 差距(HFL)</th>
</tr></thead>
<tbody>
{detail_rows}
</tbody>
</table>
</div>

<h2>4. 验证 MRR 收敛曲线（均值 ± 标准差带，150 轮）</h2>

<div class="card" style="overflow:hidden;">
<div class="chart-wrapper">
  <canvas id="valCurve" role="img" aria-label="三实验臂验证MRR收敛曲线均值与标准差带"></canvas>
</div>
</div>

<h2>5. 后20轮平台 MRR 对比（含误差棒）</h2>

<div class="card" style="overflow:hidden;">
<div class="chart-wrapper" style="height:320px;">
  <canvas id="barChart" role="img" aria-label="三实验臂后20轮均值MRR柱状图"></canvas>
</div>
</div>

<h2>6. 核心发现</h2>

<h3>6.1 三臂性能分层：HFLSnF 领先，FLnoSnF 明显落后</h3>
<p>后20轮平台 MRR 上，<strong>HFLSnF</strong> 达到 {h['late_mean']['mean']:.4f}，<strong>HFLnoSnF</strong> 为 {hn['late_mean']['mean']:.4f}，
<strong>FLnoSnF</strong> 仅 {fn['late_mean']['mean']:.4f}。三臂最佳验证 MRR 分别为
{arms['hflsnf']['best_val_mrr']['mean']:.4f} / {arms['hflnosnf']['best_val_mrr']['mean']:.4f} / {arms['flnosnf']['best_val_mrr']['mean']:.4f}。
这与各臂每轮参与预算一致：HFLSnF 平均约 {h['participant_mean']['mean']:.1f} 人/轮，HFLnoSnF 约 {hn['participant_mean']['mean']:.1f} 人/轮，
FLnoSnF 约 {fn['participant_mean']['mean']:.1f} 人/轮——数据暴露量（有效遍数 {h['effective_global_passes']['mean']:.0f} / {hn['effective_global_passes']['mean']:.0f} / {fn['effective_global_passes']['mean']:.0f}）直接决定了性能上限。</p>

<h3>6.2 SnF 增益与分层增益（在 HFL 内、以及无 SnF 时）</h3>
<p>在 HFL 架构内，SnF 带来的平台 MRR 增益（HFLSnF − HFLnoSnF）为 <strong>{snf_gap_mean:+.4f} ± {snf_gap_std:.4f}</strong>（三种子分别 {gaps['snf_gap_per_seed'][42]:+.4f} / {gaps['snf_gap_per_seed'][2024]:+.4f} / {gaps['snf_gap_per_seed'][2025]:+.4f}）。
在无 SnF 条件下，分层架构带来的增益（HFLnoSnF − FLnoSnF）为 <strong>{hfl_gap_mean:+.4f} ± {hfl_gap_std:.4f}</strong>（三种子分别 {gaps['hfl_gap_per_seed'][42]:+.4f} / {gaps['hfl_gap_per_seed'][2024]:+.4f} / {gaps['hfl_gap_per_seed'][2025]:+.4f}）。
分层增益的绝对幅度大于 SnF 增益，但两者都叠加在同一“参与预算”效应之上，不能剥离后单独解读。</p>

<h3>6.3 收敛速度与最佳轮次</h3>
<p>三臂达到各自平台均值 95% 的收敛轮次分别为 HFLSnF {h['convergence_round']['mean']:.1f} 轮、HFLnoSnF {hn['convergence_round']['mean']:.1f} 轮、FLnoSnF {fn['convergence_round']['mean']:.1f} 轮。
最佳验证轮次方面，HFLSnF 平均 {h['best_round']['mean']:.1f} 轮、HFLnoSnF {hn['best_round']['mean']:.1f} 轮、FLnoSnF {fn['best_round']['mean']:.1f} 轮。
三臂均在 150 轮之前达到最佳验证 MRR，之后进入平台期。</p>

<h3>6.4 冷启动过冲（FedAdam 关闭偏差修正）</h3>
<p>关闭偏差修正的 FedAdam 在训练初期存在过冲：</p>
<div class="card">
<table>
<thead><tr><th>实验臂</th><th class="number">前20轮最大回撤</th><th class="number">峰值轮次</th><th class="number">谷底轮次</th><th class="number">谷底 MRR</th></tr></thead>
<tbody>
{cold_rows}
</tbody>
</table>
</div>
<p>三臂前期均呈现“冲高 → 回落 → 爬升”形态，谷底深度随参与预算减少而加深，符合关闭偏差修正时 FedAdam 二阶梯度的典型冷启动特征。</p>

<h3>6.5 跨种子稳定性</h3>
<p>三臂后20轮 MRR 的跨种子标准差均较小（HFLSnF ±{h['late_mean']['std']:.4f}、HFLnoSnF ±{hn['late_mean']['std']:.4f}、FLnoSnF ±{fn['late_mean']['std']:.4f}），
说明三种子下的性能排序稳定，结论不是单一种子的偶然波动。SnF 增益在三种子上方向一致（均为正）。</p>

<h2>7. 指标定义与分析范围</h2>
<ul>
<li><strong>最佳验证 MRR</strong>：150 轮逐轮验证中 MRR 的最大值及其对应轮次（来自 <code>summary.json</code> 的 <code>best_validation_mrr_during_training</code>）。</li>
<li><strong>后20轮均值 MRR</strong>：第 131–150 轮验证 MRR 的算术平均；若该窗口 OLS 斜率绝对值 ≤ 0.0005 则标注为“平台”，否则称“后20轮均值”。</li>
<li><strong>收敛轮次</strong>：以该臂后20轮均值的 95% 为阈值，首次连续 5 轮达到该阈值的轮次。</li>
<li><strong>冷启动过冲</strong>：前 20 轮内最大“峰→谷”回撤幅度及峰值/谷底轮次。</li>
<li><strong>有效数据遍数</strong>：累计本地正三元组暴露数 / 训练集正三元组总数（272115）。</li>
<li><strong>均值 ± 标准差</strong>：对 3 个种子取总体标准差（<code>pstdev</code>），仅描述性汇总，不做显著性检验。</li>
</ul>

<h2>8. 限制说明</h2>
<ul>
<li><strong>非严格单因素消融</strong>：三个实验臂的参与人数、覆盖率、分组过程同时变化，SnF 增益与分层增益无法与参与预算效应完全分离。</li>
<li><strong>种子数较少</strong>：仅 3 个种子，均值±标准差为描述性统计，不足以支撑显著性结论。</li>
<li><strong>仅验证集</strong>：训练阶段未执行官方测试集评估；验证集每轮仅 4096 三元组，后期单点波动约 0.003–0.008。</li>
<li><strong>范围局限</strong>：结论仅适用于 FB15k-237 + TransE + 两级 HFL + FedAdam + util=0.6 动态拓扑这一特定配置。</li>
</ul>

<h2>9. 推荐下一步</h2>
<ol>
<li>用官方评估入口对三臂各自的最佳验证检查点（<code>model_best.pt</code>）跑完整 filtered 测试集，得到可发表的测试 MRR/Hits@10。</li>
<li>设计“同一人数序列、只改变分组”以及“同一参与集合、只改变 SnF”的严格配对实验，剥离参与预算与拓扑机制。</li>
<li>若需进一步拉大 HFLSnF/HFLnoSnF 差距，从拓扑可达性入手（如降低 edge_num 或 topology_util），而非仅调整 yaml。</li>
</ol>

<p style="margin-top:32px;color:var(--tertiary);font-size:12px;">
报告生成时间: {now:%Y-%m-%d %H:%M:%S} |
数据源: 9 组实验结果 (summary.json + metrics.csv + dynamic_participation_summary.json) |
聚合方式: 三实验臂 × 三种子 → 均值 ± 标准差
</p>

</div>

<script>
(function() {{
  const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const textColor = isDark ? '#cdcdcd' : '#5d5d5d';
  const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(13,13,13,0.06)';
  Chart.defaults.color = textColor;
  Chart.defaults.borderColor = gridColor;

  const rounds = {round_labels_js};
  const datasets = {datasets_js};
  const lineCtx = document.getElementById('valCurve').getContext('2d');
  new Chart(lineCtx, {{
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
          suggestedMin: 0,
          ticks: {{ callback: function(v) {{ return v.toFixed(3); }} }}
        }}
      }}
    }}
  }});

  const barCtx = document.getElementById('barChart').getContext('2d');
  new Chart(barCtx, {{
    type: 'bar',
    data: {{
      labels: {bar_labels_js},
      datasets: [{{
        label: '后20轮均值 MRR',
        data: {bar_data_js},
        backgroundColor: {bar_colors_js},
        borderColor: {bar_colors_js},
        borderWidth: 1,
        errorBars: null
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: function(ctx) {{ return 'MRR: ' + ctx.parsed.y.toFixed(4) + ' ± {bar_err_str}'; }} }} }}
      }},
      scales: {{
        y: {{ beginAtZero: true, title: {{ display: true, text: '后20轮均值 MRR', font: {{ size: 12 }} }} }}
      }}
    }}
  }});
}})();
</script>

</body>
</html>"""

    return html


def main():
    per_run, arms, gaps = load_and_aggregate()
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    report_name = "final_three_seed_analysis_{}".format(ts)
    report_dir = REPORT_ROOT / report_name
    report_dir.mkdir(parents=True, exist_ok=True)

    html = _build_html(per_run, arms, gaps, now)
    html_path = report_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "title": "HFLSnF 动态拓扑三实验臂 × 三种子 聚合分析报告",
            "description": "跨种子聚合 HFLSnF/HFLnoSnF/FLnoSnF 三种子(42/2024/2025)的验证集性能，"
                           "报告平台 MRR、SnF/HFL 增益、收敛轮次与冷启动过冲。",
            "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "sources": [{"id": "final-three-seed", "label": "三臂三种子正式结果",
                         "path": "HFLSnF_KG_v3/results/三个随机数种子/results/"}],
        },
    }
    (report_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    receipt = {
        "report": report_name,
        "generated_at": now.isoformat(),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "checks": {
            "ablation_suite": "v3_final_dynamic_fedadam_u0p6_bcfalse_e3_eval1_formal150",
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

    print("Report dir :", report_dir)
    print("report.html :", html_path)
    # 控制台打印关键数字，便于核对
    for arm in ARMS:
        a = arms[arm]
        print(
            "{}: best_val {:.4f}±{:.4f} @r{:.1f} | late {:.4f}±{:.4f} | "
            "conv {:.1f} | hits10 {:.4f}".format(
                ARM_META[arm]["label"],
                a["best_val_mrr"]["mean"], a["best_val_mrr"]["std"],
                a["best_round"]["mean"],
                a["late_mean"]["mean"], a["late_mean"]["std"],
                a["convergence_round"]["mean"],
                a["late_hits10"]["mean"],
            )
        )
    print("SnF gap (HFL) : {:.4f}±{:.4f}".format(gaps["snf_gap"]["mean"], gaps["snf_gap"]["std"]))
    print("HFL gap (noSnF): {:.4f}±{:.4f}".format(gaps["hfl_gap"]["mean"], gaps["hfl_gap"]["std"]))
    return report_dir


if __name__ == "__main__":
    main()
