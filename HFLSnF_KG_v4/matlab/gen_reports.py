"""
Generate SIMPLE HTML data analysis reports for .mat result files.
Format reference: 4 summary cards + 1 statistics table per (scheme, U-value) combo.
Covers all 6 experiment schemes, U=0.4, 0.5, 0.6, 0.7.
"""

import scipy.io
import numpy as np
import json
import os

MAT_DIR = r"D:\1\1myworkcode\HFLSnF_KG_v3\matlab"

FILES = [
    {
        "name": "result-U-6fixedge_epoch200.mat",
        "path": os.path.join(MAT_DIR, "result-U-6fixedge_epoch200.mat"),
        "output": os.path.join(MAT_DIR, "result-U-6fixedge_epoch200_U04-07_数据报告.html"),
        "title": "result-U-6fixedge_epoch200.mat — U=0.4~0.7 全方案数据分析报告",
    },
    {
        "name": "result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat",
        "path": os.path.join(MAT_DIR, "result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat"),
        "output": os.path.join(MAT_DIR, "result-U-6fixedge_epoch200_varAlpha_0p1_trainable_U04-07_数据报告.html"),
        "title": "result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat — U=0.4~0.7 全方案数据分析报告",
    },
]

U_VALUES = [0.4, 0.5, 0.6, 0.7]
U_INDICES = [3, 4, 5, 6]

# 6 schemes: (display name, prefix for fields, has_group_num)
SCHEMES = [
    ("HFLSnF_fix",   "HFLSnF_fix",   True),
    ("HFLSnF_los",   "HFLSnF_los",   True),
    ("HFLnoSnF_fix", "HFLnoSnF_fix", True),
    ("HFLnoSnF_los", "HFLnoSnF_los", True),
    ("FLSnF",        "FLSnF",        False),
    ("FLnoSnF",      "FLnoSnF",      False),
]

STAT_NAMES = {
    "group_num": "活跃边缘组数",
    "client_num": "参与客户端数",
    "max_layer": "最大聚合层数",
    "time_agg": "聚合时间(time_agg)",
}

SCHEME_DESCRIPTIONS = {
    "HFLSnF_fix":   "HFL + SnF启用 + 固定边缘",
    "HFLSnF_los":   "HFL + SnF启用 + LoS边缘",
    "HFLnoSnF_fix": "HFL + SnF关闭 + 固定边缘",
    "HFLnoSnF_los": "HFL + SnF关闭 + LoS边缘",
    "FLSnF":        "FL + SnF启用",
    "FLnoSnF":      "FL + SnF关闭",
}


def compute_stats(arr):
    """Compute statistics for a 1D array."""
    arr = arr.astype(float)
    n = len(arr)
    valid = int(np.sum(~np.isnan(arr)))
    mean = float(np.mean(arr))
    var = float(np.var(arr, ddof=0))
    std = float(np.std(arr, ddof=0))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    med = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    rng = mx - mn
    return {
        "n": n, "valid": valid, "mean": mean, "var": var, "std": std,
        "min": mn, "max": mx, "median": med, "q25": q25, "q75": q75, "range": rng,
    }


def stats_card_and_table(stats_dict, has_group_num, scheme_name, u_val, epoch_num):
    """Generate 4 cards + 1 stats table for a scheme."""
    parts = []
    # Experiment description
    desc = SCHEME_DESCRIPTIONS[scheme_name]
    parts.append(f'<p style="font-size:13px;color:var(--t2);margin-bottom:10px;"><strong>{scheme_name}</strong> &mdash; {desc} + U={u_val}，共{epoch_num}轮。</p>')

    # Card grid
    parts.append('<div class="card-grid">')
    if has_group_num:
        metrics = ["group_num", "client_num", "max_layer", "time_agg"]
    else:
        metrics = ["client_num", "max_layer", "time_agg"]
    for m_name in metrics:
        v = stats_dict[m_name]
        parts.append(f'''  <div class="card">
    <div class="card-label">{STAT_NAMES[m_name]}<br>M &plusmn; &Sigma;</div>
    <div class="card-value">{v["mean"]:.2f} <span style="font-size:14px;font-weight:400;">&plusmn;{v["std"]:.4f}</span></div>
    <div class="card-detail">方差={v["var"]:.4f} | 极差={v["range"]:.1f} | 中位数={v["median"]:.1f} | Q25={v["q25"]:.1f} | Q75={v["q75"]:.1f}</div>
  </div>''')
    parts.append('</div>')

    # Stats table
    parts.append(f'<h3 style="font-size:14px;margin:14px 0 8px;">&#x1f4ca; 完整统计量表 (N={epoch_num})</h3>')
    parts.append('<div class="table-wrap"><table>')
    parts.append('<thead><tr><th>指标</th><th>样本量</th><th>均值 &mu;</th><th>方差 &sigma;&sup2;</th><th>标准差 &sigma;</th><th>最小值</th><th>Q25</th><th>中位数</th><th>Q75</th><th>最大值</th><th>极差</th><th>变异系数 CV</th></tr></thead>')
    parts.append('<tbody>')
    for m_name in metrics:
        v = stats_dict[m_name]
        cv = v["std"] / v["mean"] if v["mean"] != 0 else 0
        parts.append(f'''  <tr>
    <td><strong>{STAT_NAMES[m_name]}</strong></td>
    <td class="sc">{v["valid"]}</td>
    <td class="sc">{v["mean"]:.4f}</td>
    <td class="sc">{v["var"]:.4f}</td>
    <td class="sc">{v["std"]:.4f}</td>
    <td class="sc">{v["min"]:.1f}</td>
    <td class="sc">{v["q25"]:.1f}</td>
    <td class="sc">{v["median"]:.1f}</td>
    <td class="sc">{v["q75"]:.1f}</td>
    <td class="sc">{v["max"]:.1f}</td>
    <td class="sc">{v["range"]:.1f}</td>
    <td class="sc">{cv:.4f}</td>
  </tr>''')
    parts.append('</tbody></table></div>')
    return "\n".join(parts)


def generate_report_html(file_info):
    """Generate the simple HTML report."""
    mat = scipy.io.loadmat(file_info["path"])

    created_at = str(mat["created_at"][0])
    schema_ver = str(mat["schema_version"][0])
    base_seed = int(mat["base_seed"][0, 0])
    topo = str(mat["TopoOption"][0])
    epoch_num = int(mat["epoch_num"][0, 0])
    num_nodes = int(mat["num_of_nodes"][0, 0])
    edge_set = mat["EdgeSet"].flatten().tolist()

    # Pre-compute stats for all scheme×U combinations
    all_stats = {}  # scheme_name -> {u_val: {metric: stats}}
    for scheme_disp, prefix, has_gn in SCHEMES:
        all_stats[scheme_disp] = {}
        for u_val, u_idx in zip(U_VALUES, U_INDICES):
            stats_dict = {}
            if has_gn:
                stats_dict["group_num"] = compute_stats(mat[f"group_num_{prefix}"][:, u_idx])
            stats_dict["client_num"] = compute_stats(mat[f"client_num_{prefix}"][:, u_idx])
            stats_dict["max_layer"] = compute_stats(mat[f"max_layer_{prefix}"][:, u_idx])
            stats_dict["time_agg"] = compute_stats(mat[f"time_agg_{prefix}"][:, u_idx])
            all_stats[scheme_disp][u_val] = stats_dict

    # ============================================================
    # Build HTML
    # ============================================================
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{file_info["title"]}</title>
<style>
:root{{--bg:#fff;--bg-card:#f8f9fb;--bg-header:#1a2332;--text:#1a1a2e;--t2:#475569;--tm:#94a3b8;--border:#e2e8f0;--accent:#2f6690;--accent2:#d68c1a;--shadow:0 1px 3px rgba(0,0,0,.08)}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text);line-height:1.7}}
.header{{background:var(--bg-header);color:#fff;padding:32px 40px 24px}}
.header h1{{font-size:24px;font-weight:700;margin-bottom:6px}}
.header .subtitle{{font-size:13px;color:#94a3b8}}
.header .meta-row{{display:flex;gap:24px;margin-top:12px;flex-wrap:wrap}}
.header .meta-item{{font-size:12px;color:#cbd5e1}}
.header .meta-item .label{{color:#64748b}}
.header .meta-item .value{{font-weight:600;color:#e2e8f0}}
.container{{max-width:1340px;margin:0 auto;padding:24px 40px}}
.section{{margin-bottom:36px}}
.section-title{{font-size:18px;font-weight:700;color:var(--text);border-bottom:3px solid var(--accent);padding-bottom:6px;margin-bottom:16px;display:flex;align-items:center;gap:8px}}
.u-tag{{display:inline-block;padding:2px 9px;border-radius:4px;font-size:12px;font-weight:700;margin-left:6px}}
.u04{{background:#dbeafe;color:#1e40af}}
.u05{{background:#dcfce7;color:#166534}}
.u06{{background:#fef9c3;color:#854d0e}}
.u07{{background:#fce7f3;color:#9d174d}}
.scheme-block{{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:14px 18px;margin-bottom:14px;box-shadow:var(--shadow)}}
.scheme-title{{font-size:14px;font-weight:700;color:var(--accent);margin-bottom:8px;display:flex;align-items:center;gap:8px}}
.card-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin-bottom:10px}}
.card{{background:#fff;border:1px solid var(--border);border-radius:8px;padding:14px 16px}}
.card .card-label{{font-size:11px;color:var(--tm);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
.card .card-value{{font-size:20px;font-weight:700;color:var(--accent)}}
.card .card-detail{{font-size:11px;color:var(--t2);margin-top:3px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:8px;margin-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead{{background:var(--bg-header);color:#fff}}
th{{padding:8px 10px;text-align:left;font-weight:600;font-size:11px;letter-spacing:.3px;white-space:nowrap}}
td{{padding:6px 10px;border-bottom:1px solid var(--border)}}
tbody tr:hover{{background:#f1f5f9}}
.sc{{text-align:center;font-variant-numeric:tabular-nums}}
.footer{{background:var(--bg-card);border-top:1px solid var(--border);padding:14px 40px;font-size:11px;color:var(--tm);text-align:center;margin-top:30px}}
@media(max-width:768px){{.container{{padding:16px}}.card-grid{{grid-template-columns:1fr 1fr}}.header{{padding:20px 16px}}}}
</style>
</head>
<body>

<div class="header">
  <h1>&#x1f4ca; {file_info["title"]}</h1>
  <p class="subtitle">基于 MATLAB 拓扑优化的层次联邦学习 (HFL) — 6种实验方案 × U=0.4~0.7，200轮完整统计</p>
  <div class="meta-row">
    <div class="meta-item"><span class="label">文件：</span><span class="value">{file_info["name"]}</span></div>
    <div class="meta-item"><span class="label">schema：</span><span class="value">v{schema_ver}</span></div>
    <div class="meta-item"><span class="label">创建时间：</span><span class="value">{created_at}</span></div>
    <div class="meta-item"><span class="label">种子：</span><span class="value">{base_seed}</span></div>
    <div class="meta-item"><span class="label">拓扑：</span><span class="value">{topo}</span></div>
    <div class="meta-item"><span class="label">节点：</span><span class="value">{num_nodes}</span></div>
    <div class="meta-item"><span class="label">轮次：</span><span class="value">{epoch_num}</span></div>
    <div class="meta-item"><span class="label">固定边缘：</span><span class="value">{len(edge_set)} 个</span></div>
  </div>
</div>

<div class="container">
'''

    # For each U value, add a section with all 6 schemes
    u_tags = {0.4: "u04", 0.5: "u05", 0.6: "u06", 0.7: "u07"}
    u_section_num = ["二", "三", "四", "五"]

    for idx, u_val in enumerate(U_VALUES):
        html += f'''
<div class="section">
  <div class="section-title">&#x1f4cb; {u_section_num[idx]}、U={u_val} <span class="u-tag {u_tags[u_val]}">U={u_val}</span> &mdash; 6种实验方案完整统计</div>
'''
        for scheme_disp, prefix, has_gn in SCHEMES:
            html += f'''
  <div class="scheme-block">
    <div class="scheme-title">&#x1f50d; {scheme_disp}</div>
'''
            html += stats_card_and_table(all_stats[scheme_disp][u_val], has_gn, scheme_disp, u_val, epoch_num)
            html += '  </div>\n'

        html += '</div>\n'

    html += f'''
</div>
<div class="footer">数据报告自动生成 | 文件: {file_info["name"]} | 分析范围: U=0.4~0.7 × 6种方案 | 分析日期: 2026-08-04</div>

</body>
</html>'''

    return html


if __name__ == "__main__":
    for file_info in FILES:
        print(f"Generating report for {file_info['name']}...")
        html = generate_report_html(file_info)
        with open(file_info["output"], "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  -> {file_info['output']}")
        print(f"  Size: {len(html):,} chars")
    print("\nDone!")