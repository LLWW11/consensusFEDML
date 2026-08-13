"""生成FedAdam阶段二筛选、复验和人数对照分析产物。"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ..tasks.kge.fedadam_stage2 import (
    BASELINE_SETTING_KEY,
    PLATFORM_END_ROUND,
    PLATFORM_START_ROUND,
    ROUND_COUNT,
    Stage2Scenario,
    scenario_from_manifest_entry,
    summarize_result,
)
from ..tasks.kge.fixed_count_four_scenarios import write_json_report


def _full_summary(entry: Mapping[str, object]) -> Dict[str, object]:
    """从一个通过项目的结果目录重新提取含逐轮数据的摘要。"""

    if entry.get("status") != "passed" or not entry.get("result_dir"):
        raise ValueError(
            "报告只能读取已通过项目：{}".format(entry.get("scenario_id"))
        )
    scenario = scenario_from_manifest_entry(entry)
    return summarize_result(Path(str(entry["result_dir"])), scenario)


def _mean(values: Iterable[float]) -> float:
    """返回报告内非空数值序列的算术平均值。"""

    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("报告不能计算空序列均值")
    return sum(materialized) / len(materialized)


def _pair_metrics(
    summaries: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """计算同seed同设置两实验臂的后20轮MRR差距。"""

    by_arm = {str(item["arm"]): item for item in summaries}
    if set(by_arm) != {"hflsnf", "hflnosnf"}:
        raise ValueError("配对分析必须同时包含HFLSnF与HFLnoSnF")
    hflsnf = float(by_arm["hflsnf"]["platform"]["mrr_mean"])
    hflnosnf = float(by_arm["hflnosnf"]["platform"]["mrr_mean"])
    return {
        "hflsnf_late_window_mrr": hflsnf,
        "hflnosnf_late_window_mrr": hflnosnf,
        "late_window_mrr_gap": hflsnf - hflnosnf,
    }


def _screen_report(payload: Mapping[str, object]) -> Dict[str, object]:
    """构造八组全因子筛选的结构化报告。"""

    entries = payload["phases"]["screen"]["entries"]
    summaries = [_full_summary(entry) for entry in entries]
    compact = [
        {key: value for key, value in item.items() if key != "rounds"}
        for item in summaries
    ]
    return {
        "title": "FedAdam阶段二全因子筛选分析",
        "selection": payload["selection"],
        "runs": compact,
        "metric_definition": {
            "late_window": [PLATFORM_START_ROUND, PLATFORM_END_ROUND],
            "convergence": "两臂较低后20轮均值的95%，连续5轮达到",
            "test_usage": "批量训练不读取测试集，选型后再对最佳验证检查点做官方测试",
        },
    }


def _confirmation_report(payload: Mapping[str, object]) -> Dict[str, object]:
    """汇总基线与候选在三个配对随机种子上的复验结果。"""

    selection = payload["selection"]
    selected_key = str(selection["selected_setting"])
    relevant = _confirmation_full_summaries(payload)
    grouped: Dict[Tuple[str, int], List[Mapping[str, object]]] = {}
    for item in relevant:
        grouped.setdefault(
            (str(item["setting"]), int(item["seed"])), []
        ).append(item)
    settings: Dict[str, object] = {}
    for setting_key in (BASELINE_SETTING_KEY, selected_key):
        per_seed: List[Dict[str, object]] = []
        for seed in (42, 2024, 2025):
            metrics = _pair_metrics(grouped[(setting_key, seed)])
            per_seed.append({"seed": seed, **metrics})
        gaps = [float(item["late_window_mrr_gap"]) for item in per_seed]
        settings[setting_key] = {
            "per_seed": per_seed,
            "gap_mean": _mean(gaps),
            "gap_std": statistics.pstdev(gaps),
            "gap_min": min(gaps),
            "gap_max": max(gaps),
        }
    baseline_mean = float(settings[BASELINE_SETTING_KEY]["gap_mean"])
    selected_mean = float(settings[selected_key]["gap_mean"])
    return {
        "title": "FedAdam阶段二三随机种子复验",
        "selection_label": selection["selection_label"],
        "baseline_setting": BASELINE_SETTING_KEY,
        "selected_setting": selected_key,
        "settings": settings,
        "paired_gap_mean_improvement": selected_mean - baseline_mean,
        "statistical_note": "仅报告三个种子的均值、标准差和范围，不做显著性结论。",
    }


def _confirmation_full_summaries(
    payload: Mapping[str, object],
) -> List[Mapping[str, object]]:
    """读取基线和候选在筛选与复验阶段的全部逐轮摘要。"""

    selected_key = str(payload["selection"]["selected_setting"])
    relevant: List[Mapping[str, object]] = []
    for phase_name in ("screen", "confirm"):
        for entry in payload["phases"][phase_name]["entries"]:
            if (
                entry.get("status") == "passed"
                and entry.get("setting")
                in {BASELINE_SETTING_KEY, selected_key}
            ):
                relevant.append(_full_summary(entry))
    return relevant


def _controls_report(payload: Mapping[str, object]) -> Dict[str, object]:
    """汇总两种固定参与人数在三个随机种子上的预算对照。"""

    entries = payload["phases"]["controls"]["entries"]
    summaries = [_full_summary(entry) for entry in entries]
    grouped: Dict[int, List[Mapping[str, object]]] = {}
    for item in summaries:
        grouped.setdefault(int(item["participant_count"]), []).append(item)
    counts: Dict[str, object] = {}
    for participant_count, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(item["seed"]))
        values = [float(item["platform"]["mrr_mean"]) for item in ordered]
        counts[str(participant_count)] = {
            "per_seed": [
                {
                    "seed": int(item["seed"]),
                    "late_window_mrr": float(item["platform"]["mrr_mean"]),
                    "client_participation_min": item["client_participation_min"],
                    "client_participation_median": item["client_participation_median"],
                    "client_participation_max": item["client_participation_max"],
                }
                for item in ordered
            ],
            "late_window_mrr_mean": _mean(values),
            "late_window_mrr_std": statistics.pstdev(values),
        }
    return {
        "title": "FedAdam阶段二固定参与人数预算对照",
        "selected_setting": payload["selection"]["selected_setting"],
        "counts": counts,
        "interpretation_limit": (
            "该对照估计参与预算影响，不完全隔离SnF机制。"
        ),
    }


def _plot_screen(
    batch_dir: Path,
    payload: Mapping[str, object],
) -> Path:
    """绘制四个全因子组合的全程、冷启动和后20轮曲线。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = [
        _full_summary(entry)
        for entry in payload["phases"]["screen"]["entries"]
    ]
    grouped: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for item in summaries:
        grouped.setdefault(str(item["setting"]), {})[str(item["arm"])] = item
    setting_order = (
        "u0p5_bctrue",
        "u0p5_bcfalse",
        "u0p6_bctrue",
        "u0p6_bcfalse",
    )
    figure, axes = plt.subplots(4, 3, figsize=(15, 15), constrained_layout=True)
    windows = ((1, ROUND_COUNT), (1, 25), (PLATFORM_START_ROUND, PLATFORM_END_ROUND))
    colors = {"hflsnf": "#1f77b4", "hflnosnf": "#d62728"}
    for row_index, setting_key in enumerate(setting_order):
        for column_index, (start, end) in enumerate(windows):
            axis = axes[row_index][column_index]
            for arm in ("hflsnf", "hflnosnf"):
                values = grouped[setting_key][arm]["rounds"]
                x = [int(item["round"]) for item in values]
                y = [float(item["val_mrr"]) for item in values]
                axis.plot(
                    x,
                    y,
                    label=arm,
                    color=colors[arm],
                    marker="o",
                    markevery=5,
                    markersize=2.5,
                    linewidth=1.2,
                )
            axis.set_xlim(start, end)
            axis.set_xlabel("Round")
            axis.set_ylabel("Validation MRR")
            axis.grid(alpha=0.25)
            if column_index == 0:
                axis.set_title("{}: all rounds".format(setting_key))
            elif column_index == 1:
                axis.set_title("{}: cold start".format(setting_key))
            else:
                axis.set_title("{}: late window".format(setting_key))
            if row_index == 0 and column_index == 0:
                axis.legend()
    path = Path(batch_dir) / "stage2_screen_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_updates(
    batch_dir: Path,
    payload: Mapping[str, object],
) -> Path:
    """绘制八组实验前25轮FedAdam服务器更新范数。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = [
        _full_summary(entry)
        for entry in payload["phases"]["screen"]["entries"]
    ]
    grouped: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for item in summaries:
        grouped.setdefault(str(item["setting"]), {})[str(item["arm"])] = item
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, setting_key in zip(
        axes.flat,
        ("u0p5_bctrue", "u0p5_bcfalse", "u0p6_bctrue", "u0p6_bcfalse"),
    ):
        for arm, color in (("hflsnf", "#1f77b4"), ("hflnosnf", "#d62728")):
            rows = grouped[setting_key][arm]["rounds"][:25]
            axis.plot(
                [item["round"] for item in rows],
                [item["server_update_l2"] for item in rows],
                label=arm,
                color=color,
                marker="o",
                markevery=5,
                markersize=3,
            )
        axis.set_title(setting_key)
        axis.set_xlabel("Round")
        axis.set_ylabel("Server update L2")
        axis.grid(alpha=0.25)
        axis.legend()
    path = Path(batch_dir) / "stage2_screen_server_updates.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_confirmation(
    batch_dir: Path,
    payload: Mapping[str, object],
) -> Path:
    """绘制基线与候选三随机种子的均值、标准差和三个时间窗口。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_key = str(payload["selection"]["selected_setting"])
    summaries = _confirmation_full_summaries(payload)
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for item in summaries:
        grouped.setdefault(
            (str(item["setting"]), str(item["arm"])), []
        ).append(item)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    settings = (BASELINE_SETTING_KEY, selected_key)
    windows = (
        (1, ROUND_COUNT),
        (1, 25),
        (PLATFORM_START_ROUND, PLATFORM_END_ROUND),
    )
    colors = {"hflsnf": "#1f77b4", "hflnosnf": "#d62728"}
    for row_index, setting_key in enumerate(settings):
        for column_index, (start, end) in enumerate(windows):
            axis = axes[row_index][column_index]
            for arm in ("hflsnf", "hflnosnf"):
                runs = grouped[(setting_key, arm)]
                x = [int(item["round"]) for item in runs[0]["rounds"]]
                per_round = [
                    [float(run["rounds"][index]["val_mrr"]) for run in runs]
                    for index in range(ROUND_COUNT)
                ]
                means = [_mean(values) for values in per_round]
                deviations = [statistics.pstdev(values) for values in per_round]
                lower = [mean - deviation for mean, deviation in zip(means, deviations)]
                upper = [mean + deviation for mean, deviation in zip(means, deviations)]
                axis.plot(
                    x,
                    means,
                    label=arm,
                    color=colors[arm],
                    marker="o",
                    markevery=5,
                    markersize=2.5,
                    linewidth=1.2,
                )
                axis.fill_between(
                    x,
                    lower,
                    upper,
                    color=colors[arm],
                    alpha=0.15,
                )
            axis.set_xlim(start, end)
            axis.set_xlabel("Round")
            axis.set_ylabel("Validation MRR")
            axis.grid(alpha=0.25)
            axis.set_title("{}: {}-{}".format(setting_key, start, end))
            if row_index == 0 and column_index == 0:
                axis.legend()
    path = Path(batch_dir) / "stage2_confirmation_curves.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_controls(
    batch_dir: Path,
    payload: Mapping[str, object],
) -> Path:
    """绘制固定参与人数与后20轮验证MRR的三种子关系。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = [
        _full_summary(entry)
        for entry in payload["phases"]["controls"]["entries"]
    ]
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for seed in (42, 2024, 2025):
        seed_items = sorted(
            (item for item in summaries if int(item["seed"]) == seed),
            key=lambda item: int(item["participant_count"]),
        )
        axis.plot(
            [int(item["participant_count"]) for item in seed_items],
            [float(item["platform"]["mrr_mean"]) for item in seed_items],
            marker="o",
            label="seed={}".format(seed),
        )
    axis.set_xlabel("Fixed participants per round")
    axis.set_ylabel("Late-window validation MRR")
    axis.grid(alpha=0.25)
    axis.legend()
    path = Path(batch_dir) / "stage2_controls_participant_budget.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_markdown(
    batch_dir: Path,
    payload: Mapping[str, object],
    phase_name: str,
    report: Mapping[str, object],
) -> Path:
    """用简体中文写入随阶段推进更新的分析说明文档。"""

    selection = payload.get("selection") or {}
    lines = [
        "# FedAdam阶段二实验分析",
        "",
        "## 当前进度",
        "",
        "- 已完成阶段：`{}`。".format(phase_name),
        "- 通信轮数：150，每轮评估一次。",
    ]
    if selection:
        lines.extend(
            [
                "- 基线组合：`{}`。".format(
                    selection["baseline_setting"]
                ),
                "- 自动选择组合：`{}`，标签为`{}`。".format(
                    selection["selected_setting"],
                    selection["selection_label"],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 指标解释",
            "",
            "- 第131至150轮斜率绝对值不超过0.0005时标记为平台，否则只称后20轮均值。",
            "- 收敛轮次使用两臂较低后20轮均值的95%作为共同阈值，并要求连续5轮达到。",
            "- 批量训练不读取测试集；完成验证选型后，再对最佳验证检查点运行官方测试。",
            "- 固定人数实验仅估计参与预算影响，不完全隔离SnF机制。",
            "",
            "## 结构化结果",
            "",
            "详细数值见同目录JSON报告：`{}`。".format(
                {
                    "screen": "stage2_screen_analysis.json",
                    "confirm": "stage2_confirmation_analysis.json",
                    "controls": "stage2_controls_analysis.json",
                }[phase_name]
            ),
            "",
        ]
    )
    path = Path(batch_dir) / "stage2_analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_phase_artifacts(
    manifest_path: Path,
    payload: Dict[str, object],
    phase_name: str,
) -> Dict[str, object]:
    """写入当前阶段的JSON、简体中文说明和要求的曲线图。"""

    batch_dir = Path(manifest_path).resolve().parent
    if phase_name == "screen":
        report = _screen_report(payload)
        json_path = batch_dir / "stage2_screen_analysis.json"
        plot_paths = [
            _plot_screen(batch_dir, payload),
            _plot_updates(batch_dir, payload),
        ]
    elif phase_name == "confirm":
        report = _confirmation_report(payload)
        json_path = batch_dir / "stage2_confirmation_analysis.json"
        plot_paths = [_plot_confirmation(batch_dir, payload)]
    elif phase_name == "controls":
        report = _controls_report(payload)
        json_path = batch_dir / "stage2_controls_analysis.json"
        plot_paths = [_plot_controls(batch_dir, payload)]
    else:
        raise ValueError("未知阶段二报告阶段：{}".format(phase_name))
    write_json_report(report, json_path)
    markdown_path = _write_markdown(batch_dir, payload, phase_name, report)
    artifacts = {
        "analysis_json": str(json_path),
        "analysis_markdown": str(markdown_path),
        "plots": [str(path) for path in plot_paths],
    }
    payload["phases"][phase_name]["artifacts"] = artifacts
    return artifacts
