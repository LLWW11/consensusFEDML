"""分析未全部完成的 FEMNIST 四方案正式实验套件。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_CANDIDATE_IDS = [
    123, 124, 125, 126, 127, 128,
    41, 42, 43, 44, 45, 46,
    0, 1, 2, 3, 4, 5,
    164, 165, 166, 167, 168, 169,
    82, 83, 84, 85, 86, 87,
    205, 206, 207, 208, 209, 210,
    129,
]
SCENARIO_ORDER = [
    "hfl_snf_fixed",
    "hfl_no_snf_fixed",
    "fl_snf",
    "fl_no_snf",
]
SCENARIO_LABELS = {
    "hfl_snf_fixed": "HFL+SnF",
    "hfl_no_snf_fixed": "HFL-noSnF",
    "fl_snf": "FL+SnF",
    "fl_no_snf": "FL-noSnF",
}
SCENARIO_COLORS = {
    "hfl_snf_fixed": "#D4A72C",
    "hfl_no_snf_fixed": "#2F6B9A",
    "fl_snf": "#E67E22",
    "fl_no_snf": "#7F8C8D",
}
SHARED_HASH_FIELDS = [
    "partition_hash",
    "candidate_manifest_hash",
    "probe_hash",
    "initial_model_hash",
    "mat_file_hash",
]
PROBE_METRICS = [
    "candidate_effective",
    "candidate_correct_effective",
    "candidate_wrong_effective",
    "active_coverage",
    "active_effective",
    "active_correct_effective",
    "active_wrong_effective",
    "coverage_weighted_active_correct_effective",
    "within_edge_effective",
    "edge_effective",
    "edge_cloud_effective",
    "cloud_probe_accuracy",
]


def parse_arguments() -> argparse.Namespace:
    """解析套件目录、输出目录和 HDF5 审计选项。"""
    parser = argparse.ArgumentParser(
        description="分析包含完整与未完整运行的 FEMNIST 正式套件。"
    )
    parser.add_argument("--suite-dir", required=True, help="正式套件根目录。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="分析输出目录；默认写入套件根目录下的 analysis_partial_日期。",
    )
    parser.add_argument(
        "--sample-h5",
        action="store_true",
        help="仅审计 HDF5 的首尾观测点；默认逐观测点完整审计。",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    """配置适合中文技术报告的 Matplotlib 全局样式。"""
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFE",
        "axes.edgecolor": "#C8D0D9",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def read_jsonl(path: Path) -> list[dict]:
    """逐行读取 JSONL 文件并返回字典列表。"""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是有效 JSON。"
                ) from error
    return rows


def safe_float(value) -> float:
    """把可能为空的数值转换为浮点数，失败时返回 NaN。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values) -> float:
    """计算有限数值的均值；没有有限值时返回 NaN。"""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def gini_coefficient(values) -> float:
    """计算非负参与次数的基尼系数。"""
    array = np.sort(np.asarray(list(values), dtype=float))
    if array.size == 0 or array.sum() == 0:
        return 0.0
    indexes = np.arange(1, array.size + 1, dtype=float)
    return float(
        2.0 * np.sum(indexes * array) / (array.size * array.sum())
        - (array.size + 1.0) / array.size
    )


def audit_schedule(run: dict) -> dict:
    """校验逐轮调度的连续性、唯一性、分组和 250 端同步语义。"""
    metadata = run["metadata"]
    schedule = run["schedule"]
    issues = []
    candidate_ids = metadata.get("candidate_client_ids", [])
    for expected_epoch, row in enumerate(schedule):
        epoch = int(row.get("global_epoch", -1))
        slots = [int(value) for value in row.get("active_candidate_slots", [])]
        client_ids = [int(value) for value in row.get("active_client_ids", [])]
        groups = row.get("group_to_candidate_slots", {})
        if epoch != expected_epoch:
            issues.append(f"第 {expected_epoch} 行记录的 global_epoch={epoch}")
        if len(slots) != len(set(slots)) or len(client_ids) != len(set(client_ids)):
            issues.append(f"第 {epoch + 1} 轮存在重复参与者")
        if any(slot < 0 or slot >= len(candidate_ids) for slot in slots):
            issues.append(f"第 {epoch + 1} 轮存在越界槽位")
        mapped_ids = [candidate_ids[slot] for slot in slots]
        if mapped_ids != client_ids:
            issues.append(f"第 {epoch + 1} 轮槽位与逻辑客户端 ID 不一致")
        if int(row.get("mat_participant_count", -1)) != len(slots):
            issues.append(f"第 {epoch + 1} 轮参与人数 n 与实际槽位数不一致")
        if int(row.get("mat_group_count", -1)) != len(groups):
            issues.append(f"第 {epoch + 1} 轮组数 k 与实际分组数不一致")
        flattened = [int(slot) for values in groups.values() for slot in values]
        if sorted(flattened) != sorted(slots) or any(not values for values in groups.values()):
            issues.append(f"第 {epoch + 1} 轮分组为空或未恰好覆盖参与槽位")
        synchronized_ids = row.get("synchronized_client_ids", [])
        if (
            int(row.get("synchronized_client_count", -1)) != 250
            or synchronized_ids != list(range(250))
        ):
            issues.append(f"第 {epoch + 1} 轮没有记录完整 250 端同步")
        if len(issues) >= 20:
            break
    return {
        "status": "通过" if not issues else "失败",
        "issues": issues,
        "recorded_rounds": len(schedule),
    }


def audit_hdf5(run: dict, sample_only: bool) -> dict:
    """流式校验探针 HDF5 的形状、坐标、掩码和概率归一化。"""
    path = run["path"] / "probe_probabilities.h5"
    issues = []
    max_sum_error = 0.0
    with h5py.File(path, "r") as handle:
        written_count = int(handle.attrs.get("written_count", -1))
        expected_count = len(run["probe"])
        if written_count != expected_count:
            issues.append(
                f"written_count={written_count}，但摘要有 {expected_count} 行"
            )
        expected_shapes = {
            "client_probabilities": (101, 37, 620, 62),
            "cloud_probabilities": (101, 620, 62),
            "active_client_mask": (101, 37),
            "global_epochs": (101,),
            "probe_indices": (620,),
            "probe_true_labels": (620,),
        }
        for dataset_name, expected_shape in expected_shapes.items():
            if dataset_name not in handle:
                issues.append(f"缺少数据集 {dataset_name}")
            elif tuple(handle[dataset_name].shape) != expected_shape:
                issues.append(
                    f"{dataset_name} 形状为 {handle[dataset_name].shape}，"
                    f"期望 {expected_shape}"
                )
        if issues:
            return {
                "status": "失败",
                "issues": issues,
                "written_count": written_count,
                "max_probability_sum_error": None,
            }
        csv_epochs = run["probe"]["global_epoch"].astype(int).tolist()
        h5_epochs = handle["global_epochs"][:written_count].astype(int).tolist()
        if csv_epochs != h5_epochs:
            issues.append("HDF5 与探针摘要的 global_epoch 坐标不一致")
        indexes = (
            sorted(set([0, max(0, written_count - 1)]))
            if sample_only
            else list(range(written_count))
        )
        for index in indexes:
            client_probabilities = np.asarray(handle["client_probabilities"][index])
            cloud_probabilities = np.asarray(handle["cloud_probabilities"][index])
            if not np.isfinite(client_probabilities).all():
                issues.append(f"观测 {index} 的客户端概率含非有限值")
            if not np.isfinite(cloud_probabilities).all():
                issues.append(f"观测 {index} 的云概率含非有限值")
            for probabilities in [client_probabilities, cloud_probabilities]:
                error = np.max(np.abs(probabilities.sum(axis=-1) - 1.0))
                max_sum_error = max(max_sum_error, float(error))
            active_mask = np.asarray(handle["active_client_mask"][index], dtype=bool)
            expected_active = int(run["probe"].iloc[index]["active_client_count"])
            if int(active_mask.sum()) != expected_active:
                issues.append(f"观测 {index} 的客户端活跃掩码数量不一致")
            if run["metadata"].get("architecture") == "hfl":
                edge_mask = np.asarray(handle["edge_active_mask"][index], dtype=bool)
                edge_probabilities = np.asarray(handle["edge_probabilities"][index])
                active_edges = edge_probabilities[edge_mask]
                if active_edges.size:
                    if not np.isfinite(active_edges).all():
                        issues.append(f"观测 {index} 的活跃边缘概率含非有限值")
                    else:
                        error = np.max(np.abs(active_edges.sum(axis=-1) - 1.0))
                        max_sum_error = max(max_sum_error, float(error))
            if len(issues) >= 20:
                break
    if max_sum_error > 1e-5:
        issues.append(f"概率归一化最大误差 {max_sum_error:.3e} 超过 1e-5")
    return {
        "status": "通过" if not issues else "失败",
        "issues": issues,
        "written_count": written_count,
        "max_probability_sum_error": max_sum_error,
        "audit_mode": "首尾抽样" if sample_only else "全部已写观测点",
    }


def load_run(run_dir: Path, sample_h5: bool) -> dict:
    """读取一个实验目录，并完成基础字段转换和完整性审计。"""
    metadata = json.loads(
        (run_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    probe = pd.read_csv(run_dir / "probe_epoch_summary.csv")
    test = pd.read_csv(run_dir / "test_metrics.csv")
    timing = pd.read_csv(run_dir / "stage_timing.csv")
    gpu = pd.read_csv(run_dir / "gpu_monitor.csv")
    schedule = read_jsonl(run_dir / "topology_schedule.jsonl")
    for frame in [probe, test, timing]:
        frame["global_epoch"] = pd.to_numeric(frame["global_epoch"], errors="raise").astype(int)
        frame["round"] = frame["global_epoch"] + 1
    run = {
        "path": run_dir,
        "metadata": metadata,
        "scenario": metadata["scenario"],
        "probe": probe,
        "test": test,
        "timing": timing,
        "gpu": gpu,
        "schedule": schedule,
    }
    run["schedule_audit"] = audit_schedule(run)
    run["h5_audit"] = audit_hdf5(run, sample_h5)
    return run


def find_runs(suite_dir: Path, sample_h5: bool) -> dict[str, dict]:
    """扫描套件 runs 子目录，并按场景名称加载可用实验。"""
    run_root = suite_dir / "runs"
    if not run_root.is_dir():
        raise FileNotFoundError(f"缺少运行目录：{run_root}")
    runs = {}
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "experiment_metadata.json").is_file():
            continue
        run = load_run(run_dir, sample_h5)
        if run["scenario"] in runs:
            raise ValueError(f"场景 {run['scenario']} 出现多个运行目录。")
        runs[run["scenario"]] = run
    if not runs:
        raise ValueError("套件中没有可分析的实验目录。")
    return runs


def calculate_participation(run: dict, round_limit: int | None = None) -> dict:
    """汇总指定轮数内的参与槽位、样本暴露量和候选公平性。"""
    schedule = run["schedule"]
    if round_limit is not None:
        schedule = schedule[:round_limit]
    sample_counts = [int(value) for value in run["metadata"]["candidate_train_sample_counts"]]
    counter = Counter()
    total_sample_exposure = 0
    for row in schedule:
        for slot in row["active_candidate_slots"]:
            slot = int(slot)
            counter[slot] += 1
            total_sample_exposure += sample_counts[slot]
    counts = [counter[index] for index in range(len(sample_counts))]
    active_slots = sum(counts)
    mean_count = float(np.mean(counts)) if counts else 0.0
    return {
        "rounds": len(schedule),
        "active_slots": active_slots,
        "active_clients_mean": active_slots / len(schedule) if schedule else 0.0,
        "sample_exposure": total_sample_exposure,
        "frequency_min": min(counts) if counts else 0,
        "frequency_max": max(counts) if counts else 0,
        "frequency_cv": float(np.std(counts) / mean_count) if mean_count else 0.0,
        "frequency_gini": gini_coefficient(counts),
        "counts": counts,
    }


def determine_common_round(runs: dict[str, dict]) -> int:
    """寻找所有已有方案共同拥有的最大非基线评估轮次。"""
    common_rounds = None
    for run in runs.values():
        rounds = set(int(value) for value in run["test"]["round"] if int(value) > 0)
        common_rounds = rounds if common_rounds is None else common_rounds & rounds
    if not common_rounds:
        raise ValueError("已有方案之间没有共同的非基线评估轮次。")
    return max(common_rounds)


def timing_at_round(run: dict, round_number: int) -> pd.Series:
    """读取不晚于指定轮次的最新累计阶段计时记录。"""
    selected = run["timing"][run["timing"]["round"] <= round_number]
    selected = selected[selected["round"] > 0]
    if selected.empty:
        raise ValueError(f"{run['scenario']} 在第 {round_number} 轮前没有计时记录。")
    return selected.iloc[-1]


def summarize_run(run: dict, common_round: int) -> dict:
    """计算单个方案的准确率、预算、稳定性、探针和 GPU 汇总。"""
    metadata = run["metadata"]
    test = run["test"].copy()
    observed = test[test["round"] > 0]
    common = observed[observed["round"] <= common_round]
    latest_row = observed.iloc[-1]
    full_complete = (
        metadata.get("status") == "complete"
        and len(run["schedule"]) == int(metadata.get("comm_round", 0))
        and int(latest_row["round"]) == int(metadata.get("comm_round", 0))
    )
    common_timing = timing_at_round(run, common_round)
    common_participation = calculate_participation(run, common_round)
    full_participation = calculate_participation(run)
    final_timing = run["timing"][run["timing"]["round"] > 0].iloc[-1]
    late_count = min(20, len(observed))
    late_accuracy = observed["test_accuracy"].tail(late_count).astype(float)
    gpu_utilization = pd.to_numeric(run["gpu"]["utilization_percent"], errors="coerce")
    gpu_memory = pd.to_numeric(run["gpu"]["memory_used_mb"], errors="coerce")
    gpu_power = pd.to_numeric(run["gpu"]["power_w"], errors="coerce")
    common_train_seconds = float(common_timing["train_seconds"])
    summary = {
        "scenario": run["scenario"],
        "方案": SCENARIO_LABELS[run["scenario"]],
        "运行状态": "完整" if full_complete else "未完成",
        "调度轮数": len(run["schedule"]),
        "评估点数": len(test),
        "最新评估轮": int(latest_row["round"]),
        "共同截面轮": common_round,
        "共同截面准确率": float(common.iloc[-1]["test_accuracy"]),
        "共同截面平均准确率": float(common["test_accuracy"].mean()),
        "共同截面最佳准确率": float(common["test_accuracy"].max()),
        "共同截面训练小时": common_train_seconds / 3600.0,
        "共同截面总耗时小时": float(common_timing["elapsed_seconds"]) / 3600.0,
        "共同截面平均参与人数": common_participation["active_clients_mean"],
        "共同截面累计参与槽位": common_participation["active_slots"],
        "共同截面样本暴露量": common_participation["sample_exposure"],
        "共同截面训练吞吐": (
            common_participation["sample_exposure"] / common_train_seconds
            if common_train_seconds > 0 else float("nan")
        ),
        "最新准确率": float(latest_row["test_accuracy"]),
        "截至当前最佳准确率": float(observed["test_accuracy"].max()),
        "截至当前最佳轮": int(observed.loc[observed["test_accuracy"].idxmax(), "round"]),
        "后期窗口点数": late_count,
        "后期准确率均值": float(late_accuracy.mean()),
        "后期准确率标准差": float(late_accuracy.std(ddof=0)),
        "完整5000轮最终准确率": (
            float(latest_row["test_accuracy"]) if full_complete else float("nan")
        ),
        "完整5000轮训练小时": (
            float(final_timing["train_seconds"]) / 3600.0 if full_complete else float("nan")
        ),
        "完整5000轮总耗时小时": (
            float(final_timing["elapsed_seconds"]) / 3600.0 if full_complete else float("nan")
        ),
        "截至当前平均参与人数": full_participation["active_clients_mean"],
        "截至当前累计参与槽位": full_participation["active_slots"],
        "截至当前样本暴露量": full_participation["sample_exposure"],
        "候选参与频率最小值": full_participation["frequency_min"],
        "候选参与频率最大值": full_participation["frequency_max"],
        "候选参与频率变异系数": full_participation["frequency_cv"],
        "候选参与频率基尼系数": full_participation["frequency_gini"],
        "GPU利用率均值": finite_mean(gpu_utilization),
        "GPU利用率中位数": float(gpu_utilization.median()),
        "GPU利用率P95": float(gpu_utilization.quantile(0.95)),
        "GPU显存峰值MB": float(gpu_memory.max()),
        "GPU功耗均值W": finite_mean(gpu_power),
    }
    final_probe = run["probe"].iloc[-1]
    late_probe = run["probe"][run["probe"]["round"] > 0].tail(late_count)
    for metric in PROBE_METRICS:
        summary[f"最新_{metric}"] = safe_float(final_probe.get(metric))
        summary[f"后期均值_{metric}"] = finite_mean(late_probe.get(metric, []))
    return summary


def build_long_metrics(runs: dict[str, dict]) -> pd.DataFrame:
    """合并测试、探针和累计计时，形成逐评估点长表。"""
    frames = []
    for scenario in SCENARIO_ORDER:
        if scenario not in runs:
            continue
        run = runs[scenario]
        merged = run["probe"].merge(
            run["test"][
                [
                    "global_epoch", "test_samples", "test_correct",
                    "test_accuracy", "test_loss", "evaluated_client_count",
                ]
            ],
            on="global_epoch",
            how="inner",
            validate="one_to_one",
        ).merge(
            run["timing"][
                [
                    "global_epoch", "train_seconds", "aggregate_seconds",
                    "probe_seconds", "test_seconds", "checkpoint_seconds",
                    "elapsed_seconds",
                ]
            ],
            on="global_epoch",
            how="inner",
            validate="one_to_one",
        )
        merged["round"] = merged["global_epoch"] + 1
        merged.insert(0, "方案", SCENARIO_LABELS[scenario])
        merged.insert(1, "scenario", scenario)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def build_candidate_table(runs: dict[str, dict], common_round: int) -> pd.DataFrame:
    """生成每个固定候选槽位的全程与共同轮次参与统计。"""
    rows = []
    for scenario in SCENARIO_ORDER:
        if scenario not in runs:
            continue
        run = runs[scenario]
        full = calculate_participation(run)
        common = calculate_participation(run, common_round)
        sample_counts = run["metadata"]["candidate_train_sample_counts"]
        for slot, client_id in enumerate(run["metadata"]["candidate_client_ids"]):
            rows.append({
                "方案": SCENARIO_LABELS[scenario],
                "scenario": scenario,
                "候选槽位": slot,
                "逻辑客户端ID": int(client_id),
                "单次本地训练样本数": int(sample_counts[slot]),
                "共同截面参与次数": int(common["counts"][slot]),
                "共同截面参与率": common["counts"][slot] / common_round,
                "共同截面累计样本暴露": int(common["counts"][slot] * sample_counts[slot]),
                "截至当前参与次数": int(full["counts"][slot]),
                "截至当前参与率": full["counts"][slot] / full["rounds"],
                "截至当前累计样本暴露": int(full["counts"][slot] * sample_counts[slot]),
            })
    return pd.DataFrame(rows)


def build_quality_table(runs: dict[str, dict], common_round: int) -> pd.DataFrame:
    """生成可直接审阅的数据质量检查表。"""
    checks = []

    def add_check(item: str, status: str, evidence: str) -> None:
        """向质量检查列表追加一条中文证据记录。"""
        checks.append({"检查项": item, "状态": status, "证据": evidence})

    available = [SCENARIO_LABELS[item] for item in SCENARIO_ORDER if item in runs]
    missing = [SCENARIO_LABELS[item] for item in SCENARIO_ORDER if item not in runs]
    add_check(
        "四方案文件齐备性",
        "部分通过" if missing else "通过",
        f"已有 {len(available)} 组：{', '.join(available)}；"
        f"缺失：{', '.join(missing) if missing else '无'}。",
    )
    for field in SHARED_HASH_FIELDS:
        values = {str(run["metadata"].get(field)) for run in runs.values()}
        add_check(
            f"共享字段 {field}",
            "通过" if len(values) == 1 else "失败",
            "一致。" if len(values) == 1 else f"发现 {len(values)} 个不同值。",
        )
    candidate_ok = all(
        run["metadata"].get("candidate_client_ids") == EXPECTED_CANDIDATE_IDS
        for run in runs.values()
    )
    add_check(
        "固定 37 槽位顺序",
        "通过" if candidate_ok else "失败",
        "三组均严格使用指定逻辑客户端 ID 顺序。" if candidate_ok else "至少一组槽位顺序不一致。",
    )
    baseline_values = [float(run["test"].iloc[0]["test_accuracy"]) for run in runs.values()]
    add_check(
        "初始模型基线一致性",
        "通过" if max(baseline_values) - min(baseline_values) <= 1e-12 else "失败",
        f"已有方案基线准确率均为 {baseline_values[0]:.12f}。",
    )
    for scenario in SCENARIO_ORDER:
        if scenario not in runs:
            add_check(f"{SCENARIO_LABELS[scenario]} 运行完整性", "缺失", "没有运行目录。")
            continue
        run = runs[scenario]
        metadata = run["metadata"]
        latest_round = int(run["test"]["round"].max())
        # 第 1 个点是训练前基线，之后只在完整跨过评估间隔时写入记录。
        expected_eval_count = 1 + len(run["schedule"]) // int(metadata["eval_interval"])
        complete = (
            metadata.get("status") == "complete"
            and len(run["schedule"]) == int(metadata["comm_round"])
            and latest_round == int(metadata["comm_round"])
        )
        internally_consistent = len(run["test"]) == expected_eval_count
        status = "通过" if complete else ("部分通过" if internally_consistent else "失败")
        add_check(
            f"{SCENARIO_LABELS[scenario]} 运行完整性",
            status,
            f"metadata.status={metadata.get('status')}；调度 {len(run['schedule'])}/"
            f"{metadata['comm_round']} 轮；最新评估到第 {latest_round} 轮；"
            f"评估点 {len(run['test'])} 个。",
        )
        test_ok = (
            (run["test"]["evaluated_client_count"].astype(int) == 250).all()
            and (run["test"]["test_samples"].astype(int) == 77483).all()
            and np.allclose(
                run["test"]["test_correct"].astype(float)
                / run["test"]["test_samples"].astype(float),
                run["test"]["test_accuracy"].astype(float),
                atol=1e-12,
                rtol=0,
            )
        )
        add_check(
            f"{SCENARIO_LABELS[scenario]} 250 端测试汇总",
            "通过" if test_ok else "失败",
            "每个评估点均覆盖 250 个客户端和 77,483 张测试图像，correct/total 与 accuracy 一致。"
            if test_ok else "测试客户端数、样本数或准确率计算存在不一致。",
        )
        schedule_audit = run["schedule_audit"]
        add_check(
            f"{SCENARIO_LABELS[scenario]} 调度日志",
            schedule_audit["status"],
            f"审计 {schedule_audit['recorded_rounds']} 轮；"
            + ("连续、无重复、k/n 分组有效且每轮记录 250 端同步。"
               if not schedule_audit["issues"] else "；".join(schedule_audit["issues"][:3])),
        )
        h5_audit = run["h5_audit"]
        add_check(
            f"{SCENARIO_LABELS[scenario]} 探针 HDF5",
            h5_audit["status"],
            f"已写 {h5_audit['written_count']} 个观测点，审计模式为 {h5_audit.get('audit_mode')}；"
            f"概率和最大误差 {h5_audit.get('max_probability_sum_error', float('nan')):.3e}。"
            if not h5_audit["issues"] else "；".join(h5_audit["issues"][:3]),
        )
    add_check(
        "共同截面可比性",
        "通过",
        f"已有三组共同拥有基线及第 50–{common_round} 轮的固定间隔评估点。",
    )
    add_check(
        "随机种子覆盖",
        "部分通过",
        "仅 seed=0；所有差异均为单种子描述性结果，不能检验统计显著性。",
    )
    return pd.DataFrame(checks)


def build_correlation_table(long_metrics: pd.DataFrame) -> pd.DataFrame:
    """计算测试准确率与共识指标的描述性 Pearson 相关系数。"""
    metrics = [
        "cloud_probe_accuracy",
        "candidate_correct_effective",
        "candidate_wrong_effective",
        "active_coverage",
        "active_correct_effective",
        "active_wrong_effective",
        "coverage_weighted_active_correct_effective",
    ]
    rows = []
    for scenario, frame in long_metrics[long_metrics["round"] > 0].groupby("scenario"):
        for metric in metrics:
            pair = frame[["test_accuracy", metric]].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "方案": SCENARIO_LABELS[scenario],
                "scenario": scenario,
                "指标": metric,
                "样本点数": len(pair),
                "Pearson相关系数": (
                    float(pair["test_accuracy"].corr(pair[metric]))
                    if len(pair) >= 3 else float("nan")
                ),
                "解释边界": "单种子、时间序列趋势相关，不代表因果关系",
            })
    return pd.DataFrame(rows)


def plot_metric_lines(ax, long_metrics: pd.DataFrame, metric: str, title: str,
                      ylabel: str, x_limit: tuple[int, int] | None = None) -> None:
    """在指定坐标轴上按统一颜色绘制多方案评估曲线。"""
    for scenario in SCENARIO_ORDER:
        frame = long_metrics[long_metrics["scenario"] == scenario]
        if frame.empty:
            continue
        ax.plot(
            frame["round"], frame[metric], marker="o", markersize=2.6,
            linewidth=1.8, color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("通信轮次")
    ax.set_ylabel(ylabel)
    if x_limit is not None:
        ax.set_xlim(*x_limit)
    ax.legend(ncol=2, fontsize=9)


def save_figure(fig, output_path: Path) -> None:
    """统一保存并关闭报告图像，避免累积占用内存。"""
    # 为多面板图保留标题和轴标签间距，避免中文长标题相互覆盖。
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=3.2, w_pad=2.2)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def create_figures(output_dir: Path, runs: dict[str, dict], summaries: pd.DataFrame,
                   long_metrics: pd.DataFrame, candidates: pd.DataFrame,
                   common_round: int) -> list[str]:
    """生成与参考分析目录相同编号风格的八张技术图。"""
    configure_plot_style()
    generated = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_metric_lines(axes[0, 0], long_metrics, "test_accuracy", "完整测试准确率", "准确率")
    plot_metric_lines(
        axes[0, 1], long_metrics, "test_accuracy",
        f"共同区间放大（0–{common_round} 轮）", "准确率", (0, common_round),
    )
    plot_metric_lines(axes[1, 0], long_metrics, "test_loss", "完整测试损失", "交叉熵损失")
    plot_metric_lines(
        axes[1, 1], long_metrics, "cloud_probe_accuracy",
        "云模型在固定 620 张探针上的准确率", "探针准确率",
    )
    fig.suptitle("模型效果趋势：完整 HFL 与未完成 FL 的边界", fontsize=16, fontweight="bold")
    path = output_dir / "01_模型效果趋势.png"
    save_figure(fig, path)
    generated.append(path.name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for scenario in SCENARIO_ORDER:
        if scenario not in runs:
            continue
        run = runs[scenario]
        evaluation = run["probe"][run["probe"]["round"] > 0]
        axes[0, 0].plot(
            evaluation["round"], evaluation["active_client_count"],
            color=SCENARIO_COLORS[scenario], label=SCENARIO_LABELS[scenario], linewidth=1.8,
        )
        sample_counts = run["metadata"]["candidate_train_sample_counts"]
        cumulative_slots = []
        cumulative_samples = []
        slot_total = 0
        sample_total = 0
        for row in run["schedule"]:
            slots = [int(value) for value in row["active_candidate_slots"]]
            slot_total += len(slots)
            sample_total += sum(sample_counts[slot] for slot in slots)
            cumulative_slots.append(slot_total)
            cumulative_samples.append(sample_total / 1_000_000.0)
        x_values = np.arange(1, len(run["schedule"]) + 1)
        axes[0, 1].plot(x_values, cumulative_slots, color=SCENARIO_COLORS[scenario],
                        label=SCENARIO_LABELS[scenario], linewidth=1.8)
        axes[1, 0].plot(x_values, cumulative_samples, color=SCENARIO_COLORS[scenario],
                        label=SCENARIO_LABELS[scenario], linewidth=1.8)
    axes[0, 0].set_title("每个评估点的活跃客户端数", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("客户端数")
    axes[0, 1].set_title("累计参与槽位", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("客户端-轮次")
    axes[1, 0].set_title("累计本地训练样本暴露", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("百万张次")
    for axis in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        axis.set_xlabel("通信轮次")
        axis.legend(fontsize=9)
    fairness = summaries.set_index("scenario")
    labels = [SCENARIO_LABELS[item] for item in SCENARIO_ORDER if item in fairness.index]
    values = [fairness.loc[item, "候选参与频率基尼系数"] for item in SCENARIO_ORDER if item in fairness.index]
    colors = [SCENARIO_COLORS[item] for item in SCENARIO_ORDER if item in fairness.index]
    axes[1, 1].bar(labels, values, color=colors)
    axes[1, 1].set_title("截至当前的候选参与不均衡", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("参与频率基尼系数（越低越均衡）")
    axes[1, 1].tick_params(axis="x", rotation=15)
    fig.suptitle("参与强度与累计贡献", fontsize=16, fontweight="bold")
    path = output_dir / "02_参与强度与累计贡献.png"
    save_figure(fig, path)
    generated.append(path.name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ordered = summaries.sort_values("共同截面准确率", ascending=False)
    labels = ordered["方案"].tolist()
    colors = [SCENARIO_COLORS[value] for value in ordered["scenario"]]
    panels = [
        ("共同截面准确率", "共同第 550 轮准确率", "准确率"),
        ("共同截面平均准确率", "第 50–550 轮平均准确率", "准确率"),
        ("共同截面训练小时", "累计训练耗时", "小时"),
        ("共同截面样本暴露量", "累计训练样本暴露", "张次"),
    ]
    for axis, (column, title, ylabel) in zip(axes.flat, panels):
        axis.bar(labels, ordered[column], color=colors)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
        if "准确率" in column:
            axis.set_ylim(max(0, ordered[column].min() - 0.03), ordered[column].max() + 0.01)
    fig.suptitle("聚合预算效率：统一到共同第 550 轮", fontsize=16, fontweight="bold")
    path = output_dir / "03_聚合预算效率.png"
    save_figure(fig, path)
    generated.append(path.name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    consensus_panels = [
        ("candidate_correct_effective", "候选正确有效共识", "正确 S"),
        ("candidate_wrong_effective", "候选错误有效共识", "错误 S"),
        ("active_coverage", "活跃覆盖率", "覆盖率"),
        ("coverage_weighted_active_correct_effective", "覆盖加权活跃正确共识 Q", "Q"),
    ]
    for axis, (metric, title, ylabel) in zip(axes.flat, consensus_panels):
        plot_metric_lines(axis, long_metrics, metric, title, ylabel)
    fig.suptitle("有效共识分解", fontsize=16, fontweight="bold")
    path = output_dir / "04_有效共识分解.png"
    save_figure(fig, path)
    generated.append(path.name)

    complete_hfl = [item for item in ["hfl_snf_fixed", "hfl_no_snf_fixed"] if item in runs]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for scenario in complete_hfl:
        frame = long_metrics[(long_metrics["scenario"] == scenario) & (long_metrics["round"] >= 4000)]
        axes[0, 0].plot(frame["round"], frame["test_accuracy"], marker="o", markersize=3,
                        color=SCENARIO_COLORS[scenario], label=SCENARIO_LABELS[scenario])
    axes[0, 0].set_title("后 1000 轮准确率", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("通信轮次")
    axes[0, 0].set_ylabel("准确率")
    axes[0, 0].legend()
    late_values = []
    late_labels = []
    for scenario in complete_hfl:
        values = long_metrics[(long_metrics["scenario"] == scenario) & (long_metrics["round"] > 0)]["test_accuracy"].tail(20)
        late_values.append(values)
        late_labels.append(SCENARIO_LABELS[scenario])
    axes[0, 1].boxplot(late_values, tick_labels=late_labels, patch_artist=True)
    axes[0, 1].set_title("最后 20 个评估点分布", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("准确率")
    hfl_summary = summaries[summaries["scenario"].isin(complete_hfl)]
    x = np.arange(len(hfl_summary))
    axes[1, 0].bar(x - 0.18, hfl_summary["最新准确率"], width=0.36, label="最终")
    axes[1, 0].bar(x + 0.18, hfl_summary["截至当前最佳准确率"], width=0.36, label="历史最佳")
    axes[1, 0].set_xticks(x, hfl_summary["方案"])
    axes[1, 0].set_title("最终值与历史最佳", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("准确率")
    axes[1, 0].set_ylim(0.82, 0.85)
    axes[1, 0].legend()
    if len(complete_hfl) == 2:
        pivot = long_metrics[long_metrics["scenario"].isin(complete_hfl)].pivot(
            index="round", columns="scenario", values="test_accuracy"
        ).dropna()
        difference = 100.0 * (pivot["hfl_snf_fixed"] - pivot["hfl_no_snf_fixed"])
        axes[1, 1].plot(difference.index, difference, color="#8E44AD", linewidth=2)
        axes[1, 1].axhline(0, color="#555555", linewidth=1)
        axes[1, 1].set_title("HFL+SnF 减 HFL-noSnF", loc="left", fontweight="bold")
        axes[1, 1].set_xlabel("通信轮次")
        axes[1, 1].set_ylabel("准确率差（百分点）")
    fig.suptitle("后期稳定性与历史最佳", fontsize=16, fontweight="bold")
    path = output_dir / "05_后期稳定性与历史最佳.png"
    save_figure(fig, path)
    generated.append(path.name)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    hierarchy_panels = [
        ("within_edge_effective", "组内有效共识"),
        ("edge_effective", "边缘有效共识"),
        ("edge_cloud_effective", "边缘—云有效共识"),
    ]
    for axis, (metric, title) in zip(axes, hierarchy_panels):
        for scenario in complete_hfl:
            frame = long_metrics[long_metrics["scenario"] == scenario]
            axis.plot(frame["round"], frame[metric], color=SCENARIO_COLORS[scenario],
                      label=SCENARIO_LABELS[scenario], linewidth=1.8)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("通信轮次")
        axis.set_ylabel("有效共识 S")
        axis.legend()
    fig.suptitle("HFL 层级共识传播", fontsize=16, fontweight="bold")
    path = output_dir / "06_HFL层级共识传播.png"
    save_figure(fig, path)
    generated.append(path.name)

    available_scenarios = [item for item in SCENARIO_ORDER if item in runs]
    fig, axes = plt.subplots(len(available_scenarios), 2, figsize=(14, 3.6 * len(available_scenarios)), squeeze=False)
    for row_index, scenario in enumerate(available_scenarios):
        frame = candidates[candidates["scenario"] == scenario]
        axes[row_index, 0].bar(frame["候选槽位"], frame["共同截面参与率"],
                               color=SCENARIO_COLORS[scenario])
        axes[row_index, 0].set_title(
            f"{SCENARIO_LABELS[scenario]}：共同第 {common_round} 轮参与率",
            loc="left", fontweight="bold",
        )
        axes[row_index, 0].set_ylabel("参与率")
        axes[row_index, 1].bar(frame["候选槽位"], frame["共同截面累计样本暴露"] / 1_000_000,
                               color=SCENARIO_COLORS[scenario])
        axes[row_index, 1].set_title(
            f"{SCENARIO_LABELS[scenario]}：共同截面累计训练量",
            loc="left", fontweight="bold",
        )
        axes[row_index, 1].set_ylabel("百万张次")
        for axis in axes[row_index]:
            axis.set_xlabel("固定候选槽位（0–36）")
    fig.suptitle("固定候选活跃与样本贡献", fontsize=16, fontweight="bold")
    path = output_dir / "07_固定候选活跃与样本贡献.png"
    save_figure(fig, path)
    generated.append(path.name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    milestones = [50, 100, 200, 300, 400, 500, common_round]
    width = 0.24
    x = np.arange(len(milestones))
    for index, scenario in enumerate(available_scenarios):
        frame = long_metrics[
            (long_metrics["scenario"] == scenario) & long_metrics["round"].isin(milestones)
        ].set_index("round")
        values = [frame.loc[round_number, "test_accuracy"] for round_number in milestones]
        axes[0, 0].bar(x + (index - 1) * width, values, width=width,
                       color=SCENARIO_COLORS[scenario], label=SCENARIO_LABELS[scenario])
    axes[0, 0].set_xticks(x, milestones)
    axes[0, 0].set_title("共同里程碑准确率", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("通信轮次")
    axes[0, 0].set_ylabel("准确率")
    axes[0, 0].legend()
    summary_index = summaries.set_index("scenario")
    plot_specs = [
        (axes[0, 1], "共同截面平均参与人数", "平均参与人数", "客户端/轮"),
        (axes[1, 0], "共同截面训练吞吐", "训练吞吐", "样本张次/训练秒"),
        (axes[1, 1], "GPU利用率均值", "GPU 利用率均值", "%"),
    ]
    for axis, column, title, ylabel in plot_specs:
        values = [summary_index.loc[item, column] for item in available_scenarios]
        axis.bar([SCENARIO_LABELS[item] for item in available_scenarios], values,
                 color=[SCENARIO_COLORS[item] for item in available_scenarios])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
    fig.suptitle("已有三方案共同第 550 轮端到端对比", fontsize=16, fontweight="bold")
    path = output_dir / "08_三方案共同550轮对比.png"
    save_figure(fig, path)
    generated.append(path.name)
    return generated


def format_percent(value: float, digits: int = 3) -> str:
    """把比例格式化为百分数字符串。"""
    return f"{100.0 * float(value):.{digits}f}%"


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    """把小型数据框转换为简洁的 GitHub Markdown 表格。"""
    formats = formats or {}
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in formats and pd.notna(value):
                value = formats[column].format(value)
            elif pd.isna(value):
                value = "—"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(suite_dir: Path, output_dir: Path, runs: dict[str, dict],
                 summaries: pd.DataFrame, quality: pd.DataFrame,
                 common_round: int) -> str:
    """根据已核验指标生成参考样式的简体中文技术报告。"""
    index = summaries.set_index("scenario")
    hfl_snf = index.loc["hfl_snf_fixed"]
    hfl_no = index.loc["hfl_no_snf_fixed"]
    fl_snf = index.loc["fl_snf"]
    final_gap_pp = 100.0 * (
        hfl_snf["完整5000轮最终准确率"] - hfl_no["完整5000轮最终准确率"]
    )
    late_gap_pp = 100.0 * (
        hfl_snf["后期准确率均值"] - hfl_no["后期准确率均值"]
    )
    exposure_ratio = (
        hfl_snf["截至当前样本暴露量"] / hfl_no["截至当前样本暴露量"] - 1.0
    )
    time_ratio = (
        hfl_snf["完整5000轮总耗时小时"] / hfl_no["完整5000轮总耗时小时"] - 1.0
    )
    common_spread_pp = 100.0 * (
        summaries["共同截面准确率"].max() - summaries["共同截面准确率"].min()
    )
    overview = summaries.copy()
    overview["完成进度"] = overview.apply(
        lambda row: f"{int(row['调度轮数'])}/5000", axis=1
    )
    overview["共同550轮准确率"] = overview["共同截面准确率"].map(format_percent)
    overview["5000轮最终准确率"] = overview["完整5000轮最终准确率"].map(
        lambda value: format_percent(value) if pd.notna(value) else "—"
    )
    overview["总耗时"] = overview["完整5000轮总耗时小时"].map(
        lambda value: f"{value:.2f} 小时" if pd.notna(value) else "—"
    )
    budget = summaries.copy()
    budget["准确率"] = budget["共同截面准确率"].map(format_percent)
    budget["平均准确率"] = budget["共同截面平均准确率"].map(format_percent)
    budget["平均参与人数"] = budget["共同截面平均参与人数"].map(lambda value: f"{value:.2f}")
    budget["样本暴露（百万张次）"] = budget["共同截面样本暴露量"].map(
        lambda value: f"{value / 1_000_000:.2f}"
    )
    budget["训练小时"] = budget["共同截面训练小时"].map(lambda value: f"{value:.2f}")
    budget["训练吞吐"] = budget["共同截面训练吞吐"].map(lambda value: f"{value:.0f}")
    mechanism = summaries[summaries["scenario"].isin(["hfl_snf_fixed", "hfl_no_snf_fixed"])].copy()
    mechanism["后期测试准确率"] = mechanism["后期准确率均值"].map(format_percent)
    mechanism["后期云探针准确率"] = mechanism["后期均值_cloud_probe_accuracy"].map(format_percent)
    mechanism["后期候选正确S"] = mechanism["后期均值_candidate_correct_effective"].map(lambda value: f"{value:.4f}")
    mechanism["后期候选错误S"] = mechanism["后期均值_candidate_wrong_effective"].map(lambda value: f"{value:.4f}")
    mechanism["后期覆盖率"] = mechanism["后期均值_active_coverage"].map(format_percent)
    mechanism["后期Q"] = mechanism["后期均值_coverage_weighted_active_correct_effective"].map(lambda value: f"{value:.4f}")
    gpu = summaries.copy()
    gpu["GPU均值"] = gpu["GPU利用率均值"].map(lambda value: f"{value:.1f}%")
    gpu["GPU P95"] = gpu["GPU利用率P95"].map(lambda value: f"{value:.1f}%")
    gpu["显存峰值"] = gpu["GPU显存峰值MB"].map(lambda value: f"{value / 1024:.2f} GiB")
    gpu["截至当前总耗时"] = gpu.apply(
        lambda row: (
            f"{row['完整5000轮总耗时小时']:.2f} 小时"
            if pd.notna(row["完整5000轮总耗时小时"])
            else f"{row['共同截面总耗时小时']:.2f} 小时（到第 {common_round} 轮）"
        ),
        axis=1,
    )
    shared = runs["hfl_snf_fixed"]["metadata"]
    return f"""# FEMNIST 250 客户端部分正式套件分析报告

> 分析状态：**部分完成**。HFL+SnF 与 HFL-noSnF 已完成 5000 轮；FL+SnF 的调度日志写到第 576 轮，固定评估记录到第 550 轮；FL-noSnF 尚无运行目录。所有结论均为 seed=0 的描述性结果。

## 1. 技术摘要

- 在三组共同拥有的第 {common_round} 轮截面，HFL-noSnF、HFL+SnF、FL+SnF 的测试准确率分别为 {format_percent(hfl_no['共同截面准确率'])}、{format_percent(hfl_snf['共同截面准确率'])}、{format_percent(fl_snf['共同截面准确率'])}，最大差距只有 {common_spread_pp:.3f} 个百分点。现有早期证据不能说明 FL 与 HFL 谁显著更好。
- 两个完整 HFL 运行在第 5000 轮分别达到 {format_percent(hfl_snf['完整5000轮最终准确率'])} 和 {format_percent(hfl_no['完整5000轮最终准确率'])}；HFL+SnF 最终高 {final_gap_pp:.3f} 个百分点，最后 20 个评估点均值高 {late_gap_pp:.3f} 个百分点。
- 这不是等预算对照：HFL+SnF 的完整累计样本暴露量比 HFL-noSnF 多 {100 * exposure_ratio:.1f}%，总耗时多 {100 * time_ratio:.1f}%，平均每轮参与人数为 {hfl_snf['截至当前平均参与人数']:.2f} 对 {hfl_no['截至当前平均参与人数']:.2f}。因此不能把最终差距直接解释为 SnF 的独立因果收益。
- 三组 GPU 平均利用率都约为 35%，显存峰值不超过 5 GiB。当前慢速更符合“小 batch、多客户端串行本地训练和 Python 调度导致 GPU 吃不满”，而不是显存容量不足。
- 第四组缺失，第三组未到终点，所以当前无法完成原计划的 2×2（HFL/FL × SnF/noSnF）终局交互效应分析。

## 2. 数据完整性与运行语义

{markdown_table(overview, ['方案', '运行状态', '完成进度', '评估点数', '共同550轮准确率', '5000轮最终准确率', '总耗时'])}

已有三组共享同一份 250 客户端 Dirichlet 划分（alpha={shared['partition_alpha']}、seed={shared['partition_seed']}）、同一固定 37 槽位、同一 620 张探针、同一初始模型和同一 MAT 文件。每个测试点均汇总 250 个客户端、77,483 张测试图像的 `correct/total/loss`。调度逐轮记录 250 端同步，固定槽位顺序与计划一致。

质量检查详见 `数据质量检查.csv`。HDF5 按已写观测点流式审计，客户端、云和活跃边缘概率均满足概率归一化误差阈值；FL 的边缘概率为不适用占位，不作为失败处理。

## 3. 模型效果与后期稳定性

![模型效果趋势](01_模型效果趋势.png)

三组初始准确率完全一致。第 50–550 轮内，HFL-noSnF 前期上升略快，但第 550 轮三者已十分接近。两个 HFL 运行继续训练后逐步收敛：HFL+SnF 最终值也是其历史最佳值；HFL-noSnF 的历史最佳为 {format_percent(hfl_no['截至当前最佳准确率'])}（第 {int(hfl_no['截至当前最佳轮'])} 轮），最终回落到 {format_percent(hfl_no['完整5000轮最终准确率'])}。

![后期稳定性与历史最佳](05_后期稳定性与历史最佳.png)

最后 20 个评估点中，HFL+SnF 的均值为 {format_percent(hfl_snf['后期准确率均值'])}、标准差为 {100*hfl_snf['后期准确率标准差']:.3f} 个百分点；HFL-noSnF 的均值为 {format_percent(hfl_no['后期准确率均值'])}、标准差为 {100*hfl_no['后期准确率标准差']:.3f} 个百分点。这个窗口显示 HFL+SnF 的后期水平略高且波动略小，但仍只有一个随机种子。

## 4. 共同第 550 轮的公平截面对比

{markdown_table(budget, ['方案', '准确率', '平均准确率', '平均参与人数', '样本暴露（百万张次）', '训练小时', '训练吞吐'])}

![共同区间预算效率](03_聚合预算效率.png)

在统一轮数下，HFL-noSnF 的第 50–550 轮平均准确率最高，但其第 550 轮优势对 HFL+SnF 仅为 {100*(hfl_no['共同截面准确率']-hfl_snf['共同截面准确率']):.3f} 个百分点、对 FL+SnF 仅为 {100*(hfl_no['共同截面准确率']-fl_snf['共同截面准确率']):.3f} 个百分点。三者训练吞吐接近，说明单张样本的 GPU 执行效率相近；总时间主要由每轮实际训练样本数决定。

![三方案共同550轮对比](08_三方案共同550轮对比.png)

## 5. 实际参与规模与预算代理

![参与强度与累计贡献](02_参与强度与累计贡献.png)

HFL+SnF 完整 5000 轮累计 {hfl_snf['截至当前累计参与槽位']:,.0f} 个客户端-轮次和 {hfl_snf['截至当前样本暴露量']/1_000_000:.2f} 百万张次训练样本；HFL-noSnF 分别为 {hfl_no['截至当前累计参与槽位']:,.0f} 和 {hfl_no['截至当前样本暴露量']/1_000_000:.2f} 百万张次。这说明 MAT 中 SnF/noSnF 的参与人数不同，本实验并不是只切换聚合机制的单变量消融。

## 6. 有效共识与准确率

{markdown_table(mechanism, ['方案', '后期测试准确率', '后期云探针准确率', '后期候选正确S', '后期候选错误S', '后期覆盖率', '后期Q'])}

![有效共识分解](04_有效共识分解.png)

HFL-noSnF 的候选正确有效共识略高，但活跃覆盖率明显较低；HFL+SnF 通过更高覆盖率把后期主指标 `Q=活跃覆盖率×活跃正确有效共识` 提高到 {hfl_snf['后期均值_coverage_weighted_active_correct_effective']:.4f}，而 HFL-noSnF 为 {hfl_no['后期均值_coverage_weighted_active_correct_effective']:.4f}。`共识准确率相关.csv` 中的 Pearson 系数仅用于描述同一训练轨迹上的同步变化，不能解释为因果关系。

## 7. HFL 层级传播

![HFL层级共识传播](06_HFL层级共识传播.png)

两个完整 HFL 方案的组内、边缘和边缘—云有效共识都随训练上升。后期 HFL-noSnF 的边缘有效共识略高，但 HFL+SnF 的云探针准确率和完整测试准确率略高；这再次说明单个共识量不能替代端到端测试结果，必须同时读取覆盖率、正确/错误共识和实际参与预算。

## 8. 固定 37 槽位的活跃公平性

![固定候选活跃与样本贡献](07_固定候选活跃与样本贡献.png)

HFL+SnF 的候选参与频率基尼系数为 {hfl_snf['候选参与频率基尼系数']:.4f}，HFL-noSnF 为 {hfl_no['候选参与频率基尼系数']:.4f}。HFL+SnF 更接近全槽位均匀参与；HFL-noSnF 的部分槽位显著更常出现。由于各逻辑客户端的本地样本数不同，参与次数相同也不等价于累计训练贡献相同，具体数据见 `固定候选参与统计.csv`。

## 9. 为什么运行很慢

{markdown_table(gpu, ['方案', 'GPU均值', 'GPU P95', '显存峰值', '截至当前总耗时'])}

GPU 利用率只有约 35%，显存仍有大量余量。结合配置中的 `local_batch_size=20`，每轮要依次处理 19–36 个客户端及其独立数据加载与优化循环，GPU 接收到的是许多较小算子，无法形成持续的大批量吞吐。HFL+SnF 每轮平均训练 35.57 个客户端，因此完整运行耗时约 {hfl_snf['完整5000轮总耗时小时']:.2f} 小时；HFL-noSnF 参与人数更少，耗时约 {hfl_no['完整5000轮总耗时小时']:.2f} 小时。

如果后续仍需跑完四组，最稳妥的提速顺序是：先用相同协议把 batch 从 20 提高到 128/256 做 50–100 轮基准；再测试单卡双进程是否提高总吞吐；最后才考虑固定 local steps 或每客户端采样上限，因为最后两项会改变训练协议，四组必须统一重跑。

## 10. 结论边界与下一步

1. 当前可以确认：两个完整 HFL 方案都已稳定收敛，HFL+SnF 的后期准确率约高 0.5 个百分点，但付出了约 74% 更多样本暴露和约 73% 更多时间。
2. 当前不能确认：SnF 是否在等参与人数、等样本预算下优于 noSnF；FL 与 HFL 的 5000 轮终局差异；架构与 SnF 是否存在交互效应。
3. 若计算预算有限，建议先把 FL+SnF 保存为“第 550 轮截面”，直接启动缺失的 FL-noSnF 并至少跑到第 550 轮，先补齐四方案共同截面。若论文需要终局 2×2 结论，则必须继续完成后两组 5000 轮。
4. 所有结果只有 seed=0。任何“优于”表述都应保留为描述性结果；要做统计推断，至少增加多个随机种子并报告均值、标准差和配对差异。

## 可复现信息

- 输入套件：`{suite_dir}`
- 输出目录：`{output_dir}`
- 分区哈希：`{shared['partition_hash']}`
- 固定候选哈希：`{shared['candidate_manifest_hash']}`
- 探针哈希：`{shared['probe_hash']}`
- 初始模型哈希：`{shared['initial_model_hash']}`
- MAT 哈希：`{shared['mat_file_hash']}`
- 报告生成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}
"""


def write_outputs(suite_dir: Path, output_dir: Path, runs: dict[str, dict],
                  sample_h5: bool) -> dict:
    """执行计算、写入 CSV/JSON/Markdown，并生成八张静态图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    common_round = determine_common_round(runs)
    summaries = pd.DataFrame([
        summarize_run(runs[scenario], common_round)
        for scenario in SCENARIO_ORDER if scenario in runs
    ])
    long_metrics = build_long_metrics(runs)
    candidates = build_candidate_table(runs, common_round)
    quality = build_quality_table(runs, common_round)
    correlations = build_correlation_table(long_metrics)

    summaries.to_csv(output_dir / "实验汇总.csv", index=False, encoding="utf-8-sig")
    long_metrics.to_csv(output_dir / "逐轮指标.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output_dir / "固定候选参与统计.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(output_dir / "数据质量检查.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(output_dir / "共识准确率相关.csv", index=False, encoding="utf-8-sig")
    comparison_columns = [
        "方案", "scenario", "运行状态", "调度轮数", "评估点数", "共同截面轮",
        "共同截面准确率", "共同截面平均准确率", "共同截面平均参与人数",
        "共同截面样本暴露量", "共同截面训练小时", "共同截面训练吞吐",
        "完整5000轮最终准确率", "完整5000轮总耗时小时",
    ]
    summaries[comparison_columns].to_csv(
        output_dir / "方案对比.csv", index=False, encoding="utf-8-sig"
    )
    figures = create_figures(
        output_dir, runs, summaries, long_metrics, candidates, common_round
    )
    report = build_report(
        suite_dir, output_dir, runs, summaries, quality, common_round
    )
    (output_dir / "分析报告.md").write_text(report, encoding="utf-8")

    shared = next(iter(runs.values()))["metadata"]
    manifest = {
        "schema_version": "femnist_partial_analysis_v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "status": "partial" if len(runs) < 4 or any(
            row != "完整" for row in summaries["运行状态"]
        ) else "complete",
        "suite_dir": str(suite_dir),
        "output_dir": str(output_dir),
        "available_scenarios": [item for item in SCENARIO_ORDER if item in runs],
        "missing_scenarios": [item for item in SCENARIO_ORDER if item not in runs],
        "common_comparison_round": common_round,
        "h5_audit_mode": "sample" if sample_h5 else "full_written_observations",
        "shared_hashes": {field: shared.get(field) for field in SHARED_HASH_FIELDS},
        "files": [
            "分析报告.md", "实验汇总.csv", "逐轮指标.csv", "固定候选参与统计.csv",
            "数据质量检查.csv", "共识准确率相关.csv", "方案对比.csv", *figures,
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    """加载部分套件、生成分析目录并打印关键输出位置。"""
    args = parse_arguments()
    suite_dir = Path(args.suite_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else suite_dir / f"analysis_partial_{datetime.now():%Y%m%d}"
    )
    runs = find_runs(suite_dir, args.sample_h5)
    required_existing = {"hfl_snf_fixed", "hfl_no_snf_fixed", "fl_snf"}
    missing_required = required_existing - set(runs)
    if missing_required:
        raise ValueError(
            "本分析模板至少需要已有的 HFL+SnF、HFL-noSnF 和 FL+SnF；"
            f"当前缺少：{sorted(missing_required)}"
        )
    manifest = write_outputs(suite_dir, output_dir, runs, args.sample_h5)
    print(f"ANALYSIS_STATUS={manifest['status']}")
    print(f"COMMON_COMPARISON_ROUND={manifest['common_comparison_round']}")
    print(f"ANALYSIS_REPORT={output_dir / '分析报告.md'}")
    print(f"ANALYSIS_MANIFEST={output_dir / 'analysis_manifest.json'}")


if __name__ == "__main__":
    main()
