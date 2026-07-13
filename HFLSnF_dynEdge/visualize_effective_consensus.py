"""快速计算并可视化客户端概率探针的有效共识指标。"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_consensus import configure_plot_font


def parse_args():
    """解析结果目录、滑动窗口和输出目录参数。"""
    parser = argparse.ArgumentParser(description="可视化有效共识及其单调达成率。")
    parser.add_argument("--result-dir", type=Path, required=True, help="包含探针 CSV 的结果目录。")
    parser.add_argument("--window", type=int, default=10, help="滑动平均窗口，默认 10 轮。")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录。")
    return parser.parse_args()


def read_probabilities(path):
    """读取客户端探针 CSV，并返回形状为 轮次×客户端×类别 的数组。"""
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row_index, row in enumerate(csv.reader(file_obj), start=1):
            if not row:
                continue
            vectors = [np.asarray(json.loads(cell), dtype=np.float64) for cell in row]
            if any(vector.shape != (10,) for vector in vectors):
                raise ValueError("第 {} 行存在非 10 维概率向量。".format(row_index))
            rows.append(np.stack(vectors))

    probabilities = np.stack(rows)
    if not np.isfinite(probabilities).all():
        raise ValueError("探针 CSV 中存在非有限数。")
    if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("探针 CSV 中存在概率和不为 1 的向量。")
    return probabilities


def normalized_entropy(probabilities):
    """沿最后一维计算归一化信息熵。"""
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=-1) / math.log(probabilities.shape[-1])


def moving_average(values, window):
    """计算完整窗口滑动均值，窗口未填满的位置保留为空值。"""
    if window <= 0 or window > values.size:
        raise ValueError("window 必须位于 1 到总轮数之间。")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def compute_metrics(probabilities, window):
    """计算概率一致性、整体确定性、有效共识和单调达成率。"""
    client_entropy = normalized_entropy(probabilities)
    mean_client_entropy = client_entropy.mean(axis=1)
    mean_probability = probabilities.mean(axis=1)
    mean_probability_entropy = normalized_entropy(mean_probability)

    # 一致性单独看会把共同均匀输出误判为高共识，因此还要乘以确定性。
    divergence = np.clip(mean_probability_entropy - mean_client_entropy, 0.0, 1.0)
    agreement = 1.0 - divergence
    certainty = 1.0 - mean_client_entropy
    effective = agreement * certainty
    effective_ma = moving_average(effective, window)
    attainment = np.maximum.accumulate(np.nan_to_num(effective_ma, nan=0.0))
    return agreement, certainty, effective, effective_ma, attainment


def write_metrics(path, metrics):
    """把逐轮共识指标写入便于复核的 CSV。"""
    agreement, certainty, effective, effective_ma, attainment = metrics
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            "round", "agreement", "certainty", "effective_consensus",
            "effective_consensus_ma", "consensus_attainment_rate",
        ])
        for index in range(effective.size):
            writer.writerow([
                index + 1,
                agreement[index],
                certainty[index],
                effective[index],
                "" if np.isnan(effective_ma[index]) else effective_ma[index],
                attainment[index],
            ])


def plot_metrics(path, result_name, metrics, window, client_count):
    """绘制指标构成和单调共识达成率两张纵向趋势图。"""
    agreement, certainty, effective, effective_ma, attainment = metrics
    rounds = np.arange(1, effective.size + 1)
    blue = "#2F6BFF"
    gold = "#D99A21"
    neutral = "#667085"

    configure_plot_font()
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    figure.suptitle("有效共识指标随训练轮次变化", fontsize=18, fontweight="bold")
    figure.text(
        0.5,
        0.94,
        "{}；每轮客户端数={}；滑动窗口={} 轮".format(result_name, client_count, window),
        ha="center",
        color=neutral,
        fontsize=10,
    )

    axes[0].plot(rounds, agreement, color=neutral, linewidth=1.4, linestyle="--", label="概率一致性 A")
    axes[0].plot(rounds, certainty, color=blue, linewidth=1.6, label="整体确定性 C")
    axes[0].plot(rounds, effective, color=gold, linewidth=1.8, label="有效共识 S=A×C")
    axes[0].set_title("一致性、确定性与有效共识", loc="left", fontweight="bold")
    axes[0].set_ylabel("得分")

    axes[1].plot(rounds, effective, color=blue, linewidth=0.9, alpha=0.25, label="瞬时有效共识")
    axes[1].plot(rounds, effective_ma, color=blue, linewidth=2.0, label="{} 轮滑动均值".format(window))
    axes[1].plot(rounds, attainment, color=gold, linewidth=2.2, label="单调共识达成率")
    axes[1].set_title("平滑共识与历史最佳持续水平", loc="left", fontweight="bold")
    axes[1].set_xlabel("训练轮次")
    axes[1].set_ylabel("得分")

    for axis in axes:
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="y", color="#E4E7EC", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, ncol=3, loc="upper left")

    figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.92))
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    """读取现有结果，生成指标 CSV、趋势图和终端摘要。"""
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or result_dir / "effective_consensus_quicklook").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = read_probabilities(result_dir / "probe_client_pre.csv")
    metrics = compute_metrics(probabilities, args.window)
    metrics_path = output_dir / "effective_consensus_metrics.csv"
    figure_path = output_dir / "effective_consensus.png"
    write_metrics(metrics_path, metrics)
    plot_metrics(figure_path, result_dir.name, metrics, args.window, probabilities.shape[1])

    effective = metrics[2]
    effective_ma = metrics[3]
    attainment = metrics[4]
    print("数据形状：{}".format(probabilities.shape))
    print("前 10 轮有效共识均值：{:.4f}".format(effective[:10].mean()))
    print("后 10 轮有效共识均值：{:.4f}".format(effective[-10:].mean()))
    print("最高 {} 轮滑动均值：{:.4f}".format(args.window, np.nanmax(effective_ma)))
    print("最终单调共识达成率：{:.4f}".format(attainment[-1]))
    print("指标 CSV：{}".format(metrics_path))
    print("趋势图：{}".format(figure_path))


if __name__ == "__main__":
    main()
