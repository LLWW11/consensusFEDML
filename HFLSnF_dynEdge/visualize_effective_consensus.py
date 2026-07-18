"""快速计算并可视化固定多图探针的有效共识。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analyze_consensus import (
    PROBE_METADATA_FILENAME,
    configure_plot_font,
    load_probe_data,
    moving_average,
    safe_mean,
)


def parse_args() -> argparse.Namespace:
    """解析结果目录、完整尾随滑动窗口和输出目录。"""
    parser = argparse.ArgumentParser(description="快速可视化固定多图探针的有效共识。")
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="包含 probe_probabilities.npz 或旧探针 CSV 的结果目录。",
    )
    parser.add_argument("--window", type=int, default=10, help="完整尾随滑动窗口，默认 10。")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录。")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window 必须大于 0。")
    return args


def read_probabilities(path: Path) -> np.ndarray:
    """读取旧客户端 CSV，并转换为 [epoch, client, probe, class]。"""
    rows = []
    expected_client_count: Optional[int] = None
    expected_class_count: Optional[int] = None
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row_index, row in enumerate(csv.reader(file_obj), start=1):
            if not row:
                continue
            if expected_client_count is None:
                expected_client_count = len(row)
            elif len(row) != expected_client_count:
                raise ValueError("第 {} 行客户端数与前面不一致。".format(row_index))
            vectors = []
            for column_index, cell in enumerate(row, start=1):
                try:
                    vector = np.asarray(json.loads(cell), dtype=np.float64)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "第 {} 行第 {} 列不是合法概率 JSON。".format(row_index, column_index)
                    ) from exc
                if vector.ndim != 1 or vector.size < 2:
                    raise ValueError("第 {} 行存在非一维概率向量。".format(row_index))
                if expected_class_count is None:
                    expected_class_count = int(vector.size)
                elif vector.size != expected_class_count:
                    raise ValueError("旧 CSV 的类别数不一致。")
                vectors.append(vector)
            rows.append(np.stack(vectors))
    if not rows:
        raise ValueError("{} 为空文件。".format(path))
    probabilities = np.stack(rows)[:, :, None, :]
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-8):
        raise ValueError("探针 CSV 中存在负数或非有限数。")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("探针 CSV 中存在概率和不为 1 的向量。")
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """沿概率张量末维计算归一化信息熵。"""
    values = np.asarray(probabilities, dtype=np.float64)
    safe = np.where(values > 0.0, values, 1.0)
    terms = np.where(values > 0.0, values * np.log(safe), 0.0)
    return -np.sum(terms, axis=-1) / math.log(values.shape[-1])


def _normalize_labels(
    true_labels: Optional[np.ndarray], epoch_count: int, probe_count: int
) -> np.ndarray:
    """把可选真实标签统一为 [epoch, probe]，缺失位置使用 -1。"""
    if true_labels is None:
        return np.full((epoch_count, probe_count), -1, dtype=np.int64)
    labels = np.asarray(true_labels, dtype=np.int64)
    if labels.ndim == 1:
        if labels.size != probe_count:
            raise ValueError("一维 true_labels 的长度必须等于探针图数。")
        labels = np.repeat(labels[None, :], epoch_count, axis=0)
    if labels.shape != (epoch_count, probe_count):
        raise ValueError("true_labels 必须为 [probe] 或 [epoch, probe]。")
    return labels


def _majority_labels(probabilities: np.ndarray) -> np.ndarray:
    """计算每个 epoch、每张图的客户端多数标签，平票时取较小标签。"""
    epoch_count, _, probe_count, class_count = probabilities.shape
    predictions = np.argmax(probabilities, axis=-1)
    result = np.empty((epoch_count, probe_count), dtype=np.int64)
    for epoch_index in range(epoch_count):
        for probe_index in range(probe_count):
            counts = np.bincount(
                predictions[epoch_index, :, probe_index], minlength=class_count
            )
            result[epoch_index, probe_index] = int(np.argmax(counts))
    return result


def compute_metrics(
    probabilities: np.ndarray,
    window: int,
    true_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """每图先计算 A、C、S，再做 epoch 平均和完整尾随平滑。"""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 3:
        # 兼容调用者直接传入旧 [epoch, client, class] 数组。
        values = values[:, :, None, :]
    if values.ndim != 4 or values.shape[1] < 2:
        raise ValueError("probabilities 必须为 [epoch, client, probe, class] 且至少含两个客户端。")
    if window > values.shape[0]:
        raise ValueError("window 不能大于总 epoch 数。")
    if not np.all(np.isfinite(values)):
        raise ValueError("客户端概率包含非有限数。")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("客户端概率存在末维和不为 1 的向量。")

    # 轴 1 是客户端；每张图的熵和 GJSD 都在保留 probe 轴时计算。
    client_entropy = normalized_entropy(values)
    mean_client_entropy = client_entropy.mean(axis=1)
    mean_probability = values.mean(axis=1)
    mean_probability_entropy = normalized_entropy(mean_probability)
    agreement_by_probe = 1.0 - np.clip(
        mean_probability_entropy - mean_client_entropy, 0.0, 1.0
    )
    certainty_by_probe = 1.0 - mean_client_entropy
    effective_by_probe = agreement_by_probe * certainty_by_probe

    agreement = agreement_by_probe.mean(axis=1)
    certainty = certainty_by_probe.mean(axis=1)
    effective = effective_by_probe.mean(axis=1)
    labels = _normalize_labels(true_labels, values.shape[0], values.shape[2])
    valid_labels = labels >= 0
    majority = _majority_labels(values)
    correct_by_probe = np.full(effective_by_probe.shape, np.nan, dtype=np.float64)
    wrong_by_probe = np.full(effective_by_probe.shape, np.nan, dtype=np.float64)
    correct_by_probe[valid_labels] = effective_by_probe[valid_labels] * (
        majority[valid_labels] == labels[valid_labels]
    )
    wrong_by_probe[valid_labels] = effective_by_probe[valid_labels] * (
        majority[valid_labels] != labels[valid_labels]
    )
    correct = np.asarray([safe_mean(row) for row in correct_by_probe], dtype=np.float64)
    wrong = np.asarray([safe_mean(row) for row in wrong_by_probe], dtype=np.float64)
    effective_ma = moving_average(effective, window)
    # 历史最佳仅作为辅助展示，不能代替原始或平滑曲线结论。
    attainment = np.maximum.accumulate(np.nan_to_num(effective_ma, nan=0.0))
    return agreement, certainty, effective, correct, wrong, effective_ma, attainment


def write_metrics(path: Path, metrics: Tuple[np.ndarray, ...]) -> None:
    """把逐 epoch 有效共识指标写入可复核 CSV。"""
    agreement, certainty, effective, correct, wrong, effective_ma, attainment = metrics
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow([
            "epoch",
            "agreement",
            "certainty",
            "effective_consensus",
            "correct_effective_consensus",
            "wrong_effective_consensus",
            "effective_consensus_ma",
            "consensus_attainment_rate_auxiliary",
        ])
        for index in range(effective.size):
            writer.writerow([
                index + 1,
                agreement[index],
                certainty[index],
                effective[index],
                "" if np.isnan(correct[index]) else correct[index],
                "" if np.isnan(wrong[index]) else wrong[index],
                "" if np.isnan(effective_ma[index]) else effective_ma[index],
                attainment[index],
            ])


def _style_axis(axis: plt.Axes) -> None:
    """统一设置快速趋势图坐标轴样式。"""
    axis.set_ylim(0.0, 1.02)
    axis.grid(axis="y", color="#E4E7EC", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper left")


def plot_metrics(
    path: Path,
    result_name: str,
    metrics: Tuple[np.ndarray, ...],
    window: int,
    client_count: int,
    probe_count: int,
) -> None:
    """绘制指标构成、正确性拆分和完整尾随趋势。"""
    agreement, certainty, effective, correct, wrong, effective_ma, attainment = metrics
    epochs = np.arange(1, effective.size + 1)
    configure_plot_font()
    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    figure.suptitle("固定探针有效共识随训练变化", fontsize=18, fontweight="bold")
    figure.text(
        0.5,
        0.945,
        "{}；候选客户端 {}；每 epoch 探针 {}；完整尾随窗口 {}".format(
            result_name, client_count, probe_count, window
        ),
        ha="center",
        color="#667085",
        fontsize=10,
    )
    axes[0].plot(epochs, agreement, color="#667085", linestyle="--", label="一致性 A")
    axes[0].plot(epochs, certainty, color="#2F6BFF", label="确定性 C")
    axes[0].plot(epochs, effective, color="#D99A21", label="纯有效共识 S=A×C")
    axes[0].set_title("每图先计算，再做 epoch 平均", loc="left", fontweight="bold")
    axes[0].set_ylabel("得分")

    axes[1].plot(epochs, effective, color="#667085", alpha=0.25, label="纯有效共识")
    if np.any(np.isfinite(correct)):
        axes[1].plot(epochs, correct, color="#039855", label="正确有效共识")
        axes[1].plot(epochs, wrong, color="#D92D20", label="错误有效共识")
    axes[1].set_title("正确性护栏", loc="left", fontweight="bold")
    axes[1].set_ylabel("得分")

    axes[2].plot(epochs, effective, color="#2F6BFF", linewidth=0.8, alpha=0.2, label="瞬时有效共识")
    axes[2].plot(epochs, effective_ma, color="#2F6BFF", linewidth=2.0, label="{} epoch 尾随均值".format(window))
    axes[2].plot(epochs, attainment, color="#D99A21", linewidth=2.0, label="历史最佳（辅助）")
    axes[2].set_title("平滑趋势与辅助历史最佳", loc="left", fontweight="bold")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("得分")
    for axis in axes:
        _style_axis(axis)
    figure.tight_layout(rect=(0.04, 0.04, 0.98, 0.92))
    figure.savefig(str(path), dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    """优先读取 NPZ，缺失时回退旧 CSV，并输出快速分析结果。"""
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or result_dir / "effective_consensus_quicklook").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = result_dir / PROBE_METADATA_FILENAME
    probe_data = load_probe_data(
        result_dir, metadata_path if metadata_path.is_file() else None
    )
    probabilities = probe_data.client_probabilities
    metrics = compute_metrics(probabilities, args.window, probe_data.true_labels)
    metrics_path = output_dir / "effective_consensus_metrics.csv"
    figure_path = output_dir / "effective_consensus.png"
    write_metrics(metrics_path, metrics)
    plot_metrics(
        figure_path,
        result_dir.name,
        metrics,
        args.window,
        probabilities.shape[1],
        probabilities.shape[2],
    )
    effective, correct, wrong, effective_ma, attainment = (
        metrics[2], metrics[3], metrics[4], metrics[5], metrics[6]
    )
    print("数据格式：{}；数据形状：{}".format(probe_data.source_format, probabilities.shape))
    print("前 10 epoch 纯有效共识均值：{:.4f}".format(safe_mean(effective[:10])))
    print("后 10 epoch 纯有效共识均值：{:.4f}".format(safe_mean(effective[-10:])))
    if np.any(np.isfinite(correct)):
        print("全程正确/错误有效共识：{:.4f} / {:.4f}".format(safe_mean(correct), safe_mean(wrong)))
    print("最高 {} epoch 尾随均值：{:.4f}".format(args.window, safe_mean([np.nanmax(effective_ma)])))
    print("最终历史最佳辅助值：{:.4f}".format(attainment[-1]))
    print("指标 CSV：{}".format(metrics_path))
    print("趋势图：{}".format(figure_path))


if __name__ == "__main__":
    main()
