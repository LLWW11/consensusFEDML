"""SevereTest 结果完整性校验、共识重算和图表生成。"""

from __future__ import absolute_import

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from probe_metrics import (
    calculate_population_probe_metrics,
    validate_probability_tensor,
)


REQUIRED_FILES = [
    "experiment_metadata.json",
    "partition_manifest.csv",
    "test_metrics.csv",
    "class_test_metrics.csv",
    "training_schedule.jsonl",
    "probe_probabilities.npz",
    "probe_epoch_summary.csv",
]


def parse_args(argv=None):
    """解析结果目录；省略时自动选择最新完成实验。"""
    parser = argparse.ArgumentParser(description="分析 SevereTest 实验结果")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="具体实验结果目录；省略时从 result/SevereTest 选择最新目录。",
    )
    return parser.parse_args(argv)


def find_latest_result_dir(result_root):
    """从结果根目录中选择最近修改且包含元数据的实验目录。"""
    result_root = Path(result_root)
    candidates = [
        path
        for path in result_root.iterdir()
        if path.is_dir() and (path / "experiment_metadata.json").is_file()
    ] if result_root.is_dir() else []
    if not candidates:
        raise FileNotFoundError("没有找到可分析的 SevereTest 结果目录。")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_csv(path):
    """读取 CSV 并返回字典行列表。"""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _read_jsonl(path):
    """读取逐行 JSON 文件并返回对象列表。"""
    rows = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{} 第 {} 行不是合法 JSON。".format(path, line_number)
                ) from exc
    return rows


def _require_complete_files(result_dir):
    """确认训练和分析所需的全部文件均存在。"""
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (Path(result_dir) / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError("结果目录缺少文件：{}。".format(missing))


def _float(row, field_name):
    """读取 CSV 数值字段并提供带字段名的错误。"""
    try:
        return float(row[field_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("字段 {} 不是合法数值。".format(field_name)) from exc


def _validate_partition_manifest(rows):
    """校验 200 个客户端、单标签映射和各类别客户端数量。"""
    if len(rows) != 200:
        raise ValueError("partition_manifest.csv 必须包含 200 个客户端。")
    client_ids = [int(row["client_id"]) for row in rows]
    if client_ids != list(range(200)):
        raise ValueError("划分清单中的客户端编号必须严格为 0 到 199。")

    label_counts = {label: 0 for label in range(10)}
    total_train = 0
    total_test = 0
    for row in rows:
        client_id = int(row["client_id"])
        label = int(row["label"])
        train_count = int(row["train_count"])
        test_count = int(row["test_count"])
        if label != client_id % 10:
            raise ValueError("客户端 {} 的标签映射不正确。".format(client_id))
        if train_count <= 0 or test_count <= 0:
            raise ValueError("客户端 {} 存在空训练或测试分区。".format(client_id))
        label_counts[label] += 1
        total_train += train_count
        total_test += test_count
    if set(label_counts.values()) != {20}:
        raise ValueError("每个类别必须严格对应 20 个客户端。")
    if total_train != 60000 or total_test != 10000:
        raise ValueError("划分清单没有完整覆盖 MNIST 训练集和测试集。")
    return {
        "client_count": 200,
        "clients_per_class": 20,
        "train_sample_count": total_train,
        "test_sample_count": total_test,
    }


def _validate_schedule(rows, epoch_count, metadata):
    """按元数据校验固定训练名单、边缘分组和全量下发。"""
    if len(rows) != epoch_count:
        raise ValueError("训练调度行数与正式轮数不一致。")
    expected_active = [
        int(value) for value in metadata["training_client_ids"]
    ]
    expected_distributed = list(range(int(metadata["client_num_in_total"])))
    expected_groups = [
        [int(value) for value in group]
        for group in metadata.get("edge_client_groups", [])
    ]
    for epoch, row in enumerate(rows):
        if int(row["global_epoch"]) != epoch:
            raise ValueError("调度 epoch 必须从 0 连续递增。")
        if row["active_client_ids"] != expected_active:
            raise ValueError("第 {} 轮活跃客户端与元数据不一致。".format(epoch))
        if row["distributed_client_ids"] != expected_distributed:
            raise ValueError("第 {} 轮没有向全部客户端下发。".format(epoch))
        if int(row["active_client_count"]) != len(expected_active):
            raise ValueError("第 {} 轮活跃客户端数量不正确。".format(epoch))
        if int(row["distributed_client_count"]) != len(expected_distributed):
            raise ValueError("第 {} 轮下发客户端数量不正确。".format(epoch))
        if expected_groups:
            if row.get("edge_client_groups") != expected_groups:
                raise ValueError("第 {} 轮边缘分组不正确。".format(epoch))
            if row.get("edge_active_group_ids") != list(
                    range(len(expected_groups))
            ):
                raise ValueError("第 {} 轮边缘组没有全部激活。".format(epoch))
            if row.get("aggregation") != (
                    "hierarchical_sample_weighted_fedavg"
            ):
                raise ValueError("第 {} 轮没有执行分层样本加权聚合。".format(epoch))


def _load_and_validate_probe(result_dir, metadata, epoch_count):
    """读取 NPZ 并验证形状、编号、掩码、概率和探针标签。"""
    npz_path = Path(result_dir) / str(
        metadata.get("probe_npz_file", "probe_probabilities.npz")
    )
    with np.load(str(npz_path), allow_pickle=False) as payload:
        probe = {key: np.asarray(payload[key]) for key in payload.files}

    client_ids = [int(value) for value in metadata["training_client_ids"]]
    client_count = len(client_ids)
    edge_count = int(metadata.get("edge_group_count", 0))
    probe_count = int(metadata.get("probe_sample_count", 100))
    class_count = int(metadata.get("probe_class_count", 10))
    expected_client_shape = (
        epoch_count, client_count, probe_count, class_count
    )
    if probe["client_probabilities"].shape != expected_client_shape:
        raise ValueError(
            "客户端探针概率形状为 {}，期望 {}。".format(
                probe["client_probabilities"].shape, expected_client_shape
            )
        )
    if probe["cloud_probabilities"].shape != (
            epoch_count, probe_count, class_count
    ):
        raise ValueError("云端探针概率形状不正确。")
    if probe["edge_probabilities"].shape != (
            epoch_count, edge_count, probe_count, class_count
    ):
        raise ValueError("边缘探针概率形状与元数据不一致。")
    if probe["client_ids"].tolist() != client_ids:
        raise ValueError("NPZ 客户端编号与固定训练名单不一致。")
    if probe["active_client_mask"].shape != (epoch_count, client_count):
        raise ValueError("NPZ 活跃掩码形状不正确。")
    if not np.all(probe["active_client_mask"]):
        raise ValueError("NPZ 每轮固定训练客户端必须全部活跃。")
    if probe["edge_active_mask"].shape != (epoch_count, edge_count):
        raise ValueError("NPZ 边缘活跃掩码形状不正确。")
    if edge_count > 0 and not np.all(probe["edge_active_mask"]):
        raise ValueError("NPZ 每轮三个边缘组必须全部活跃。")
    if probe["global_epochs"].tolist() != list(range(epoch_count)):
        raise ValueError("NPZ global_epochs 必须从 0 连续递增。")
    if int(probe["completed_epochs"]) != epoch_count:
        raise ValueError("NPZ completed_epochs 与正式轮数不一致。")
    true_labels = probe["true_labels"].astype(np.int64)
    if true_labels.shape != (probe_count,):
        raise ValueError("探针真实标签数量与元数据不一致。")
    expected_per_class = probe_count // class_count
    if np.bincount(
            true_labels, minlength=class_count
    ).tolist() != [expected_per_class] * class_count:
        raise ValueError("固定探针必须每类恰好 10 张。")

    # 公共校验器只接受 [模型, 探针, 类别]，因此把 epoch 与模型维合并。
    validate_probability_tensor(
        probe["client_probabilities"].reshape(
            -1, probe_count, class_count
        ),
        "客户端探针概率",
    )
    if edge_count > 0:
        validate_probability_tensor(
            probe["edge_probabilities"].reshape(
                -1, probe_count, class_count
            ),
            "边缘探针概率",
        )
    validate_probability_tensor(
        probe["cloud_probabilities"], "云端探针概率"
    )
    return probe


def _recompute_round_metrics(probe, summary_rows, test_rows):
    """从 NPZ 重算客户端与边缘 A、C、S，并核对训练期摘要。"""
    epoch_count = int(probe["client_probabilities"].shape[0])
    if len(summary_rows) != epoch_count or len(test_rows) != epoch_count:
        raise ValueError("探针摘要或测试指标行数与正式轮数不一致。")
    true_labels = probe["true_labels"].astype(np.int64)
    output_rows = []
    comparisons = [
        ("candidate_agreement_mean", "agreement_mean"),
        ("candidate_certainty_mean", "certainty_mean"),
        ("candidate_effective_mean", "effective_mean"),
        ("candidate_correct_effective_mean", "correct_effective_mean"),
        ("candidate_wrong_effective_mean", "wrong_effective_mean"),
    ]
    for epoch in range(epoch_count):
        client_values = probe["client_probabilities"][epoch]
        edge_values = probe["edge_probabilities"][epoch]
        cloud_values = probe["cloud_probabilities"][epoch]
        metrics = calculate_population_probe_metrics(client_values, true_labels)
        edge_metrics = (
            calculate_population_probe_metrics(edge_values, true_labels)
            if edge_values.shape[0] > 0 else None
        )
        saved_row = summary_rows[epoch]
        for saved_field, computed_field in comparisons:
            saved_value = _float(saved_row, saved_field)
            computed_value = float(metrics[computed_field])
            if not np.isclose(
                    saved_value, computed_value, rtol=1e-8, atol=1e-10
            ):
                raise ValueError(
                    "第 {} 轮 {} 与 NPZ 重算值不一致。".format(
                        epoch, saved_field
                    )
                )
        if edge_metrics is not None:
            edge_comparisons = [
                ("edge_effective_mean", "effective_mean"),
                ("edge_correct_effective_mean", "correct_effective_mean"),
            ]
            for saved_field, computed_field in edge_comparisons:
                if not np.isclose(
                        _float(saved_row, saved_field),
                        float(edge_metrics[computed_field]),
                        rtol=1e-8,
                        atol=1e-10,
                ):
                    raise ValueError(
                        "第 {} 轮 {} 与边缘 NPZ 重算值不一致。".format(
                            epoch, saved_field
                        )
                    )
        cloud_accuracy = float(
            np.mean(np.argmax(cloud_values, axis=1) == true_labels)
        )
        saved_cloud_accuracy = _float(saved_row, "cloud_probe_accuracy")
        if not np.isclose(
                saved_cloud_accuracy, cloud_accuracy, rtol=1e-8, atol=1e-10
        ):
            raise ValueError("第 {} 轮云端探针准确率与 NPZ 不一致。".format(epoch))

        output_rows.append({
            "global_epoch": epoch,
            "agreement_a": float(metrics["agreement_mean"]),
            "certainty_c": float(metrics["certainty_mean"]),
            "effective_consensus_s": float(metrics["effective_mean"]),
            "correct_effective_consensus": float(
                metrics["correct_effective_mean"]
            ),
            "wrong_effective_consensus": float(
                metrics["wrong_effective_mean"]
            ),
            "edge_agreement_a": (
                float(edge_metrics["agreement_mean"])
                if edge_metrics is not None else math.nan
            ),
            "edge_certainty_c": (
                float(edge_metrics["certainty_mean"])
                if edge_metrics is not None else math.nan
            ),
            "edge_effective_consensus_s": (
                float(edge_metrics["effective_mean"])
                if edge_metrics is not None else math.nan
            ),
            "edge_correct_effective_consensus": (
                float(edge_metrics["correct_effective_mean"])
                if edge_metrics is not None else math.nan
            ),
            "edge_wrong_effective_consensus": (
                float(edge_metrics["wrong_effective_mean"])
                if edge_metrics is not None else math.nan
            ),
            "cloud_probe_accuracy": cloud_accuracy,
            "test_accuracy": _float(test_rows[epoch], "test_acc"),
            "test_loss": _float(test_rows[epoch], "test_loss"),
        })
    return output_rows


def _write_csv(path, rows):
    """按首行字段顺序写出分析 CSV。"""
    if not rows:
        raise ValueError("不能写出空 CSV。")
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _configure_chinese_font():
    """选择本机可用中文字体并返回字体名称。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    selected = next(
        (font_name for font_name in candidates if font_name in available),
        "DejaVu Sans",
    )
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [selected, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
    })
    return selected


def _plot_consensus(round_rows, output_path, client_count):
    """绘制全部固定训练客户端的 A、C、S 趋势图。"""
    epochs = [row["global_epoch"] + 1 for row in round_rows]
    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.plot(
        epochs,
        [row["agreement_a"] for row in round_rows],
        color="#315A8C",
        linestyle="-",
        linewidth=1.8,
        label="概率一致性 A",
    )
    axis.plot(
        epochs,
        [row["certainty_c"] for row in round_rows],
        color="#C97A2B",
        linestyle="--",
        linewidth=1.8,
        label="预测确定性 C",
    )
    axis.plot(
        epochs,
        [row["effective_consensus_s"] for row in round_rows],
        color="#5B6F3A",
        linestyle="-.",
        linewidth=2.1,
        label="有效共识 S",
    )
    axis.set_title("固定 {} 客户端探针共识指标".format(client_count))
    axis.set_xlabel("通信轮次")
    axis.set_ylabel("指标值")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(str(output_path), bbox_inches="tight")
    plt.close(figure)


def _plot_hierarchical_consensus(round_rows, output_path):
    """绘制 30 个本地模型与三个边缘模型的有效共识对比。"""
    epochs = [row["global_epoch"] + 1 for row in round_rows]
    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.plot(
        epochs,
        [row["effective_consensus_s"] for row in round_rows],
        color="#315A8C",
        linewidth=2.0,
        label="30 个客户端有效共识 S",
    )
    axis.plot(
        epochs,
        [row["edge_effective_consensus_s"] for row in round_rows],
        color="#C97A2B",
        linestyle="--",
        linewidth=2.0,
        label="3 个边缘模型有效共识 S",
    )
    axis.set_title("客户端层与边缘层有效共识")
    axis.set_xlabel("通信轮次")
    axis.set_ylabel("有效共识 S")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(str(output_path), bbox_inches="tight")
    plt.close(figure)


def _plot_accuracy(round_rows, output_path):
    """绘制完整测试集与固定探针云端准确率。"""
    epochs = [row["global_epoch"] + 1 for row in round_rows]
    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.plot(
        epochs,
        [row["test_accuracy"] for row in round_rows],
        color="#315A8C",
        linewidth=2.0,
        label="完整测试集准确率",
    )
    axis.plot(
        epochs,
        [row["cloud_probe_accuracy"] for row in round_rows],
        color="#C97A2B",
        linestyle="--",
        linewidth=1.7,
        label="云端固定探针准确率",
    )
    axis.set_title("模型准确率")
    axis.set_xlabel("通信轮次")
    axis.set_ylabel("准确率")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(str(output_path), bbox_inches="tight")
    plt.close(figure)


def _plot_class_heatmap(class_rows, epoch_count, output_path):
    """绘制 200 轮乘 10 类的测试准确率热力图。"""
    matrix = np.full((10, epoch_count), np.nan, dtype=np.float64)
    for row in class_rows:
        epoch = int(row["global_epoch"])
        label = int(row["label"])
        matrix[label, epoch] = float(row["test_acc"])
    if np.any(~np.isfinite(matrix)):
        raise ValueError("逐类准确率无法组成完整的 10×轮次矩阵。")

    figure, axis = plt.subplots(figsize=(12, 5.8))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="YlGnBu",
    )
    axis.set_title("MNIST 逐类测试准确率")
    axis.set_xlabel("通信轮次")
    axis.set_ylabel("真实标签")
    axis.set_yticks(np.arange(10))
    axis.set_yticklabels([str(label) for label in range(10)])
    tick_positions = np.linspace(0, epoch_count - 1, num=9, dtype=int)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels([str(value + 1) for value in tick_positions])
    color_bar = figure.colorbar(image, ax=axis)
    color_bar.set_label("准确率")
    figure.tight_layout()
    figure.savefig(str(output_path), bbox_inches="tight")
    plt.close(figure)


def _tail_stats(values, window=20):
    """返回最终值及尾部窗口均值和标准差。"""
    array = np.asarray(values, dtype=np.float64)
    tail = array[-min(int(window), array.size):]
    return {
        "final": float(array[-1]),
        "last_20_mean": float(np.mean(tail)),
        "last_20_std": float(np.std(tail)),
        "best": float(np.max(array)),
        "best_epoch": int(np.argmax(array)) + 1,
    }


def _build_analysis_summary(metadata, partition_audit, round_rows, class_rows):
    """汇总报告所需的关键数值、相关性、类别末轮表现与局限。"""
    s_values = [row["effective_consensus_s"] for row in round_rows]
    correct_values = [row["correct_effective_consensus"] for row in round_rows]
    wrong_values = [row["wrong_effective_consensus"] for row in round_rows]
    test_values = [row["test_accuracy"] for row in round_rows]
    cloud_values = [row["cloud_probe_accuracy"] for row in round_rows]
    if np.std(s_values) > 0.0 and np.std(test_values) > 0.0:
        s_test_correlation = float(np.corrcoef(s_values, test_values)[0, 1])
    else:
        s_test_correlation = math.nan

    last_epoch = len(round_rows) - 1
    final_class_rows = [
        row for row in class_rows if int(row["global_epoch"]) == last_epoch
    ]
    final_class_accuracy = {
        str(int(row["label"])): float(row["test_acc"])
        for row in sorted(final_class_rows, key=lambda item: int(item["label"]))
    }
    class_history = {label: [] for label in range(10)}
    for row in sorted(
            class_rows,
            key=lambda item: (int(item["global_epoch"]), int(item["label"])),
    ):
        class_history[int(row["label"])].append(float(row["test_acc"]))
    last_20_class_accuracy = {
        str(label): float(np.mean(values[-20:]))
        for label, values in class_history.items()
    }
    worst_last_20_label = min(
        last_20_class_accuracy,
        key=lambda label: last_20_class_accuracy[label],
    )
    if int(metadata.get("edge_group_count", 0)) > 0:
        topology_limitation = (
            "本实验不使用 MATLAB 或 SnF 拓扑；三个边缘组由 YAML 固定指定。"
        )
    else:
        topology_limitation = (
            "本实验为普通同步 FedAvg，不使用 MATLAB、HFL 或 SnF 拓扑。"
        )
    summary = {
        "experiment_status": metadata["status"],
        "completed_epochs": len(round_rows),
        "partition_audit": partition_audit,
        "active_train_samples": int(metadata["active_train_samples"]),
        "active_train_fraction": float(metadata["active_train_fraction"]),
        "effective_consensus_s": _tail_stats(s_values),
        "correct_effective_consensus": _tail_stats(correct_values),
        "wrong_effective_consensus": _tail_stats(wrong_values),
        "test_accuracy": _tail_stats(test_values),
        "cloud_probe_accuracy": _tail_stats(cloud_values),
        "s_test_accuracy_correlation": s_test_correlation,
        "final_class_accuracy": final_class_accuracy,
        "last_20_class_accuracy": last_20_class_accuracy,
        "worst_last_20_label": int(worst_last_20_label),
        "worst_last_20_class_accuracy": float(
            last_20_class_accuracy[worst_last_20_label]
        ),
        "limitations": [
            "仅运行一个随机种子，结果只能作为描述性证据。",
            "每轮固定 {} 个客户端参与训练，对应 {:.2%} 的 MNIST 训练样本。".format(
                len(metadata["training_client_ids"]),
                float(metadata["active_train_fraction"]),
            ),
            topology_limitation,
        ],
    }
    edge_values = [
        row["edge_effective_consensus_s"] for row in round_rows
    ]
    if np.all(np.isfinite(edge_values)):
        summary["edge_effective_consensus_s"] = _tail_stats(edge_values)
    return summary


def analyze_result(result_dir):
    """验证一个实验、重算指标并生成可审计分析数据和静态图。"""
    result_dir = Path(result_dir).resolve()
    _require_complete_files(result_dir)
    metadata = json.loads(
        (result_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    epoch_count = int(metadata["comm_round"])
    if metadata.get("status") != "complete":
        raise ValueError("只有 status=complete 的正式结果才能生成最终分析。")
    if int(metadata.get("completed_epochs", -1)) != epoch_count:
        raise ValueError("元数据中的完成轮数与配置不一致。")

    partition_rows = _read_csv(result_dir / "partition_manifest.csv")
    test_rows = _read_csv(result_dir / "test_metrics.csv")
    class_rows = _read_csv(result_dir / "class_test_metrics.csv")
    summary_rows = _read_csv(result_dir / "probe_epoch_summary.csv")
    schedule_rows = _read_jsonl(result_dir / "training_schedule.jsonl")
    partition_audit = _validate_partition_manifest(partition_rows)
    _validate_schedule(schedule_rows, epoch_count, metadata)
    if len(class_rows) != epoch_count * 10:
        raise ValueError("逐类指标必须每轮恰好包含 10 行。")
    probe = _load_and_validate_probe(result_dir, metadata, epoch_count)
    round_rows = _recompute_round_metrics(probe, summary_rows, test_rows)

    analysis_dir = result_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "round_metrics.csv", round_rows)
    _write_csv(analysis_dir / "class_metrics.csv", class_rows)
    summary = _build_analysis_summary(
        metadata, partition_audit, round_rows, class_rows
    )
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    chart_map = [
            {
                "section": "逐轮有效共识",
                "question": "A、C、S 在 200 轮内如何变化",
                "chart": "multi_series_line",
                "source": "round_metrics.csv",
            },
            {
                "section": "模型准确率",
                "question": "完整测试集与固定探针准确率是否一致",
                "chart": "two_series_line",
                "source": "round_metrics.csv",
            },
            {
                "section": "类别表现",
                "question": "10 个标签是否出现持续性性能短板",
                "chart": "heatmap",
                "source": "class_metrics.csv",
            },
        ]
    if int(metadata.get("edge_group_count", 0)) > 0:
        chart_map.append({
            "section": "分层有效共识",
            "question": "客户端层与边缘层的有效共识是否同步",
            "chart": "two_series_line",
            "source": "round_metrics.csv",
        })
    (analysis_dir / "chart_map.json").write_text(
        json.dumps(chart_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _configure_chinese_font()
    _plot_consensus(
        round_rows,
        figures_dir / "01_有效共识趋势.png",
        len(metadata["training_client_ids"]),
    )
    _plot_accuracy(round_rows, figures_dir / "02_准确率趋势.png")
    _plot_class_heatmap(
        class_rows, epoch_count, figures_dir / "03_逐类准确率热力图.png"
    )
    if int(metadata.get("edge_group_count", 0)) > 0:
        _plot_hierarchical_consensus(
            round_rows, figures_dir / "04_分层有效共识趋势.png"
        )
    return analysis_dir


def main():
    """命令行分析入口。"""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    result_dir = args.result_dir
    if result_dir is None:
        result_dir = find_latest_result_dir(project_root / "result" / "SevereTest")
    analysis_dir = analyze_result(result_dir)
    print("SEVERE_TEST_ANALYSIS_DIR={}".format(analysis_dir))


if __name__ == "__main__":
    main()
