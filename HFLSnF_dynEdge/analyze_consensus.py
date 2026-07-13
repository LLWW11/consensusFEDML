"""计算层次联邦学习概率探针的逐轮共识指标并生成可视化结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# 分析脚本可能在无桌面的服务器上运行，因此固定使用非交互式后端。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
from scipy.io import loadmat


PROBE_FILENAMES = {
    "client": "probe_client_pre.csv",
    "edge": "probe_edge_post.csv",
    "cloud": "probe_cloud_post.csv",
}

METRIC_DEFINITIONS = [
    {
        "metric": "client_vote_consensus_ratio",
        "name_zh": "全客户端标签共识比例",
        "formula": "max_y sum_i I[argmax(p_i)=y] / M",
        "range": "[0,1]",
        "direction": "越大越一致",
        "meaning": "37 个客户端中支持同一多数标签的最大比例，只比较最终类别。",
    },
    {
        "metric": "client_label_consensus_score",
        "name_zh": "全客户端标签熵共识得分",
        "formula": "1 - H(q) / ln(K)",
        "range": "[0,1]",
        "direction": "越大越一致",
        "meaning": "根据客户端投票标签分布 q 衡量意见是否集中，可区分反对意见集中或分散。",
    },
    {
        "metric": "client_probability_consensus_score",
        "name_zh": "全客户端概率共识得分",
        "formula": "1 - [H(mean(p_i)) - mean(H(p_i))] / ln(K)",
        "range": "[0,1]",
        "direction": "越大越一致",
        "meaning": "比较完整的 10 维概率分布；即使最终标签相同，置信结构不同也会降低该得分。",
    },
    {
        "metric": "within_edge_probability_consensus_score",
        "name_zh": "域内概率共识得分",
        "formula": "1 - weighted_mean_e(GJSD({p_i | i in C_e}))",
        "range": "[0,1]",
        "direction": "越大越一致",
        "meaning": "先在每个动态边缘组内计算客户端概率分歧，再按组内客户端数加权。",
    },
    {
        "metric": "edge_probability_consensus_score",
        "name_zh": "边缘模型间概率共识得分",
        "formula": "1 - GJSD({p_e})",
        "range": "[0,1]",
        "direction": "越大越一致",
        "meaning": "衡量当前轮各有效边缘聚合模型对同一探针的输出是否一致；少于两个边缘时不定义。",
    },
    {
        "metric": "client_cloud_alignment_score",
        "name_zh": "客户端到云概率对齐得分",
        "formula": "1 - mean_i(JS(p_i, p_g) / ln(2))",
        "range": "[0,1]",
        "direction": "越大越对齐",
        "meaning": "衡量每个客户端概率分布与云模型概率分布的平均接近程度。",
    },
    {
        "metric": "edge_cloud_alignment_score",
        "name_zh": "边缘到云概率对齐得分",
        "formula": "1 - mean_e(JS(p_e, p_g) / ln(2))",
        "range": "[0,1]",
        "direction": "越大越对齐",
        "meaning": "衡量有效边缘模型与云模型之间的概率输出接近程度。",
    },
    {
        "metric": "client_cloud_label_agreement",
        "name_zh": "客户端与云标签一致比例",
        "formula": "sum_i I[argmax(p_i)=argmax(p_g)] / M",
        "range": "[0,1]",
        "direction": "越大越一致",
        "meaning": "37 个客户端中最终预测标签与云模型标签相同的比例。",
    },
    {
        "metric": "cloud_confidence",
        "name_zh": "云模型置信度",
        "formula": "max_y p_g(y)",
        "range": "[0,1]",
        "direction": "越大表示云输出更集中",
        "meaning": "云模型对其首选类别给出的概率，不等价于多客户端共识，也不保证预测正确。",
    },
    {
        "metric": "cloud_margin",
        "name_zh": "云模型概率间隔",
        "formula": "top1(p_g) - top2(p_g)",
        "range": "[0,1]",
        "direction": "越大表示决策边界更明确",
        "meaning": "云模型最高概率与次高概率之差，用于判断当前决策是否犹豫。",
    },
    {
        "metric": "correct_soft_consensus",
        "name_zh": "正确软共识事件",
        "formula": "I[A >= tau_soft and cloud_prediction = true_label]",
        "range": "{0,1}",
        "direction": "平均值越大越好",
        "meaning": "达到软共识且云端决策正确；需要 probe_meta.csv 中的 true_label。",
    },
    {
        "metric": "wrong_soft_consensus",
        "name_zh": "错误软共识事件",
        "formula": "I[A >= tau_soft and cloud_prediction != true_label]",
        "range": "{0,1}",
        "direction": "平均值越小越好",
        "meaning": "达到软共识但云端决策错误；这是防止高一致性掩盖集体错误的关键护栏。",
    },
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回分析配置。"""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="计算客户端、边缘和云概率探针的逐轮共识指标并绘图。"
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="包含三份 probe CSV 的结果目录；默认选择 result/ 下最近修改的有效目录。",
    )
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=script_dir / "matlab" / "my_data_150.mat",
        help="动态分组 .mat 文件路径。",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="可选探针元数据 CSV，需包含 true_label 列。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认使用 <result-dir>/consensus_analysis。",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="趋势图滑动平均窗口，默认 5 轮。",
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=2.0 / 3.0,
        help="软共识标签比例阈值，默认 2/3。",
    )
    parser.add_argument(
        "--strong-threshold",
        type=float,
        default=0.8,
        help="强共识标签比例阈值，默认 0.8。",
    )
    args = parser.parse_args()
    if args.smooth_window <= 0:
        parser.error("--smooth-window 必须大于 0。")
    if not 0.0 <= args.soft_threshold <= 1.0:
        parser.error("--soft-threshold 必须位于 [0,1]。")
    if not 0.0 <= args.strong_threshold <= 1.0:
        parser.error("--strong-threshold 必须位于 [0,1]。")
    return args


def find_latest_result_dir(result_root: Path) -> Path:
    """在 result 根目录中寻找最近修改且包含三份探针文件的运行目录。"""
    candidates = []
    if result_root.exists():
        for path in result_root.iterdir():
            if not path.is_dir():
                continue
            if all((path / filename).exists() for filename in PROBE_FILENAMES.values()):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("未在 {} 下找到包含三份探针 CSV 的结果目录。".format(result_root))
    return max(candidates, key=lambda item: item.stat().st_mtime)


def resolve_input_path(path: Path, base_dir: Path) -> Path:
    """将输入路径解析为绝对路径，优先以脚本目录为相对路径基准。"""
    if path.is_absolute():
        return path
    script_candidate = (base_dir / path).resolve()
    if script_candidate.exists():
        return script_candidate
    return path.resolve()


def read_probability_csv(path: Path) -> List[List[Optional[np.ndarray]]]:
    """读取概率探针 CSV，并校验非空单元格均为合法的 10 维概率向量。"""
    rows: List[List[Optional[np.ndarray]]] = []
    expected_width: Optional[int] = None
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row_index, raw_row in enumerate(csv.reader(file_obj), start=1):
            if expected_width is None:
                expected_width = len(raw_row)
            elif len(raw_row) != expected_width:
                raise ValueError(
                    "{} 第 {} 行列数为 {}，预期为 {}。".format(
                        path, row_index, len(raw_row), expected_width
                    )
                )

            parsed_row: List[Optional[np.ndarray]] = []
            for column_index, cell in enumerate(raw_row, start=1):
                if not cell.strip():
                    parsed_row.append(None)
                    continue
                try:
                    vector = np.asarray(json.loads(cell), dtype=np.float64)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "{} 第 {} 行第 {} 列不是合法 JSON 概率向量。".format(
                            path, row_index, column_index
                        )
                    ) from exc
                if vector.shape != (10,):
                    raise ValueError(
                        "{} 第 {} 行第 {} 列的向量形状为 {}，预期为 (10,)。".format(
                            path, row_index, column_index, vector.shape
                        )
                    )
                if not np.all(np.isfinite(vector)) or np.any(vector < -1e-8):
                    raise ValueError(
                        "{} 第 {} 行第 {} 列包含负值或非有限数。".format(
                            path, row_index, column_index
                        )
                    )
                probability_sum = float(vector.sum())
                if not math.isclose(probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(
                        "{} 第 {} 行第 {} 列概率和为 {:.8f}，不接近 1。".format(
                            path, row_index, column_index, probability_sum
                        )
                    )
                # 归一化可吸收浮点序列化造成的极小误差。
                parsed_row.append(vector / probability_sum)
            rows.append(parsed_row)
    if not rows:
        raise ValueError("{} 为空文件。".format(path))
    return rows


def read_metric_series(path: Path, round_count: int) -> np.ndarray:
    """读取每行一个浮点数的训练指标文件，并对齐到探针轮数。"""
    values = np.full(round_count, np.nan, dtype=np.float64)
    if not path.exists():
        return values
    raw_values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_values.append(float(line.strip()))
    usable_count = min(round_count, len(raw_values))
    values[:usable_count] = raw_values[:usable_count]
    return values


def read_true_labels(path: Optional[Path], round_count: int) -> List[Optional[int]]:
    """从可选元数据文件读取真实标签；缺失时返回全空标签列表。"""
    labels: List[Optional[int]] = [None for _ in range(round_count)]
    if path is None or not path.exists():
        return labels
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None or "true_label" not in reader.fieldnames:
            raise ValueError("{} 必须包含 true_label 列。".format(path))
        for row_index, row in enumerate(reader):
            if row_index >= round_count:
                break
            raw_label = (row.get("true_label") or "").strip()
            if raw_label:
                label = int(raw_label)
                if label < 0 or label > 9:
                    raise ValueError("true_label 必须位于 0 到 9。")
                labels[row_index] = label
    return labels


def normalized_entropy(probability: np.ndarray) -> float:
    """计算以类别数对数归一化的信息熵，结果位于 [0,1]。"""
    positive = probability[probability > 0.0]
    if positive.size == 0:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive)))
    return entropy / math.log(probability.size)


def generalized_js_divergence(probabilities: np.ndarray) -> float:
    """计算多分布广义 Jensen-Shannon 分歧度，并按 ln(类别数) 归一化。"""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        return float("nan")
    if probabilities.shape[0] == 1:
        return 0.0
    mean_probability = np.mean(probabilities, axis=0)
    mean_entropy = normalized_entropy(mean_probability)
    component_entropy = float(np.mean([normalized_entropy(item) for item in probabilities]))
    # 两项均已除以 ln(K)，因此差值仍是归一化后的广义 JSD。
    return max(0.0, min(1.0, mean_entropy - component_entropy))


def pairwise_js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个概率分布的 Jensen-Shannon 分歧度，并按 ln(2) 归一化。"""
    midpoint = (left + right) / 2.0

    def kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
        """计算离散概率分布 source 到 target 的 KL 散度。"""
        mask = source > 0.0
        return float(np.sum(source[mask] * np.log(source[mask] / target[mask])))

    divergence = 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)
    return max(0.0, min(1.0, divergence / math.log(2.0)))


def vote_metrics(probabilities: np.ndarray) -> Tuple[float, float, float, int]:
    """计算标签多数比例、归一化投票熵、标签共识得分和多数标签。"""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        return float("nan"), float("nan"), float("nan"), -1
    labels = np.argmax(probabilities, axis=1)
    counts = np.bincount(labels, minlength=probabilities.shape[1]).astype(np.float64)
    distribution = counts / counts.sum()
    vote_entropy = normalized_entropy(distribution)
    majority_label = int(np.argmax(counts))
    return float(counts.max() / counts.sum()), vote_entropy, 1.0 - vote_entropy, majority_label


def probability_margin(probability: np.ndarray) -> float:
    """计算概率向量中最高概率与次高概率的差值。"""
    ordered = np.sort(probability)
    return float(ordered[-1] - ordered[-2])


def safe_mean(values: Sequence[float]) -> float:
    """忽略非有限数计算均值；没有有效值时返回 NaN。"""
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def load_dynamic_groups(mat_path: Path, round_count: int, client_count: int) -> List[List[List[int]]]:
    """读取动态分组文件，并还原每轮各边缘组对应的客户端编号。"""
    data = loadmat(mat_path)
    if "group_num" not in data or "client_num" not in data:
        raise KeyError("{} 必须包含 group_num 和 client_num。".format(mat_path))
    group_num = np.asarray(data["group_num"]).reshape(-1)
    client_num = np.asarray(data["client_num"])
    if group_num.shape[0] < round_count or client_num.shape[0] < round_count:
        raise ValueError("动态分组文件轮数少于探针 CSV 轮数。")

    groups_by_round: List[List[List[int]]] = []
    for round_index in range(round_count):
        enabled_group_count = int(group_num[round_index])
        counts = np.asarray(client_num[round_index]).astype(int).reshape(-1)
        if enabled_group_count > counts.size:
            raise ValueError("第 {} 轮启用组数超过 client_num 列数。".format(round_index + 1))
        groups: List[List[int]] = []
        next_client_index = 0
        for count in counts[:enabled_group_count]:
            if count < 0:
                raise ValueError("第 {} 轮存在负客户端数量。".format(round_index + 1))
            group_clients = list(range(next_client_index, next_client_index + int(count)))
            next_client_index += int(count)
            if group_clients:
                groups.append(group_clients)
        if next_client_index > client_count:
            raise ValueError("第 {} 轮动态分组客户端数量超过 {}。".format(round_index + 1, client_count))
        groups_by_round.append(groups)
    return groups_by_round


def compute_within_edge_metrics(
    client_probabilities: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> Tuple[float, float, float]:
    """按组内客户端数量加权计算域内标签比例和概率分歧。"""
    total_client_count = sum(len(group) for group in groups)
    if total_client_count == 0:
        return float("nan"), float("nan"), float("nan")
    weighted_vote_ratio = 0.0
    weighted_divergence = 0.0
    for group in groups:
        group_probabilities = client_probabilities[np.asarray(group, dtype=int)]
        vote_ratio, _, _, _ = vote_metrics(group_probabilities)
        divergence = generalized_js_divergence(group_probabilities)
        group_weight = len(group) / total_client_count
        weighted_vote_ratio += group_weight * vote_ratio
        weighted_divergence += group_weight * divergence
    return weighted_vote_ratio, weighted_divergence, 1.0 - weighted_divergence


def compute_round_metrics(
    client_rows: List[List[Optional[np.ndarray]]],
    edge_rows: List[List[Optional[np.ndarray]]],
    cloud_rows: List[List[Optional[np.ndarray]]],
    groups_by_round: List[List[List[int]]],
    true_labels: Sequence[Optional[int]],
    quality_series: Dict[str, np.ndarray],
    soft_threshold: float,
    strong_threshold: float,
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    """逐轮计算全部共识指标，并返回客户端概率、标签对齐矩阵和参与矩阵。"""
    round_count = len(client_rows)
    client_count = len(client_rows[0])
    metrics: List[Dict[str, object]] = []
    cloud_alignment_matrix = np.full((client_count, round_count), np.nan, dtype=np.float64)
    cloud_agreement_matrix = np.zeros((client_count, round_count), dtype=np.int8)
    active_matrix = np.zeros((client_count, round_count), dtype=np.int8)

    for round_index in range(round_count):
        if any(item is None for item in client_rows[round_index]):
            raise ValueError("客户端探针第 {} 轮存在空单元格。".format(round_index + 1))
        if len(cloud_rows[round_index]) != 1 or cloud_rows[round_index][0] is None:
            raise ValueError("云探针第 {} 轮必须恰好包含一个概率向量。".format(round_index + 1))

        client_probabilities = np.vstack(client_rows[round_index])
        edge_probabilities_list = [item for item in edge_rows[round_index] if item is not None]
        edge_probabilities = (
            np.vstack(edge_probabilities_list)
            if edge_probabilities_list
            else np.empty((0, client_probabilities.shape[1]), dtype=np.float64)
        )
        cloud_probability = cloud_rows[round_index][0]
        groups = groups_by_round[round_index]
        active_client_indexes = sorted({client for group in groups for client in group})
        active_probabilities = (
            client_probabilities[np.asarray(active_client_indexes, dtype=int)]
            if active_client_indexes
            else np.empty((0, client_probabilities.shape[1]), dtype=np.float64)
        )

        client_vote_ratio, client_vote_entropy, client_label_score, majority_label = vote_metrics(
            client_probabilities
        )
        active_vote_ratio, active_vote_entropy, active_label_score, _ = vote_metrics(active_probabilities)
        within_vote_ratio, within_divergence, within_score = compute_within_edge_metrics(
            client_probabilities, groups
        )

        client_divergence = generalized_js_divergence(client_probabilities)
        active_divergence = generalized_js_divergence(active_probabilities)
        client_cloud_divergences = np.asarray(
            [pairwise_js_divergence(item, cloud_probability) for item in client_probabilities],
            dtype=np.float64,
        )
        client_cloud_divergence = safe_mean(client_cloud_divergences)
        cloud_prediction = int(np.argmax(cloud_probability))
        client_predictions = np.argmax(client_probabilities, axis=1)
        client_cloud_label_agreement = float(np.mean(client_predictions == cloud_prediction))
        majority_cloud_label_match = int(majority_label == cloud_prediction)

        # 至少两个边缘输出时，边缘间共识才有比较意义。
        if edge_probabilities.shape[0] >= 2:
            edge_vote_ratio, edge_vote_entropy, edge_label_score, _ = vote_metrics(edge_probabilities)
            edge_divergence = generalized_js_divergence(edge_probabilities)
        else:
            edge_vote_ratio = float("nan")
            edge_vote_entropy = float("nan")
            edge_label_score = float("nan")
            edge_divergence = float("nan")
        edge_cloud_divergence = safe_mean(
            [pairwise_js_divergence(item, cloud_probability) for item in edge_probabilities]
        )
        edge_cloud_label_agreement = safe_mean(
            [float(np.argmax(item) == cloud_prediction) for item in edge_probabilities]
        )

        true_label = true_labels[round_index]
        cloud_correct = None if true_label is None else int(cloud_prediction == true_label)
        soft_consensus = int(client_vote_ratio >= soft_threshold)
        strong_consensus = int(client_vote_ratio >= strong_threshold)
        correct_soft_consensus = None
        wrong_soft_consensus = None
        correct_strong_consensus = None
        wrong_strong_consensus = None
        if cloud_correct is not None:
            correct_soft_consensus = int(bool(soft_consensus) and bool(cloud_correct))
            wrong_soft_consensus = int(bool(soft_consensus) and not bool(cloud_correct))
            correct_strong_consensus = int(bool(strong_consensus) and bool(cloud_correct))
            wrong_strong_consensus = int(bool(strong_consensus) and not bool(cloud_correct))

        cloud_alignment_matrix[:, round_index] = 1.0 - client_cloud_divergences
        cloud_agreement_matrix[:, round_index] = (client_predictions == cloud_prediction).astype(np.int8)
        if active_client_indexes:
            active_matrix[np.asarray(active_client_indexes, dtype=int), round_index] = 1

        row: Dict[str, object] = {
            "epoch": round_index + 1,
            "client_count": client_count,
            "active_client_count": len(active_client_indexes),
            "active_edge_count": edge_probabilities.shape[0],
            "client_vote_consensus_ratio": client_vote_ratio,
            "active_client_vote_consensus_ratio": active_vote_ratio,
            "within_edge_vote_consensus_ratio": within_vote_ratio,
            "edge_vote_consensus_ratio": edge_vote_ratio,
            "client_vote_entropy_norm": client_vote_entropy,
            "client_label_consensus_score": client_label_score,
            "active_client_vote_entropy_norm": active_vote_entropy,
            "active_client_label_consensus_score": active_label_score,
            "edge_vote_entropy_norm": edge_vote_entropy,
            "edge_label_consensus_score": edge_label_score,
            "client_probability_divergence_gjs": client_divergence,
            "client_probability_consensus_score": 1.0 - client_divergence,
            "active_client_probability_divergence_gjs": active_divergence,
            "active_client_probability_consensus_score": (
                float("nan") if not math.isfinite(active_divergence) else 1.0 - active_divergence
            ),
            "within_edge_probability_divergence_gjs": within_divergence,
            "within_edge_probability_consensus_score": within_score,
            "edge_probability_divergence_gjs": edge_divergence,
            "edge_probability_consensus_score": (
                float("nan") if not math.isfinite(edge_divergence) else 1.0 - edge_divergence
            ),
            "client_cloud_js_divergence": client_cloud_divergence,
            "client_cloud_alignment_score": 1.0 - client_cloud_divergence,
            "edge_cloud_js_divergence": edge_cloud_divergence,
            "edge_cloud_alignment_score": (
                float("nan")
                if not math.isfinite(edge_cloud_divergence)
                else 1.0 - edge_cloud_divergence
            ),
            "client_cloud_label_agreement": client_cloud_label_agreement,
            "edge_cloud_label_agreement": edge_cloud_label_agreement,
            "majority_cloud_label_match": majority_cloud_label_match,
            "client_mean_confidence": float(np.mean(np.max(client_probabilities, axis=1))),
            "cloud_confidence": float(np.max(cloud_probability)),
            "cloud_entropy_norm": normalized_entropy(cloud_probability),
            "cloud_margin": probability_margin(cloud_probability),
            "majority_label": majority_label,
            "cloud_predicted_label": cloud_prediction,
            "true_label": true_label,
            "soft_consensus": soft_consensus,
            "strong_consensus": strong_consensus,
            "cloud_correct": cloud_correct,
            "correct_soft_consensus": correct_soft_consensus,
            "wrong_soft_consensus": wrong_soft_consensus,
            "correct_strong_consensus": correct_strong_consensus,
            "wrong_strong_consensus": wrong_strong_consensus,
        }
        for metric_name, values in quality_series.items():
            row[metric_name] = float(values[round_index])
        metrics.append(row)

    return metrics, cloud_alignment_matrix, cloud_agreement_matrix, active_matrix


def csv_value(value: object) -> object:
    """将 NaN 和 None 转换为空单元格，其他值保持可读精度。"""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return "{:.10f}".format(float(value))
    return value


def write_metrics_csv(path: Path, metrics: Sequence[Dict[str, object]]) -> None:
    """将逐轮共识指标写入 CSV。"""
    fieldnames = list(metrics[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def write_metric_definitions(path: Path) -> None:
    """将指标名称、公式和中文解释写入独立 CSV。"""
    fieldnames = ["metric", "name_zh", "formula", "range", "direction", "meaning"]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(METRIC_DEFINITIONS)


def metric_array(metrics: Sequence[Dict[str, object]], metric_name: str) -> np.ndarray:
    """从逐轮指标字典中抽取指定数值列，并将空值转换为 NaN。"""
    values = []
    for row in metrics:
        value = row.get(metric_name)
        values.append(float("nan") if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def write_summary_csv(path: Path, metrics: Sequence[Dict[str, object]]) -> None:
    """汇总关键指标的全程统计量和前中后阶段均值。"""
    definition_map = {item["metric"]: item for item in METRIC_DEFINITIONS}
    summary_metric_names = [item["metric"] for item in METRIC_DEFINITIONS]
    summary_metric_names.extend(
        [
            "active_client_vote_consensus_ratio",
            "within_edge_vote_consensus_ratio",
            "edge_vote_consensus_ratio",
            "active_client_probability_consensus_score",
            "test_acc",
            "test_loss",
        ]
    )
    round_groups = np.array_split(np.arange(len(metrics)), 3)
    fieldnames = [
        "metric",
        "name_zh",
        "valid_rounds",
        "mean",
        "median",
        "min",
        "max",
        "first_third_mean",
        "middle_third_mean",
        "last_third_mean",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in summary_metric_names:
            if metric_name not in metrics[0]:
                continue
            values = metric_array(metrics, metric_name)
            valid_values = values[np.isfinite(values)]
            if valid_values.size == 0:
                continue
            definition = definition_map.get(metric_name, {})
            row = {
                "metric": metric_name,
                "name_zh": definition.get("name_zh", metric_name),
                "valid_rounds": int(valid_values.size),
                "mean": float(np.mean(valid_values)),
                "median": float(np.median(valid_values)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "first_third_mean": safe_mean(values[round_groups[0]]),
                "middle_third_mean": safe_mean(values[round_groups[1]]),
                "last_third_mean": safe_mean(values[round_groups[2]]),
            }
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def write_client_summary_csv(
    path: Path,
    cloud_alignment_matrix: np.ndarray,
    cloud_agreement_matrix: np.ndarray,
    active_matrix: np.ndarray,
) -> None:
    """汇总每个客户端在全部、活跃和非活跃轮次中的云对齐表现。"""
    fieldnames = [
        "client_id",
        "active_rounds",
        "active_rate",
        "mean_probability_alignment_to_cloud",
        "median_probability_alignment_to_cloud",
        "cloud_label_agreement_rate",
        "mean_alignment_when_active",
        "mean_alignment_when_inactive",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for client_id in range(cloud_alignment_matrix.shape[0]):
            alignment = cloud_alignment_matrix[client_id]
            agreement = cloud_agreement_matrix[client_id].astype(np.float64)
            active_mask = active_matrix[client_id].astype(bool)
            inactive_mask = ~active_mask
            row = {
                "client_id": client_id,
                "active_rounds": int(np.sum(active_mask)),
                "active_rate": float(np.mean(active_mask)),
                "mean_probability_alignment_to_cloud": safe_mean(alignment),
                "median_probability_alignment_to_cloud": float(np.nanmedian(alignment)),
                "cloud_label_agreement_rate": safe_mean(agreement),
                "mean_alignment_when_active": safe_mean(alignment[active_mask]),
                "mean_alignment_when_inactive": safe_mean(alignment[inactive_mask]),
            }
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def configure_plot_font() -> str:
    """选择可用中文字体，并配置 Matplotlib 的通用显示样式。"""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    selected = next((name for name in candidates if name in available_fonts), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#475467",
            "axes.labelcolor": "#344054",
            "xtick.color": "#667085",
            "ytick.color": "#667085",
            "text.color": "#101828",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )
    return selected


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """计算忽略 NaN 的居中滑动平均。"""
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64)
    valid = np.isfinite(values)
    value_sum = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    valid_count = np.convolve(valid.astype(np.float64), kernel, mode="same")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(value_sum, valid_count, out=result, where=valid_count > 0.0)
    return result


def style_axis(axis: plt.Axes, y_label: str, y_limit: Optional[Tuple[float, float]] = None) -> None:
    """统一设置趋势图坐标轴、网格和边框样式。"""
    axis.set_xlabel("迭代轮次")
    axis.set_ylabel(y_label)
    if y_limit is not None:
        axis.set_ylim(*y_limit)
    axis.grid(axis="y", color="#EAECF0", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_series(
    axis: plt.Axes,
    rounds: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
    line_style: str,
    window: int,
) -> None:
    """绘制浅色原始序列和醒目的滑动平均序列。"""
    axis.plot(rounds, values, color=color, alpha=0.16, linewidth=0.8)
    axis.plot(
        rounds,
        moving_average(values, window),
        color=color,
        linestyle=line_style,
        linewidth=2.0,
        label=label,
    )


def plot_consensus_overview(
    path: Path,
    metrics: Sequence[Dict[str, object]],
    smooth_window: int,
    soft_threshold: float,
    strong_threshold: float,
) -> None:
    """绘制标签、概率、层级对齐、覆盖率、置信度和模型质量总览图。"""
    rounds = metric_array(metrics, "epoch")
    blue = "#2F6690"
    blue_dark = "#1F4E6D"
    gold = "#C4871A"
    gold_dark = "#8A5A08"
    neutral = "#667085"
    neutral_light = "#98A2B3"

    figure, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=False)
    figure.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.07, hspace=0.38, wspace=0.22)
    figure.suptitle("MNIST 层次共识指标随迭代轮次的变化", fontsize=20, fontweight="bold", x=0.08, ha="left")
    figure.text(
        0.08,
        0.925,
        "浅线为单轮值，深线为 {} 轮居中滑动平均；每轮探针图片不同，趋势用于描述输出趋同而非严格同样本收敛。".format(
            smooth_window
        ),
        fontsize=10.5,
        color="#475467",
    )

    axis = axes[0, 0]
    plot_series(axis, rounds, metric_array(metrics, "client_vote_consensus_ratio"), "全部客户端", blue, "-", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "active_client_vote_consensus_ratio"), "活跃客户端", neutral, "--", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "within_edge_vote_consensus_ratio"), "域内客户端", neutral_light, ":", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "edge_vote_consensus_ratio"), "边缘模型", gold, "-", smooth_window)
    axis.axhline(soft_threshold, color="#475467", linestyle="--", linewidth=1.0, label="软共识阈值")
    axis.axhline(strong_threshold, color="#475467", linestyle=":", linewidth=1.0, label="强共识阈值")
    axis.set_title("A. 标签共识比例", loc="left", fontweight="bold")
    style_axis(axis, "多数标签占比", (0.0, 1.02))
    axis.legend(ncol=2, fontsize=8.5, frameon=False, loc="lower right")

    axis = axes[0, 1]
    plot_series(axis, rounds, metric_array(metrics, "client_probability_consensus_score"), "全部客户端", blue, "-", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "active_client_probability_consensus_score"), "活跃客户端", neutral, "--", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "within_edge_probability_consensus_score"), "域内客户端", neutral_light, ":", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "edge_probability_consensus_score"), "边缘模型", gold, "-", smooth_window)
    axis.set_title("B. 完整概率分布共识", loc="left", fontweight="bold")
    style_axis(axis, "1 - 归一化广义 JSD", (0.0, 1.02))
    axis.legend(ncol=2, fontsize=8.5, frameon=False, loc="lower right")

    axis = axes[1, 0]
    plot_series(axis, rounds, metric_array(metrics, "client_cloud_alignment_score"), "客户端到云", blue, "-", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "edge_cloud_alignment_score"), "边缘到云", gold, "-", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "client_cloud_label_agreement"), "客户端与云标签一致", neutral, "--", smooth_window)
    axis.set_title("C. 与云模型的对齐程度", loc="left", fontweight="bold")
    style_axis(axis, "对齐得分 / 标签比例", (0.0, 1.02))
    axis.legend(fontsize=8.5, frameon=False, loc="lower right")

    axis = axes[1, 1]
    soft_coverage = moving_average(metric_array(metrics, "soft_consensus"), max(10, smooth_window))
    strong_coverage = moving_average(metric_array(metrics, "strong_consensus"), max(10, smooth_window))
    majority_match = moving_average(metric_array(metrics, "majority_cloud_label_match"), max(10, smooth_window))
    axis.plot(rounds, soft_coverage, color=blue, linewidth=2.2, label="软共识覆盖率")
    axis.plot(rounds, strong_coverage, color=gold, linewidth=2.2, label="强共识覆盖率")
    axis.plot(rounds, majority_match, color=neutral, linestyle="--", linewidth=1.8, label="多数标签与云一致率")
    axis.set_title("D. 最近 10 轮共识覆盖率", loc="left", fontweight="bold")
    style_axis(axis, "窗口内事件比例", (0.0, 1.02))
    axis.legend(fontsize=8.5, frameon=False, loc="lower right")

    axis = axes[2, 0]
    plot_series(axis, rounds, metric_array(metrics, "cloud_confidence"), "云置信度", blue_dark, "-", smooth_window)
    plot_series(axis, rounds, metric_array(metrics, "cloud_margin"), "云 Top1-Top2 间隔", gold_dark, "-", smooth_window)
    plot_series(axis, rounds, 1.0 - metric_array(metrics, "cloud_entropy_norm"), "云输出集中度", neutral, "--", smooth_window)
    axis.set_title("E. 云模型决策确定性", loc="left", fontweight="bold")
    style_axis(axis, "得分", (0.0, 1.02))
    axis.legend(fontsize=8.5, frameon=False, loc="lower right")

    axis = axes[2, 1]
    test_acc = metric_array(metrics, "test_acc")
    train_acc = metric_array(metrics, "train_acc")
    test_loss = metric_array(metrics, "test_loss")
    plot_series(axis, rounds, test_acc, "测试准确率", blue, "-", smooth_window)
    plot_series(axis, rounds, train_acc, "训练准确率", neutral, "--", smooth_window)
    axis.set_title("F. 模型质量护栏", loc="left", fontweight="bold")
    style_axis(axis, "准确率", (0.0, 1.02))
    loss_axis = axis.twinx()
    loss_axis.plot(rounds, moving_average(test_loss, smooth_window), color=gold, linewidth=1.7, label="测试损失")
    loss_axis.set_ylabel("测试损失", color=gold_dark)
    loss_axis.tick_params(axis="y", colors=gold_dark)
    loss_axis.spines["top"].set_visible(False)
    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = loss_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right, fontsize=8.5, frameon=False, loc="center right")

    figure.text(
        0.08,
        0.025,
        "说明：边缘模型间共识在有效边缘少于 2 个时记为空值；云置信度高不代表多客户端已经达成共识，也不代表预测正确。",
        fontsize=9.5,
        color="#475467",
    )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_hierarchy_and_clients(
    path: Path,
    metrics: Sequence[Dict[str, object]],
    cloud_alignment_matrix: np.ndarray,
    cloud_agreement_matrix: np.ndarray,
    active_matrix: np.ndarray,
) -> None:
    """绘制动态参与趋势、逐客户端概率/标签对齐热图和活跃状态热图。"""
    rounds = metric_array(metrics, "epoch")
    blue = "#2F6690"
    gold = "#C4871A"
    neutral = "#667085"

    figure = plt.figure(figsize=(17, 15))
    grid = figure.add_gridspec(4, 1, height_ratios=[1.0, 1.7, 1.45, 1.45], hspace=0.34)
    figure.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.07)
    figure.suptitle("动态参与和逐客户端共识诊断", fontsize=20, fontweight="bold", x=0.08, ha="left")
    figure.text(
        0.08,
        0.915,
        "依次展示每轮参与规模、每个客户端到云的概率对齐、标签一致状态和聚合参与状态。",
        fontsize=10.5,
        color="#475467",
    )

    axis = figure.add_subplot(grid[0, 0])
    axis.plot(rounds, metric_array(metrics, "active_client_count"), color=blue, linewidth=1.7, label="活跃客户端数")
    axis.fill_between(rounds, metric_array(metrics, "active_client_count"), color=blue, alpha=0.10)
    axis.set_title("A. 动态参与规模", loc="left", fontweight="bold")
    axis.set_xlabel("迭代轮次")
    axis.set_ylabel("客户端数")
    axis.set_ylim(0, max(metric_array(metrics, "client_count")) + 2)
    axis.grid(axis="y", color="#EAECF0", linewidth=0.8)
    edge_axis = axis.twinx()
    edge_axis.plot(rounds, metric_array(metrics, "active_edge_count"), color=gold, linewidth=1.7, label="有效边缘数")
    edge_axis.set_ylabel("边缘数", color="#8A5A08")
    edge_axis.set_ylim(0, max(6.5, float(np.nanmax(metric_array(metrics, "active_edge_count"))) + 0.5))
    edge_axis.tick_params(axis="y", colors="#8A5A08")
    zero_edge_rounds = rounds[metric_array(metrics, "active_edge_count") == 0]
    if zero_edge_rounds.size:
        axis.scatter(zero_edge_rounds, np.zeros_like(zero_edge_rounds), marker="x", color="#8A5A08", s=35, label="空聚合轮")
    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = edge_axis.get_legend_handles_labels()
    axis.legend(handles_left + handles_right, labels_left + labels_right, ncol=3, frameon=False, fontsize=9, loc="upper right")
    axis.spines["top"].set_visible(False)
    edge_axis.spines["top"].set_visible(False)

    alignment_axis = figure.add_subplot(grid[1, 0])
    alignment_cmap = LinearSegmentedColormap.from_list(
        "alignment",
        ["#F4D7A1", "#F8F9FB", blue],
    )
    alignment_image = alignment_axis.imshow(
        cloud_alignment_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=alignment_cmap,
        vmin=0,
        vmax=1,
    )
    alignment_axis.set_title("B. 每个客户端与云模型的概率分布对齐得分", loc="left", fontweight="bold")
    alignment_axis.set_ylabel("客户端编号")
    alignment_axis.set_xlabel("迭代轮次")
    alignment_axis.set_yticks(np.arange(0, cloud_alignment_matrix.shape[0], 4))
    alignment_axis.set_yticklabels(np.arange(0, cloud_alignment_matrix.shape[0], 4))
    alignment_axis.set_xticks(np.arange(0, cloud_alignment_matrix.shape[1], 10))
    alignment_axis.set_xticklabels(np.arange(1, cloud_alignment_matrix.shape[1] + 1, 10))
    colorbar = figure.colorbar(alignment_image, ax=alignment_axis, fraction=0.018, pad=0.012)
    colorbar.set_label("1 - JS(p_i, p_g)")

    agreement_axis = figure.add_subplot(grid[2, 0])
    agreement_cmap = ListedColormap(["#F4D7A1", blue])
    agreement_image = agreement_axis.imshow(
        cloud_agreement_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=agreement_cmap,
        vmin=0,
        vmax=1,
    )
    agreement_axis.set_title("C. 客户端预测标签与云模型标签是否一致", loc="left", fontweight="bold")
    agreement_axis.set_ylabel("客户端编号")
    agreement_axis.set_xlabel("迭代轮次")
    agreement_axis.set_yticks(np.arange(0, cloud_agreement_matrix.shape[0], 4))
    agreement_axis.set_yticklabels(np.arange(0, cloud_agreement_matrix.shape[0], 4))
    agreement_axis.set_xticks(np.arange(0, cloud_agreement_matrix.shape[1], 10))
    agreement_axis.set_xticklabels(np.arange(1, cloud_agreement_matrix.shape[1] + 1, 10))
    colorbar = figure.colorbar(agreement_image, ax=agreement_axis, fraction=0.018, pad=0.012, ticks=[0.25, 0.75])
    colorbar.ax.set_yticklabels(["不一致", "一致"])

    active_axis = figure.add_subplot(grid[3, 0])
    active_cmap = ListedColormap(["#EAECF0", gold])
    active_image = active_axis.imshow(
        active_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=active_cmap,
        vmin=0,
        vmax=1,
    )
    active_axis.set_title("D. 客户端是否参与当前轮聚合", loc="left", fontweight="bold")
    active_axis.set_ylabel("客户端编号")
    active_axis.set_xlabel("迭代轮次")
    active_axis.set_yticks(np.arange(0, active_matrix.shape[0], 4))
    active_axis.set_yticklabels(np.arange(0, active_matrix.shape[0], 4))
    active_axis.set_xticks(np.arange(0, active_matrix.shape[1], 10))
    active_axis.set_xticklabels(np.arange(1, active_matrix.shape[1] + 1, 10))
    colorbar = figure.colorbar(active_image, ax=active_axis, fraction=0.018, pad=0.012, ticks=[0.25, 0.75])
    colorbar.ax.set_yticklabels(["未参与", "参与"])

    figure.text(
        0.08,
        0.025,
        "概率对齐热图保留了 10 维输出差异；标签热图只比较 argmax 类别，两者应与模型质量护栏一起解释。",
        fontsize=9.5,
        color="#475467",
    )
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def print_key_summary(metrics: Sequence[Dict[str, object]], output_dir: Path) -> None:
    """在终端打印最关键的汇总结果和输出位置。"""
    vote_ratio = metric_array(metrics, "client_vote_consensus_ratio")
    probability_score = metric_array(metrics, "client_probability_consensus_score")
    edge_score = metric_array(metrics, "edge_probability_consensus_score")
    soft_consensus = metric_array(metrics, "soft_consensus")
    print("共识分析完成：")
    print("  全客户端标签共识比例均值: {:.4f}".format(safe_mean(vote_ratio)))
    print("  全客户端概率共识得分均值: {:.4f}".format(safe_mean(probability_score)))
    print("  边缘模型概率共识得分均值: {:.4f}".format(safe_mean(edge_score)))
    print("  软共识覆盖率: {:.2%}".format(safe_mean(soft_consensus)))
    print("  输出目录: {}".format(output_dir))


def main() -> None:
    """加载探针数据、计算指标、导出 CSV，并生成两张可视化图。"""
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    result_dir = (
        find_latest_result_dir(script_dir / "result")
        if args.result_dir is None
        else resolve_input_path(args.result_dir, script_dir)
    )
    mat_path = resolve_input_path(args.mat_file, script_dir)
    metadata_path = (
        None if args.metadata_file is None else resolve_input_path(args.metadata_file, script_dir)
    )
    output_dir = (
        result_dir / "consensus_analysis"
        if args.output_dir is None
        else resolve_input_path(args.output_dir, result_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_paths = {name: result_dir / filename for name, filename in PROBE_FILENAMES.items()}
    for probe_path in probe_paths.values():
        if not probe_path.exists():
            raise FileNotFoundError("缺少探针文件：{}".format(probe_path))
    if not mat_path.exists():
        raise FileNotFoundError("缺少动态分组文件：{}".format(mat_path))

    client_rows = read_probability_csv(probe_paths["client"])
    edge_rows = read_probability_csv(probe_paths["edge"])
    cloud_rows = read_probability_csv(probe_paths["cloud"])
    round_count = len(client_rows)
    if len(edge_rows) != round_count or len(cloud_rows) != round_count:
        raise ValueError("客户端、边缘和云探针 CSV 的行数必须一致。")
    client_count = len(client_rows[0])
    groups_by_round = load_dynamic_groups(mat_path, round_count, client_count)
    true_labels = read_true_labels(metadata_path, round_count)
    quality_series = {
        "train_acc": read_metric_series(result_dir / "train_acc.txt", round_count),
        "train_loss": read_metric_series(result_dir / "train_loss.txt", round_count),
        "test_acc": read_metric_series(result_dir / "test_acc.txt", round_count),
        "test_loss": read_metric_series(result_dir / "test_loss.txt", round_count),
    }

    metrics, cloud_alignment_matrix, cloud_agreement_matrix, active_matrix = compute_round_metrics(
        client_rows=client_rows,
        edge_rows=edge_rows,
        cloud_rows=cloud_rows,
        groups_by_round=groups_by_round,
        true_labels=true_labels,
        quality_series=quality_series,
        soft_threshold=args.soft_threshold,
        strong_threshold=args.strong_threshold,
    )

    configure_plot_font()
    write_metrics_csv(output_dir / "consensus_metrics_by_round.csv", metrics)
    write_summary_csv(output_dir / "consensus_metrics_summary.csv", metrics)
    write_metric_definitions(output_dir / "consensus_metric_definitions.csv")
    write_client_summary_csv(
        output_dir / "consensus_client_summary.csv",
        cloud_alignment_matrix,
        cloud_agreement_matrix,
        active_matrix,
    )
    plot_consensus_overview(
        output_dir / "consensus_overview.png",
        metrics,
        args.smooth_window,
        args.soft_threshold,
        args.strong_threshold,
    )
    plot_hierarchy_and_clients(
        output_dir / "consensus_hierarchy_and_clients.png",
        metrics,
        cloud_alignment_matrix,
        cloud_agreement_matrix,
        active_matrix,
    )
    print_key_summary(metrics, output_dir)


if __name__ == "__main__":
    main()
