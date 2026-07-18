"""分析客户端、边缘与云端固定概率探针，并生成逐 epoch 共识诊断结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# 服务器通常没有图形桌面，固定使用非交互式后端。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.io import loadmat


PROBE_FILENAMES = {
    "client": "probe_client_pre.csv",
    "edge": "probe_edge_post.csv",
    "cloud": "probe_cloud_post.csv",
}
PROBE_NPZ_FILENAME = "probe_probabilities.npz"
PROBE_METADATA_FILENAME = "probe_meta.csv"
TOPOLOGY_SCHEDULE_FILENAME = "topology_schedule.jsonl"

METRIC_DEFINITIONS = [
    {
        "metric": "client_probability_consensus_score",
        "name_zh": "候选客户端概率一致性 A",
        "formula": "mean_j(1-GJSD({p_i,j}))",
        "range": "[0,1]",
        "direction": "越大越一致",
        "meaning": "先对每张探针图计算客户端概率一致性，再在本 epoch 内取平均。",
    },
    {
        "metric": "client_certainty_score",
        "name_zh": "候选客户端确定性 C",
        "formula": "mean_j(1-mean_i(H(p_i,j)/ln(K)))",
        "range": "[0,1]",
        "direction": "越大越确定",
        "meaning": "抑制所有客户端共同输出均匀分布造成的虚假高一致性。",
    },
    {
        "metric": "client_effective_consensus_score",
        "name_zh": "候选客户端纯有效共识 S",
        "formula": "mean_j(A_j*C_j)",
        "range": "[0,1]",
        "direction": "越大越好",
        "meaning": "每张图先计算 A×C，禁止先平均不同图片的概率向量。",
    },
    {
        "metric": "client_correct_effective_consensus_score",
        "name_zh": "候选客户端正确有效共识",
        "formula": "mean_j(S_j*I(majority_j=y_j))",
        "range": "[0,1]",
        "direction": "越大越好",
        "meaning": "只保留客户端多数标签与真实标签一致的有效共识。",
    },
    {
        "metric": "client_wrong_effective_consensus_score",
        "name_zh": "候选客户端错误有效共识",
        "formula": "mean_j(S_j*I(majority_j!=y_j))",
        "range": "[0,1]",
        "direction": "越小越好",
        "meaning": "用于识别高一致但集体预测错误的情况。",
    },
    {
        "metric": "active_client_correct_effective_consensus_score",
        "name_zh": "活跃客户端正确有效共识",
        "formula": "mean_j(S_active,j*I(majority_active,j=y_j))",
        "range": "[0,1]",
        "direction": "越大越好",
        "meaning": "只比较当轮实际参与聚合的客户端；不足 2 个时为空。",
    },
    {
        "metric": "coverage_weighted_active_correct_effective_consensus",
        "name_zh": "覆盖率加权活跃正确有效共识",
        "formula": "active_coverage*active_correct_effective_consensus",
        "range": "[0,1]",
        "direction": "越大越好",
        "meaning": "同时反映参与规模与活跃客户端的正确共识质量。",
    },
    {
        "metric": "cloud_probe_accuracy",
        "name_zh": "固定探针云端准确率",
        "formula": "mean_j(I(argmax(p_g,j)=y_j))",
        "range": "[0,1]",
        "direction": "越大越好",
        "meaning": "固定探针护栏，最终性能仍应结合完整测试集 test_acc。",
    },
]


@dataclass
class ProbeData:
    """保存统一后的固定探针数组及其坐标信息。"""

    client_probabilities: np.ndarray
    edge_probabilities: np.ndarray
    cloud_probabilities: np.ndarray
    client_ids: np.ndarray
    true_labels: np.ndarray
    global_epochs: np.ndarray
    source_format: str
    probe_set_hash: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """解析命令行参数并校验平滑窗口和阈值。"""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="分析固定多图概率探针的逐 epoch 共识。")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="结果目录；未指定时选择 result 下最近修改的有效目录。",
    )
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=script_dir / "matlab" / "my_data_150.mat",
        help="缺少 topology_schedule.jsonl 时使用的旧版动态分组 MAT。",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="旧 CSV 探针的真实标签文件；NPZ 自带真实标签。",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录。")
    parser.add_argument("--smooth-window", type=int, default=10, help="完整尾随滑动窗口。")
    parser.add_argument("--soft-threshold", type=float, default=2.0 / 3.0)
    parser.add_argument("--strong-threshold", type=float, default=0.8)
    args = parser.parse_args()
    if args.smooth_window <= 0:
        parser.error("--smooth-window 必须大于 0。")
    if not 0.0 <= args.soft_threshold <= 1.0:
        parser.error("--soft-threshold 必须位于 [0,1]。")
    if not 0.0 <= args.strong_threshold <= 1.0:
        parser.error("--strong-threshold 必须位于 [0,1]。")
    return args


def _has_probe_files(path: Path) -> bool:
    """判断目录是否含新 NPZ 或完整的三份旧 CSV 探针。"""
    return (path / PROBE_NPZ_FILENAME).is_file() or all(
        (path / filename).is_file() for filename in PROBE_FILENAMES.values()
    )


def find_latest_result_dir(result_root: Path) -> Path:
    """在结果根目录中寻找最近修改的有效探针目录。"""
    candidates = []
    if result_root.exists():
        candidates = [path for path in result_root.iterdir() if path.is_dir() and _has_probe_files(path)]
    if not candidates:
        raise FileNotFoundError("未在 {} 下找到 NPZ 或完整旧 CSV 探针。".format(result_root))
    return max(candidates, key=lambda item: item.stat().st_mtime)


def resolve_input_path(path: Path, base_dir: Path) -> Path:
    """将输入路径解析为绝对路径，并优先按脚本目录解析相对路径。"""
    if path.is_absolute():
        return path
    script_candidate = (base_dir / path).resolve()
    return script_candidate if script_candidate.exists() else path.resolve()


def _validate_probability_vector(vector: np.ndarray, context: str) -> np.ndarray:
    """校验并归一化一维概率向量。"""
    if vector.ndim != 1 or vector.size < 2:
        raise ValueError("{} 必须是一维且至少含两个类别。".format(context))
    if not np.all(np.isfinite(vector)) or np.any(vector < -1e-8):
        raise ValueError("{} 含负值或非有限数。".format(context))
    probability_sum = float(vector.sum())
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("{} 的概率和 {:.8f} 不接近 1。".format(context, probability_sum))
    return vector / probability_sum


def read_probability_csv(path: Path) -> List[List[Optional[np.ndarray]]]:
    """读取旧探针 CSV，并保留边缘文件中的空槽位。"""
    rows: List[List[Optional[np.ndarray]]] = []
    expected_width: Optional[int] = None
    expected_classes: Optional[int] = None
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row_index, raw_row in enumerate(csv.reader(file_obj), start=1):
            if expected_width is None:
                expected_width = len(raw_row)
            elif len(raw_row) != expected_width:
                raise ValueError("{} 第 {} 行列数不一致。".format(path, row_index))
            parsed_row: List[Optional[np.ndarray]] = []
            for column_index, cell in enumerate(raw_row, start=1):
                if not cell.strip():
                    parsed_row.append(None)
                    continue
                try:
                    vector = np.asarray(json.loads(cell), dtype=np.float64)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "{} 第 {} 行第 {} 列不是合法 JSON。".format(path, row_index, column_index)
                    ) from exc
                vector = _validate_probability_vector(
                    vector, "{} 第 {} 行第 {} 列".format(path, row_index, column_index)
                )
                if expected_classes is None:
                    expected_classes = int(vector.size)
                elif vector.size != expected_classes:
                    raise ValueError("{} 的概率类别数不一致。".format(path))
                parsed_row.append(vector)
            rows.append(parsed_row)
    if not rows:
        raise ValueError("{} 为空文件。".format(path))
    return rows


def read_metric_series(path: Path, round_count: int) -> np.ndarray:
    """读取每行一个数值的训练指标，并截断或补 NaN 到探针轮数。"""
    values = np.full(round_count, np.nan, dtype=np.float64)
    if not path.exists():
        return values
    raw_values = [float(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usable_count = min(round_count, len(raw_values))
    values[:usable_count] = raw_values[:usable_count]
    return values


def read_true_labels(path: Optional[Path], round_count: int) -> List[Optional[int]]:
    """读取旧探针真实标签，并优先按 global_epoch 显式对齐。"""
    labels: List[Optional[int]] = [None for _ in range(round_count)]
    if path is None or not path.exists():
        return labels
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None or "true_label" not in reader.fieldnames:
            raise ValueError("{} 必须包含 true_label 列。".format(path))
        has_global_epoch = "global_epoch" in reader.fieldnames
        seen = set()
        for row_index, row in enumerate(reader):
            if not has_global_epoch and row_index >= round_count:
                break
            target_index = row_index
            if has_global_epoch:
                raw_epoch = (row.get("global_epoch") or "").strip()
                if not raw_epoch:
                    raise ValueError("{} 第 {} 行缺少 global_epoch。".format(path, row_index + 2))
                target_index = int(raw_epoch)
                if target_index in seen:
                    raise ValueError("{} 存在重复 global_epoch：{}。".format(path, target_index))
                seen.add(target_index)
            if target_index < 0 or target_index >= round_count:
                raise ValueError("{} 的标签轮次 {} 越界。".format(path, target_index))
            raw_label = (row.get("true_label") or "").strip()
            if raw_label:
                label = int(raw_label)
                if label < 0:
                    raise ValueError("true_label 不能为负数。")
                labels[target_index] = label
    return labels


def _validate_dense_probabilities(values: np.ndarray, name: str) -> None:
    """校验稠密概率张量的有限性、范围和末维概率和。"""
    if not np.all(np.isfinite(values)):
        raise ValueError("{} 包含非有限数。".format(name))
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ValueError("{} 存在超出 [0,1] 的值。".format(name))
    if not np.allclose(np.sum(values, axis=-1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("{} 存在概率和不为 1 的向量。".format(name))


def load_npz_probe_data(path: Path) -> ProbeData:
    """以禁止对象反序列化的方式加载并严格校验固定探针 NPZ。"""
    required_keys = {
        "client_probabilities",
        "edge_probabilities",
        "cloud_probabilities",
        "client_ids",
        "probe_indices",
        "true_labels",
        "global_epochs",
    }
    with np.load(str(path), allow_pickle=False) as archive:
        missing = sorted(required_keys.difference(archive.files))
        if missing:
            raise ValueError("{} 缺少字段：{}。".format(path, ", ".join(missing)))
        clients = np.asarray(archive["client_probabilities"], dtype=np.float64)
        edges = np.asarray(archive["edge_probabilities"], dtype=np.float64)
        cloud = np.asarray(archive["cloud_probabilities"], dtype=np.float64)
        client_ids = np.asarray(archive["client_ids"], dtype=np.int64).reshape(-1)
        probe_indices = np.asarray(archive["probe_indices"], dtype=np.int64).reshape(-1)
        true_labels = np.asarray(archive["true_labels"], dtype=np.int64).reshape(-1)
        global_epochs = np.asarray(archive["global_epochs"], dtype=np.int64).reshape(-1)
        completed = int(np.asarray(archive["completed_epochs"]).item()) if "completed_epochs" in archive else clients.shape[0]
        probe_hash = str(np.asarray(archive["probe_set_hash"]).item()) if "probe_set_hash" in archive else None

    if clients.ndim != 4 or edges.ndim != 4 or cloud.ndim != 3:
        raise ValueError("NPZ 概率形状必须分别为 [E,M,P,K]、[E,G,P,K]、[E,P,K]。")
    if completed < 0 or completed > clients.shape[0]:
        raise ValueError("completed_epochs 与客户端概率轮数不一致。")
    clients, edges, cloud, global_epochs = (
        clients[:completed], edges[:completed], cloud[:completed], global_epochs[:completed]
    )
    epoch_count, client_count, probe_count, class_count = clients.shape
    if epoch_count == 0:
        raise ValueError("NPZ 没有已完成 epoch。")
    if edges.shape[0] != epoch_count or edges.shape[2:] != (probe_count, class_count):
        raise ValueError("边缘概率与客户端概率的 epoch、探针或类别维不一致。")
    if cloud.shape != (epoch_count, probe_count, class_count):
        raise ValueError("云端概率形状与客户端概率不一致。")
    if client_ids.size != client_count or np.unique(client_ids).size != client_ids.size:
        raise ValueError("client_ids 数量错误或包含重复值。")
    if probe_indices.size != probe_count or np.unique(probe_indices).size != probe_indices.size:
        raise ValueError("probe_indices 数量错误或包含重复值。")
    if true_labels.size != probe_count or np.any(true_labels < 0) or np.any(true_labels >= class_count):
        raise ValueError("true_labels 数量或范围不合法。")
    expected_epochs = np.arange(epoch_count, dtype=np.int64)
    if not np.array_equal(global_epochs, expected_epochs):
        # 固定探针文件按训练完成顺序保存；跳轮、乱序或重复都会破坏跨文件对齐。
        raise ValueError("global_epochs 必须从 0 开始连续递增，并与已完成 epoch 数一致。")
    _validate_dense_probabilities(clients, "客户端概率")
    _validate_dense_probabilities(cloud, "云端概率")
    # 未启用边缘槽位必须整块为 NaN；启用槽位则必须全部为有效概率。
    for epoch_index in range(epoch_count):
        for edge_index in range(edges.shape[1]):
            block = edges[epoch_index, edge_index]
            if np.all(np.isnan(block)):
                continue
            if np.any(np.isnan(block)):
                raise ValueError("边缘槽位只能整块为 NaN，不能部分缺失。")
            _validate_dense_probabilities(block, "边缘概率")
    labels_by_epoch = np.repeat(true_labels[None, :], epoch_count, axis=0)
    return ProbeData(
        clients, edges, cloud, client_ids, labels_by_epoch, global_epochs, "npz", probe_hash
    )


def load_legacy_probe_data(result_dir: Path, metadata_path: Optional[Path]) -> ProbeData:
    """加载旧版单图 CSV，并扩展为统一的四维/三维概率张量。"""
    client_rows = read_probability_csv(result_dir / PROBE_FILENAMES["client"])
    edge_rows = read_probability_csv(result_dir / PROBE_FILENAMES["edge"])
    cloud_rows = read_probability_csv(result_dir / PROBE_FILENAMES["cloud"])
    epoch_count = len(client_rows)
    if len(edge_rows) != epoch_count or len(cloud_rows) != epoch_count:
        raise ValueError("客户端、边缘和云端旧 CSV 行数必须一致。")
    if any(any(item is None for item in row) for row in client_rows):
        raise ValueError("客户端旧 CSV 不能含空单元格。")
    if any(len(row) != 1 or row[0] is None for row in cloud_rows):
        raise ValueError("云端旧 CSV 每行必须恰好有一个概率向量。")
    clients = np.asarray([[item for item in row] for row in client_rows], dtype=np.float64)[:, :, None, :]
    class_count = clients.shape[-1]
    edge_count = len(edge_rows[0])
    edges = np.full((epoch_count, edge_count, 1, class_count), np.nan, dtype=np.float64)
    for epoch_index, row in enumerate(edge_rows):
        if len(row) != edge_count:
            raise ValueError("边缘旧 CSV 各行槽位数必须一致。")
        for edge_index, item in enumerate(row):
            if item is not None:
                edges[epoch_index, edge_index, 0] = item
    cloud = np.asarray([row[0] for row in cloud_rows], dtype=np.float64)[:, None, :]
    raw_labels = read_true_labels(metadata_path, epoch_count)
    labels = np.asarray([[-1 if value is None else int(value)] for value in raw_labels], dtype=np.int64)
    if np.any(labels >= class_count):
        raise ValueError("旧探针真实标签超出概率类别范围。")
    return ProbeData(
        clients,
        edges,
        cloud,
        np.arange(clients.shape[1], dtype=np.int64),
        labels,
        np.arange(epoch_count, dtype=np.int64),
        "legacy_csv",
    )


def load_probe_data(result_dir: Path, metadata_path: Optional[Path]) -> ProbeData:
    """优先读取固定探针 NPZ，缺少时回退旧版三份单图 CSV。"""
    npz_path = result_dir / PROBE_NPZ_FILENAME
    if npz_path.is_file():
        return load_npz_probe_data(npz_path)
    missing = [name for name in PROBE_FILENAMES.values() if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("缺少 NPZ，且旧 CSV 不完整：{}。".format(", ".join(missing)))
    return load_legacy_probe_data(result_dir, metadata_path)


def normalized_entropy(probability: np.ndarray) -> float:
    """计算一个概率向量按 ln(K) 归一化的信息熵。"""
    vector = np.asarray(probability, dtype=np.float64)
    positive = vector[vector > 0.0]
    if positive.size == 0:
        return 0.0
    return -float(np.sum(positive * np.log(positive))) / math.log(vector.size)


def normalized_entropy_array(probabilities: np.ndarray) -> np.ndarray:
    """沿概率张量末维批量计算归一化信息熵。"""
    values = np.asarray(probabilities, dtype=np.float64)
    safe = np.where(values > 0.0, values, 1.0)
    terms = np.where(values > 0.0, values * np.log(safe), 0.0)
    return -np.sum(terms, axis=-1) / math.log(values.shape[-1])


def generalized_js_divergence(probabilities: np.ndarray) -> float:
    """计算一组模型对同一张图输出的归一化广义 JSD。"""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return float("nan")
    if values.shape[0] == 1:
        return 0.0
    divergence = normalized_entropy(np.mean(values, axis=0)) - float(
        np.mean(normalized_entropy_array(values))
    )
    return float(np.clip(divergence, 0.0, 1.0))


def pairwise_js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个概率分布按 ln(2) 归一化的 Jensen-Shannon 分歧。"""
    midpoint = (left + right) / 2.0

    def kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
        """计算 source 到 target 的离散 KL 散度。"""
        mask = source > 0.0
        return float(np.sum(source[mask] * np.log(source[mask] / target[mask])))

    divergence = 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(right, midpoint)
    return float(np.clip(divergence / math.log(2.0), 0.0, 1.0))


def vote_metrics(probabilities: np.ndarray) -> Tuple[float, float, float, int]:
    """计算单张图的多数标签比例、投票熵、标签共识和多数标签。"""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return float("nan"), float("nan"), float("nan"), -1
    labels = np.argmax(values, axis=1)
    counts = np.bincount(labels, minlength=values.shape[1]).astype(np.float64)
    distribution = counts / counts.sum()
    entropy = normalized_entropy(distribution)
    return float(counts.max() / counts.sum()), entropy, 1.0 - entropy, int(np.argmax(counts))


def probability_margin(probability: np.ndarray) -> float:
    """计算一个概率向量最高概率与次高概率的差。"""
    ordered = np.sort(probability)
    return float(ordered[-1] - ordered[-2])


def safe_mean(values: Sequence[float]) -> float:
    """忽略非有限值求均值；没有有效值时返回 NaN。"""
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    return float(np.mean(valid)) if valid.size else float("nan")


def _population_sample_metrics(probabilities: np.ndarray, true_labels: np.ndarray) -> Dict[str, np.ndarray]:
    """对每张图独立计算 A、C、S 及正确/错误有效共识。"""
    values = np.asarray(probabilities, dtype=np.float64)
    probe_count = values.shape[1]
    empty = np.full(probe_count, np.nan, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        return {
            "agreement": empty.copy(), "certainty": empty.copy(), "effective": empty.copy(),
            "correct": empty.copy(), "wrong": empty.copy(), "vote_ratio": empty.copy(),
            "vote_entropy": empty.copy(), "label_score": empty.copy(),
            "majority": np.full(probe_count, -1, dtype=np.int64),
        }
    client_entropy = normalized_entropy_array(values)
    mean_client_entropy = np.mean(client_entropy, axis=0)
    mean_probability_entropy = normalized_entropy_array(np.mean(values, axis=0))
    agreement = 1.0 - np.clip(mean_probability_entropy - mean_client_entropy, 0.0, 1.0)
    certainty = 1.0 - mean_client_entropy
    effective = agreement * certainty
    vote_ratio = np.empty(probe_count, dtype=np.float64)
    vote_entropy = np.empty(probe_count, dtype=np.float64)
    label_score = np.empty(probe_count, dtype=np.float64)
    majority = np.empty(probe_count, dtype=np.int64)
    for probe_index in range(probe_count):
        vote_ratio[probe_index], vote_entropy[probe_index], label_score[probe_index], majority[probe_index] = vote_metrics(
            values[:, probe_index, :]
        )
    labels = np.asarray(true_labels, dtype=np.int64)
    valid_labels = labels >= 0
    correct = np.full(probe_count, np.nan, dtype=np.float64)
    wrong = np.full(probe_count, np.nan, dtype=np.float64)
    correct[valid_labels] = effective[valid_labels] * (majority[valid_labels] == labels[valid_labels])
    wrong[valid_labels] = effective[valid_labels] * (majority[valid_labels] != labels[valid_labels])
    return {
        "agreement": agreement, "certainty": certainty, "effective": effective,
        "correct": correct, "wrong": wrong, "vote_ratio": vote_ratio,
        "vote_entropy": vote_entropy, "label_score": label_score, "majority": majority,
    }


def load_dynamic_groups(mat_path: Path, round_count: int, client_count: int) -> List[List[List[int]]]:
    """从旧 MAT 的组规模还原顺序槽位分组，仅作为运行时拓扑缺失时的回退。"""
    data = loadmat(str(mat_path))
    if "group_num" not in data or "client_num" not in data:
        raise KeyError("{} 必须包含 group_num 和 client_num。".format(mat_path))
    group_num = np.asarray(data["group_num"]).reshape(-1)
    client_num = np.asarray(data["client_num"])
    if group_num.shape[0] < round_count or client_num.shape[0] < round_count:
        raise ValueError("动态分组 MAT 轮数少于探针轮数。")
    result: List[List[List[int]]] = []
    for epoch_index in range(round_count):
        enabled = int(group_num[epoch_index])
        counts = np.asarray(client_num[epoch_index]).astype(int).reshape(-1)
        groups: List[List[int]] = []
        next_slot = 0
        for count in counts[:enabled]:
            if count < 0:
                raise ValueError("第 {} 轮存在负客户端数。".format(epoch_index + 1))
            group = list(range(next_slot, next_slot + int(count)))
            next_slot += int(count)
            if group:
                groups.append(group)
        if next_slot > client_count:
            raise ValueError("第 {} 轮 MAT 活跃槽位数超过候选数。".format(epoch_index + 1))
        result.append(groups)
    return result


def load_runtime_groups(
    path: Path,
    global_epochs: np.ndarray,
    client_count: int,
    expected_client_ids: np.ndarray,
    allow_sequential_placeholder: bool = False,
) -> Tuple[List[List[List[int]]], List[List[int]], np.ndarray]:
    """从 topology_schedule.jsonl 精确读取每轮分组和活跃候选槽位。"""
    records: Dict[int, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "global_epoch" not in record:
                raise ValueError("{} 第 {} 行缺少 global_epoch。".format(path, line_number))
            epoch = int(record["global_epoch"])
            if epoch in records:
                raise ValueError("{} 存在重复 global_epoch：{}。".format(path, epoch))
            records[epoch] = record
    groups_by_round: List[List[List[int]]] = []
    active_by_round: List[List[int]] = []
    schedule_ids: Optional[np.ndarray] = None
    for epoch in global_epochs:
        if int(epoch) not in records:
            raise ValueError("{} 缺少 global_epoch={} 的记录。".format(path, int(epoch)))
        record = records[int(epoch)]
        raw_ids = record.get("candidate_client_indexes")
        if raw_ids is None or len(raw_ids) != client_count:
            raise ValueError("global_epoch={} 的候选客户端列表不合法。".format(int(epoch)))
        current_ids = np.asarray(raw_ids, dtype=np.int64)
        if schedule_ids is None:
            schedule_ids = current_ids
        elif not np.array_equal(schedule_ids, current_ids):
            raise ValueError("运行时拓扑中的固定候选客户端在 epoch 间发生变化。")
        raw_groups = record.get("mat_group_to_candidate_slots") or {}
        if not raw_groups and record.get("group_to_client_indexes"):
            # 兼容早期运行记录：把真实客户端编号显式映射回固定候选槽位。
            id_to_slot = {int(client_id): slot for slot, client_id in enumerate(current_ids)}
            try:
                raw_groups = {
                    key: [id_to_slot[int(client_id)] for client_id in client_ids]
                    for key, client_ids in record["group_to_client_indexes"].items()
                }
            except KeyError as exc:
                raise ValueError(
                    "global_epoch={} 的分组客户端不在固定候选列表中。".format(int(epoch))
                ) from exc
        groups = [
            [int(slot) for slot in raw_groups[key]]
            for key in sorted(raw_groups, key=lambda value: int(value))
            if raw_groups[key]
        ]
        if not groups and record.get("mat_active_candidate_slots"):
            groups = [[int(slot) for slot in record["mat_active_candidate_slots"]]]
        active_slots = [int(slot) for slot in record.get("mat_active_candidate_slots", [])]
        if not active_slots and record.get("active_client_indexes"):
            id_to_slot = {int(client_id): slot for slot, client_id in enumerate(current_ids)}
            try:
                active_slots = [
                    id_to_slot[int(client_id)]
                    for client_id in record["active_client_indexes"]
                ]
            except KeyError as exc:
                raise ValueError(
                    "global_epoch={} 的活跃客户端不在固定候选列表中。".format(int(epoch))
                ) from exc
        if not active_slots:
            active_slots = sorted({slot for group in groups for slot in group})
        all_slots = [slot for group in groups for slot in group] + active_slots
        if any(slot < 0 or slot >= client_count for slot in all_slots):
            raise ValueError("global_epoch={} 含越界候选槽位。".format(int(epoch)))
        groups_by_round.append(groups)
        active_by_round.append(sorted(set(active_slots)))
    assert schedule_ids is not None
    if expected_client_ids.size and not np.array_equal(expected_client_ids, schedule_ids):
        # 只有旧 CSV 的顺序占位编号允许被运行时记录替换；NPZ 必须严格一致。
        if not allow_sequential_placeholder:
            raise ValueError("NPZ client_ids 与 topology_schedule.jsonl 不一致。")
    return groups_by_round, active_by_round, schedule_ids


def resolve_groups(
    result_dir: Path, mat_path: Path, probe_data: ProbeData
) -> Tuple[List[List[List[int]]], List[List[int]], np.ndarray]:
    """优先使用运行时 JSONL；仅在缺失时退回旧 MAT 分组。"""
    schedule_path = result_dir / TOPOLOGY_SCHEDULE_FILENAME
    if schedule_path.is_file():
        return load_runtime_groups(
            schedule_path,
            probe_data.global_epochs,
            probe_data.client_probabilities.shape[1],
            probe_data.client_ids,
            probe_data.source_format == "legacy_csv",
        )
    if not mat_path.is_file():
        raise FileNotFoundError("缺少运行时拓扑且旧 MAT 不存在：{}。".format(mat_path))
    groups = load_dynamic_groups(
        mat_path, probe_data.client_probabilities.shape[0], probe_data.client_probabilities.shape[1]
    )
    active = [sorted({slot for group in epoch_groups for slot in group}) for epoch_groups in groups]
    return groups, active, probe_data.client_ids


def compute_within_edge_metrics(
    client_probabilities: np.ndarray, groups: Sequence[Sequence[int]]
) -> Tuple[float, float, float]:
    """逐图计算各组共识，再按组客户端数加权并在探针图上平均。"""
    values = np.asarray(client_probabilities, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, None, :]
    total = sum(len(group) for group in groups)
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    weighted_vote = np.zeros(values.shape[1], dtype=np.float64)
    weighted_divergence = np.zeros(values.shape[1], dtype=np.float64)
    for group in groups:
        group_values = values[np.asarray(group, dtype=int)]
        weight = float(len(group)) / float(total)
        for probe_index in range(values.shape[1]):
            vote_ratio, _, _, _ = vote_metrics(group_values[:, probe_index, :])
            weighted_vote[probe_index] += weight * vote_ratio
            weighted_divergence[probe_index] += weight * generalized_js_divergence(
                group_values[:, probe_index, :]
            )
    return safe_mean(weighted_vote), safe_mean(weighted_divergence), safe_mean(1.0 - weighted_divergence)


def _mean_pairwise_alignment(models: np.ndarray, cloud: np.ndarray) -> Tuple[float, np.ndarray]:
    """计算模型集合到同图云端输出的平均 JS 对齐及逐模型均值。"""
    if models.shape[0] == 0:
        return float("nan"), np.empty(0, dtype=np.float64)
    per_model = np.asarray(
        [
            safe_mean([pairwise_js_divergence(model[p], cloud[p]) for p in range(cloud.shape[0])])
            for model in models
        ],
        dtype=np.float64,
    )
    return safe_mean(per_model), per_model


def compute_round_metrics(
    client_probabilities: np.ndarray,
    edge_probabilities: np.ndarray,
    cloud_probabilities: np.ndarray,
    groups_by_round: List[List[List[int]]],
    active_slots_by_round: List[List[int]],
    true_labels: np.ndarray,
    global_epochs: np.ndarray,
    quality_series: Dict[str, np.ndarray],
    soft_threshold: float,
    strong_threshold: float,
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    """逐 epoch、逐探针图计算指标，并返回客户端热图矩阵。"""
    epoch_count, client_count, probe_count, _ = client_probabilities.shape
    metrics: List[Dict[str, object]] = []
    cloud_alignment_matrix = np.full((client_count, epoch_count), np.nan, dtype=np.float64)
    cloud_agreement_matrix = np.full((client_count, epoch_count), np.nan, dtype=np.float64)
    active_matrix = np.zeros((client_count, epoch_count), dtype=np.int8)
    for epoch_index in range(epoch_count):
        clients = client_probabilities[epoch_index]
        cloud = cloud_probabilities[epoch_index]
        labels = true_labels[epoch_index]
        groups = groups_by_round[epoch_index]
        active_slots = active_slots_by_round[epoch_index]
        active = clients[np.asarray(active_slots, dtype=int)] if active_slots else clients[:0]
        valid_edges = np.all(np.isfinite(edge_probabilities[epoch_index]), axis=(1, 2))
        edges = edge_probabilities[epoch_index, valid_edges]

        candidate_metrics = _population_sample_metrics(clients, labels)
        active_metrics = _population_sample_metrics(active, labels)
        edge_metrics = _population_sample_metrics(edges, labels)
        within_vote, within_divergence, within_score = compute_within_edge_metrics(clients, groups)
        candidate_divergence = 1.0 - candidate_metrics["agreement"]
        active_divergence = 1.0 - active_metrics["agreement"]
        edge_divergence = 1.0 - edge_metrics["agreement"]
        cloud_predictions = np.argmax(cloud, axis=1)
        client_predictions = np.argmax(clients, axis=2)
        valid_labels = labels >= 0

        client_cloud_divergence, client_divergences = _mean_pairwise_alignment(clients, cloud)
        edge_cloud_divergence, _ = _mean_pairwise_alignment(edges, cloud)
        cloud_alignment_matrix[:, epoch_index] = 1.0 - client_divergences
        cloud_agreement_matrix[:, epoch_index] = np.mean(
            client_predictions == cloud_predictions[None, :], axis=1
        )
        if active_slots:
            active_matrix[np.asarray(active_slots, dtype=int), epoch_index] = 1

        majority = candidate_metrics["majority"]
        vote_ratio = candidate_metrics["vote_ratio"]
        cloud_correct_by_probe = np.full(probe_count, np.nan, dtype=np.float64)
        cloud_correct_by_probe[valid_labels] = (
            cloud_predictions[valid_labels] == labels[valid_labels]
        ).astype(np.float64)
        soft_by_probe = (vote_ratio >= soft_threshold).astype(np.float64)
        strong_by_probe = (vote_ratio >= strong_threshold).astype(np.float64)
        active_coverage = float(len(active_slots)) / float(client_count)
        active_correct = safe_mean(active_metrics["correct"])
        coverage_weighted = active_coverage * active_correct if math.isfinite(active_correct) else float("nan")
        true_class_probability = (
            cloud[np.arange(probe_count)[valid_labels], labels[valid_labels]]
            if np.any(valid_labels)
            else np.empty(0, dtype=np.float64)
        )

        row: Dict[str, object] = {
            "epoch": int(global_epochs[epoch_index]) + 1,
            "probe_count": probe_count,
            "client_count": client_count,
            "active_client_count": len(active_slots),
            "active_client_coverage": active_coverage,
            "active_edge_count": int(edges.shape[0]),
            "client_vote_consensus_ratio": safe_mean(candidate_metrics["vote_ratio"]),
            "active_client_vote_consensus_ratio": safe_mean(active_metrics["vote_ratio"]),
            "within_edge_vote_consensus_ratio": within_vote,
            "edge_vote_consensus_ratio": safe_mean(edge_metrics["vote_ratio"]),
            "client_vote_entropy_norm": safe_mean(candidate_metrics["vote_entropy"]),
            "client_label_consensus_score": safe_mean(candidate_metrics["label_score"]),
            "active_client_vote_entropy_norm": safe_mean(active_metrics["vote_entropy"]),
            "active_client_label_consensus_score": safe_mean(active_metrics["label_score"]),
            "edge_vote_entropy_norm": safe_mean(edge_metrics["vote_entropy"]),
            "edge_label_consensus_score": safe_mean(edge_metrics["label_score"]),
            "client_probability_divergence_gjs": safe_mean(candidate_divergence),
            "client_probability_consensus_score": safe_mean(candidate_metrics["agreement"]),
            "client_certainty_score": safe_mean(candidate_metrics["certainty"]),
            "client_effective_consensus_score": safe_mean(candidate_metrics["effective"]),
            "client_correct_effective_consensus_score": safe_mean(candidate_metrics["correct"]),
            "client_wrong_effective_consensus_score": safe_mean(candidate_metrics["wrong"]),
            "client_effective_consensus_q25": float(np.quantile(candidate_metrics["effective"], 0.25)),
            "client_effective_consensus_q50": float(np.quantile(candidate_metrics["effective"], 0.50)),
            "client_effective_consensus_q75": float(np.quantile(candidate_metrics["effective"], 0.75)),
            "active_client_probability_divergence_gjs": safe_mean(active_divergence),
            "active_client_probability_consensus_score": safe_mean(active_metrics["agreement"]),
            "active_client_certainty_score": safe_mean(active_metrics["certainty"]),
            "active_client_effective_consensus_score": safe_mean(active_metrics["effective"]),
            "active_client_correct_effective_consensus_score": active_correct,
            "active_client_wrong_effective_consensus_score": safe_mean(active_metrics["wrong"]),
            "coverage_weighted_active_correct_effective_consensus": coverage_weighted,
            "within_edge_probability_divergence_gjs": within_divergence,
            "within_edge_probability_consensus_score": within_score,
            "edge_probability_divergence_gjs": safe_mean(edge_divergence),
            "edge_probability_consensus_score": safe_mean(edge_metrics["agreement"]),
            "edge_effective_consensus_score": safe_mean(edge_metrics["effective"]),
            "edge_correct_effective_consensus_score": safe_mean(edge_metrics["correct"]),
            "client_cloud_js_divergence": client_cloud_divergence,
            "client_cloud_alignment_score": 1.0 - client_cloud_divergence,
            "edge_cloud_js_divergence": edge_cloud_divergence,
            "edge_cloud_alignment_score": 1.0 - edge_cloud_divergence if math.isfinite(edge_cloud_divergence) else float("nan"),
            "client_cloud_label_agreement": float(np.mean(client_predictions == cloud_predictions[None, :])),
            "edge_cloud_label_agreement": safe_mean(
                [float(np.mean(np.argmax(edge, axis=1) == cloud_predictions)) for edge in edges]
            ),
            "majority_cloud_label_match": float(np.mean(majority == cloud_predictions)),
            "client_mean_confidence": float(np.mean(np.max(clients, axis=2))),
            "cloud_confidence": float(np.mean(np.max(cloud, axis=1))),
            "cloud_entropy_norm": safe_mean(normalized_entropy_array(cloud)),
            "cloud_margin": safe_mean([probability_margin(item) for item in cloud]),
            "majority_label": int(majority[0]) if probe_count == 1 else None,
            "cloud_predicted_label": int(cloud_predictions[0]) if probe_count == 1 else None,
            "true_label": int(labels[0]) if probe_count == 1 and labels[0] >= 0 else None,
            "soft_consensus": safe_mean(soft_by_probe),
            "strong_consensus": safe_mean(strong_by_probe),
            "cloud_correct": safe_mean(cloud_correct_by_probe),
            "cloud_probe_accuracy": safe_mean(cloud_correct_by_probe),
            "cloud_true_class_probability_mean": safe_mean(true_class_probability),
            "correct_soft_consensus": safe_mean(soft_by_probe * cloud_correct_by_probe),
            "wrong_soft_consensus": safe_mean(soft_by_probe * (1.0 - cloud_correct_by_probe)),
            "correct_strong_consensus": safe_mean(strong_by_probe * cloud_correct_by_probe),
            "wrong_strong_consensus": safe_mean(strong_by_probe * (1.0 - cloud_correct_by_probe)),
        }
        for metric_name, values in quality_series.items():
            row[metric_name] = float(values[epoch_index])
        metrics.append(row)
    return metrics, cloud_alignment_matrix, cloud_agreement_matrix, active_matrix


def csv_value(value: object) -> object:
    """把 None 和非有限数转换为空 CSV 单元格。"""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "{:.10f}".format(float(value)) if math.isfinite(float(value)) else ""
    return value


def write_metrics_csv(path: Path, metrics: Sequence[Dict[str, object]]) -> None:
    """写出逐 epoch 共识明细 CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def write_metric_definitions(path: Path) -> None:
    """写出核心指标的中文定义。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["metric", "name_zh", "formula", "range", "direction", "meaning"])
        writer.writeheader()
        writer.writerows(METRIC_DEFINITIONS)


def metric_array(metrics: Sequence[Dict[str, object]], metric_name: str) -> np.ndarray:
    """从逐轮字典抽取数值列，空值转换为 NaN。"""
    return np.asarray(
        [float("nan") if row.get(metric_name) is None else float(row[metric_name]) for row in metrics],
        dtype=np.float64,
    )


def write_summary_csv(path: Path, metrics: Sequence[Dict[str, object]]) -> None:
    """汇总核心指标的全程和前中后阶段统计量。"""
    definition_map = {item["metric"]: item for item in METRIC_DEFINITIONS}
    names = [item["metric"] for item in METRIC_DEFINITIONS] + [
        "active_client_coverage", "active_client_effective_consensus_score",
        "client_vote_consensus_ratio", "edge_effective_consensus_score", "test_acc", "test_loss",
    ]
    thirds = np.array_split(np.arange(len(metrics)), 3)
    fields = ["metric", "name_zh", "valid_rounds", "mean", "median", "min", "max", "first_third_mean", "middle_third_mean", "last_third_mean"]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for name in names:
            if name not in metrics[0]:
                continue
            values = metric_array(metrics, name)
            valid = values[np.isfinite(values)]
            if not valid.size:
                continue
            row = {
                "metric": name,
                "name_zh": definition_map.get(name, {}).get("name_zh", name),
                "valid_rounds": int(valid.size),
                "mean": float(np.mean(valid)), "median": float(np.median(valid)),
                "min": float(np.min(valid)), "max": float(np.max(valid)),
                "first_third_mean": safe_mean(values[thirds[0]]),
                "middle_third_mean": safe_mean(values[thirds[1]]),
                "last_third_mean": safe_mean(values[thirds[2]]),
            }
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def write_client_summary_csv(
    path: Path,
    cloud_alignment_matrix: np.ndarray,
    cloud_agreement_matrix: np.ndarray,
    active_matrix: np.ndarray,
    client_ids: Optional[np.ndarray] = None,
) -> None:
    """汇总每个候选客户端在活跃和非活跃 epoch 的云端对齐。"""
    ids = np.arange(cloud_alignment_matrix.shape[0]) if client_ids is None else client_ids
    fields = [
        "client_id", "active_rounds", "active_rate", "mean_probability_alignment_to_cloud",
        "median_probability_alignment_to_cloud", "cloud_label_agreement_rate",
        "mean_alignment_when_active", "mean_alignment_when_inactive",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for slot, client_id in enumerate(ids):
            alignment = cloud_alignment_matrix[slot]
            agreement = cloud_agreement_matrix[slot]
            active = active_matrix[slot].astype(bool)
            row = {
                "client_id": int(client_id), "active_rounds": int(np.sum(active)),
                "active_rate": float(np.mean(active)),
                "mean_probability_alignment_to_cloud": safe_mean(alignment),
                "median_probability_alignment_to_cloud": float(np.nanmedian(alignment)),
                "cloud_label_agreement_rate": safe_mean(agreement),
                "mean_alignment_when_active": safe_mean(alignment[active]),
                "mean_alignment_when_inactive": safe_mean(alignment[~active]),
            }
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def configure_plot_font() -> str:
    """选择可用中文字体并设置 Matplotlib 通用样式。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"]
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [selected, "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF",
    })
    return selected


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """计算完整尾随滑动平均，窗口未填满的位置保留为 NaN。"""
    array = np.asarray(values, dtype=np.float64)
    if window <= 0:
        raise ValueError("滑动窗口必须大于 0。")
    if window == 1:
        return array.copy()
    result = np.full(array.shape, np.nan, dtype=np.float64)
    for index in range(window - 1, array.size):
        # 时间窗口必须完整；窗口内部的指标空值按现有有效项求均值。
        result[index] = safe_mean(array[index - window + 1:index + 1])
    return result


def _style_axis(axis: plt.Axes, ylabel: str) -> None:
    """统一设置趋势图坐标轴样式。"""
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, 1.02)
    axis.grid(axis="y", color="#EAECF0", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_smoothed(axis: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str, window: int) -> None:
    """同时绘制半透明原始曲线和完整尾随滑动平均。"""
    axis.plot(x, y, color=color, linewidth=0.8, alpha=0.18)
    axis.plot(x, moving_average(y, window), color=color, linewidth=2.0, label=label)


def plot_consensus_overview(
    path: Path,
    metrics: Sequence[Dict[str, object]],
    smooth_window: int,
    soft_threshold: float,
    strong_threshold: float,
) -> None:
    """绘制候选/活跃共识、正确性护栏和测试性能总览。"""
    del soft_threshold, strong_threshold
    epochs = metric_array(metrics, "epoch")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    figure.suptitle("固定探针逐 epoch 共识诊断", fontsize=19, fontweight="bold")
    _plot_smoothed(axes[0, 0], epochs, metric_array(metrics, "client_probability_consensus_score"), "一致性 A", "#667085", smooth_window)
    _plot_smoothed(axes[0, 0], epochs, metric_array(metrics, "client_certainty_score"), "确定性 C", "#2F6BFF", smooth_window)
    _plot_smoothed(axes[0, 0], epochs, metric_array(metrics, "client_effective_consensus_score"), "纯有效共识 S", "#D99A21", smooth_window)
    axes[0, 0].set_title("候选客户端共识构成", loc="left", fontweight="bold")
    _style_axis(axes[0, 0], "得分")

    _plot_smoothed(axes[0, 1], epochs, metric_array(metrics, "client_correct_effective_consensus_score"), "候选正确 S", "#039855", smooth_window)
    _plot_smoothed(axes[0, 1], epochs, metric_array(metrics, "client_wrong_effective_consensus_score"), "候选错误 S", "#D92D20", smooth_window)
    _plot_smoothed(axes[0, 1], epochs, metric_array(metrics, "active_client_correct_effective_consensus_score"), "活跃正确 S", "#2F6BFF", smooth_window)
    axes[0, 1].set_title("正确与错误有效共识", loc="left", fontweight="bold")
    _style_axis(axes[0, 1], "得分")

    _plot_smoothed(axes[1, 0], epochs, metric_array(metrics, "active_client_coverage"), "活跃覆盖率", "#667085", smooth_window)
    _plot_smoothed(axes[1, 0], epochs, metric_array(metrics, "coverage_weighted_active_correct_effective_consensus"), "覆盖率×活跃正确 S", "#D99A21", smooth_window)
    axes[1, 0].set_title("参与机制", loc="left", fontweight="bold")
    _style_axis(axes[1, 0], "比例/得分")

    _plot_smoothed(axes[1, 1], epochs, metric_array(metrics, "cloud_probe_accuracy"), "固定探针云端准确率", "#2F6BFF", smooth_window)
    _plot_smoothed(axes[1, 1], epochs, metric_array(metrics, "test_acc"), "完整测试集准确率", "#039855", smooth_window)
    axes[1, 1].set_title("性能护栏", loc="left", fontweight="bold")
    _style_axis(axes[1, 1], "准确率")
    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.legend(frameon=False, fontsize=9)
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.94))
    figure.savefig(str(path), dpi=180, facecolor="white")
    plt.close(figure)


def plot_hierarchy_and_clients(
    path: Path,
    metrics: Sequence[Dict[str, object]],
    cloud_alignment_matrix: np.ndarray,
    cloud_agreement_matrix: np.ndarray,
    active_matrix: np.ndarray,
) -> None:
    """绘制参与规模以及逐候选客户端的云对齐与活跃热图。"""
    epochs = metric_array(metrics, "epoch")
    figure, axes = plt.subplots(4, 1, figsize=(15, 14), gridspec_kw={"height_ratios": [1.0, 1.5, 1.5, 1.5]})
    figure.suptitle("动态参与和逐客户端诊断", fontsize=19, fontweight="bold")
    axes[0].plot(epochs, metric_array(metrics, "active_client_count"), color="#2F6BFF", label="活跃客户端数")
    axes[0].set_ylabel("客户端数")
    axes[0].grid(axis="y", color="#EAECF0")
    axes[0].legend(frameon=False)
    images = [
        axes[1].imshow(cloud_alignment_matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="Blues"),
        axes[2].imshow(cloud_agreement_matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="Blues"),
        axes[3].imshow(active_matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="YlOrBr"),
    ]
    titles = ["逐客户端到云端的概率对齐", "逐客户端与云端标签一致比例", "逐客户端活跃状态"]
    for axis, image, title in zip(axes[1:], images, titles):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel("候选槽位")
        axis.set_xlabel("epoch 索引")
        figure.colorbar(image, ax=axis, fraction=0.018, pad=0.012)
    figure.tight_layout(rect=(0.03, 0.03, 0.99, 0.95))
    figure.savefig(str(path), dpi=180, facecolor="white")
    plt.close(figure)


def print_key_summary(metrics: Sequence[Dict[str, object]], output_dir: Path, source_format: str) -> None:
    """在终端打印固定探针最关键的共识与护栏摘要。"""
    print("共识分析完成（数据源：{}）。".format(source_format))
    for metric_name, label in [
        ("client_effective_consensus_score", "候选纯有效共识均值"),
        ("client_correct_effective_consensus_score", "候选正确有效共识均值"),
        ("client_wrong_effective_consensus_score", "候选错误有效共识均值"),
        ("coverage_weighted_active_correct_effective_consensus", "覆盖率加权活跃正确共识均值"),
        ("cloud_probe_accuracy", "固定探针云端准确率均值"),
    ]:
        print("  {}：{:.4f}".format(label, safe_mean(metric_array(metrics, metric_name))))
    print("  输出目录：{}".format(output_dir))


def main() -> None:
    """加载新旧探针、解析真实拓扑、计算指标并写出 CSV 和图片。"""
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    result_dir = find_latest_result_dir(script_dir / "result") if args.result_dir is None else resolve_input_path(args.result_dir, script_dir)
    if args.metadata_file is None:
        default_metadata = result_dir / PROBE_METADATA_FILENAME
        metadata_path = default_metadata if default_metadata.is_file() else None
    else:
        metadata_path = resolve_input_path(args.metadata_file, script_dir)
    probe_data = load_probe_data(result_dir, metadata_path)
    mat_path = resolve_input_path(args.mat_file, script_dir)
    groups, active_slots, schedule_client_ids = resolve_groups(result_dir, mat_path, probe_data)
    # 旧 CSV 没有真实客户端编号，使用运行时拓扑中的固定候选编号补全。
    if probe_data.source_format == "legacy_csv":
        probe_data.client_ids = schedule_client_ids
    epoch_count = probe_data.client_probabilities.shape[0]
    quality_series = {
        name: read_metric_series(result_dir / "{}.txt".format(name), epoch_count)
        for name in ["train_acc", "train_loss", "test_acc", "test_loss"]
    }
    metrics, alignment, agreement, active = compute_round_metrics(
        probe_data.client_probabilities,
        probe_data.edge_probabilities,
        probe_data.cloud_probabilities,
        groups,
        active_slots,
        probe_data.true_labels,
        probe_data.global_epochs,
        quality_series,
        args.soft_threshold,
        args.strong_threshold,
    )
    output_dir = (result_dir / "consensus_analysis") if args.output_dir is None else resolve_input_path(args.output_dir, result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_font()
    write_metrics_csv(output_dir / "consensus_metrics_by_round.csv", metrics)
    write_summary_csv(output_dir / "consensus_metrics_summary.csv", metrics)
    write_metric_definitions(output_dir / "consensus_metric_definitions.csv")
    write_client_summary_csv(output_dir / "consensus_client_summary.csv", alignment, agreement, active, probe_data.client_ids)
    plot_consensus_overview(output_dir / "consensus_overview.png", metrics, args.smooth_window, args.soft_threshold, args.strong_threshold)
    plot_hierarchy_and_clients(output_dir / "consensus_hierarchy_and_clients.png", metrics, alignment, agreement, active)
    print_key_summary(metrics, output_dir, probe_data.source_format)


if __name__ == "__main__":
    main()
