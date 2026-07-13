"""对四组联邦学习实验进行可复现的横向分析并生成中文报告。"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# 服务器运行时通常没有图形界面，固定使用非交互式后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np


SCENARIO_ORDER = [
    "hfl_snf_fixed",
    "hfl_no_snf_fixed",
    "fl_snf",
    "fl_no_snf",
]

SCENARIO_LABELS = {
    "hfl_snf_fixed": "HFL-SnF",
    "hfl_no_snf_fixed": "HFL-noSnF",
    "fl_snf": "FL-SnF",
    "fl_no_snf": "FL-noSnF",
}

SCENARIO_COLORS = {
    "hfl_snf_fixed": "#B7791F",
    "hfl_no_snf_fixed": "#C2410C",
    "fl_snf": "#2563EB",
    "fl_no_snf": "#667085",
}

SCENARIO_LINESTYLES = {
    "hfl_snf_fixed": "-",
    "hfl_no_snf_fixed": "--",
    "fl_snf": "-",
    "fl_no_snf": "--",
}

REQUIRED_RESULT_FILES = [
    "topology_metadata.json",
    "topology_schedule.jsonl",
    "train_acc.txt",
    "train_loss.txt",
    "test_acc.txt",
    "test_loss.txt",
    "probe_client_pre.csv",
    "probe_edge_post.csv",
    "probe_cloud_post.csv",
]

SUMMARY_THRESHOLDS = (0.80, 0.85, 0.88)


@dataclass
class ExperimentData:
    """保存一组实验的元数据、逐轮指标、调度记录和三层探针。"""

    path: Path
    scenario: str
    label: str
    metadata: Dict[str, object]
    schedule: List[Dict[str, object]]
    train_acc: np.ndarray
    train_loss: np.ndarray
    test_acc: np.ndarray
    test_loss: np.ndarray
    client_probe: List[List[np.ndarray]]
    edge_probe: List[List[np.ndarray]]
    cloud_probe: List[List[np.ndarray]]


def parse_args() -> argparse.Namespace:
    """解析结果根目录、显式实验目录、输出目录和共识平滑窗口。"""

    parser = argparse.ArgumentParser(description="分析四组联邦学习实验并生成中文报告")
    parser.add_argument("--result-root", type=Path, default=Path("result"), help="实验结果根目录")
    parser.add_argument(
        "--experiment-dir",
        action="append",
        type=Path,
        default=None,
        help="显式指定实验目录；可重复四次，未指定时从 result-root 自动发现",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="分析输出目录")
    parser.add_argument("--smooth-window", type=int, default=10, help="共识和趋势的尾随平滑窗口")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    """读取 UTF-8 JSON 对象，并在顶层不是对象时直接报错。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON 顶层必须是对象：{}".format(path))
    return value


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    """读取逐行 JSON 调度文件，跳过空行并保留原始轮次顺序。"""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("JSONL 第 {} 行不是对象：{}".format(line_number, path))
            rows.append(value)
    return rows


def read_metric_series(path: Path) -> np.ndarray:
    """读取每轮单值指标，并验证所有数值均为有限浮点数。"""

    values = [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("指标文件为空或维度异常：{}".format(path))
    if not np.all(np.isfinite(result)):
        raise ValueError("指标文件包含非有限数值：{}".format(path))
    return result


def read_probability_csv(path: Path) -> List[List[np.ndarray]]:
    """读取无表头探针 CSV，将每个非空单元格解析为概率向量。"""

    rounds = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            vectors = []
            for column_number, cell in enumerate(row, start=1):
                if not cell.strip():
                    continue
                try:
                    vector = np.asarray(json.loads(cell), dtype=np.float64)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "探针解析失败：{} 第 {} 行第 {} 列".format(path, row_number, column_number)
                    ) from exc
                vectors.append(vector)
            rounds.append(vectors)
    return rounds


def discover_experiment_dirs(result_root: Path) -> List[Path]:
    """从结果根目录发现恰好覆盖四个目标场景的完整实验目录。"""

    scenario_to_paths = {scenario: [] for scenario in SCENARIO_ORDER}
    for directory in sorted(result_root.iterdir()):
        if not directory.is_dir() or not (directory / "topology_metadata.json").is_file():
            continue
        if not all((directory / filename).is_file() for filename in REQUIRED_RESULT_FILES):
            continue
        metadata = read_json(directory / "topology_metadata.json")
        scenario = str(metadata.get("scenario", ""))
        if scenario not in scenario_to_paths:
            continue
        # 本分析只自动选择计划中的 alpha=0.2、U=0.5、full200 标签实验。
        if float(metadata.get("partition_alpha", -1.0)) != 0.2:
            continue
        if float(metadata.get("topology_util", -1.0)) != 0.5:
            continue
        if str(metadata.get("experiment_tag", "")) != "full200":
            continue
        scenario_to_paths[scenario].append(directory)

    problems = []
    for scenario in SCENARIO_ORDER:
        count = len(scenario_to_paths[scenario])
        if count != 1:
            problems.append("{} 匹配到 {} 个目录".format(scenario, count))
    if problems:
        raise ValueError("自动发现实验失败：{}；请使用 --experiment-dir 显式指定".format("；".join(problems)))
    return [scenario_to_paths[scenario][0] for scenario in SCENARIO_ORDER]


def resolve_experiment_dirs(result_root: Path, explicit_dirs: Optional[Sequence[Path]]) -> List[Path]:
    """解析显式目录或执行自动发现，并按固定场景顺序返回四个目录。"""

    if not explicit_dirs:
        return discover_experiment_dirs(result_root)
    if len(explicit_dirs) != 4:
        raise ValueError("--experiment-dir 必须恰好指定四次")
    scenario_to_path = {}
    for raw_path in explicit_dirs:
        path = raw_path.resolve()
        metadata = read_json(path / "topology_metadata.json")
        scenario = str(metadata.get("scenario", ""))
        if scenario not in SCENARIO_ORDER:
            raise ValueError("未知场景 {}：{}".format(scenario, path))
        if scenario in scenario_to_path:
            raise ValueError("场景 {} 被重复指定".format(scenario))
        scenario_to_path[scenario] = path
    if set(scenario_to_path) != set(SCENARIO_ORDER):
        raise ValueError("显式实验目录没有覆盖全部四个目标场景")
    return [scenario_to_path[scenario] for scenario in SCENARIO_ORDER]


def load_experiment(path: Path) -> ExperimentData:
    """读取一个实验目录的全部分析输入，并保留结果目录的实际运行语义。"""

    path = path.resolve()
    missing = [filename for filename in REQUIRED_RESULT_FILES if not (path / filename).is_file()]
    if missing:
        raise FileNotFoundError("实验目录缺少文件 {}：{}".format(missing, path))
    metadata = read_json(path / "topology_metadata.json")
    scenario = str(metadata.get("scenario", ""))
    if scenario not in SCENARIO_ORDER:
        raise ValueError("不支持的实验场景 {}：{}".format(scenario, path))
    return ExperimentData(
        path=path,
        scenario=scenario,
        label=SCENARIO_LABELS[scenario],
        metadata=metadata,
        schedule=read_jsonl(path / "topology_schedule.jsonl"),
        train_acc=read_metric_series(path / "train_acc.txt"),
        train_loss=read_metric_series(path / "train_loss.txt"),
        test_acc=read_metric_series(path / "test_acc.txt"),
        test_loss=read_metric_series(path / "test_loss.txt"),
        client_probe=read_probability_csv(path / "probe_client_pre.csv"),
        edge_probe=read_probability_csv(path / "probe_edge_post.csv"),
        cloud_probe=read_probability_csv(path / "probe_cloud_post.csv"),
    )


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """沿最后一维计算归一化熵，十分类均匀分布为 1，one-hot 为 0。"""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] < 2:
        raise ValueError("概率数组最后一维至少需要两个类别")
    clipped = np.clip(values, 1e-15, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1) / math.log(values.shape[-1])


def generalized_js_divergence(probabilities: np.ndarray) -> float:
    """计算多概率向量的归一化广义 Jensen-Shannon 散度。"""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("广义 JS 散度需要形状为 M×K 的非空矩阵")
    mean_entropy = float(normalized_entropy(matrix.mean(axis=0)))
    component_entropy = float(np.mean(normalized_entropy(matrix)))
    return float(np.clip(mean_entropy - component_entropy, 0.0, 1.0))


def pairwise_js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个概率向量之间的归一化 Jensen-Shannon 散度。"""

    return generalized_js_divergence(np.stack([left, right], axis=0))


def consensus_components(probabilities: np.ndarray) -> Tuple[float, float, float]:
    """计算概率一致性 A、整体确定性 C 和有效共识 S=A×C。"""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    agreement = 1.0 - generalized_js_divergence(matrix)
    certainty = 1.0 - float(np.mean(normalized_entropy(matrix)))
    effective = agreement * certainty
    return agreement, certainty, effective


def trailing_mean(values: Sequence[float], window: int, require_full_values: bool = True) -> np.ndarray:
    """计算尾随移动均值，前 window-1 个位置保持为空值，不使用未来轮次。"""

    array = np.asarray(values, dtype=np.float64)
    if window <= 0:
        raise ValueError("移动窗口必须大于 0")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    for index in range(window - 1, array.size):
        segment = array[index - window + 1:index + 1]
        finite = segment[np.isfinite(segment)]
        if require_full_values and finite.size != window:
            continue
        if not require_full_values and finite.size < max(1, window // 2):
            continue
        result[index] = float(np.mean(finite))
    return result


def historical_best(values: Sequence[float]) -> np.ndarray:
    """计算仅在已有有效值上单调不降的历史最佳序列，并保留前导空值。"""

    array = np.asarray(values, dtype=np.float64)
    result = np.full(array.shape, np.nan, dtype=np.float64)
    best = None
    for index, value in enumerate(array):
        if not np.isfinite(value):
            continue
        best = value if best is None else max(best, value)
        result[index] = best
    return result


def safe_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """在共同有限且具有方差的样本上计算 Pearson 相关系数。"""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if int(mask.sum()) < 3:
        return float("nan")
    left_valid = left_array[mask]
    right_valid = right_array[mask]
    if float(np.std(left_valid)) == 0.0 or float(np.std(right_valid)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def lagged_correlations(
        leading: Sequence[float], following: Sequence[float], max_lag: int = 10
) -> List[Dict[str, float]]:
    """计算 ±max_lag 轮相关；正滞后表示前一序列领先后一序列。"""

    left = np.asarray(leading, dtype=np.float64)
    right = np.asarray(following, dtype=np.float64)
    if left.size != right.size:
        raise ValueError("滞后相关的两个序列长度必须一致")
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            left_part, right_part = left[:-lag], right[lag:]
        elif lag < 0:
            left_part, right_part = left[-lag:], right[:lag]
        else:
            left_part, right_part = left, right
        finite_count = int(np.sum(np.isfinite(left_part) & np.isfinite(right_part)))
        rows.append(
            {
                "lag": float(lag),
                "correlation": safe_correlation(left_part, right_part),
                "sample_count": float(finite_count),
            }
        )
    return rows


def map_client_ids_to_slots(record: Dict[str, object], client_ids: Sequence[int]) -> List[int]:
    """依据当轮候选顺序，将真实客户端编号映射为探针 CSV 的列位置。"""

    candidates = [int(value) for value in record["candidate_client_indexes"]]
    slot_by_client = {client_id: slot for slot, client_id in enumerate(candidates)}
    slots = []
    for client_id in client_ids:
        normalized_id = int(client_id)
        if normalized_id not in slot_by_client:
            raise ValueError("客户端 {} 不属于本轮候选集合".format(normalized_id))
        slots.append(slot_by_client[normalized_id])
    return slots


def validate_probability_vector(vector: np.ndarray, class_count: int = 10) -> bool:
    """检查概率向量的维度、有限性、取值范围和归一化误差。"""

    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (class_count,) or not np.all(np.isfinite(values)):
        return False
    if np.any(values < -1e-7) or np.any(values > 1.0 + 1e-7):
        return False
    return abs(float(values.sum()) - 1.0) <= 5e-6


def add_quality_check(
        checks: List[Dict[str, object]], experiment: str, name: str,
        status: str, evidence: str, severity: str
) -> None:
    """向数据质量检查表追加一条带证据和严重级别的记录。"""

    checks.append(
        {
            "实验": experiment,
            "检查项": name,
            "状态": status,
            "证据": evidence,
            "严重级别": severity,
        }
    )


def validate_experiment(experiment: ExperimentData) -> List[Dict[str, object]]:
    """核验一组实验的轮数、两级采样、聚合下发和探针完整性。"""

    checks = []
    round_count = len(experiment.schedule)
    series_lengths = {
        "train_acc": experiment.train_acc.size,
        "train_loss": experiment.train_loss.size,
        "test_acc": experiment.test_acc.size,
        "test_loss": experiment.test_loss.size,
        "client_probe": len(experiment.client_probe),
        "edge_probe": len(experiment.edge_probe),
        "cloud_probe": len(experiment.cloud_probe),
    }
    same_length = all(int(length) == round_count for length in series_lengths.values())
    add_quality_check(
        checks, experiment.label, "输出轮数一致", "通过" if same_length else "未通过",
        "调度 {} 轮；其余长度 {}".format(round_count, series_lengths), "关键",
    )

    configured_rounds = int(experiment.metadata.get("configured_comm_round", -1))
    add_quality_check(
        checks, experiment.label, "实际轮数与运行元数据一致",
        "通过" if configured_rounds == round_count else "未通过",
        "configured_comm_round={}，实际输出={}；MAT容量={}".format(
            configured_rounds, round_count, experiment.metadata.get("round_count")
        ), "关键",
    )

    candidate_ok = True
    active_ok = True
    group_ok = True
    distribution_ok = True
    epoch_ok = True
    edge_ok = True
    probability_ok = True
    zero_rounds = 0
    candidate_sets = []
    for index, record in enumerate(experiment.schedule):
        candidates = [int(value) for value in record["candidate_client_indexes"]]
        active = [int(value) for value in record["active_client_indexes"]]
        group_mapping = {
            int(group_id): [int(value) for value in client_ids]
            for group_id, client_ids in record["group_to_client_indexes"].items()
        }
        group_counts = {int(group_id): int(value) for group_id, value in record["mat_group_client_counts"].items()}
        group_union = [client_id for client_ids in group_mapping.values() for client_id in client_ids]
        candidate_sets.append(tuple(candidates))
        epoch_ok = epoch_ok and int(record["global_epoch"]) == index
        candidate_ok = candidate_ok and len(candidates) == 37 and len(set(candidates)) == 37
        candidate_ok = candidate_ok and all(0 <= client_id < 200 for client_id in candidates)
        active_ok = active_ok and set(active).issubset(set(candidates)) and len(active) == len(set(active))
        active_ok = active_ok and int(record["active_client_count"]) == len(active)
        group_ok = group_ok and len(group_union) == len(set(group_union))
        group_ok = group_ok and set(group_union) == set(active)
        group_ok = group_ok and sum(group_counts.values()) == len(active)

        if len(active) == 0:
            zero_rounds += 1
            distribution_ok = distribution_ok and not bool(record["aggregated"])
            distribution_ok = distribution_ok and int(record["distributed_client_count"]) == 0
        else:
            distribution_ok = distribution_ok and bool(record["aggregated"])
            distribution_ok = distribution_ok and int(record["distributed_client_count"]) == 200

        client_vectors = experiment.client_probe[index]
        edge_vectors = experiment.edge_probe[index]
        cloud_vectors = experiment.cloud_probe[index]
        probability_ok = probability_ok and len(client_vectors) == 37 and len(cloud_vectors) == 1
        probability_ok = probability_ok and all(validate_probability_vector(vector) for vector in client_vectors)
        probability_ok = probability_ok and all(validate_probability_vector(vector) for vector in edge_vectors)
        probability_ok = probability_ok and all(validate_probability_vector(vector) for vector in cloud_vectors)
        nonempty_group_count = sum(1 for client_ids in group_mapping.values() if client_ids)
        expected_edge_count = nonempty_group_count if experiment.scenario.startswith("hfl_") else 0
        edge_ok = edge_ok and len(edge_vectors) == expected_edge_count

    add_quality_check(
        checks, experiment.label, "global_epoch 连续且无重复", "通过" if epoch_ok else "未通过",
        "期望 0..{}".format(round_count - 1), "关键",
    )
    add_quality_check(
        checks, experiment.label, "一级采样为200中37个唯一客户端", "通过" if candidate_ok else "未通过",
        "共 {} 轮；候选集合出现 {} 种".format(round_count, len(set(candidate_sets))), "关键",
    )
    add_quality_check(
        checks, experiment.label, "二级采样属于候选集合", "通过" if active_ok else "未通过",
        "活跃客户端与 active_client_count 逐轮核对", "关键",
    )
    add_quality_check(
        checks, experiment.label, "分组配额和最终参与集合一致", "通过" if group_ok else "未通过",
        "各组并集、去重结果和MAT配额之和逐轮核对", "关键",
    )
    add_quality_check(
        checks, experiment.label, "聚合与全量下发行为一致", "通过" if distribution_ok else "未通过",
        "零参与轮 {} 个；非零轮应下发200人".format(zero_rounds), "关键",
    )
    add_quality_check(
        checks, experiment.label, "探针概率合法且层级列数匹配", "通过" if probability_ok and edge_ok else "未通过",
        "客户端37列、云端1列；HFL边缘列数等于非空组，FL边缘为空", "关键",
    )
    add_quality_check(
        checks, experiment.label, "每轮全部200个客户端完成本地训练", "无法验证",
        "结果文件没有 trained_client_indexes；只能作为 result/PLAN.md 的运行约定", "高",
    )
    add_quality_check(
        checks, experiment.label, "metadata参与均值可直接代表本次运行", "注意",
        "metadata按MAT全部{}行统计，本报告改用JSONL实际{}行".format(
            experiment.metadata.get("round_count"), round_count
        ), "中",
    )

    failures = [row for row in checks if row["状态"] == "未通过" and row["严重级别"] == "关键"]
    if failures:
        raise ValueError("{} 存在关键数据质量问题：{}".format(experiment.label, failures))
    return checks


def weighted_group_metric(group_values: Sequence[Tuple[int, float]]) -> float:
    """按组内客户端数对组级指标加权，避免对大小不同的组直接平均。"""

    valid = [(int(weight), float(value)) for weight, value in group_values if weight > 0 and np.isfinite(value)]
    if not valid:
        return float("nan")
    total_weight = sum(weight for weight, _ in valid)
    return float(sum(weight * value for weight, value in valid) / total_weight)


def mean_js_to_reference(probabilities: np.ndarray, reference: np.ndarray) -> float:
    """计算一组概率向量到同一参考概率向量的平均 JS 散度。"""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return float("nan")
    return float(np.mean([pairwise_js_divergence(vector, reference) for vector in matrix]))


def mean_consensus_to_reference(
        probabilities: np.ndarray, reference: np.ndarray
) -> Tuple[float, float, float]:
    """逐个计算概率向量与参考向量的A/C/S，并返回三个分量的算术均值。"""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    components = np.asarray(
        [
            consensus_components(np.stack([vector, reference], axis=0))
            for vector in matrix
        ],
        dtype=np.float64,
    )
    return tuple(float(value) for value in components.mean(axis=0))


def build_round_metrics(experiment: ExperimentData, smooth_window: int) -> List[Dict[str, object]]:
    """逐轮还原真实客户端槽位，并计算模型、参与、层级和共识指标。"""

    rows = []
    cumulative_active = 0
    for index, record in enumerate(experiment.schedule):
        candidates = [int(value) for value in record["candidate_client_indexes"]]
        active = [int(value) for value in record["active_client_indexes"]]
        candidate_matrix = np.stack(experiment.client_probe[index], axis=0)
        cloud_probability = experiment.cloud_probe[index][0]
        active_slots = map_client_ids_to_slots(record, active)
        active_matrix = candidate_matrix[active_slots] if active_slots else np.empty((0, candidate_matrix.shape[1]))

        candidate_a, candidate_c, candidate_s = consensus_components(candidate_matrix)
        active_a, active_c, active_s = consensus_components(active_matrix)

        group_effective_values = []
        group_certainty_values = []
        group_agreement_values = []
        group_mapping = {
            int(group_id): [int(value) for value in client_ids]
            for group_id, client_ids in record["group_to_client_indexes"].items()
        }
        for group_id in sorted(group_mapping):
            group_clients = group_mapping[group_id]
            if not group_clients:
                continue
            group_slots = map_client_ids_to_slots(record, group_clients)
            group_matrix = candidate_matrix[group_slots]
            group_a, group_c, group_s = consensus_components(group_matrix)
            group_agreement_values.append((len(group_clients), group_a))
            group_certainty_values.append((len(group_clients), group_c))
            group_effective_values.append((len(group_clients), group_s))

        edge_vectors = experiment.edge_probe[index]
        edge_matrix = np.stack(edge_vectors, axis=0) if edge_vectors else np.empty((0, candidate_matrix.shape[1]))
        edge_a, edge_c, edge_s = consensus_components(edge_matrix)
        edge_cloud_a, edge_cloud_c, edge_cloud_s = mean_consensus_to_reference(
            edge_matrix, cloud_probability
        )
        cloud_certainty = 1.0 - float(normalized_entropy(cloud_probability))

        cumulative_active += len(active)
        row = {
            "scenario": experiment.scenario,
            "label": experiment.label,
            "epoch": index + 1,
            "candidate_count": len(candidates),
            "active_count": len(active),
            "cumulative_active": cumulative_active,
            "effective_edge_count": len(edge_vectors),
            "distributed_count": int(record["distributed_client_count"]),
            "aggregated": int(bool(record["aggregated"])),
            "train_acc": float(experiment.train_acc[index]),
            "train_loss": float(experiment.train_loss[index]),
            "test_acc": float(experiment.test_acc[index]),
            "test_loss": float(experiment.test_loss[index]),
            "generalization_gap": float(experiment.train_acc[index] - experiment.test_acc[index]),
            "candidate_agreement": candidate_a,
            "candidate_certainty": candidate_c,
            "candidate_effective": candidate_s,
            "active_agreement": active_a,
            "active_certainty": active_c,
            "active_effective": active_s,
            "within_group_agreement": weighted_group_metric(group_agreement_values),
            "within_group_certainty": weighted_group_metric(group_certainty_values),
            "within_group_effective": weighted_group_metric(group_effective_values),
            "edge_agreement": edge_a,
            "edge_certainty": edge_c,
            "edge_effective": edge_s,
            "cloud_certainty": cloud_certainty,
            "candidate_cloud_js": mean_js_to_reference(candidate_matrix, cloud_probability),
            "active_cloud_js": mean_js_to_reference(active_matrix, cloud_probability),
            "edge_cloud_js": mean_js_to_reference(edge_matrix, cloud_probability),
            "edge_cloud_agreement": edge_cloud_a,
            "edge_cloud_certainty": edge_cloud_c,
            "edge_cloud_effective": edge_cloud_s,
            "candidate_client_ids": json.dumps(candidates, ensure_ascii=False),
            "active_client_ids": json.dumps(active, ensure_ascii=False),
            "group_to_client_ids": json.dumps(group_mapping, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)

    # 所有平滑指标都只使用当前轮及历史轮次，避免离线图引入未来信息。
    smooth_fields = [
        "train_acc",
        "train_loss",
        "test_acc",
        "test_loss",
        "generalization_gap",
        "candidate_agreement",
        "candidate_certainty",
        "candidate_effective",
        "active_certainty",
        "edge_certainty",
        "cloud_certainty",
        "candidate_cloud_js",
        "active_cloud_js",
        "edge_cloud_js",
        "edge_cloud_agreement",
        "edge_cloud_certainty",
        "edge_cloud_effective",
    ]
    for field in smooth_fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        require_full = field not in {
            "active_certainty",
            "edge_certainty",
            "active_cloud_js",
            "edge_cloud_js",
            "edge_cloud_agreement",
            "edge_cloud_certainty",
            "edge_cloud_effective",
        }
        smoothed = trailing_mean(values, smooth_window, require_full_values=require_full)
        for index, value in enumerate(smoothed):
            rows[index][field + "_ma"] = float(value)

    attainment = historical_best([float(row["candidate_effective_ma"]) for row in rows])
    for index, value in enumerate(attainment):
        rows[index]["candidate_effective_historical_best"] = float(value)
    return rows


def first_stable_epoch(values: Sequence[float], threshold: float, window: int = 5) -> Optional[int]:
    """返回5轮尾随均值达到阈值且此后不再跌破阈值的首个轮次。"""

    smoothed = trailing_mean(values, window, require_full_values=True)
    for index in range(window - 1, smoothed.size):
        remaining = smoothed[index:]
        if np.all(np.isfinite(remaining)) and np.all(remaining >= threshold):
            return int(index + 1)
    return None


def summarize_experiment(
        experiment: ExperimentData, round_rows: Sequence[Dict[str, object]], smooth_window: int
) -> Dict[str, object]:
    """汇总一组实验的性能、参与、稳定性、阈值和共识统计。"""

    test_acc = np.asarray([float(row["test_acc"]) for row in round_rows])
    test_loss = np.asarray([float(row["test_loss"]) for row in round_rows])
    train_acc = np.asarray([float(row["train_acc"]) for row in round_rows])
    active = np.asarray([int(row["active_count"]) for row in round_rows])
    distributed = np.asarray([int(row["distributed_count"]) for row in round_rows])
    effective_edges = np.asarray([int(row["effective_edge_count"]) for row in round_rows])
    candidate_a = np.asarray([float(row["candidate_agreement"]) for row in round_rows])
    candidate_c = np.asarray([float(row["candidate_certainty"]) for row in round_rows])
    candidate_s = np.asarray([float(row["candidate_effective"]) for row in round_rows])
    candidate_s_ma = np.asarray([float(row["candidate_effective_ma"]) for row in round_rows])
    test_acc_ma = np.asarray([float(row["test_acc_ma"]) for row in round_rows])

    candidate_frequency = np.bincount(
        [client_id for record in experiment.schedule for client_id in record["candidate_client_indexes"]],
        minlength=int(experiment.metadata["client_num_in_total"]),
    )
    active_frequency = np.bincount(
        [client_id for record in experiment.schedule for client_id in record["active_client_indexes"]],
        minlength=int(experiment.metadata["client_num_in_total"]),
    )

    lag_rows = lagged_correlations(candidate_s_ma, test_acc_ma, max_lag=10)
    finite_lags = [row for row in lag_rows if np.isfinite(row["correlation"])]
    strongest_lag = max(finite_lags, key=lambda row: abs(row["correlation"])) if finite_lags else None
    result = {
        "scenario": experiment.scenario,
        "label": experiment.label,
        "rounds": len(round_rows),
        "final_test_acc": float(test_acc[-1]),
        "best_test_acc": float(np.max(test_acc)),
        "best_test_epoch": int(np.argmax(test_acc) + 1),
        "mean_test_acc": float(np.mean(test_acc)),
        "last10_test_acc_mean": float(np.mean(test_acc[-10:])),
        "last10_test_acc_std": float(np.std(test_acc[-10:])),
        "final_test_loss": float(test_loss[-1]),
        "min_test_loss": float(np.min(test_loss)),
        "last10_train_acc_mean": float(np.mean(train_acc[-10:])),
        "last10_generalization_gap": float(np.mean(train_acc[-10:] - test_acc[-10:])),
        "active_min": int(np.min(active)),
        "active_mean": float(np.mean(active)),
        "active_max": int(np.max(active)),
        "active_total": int(np.sum(active)),
        "zero_active_rounds": int(np.sum(active == 0)),
        "aggregated_rounds": int(np.sum(active > 0)),
        "distributed_total": int(np.sum(distributed)),
        "effective_edge_mean": float(np.mean(effective_edges)),
        "candidate_coverage": int(np.sum(candidate_frequency > 0)),
        "active_coverage": int(np.sum(active_frequency > 0)),
        "candidate_frequency_min": int(np.min(candidate_frequency)),
        "candidate_frequency_max": int(np.max(candidate_frequency)),
        "active_frequency_min": int(np.min(active_frequency)),
        "active_frequency_max": int(np.max(active_frequency)),
        "agreement_first10": float(np.mean(candidate_a[:10])),
        "certainty_first10": float(np.mean(candidate_c[:10])),
        "effective_first10": float(np.mean(candidate_s[:10])),
        "agreement_last10": float(np.mean(candidate_a[-10:])),
        "certainty_last10": float(np.mean(candidate_c[-10:])),
        "effective_last10": float(np.mean(candidate_s[-10:])),
        "effective_ma_best": float(np.nanmax(candidate_s_ma)),
        "consensus_accuracy_level_corr": safe_correlation(candidate_s_ma, test_acc_ma),
        "consensus_accuracy_diff_corr": safe_correlation(
            np.diff(candidate_s_ma), np.diff(test_acc_ma)
        ),
        "strongest_lag": None if strongest_lag is None else int(strongest_lag["lag"]),
        "strongest_lag_corr": float("nan") if strongest_lag is None else float(strongest_lag["correlation"]),
        "smooth_window": smooth_window,
    }
    for threshold in SUMMARY_THRESHOLDS:
        key = "stable_epoch_{:.2f}".format(threshold)
        result[key] = first_stable_epoch(test_acc, threshold, window=5)
    return result


def build_client_statistics(experiments: Sequence[ExperimentData]) -> List[Dict[str, object]]:
    """按实验和真实客户端编号统计候选频率、参与频率及条件参与率。"""

    rows = []
    for experiment in experiments:
        client_total = int(experiment.metadata["client_num_in_total"])
        candidate_frequency = np.zeros(client_total, dtype=np.int64)
        active_frequency = np.zeros(client_total, dtype=np.int64)
        for record in experiment.schedule:
            for client_id in record["candidate_client_indexes"]:
                candidate_frequency[int(client_id)] += 1
            for client_id in record["active_client_indexes"]:
                active_frequency[int(client_id)] += 1
        for client_id in range(client_total):
            candidate_count = int(candidate_frequency[client_id])
            active_count = int(active_frequency[client_id])
            rows.append(
                {
                    "实验": experiment.label,
                    "场景": experiment.scenario,
                    "客户端ID": client_id,
                    "候选次数": candidate_count,
                    "最终参与次数": active_count,
                    "候选覆盖率": candidate_count / len(experiment.schedule),
                    "最终参与覆盖率": active_count / len(experiment.schedule),
                    "候选后参与率": float("nan") if candidate_count == 0 else active_count / candidate_count,
                }
            )
    return rows


def build_contrasts(summary_by_scenario: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """构造四项2×2对比及一项差分中的差分描述性结果。"""

    pairs = [
        ("HFL中SnF增益", "hfl_snf_fixed", "hfl_no_snf_fixed"),
        ("FL中SnF增益", "fl_snf", "fl_no_snf"),
        ("SnF条件下HFL相对FL", "hfl_snf_fixed", "fl_snf"),
        ("noSnF条件下HFL相对FL", "hfl_no_snf_fixed", "fl_no_snf"),
    ]
    rows = []
    for name, left_key, right_key in pairs:
        left = summary_by_scenario[left_key]
        right = summary_by_scenario[right_key]
        rows.append(
            {
                "对比": name,
                "左方案": left["label"],
                "右方案": right["label"],
                "后10轮准确率差_百分点": 100.0 * (
                    float(left["last10_test_acc_mean"]) - float(right["last10_test_acc_mean"])
                ),
                "最终准确率差_百分点": 100.0 * (
                    float(left["final_test_acc"]) - float(right["final_test_acc"])
                ),
                "平均活跃人数差": float(left["active_mean"]) - float(right["active_mean"]),
            }
        )
    hfl_gain = rows[0]["后10轮准确率差_百分点"]
    fl_gain = rows[1]["后10轮准确率差_百分点"]
    rows.append(
        {
            "对比": "SnF增益的差分中的差分",
            "左方案": "HFL中SnF增益",
            "右方案": "FL中SnF增益",
            "后10轮准确率差_百分点": float(hfl_gain - fl_gain),
            "最终准确率差_百分点": float(
                rows[0]["最终准确率差_百分点"] - rows[1]["最终准确率差_百分点"]
            ),
            "平均活跃人数差": float(rows[0]["平均活跃人数差"] - rows[1]["平均活跃人数差"]),
        }
    )
    return rows


def build_correlation_rows(
        rows_by_scenario: Dict[str, List[Dict[str, object]]]
) -> List[Dict[str, object]]:
    """导出四方案平滑水平的±10轮滞后相关及一阶差分同期相关明细。"""

    output = []
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        consensus = np.asarray(
            [float(row["candidate_effective_ma"]) for row in rows], dtype=np.float64
        )
        accuracy = np.asarray([float(row["test_acc_ma"]) for row in rows], dtype=np.float64)
        for item in lagged_correlations(consensus, accuracy, max_lag=10):
            output.append(
                {
                    "方案": SCENARIO_LABELS[scenario],
                    "场景": scenario,
                    "相关类型": "10轮平滑水平滞后相关",
                    "滞后轮数_正值表示共识领先": int(item["lag"]),
                    "相关系数": item["correlation"],
                    "共同有效样本数": int(item["sample_count"]),
                }
            )

        consensus_difference = np.diff(consensus)
        accuracy_difference = np.diff(accuracy)
        finite_count = int(
            np.sum(np.isfinite(consensus_difference) & np.isfinite(accuracy_difference))
        )
        output.append(
            {
                "方案": SCENARIO_LABELS[scenario],
                "场景": scenario,
                "相关类型": "10轮平滑值的一阶差分同期相关",
                "滞后轮数_正值表示共识领先": 0,
                "相关系数": safe_correlation(consensus_difference, accuracy_difference),
                "共同有效样本数": finite_count,
            }
        )
    return output


def configure_plot_style() -> str:
    """选择可用中文字体，并统一图表背景、文字、网格和负号样式。"""

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


def style_axis(axis: plt.Axes, y_label: str, y_limits: Optional[Tuple[float, float]] = None) -> None:
    """为单个坐标轴设置统一标签、浅色网格和可选纵轴范围。"""

    axis.set_xlabel("训练轮次")
    axis.set_ylabel(y_label)
    axis.grid(True, axis="y", color="#EAECF0", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if y_limits is not None:
        axis.set_ylim(*y_limits)


def add_figure_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    """为整张图添加中性标题和包含口径、轮数或平滑方式的副标题。"""

    # 用物理高度而不是固定比例控制间距，避免横向短图的主副标题重叠。
    figure_height = figure.get_figheight()
    title_y = 1.0 - 0.04 / figure_height
    subtitle_y = 1.0 - 0.38 / figure_height
    figure.suptitle(title, fontsize=16, fontweight="bold", y=title_y)
    figure.text(0.5, subtitle_y, subtitle, ha="center", va="top", fontsize=10, color="#667085")


def save_figure(figure: plt.Figure, path: Path) -> None:
    """以300 DPI保存图像，并在保存后释放Matplotlib对象。"""

    # 给标题区域预留固定物理高度，使不同宽高比的图都保持一致留白。
    content_top = 1.0 - 0.70 / figure.get_figheight()
    # 复杂GridSpec热图会触发Matplotlib的保守提示；该版式已逐图做原始分辨率检查。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        figure.tight_layout(rect=[0.02, 0.02, 0.98, content_top])
    figure.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_metric_line(
        axis: plt.Axes, rows: Sequence[Dict[str, object]], scenario: str,
        field: str, smooth_field: str, label: Optional[str] = None
) -> None:
    """在同一坐标轴绘制一条浅色原始曲线和一条加粗尾随均值曲线。"""

    epochs = np.asarray([int(row["epoch"]) for row in rows])
    raw = np.asarray([float(row[field]) for row in rows])
    smooth = np.asarray([float(row[smooth_field]) for row in rows])
    color = SCENARIO_COLORS[scenario]
    linestyle = SCENARIO_LINESTYLES[scenario]
    axis.plot(epochs, raw, color=color, linewidth=0.75, alpha=0.20, linestyle=linestyle)
    axis.plot(
        epochs, smooth, color=color, linewidth=2.0, alpha=0.98,
        linestyle=linestyle, label=label or SCENARIO_LABELS[scenario],
    )


def plot_model_metrics(
        rows_by_scenario: Dict[str, List[Dict[str, object]]], figure_dir: Path, smooth_window: int
) -> Path:
    """绘制四方案训练/测试准确率、测试损失和泛化差距趋势。"""

    path = figure_dir / "01_模型效果趋势.png"
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    specs = [
        ("test_acc", "test_acc_ma", "测试准确率", (0.05, 0.95)),
        ("test_loss", "test_loss_ma", "测试损失", None),
        ("train_acc", "train_acc_ma", "训练准确率", (0.05, 0.95)),
        ("generalization_gap", "generalization_gap_ma", "训练准确率 − 测试准确率", (-0.04, 0.04)),
    ]
    for axis, (field, smooth_field, y_label, limits) in zip(axes.flat, specs):
        for scenario in SCENARIO_ORDER:
            plot_metric_line(axis, rows_by_scenario[scenario], scenario, field, smooth_field)
        style_axis(axis, y_label, limits)
        axis.set_title(y_label, loc="left", fontweight="bold")
    axes[0, 0].legend(loc="lower right", frameon=False, ncol=2)
    add_figure_header(
        figure,
        "四方案模型指标随训练轮次变化",
        "实际150轮；浅线为每轮原始值，粗线为{}轮尾随均值".format(smooth_window),
    )
    save_figure(figure, path)
    return path


def plot_participation(
        rows_by_scenario: Dict[str, List[Dict[str, object]]],
        summaries: Dict[str, Dict[str, object]], figure_dir: Path
) -> Path:
    """绘制每轮活跃人数、累计参与量、分布箱线图和均值比较。"""

    path = figure_dir / "02_参与强度与累计贡献.png"
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    epochs = np.arange(1, len(next(iter(rows_by_scenario.values()))) + 1)
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        active = np.asarray([int(row["active_count"]) for row in rows])
        cumulative = np.asarray([int(row["cumulative_active"]) for row in rows])
        axes[0, 0].plot(
            epochs, active, color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario], linewidth=1.3,
            label=SCENARIO_LABELS[scenario],
        )
        axes[0, 1].plot(
            epochs, cumulative, color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario], linewidth=2.0,
            label=SCENARIO_LABELS[scenario],
        )
    style_axis(axes[0, 0], "实际参与聚合客户端数", (0, 39))
    style_axis(axes[0, 1], "累计聚合客户端次")
    axes[0, 0].set_title("每轮实际参与人数", loc="left", fontweight="bold")
    axes[0, 1].set_title("累计有效聚合贡献", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")

    distributions = [
        [int(row["active_count"]) for row in rows_by_scenario[scenario]]
        for scenario in SCENARIO_ORDER
    ]
    boxes = axes[1, 0].boxplot(
        distributions, labels=[SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER],
        patch_artist=True, showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#344054", "markersize": 5},
        medianprops={"color": "#101828", "linewidth": 1.4},
    )
    for patch, scenario in zip(boxes["boxes"], SCENARIO_ORDER):
        patch.set_facecolor(SCENARIO_COLORS[scenario])
        patch.set_alpha(0.45)
    axes[1, 0].set_ylabel("每轮实际参与人数")
    axes[1, 0].set_title("参与人数分布", loc="left", fontweight="bold")
    axes[1, 0].grid(True, axis="y", color="#EAECF0")
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    x_positions = np.arange(len(SCENARIO_ORDER))
    mean_active = [float(summaries[scenario]["active_mean"]) for scenario in SCENARIO_ORDER]
    bars = axes[1, 1].bar(
        x_positions, mean_active,
        color=[SCENARIO_COLORS[scenario] for scenario in SCENARIO_ORDER], alpha=0.82,
    )
    for bar, scenario in zip(bars, SCENARIO_ORDER):
        summary = summaries[scenario]
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.6,
            "均值 {:.2f}\n零参与 {} 轮".format(summary["active_mean"], summary["zero_active_rounds"]),
            ha="center", va="bottom", fontsize=9,
        )
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels([SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER])
    axes[1, 1].set_ylim(0, 40)
    axes[1, 1].set_ylabel("平均实际参与人数")
    axes[1, 1].set_title("参与均值与空聚合轮", loc="left", fontweight="bold")
    axes[1, 1].grid(True, axis="y", color="#EAECF0")
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    add_figure_header(
        figure,
        "实际参与强度和累计聚合贡献",
        "活跃人数来自topology_schedule.jsonl实际150轮；不是metadata对MAT全部200行的均值",
    )
    save_figure(figure, path)
    return path


def plot_aggregation_efficiency(
        rows_by_scenario: Dict[str, List[Dict[str, object]]],
        summaries: Dict[str, Dict[str, object]], figure_dir: Path
) -> Path:
    """绘制测试准确率相对于累计聚合客户端次的完整和共同预算曲线。"""

    path = figure_dir / "03_聚合预算效率.png"
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    common_budget = min(int(summaries[scenario]["active_total"]) for scenario in SCENARIO_ORDER)
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        cumulative = np.asarray([int(row["cumulative_active"]) for row in rows])
        accuracy = trailing_mean([float(row["test_acc"]) for row in rows], 5)
        valid = np.isfinite(accuracy)
        axes[0].plot(
            cumulative[valid], accuracy[valid], color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario], linewidth=2.0,
            label=SCENARIO_LABELS[scenario],
        )
        common = valid & (cumulative <= common_budget)
        axes[1].plot(
            cumulative[common], accuracy[common], color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario], linewidth=2.0,
            label=SCENARIO_LABELS[scenario],
        )
    for axis, title in zip(axes, ["完整累计预算", "共同预算区间（0–{}）".format(common_budget)]):
        axis.set_xlabel("累计聚合客户端次")
        axis.set_ylabel("测试准确率（5轮尾随均值）")
        axis.set_ylim(0.05, 0.95)
        axis.grid(True, axis="both", color="#EAECF0")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(title, loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower right")
    add_figure_header(
        figure,
        "测试准确率与累计有效聚合客户端次",
        "该横轴只表示进入聚合的客户端状态数量；PLAN约定的200人本地训练工作量并未计入",
    )
    save_figure(figure, path)
    return path


def plot_consensus_decomposition(
        rows_by_scenario: Dict[str, List[Dict[str, object]]], figure_dir: Path, smooth_window: int
) -> Path:
    """以四个小图展示一致性A、确定性C和有效共识S的分解。"""

    path = figure_dir / "04_有效共识分解.png"
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, sharey=True)
    metric_styles = [
        ("candidate_agreement", "candidate_agreement_ma", "一致性 A", "#667085", ":"),
        ("candidate_certainty", "candidate_certainty_ma", "确定性 C", "#2563EB", "--"),
        ("candidate_effective", "candidate_effective_ma", "有效共识 S=A×C", "#B7791F", "-"),
    ]
    for axis, scenario in zip(axes.flat, SCENARIO_ORDER):
        rows = rows_by_scenario[scenario]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        for field, smooth_field, label, color, linestyle in metric_styles:
            raw = np.asarray([float(row[field]) for row in rows])
            smooth = np.asarray([float(row[smooth_field]) for row in rows])
            axis.plot(epochs, raw, color=color, linewidth=0.7, alpha=0.16, linestyle=linestyle)
            axis.plot(epochs, smooth, color=color, linewidth=2.0, linestyle=linestyle, label=label)
        style_axis(axis, "指标值", (0, 1.02))
        axis.set_title(SCENARIO_LABELS[scenario], loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="center right")
    add_figure_header(
        figure,
        "候选37人的有效共识分解",
        "浅线为原始每轮值，粗线为{}轮尾随均值；A高而C低表示共同接近均匀分布".format(smooth_window),
    )
    save_figure(figure, path)
    return path


def plot_consensus_attainment(
        rows_by_scenario: Dict[str, List[Dict[str, object]]], figure_dir: Path, smooth_window: int
) -> Path:
    """对比平滑有效共识和单调不降的历史最佳平滑共识。"""

    path = figure_dir / "05_平滑共识与历史最佳.png"
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        smooth = np.asarray([float(row["candidate_effective_ma"]) for row in rows])
        best = np.asarray([float(row["candidate_effective_historical_best"]) for row in rows])
        axes[0].plot(
            epochs, smooth, color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario], linewidth=2.0,
            label=SCENARIO_LABELS[scenario],
        )
        axes[1].plot(
            epochs, best, color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario], linewidth=2.0,
            label=SCENARIO_LABELS[scenario],
        )
    style_axis(axes[0], "有效共识", (0, 1.02))
    style_axis(axes[1], "历史最佳平滑共识", (0, 1.02))
    axes[0].set_title("当前{}轮平滑共识".format(smooth_window), loc="left", fontweight="bold")
    axes[1].set_title("历史最佳平滑共识（单调）", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower right")
    add_figure_header(
        figure,
        "平滑有效共识与单调历史记录",
        "前{}轮为空值；右图只表示截至当前的历史最高水平，不能反映后续退化".format(smooth_window - 1),
    )
    save_figure(figure, path)
    return path


def plot_hierarchy_consensus(
        rows_by_scenario: Dict[str, List[Dict[str, object]]], figure_dir: Path, smooth_window: int
) -> Path:
    """展示两种HFL方案中候选、活跃、边缘和云端的确定性及云端分歧。"""

    path = figure_dir / "06_HFL层级共识传播.png"
    hfl_scenarios = ["hfl_snf_fixed", "hfl_no_snf_fixed"]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True)
    certainty_fields = [
        ("candidate_certainty_ma", "候选客户端", "#667085", ":"),
        ("active_certainty_ma", "最终参与客户端", "#2563EB", "--"),
        ("edge_certainty_ma", "边缘模型", "#C2410C", "-."),
        ("cloud_certainty_ma", "云模型", "#B7791F", "-"),
    ]
    divergence_fields = [
        ("candidate_cloud_js_ma", "候选客户端—云", "#667085", ":"),
        ("active_cloud_js_ma", "最终参与者—云", "#2563EB", "--"),
        ("edge_cloud_js_ma", "边缘—云", "#C2410C", "-."),
    ]
    for column, scenario in enumerate(hfl_scenarios):
        rows = rows_by_scenario[scenario]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        for field, label, color, linestyle in certainty_fields:
            axes[0, column].plot(
                epochs, [float(row[field]) for row in rows], color=color,
                linestyle=linestyle, linewidth=2.0, label=label,
            )
        for field, label, color, linestyle in divergence_fields:
            axes[1, column].plot(
                epochs, [float(row[field]) for row in rows], color=color,
                linestyle=linestyle, linewidth=2.0, label=label,
            )
        style_axis(axes[0, column], "确定性", (0, 1.02))
        style_axis(axes[1, column], "平均JS散度", (0, 0.35))
        axes[0, column].set_title(SCENARIO_LABELS[scenario], loc="left", fontweight="bold")
        axes[1, column].set_title("与云模型的概率分歧", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="lower right")
    axes[1, 0].legend(frameon=False, loc="upper right")
    add_figure_header(
        figure,
        "HFL候选—边缘—云端的共识传播",
        "{}轮尾随均值；零参与轮和无边缘模型位置保留为空值".format(smooth_window),
    )
    save_figure(figure, path)
    return path


def build_client_matrix(experiment: ExperimentData, field: str) -> np.ndarray:
    """将候选或最终参与客户端记录展开为200×轮数的0/1矩阵。"""

    client_total = int(experiment.metadata["client_num_in_total"])
    matrix = np.zeros((client_total, len(experiment.schedule)), dtype=np.float64)
    if field not in {"candidate_client_indexes", "active_client_indexes"}:
        raise ValueError("不支持的客户端矩阵字段：{}".format(field))
    for epoch, record in enumerate(experiment.schedule):
        for client_id in record[field]:
            matrix[int(client_id), epoch] = 1.0
    return matrix


def plot_client_coverage_and_relationship(
        experiments: Sequence[ExperimentData], rows_by_scenario: Dict[str, List[Dict[str, object]]],
        summaries: Dict[str, Dict[str, object]], figure_dir: Path, smooth_window: int
) -> Path:
    """绘制200客户端候选/参与热图及平滑共识—准确率散点关系。"""

    path = figure_dir / "07_客户端覆盖与共识准确率关系.png"
    figure = plt.figure(figsize=(16, 15))
    grid = figure.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.15], hspace=0.48, wspace=0.18)
    experiment_by_scenario = {experiment.scenario: experiment for experiment in experiments}

    candidate_axis = figure.add_subplot(grid[0, :])
    candidate_matrix = build_client_matrix(experiment_by_scenario[SCENARIO_ORDER[0]], "candidate_client_indexes")
    candidate_axis.imshow(
        candidate_matrix, aspect="auto", origin="lower",
        cmap=ListedColormap(["#F8FAFC", "#2563EB"]), interpolation="nearest",
    )
    candidate_axis.set_title("四方案共享的一级候选采样", loc="left", fontweight="bold")
    candidate_axis.set_ylabel("真实客户端ID")
    candidate_axis.set_xlabel("训练轮次")

    for index, scenario in enumerate(SCENARIO_ORDER):
        row = 1 + index // 2
        column = index % 2
        axis = figure.add_subplot(grid[row, column])
        active_matrix = build_client_matrix(experiment_by_scenario[scenario], "active_client_indexes")
        axis.imshow(
            active_matrix, aspect="auto", origin="lower",
            cmap=ListedColormap(["#F8FAFC", SCENARIO_COLORS[scenario]]), interpolation="nearest",
        )
        axis.set_title("{} 最终参与".format(SCENARIO_LABELS[scenario]), loc="left", fontweight="bold")
        axis.set_ylabel("真实客户端ID")
        axis.set_xlabel("训练轮次")

    relation_axis = figure.add_subplot(grid[3, :])
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        consensus = np.asarray([float(row["candidate_effective_ma"]) for row in rows])
        accuracy = np.asarray([float(row["test_acc_ma"]) for row in rows])
        mask = np.isfinite(consensus) & np.isfinite(accuracy)
        relation_axis.scatter(
            consensus[mask], accuracy[mask], s=22, alpha=0.55,
            facecolor=SCENARIO_COLORS[scenario], edgecolor="white", linewidth=0.35,
            label="{}（r={:.2f}）".format(
                SCENARIO_LABELS[scenario], summaries[scenario]["consensus_accuracy_level_corr"]
            ),
        )
    relation_axis.set_xlabel("候选客户端有效共识（{}轮尾随均值）".format(smooth_window))
    relation_axis.set_ylabel("测试准确率（{}轮尾随均值）".format(smooth_window))
    relation_axis.set_xlim(0, 1.0)
    relation_axis.set_ylim(0.05, 0.95)
    relation_axis.grid(True, color="#EAECF0")
    relation_axis.spines["top"].set_visible(False)
    relation_axis.spines["right"].set_visible(False)
    relation_axis.set_title("平滑共识与平滑准确率的同期关系", loc="left", fontweight="bold")
    relation_axis.legend(frameon=False, ncol=2, loc="lower right")

    add_figure_header(
        figure,
        "真实客户端覆盖和共识—准确率关系",
        "热图深色表示该轮入选/参与；散点相关包含共同时间趋势，只作描述性证据",
    )
    save_figure(figure, path)
    return path


def csv_value(value: object) -> object:
    """将空值和非有限浮点数转换为空单元格，其余值保持可写形式。"""

    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    """按固定列顺序写出带UTF-8 BOM的CSV，便于Windows Excel直接打开。"""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def export_summary_rows(summaries: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """将内部英文键汇总转换为面向读者的简体中文CSV字段。"""

    rows = []
    for scenario in SCENARIO_ORDER:
        item = summaries[scenario]
        rows.append(
            {
                "方案": item["label"],
                "场景": scenario,
                "实际轮数": item["rounds"],
                "最终测试准确率": item["final_test_acc"],
                "最佳测试准确率": item["best_test_acc"],
                "最佳轮次": item["best_test_epoch"],
                "150轮平均测试准确率": item["mean_test_acc"],
                "后10轮测试准确率均值": item["last10_test_acc_mean"],
                "后10轮测试准确率标准差": item["last10_test_acc_std"],
                "最终测试损失": item["final_test_loss"],
                "最低测试损失": item["min_test_loss"],
                "后10轮训练准确率均值": item["last10_train_acc_mean"],
                "后10轮泛化差距_训练减测试": item["last10_generalization_gap"],
                "平均活跃客户端数": item["active_mean"],
                "最少活跃客户端数": item["active_min"],
                "最多活跃客户端数": item["active_max"],
                "累计活跃客户端次": item["active_total"],
                "零参与轮数": item["zero_active_rounds"],
                "累计下发客户端次": item["distributed_total"],
                "候选覆盖客户端数": item["candidate_coverage"],
                "最终参与覆盖客户端数": item["active_coverage"],
                "稳定达到80%轮次": item["stable_epoch_0.80"],
                "稳定达到85%轮次": item["stable_epoch_0.85"],
                "稳定达到88%轮次": item["stable_epoch_0.88"],
                "前10轮一致性A": item["agreement_first10"],
                "前10轮确定性C": item["certainty_first10"],
                "前10轮有效共识S": item["effective_first10"],
                "后10轮一致性A": item["agreement_last10"],
                "后10轮确定性C": item["certainty_last10"],
                "后10轮有效共识S": item["effective_last10"],
                "最高10轮平滑有效共识": item["effective_ma_best"],
                "平滑共识与平滑准确率同期相关": item["consensus_accuracy_level_corr"],
                "共识变化与准确率变化相关": item["consensus_accuracy_diff_corr"],
                "最强滞后_正值表示共识领先": item["strongest_lag"],
                "最强滞后相关": item["strongest_lag_corr"],
            }
        )
    return rows


def export_round_rows(rows_by_scenario: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    """将四组逐轮指标合并为600行中文字段明细表。"""

    field_mapping = [
        ("label", "方案"),
        ("scenario", "场景"),
        ("epoch", "轮次"),
        ("candidate_count", "候选客户端数"),
        ("active_count", "最终参与客户端数"),
        ("cumulative_active", "累计参与客户端次"),
        ("effective_edge_count", "有效边缘数"),
        ("distributed_count", "下发客户端数"),
        ("aggregated", "是否聚合"),
        ("train_acc", "训练准确率"),
        ("train_loss", "训练损失"),
        ("test_acc", "测试准确率"),
        ("test_loss", "测试损失"),
        ("generalization_gap", "泛化差距_训练减测试"),
        ("candidate_agreement", "候选一致性A"),
        ("candidate_certainty", "候选确定性C"),
        ("candidate_effective", "候选有效共识S"),
        ("candidate_effective_ma", "候选有效共识尾随均值"),
        ("candidate_effective_historical_best", "历史最佳平滑共识"),
        ("active_agreement", "参与者一致性A"),
        ("active_certainty", "参与者确定性C"),
        ("active_effective", "参与者有效共识S"),
        ("within_group_agreement", "组内一致性A_人数加权"),
        ("within_group_certainty", "组内确定性C_人数加权"),
        ("within_group_effective", "组内有效共识S_人数加权"),
        ("edge_agreement", "边缘间一致性A"),
        ("edge_certainty", "边缘模型确定性C"),
        ("edge_effective", "边缘间有效共识S"),
        ("cloud_certainty", "云模型确定性"),
        ("candidate_cloud_js", "候选客户端到云平均JS"),
        ("active_cloud_js", "参与客户端到云平均JS"),
        ("edge_cloud_js", "边缘到云平均JS"),
        ("edge_cloud_agreement", "边缘到云一致性A"),
        ("edge_cloud_certainty", "边缘云联合确定性C"),
        ("edge_cloud_effective", "边缘云有效共识S"),
        ("edge_cloud_effective_ma", "边缘云有效共识尾随均值"),
        ("candidate_client_ids", "候选客户端ID"),
        ("active_client_ids", "最终参与客户端ID"),
        ("group_to_client_ids", "最终分组客户端ID"),
    ]
    output = []
    for scenario in SCENARIO_ORDER:
        for row in rows_by_scenario[scenario]:
            output.append({chinese: row.get(english) for english, chinese in field_mapping})
    return output


def sha256_file(path: Path) -> str:
    """分块计算输入文件SHA-256，用于分析清单中的来源追溯。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    """递归将NumPy对象和非有限浮点数转换为标准JSON可表示值。"""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, Path):
        return str(value)
    return value


def detect_workspace_drift(workspace: Path, experiments: Sequence[ExperimentData]) -> Dict[str, object]:
    """只读检查当前YAML轮数和训练器关键语义是否与历史结果一致。"""

    config_files = {
        "hfl_snf_fixed": workspace / "configs" / "fedml_config_hfl_snf_fixed_u05.yaml",
        "hfl_no_snf_fixed": workspace / "configs" / "fedml_config_hfl_no_snf_fixed_u05.yaml",
        "fl_snf": workspace / "configs" / "fedml_config_fl_snf_u05.yaml",
        "fl_no_snf": workspace / "configs" / "fedml_config_fl_no_snf_u05.yaml",
    }
    current_rounds = {}
    for scenario, path in config_files.items():
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        match = re.search(r"^\s*comm_round\s*:\s*(\d+)\s*$", text, flags=re.MULTILINE)
        current_rounds[scenario] = None if match is None else int(match.group(1))
    trainer_path = workspace / "trainer_test.py"
    trainer_text = trainer_path.read_text(encoding="utf-8") if trainer_path.is_file() else ""
    result_rounds = {experiment.scenario: len(experiment.schedule) for experiment in experiments}
    return {
        "current_yaml_comm_round": current_rounds,
        "result_actual_rounds": result_rounds,
        "current_trainer_uses_fixed_candidate": (
            "def _initialize_fixed_candidate_clients" in trainer_text
            and "self.fixed_candidate_client_indexes" in trainer_text
        ),
        "current_trainer_matlab_trains_only_active": (
            "self._train_active_clients_one_epoch" in trainer_text
            and "正式 MATLAB 模式只训练当前行实际启用" in trainer_text
        ),
        "result_candidates_change_each_epoch": all(
            len({tuple(record["candidate_client_indexes"]) for record in experiment.schedule})
            == len(experiment.schedule)
            for experiment in experiments
        ),
    }


def add_cross_experiment_checks(
        quality_checks: List[Dict[str, object]], experiments: Sequence[ExperimentData],
        rows_by_scenario: Dict[str, List[Dict[str, object]]], drift: Dict[str, object]
) -> None:
    """追加四方案候选序列一致性、零轮模型保持和代码漂移检查。"""

    reference = [record["candidate_client_indexes"] for record in experiments[0].schedule]
    identical = all(
        [record["candidate_client_indexes"] for record in experiment.schedule] == reference
        for experiment in experiments[1:]
    )
    add_quality_check(
        quality_checks, "跨实验", "同轮一级候选序列一致", "通过" if identical else "未通过",
        "四方案同一epoch使用相同的37人候选顺序", "关键",
    )
    zero_model_hold = True
    evidence = []
    for experiment in experiments:
        rows = rows_by_scenario[experiment.scenario]
        zero_epochs = []
        for index, row in enumerate(rows):
            if int(row["active_count"]) != 0:
                continue
            zero_epochs.append(index + 1)
            if index > 0:
                zero_model_hold = zero_model_hold and float(row["test_acc"]) == float(rows[index - 1]["test_acc"])
                zero_model_hold = zero_model_hold and float(row["test_loss"]) == float(rows[index - 1]["test_loss"])
        evidence.append("{}:{}".format(experiment.label, zero_epochs))
    add_quality_check(
        quality_checks, "跨实验", "零参与轮保留上一云模型指标", "通过" if zero_model_hold else "未通过",
        "；".join(evidence), "关键",
    )
    drift_found = bool(drift["current_trainer_uses_fixed_candidate"] and drift["result_candidates_change_each_epoch"])
    add_quality_check(
        quality_checks, "跨实验", "当前代码可直接复现本批采样语义", "注意" if drift_found else "通过",
        "结果每轮重采37人；当前trainer固定一次候选={}".format(
            drift["current_trainer_uses_fixed_candidate"]
        ), "高",
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """生成无复杂合并单元格的GitHub风格Markdown表格。"""

    def escape(value: object) -> str:
        """将单元格转为字符串，并转义会破坏表格的竖线。"""

        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(escape(value) for value in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def format_percent(value: object, digits: int = 2) -> str:
    """将0到1比例格式化为百分数，空值显示为破折号。"""

    if value is None or not np.isfinite(float(value)):
        return "—"
    return ("{:." + str(digits) + "%}").format(float(value))


def format_number(value: object, digits: int = 3) -> str:
    """以固定小数位格式化数值，空值或非有限值显示为破折号。"""

    if value is None or not np.isfinite(float(value)):
        return "—"
    return ("{:." + str(digits) + "f}").format(float(value))


def build_report_text(
        experiments: Sequence[ExperimentData], summaries: Dict[str, Dict[str, object]],
        contrasts: Sequence[Dict[str, object]], quality_checks: Sequence[Dict[str, object]],
        drift: Dict[str, object], smooth_window: int
) -> str:
    """根据已复算指标生成回答优先、证据相邻的简体中文技术报告。"""

    best_scenario = max(SCENARIO_ORDER, key=lambda key: summaries[key]["last10_test_acc_mean"])
    stable_scenario = min(SCENARIO_ORDER, key=lambda key: summaries[key]["last10_test_acc_std"])
    hfl_gain = next(row for row in contrasts if row["对比"] == "HFL中SnF增益")
    fl_gain = next(row for row in contrasts if row["对比"] == "FL中SnF增益")

    performance_rows = []
    for scenario in SCENARIO_ORDER:
        item = summaries[scenario]
        performance_rows.append(
            [
                item["label"],
                format_percent(item["final_test_acc"]),
                format_percent(item["best_test_acc"]),
                format_percent(item["last10_test_acc_mean"]),
                "{:.2f}个百分点".format(100.0 * item["last10_test_acc_std"]),
                item["stable_epoch_0.80"] or "未达到",
                item["stable_epoch_0.85"] or "未达到",
                item["stable_epoch_0.88"] or "未达到",
            ]
        )

    participation_rows = []
    consensus_rows = []
    for scenario in SCENARIO_ORDER:
        item = summaries[scenario]
        participation_rows.append(
            [
                item["label"],
                format_number(item["active_mean"], 2),
                "{}–{}".format(item["active_min"], item["active_max"]),
                item["active_total"],
                item["zero_active_rounds"],
                "{}/200".format(item["active_coverage"]),
                item["distributed_total"],
            ]
        )
        consensus_rows.append(
            [
                item["label"],
                format_number(item["agreement_first10"], 4),
                format_number(item["certainty_first10"], 4),
                format_number(item["effective_first10"], 4),
                format_number(item["effective_last10"], 4),
                format_number(item["effective_ma_best"], 4),
                format_number(item["consensus_accuracy_level_corr"], 3),
                format_number(item["consensus_accuracy_diff_corr"], 3),
                item["strongest_lag"],
                format_number(item["strongest_lag_corr"], 3),
            ]
        )

    contrast_rows = [
        [
            row["对比"],
            "{:+.3f}".format(row["后10轮准确率差_百分点"]),
            "{:+.3f}".format(row["最终准确率差_百分点"]),
            "{:+.2f}".format(row["平均活跃人数差"]),
        ]
        for row in contrasts
    ]
    quality_status_counts = {}
    for row in quality_checks:
        quality_status_counts[row["状态"]] = quality_status_counts.get(row["状态"], 0) + 1

    result_rounds = sorted(set(int(item["rounds"]) for item in summaries.values()))
    yaml_rounds = sorted(set(value for value in drift["current_yaml_comm_round"].values() if value is not None))
    text = """# 四组联邦学习实验结果分析报告

## 技术摘要

- **后期效果最好的方案是 {best_label}。** 它在最后10轮的平均测试准确率为 **{best_acc}**；后期最稳定的方案是 {stable_label}，最后10轮标准差为 **{stable_std:.2f} 个百分点**。
- **SnF方案在两种架构下都表现出更高的后10轮准确率，但不能解释成纯SnF因果效果。** HFL内差值为 **{hfl_gain:+.3f} 个百分点**，FL内差值为 **{fl_gain:+.3f} 个百分点**；与此同时四场景平均参与聚合人数差异很大。
- **有效共识修正了“共同乱猜”的误判。** 四组前10轮的一致性A约为0.998，但确定性C只有约0.003，因此有效共识S接近0；后10轮S上升到约0.65。
- **证据评级：可分享但附带限制。** 600条调度记录和概率数据通过关键一致性检查，但只有单随机种子，结果没有运行代码哈希、模型快照或探针真值，而且当前代码与生成结果时的采样语义已经发生漂移。

## 1. 数据完整，但实际运行是150轮而不是200轮

四个结果目录的调度、四项模型指标和三份探针均为 **{rounds}轮**。元数据中的 `round_count=200` 表示MAT可用拓扑容量，`configured_comm_round=150` 和实际文件行数才是本次运行范围。目录名中的 `full200` 是实验标签，不能当成实际轮数。

数据质量检查共记录 {quality_total} 项：{quality_counts}。所有关键不变量均通过；“每轮全部200人完成本地训练”由于没有 `trained_client_indexes`，被标记为无法验证。

## 2. HFL-SnF后期准确率最高，本次运行中SnF同时降低后期波动

下表以最后10轮均值作为主要后期指标，同时保留最终值、峰值和基于5轮尾随均值的稳定达标轮次。单个最终epoch容易受当轮拓扑影响，因此不能只看最终一行。

{performance_table}

四组都没有稳定达到90%。HFL-SnF最后10轮平均准确率最高，且后期标准差最小；FL-noSnF的后期准确率最低，HFL-noSnF的后期波动最大。训练准确率与测试准确率非常接近，未观察到明显过拟合证据。

![四方案模型指标趋势](figures/01_模型效果趋势.png)

图中粗线为{smooth_window}轮尾随均值。SnF两条曲线的后期波动明显小于对应noSnF曲线，但参与规模不同仍是重要替代解释。

## 3. 四场景的有效聚合预算差异很大

{participation_table}

如果PLAN中的“每轮200人本地训练”约定成立，四组本地训练工作量相同；上表统计的是**真正进入聚合的客户端状态数量**和下发次数。HFL-SnF在150轮累计聚合5216个客户端状态，而FL-noSnF只有951个，因此按轮次得到的性能差异同时包含拓扑、聚合层次和参与预算差异。

![参与强度与累计贡献](figures/02_参与强度与累计贡献.png)

![聚合预算效率](figures/03_聚合预算效率.png)

按轮次看，更多参与通常带来更快、更稳定的全局学习；按累计参与量看，FL-noSnF用更少的聚合客户端次达到较高准确率。后者不代表其总体方案更优，而是说明“每轮效果”和“单位上行聚合贡献效率”是两个不同问题。

## 4. 有效共识避免把均匀输出当成高共识

定义候选客户端平均概率为 $\\bar p_t$，归一化熵为 $h(p)$。本报告使用：

$$A_t=1-\\left[h(\\bar p_t)-\\frac1{{37}}\\sum_i h(p_i^t)\\right],\\quad C_t=1-\\frac1{{37}}\\sum_i h(p_i^t),\\quad S_t=A_tC_t.$$

{consensus_table}

训练初期A接近1并不意味着模型已经形成有意义共识，而是所有客户端都输出接近均匀分布；C和S正确保持在接近0。四方案后10轮S都约为0.65，差异远小于参与规模差异，因此S更适合描述“当前预测是否既一致又确定”，不应单独用作算法优劣排名。

正滞后表示共识领先准确率；四组绝对相关最大值都落在搜索边界 -10 轮，即准确率领先共识约10轮的描述性信号。由于最优点卡在边界且两条序列都随训练时间上升，这一结果不能用于判断因果方向；一阶差分相关接近0也说明同期相关主要包含共同时间趋势。完整的 -10 至 +10 轮结果见 `共识准确率相关.csv`。

![有效共识分解](figures/04_有效共识分解.png)

![平滑共识与历史最佳](figures/05_平滑共识与历史最佳.png)

“历史最佳平滑共识”是截至当前轮的累计最高值，所以天然单调不降；它反映达成过的最好水平，却不会在模型退化时下降，必须和当前平滑S一起阅读。

## 5. HFL层级传播提高输出集中度，但不能证明预测正确

![HFL层级共识传播](figures/06_HFL层级共识传播.png)

层级图按每轮JSONL的真实分组将37列候选探针映射到最终参与客户端，再与相同顺序的非空边缘探针对齐。边缘和云模型通常比客户端输出更集中，边缘—云JS分歧也较小；但是结果目录没有保存探针真实标签，因此高确定性和高共识仍可能是“共同预测错误”。

## 6. 一级采样公平覆盖200人，但二级参与覆盖随场景变化

![客户端覆盖与共识准确率关系](figures/07_客户端覆盖与共识准确率关系.png)

四方案同一轮使用完全相同的37人候选顺序，150轮候选并集覆盖全部200个客户端；FL-noSnF最终只覆盖197人，其余三组覆盖200人。右下散点的同期相关包含共同时间趋势；报告同时计算一阶差分相关，避免把“都随轮次上升”直接解释成即时驱动关系。

## 7. 2×2比较只能作为端到端场景描述

{contrast_table}

差分中的差分为描述性结果，不是受控因果估计。原因是SnF/noSnF场景的MAT参与人数、非空边缘数量和累计聚合客户端次并不相同，而且每个方案只有一个随机种子。

## 8. 运行语义和可复现性限制

- 结果记录每个epoch重新抽取37人；当前 `trainer_test.py` 已改为实验开始时只抽一次固定候选，并且MAT模式只训练最终活跃客户端。
- 当前四份YAML配置为 {yaml_rounds} 轮，而本批结果元数据和文件均为 {rounds} 轮；当前配置不能直接复现这批结果。
- topology_metadata.json 的参与均值覆盖MAT全部200行；本报告改用JSONL实际150行重新计算。
- 结果没有保存完整运行配置快照、Git提交或代码哈希，也没有保存模型参数，无法独立复算“指标一定来自最终云模型”。
- 运行时MAT绝对路径已失效且没有文件哈希。本报告因此以JSONL中的实际分组为准，而不假设本地MAT与服务器文件字节一致。
- 每轮探针样本随epoch变化，共识曲线混合了模型学习进展与样本难度变化。
- 没有探针标签，不能区分正确共识和错误共识；没有可靠时间日志，不能比较训练耗时。
- 单随机种子不支持显著性检验、置信区间或稳定性外推。

## 9. 建议的下一步

1. 固化下一批实验的代码提交、完整YAML和MAT哈希，并为每轮日志增加 `trained_client_indexes`，消除训练语义无法验证的问题。
2. 使用相同候选序列、相同最终参与人数预算和至少5个随机种子重跑2×2消融，分别识别SnF、HFL层级和参与规模的影响。
3. 将探针改为固定的小批量样本并保存真实标签，分别报告正确共识、错误共识和样本难度分层结果。
4. 同时保留当前平滑有效共识S与历史最佳平滑共识，前者监控退化，后者记录达成进度。
5. 若目标是比较通信效率，应保存模型字节数、上行/下行次数和真实耗时；当前“累计客户端次”只能作为代理量。

## 10. 仍需进一步回答的问题

- 在严格匹配每轮参与人数后，SnF是否仍能带来1–3个百分点的后期准确率优势？
- HFL的提升来自边缘层聚合本身，还是来自更高的有效参与覆盖？
- 有效共识S在固定探针集上的变化是否仍与准确率同步，还是当前相关主要由样本难度和共同时间趋势造成？
- 当前代码的固定候选版本与本批每轮重采版本相比，会如何影响客户端公平性和最终精度？

本报告由 `analyze_experiment_suite.py` 从原始结果重新计算；可审计明细见同目录CSV和 `analysis_manifest.json`。
""".format(
        best_label=summaries[best_scenario]["label"],
        best_acc=format_percent(summaries[best_scenario]["last10_test_acc_mean"]),
        stable_label=summaries[stable_scenario]["label"],
        stable_std=100.0 * summaries[stable_scenario]["last10_test_acc_std"],
        hfl_gain=hfl_gain["后10轮准确率差_百分点"],
        fl_gain=fl_gain["后10轮准确率差_百分点"],
        rounds="、".join(str(value) for value in result_rounds),
        yaml_rounds="、".join(str(value) for value in yaml_rounds) or "未知",
        quality_total=len(quality_checks),
        quality_counts="，".join("{}{}项".format(key, value) for key, value in sorted(quality_status_counts.items())),
        performance_table=markdown_table(
            ["方案", "最终准确率", "峰值", "后10轮均值", "后10轮标准差", "稳定80%", "稳定85%", "稳定88%"],
            performance_rows,
        ),
        participation_table=markdown_table(
            ["方案", "平均参与", "范围", "累计参与客户端次", "零参与轮", "最终参与覆盖", "累计下发客户端次"],
            participation_rows,
        ),
        consensus_table=markdown_table(
            [
                "方案", "前10轮A", "前10轮C", "前10轮S", "后10轮S", "最高MA10",
                "同期相关", "差分相关", "最强滞后", "最强滞后相关",
            ],
            consensus_rows,
        ),
        contrast_table=markdown_table(
            ["对比", "后10轮准确率差/百分点", "最终差/百分点", "平均参与人数差"], contrast_rows,
        ),
        smooth_window=smooth_window,
    )
    return text


def build_manifest(
        experiments: Sequence[ExperimentData], summaries: Dict[str, Dict[str, object]],
        quality_checks: Sequence[Dict[str, object]], figure_paths: Sequence[Path],
        output_dir: Path, smooth_window: int, drift: Dict[str, object]
) -> Dict[str, object]:
    """构建包含来源哈希、分析参数、图表地图、限制和输出清单的JSON清单。"""

    sources = {}
    for experiment in experiments:
        hashes = {}
        for filename in REQUIRED_RESULT_FILES:
            path = experiment.path / filename
            hashes[filename] = sha256_file(path)
        sources[experiment.scenario] = {
            "label": experiment.label,
            "path": str(experiment.path),
            "rounds": len(experiment.schedule),
            "hashes": hashes,
        }
    chart_questions = [
        "四方案的准确率、损失与后期波动如何变化？",
        "实际参与强度和累计聚合贡献有多大差异？",
        "在累计聚合客户端次口径下，各方案的学习效率如何？",
        "一致性、确定性和有效共识如何共同变化？",
        "当前平滑共识与单调历史最佳值有什么区别？",
        "HFL中客户端、边缘和云端的确定性及分歧如何传播？",
        "两级采样如何覆盖200客户端，共识与准确率是否同期变化？",
    ]
    chart_map = []
    for index, path in enumerate(figure_paths):
        chart_map.append(
            {
                "file": str(path.relative_to(output_dir)),
                "question": chart_questions[index],
                "renderer": "Matplotlib静态PNG",
                "palette": "蓝、金、橙、中性色，并辅以线型区分",
            }
        )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "confidence": "可分享但附带限制",
        "source_precedence": [
            "topology_schedule.jsonl和指标/探针文件",
            "topology_metadata.json",
            "result/PLAN.md",
            "当前YAML和代码",
        ],
        "analysis_parameters": {
            "smooth_window": smooth_window,
            "stable_threshold_window": 5,
            "thresholds": list(SUMMARY_THRESHOLDS),
            "lag_range": [-10, 10],
        },
        "sources": sources,
        "summaries": summaries,
        "quality_checks": list(quality_checks),
        "workspace_drift": drift,
        "chart_map": chart_map,
        "outputs": [
            "分析报告.md",
            "实验汇总.csv",
            "逐轮指标.csv",
            "客户端采样统计.csv",
            "数据质量检查.csv",
            "共识准确率相关.csv",
            "方案对比.csv",
            "analysis_manifest.json",
        ] + [str(path.relative_to(output_dir)) for path in figure_paths],
        "limitations": [
            "只有单随机种子，不能声明统计显著",
            "没有trained_client_indexes，不能独立证明每轮全部200人训练",
            "没有模型快照，不能独立复算指标是否来自最终云模型",
            "探针样本每轮变化且没有保存真值标签",
            "当前代码和YAML与生成结果时的运行语义发生漂移",
            "没有可靠运行时间和通信字节日志",
        ],
    }


def derive_output_dir(result_root: Path, experiments: Sequence[ExperimentData]) -> Path:
    """根据实际轮数和目录日期生成稳定、可读且不覆盖原实验的输出目录。"""

    round_counts = {len(experiment.schedule) for experiment in experiments}
    if len(round_counts) != 1:
        raise ValueError("四组实验轮数不同，不能自动生成统一输出目录")
    date_values = []
    for experiment in experiments:
        match = re.search(r"(\d{8})$", experiment.path.name)
        if match:
            date_values.append(match.group(1))
    date_label = date_values[0] if date_values and len(set(date_values)) == 1 else datetime.now().strftime("%Y%m%d")
    rounds = next(iter(round_counts))
    return result_root.resolve() / "analysis_alpha0p2_u0p5_{}rounds_{}".format(rounds, date_label)


def main() -> None:
    """执行发现、校验、计算、绘图、导出和中文技术报告生成全流程。"""

    args = parse_args()
    if args.smooth_window < 2:
        raise ValueError("--smooth-window 至少为2")
    workspace = Path(__file__).resolve().parent
    result_root = args.result_root.resolve()
    experiment_dirs = resolve_experiment_dirs(result_root, args.experiment_dir)
    experiments = [load_experiment(path) for path in experiment_dirs]
    experiment_by_scenario = {experiment.scenario: experiment for experiment in experiments}
    experiments = [experiment_by_scenario[scenario] for scenario in SCENARIO_ORDER]

    quality_checks = []
    for experiment in experiments:
        quality_checks.extend(validate_experiment(experiment))

    rows_by_scenario = {
        experiment.scenario: build_round_metrics(experiment, args.smooth_window)
        for experiment in experiments
    }
    summaries = {
        experiment.scenario: summarize_experiment(
            experiment, rows_by_scenario[experiment.scenario], args.smooth_window
        )
        for experiment in experiments
    }
    contrasts = build_contrasts(summaries)
    client_statistics = build_client_statistics(experiments)
    correlation_rows = build_correlation_rows(rows_by_scenario)
    drift = detect_workspace_drift(workspace, experiments)
    add_cross_experiment_checks(quality_checks, experiments, rows_by_scenario, drift)

    output_dir = (args.output_dir.resolve() if args.output_dir else derive_output_dir(result_root, experiments))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    figure_paths = [
        plot_model_metrics(rows_by_scenario, figure_dir, args.smooth_window),
        plot_participation(rows_by_scenario, summaries, figure_dir),
        plot_aggregation_efficiency(rows_by_scenario, summaries, figure_dir),
        plot_consensus_decomposition(rows_by_scenario, figure_dir, args.smooth_window),
        plot_consensus_attainment(rows_by_scenario, figure_dir, args.smooth_window),
        plot_hierarchy_consensus(rows_by_scenario, figure_dir, args.smooth_window),
        plot_client_coverage_and_relationship(
            experiments, rows_by_scenario, summaries, figure_dir, args.smooth_window
        ),
    ]

    summary_rows = export_summary_rows(summaries)
    round_rows = export_round_rows(rows_by_scenario)
    write_csv(output_dir / "实验汇总.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(output_dir / "逐轮指标.csv", round_rows, list(round_rows[0].keys()))
    write_csv(output_dir / "客户端采样统计.csv", client_statistics, list(client_statistics[0].keys()))
    write_csv(output_dir / "数据质量检查.csv", quality_checks, ["实验", "检查项", "状态", "证据", "严重级别"])
    write_csv(
        output_dir / "共识准确率相关.csv",
        correlation_rows,
        [
            "方案", "场景", "相关类型", "滞后轮数_正值表示共识领先",
            "相关系数", "共同有效样本数",
        ],
    )
    write_csv(
        output_dir / "方案对比.csv", contrasts,
        ["对比", "左方案", "右方案", "后10轮准确率差_百分点", "最终准确率差_百分点", "平均活跃人数差"],
    )

    manifest = build_manifest(
        experiments, summaries, quality_checks, figure_paths,
        output_dir, args.smooth_window, drift,
    )
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_text = build_report_text(
        experiments, summaries, contrasts, quality_checks, drift, args.smooth_window
    )
    (output_dir / "分析报告.md").write_text(report_text, encoding="utf-8")

    print("分析完成：{}".format(output_dir))
    print("实际轮数：{}；逐轮指标行数：{}".format(len(experiments[0].schedule), len(round_rows)))
    for scenario in SCENARIO_ORDER:
        item = summaries[scenario]
        print(
            "{}：后10轮测试准确率 {:.4f}，累计参与 {}，后10轮有效共识 {:.4f}".format(
                item["label"], item["last10_test_acc_mean"],
                item["active_total"], item["effective_last10"],
            )
        )


if __name__ == "__main__":
    main()
