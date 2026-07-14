"""对四组联邦学习实验进行可复现的横向分析并生成中文报告。"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

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


def locate_project_root() -> Path:
    """根据分析脚本的新固定位置返回训练项目根目录。"""

    # 当前文件位于“项目根目录/result/originalData”，向上两级即项目根目录。
    return Path(__file__).resolve().parent.parent.parent


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
    client_probe: List[List[Optional[np.ndarray]]]
    edge_probe: List[List[Optional[np.ndarray]]]
    cloud_probe: List[List[Optional[np.ndarray]]]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析输入批次、兼容参数、输出位置和共识平滑窗口。"""

    parser = argparse.ArgumentParser(description="分析四组联邦学习实验并生成中文报告")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="包含四个实验目录的批次目录，例如 result/originalData/1",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="兼容旧命令的实验结果根目录；请优先使用 --input-dir",
    )
    parser.add_argument(
        "--experiment-dir",
        action="append",
        type=Path,
        default=None,
        help="显式指定实验目录；可重复四次，未指定时从输入批次目录自动发现",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="分析输出目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="自动生成分析目录时使用的输出根目录",
    )
    parser.add_argument("--smooth-window", type=int, default=10, help="共识和趋势的尾随平滑窗口")
    args = parser.parse_args(argv)
    if args.input_dir is not None and args.result_root is not None:
        parser.error("--input-dir 与 --result-root 不能同时使用")
    if args.output_dir is not None and args.output_root is not None:
        parser.error("--output-dir 与 --output-root 不能同时使用")
    return args


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


def read_probability_csv(path: Path) -> List[List[Optional[np.ndarray]]]:
    """读取无表头探针 CSV，并保留空单元格对应的原始槽位。"""

    rounds = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            vectors: List[Optional[np.ndarray]] = []
            for column_number, cell in enumerate(row, start=1):
                if not cell.strip():
                    # 边缘探针用空单元格表示该边缘槽位当前未启用，不能删除其位置。
                    vectors.append(None)
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


def nonempty_probability_vectors(
        row: Sequence[Optional[np.ndarray]]
) -> List[np.ndarray]:
    """按原始列顺序返回探针行中的全部非空概率向量。"""

    return [vector for vector in row if vector is not None]


def discover_experiment_dirs(result_root: Path) -> List[Path]:
    """从指定批次目录发现恰好覆盖四个目标场景的完整实验目录。"""

    result_root = result_root.resolve()
    if not result_root.is_dir():
        raise FileNotFoundError("输入批次目录不存在：{}".format(result_root))
    scenario_to_paths = {scenario: [] for scenario in SCENARIO_ORDER}
    incomplete_directories = []
    for directory in sorted(result_root.iterdir()):
        if not directory.is_dir() or not (directory / "topology_metadata.json").is_file():
            continue
        metadata = read_json(directory / "topology_metadata.json")
        scenario = str(metadata.get("scenario", ""))
        if scenario not in scenario_to_paths:
            continue
        missing = [
            filename for filename in REQUIRED_RESULT_FILES
            if not (directory / filename).is_file()
        ]
        if missing:
            incomplete_directories.append(
                "{} 缺少 {}".format(directory.name, "、".join(missing))
            )
            continue
        scenario_to_paths[scenario].append(directory)

    problems = []
    for scenario in SCENARIO_ORDER:
        count = len(scenario_to_paths[scenario])
        if count != 1:
            problems.append("{} 匹配到 {} 个目录".format(scenario, count))
    problems.extend(incomplete_directories)
    if problems:
        raise ValueError(
            "自动发现实验失败：{}；可修正批次目录，或使用 --experiment-dir 显式指定"
            .format("；".join(problems))
        )
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


def infer_probability_class_count(experiment: ExperimentData) -> int:
    """从首个非空探针向量推导类别数，并拒绝无法识别的输入。"""

    for probe_rows in (
            experiment.client_probe, experiment.edge_probe, experiment.cloud_probe
    ):
        for row in probe_rows:
            for vector in row:
                if vector is None:
                    continue
                values = np.asarray(vector)
                if values.ndim != 1 or values.size < 2:
                    raise ValueError("无法从探针推导有效类别数：{}".format(experiment.path))
                return int(values.size)
    raise ValueError("实验没有任何非空概率探针：{}".format(experiment.path))


def build_batch_profile(
        input_dir: Path, experiments: Sequence[ExperimentData]
) -> Dict[str, object]:
    """汇总批次公共元数据，并在四方案关键比较口径不一致时终止。"""

    if len(experiments) != len(SCENARIO_ORDER):
        raise ValueError("批次必须包含四个目标实验")
    profile_fields = {
        "client_num_in_total": [
            int(experiment.metadata.get("client_num_in_total", -1))
            for experiment in experiments
        ],
        "client_num_per_round": [
            int(experiment.metadata.get("client_num_per_round", -1))
            for experiment in experiments
        ],
        "configured_comm_round": [
            int(experiment.metadata.get("configured_comm_round", -1))
            for experiment in experiments
        ],
        "partition_alpha": [
            experiment.metadata.get("partition_alpha") for experiment in experiments
        ],
        "topology_util": [
            experiment.metadata.get("topology_util") for experiment in experiments
        ],
        "random_seed": [
            experiment.metadata.get("random_seed") for experiment in experiments
        ],
        "model_distribution_scope": [
            experiment.metadata.get("model_distribution_scope") for experiment in experiments
        ],
        "experiment_tag": [
            experiment.metadata.get("experiment_tag") for experiment in experiments
        ],
        "probability_class_count": [
            infer_probability_class_count(experiment) for experiment in experiments
        ],
        "actual_rounds": [len(experiment.schedule) for experiment in experiments],
    }
    inconsistent = {
        key: values for key, values in profile_fields.items() if len(set(values)) != 1
    }
    if inconsistent:
        raise ValueError("四方案关键元数据不一致：{}".format(inconsistent))

    hfl_edge_slots = {
        int(experiment.metadata.get("group_capacity", len(experiment.edge_probe[0])))
        for experiment in experiments
        if experiment.scenario.startswith("hfl_") and experiment.edge_probe
    }
    if len(hfl_edge_slots) != 1:
        raise ValueError("两个HFL方案的边缘槽位数不一致：{}".format(sorted(hfl_edge_slots)))
    fl_edge_slots = {
        len(experiment.edge_probe[0])
        for experiment in experiments
        if experiment.scenario.startswith("fl_") and experiment.edge_probe
    }
    if len(fl_edge_slots) != 1:
        raise ValueError("两个FL方案的边缘探针槽位数不一致：{}".format(sorted(fl_edge_slots)))

    profile = {key: values[0] for key, values in profile_fields.items()}
    profile.update(
        {
            "input_dir": str(input_dir.resolve()),
            "batch_name": input_dir.resolve().name,
            "hfl_edge_slot_count": next(iter(hfl_edge_slots)),
            "fl_edge_slot_count": next(iter(fl_edge_slots)),
        }
    )
    return profile


def add_batch_name_check(
        quality_checks: List[Dict[str, object]], profile: Dict[str, object]
) -> None:
    """核对批次名中可明确识别的客户端数和利用率，不猜测含糊的变量含义。"""

    batch_name = str(profile["batch_name"])
    mismatches = []
    client_match = re.search(r"client(\d+)", batch_name, flags=re.IGNORECASE)
    if client_match and int(client_match.group(1)) != int(profile["client_num_in_total"]):
        mismatches.append(
            "目录名client{}，元数据client_num_in_total={}"
            .format(client_match.group(1), profile["client_num_in_total"])
        )
    util_match = re.search(r"util(\d+(?:p\d+)?)", batch_name, flags=re.IGNORECASE)
    if util_match:
        name_util = float(util_match.group(1).replace("p", "."))
        if not math.isclose(name_util, float(profile["topology_util"]), rel_tol=0.0, abs_tol=1e-12):
            mismatches.append(
                "目录名util{}，元数据topology_util={}"
                .format(util_match.group(1), profile["topology_util"])
            )
    if mismatches:
        add_quality_check(
            quality_checks, "跨实验", "批次目录名与结构化元数据一致",
            "注意", "；".join(mismatches), "中",
        )


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """沿最后一维计算归一化熵，均匀分布为1，one-hot分布为0。"""

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


def validate_probability_vector(vector: np.ndarray, class_count: int) -> bool:
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
    fixed_mapping_ok = True
    zero_rounds = 0
    candidate_sets = []
    client_total = int(experiment.metadata.get("client_num_in_total", -1))
    candidate_total = int(experiment.metadata.get("client_num_per_round", -1))
    class_count = infer_probability_class_count(experiment)
    hfl_edge_slot_count = int(
        experiment.metadata.get(
            "group_capacity", len(experiment.edge_probe[0]) if experiment.edge_probe else 0
        )
    )
    distribution_scope = str(experiment.metadata.get("model_distribution_scope", "active"))
    metadata_candidates = [
        int(value) for value in experiment.metadata.get(
            "fixed_candidate_client_indexes", []
        )
    ]
    metadata_slot_mapping = {
        int(slot): int(client_id)
        for slot, client_id in experiment.metadata.get(
            "mat_candidate_slot_to_client_index", {}
        ).items()
    }
    for index, record in enumerate(experiment.schedule):
        candidates = [int(value) for value in record["candidate_client_indexes"]]
        active = [int(value) for value in record["active_client_indexes"]]
        active_slots = [int(value) for value in record["mat_active_candidate_slots"]]
        group_mapping = {
            int(group_id): [int(value) for value in client_ids]
            for group_id, client_ids in record["group_to_client_indexes"].items()
        }
        group_slot_mapping = {
            int(group_id): [int(value) for value in slots]
            for group_id, slots in record["mat_group_to_candidate_slots"].items()
        }
        group_counts = {int(group_id): int(value) for group_id, value in record["mat_group_client_counts"].items()}
        group_union = [client_id for client_ids in group_mapping.values() for client_id in client_ids]
        candidate_sets.append(tuple(candidates))
        epoch_ok = epoch_ok and int(record["global_epoch"]) == index
        candidate_ok = candidate_ok and len(candidates) == candidate_total
        candidate_ok = candidate_ok and len(set(candidates)) == candidate_total
        candidate_ok = candidate_ok and all(0 <= client_id < client_total for client_id in candidates)
        active_ok = active_ok and set(active).issubset(set(candidates)) and len(active) == len(set(active))
        active_ok = active_ok and int(record["active_client_count"]) == len(active)
        group_ok = group_ok and len(group_union) == len(set(group_union))
        group_ok = group_ok and set(group_union) == set(active)
        group_ok = group_ok and sum(group_counts.values()) == len(active)
        group_ok = group_ok and sorted(
            slot for slots in group_slot_mapping.values() for slot in slots
        ) == sorted(active_slots)
        group_ok = group_ok and all(0 <= slot < len(candidates) for slot in active_slots)
        group_ok = group_ok and sorted(candidates[slot] for slot in active_slots) == sorted(active)
        group_ok = group_ok and all(
            [candidates[slot] for slot in group_slot_mapping.get(group_id, [])]
            == group_mapping.get(group_id, [])
            for group_id in set(group_slot_mapping) | set(group_mapping)
        )

        if metadata_candidates:
            fixed_mapping_ok = fixed_mapping_ok and candidates == metadata_candidates
        if metadata_slot_mapping:
            fixed_mapping_ok = fixed_mapping_ok and all(
                metadata_slot_mapping.get(slot) == client_id
                for slot, client_id in enumerate(candidates)
            )

        if len(active) == 0:
            zero_rounds += 1
            distribution_ok = distribution_ok and not bool(record["aggregated"])
        else:
            distribution_ok = distribution_ok and bool(record["aggregated"])
        expected_distributed = list(range(client_total)) if distribution_scope == "all" else active
        distribution_ok = distribution_ok and [
            int(value) for value in record["distributed_client_indexes"]
        ] == expected_distributed
        distribution_ok = distribution_ok and int(
            record["distributed_client_count"]
        ) == len(expected_distributed)

        client_vectors = experiment.client_probe[index]
        edge_vectors = experiment.edge_probe[index]
        cloud_vectors = experiment.cloud_probe[index]
        probability_ok = probability_ok and len(client_vectors) == candidate_total
        probability_ok = probability_ok and len(cloud_vectors) == 1
        probability_ok = probability_ok and all(vector is not None for vector in client_vectors)
        probability_ok = probability_ok and all(vector is not None for vector in cloud_vectors)
        probability_ok = probability_ok and all(
            validate_probability_vector(vector, class_count)
            for vector in client_vectors + edge_vectors + cloud_vectors
            if vector is not None
        )
        nonempty_group_count = sum(1 for client_ids in group_mapping.values() if client_ids)
        if experiment.scenario.startswith("hfl_"):
            edge_ok = edge_ok and len(edge_vectors) == hfl_edge_slot_count
            edge_ok = edge_ok and all(
                (edge_vectors[group_id] is not None) == bool(group_mapping.get(group_id, []))
                for group_id in range(hfl_edge_slot_count)
            )
            edge_ok = edge_ok and len(nonempty_probability_vectors(edge_vectors)) == nonempty_group_count
        else:
            edge_ok = edge_ok and len(edge_vectors) == 1 and edge_vectors[0] is None

    add_quality_check(
        checks, experiment.label, "global_epoch 连续且无重复", "通过" if epoch_ok else "未通过",
        "期望 0..{}".format(round_count - 1), "关键",
    )
    add_quality_check(
        checks, experiment.label,
        "一级采样为{}中{}个唯一客户端".format(client_total, candidate_total),
        "通过" if candidate_ok else "未通过",
        "共 {} 轮；候选集合出现 {} 种".format(round_count, len(set(candidate_sets))), "关键",
    )
    add_quality_check(
        checks, experiment.label, "固定候选与MAT槽位映射稳定",
        "通过" if fixed_mapping_ok and len(set(candidate_sets)) == 1 else "未通过",
        "metadata和每轮JSONL均应使用同一组、同一顺序的{}个客户端".format(candidate_total),
        "关键",
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
        "下发范围={}；零参与轮{}个仍应下发{}人且不产生新聚合".format(
            distribution_scope, zero_rounds, client_total if distribution_scope == "all" else 0
        ), "关键",
    )
    add_quality_check(
        checks, experiment.label, "探针概率合法且层级列数匹配", "通过" if probability_ok and edge_ok else "未通过",
        "客户端{}列、概率{}维、云端1列；HFL边缘{}槽位，FL边缘为空".format(
            candidate_total, class_count, hfl_edge_slot_count
            if experiment.scenario.startswith("hfl_") else 0
        ), "关键",
    )
    add_quality_check(
        checks, experiment.label, "实际训练调用严格等于MAT活跃集合", "无法独立验证",
        "JSONL记录了活跃集合，但没有逐客户端训练事件或模型更新哈希", "中",
    )
    add_quality_check(
        checks, experiment.label, "运行时MAT绝对路径仍可访问",
        "通过" if Path(str(experiment.metadata.get("mat_file", ""))).is_file() else "注意",
        "报告计算以JSONL实际{}行为准；metadata路径={}".format(
            round_count, experiment.metadata.get("mat_file")
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
        candidate_vectors = nonempty_probability_vectors(experiment.client_probe[index])
        cloud_vectors = nonempty_probability_vectors(experiment.cloud_probe[index])
        candidate_matrix = np.stack(candidate_vectors, axis=0)
        cloud_probability = cloud_vectors[0]
        # JSONL保存的是MAT真实槽位，优先直接使用，避免按连续人数错误还原分组。
        active_slots = [int(value) for value in record["mat_active_candidate_slots"]]
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
        group_slot_mapping = {
            int(group_id): [int(value) for value in slots]
            for group_id, slots in record["mat_group_to_candidate_slots"].items()
        }
        for group_id in sorted(group_slot_mapping):
            group_slots = group_slot_mapping[group_id]
            if not group_slots:
                continue
            group_matrix = candidate_matrix[group_slots]
            group_a, group_c, group_s = consensus_components(group_matrix)
            group_agreement_values.append((len(group_slots), group_a))
            group_certainty_values.append((len(group_slots), group_c))
            group_effective_values.append((len(group_slots), group_s))

        edge_vectors = nonempty_probability_vectors(experiment.edge_probe[index])
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
            "active_candidate_slots": json.dumps(active_slots, ensure_ascii=False),
            "active_client_ids": json.dumps(active, ensure_ascii=False),
            "group_to_candidate_slots": json.dumps(
                group_slot_mapping, ensure_ascii=False, sort_keys=True
            ),
            "group_to_client_ids": json.dumps(group_mapping, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)

    for index, row in enumerate(rows):
        # 第一轮没有前序模型，把变化量定义为0，后续用于参与规模与精度波动关系图。
        row["test_acc_delta"] = 0.0 if index == 0 else float(
            row["test_acc"] - rows[index - 1]["test_acc"]
        )

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
        "active_agreement",
        "active_certainty",
        "active_effective",
        "within_group_agreement",
        "within_group_certainty",
        "within_group_effective",
        "edge_agreement",
        "edge_certainty",
        "edge_effective",
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
            "active_agreement",
            "active_effective",
            "within_group_agreement",
            "within_group_certainty",
            "within_group_effective",
            "edge_agreement",
            "edge_certainty",
            "edge_effective",
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
    active_attainment = historical_best([float(row["active_effective_ma"]) for row in rows])
    for index, value in enumerate(active_attainment):
        rows[index]["active_effective_historical_best"] = float(value)
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
    active_s = np.asarray([float(row["active_effective"]) for row in round_rows])
    candidate_s_ma = np.asarray([float(row["candidate_effective_ma"]) for row in round_rows])
    active_s_ma = np.asarray([float(row["active_effective_ma"]) for row in round_rows])
    test_acc_ma = np.asarray([float(row["test_acc_ma"]) for row in round_rows])
    test_acc_delta = np.asarray([float(row["test_acc_delta"]) for row in round_rows])
    test_acc_changes = np.diff(test_acc)

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
        "last20_test_acc_mean": float(np.mean(test_acc[-20:])),
        "last20_test_acc_std": float(np.std(test_acc[-20:])),
        "max_single_round_drop": float(np.min(test_acc_changes)),
        "max_drop_epoch": int(np.argmin(test_acc_changes) + 2),
        "final_test_loss": float(test_loss[-1]),
        "min_test_loss": float(np.min(test_loss)),
        "last20_test_loss_mean": float(np.mean(test_loss[-20:])),
        "last10_train_acc_mean": float(np.mean(train_acc[-10:])),
        "last20_train_acc_mean": float(np.mean(train_acc[-20:])),
        "last10_generalization_gap": float(np.mean(train_acc[-10:] - test_acc[-10:])),
        "last20_generalization_gap": float(np.mean(train_acc[-20:] - test_acc[-20:])),
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
        "effective_last20": float(np.mean(candidate_s[-20:])),
        "effective_last20_std": float(np.std(candidate_s[-20:])),
        "active_effective_last10": float(np.nanmean(active_s[-10:])),
        "active_effective_last20": float(np.nanmean(active_s[-20:])),
        "effective_ma_best": float(np.nanmax(candidate_s_ma)),
        "active_effective_ma_best": float(np.nanmax(active_s_ma)),
        "consensus_accuracy_level_corr": safe_correlation(candidate_s_ma, test_acc_ma),
        "consensus_accuracy_diff_corr": safe_correlation(
            np.diff(candidate_s_ma), np.diff(test_acc_ma)
        ),
        "strongest_lag": None if strongest_lag is None else int(strongest_lag["lag"]),
        "strongest_lag_corr": float("nan") if strongest_lag is None else float(strongest_lag["correlation"]),
        "active_count_accuracy_delta_corr": safe_correlation(active, test_acc_delta),
        "smooth_window": smooth_window,
    }
    for threshold in SUMMARY_THRESHOLDS:
        key = "stable_epoch_{:.2f}".format(threshold)
        result[key] = first_stable_epoch(test_acc, threshold, window=5)
    return result


def build_client_statistics(experiments: Sequence[ExperimentData]) -> List[Dict[str, object]]:
    """按实验和固定MAT槽位统计候选客户端的实际参与频率。"""

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
        fixed_candidates = [
            int(value) for value in experiment.metadata.get(
                "fixed_candidate_client_indexes",
                experiment.schedule[0]["candidate_client_indexes"],
            )
        ]
        for candidate_slot, client_id in enumerate(fixed_candidates):
            candidate_count = int(candidate_frequency[client_id])
            active_count = int(active_frequency[client_id])
            rows.append(
                {
                    "实验": experiment.label,
                    "场景": experiment.scenario,
                    "MAT槽位": candidate_slot,
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
                "后20轮准确率差_百分点": 100.0 * (
                    float(left["last20_test_acc_mean"]) - float(right["last20_test_acc_mean"])
                ),
                "后10轮准确率差_百分点": 100.0 * (
                    float(left["last10_test_acc_mean"]) - float(right["last10_test_acc_mean"])
                ),
                "最终准确率差_百分点": 100.0 * (
                    float(left["final_test_acc"]) - float(right["final_test_acc"])
                ),
                "平均活跃人数差": float(left["active_mean"]) - float(right["active_mean"]),
            }
        )
    hfl_gain = rows[0]["后20轮准确率差_百分点"]
    fl_gain = rows[1]["后20轮准确率差_百分点"]
    rows.append(
        {
            "对比": "SnF增益的差分中的差分",
            "左方案": "HFL中SnF增益",
            "右方案": "FL中SnF增益",
            "后20轮准确率差_百分点": float(hfl_gain - fl_gain),
            "后10轮准确率差_百分点": float(
                rows[0]["后10轮准确率差_百分点"]
                - rows[1]["后10轮准确率差_百分点"]
            ),
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


def draw_compatible_boxplot(
        axis: plt.Axes, distributions: Sequence[Sequence[float]], labels: Sequence[str],
        **kwargs: object
) -> Dict[str, object]:
    """兼容新旧Matplotlib的箱线图标签参数并返回图形对象。"""

    try:
        # Matplotlib 3.9及以后使用tick_labels，旧版仍使用labels。
        return axis.boxplot(distributions, tick_labels=labels, **kwargs)
    except TypeError:
        return axis.boxplot(distributions, labels=labels, **kwargs)


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
        "实际{}轮；浅线为每轮原始值，粗线为{}轮尾随均值".format(
            len(next(iter(rows_by_scenario.values()))), smooth_window
        ),
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
    max_active = max(
        int(row["active_count"])
        for rows in rows_by_scenario.values()
        for row in rows
    )
    active_upper = max(1.0, max_active * 1.08)
    style_axis(axes[0, 0], "实际参与聚合客户端数", (0, active_upper))
    style_axis(axes[0, 1], "累计聚合客户端次")
    axes[0, 0].set_title("每轮实际参与人数", loc="left", fontweight="bold")
    axes[0, 1].set_title("累计有效聚合贡献", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")

    distributions = [
        [int(row["active_count"]) for row in rows_by_scenario[scenario]]
        for scenario in SCENARIO_ORDER
    ]
    boxes = draw_compatible_boxplot(
        axes[1, 0], distributions,
        [SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER],
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
    axes[1, 1].set_ylim(0, max(1.0, max(mean_active) * 1.25))
    axes[1, 1].set_ylabel("平均实际参与人数")
    axes[1, 1].set_title("参与均值与空聚合轮", loc="left", fontweight="bold")
    axes[1, 1].grid(True, axis="y", color="#EAECF0")
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    add_figure_header(
        figure,
        "实际参与强度和累计聚合贡献",
        "活跃人数来自topology_schedule.jsonl实际{}轮；零参与轮保留上一云模型".format(
            len(next(iter(rows_by_scenario.values())))
        ),
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
        "该横轴是活跃客户端次数代理量，不等同于样本量、计算量、通信字节数或真实耗时",
    )
    save_figure(figure, path)
    return path


def plot_consensus_decomposition(
        rows_by_scenario: Dict[str, List[Dict[str, object]]], figure_dir: Path, smooth_window: int
) -> Path:
    """以四个小图展示候选共识分解，并与MAT活跃客户端共识对照。"""

    path = figure_dir / "04_有效共识分解.png"
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, sharey=True)
    metric_styles = [
        ("candidate_agreement", "candidate_agreement_ma", "候选一致性 A", "#667085", ":"),
        ("candidate_certainty", "candidate_certainty_ma", "候选确定性 C", "#2563EB", "--"),
        ("candidate_effective", "candidate_effective_ma", "候选有效共识 S", "#B7791F", "-"),
        ("active_effective", "active_effective_ma", "活跃客户端有效共识 S", "#C2410C", "-."),
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
    candidate_total = int(rows_by_scenario[SCENARIO_ORDER[0]][0]["candidate_count"])
    add_figure_header(
        figure,
        "固定{}候选与MAT活跃客户端的有效共识分解".format(candidate_total),
        "浅线为原始值，粗线为{}轮尾随均值；活跃共识排除未训练候选保留旧云模型造成的多数效应".format(smooth_window),
    )
    save_figure(figure, path)
    return path


def plot_candidate_consensus_comparison(
        rows_by_scenario: Dict[str, List[Dict[str, object]]],
        summaries: Dict[str, Dict[str, object]], figure_dir: Path, smooth_window: int
) -> Path:
    """在统一坐标中直接比较四方案固定候选的有效共识S。"""

    path = figure_dir / "08_四方案候选有效共识S对比.png"
    figure, axes = plt.subplots(
        1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [2.15, 1.0]}
    )

    # 左图保留逐轮原始值作为波动背景，并用粗线突出相同窗口的平滑趋势。
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        raw = np.asarray([float(row["candidate_effective"]) for row in rows])
        smooth = np.asarray([float(row["candidate_effective_ma"]) for row in rows])
        color = SCENARIO_COLORS[scenario]
        linestyle = SCENARIO_LINESTYLES[scenario]
        axes[0].plot(
            epochs, raw, color=color, linestyle=linestyle, linewidth=0.7, alpha=0.14
        )
        axes[0].plot(
            epochs, smooth, color=color, linestyle=linestyle, linewidth=2.2,
            label=SCENARIO_LABELS[scenario],
        )
    style_axis(axes[0], "候选有效共识 S", (0, 1.02))
    axes[0].set_title("逐轮趋势与{}轮尾随均值".format(smooth_window), loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)

    # 右图从零开始显示绝对水平；误差线仅表示最后20轮的轮间波动。
    positions = np.arange(len(SCENARIO_ORDER))
    last20_means = np.asarray([
        float(summaries[scenario]["effective_last20"]) for scenario in SCENARIO_ORDER
    ])
    last20_stds = np.asarray([
        float(summaries[scenario]["effective_last20_std"]) for scenario in SCENARIO_ORDER
    ])
    bars = axes[1].bar(
        positions, last20_means, width=0.68,
        color=[SCENARIO_COLORS[scenario] for scenario in SCENARIO_ORDER],
        edgecolor="#344054", linewidth=0.7, zorder=2,
    )
    axes[1].errorbar(
        positions, last20_means, yerr=last20_stds, fmt="none", ecolor="#344054",
        elinewidth=1.2, capsize=4, capthick=1.2, zorder=3,
    )
    upper_limit = min(1.02, max(0.82, float(np.max(last20_means + last20_stds)) * 1.15))
    for bar, mean, std in zip(bars, last20_means, last20_stds):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0, mean + std + 0.018,
            "{:.4f}".format(mean), ha="center", va="bottom", fontsize=9,
            color="#101828", fontweight="bold",
        )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER], rotation=14)
    axes[1].set_xlabel("方案")
    axes[1].set_ylabel("候选有效共识 S")
    axes[1].set_ylim(0, upper_limit)
    axes[1].grid(True, axis="y", color="#EAECF0", linewidth=0.8, zorder=0)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_title("最后20轮均值与轮间标准差", loc="left", fontweight="bold")

    candidate_total = int(rows_by_scenario[SCENARIO_ORDER[0]][0]["candidate_count"])
    add_figure_header(
        figure,
        "四方案固定{}候选的有效共识S对比".format(candidate_total),
        "左图浅线为逐轮值、粗线为{}轮尾随均值；右图误差线不是多种子置信区间".format(smooth_window),
    )
    save_figure(figure, path)
    return path


def plot_consensus_attainment(
        rows_by_scenario: Dict[str, List[Dict[str, object]]], figure_dir: Path, smooth_window: int
) -> Path:
    """逐方案对比候选与活跃客户端的当前平滑共识和历史最佳值。"""

    path = figure_dir / "05_平滑共识与历史最佳.png"
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, sharey=True)
    for axis, scenario in zip(axes.flat, SCENARIO_ORDER):
        rows = rows_by_scenario[scenario]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        candidate_smooth = np.asarray([float(row["candidate_effective_ma"]) for row in rows])
        active_smooth = np.asarray([float(row["active_effective_ma"]) for row in rows])
        candidate_best = np.asarray([
            float(row["candidate_effective_historical_best"]) for row in rows
        ])
        active_best = np.asarray([
            float(row["active_effective_historical_best"]) for row in rows
        ])
        axis.plot(
            epochs, candidate_smooth, color="#2563EB", linewidth=2.0,
            label="候选当前平滑S",
        )
        axis.plot(
            epochs, active_smooth, color="#C2410C", linestyle="-.", linewidth=2.0,
            label="活跃当前平滑S",
        )
        axis.plot(
            epochs, candidate_best, color="#2563EB", linestyle=":", linewidth=1.5,
            label="候选历史最佳",
        )
        axis.plot(
            epochs, active_best, color="#C2410C", linestyle=":", linewidth=1.5,
            label="活跃历史最佳",
        )
        style_axis(axis, "有效共识", (0, 1.02))
        axis.set_title(SCENARIO_LABELS[scenario], loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=8.5)
    add_figure_header(
        figure,
        "当前平滑共识、历史最佳与退化",
        "前{}轮为空值；历史最佳天然单调，当前平滑线下降才表示共识退化".format(smooth_window - 1),
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
    # 根据两种HFL方案的真实取值共同确定纵轴，避免很小的JS散度被固定宽轴压扁。
    divergence_values = [
        float(row[field])
        for scenario in hfl_scenarios
        for row in rows_by_scenario[scenario]
        for field, _, _, _ in divergence_fields
        if math.isfinite(float(row[field]))
    ]
    divergence_max = max(divergence_values) if divergence_values else 0.0
    divergence_upper = min(0.35, max(0.03, math.ceil(divergence_max * 1.2 * 100.0) / 100.0))
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
        style_axis(axes[1, column], "平均JS散度", (0, divergence_upper))
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


def build_fixed_candidate_activity_matrix(experiment: ExperimentData) -> np.ndarray:
    """将每轮MAT活跃槽位展开为固定候选槽位乘轮数的0/1矩阵。"""

    candidate_total = int(experiment.metadata["client_num_per_round"])
    matrix = np.zeros((candidate_total, len(experiment.schedule)), dtype=np.float64)
    for epoch, record in enumerate(experiment.schedule):
        for candidate_slot in record["mat_active_candidate_slots"]:
            matrix[int(candidate_slot), epoch] = 1.0
    return matrix


def plot_client_coverage_and_relationship(
        experiments: Sequence[ExperimentData], rows_by_scenario: Dict[str, List[Dict[str, object]]],
        summaries: Dict[str, Dict[str, object]], figure_dir: Path, smooth_window: int
) -> Path:
    """绘制固定候选槽位活跃热图及参与人数与测试准确率变化关系。"""

    path = figure_dir / "07_固定候选活跃与精度波动.png"
    figure = plt.figure(figsize=(16, 12.5))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25], hspace=0.48, wspace=0.18)
    experiment_by_scenario = {experiment.scenario: experiment for experiment in experiments}
    candidate_total = int(experiments[0].metadata["client_num_per_round"])

    for index, scenario in enumerate(SCENARIO_ORDER):
        row = index // 2
        column = index % 2
        axis = figure.add_subplot(grid[row, column])
        experiment = experiment_by_scenario[scenario]
        active_matrix = build_fixed_candidate_activity_matrix(experiment)
        axis.imshow(
            active_matrix, aspect="auto", origin="lower",
            cmap=ListedColormap(["#F8FAFC", SCENARIO_COLORS[scenario]]), interpolation="nearest",
        )
        fixed_candidates = [
            int(value) for value in experiment.metadata["fixed_candidate_client_indexes"]
        ]
        tick_slots = np.arange(0, len(fixed_candidates), 4)
        axis.set_yticks(tick_slots)
        axis.set_yticklabels([
            "{} ({})".format(slot, fixed_candidates[slot]) for slot in tick_slots
        ])
        axis.set_title("{} 的MAT活跃槽位".format(SCENARIO_LABELS[scenario]), loc="left", fontweight="bold")
        axis.set_ylabel("MAT槽位（真实客户端ID）")
        axis.set_xlabel("训练轮次")

    relation_axis = figure.add_subplot(grid[2, :])
    all_changes = []
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        active_count = np.asarray([float(row["active_count"]) for row in rows])
        accuracy_change = np.asarray([float(row["test_acc_delta"]) for row in rows])
        all_changes.extend(accuracy_change[1:].tolist())
        relation_axis.scatter(
            active_count[1:], accuracy_change[1:], s=24, alpha=0.55,
            facecolor=SCENARIO_COLORS[scenario], edgecolor="white", linewidth=0.35,
            label="{}（r={:.2f}）".format(
                SCENARIO_LABELS[scenario], summaries[scenario]["active_count_accuracy_delta_corr"]
            ),
        )
    relation_axis.axhline(0.0, color="#344054", linewidth=1.0, linestyle="--")
    relation_axis.set_xlabel("本轮MAT活跃客户端数")
    relation_axis.set_ylabel("相对上一轮的测试准确率变化")
    relation_axis.set_xlim(-0.5, candidate_total + 0.5)
    if all_changes:
        padding = max(0.02, 0.08 * (max(all_changes) - min(all_changes)))
        relation_axis.set_ylim(min(all_changes) - padding, max(all_changes) + padding)
    relation_axis.grid(True, color="#EAECF0")
    relation_axis.spines["top"].set_visible(False)
    relation_axis.spines["right"].set_visible(False)
    relation_axis.set_title("参与规模与本轮测试准确率变化的描述性关系", loc="left", fontweight="bold")
    relation_axis.legend(frameon=False, ncol=2, loc="lower right")

    add_figure_header(
        figure,
        "固定{}候选客户端的MAT活跃轨迹与精度波动".format(candidate_total),
        "热图深色表示该槽位实际训练并参与聚合；散点相关只描述同期关系，不构成因果证据",
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
                "全程平均测试准确率": item["mean_test_acc"],
                "后10轮测试准确率均值": item["last10_test_acc_mean"],
                "后10轮测试准确率标准差": item["last10_test_acc_std"],
                "后20轮测试准确率均值": item["last20_test_acc_mean"],
                "后20轮测试准确率标准差": item["last20_test_acc_std"],
                "最大单轮准确率下降": item["max_single_round_drop"],
                "最大下降发生轮次": item["max_drop_epoch"],
                "最终测试损失": item["final_test_loss"],
                "最低测试损失": item["min_test_loss"],
                "后20轮测试损失均值": item["last20_test_loss_mean"],
                "后10轮训练准确率均值": item["last10_train_acc_mean"],
                "后20轮训练准确率均值": item["last20_train_acc_mean"],
                "后10轮泛化差距_训练减测试": item["last10_generalization_gap"],
                "后20轮泛化差距_训练减测试": item["last20_generalization_gap"],
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
                "后20轮有效共识S": item["effective_last20"],
                "后20轮有效共识S标准差": item["effective_last20_std"],
                "后10轮活跃客户端有效共识S": item["active_effective_last10"],
                "后20轮活跃客户端有效共识S": item["active_effective_last20"],
                "最高10轮平滑有效共识": item["effective_ma_best"],
                "最高10轮平滑活跃客户端有效共识": item["active_effective_ma_best"],
                "平滑共识与平滑准确率同期相关": item["consensus_accuracy_level_corr"],
                "共识变化与准确率变化相关": item["consensus_accuracy_diff_corr"],
                "活跃人数与准确率变化同期相关": item["active_count_accuracy_delta_corr"],
                "最强滞后_正值表示共识领先": item["strongest_lag"],
                "最强滞后相关": item["strongest_lag_corr"],
            }
        )
    return rows


def export_round_rows(rows_by_scenario: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    """将四组逐轮指标合并为统一中文字段明细表。"""

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
        ("test_acc_delta", "测试准确率较上一轮变化"),
        ("test_loss", "测试损失"),
        ("generalization_gap", "泛化差距_训练减测试"),
        ("candidate_agreement", "候选一致性A"),
        ("candidate_certainty", "候选确定性C"),
        ("candidate_effective", "候选有效共识S"),
        ("candidate_effective_ma", "候选有效共识尾随均值"),
        ("candidate_effective_historical_best", "历史最佳平滑共识"),
        ("active_effective_historical_best", "参与者历史最佳平滑共识"),
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
        ("active_candidate_slots", "MAT活跃候选槽位"),
        ("active_client_ids", "最终参与客户端ID"),
        ("group_to_candidate_slots", "MAT分组候选槽位"),
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
    """只读检查当前YAML、训练器和结果记录的关键运行语义是否一致。"""

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
        "result_uses_fixed_candidates": all(
            len({tuple(record["candidate_client_indexes"]) for record in experiment.schedule}) == 1
            and str(experiment.metadata.get("candidate_sampling_mode", ""))
            == "fixed_once_by_random_seed"
            for experiment in experiments
        ),
        "result_distribution_scopes": {
            experiment.scenario: experiment.metadata.get("model_distribution_scope")
            for experiment in experiments
        },
        "runtime_mat_paths_exist": {
            experiment.scenario: Path(str(experiment.metadata.get("mat_file", ""))).is_file()
            for experiment in experiments
        },
    }


def add_cross_experiment_checks(
        quality_checks: List[Dict[str, object]], experiments: Sequence[ExperimentData],
        rows_by_scenario: Dict[str, List[Dict[str, object]]], drift: Dict[str, object]
) -> None:
    """追加四方案固定候选一致性、零轮模型保持和代码语义检查。"""

    reference = [record["candidate_client_indexes"] for record in experiments[0].schedule]
    identical = all(
        [record["candidate_client_indexes"] for record in experiment.schedule] == reference
        for experiment in experiments[1:]
    )
    candidate_total = int(experiments[0].metadata["client_num_per_round"])
    add_quality_check(
        quality_checks, "跨实验", "四方案固定候选顺序一致", "通过" if identical else "未通过",
        "四方案全部epoch共享同一组、同一顺序的{}个真实客户端".format(candidate_total), "关键",
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
    semantics_match = bool(
        drift["current_trainer_uses_fixed_candidate"]
        and drift["current_trainer_matlab_trains_only_active"]
        and drift["result_uses_fixed_candidates"]
    )
    add_quality_check(
        quality_checks, "跨实验", "当前代码与结果关键采样语义一致",
        "通过" if semantics_match else "注意",
        "固定候选={}；仅训练MAT活跃客户端={}；结果固定候选={}".format(
            drift["current_trainer_uses_fixed_candidate"],
            drift["current_trainer_matlab_trains_only_active"],
            drift["result_uses_fixed_candidates"],
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




def build_current_report_text(
        experiments: Sequence[ExperimentData], summaries: Dict[str, Dict[str, object]],
        contrasts: Sequence[Dict[str, object]], quality_checks: Sequence[Dict[str, object]],
        drift: Dict[str, object], profile: Dict[str, object], smooth_window: int
) -> str:
    """根据批次元数据和实际结果生成简体中文分析报告。"""

    best_scenario = max(SCENARIO_ORDER, key=lambda key: summaries[key]["last20_test_acc_mean"])
    stable_scenario = min(SCENARIO_ORDER, key=lambda key: summaries[key]["last20_test_acc_std"])
    hfl_gain = next(row for row in contrasts if row["对比"] == "HFL中SnF增益")
    fl_gain = next(row for row in contrasts if row["对比"] == "FL中SnF增益")

    candidate_total = int(profile["client_num_per_round"])
    client_total = int(profile["client_num_in_total"])
    class_count = int(profile["probability_class_count"])
    hfl_edge_slot_count = int(profile["hfl_edge_slot_count"])
    fl_edge_slot_count = int(profile["fl_edge_slot_count"])
    performance_rows = []
    participation_rows = []
    consensus_rows = []
    for scenario in SCENARIO_ORDER:
        item = summaries[scenario]
        performance_rows.append([
            item["label"], format_percent(item["final_test_acc"]),
            format_percent(item["best_test_acc"]),
            format_percent(item["last20_test_acc_mean"]),
            "{:.2f}个百分点".format(100.0 * item["last20_test_acc_std"]),
            "{:+.2f}个百分点（第{}轮）".format(
                100.0 * item["max_single_round_drop"], item["max_drop_epoch"]
            ),
            item["stable_epoch_0.80"] or "未稳定达到",
            item["stable_epoch_0.85"] or "未稳定达到",
            item["stable_epoch_0.88"] or "未稳定达到",
        ])
        participation_rows.append([
            item["label"], format_number(item["active_mean"], 2),
            "{}–{}".format(item["active_min"], item["active_max"]),
            item["active_total"], item["zero_active_rounds"],
            "{}/{}".format(item["active_coverage"], candidate_total), item["distributed_total"],
        ])
        consensus_rows.append([
            item["label"], format_number(item["effective_last20"], 4),
            format_number(item["effective_last20_std"], 4),
            format_number(item["active_effective_last20"], 4),
            format_number(item["effective_ma_best"], 4),
            format_number(item["active_effective_ma_best"], 4),
            format_number(item["active_count_accuracy_delta_corr"], 3),
        ])

    contrast_rows = [[
        row["对比"], "{:+.3f}".format(row["后20轮准确率差_百分点"]),
        "{:+.3f}".format(row["最终准确率差_百分点"]),
        "{:+.2f}".format(row["平均活跃人数差"]),
    ] for row in contrasts]
    quality_status_counts: Dict[str, int] = {}
    for row in quality_checks:
        quality_status_counts[row["状态"]] = quality_status_counts.get(row["状态"], 0) + 1

    round_counts = sorted({int(item["rounds"]) for item in summaries.values()})
    fixed_candidates = [
        int(value) for value in experiments[0].metadata["fixed_candidate_client_indexes"]
    ]
    yaml_rounds = sorted({
        value for value in drift["current_yaml_comm_round"].values() if value is not None
    })
    missing_mat_scenarios = [
        SCENARIO_LABELS[scenario]
        for scenario, exists in drift["runtime_mat_paths_exist"].items()
        if not exists
    ]
    candidate_consensus_ranking = " > ".join(
        "{}（{:.4f}）".format(summaries[scenario]["label"], summaries[scenario]["effective_last20"])
        for scenario in sorted(
            SCENARIO_ORDER, key=lambda key: summaries[key]["effective_last20"], reverse=True
        )
    )
    candidate_top_label = summaries[
        max(SCENARIO_ORDER, key=lambda key: summaries[key]["effective_last20"])
    ]["label"]
    if len(missing_mat_scenarios) == len(SCENARIO_ORDER):
        mat_path_statement = (
            "运行元数据中的MAT绝对路径在当前工作区均不可访问（{}）"
            .format("、".join(missing_mat_scenarios))
        )
    elif missing_mat_scenarios:
        mat_path_statement = (
            "部分运行元数据中的MAT绝对路径在当前工作区不可访问（{}）"
            .format("、".join(missing_mat_scenarios))
        )
    else:
        mat_path_statement = "四方案运行元数据中的MAT路径在当前工作区均可访问"
    trainer_semantics_ok = bool(
        drift["current_trainer_uses_fixed_candidate"]
        and drift["current_trainer_matlab_trains_only_active"]
    )
    trainer_semantics_text = (
        "当前 `trainer_test.py` 采用固定候选、仅训练MAT活跃客户端和全量下发语义"
        if trainer_semantics_ok
        else "当前 `trainer_test.py` 与结果记录的关键运行语义存在差异，详情见数据质量检查"
    )

    return """# 批次 `{batch_name}` 四组联邦学习实验分析报告

## 技术摘要

- **后20轮效果最好的是 {best_label}，波动最小的是 {stable_label}。** 最佳方案的后20轮平均测试准确率为 **{best_acc}**，其标准差为 **{best_std:.2f} 个百分点**。
- **本批SnF相对noSnF的后20轮差值在HFL内为 {hfl_gain:+.3f} 个百分点，在FL内为 {fl_gain:+.3f} 个百分点。** 这些差值不是纯SnF因果效应，因为对应方案的累计活跃客户端次数也可能不同。
- **四方案均记录了{rounds}轮，但有效参与预算并不相同。** HFL-SnF、HFL-noSnF、FL-SnF、FL-noSnF累计活跃客户端次数分别为 **{hfl_snf_total}、{hfl_no_total}、{fl_snf_total}、{fl_no_total}**。
- **证据评级：可分享但附带限制。** 结果完整且关键运行不变量通过检查，但只有单随机种子、探针每轮更换样本且没有真值标签，也没有可靠耗时和硬件日志。

## 1. 数据完整性与运行语义

四个实验的调度、训练/测试准确率、训练/测试损失和三层概率探针均为 **{rounds}轮**。客户端探针逐轮固定{candidate_total}列；HFL边缘探针固定{hfl_edge_slot_count}个物理槽位并保留未启用组的空单元格；FL边缘探针固定{fl_edge_slot_count}个槽位；云探针固定1列。所有非空概率向量均为{class_count}维、数值有限且概率和接近1。

固定候选顺序为：`{fixed_candidates}`。四方案、全部epoch均使用同一顺序，MAT活跃槽位、真实客户端分组和人数之和逐轮一致。正常轮产生聚合，零参与轮不产生新聚合；两种情况下都向0至{last_client_id}号全部客户端下发当前云模型。

数据质量检查共 {quality_total} 项：{quality_counts}。{mat_path_statement}，因此本报告以 `topology_schedule.jsonl` 的实际运行记录为控制来源，不使用本地MAT重新推测分组。

## 2. 模型效果与后期稳定性

最后20轮均值是主要后期指标；最终轮和峰值用于补充，不能用单个终点替代稳定性判断。

{performance_table}

HFL-SnF最后20轮平均准确率为 **{hfl_snf_acc}**，FL-SnF、HFL-noSnF、FL-noSnF依次为 **{fl_snf_acc}、{hfl_no_acc}、{fl_no_acc}**。当前后20轮均值最高的方案是 **{best_label}**，后20轮标准差最小的方案是 **{stable_label}**。

![四方案模型效果趋势](figures/01_模型效果趋势.png)

图中浅线为逐轮原始值，粗线为{smooth_window}轮尾随均值。训练准确率和测试准确率接近，没有出现明显的持续性训练—测试分离；但这只是当前单次运行的描述性现象。

## 3. 实际参与规模与预算代理

{participation_table}

这里的“累计活跃客户端次”只统计真正完成本地训练并进入聚合的客户端状态。它可以作为参与预算代理，但不能等同于样本量、FLOPs、通信字节数或真实训练时间。

![参与强度与累计贡献](figures/02_参与强度与累计贡献.png)

![累计参与预算效率](figures/03_聚合预算效率.png)

按通信轮次和按共同累计参与量比较时，方案排序可能不同。因此SnF/noSnF差值同时包含拓扑保留、参与规模和聚合路径的综合影响。

## 4. 固定候选、活跃客户端与有效共识

有效共识分成概率一致性A、预测确定性C和二者乘积S。只看A会把“所有模型共同接近均匀分布”误判成高共识，因此报告以S为主要共识指标。

{consensus_table}

全部{candidate_total}个候选的共识和MAT活跃客户端的共识必须分开解释。noSnF场景每轮只有少量候选训练，其余候选在上轮全量下发后保留相同云模型，容易形成较高的多数一致；活跃客户端指标更能反映当轮真正贡献聚合的模型差异。

![候选与活跃客户端有效共识](figures/04_有效共识分解.png)

![四方案候选有效共识S对比](figures/08_四方案候选有效共识S对比.png)

按候选有效共识S的最后20轮均值排序为：**{candidate_consensus_ranking}**。当前排序第一的是{candidate_top_label}；候选S会受到轮内未训练候选保留旧云模型的影响，因此该排序不能直接解释成真实参与客户端之间形成了更强共识，应与同表中的活跃客户端S一起阅读。

![当前平滑共识与历史最佳](figures/05_平滑共识与历史最佳.png)

历史最佳曲线天然单调不降，只表示曾经达到过的水平；判断退化必须读取当前平滑共识。每轮探针使用不同测试样本，所以曲线同时包含模型学习进展和样本难度变化，不能当成同一样本上的纯收敛轨迹。

## 5. HFL层级传播

![HFL层级共识传播](figures/06_HFL层级共识传播.png)

层级指标直接使用JSONL的MAT槽位到边缘组映射，组内指标按实际组人数加权；边缘空槽位不会被压缩后错误对应到其他组。边缘和云输出更集中并不自动表示预测正确，因为结果没有保存探针真实标签，无法区分正确共识和集体错误。

## 6. 固定{candidate_total}人的活跃公平性与精度波动

![固定候选活跃轨迹与精度波动](figures/07_固定候选活跃与精度波动.png)

热图纵轴同时给出MAT槽位和真实客户端ID，可检查{candidate_total}列身份是否稳定以及不同拓扑对候选参与频率的影响。散点图展示本轮活跃人数与相对上一轮测试准确率变化的同期关系；相关系数仅作描述性证据，不支持“增加参与人数必然导致当轮精度提升”的因果结论。

## 7. 2×2端到端方案对比

{contrast_table}

差分中的差分同样只是描述性结果。四个场景的MAT活跃人数、非空边缘数和累计参与预算不同，而且每个方案只有一个种子，不能据此做显著性检验或把差值归因给单一机制。

## 8. 可复现性和解释限制

- 当前工作区四份YAML配置轮数为 {yaml_rounds}，本批实际结果轮数为 {rounds}；两者是否一致见数据质量检查。{trainer_semantics_text}。
- 结果没有保存Git提交、完整运行环境、模型快照和逐客户端训练事件，因此不能从结果文件独立证明每一次本地训练调用确实发生。
- 运行时MAT绝对路径已失效，但JSONL保存了逐轮真实槽位、分组和下发对象，足以完成本报告的运行后分析。
- 探针每轮样本不同且没有真实标签，不能计算正确共识、错误共识或固定样本的纵向变化。
- 结果没有阶段耗时、GPU利用率、驱动和PyTorch CUDA信息，不能用于比较4090与4060训练速度。
- 单随机种子不支持置信区间、显著性检验或稳定性外推。

## 9. 建议的下一步

1. 下一批实验保存Git提交、完整YAML、MAT文件哈希、CUDA环境和分阶段耗时。
2. 使用至少5个随机种子，并在SnF/noSnF之间匹配每轮活跃人数或累计样本预算。
3. 将探针改为固定小批量并保存真实标签，分别报告正确共识、错误共识和样本难度分层结果。
4. 若研究通信效率，额外保存上下行模型字节数和真实传输次数，避免只使用客户端次数代理。

本报告由 `analyze_experiment_suite.py` 从原始结果重新计算；逐轮证据、质量检查和来源哈希见同目录CSV及 `analysis_manifest.json`。
""".format(
        batch_name=profile["batch_name"],
        best_label=summaries[best_scenario]["label"],
        best_acc=format_percent(summaries[best_scenario]["last20_test_acc_mean"]),
        best_std=100.0 * summaries[best_scenario]["last20_test_acc_std"],
        stable_label=summaries[stable_scenario]["label"],
        hfl_gain=hfl_gain["后20轮准确率差_百分点"],
        fl_gain=fl_gain["后20轮准确率差_百分点"],
        hfl_snf_total=summaries["hfl_snf_fixed"]["active_total"],
        hfl_no_total=summaries["hfl_no_snf_fixed"]["active_total"],
        fl_snf_total=summaries["fl_snf"]["active_total"],
        fl_no_total=summaries["fl_no_snf"]["active_total"],
        rounds="、".join(str(value) for value in round_counts),
        fixed_candidates=json.dumps(fixed_candidates, ensure_ascii=False),
        candidate_total=candidate_total,
        hfl_edge_slot_count=hfl_edge_slot_count,
        fl_edge_slot_count=fl_edge_slot_count,
        class_count=class_count,
        last_client_id=client_total - 1,
        quality_total=len(quality_checks),
        quality_counts="，".join(
            "{}{}项".format(key, value) for key, value in sorted(quality_status_counts.items())
        ),
        mat_path_statement=mat_path_statement,
        performance_table=markdown_table(
            ["方案", "最终准确率", "峰值", "后20轮均值", "后20轮标准差", "最大单轮下降", "稳定80%", "稳定85%", "稳定88%"],
            performance_rows,
        ),
        hfl_snf_acc=format_percent(summaries["hfl_snf_fixed"]["last20_test_acc_mean"]),
        fl_snf_acc=format_percent(summaries["fl_snf"]["last20_test_acc_mean"]),
        hfl_no_acc=format_percent(summaries["hfl_no_snf_fixed"]["last20_test_acc_mean"]),
        fl_no_acc=format_percent(summaries["fl_no_snf"]["last20_test_acc_mean"]),
        smooth_window=smooth_window,
        participation_table=markdown_table(
            ["方案", "平均活跃人数", "范围", "累计活跃客户端次", "零参与轮", "{}人中活跃覆盖".format(candidate_total), "累计下发客户端次"],
            participation_rows,
        ),
        consensus_table=markdown_table(
            ["方案", "候选后20轮S", "候选后20轮S标准差", "活跃后20轮S", "候选最高MA10", "活跃最高MA10", "活跃人数—精度变化相关"],
            consensus_rows,
        ),
        candidate_consensus_ranking=candidate_consensus_ranking,
        candidate_top_label=candidate_top_label,
        contrast_table=markdown_table(
            ["对比", "后20轮准确率差/百分点", "最终差/百分点", "平均活跃人数差"],
            contrast_rows,
        ),
        yaml_rounds="、".join(str(value) for value in yaml_rounds) or "未知",
        trainer_semantics_text=trainer_semantics_text,
    )


def build_manifest(
        experiments: Sequence[ExperimentData], summaries: Dict[str, Dict[str, object]],
        quality_checks: Sequence[Dict[str, object]], figure_paths: Sequence[Path],
        output_dir: Path, smooth_window: int, drift: Dict[str, object],
        profile: Dict[str, object]
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
    candidate_total = int(profile["client_num_per_round"])
    chart_questions = [
        "四方案的准确率、损失与后期波动如何变化？",
        "实际参与强度和累计聚合贡献有多大差异？",
        "在累计聚合客户端次口径下，各方案的学习效率如何？",
        "一致性、确定性和有效共识如何共同变化？",
        "当前平滑共识与单调历史最佳值有什么区别？",
        "HFL中客户端、边缘和云端的确定性及分歧如何传播？",
        "固定{}槽位如何活跃，参与人数与当轮准确率变化有什么关系？".format(candidate_total),
        "四种方案的固定{}候选有效共识S如何变化，后20轮水平有何差异？".format(candidate_total),
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
        "schema_version": "3.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "confidence": "可分享但附带限制",
        "source_precedence": [
            "topology_schedule.jsonl和指标/探针文件",
            "topology_metadata.json",
            "当前YAML和代码",
        ],
        "analysis_parameters": {
            "smooth_window": smooth_window,
            "stable_threshold_window": 5,
            "thresholds": list(SUMMARY_THRESHOLDS),
            "lag_range": [-10, 10],
        },
        "batch_profile": profile,
        "sources": sources,
        "summaries": summaries,
        "quality_checks": list(quality_checks),
        "workspace_drift": drift,
        "chart_map": chart_map,
        "outputs": [
            "分析报告.md",
            "实验汇总.csv",
            "逐轮指标.csv",
            "固定候选参与统计.csv",
            "数据质量检查.csv",
            "共识准确率相关.csv",
            "方案对比.csv",
            "analysis_manifest.json",
        ] + [str(path.relative_to(output_dir)) for path in figure_paths],
        "limitations": [
            "只有单随机种子，不能声明统计显著",
            "没有逐客户端训练事件或更新哈希，不能从结果文件独立证明每次本地训练调用",
            "没有模型快照，不能独立复算指标是否来自最终云模型",
            "探针样本每轮变化且没有保存真值标签",
            "运行时MAT绝对路径在当前工作区失效，分析以JSONL为准",
            "没有可靠运行时间和通信字节日志",
        ],
    }


def sanitize_path_component(value: str) -> str:
    """把批次名转换为可安全用于Windows目录名的短文本。"""

    # 保留中文、字母、数字、点、下划线和连字符，其他字符统一替换为下划线。
    sanitized = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", value, flags=re.UNICODE).strip("._")
    return sanitized or "未命名批次"


def derive_output_dir(
        output_root: Path, input_dir: Path, experiments: Sequence[ExperimentData]
) -> Path:
    """根据批次名、实际轮数和数据日期生成不含实验参数猜测的输出目录。"""

    round_counts = {len(experiment.schedule) for experiment in experiments}
    if len(round_counts) != 1:
        raise ValueError("四组实验轮数不同，不能自动生成统一输出目录")
    date_values = []
    for experiment in experiments:
        match = re.search(r"(\d{8})$", experiment.path.name)
        if match:
            date_values.append(match.group(1))
    date_label = (
        date_values[0]
        if date_values and len(set(date_values)) == 1
        else datetime.now().strftime("%Y%m%d")
    )
    rounds = next(iter(round_counts))
    batch_name = sanitize_path_component(input_dir.resolve().name)
    return output_root.resolve() / "analysis_{}_{}rounds_{}".format(
        batch_name, rounds, date_label
    )


def choose_available_output_dir(candidate: Path) -> Path:
    """在自动输出目录已存在时追加序号，避免覆盖任何旧分析包。"""

    candidate = candidate.resolve()
    if not candidate.exists():
        return candidate
    for sequence in range(2, 10000):
        alternative = candidate.with_name("{}_{}".format(candidate.name, sequence))
        if not alternative.exists():
            return alternative
    raise RuntimeError("无法为分析结果找到可用输出目录：{}".format(candidate.parent))


def validate_explicit_output_dir(output_dir: Path) -> Path:
    """允许新目录或空目录作为显式输出，拒绝覆盖已有文件。"""

    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("输出目录已存在且非空，为避免覆盖已停止：{}".format(output_dir))
    return output_dir


def _run_analysis(
        input_dir: Path, output_dir: Optional[Path], smooth_window: int,
        output_root: Optional[Path] = None,
        explicit_dirs: Optional[Sequence[Path]] = None
) -> Path:
    """执行共享分析流程，并支持终端入口传入额外的输出根目录和实验目录。"""

    if smooth_window < 2:
        raise ValueError("平滑窗口至少为2")
    workspace = locate_project_root()
    input_dir = input_dir.resolve()
    experiment_dirs = resolve_experiment_dirs(input_dir, explicit_dirs)
    experiments = [load_experiment(path) for path in experiment_dirs]
    experiment_by_scenario = {experiment.scenario: experiment for experiment in experiments}
    experiments = [experiment_by_scenario[scenario] for scenario in SCENARIO_ORDER]
    profile = build_batch_profile(input_dir, experiments)

    quality_checks = []
    for experiment in experiments:
        quality_checks.extend(validate_experiment(experiment))
    critical_failures = [
        check for check in quality_checks
        if check["严重级别"] == "关键" and check["状态"] == "未通过"
    ]
    if critical_failures:
        # 非法概率、轮次错位、候选/下发异常等关键问题不能只写入报告后继续运行。
        failure_text = "；".join(
            "{}：{}（{}）".format(check["实验"], check["检查项"], check["证据"])
            for check in critical_failures
        )
        raise ValueError("关键数据校验未通过：{}".format(failure_text))
    add_quality_check(
        quality_checks, "跨实验", "四方案关键元数据一致", "通过",
        "客户端总数{}、每轮候选{}、实际轮数{}、类别数{}"
        .format(
            profile["client_num_in_total"], profile["client_num_per_round"],
            profile["actual_rounds"], profile["probability_class_count"],
        ),
        "关键",
    )
    add_batch_name_check(quality_checks, profile)

    rows_by_scenario = {
        experiment.scenario: build_round_metrics(experiment, smooth_window)
        for experiment in experiments
    }
    summaries = {
        experiment.scenario: summarize_experiment(
            experiment, rows_by_scenario[experiment.scenario], smooth_window
        )
        for experiment in experiments
    }
    contrasts = build_contrasts(summaries)
    client_statistics = build_client_statistics(experiments)
    correlation_rows = build_correlation_rows(rows_by_scenario)
    drift = detect_workspace_drift(workspace, experiments)
    add_cross_experiment_checks(quality_checks, experiments, rows_by_scenario, drift)

    if output_dir is not None:
        resolved_output_dir = validate_explicit_output_dir(output_dir)
    else:
        resolved_output_root = (
            output_root.resolve()
            if output_root is not None
            else workspace / "result" / "1结果和分析"
        )
        resolved_output_dir = choose_available_output_dir(
            derive_output_dir(resolved_output_root, input_dir, experiments)
        )
    output_dir = resolved_output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    figure_paths = [
        plot_model_metrics(rows_by_scenario, figure_dir, smooth_window),
        plot_participation(rows_by_scenario, summaries, figure_dir),
        plot_aggregation_efficiency(rows_by_scenario, summaries, figure_dir),
        plot_consensus_decomposition(rows_by_scenario, figure_dir, smooth_window),
        plot_consensus_attainment(rows_by_scenario, figure_dir, smooth_window),
        plot_hierarchy_consensus(rows_by_scenario, figure_dir, smooth_window),
        plot_client_coverage_and_relationship(
            experiments, rows_by_scenario, summaries, figure_dir, smooth_window
        ),
        plot_candidate_consensus_comparison(
            rows_by_scenario, summaries, figure_dir, smooth_window
        ),
    ]

    summary_rows = export_summary_rows(summaries)
    round_rows = export_round_rows(rows_by_scenario)
    write_csv(output_dir / "实验汇总.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(output_dir / "逐轮指标.csv", round_rows, list(round_rows[0].keys()))
    write_csv(output_dir / "固定候选参与统计.csv", client_statistics, list(client_statistics[0].keys()))
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
        [
            "对比", "左方案", "右方案", "后20轮准确率差_百分点",
            "后10轮准确率差_百分点", "最终准确率差_百分点", "平均活跃人数差",
        ],
    )

    manifest = build_manifest(
        experiments, summaries, quality_checks, figure_paths,
        output_dir, smooth_window, drift, profile,
    )
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_text = build_current_report_text(
        experiments, summaries, contrasts, quality_checks, drift, profile, smooth_window
    )
    (output_dir / "分析报告.md").write_text(report_text, encoding="utf-8")

    print("输入目录：{}".format(input_dir))
    print("输出目录：{}".format(output_dir))
    print("报告路径：{}".format((output_dir / "分析报告.md").resolve()))
    print("实际轮数：{}；逐轮指标行数：{}".format(len(experiments[0].schedule), len(round_rows)))
    for scenario in SCENARIO_ORDER:
        item = summaries[scenario]
        print(
            "{}：后20轮测试准确率 {:.4f}，累计参与 {}，后20轮活跃有效共识 {:.4f}".format(
                item["label"], item["last20_test_acc_mean"],
                item["active_total"], item["active_effective_last20"],
            )
        )
    return output_dir


def run_analysis(
        input_dir: Path, output_dir: Optional[Path] = None, smooth_window: int = 10
) -> Path:
    """从指定批次读取四组实验并生成完整分析包，返回输出目录。"""

    return _run_analysis(Path(input_dir), output_dir, smooth_window)


def main() -> None:
    """解析终端参数并调用与IDE入口相同的共享分析流程。"""

    args = parse_args()
    workspace = locate_project_root()
    ##修改这里尝试使用不同的数据
    input_dir = args.input_dir or args.result_root or workspace / "result" / "originalData" / "varAlpha_0p1_client200_util0p6"
    _run_analysis(
        input_dir=Path(input_dir),
        output_dir=args.output_dir,
        smooth_window=args.smooth_window,
        output_root=args.output_root,
        explicit_dirs=args.experiment_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # 终端入口将常见数据或路径异常压缩为一行明确错误。
        print("分析失败：{}".format(exc), file=sys.stderr)
        raise SystemExit(1)
