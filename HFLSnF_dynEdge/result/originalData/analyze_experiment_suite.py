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

# 分析脚本位于 result/originalData；显式加入项目根目录以复用训练端同一指标公式。
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from probe_metrics import calculate_population_probe_metrics


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

COMMON_REQUIRED_RESULT_FILES = [
    "topology_metadata.json",
    "topology_schedule.jsonl",
    "train_acc.txt",
    "train_loss.txt",
    "test_acc.txt",
    "test_loss.txt",
]

LEGACY_PROBE_FILES = [
    "probe_client_pre.csv",
    "probe_edge_post.csv",
    "probe_cloud_post.csv",
]

NPZ_PROBE_FILES = [
    "probe_probabilities.npz",
    "probe_epoch_summary.csv",
]

# 保留旧公开常量，避免已有测试和外部脚本导入后失效。
REQUIRED_RESULT_FILES = COMMON_REQUIRED_RESULT_FILES + LEGACY_PROBE_FILES

# 新训练会额外保存真实标签；旧批次没有该文件时仍可继续生成兼容报告。
OPTIONAL_RESULT_FILES = [
    "probe_meta.csv",
]

SUMMARY_THRESHOLDS = (0.80, 0.85, 0.88)
CONSENSUS_THRESHOLDS = (0.60, 0.70, 0.80)
# 参与机制指标同时受覆盖率和正确共识质量约束，量级低于候选共识。
# 这些阈值从本版报告起固定使用，避免通过调整阈值追逐单批结果。
MECHANISM_THRESHOLDS = (0.20, 0.40, 0.50)


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
    true_labels: np.ndarray
    probe_indices: np.ndarray
    global_epochs: np.ndarray
    client_ids: np.ndarray
    active_client_mask: np.ndarray
    probe_set_hash: str
    probe_format: str


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


def get_probe_input_format(path: Path, metadata: Dict[str, object]) -> str:
    """根据结构化声明和实际文件选择NPZ或历史CSV输入，不静默回退损坏NPZ。"""
    declared = str(metadata.get("probe_output_format", "")).strip().lower()
    if declared in {"csv", "legacy", "legacy_csv"}:
        return "legacy_csv"
    if declared == "npz":
        return "npz"
    npz_filename = str(metadata.get("probe_npz_file", "probe_probabilities.npz"))
    if (path / npz_filename).is_file():
        return "npz"
    return "legacy_csv"


def get_probe_input_files(path: Path, metadata: Dict[str, object]) -> List[str]:
    """返回当前实验格式必须存在的探针文件名。"""
    if get_probe_input_format(path, metadata) == "npz":
        return [
            str(metadata.get("probe_npz_file", "probe_probabilities.npz")),
            str(metadata.get("probe_summary_file", "probe_epoch_summary.csv")),
        ]
    return list(LEGACY_PROBE_FILES)


def missing_experiment_files(path: Path, metadata: Dict[str, object]) -> List[str]:
    """列出公共输入和所选探针格式缺失的全部文件。"""
    required = COMMON_REQUIRED_RESULT_FILES + get_probe_input_files(path, metadata)
    return [filename for filename in required if not (path / filename).is_file()]


def _legacy_probe_rows_to_batches(
        rows: List[List[Optional[np.ndarray]]]
) -> List[List[Optional[np.ndarray]]]:
    """把旧单图CSV统一扩展为每个模型 [1,K] 的探针批次。"""
    converted = []
    for row in rows:
        converted.append([
            None if vector is None else np.asarray(vector, dtype=np.float64).reshape(1, -1)
            for vector in row
        ])
    return converted


def _read_legacy_probe_labels(path: Path, round_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """读取旧probe_meta真实标签；历史结果缺失时使用-1明确表示未知。"""
    labels = np.full((round_count, 1), -1, dtype=np.int64)
    indices = np.full((round_count, 1), -1, dtype=np.int64)
    metadata_path = path / "probe_meta.csv"
    if not metadata_path.is_file():
        return labels, indices
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != round_count:
        raise ValueError(
            "probe_meta.csv行数{}与实验轮数{}不一致：{}".format(
                len(rows), round_count, metadata_path
            )
        )
    for row_index, row in enumerate(rows):
        if int(row.get("global_epoch", row_index)) != row_index:
            raise ValueError("probe_meta.csv的global_epoch必须从0连续递增。")
        labels[row_index, 0] = int(row["true_label"])
        indices[row_index, 0] = int(row["probe_index"])
    return labels, indices


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    """读取禁止object类型的NPZ标量，并拒绝意外维度。"""
    value = np.asarray(data[key])
    if value.dtype.kind == "O" or value.ndim != 0:
        raise ValueError("NPZ字段{}必须是非object标量。".format(key))
    return value.item()


def read_probe_npz(path: Path, metadata: Dict[str, object]) -> Dict[str, object]:
    """严格读取固定多图NPZ并转换为报告内部统一的逐模型探针批次。"""
    npz_filename = str(metadata.get("probe_npz_file", "probe_probabilities.npz"))
    npz_path = path / npz_filename
    required_keys = {
        "schema_version",
        "client_probabilities",
        "edge_probabilities",
        "cloud_probabilities",
        "active_client_mask",
        "client_ids",
        "probe_indices",
        "true_labels",
        "global_epochs",
        "completed_epochs",
        "probe_set_hash",
        "probe_source",
    }
    try:
        with np.load(str(npz_path), allow_pickle=False) as data:
            missing_keys = sorted(required_keys.difference(data.files))
            if missing_keys:
                raise ValueError("NPZ缺少字段：{}".format(missing_keys))
            # 保留训练端float32，四方案同时加载时可把固定探针内存占用控制在约一半。
            clients = np.asarray(data["client_probabilities"], dtype=np.float32)
            edges = np.asarray(data["edge_probabilities"], dtype=np.float32)
            cloud = np.asarray(data["cloud_probabilities"], dtype=np.float32)
            active_mask = np.asarray(data["active_client_mask"], dtype=np.bool_)
            client_ids = np.asarray(data["client_ids"], dtype=np.int64)
            probe_indices = np.asarray(data["probe_indices"], dtype=np.int64)
            true_labels = np.asarray(data["true_labels"], dtype=np.int64)
            global_epochs = np.asarray(data["global_epochs"], dtype=np.int64)
            completed_epochs = int(_npz_scalar(data, "completed_epochs"))
            probe_set_hash = str(_npz_scalar(data, "probe_set_hash"))
            schema_version = str(_npz_scalar(data, "schema_version"))
            probe_source = str(_npz_scalar(data, "probe_source"))
            edge_active_mask = (
                np.asarray(data["edge_active_mask"], dtype=np.bool_)
                if "edge_active_mask" in data.files else None
            )
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("固定探针NPZ损坏或格式不合法：{}；{}".format(npz_path, exc)) from exc

    if clients.ndim != 4 or edges.ndim != 4 or cloud.ndim != 3:
        raise ValueError("NPZ概率维度必须分别为[E,M,P,K]、[E,G,P,K]和[E,P,K]。")
    if schema_version != "fixed_probe_v1":
        raise ValueError("不支持的固定探针NPZ版本：{}。".format(schema_version))
    if probe_source not in {"test", "train"}:
        raise ValueError("NPZ probe_source只能是test或train。")
    metadata_source = str(metadata.get("probe_source", probe_source))
    if metadata_source != probe_source:
        raise ValueError("NPZ probe_source与topology_metadata.json不一致。")
    epoch_count, candidate_count, probe_count, class_count = clients.shape
    if edges.shape[0] != epoch_count or edges.shape[2:] != (probe_count, class_count):
        raise ValueError("边缘概率与客户端概率的epoch、探针或类别维度不一致。")
    if cloud.shape != (epoch_count, probe_count, class_count):
        raise ValueError("云端概率形状与客户端概率不一致。")
    if completed_epochs != epoch_count or global_epochs.shape != (epoch_count,):
        raise ValueError("completed_epochs、global_epochs与NPZ首维不一致。")
    if not np.array_equal(global_epochs, np.arange(epoch_count, dtype=np.int64)):
        raise ValueError("NPZ global_epochs必须从0严格连续递增。")
    if client_ids.shape != (candidate_count,) or np.unique(client_ids).size != candidate_count:
        raise ValueError("NPZ client_ids形状异常或包含重复编号。")
    if active_mask.shape != (epoch_count, candidate_count):
        raise ValueError("NPZ active_client_mask形状异常。")
    if probe_indices.shape != (probe_count,) or np.unique(probe_indices).size != probe_count:
        raise ValueError("NPZ probe_indices形状异常或索引不唯一。")
    if true_labels.shape != (probe_count,) or np.any(true_labels < 0) or np.any(true_labels >= class_count):
        raise ValueError("NPZ true_labels形状或标签范围异常。")
    expected_per_class = int(metadata.get("probe_samples_per_class", 10))
    label_counts = np.bincount(true_labels, minlength=class_count)
    if not np.array_equal(label_counts, np.full(class_count, expected_per_class, dtype=np.int64)):
        raise ValueError(
            "固定探针必须每类恰好{}张，实际计数为{}。".format(
                expected_per_class, label_counts.tolist()
            )
        )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", probe_set_hash):
        raise ValueError("probe_set_hash必须是64位SHA-256十六进制字符串。")
    metadata_hash = str(metadata.get("probe_set_hash", probe_set_hash))
    if metadata_hash != probe_set_hash:
        raise ValueError("NPZ探针哈希与topology_metadata.json不一致。")
    metadata_probe_count = int(metadata.get("probe_sample_count", probe_count))
    if metadata_probe_count != probe_count:
        raise ValueError("NPZ探针数与topology_metadata.json不一致。")

    if not validate_probability_vector(clients, class_count):
        raise ValueError("NPZ客户端概率包含非法值。")
    if not validate_probability_vector(cloud, class_count):
        raise ValueError("NPZ云端概率包含非法值。")
    inferred_edge_mask = np.all(np.isfinite(edges), axis=(2, 3))
    partial_edge_mask = np.any(np.isfinite(edges), axis=(2, 3)) & ~inferred_edge_mask
    if np.any(partial_edge_mask):
        raise ValueError("NPZ边缘槽位只能整块为空，不能部分包含NaN。")
    if edge_active_mask is not None:
        if edge_active_mask.shape != inferred_edge_mask.shape or not np.array_equal(
                edge_active_mask, inferred_edge_mask
        ):
            raise ValueError("NPZ edge_active_mask与边缘概率有限性不一致。")
    for epoch_index, edge_slot in np.argwhere(inferred_edge_mask):
        if not validate_probability_vector(edges[epoch_index, edge_slot], class_count):
            raise ValueError("NPZ边缘概率包含非法值。")

    client_rows = [
        [clients[epoch_index, model_index] for model_index in range(candidate_count)]
        for epoch_index in range(epoch_count)
    ]
    edge_rows = [
        [
            edges[epoch_index, edge_slot] if inferred_edge_mask[epoch_index, edge_slot] else None
            for edge_slot in range(edges.shape[1])
        ]
        for epoch_index in range(epoch_count)
    ]
    cloud_rows = [[cloud[epoch_index]] for epoch_index in range(epoch_count)]
    return {
        "client_probe": client_rows,
        "edge_probe": edge_rows,
        "cloud_probe": cloud_rows,
        "true_labels": np.tile(true_labels[None, :], (epoch_count, 1)),
        "probe_indices": np.tile(probe_indices[None, :], (epoch_count, 1)),
        "global_epochs": global_epochs,
        "client_ids": client_ids,
        "active_client_mask": active_mask,
        "probe_set_hash": probe_set_hash,
        "probe_format": "npz",
    }


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
        missing = missing_experiment_files(directory, metadata)
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
    metadata = read_json(path / "topology_metadata.json")
    missing = missing_experiment_files(path, metadata)
    if missing:
        raise FileNotFoundError("实验目录缺少文件 {}：{}".format(missing, path))
    scenario = str(metadata.get("scenario", ""))
    if scenario not in SCENARIO_ORDER:
        raise ValueError("不支持的实验场景 {}：{}".format(scenario, path))
    schedule = read_jsonl(path / "topology_schedule.jsonl")
    probe_format = get_probe_input_format(path, metadata)
    if probe_format == "npz":
        probe_data = read_probe_npz(path, metadata)
    else:
        client_probe = _legacy_probe_rows_to_batches(
            read_probability_csv(path / "probe_client_pre.csv")
        )
        edge_probe = _legacy_probe_rows_to_batches(
            read_probability_csv(path / "probe_edge_post.csv")
        )
        cloud_probe = _legacy_probe_rows_to_batches(
            read_probability_csv(path / "probe_cloud_post.csv")
        )
        true_labels, probe_indices = _read_legacy_probe_labels(path, len(schedule))
        if schedule:
            client_ids = np.asarray(
                schedule[0].get("candidate_client_indexes", []), dtype=np.int64
            )
            active_mask = np.zeros((len(schedule), client_ids.shape[0]), dtype=np.bool_)
            for epoch_index, record in enumerate(schedule):
                active_slots = [
                    int(value) for value in record.get("mat_active_candidate_slots", [])
                ]
                active_mask[epoch_index, active_slots] = True
        else:
            client_ids = np.asarray([], dtype=np.int64)
            active_mask = np.empty((0, 0), dtype=np.bool_)
        probe_data = {
            "client_probe": client_probe,
            "edge_probe": edge_probe,
            "cloud_probe": cloud_probe,
            "true_labels": true_labels,
            "probe_indices": probe_indices,
            "global_epochs": np.arange(len(schedule), dtype=np.int64),
            "client_ids": client_ids,
            "active_client_mask": active_mask,
            "probe_set_hash": "",
            "probe_format": "legacy_csv",
        }
    return ExperimentData(
        path=path,
        scenario=scenario,
        label=SCENARIO_LABELS[scenario],
        metadata=metadata,
        schedule=schedule,
        train_acc=read_metric_series(path / "train_acc.txt"),
        train_loss=read_metric_series(path / "train_loss.txt"),
        test_acc=read_metric_series(path / "test_acc.txt"),
        test_loss=read_metric_series(path / "test_loss.txt"),
        client_probe=probe_data["client_probe"],
        edge_probe=probe_data["edge_probe"],
        cloud_probe=probe_data["cloud_probe"],
        true_labels=probe_data["true_labels"],
        probe_indices=probe_data["probe_indices"],
        global_epochs=probe_data["global_epochs"],
        client_ids=probe_data["client_ids"],
        active_client_mask=probe_data["active_client_mask"],
        probe_set_hash=str(probe_data["probe_set_hash"]),
        probe_format=str(probe_data["probe_format"]),
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
                if values.ndim not in {1, 2} or values.shape[-1] < 2:
                    raise ValueError("无法从探针推导有效类别数：{}".format(experiment.path))
                return int(values.shape[-1])
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
        "probe_format": [experiment.probe_format for experiment in experiments],
        "probe_count": [int(experiment.true_labels.shape[1]) for experiment in experiments],
    }
    inconsistent = {
        key: values for key, values in profile_fields.items() if len(set(values)) != 1
    }
    if inconsistent:
        raise ValueError("四方案关键元数据不一致：{}".format(inconsistent))

    if profile_fields["probe_format"][0] == "npz":
        reference = experiments[0]
        for experiment in experiments[1:]:
            if experiment.probe_set_hash != reference.probe_set_hash:
                raise ValueError(
                    "四方案固定探针哈希不一致：{}与{}。".format(
                        reference.label, experiment.label
                    )
                )
            if not np.array_equal(
                    experiment.probe_indices[0], reference.probe_indices[0]
            ):
                raise ValueError("四方案固定探针索引不一致。")
            if not np.array_equal(
                    experiment.true_labels[0], reference.true_labels[0]
            ):
                raise ValueError("四方案固定探针真实标签不一致。")

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
            "probe_set_hash": experiments[0].probe_set_hash,
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
    """逐图计算概率一致性A、确定性C和有效共识S，再对探针取平均。"""

    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 2:
        matrix = matrix[:, None, :]
    if matrix.ndim != 3 or matrix.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")
    metrics = calculate_population_probe_metrics(matrix, true_labels=None)
    return (
        float(metrics["agreement_mean"]),
        float(metrics["certainty_mean"]),
        float(metrics["effective_mean"]),
    )


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


def finite_mean(values: Sequence[float]) -> float:
    """仅对有限值取平均；全部为空时返回NaN且不产生运行时警告。"""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


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
    """检查单个或成批概率向量的类别维、有限性、范围和归一化误差。"""

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != class_count or not np.all(np.isfinite(values)):
        return False
    if np.any(values < -1e-7) or np.any(values > 1.0 + 1e-7):
        return False
    return bool(np.all(np.abs(np.sum(values, axis=-1) - 1.0) <= 5e-6))


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
    probe_alignment_ok = True
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
        probe_alignment_ok = probe_alignment_ok and int(experiment.global_epochs[index]) == index
        probe_alignment_ok = probe_alignment_ok and candidates == [
            int(value) for value in experiment.client_ids
        ]
        expected_active_mask = np.zeros(len(candidates), dtype=np.bool_)
        expected_active_mask[active_slots] = True
        probe_alignment_ok = probe_alignment_ok and np.array_equal(
            experiment.active_client_mask[index], expected_active_mask
        )
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
        "格式{}；每轮{}张；客户端{}列、概率{}维、云端1列；HFL边缘{}槽位，FL边缘为空".format(
            experiment.probe_format, experiment.true_labels.shape[1], candidate_total, class_count, hfl_edge_slot_count
            if experiment.scenario.startswith("hfl_") else 0
        ), "关键",
    )
    add_quality_check(
        checks, experiment.label, "探针epoch、候选顺序和活跃掩码与JSONL一致",
        "通过" if probe_alignment_ok else "未通过",
        "global_epochs、client_ids和active_client_mask逐轮交叉核对", "关键",
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
    """逐模型、逐探针计算到对应云概率的JS散度，再取总体平均。"""

    matrix = np.asarray(probabilities, dtype=np.float64)
    reference_matrix = np.asarray(reference, dtype=np.float64)
    if matrix.ndim == 2:
        matrix = matrix[:, None, :]
    if reference_matrix.ndim == 1:
        reference_matrix = reference_matrix[None, :]
    if matrix.ndim != 3 or matrix.shape[0] == 0 or reference_matrix.shape != matrix.shape[1:]:
        return float("nan")
    # 使用广播一次计算全部“模型-探针”对，数学口径与逐项pairwise JSD完全一致。
    reference_entropy = normalized_entropy(reference_matrix)[None, :]
    model_entropy = normalized_entropy(matrix)
    midpoint_entropy = normalized_entropy(
        0.5 * (matrix + reference_matrix[None, :, :])
    )
    divergences = np.clip(
        midpoint_entropy - 0.5 * (model_entropy + reference_entropy),
        0.0,
        1.0,
    )
    return float(np.mean(divergences))


def mean_consensus_to_reference(
        probabilities: np.ndarray, reference: np.ndarray
) -> Tuple[float, float, float]:
    """逐模型、逐探针计算与对应云概率的A/C/S，并返回算术均值。"""

    matrix = np.asarray(probabilities, dtype=np.float64)
    reference_matrix = np.asarray(reference, dtype=np.float64)
    if matrix.ndim == 2:
        matrix = matrix[:, None, :]
    if reference_matrix.ndim == 1:
        reference_matrix = reference_matrix[None, :]
    if matrix.ndim != 3 or matrix.shape[0] == 0 or reference_matrix.shape != matrix.shape[1:]:
        return float("nan"), float("nan"), float("nan")
    # 将每个模型与对应云概率组成一个二模型探针，再批量复用统一的A/C/S实现。
    model_count, probe_count, class_count = matrix.shape
    expanded_reference = np.broadcast_to(
        reference_matrix[None, :, :], matrix.shape
    )
    paired_probabilities = np.stack(
        [
            matrix.reshape(model_count * probe_count, class_count),
            expanded_reference.reshape(model_count * probe_count, class_count),
        ],
        axis=0,
    )
    metrics = calculate_population_probe_metrics(
        paired_probabilities, true_labels=None
    )
    return (
        float(metrics["agreement_mean"]),
        float(metrics["certainty_mean"]),
        float(metrics["effective_mean"]),
    )


def calculate_participation_margin_metrics(
        candidate_matrix: np.ndarray,
        active_slots: Sequence[int],
        true_labels: Optional[np.ndarray],
) -> Tuple[float, float]:
    """计算参与加权的正边界质量和有符号边界质量。

    对每个活跃客户端和探针计算真实类别概率减去最强错误类别概率，再以
    “候选总数乘探针数”归一化。正边界质量只累计正边界；有符号版本
    同时保留错误预测的负贡献。没有活跃客户端时二者为0，缺少真值时为空值。
    """

    probabilities = np.asarray(candidate_matrix, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] == 0:
        raise ValueError("候选概率必须是非空的[候选, 探针, 类别]三维数组。")
    labels = (
        None
        if true_labels is None
        else np.asarray(true_labels, dtype=np.int64).reshape(-1)
    )
    probe_count = int(probabilities.shape[1])
    class_count = int(probabilities.shape[2])
    if labels is None or labels.shape != (probe_count,) or np.any(labels < 0):
        return float("nan"), float("nan")
    if np.any(labels >= class_count):
        raise ValueError("真实标签超出候选概率的类别范围。")

    active_indexes = np.asarray(list(active_slots), dtype=np.int64)
    if active_indexes.size == 0:
        # 这是“正确证据质量”而非至少需要两人的共识，因此零参与自然贡献0。
        return 0.0, 0.0
    if np.any(active_indexes < 0) or np.any(active_indexes >= probabilities.shape[0]):
        raise ValueError("活跃候选槽位超出候选概率范围。")
    active_probabilities = probabilities[active_indexes]
    true_probabilities = np.take_along_axis(
        active_probabilities,
        labels[None, :, None],
        axis=2,
    )[:, :, 0]
    wrong_probabilities = active_probabilities.copy()
    label_positions = np.broadcast_to(
        labels[None, :, None],
        (active_probabilities.shape[0], probe_count, 1),
    )
    np.put_along_axis(wrong_probabilities, label_positions, -np.inf, axis=2)
    strongest_wrong = np.max(wrong_probabilities, axis=2)
    signed_margin = true_probabilities - strongest_wrong
    denominator = float(probabilities.shape[0] * probe_count)
    positive_margin_mass = float(np.sum(np.maximum(signed_margin, 0.0)) / denominator)
    signed_margin_mass = float(np.sum(signed_margin) / denominator)
    return positive_margin_mass, signed_margin_mass


def build_round_metrics(experiment: ExperimentData, smooth_window: int) -> List[Dict[str, object]]:
    """逐轮还原真实客户端槽位，并计算模型、参与、层级和共识指标。"""

    rows = []
    cumulative_active = 0
    cumulative_mechanism_evidence = 0.0
    cumulative_positive_margin = 0.0
    has_mechanism_evidence = False
    has_positive_margin = False
    for index, record in enumerate(experiment.schedule):
        candidates = [int(value) for value in record["candidate_client_indexes"]]
        active = [int(value) for value in record["active_client_indexes"]]
        candidate_vectors = nonempty_probability_vectors(experiment.client_probe[index])
        cloud_vectors = nonempty_probability_vectors(experiment.cloud_probe[index])
        candidate_matrix = np.stack(candidate_vectors, axis=0)
        cloud_probability = cloud_vectors[0]
        true_labels = experiment.true_labels[index]
        labels_for_metrics = true_labels if np.all(true_labels >= 0) else None
        # JSONL保存的是MAT真实槽位，优先直接使用，避免按连续人数错误还原分组。
        active_slots = [int(value) for value in record["mat_active_candidate_slots"]]
        active_matrix = (
            candidate_matrix[active_slots]
            if active_slots else np.empty((0,) + candidate_matrix.shape[1:])
        )

        candidate_metrics = calculate_population_probe_metrics(
            candidate_matrix, labels_for_metrics
        )
        active_metrics = calculate_population_probe_metrics(
            active_matrix, labels_for_metrics
        )
        candidate_a = float(candidate_metrics["agreement_mean"])
        candidate_c = float(candidate_metrics["certainty_mean"])
        candidate_s = float(candidate_metrics["effective_mean"])
        active_a = float(active_metrics["agreement_mean"])
        active_c = float(active_metrics["certainty_mean"])
        active_s = float(active_metrics["effective_mean"])

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
            group_metrics = calculate_population_probe_metrics(
                group_matrix, labels_for_metrics
            )
            group_a = float(group_metrics["agreement_mean"])
            group_c = float(group_metrics["certainty_mean"])
            group_s = float(group_metrics["effective_mean"])
            group_agreement_values.append((len(group_slots), group_a))
            group_certainty_values.append((len(group_slots), group_c))
            group_effective_values.append((len(group_slots), group_s))

        edge_vectors = nonempty_probability_vectors(experiment.edge_probe[index])
        edge_matrix = (
            np.stack(edge_vectors, axis=0)
            if edge_vectors else np.empty((0,) + candidate_matrix.shape[1:])
        )
        edge_metrics = calculate_population_probe_metrics(
            edge_matrix, labels_for_metrics
        )
        edge_a = float(edge_metrics["agreement_mean"])
        edge_c = float(edge_metrics["certainty_mean"])
        edge_s = float(edge_metrics["effective_mean"])
        edge_cloud_a, edge_cloud_c, edge_cloud_s = mean_consensus_to_reference(
            edge_matrix, cloud_probability
        )
        cloud_certainty = 1.0 - float(np.mean(normalized_entropy(cloud_probability)))
        if labels_for_metrics is None:
            cloud_probe_accuracy = float("nan")
            cloud_true_probability = float("nan")
        else:
            cloud_probe_accuracy = float(
                np.mean(np.argmax(cloud_probability, axis=1) == true_labels)
            )
            cloud_true_probability = float(np.mean(
                cloud_probability[np.arange(true_labels.shape[0]), true_labels]
            ))
        active_coverage_ratio = float(len(active_slots)) / float(len(candidates))
        if np.isfinite(active_metrics["correct_effective_mean"]):
            coverage_weighted_active_correct = (
                active_coverage_ratio
                * float(active_metrics["correct_effective_mean"])
            )
        else:
            coverage_weighted_active_correct = float("nan")
        positive_margin_mass, signed_margin_mass = (
            calculate_participation_margin_metrics(
                candidate_matrix, active_slots, labels_for_metrics
            )
        )

        cumulative_active += len(active)
        # 共识不足两人或缺少真值时Q不可用；边界质量只在缺少真值时不可用。
        # 累计列在首次出现有效值前保持空值，避免把旧CSV的缺失证据误写成零。
        if np.isfinite(coverage_weighted_active_correct):
            cumulative_mechanism_evidence += coverage_weighted_active_correct
            has_mechanism_evidence = True
        if np.isfinite(positive_margin_mass):
            cumulative_positive_margin += positive_margin_mass
            has_positive_margin = True
        row = {
            "scenario": experiment.scenario,
            "label": experiment.label,
            "epoch": index + 1,
            "probe_count": int(candidate_matrix.shape[1]),
            "candidate_count": len(candidates),
            "active_count": len(active),
            "active_coverage_ratio": active_coverage_ratio,
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
            "candidate_correct_effective": float(candidate_metrics["correct_effective_mean"]),
            "candidate_wrong_effective": float(candidate_metrics["wrong_effective_mean"]),
            "candidate_effective_q25": float(candidate_metrics["effective_q25"]),
            "candidate_effective_q50": float(candidate_metrics["effective_q50"]),
            "candidate_effective_q75": float(candidate_metrics["effective_q75"]),
            "active_agreement": active_a,
            "active_certainty": active_c,
            "active_effective": active_s,
            "active_correct_effective": float(active_metrics["correct_effective_mean"]),
            "active_wrong_effective": float(active_metrics["wrong_effective_mean"]),
            "active_effective_q25": float(active_metrics["effective_q25"]),
            "active_effective_q50": float(active_metrics["effective_q50"]),
            "active_effective_q75": float(active_metrics["effective_q75"]),
            "coverage_weighted_active_correct_effective": coverage_weighted_active_correct,
            "cumulative_coverage_weighted_active_correct_effective": (
                cumulative_mechanism_evidence
                if has_mechanism_evidence else float("nan")
            ),
            "participation_weighted_positive_margin": positive_margin_mass,
            "participation_weighted_signed_margin": signed_margin_mass,
            "cumulative_participation_weighted_positive_margin": (
                cumulative_positive_margin
                if has_positive_margin else float("nan")
            ),
            "within_group_agreement": weighted_group_metric(group_agreement_values),
            "within_group_certainty": weighted_group_metric(group_certainty_values),
            "within_group_effective": weighted_group_metric(group_effective_values),
            "edge_agreement": edge_a,
            "edge_certainty": edge_c,
            "edge_effective": edge_s,
            "edge_correct_effective": float(edge_metrics["correct_effective_mean"]),
            "edge_wrong_effective": float(edge_metrics["wrong_effective_mean"]),
            "cloud_certainty": cloud_certainty,
            "cloud_probe_accuracy": cloud_probe_accuracy,
            "cloud_true_class_probability": cloud_true_probability,
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
        "candidate_correct_effective",
        "candidate_wrong_effective",
        "active_agreement",
        "active_certainty",
        "active_effective",
        "active_correct_effective",
        "active_wrong_effective",
        "active_coverage_ratio",
        "coverage_weighted_active_correct_effective",
        "participation_weighted_positive_margin",
        "participation_weighted_signed_margin",
        "within_group_agreement",
        "within_group_certainty",
        "within_group_effective",
        "edge_agreement",
        "edge_certainty",
        "edge_effective",
        "edge_correct_effective",
        "edge_wrong_effective",
        "cloud_certainty",
        "cloud_probe_accuracy",
        "cloud_true_class_probability",
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
            "active_correct_effective",
            "active_wrong_effective",
            "coverage_weighted_active_correct_effective",
            "participation_weighted_positive_margin",
            "participation_weighted_signed_margin",
            "within_group_agreement",
            "within_group_certainty",
            "within_group_effective",
            "edge_agreement",
            "edge_certainty",
            "edge_effective",
            "edge_correct_effective",
            "edge_wrong_effective",
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
    correct_attainment = historical_best([
        float(row["candidate_correct_effective_ma"]) for row in rows
    ])
    for index, value in enumerate(correct_attainment):
        rows[index]["candidate_correct_effective_historical_best"] = float(value)
    return rows


def validate_npz_summary(
        experiment: ExperimentData, round_rows: Sequence[Dict[str, object]]
) -> None:
    """用NPZ重算结果核对训练端摘要CSV，防止把派生文件直接当作可信来源。"""
    if experiment.probe_format != "npz":
        return
    summary_filename = str(
        experiment.metadata.get("probe_summary_file", "probe_epoch_summary.csv")
    )
    summary_path = experiment.path / summary_filename
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != len(round_rows):
        raise ValueError(
            "{}摘要CSV为{}行，NPZ重算为{}行。".format(
                experiment.label, len(summary_rows), len(round_rows)
            )
        )
    field_mapping = {
        "candidate_agreement_mean": "candidate_agreement",
        "candidate_certainty_mean": "candidate_certainty",
        "candidate_effective_mean": "candidate_effective",
        "candidate_correct_effective_mean": "candidate_correct_effective",
        "candidate_wrong_effective_mean": "candidate_wrong_effective",
        "candidate_effective_q25": "candidate_effective_q25",
        "candidate_effective_q50": "candidate_effective_q50",
        "candidate_effective_q75": "candidate_effective_q75",
        "active_coverage": "active_coverage_ratio",
        "active_effective_mean": "active_effective",
        "active_correct_effective_mean": "active_correct_effective",
        "active_wrong_effective_mean": "active_wrong_effective",
        "active_effective_q25": "active_effective_q25",
        "active_effective_q50": "active_effective_q50",
        "active_effective_q75": "active_effective_q75",
        "coverage_weighted_active_correct_effective": "coverage_weighted_active_correct_effective",
        "edge_effective_mean": "edge_effective",
        "edge_correct_effective_mean": "edge_correct_effective",
        "cloud_probe_accuracy": "cloud_probe_accuracy",
        "cloud_true_class_probability_mean": "cloud_true_class_probability",
    }
    for row_index, (saved, recomputed) in enumerate(
            zip(summary_rows, round_rows)
    ):
        if int(saved["global_epoch"]) != row_index:
            raise ValueError("探针摘要global_epoch未从0连续递增。")
        if int(saved["active_count"]) != int(recomputed["active_count"]):
            raise ValueError("探针摘要第{}轮活跃人数与NPZ重算不一致。".format(row_index))
        if int(saved["probe_count"]) != int(recomputed["probe_count"]):
            raise ValueError("探针摘要第{}轮探针数与NPZ重算不一致。".format(row_index))
        if int(saved["candidate_count"]) != int(recomputed["candidate_count"]):
            raise ValueError("探针摘要第{}轮候选数与NPZ重算不一致。".format(row_index))
        for saved_field, computed_field in field_mapping.items():
            text = str(saved.get(saved_field, "")).strip()
            computed_value = float(recomputed[computed_field])
            if not text:
                if np.isfinite(computed_value):
                    raise ValueError(
                        "探针摘要第{}轮字段{}为空，但NPZ重算为有限值。".format(
                            row_index, saved_field
                        )
                    )
                continue
            saved_value = float(text)
            if not np.isfinite(computed_value) or not math.isclose(
                    saved_value, computed_value, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    "探针摘要第{}轮字段{}与NPZ重算不一致：{} vs {}。".format(
                        row_index, saved_field, saved_value, computed_value
                    )
                )
        # 每图拆分由共享公式保证，这里再校验epoch均值恒等式作为输出护栏。
        if not math.isclose(
                float(recomputed["candidate_correct_effective"])
                + float(recomputed["candidate_wrong_effective"]),
                float(recomputed["candidate_effective"]),
                rel_tol=0.0,
                abs_tol=1e-10,
        ):
            raise ValueError("正确与错误有效共识之和不等于纯有效共识。")


def first_stable_epoch(values: Sequence[float], threshold: float, window: int = 5) -> Optional[int]:
    """返回5轮尾随均值达到阈值且此后不再跌破阈值的首个轮次。"""

    smoothed = trailing_mean(values, window, require_full_values=True)
    for index in range(window - 1, smoothed.size):
        remaining = smoothed[index:]
        if np.all(np.isfinite(remaining)) and np.all(remaining >= threshold):
            return int(index + 1)
    return None


def normalized_curve_area(values: Sequence[float], limit: int) -> float:
    """计算前limit轮曲线的单位epoch归一化梯形面积，空序列返回NaN。"""
    array = np.asarray(values, dtype=np.float64)[:int(limit)]
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan")
    if array.size == 1:
        return float(array[0])
    area = np.sum((array[:-1] + array[1:]) * 0.5)
    return float(area / float(array.size - 1))


def mechanism_speed_curve(values: Sequence[float]) -> np.ndarray:
    """把偶发无有效活跃共识的轮次记为零证据，同时保留整列缺失语义。"""

    array = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(array)):
        return array.copy()
    # 新NPZ有真值但活跃不足两人时，原始共识为空；速度/累计口径解释为本轮零证据。
    return np.where(np.isfinite(array), array, 0.0)


def first_stable_smoothed_epoch(
        smoothed_values: Sequence[float], threshold: float, minimum_tail: int = 5
) -> Optional[int]:
    """返回MA曲线首次越线且之后不跌破，并至少保留minimum_tail个观察点的epoch。"""
    values = np.asarray(smoothed_values, dtype=np.float64)
    for index, value in enumerate(values):
        remaining = values[index:]
        if remaining.size < int(minimum_tail) or not np.isfinite(value):
            continue
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
    candidate_correct_s = np.asarray([
        float(row["candidate_correct_effective"]) for row in round_rows
    ])
    candidate_wrong_s = np.asarray([
        float(row["candidate_wrong_effective"]) for row in round_rows
    ])
    active_s = np.asarray([float(row["active_effective"]) for row in round_rows])
    active_correct_s = np.asarray([
        float(row["active_correct_effective"]) for row in round_rows
    ])
    coverage_weighted_correct = np.asarray([
        float(row["coverage_weighted_active_correct_effective"])
        for row in round_rows
    ])
    positive_margin = np.asarray([
        float(row["participation_weighted_positive_margin"])
        for row in round_rows
    ])
    signed_margin = np.asarray([
        float(row["participation_weighted_signed_margin"])
        for row in round_rows
    ])
    mechanism_speed = mechanism_speed_curve(coverage_weighted_correct)
    candidate_s_ma = np.asarray([float(row["candidate_effective_ma"]) for row in round_rows])
    # 达成速度固定使用MA10，避免用户只调整绘图窗口就改变论文主口径。
    candidate_s_ma10 = trailing_mean(candidate_s, 10, require_full_values=True)
    candidate_correct_s_ma10 = trailing_mean(
        candidate_correct_s, 10, require_full_values=True
    )
    mechanism_speed_ma10 = trailing_mean(
        mechanism_speed, 10, require_full_values=True
    )
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
        "correct_effective_last10": finite_mean(candidate_correct_s[-10:]),
        "correct_effective_last20": finite_mean(candidate_correct_s[-20:]),
        "wrong_effective_last10": finite_mean(candidate_wrong_s[-10:]),
        "wrong_effective_last20": finite_mean(candidate_wrong_s[-20:]),
        "active_effective_last10": finite_mean(active_s[-10:]),
        "active_effective_last20": finite_mean(active_s[-20:]),
        "active_correct_effective_last20": finite_mean(active_correct_s[-20:]),
        "coverage_weighted_active_correct_last20": finite_mean(
            coverage_weighted_correct[-20:]
        ),
        "coverage_weighted_active_correct_total": (
            float(np.sum(mechanism_speed))
            if np.any(np.isfinite(mechanism_speed)) else float("nan")
        ),
        "coverage_weighted_active_correct_auc50": normalized_curve_area(
            mechanism_speed, 50
        ),
        "coverage_weighted_active_correct_auc100": normalized_curve_area(
            mechanism_speed, 100
        ),
        "participation_positive_margin_last20": finite_mean(
            positive_margin[-20:]
        ),
        "participation_signed_margin_last20": finite_mean(
            signed_margin[-20:]
        ),
        "participation_positive_margin_total": (
            float(np.nansum(positive_margin))
            if np.any(np.isfinite(positive_margin)) else float("nan")
        ),
        "participation_positive_margin_auc50": normalized_curve_area(
            positive_margin, 50
        ),
        "participation_positive_margin_auc100": normalized_curve_area(
            positive_margin, 100
        ),
        "effective_auc50": normalized_curve_area(candidate_s, 50),
        "effective_auc100": normalized_curve_area(candidate_s, 100),
        "correct_effective_auc50": normalized_curve_area(candidate_correct_s, 50),
        "correct_effective_auc100": normalized_curve_area(candidate_correct_s, 100),
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
    for threshold in CONSENSUS_THRESHOLDS:
        result["effective_ma10_stable_epoch_{:.2f}".format(threshold)] = (
            first_stable_smoothed_epoch(candidate_s_ma10, threshold)
        )
        result["correct_effective_ma10_stable_epoch_{:.2f}".format(threshold)] = (
            first_stable_smoothed_epoch(candidate_correct_s_ma10, threshold)
        )
    for threshold in MECHANISM_THRESHOLDS:
        result["mechanism_ma10_stable_epoch_{:.2f}".format(threshold)] = (
            first_stable_smoothed_epoch(mechanism_speed_ma10, threshold)
        )
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
    """比较参与机制共识吞吐量、累计证据、相对差值和正确边界质量。"""

    path = figure_dir / "08_参与加权正确共识与边界质量.png"
    figure, axes = plt.subplots(2, 2, figsize=(16, 10.2))
    mechanism_available = all(
        np.any(np.isfinite([
            float(row["coverage_weighted_active_correct_effective"])
            for row in rows_by_scenario[scenario]
        ]))
        for scenario in SCENARIO_ORDER
    )
    raw_field = (
        "coverage_weighted_active_correct_effective"
        if mechanism_available else "candidate_effective"
    )
    smooth_field = (
        "coverage_weighted_active_correct_effective_ma"
        if mechanism_available else "candidate_effective_ma"
    )

    cumulative_curves = {}
    smoothed_curves = {}
    for scenario in SCENARIO_ORDER:
        rows = rows_by_scenario[scenario]
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        raw = np.asarray([float(row[raw_field]) for row in rows], dtype=np.float64)
        smooth = np.asarray([float(row[smooth_field]) for row in rows], dtype=np.float64)
        if mechanism_available:
            cumulative = np.asarray([
                float(row["cumulative_coverage_weighted_active_correct_effective"])
                for row in rows
            ])
        else:
            cumulative = np.cumsum(np.where(np.isfinite(raw), raw, 0.0))
        cumulative_curves[scenario] = cumulative
        smoothed_curves[scenario] = smooth
        color = SCENARIO_COLORS[scenario]
        linestyle = SCENARIO_LINESTYLES[scenario]
        axes[0, 0].plot(
            epochs, raw, color=color, linestyle=linestyle, linewidth=0.7, alpha=0.14
        )
        axes[0, 0].plot(
            epochs, smooth, color=color, linestyle=linestyle, linewidth=2.2,
            label=SCENARIO_LABELS[scenario],
        )
        axes[0, 1].plot(
            epochs, cumulative, color=color, linestyle=linestyle, linewidth=2.2,
            label=SCENARIO_LABELS[scenario],
        )
    style_axis(
        axes[0, 0],
        "覆盖加权活跃正确共识" if mechanism_available else "候选有效共识 S",
        (0, 1.02),
    )
    axes[0, 0].set_title(
        "逐轮值与{}轮尾随均值".format(smooth_window),
        loc="left", fontweight="bold",
    )
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=2)
    style_axis(
        axes[0, 1],
        "累计机制共识证据" if mechanism_available else "累计候选有效共识",
    )
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_title("累计量：越早越陡、总量越大", loc="left", fontweight="bold")

    reference = smoothed_curves["hfl_snf_fixed"]
    epochs = np.asarray([
        int(row["epoch"]) for row in rows_by_scenario["hfl_snf_fixed"]
    ])
    for scenario in SCENARIO_ORDER[1:]:
        difference = reference - smoothed_curves[scenario]
        axes[1, 0].plot(
            epochs,
            difference,
            color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_LINESTYLES[scenario],
            linewidth=2.0,
            label="HFL-SnF − {}".format(SCENARIO_LABELS[scenario]),
        )
    axes[1, 0].axhline(0.0, color="#344054", linewidth=1.0)
    style_axis(axes[1, 0], "相对差值")
    axes[1, 0].set_title("相对基线差值，零线以上表示HFL-SnF更高", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8.5)

    positions = np.arange(len(SCENARIO_ORDER))
    if mechanism_available:
        positive_margin = np.asarray([
            float(summaries[scenario]["participation_positive_margin_last20"])
            for scenario in SCENARIO_ORDER
        ])
        signed_margin = np.asarray([
            float(summaries[scenario]["participation_signed_margin_last20"])
            for scenario in SCENARIO_ORDER
        ])
        width = 0.34
        axes[1, 1].bar(
            positions - width / 2.0,
            positive_margin,
            width=width,
            color="#60A5FA",
            edgecolor="#344054",
            linewidth=0.6,
            label="正边界质量",
        )
        axes[1, 1].bar(
            positions + width / 2.0,
            signed_margin,
            width=width,
            color="#F59E0B",
            edgecolor="#344054",
            linewidth=0.6,
            label="有符号边界质量",
        )
        axes[1, 1].set_ylabel("后20轮均值")
        axes[1, 1].set_title("真实类别相对最强错误类别的概率边界", loc="left", fontweight="bold")
        axes[1, 1].legend(frameon=False, fontsize=8.5)
    else:
        last20_means = np.asarray([
            float(summaries[scenario]["effective_last20"])
            for scenario in SCENARIO_ORDER
        ])
        axes[1, 1].bar(
            positions,
            last20_means,
            width=0.68,
            color=[SCENARIO_COLORS[scenario] for scenario in SCENARIO_ORDER],
            edgecolor="#344054",
            linewidth=0.7,
        )
        axes[1, 1].set_ylabel("候选有效共识 S")
        axes[1, 1].set_title("历史CSV缺少真值时的候选S后20轮均值", loc="left", fontweight="bold")
    axes[1, 1].axhline(0.0, color="#344054", linewidth=0.8)
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(
        [SCENARIO_LABELS[scenario] for scenario in SCENARIO_ORDER],
        rotation=12,
    )
    axes[1, 1].set_xlabel("方案")
    axes[1, 1].grid(True, axis="y", color="#EAECF0", linewidth=0.8, zorder=0)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    candidate_total = int(rows_by_scenario[SCENARIO_ORDER[0]][0]["candidate_count"])
    subtitle = (
        "主曲线=活跃覆盖率×活跃正确有效共识；边界质量使用完整概率向量，未活跃候选贡献0"
        if mechanism_available else
        "历史CSV缺少完整真值，保留候选有效共识兼容展示"
    )
    add_figure_header(
        figure,
        "固定{}候选下的参与机制共识吞吐量".format(candidate_total),
        subtitle,
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
                "后20轮正确有效共识S": item["correct_effective_last20"],
                "后20轮错误有效共识S": item["wrong_effective_last20"],
                "后10轮活跃客户端有效共识S": item["active_effective_last10"],
                "后20轮活跃客户端有效共识S": item["active_effective_last20"],
                "后20轮活跃客户端正确有效共识S": item["active_correct_effective_last20"],
                "后20轮覆盖加权活跃正确有效共识S": item["coverage_weighted_active_correct_last20"],
                "累计覆盖加权活跃正确有效共识S": item[
                    "coverage_weighted_active_correct_total"
                ],
                "覆盖加权活跃正确有效共识前50轮归一化面积": item[
                    "coverage_weighted_active_correct_auc50"
                ],
                "覆盖加权活跃正确有效共识前100轮归一化面积": item[
                    "coverage_weighted_active_correct_auc100"
                ],
                "后20轮参与加权正确边界质量": item[
                    "participation_positive_margin_last20"
                ],
                "后20轮参与加权有符号边界质量": item[
                    "participation_signed_margin_last20"
                ],
                "累计参与加权正确边界质量": item[
                    "participation_positive_margin_total"
                ],
                "参与加权正确边界质量前50轮归一化面积": item[
                    "participation_positive_margin_auc50"
                ],
                "参与加权正确边界质量前100轮归一化面积": item[
                    "participation_positive_margin_auc100"
                ],
                "纯有效共识前50轮归一化面积": item["effective_auc50"],
                "纯有效共识前100轮归一化面积": item["effective_auc100"],
                "正确有效共识前50轮归一化面积": item["correct_effective_auc50"],
                "正确有效共识前100轮归一化面积": item["correct_effective_auc100"],
                "纯有效共识MA10稳定达到0.60轮次": item["effective_ma10_stable_epoch_0.60"],
                "纯有效共识MA10稳定达到0.70轮次": item["effective_ma10_stable_epoch_0.70"],
                "纯有效共识MA10稳定达到0.80轮次": item["effective_ma10_stable_epoch_0.80"],
                "正确有效共识MA10稳定达到0.60轮次": item["correct_effective_ma10_stable_epoch_0.60"],
                "正确有效共识MA10稳定达到0.70轮次": item["correct_effective_ma10_stable_epoch_0.70"],
                "正确有效共识MA10稳定达到0.80轮次": item["correct_effective_ma10_stable_epoch_0.80"],
                "机制共识MA10稳定达到0.20轮次": item[
                    "mechanism_ma10_stable_epoch_0.20"
                ],
                "机制共识MA10稳定达到0.40轮次": item[
                    "mechanism_ma10_stable_epoch_0.40"
                ],
                "机制共识MA10稳定达到0.50轮次": item[
                    "mechanism_ma10_stable_epoch_0.50"
                ],
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
        ("probe_count", "固定探针图片数"),
        ("candidate_count", "候选客户端数"),
        ("active_count", "最终参与客户端数"),
        ("active_coverage_ratio", "活跃候选覆盖率"),
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
        ("candidate_correct_effective", "候选正确有效共识S"),
        ("candidate_wrong_effective", "候选错误有效共识S"),
        ("candidate_effective_q25", "候选有效共识S_P25"),
        ("candidate_effective_q50", "候选有效共识S_P50"),
        ("candidate_effective_q75", "候选有效共识S_P75"),
        ("candidate_effective_ma", "候选有效共识尾随均值"),
        ("candidate_correct_effective_ma", "候选正确有效共识尾随均值"),
        ("candidate_wrong_effective_ma", "候选错误有效共识尾随均值"),
        ("candidate_effective_historical_best", "历史最佳平滑共识"),
        ("candidate_correct_effective_historical_best", "正确有效共识历史最佳"),
        ("active_effective_historical_best", "参与者历史最佳平滑共识"),
        ("active_agreement", "参与者一致性A"),
        ("active_certainty", "参与者确定性C"),
        ("active_effective", "参与者有效共识S"),
        ("active_correct_effective", "参与者正确有效共识S"),
        ("active_wrong_effective", "参与者错误有效共识S"),
        ("coverage_weighted_active_correct_effective", "覆盖率加权参与者正确有效共识S"),
        (
            "cumulative_coverage_weighted_active_correct_effective",
            "累计覆盖率加权参与者正确有效共识S",
        ),
        (
            "participation_weighted_positive_margin",
            "参与加权正确边界质量",
        ),
        (
            "participation_weighted_signed_margin",
            "参与加权有符号边界质量",
        ),
        (
            "cumulative_participation_weighted_positive_margin",
            "累计参与加权正确边界质量",
        ),
        (
            "coverage_weighted_active_correct_effective_ma",
            "覆盖率加权参与者正确有效共识尾随均值",
        ),
        (
            "participation_weighted_positive_margin_ma",
            "参与加权正确边界质量尾随均值",
        ),
        (
            "participation_weighted_signed_margin_ma",
            "参与加权有符号边界质量尾随均值",
        ),
        ("within_group_agreement", "组内一致性A_人数加权"),
        ("within_group_certainty", "组内确定性C_人数加权"),
        ("within_group_effective", "组内有效共识S_人数加权"),
        ("edge_agreement", "边缘间一致性A"),
        ("edge_certainty", "边缘模型确定性C"),
        ("edge_effective", "边缘间有效共识S"),
        ("edge_correct_effective", "边缘正确有效共识S"),
        ("edge_wrong_effective", "边缘错误有效共识S"),
        ("cloud_certainty", "云模型确定性"),
        ("cloud_probe_accuracy", "固定探针云端准确率"),
        ("cloud_true_class_probability", "云端真实类别平均概率"),
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
    candidate_consensus_rows = []
    mechanism_rows = []
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
        candidate_consensus_rows.append([
            item["label"], format_number(item["effective_last20"], 4),
            format_number(item["correct_effective_last20"], 4),
            format_number(item["wrong_effective_last20"], 4),
            format_number(item["active_correct_effective_last20"], 4),
            "{}/{}".format(
                format_number(item["effective_auc50"], 4),
                format_number(item["effective_auc100"], 4),
            ),
            "{}/{}".format(
                format_number(item["correct_effective_auc50"], 4),
                format_number(item["correct_effective_auc100"], 4),
            ),
            item["correct_effective_ma10_stable_epoch_0.60"] or "未稳定达到",
        ])
        mechanism_rows.append([
            item["label"],
            format_number(item["coverage_weighted_active_correct_last20"], 4),
            format_number(item["coverage_weighted_active_correct_total"], 4),
            "{}/{}".format(
                format_number(item["coverage_weighted_active_correct_auc50"], 4),
                format_number(item["coverage_weighted_active_correct_auc100"], 4),
            ),
            "{}/{}/{}".format(
                item["mechanism_ma10_stable_epoch_0.20"] or "未达到",
                item["mechanism_ma10_stable_epoch_0.40"] or "未达到",
                item["mechanism_ma10_stable_epoch_0.50"] or "未达到",
            ),
            format_number(item["participation_positive_margin_last20"], 4),
            format_number(item["participation_signed_margin_last20"], 4),
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
    finite_mechanism_scenarios = [
        scenario for scenario in SCENARIO_ORDER
        if np.isfinite(float(
            summaries[scenario]["coverage_weighted_active_correct_last20"]
        ))
    ]
    mechanism_ranking = " > ".join(
        "{}（{:.4f}）".format(
            summaries[scenario]["label"],
            summaries[scenario]["coverage_weighted_active_correct_last20"],
        )
        for scenario in sorted(
            finite_mechanism_scenarios,
            key=lambda key: summaries[key]["coverage_weighted_active_correct_last20"],
            reverse=True,
        )
    )
    if not mechanism_ranking:
        mechanism_ranking = "当前格式无法计算"
    fixed_probe_mode = str(profile.get("probe_format")) == "npz"
    if fixed_probe_mode:
        probe_evidence_limit = (
            "固定探针已保存真实标签并通过四方案哈希核对；当前主要限制是只有单随机种子，"
            "且没有可靠耗时和硬件日志"
        )
        probe_storage_statement = (
            "探针使用固定、类别均衡的{probe_count}张图片，每类10张；客户端、边缘和云概率"
            "分别来自压缩NPZ，未启用边缘槽位为整块NaN。四方案探针SHA-256为"
            "`{probe_hash}`。每张图片先独立计算A、C、S，再在epoch内取平均。"
        ).format(
            probe_count=profile["probe_count"], probe_hash=profile["probe_set_hash"]
        )
        probe_longitudinal_statement = (
            "全部epoch使用同一批固定图片与顺序，因此MA10可解释为相同探针集合上的纵向趋势。"
            "历史最佳仍只作辅助展示；候选收敛速度读取候选S的前50/100轮面积，"
            "参与机制速度则读取Q的前50/100轮面积、累计量及MA10稳定越过0.20/0.40/0.50的轮次。"
        )
        correctness_statement = (
            "真实标签允许把纯有效共识拆成正确和错误两部分，并同时报告固定探针云端准确率；"
            "这仍不能替代完整测试集test_acc。"
        )
        probe_limitation_bullet = (
            "- 固定探针只覆盖测试集中的100张类别均衡图片，用于机制监控；最终性能仍以完整测试集 `test_acc` 为准。"
        )
        probe_next_step = (
            "3. 在至少5个匹配随机种子上重复当前固定探针方案，并报告均值、标准差或置信区间。"
        )
        mechanism_evidence_statement = (
            "**参与机制主指标中，HFL-SnF后20轮覆盖加权活跃正确共识为"
            "{last20}，{rounds_total}轮累计为{total}，前100轮归一化面积为{auc100}。**"
            " 该口径把参与覆盖率和正确共识质量同时纳入。"
        ).format(
            last20=format_number(
                summaries["hfl_snf_fixed"][
                    "coverage_weighted_active_correct_last20"
                ],
                4,
            ),
            total=format_number(
                summaries["hfl_snf_fixed"][
                    "coverage_weighted_active_correct_total"
                ],
                4,
            ),
            auc100=format_number(
                summaries["hfl_snf_fixed"][
                    "coverage_weighted_active_correct_auc100"
                ],
                4,
            ),
            rounds_total=summaries["hfl_snf_fixed"]["rounds"],
        )
        mechanism_definition_statement = (
            "本报告把“活跃覆盖率乘活跃正确有效共识”记为参与机制主指标Q。"
            "Q衡量固定候选池中每轮实际产生的正确共识证据，而不是只看候选模型是否相似。"
            "同时计算参与加权正确边界质量：对每个活跃客户端和探针取"
            "“真实类别概率减最强错误类别概率”的正部，并除以候选总数与探针数；"
            "有符号版本保留错误预测的负贡献作为护栏。"
        )
    else:
        probe_evidence_limit = (
            "历史探针每轮更换样本且部分批次没有真值标签；同时只有单随机种子，"
            "也没有可靠耗时和硬件日志"
        )
        probe_storage_statement = (
            "本批使用历史单图CSV：每轮只有1张探针图片，边缘空单元格表示未启用槽位。"
            "该格式不具备固定多图哈希。"
        )
        probe_longitudinal_statement = (
            "每轮探针使用不同测试样本，所以曲线同时包含模型学习进展和样本难度变化，"
            "不能当成同一样本上的纯收敛轨迹。历史最佳只作辅助展示。"
        )
        correctness_statement = (
            "历史结果缺少完整真实标签时，无法区分正确共识和集体错误。"
        )
        probe_limitation_bullet = (
            "- 历史单图探针每轮样本不同且可能没有真实标签，不能可靠比较固定样本的纵向正确/错误共识。"
        )
        probe_next_step = (
            "3. 使用当前代码重新运行固定100图探针实验，获得正确/错误共识和跨方案探针哈希。"
        )
        mechanism_evidence_statement = (
            "**历史CSV缺少稳定的固定多图真值，无法可靠计算参与机制主指标和正确边界质量。**"
        )
        mechanism_definition_statement = (
            "历史格式继续保留候选纯S与活跃S；参与机制速度和概率边界相关字段标为空值，"
            "不会把缺失真值解释成零表现。"
        )
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
- {mechanism_evidence_statement}
- **证据评级：可分享但附带限制。** 结果完整且关键运行不变量通过检查；{probe_evidence_limit}。

## 1. 数据完整性与运行语义

四个实验的调度、训练/测试准确率、训练/测试损失和三层概率探针均为 **{rounds}轮**。客户端探针逐轮固定{candidate_total}列；HFL边缘探针固定{hfl_edge_slot_count}个物理槽位，FL边缘探针固定{fl_edge_slot_count}个槽位；云探针固定1列。所有非空概率向量均为{class_count}维、数值有限且概率和接近1。

{probe_storage_statement}

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

有效共识分成概率一致性A、各客户端预测确定性C和二者乘积S。只看A会把“所有模型共同接近均匀分布”误判成高共识。候选纯/正确S继续作为收敛护栏，错误有效共识继续用于识别集体错误，但这些候选口径不再单独承担解释参与机制的任务。

{candidate_consensus_table}

全部{candidate_total}个候选的共识和MAT活跃客户端的共识必须分开解释。noSnF场景每轮只有少量候选训练，其余候选在上轮全量下发后保留相同云模型，容易形成较高的多数一致；活跃客户端指标更能反映当轮真正贡献聚合的模型差异。

![候选与活跃客户端有效共识](figures/04_有效共识分解.png)

按候选有效共识S的最后20轮均值排序为：**{candidate_consensus_ranking}**。当前排序第一的是{candidate_top_label}；候选S会受到轮内未训练候选保留旧云模型的影响，因此该排序不能直接解释成真实参与客户端之间形成了更强共识，应与同表中的活跃客户端S一起阅读。

### 4.1 参与机制共识吞吐量

{mechanism_definition_statement}

{mechanism_table}

当前覆盖加权活跃正确共识的后20轮排序为：**{mechanism_ranking}**。前50/100轮面积衡量单位epoch内累积正确共识的速度；累计量衡量整段训练产生的机制证据总量；MA10越线轮次越早，表示越早进入稳定水平。该指标有意保留参与人数差异，适合检验“每轮聚合更多客户端是否更快形成正确共识”，但不能替代相同参与预算下的效率比较。

![参与加权正确共识与边界质量](figures/08_参与加权正确共识与边界质量.png)

![当前平滑共识与历史最佳](figures/05_平滑共识与历史最佳.png)

历史最佳曲线天然单调不降，只表示曾经达到过的水平；判断退化必须读取当前平滑共识。{probe_longitudinal_statement}

## 5. HFL层级传播

![HFL层级共识传播](figures/06_HFL层级共识传播.png)

层级指标直接使用JSONL的MAT槽位到边缘组映射，组内指标按实际组人数加权；边缘空槽位不会被压缩后错误对应到其他组。{correctness_statement}

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
{probe_limitation_bullet}
- 结果没有阶段耗时、GPU利用率、驱动和PyTorch CUDA信息，不能用于比较4090与4060训练速度。
- 单随机种子不支持置信区间、显著性检验或稳定性外推。

## 9. 建议的下一步

1. 下一批实验保存Git提交、完整YAML、MAT文件哈希、CUDA环境和分阶段耗时。
2. 使用至少5个随机种子，并在SnF/noSnF之间匹配每轮活跃人数或累计样本预算。
{probe_next_step}
4. 若研究通信效率，额外保存上下行模型字节数和真实传输次数，避免只使用客户端次数代理。

## 10. 待进一步回答的问题

- 在逐轮活跃人数或累计样本预算严格匹配后，HFL-SnF的Q、正确边界质量和 `test_acc` 优势是否仍然存在？
- Q的提升主要来自覆盖率增加，还是来自活跃客户端正确有效共识本身提高？建议在多随机种子结果中同时报告两项分量。
- 若按客户端样本数、FLOPs或上传字节数归一化，当前“每epoch产生更多正确共识证据”的排序是否改变？

本报告由 `analyze_experiment_suite.py` 从原始结果重新计算；逐轮证据、质量检查和来源哈希见同目录CSV及 `analysis_manifest.json`。
""".format(
        batch_name=profile["batch_name"],
        best_label=summaries[best_scenario]["label"],
        best_acc=format_percent(summaries[best_scenario]["last20_test_acc_mean"]),
        best_std=100.0 * summaries[best_scenario]["last20_test_acc_std"],
        stable_label=summaries[stable_scenario]["label"],
        mechanism_evidence_statement=mechanism_evidence_statement,
        mechanism_definition_statement=mechanism_definition_statement,
        probe_evidence_limit=probe_evidence_limit,
        probe_storage_statement=probe_storage_statement,
        probe_longitudinal_statement=probe_longitudinal_statement,
        correctness_statement=correctness_statement,
        probe_limitation_bullet=probe_limitation_bullet,
        probe_next_step=probe_next_step,
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
        candidate_consensus_table=markdown_table(
            [
                "方案", "候选后20轮纯S", "候选后20轮正确S", "候选后20轮错误S",
                "活跃后20轮正确S", "纯S面积50/100",
                "正确S面积50/100", "正确S的MA10稳定达到0.60",
            ],
            candidate_consensus_rows,
        ),
        mechanism_table=markdown_table(
            [
                "方案", "Q后20轮", "Q累计量", "Q面积50/100",
                "Q的MA10稳定达到0.20/0.40/0.50",
                "后20轮正确边界质量", "后20轮有符号边界质量",
            ],
            mechanism_rows,
        ),
        candidate_consensus_ranking=candidate_consensus_ranking,
        candidate_top_label=candidate_top_label,
        mechanism_ranking=mechanism_ranking,
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
        source_filenames = (
            COMMON_REQUIRED_RESULT_FILES
            + get_probe_input_files(experiment.path, experiment.metadata)
            + OPTIONAL_RESULT_FILES
        )
        for filename in source_filenames:
            path = experiment.path / filename
            if path.is_file():
                hashes[filename] = sha256_file(path)
        sources[experiment.scenario] = {
            "label": experiment.label,
            "path": str(experiment.path),
            "rounds": len(experiment.schedule),
            "probe_format": experiment.probe_format,
            "probe_count": int(experiment.true_labels.shape[1]),
            "probe_set_hash": experiment.probe_set_hash or None,
            "has_probe_meta": (experiment.path / "probe_meta.csv").is_file(),
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
        "四方案的参与机制共识吞吐量、累计量、相对差值和正确边界质量有何差异？",
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
    limitations = [
        "只有单随机种子，不能声明统计显著",
        "没有逐客户端训练事件或更新哈希，不能从结果文件独立证明每次本地训练调用",
        "没有模型快照，不能独立复算指标是否来自最终云模型",
        "运行时MAT绝对路径在当前工作区失效，分析以JSONL为准",
        "没有可靠运行时间和通信字节日志",
    ]
    if any(
            experiment.probe_format == "legacy_csv"
            and np.any(experiment.true_labels < 0)
            for experiment in experiments
    ):
        # 仅对缺失真值的历史格式保留这项解释限制。
        limitations.insert(3, "至少一组历史单图探针没有保存真值标签")

    return {
        "schema_version": "5.0",
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
            "consensus_thresholds": list(CONSENSUS_THRESHOLDS),
            "mechanism_thresholds": list(MECHANISM_THRESHOLDS),
            "mechanism_primary_metric": (
                "active_coverage_ratio × active_correct_effective"
            ),
            "participation_margin_metric": (
                "sum_active_probe(max(true_probability - "
                "max_wrong_probability, 0)) / "
                "(candidate_count × probe_count)"
            ),
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
        "limitations": limitations,
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
    for experiment in experiments:
        validate_npz_summary(
            experiment, rows_by_scenario[experiment.scenario]
        )
        add_quality_check(
            quality_checks,
            experiment.label,
            "训练端探针摘要可由原始探针重算",
            "通过" if experiment.probe_format == "npz" else "不适用",
            (
                "逐字段在1e-6容差内核对"
                if experiment.probe_format == "npz"
                else "历史CSV没有独立的逐epoch摘要文件"
            ),
            "关键" if experiment.probe_format == "npz" else "低",
        )
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
            "{}：后20轮测试准确率 {:.4f}，累计参与 {}，"
            "后20轮机制共识 {}，后20轮正确边界质量 {}".format(
                item["label"], item["last20_test_acc_mean"],
                item["active_total"],
                format_number(
                    item["coverage_weighted_active_correct_last20"], 4
                ),
                format_number(item["participation_positive_margin_last20"], 4),
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
