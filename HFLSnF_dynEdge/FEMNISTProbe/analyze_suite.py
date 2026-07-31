"""校验四组正式结果并生成简体中文机制探索报告。"""

from __future__ import absolute_import

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_SCENARIOS = {
    "hfl_snf_fixed",
    "hfl_no_snf_fixed",
    "fl_snf",
    "fl_no_snf",
}


def parse_arguments():
    """解析正式套件目录与报告输出路径。"""
    parser = argparse.ArgumentParser(description="分析FEMNIST四方案结果。")
    parser.add_argument("--suite_dir", required=True, type=str)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _read_csv(path):
    """读取CSV并返回字典行。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _float(row, key):
    """读取CSV有限浮点字段。"""
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError("{}包含非有限字段{}。".format(key, value))
    return value


def _optional_float(row, key):
    """读取允许为空的CSV浮点字段，空值返回NaN。"""
    text = str(row.get(key, "")).strip()
    if not text:
        return np.nan
    value = float(text)
    return value if np.isfinite(value) else np.nan


def _mean_optional(rows, key):
    """计算一个允许部分时间点为空的指标均值。"""
    values = np.asarray([
        _optional_float(row, key) for row in rows
    ], dtype=np.float64)
    valid = np.isfinite(values)
    return float(np.mean(values[valid])) if np.any(valid) else np.nan


def _format_optional(value):
    """把可选浮点值格式化为四位小数或中文缺失标记。"""
    return "{:.4f}".format(value) if np.isfinite(value) else "—"


def _validate_h5(run_dir, metadata):
    """校验流式HDF5的形状、坐标、概率和有效NaN位置。"""
    path = run_dir / "probe_probabilities.h5"
    with h5py.File(str(path), "r") as archive:
        written_count = int(archive.attrs["written_count"])
        if written_count != 101:
            raise ValueError("{}的HDF5时间点不是101。".format(run_dir.name))
        clients = archive["client_probabilities"][:written_count]
        edges = archive["edge_probabilities"][:written_count]
        cloud = archive["cloud_probabilities"][:written_count]
        active = archive["active_client_mask"][:written_count]
        edge_active = archive["edge_active_mask"][:written_count]
        epochs = archive["global_epochs"][:written_count]
        cycles = archive["topology_cycle_indexes"][:written_count]
        mat_indexes = archive["mat_topology_indexes"][:written_count]
        if clients.shape != (101, 37, 620, 62):
            raise ValueError("{}客户端探针形状错误。".format(run_dir.name))
        expected_edge_slots = 6 if metadata["architecture"] == "hfl" else 1
        if edges.shape != (101, expected_edge_slots, 620, 62):
            raise ValueError("{}边缘探针形状错误。".format(run_dir.name))
        if cloud.shape != (101, 620, 62):
            raise ValueError("{}云探针形状错误。".format(run_dir.name))
        if epochs.tolist() != [-1] + list(range(49, 5000, 50)):
            raise ValueError("{}探针epoch坐标错误。".format(run_dir.name))
        expected_cycles = [-1] + [
            epoch // 200 for epoch in range(49, 5000, 50)
        ]
        expected_mat_indexes = [-1] + [
            epoch % 200 for epoch in range(49, 5000, 50)
        ]
        if cycles.tolist() != expected_cycles:
            raise ValueError("{}探针MAT循环编号错误。".format(run_dir.name))
        if mat_indexes.tolist() != expected_mat_indexes:
            raise ValueError("{}探针MAT行坐标错误。".format(run_dir.name))
        if not np.all(np.isfinite(clients)) or not np.all(np.isfinite(cloud)):
            raise ValueError("{}客户端或云探针包含NaN/Inf。".format(run_dir.name))
        if not np.allclose(np.sum(clients, axis=-1), 1.0, atol=1e-5):
            raise ValueError("{}客户端概率行和错误。".format(run_dir.name))
        if not np.allclose(np.sum(cloud, axis=-1), 1.0, atol=1e-5):
            raise ValueError("{}云概率行和错误。".format(run_dir.name))
        valid_edge_values = edges[edge_active]
        inactive_edge_values = edges[~edge_active]
        if np.any(~np.isfinite(valid_edge_values)):
            raise ValueError("{}活跃边缘探针包含NaN/Inf。".format(run_dir.name))
        if not np.allclose(
                np.sum(valid_edge_values, axis=-1), 1.0, atol=1e-5
        ):
            raise ValueError("{}活跃边缘概率行和错误。".format(run_dir.name))
        if np.any(np.isfinite(inactive_edge_values)):
            raise ValueError("{}非活跃边缘槽位没有保持NaN。".format(run_dir.name))
        if metadata["architecture"] == "fl" and np.any(edge_active):
            raise ValueError("{}普通FL不应标记活跃边缘。".format(run_dir.name))
        return {
            "written_count": written_count,
            "probe_hash": str(archive.attrs["probe_hash"]),
            "mean_active_coverage": float(np.mean(active[1:])),
        }


def _validate_schedule(schedule_lines, metadata, run_name):
    """逐轮校验5000条全局epoch、MAT循环、候选映射和全量同步记录。"""
    is_v2 = str(metadata.get("schema_version", "")).endswith("_v2")
    candidate_clients = list(
        metadata.get(
            "candidate_client_ids",
            metadata.get("candidate_writer_ids", []),
        )
    )
    if len(candidate_clients) != 37:
        raise ValueError("{}候选客户端清单不是37人。".format(run_name))
    candidate_hash = str(metadata["candidate_manifest_hash"])
    for expected_epoch, line in enumerate(schedule_lines):
        row = json.loads(line)
        if int(row["global_epoch"]) != expected_epoch:
            raise ValueError("{}拓扑epoch不连续。".format(run_name))
        if int(row["topology_cycle_index"]) != expected_epoch // 200:
            raise ValueError("{}拓扑循环编号错误。".format(run_name))
        if int(row["mat_topology_index"]) != expected_epoch % 200:
            raise ValueError("{}拓扑MAT行错误。".format(run_name))
        if str(row["candidate_manifest_hash"]) != candidate_hash:
            raise ValueError("{}拓扑候选哈希错误。".format(run_name))
        active_slots = [int(value) for value in row["active_candidate_slots"]]
        expected_clients = [
            candidate_clients[slot] for slot in active_slots
        ]
        active_key = "active_client_ids" if is_v2 else "active_writer_ids"
        if list(row[active_key]) != expected_clients:
            raise ValueError("{}活跃槽位与客户端未对齐。".format(run_name))
        if is_v2:
            population_count = int(metadata["population_client_count"])
            if row.get("synchronized_client_ids") != list(
                    range(population_count)
            ):
                raise ValueError("{}未记录250个逻辑客户端全量同步。".format(run_name))
            if int(row.get("synchronized_client_count", -1)) != population_count:
                raise ValueError("{}全量同步客户端数量错误。".format(run_name))
            if int(row.get("mat_participant_count", -1)) != len(active_slots):
                raise ValueError("{}MAT参与人数与活跃槽位不一致。".format(run_name))
        else:
            if row.get("synchronized_candidate_slots") != list(range(37)):
                raise ValueError("{}未记录37名候选全量同步。".format(run_name))
            if row.get("synchronized_writer_ids") != candidate_clients:
                raise ValueError("{}同步槽位与书写者未对齐。".format(run_name))


def _stage_summary(rows, start_epoch, end_epoch):
    """计算一个训练阶段的核心指标均值。"""
    selected = [
        row for row in rows
        if start_epoch <= int(float(row["global_epoch"])) + 1 <= end_epoch
    ]
    if not selected:
        raise ValueError("阶段{}–{}没有指标行。".format(start_epoch, end_epoch))
    return {
        "test_accuracy": float(np.mean([
            _float(row, "test_accuracy") for row in selected
        ])),
        "candidate_effective": float(np.mean([
            _float(row, "candidate_effective") for row in selected
        ])),
        "active_coverage": float(np.mean([
            _float(row, "active_coverage") for row in selected
        ])),
        "active_effective": float(np.mean([
            _float(row, "active_effective") for row in selected
        ])),
        "active_correct_effective": float(np.mean([
            _float(row, "active_correct_effective") for row in selected
        ])),
        "active_wrong_effective": float(np.mean([
            _float(row, "active_wrong_effective") for row in selected
        ])),
        "q": float(np.mean([
            _float(row, "coverage_weighted_active_correct_effective")
            for row in selected
        ])),
        "within_edge_effective": _mean_optional(
            selected, "within_edge_effective"
        ),
        "edge_effective": _mean_optional(selected, "edge_effective"),
        "edge_cloud_effective": _mean_optional(
            selected, "edge_cloud_effective"
        ),
    }


def _cycle_summary(rows, cycle_index):
    """汇总一个200轮MAT周期内的四个固定评估时间点。"""
    selected = [
        row for row in rows
        if int(float(row["topology_cycle_index"])) == int(cycle_index)
    ]
    if len(selected) != 4:
        raise ValueError(
            "MAT周期{}应有4个评估点，实际为{}。".format(
                cycle_index, len(selected)
            )
        )
    summary = _stage_summary(
        selected,
        int(cycle_index) * 200 + 1,
        (int(cycle_index) + 1) * 200,
    )
    summary["cycle_index"] = int(cycle_index)
    return summary


def _load_run(run_dir):
    """读取并完整校验一个正式方案目录。"""
    metadata = json.loads(
        (run_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("status") != "complete":
        raise ValueError("{}尚未完成。".format(run_dir.name))
    if int(metadata.get("comm_round", 0)) != 5000:
        raise ValueError("{}不是5000轮正式结果。".format(run_dir.name))
    schedule_lines = (
        run_dir / "topology_schedule.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    if len(schedule_lines) != 5000:
        raise ValueError("{}拓扑日志不是5000行。".format(run_dir.name))
    _validate_schedule(schedule_lines, metadata, run_dir.name)
    last_schedule = json.loads(schedule_lines[-1])
    if (
            int(last_schedule["topology_cycle_index"]) != 24
            or int(last_schedule["mat_topology_index"]) != 199
    ):
        raise ValueError("{}最后一轮MAT循环坐标错误。".format(run_dir.name))
    probe_rows = _read_csv(run_dir / "probe_epoch_summary.csv")
    test_rows = _read_csv(run_dir / "test_metrics.csv")
    if len(probe_rows) != 101 or len(test_rows) != 101:
        raise ValueError("{}摘要或测试时间点不是101。".format(run_dir.name))
    merged = []
    for probe_row, test_row in zip(probe_rows, test_rows):
        if probe_row["global_epoch"] != test_row["global_epoch"]:
            raise ValueError("{}探针和测试epoch未对齐。".format(run_dir.name))
        if (
                probe_row["topology_cycle_index"]
                != test_row["topology_cycle_index"]
                or probe_row["mat_topology_index"]
                != test_row["mat_topology_index"]
        ):
            raise ValueError("{}探针和测试MAT坐标未对齐。".format(run_dir.name))
        merged.append(dict(probe_row, **{
            "test_accuracy": test_row["test_accuracy"],
            "test_loss": test_row["test_loss"],
        }))
        if str(metadata.get("schema_version", "")).endswith("_v2"):
            if int(float(test_row["evaluated_client_count"])) != 250:
                raise ValueError("{}测试没有覆盖250个客户端。".format(run_dir.name))
            if int(float(test_row["test_samples"])) != 77483:
                raise ValueError("{}测试样本数不是77483。".format(run_dir.name))
    h5_audit = _validate_h5(run_dir, metadata)
    return {
        "path": run_dir,
        "metadata": metadata,
        "rows": merged,
        "h5_audit": h5_audit,
        "stages": {
            "0–1000": _stage_summary(merged, 1, 1000),
            "1000–3000": _stage_summary(merged, 1001, 3000),
            "3000–5000": _stage_summary(merged, 3001, 5000),
        },
        "cycles": [
            _cycle_summary(merged, cycle_index)
            for cycle_index in range(25)
        ],
    }


def _find_run_directories(suite_dir):
    """在套件目录中寻找四个完成的实验目录。"""
    run_root = suite_dir / "runs"
    if not run_root.is_dir():
        raise FileNotFoundError("套件目录缺少runs子目录。")
    candidates = [
        path for path in run_root.iterdir()
        if path.is_dir() and (path / "experiment_metadata.json").is_file()
    ]
    if len(candidates) != 4:
        raise ValueError("正式套件必须恰好包含四个实验目录。")
    return candidates


def _markdown_table(headers, rows):
    """生成GitHub Markdown表格。"""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def build_report(runs):
    """根据四方案最终值和阶段均值生成简体中文报告。"""
    ordered = sorted(runs, key=lambda item: item["metadata"]["scenario"])
    overview_rows = []
    candidate_rows = []
    active_rows = []
    hierarchy_rows = []
    for run in ordered:
        row = run["rows"][-1]
        scenario = run["metadata"]["scenario"]
        overview_rows.append([
            scenario,
            "{:.4f}".format(_float(row, "test_accuracy")),
            "{:.4f}".format(_float(row, "cloud_probe_accuracy")),
        ])
        candidate_rows.append([
            scenario,
            "{:.4f}".format(_float(row, "candidate_agreement")),
            "{:.4f}".format(_float(row, "candidate_certainty")),
            "{:.4f}".format(_float(row, "candidate_effective")),
            "{:.4f}".format(_float(
                row, "candidate_correct_effective"
            )),
            "{:.4f}".format(_float(
                row, "candidate_wrong_effective"
            )),
        ])
        active_rows.append([
            scenario,
            "{:.4f}".format(_float(row, "active_coverage")),
            "{:.4f}".format(_float(row, "active_agreement")),
            "{:.4f}".format(_float(row, "active_certainty")),
            "{:.4f}".format(_float(row, "active_effective")),
            "{:.4f}".format(_float(row, "active_correct_effective")),
            "{:.4f}".format(_float(row, "active_wrong_effective")),
            "{:.4f}".format(_float(
                row, "coverage_weighted_active_correct_effective"
            )),
        ])
        hierarchy_rows.append([
            scenario,
            _format_optional(_optional_float(
                row, "within_edge_effective"
            )),
            _format_optional(_optional_float(row, "edge_effective")),
            _format_optional(_optional_float(
                row, "edge_cloud_effective"
            )),
        ])
    stage_rows = []
    for run in ordered:
        for stage_name, values in run["stages"].items():
            stage_rows.append([
                run["metadata"]["scenario"],
                stage_name,
                "{:.4f}".format(values["test_accuracy"]),
                "{:.4f}".format(values["candidate_effective"]),
                "{:.4f}".format(values["active_coverage"]),
                "{:.4f}".format(values["active_effective"]),
                "{:.4f}".format(values["active_correct_effective"]),
                "{:.4f}".format(values["active_wrong_effective"]),
                "{:.4f}".format(values["q"]),
                _format_optional(values["edge_effective"]),
            ])
    cycle_rows = []
    for run in ordered:
        for values in run["cycles"]:
            cycle_rows.append([
                run["metadata"]["scenario"],
                values["cycle_index"] + 1,
                "{}–{}".format(
                    values["cycle_index"] * 200 + 1,
                    (values["cycle_index"] + 1) * 200,
                ),
                "{:.4f}".format(values["test_accuracy"]),
                "{:.4f}".format(values["active_coverage"]),
                "{:.4f}".format(values["active_correct_effective"]),
                "{:.4f}".format(values["active_wrong_effective"]),
                "{:.4f}".format(values["q"]),
            ])
    shared = ordered[0]["metadata"]
    if str(shared.get("schema_version", "")).endswith("_v2"):
        data_description = (
            "四方案共同使用完整FEMNIST的250客户端Dirichlet划分"
            "（alpha={alpha:g}、seed={seed}）、固定37个逻辑客户端槽位、"
            "固定620张探针、同一初始模型和同一varAlpha=0.1 MAT。"
        ).format(
            alpha=float(shared["partition_alpha"]),
            seed=int(shared["partition_seed"]),
        )
    else:
        data_description = (
            "四方案共同使用固定37名书写者、固定620张探针、"
            "同一初始模型和同一varAlpha=0.1 MAT。"
        )
    return """# FEMNIST MAT四方案探针机制探索报告

## 结论边界

本报告基于单随机种子0，仅用于描述四种MAT拓扑机制下共识、覆盖率和准确率的变化，不声明统计显著性或因果关系。{data_description}

## 完整性审计

- 每组训练5000轮，MAT的200行显式循环25次。
- 每组包含5000条拓扑日志和101个固定探针/完整测试时间点。
- 候选清单哈希：`{candidate_hash}`
- 探针哈希：`{probe_hash}`
- 初始模型哈希：`{model_hash}`
- MAT文件哈希：`{mat_hash}`

## 最终时间点

### 准确率

{overview_table}

### 候选客户端共识

{candidate_table}

### 活跃客户端共识与主指标

{active_table}

### 分层共识

{hierarchy_table}

## 分阶段均值

{stage_table}

## 25个MAT周期均值

{cycle_table}

## 阅读建议

原始候选有效共识S会受到未活跃候选仍保留云模型的影响，方案比较必须同时读取活跃正确/错误有效共识与主指标 `Q=活跃覆盖率×活跃正确有效共识`。SnF和no-SnF的MAT参与人数不同，因此不能把准确率或S的单一差异直接解释成SnF的独立因果效应。
""".format(
        candidate_hash=shared["candidate_manifest_hash"],
        data_description=data_description,
        probe_hash=shared["probe_hash"],
        model_hash=shared["initial_model_hash"],
        mat_hash=shared["mat_file_hash"],
        overview_table=_markdown_table(
            ["方案", "完整测试准确率", "云探针准确率"],
            overview_rows,
        ),
        candidate_table=_markdown_table(
            ["方案", "候选A", "候选C", "候选S", "候选正确S", "候选错误S"],
            candidate_rows,
        ),
        active_table=_markdown_table(
            [
                "方案", "活跃覆盖率", "活跃A", "活跃C",
                "活跃S", "活跃正确S", "活跃错误S", "Q",
            ],
            active_rows,
        ),
        hierarchy_table=_markdown_table(
            ["方案", "组内S", "边缘S", "边缘-云S"],
            hierarchy_rows,
        ),
        stage_table=_markdown_table(
            [
                "方案", "阶段", "准确率", "候选S", "覆盖率",
                "活跃S", "活跃正确S", "活跃错误S", "Q", "边缘S",
            ],
            stage_rows,
        ),
        cycle_table=_markdown_table(
            [
                "方案", "周期", "轮次", "准确率", "覆盖率",
                "活跃正确S", "活跃错误S", "Q",
            ],
            cycle_rows,
        ),
    )


def main():
    """校验四组公共哈希并写入中文报告。"""
    args = parse_arguments()
    suite_dir = Path(args.suite_dir).resolve()
    runs = [_load_run(path) for path in _find_run_directories(suite_dir)]
    scenarios = {run["metadata"]["scenario"] for run in runs}
    if scenarios != EXPECTED_SCENARIOS:
        raise ValueError("四方案集合错误：{}。".format(sorted(scenarios)))
    for key in [
        "candidate_manifest_hash",
        "partition_hash",
        "probe_hash",
        "initial_model_hash",
        "mat_file_hash",
        "gpu_name",
        "amp_enabled",
    ]:
        values = {str(run["metadata"].get(key)) for run in runs}
        if len(values) != 1:
            raise ValueError("四方案公共字段{}不一致：{}。".format(key, values))
    report = build_report(runs)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else suite_dir / "FEMNIST_MAT四方案探针报告.md"
    )
    output_path.write_text(report, encoding="utf-8")
    print("FEMNIST_PROBE_REPORT={}".format(output_path))


if __name__ == "__main__":
    main()
