"""准备、校验并独立运行多数据集多拓扑联邦实验，支持批次恢复。"""

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .core.controlled_schedule import prepare_open_schedule
from .multi_experiment import file_sha256, validate_configurations
from .selection import PACKAGE_DIR, package_path


DEFAULT_CONFIGS = [PACKAGE_DIR / "configs/multi_topology_dataset" / ("open_wn18rr_" + arm + "_seed42.yaml") for arm in ("hflsnf", "hflnosnf", "flnosnf")]
DEFAULT_SOURCE = PACKAGE_DIR.parent / "Topo_other/OPEN/Result_JICN3/variance_control/jieguo_OPEN_epoch1000_varAlpha_0p1_analysis.mat"
DEFAULT_SCHEDULE = PACKAGE_DIR / "artifacts/topologies/open_varalpha0p1_reconstructed.mat"


def write_json(path, payload):
    """原子保存有限值清单，确保失败或中断后仍能恢复批次。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def binding(entry):
    """冻结当前配置、数据、参与调度与共同初始化，防止跨实验续跑。"""

    result = {key: entry[key] for key in ("config_sha256", "data_sha256", "mat_sha256", "partition_hash", "initial_model_hash")}
    result["schedule_hash"] = entry["schedule"]["schedule_hash"]
    return result


def completed_metrics(path):
    """只解析已经完整写入的逐轮CSV，允许训练进程仍在写末行。"""

    if not Path(path).is_file():
        return []
    text = Path(path).read_text(encoding="utf-8-sig")
    end = text.rfind("\n")
    return list(csv.DictReader(io.StringIO(text[:end + 1]))) if end >= 0 else []


def check_finite(rows):
    """拒绝真实损失及已评估验证指标中的非有限值，不检查计时占位列。"""

    for row in rows:
        for key in ("mean_client_train_loss", "val_mrr", "val_hits_at_1", "val_hits_at_3", "val_hits_at_10"):
            if key not in row or not math.isfinite(float(row[key])):
                raise ValueError("第{}轮{}非有限或缺失".format(row.get("round"), key))


def effective_configuration(entry, output_path):
    """把预检后的路径、客户端数与期望哈希写入独立启动配置。"""

    sections = yaml.safe_load(Path(entry["config"]).read_text(encoding="utf-8"))
    values = dict(entry["configuration"])
    values["expected_partition_hash"] = entry["partition_hash"]
    values["expected_topology_schedule_hash"] = entry["schedule"]["schedule_hash"]
    for key, value in values.items():
        targets = [section for section in sections.values() if key in section]
        target = targets[0] if targets else sections.setdefault("train_args", {})
        target[key] = value
    Path(output_path).write_text("# Frozen effective configuration for this batch attempt.\n" + yaml.safe_dump(sections, sort_keys=False), encoding="utf-8")


def validate_result(directory, entry):
    """验证150轮等预算记录、检查点及实际分区和调度身份。"""

    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = completed_metrics(directory / "metrics.csv")
    config = entry["configuration"]
    check_finite(rows)
    if [int(row["round"]) for row in rows] != list(range(1, int(config["comm_round"]) + 1)):
        raise ValueError("逐轮记录不完整或顺序错误")
    # 验证集指标严格决定最佳轮次，拒绝摘要与逐轮记录不一致的结果。
    best_row = max(rows, key=lambda row: float(row["val_mrr"]))
    best_mrr = float(summary["best_validation_mrr_during_training"])
    if not math.isfinite(best_mrr) or abs(best_mrr - float(best_row["val_mrr"])) > 1e-10 or int(summary["best_round"]) != int(best_row["round"]):
        raise ValueError("最佳模型选择与验证MRR记录不一致")
    for key, expected in (("partition_hash", entry["partition_hash"]), ("topology_schedule_hash", entry["schedule"]["schedule_hash"]), ("initial_model_hash", entry["initial_model_hash"])):
        if summary.get(key) != expected:
            raise ValueError("正式结果{}不匹配".format(key))
    if summary.get("dataset") != config["dataset"]:
        raise ValueError("正式结果数据集不匹配")
    if not str(summary.get("device", "")).startswith("cuda"):
        raise ValueError("正式结果不是CUDA训练")
    if summary.get("final_test_metrics") is not None or summary.get("test_evaluation_performed"):
        raise ValueError("首轮应只用验证选模，完整测试使用独立评估入口")
    snapshot = json.loads((directory / "config_snapshot.json").read_text(encoding="utf-8"))
    for key, expected in config.items():
        if snapshot.get(key) != expected:
            raise ValueError("实际配置{}与预检不一致".format(key))
    model_path = directory / "model_best.pt"
    if not model_path.is_file():
        raise ValueError("缺少最佳模型检查点")
    return {"best_validation_mrr": float(summary["best_validation_mrr_during_training"]), "best_round": int(summary["best_round"]), "checkpoint_sha256": file_sha256(model_path)}


def run_child(entry, directory, attempt):
    """监测一个独立训练子进程，数值异常或中断时回收该进程。"""

    config = entry["configuration"]
    output_root = package_path(config["result_root"])
    prefix = config["run_name"] + "_"
    before = set(output_root.glob(prefix + "*"))
    launch = directory / ("launch_{}_{}.yaml".format(config["run_name"], attempt))
    effective_configuration(entry, launch)
    log_path = launch.with_suffix(".log")
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    process = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "HFLSnF_KG_v5.run_federated_transe", "--cf", str(launch)], cwd=str(PACKAGE_DIR.parent), env=environment, stdout=log, stderr=subprocess.STDOUT)
            last_round = 0
            while True:
                created = sorted(set(output_root.glob(prefix + "*")) - before)
                if len(created) > 1:
                    raise ValueError("一次训练出现多个结果目录")
                if created:
                    rows = completed_metrics(created[0] / "metrics.csv")
                    check_finite(rows)
                    if rows and int(rows[-1]["round"]) > last_round:
                        last_round = int(rows[-1]["round"])
                        print("{}：第{}轮，验证MRR={}".format(config["ablation_arm"], last_round, rows[-1]["val_mrr"]), flush=True)
                code = process.poll()
                if code is not None:
                    created = sorted(set(output_root.glob(prefix + "*")) - before)
                    if code or len(created) != 1:
                        raise RuntimeError("训练失败，退出码{}，日志{}".format(code, log_path))
                    return created[0], validate_result(created[0], entry)
                time.sleep(1)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_batch(validation, resume=None):
    """顺序执行所选实验，恢复时跳过已通过且检查点哈希不变的组。"""

    entries = validation["entries"]
    if resume is None:
        root = package_path(entries[0]["configuration"]["result_root"])
        path = root / ("batch_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")) / "batch_summary.json"
        manifest = {"status": "pending", "validation": validation, "entries": [{"binding": binding(entry), "status": "pending", "attempts": 0} for entry in entries]}
        write_json(path, manifest)
    else:
        path = Path(resume).resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if [entry["binding"] for entry in manifest["entries"]] != [binding(entry) for entry in entries]:
            raise ValueError("恢复清单与当前配置、数据或调度不匹配")
    manifest["status"] = "running"
    write_json(path, manifest)
    for index, entry in enumerate(entries):
        state = manifest["entries"][index]
        try:
            if state["status"] == "passed":
                current = validate_result(state["result_dir"], entry)
                if current["checkpoint_sha256"] != state["result"]["checkpoint_sha256"]:
                    raise ValueError("已完成结果检查点发生变化，拒绝跳过")
                continue
            state["status"] = "running"
            state["attempts"] += 1
            write_json(path, manifest)
            result_dir, result = run_child(entry, path.parent, state["attempts"])
            state.update(status="passed", result_dir=str(result_dir), result=result)
            state.pop("error", None)
        except BaseException as error:
            state.update(status="failed", error="{}: {}".format(type(error).__name__, error))
            manifest["status"] = "failed"
            write_json(path, manifest)
            raise
        write_json(path, manifest)
    manifest["status"] = "passed"
    write_json(path, manifest)
    print("批次完成：{}".format(path), flush=True)
    return path


def report_batch(path):
    """重新验证结果后生成中文验证指标与实际参与预算报告。"""

    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    lines = ["# 多拓扑多数据集联邦实验结果", "", "本报告展示验证集指标，训练未执行完整测试。"]
    if any(entry["topology"].get("schedule_provenance") == "synthetic_reconstruction_from_varctrl_counts" for entry in manifest["validation"]["entries"]):
        lines += ["", "本批次包含基于统计人数重建的合成参与调度，其物理可行性未验证。"]
    lines += ["", "| 方法 | 状态 | 最佳验证MRR | 平均参与人数 | 有效数据遍数 |", "|---|---|---:|---:|---:|"]
    for state, entry in zip(manifest["entries"], manifest["validation"]["entries"]):
        best = "待完成"
        if state["status"] == "passed":
            result = validate_result(state["result_dir"], entry)
            if result["checkpoint_sha256"] != state["result"]["checkpoint_sha256"]:
                raise ValueError("报告检查点哈希不匹配")
            best = "{:.6f}".format(result["best_validation_mrr"])
        lines.append("| {} | {} | {} | {:.3f} | {:.3f} |".format(entry["configuration"]["ablation_arm"], state["status"], best, entry["schedule"]["participant_mean"], entry["schedule"]["effective_full_data_passes"]))
    lines += ["", "各方法实际参与次数和有效训练数据量不同，不能将验证差异单独归因于聚合算子。", ""]
    destination = path.with_name("实验结果.md")
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(str(destination))


def main(argv=None):
    """先检查环境与合同，再准备或运行指定配置集合。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "formal3", "report"))
    parser.add_argument("--configs", nargs="+", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--source-mat", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-mat", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch", type=Path)
    args = parser.parse_args(argv)
    if args.resume is not None and args.action != "formal3":
        parser.error("--resume仅用于formal3")
    if args.batch is not None and args.action != "report":
        parser.error("--batch仅用于report")
    if args.action == "prepare":
        audit = prepare_open_schedule(args.source_mat, args.output_mat)
        write_json(args.output_mat.with_suffix(".audit.json"), audit)
        print("OPEN统计重建调度准备完成：{}".format(args.output_mat))
        return 0
    if args.action == "report":
        if args.batch is None:
            parser.error("report需要--batch")
        report_batch(args.batch)
        return 0
    if args.action == "formal3":
        if not torch.cuda.is_available():
            raise RuntimeError("正式联邦训练要求CUDA，禁止退回CPU；未创建正式批次")
        import fedml

    validation = validate_configurations(args.configs)
    if args.action == "validate":
        destination = PACKAGE_DIR / "artifacts/multi_topology_dataset/validation.json"
        write_json(destination, validation)
        print("真实分区及调度校验通过：{}".format(destination))
    else:
        if any(not entry["configuration"].get("require_cuda") or not entry["configuration"].get("using_gpu") for entry in validation["entries"]):
            raise ValueError("正式批次所有配置必须开启using_gpu和require_cuda")
        run_batch(validation, args.resume)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("多拓扑数据集入口失败：{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(1)
