"""独立运行HFLSnF种子42的FedAvg四学习率实验并监测数值稳定性。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .tasks.kge.fixed_count_four_scenarios import PACKAGE_DIR, _load_json, write_json_report
from .tasks.kge.graph_semantic_ablation import CALIBRATION_CONTRACT_PATH
from .tasks.kge.graph_semantic_fedavg_lr_sweep import (
    RESULT_ROOT, ROUND_COUNT, SUITE_NAME, LearningRateScenario,
    expected_flat_config, scenarios, validate_configs, validate_result,
)


CONTRACT_FILE = "fedavg_lr_sweep_formal150_contract.json"
TERMINAL_STATUSES = ("passed", "unstable")
POLL_SECONDS = 1.0
COMPARISON_METRICS = ("best_validation_mrr", "round150_mrr", "last30_mean_mrr")


@dataclass(frozen=True)
class ExperimentDefinition:
    """绑定实验身份和合同，使多个批次复用同一监测与恢复流程。"""

    suite: str
    result_root: str
    scenarios: Callable[[], Tuple[LearningRateScenario, ...]]
    expected_config: Callable[[LearningRateScenario], Dict[str, object]]
    validate_result: Callable[[Path, LearningRateScenario], Dict[str, object]]
    contract_file: str
    batch_prefix: str
    report_stem: str
    report_title: str
    report_description: str
    topology_report: bool = False


def _definition(value: Optional[ExperimentDefinition]) -> ExperimentDefinition:
    """省略实验定义时保持原四学习率入口和既有批次格式不变。"""

    if value is not None:
        return value
    return ExperimentDefinition(
        SUITE_NAME, RESULT_ROOT, scenarios, expected_flat_config, validate_result,
        CONTRACT_FILE, "lr_sweep_batch_", "lr_sweep", "FedAvg四学习率实验报告",
        "固定HFLSnF、种子42，四组从同一初始化独立训练。",
    )


class TrainingRunError(RuntimeError):
    """携带当前尝试产生目录的训练或数值监测错误。"""

    def __init__(self, message: str, created_results: Sequence[Path] = ()) -> None:
        """保存错误原因及仅属于当前尝试的结果目录。"""

        super().__init__(message)
        self.created_results = tuple(Path(path).resolve() for path in created_results)


class NumericalInstability(TrainingRunError):
    """表示完整逐轮记录出现非有限损失或验证指标。"""


def _timestamp() -> str:
    """返回含时区的状态记录时间。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    """分块计算配置或检查点的内容哈希。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(item: LearningRateScenario) -> Dict[str, object]:
    """冻结一个场景的配置、来源、分区和拓扑身份。"""

    return {
        "scenario_id": item.scenario.scenario_id, "learning_rate": item.learning_rate,
        "config": str(item.scenario.config_path),
        "config_sha256": _file_sha256(item.scenario.config_path),
        "source_config_sha256": _file_sha256(item.scenario.source_config_path),
        "calibration_sha256": _file_sha256(CALIBRATION_CONTRACT_PATH),
        "partition_hash": item.scenario.partition_hash,
        "topology_hash": item.scenario.topology_contract.schedule_hash,
    }


def _empty_batch_payload(validation: Mapping[str, object],
                         definition: Optional[ExperimentDefinition] = None) -> Dict[str, object]:
    """创建当前实验所有场景均待重新训练的批次，不引用历史结果。"""

    definition = _definition(definition)
    entries = []
    for item in definition.scenarios():
        entries.append(dict(_binding(item), status="pending", attempts=[],
                            result_dir=None, checkpoint_sha256=None, error=None))
    return {
        "schema_version": 1, "suite": definition.suite, "status": "pending",
        "started_at": _timestamp(), "completed_at": None,
        "validation": dict(validation), "entries": entries,
    }


def _save_batch_manifest(path: Path, payload: Dict[str, object]) -> None:
    """先写临时文件再替换清单，避免中断损坏恢复信息。"""

    payload["updated_at"] = _timestamp()
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def create_batch_manifest(validation: Mapping[str, object],
                          definition: Optional[ExperimentDefinition] = None) -> Path:
    """在新时间戳目录创建通过真实分区复算的正式批次。"""

    if validation.get("status") != "passed" or not validation.get("recompute_partitions"):
        raise ValueError("正式批次必须通过真实分区复算")
    definition = _definition(definition)
    directory = PACKAGE_DIR / definition.result_root / (
        definition.batch_prefix + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "batch_summary.json"
    _save_batch_manifest(path, _empty_batch_payload(validation, definition))
    return path.resolve()


def _verify_passed_entry(entry: Mapping[str, object], item: LearningRateScenario,
                         definition: Optional[ExperimentDefinition] = None) -> None:
    """恢复或报告前重新校验已通过结果及检查点，拒绝过期通过状态。"""

    if not entry.get("result_dir") or not entry.get("checkpoint_sha256"):
        raise ValueError("已通过项缺少结果目录或检查点哈希")
    directory = Path(str(entry["result_dir"]))
    if _definition(definition).validate_result(directory, item)["status"] != "passed":
        raise ValueError("已通过项的结果合同已失效：" + item.scenario.scenario_id)
    if _file_sha256(directory / "model_best.pt") != entry["checkpoint_sha256"]:
        raise ValueError("已通过项的最佳检查点内容发生变化")


def _load_batch_manifest(path: Path,
                         definition: Optional[ExperimentDefinition] = None) -> Dict[str, object]:
    """验证恢复清单的固定顺序、配置绑定和已完成结果。"""

    definition = _definition(definition)
    payload = _load_json(path)
    if payload.get("schema_version") != 1 or payload.get("suite") != definition.suite:
        raise ValueError("实验批次身份不一致")
    entries = payload.get("entries")
    items = definition.scenarios()
    if not isinstance(entries, list) or len(entries) != len(items):
        raise ValueError("实验批次必须包含{}项".format(len(items)))
    if payload.get("validation", {}).get("status") != "passed" or not (
        payload.get("validation", {}).get("recompute_partitions")
    ):
        raise ValueError("批次没有有效的正式启动校验")
    for entry, item in zip(entries, items):
        for key, value in _binding(item).items():
            if entry.get(key) != value:
                raise ValueError("恢复绑定不一致：{} {}".format(item.tag, key))
        if entry.get("status") not in (
            "pending", "running", "passed", "unstable", "failed", "interrupted",
        ) or not isinstance(entry.get("attempts"), list):
            raise ValueError("批次项目状态或尝试记录非法")
        if entry["status"] == "passed":
            _verify_passed_entry(entry, item, definition)
    return payload


def check_formal_environment() -> None:
    """在创建批次或启动训练前检查当前解释器的CUDA和FedML。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("正式FedAvg训练要求CUDA；当前PyTorch不可用，禁止退回CPU。")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("配置要求cuda:0，但没有可用CUDA设备。")
    # 实际导入用于发现安装不完整的FedML，不能只检测包目录是否存在。
    importlib.import_module("fedml")


def _result_directories(item: LearningRateScenario,
                        definition: Optional[ExperimentDefinition] = None) -> Tuple[Path, ...]:
    """仅列出当前实验根目录及场景运行名对应的训练目录。"""

    definition = _definition(definition)
    root = PACKAGE_DIR / definition.result_root
    prefix = str(definition.expected_config(item)["run_name"]) + "_"
    return tuple(sorted(path.resolve() for path in root.glob(prefix + "*") if path.is_dir()))


def _read_completed_metrics(path: Path) -> List[Dict[str, str]]:
    """只解析已写到换行符的完整CSV记录，忽略正在写入的末行。"""

    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        text = handle.read()
    end = text.rfind("\n")
    if end < 0:
        return []
    rows = []
    for row in csv.DictReader(io.StringIO(text[:end + 1])):
        if None in row or any(value is None for value in row.values()):
            raise TrainingRunError("完整CSV记录的列数与表头不一致")
        rows.append(dict(row))
    return rows


def _check_finite_metrics(rows: Sequence[Mapping[str, str]]) -> None:
    """检查有限损失与全部验证数值，允许关闭剖析时的占位NaN。"""

    for row in rows:
        if "round" not in row or "mean_client_train_loss" not in row or "val_mrr" not in row:
            raise TrainingRunError("逐轮CSV缺少轮号、损失或验证MRR")
        fields = ["mean_client_train_loss"] + [key for key in row if key.startswith("val_")]
        for field in fields:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as error:
                raise TrainingRunError("第{}轮字段{}不是数值".format(row["round"], field)) from error
            if not math.isfinite(value):
                raise NumericalInstability("第{}轮{}={}，停止当前候选".format(
                    row["round"], field, row[field],
                ))


def _stop_process(process: subprocess.Popen) -> None:
    """只停止本入口创建的子进程，正常终止超时后再强制结束。"""

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        # 子进程可能在poll和终止请求之间退出，仍须回收而不能覆盖原始错误。
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _run_scenario(item: LearningRateScenario, log_path: Path,
                  definition: Optional[ExperimentDefinition] = None) -> Path:
    """调用原训练入口并逐轮监测，成功时绑定唯一新增结果目录。"""

    definition = _definition(definition)
    before = set(_result_directories(item, definition))
    process = None
    created: Tuple[Path, ...] = ()
    last_round = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-m", "HFLSnF_KG_v5.run_federated_transe",
               "--cf", str(item.scenario.config_path)]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(PACKAGE_DIR.parent),
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
            while True:
                created = tuple(path for path in _result_directories(item, definition) if path not in before)
                if len(created) > 1:
                    raise TrainingRunError("一次尝试出现多个结果目录，拒绝绑定")
                if created:
                    rows = _read_completed_metrics(created[0] / "metrics.csv")
                    _check_finite_metrics(rows)
                    if rows and int(rows[-1]["round"]) > last_round:
                        last_round = int(rows[-1]["round"])
                        print("{}：第{}轮，验证MRR={}，损失={}".format(
                            item.tag, last_round, rows[-1]["val_mrr"],
                            rows[-1]["mean_client_train_loss"],
                        ), flush=True)
                code = process.poll()
                if code is not None:
                    # 退出瞬间可能刚写出末轮；再次读取，避免漏检最后一轮。
                    created = tuple(path for path in _result_directories(item, definition) if path not in before)
                    if len(created) != 1:
                        raise TrainingRunError("训练退出后期望唯一结果目录，实际{}个".format(len(created)))
                    _check_finite_metrics(_read_completed_metrics(created[0] / "metrics.csv"))
                    if code != 0:
                        raise TrainingRunError("训练子进程退出码{}，详见{}".format(code, log_path))
                    return created[0]
                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt as error:
        error.created_results = created
        raise
    except Exception as error:
        if isinstance(error, TrainingRunError):
            error.created_results = created
            raise
        raise TrainingRunError(str(error), created) from error
    finally:
        if process is not None:
            _stop_process(process)


def _record_attempt(manifest: Path, payload: Dict[str, object], entry: Dict[str, object],
                    attempt: Dict[str, object], status: str,
                    error: Optional[BaseException] = None) -> None:
    """持久化当前尝试的完成状态、错误及部分结果目录。"""

    message = "{}: {}".format(type(error).__name__, error) if error is not None else None
    if error is not None:
        created = getattr(error, "created_results", ())
        if created:
            attempt["created_results"] = [str(path) for path in created]
            if len(created) == 1:
                attempt["result_dir"] = str(created[0])
    attempt.update(status=status, completed_at=_timestamp(), error=message)
    entry.update(status=status, result_dir=attempt.get("result_dir"), error=message)
    payload["status"] = status if status in ("failed", "interrupted") else "running"
    _save_batch_manifest(manifest, payload)


def run_training_batch(manifest_path: Path,
                        definition: Optional[ExperimentDefinition] = None) -> int:
    """按固定场景顺序训练，数值不稳定继续，执行失败保留恢复信息。"""

    definition = _definition(definition)
    manifest = Path(manifest_path).resolve()
    payload = _load_batch_manifest(manifest, definition)
    payload.update(status="running", completed_at=None)
    _save_batch_manifest(manifest, payload)
    for entry, item in zip(payload["entries"], definition.scenarios()):
        if entry["status"] in TERMINAL_STATUSES:
            print("跳过已结束候选：{} ({})".format(item.tag, entry["status"]), flush=True)
            continue
        number = len(entry["attempts"]) + 1
        log_path = manifest.parent / "logs" / ("{}_attempt{}.log".format(item.tag, number))
        attempt = {
            "attempt": number, "status": "running", "started_at": _timestamp(),
            "completed_at": None, "log_file": str(log_path), "result_dir": None,
            "created_results": [], "error": None,
        }
        entry["attempts"].append(attempt)
        entry.update(status="running", result_dir=None, checkpoint_sha256=None, error=None)
        _save_batch_manifest(manifest, payload)
        print("开始{}，学习率={}，日志={}".format(item.tag, item.learning_rate, log_path), flush=True)
        try:
            directory = _run_scenario(item, log_path, definition)
            attempt.update(result_dir=str(directory), created_results=[str(directory)])
            contract = definition.validate_result(directory, item)
            contract_path = directory / definition.contract_file
            write_json_report(contract, contract_path)
            attempt["contract_file"] = str(contract_path)
            if contract["status"] != "passed":
                raise TrainingRunError("结果合同未通过，详见{}".format(contract_path), [directory])
            entry["checkpoint_sha256"] = _file_sha256(directory / "model_best.pt")
            _record_attempt(manifest, payload, entry, attempt, "passed")
        except NumericalInstability as error:
            _record_attempt(manifest, payload, entry, attempt, "unstable", error)
            print("{}数值不稳定，继续下一组：{}".format(item.tag, error), flush=True)
        except KeyboardInterrupt as error:
            _record_attempt(manifest, payload, entry, attempt, "interrupted", error)
            print("已保存中断状态：{}".format(manifest), flush=True)
            return 130
        except Exception as error:
            _record_attempt(manifest, payload, entry, attempt, "failed", error)
            print("批次停止：{}；恢复清单：{}".format(error, manifest), flush=True)
            return 1
    payload.update(
        status="completed_with_unstable" if any(entry["status"] == "unstable"
                                               for entry in payload["entries"]) else "passed",
        completed_at=_timestamp(),
    )
    _save_batch_manifest(manifest, payload)
    generate_report(manifest, definition)
    return 0


def summarize_metrics(rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    """从有限训练前缀计算最佳值、末轮值及明确口径的波动和回撤。"""

    finite_rows = []
    for row in rows:
        try:
            _check_finite_metrics([row])
        except NumericalInstability:
            break
        finite_rows.append(row)
    result: Dict[str, object] = {
        "completed_rounds": len(rows), "finite_rounds": len(finite_rows),
        "best_validation_mrr": None, "best_round": None, "round150_mrr": None,
        "last30_mean_mrr": None, "last30_std_mrr": None,
        "max_mrr_drawdown": None, "total_round_seconds": None,
    }
    if not finite_rows:
        return result
    values = [float(row["val_mrr"]) for row in finite_rows]
    best_index = values.index(max(values))
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    seconds = [float(row["round_seconds"]) for row in finite_rows]
    if not all(math.isfinite(value) and value >= 0 for value in seconds):
        raise ValueError("逐轮时间必须是有限非负数")
    result.update(best_validation_mrr=values[best_index],
                  best_round=int(finite_rows[best_index]["round"]),
                  max_mrr_drawdown=drawdown, total_round_seconds=sum(seconds))
    if [int(row["round"]) for row in finite_rows] == list(range(1, ROUND_COUNT + 1)):
        result.update(round150_mrr=values[-1], last30_mean_mrr=statistics.mean(values[-30:]),
                      last30_std_mrr=statistics.pstdev(values[-30:]))
    return result


def _aggregate_seed_values(complete_units: Sequence[Mapping[str, object]],
                           expected_count: int) -> Dict[str, object]:
    """仅在所有计划种子均有效时计算跨种子均值与样本标准差。"""

    result = {"completed_seeds": len(complete_units), "expected_seeds": expected_count}
    valid = len(complete_units) == expected_count and expected_count > 1
    for field in COMPARISON_METRICS:
        values = [float(unit[field]) for unit in complete_units] if valid else []
        result[field + "_mean"] = statistics.mean(values) if valid else None
        result[field + "_std"] = statistics.stdev(values) if valid else None
    return result


def _topology_comparisons(units: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """先按种子配对，再聚合完整种子，避免不同种子相互覆盖或错配。"""

    seeds = tuple(dict.fromkeys(unit["seed"] for unit in units))
    pairs = []
    for seed in seeds:
        by_arm = {unit["arm"]: unit for unit in units if unit["seed"] == seed}
        for left, right in (("hflsnf", "hflnosnf"), ("hflsnf", "flnosnf"), ("hflnosnf", "flnosnf")):
            valid = all(by_arm[arm]["status"] == "passed" and by_arm[arm]["finite_rounds"] == ROUND_COUNT
                        for arm in (left, right))
            pair = {"seed": seed, "comparison": left + "-" + right, "complete": valid}
            for field in COMPARISON_METRICS:
                pair[field] = by_arm[left][field] - by_arm[right][field] if valid else None
            pairs.append(pair)
    result = {"paired_differences": pairs}
    if len(seeds) > 1:
        arms = []
        for arm in ("hflsnf", "hflnosnf", "flnosnf"):
            complete = [unit for unit in units if unit["arm"] == arm and
                        unit["status"] == "passed" and unit["finite_rounds"] == ROUND_COUNT]
            arms.append(dict(_aggregate_seed_values(complete, len(seeds)), arm=arm))
        contrasts = []
        for comparison in ("hflsnf-hflnosnf", "hflsnf-flnosnf", "hflnosnf-flnosnf"):
            complete = [pair for pair in pairs if pair["comparison"] == comparison and pair["complete"]]
            contrasts.append(dict(_aggregate_seed_values(complete, len(seeds)), comparison=comparison))
        result.update(cross_seed_summary=arms, cross_seed_paired_summary=contrasts)
    return result


def _mean_std_cell(record: Mapping[str, object], field: str) -> str:
    """格式化跨种子均值和样本标准差，缺失计划种子时显示未完成。"""

    if record[field + "_mean"] is None:
        return "未完成"
    return "{:.6g} ± {:.6g}".format(record[field + "_mean"], record[field + "_std"])


def generate_report(manifest_path: Path,
                     definition: Optional[ExperimentDefinition] = None) -> Path:
    """输出当前批次JSON、CSV和中文报告，部分结果不冒充150轮结果。"""

    definition = _definition(definition)
    manifest = Path(manifest_path).resolve()
    payload = _load_batch_manifest(manifest, definition)
    units = []
    for entry, item in zip(payload["entries"], definition.scenarios()):
        rows = []
        if entry.get("result_dir"):
            rows = _read_completed_metrics(Path(entry["result_dir"]) / "metrics.csv")
        unit = {key: entry.get(key) for key in (
            "scenario_id", "learning_rate", "status", "result_dir", "error",
        )}
        unit.update(summarize_metrics(rows))
        if definition.topology_report:
            unit["arm"] = item.scenario.arm
            unit["seed"] = item.scenario.seed
            # 完整结果从审计过的汇总读取预算；未完成组不冒充完整150轮预算。
            summary = _load_json(Path(entry["result_dir"]) / "summary.json") if entry["status"] == "passed" else {}
            for field in ("participant_count_mean", "group_count_mean", "effective_global_passes"):
                unit[field] = summary.get(field)
            unit["initial_model_hash"] = summary.get("initial_model_hash")
            unit["partition_hash"] = entry["partition_hash"]
            unit["topology_hash"] = entry["topology_hash"]
        units.append(unit)
    directory = manifest.parent / "analysis"
    directory.mkdir(exist_ok=True)
    report_data = {"suite": definition.suite, "status": payload["status"],
                   "batch": str(manifest), "units": units}
    if definition.topology_report:
        report_data.update(_topology_comparisons(units))
    write_json_report(report_data, directory / (definition.report_stem + "_summary.json"))
    with (directory / (definition.report_stem + "_units.csv")).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(units[0]))
        writer.writeheader()
        writer.writerows(units)
    if "cross_seed_summary" in report_data:
        for name in ("paired_differences", "cross_seed_summary", "cross_seed_paired_summary"):
            records = report_data[name]
            with (directory / (definition.report_stem + "_" + name + ".csv")).open(
                "w", encoding="utf-8-sig", newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    columns = [("learning_rate", "学习率"), ("status", "状态"), ("completed_rounds", "完整轮数"),
               ("best_validation_mrr", "最佳验证MRR"), ("best_round", "最佳轮"),
               ("round150_mrr", "第150轮MRR"), ("last30_mean_mrr", "末30轮均值"),
               ("last30_std_mrr", "末30轮标准差"), ("max_mrr_drawdown", "最大回撤"),
               ("total_round_seconds", "训练秒数")]
    if definition.topology_report:
        columns.insert(0, ("arm", "拓扑"))
        columns.insert(0, ("seed", "种子"))
        columns.extend([("participant_count_mean", "平均参与人数"),
                        ("effective_global_passes", "有效数据遍数")])
    lines = ["# " + definition.report_title, "", definition.report_description, "",
             "| " + " | ".join(title for _, title in columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for unit in units:
        cells = []
        for field, _ in columns:
            value = unit[field]
            cells.append("未完成" if value is None else (
                "{:.6g}".format(value) if isinstance(value, float) else str(value)
            ))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "末30轮统计只对完整有限的150轮曲线计算，标准差采用总体标准差。",
                  "最大回撤为历次验证MRR峰值与后续MRR的最大绝对差；训练秒数为轮级时间之和。",
                  "不稳定或未完成组的最佳值仅描述已完成的有限前缀，不参与完整150轮候选比较。",
                  "状态passed表示结果合同通过；unstable表示数值监测终止，并不表示性能达标。",
                  "本报告仅使用验证集，不包含测试集评估。", ""])
    if definition.topology_report:
        lines.extend(["## 同种子配对差", "", "差值为左侧减右侧，仅对均已通过的完整150轮结果计算。",
                      "", "| 种子 | 比较 | 最佳验证MRR差 | 第150轮MRR差 | 末30轮均值差 |", "|---|---|---:|---:|---:|"])
        for pair in report_data["paired_differences"]:
            cells = [str(pair["seed"]), pair["comparison"]] + [
                "未完成" if pair[field] is None else "{:.6g}".format(pair[field]) for field in COMPARISON_METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        if "cross_seed_summary" in report_data:
            lines.extend(["", "## 跨种子统计", "",
                          "各拓扑及配对差只在全部计划种子有效时计算均值与样本标准差；不以部分种子替代完整结果。",
                          "只有三个种子，不将均值差自动称为统计显著差异。", ""])
            for key, label in (("cross_seed_summary", "拓扑"), ("cross_seed_paired_summary", "比较")):
                lines.extend(["| " + label + " | 有效种子数 | 最佳验证MRR均值及标准差 | 第150轮MRR均值及标准差 | 末30轮均值MRR的跨种子统计 |",
                              "|---|---:|---:|---:|---:|"])
                for record in report_data[key]:
                    cells = [record["arm"] if label == "拓扑" else record["comparison"],
                             "{}/{}".format(record["completed_seeds"], record["expected_seeds"])]
                    cells.extend(_mean_std_cell(record, field) for field in COMPARISON_METRICS)
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")
        lines.extend(["", "三拓扑参与人数和数据暴露量不同，差值属于固定通信轮预算下的系统级收益，不代表纯聚合机制净收益。",
                      "单种子和逐轮标准差不构成跨种子显著性证据；报告不根据预期排序判定通过。", ""])
    report = directory / (definition.report_title + ".md")
    report.write_text("\n".join(lines), encoding="utf-8")
    payload["report_markdown"] = str(report)
    _save_batch_manifest(manifest, payload)
    print("报告：{}".format(report), flush=True)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    """提供配置校验、四组CUDA训练、批次恢复及报告命令。"""

    parser = argparse.ArgumentParser(description="固定HFLSnF种子42的FedAvg四学习率实验")
    parser.add_argument("action", choices=("validate", "formal4", "report"))
    parser.add_argument("--resume", type=Path, help="formal4恢复使用的batch_summary.json")
    parser.add_argument("--batch", type=Path, help="report使用的batch_summary.json")
    parser.add_argument("--skip-partition-recompute", action="store_true",
                        help="仅validate可用；正式训练必须重算真实分区")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """校验命令组合并执行独立四组工作流，环境错误不创建训练批次。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.resume is not None and args.action != "formal4":
        parser.error("--resume仅用于formal4")
    if args.batch is not None and args.action != "report":
        parser.error("--batch仅用于report")
    if args.skip_partition_recompute and args.action != "validate":
        parser.error("只有validate允许跳过分区复算")
    if args.action == "report":
        if args.batch is None:
            parser.error("report需要--batch")
        generate_report(args.batch)
        return 0
    if args.action == "formal4":
        check_formal_environment()
    report = validate_configs(recompute_partitions=not args.skip_partition_recompute)
    print("四组配置合同：{}；真实分区复算：{}".format(
        report["status"], report["recompute_partitions"],
    ), flush=True)
    if report["status"] != "passed":
        for check in report["checks"]:
            if not check["passed"]:
                print("失败项：{}".format(check["name"]), flush=True)
        return 1
    if args.action == "validate":
        print("分区哈希：{}\n拓扑哈希：{}".format(report["partition_hash"], report["topology_hash"]))
        return 0
    manifest = args.resume.resolve() if args.resume is not None else create_batch_manifest(report)
    print("批次清单：{}".format(manifest), flush=True)
    return run_training_batch(manifest)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("四组实验入口失败：{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(1)
