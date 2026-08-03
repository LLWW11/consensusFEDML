"""依次或受控并行运行四个FEMNIST MAT探针方案。"""

from __future__ import absolute_import

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CONFIG_NAMES = [
    "femnist_hfl_snf_u05_5000.yaml",
    "femnist_hfl_no_snf_u05_5000.yaml",
    "femnist_fl_snf_u05_5000.yaml",
    "femnist_fl_no_snf_u05_5000.yaml",
]


def parse_arguments():
    """解析套件模式、GPU编号和显式并行数。"""
    parser = argparse.ArgumentParser(description="运行FEMNIST四方案套件。")
    parser.add_argument(
        "--mode",
        choices=["smoke", "calibrate", "benchmark", "formal"],
        required=True,
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--parallel", type=int, choices=[1, 2], default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="覆盖正式四方案的本地批大小；不传时使用YAML配置。",
    )
    return parser.parse_args()


def _build_command(
        config_path,
        rounds,
        result_root,
        amp_enabled,
        experiment_tag,
        gpu_id,
        reference_baseline=False,
        batch_size=None,
):
    """构造单方案子进程命令。"""
    command = [
        sys.executable,
        "-m",
        "FEMNISTProbe.run_experiment",
        "--yaml_config_file",
        str(config_path),
        "--comm_round_override",
        str(int(rounds)),
        "--eval_interval_override",
        str(min(50, int(rounds))),
        "--amp_override",
        "true" if amp_enabled else "false",
        "--result_root_override",
        str(result_root),
        "--experiment_tag_override",
        str(experiment_tag),
        "--gpu_id_override",
        str(int(gpu_id)),
    ]
    if reference_baseline:
        command.append("--reference_baseline")
    if batch_size is not None:
        command.extend(["--batch_size_override", str(int(batch_size))])
    return command


def _gpu_identity(gpu_id):
    """返回当前套件绑定GPU的名称和总显存，并禁止CPU回退。"""
    if not torch.cuda.is_available():
        raise RuntimeError("套件模式要求CUDA，当前PyTorch无法访问GPU。")
    gpu_id = int(gpu_id)
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise RuntimeError("gpu_id={}超出可见GPU范围。".format(gpu_id))
    properties = torch.cuda.get_device_properties(gpu_id)
    return {
        "gpu_id": gpu_id,
        "gpu_name": str(properties.name),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "cuda_build_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
    }


def _validate_gate_identity(gate, gpu_identity, gate_name):
    """拒绝复用另一GPU型号或另一CUDA运行时生成的门禁。"""
    keys = [
        "gpu_name",
        "gpu_total_memory_bytes",
        "torch_version",
        "cuda_build_version",
        "cudnn_version",
    ]
    mismatches = [
        key for key in keys
        if str(gate.get(key)) != str(gpu_identity.get(key))
    ]
    if mismatches:
        raise RuntimeError(
            "{}门禁与当前GPU环境不一致：{}。".format(
                gate_name, mismatches
            )
        )


def _run_group(commands, log_dir, parallelism):
    """按并行上限运行子进程并返回总耗时和退出码。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    parallelism = int(parallelism)
    start_time = time.perf_counter()

    if parallelism == 1:
        return_codes = []
        child_environment = os.environ.copy()
        # 强制子进程逐行刷新UTF-8输出，保证每轮日志实时到达终端。
        child_environment["PYTHONUNBUFFERED"] = "1"
        child_environment["PYTHONIOENCODING"] = "utf-8"
        for job_index, command in enumerate(commands):
            log_path = log_dir / "job_{:02d}.log".format(job_index)
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=child_environment,
                )
                if process.stdout is None:
                    raise RuntimeError("无法读取子进程标准输出。")
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    print(line, end="", flush=True)
                process.stdout.close()
                return_codes.append(int(process.wait()))
        elapsed = time.perf_counter() - start_time
        if any(code != 0 for code in return_codes):
            raise RuntimeError(
                "至少一个套件子任务失败，退出码为{}；请检查{}。".format(
                    return_codes, log_dir
                )
            )
        return elapsed, return_codes

    pending = list(enumerate(commands))
    running = []
    return_codes = {}
    while pending or running:
        while pending and len(running) < parallelism:
            job_index, command = pending.pop(0)
            log_path = log_dir / "job_{:02d}.log".format(job_index)
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running.append((job_index, process, log_file))
        next_running = []
        for job_index, process, log_file in running:
            return_code = process.poll()
            if return_code is None:
                next_running.append((job_index, process, log_file))
            else:
                log_file.close()
                return_codes[job_index] = int(return_code)
        running = next_running
        if running:
            time.sleep(1.0)
    elapsed = time.perf_counter() - start_time
    ordered_codes = [return_codes[index] for index in range(len(commands))]
    if any(code != 0 for code in ordered_codes):
        raise RuntimeError(
            "至少一个套件子任务失败，退出码为{}；请检查{}。".format(
                ordered_codes, log_dir
            )
        )
    return elapsed, ordered_codes


def _new_result_directories(result_root, before):
    """返回一次子任务后新增的实验目录。"""
    after = {path.resolve() for path in result_root.iterdir() if path.is_dir()}
    return sorted(after.difference(before), key=lambda path: path.stat().st_mtime)


def _read_metric_column(path, column):
    """从CSV读取一个有限浮点指标列。"""
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            text = str(row.get(column, "")).strip()
            if text:
                values.append(float(text))
    return values


def _read_final_timing(run_dir):
    """读取一次100轮运行的最终累计阶段耗时。"""
    rows = []
    with (run_dir / "stage_timing.csv").open(
            "r", encoding="utf-8-sig", newline=""
    ) as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        raise ValueError("{}缺少阶段耗时行。".format(run_dir))
    final = rows[-1]
    fields = [
        "train_seconds",
        "aggregate_seconds",
        "probe_seconds",
        "test_seconds",
        "io_seconds",
        "checkpoint_seconds",
        "elapsed_seconds",
    ]
    return {field: float(final[field]) for field in fields}


def _maximum_gpu_memory_ratio(run_directories):
    """读取监控CSV并返回所有进程观察到的最大整卡显存占用率。"""
    ratios = []
    for run_dir in run_directories:
        monitor_path = run_dir / "gpu_monitor.csv"
        if not monitor_path.is_file():
            raise FileNotFoundError("{}缺少GPU监控文件。".format(run_dir))
        with monitor_path.open(
                "r", encoding="utf-8-sig", newline=""
        ) as file_obj:
            for row in csv.DictReader(file_obj):
                used = str(row.get("memory_used_mb", "")).strip()
                total = str(row.get("memory_total_mb", "")).strip()
                if used and total and float(total) > 0:
                    ratios.append(float(used) / float(total))
    if not ratios:
        raise ValueError("GPU监控没有可用显存样本。")
    return max(ratios)


def _find_tagged_run(run_root, tag):
    """按实验标签唯一定位子进程生成的结果目录。"""
    matches = [
        path for path in run_root.iterdir()
        if path.is_dir()
        and tag in path.name
        and (path / "experiment_metadata.json").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(
            "标签{}应唯一匹配结果目录，实际为{}。".format(tag, matches)
        )
    return matches[0]


def _validate_benchmark_pair(reference_dir, fast_dir):
    """校验参考路径与快速路径使用相同实验输入和精度模式。"""
    reference = json.loads(
        (reference_dir / "experiment_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    fast = json.loads(
        (fast_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    for key in [
        "candidate_manifest_hash",
        "partition_hash",
        "probe_hash",
        "initial_model_hash",
        "mat_file_hash",
        "scenario",
        "comm_round",
        "amp_enabled",
        "gpu_name",
    ]:
        if reference.get(key) != fast.get(key):
            raise ValueError("性能对照字段{}不一致。".format(key))
    if not bool(reference.get("reference_baseline", False)):
        raise ValueError("性能对照参考运行没有启用未优化参考路径。")
    if bool(fast.get("reference_baseline", True)):
        raise ValueError("性能对照快速运行意外启用了参考路径。")


def _compare_amp_runs(fp32_dir, amp_dir):
    """根据200轮准确率、S和Q生成AMP数值门控结果。"""
    fp32_metadata = json.loads(
        (fp32_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    amp_metadata = json.loads(
        (amp_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    for key in [
        "candidate_manifest_hash",
        "partition_hash",
        "probe_hash",
        "initial_model_hash",
        "mat_file_hash",
        "gpu_name",
        "comm_round",
    ]:
        if fp32_metadata.get(key) != amp_metadata.get(key):
            raise ValueError("FP32与AMP校准字段{}不一致。".format(key))
    if bool(fp32_metadata.get("amp_enabled", True)):
        raise ValueError("FP32校准运行意外启用了AMP。")
    if not bool(amp_metadata.get("amp_enabled", False)):
        raise ValueError("AMP校准运行没有真正启用AMP。")
    fp32_test = _read_metric_column(
        fp32_dir / "test_metrics.csv", "test_accuracy"
    )
    amp_test = _read_metric_column(
        amp_dir / "test_metrics.csv", "test_accuracy"
    )
    fp32_s = _read_metric_column(
        fp32_dir / "probe_epoch_summary.csv", "candidate_effective"
    )
    amp_s = _read_metric_column(
        amp_dir / "probe_epoch_summary.csv", "candidate_effective"
    )
    fp32_q = _read_metric_column(
        fp32_dir / "probe_epoch_summary.csv",
        "coverage_weighted_active_correct_effective",
    )
    amp_q = _read_metric_column(
        amp_dir / "probe_epoch_summary.csv",
        "coverage_weighted_active_correct_effective",
    )
    if not (
            len(fp32_test) == len(amp_test)
            and len(fp32_s) == len(amp_s)
            and len(fp32_q) == len(amp_q)
    ):
        raise ValueError("FP32与AMP校准指标行数不一致。")
    correlations = []
    for left, right in [(fp32_test, amp_test), (fp32_s, amp_s), (fp32_q, amp_q)]:
        if len(left) >= 3:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    finite_correlations = [
        value for value in correlations if math.isfinite(value)
    ]
    minimum_correlation = (
        min(finite_correlations)
        if len(finite_correlations) == len(correlations)
        else float("nan")
    )
    accuracy_difference = abs(fp32_test[-1] - amp_test[-1])
    s_difference = abs(fp32_s[-1] - amp_s[-1])
    q_difference = abs(fp32_q[-1] - amp_q[-1])
    final_amp_scale = float(amp_metadata.get("amp_scale", float("nan")))
    numerical_passed = (
        accuracy_difference <= 0.01
        and s_difference <= 0.02
        and q_difference <= 0.02
        and minimum_correlation >= 0.98
        and math.isfinite(final_amp_scale)
        and final_amp_scale > 0
        and int(amp_metadata.get("amp_max_consecutive_backoffs", 0)) <= 2
    )
    fp32_timing = _read_final_timing(fp32_dir)
    amp_timing = _read_final_timing(amp_dir)
    stage_keys = [
        "train_seconds",
        "aggregate_seconds",
        "probe_seconds",
        "test_seconds",
        "io_seconds",
        "checkpoint_seconds",
    ]
    fp32_stage_seconds = sum(fp32_timing[key] for key in stage_keys)
    amp_stage_seconds = sum(amp_timing[key] for key in stage_keys)
    amp_speedup = fp32_stage_seconds / amp_stage_seconds
    performance_passed = amp_speedup > 1.0
    passed = numerical_passed and performance_passed
    return {
        "passed": bool(passed),
        "numerical_passed": bool(numerical_passed),
        "performance_passed": bool(performance_passed),
        "amp_speedup": amp_speedup,
        "fp32_stage_timing": fp32_timing,
        "amp_stage_timing": amp_timing,
        "accuracy_difference": accuracy_difference,
        "candidate_s_difference": s_difference,
        "q_difference": q_difference,
        "minimum_curve_correlation": minimum_correlation,
        "final_amp_scale": final_amp_scale,
        "amp_scale_backoff_count": int(
            amp_metadata.get("amp_scale_backoff_count", 0)
        ),
        "amp_max_consecutive_backoffs": int(
            amp_metadata.get("amp_max_consecutive_backoffs", 0)
        ),
        "fp32_result_dir": str(fp32_dir),
        "amp_result_dir": str(amp_dir),
    }


def _build_benchmark_report(result):
    """把100轮快速路径及并行门禁结果整理为简体中文报告。"""
    parallel_text = (
        "8 GB档按规则未执行双进程测试。"
        if result["parallel_speedup"] is None
        else (
            "双进程相对串行加速为{:.3f}倍，峰值整卡显存占用率为{:.2%}。"
            .format(
                result["parallel_speedup"],
                result["parallel_maximum_gpu_memory_ratio"],
            )
        )
    )
    timing = result["fast_stage_timing"]
    return """# FEMNIST GPU 100轮性能基准报告

## 环境

- GPU：{gpu_name}
- 总显存：{memory_gb:.2f} GB
- PyTorch：{torch_version}
- CUDA构建：{cuda_version}
- cuDNN：{cudnn_version}
- AMP：{amp_enabled}

## 快速路径对照

受控未优化参考路径与正式快速路径使用相同方案、MAT、候选清单、初始模型、批大小和精度模式。快速路径阶段总耗时相对参考路径的加速为 **{fast_speedup:.3f}倍**，1.5倍门槛结果为 `{fast_passed}`。

正式快速路径累计阶段耗时：

- 本地训练：{train_seconds:.3f} 秒
- 聚合：{aggregate_seconds:.3f} 秒
- 探针：{probe_seconds:.3f} 秒
- 完整测试：{test_seconds:.3f} 秒
- I/O等待：{io_seconds:.3f} 秒
- 检查点：{checkpoint_seconds:.3f} 秒

## 并行门禁

{parallel_text}

正式推荐并行度为 `{parallelism}`。该门禁只对本报告记录的同型号GPU、PyTorch、CUDA和cuDNN环境有效。
""".format(
        gpu_name=result["gpu_name"],
        memory_gb=result["gpu_total_memory_bytes"] / float(1024 ** 3),
        torch_version=result["torch_version"],
        cuda_version=result["cuda_build_version"],
        cudnn_version=result["cudnn_version"],
        amp_enabled=result["amp_enabled"],
        fast_speedup=result["fast_path_speedup"],
        fast_passed=result["fast_path_passed_1p5_gate"],
        train_seconds=timing["train_seconds"],
        aggregate_seconds=timing["aggregate_seconds"],
        probe_seconds=timing["probe_seconds"],
        test_seconds=timing["test_seconds"],
        io_seconds=timing["io_seconds"],
        checkpoint_seconds=timing["checkpoint_seconds"],
        parallel_text=parallel_text,
        parallelism=result["recommended_parallelism"],
    )


def main():
    """根据模式执行冒烟、AMP校准、并行基准或正式套件。"""
    args = parse_arguments()
    gpu_identity = _gpu_identity(args.gpu_id)
    compact_gpu = gpu_identity["gpu_total_memory_bytes"] < 16 * 1024 ** 3
    if compact_gpu and args.parallel == 2:
        raise RuntimeError("16GB以下GPU的全部套件模式均禁止双进程。")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = PROJECT_ROOT / "result" / "FEMNISTProbe" / (
        "suite_{}_{}".format(args.mode, timestamp)
    )
    run_root = suite_root / "runs"
    log_root = suite_root / "logs"
    run_root.mkdir(parents=True, exist_ok=False)
    configs = [CONFIG_DIR / name for name in CONFIG_NAMES]

    if args.mode == "smoke":
        commands = [
            _build_command(config, 2, run_root, True, "smoke2", args.gpu_id)
            for config in configs
        ]
        elapsed, _ = _run_group(commands, log_root, args.parallel or 1)
        result = {
            "mode": "smoke",
            "elapsed_seconds": elapsed,
            "passed": True,
        }
        result.update(gpu_identity)
    elif args.mode == "calibrate":
        before = {path.resolve() for path in run_root.iterdir() if path.is_dir()}
        commands = [
            _build_command(
                configs[0],
                200,
                run_root,
                False,
                "calibrate_fp32",
                args.gpu_id,
            ),
            _build_command(
                configs[0],
                200,
                run_root,
                True,
                "calibrate_amp",
                args.gpu_id,
            ),
        ]
        elapsed, _ = _run_group(commands, log_root, 1)
        new_dirs = _new_result_directories(run_root, before)
        if len(new_dirs) != 2:
            raise RuntimeError("AMP校准没有生成恰好两个结果目录。")
        fp32_dir = next(path for path in new_dirs if "fp32" in path.name)
        amp_dir = next(path for path in new_dirs if "amp" in path.name)
        result = _compare_amp_runs(fp32_dir, amp_dir)
        result.update({"mode": "calibrate", "elapsed_seconds": elapsed})
        result.update(gpu_identity)
        gate_path = PROJECT_ROOT / "result" / "FEMNISTProbe" / "amp_gate.json"
        gate_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif args.mode == "benchmark":
        amp_gate_path = (
            PROJECT_ROOT / "result" / "FEMNISTProbe" / "amp_gate.json"
        )
        if not amp_gate_path.is_file():
            raise RuntimeError("性能基准前必须先执行--mode calibrate。")
        amp_gate = json.loads(amp_gate_path.read_text(encoding="utf-8"))
        _validate_gate_identity(amp_gate, gpu_identity, "AMP")
        amp_enabled = bool(amp_gate.get("passed", False))

        reference_command = _build_command(
            configs[0],
            100,
            run_root,
            amp_enabled,
            "benchmark_reference",
            args.gpu_id,
            reference_baseline=True,
        )
        reference_seconds, _ = _run_group(
            [reference_command], log_root / "reference", 1
        )
        fast_command = _build_command(
            configs[0],
            100,
            run_root,
            amp_enabled,
            "benchmark_fast",
            args.gpu_id,
        )
        fast_seconds, _ = _run_group(
            [fast_command], log_root / "fast", 1
        )
        reference_dir = _find_tagged_run(
            run_root, "benchmark_reference"
        )
        fast_dir = _find_tagged_run(run_root, "benchmark_fast")
        _validate_benchmark_pair(reference_dir, fast_dir)
        reference_timing = _read_final_timing(reference_dir)
        fast_timing = _read_final_timing(fast_dir)
        reference_stage_seconds = sum(
            reference_timing[key]
            for key in [
                "train_seconds",
                "aggregate_seconds",
                "probe_seconds",
                "test_seconds",
                "io_seconds",
                "checkpoint_seconds",
            ]
        )
        fast_stage_seconds = sum(
            fast_timing[key]
            for key in [
                "train_seconds",
                "aggregate_seconds",
                "probe_seconds",
                "test_seconds",
                "io_seconds",
                "checkpoint_seconds",
            ]
        )
        fast_path_speedup = reference_stage_seconds / fast_stage_seconds
        if compact_gpu:
            serial_seconds = None
            parallel_seconds = None
            parallel_speedup = None
            maximum_memory_ratio = None
            memory_pressure_ok = True
            parallel_gate_passed = False
        else:
            serial_commands = [
                _build_command(
                    config,
                    100,
                    run_root,
                    amp_enabled,
                    "benchmark_serial{}".format(index),
                    args.gpu_id,
                )
                for index, config in enumerate(configs[:2])
            ]
            parallel_commands = [
                _build_command(
                    config,
                    100,
                    run_root,
                    amp_enabled,
                    "benchmark_parallel{}".format(index),
                    args.gpu_id,
                )
                for index, config in enumerate(configs[:2])
            ]
            serial_seconds, _ = _run_group(
                serial_commands, log_root / "serial", 1
            )
            parallel_seconds, _ = _run_group(
                parallel_commands, log_root / "parallel", 2
            )
            parallel_dirs = [
                _find_tagged_run(
                    run_root, "benchmark_parallel{}".format(index)
                )
                for index in range(2)
            ]
            parallel_speedup = serial_seconds / parallel_seconds
            maximum_memory_ratio = _maximum_gpu_memory_ratio(
                parallel_dirs
            )
            memory_pressure_ok = maximum_memory_ratio <= 0.90
            parallel_gate_passed = (
                parallel_speedup >= 1.5 and memory_pressure_ok
            )
        result = {
            "mode": "benchmark",
            "amp_enabled": amp_enabled,
            "reference_wall_seconds": reference_seconds,
            "fast_wall_seconds": fast_seconds,
            "reference_stage_timing": reference_timing,
            "fast_stage_timing": fast_timing,
            "fast_path_speedup": fast_path_speedup,
            "fast_path_passed_1p5_gate": bool(fast_path_speedup >= 1.5),
            "serial_seconds": serial_seconds,
            "parallel_seconds": parallel_seconds,
            "parallel_speedup": parallel_speedup,
            "parallel_maximum_gpu_memory_ratio": maximum_memory_ratio,
            "parallel_memory_pressure_ok": memory_pressure_ok,
            "recommended_parallelism": 2 if parallel_gate_passed else 1,
            "parallel_passed_1p5_gate": bool(parallel_gate_passed),
        }
        result.update(gpu_identity)
        gate_path = (
            PROJECT_ROOT / "result" / "FEMNISTProbe" / "parallel_gate.json"
        )
        gate_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (suite_root / "性能基准报告.md").write_text(
            _build_benchmark_report(result),
            encoding="utf-8",
        )
    else:
        amp_gate_path = (
            PROJECT_ROOT / "result" / "FEMNISTProbe" / "amp_gate.json"
        )
        if not amp_gate_path.is_file():
            raise RuntimeError("正式运行前必须先执行--mode calibrate。")
        amp_gate = json.loads(amp_gate_path.read_text(encoding="utf-8"))
        _validate_gate_identity(amp_gate, gpu_identity, "AMP")
        amp_enabled = bool(amp_gate.get("passed", False))
        parallelism = args.parallel or 1
        parallel_gate_path = (
            PROJECT_ROOT / "result" / "FEMNISTProbe" / "parallel_gate.json"
        )
        if not parallel_gate_path.is_file():
            raise RuntimeError("正式运行前必须先执行--mode benchmark。")
        parallel_gate = json.loads(
            parallel_gate_path.read_text(encoding="utf-8")
        )
        _validate_gate_identity(parallel_gate, gpu_identity, "性能")
        recommended_parallelism = int(
            parallel_gate.get("recommended_parallelism", 1)
        )
        if args.parallel == 2 and recommended_parallelism != 2:
            raise RuntimeError("性能门禁未通过，禁止强制双方案并行。")
        if (
                not compact_gpu
                and args.parallel is None
        ):
            parallelism = recommended_parallelism
        commands = [
            _build_command(
                config,
                5000,
                run_root,
                amp_enabled,
                "formal5000",
                args.gpu_id,
                batch_size=args.batch_size,
            )
            for config in configs
        ]
        elapsed, _ = _run_group(commands, log_root, parallelism)
        result = {
            "mode": "formal",
            "elapsed_seconds": elapsed,
            "amp_enabled": amp_enabled,
            "parallelism": parallelism,
            "batch_size": int(args.batch_size) if args.batch_size else None,
            "passed": True,
        }
        result.update(gpu_identity)

    (suite_root / "suite_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
