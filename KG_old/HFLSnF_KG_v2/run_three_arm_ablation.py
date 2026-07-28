"""校验、运行并汇总V2同MAT三臂TransE消融实验。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .tasks.kge.ablation import (
    THREE_ARM_SPECS,
    compare_three_arm_results,
    validate_three_arm_configs,
    write_comparison_outputs,
)


def _package_dir() -> Path:
    """返回HFLSnF_KG_v2包目录的绝对路径。"""

    return Path(__file__).resolve().parent


def _project_root() -> Path:
    """返回包含HFLSnF_KG_v2包的项目根目录。"""

    return _package_dir().parent


def _timestamp() -> str:
    """生成适合结果目录名称的微秒级时间戳。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """把字典以可读的UTF-8 JSON写入指定路径。"""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def _arm_names(values: Optional[Sequence[str]]) -> List[str]:
    """把命令行实验臂参数整理成固定的A、B、C执行顺序。"""

    requested = list(values or ["all"])
    if "all" in requested:
        requested = [spec.arm for spec in THREE_ARM_SPECS]
    allowed = {spec.arm for spec in THREE_ARM_SPECS}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(
            "未知实验臂：{}；可选值为{}".format(
                ", ".join(unknown),
                "、".join(sorted(allowed)),
            )
        )
    # 去重后仍按A、B、C顺序执行，避免参数书写顺序影响实验管理。
    return [
        spec.arm for spec in THREE_ARM_SPECS if spec.arm in requested
    ]


def _contract_arm_map(
    contract: Mapping[str, object],
) -> Dict[str, Mapping[str, object]]:
    """把公平合同中的实验臂列表转换成按名称索引的字典。"""

    arms = contract.get("arms", [])
    if not isinstance(arms, list):
        raise TypeError("公平合同中的arms必须是列表")
    return {
        str(item["arm"]): item
        for item in arms
        if isinstance(item, dict)
    }


def _existing_result_directories(
    result_root: Path,
    run_name: str,
) -> set:
    """记录一次训练开始前已经存在的同名前缀结果目录。"""

    result_root = Path(result_root).expanduser().resolve()
    if not result_root.is_dir():
        return set()
    return {
        path.resolve()
        for path in result_root.glob("{}_*".format(run_name))
        if path.is_dir()
    }


def _result_root(
    contract: Mapping[str, object],
) -> Path:
    """按照公平合同解析单个训练任务实际使用的结果根目录。"""

    shared = contract.get("shared_contract", {})
    if not isinstance(shared, dict):
        raise TypeError("公平合同中的shared_contract必须是字典")
    path = Path(str(shared.get("result_root", "results"))).expanduser()
    if not path.is_absolute():
        path = _package_dir() / path
    return path.resolve()


def _find_new_result_directory(
    result_root: Path,
    run_name: str,
    before: Iterable[Path],
) -> Path:
    """从训练前后目录差集中找出本次生成的唯一结果目录。"""

    before_set = {Path(path).resolve() for path in before}
    candidates = sorted(
        (
            path.resolve()
            for path in Path(result_root).glob(
                "{}_*".format(run_name)
            )
            if path.is_dir() and path.resolve() not in before_set
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "无法唯一定位{}本次结果目录，新目录数量为{}".format(
                run_name, len(candidates)
            )
        )
    return candidates[0]


def _require_cuda_for_formal_run() -> None:
    """正式三臂训练前快速检查CUDA，避免读完数据后才失败。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "三臂正式配置要求CUDA，但当前Python环境检测不到GPU。"
            "请在服务器CUDA环境运行；本机可先使用--action validate。"
        )


def _run_one_arm(
    arm: str,
    contract: Mapping[str, object],
    suite_dir: Path,
) -> Path:
    """在独立子进程中运行一个实验臂并返回它的新结果目录。"""

    arm_map = _contract_arm_map(contract)
    arm_contract = arm_map[arm]
    config_path = Path(str(arm_contract["config_path"])).resolve()
    run_name = str(arm_contract["run_name"])
    result_root = _result_root(contract)
    before = _existing_result_directories(result_root, run_name)
    log_path = Path(suite_dir) / "{}.log".format(arm)
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v2.run_dynamic_federated_transe",
        "--cf",
        str(config_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    print("\n开始运行{}，日志：{}".format(arm, log_path), flush=True)
    print("命令：{}".format(" ".join(command)), flush=True)

    # 标准输出一边显示、一边落盘，服务器终端可以实时观察每轮进度。
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(_project_root()),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            # 用户在服务器终端中止总任务时，同时结束当前训练子进程。
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code != 0:
        raise RuntimeError(
            "{}训练失败，退出码为{}；请查看{}".format(
                arm, return_code, log_path
            )
        )
    return _find_new_result_directory(result_root, run_name, before)


def _parse_result_arguments(
    values: Optional[Sequence[str]],
) -> Dict[str, Path]:
    """解析`实验臂=结果目录`形式的三组汇总参数。"""

    result_dirs: Dict[str, Path] = {}
    for value in values or []:
        arm, separator, path_text = str(value).partition("=")
        arm = arm.strip()
        path_text = path_text.strip()
        if not separator or not arm or not path_text:
            raise ValueError(
                "--result必须写成实验臂=结果目录，例如"
                "dense_margin=D:\\results\\run"
            )
        if arm in result_dirs:
            raise ValueError("实验臂{}重复提供了结果目录".format(arm))
        result_dirs[arm] = Path(path_text).expanduser().resolve()
    expected = {spec.arm for spec in THREE_ARM_SPECS}
    if set(result_dirs) != expected:
        raise ValueError(
            "汇总必须同时提供{}三组结果".format(
                "、".join(sorted(expected))
            )
        )
    return result_dirs


def _print_contract(contract: Mapping[str, object]) -> None:
    """把公平合同的关键信息用大白话打印到终端。"""

    shared = contract["shared_contract"]
    print("三臂配置校验通过。")
    print("公平合同哈希：{}".format(contract["contract_hash"]))
    print("客户端划分哈希：{}".format(
        shared["expected_partition_hash"]
    ))
    print("MAT调度哈希：{}".format(
        shared["expected_topology_schedule_hash"]
    ))
    for arm in contract["arms"]:
        print(
            "- {arm}: {aggregation_mode} + {local_objective}，配置={config_path}".format(
                **arm
            )
        )


def validate_action() -> Dict[str, object]:
    """只检查A、B、C配置和数据指纹，不启动任何训练。"""

    contract = validate_three_arm_configs(_package_dir())
    _print_contract(contract)
    return contract


def run_action(
    arms: Optional[Sequence[str]] = None,
    mrr_threshold: float = 0.003,
) -> Path:
    """按固定顺序运行所选实验臂，三臂齐全时自动生成比较报告。"""

    contract = validate_three_arm_configs(_package_dir())
    selected_arms = _arm_names(arms)
    _print_contract(contract)
    _require_cuda_for_formal_run()

    suite_dir = (
        _package_dir()
        / "results"
        / "three_arm_ablation_{}".format(_timestamp())
    ).resolve()
    suite_dir.mkdir(parents=True, exist_ok=False)
    _write_json(suite_dir / "ablation_contract.json", contract)
    status: Dict[str, object] = {
        "status": "running",
        "selected_arms": selected_arms,
        "result_dirs": {},
    }
    _write_json(suite_dir / "suite_status.json", status)

    try:
        result_dirs: Dict[str, Path] = {}
        for arm in selected_arms:
            result_dirs[arm] = _run_one_arm(
                arm, contract, suite_dir
            )
            status["result_dirs"] = {
                key: str(value)
                for key, value in result_dirs.items()
            }
            _write_json(suite_dir / "suite_status.json", status)

        if len(result_dirs) == len(THREE_ARM_SPECS):
            comparison = compare_three_arm_results(
                _package_dir(),
                result_dirs,
                mrr_threshold=mrr_threshold,
            )
            write_comparison_outputs(suite_dir, comparison)
            status["status"] = "completed_and_compared"
        else:
            status["status"] = "completed_partial"
        _write_json(suite_dir / "suite_status.json", status)
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = "{}: {}".format(
            type(error).__name__, error
        )
        _write_json(suite_dir / "suite_status.json", status)
        raise

    print("\n三臂任务完成，管理目录：{}".format(suite_dir))
    return suite_dir


def summarize_action(
    result_values: Optional[Sequence[str]],
    output_dir: Optional[Path],
    mrr_threshold: float = 0.003,
) -> Path:
    """对三次已经完成的训练做公平性审计并生成统一报告。"""

    result_dirs = _parse_result_arguments(result_values)
    comparison = compare_three_arm_results(
        _package_dir(),
        result_dirs,
        mrr_threshold=mrr_threshold,
    )
    target = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            _package_dir()
            / "results"
            / "three_arm_comparison_{}".format(_timestamp())
        ).resolve()
    )
    written = write_comparison_outputs(target, comparison)
    print("三组结果可比，汇总目录：{}".format(target))
    print("大白话报告：{}".format(written["comparison_report"]))
    return target


def build_argument_parser() -> argparse.ArgumentParser:
    """创建三臂消融命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="V2同MAT三臂TransE消融的一键校验、训练和汇总入口"
    )
    parser.add_argument(
        "--action",
        choices=("validate", "run", "summarize"),
        default="validate",
        help="validate只检查；run训练；summarize汇总已有结果",
    )
    parser.add_argument(
        "--arm",
        action="append",
        choices=tuple(spec.arm for spec in THREE_ARM_SPECS)
        + ("all",),
        help="run时选择实验臂；不填或填all表示按A、B、C顺序全跑",
    )
    parser.add_argument(
        "--result",
        action="append",
        help="summarize时填写实验臂=结果目录，共填写三次",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="summarize输出目录；不填则自动创建时间戳目录",
    )
    parser.add_argument(
        "--mrr-threshold",
        type=float,
        default=0.003,
        help="决定是否继续补D臂的MRR初筛阈值，默认0.003",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """执行用户选择的三臂校验、正式训练或结果汇总动作。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.action == "validate":
        validate_action()
    elif args.action == "run":
        run_action(args.arm, mrr_threshold=args.mrr_threshold)
    else:
        summarize_action(
            args.result,
            args.output_dir,
            mrr_threshold=args.mrr_threshold,
        )


if __name__ == "__main__":
    main()
