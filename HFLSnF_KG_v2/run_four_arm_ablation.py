"""校验、补跑D臂并汇总V2同MAT四臂二乘二实验。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from .run_three_arm_ablation import (
    _run_one_arm,
    _write_json,
)
from .tasks.kge.factorial_ablation import (
    D_ARM_SPEC,
    FOUR_ARM_SPECS,
    compare_four_arm_results,
    validate_four_arm_configs,
    write_factorial_outputs,
)


def _package_dir() -> Path:
    """返回HFLSnF_KG_v2包目录的绝对路径。"""

    return Path(__file__).resolve().parent


def _timestamp() -> str:
    """生成不会覆盖旧结果的微秒级时间戳。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _require_cuda_for_d_run() -> None:
    """补跑D臂前快速检查CUDA，避免读取大数据后才失败。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "D臂正式配置要求CUDA，但当前Python环境检测不到GPU。"
            "本机请使用--action validate。"
        )


def _read_result_arm(result_dir: Path) -> Optional[str]:
    """从结果目录摘要中读取实验臂名称；非训练目录返回None。"""

    summary_path = Path(result_dir) / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    arm = str(payload.get("ablation_arm", "")).strip()
    return arm or None


def discover_result_dirs(
    result_root: Path,
    expected_arms: Sequence[str],
) -> Dict[str, Path]:
    """在结果根目录中按摘要发现每个实验臂最新的一次运行。"""

    result_root = Path(result_root).expanduser().resolve()
    if not result_root.is_dir():
        raise FileNotFoundError("找不到已有结果目录：{}".format(result_root))
    expected = {str(arm) for arm in expected_arms}
    candidates: Dict[str, list] = {arm: [] for arm in expected}
    for path in result_root.iterdir():
        if not path.is_dir():
            continue
        arm = _read_result_arm(path)
        if arm in candidates:
            candidates[arm].append(path.resolve())

    discovered: Dict[str, Path] = {}
    for arm, paths in candidates.items():
        if paths:
            # 多次运行时明确选最新结果，同时在终端打印实际选择。
            discovered[arm] = max(
                paths, key=lambda item: item.stat().st_mtime_ns
            )
    return discovered


def _parse_result_arguments(
    values: Optional[Sequence[str]],
) -> Dict[str, Path]:
    """解析四个`实验臂=结果目录`参数并检查名称完整。"""

    parsed: Dict[str, Path] = {}
    for value in values or []:
        arm, separator, path_text = str(value).partition("=")
        arm = arm.strip()
        path_text = path_text.strip()
        if not separator or not arm or not path_text:
            raise ValueError("--result必须写成实验臂=结果目录")
        if arm in parsed:
            raise ValueError("实验臂{}被重复提供".format(arm))
        parsed[arm] = Path(path_text).expanduser().resolve()
    expected = {spec.arm for spec in FOUR_ARM_SPECS}
    if set(parsed) != expected:
        raise ValueError(
            "汇总必须同时提供{}".format("、".join(sorted(expected)))
        )
    return parsed


def _print_contract(contract: Mapping[str, object]) -> None:
    """把四臂公平合同和D臂运行信息打印到终端。"""

    shared = contract["shared_contract"]
    print("四臂二乘二配置校验通过。")
    print("公平合同哈希：{}".format(contract["contract_hash"]))
    print("客户端划分哈希：{}".format(
        shared["expected_partition_hash"]
    ))
    print("MAT调度哈希：{}".format(
        shared["expected_topology_schedule_hash"]
    ))
    for arm in contract["arms"]:
        print(
            "- {label}：{aggregation_mode} + {local_objective}".format(
                **arm
            )
        )


def validate_action() -> Dict[str, object]:
    """只校验A、B、C、D配置和数据指纹，不启动训练。"""

    contract = validate_four_arm_configs(_package_dir())
    _print_contract(contract)
    return contract


def summarize_action(
    result_dirs: Mapping[str, Path],
    output_dir: Optional[Path] = None,
    mrr_threshold: float = 0.003,
) -> Path:
    """审计四个已有结果并写出二乘二比较报告。"""

    comparison = compare_four_arm_results(
        _package_dir(),
        result_dirs,
        mrr_threshold=float(mrr_threshold),
    )
    target = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            _package_dir()
            / "results"
            / "four_arm_comparison_{}".format(_timestamp())
        ).resolve()
    )
    written = write_factorial_outputs(target, comparison)
    print("四组结果可比，汇总目录：{}".format(target))
    print("大白话报告：{}".format(written["factorial_report"]))
    return target


def run_d_action(
    existing_root: Optional[Path] = None,
    mrr_threshold: float = 0.003,
) -> Path:
    """只补跑D臂，并在找到既有A、B、C时自动生成四臂报告。"""

    contract = validate_four_arm_configs(_package_dir())
    _print_contract(contract)
    _require_cuda_for_d_run()
    suite_dir = (
        _package_dir()
        / "results"
        / "four_arm_ablation_{}".format(_timestamp())
    ).resolve()
    suite_dir.mkdir(parents=True, exist_ok=False)
    _write_json(suite_dir / "factorial_contract.json", contract)
    status: Dict[str, object] = {
        "status": "running_d",
        "result_dirs": {},
    }
    _write_json(suite_dir / "suite_status.json", status)

    try:
        d_result = _run_one_arm(
            D_ARM_SPEC.arm,
            contract,
            suite_dir,
        )
        result_dirs: Dict[str, Path] = {
            D_ARM_SPEC.arm: d_result
        }
        if existing_root is not None:
            prior_arms = [
                spec.arm
                for spec in FOUR_ARM_SPECS
                if spec.arm != D_ARM_SPEC.arm
            ]
            discovered = discover_result_dirs(
                existing_root, prior_arms
            )
            result_dirs.update(discovered)
            for arm, path in sorted(discovered.items()):
                print("复用{}结果：{}".format(arm, path))

        status["result_dirs"] = {
            arm: str(path) for arm, path in result_dirs.items()
        }
        expected = {spec.arm for spec in FOUR_ARM_SPECS}
        if set(result_dirs) == expected:
            comparison = compare_four_arm_results(
                _package_dir(),
                result_dirs,
                mrr_threshold=float(mrr_threshold),
            )
            write_factorial_outputs(suite_dir, comparison)
            status["status"] = "completed_and_compared"
        else:
            missing = sorted(expected.difference(result_dirs))
            status["status"] = "completed_d_only"
            status["missing_for_comparison"] = missing
            print(
                "D臂已完成，但未自动汇总；缺少已有结果：{}".format(
                    "、".join(missing)
                )
            )
        _write_json(suite_dir / "suite_status.json", status)
    except BaseException as error:
        status["status"] = "failed"
        status["error"] = "{}: {}".format(
            type(error).__name__, error
        )
        _write_json(suite_dir / "suite_status.json", status)
        raise

    print("D臂任务完成，管理目录：{}".format(suite_dir))
    return suite_dir


def build_argument_parser() -> argparse.ArgumentParser:
    """创建四臂校验、补跑和汇总的命令行解析器。"""

    parser = argparse.ArgumentParser(
        description="V2同MAT四臂二乘二消融的一键入口"
    )
    parser.add_argument(
        "--action",
        choices=("validate", "run-d", "summarize"),
        default="validate",
        help="validate只检查；run-d只补D；summarize汇总四组结果",
    )
    parser.add_argument(
        "--existing-root",
        type=Path,
        help="run-d时存放已有A、B、C结果的目录",
    )
    parser.add_argument(
        "--result",
        action="append",
        help="summarize时填写实验臂=结果目录，共填写四次",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="summarize的输出目录；不填则自动创建",
    )
    parser.add_argument(
        "--mrr-threshold",
        type=float,
        default=0.003,
        help="描述性初筛阈值，默认0.003",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """执行四臂配置校验、D臂训练或已有结果汇总。"""

    args = build_argument_parser().parse_args(argv)
    if args.action == "validate":
        validate_action()
    elif args.action == "run-d":
        run_d_action(
            existing_root=args.existing_root,
            mrr_threshold=args.mrr_threshold,
        )
    else:
        summarize_action(
            _parse_result_arguments(args.result),
            output_dir=args.output_dir,
            mrr_threshold=args.mrr_threshold,
        )


if __name__ == "__main__":
    main()
