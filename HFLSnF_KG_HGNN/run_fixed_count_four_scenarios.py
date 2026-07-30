"""执行固定人数和MAT动态拓扑四组对照实验。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from .tasks.kge.fixed_count_four_scenarios import (
    DYNAMIC_SCENARIOS,
    PACKAGE_DIR,
    SCENARIOS,
    SMOKE_CONFIG,
    DynamicMatScenario,
    FixedCountScenario,
    dynamic_scenario_by_arm,
    load_flat_config,
    scenario_by_arm,
    scenario_from_config,
    validate_dynamic_mat_result,
    validate_fixed_count_result,
    validate_four_scenario_configs,
    validate_smoke_result,
    write_json_report,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建四组对照实验的命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "校验固定/动态四组配置、执行两轮CPU烟雾或150轮正式实验"
        )
    )
    parser.add_argument(
        "action",
        choices=("validate", "smoke", "formal150", "dynamic150"),
        help="配置校验、两轮CPU烟雾、固定人数或动态MAT正式实验",
    )
    parser.add_argument(
        "--arm",
        choices=(
            ("all",)
            + tuple(item.arm for item in SCENARIOS)
            + tuple(item.arm for item in DYNAMIC_SCENARIOS)
        ),
        default="all",
        help="根据正式实验类型选择全部或单个实验臂",
    )
    return parser


def _selected_scenarios(
    arm: str,
) -> Sequence[FixedCountScenario]:
    """根据all或单个实验臂名称返回需要执行的正式场景。"""

    if str(arm) == "all":
        return SCENARIOS
    return (scenario_by_arm(str(arm)),)


def _selected_dynamic_scenarios(
    arm: str,
) -> Sequence[DynamicMatScenario]:
    """根据all或单个名称返回需要执行的动态MAT场景。"""

    if str(arm) == "all":
        return DYNAMIC_SCENARIOS
    return (dynamic_scenario_by_arm(str(arm)),)


def _validate_action_arm(
    parser: argparse.ArgumentParser,
    action: str,
    arm: str,
) -> None:
    """校验实验动作与固定或动态实验臂名称是否匹配。"""

    if str(arm) == "all" or str(action) in {"validate", "smoke"}:
        return
    fixed_arms = {item.arm for item in SCENARIOS}
    dynamic_arms = {item.arm for item in DYNAMIC_SCENARIOS}
    if str(action) == "formal150" and str(arm) not in fixed_arms:
        parser.error(
            "formal150只能使用固定人数实验臂：{}".format(
                ", ".join(sorted(fixed_arms))
            )
        )
    if str(action) == "dynamic150" and str(arm) not in dynamic_arms:
        parser.error(
            "dynamic150只能使用动态MAT实验臂：{}".format(
                ", ".join(sorted(dynamic_arms))
            )
        )


def _discover_new_result(
    run_name: str,
    before: Sequence[Path],
) -> Path:
    """从运行前后目录差集中找到刚完成的唯一结果目录。"""

    result_root = PACKAGE_DIR / "results"
    before_set = {path.resolve() for path in before}
    after = {
        path.resolve()
        for path in result_root.glob("{}_*".format(run_name))
        if path.is_dir()
    }
    created = sorted(
        after - before_set,
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(created) != 1:
        raise RuntimeError(
            "期望新建1个结果目录，实际为{}：{}".format(
                len(created), created
            )
        )
    return created[0]


def _run_config(config_name: str) -> Path:
    """调用统一训练入口运行一份YAML并返回新结果目录。"""

    config_path = PACKAGE_DIR / "configs" / config_name
    config = load_flat_config(config_path)
    run_name = str(config["run_name"])
    before = tuple(
        path
        for path in (PACKAGE_DIR / "results").glob(
            "{}_*".format(run_name)
        )
        if path.is_dir()
    )
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v3.run_federated_transe",
        "--cf",
        str(config_path),
    ]
    # 子进程继承当前环境，工作目录固定为包目录的父目录。
    subprocess.run(
        command,
        cwd=str(PACKAGE_DIR.parent),
        check=True,
    )
    return _discover_new_result(run_name, before)


def _run_smoke() -> int:
    """执行唯一两轮CPU烟雾并校验端到端训练产物。"""

    print("开始两轮CPU烟雾测试")
    result_dir = _run_config(SMOKE_CONFIG)
    report = validate_smoke_result(result_dir)
    report_path = result_dir / "smoke_contract.json"
    write_json_report(report, report_path)
    print("烟雾测试状态：{}".format(report["status"]))
    print("烟雾测试结果：{}".format(result_dir))
    if report["status"] == "passed":
        _delete_passed_smoke_result(result_dir)
    return 0 if report["status"] == "passed" else 2


def _delete_passed_smoke_result(result_dir: Path) -> None:
    """安全删除已经通过合同校验的临时烟雾结果目录。"""

    result_root = (PACKAGE_DIR / "results").resolve()
    resolved_result = Path(result_dir).resolve()
    expected_prefix = "hflsnf_kg_v3_smoke_four_scenario_pipeline_cpu_"
    if (
        resolved_result.parent != result_root
        or not resolved_result.name.startswith(expected_prefix)
    ):
        raise RuntimeError(
            "拒绝删除非规范烟雾结果目录：{}".format(resolved_result)
        )
    shutil.rmtree(str(resolved_result))
    print("烟雾测试已通过，临时结果已自动删除")


def _run_formal150(
    scenarios: Sequence[FixedCountScenario],
) -> int:
    """直接运行所选四组150轮实验并校验正式结果合同。"""

    all_passed = True
    for template in scenarios:
        scenario = scenario_from_config(
            template, template.formal_config
        )
        print("开始150轮正式实验：{}".format(scenario.arm))
        result_dir = _run_config(scenario.formal_config)
        report = validate_fixed_count_result(
            result_dir,
            scenario,
            expected_rounds=150,
        )
        report_path = (
            result_dir / "fixed_count_formal150_contract.json"
        )
        write_json_report(report, report_path)
        print(
            "{}正式结果合同：{}".format(
                scenario.arm, report["status"]
            )
        )
        all_passed = all_passed and report["status"] == "passed"
        if report["status"] != "passed":
            # 正式结果合同失败时停止后续实验，避免继续占用GPU。
            break
    return 0 if all_passed else 3


def _run_dynamic150(
    scenarios: Sequence[DynamicMatScenario],
) -> int:
    """运行所选四组MAT原样回放150轮实验并校验结果。"""

    all_passed = True
    for scenario in scenarios:
        print("开始150轮动态MAT实验：{}".format(scenario.arm))
        result_dir = _run_config(scenario.formal_config)
        report = validate_dynamic_mat_result(
            result_dir,
            scenario,
            expected_rounds=150,
        )
        report_path = (
            result_dir / "dynamic_mat_formal150_contract.json"
        )
        write_json_report(report, report_path)
        print(
            "{}动态MAT结果合同：{}".format(
                scenario.arm, report["status"]
            )
        )
        all_passed = all_passed and report["status"] == "passed"
        if report["status"] != "passed":
            # 动态结果合同失败时停止后续实验，避免继续占用GPU。
            break
    return 0 if all_passed else 4


def main(argv: Optional[Sequence[str]] = None) -> None:
    """校验配置并执行烟雾、固定人数或动态MAT实验。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    _validate_action_arm(parser, args.action, args.arm)
    config_report = validate_four_scenario_configs()
    print("配置合同：{}".format(config_report["status"]))
    if config_report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        return
    if args.action == "smoke":
        exit_code = _run_smoke()
    elif args.action == "dynamic150":
        exit_code = _run_dynamic150(
            _selected_dynamic_scenarios(args.arm)
        )
    else:
        exit_code = _run_formal150(
            _selected_scenarios(args.arm)
        )
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
