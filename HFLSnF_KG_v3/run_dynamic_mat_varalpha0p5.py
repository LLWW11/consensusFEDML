"""校验并运行varAlpha=0.5动态MAT正式实验。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from .tasks.kge.dynamic_mat_varalpha0p5 import (
    SCENARIOS,
    VarAlpha05Scenario,
    selected_scenarios,
    validate_configs,
    validate_result,
)
from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    load_flat_config,
    write_json_report,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建0p5配置校验和150轮正式实验的命令行解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "校验或运行varAlpha=0.5动态MAT的四种联邦TransE场景"
        )
    )
    parser.add_argument(
        "action",
        choices=("validate", "dynamic150"),
        help="只校验配置，或运行150轮正式实验",
    )
    parser.add_argument(
        "--arm",
        choices=("all",) + tuple(item.arm for item in SCENARIOS),
        default="hflsnf",
        help="默认只运行HFLSnF；使用all可顺序运行四个场景",
    )
    return parser


def _discover_new_result(run_name: str, before: Sequence[Path]) -> Path:
    """从运行前后的目录差集中定位唯一的新结果目录。"""

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
    """使用统一训练入口运行一份0p5 YAML并返回结果目录。"""

    config_path = PACKAGE_DIR / "configs" / config_name
    config = load_flat_config(config_path)
    run_name = str(config["run_name"])
    result_root = PACKAGE_DIR / "results"
    before = tuple(
        path
        for path in result_root.glob("{}_*".format(run_name))
        if path.is_dir()
    )
    command = [
        sys.executable,
        "-m",
        "HFLSnF_KG_v3.run_federated_transe",
        "--cf",
        str(config_path),
    ]
    # 子进程继承服务器当前Python与CUDA环境。
    subprocess.run(
        command,
        cwd=str(PACKAGE_DIR.parent),
        check=True,
    )
    return _discover_new_result(run_name, before)


def _run_dynamic150(
    scenarios: Sequence[VarAlpha05Scenario],
) -> int:
    """顺序运行所选0p5场景并校验每份正式结果。"""

    for scenario in scenarios:
        print("开始varAlpha=0.5动态实验：{}".format(scenario.arm))
        result_dir = _run_config(scenario.formal_config)
        report = validate_result(result_dir, scenario, expected_rounds=150)
        write_json_report(
            report,
            result_dir / "dynamic_mat_varalpha0p5_formal150_contract.json",
        )
        print(
            "{}结果合同：{}".format(scenario.arm, report["status"])
        )
        if report["status"] != "passed":
            # 合同失败时停止后续场景，防止无效实验继续占用GPU。
            return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """先校验0p5配置和MAT调度，再按请求运行正式实验。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config_report = validate_configs()
    print("varAlpha=0.5动态MAT配置合同：{}".format(config_report["status"]))
    if config_report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        return
    exit_code = _run_dynamic150(selected_scenarios(args.arm))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
