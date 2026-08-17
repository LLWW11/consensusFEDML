"""运行HFLKGE每轮客户端人数单因素消融实验。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    load_flat_config,
    write_json_report,
)
from .tasks.kge.hflkge_client_count_ablation import (
    SCENARIOS,
    HFLKGECountScenario,
    selected_scenarios,
    validate_configs,
    validate_result,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建配置校验和150轮正式实验的命令行解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "校验或运行HFLKGE的K=36、30、24、18、12、6随机客户端人数消融"
        )
    )
    parser.add_argument(
        "action",
        choices=("validate", "formal150"),
        help="只校验配置，或运行150轮正式实验",
    )
    parser.add_argument(
        "--arm",
        choices=("all",) + tuple(item.arm for item in SCENARIOS),
        default="all",
        help="运行全部人数，或只运行一个人数实验臂",
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
    """使用统一训练入口运行一份YAML并返回新结果目录。"""

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
    # 子进程复用当前Python环境，保证CUDA和依赖版本与入口一致。
    subprocess.run(
        command,
        cwd=str(PACKAGE_DIR.parent),
        check=True,
    )
    return _discover_new_result(run_name, before)


def _run_formal150(
    scenarios: Sequence[HFLKGECountScenario],
) -> int:
    """顺序运行所选人数并为每份正式结果写入合同校验报告。"""

    for scenario in scenarios:
        print("开始HFLKGE人数实验：{}".format(scenario.arm))
        result_dir = _run_config(scenario.formal_config)
        report = validate_result(result_dir, scenario, expected_rounds=150)
        write_json_report(
            report,
            result_dir / "hflkge_client_count_formal150_contract.json",
        )
        print(
            "{}结果合同：{}".format(scenario.arm, report["status"])
        )
        if report["status"] != "passed":
            # 一旦合同失败便停止，避免继续占用GPU并混入无效结果。
            return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """先校验单因素配置，再按请求执行一个或全部正式实验。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config_report = validate_configs()
    print("HFLKGE人数消融配置合同：{}".format(config_report["status"]))
    if config_report["status"] != "passed":
        raise SystemExit(1)
    if args.action == "validate":
        return
    exit_code = _run_formal150(selected_scenarios(args.arm))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
