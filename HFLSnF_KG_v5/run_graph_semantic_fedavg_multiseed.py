"""独立运行三种子乘三拓扑的FedAvg九实验，固定本地学习率5e-4。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import run_graph_semantic_fedavg_lr_sweep as monitored
from .tasks.kge import graph_semantic_fedavg_multiseed as contract


def experiment_definition() -> monitored.ExperimentDefinition:
    """绑定独立九实验配置、输出、合同和跨种子拓扑报告。"""

    return monitored.ExperimentDefinition(
        suite=contract.SUITE_NAME, result_root=contract.RESULT_ROOT,
        scenarios=contract.scenarios, expected_config=contract.expected_flat_config,
        validate_result=contract.validate_result,
        contract_file="fedavg_multiseed_lr5e-4_formal150_contract.json",
        batch_prefix="multiseed_topology_batch_", report_stem="multiseed_topology",
        report_title="FedAvg三种子三拓扑九实验报告",
        report_description="固定FedAvg和本地学习率5e-4，种子42、2024、2025各自独立重跑三拓扑150轮。",
        topology_report=True,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """提供九实验校验、正式训练、恢复和报告命令。"""

    parser = argparse.ArgumentParser(description="FedAvg本地学习率5e-4的三种子三拓扑九实验")
    parser.add_argument("action", choices=("validate", "formal9", "report"))
    parser.add_argument("--resume", type=Path, help="formal9恢复使用的batch_summary.json")
    parser.add_argument("--batch", type=Path, help="report使用的batch_summary.json")
    parser.add_argument("--skip-partition-recompute", action="store_true",
                        help="仅validate可用；正式训练必须复算三个种子的真实分区")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """真实校验后顺序完成九组，不复用已有三组或扫描结果。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.resume is not None and args.action != "formal9":
        parser.error("--resume仅用于formal9")
    if args.batch is not None and args.action != "report":
        parser.error("--batch仅用于report")
    if args.skip_partition_recompute and args.action != "validate":
        parser.error("只有validate允许跳过分区复算")
    definition = experiment_definition()
    if args.action == "report":
        if args.batch is None:
            parser.error("report需要--batch")
        monitored.generate_report(args.batch, definition)
        return 0
    if args.action == "formal9":
        monitored.check_formal_environment()
    report = contract.validate_configs(recompute_partitions=not args.skip_partition_recompute)
    print("三种子九实验配置合同：{}；真实分区复算：{}".format(
        report["status"], report["recompute_partitions"]), flush=True)
    if report["status"] != "passed":
        for check in report["checks"]:
            if not check["passed"]:
                print("失败项：{}".format(check["name"]), flush=True)
        return 1
    if args.action == "validate":
        for seed, digest in report["partition_hashes"].items():
            print("seed{}分区哈希：{}".format(seed, digest))
        for arm, digest in report["topology_hashes"].items():
            print("{}调度哈希：{}".format(arm, digest))
        return 0
    manifest = args.resume.resolve() if args.resume is not None else monitored.create_batch_manifest(report, definition)
    print("批次清单：{}".format(manifest), flush=True)
    return monitored.run_training_batch(manifest, definition)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("三种子九实验入口失败：{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(1)
