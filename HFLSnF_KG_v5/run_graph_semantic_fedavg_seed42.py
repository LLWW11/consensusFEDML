"""独立运行本地学习率5e-4、种子42的FedAvg三拓扑实验。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import run_graph_semantic_fedavg_lr_sweep as monitored
from .tasks.kge import graph_semantic_fedavg_seed42 as contract


def experiment_definition() -> monitored.ExperimentDefinition:
    """为已有监测引擎提供三拓扑合同和独立输出位置。"""

    return monitored.ExperimentDefinition(
        suite=contract.SUITE_NAME, result_root=contract.RESULT_ROOT,
        scenarios=contract.scenarios, expected_config=contract.expected_flat_config,
        validate_result=contract.validate_result,
        contract_file="fedavg_seed42_lr5e-4_formal150_contract.json",
        batch_prefix="seed42_topology_batch_", report_stem="seed42_topology",
        report_title="FedAvg种子42三拓扑实验报告",
        report_description="固定FedAvg、本地学习率5e-4和种子42，三拓扑从相同初始化独立重跑150轮。",
        topology_report=True,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """提供三拓扑校验、正式训练、批次恢复和报告命令。"""

    parser = argparse.ArgumentParser(description="FedAvg本地学习率5e-4的seed42三拓扑实验")
    parser.add_argument("action", choices=("validate", "formal3", "report"))
    parser.add_argument("--resume", type=Path, help="formal3恢复使用的batch_summary.json")
    parser.add_argument("--batch", type=Path, help="report使用的batch_summary.json")
    parser.add_argument("--skip-partition-recompute", action="store_true",
                        help="仅validate可用；正式训练必须重算真实分区")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """先校验正式环境和真实合同，再逐个独立训练；不会启动其他种子。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.resume is not None and args.action != "formal3":
        parser.error("--resume仅用于formal3")
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
    if args.action == "formal3":
        monitored.check_formal_environment()
    report = contract.validate_configs(recompute_partitions=not args.skip_partition_recompute)
    print("seed42三拓扑配置合同：{}；真实分区复算：{}".format(
        report["status"], report["recompute_partitions"]), flush=True)
    if report["status"] != "passed":
        for check in report["checks"]:
            if not check["passed"]:
                print("失败项：{}".format(check["name"]), flush=True)
        return 1
    if args.action == "validate":
        print("共同分区哈希：{}".format(report["partition_hash"]))
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
        print("seed42三拓扑入口失败：{}: {}".format(type(error).__name__, error), file=sys.stderr)
        sys.exit(1)
