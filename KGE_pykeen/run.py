"""PyKEEN双口径TransE工程的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .configuration import PACKAGE_DIR
from .experiment import run_experiment


DEFAULT_CONFIG = (
    PACKAGE_DIR
    / "configs"
    / "matched_fb15k237_seed42_cuda.yaml"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建集中式TransE命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="运行PyKEEN双口径TransE知识图谱补全实验"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="集中式实验YAML，默认使用正式CUDA强配置",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """解析配置、运行集中式实验并打印主要结果。"""

    args = build_argument_parser().parse_args(argv)
    result_dir, summary = run_experiment(args.config)
    test_metrics = summary["final_test_metrics"]
    print("PyKEEN TransE训练完成")
    print("结果目录：{}".format(result_dir))
    print("最佳epoch：{}".format(summary["best_epoch"]))
    print("完整测试MRR：{:.6f}".format(float(test_metrics["mrr"])))
    print(
        "完整测试Hits@10：{:.6f}".format(
            float(test_metrics["hits_at_10"])
        )
    )


if __name__ == "__main__":
    main()
