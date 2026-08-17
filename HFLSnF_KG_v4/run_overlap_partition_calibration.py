"""无训练地校准 FB15k-237 三个种子的低中高实体重叠档位。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .tasks.kge import (
    calibrate_entity_overlap_levels,
    load_fb15k237,
)


PACKAGE_ROOT = Path(__file__).resolve().parent


def build_argument_parser() -> argparse.ArgumentParser:
    """创建只执行数据划分校准、不触发模型训练的命令行解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "校准 balanced_head_entity_overlap_target 的低中高档位；"
            "本命令不会创建模型或启动 CUDA 训练。"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "FB15k-237",
        help="包含 train.txt、valid.txt 和 test.txt 的数据目录",
    )
    parser.add_argument(
        "--client-count",
        type=int,
        default=37,
        help="参与划分的客户端数量",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 2024, 2025],
        help="需要求共同可达区间的确定性随机种子",
    )
    parser.add_argument(
        "--overlap-tolerance",
        type=float,
        default=0.005,
        help="实际实体重叠率与目标值的最大绝对误差",
    )
    parser.add_argument(
        "--load-tolerance",
        type=float,
        default=0.05,
        help="客户端三元组数相对均值的最大偏差",
    )
    parser.add_argument(
        "--relation-overlap-tolerance",
        type=float,
        default=0.02,
        help="关系重叠率相对原始划分的最大绝对偏差",
    )
    parser.add_argument(
        "--search-restarts",
        type=int,
        default=8,
        help="每个目标执行的确定性贪心搜索次数",
    )
    parser.add_argument(
        "--minimum-overlap-span",
        type=float,
        default=0.06,
        help="共同可达高低档之间要求的最小间距",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选 JSON 输出路径；不指定时仅写入标准输出",
    )
    return parser


def run_calibration(args: argparse.Namespace) -> dict:
    """读取数据并运行三个种子的无训练重叠率校准。"""

    dataset = load_fb15k237(Path(args.data_dir))
    # 校准函数只操作训练三元组分区，不导入训练器和 CUDA 设备。
    return calibrate_entity_overlap_levels(
        dataset=dataset,
        client_count=int(args.client_count),
        seeds=tuple(int(seed) for seed in args.seeds),
        overlap_tolerance=float(args.overlap_tolerance),
        load_tolerance=float(args.load_tolerance),
        relation_overlap_tolerance=float(
            args.relation_overlap_tolerance
        ),
        search_restarts=int(args.search_restarts),
        minimum_overlap_span=float(args.minimum_overlap_span),
    )


def write_report(report: dict, output_path: Optional[Path]) -> None:
    """把校准报告写入标准输出，并按需保存为 UTF-8 JSON。"""

    serialized = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    print(serialized)
    if output_path is None:
        return
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(serialized + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析参数、执行校准并返回命令行退出码。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    report = run_calibration(args)
    write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
