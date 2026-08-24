"""无训练地校准 FB15k-237 三个种子的低中高实体重叠档位。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from .tasks.kge import (
    BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
    calibrate_entity_overlap_levels,
    load_fb15k237,
    partition_train_triples_by_overlap_target,
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
    parser.add_argument(
        "--verification-source",
        type=Path,
        default=None,
        help="可选的首次校准报告；提供后要求目标值和九个哈希完全复现",
    )
    parser.add_argument(
        "--skip-reproduction-verification",
        action="store_true",
        help="仅供开发端点探测；正式合同禁止跳过九个分区独立复算",
    )
    return parser


def _file_sha256(path: Path) -> str:
    """分块计算数据文件的SHA-256，避免一次性读入大文件。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reproduction_payload(report: dict) -> dict:
    """提取必须跨两次正式校准完全一致的目标值和分区哈希。"""

    return {
        "common_reachable_interval": report["common_reachable_interval"],
        "targets": {
            level_name: level["target_entity_overlap"]
            for level_name, level in report["levels"].items()
        },
        "partition_hashes": {
            level_name: {
                seed: summary["partition_hash"]
                for seed, summary in level["per_seed"].items()
            }
            for level_name, level in report["levels"].items()
        },
    }


def _load_verification_source(path: Path) -> dict:
    """读取首次校准报告并校验其顶层结构。"""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("找不到首次校准报告：{}".format(resolved))
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("首次校准报告顶层必须是对象")
    return payload


def _verify_source_partitions(
    dataset,
    source_report: dict,
    args: argparse.Namespace,
) -> dict:
    """按首次报告的三个目标独立重算九个分区并核对哈希。"""

    source_constraints = source_report.get("constraints")
    if not isinstance(source_constraints, dict):
        raise TypeError("首次校准报告缺少constraints")
    expected_arguments = {
        "client_count": int(args.client_count),
        "overlap_tolerance": float(args.overlap_tolerance),
        "load_tolerance": float(args.load_tolerance),
        "relation_overlap_tolerance": float(
            args.relation_overlap_tolerance
        ),
        "search_restarts": int(args.search_restarts),
        "minimum_overlap_span": float(args.minimum_overlap_span),
    }
    if int(source_report.get("client_count", -1)) != expected_arguments[
        "client_count"
    ]:
        raise ValueError("复现客户端数量与首次校准不一致")
    for field in (
        "overlap_tolerance",
        "load_tolerance",
        "relation_overlap_tolerance",
        "search_restarts",
        "minimum_overlap_span",
    ):
        if float(source_constraints.get(field, float("nan"))) != float(
            expected_arguments[field]
        ):
            raise ValueError("复现参数{}与首次校准不一致".format(field))
    levels = source_report.get("levels")
    if not isinstance(levels, dict):
        raise TypeError("首次校准报告缺少levels")
    for level_name in ("low", "medium", "high"):
        level = levels.get(level_name)
        if not isinstance(level, dict):
            raise KeyError("首次校准报告缺少{}档".format(level_name))
        target = float(level["target_entity_overlap"])
        per_seed = level.get("per_seed")
        if not isinstance(per_seed, dict):
            raise TypeError("首次校准档位per_seed必须是对象")
        for seed in tuple(int(value) for value in args.seeds):
            reproduced = partition_train_triples_by_overlap_target(
                dataset=dataset,
                client_count=int(args.client_count),
                seed=seed,
                target_entity_overlap=target,
                overlap_tolerance=float(args.overlap_tolerance),
                load_tolerance=float(args.load_tolerance),
                relation_overlap_tolerance=float(
                    args.relation_overlap_tolerance
                ),
                search_restarts=int(args.search_restarts),
                search_seed=seed,
                strict=True,
            )
            expected_hash = str(per_seed[str(seed)]["partition_hash"])
            if reproduced.partition_hash != expected_hash:
                raise RuntimeError(
                    "{}档seed{}分区哈希复现失败".format(
                        level_name, seed
                    )
                )
    # 通过JSON往返构造深拷贝，避免修改首次报告对象。
    return json.loads(json.dumps(source_report, ensure_ascii=False))


def run_calibration(args: argparse.Namespace) -> dict:
    """读取数据并运行三个种子的无训练重叠率校准。"""

    data_dir = Path(args.data_dir).expanduser().resolve()
    dataset = load_fb15k237(data_dir)
    verification_source = getattr(args, "verification_source", None)
    if verification_source is None:
        # 首次校准搜索共同区间和三个正式目标，不导入训练器或CUDA。
        report = calibrate_entity_overlap_levels(
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
    else:
        source_report = _load_verification_source(verification_source)
        report = _verify_source_partitions(dataset, source_report, args)
    report.update(
        {
            "contract_schema_version": 1,
            "partition_strategy": BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
            "data_files": {
                name: {
                    "relative_path": "data/FB15k-237/{}".format(name),
                    "sha256": _file_sha256(data_dir / name),
                }
                for name in ("train.txt", "valid.txt", "test.txt")
            },
            "dataset_summary": dataset.summary(),
            "search_seed_policy": "experiment_seed",
        }
    )
    if verification_source is None and bool(
        getattr(args, "skip_reproduction_verification", False)
    ):
        report["reproduction_verification"] = {
            "status": "not_requested",
            "source_report_sha256": None,
        }
        return report
    if verification_source is None:
        source_payload = _reproduction_payload(report)
        report = _verify_source_partitions(dataset, report, args)
        if source_payload != _reproduction_payload(report):
            raise RuntimeError("独立复算前后的正式目标或九个哈希不一致")
        report["reproduction_verification"] = {
            "status": "passed",
            "verification_mode": "independent_in_process_recompute",
            "source_report_sha256": None,
        }
        return report
    source_path = Path(verification_source).expanduser().resolve()
    source_report = _load_verification_source(source_path)
    if _reproduction_payload(source_report) != _reproduction_payload(report):
        raise RuntimeError("复现报告的目标值或九个分区哈希不一致")
    report["reproduction_verification"] = {
        "status": "passed",
        "verification_mode": "independent_source_report_recompute",
        "source_report_sha256": _file_sha256(source_path),
    }
    return report


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
