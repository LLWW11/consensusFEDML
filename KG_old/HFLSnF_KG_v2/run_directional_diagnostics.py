"""从终端运行TransE最佳检查点的无需重训方向诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from .run_four_arm_ablation import discover_result_dirs
from .tasks.kge.data import load_fb15k237
from .tasks.kge.directional_diagnostics import (
    create_directional_result_dir,
    run_directional_diagnostics,
)
from .tasks.kge.factorial_ablation import FOUR_ARM_SPECS


PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve_package_path(path: Path) -> Path:
    """解析绝对、当前工作目录相对或V2包相对路径。"""

    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    working_directory_candidate = (Path.cwd() / path).resolve()
    if working_directory_candidate.exists():
        return working_directory_candidate
    return (PACKAGE_DIR / path).resolve()


def _parse_result_overrides(
    values: Optional[Sequence[str]],
) -> Dict[str, Path]:
    """解析可重复提供的`实验臂=结果目录`覆盖参数。"""

    allowed = {spec.arm for spec in FOUR_ARM_SPECS}
    parsed: Dict[str, Path] = {}
    for value in values or []:
        arm, separator, path_text = str(value).partition("=")
        arm = arm.strip()
        path_text = path_text.strip()
        if not separator or not arm or not path_text:
            raise ValueError("--result必须写成实验臂=结果目录")
        if arm not in allowed:
            raise ValueError(
                "未知实验臂{}；可选{}".format(
                    arm, "、".join(sorted(allowed))
                )
            )
        parsed[arm] = Path(path_text).expanduser().resolve()
    return parsed


def collect_result_dirs(
    result_root: Optional[Path],
    result_values: Optional[Sequence[str]],
) -> Dict[str, Path]:
    """合并自动发现结果和命令行显式指定结果，显式参数优先。"""

    allowed = [spec.arm for spec in FOUR_ARM_SPECS]
    result_dirs: Dict[str, Path] = {}
    if result_root is not None:
        result_dirs.update(
            discover_result_dirs(_resolve_package_path(result_root), allowed)
        )
    result_dirs.update(_parse_result_overrides(result_values))
    if len(result_dirs) < 2:
        raise ValueError(
            "至少需要两个实验臂结果；请检查--result-root或--result"
        )
    return {
        spec.arm: result_dirs[spec.arm]
        for spec in FOUR_ARM_SPECS
        if spec.arm in result_dirs
    }


def resolve_device(
    using_gpu: bool,
    gpu_id: int,
    require_cuda: bool,
) -> torch.device:
    """在读取数据前解析CPU或CUDA设备，正式配置无CUDA时快速失败。"""

    if bool(using_gpu):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "方向诊断要求GPU，但当前PyTorch检测不到CUDA。"
            )
        return torch.device("cuda:{}".format(int(gpu_id)))
    if bool(require_cuda):
        raise RuntimeError("require_cuda=true时禁止回退CPU")
    return torch.device("cpu")


def _write_config_snapshot(
    output_dir: Path,
    args: argparse.Namespace,
    result_dirs: Dict[str, Path],
) -> None:
    """把本次只读诊断参数和实际检查点目录写入JSON快照。"""

    payload = {
        "data_dir": str(_resolve_package_path(args.data_dir)),
        "result_dirs": {
            arm: str(path) for arm, path in result_dirs.items()
        },
        "using_gpu": bool(args.using_gpu),
        "gpu_id": int(args.gpu_id),
        "require_cuda": bool(args.require_cuda),
        "max_triples": int(args.max_triples),
        "selection_seed": int(args.selection_seed),
        "query_batch_size": int(args.query_batch_size),
        "candidate_batch_size": int(args.candidate_batch_size),
        "progress_every": int(args.progress_every),
    }
    with (Path(output_dir) / "diagnostic_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_argument_parser() -> argparse.ArgumentParser:
    """创建方向诊断的终端参数解析器。"""

    parser = argparse.ArgumentParser(
        description="读取最佳检查点，诊断头尾、逐关系和逐查询差异"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/FB15k-237"),
        help="FB15k-237目录，默认使用V2自带数据",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/三种"),
        help="自动发现A、B、C、D结果的目录",
    )
    parser.add_argument(
        "--result",
        action="append",
        help="额外或覆盖实验臂结果，格式为实验臂=结果目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；不填则在V2/results下创建时间戳目录",
    )
    parser.add_argument(
        "--using-gpu",
        action="store_true",
        help="使用指定CUDA设备",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="无CUDA时立即报错，禁止回退CPU",
    )
    parser.add_argument(
        "--max-triples",
        type=int,
        default=8,
        help="测试事实数量；0表示完整官方测试集，默认8用于CPU冒烟",
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument(
        "--candidate-batch-size", type=int, default=4096
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """解析参数并执行只读方向诊断，不修改任何模型检查点。"""

    args = build_argument_parser().parse_args(argv)
    if int(args.max_triples) < 0:
        raise ValueError("max_triples不能小于0")
    if int(args.query_batch_size) <= 0:
        raise ValueError("query_batch_size必须大于0")
    if int(args.candidate_batch_size) <= 0:
        raise ValueError("candidate_batch_size必须大于0")

    device = resolve_device(
        args.using_gpu,
        args.gpu_id,
        args.require_cuda,
    )
    result_dirs = collect_result_dirs(
        args.result_root,
        args.result,
    )
    data_dir = _resolve_package_path(args.data_dir)
    dataset = load_fb15k237(data_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else create_directional_result_dir(
            PACKAGE_DIR / "results",
            "directional_diagnostics",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(output_dir, args, result_dirs)
    summary = run_directional_diagnostics(
        dataset=dataset,
        result_dirs=result_dirs,
        output_dir=output_dir,
        device=device,
        max_triples=args.max_triples,
        selection_seed=args.selection_seed,
        query_batch_size=args.query_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        progress_every=args.progress_every,
    )
    print("无需重训方向诊断完成")
    print("结果目录：{}".format(output_dir))
    for arm, metrics in summary["model_metrics"].items():
        print(
            "{} 头MRR/尾MRR/综合MRR：{:.6f} / {:.6f} / {:.6f}".format(
                arm,
                float(metrics["head"]["mrr"]),
                float(metrics["tail"]["mrr"]),
                float(metrics["combined"]["mrr"]),
            )
        )


if __name__ == "__main__":
    main()
