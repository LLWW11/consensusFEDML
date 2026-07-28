"""运行阶段1单检查点完整头尾方向诊断。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch

from .runtime import PACKAGE_DIR, resolve_package_path
from .tasks.kge import load_fb15k237, run_directional_diagnostic


DEFAULT_C_RESULT = Path(
    "../HFLSnF_KG_v2/results/三种/"
    "hflsnf_kg_v2_three_arm_masked_fede_fair_cuda_"
    "20260727_162802_515404"
)


def resolve_device(
    using_gpu: bool,
    gpu_id: int,
    require_cuda: bool,
) -> torch.device:
    """在读取数据和检查点前解析诊断设备。"""

    if bool(using_gpu):
        if not torch.cuda.is_available():
            raise RuntimeError("方向诊断要求GPU，但当前没有CUDA")
        return torch.device("cuda:{}".format(int(gpu_id)))
    if bool(require_cuda):
        raise RuntimeError("require_cuda=true时禁止回退CPU")
    return torch.device("cpu")


def build_argument_parser() -> argparse.ArgumentParser:
    """创建阶段1方向诊断命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="只读评估V2 C臂或V3检查点的头尾排名"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/FB15k-237"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_C_RESULT,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--using-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--max-triples",
        type=int,
        default=8,
        help="0表示完整官方测试集，默认8用于CPU链路验证",
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=4096,
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--distance-norm", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """解析参数并执行不修改模型的头尾方向诊断。"""

    args = build_argument_parser().parse_args(argv)
    if int(args.max_triples) < 0:
        raise ValueError("max_triples不能小于0")
    device = resolve_device(
        args.using_gpu,
        args.gpu_id,
        args.require_cuda,
    )
    dataset = load_fb15k237(
        resolve_package_path(str(args.data_dir))
    )
    checkpoint = resolve_package_path(str(args.checkpoint))
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else PACKAGE_DIR
        / "results"
        / "directional_{}".format(
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    )
    summary = run_directional_diagnostic(
        dataset=dataset,
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        max_triples=args.max_triples,
        selection_seed=args.selection_seed,
        query_batch_size=args.query_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        progress_every=args.progress_every,
        distance_norm_override=args.distance_norm,
    )
    print("方向诊断完成")
    print("结果目录：{}".format(output_dir))
    print(
        "头/尾/综合MRR：{:.6f} / {:.6f} / {:.6f}".format(
            float(summary["head_metrics"]["mrr"]),
            float(summary["tail_metrics"]["mrr"]),
            float(summary["combined_metrics"]["mrr"]),
        )
    )


if __name__ == "__main__":
    main()
