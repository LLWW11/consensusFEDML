"""运行V5或冻结基线最佳检查点的完整FB15k-237官方测试。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch

from .tasks.kge import (
    load_fb15k237,
    run_best_checkpoint_official_evaluation,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def resolve_package_path(path_value: str) -> Path:
    """把相对路径解析为相对于V5包目录的绝对路径。"""

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PACKAGE_DIR / path).resolve()


def resolve_device(
    using_gpu: bool,
    gpu_id: int,
    require_cuda: bool,
) -> torch.device:
    """解析独立评估设备，不导入FedML训练运行时。"""

    if bool(using_gpu):
        if not torch.cuda.is_available():
            raise RuntimeError("完整官方测试要求GPU，但当前没有CUDA")
        return torch.device("cuda:{}".format(int(gpu_id)))
    if bool(require_cuda):
        raise RuntimeError("require_cuda=true时禁止回退CPU")
    return torch.device("cpu")


def build_argument_parser() -> argparse.ArgumentParser:
    """创建最佳检查点完整官方测试参数解析器。"""

    parser = argparse.ArgumentParser(
        description="对V5或冻结基线最佳检查点执行全部FB15k-237官方测试"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/FB15k-237"))
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="一个已通过训练合同的正式实验结果目录",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--using-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--candidate-batch-size", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """解析参数并执行不可降级为子集的完整官方测试。"""

    args = build_argument_parser().parse_args(argv)
    device = resolve_device(args.using_gpu, args.gpu_id, args.require_cuda)
    dataset = load_fb15k237(resolve_package_path(str(args.data_dir)))
    result_dir = resolve_package_path(str(args.result_dir))
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else result_dir / "full_official_evaluation"
    )
    contract = run_best_checkpoint_official_evaluation(
        dataset=dataset,
        result_dir=result_dir,
        output_dir=output_dir,
        device=device,
        query_batch_size=args.query_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        progress_every=args.progress_every,
    )
    print("完整官方测试完成：{}".format(output_dir))
    print(
        "头/尾/综合MRR：{:.6f} / {:.6f} / {:.6f}".format(
            float(contract["head_metrics"]["mrr"]),
            float(contract["tail_metrics"]["mrr"]),
            float(contract["combined_metrics"]["mrr"]),
        )
    )


if __name__ == "__main__":
    main()
