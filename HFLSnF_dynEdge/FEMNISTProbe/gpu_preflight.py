"""在4090D或4060 Laptop服务器上执行CUDA与AMP快速自检。"""

from __future__ import absolute_import

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn


def parse_arguments():
    """解析GPU编号和结果JSON路径。"""
    parser = argparse.ArgumentParser(description="检查FEMNIST实验GPU环境。")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="result/FEMNISTProbe/gpu_preflight.json",
    )
    return parser.parse_args()


def run_preflight(gpu_id):
    """执行卷积、FP32和AMP训练步并返回环境与耗时信息。"""
    if sys.version_info[:2] != (3, 7):
        raise RuntimeError("服务器兼容环境要求Python 3.7。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前PyTorch无法访问CUDA。")
    device = torch.device("cuda:{}".format(int(gpu_id)))
    properties = torch.cuda.get_device_properties(device)
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 26 * 26, 62),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    inputs = torch.rand(20, 1, 28, 28, device=device)
    labels = torch.randint(0, 62, (20,), device=device)

    timings = {}
    for mode_name, amp_enabled in [("fp32", False), ("fp16_amp", True)]:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = nn.functional.cross_entropy(model(inputs), labels)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize(device)
        timings[mode_name + "_20_steps_seconds"] = (
            time.perf_counter() - start
        )
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("GPU预检后模型参数包含NaN或无穷值。")
    return {
        "status": "passed",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_id": int(gpu_id),
        "gpu_name": properties.name,
        "gpu_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "timings": timings,
    }


def main():
    """运行预检并写入JSON。"""
    args = parse_arguments()
    result = run_preflight(args.gpu_id)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
