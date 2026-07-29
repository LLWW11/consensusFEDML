"""SevereTest 训练与分析的一键入口。"""

from __future__ import absolute_import

import logging
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fedml  # noqa: E402
from fedml.constants import (  # noqa: E402
    FEDML_SIMULATION_TYPE_SP,
    FEDML_TRAINING_PLATFORM_SIMULATION,
)

import init_test  # noqa: E402
from SevereTest.analyze_result import analyze_result  # noqa: E402
from SevereTest.data_partition import load_severe_mnist_data  # noqa: E402
from SevereTest.trainer import (  # noqa: E402
    SevereFixedFedAvg,
    SevereHierarchicalFedAvg,
)


def select_trainer_class(args):
    """根据 aggregation_mode 选择普通或三边缘组分层训练器。"""
    aggregation_mode = str(
        getattr(args, "aggregation_mode", "fedavg")
    ).strip().lower()
    if aggregation_mode == "fedavg":
        return SevereFixedFedAvg
    if aggregation_mode == "hierarchical_fedavg":
        return SevereHierarchicalFedAvg
    raise ValueError(
        "aggregation_mode 仅支持 fedavg 或 hierarchical_fedavg。"
    )


def validate_requested_device(args, device):
    """确保请求 GPU 时 CUDA 可用且 FedML 实际选择了目标 CUDA 设备。"""
    if not bool(getattr(args, "using_gpu", False)):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "配置要求使用 GPU，但当前 PyTorch 无法访问 CUDA。"
            "请安装 CUDA 版 PyTorch，并检查 NVIDIA 驱动和 CUDA_VISIBLE_DEVICES。"
        )
    gpu_id = int(getattr(args, "gpu_id", 0))
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            "gpu_id={} 超出当前可见 GPU 范围 0–{}。".format(
                gpu_id, torch.cuda.device_count() - 1
            )
        )
    if getattr(device, "type", str(device).split(":")[0]) != "cuda":
        raise RuntimeError(
            "配置要求使用 GPU，但 FedML 实际选择的设备是 {}。".format(device)
        )
    logging.info(
        "SevereTest 使用 GPU：device=%s, name=%s",
        device,
        torch.cuda.get_device_name(gpu_id),
    )


def main():
    """读取 YAML、运行所选 SevereTest 实验并按配置自动分析结果。"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
    fedml._global_comm_backend = FEDML_SIMULATION_TYPE_SP
    args = init_test.init()
    device = fedml.device.get_device(args)
    validate_requested_device(args, device)
    dataset, output_dim, partition_manifest = load_severe_mnist_data(args)
    model = fedml.model.create(args, output_dim)
    trainer_class = select_trainer_class(args)
    trainer = trainer_class(
        args, device, dataset, model, partition_manifest
    )
    result_dir = trainer.train()
    logging.info("SevereTest 训练完成：%s", result_dir)
    if bool(getattr(args, "auto_analyze", True)):
        analysis_dir = analyze_result(result_dir)
        logging.info("SevereTest 分析完成：%s", analysis_dir)
    print("SEVERE_TEST_RESULT_DIR={}".format(result_dir))


if __name__ == "__main__":
    main()
