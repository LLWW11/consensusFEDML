"""FedGCN核心复现的FedML命令行入口。"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import fedml
import numpy as np
import torch
from fedml.arguments import load_arguments
from fedml.constants import (
    FEDML_SIMULATION_TYPE_SP,
    FEDML_TRAINING_PLATFORM_SIMULATION,
)

from .data import build_federated_graph_data
from .device import resolve_fedml_device
from .model import GCN
from .simulator import FedGCNSimulator


def initialize_fedml_runtime():
    """使用FedML官方YAML解析器初始化单进程参数和确定性随机种子。

    FedML 0.7.600 的 ``fedml.init`` 会在本地模拟开始前强制验证内置S3凭据，
    该网络诊断与本实验无关，并会在无有效云凭据时阻断离线训练。因此这里直接
    调用同一套官方 ``load_arguments``，仍保留FedML配置对象和设备初始化，
    但不启动MLOps与远程存储诊断。
    """

    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
    fedml._global_comm_backend = FEDML_SIMULATION_TYPE_SP
    args = load_arguments(
        FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_SIMULATION_TYPE_SP
    )
    seed = int(args.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
    return args


def _resolve_package_relative_path(path_value: str) -> Path:
    """把配置中的相对路径稳定解析为相对于FedGCN模块目录的路径。"""

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def _create_result_directory(args) -> Path:
    """为当前运行创建带名称和微秒时间戳的独立结果目录。"""

    result_root = _resolve_package_relative_path(
        getattr(args, "result_root", "results")
    )
    run_name = str(getattr(args, "run_name", "fedgcn")).strip() or "fedgcn"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = result_root / "{}_{}".format(run_name, timestamp)
    result_dir.mkdir(parents=True, exist_ok=False)
    args.result_dir = str(result_dir)
    return result_dir


def main() -> None:
    """初始化FedML、加载图数据并启动专用FedGCN模拟器。"""

    args = initialize_fedml_runtime()
    device = resolve_fedml_device(args)

    data_dir = _resolve_package_relative_path(args.data_cache_dir)
    dataset = build_federated_graph_data(
        dataset_name=args.dataset,
        data_dir=data_dir,
        client_count=int(args.client_num_in_total),
        iid_fraction=float(args.iid_fraction),
        seed=int(args.random_seed),
    )
    model = GCN(
        input_dim=dataset.num_features,
        hidden_dim=int(args.hidden_dim),
        output_dim=dataset.num_classes,
        dropout=float(args.dropout),
    )
    result_dir = _create_result_directory(args)
    simulator = FedGCNSimulator(args, device, dataset, model)
    summary = simulator.train(result_dir)
    logging.info("FedGCN训练完成：%s", json.dumps(summary, ensure_ascii=False))
    print("FedGCN训练完成，结果目录：{}".format(result_dir))
    print("最终测试准确率：{:.6f}".format(summary["final_test_accuracy"]))


if __name__ == "__main__":
    main()
