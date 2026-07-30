"""加载YAML并启动一次FEMNIST MAT探针实验。"""

from __future__ import absolute_import

import argparse
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from FEMNISTProbe.data import load_femnist_experiment_data
from FEMNISTProbe.model import FEMNISTChannelsLastCNN
from FEMNISTProbe.topology import CyclicMatlabTopology
from FEMNISTProbe.trainer import FastFEMNISTMatTrainer
from topology_schedule import MatlabTopologySchedule


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_cli_arguments():
    """解析配置路径、训练轮数、AMP和恢复位置等运行时覆盖项。"""
    parser = argparse.ArgumentParser(
        description="运行FEMNIST循环MAT固定探针实验。"
    )
    parser.add_argument(
        "--yaml_config_file", "--cf", required=True, type=str
    )
    parser.add_argument("--comm_round_override", type=int, default=None)
    parser.add_argument("--eval_interval_override", type=int, default=None)
    parser.add_argument(
        "--amp_override", choices=["true", "false"], default=None
    )
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--result_root_override", type=str, default=None)
    parser.add_argument("--experiment_tag_override", type=str, default=None)
    parser.add_argument("--gpu_id_override", type=int, default=None)
    parser.add_argument(
        "--reference_baseline",
        action="store_true",
        help="仅用于100轮未优化GPU参考基准。",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="仅供本地冒烟与测试；正式配置仍强制CUDA。",
    )
    return parser.parse_args()


def _resolve_project_path(value):
    """把配置中的相对路径解析到项目根目录。"""
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_flat_yaml(config_path):
    """读取FedML风格分组YAML并展平为属性命名空间。"""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file_obj:
        configuration = yaml.safe_load(file_obj)
    if not isinstance(configuration, dict):
        raise ValueError("YAML顶层必须是映射。")
    flattened = {}
    for family_name, family_values in configuration.items():
        if not isinstance(family_values, dict):
            raise ValueError("YAML分组{}必须是映射。".format(family_name))
        for key, value in family_values.items():
            if key in flattened:
                raise ValueError("YAML字段{}在多个分组中重复。".format(key))
            flattened[key] = value
    flattened["yaml_config_file"] = str(config_path.resolve())
    return SimpleNamespace(**flattened)


def apply_cli_overrides(args, cli_args):
    """把明确的命令行覆盖项写入配置对象。"""
    if cli_args.comm_round_override is not None:
        args.comm_round = int(cli_args.comm_round_override)
    if cli_args.eval_interval_override is not None:
        args.eval_interval = int(cli_args.eval_interval_override)
        args.checkpoint_interval = int(cli_args.eval_interval_override)
    if cli_args.amp_override is not None:
        args.amp_enabled = cli_args.amp_override == "true"
    if cli_args.resume_checkpoint is not None:
        args.resume_checkpoint = cli_args.resume_checkpoint
    if cli_args.result_root_override is not None:
        args.result_root = cli_args.result_root_override
    if cli_args.experiment_tag_override is not None:
        args.experiment_tag = cli_args.experiment_tag_override
    if cli_args.gpu_id_override is not None:
        args.gpu_id = int(cli_args.gpu_id_override)
    if cli_args.reference_baseline:
        args.reference_baseline = True
    if cli_args.allow_cpu:
        args.using_gpu = False
    return args


def seed_everything(seed):
    """固定Python、NumPy、PyTorch和CUDA的初始模型随机种子。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(args):
    """根据配置选择CUDA设备，并禁止正式任务静默回退CPU。"""
    using_gpu = bool(getattr(args, "using_gpu", True))
    if using_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "正式配置要求CUDA；当前PyTorch={}无法访问GPU。".format(
                    torch.__version__
                )
            )
        gpu_id = int(getattr(args, "gpu_id", 0))
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise RuntimeError("gpu_id={}超出可见GPU范围。".format(gpu_id))
        return torch.device("cuda:{}".format(gpu_id))
    return torch.device("cpu")


def build_topology(args):
    """加载varAlpha MAT并创建显式循环拓扑。"""
    mat_path = _resolve_project_path(args.dynamic_group_mat_file)
    schedule = MatlabTopologySchedule(
        mat_path=str(mat_path),
        architecture=str(args.topology_architecture),
        snf_enabled=bool(args.topology_snf),
        edge_mode=str(getattr(args, "topology_edge_mode", "fixed")),
        util=float(args.topology_util),
        client_num_in_total=int(args.client_num_in_total),
        candidate_client_count=int(args.client_num_per_round),
    )
    return CyclicMatlabTopology(
        schedule,
        repeat_mode=str(getattr(args, "topology_repeat_mode", "error")),
    )


def main():
    """执行一次完整或覆盖轮数后的FEMNIST MAT探针实验。"""
    cli_args = parse_cli_arguments()
    args = apply_cli_overrides(
        load_flat_yaml(cli_args.yaml_config_file), cli_args
    )
    if str(args.dataset).lower() != "femnist":
        raise ValueError("新入口当前只接受dataset=femnist。")
    seed_everything(getattr(args, "random_seed", 0))
    device = select_device(args)
    data_bundle = load_femnist_experiment_data(
        data_dir=_resolve_project_path(args.data_cache_dir),
        candidate_count=int(args.client_num_per_round),
        candidate_seed=int(args.candidate_seed),
        probe_samples_per_class=int(args.probe_samples_per_class),
        probe_seed=int(args.probe_seed),
    )
    # 数据加载不接触全局模型随机数；此处再次固定种子，确保四方案初始权重一致。
    seed_everything(getattr(args, "random_seed", 0))
    model = FEMNISTChannelsLastCNN()
    topology = build_topology(args)
    trainer = FastFEMNISTMatTrainer(
        args=args,
        device=device,
        data_bundle=data_bundle,
        model=model,
        cyclic_topology=topology,
    )
    result_dir = trainer.train()
    print("FEMNIST_PROBE_RESULT_DIR={}".format(result_dir), flush=True)


if __name__ == "__main__":
    main()
