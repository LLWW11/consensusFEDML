"""任务无关分层联邦框架的FedML命令行入口。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import fedml
import torch
from fedml.arguments import load_arguments
from fedml.constants import (
    FEDML_SIMULATION_TYPE_SP,
    FEDML_TRAINING_PLATFORM_SIMULATION,
)

from FedGCN_fedml.data import build_federated_graph_data
from FedGCN_fedml.model import GCN

from .core.device import as_bool, resolve_fedml_device
from .core.randomness import seed_everything
from .core.topology import (
    MatlabTopologyProvider,
    StaticTopologyProvider,
    TopologyProvider,
)
from .fedml_framework import FedMLRunner
from .tasks.gcn import CoraGCNTask


def initialize_fedml_runtime():
    """使用FedML官方YAML解析器初始化参数和统一随机种子。"""

    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
    fedml._global_comm_backend = FEDML_SIMULATION_TYPE_SP
    args = load_arguments(
        FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_SIMULATION_TYPE_SP
    )
    seed_everything(int(args.random_seed))
    return args


def resolve_package_relative_path(path_value: str) -> Path:
    """把配置中的相对路径解析为相对于新实验包目录的绝对路径。"""

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def create_result_directory(args) -> Path:
    """创建包含运行名称和微秒时间戳的独立结果目录。"""

    result_root = resolve_package_relative_path(
        getattr(args, "result_root", "results")
    )
    run_name = str(getattr(args, "run_name", "hflsnf_kg")).strip()
    if not run_name:
        run_name = "hflsnf_kg"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = result_root / "{}_{}".format(run_name, timestamp)
    result_dir.mkdir(parents=True, exist_ok=False)
    args.result_dir = str(result_dir)
    return result_dir


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """把运行配置或最终汇总写入UTF-8 JSON文件。"""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def build_topology_provider(
    args, client_ids
) -> TopologyProvider:
    """根据YAML选择静态分层拓扑或旧MATLAB调度适配器。"""

    topology_type = str(
        getattr(args, "topology_type", "static")
    ).strip().lower()
    client_ids = tuple(int(value) for value in client_ids)
    if topology_type == "static":
        expected_clients = int(
            getattr(args, "client_num_per_round", len(client_ids))
        )
        if expected_clients != len(client_ids):
            raise ValueError(
                "阶段二静态拓扑要求每轮全部客户端参与："
                "client_num_per_round={}，实际客户端数={}".format(
                    expected_clients, len(client_ids)
                )
            )
        edge_num = int(getattr(args, "edge_num", 1))
        return StaticTopologyProvider.round_robin(client_ids, edge_num)

    if topology_type == "matlab":
        mat_path = resolve_package_relative_path(
            getattr(
                args,
                "dynamic_group_mat_file",
                "../HFLSnF_dynEdge/matlab/result-U-6fixedge_epoch200.mat",
            )
        )
        provider = MatlabTopologyProvider(
            mat_path=mat_path,
            architecture=getattr(args, "topology_architecture", "hfl"),
            snf_enabled=as_bool(getattr(args, "topology_snf", True)),
            edge_mode=getattr(args, "topology_edge_mode", "fixed"),
            util=float(getattr(args, "topology_util", 0.5)),
            client_count=len(client_ids),
        )
        if int(args.comm_round) > provider.round_count:
            raise ValueError(
                "comm_round={}超过MAT拓扑轮数{}".format(
                    int(args.comm_round), provider.round_count
                )
            )
        return provider

    raise ValueError(
        "topology_type 必须是static或matlab，实际为{}".format(
            topology_type
        )
    )


def build_gcn_task(args, device: torch.device) -> CoraGCNTask:
    """加载Planetoid图数据并创建Cora/Citeseer任务适配器。"""

    dataset_name = str(args.dataset).strip().lower()
    if dataset_name not in {"cora", "citeseer"}:
        raise ValueError(
            "阶段二只支持cora或citeseer，实际为{}".format(dataset_name)
        )
    data_dir = resolve_package_relative_path(args.data_cache_dir)
    dataset = build_federated_graph_data(
        dataset_name=dataset_name,
        data_dir=data_dir,
        client_count=int(args.client_num_in_total),
        iid_fraction=float(args.iid_fraction),
        seed=int(args.random_seed),
    )
    return CoraGCNTask(
        dataset=dataset,
        device=device,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        learning_rate=float(
            getattr(args, "learning_rate", getattr(args, "lr", 0.5))
        ),
        weight_decay=float(getattr(args, "weight_decay", 5e-4)),
        seed=int(args.random_seed),
    )


def build_gcn_dataset_and_model(args):
    """加载Planetoid联邦图数据并创建供FedML ClientTrainer使用的GCN。"""

    dataset_name = str(args.dataset).strip().lower()
    if dataset_name not in {"cora", "citeseer"}:
        raise ValueError(
            "阶段二只支持cora或citeseer，实际为{}".format(dataset_name)
        )
    data_dir = resolve_package_relative_path(args.data_cache_dir)
    dataset = build_federated_graph_data(
        dataset_name=dataset_name,
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
    return dataset, model


def main() -> None:
    """初始化环境并通过FedML Runner链路启动分层GCN训练。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = initialize_fedml_runtime()
    device = resolve_fedml_device(args)
    dataset, model = build_gcn_dataset_and_model(args)
    client_ids = tuple(
        int(partition.client_id) for partition in dataset.partitions
    )
    topology_provider = build_topology_provider(args, client_ids)
    result_dir = create_result_directory(args)
    write_json(
        result_dir / "config_snapshot.json",
        {
            key: value
            for key, value in sorted(vars(args).items())
        },
    )

    fedml_runner = FedMLRunner(
        args=args,
        device=device,
        dataset=dataset,
        model=model,
        topology_provider=topology_provider,
    )
    summary = fedml_runner.run(result_dir)
    summary["device"] = str(device)
    summary["result_dir"] = str(result_dir)
    write_json(result_dir / "summary.json", summary)
    torch.save(
        fedml_runner.get_global_state(), result_dir / "model_final.pt"
    )

    final_metrics = summary["final_metrics"]
    logging.info(
        "分层GCN训练完成：%s",
        json.dumps(summary, ensure_ascii=False, default=str),
    )
    print("分层GCN训练完成，结果目录：{}".format(result_dir))
    print(
        "最终验证集/测试集准确率：{:.6f} / {:.6f}".format(
            float(final_metrics["val_accuracy"]),
            float(final_metrics["test_accuracy"]),
        )
    )


if __name__ == "__main__":
    main()
