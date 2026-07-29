"""固定人数四组实验的FedML配置、路径和结果目录工具。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import fedml
from fedml.arguments import load_arguments
from fedml.constants import (
    FEDML_SIMULATION_TYPE_SP,
    FEDML_TRAINING_PLATFORM_SIMULATION,
)

from .core.randomness import seed_everything
from .core.device import as_bool
from .core.topology import (
    FixedCountTopologyProvider,
    MatlabTopologyProvider,
    TopologyProvider,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def initialize_fedml_runtime():
    """使用FedML YAML解析器加载参数并固定全部随机种子。"""

    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
    fedml._global_comm_backend = FEDML_SIMULATION_TYPE_SP
    args = load_arguments(
        FEDML_TRAINING_PLATFORM_SIMULATION,
        FEDML_SIMULATION_TYPE_SP,
    )
    seed_everything(int(args.random_seed))
    return args


def resolve_package_path(path_value: str) -> Path:
    """把相对路径解析为相对于HFLSnF_KG_v3目录的绝对路径。"""

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PACKAGE_DIR / path).resolve()


def create_result_directory(args) -> Path:
    """根据运行名称创建不会覆盖旧实验的微秒时间戳目录。"""

    root = resolve_package_path(
        getattr(args, "result_root", "results")
    )
    run_name = str(
        getattr(args, "run_name", "hflsnf_kg_v3")
    ).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = root / "{}_{}".format(run_name, timestamp)
    result_dir.mkdir(parents=True, exist_ok=False)
    args.result_dir = str(result_dir)
    return result_dir


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """把配置、指标或审计信息写为简体中文友好的UTF-8 JSON。"""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def build_topology_provider(
    args,
    client_ids,
) -> TopologyProvider:
    """根据配置创建固定人数或MAT原样回放拓扑。"""

    topology_type = str(
        getattr(args, "topology_type", "fixed_count")
    ).strip().lower()
    client_ids = tuple(int(value) for value in client_ids)
    architecture = str(
        getattr(args, "topology_architecture", "hfl")
    ).strip().lower()
    snf_enabled = as_bool(getattr(args, "topology_snf", True))
    if topology_type == "matlab_direct":
        mat_path = resolve_package_path(
            getattr(
                args,
                "dynamic_group_mat_file",
                "matlab/result-U-6fixedge_epoch200_"
                "varAlpha_0p1_trainable.mat",
            )
        )
        return MatlabTopologyProvider(
            mat_path=mat_path,
            architecture=architecture,
            snf_enabled=snf_enabled,
            edge_mode=str(
                getattr(args, "topology_edge_mode", "fixed")
            ),
            util=float(getattr(args, "topology_util", 0.5)),
            client_count=len(client_ids),
            schedule_policy=str(
                getattr(
                    args,
                    "topology_schedule_policy",
                    "strict",
                )
            ),
        )
    if topology_type != "fixed_count":
        raise ValueError(
            "topology_type必须是fixed_count或matlab_direct"
        )
    selection_mode = str(
        getattr(
            args,
            "fixed_count_selection_mode",
            (
                "snf_mat_projected"
                if snf_enabled
                else "seeded_round_robin"
            ),
        )
    ).strip().lower()
    source_provider = None
    if selection_mode == "snf_mat_projected":
        mat_path = resolve_package_path(
            getattr(
                args,
                "dynamic_group_mat_file",
                "matlab/result-U-6fixedge_epoch200_"
                "varAlpha_0p1_trainable.mat",
            )
        )
        source_provider = MatlabTopologyProvider(
            mat_path=mat_path,
            architecture=architecture,
            snf_enabled=True,
            edge_mode=str(
                getattr(args, "topology_edge_mode", "fixed")
            ),
            util=float(getattr(args, "topology_util", 0.5)),
            client_count=len(client_ids),
            schedule_policy=str(
                getattr(
                    args,
                    "topology_schedule_policy",
                    "strict",
                )
            ),
        )
    return FixedCountTopologyProvider(
        client_ids=client_ids,
        participant_count=int(args.client_num_per_round),
        architecture=architecture,
        group_count=int(getattr(args, "edge_num", 1)),
        selection_mode=selection_mode,
        seed=int(getattr(args, "fixed_count_seed", args.random_seed)),
        source_provider=source_provider,
    )
