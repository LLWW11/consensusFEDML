"""固定人数四组联邦TransE入口共享的实验构造函数。"""

from __future__ import annotations

from typing import Dict

import torch

from .runtime import resolve_package_path
from .tasks.kge import (
    BALANCED_HEAD_ENTITY,
    TransE,
    build_synthetic_knowledge_graph,
    load_fb15k237,
    partition_train_triples_by_head,
)


def load_configured_dataset(args):
    """根据FedML配置加载合成知识图谱或标准FB15k-237。"""

    name = str(getattr(args, "dataset", "")).strip().lower()
    if name in {"synthetic-kg", "synthetic_kg"}:
        return build_synthetic_knowledge_graph()
    if name in {"fb15k-237", "fb15k237"}:
        return load_fb15k237(
            resolve_package_path(
                getattr(args, "data_dir", "data/FB15k-237")
            )
        )
    raise ValueError(
        "dataset必须是synthetic-kg或fb15k-237，实际为{}".format(
            name
        )
    )


def build_transe(
    args,
    num_entities: int,
    num_relations: int,
) -> TransE:
    """根据配置创建V3唯一支持的TransE骨干模型。"""

    model_name = str(
        getattr(args, "model", "transe")
    ).strip().lower()
    if model_name != "transe":
        raise ValueError("阶段0至阶段2只支持TransE")
    return TransE(
        num_entities=int(num_entities),
        num_relations=int(num_relations),
        embedding_dim=int(getattr(args, "embedding_dim", 256)),
        distance_norm=int(getattr(args, "distance_norm", 1)),
    )


def build_federated_data(args, dataset):
    """按头实体均衡策略构造四组实验共享的客户端数据。"""

    strategy = str(
        getattr(args, "partition_strategy", BALANCED_HEAD_ENTITY)
    ).strip().lower()
    if strategy != BALANCED_HEAD_ENTITY:
        raise ValueError(
            "四组对照只支持partition_strategy={}".format(
                BALANCED_HEAD_ENTITY
            )
        )
    partition_arguments = {
        "dataset": dataset,
        "client_count": int(args.client_num_in_total),
        "seed": int(args.random_seed),
    }
    return partition_train_triples_by_head(**partition_arguments)


def checkpoint_payload(
    model_state: Dict[str, torch.Tensor],
    dataset,
    summary: Dict[str, object],
    federated_data=None,
) -> Dict[str, object]:
    """构造包含映射、模型状态和可选客户端分区的标准检查点。"""

    payload: Dict[str, object] = {
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model_state.items()
        },
        "entity_to_id": dict(dataset.entity_to_id),
        "relation_to_id": dict(dataset.relation_to_id),
        "dataset_summary": dataset.summary(),
        "training_summary": dict(summary),
    }
    if federated_data is not None:
        payload["client_partition_summary"] = (
            federated_data.summary()
        )
    return payload
