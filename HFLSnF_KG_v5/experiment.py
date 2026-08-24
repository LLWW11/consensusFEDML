"""固定人数四组联邦TransE入口共享的实验构造函数。"""

from __future__ import annotations

from typing import Dict

import torch

from .tasks.kge import (
    BALANCED_HEAD_ENTITY,
    BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
    DOMAIN_EXTRACTOR,
    SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
    TransE,
    build_synthetic_knowledge_graph,
    load_fb15k237,
    partition_train_triples_by_head,
    partition_train_triples_by_overlap_target,
    partition_train_triples_by_semantic_domain_graph_local,
)


def load_configured_dataset(args):
    """根据FedML配置加载合成知识图谱或标准FB15k-237。"""

    name = str(getattr(args, "dataset", "")).strip().lower()
    if name in {"synthetic-kg", "synthetic_kg"}:
        return build_synthetic_knowledge_graph()
    if name in {"fb15k-237", "fb15k237"}:
        # 仅真实训练的数据路径解析需要FedML运行时，分区单测无需加载它。
        from .runtime import resolve_package_path

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
    """根据配置创建V5正式实验唯一支持的TransE骨干模型。"""

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
    """按配置构造原始、目标重叠或图语义客户端分区。"""

    strategy = str(
        getattr(args, "partition_strategy", BALANCED_HEAD_ENTITY)
    ).strip().lower()
    partition_arguments = {
        "dataset": dataset,
        "client_count": int(args.client_num_in_total),
        "seed": int(args.random_seed),
    }
    if strategy == BALANCED_HEAD_ENTITY:
        return partition_train_triples_by_head(**partition_arguments)
    if strategy == BALANCED_HEAD_ENTITY_OVERLAP_TARGET:
        if not hasattr(args, "partition_target_entity_overlap"):
            raise ValueError(
                "目标重叠划分缺少partition_target_entity_overlap配置"
            )
        return partition_train_triples_by_overlap_target(
            **partition_arguments,
            target_entity_overlap=float(
                args.partition_target_entity_overlap
            ),
            overlap_tolerance=float(
                getattr(args, "partition_overlap_tolerance", 0.005)
            ),
            load_tolerance=float(
                getattr(args, "partition_load_tolerance", 0.05)
            ),
            relation_overlap_tolerance=float(
                getattr(args, "relation_overlap_tolerance", 0.02)
            ),
            search_restarts=int(
                getattr(args, "partition_search_restarts", 8)
            ),
            search_seed=int(
                getattr(
                    args,
                    "partition_search_seed",
                    int(args.random_seed),
                )
            ),
            strict=True,
        )
    if strategy == SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED:
        extractor = str(
            getattr(args, "partition_domain_extractor", DOMAIN_EXTRACTOR)
        ).strip().lower()
        if extractor != DOMAIN_EXTRACTOR:
            raise ValueError(
                "partition_domain_extractor必须是{}".format(
                    DOMAIN_EXTRACTOR
                )
            )
        search_seed = int(
            getattr(
                args,
                "partition_search_seed",
                partition_arguments["seed"],
            )
        )
        if search_seed != partition_arguments["seed"]:
            raise ValueError(
                "图语义partition_search_seed必须与random_seed一致"
            )
        return partition_train_triples_by_semantic_domain_graph_local(
            **partition_arguments,
            load_tolerance=float(
                getattr(args, "partition_load_tolerance", 0.05)
            ),
            search_restarts=int(
                getattr(args, "partition_search_restarts", 8)
            ),
        )
    raise ValueError(
        "不支持的partition_strategy={}；当前支持{}、{}和{}".format(
            strategy,
            BALANCED_HEAD_ENTITY,
            BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
            SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
        )
    )


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
