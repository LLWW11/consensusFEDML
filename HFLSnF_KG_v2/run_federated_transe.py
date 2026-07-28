"""使用FedML Client和ClientTrainer运行普通联邦TransE。"""

from __future__ import annotations

import json
import logging
from typing import Dict

import torch

from .core.device import resolve_fedml_device
from .fedml_kge import FedMLTransERunner
from .run_hfl_kg import (
    create_result_directory,
    initialize_fedml_runtime,
    write_json,
)
from .run_transe import build_transe, load_configured_dataset
from .tasks.kge import partition_train_triples_by_head


def build_federated_data(args, dataset):
    """校验划分策略并构造头实体均衡的知识客户端数据。"""

    strategy = str(
        getattr(args, "partition_strategy", "balanced_head_entity")
    ).strip().lower()
    if strategy != "balanced_head_entity":
        raise ValueError(
            "阶段四partition_strategy必须是balanced_head_entity"
        )
    return partition_train_triples_by_head(
        dataset,
        client_count=int(args.client_num_in_total),
        seed=int(args.random_seed),
    )


def _checkpoint_payload(
    runner: FedMLTransERunner,
    federated_data,
    summary: Dict[str, object],
) -> Dict[str, object]:
    """构造包含最佳全局模型、编号映射和分区信息的检查点。"""

    dataset = federated_data.dataset
    return {
        "model_state_dict": runner.get_global_state(),
        "entity_to_id": dict(dataset.entity_to_id),
        "relation_to_id": dict(dataset.relation_to_id),
        "dataset_summary": dataset.summary(),
        "client_partition_summary": federated_data.summary(),
        "training_summary": summary,
    }


def main() -> None:
    """初始化FedML环境并执行阶段四普通联邦TransE训练。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = initialize_fedml_runtime()
    device = resolve_fedml_device(args)
    dataset = load_configured_dataset(args)
    federated_data = build_federated_data(args, dataset)
    model = build_transe(
        args, dataset.num_entities, dataset.num_relations
    )
    result_dir = create_result_directory(args)
    write_json(
        result_dir / "config_snapshot.json",
        {
            key: value
            for key, value in sorted(vars(args).items())
        },
    )
    write_json(result_dir / "dataset_summary.json", dataset.summary())
    write_json(
        result_dir / "client_partition_summary.json",
        federated_data.summary(),
    )
    write_json(result_dir / "entity2id.json", dataset.entity_to_id)
    write_json(
        result_dir / "relation2id.json", dataset.relation_to_id
    )

    runner = FedMLTransERunner(
        args=args,
        device=device,
        federated_data=federated_data,
        model=model,
    )
    summary = runner.run(result_dir)
    summary["device"] = str(device)
    summary["result_dir"] = str(result_dir)
    write_json(result_dir / "summary.json", summary)
    torch.save(
        _checkpoint_payload(runner, federated_data, summary),
        result_dir / "model_best.pt",
    )

    logging.info(
        "阶段四普通联邦TransE训练完成：%s",
        json.dumps(summary, ensure_ascii=False, default=str),
    )
    test_metrics = summary["final_test_metrics"]
    print("普通联邦TransE训练完成，结果目录：{}".format(result_dir))
    print(
        "最终filtered MRR/Hits@10：{:.6f} / {:.6f}".format(
            float(test_metrics["mrr"]),
            float(test_metrics["hits_at_10"]),
        )
    )
    print(
        "相对集中式测试MRR差值：{:.6f}".format(
            float(summary["test_mrr_delta_vs_centralized"])
        )
    )


if __name__ == "__main__":
    main()
