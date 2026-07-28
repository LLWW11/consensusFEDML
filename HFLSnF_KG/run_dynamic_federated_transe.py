"""运行由MATLAB逐轮拓扑驱动的动态分层联邦TransE。"""

from __future__ import annotations

import json
import logging
from typing import Dict

import torch

from .core.device import resolve_fedml_device
from .fedml_kge import FedMLDynamicTopologyTransERunner
from .run_federated_transe import build_federated_data
from .run_hfl_kg import (
    build_topology_provider,
    create_result_directory,
    initialize_fedml_runtime,
    write_json,
)
from .run_transe import build_transe, load_configured_dataset


def _checkpoint_payload(
    runner: FedMLDynamicTopologyTransERunner,
    federated_data,
    summary: Dict[str, object],
) -> Dict[str, object]:
    """构造包含最佳模型、MAT调度摘要、映射和分区的检查点。"""

    dataset = federated_data.dataset
    return {
        "model_state_dict": runner.get_global_state(),
        "entity_to_id": dict(dataset.entity_to_id),
        "relation_to_id": dict(dataset.relation_to_id),
        "dataset_summary": dataset.summary(),
        "client_partition_summary": federated_data.summary(),
        "dynamic_participation_summary": (
            runner.get_dynamic_participation_summary()
        ),
        "training_summary": summary,
    }


def main() -> None:
    """初始化FedML、读取MAT并启动动态采样分层TransE实验。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = initialize_fedml_runtime()
    # 正式配置先检查CUDA，避免加载FB15k-237后才发现服务器环境错误。
    device = resolve_fedml_device(args)
    dataset = load_configured_dataset(args)
    federated_data = build_federated_data(args, dataset)
    client_ids = tuple(
        int(partition.client_id)
        for partition in federated_data.partitions
    )
    topology_provider = build_topology_provider(args, client_ids)
    model = build_transe(
        args, dataset.num_entities, dataset.num_relations
    )
    runner = FedMLDynamicTopologyTransERunner(
        args=args,
        device=device,
        federated_data=federated_data,
        model=model,
        topology_provider=topology_provider,
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
    write_json(
        result_dir / "topology_metadata.json",
        topology_provider.describe(),
    )
    write_json(
        result_dir / "dynamic_participation_summary.json",
        runner.get_dynamic_participation_summary(),
    )
    write_json(result_dir / "entity2id.json", dataset.entity_to_id)
    write_json(
        result_dir / "relation2id.json", dataset.relation_to_id
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
        "动态MAT联邦TransE训练完成：%s",
        json.dumps(summary, ensure_ascii=False, default=str),
    )
    test_metrics = summary["final_test_metrics"]
    print("动态MAT联邦TransE训练完成")
    print("结果目录：{}".format(result_dir))
    print(
        "参与客户端范围/均值：{}–{} / {:.3f}".format(
            summary["participant_count_min"],
            summary["participant_count_max"],
            float(summary["participant_count_mean"]),
        )
    )
    print(
        "动态分组范围/均值：{}–{} / {:.3f}".format(
            summary["group_count_min"],
            summary["group_count_max"],
            float(summary["group_count_mean"]),
        )
    )
    print(
        "最终filtered MRR/Hits@3/Hits@10："
        "{:.6f} / {:.6f} / {:.6f}".format(
            float(test_metrics["mrr"]),
            float(test_metrics["hits_at_3"]),
            float(test_metrics["hits_at_10"]),
        )
    )


if __name__ == "__main__":
    main()
