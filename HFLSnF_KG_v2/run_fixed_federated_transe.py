"""运行四种固定参与集合的FL/HFL联邦TransE对照实验。"""

from __future__ import annotations

import json
import logging
from typing import Dict

import torch

from .core.device import resolve_fedml_device
from .fedml_kge import FedMLFixedTopologyTransERunner
from .run_federated_transe import build_federated_data
from .run_hfl_kg import (
    create_result_directory,
    initialize_fedml_runtime,
    write_json,
)
from .run_transe import build_transe, load_configured_dataset


def _checkpoint_payload(
    runner: FedMLFixedTopologyTransERunner,
    federated_data,
    summary: Dict[str, object],
) -> Dict[str, object]:
    """构造包含最佳模型、固定拓扑、编号映射和分区的检查点。"""

    dataset = federated_data.dataset
    return {
        "model_state_dict": runner.get_global_state(),
        "entity_to_id": dict(dataset.entity_to_id),
        "relation_to_id": dict(dataset.relation_to_id),
        "dataset_summary": dataset.summary(),
        "client_partition_summary": federated_data.summary(),
        "fixed_topology_summary": (
            runner.get_fixed_topology_summary()
        ),
        "training_summary": summary,
    }


def main() -> None:
    """初始化FedML并执行一种固定FL/HFL TransE对照实验。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = initialize_fedml_runtime()
    # 正式配置必须在读取FB15k-237之前验证CUDA，避免误用CPU长时间运行。
    device = resolve_fedml_device(args)
    dataset = load_configured_dataset(args)
    federated_data = build_federated_data(args, dataset)
    model = build_transe(
        args, dataset.num_entities, dataset.num_relations
    )
    runner = FedMLFixedTopologyTransERunner(
        args=args,
        device=device,
        federated_data=federated_data,
        model=model,
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
        result_dir / "fixed_participation.json",
        runner.get_fixed_topology_summary(),
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
        "固定拓扑联邦TransE训练完成：%s",
        json.dumps(summary, ensure_ascii=False, default=str),
    )
    test_metrics = summary["final_test_metrics"]
    print(
        "{}训练完成，固定客户端：{}".format(
            summary["scenario"],
            summary["active_client_ids"],
        )
    )
    print("结果目录：{}".format(result_dir))
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
