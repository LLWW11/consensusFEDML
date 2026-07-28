"""运行阶段0的37客户端HFLSnF动态MAT TransE链路。"""

from __future__ import annotations

import json
import logging

import torch

from .core.device import resolve_fedml_device
from .experiment import (
    build_federated_data,
    build_transe,
    checkpoint_payload,
    load_configured_dataset,
)
from .fedml_kge import FedMLDynamicTopologyTransERunner
from .runtime import (
    build_topology_provider,
    create_result_directory,
    initialize_fedml_runtime,
    write_json,
)


def main() -> None:
    """加载37客户端、动态拓扑并完成HFLSnF训练与结果保存。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = initialize_fedml_runtime()
    device = resolve_fedml_device(args)
    dataset = load_configured_dataset(args)
    federated_data = build_federated_data(args, dataset)
    client_ids = tuple(
        int(partition.client_id)
        for partition in federated_data.partitions
    )
    topology_provider = build_topology_provider(args, client_ids)
    model = build_transe(
        args,
        dataset.num_entities,
        dataset.num_relations,
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
    write_json(result_dir / "entity2id.json", dataset.entity_to_id)
    write_json(result_dir / "relation2id.json", dataset.relation_to_id)

    summary = runner.run(result_dir)
    summary["device"] = str(device)
    summary["result_dir"] = str(result_dir)
    write_json(result_dir / "summary.json", summary)
    torch.save(
        checkpoint_payload(
            runner.get_global_state(),
            dataset,
            summary,
            federated_data=federated_data,
        ),
        result_dir / "model_best.pt",
    )
    logging.info(
        "V3 HFLSnF训练完成：%s",
        json.dumps(summary, ensure_ascii=False, default=str),
    )
    print("V3 HFLSnF训练完成")
    print("结果目录：{}".format(result_dir))


if __name__ == "__main__":
    main()
