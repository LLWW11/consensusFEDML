"""使用FedML配置和设备接口运行集中式TransE基线。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import torch

from .core.device import resolve_fedml_device
from .run_hfl_kg import (
    create_result_directory,
    initialize_fedml_runtime,
    resolve_package_relative_path,
    write_json,
)
from .tasks.kge import (
    CentralizedTransETrainer,
    TransE,
    build_synthetic_knowledge_graph,
    load_fb15k237,
)


def load_configured_dataset(args):
    """根据配置加载内置合成图或标准FB15k-237目录。"""

    dataset_name = str(getattr(args, "dataset", "")).strip().lower()
    if dataset_name in {"synthetic-kg", "synthetic_kg"}:
        return build_synthetic_knowledge_graph()
    if dataset_name in {"fb15k-237", "fb15k237"}:
        data_dir = resolve_package_relative_path(
            getattr(args, "data_dir", "data/FB15k-237")
        )
        try:
            return load_fb15k237(data_dir)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "{}；请按HFLSnF_KG/data/README.md放置"
                "train.txt、valid.txt和test.txt。".format(error)
            )
    raise ValueError(
        "阶段三dataset必须是synthetic-kg或fb15k-237，实际为{}".format(
            dataset_name
        )
    )


def build_transe(args, num_entities: int, num_relations: int) -> TransE:
    """根据YAML参数创建集中式TransE模型。"""

    model_name = str(getattr(args, "model", "transe")).strip().lower()
    if model_name != "transe":
        raise ValueError("阶段三model必须是transe，实际为{}".format(model_name))
    return TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=int(getattr(args, "embedding_dim", 128)),
        distance_norm=int(getattr(args, "distance_norm", 1)),
    )


def _checkpoint_payload(
    model: TransE, dataset, summary: Dict[str, object]
) -> Dict[str, object]:
    """构造包含模型参数和全局编号映射的可恢复检查点。"""

    return {
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "entity_to_id": dict(dataset.entity_to_id),
        "relation_to_id": dict(dataset.relation_to_id),
        "dataset_summary": dataset.summary(),
        "training_summary": summary,
    }


def main() -> None:
    """初始化FedML运行环境并执行阶段三集中式TransE训练。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = initialize_fedml_runtime()
    device = resolve_fedml_device(args)
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    print(
        "TransE运行设备：{}（{}）".format(device, device_name),
        flush=True,
    )
    dataset = load_configured_dataset(args)
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
        result_dir / "entity2id.json", dataset.entity_to_id
    )
    write_json(
        result_dir / "relation2id.json", dataset.relation_to_id
    )

    trainer = CentralizedTransETrainer(
        args=args,
        dataset=dataset,
        model=model,
        device=device,
    )
    summary = trainer.train(result_dir)
    summary["device"] = str(device)
    summary["result_dir"] = str(result_dir)
    write_json(result_dir / "summary.json", summary)
    torch.save(
        _checkpoint_payload(model, dataset, summary),
        result_dir / "model_best.pt",
    )

    logging.info(
        "阶段三集中式TransE训练完成：%s",
        json.dumps(summary, ensure_ascii=False, default=str),
    )
    test_metrics = summary["final_test_metrics"]
    print("集中式TransE训练完成，结果目录：{}".format(result_dir))
    print(
        "最终filtered MRR/Hits@10：{:.6f} / {:.6f}".format(
            float(test_metrics["mrr"]),
            float(test_metrics["hits_at_10"]),
        )
    )


if __name__ == "__main__":
    main()
