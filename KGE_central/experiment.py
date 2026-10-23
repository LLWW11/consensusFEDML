"""加载数据并执行集中式TransE训练、评估和检查点保存。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .configuration import (
    as_namespace,
    load_flat_config,
    normalize_dataset_name,
    resolve_dataset_path,
)
from .engine import (
    CentralizedTransETrainer,
    TransE,
    build_synthetic_knowledge_graph,
    load_fb15k237,
    load_wn18rr,
)
from .runtime import (
    create_result_directory,
    resolve_device,
    seed_everything,
    write_json,
)


def _load_dataset(config: Dict[str, object]):
    """按规范名称分派真实数据加载器，或构造内存合成图谱。"""

    dataset_name = normalize_dataset_name(config["dataset"])
    if dataset_name == "synthetic-kg":
        return build_synthetic_knowledge_graph()
    loaders = {"fb15k-237": load_fb15k237, "wn18rr": load_wn18rr}
    return loaders[dataset_name](resolve_dataset_path(config))


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """计算模型参数名称、形状、类型和数值的稳定SHA-256。"""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_payload(
    model: TransE,
    dataset,
    summary: Dict[str, object],
) -> Dict[str, object]:
    """构造包含模型、映射、数据摘要和训练合同的检查点。"""

    return {
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "entity_to_id": dict(dataset.entity_to_id),
        "relation_to_id": dict(dataset.relation_to_id),
        "dataset_summary": dataset.summary(),
        "training_summary": dict(summary),
    }


def run_experiment(
    config_path: Path,
    result_root_override: Optional[Path] = None,
) -> Tuple[Path, Dict[str, object]]:
    """执行一份集中式YAML并返回结果目录和完整训练摘要。"""

    config = load_flat_config(config_path)
    if result_root_override is not None:
        config["result_root"] = str(
            Path(result_root_override).expanduser().resolve()
        )
    seed_everything(int(config["random_seed"]))
    device = resolve_device(config)
    dataset = _load_dataset(config)
    model = TransE(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        embedding_dim=int(config["embedding_dim"]),
        distance_norm=int(config["distance_norm"]),
    )
    initial_model_hash = _state_dict_sha256(model)
    result_dir = create_result_directory(config)
    write_json(result_dir / "config_snapshot.json", config)
    write_json(result_dir / "dataset_summary.json", dataset.summary())
    write_json(result_dir / "entity2id.json", dataset.entity_to_id)
    write_json(result_dir / "relation2id.json", dataset.relation_to_id)

    trainer = CentralizedTransETrainer(
        args=as_namespace(config),
        dataset=dataset,
        model=model,
        device=device,
    )
    summary = trainer.train(result_dir)
    summary.update(
        {
            "runtime": "standalone_centralized_transe",
            "device": str(device),
            "result_dir": str(result_dir),
            "initial_model_hash": initial_model_hash,
            "best_model_hash": _state_dict_sha256(model),
            "implementation_kernel": (
                "KGE_central"
            ),
        }
    )
    write_json(result_dir / "summary.json", summary)
    torch.save(
        _checkpoint_payload(model, dataset, summary),
        result_dir / "model_best.pt",
    )
    return result_dir, summary
