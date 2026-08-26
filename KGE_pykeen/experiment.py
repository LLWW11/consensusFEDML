"""加载数据并执行PyKEEN双口径集中式TransE实验。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .configuration import (
    as_namespace,
    load_flat_config,
    resolve_project_path,
)
from .data import (
    load_fb15k237,
)
from .model import TransE
from .native_runner import run_native_pipeline
from .pykeen_bridge import (
    build_triples_factories,
    dataset_contract,
    environment_snapshot,
)
from .runtime import (
    create_result_directory,
    resolve_device,
    seed_everything,
    write_json,
)
from .trainer import CentralizedTransETrainer


def _load_dataset(config: Dict[str, object]):
    """按配置加载正式FB15k-237知识图谱。"""

    data_dir = resolve_project_path(config["data_dir"])
    return load_fb15k237(data_dir)


def _state_dict_sha256(model: torch.nn.Module) -> str:
    """计算规范化模型参数的稳定SHA-256。"""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _comparison_contract(
    mode: str,
    dataset_audit: Dict[str, object],
) -> Dict[str, object]:
    """返回当前模式公开声明的等价项和差异项。"""

    matched = str(mode) == "matched_recipe"
    return {
        "comparison_mode": str(mode),
        "dataset_contract": dict(dataset_audit),
        "shared_hyperparameters": [
            "random_seed",
            "embedding_dim",
            "distance_norm",
            "epochs",
            "batch_size",
            "learning_rate",
            "negative_sample_count",
            "fede_gamma",
            "adversarial_temperature",
        ],
        "strict_all_true_negative_filtering": matched,
        "batch_alternating_head_tail_corruption": matched,
        "frequency_subsampling_weights": matched,
        "epoch_one_validation_selection": matched,
        "canonical_optimistic_filtered_evaluation": True,
        "native_semantic_differences": (
            []
            if matched
            else [
                "训练负采样只过滤训练事实",
                "BasicNegativeSampler不按整批交替头尾方向",
                "PyKEEN原生NSSALoss不使用频率子采样权重",
                "EarlyStopper从第eval_every个epoch开始评估",
            ]
        ),
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    dataset,
    summary: Dict[str, object],
) -> Dict[str, object]:
    """构造规范化权重、PyKEEN原始状态和完整训练合同。"""

    return {
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "pykeen_model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.pykeen_model.state_dict().items()
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
    """执行一份PyKEEN双口径YAML并返回结果目录和摘要。"""

    config = load_flat_config(config_path)
    if result_root_override is not None:
        config["result_root"] = str(
            Path(result_root_override).expanduser().resolve()
        )
    seed_everything(int(config["random_seed"]))
    device = resolve_device(config)
    dataset = _load_dataset(config)
    dataset_audit = dataset_contract(dataset)
    mode = str(config["comparison_mode"]).strip().lower()
    result_dir = create_result_directory(config)
    write_json(result_dir / "config_snapshot.json", config)
    write_json(result_dir / "dataset_summary.json", dataset.summary())
    write_json(result_dir / "entity2id.json", dataset.entity_to_id)
    write_json(result_dir / "relation2id.json", dataset.relation_to_id)
    write_json(
        result_dir / "environment_snapshot.json",
        environment_snapshot(device),
    )
    contract = _comparison_contract(mode, dataset_audit)
    write_json(result_dir / "comparison_contract.json", contract)

    pykeen_metrics: Dict[str, object] = {
        "comparison_mode": mode,
        "note": "matched模式的正式指标由canonical评估器给出",
    }
    initial_model_hash = None
    if mode == "matched_recipe":
        training_factory, _, _ = build_triples_factories(dataset)
        model = TransE(
            triples_factory=training_factory,
            embedding_dim=int(config["embedding_dim"]),
            distance_norm=int(config["distance_norm"]),
            random_seed=int(config["random_seed"]),
        )
        initial_model_hash = _state_dict_sha256(model)
        trainer = CentralizedTransETrainer(
            args=as_namespace(config),
            dataset=dataset,
            model=model,
            device=device,
        )
        summary = trainer.train(result_dir)
    else:
        model, summary, pykeen_metrics = run_native_pipeline(
            args=as_namespace(config),
            dataset=dataset,
            device=device,
            result_dir=result_dir,
        )

    summary.update(
        {
            "comparison_mode": mode,
            "device": str(device),
            "result_dir": str(result_dir),
            "initial_model_hash": (
                initial_model_hash
                if initial_model_hash is not None
                else summary.get("initial_model_hash")
            ),
            "best_model_hash": _state_dict_sha256(model),
            "implementation_kernel": "KGE_pykeen",
            "dataset_contract": dataset_audit,
            "comparison_contract_file": str(
                result_dir / "comparison_contract.json"
            ),
            "environment_snapshot_file": str(
                result_dir / "environment_snapshot.json"
            ),
            "pykeen_metrics_file": str(
                result_dir / "pykeen_metrics.json"
            ),
        }
    )
    write_json(result_dir / "pykeen_metrics.json", pykeen_metrics)
    write_json(result_dir / "summary.json", summary)
    payload = _checkpoint_payload(model, dataset, summary)
    torch.save(payload, result_dir / "model_best.pt")
    torch.save(
        payload["pykeen_model_state_dict"],
        result_dir / "pykeen_model_raw.pt",
    )
    return result_dir, summary
