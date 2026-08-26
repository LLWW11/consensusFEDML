"""PyKEEN原生pipeline训练、最佳模型选择与双评估流程。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import torch

from .evaluator import FilteredRankingEvaluator
from .model import PyKEENTransEView
from .pykeen_bridge import (
    build_pykeen_transe,
    build_triples_factories,
    canonical_model_sha256,
    flatten_metric_results,
    require_pykeen,
)


def _metric_value(
    metrics: Dict[str, float],
    *suffixes: str
) -> float:
    """按完整键或后缀从扁平PyKEEN指标中读取一个值。"""

    for suffix in suffixes:
        for key, value in metrics.items():
            if key == suffix or key.endswith(suffix):
                return float(value)
    return float("nan")


def _write_native_metrics(
    path: Path,
    losses,
    validation_results,
    eval_every: int,
) -> None:
    """把PyKEEN逐epoch损失写入与原工程兼容的CSV字段。"""

    from .trainer import CentralizedTransETrainer

    validation_by_epoch = {
        int(eval_every) * (index + 1): float(value)
        for index, value in enumerate(validation_results)
    }
    with Path(path).open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CentralizedTransETrainer.METRIC_FIELDS),
        )
        writer.writeheader()
        for epoch, loss in enumerate(losses, start=1):
            row = {
                field: float("nan")
                for field in CentralizedTransETrainer.METRIC_FIELDS
            }
            row["epoch"] = int(epoch)
            row["train_loss"] = float(loss)
            if epoch in validation_by_epoch:
                row["val_mrr"] = validation_by_epoch[epoch]
            writer.writerow(row)


def _evaluate_with_pykeen(
    model,
    dataset,
    query_batch_size: int,
    candidate_batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """用PyKEEN RankBasedEvaluator计算完整验证和测试指标。"""

    require_pykeen()
    from pykeen.evaluation import RankBasedEvaluator

    evaluator = RankBasedEvaluator(filtered=True)
    validation = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.valid_triples,
        additional_filter_triples=[
            dataset.train_triples,
            dataset.test_triples,
        ],
        batch_size=int(query_batch_size),
        slice_size=int(candidate_batch_size),
        use_tqdm=True,
    )
    testing = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.test_triples,
        additional_filter_triples=[
            dataset.train_triples,
            dataset.valid_triples,
        ],
        batch_size=int(query_batch_size),
        slice_size=int(candidate_batch_size),
        use_tqdm=True,
    )
    return {
        "full_validation": flatten_metric_results(validation),
        "full_test": flatten_metric_results(testing),
    }


def run_native_pipeline(
    args,
    dataset,
    device: torch.device,
    result_dir: Path,
) -> Tuple[PyKEENTransEView, Dict[str, object], Dict[str, object]]:
    """运行PyKEEN原生sLCWA训练并返回统一结果合同。"""

    require_pykeen()
    from pykeen.losses import NSSALoss
    from pykeen.pipeline import pipeline

    seed = int(getattr(args, "random_seed", 0))
    eval_every = int(getattr(args, "eval_every", 10))
    validation_max = int(
        getattr(args, "validation_max_triples", 0)
    )
    validation_selection = str(
        getattr(args, "validation_selection", "random")
    ).strip().lower()
    selected_validation = FilteredRankingEvaluator._select_triples(
        dataset.valid_triples,
        validation_max,
        seed + 17,
        relation_stratified=(
            validation_selection == "relation_stratified"
        ),
    )
    training_factory, validation_factory, testing_factory = (
        build_triples_factories(
            dataset,
            validation_triples=selected_validation,
        )
    )
    pykeen_model = build_pykeen_transe(
        triples_factory=training_factory,
        embedding_dim=int(getattr(args, "embedding_dim")),
        distance_norm=int(getattr(args, "distance_norm")),
        random_seed=seed,
    )
    pykeen_model.loss = NSSALoss(
        margin=float(getattr(args, "fede_gamma", 9.0)),
        adversarial_temperature=float(
            getattr(args, "adversarial_temperature", 1.0)
        ),
        reduction="mean",
    )
    # 原生pipeline首批训练前也执行与严格配方相同的实体约束。
    pykeen_model.post_parameter_update()
    initial_model_hash = canonical_model_sha256(pykeen_model)

    query_batch_size = int(
        getattr(args, "evaluation_query_batch_size", 1)
    )
    candidate_batch_size = int(
        getattr(args, "evaluation_candidate_batch_size", 4096)
    )
    result = pipeline(
        training=training_factory,
        validation=validation_factory,
        testing=testing_factory,
        model=pykeen_model,
        optimizer="Adam",
        optimizer_kwargs={
            "lr": float(getattr(args, "learning_rate"))
        },
        training_loop="sLCWA",
        negative_sampler="BasicNegativeSampler",
        negative_sampler_kwargs={
            "num_negs_per_pos": int(
                getattr(args, "negative_sample_count")
            ),
            "corruption_scheme": ("head", "tail"),
            "filtered": True,
        },
        training_kwargs={
            "num_epochs": int(getattr(args, "epochs")),
            "batch_size": int(getattr(args, "batch_size")),
            "use_tqdm": True,
        },
        stopper="early",
        stopper_kwargs={
            "frequency": eval_every,
            "patience": 1000,
            "relative_delta": 0.0,
            "metric": (
                "both.optimistic.inverse_harmonic_mean_rank"
            ),
        },
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={"filtered": True},
        evaluation_kwargs={
            "batch_size": query_batch_size,
            "slice_size": candidate_batch_size,
            "use_tqdm": True,
        },
        device=torch.device(device),
        random_seed=seed,
        evaluation_fallback=False,
        filter_validation_when_testing=True,
        use_tqdm=True,
    )
    model_view = PyKEENTransEView(
        result.model,
        distance_norm=int(getattr(args, "distance_norm")),
    )
    canonical_evaluator = FilteredRankingEvaluator(dataset)
    canonical_validation = canonical_evaluator.evaluate(
        model_view,
        dataset.valid_triples,
        device,
        max_triples=int(
            getattr(args, "final_validation_max_triples", 0)
        ),
        seed=seed + 17,
        candidate_batch_size=candidate_batch_size,
        query_batch_size=query_batch_size,
    )
    canonical_test = canonical_evaluator.evaluate(
        model_view,
        dataset.test_triples,
        device,
        max_triples=int(getattr(args, "test_max_triples", 0)),
        seed=seed + 29,
        candidate_batch_size=candidate_batch_size,
        query_batch_size=query_batch_size,
    )
    pykeen_metrics = _evaluate_with_pykeen(
        result.model,
        dataset,
        query_batch_size=query_batch_size,
        candidate_batch_size=candidate_batch_size,
    )
    stopper = result.stopper
    validation_results = list(getattr(stopper, "results", []))
    _write_native_metrics(
        Path(result_dir) / "metrics.csv",
        result.losses,
        validation_results,
        eval_every=eval_every,
    )
    best_epoch = getattr(stopper, "best_epoch", None)
    best_metric = getattr(stopper, "best_metric", float("nan"))
    summary = {
        "task": "centralized_knowledge_graph_completion",
        "initial_model_hash": initial_model_hash,
        "runtime": "pykeen_native_pipeline",
        "dataset": dataset.dataset_name,
        "local_objective": "bidirectional_self_adversarial",
        "negative_sample_count": int(
            getattr(args, "negative_sample_count")
        ),
        "negative_sampling_backend": (
            "pykeen_basic_filtered_training_only"
        ),
        "subsampling_weights_precomputed": False,
        "profile_training_timing": False,
        "fede_gamma": float(getattr(args, "fede_gamma", 9.0)),
        "adversarial_temperature": float(
            getattr(args, "adversarial_temperature", 1.0)
        ),
        "epochs_configured": int(getattr(args, "epochs")),
        "epochs_ran": len(result.losses),
        "best_epoch": (
            int(best_epoch)
            if best_epoch is not None
            else len(result.losses)
        ),
        "monitor_every_epoch": False,
        "monitor_validation_max_triples": 0,
        "selection_eval_every": eval_every,
        "selection_validation_max_triples": validation_max,
        "validation_selection": validation_selection,
        "evaluation_query_batch_size": query_batch_size,
        "evaluation_candidate_batch_size": candidate_batch_size,
        "best_validation_mrr_during_training": float(best_metric),
        "final_validation_metrics": canonical_validation,
        "final_test_metrics": canonical_test,
        "metrics_file": str(Path(result_dir) / "metrics.csv"),
        "pykeen_pipeline_test_realistic_mrr": _metric_value(
            pykeen_metrics["full_test"],
            "both.realistic.inverse_harmonic_mean_rank",
            "both.realistic.mean_reciprocal_rank",
        ),
        "pykeen_pipeline_test_optimistic_mrr": _metric_value(
            pykeen_metrics["full_test"],
            "both.optimistic.inverse_harmonic_mean_rank",
            "both.optimistic.mean_reciprocal_rank",
        ),
    }
    return model_view, summary, pykeen_metrics
