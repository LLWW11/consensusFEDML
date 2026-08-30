"""集中式TransE训练、验证选择和最终测试流程。"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .runtime import as_bool, should_run_selection_evaluation
from .utils import clone_state_dict
from .data import KnowledgeGraphDataset
from .evaluator import FilteredRankingEvaluator
from .model import TransE
from .negative_sampling import VectorizedFilteredNegativeSampler
from .objectives import self_adversarial_loss
from .subsampling import TripleFrequencySubsampler


class CentralizedTransETrainer:
    """在完整训练三元组上训练TransE并使用filtered指标选模。"""

    METRIC_FIELDS = (
        "epoch",
        "train_loss",
        "head_train_loss",
        "tail_train_loss",
        "head_positive_count",
        "tail_positive_count",
        "sampling_seconds",
        "transfer_seconds",
        "forward_backward_seconds",
        "epoch_seconds",
        "monitor_mrr",
        "monitor_hits_at_3",
        "monitor_evaluated_triple_count",
        "val_mrr",
        "val_mean_rank",
        "val_hits_at_1",
        "val_hits_at_3",
        "val_hits_at_10",
        "val_evaluated_triple_count",
    )

    def __init__(
        self,
        args,
        dataset: KnowledgeGraphDataset,
        model: TransE,
        device: torch.device,
    ):
        """读取训练超参数并创建负采样器和filtered评估器。"""

        self.args = args
        self.dataset = dataset
        self.model = model
        self.device = torch.device(device)
        self.epochs = int(getattr(args, "epochs", 1))
        self.batch_size = int(getattr(args, "batch_size", 1024))
        self.learning_rate = float(
            getattr(args, "learning_rate", getattr(args, "lr", 0.001))
        )
        self.margin = float(getattr(args, "margin", 1.0))
        self.negative_sample_count = int(
            getattr(args, "negative_sample_count", 1)
        )
        self.local_objective = str(
            getattr(args, "local_objective", "margin_ranking")
        ).strip().lower()
        self.gamma = float(getattr(args, "fede_gamma", 9.0))
        self.adversarial_temperature = float(
            getattr(args, "adversarial_temperature", 1.0)
        )
        self.eval_every = int(getattr(args, "eval_every", 1))
        self.validation_max_triples = int(
            getattr(args, "validation_max_triples", 0)
        )
        self.validation_selection = str(
            getattr(args, "validation_selection", "random")
        ).strip().lower()
        self.monitor_every_epoch = as_bool(
            getattr(args, "monitor_every_epoch", False)
        )
        self.monitor_validation_max_triples = int(
            getattr(
                args,
                "monitor_validation_max_triples",
                self.validation_max_triples,
            )
        )
        self.final_validation_max_triples = int(
            getattr(args, "final_validation_max_triples", 0)
        )
        self.test_max_triples = int(
            getattr(args, "test_max_triples", 0)
        )
        self.candidate_batch_size = int(
            getattr(args, "evaluation_candidate_batch_size", 4096)
        )
        self.query_batch_size = int(
            getattr(args, "evaluation_query_batch_size", 1)
        )
        self.profile_training_timing = as_bool(
            getattr(args, "profile_training_timing", False)
        )
        self.early_stopping_patience = int(
            getattr(args, "early_stopping_patience", 0)
        )
        self.seed = int(getattr(args, "random_seed", 0))
        self._validate_hyperparameters()
        self.negative_sampler = VectorizedFilteredNegativeSampler(
            dataset.num_entities,
            dataset.all_true_triples,
            self.seed + 1009,
            num_relations=dataset.num_relations,
        )
        self.subsampler = TripleFrequencySubsampler(
            dataset.train_triples,
            num_relations=dataset.num_relations,
        )
        self._direction_step = 0
        self.evaluator = FilteredRankingEvaluator(dataset)

    def _validate_hyperparameters(self) -> None:
        """校验集中式TransE训练和评估参数。"""

        if self.epochs <= 0:
            raise ValueError("epochs必须大于0")
        if self.batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate必须大于0")
        if self.margin <= 0.0:
            raise ValueError("margin必须大于0")
        if self.negative_sample_count <= 0:
            raise ValueError("negative_sample_count必须大于0")
        if self.local_objective not in {
            "margin_ranking",
            "bidirectional_self_adversarial",
        }:
            raise ValueError(
                "local_objective必须是margin_ranking或"
                "bidirectional_self_adversarial"
            )
        if self.gamma <= 0.0:
            raise ValueError("fede_gamma必须大于0")
        if self.adversarial_temperature <= 0.0:
            raise ValueError("adversarial_temperature必须大于0")
        if self.eval_every <= 0:
            raise ValueError("eval_every必须大于0")
        if self.monitor_validation_max_triples < 0:
            raise ValueError("monitor_validation_max_triples不能小于0")
        if self.validation_selection not in {
            "random",
            "relation_stratified",
        }:
            raise ValueError(
                "validation_selection必须是random或"
                "relation_stratified"
            )
        if self.candidate_batch_size <= 0:
            raise ValueError("evaluation_candidate_batch_size必须大于0")
        if self.query_batch_size <= 0:
            raise ValueError("evaluation_query_batch_size必须大于0")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience不能小于0")

    def _build_data_loader(self) -> DataLoader:
        """创建使用独立固定种子的训练三元组随机批次。"""

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            TensorDataset(
                self.dataset.train_triples,
                self.subsampler.precomputed_weights,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

    def _synchronize_for_timing(self) -> None:
        """仅在显式性能剖析时同步CUDA，以免正式训练被频繁同步拖慢。"""

        if (
            self.profile_training_timing
            and self.device.type == "cuda"
        ):
            torch.cuda.synchronize(self.device)

    def _train_epoch(
        self, data_loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """完成一个集中式epoch并返回综合及头尾方向训练统计。"""

        self.model.train()
        loss_sum = 0.0
        positive_count = 0
        directional_loss_sums = {"head": 0.0, "tail": 0.0}
        directional_counts = {"head": 0, "tail": 0}
        timing = {
            "sampling_seconds": 0.0,
            "transfer_seconds": 0.0,
            "forward_backward_seconds": 0.0,
        }
        for positive_cpu, sample_weight_cpu in data_loader:
            corruption_mode = (
                "head"
                if self._direction_step % 2 == 0
                else "tail"
            )
            self._synchronize_for_timing()
            started_at = time.perf_counter()
            positives = positive_cpu.to(
                self.device, non_blocking=True
            )
            sample_weights = sample_weight_cpu.to(
                self.device, non_blocking=True
            )
            self._synchronize_for_timing()
            timing["transfer_seconds"] += (
                time.perf_counter() - started_at
            )

            started_at = time.perf_counter()
            negatives = self.negative_sampler.sample(
                positives,
                self.negative_sample_count,
                corruption_mode=(
                    corruption_mode
                    if self.local_objective
                    == "bidirectional_self_adversarial"
                    else "head_tail"
                ),
            )
            self._synchronize_for_timing()
            timing["sampling_seconds"] += (
                time.perf_counter() - started_at
            )

            started_at = time.perf_counter()
            optimizer.zero_grad()
            positive_scores = self.model.score_triples(positives)
            negative_scores = self.model.score_triples(negatives)
            if self.local_objective == "margin_ranking":
                repeated_positive_scores = (
                    positive_scores.repeat_interleave(
                        self.negative_sample_count,
                        dim=0,
                    )
                )
                loss = F.relu(
                    self.margin
                    + repeated_positive_scores
                    - negative_scores
                ).mean()
            else:
                loss = self_adversarial_loss(
                    positive_distances=positive_scores,
                    negative_distances=negative_scores.reshape(
                        int(positive_cpu.shape[0]),
                        self.negative_sample_count,
                    ),
                    sample_weights=sample_weights,
                    gamma=self.gamma,
                    temperature=self.adversarial_temperature,
                )
            loss.backward()
            optimizer.step()
            self.model.normalize_entity_embeddings()
            self._synchronize_for_timing()
            timing["forward_backward_seconds"] += (
                time.perf_counter() - started_at
            )
            batch_positive_count = int(positive_cpu.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * batch_positive_count
            positive_count += batch_positive_count
            directional_loss_sums[corruption_mode] += (
                float(loss.detach().cpu().item())
                * batch_positive_count
            )
            directional_counts[corruption_mode] += batch_positive_count
            self._direction_step += 1
        if positive_count <= 0:
            raise RuntimeError("训练DataLoader没有产生三元组")
        if not self.profile_training_timing:
            # 异步CUDA未同步时的分段墙钟不具备解释性，因此明确写为缺失值。
            timing = {
                key: float("nan") for key in timing
            }
        return {
            "train_loss": loss_sum / float(positive_count),
            "head_train_loss": (
                directional_loss_sums["head"]
                / float(directional_counts["head"])
                if directional_counts["head"] > 0
                else float("nan")
            ),
            "tail_train_loss": (
                directional_loss_sums["tail"]
                / float(directional_counts["tail"])
                if directional_counts["tail"] > 0
                else float("nan")
            ),
            "head_positive_count": float(
                directional_counts["head"]
            ),
            "tail_positive_count": float(
                directional_counts["tail"]
            ),
            **timing,
        }

    def _evaluate_validation(self, max_triples: int) -> Dict[str, float]:
        """使用指定采样上限计算验证集filtered指标。"""

        return self.evaluator.evaluate(
            self.model,
            self.dataset.valid_triples,
            self.device,
            max_triples=max_triples,
            seed=self.seed + 17,
            candidate_batch_size=self.candidate_batch_size,
            query_batch_size=self.query_batch_size,
            relation_stratified=(
                self.validation_selection
                == "relation_stratified"
            ),
        )

    def _metric_row(
        self,
        epoch: int,
        training_metrics: Dict[str, float],
        epoch_seconds: float,
        monitoring_metrics: Optional[Dict[str, float]],
        validation_metrics: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """构造字段固定的逐epoch CSV记录。"""

        monitoring_metrics = monitoring_metrics or {}
        validation_metrics = validation_metrics or {}
        return {
            "epoch": int(epoch),
            "train_loss": float(training_metrics["train_loss"]),
            "head_train_loss": float(
                training_metrics["head_train_loss"]
            ),
            "tail_train_loss": float(
                training_metrics["tail_train_loss"]
            ),
            "head_positive_count": float(
                training_metrics["head_positive_count"]
            ),
            "tail_positive_count": float(
                training_metrics["tail_positive_count"]
            ),
            "sampling_seconds": float(
                training_metrics["sampling_seconds"]
            ),
            "transfer_seconds": float(
                training_metrics["transfer_seconds"]
            ),
            "forward_backward_seconds": float(
                training_metrics["forward_backward_seconds"]
            ),
            "epoch_seconds": float(epoch_seconds),
            "monitor_mrr": float(
                monitoring_metrics.get("mrr", float("nan"))
            ),
            "monitor_hits_at_3": float(
                monitoring_metrics.get("hits_at_3", float("nan"))
            ),
            "monitor_evaluated_triple_count": float(
                monitoring_metrics.get(
                    "evaluated_triple_count", float("nan")
                )
            ),
            "val_mrr": float(
                validation_metrics.get("mrr", float("nan"))
            ),
            "val_mean_rank": float(
                validation_metrics.get("mean_rank", float("nan"))
            ),
            "val_hits_at_1": float(
                validation_metrics.get("hits_at_1", float("nan"))
            ),
            "val_hits_at_3": float(
                validation_metrics.get("hits_at_3", float("nan"))
            ),
            "val_hits_at_10": float(
                validation_metrics.get("hits_at_10", float("nan"))
            ),
            "val_evaluated_triple_count": float(
                validation_metrics.get(
                    "evaluated_triple_count", float("nan")
                )
            ),
        }

    def _print_epoch_progress(
        self,
        epoch: int,
        training_metrics: Dict[str, float],
        epoch_seconds: float,
        monitoring_metrics: Optional[Dict[str, float]],
        validation_metrics: Optional[Dict[str, float]],
    ) -> None:
        """向终端逐epoch打印损失、监控指标和正式选模指标。"""

        parts = [
            "[Epoch {}/{}]".format(epoch, self.epochs),
            "loss={:.6f}".format(
                float(training_metrics["train_loss"])
            ),
            "头损失={:.6f}".format(
                float(training_metrics["head_train_loss"])
            ),
            "尾损失={:.6f}".format(
                float(training_metrics["tail_train_loss"])
            ),
            "耗时={:.2f}秒".format(epoch_seconds),
        ]
        if self.profile_training_timing:
            parts.extend(
                [
                    "采样={:.2f}秒".format(
                        float(training_metrics["sampling_seconds"])
                    ),
                    "传输={:.2f}秒".format(
                        float(training_metrics["transfer_seconds"])
                    ),
                    "前后向={:.2f}秒".format(
                        float(
                            training_metrics[
                                "forward_backward_seconds"
                            ]
                        )
                    ),
                ]
            )
        if monitoring_metrics is not None:
            parts.extend(
                [
                    "监控MRR={:.6f}".format(
                        float(monitoring_metrics["mrr"])
                    ),
                    "监控Hits@3={:.6f}".format(
                        float(monitoring_metrics["hits_at_3"])
                    ),
                    "监控三元组={}".format(
                        int(
                            monitoring_metrics[
                                "evaluated_triple_count"
                            ]
                        )
                    ),
                ]
            )
        if validation_metrics is not None:
            parts.extend(
                [
                    "选模MRR={:.6f}".format(
                        float(validation_metrics["mrr"])
                    ),
                    "选模Hits@3={:.6f}".format(
                        float(validation_metrics["hits_at_3"])
                    ),
                    "选模三元组={}".format(
                        int(
                            validation_metrics[
                                "evaluated_triple_count"
                            ]
                        )
                    ),
                ]
            )
        print(" | ".join(parts), flush=True)

    def train(self, result_dir: Path) -> Dict[str, object]:
        """训练TransE、恢复最佳验证模型并返回最终验证测试汇总。"""

        result_dir = Path(result_dir).expanduser().resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = result_dir / "metrics.csv"
        self.model.to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        data_loader = self._build_data_loader()
        best_state = clone_state_dict(
            self.model.state_dict(), to_cpu=True
        )
        best_val_mrr = -math.inf
        best_epoch = 0
        evaluations_without_improvement = 0
        epochs_ran = 0

        with metrics_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as metrics_file:
            writer = csv.DictWriter(
                metrics_file, fieldnames=list(self.METRIC_FIELDS)
            )
            writer.writeheader()
            for epoch in range(1, self.epochs + 1):
                epoch_started_at = time.perf_counter()
                training_metrics = self._train_epoch(
                    data_loader, optimizer
                )
                monitoring_metrics = None
                if self.monitor_every_epoch:
                    monitoring_metrics = self._evaluate_validation(
                        self.monitor_validation_max_triples
                    )
                should_evaluate = should_run_selection_evaluation(
                    epoch,
                    self.eval_every,
                )
                validation_metrics = None
                if should_evaluate:
                    # 监控集与选模集完全相同时复用结果，避免重复评估。
                    if (
                        monitoring_metrics is not None
                        and self.monitor_validation_max_triples
                        == self.validation_max_triples
                    ):
                        validation_metrics = dict(monitoring_metrics)
                    else:
                        validation_metrics = self._evaluate_validation(
                            self.validation_max_triples
                        )
                    current_mrr = float(validation_metrics["mrr"])
                    if current_mrr > best_val_mrr:
                        best_val_mrr = current_mrr
                        best_epoch = epoch
                        best_state = clone_state_dict(
                            self.model.state_dict(), to_cpu=True
                        )
                        evaluations_without_improvement = 0
                    else:
                        evaluations_without_improvement += 1
                epoch_seconds = time.perf_counter() - epoch_started_at
                self._print_epoch_progress(
                    epoch,
                    training_metrics,
                    epoch_seconds,
                    monitoring_metrics,
                    validation_metrics,
                )
                writer.writerow(
                    self._metric_row(
                        epoch,
                        training_metrics,
                        epoch_seconds,
                        monitoring_metrics,
                        validation_metrics,
                    )
                )
                metrics_file.flush()
                epochs_ran = epoch
                if (
                    self.early_stopping_patience > 0
                    and evaluations_without_improvement
                    >= self.early_stopping_patience
                ):
                    break

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        final_validation = self._evaluate_validation(
            self.final_validation_max_triples
        )
        final_test = self.evaluator.evaluate(
            self.model,
            self.dataset.test_triples,
            self.device,
            max_triples=self.test_max_triples,
            seed=self.seed + 29,
            candidate_batch_size=self.candidate_batch_size,
            query_batch_size=self.query_batch_size,
        )
        return {
            "task": "centralized_knowledge_graph_completion",
            "runtime": "standalone_centralized_transe",
            "dataset": self.dataset.dataset_name,
            "local_objective": self.local_objective,
            "negative_sample_count": self.negative_sample_count,
            "negative_sampling_backend": (
                self.negative_sampler.backend
            ),
            "subsampling_weights_precomputed": True,
            "profile_training_timing": (
                self.profile_training_timing
            ),
            "fede_gamma": self.gamma,
            "adversarial_temperature": self.adversarial_temperature,
            "epochs_configured": self.epochs,
            "epochs_ran": epochs_ran,
            "best_epoch": best_epoch,
            "monitor_every_epoch": self.monitor_every_epoch,
            "monitor_validation_max_triples": (
                self.monitor_validation_max_triples
            ),
            "selection_eval_every": self.eval_every,
            "selection_validation_max_triples": (
                self.validation_max_triples
            ),
            "validation_selection": self.validation_selection,
            "evaluation_query_batch_size": self.query_batch_size,
            "evaluation_candidate_batch_size": self.candidate_batch_size,
            "best_validation_mrr_during_training": best_val_mrr,
            "final_validation_metrics": final_validation,
            "final_test_metrics": final_test,
            "metrics_file": str(metrics_path),
        }

