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

from ...core.device import as_bool
from ...core.types import clone_state_dict
from .data import KnowledgeGraphDataset
from .evaluator import FilteredRankingEvaluator
from .model import TransE
from .negative_sampling import FilteredNegativeSampler


class CentralizedTransETrainer:
    """在完整训练三元组上训练TransE并使用filtered指标选模。"""

    METRIC_FIELDS = (
        "epoch",
        "train_loss",
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
        self.eval_every = int(getattr(args, "eval_every", 1))
        self.validation_max_triples = int(
            getattr(args, "validation_max_triples", 0)
        )
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
        self.early_stopping_patience = int(
            getattr(args, "early_stopping_patience", 0)
        )
        self.seed = int(getattr(args, "random_seed", 0))
        self._validate_hyperparameters()
        self.negative_sampler = FilteredNegativeSampler(
            dataset.num_entities,
            dataset.all_true_triples,
            self.seed + 1009,
        )
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
        if self.eval_every <= 0:
            raise ValueError("eval_every必须大于0")
        if self.monitor_validation_max_triples < 0:
            raise ValueError("monitor_validation_max_triples不能小于0")
        if self.candidate_batch_size <= 0:
            raise ValueError("evaluation_candidate_batch_size必须大于0")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience不能小于0")

    def _build_data_loader(self) -> DataLoader:
        """创建使用独立固定种子的训练三元组随机批次。"""

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            TensorDataset(self.dataset.train_triples),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

    def _train_epoch(
        self, data_loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """完成一个集中式epoch并返回按正三元组数加权的平均损失。"""

        self.model.train()
        loss_sum = 0.0
        positive_count = 0
        for (positive_cpu,) in data_loader:
            negatives_cpu = self.negative_sampler.sample(
                positive_cpu, self.negative_sample_count
            )
            positives = positive_cpu.repeat_interleave(
                self.negative_sample_count, dim=0
            ).to(self.device)
            negatives = negatives_cpu.to(self.device)
            optimizer.zero_grad()
            positive_scores = self.model.score_triples(positives)
            negative_scores = self.model.score_triples(negatives)
            loss = F.relu(
                self.margin + positive_scores - negative_scores
            ).mean()
            loss.backward()
            optimizer.step()
            self.model.normalize_entity_embeddings()
            batch_positive_count = int(positive_cpu.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * batch_positive_count
            positive_count += batch_positive_count
        if positive_count <= 0:
            raise RuntimeError("训练DataLoader没有产生三元组")
        return loss_sum / float(positive_count)

    def _evaluate_validation(self, max_triples: int) -> Dict[str, float]:
        """使用指定采样上限计算验证集filtered指标。"""

        return self.evaluator.evaluate(
            self.model,
            self.dataset.valid_triples,
            self.device,
            max_triples=max_triples,
            seed=self.seed + 17,
            candidate_batch_size=self.candidate_batch_size,
        )

    def _metric_row(
        self,
        epoch: int,
        train_loss: float,
        epoch_seconds: float,
        monitoring_metrics: Optional[Dict[str, float]],
        validation_metrics: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """构造字段固定的逐epoch CSV记录。"""

        monitoring_metrics = monitoring_metrics or {}
        validation_metrics = validation_metrics or {}
        return {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
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
        train_loss: float,
        epoch_seconds: float,
        monitoring_metrics: Optional[Dict[str, float]],
        validation_metrics: Optional[Dict[str, float]],
    ) -> None:
        """向终端逐epoch打印损失、监控指标和正式选模指标。"""

        parts = [
            "[Epoch {}/{}]".format(epoch, self.epochs),
            "loss={:.6f}".format(train_loss),
            "耗时={:.2f}秒".format(epoch_seconds),
        ]
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
                train_loss = self._train_epoch(data_loader, optimizer)
                monitoring_metrics = None
                if self.monitor_every_epoch:
                    monitoring_metrics = self._evaluate_validation(
                        self.monitor_validation_max_triples
                    )
                should_evaluate = (
                    epoch == 1
                    or epoch % self.eval_every == 0
                    or epoch == self.epochs
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
                    train_loss,
                    epoch_seconds,
                    monitoring_metrics,
                    validation_metrics,
                )
                writer.writerow(
                    self._metric_row(
                        epoch,
                        train_loss,
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
        )
        return {
            "task": "centralized_knowledge_graph_completion",
            "runtime": "fedml_configured_centralized_transe",
            "dataset": self.dataset.dataset_name,
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
            "best_validation_mrr_during_training": best_val_mrr,
            "final_validation_metrics": final_validation,
            "final_test_metrics": final_test,
            "metrics_file": str(metrics_path),
        }
