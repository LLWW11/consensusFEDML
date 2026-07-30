"""基于FedML ClientTrainer接口实现的TransE本地训练器。"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import torch
from fedml.core import ClientTrainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..core.device import as_bool
from ..core.randomness import derive_client_seed
from ..core.types import clone_state_dict
from ..tasks.kge.data import KnowledgeGraphDataset
from ..tasks.kge.federated_data import KnowledgeGraphClientPartition
from ..tasks.kge.negative_sampling import (
    LegacyFilteredNegativeSampler,
    VectorizedFilteredNegativeSampler,
)
from ..tasks.kge.objectives import self_adversarial_loss
from ..tasks.kge.subsampling import TripleFrequencySubsampler


class FedMLTransEModelTrainer(ClientTrainer):
    """让FedML客户端在本地正三元组上训练共享TransE模型。"""

    def __init__(
        self,
        model: torch.nn.Module,
        args,
        dataset: KnowledgeGraphDataset,
    ):
        """初始化共享模型、完整真三元组集合和本地训练指标。"""

        super().__init__(model, args)
        self.dataset = dataset
        self.last_train_metrics: Dict[str, object] = {}
        self._device_negative_sampler = (
            VectorizedFilteredNegativeSampler(
                dataset.num_entities,
                dataset.all_true_triples,
                seed=int(getattr(args, "random_seed", 0)) + 313,
                num_relations=dataset.num_relations,
            )
        )
        self._subsampling_weight_cache: Dict[int, torch.Tensor] = {}
        self._client_optimizer_state_mode = (
            self._resolve_optimizer_state_mode(args)
        )
        # 持久模式让每个客户端保留独立Adam动量，同时继续共享当前模型参数对象。
        self._client_optimizer_cache: Dict[
            int, torch.optim.Optimizer
        ] = {}

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """返回共享TransE当前参数的CPU深拷贝。"""

        return clone_state_dict(self.model.state_dict(), to_cpu=True)

    def set_model_params(
        self, model_parameters: Dict[str, torch.Tensor]
    ) -> None:
        """把云端下发的完整实体和关系嵌入加载到共享模型。"""

        self.model.load_state_dict(model_parameters)

    @staticmethod
    def _build_optimizer(model, args) -> torch.optim.Optimizer:
        """按配置为共享TransE参数创建一个Adam优化器。"""

        optimizer_name = str(
            getattr(args, "client_optimizer", "adam")
        ).strip().lower()
        if optimizer_name != "adam":
            raise ValueError(
                "阶段四client_optimizer必须是adam，实际为{}".format(
                    optimizer_name
                )
            )
        learning_rate = float(
            getattr(args, "learning_rate", getattr(args, "lr", 0.001))
        )
        if learning_rate <= 0.0:
            raise ValueError("learning_rate必须大于0")
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    @staticmethod
    def _resolve_optimizer_state_mode(args) -> str:
        """读取并校验客户端Adam状态采用重置或逐客户端持久模式。"""

        mode = str(
            getattr(args, "client_optimizer_state_mode", "reset")
        ).strip().lower()
        if mode not in {"reset", "persistent_per_client"}:
            raise ValueError(
                "client_optimizer_state_mode必须是reset或"
                "persistent_per_client"
            )
        return mode

    def _optimizer_for_client(
        self,
        client_id: int,
        args,
    ) -> Tuple[torch.optim.Optimizer, bool]:
        """返回本次客户端调用的Adam及其状态是否来自先前通信轮。"""

        runtime_mode = self._resolve_optimizer_state_mode(args)
        if runtime_mode != self._client_optimizer_state_mode:
            raise ValueError(
                "运行期间不能修改client_optimizer_state_mode"
            )
        if runtime_mode == "reset":
            return self._build_optimizer(self.model, args), False

        normalized_client_id = int(client_id)
        optimizer = self._client_optimizer_cache.get(
            normalized_client_id
        )
        reused = optimizer is not None
        if optimizer is None:
            optimizer = self._build_optimizer(self.model, args)
            self._client_optimizer_cache[
                normalized_client_id
            ] = optimizer
        return optimizer, reused

    @staticmethod
    def _optimizer_step(optimizer: torch.optim.Optimizer) -> float:
        """返回Adam参数状态中的最大步数，尚未更新时返回零。"""

        steps = []
        for state in optimizer.state.values():
            raw_step = state.get("step")
            if raw_step is None:
                continue
            if torch.is_tensor(raw_step):
                steps.append(float(raw_step.detach().cpu().item()))
            else:
                steps.append(float(raw_step))
        return max(steps) if steps else 0.0

    @property
    def optimizer_state_cache_size(self) -> int:
        """返回当前已经建立持久Adam状态的客户端数量。"""

        return len(self._client_optimizer_cache)

    @staticmethod
    def _synchronize_for_timing(
        device: torch.device, enabled: bool
    ) -> None:
        """仅在性能剖析开启时同步CUDA分段，避免正式训练额外阻塞。"""

        if bool(enabled) and torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))

    def _client_subsampling_weights(
        self, partition: KnowledgeGraphClientPartition
    ) -> torch.Tensor:
        """按客户端缓存与本地三元组行对齐的频率子采样权重。"""

        client_id = int(partition.client_id)
        cached = self._subsampling_weight_cache.get(client_id)
        expected_count = int(partition.train_triples.shape[0])
        if cached is None or int(cached.shape[0]) != expected_count:
            subsampler = TripleFrequencySubsampler(
                partition.train_triples,
                num_relations=self.dataset.num_relations,
            )
            cached = subsampler.precomputed_weights
            self._subsampling_weight_cache[client_id] = cached
        return cached

    def train(self, train_data, device, args) -> Dict[str, object]:
        """通过FedML生命周期完成一个知识客户端的本地TransE训练。"""

        if not isinstance(train_data, KnowledgeGraphClientPartition):
            raise TypeError(
                "FedMLTransEModelTrainer只接受KnowledgeGraphClientPartition"
            )
        local_epochs = int(getattr(args, "epochs", 1))
        batch_size = int(getattr(args, "batch_size", 1024))
        margin = float(getattr(args, "margin", 1.0))
        negative_sample_count = int(
            getattr(args, "negative_sample_count", 1)
        )
        local_objective = str(
            getattr(args, "local_objective", "margin_ranking")
        ).strip().lower()
        profile_training_timing = as_bool(
            getattr(args, "profile_training_timing", False)
        )
        if local_epochs <= 0:
            raise ValueError("本地epochs必须大于0")
        if batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        if margin <= 0.0:
            raise ValueError("margin必须大于0")
        if negative_sample_count <= 0:
            raise ValueError("negative_sample_count必须大于0")
        if local_objective not in {
            "margin_ranking",
            "fede_self_adversarial",
            "bidirectional_self_adversarial",
        }:
            raise ValueError(
                "local_objective必须是margin_ranking或"
                "fede_self_adversarial或"
                "bidirectional_self_adversarial"
            )

        client_seed = derive_client_seed(
            int(getattr(args, "random_seed", 0)),
            int(getattr(args, "round_idx", 0)),
            int(self.id),
        )
        generator = torch.Generator()
        generator.manual_seed(client_seed)
        device = torch.device(device)
        if local_objective == "bidirectional_self_adversarial":
            sample_weights = self._client_subsampling_weights(
                train_data
            )
        else:
            # 旧目标不消费该张量；占位张量不会改变V2批次和随机序列。
            sample_weights = torch.ones(
                int(train_data.train_triples.shape[0]),
                dtype=torch.float32,
            )
        data_loader = DataLoader(
            TensorDataset(
                train_data.train_triples,
                sample_weights,
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        if local_objective == "bidirectional_self_adversarial":
            # 37个客户端复用真三元组索引，仅为每次本地调用重置随机序列。
            negative_sampler = self._device_negative_sampler
            negative_sampler.reset_seed(client_seed + 313)
        else:
            negative_sampler = LegacyFilteredNegativeSampler(
                self.dataset.num_entities,
                self.dataset.all_true_triples,
                client_seed + 313,
            )
        self.model.to(device)
        optimizer, optimizer_state_reused = (
            self._optimizer_for_client(
                int(train_data.client_id),
                args,
            )
        )
        optimizer_step_before = self._optimizer_step(optimizer)
        loss_sum = 0.0
        positive_count = 0
        directional_loss_sums = {"head": 0.0, "tail": 0.0}
        directional_counts = {"head": 0, "tail": 0}
        timing = {
            "sampling_seconds": 0.0,
            "transfer_seconds": 0.0,
            "forward_backward_seconds": 0.0,
        }
        # 极小客户端每轮可能只有一个批次，因此把轮次纳入方向起点。
        batch_sequence = int(getattr(args, "round_idx", 0))
        for _ in range(local_epochs):
            for positive_cpu, sample_weight_cpu in data_loader:
                direction = (
                    "head" if batch_sequence % 2 == 0 else "tail"
                )
                self._synchronize_for_timing(
                    device, profile_training_timing
                )
                started_at = time.perf_counter()
                positives = positive_cpu.to(
                    device, non_blocking=True
                )
                weights = sample_weight_cpu.to(
                    device, non_blocking=True
                )
                self._synchronize_for_timing(
                    device, profile_training_timing
                )
                timing["transfer_seconds"] += (
                    time.perf_counter() - started_at
                )

                started_at = time.perf_counter()
                if local_objective == "bidirectional_self_adversarial":
                    negatives = negative_sampler.sample(
                        positives,
                        negative_sample_count,
                        corruption_mode=direction,
                    )
                else:
                    negatives_cpu = negative_sampler.sample(
                        positive_cpu,
                        negative_sample_count,
                        corruption_mode=(
                            "tail"
                            if local_objective
                            == "fede_self_adversarial"
                            else "head_tail"
                        ),
                    )
                    negatives = negatives_cpu.to(
                        device, non_blocking=True
                    )
                self._synchronize_for_timing(
                    device, profile_training_timing
                )
                timing["sampling_seconds"] += (
                    time.perf_counter() - started_at
                )

                started_at = time.perf_counter()
                self.model.train()
                optimizer.zero_grad()
                positive_scores = self.model.score_triples(positives)
                negative_scores = self.model.score_triples(negatives)
                if local_objective == "margin_ranking":
                    repeated_positive_scores = (
                        positive_scores.repeat_interleave(
                            negative_sample_count,
                            dim=0,
                        )
                    )
                    loss = F.relu(
                        margin
                        + repeated_positive_scores
                        - negative_scores
                    ).mean()
                elif local_objective == "fede_self_adversarial":
                    loss = self._fede_self_adversarial_loss(
                        positive_scores=positive_scores,
                        negative_scores=negative_scores,
                        positive_batch_size=int(
                            positive_cpu.shape[0]
                        ),
                        negative_sample_count=negative_sample_count,
                        args=args,
                    )
                else:
                    loss = self_adversarial_loss(
                        positive_distances=positive_scores,
                        negative_distances=negative_scores.reshape(
                            int(positive_cpu.shape[0]),
                            negative_sample_count,
                        ),
                        sample_weights=weights,
                        gamma=float(
                            getattr(args, "fede_gamma", 9.0)
                        ),
                        temperature=float(
                            getattr(
                                args,
                                "adversarial_temperature",
                                1.0,
                            )
                        ),
                    )
                loss.backward()
                optimizer.step()
                self.model.normalize_entity_embeddings()
                self._synchronize_for_timing(
                    device, profile_training_timing
                )
                timing["forward_backward_seconds"] += (
                    time.perf_counter() - started_at
                )
                batch_positive_count = int(positive_cpu.shape[0])
                loss_sum += (
                    float(loss.detach().cpu().item())
                    * batch_positive_count
                )
                positive_count += batch_positive_count
                metric_direction = (
                    "tail"
                    if local_objective == "fede_self_adversarial"
                    else direction
                )
                directional_loss_sums[metric_direction] += (
                    float(loss.detach().cpu().item())
                    * batch_positive_count
                )
                directional_counts[metric_direction] += (
                    batch_positive_count
                )
                batch_sequence += 1
        if positive_count <= 0:
            raise RuntimeError("知识客户端没有产生本地训练批次")
        optimizer_step_after = self._optimizer_step(optimizer)
        if not profile_training_timing:
            timing = {
                key: float("nan") for key in timing
            }
        self.last_train_metrics = {
            "train_loss": loss_sum / float(positive_count),
            "train_triple_count": float(train_data.triple_count),
            "optimizer_step_positive_count": float(positive_count),
            "client_optimizer_state_mode": (
                self._client_optimizer_state_mode
            ),
            "optimizer_state_reused": float(
                bool(optimizer_state_reused)
            ),
            "optimizer_step_before": optimizer_step_before,
            "optimizer_step_after": optimizer_step_after,
            "optimizer_state_cache_size": float(
                self.optimizer_state_cache_size
            ),
            "local_objective": local_objective,
            "negative_sampling_backend": (
                negative_sampler.backend
                if local_objective
                == "bidirectional_self_adversarial"
                else "legacy_numpy_per_triple"
            ),
            "subsampling_weights_precomputed": (
                local_objective
                == "bidirectional_self_adversarial"
            ),
            "profile_training_timing": profile_training_timing,
            **timing,
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
        }
        return dict(self.last_train_metrics)

    @staticmethod
    def _fede_self_adversarial_loss(
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
        positive_batch_size: int,
        negative_sample_count: int,
        args,
    ) -> torch.Tensor:
        """按FedE代码使用的自对抗逻辑目标计算一批本地损失。"""

        gamma = float(getattr(args, "fede_gamma", 10.0))
        temperature = float(
            getattr(args, "adversarial_temperature", 1.0)
        )
        if gamma <= 0.0:
            raise ValueError("fede_gamma必须大于0")
        if temperature <= 0.0:
            raise ValueError("adversarial_temperature必须大于0")

        # 当前模型输出距离；FedE分数为gamma减去TransE距离。
        positive_logits = gamma - positive_scores.reshape(
            int(positive_batch_size)
        )
        negative_logits = (
            gamma
            - negative_scores.reshape(
                int(positive_batch_size),
                int(negative_sample_count),
            )
        )
        adversarial_weights = F.softmax(
            negative_logits * temperature, dim=1
        ).detach()
        positive_loss = -F.logsigmoid(positive_logits).mean()
        negative_loss = -(
            adversarial_weights * F.logsigmoid(-negative_logits)
        ).sum(dim=1).mean()
        return (positive_loss + negative_loss) / 2.0

    def test(self, test_data, device, args) -> Dict[str, float]:
        """返回客户端本地正三元组的平均TransE距离。"""

        del args
        if not isinstance(test_data, KnowledgeGraphClientPartition):
            raise TypeError(
                "FedMLTransEModelTrainer只接受KnowledgeGraphClientPartition"
            )
        device = torch.device(device)
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            scores = self.model.score_triples(
                test_data.train_triples.to(device)
            )
        return {
            "test_total": int(test_data.triple_count),
            "mean_positive_score": float(scores.mean().cpu().item()),
        }
