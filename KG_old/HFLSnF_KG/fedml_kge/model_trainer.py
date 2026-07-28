"""基于FedML ClientTrainer接口实现的TransE本地训练器。"""

from __future__ import annotations

from typing import Dict

import torch
from fedml.core import ClientTrainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..core.randomness import derive_client_seed
from ..core.types import clone_state_dict
from ..tasks.kge.data import KnowledgeGraphDataset
from ..tasks.kge.federated_data import KnowledgeGraphClientPartition
from ..tasks.kge.negative_sampling import FilteredNegativeSampler


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
        self.last_train_metrics: Dict[str, float] = {}

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
        """按配置创建每次本地调用独立的Adam优化器。"""

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

    def train(self, train_data, device, args) -> Dict[str, float]:
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
        if local_epochs <= 0:
            raise ValueError("本地epochs必须大于0")
        if batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        if margin <= 0.0:
            raise ValueError("margin必须大于0")
        if negative_sample_count <= 0:
            raise ValueError("negative_sample_count必须大于0")

        client_seed = derive_client_seed(
            int(getattr(args, "random_seed", 0)),
            int(getattr(args, "round_idx", 0)),
            int(self.id),
        )
        generator = torch.Generator()
        generator.manual_seed(client_seed)
        data_loader = DataLoader(
            TensorDataset(train_data.train_triples),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        negative_sampler = FilteredNegativeSampler(
            self.dataset.num_entities,
            self.dataset.all_true_triples,
            client_seed + 313,
        )
        device = torch.device(device)
        self.model.to(device)
        optimizer = self._build_optimizer(self.model, args)
        loss_sum = 0.0
        positive_count = 0
        for _ in range(local_epochs):
            for (positive_cpu,) in data_loader:
                negatives_cpu = negative_sampler.sample(
                    positive_cpu, negative_sample_count
                )
                positives = positive_cpu.repeat_interleave(
                    negative_sample_count, dim=0
                ).to(device)
                negatives = negatives_cpu.to(device)
                self.model.train()
                optimizer.zero_grad()
                positive_scores = self.model.score_triples(positives)
                negative_scores = self.model.score_triples(negatives)
                loss = F.relu(
                    margin + positive_scores - negative_scores
                ).mean()
                loss.backward()
                optimizer.step()
                self.model.normalize_entity_embeddings()
                batch_positive_count = int(positive_cpu.shape[0])
                loss_sum += (
                    float(loss.detach().cpu().item())
                    * batch_positive_count
                )
                positive_count += batch_positive_count
        if positive_count <= 0:
            raise RuntimeError("知识客户端没有产生本地训练批次")
        self.last_train_metrics = {
            "train_loss": loss_sum / float(positive_count),
            "train_triple_count": float(train_data.triple_count),
            "optimizer_step_positive_count": float(positive_count),
        }
        return dict(self.last_train_metrics)

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
