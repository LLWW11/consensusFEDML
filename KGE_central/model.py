"""TransE知识图谱嵌入模型。"""

from __future__ import annotations

import math

import torch
from torch import nn


class TransE(nn.Module):
    """使用头实体加关系接近尾实体的距离函数评价三元组。"""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        distance_norm: int = 1,
    ):
        """创建实体和关系嵌入表并初始化参数。"""

        super().__init__()
        if int(num_entities) <= 1:
            raise ValueError("TransE至少需要两个实体")
        if int(num_relations) <= 0:
            raise ValueError("TransE至少需要一个关系")
        if int(embedding_dim) <= 0:
            raise ValueError("embedding_dim必须大于0")
        if int(distance_norm) not in {1, 2}:
            raise ValueError("distance_norm必须是1或2")
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.distance_norm = int(distance_norm)
        self.entity_embeddings = nn.Embedding(
            self.num_entities, self.embedding_dim
        )
        self.relation_embeddings = nn.Embedding(
            self.num_relations, self.embedding_dim
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """按TransE常用范围均匀初始化并归一化实体嵌入。"""

        bound = 6.0 / math.sqrt(float(self.embedding_dim))
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        self.normalize_entity_embeddings()

    @torch.no_grad()
    def normalize_entity_embeddings(self) -> None:
        """将每一行实体向量投影到单位L2球面。"""

        normalized = torch.nn.functional.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
        self.entity_embeddings.weight.data.copy_(normalized)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        """返回一批三元组的TransE距离，距离越小表示可信度越高。"""

        if triples.dtype != torch.long:
            raise TypeError("TransE三元组必须使用torch.long")
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("TransE输入形状必须为[N, 3]")
        heads = self.entity_embeddings(triples[:, 0])
        relations = self.relation_embeddings(triples[:, 1])
        tails = self.entity_embeddings(triples[:, 2])
        return torch.linalg.vector_norm(
            heads + relations - tails,
            ord=self.distance_norm,
            dim=1,
        )

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """按照PyTorch模块接口计算三元组距离。"""

        return self.score_triples(triples)


