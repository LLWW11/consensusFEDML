"""PyKEEN TransE模型与现有集中式训练接口之间的适配层。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Optional, Tuple

import torch
from torch import nn

from .pykeen_bridge import build_pykeen_transe, extract_embedding_weights


class _EmbeddingWeightView(nn.Module):
    """仅公开weight属性且不重复注册参数的嵌入兼容视图。"""

    def __init__(self, weight: torch.Tensor):
        """保存底层权重引用。"""

        super().__init__()
        object.__setattr__(self, "_weight_reference", weight)

    @property
    def weight(self) -> torch.Tensor:
        """返回底层PyKEEN嵌入权重。"""

        return object.__getattribute__(self, "_weight_reference")


class PyKEENTransEView(nn.Module):
    """把PyKEEN TransE公开为原工程使用的正距离接口。"""

    def __init__(self, pykeen_model: nn.Module, distance_norm: int):
        """保存PyKEEN模型并校验距离范数。"""

        super().__init__()
        if int(distance_norm) not in {1, 2}:
            raise ValueError("distance_norm必须是1或2")
        self.pykeen_model = pykeen_model
        self.distance_norm = int(distance_norm)
        self.num_entities = int(pykeen_model.num_entities)
        self.num_relations = int(pykeen_model.num_relations)

    @property
    def entity_embeddings(self) -> nn.Module:
        """返回兼容旧评估器的实体嵌入模块。"""

        entity_weight, _ = extract_embedding_weights(self.pykeen_model)
        return _EmbeddingWeightView(entity_weight)

    @property
    def relation_embeddings(self) -> nn.Module:
        """返回兼容旧评估器的关系嵌入模块。"""

        _, relation_weight = extract_embedding_weights(self.pykeen_model)
        return _EmbeddingWeightView(relation_weight)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        """返回正TransE距离，使其与原工程的距离语义一致。"""

        if triples.dtype != torch.long:
            raise TypeError("TransE三元组必须使用torch.long")
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("TransE输入形状必须为[N, 3]")
        # PyKEEN的TransE分数越大越好，因此取反恢复正距离。
        return -self.pykeen_model.score_hrt(triples).reshape(-1)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """按照PyTorch模块接口计算正TransE距离。"""

        return self.score_triples(triples)

    @torch.no_grad()
    def normalize_entity_embeddings(self) -> None:
        """调用PyKEEN参数更新钩子执行实体L2约束。"""

        self.pykeen_model.post_parameter_update()

    def canonical_state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        """返回与原工程检查点键名一致的规范化参数。"""

        entity_weight, relation_weight = extract_embedding_weights(
            self.pykeen_model
        )
        return OrderedDict(
            (
                ("entity_embeddings.weight", entity_weight),
                ("relation_embeddings.weight", relation_weight),
            )
        )

    def state_dict(self, *args, **kwargs):
        """让原训练器保存规范化的实体和关系权重。"""

        keep_vars = bool(kwargs.get("keep_vars", False))
        return OrderedDict(
            (name, tensor if keep_vars else tensor.detach())
            for name, tensor in self.canonical_state_dict().items()
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ):
        """从规范化键名恢复PyKEEN实体和关系权重。"""

        expected = {
            "entity_embeddings.weight",
            "relation_embeddings.weight",
        }
        actual = set(state_dict)
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        if strict and (missing or unexpected):
            raise RuntimeError(
                "规范化检查点键不匹配，缺少{}，多出{}".format(
                    missing, unexpected
                )
            )
        entity_weight, relation_weight = extract_embedding_weights(
            self.pykeen_model
        )
        with torch.no_grad():
            if "entity_embeddings.weight" in state_dict:
                entity_weight.copy_(state_dict["entity_embeddings.weight"])
            if "relation_embeddings.weight" in state_dict:
                relation_weight.copy_(
                    state_dict["relation_embeddings.weight"]
                )
        return torch.nn.modules.module._IncompatibleKeys(
            missing, unexpected
        )


class TransE(PyKEENTransEView):
    """使用已验证初始化配方构造PyKEEN TransE。"""

    def __init__(
        self,
        triples_factory,
        embedding_dim: int,
        distance_norm: int = 1,
        random_seed: Optional[int] = None,
    ):
        """创建PyKEEN TransE并立即执行实体归一化。"""

        model = build_pykeen_transe(
            triples_factory=triples_factory,
            embedding_dim=int(embedding_dim),
            distance_norm=int(distance_norm),
            random_seed=random_seed,
        )
        super().__init__(model, distance_norm=distance_norm)
        self.embedding_dim = int(embedding_dim)
        self.normalize_entity_embeddings()


def canonical_weights(
    model: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回任意支持模型的实体和关系权重。"""

    if isinstance(model, PyKEENTransEView):
        return extract_embedding_weights(model.pykeen_model)
    return extract_embedding_weights(model)
