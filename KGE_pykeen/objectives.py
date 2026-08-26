"""PyKEEN NSSA损失的频率加权适配与独立参考公式。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .pykeen_bridge import require_pykeen


def self_adversarial_loss(
    positive_distances: torch.Tensor,
    negative_distances: torch.Tensor,
    sample_weights: torch.Tensor,
    gamma: float,
    temperature: float,
) -> torch.Tensor:
    """按原工程公式计算频率加权自对抗逻辑损失。"""

    return _validate_and_compute_weighted_nssa(
        positive_distances=positive_distances,
        negative_distances=negative_distances,
        sample_weights=sample_weights,
        gamma=gamma,
        temperature=temperature,
    )


def _validate_and_compute_weighted_nssa(
    positive_distances: torch.Tensor,
    negative_distances: torch.Tensor,
    sample_weights: torch.Tensor,
    gamma: float,
    temperature: float,
) -> torch.Tensor:
    """校验张量并执行与PyKEEN 1.10.1一致的逐项NSSA公式。"""

    if positive_distances.ndim != 1:
        raise ValueError("正样本距离必须是一维张量")
    if negative_distances.ndim != 2:
        raise ValueError("负样本距离必须是二维张量")
    if int(negative_distances.shape[0]) != int(
        positive_distances.shape[0]
    ):
        raise ValueError("正负样本批次大小不一致")
    if sample_weights.ndim != 1 or int(sample_weights.shape[0]) != int(
        positive_distances.shape[0]
    ):
        raise ValueError("子采样权重形状与正样本批次不一致")
    if float(gamma) <= 0.0:
        raise ValueError("gamma必须大于0")
    if float(temperature) <= 0.0:
        raise ValueError("temperature必须大于0")

    # PyKEEN分数是负距离；下面保持其NSSALoss的分数方向。
    positive_scores = -positive_distances
    negative_scores = -negative_distances
    adversarial_weights = F.softmax(
        negative_scores.detach() * float(temperature),
        dim=1,
    )
    positive_losses = -F.logsigmoid(float(gamma) + positive_scores)
    negative_losses = -(
        adversarial_weights
        * F.logsigmoid(-negative_scores - float(gamma))
    ).sum(dim=1)
    normalized_weights = sample_weights / sample_weights.sum().clamp_min(
        torch.finfo(sample_weights.dtype).eps
    )
    return (
        (normalized_weights * positive_losses).sum()
        + (normalized_weights * negative_losses).sum()
    ) / 2.0


class WeightedNSSALossAdapter(nn.Module):
    """为PyKEEN NSSALoss增加原工程的逐正样本频率权重。"""

    def __init__(self, gamma: float, temperature: float):
        """创建并保存固定版本PyKEEN NSSALoss合同。"""

        super().__init__()
        require_pykeen()
        from pykeen.losses import NSSALoss

        self.pykeen_loss = NSSALoss(
            margin=float(gamma),
            adversarial_temperature=float(temperature),
            reduction="mean",
        )

    def forward(
        self,
        positive_distances: torch.Tensor,
        negative_distances: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        """使用PyKEEN参数语义执行严格配方的加权归约。"""

        # PyKEEN原生接口没有逐正样本权重，因此仅适配外层归约。
        return _validate_and_compute_weighted_nssa(
            positive_distances=positive_distances,
            negative_distances=negative_distances,
            sample_weights=sample_weights,
            gamma=self.pykeen_loss.margin,
            temperature=self.pykeen_loss.inverse_softmax_temperature,
        )
