"""TransE间隔损失和双向自对抗逻辑损失。"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def self_adversarial_loss(
    positive_distances: torch.Tensor,
    negative_distances: torch.Tensor,
    sample_weights: torch.Tensor,
    gamma: float,
    temperature: float,
) -> torch.Tensor:
    """按正样本频率和困难负样本分数计算自对抗逻辑损失。"""

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

    positive_logits = float(gamma) - positive_distances
    negative_logits = float(gamma) - negative_distances
    adversarial_weights = F.softmax(
        negative_logits * float(temperature),
        dim=1,
    ).detach()
    positive_losses = -F.logsigmoid(positive_logits)
    negative_losses = -(
        adversarial_weights * F.logsigmoid(-negative_logits)
    ).sum(dim=1)
    normalized_weights = sample_weights / sample_weights.sum().clamp_min(
        torch.finfo(sample_weights.dtype).eps
    )
    positive_loss = (normalized_weights * positive_losses).sum()
    negative_loss = (normalized_weights * negative_losses).sum()
    return (positive_loss + negative_loss) / 2.0
