"""全局随机种子和客户端本地训练种子的统一管理。"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """统一设置Python、NumPy、PyTorch和CUDA随机种子。"""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_client_seed(base_seed: int, round_index: int, client_id: int) -> int:
    """从实验种子、通信轮和客户端编号确定性派生本地训练种子。"""

    modulus = 2147483647
    derived = (
        int(base_seed) * 1000003
        + int(round_index) * 1009
        + int(client_id) * 9176
        + 97
    ) % modulus
    return int(derived)
