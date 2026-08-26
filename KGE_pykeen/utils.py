"""集中式训练使用的模型状态工具。"""

from __future__ import annotations

from typing import Dict, Mapping

import torch


def clone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    to_cpu: bool = True,
) -> Dict[str, torch.Tensor]:
    """深拷贝模型状态，并按需移动到CPU以降低显存占用。"""

    cloned: Dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        tensor = value.detach().clone()
        if to_cpu:
            tensor = tensor.cpu()
        cloned[str(name)] = tensor
    return cloned
