"""FedML设备选择以及服务器CUDA强制约束。"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def as_bool(value) -> bool:
    """把YAML、字符串或数值配置稳定转换成布尔值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def resolve_fedml_device(
    args,
    fedml_device_getter: Optional[Callable] = None,
    cuda_available: Optional[bool] = None,
) -> torch.device:
    """通过FedML解析设备，并禁止正式CUDA配置静默退回CPU。"""

    using_gpu = as_bool(getattr(args, "using_gpu", False))
    require_cuda = as_bool(getattr(args, "require_cuda", False))
    if require_cuda and not using_gpu:
        raise RuntimeError("require_cuda=true 时必须同时设置 using_gpu=true。")

    detected_cuda = (
        torch.cuda.is_available()
        if cuda_available is None
        else bool(cuda_available)
    )
    if require_cuda and not detected_cuda:
        raise RuntimeError(
            "正式配置要求CUDA，但当前PyTorch未检测到可用GPU；"
            "请检查服务器驱动和CUDA版PyTorch。"
        )

    if fedml_device_getter is None:
        import fedml

        fedml_device_getter = fedml.device.get_device
    device = torch.device(fedml_device_getter(args))
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "正式配置要求CUDA，但FedML实际返回设备{}。".format(device)
        )
    return device
