"""集中式实验的随机性、设备、结果目录和JSON工具。"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .configuration import resolve_project_path


def as_bool(value: object) -> bool:
    """把布尔值或常见字符串解析为严格布尔值。"""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("无法解析布尔值：{}".format(value))


def should_run_selection_evaluation(
    epoch: int,
    eval_every: int,
) -> bool:
    """判断当前epoch是否到达严格的周期性验证选模时点。"""

    normalized_epoch = int(epoch)
    normalized_interval = int(eval_every)
    if normalized_epoch <= 0:
        raise ValueError("epoch必须大于0")
    if normalized_interval <= 0:
        raise ValueError("eval_every必须大于0")
    # 只在完整周期点选模，避免epoch 1或最终轮特例破坏三组对齐。
    return normalized_epoch % normalized_interval == 0


def seed_everything(seed: int) -> None:
    """固定Python、NumPy、PyTorch CPU和CUDA随机种子。"""

    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)


def resolve_device(config: Mapping[str, object]) -> torch.device:
    """按配置选择CUDA或CPU，并在正式配置缺少CUDA时快速失败。"""

    using_gpu = as_bool(config.get("using_gpu", False))
    require_cuda = as_bool(config.get("require_cuda", False))
    gpu_id = int(config.get("gpu_id", 0))
    if using_gpu and torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                "gpu_id={}超出可见CUDA设备范围0..{}".format(
                    gpu_id,
                    torch.cuda.device_count() - 1,
                )
            )
        return torch.device("cuda:{}".format(gpu_id))
    if require_cuda:
        raise RuntimeError(
            "配置要求CUDA，但当前PyTorch没有可用CUDA；已拒绝降级到CPU"
        )
    return torch.device("cpu")


def create_result_directory(config: Mapping[str, object]) -> Path:
    """创建带微秒时间戳且不会覆盖旧实验的结果目录。"""

    result_root = resolve_project_path(config["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    run_name = str(config["run_name"]).strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = result_root / "{}_{}".format(run_name, timestamp)
    result_dir.mkdir(parents=False, exist_ok=False)
    return result_dir


def write_json(
    path: Path,
    payload: Mapping[str, object],
) -> Path:
    """把映射写为便于人工审计的UTF-8 JSON。"""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return path
