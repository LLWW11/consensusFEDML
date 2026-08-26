"""PyKEEN版本、数据工厂、初始化和审计信息的共享桥接工具。"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml


EXPECTED_PYKEEN_VERSION = "1.10.1"


def installed_distribution_version(distribution_name: str) -> str:
    """从Python发行包元数据读取指定依赖的版本。"""

    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version
    return str(version(str(distribution_name)))


def installed_pykeen_version() -> str:
    """从Python发行包元数据读取已安装的PyKEEN版本。"""

    return installed_distribution_version("pykeen")


def require_pykeen():
    """导入并校验固定的PyKEEN版本。"""

    try:
        import pykeen
    except ImportError as error:
        raise RuntimeError(
            "KGE_pykeen需要pykeen=={}；请先按README创建服务器环境".format(
                EXPECTED_PYKEEN_VERSION
            )
        ) from error
    actual_version = installed_pykeen_version()
    if actual_version != EXPECTED_PYKEEN_VERSION:
        raise RuntimeError(
            "PyKEEN版本必须是{}，实际为{}".format(
                EXPECTED_PYKEEN_VERSION,
                actual_version,
            )
        )
    return pykeen


def verified_uniform_initializer(embedding_dim: int):
    """创建与原TransE范围一致的PyKEEN初始化函数。"""

    dimension = int(embedding_dim)
    if dimension <= 0:
        raise ValueError("embedding_dim必须大于0")
    bound = 6.0 / float(dimension) ** 0.5

    def initialize(tensor: torch.Tensor) -> torch.Tensor:
        """在原强基线的均匀范围内原位初始化张量。"""

        return torch.nn.init.uniform_(tensor, -bound, bound)

    return initialize


def build_triples_factories(
    dataset,
    validation_triples: Optional[torch.Tensor] = None,
):
    """使用现有全局映射构造共享编号的训练验证测试工厂。"""

    require_pykeen()
    from pykeen.triples import TriplesFactory

    training = TriplesFactory(
        mapped_triples=dataset.train_triples,
        entity_to_id=dict(dataset.entity_to_id),
        relation_to_id=dict(dataset.relation_to_id),
        create_inverse_triples=False,
    )
    validation = training.clone_and_exchange_triples(
        mapped_triples=(
            dataset.valid_triples
            if validation_triples is None
            else validation_triples
        ),
        create_inverse_triples=False,
    )
    testing = training.clone_and_exchange_triples(
        mapped_triples=dataset.test_triples,
        create_inverse_triples=False,
    )
    return training, validation, testing


def build_pykeen_transe(
    triples_factory,
    embedding_dim: int,
    distance_norm: int,
    random_seed: Optional[int],
):
    """按原强基线初始化分布构造PyKEEN TransE。"""

    require_pykeen()
    from pykeen.models import TransE as PyKEENTransE

    initializer = verified_uniform_initializer(embedding_dim)
    return PyKEENTransE(
        triples_factory=triples_factory,
        embedding_dim=int(embedding_dim),
        scoring_fct_norm=int(distance_norm),
        entity_initializer=initializer,
        entity_constrainer=torch.nn.functional.normalize,
        relation_initializer=initializer,
        relation_constrainer=None,
        random_seed=(None if random_seed is None else int(random_seed)),
    )


def extract_embedding_weights(
    pykeen_model,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从PyKEEN 1.10.1 TransE中提取底层嵌入权重。"""

    try:
        entity = pykeen_model.entity_representations[0]
        relation = pykeen_model.relation_representations[0]
        return entity._embeddings.weight, relation._embeddings.weight
    except (AttributeError, IndexError) as error:
        raise TypeError("无法识别PyKEEN TransE嵌入结构") from error


def _mapping_sha256(mapping: Mapping[str, int]) -> str:
    """计算字符串到整数映射的稳定SHA-256。"""

    digest = hashlib.sha256()
    for label, value in sorted(mapping.items(), key=lambda item: item[1]):
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _triples_sha256(triples: torch.Tensor) -> str:
    """计算三元组张量的稳定SHA-256。"""

    value = triples.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def dataset_contract(dataset) -> Dict[str, object]:
    """返回数据映射和三个划分的审计哈希。"""

    return {
        "dataset": dataset.dataset_name,
        "entity_mapping_sha256": _mapping_sha256(dataset.entity_to_id),
        "relation_mapping_sha256": _mapping_sha256(
            dataset.relation_to_id
        ),
        "train_triples_sha256": _triples_sha256(dataset.train_triples),
        "valid_triples_sha256": _triples_sha256(dataset.valid_triples),
        "test_triples_sha256": _triples_sha256(dataset.test_triples),
        **dataset.summary(),
    }


def environment_snapshot(device: torch.device) -> Dict[str, object]:
    """采集可复现实验所需的解释器和核心依赖版本。"""

    require_pykeen()
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(torch.device(device)),
        "pykeen_version": installed_pykeen_version(),
        "torch_max_mem_version": installed_distribution_version(
            "torch-max-mem"
        ),
        "numpy_version": np.__version__,
        "pyyaml_version": yaml.__version__,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if torch.device(device).type == "cuda"
            else None
        ),
    }


def canonical_model_sha256(pykeen_model) -> str:
    """计算PyKEEN实体和关系权重的规范化SHA-256。"""

    entity_weight, relation_weight = extract_embedding_weights(
        pykeen_model
    )
    digest = hashlib.sha256()
    for name, tensor in (
        ("entity_embeddings.weight", entity_weight),
        ("relation_embeddings.weight", relation_weight),
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def flatten_metric_results(metric_results) -> Dict[str, float]:
    """把PyKEEN指标对象转换为稳定的扁平数值映射。"""

    if hasattr(metric_results, "to_flat_dict"):
        raw = metric_results.to_flat_dict()
    else:
        raw = metric_results.to_dict()
    flattened: Dict[str, float] = {}

    def visit(prefix: str, value) -> None:
        """递归展开PyKEEN可能返回的嵌套指标结构。"""

        if isinstance(value, Mapping):
            for key, child in value.items():
                child_prefix = (
                    "{}.{}".format(prefix, key) if prefix else str(key)
                )
                visit(child_prefix, child)
        else:
            try:
                flattened[prefix] = float(value)
            except (TypeError, ValueError):
                return

    visit("", raw)
    return flattened
