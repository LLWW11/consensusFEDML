"""集中式实验的YAML读取、路径解析和配置校验。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIRS = {
    "fb15k-237": "../HFLSnF_KG_v5/data/FB15k-237",
    "wn18rr": "../HFLSnF_KG_v5/data/wn18rr",
}


def normalize_dataset_name(value: object) -> str:
    """规范化受支持的数据集名称，并兼容已有配置别名。"""

    name = str(value).strip().lower()
    aliases = {"fb15k237": "fb15k-237", "synthetic_kg": "synthetic-kg"}
    name = aliases.get(name, name)
    if name not in {"fb15k-237", "wn18rr", "synthetic-kg"}:
        raise ValueError("dataset必须是fb15k-237、wn18rr或synthetic-kg")
    return name


def resolve_dataset_path(config: Mapping[str, object]) -> Path:
    """优先使用显式路径，否则按数据集选择默认目录；不回退数据源。"""

    name = normalize_dataset_name(config["dataset"])
    value = config.get("data_dir")
    if value is None or not str(value).strip():
        if name not in DEFAULT_DATA_DIRS:
            raise ValueError("合成数据在内存构造，没有默认数据目录")
        value = DEFAULT_DATA_DIRS[name]
    return resolve_project_path(value)


def load_flat_config(config_path: Path) -> Dict[str, object]:
    """读取分节YAML并合并为训练器可直接访问的扁平配置。"""

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        sections = yaml.safe_load(handle)
    if not isinstance(sections, dict):
        raise ValueError("配置顶层必须是对象：{}".format(config_path))
    flattened: Dict[str, object] = {}
    for section_name, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError(
                "配置分节{}必须是对象".format(section_name)
            )
        flattened.update(section)
    flattened["config_file"] = str(config_path)
    validate_config(flattened)
    flattened["dataset"] = normalize_dataset_name(flattened["dataset"])
    if flattened["dataset"] != "synthetic-kg":
        # 快照保存实际使用的目录，便于在其他机器追溯数据来源。
        flattened["data_dir"] = str(resolve_dataset_path(flattened))
    flattened["run_name"] = str(flattened["run_name"]).replace(
        "{dataset}", flattened["dataset"]
    )
    return flattened


def validate_config(config: Mapping[str, object]) -> None:
    """校验集中式TransE正式训练所需的核心配置字段。"""

    required = (
        "random_seed",
        "dataset",
        "embedding_dim",
        "distance_norm",
        "epochs",
        "batch_size",
        "learning_rate",
        "negative_sample_count",
        "eval_every",
        "using_gpu",
        "require_cuda",
        "run_name",
        "result_root",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError("集中式配置缺少字段：{}".format(missing))
    normalize_dataset_name(config["dataset"])
    for field in (
        "embedding_dim",
        "epochs",
        "batch_size",
        "negative_sample_count",
        "eval_every",
    ):
        if int(config[field]) <= 0:
            raise ValueError("{}必须大于0".format(field))
    if int(config["distance_norm"]) not in {1, 2}:
        raise ValueError("distance_norm必须是1或2")
    if float(config["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate必须大于0")


def resolve_project_path(path_value: object) -> Path:
    """把相对路径解析为相对于KGE_central目录的绝对路径。"""

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PACKAGE_DIR / path).resolve()


def as_namespace(config: Mapping[str, object]) -> SimpleNamespace:
    """把已校验配置转换为兼容训练器属性访问的命名空间。"""

    return SimpleNamespace(**dict(config))
