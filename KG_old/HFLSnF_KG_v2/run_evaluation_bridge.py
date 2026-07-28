"""从终端或IDE运行FedE与当前TransE检查点的统一评估桥接。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

from .tasks.kge.data import KnowledgeGraphDataset, load_fb15k237
from .tasks.kge.evaluation_bridge import (
    CommonHoldout,
    FedEAssignment,
    FedEDataBundle,
    TransEEmbeddingBundle,
    bootstrap_mrr_interval,
    bootstrap_paired_delta_interval,
    build_common_holdout,
    evaluate_fede_original_protocol,
    evaluate_global_protocol,
    hash_json,
    hash_triples,
    load_fede_data_bundle,
    load_fede_embedding_bundle,
    load_project_embedding_bundle,
    select_common_test_assignments,
    select_fede_test_assignments,
    sha256_file,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def _as_bool(value: object) -> bool:
    """把YAML中的布尔、字符串或数值转换成明确布尔值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _flatten_yaml_sections(payload: Mapping[str, object]) -> Dict[str, object]:
    """把与FedML配置相似的分区YAML合并为一层参数字典。"""

    flattened: Dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and str(key).endswith("_args"):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def _resolve_package_path(path_value: object) -> Path:
    """把相对路径解析为相对于HFLSnF_KG_v2目录的绝对路径。"""

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PACKAGE_DIR / path).resolve()


@dataclass(frozen=True)
class EvaluationBridgeSettings:
    """保存无需重训评估桥接的全部运行参数。"""

    config_path: Path
    stage: str
    standard_data_dir: Path
    fede_data_path: Path
    fede_checkpoint_path: Path
    project_checkpoints: Tuple[Dict[str, object], ...]
    using_gpu: bool
    gpu_id: int
    require_cuda: bool
    result_root: Path
    run_name: str
    max_fede_triples: int
    max_common_triples: int
    selection_seed: int
    query_batch_size: int
    candidate_batch_size: int
    progress_every: int
    bootstrap_samples: int
    bootstrap_seed: int
    reference_model: str
    expected_values: Dict[str, object]

    @classmethod
    def from_yaml(cls, path: Path) -> "EvaluationBridgeSettings":
        """读取简体中文桥接配置并解析所有相对路径。"""

        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("找不到评估桥接配置：{}".format(path))
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise TypeError("评估桥接YAML顶层必须是对象")
        values = _flatten_yaml_sections(raw)
        checkpoints = values.get("project_checkpoints", [])
        if not isinstance(checkpoints, list):
            raise TypeError("project_checkpoints必须是列表")
        normalized_checkpoints: List[Dict[str, object]] = []
        for index, item in enumerate(checkpoints):
            if not isinstance(item, dict):
                raise TypeError(
                    "project_checkpoints第{}项必须是对象".format(index)
                )
            if "name" not in item or "path" not in item:
                raise ValueError(
                    "project_checkpoints第{}项必须包含name和path".format(
                        index
                    )
                )
            normalized = dict(item)
            normalized["name"] = str(item["name"]).strip()
            normalized["path"] = str(
                _resolve_package_path(item["path"])
            )
            normalized["optional"] = _as_bool(item.get("optional", False))
            normalized_checkpoints.append(normalized)

        expected_values = values.get("expected_values", {})
        if not isinstance(expected_values, dict):
            raise TypeError("expected_values必须是对象")
        settings = cls(
            config_path=path,
            stage=str(values.get("stage", "all")).strip().lower(),
            standard_data_dir=_resolve_package_path(
                values.get("standard_data_dir", "data/FB15k-237")
            ),
            fede_data_path=_resolve_package_path(
                values.get(
                    "fede_data_path",
                    "../1paperAbout/FedE-master/data/FB15k237-Fed3.pkl",
                )
            ),
            fede_checkpoint_path=_resolve_package_path(
                values.get(
                    "fede_checkpoint_path",
                    "../1paperAbout/FedE-master/state/"
                    "fb15k237_fed3_transe_fede.best",
                )
            ),
            project_checkpoints=tuple(normalized_checkpoints),
            using_gpu=_as_bool(values.get("using_gpu", False)),
            gpu_id=int(values.get("gpu_id", 0)),
            require_cuda=_as_bool(values.get("require_cuda", False)),
            result_root=_resolve_package_path(
                values.get("result_root", "results")
            ),
            run_name=str(
                values.get("run_name", "evaluation_bridge")
            ).strip(),
            max_fede_triples=int(values.get("max_fede_triples", 0)),
            max_common_triples=int(values.get("max_common_triples", 0)),
            selection_seed=int(values.get("selection_seed", 42)),
            query_batch_size=int(values.get("query_batch_size", 16)),
            candidate_batch_size=int(
                values.get("candidate_batch_size", 4096)
            ),
            progress_every=int(values.get("progress_every", 100)),
            bootstrap_samples=int(values.get("bootstrap_samples", 1000)),
            bootstrap_seed=int(values.get("bootstrap_seed", 2026)),
            reference_model=str(
                values.get("reference_model", "centralized_fast")
            ).strip(),
            expected_values=dict(expected_values),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """校验阶段名称、批次、抽样和输出名称等基础约束。"""

        if self.stage not in {"data", "fede", "common", "all"}:
            raise ValueError("stage必须是data、fede、common或all")
        if self.require_cuda and not self.using_gpu:
            raise ValueError(
                "require_cuda=true时必须同时设置using_gpu=true"
            )
        if self.gpu_id < 0:
            raise ValueError("gpu_id不能为负数")
        for name, value in (
            ("max_fede_triples", self.max_fede_triples),
            ("max_common_triples", self.max_common_triples),
            ("bootstrap_samples", self.bootstrap_samples),
        ):
            if int(value) < 0:
                raise ValueError("{}不能为负数".format(name))
        if self.query_batch_size <= 0 or self.candidate_batch_size <= 0:
            raise ValueError("查询批次和候选批次必须大于0")
        if not self.run_name:
            raise ValueError("run_name不能为空")

    def snapshot(self) -> Dict[str, object]:
        """返回隐藏了Python对象细节的可复现配置快照。"""

        return {
            "config_path": str(self.config_path),
            "stage": self.stage,
            "standard_data_dir": str(self.standard_data_dir),
            "fede_data_path": str(self.fede_data_path),
            "fede_checkpoint_path": str(self.fede_checkpoint_path),
            "project_checkpoints": [
                dict(item) for item in self.project_checkpoints
            ],
            "using_gpu": self.using_gpu,
            "gpu_id": self.gpu_id,
            "require_cuda": self.require_cuda,
            "result_root": str(self.result_root),
            "run_name": self.run_name,
            "max_fede_triples": self.max_fede_triples,
            "max_common_triples": self.max_common_triples,
            "selection_seed": self.selection_seed,
            "query_batch_size": self.query_batch_size,
            "candidate_batch_size": self.candidate_batch_size,
            "progress_every": self.progress_every,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "reference_model": self.reference_model,
            "expected_values": dict(self.expected_values),
        }


def resolve_bridge_device(settings: EvaluationBridgeSettings) -> torch.device:
    """在读取大数据和检查点前解析CPU或CUDA设备并快速失败。"""

    if settings.using_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "评估配置要求GPU，但当前PyTorch没有检测到CUDA；"
                "请安装与服务器驱动匹配的CUDA版PyTorch。"
            )
        device = torch.device("cuda:{}".format(settings.gpu_id))
        if settings.require_cuda and device.type != "cuda":
            raise RuntimeError("正式评估配置要求CUDA")
        return device
    if settings.require_cuda:
        raise RuntimeError("正式评估配置禁止回退CPU")
    return torch.device("cpu")


def _create_result_directory(settings: EvaluationBridgeSettings) -> Path:
    """在V2结果根目录创建不会覆盖旧结果的时间戳目录。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = (
        settings.result_root / "{}_{}".format(settings.run_name, timestamp)
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def _write_json(path: Path, payload: object) -> None:
    """以UTF-8和缩进格式写入JSON结果。"""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def _reverse_mapping(mapping: Mapping[str, int]) -> Dict[int, str]:
    """把名称到编号映射转换成编号到名称映射并校验唯一性。"""

    reversed_mapping = {int(value): str(key) for key, value in mapping.items()}
    if len(reversed_mapping) != len(mapping):
        raise ValueError("实体或关系编号映射不是一一对应")
    return reversed_mapping


def _write_common_tsv(
    path: Path,
    assignments: Sequence[FedEAssignment],
    dataset: KnowledgeGraphDataset,
) -> None:
    """写出公共留出集的编号、名称和FedE客户端归属。"""

    id_to_entity = _reverse_mapping(dataset.entity_to_id)
    id_to_relation = _reverse_mapping(dataset.relation_to_id)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "head_id",
                "relation_id",
                "tail_id",
                "client_id",
                "local_relation_id",
                "head",
                "relation",
                "tail",
            ]
        )
        for head, relation, tail, client_id, local_relation in assignments:
            writer.writerow(
                [
                    head,
                    relation,
                    tail,
                    client_id,
                    local_relation,
                    id_to_entity[int(head)],
                    id_to_relation[int(relation)],
                    id_to_entity[int(tail)],
                ]
            )


def _write_rank_csv(
    path: Path, records: Sequence[Mapping[str, object]]
) -> None:
    """把所有协议和模型的逐查询排名写入一个CSV文件。"""

    fieldnames = [
        "stage",
        "model",
        "protocol",
        "head",
        "relation",
        "tail",
        "client_id",
        "direction",
        "rank",
        "reciprocal_rank",
    ]
    sorted_records = sorted(
        records,
        key=lambda row: (
            str(row.get("stage", "")),
            str(row.get("model", "")),
            int(row.get("head", 0)),
            int(row.get("relation", 0)),
            int(row.get("tail", 0)),
            str(row.get("direction", "")),
        ),
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(
                {field: record.get(field) for field in fieldnames}
            )


def _validate_expected(
    actual: object, expected: object, label: str
) -> None:
    """对黄金数量或哈希执行快速失败校验。"""

    if expected is None:
        return
    if isinstance(actual, str):
        matches = str(actual).lower() == str(expected).lower()
    elif isinstance(actual, float):
        matches = abs(float(actual) - float(expected)) <= 1e-12
    else:
        matches = actual == expected
    if not matches:
        raise RuntimeError(
            "{}不符合预期：实际{}，预期{}".format(label, actual, expected)
        )


def _validate_data_goldens(
    settings: EvaluationBridgeSettings,
    fede_data: FedEDataBundle,
    holdout: CommonHoldout,
) -> None:
    """根据配置核验Fed3文件、全集和公共留出集黄金值。"""

    expected = settings.expected_values
    summary = holdout.summary()
    checks = (
        (
            fede_data.source_sha256,
            expected.get("fede_data_sha256"),
            "FedE数据文件SHA-256",
        ),
        (
            hash_triples(fede_data.all_true_triples),
            expected.get("universe_hash"),
            "三元组全集哈希",
        ),
        (
            summary["common_valid_count"],
            expected.get("common_valid_count"),
            "公共验证集数量",
        ),
        (
            summary["common_valid_hash"],
            expected.get("common_valid_hash"),
            "公共验证集哈希",
        ),
        (
            summary["common_test_count"],
            expected.get("common_test_count"),
            "公共测试集数量",
        ),
        (
            summary["common_test_hash"],
            expected.get("common_test_hash"),
            "公共测试集哈希",
        ),
    )
    for actual, expected_value, label in checks:
        _validate_expected(actual, expected_value, label)


def _build_data_audit(
    settings: EvaluationBridgeSettings,
    dataset: KnowledgeGraphDataset,
    fede_data: FedEDataBundle,
    holdout: CommonHoldout,
) -> Dict[str, object]:
    """汇总文件身份、直接交叉测试泄漏和公共集覆盖范围。"""

    standard_train = set(map(tuple, dataset.train_triples.tolist()))
    standard_valid = set(map(tuple, dataset.valid_triples.tolist()))
    standard_test = set(map(tuple, dataset.test_triples.tolist()))
    fede_train = set(fede_data.triples("train"))
    fede_valid = set(fede_data.triples("valid"))
    fede_test = set(fede_data.triples("test"))
    common_valid = set(holdout.valid_triples)
    common_test = set(holdout.test_triples)
    common_valid_entities = {
        entity
        for head, _, tail in common_valid
        for entity in (head, tail)
    }
    common_test_entities = {
        entity
        for head, _, tail in common_test
        for entity in (head, tail)
    }
    standard_valid_entities = {
        entity
        for head, _, tail in standard_valid
        for entity in (head, tail)
    }
    standard_test_entities = {
        entity
        for head, _, tail in standard_test
        for entity in (head, tail)
    }
    return {
        "schema_version": 1,
        "standard_dataset": dataset.summary(),
        "standard_files": {
            split: {
                "path": str(settings.standard_data_dir / "{}.txt".format(split)),
                "sha256": sha256_file(
                    settings.standard_data_dir / "{}.txt".format(split)
                ),
            }
            for split in ("train", "valid", "test")
        },
        "entity_mapping_hash": hash_json(dataset.entity_to_id),
        "relation_mapping_hash": hash_json(dataset.relation_to_id),
        "fede_data": fede_data.summary(),
        "common_holdout": holdout.summary(),
        "unsafe_cross_test_leakage": {
            "standard_test_seen_in_fede_train_count": len(
                standard_test.intersection(fede_train)
            ),
            "standard_test_seen_in_fede_train_fraction": (
                len(standard_test.intersection(fede_train))
                / float(len(standard_test))
            ),
            "fede_test_seen_in_standard_train_count": len(
                fede_test.intersection(standard_train)
            ),
            "fede_test_seen_in_standard_train_fraction": (
                len(fede_test.intersection(standard_train))
                / float(len(fede_test))
            ),
        },
        "common_valid_coverage": {
            "standard_valid_fraction": len(common_valid)
            / float(len(standard_valid)),
            "fede_valid_fraction": len(common_valid) / float(len(fede_valid)),
            "relation_count": len({row[1] for row in common_valid}),
            "entity_count": len(common_valid_entities),
            "standard_valid_entity_fraction": len(common_valid_entities)
            / float(len(standard_valid_entities)),
        },
        "common_test_coverage": {
            "standard_test_fraction": len(common_test)
            / float(len(standard_test)),
            "fede_test_fraction": len(common_test) / float(len(fede_test)),
            "relation_count": len({row[1] for row in common_test}),
            "entity_count": len(common_test_entities),
            "standard_test_entity_fraction": len(common_test_entities)
            / float(len(standard_test_entities)),
        },
        "strict_exclusion_checks": {
            "common_valid_vs_standard_other": len(
                common_valid.intersection(standard_train.union(standard_test))
            ),
            "common_valid_vs_fede_other": len(
                common_valid.intersection(fede_train.union(fede_test))
            ),
            "common_test_vs_standard_train_valid": len(
                common_test.intersection(standard_train.union(standard_valid))
            ),
            "common_test_vs_fede_train_valid": len(
                common_test.intersection(fede_train.union(fede_valid))
            ),
        },
        "usage_note": (
            "公共测试集只用于在同一批查询上校准检查点差异，"
            "不能替代各自完整标准测试集。"
        ),
    }


def _load_project_models(
    settings: EvaluationBridgeSettings,
    dataset: KnowledgeGraphDataset,
) -> List[TransEEmbeddingBundle]:
    """按配置加载所有存在且映射一致的当前项目检查点。"""

    models: List[TransEEmbeddingBundle] = []
    seen_names = set()
    for item in settings.project_checkpoints:
        name = str(item["name"]).strip()
        if not name or name in seen_names:
            raise ValueError("项目检查点名称不能为空或重复")
        seen_names.add(name)
        path = Path(str(item["path"]))
        optional = _as_bool(item.get("optional", False))
        if not path.exists() and optional:
            print("跳过不存在的可选检查点：{}".format(path), flush=True)
            continue
        override = item.get("distance_norm")
        models.append(
            load_project_embedding_bundle(
                name=name,
                path_or_directory=path,
                dataset=dataset,
                distance_norm_override=(
                    int(override) if override is not None else None
                ),
            )
        )
    return models


def _evaluate_fede_stages(
    settings: EvaluationBridgeSettings,
    fede_model: TransEEmbeddingBundle,
    fede_data: FedEDataBundle,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """依次运行E0及E1a、E1b、E1c，并复用E1头尾排名。"""

    e0_summary, e0_records = evaluate_fede_original_protocol(
        fede_model,
        fede_data,
        device=device,
        max_triples=settings.max_fede_triples,
        seed=settings.selection_seed,
        query_batch_size=settings.query_batch_size,
        candidate_batch_size=settings.candidate_batch_size,
        progress_every=settings.progress_every,
    )
    selected = select_fede_test_assignments(
        fede_data,
        maximum=settings.max_fede_triples,
        seed=settings.selection_seed,
    )
    triples = [(row[0], row[1], row[2]) for row in selected]
    clients = [row[3] for row in selected]
    e1_summary, e1_records, _ = evaluate_global_protocol(
        stage="E1",
        bundle=fede_model,
        triples=triples,
        all_true_triples=fede_data.all_true_triples,
        device=device,
        query_batch_size=settings.query_batch_size,
        candidate_batch_size=settings.candidate_batch_size,
        client_ids=clients,
        progress_every=settings.progress_every,
    )
    expected_e0 = settings.expected_values.get("e0_mrr")
    tolerance = float(
        settings.expected_values.get("e0_mrr_tolerance", 0.0001)
    )
    if (
        expected_e0 is not None
        and settings.max_fede_triples == 0
        and abs(
            float(e0_summary["metrics"]["mrr"]) - float(expected_e0)
        )
        > tolerance
    ):
        raise RuntimeError(
            "E0完整MRR未复现FedE日志：实际{:.8f}，预期{:.8f}±{}".format(
                float(e0_summary["metrics"]["mrr"]),
                float(expected_e0),
                tolerance,
            )
        )
    return (
        {
            "E0": e0_summary,
            "E1a_global_tail_only": e1_summary["tail_metrics"],
            "E1b_global_head_only": e1_summary["head_metrics"],
            "E1c_global_head_tail": e1_summary["combined_metrics"],
            "E1_protocol": {
                key: value
                for key, value in e1_summary.items()
                if key
                not in {"head_metrics", "tail_metrics", "combined_metrics"}
            },
        },
        e0_records + e1_records,
    )


def _evaluate_common_stage(
    settings: EvaluationBridgeSettings,
    models: Sequence[TransEEmbeddingBundle],
    fede_data: FedEDataBundle,
    holdout: CommonHoldout,
    device: torch.device,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """在严格公共测试集上统一评估FedE和当前项目检查点。"""

    selected = select_common_test_assignments(
        holdout,
        maximum=settings.max_common_triples,
        seed=settings.selection_seed,
    )
    triples = [(row[0], row[1], row[2]) for row in selected]
    client_ids = [row[3] for row in selected]
    summaries: Dict[str, object] = {}
    all_records: List[Dict[str, object]] = []
    per_model_scores: Dict[str, object] = {}
    for model in models:
        summary, records, per_triple_mrr = evaluate_global_protocol(
            stage="E2",
            bundle=model,
            triples=triples,
            all_true_triples=fede_data.all_true_triples,
            device=device,
            query_batch_size=settings.query_batch_size,
            candidate_batch_size=settings.candidate_batch_size,
            client_ids=client_ids,
            progress_every=settings.progress_every,
        )
        summary["bootstrap_mrr"] = bootstrap_mrr_interval(
            per_triple_mrr,
            bootstrap_samples=settings.bootstrap_samples,
            seed=settings.bootstrap_seed,
        )
        summaries[model.name] = summary
        per_model_scores[model.name] = per_triple_mrr
        all_records.extend(records)

    reference_name = settings.reference_model
    paired: Dict[str, object] = {}
    if reference_name in per_model_scores:
        reference_scores = per_model_scores[reference_name]
        for model_name, scores in per_model_scores.items():
            if model_name == reference_name:
                continue
            paired[model_name] = bootstrap_paired_delta_interval(
                scores,
                reference_scores,
                bootstrap_samples=settings.bootstrap_samples,
                seed=settings.bootstrap_seed,
            )
    return (
        {
            "selected_common_test_count": len(selected),
            "selected_common_test_hash": hash_triples(triples),
            "reference_model": (
                reference_name if reference_name in per_model_scores else None
            ),
            "models": summaries,
            "paired_delta_vs_reference": paired,
        },
        all_records,
    )


def run_evaluation_bridge(
    settings: EvaluationBridgeSettings,
) -> Tuple[Path, Dict[str, object]]:
    """执行数据审计、E0/E1/E2评估并写出完整桥接结果。"""

    # 正式配置必须在读取22MB文本和pickle前确认CUDA可用。
    device = resolve_bridge_device(settings)
    print("评估桥接设备：{}".format(device), flush=True)
    result_dir = _create_result_directory(settings)
    _write_json(result_dir / "config_snapshot.json", settings.snapshot())

    expected_data_hash = settings.expected_values.get("fede_data_sha256")
    if expected_data_hash is not None:
        actual_hash = sha256_file(settings.fede_data_path)
        _validate_expected(
            actual_hash, expected_data_hash, "FedE数据文件SHA-256"
        )
    expected_checkpoint_hash = settings.expected_values.get(
        "fede_checkpoint_sha256"
    )
    if expected_checkpoint_hash is not None:
        actual_hash = sha256_file(settings.fede_checkpoint_path)
        _validate_expected(
            actual_hash, expected_checkpoint_hash, "FedE检查点SHA-256"
        )

    print("读取标准FB15k-237和FedE Fed3数据……", flush=True)
    dataset = load_fb15k237(settings.standard_data_dir)
    fede_data = load_fede_data_bundle(
        settings.fede_data_path,
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
    )
    holdout = build_common_holdout(dataset, fede_data)
    _validate_data_goldens(settings, fede_data, holdout)
    data_audit = _build_data_audit(
        settings, dataset, fede_data, holdout
    )
    _write_json(result_dir / "data_audit.json", data_audit)
    _write_common_tsv(
        result_dir / "common_valid.tsv",
        holdout.valid_assignments,
        dataset,
    )
    _write_common_tsv(
        result_dir / "common_test.tsv",
        holdout.test_assignments,
        dataset,
    )
    print(
        "严格公共验证/测试集：{} / {} 条".format(
            len(holdout.valid_assignments),
            len(holdout.test_assignments),
        ),
        flush=True,
    )

    models: List[TransEEmbeddingBundle] = []
    fede_model: Optional[TransEEmbeddingBundle] = None
    if settings.stage != "data":
        fede_model = load_fede_embedding_bundle(
            settings.fede_checkpoint_path, fede_data, name="fede"
        )
        models.append(fede_model)
    if settings.stage in {"common", "all"}:
        models.extend(_load_project_models(settings, dataset))
    _write_json(
        result_dir / "model_manifest.json",
        {
            "models": [model.summary() for model in models],
            "manifest_hash": hash_json(
                [model.summary() for model in models]
            ),
        },
    )

    protocol_metrics: Dict[str, object] = {}
    rank_records: List[Dict[str, object]] = []
    if settings.stage in {"fede", "all"}:
        if fede_model is None:
            raise RuntimeError("E0/E1缺少FedE检查点")
        print("开始E0和E1：只评估FedE检查点，不训练。", flush=True)
        fede_metrics, records = _evaluate_fede_stages(
            settings, fede_model, fede_data, device
        )
        protocol_metrics.update(fede_metrics)
        rank_records.extend(records)
    if settings.stage in {"common", "all"}:
        if fede_model is None:
            raise RuntimeError("E2缺少FedE检查点")
        print("开始E2：在严格公共测试集统一复评。", flush=True)
        common_metrics, records = _evaluate_common_stage(
            settings,
            models,
            fede_data,
            holdout,
            device,
        )
        protocol_metrics["E2_common_test"] = common_metrics
        rank_records.extend(records)

    _write_json(result_dir / "protocol_metrics.json", protocol_metrics)
    _write_rank_csv(result_dir / "query_ranks.csv", rank_records)
    summary = {
        "status": "completed",
        "stage": settings.stage,
        "device": str(device),
        "result_dir": str(result_dir),
        "training_performed": False,
        "standard_dataset_summary": dataset.summary(),
        "fede_data_summary": fede_data.summary(),
        "common_holdout_summary": holdout.summary(),
        "evaluated_models": [model.summary() for model in models],
        "rank_record_count": len(rank_records),
        "protocol_metrics": protocol_metrics,
        "output_files": [
            "config_snapshot.json",
            "data_audit.json",
            "common_valid.tsv",
            "common_test.tsv",
            "model_manifest.json",
            "protocol_metrics.json",
            "query_ranks.csv",
            "summary.json",
        ],
    }
    _write_json(result_dir / "summary.json", summary)
    return result_dir, summary


def build_argument_parser() -> argparse.ArgumentParser:
    """创建同时支持终端与IDE的轻量级命令行解析器。"""

    parser = argparse.ArgumentParser(
        description="无需重训地统一复评FedE与HFLSnF_KG TransE检查点"
    )
    parser.add_argument(
        "--cf",
        type=str,
        default=str(
            PACKAGE_DIR / "configs" / "evaluation_bridge_smoke_cpu.yaml"
        ),
        help="评估桥接YAML配置路径",
    )
    parser.add_argument(
        "--stage",
        choices=["data", "fede", "common", "all"],
        default=None,
        help="可选：临时覆盖YAML中的运行阶段",
    )
    return parser


def main() -> None:
    """读取YAML并启动不会修改任何检查点的评估桥接。"""

    parser = build_argument_parser()
    arguments = parser.parse_args()
    settings = EvaluationBridgeSettings.from_yaml(Path(arguments.cf))
    if arguments.stage is not None:
        # dataclass被冻结，使用构造器显式替换单个字段。
        settings = EvaluationBridgeSettings(
            **{
                **settings.__dict__,
                "stage": str(arguments.stage),
            }
        )
        settings.validate()
    result_dir, summary = run_evaluation_bridge(settings)
    print("评估桥接完成，未执行任何训练。", flush=True)
    print("结果目录：{}".format(result_dir), flush=True)
    if "E2_common_test" in summary["protocol_metrics"]:
        models = summary["protocol_metrics"]["E2_common_test"]["models"]
        print("严格公共测试集MRR：", flush=True)
        for model_name, result in models.items():
            print(
                "  {}：{:.6f}".format(
                    model_name,
                    float(result["combined_metrics"]["mrr"]),
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
