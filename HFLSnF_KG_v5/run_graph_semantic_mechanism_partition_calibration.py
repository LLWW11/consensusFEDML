"""生成V5语义—图局部性双消融的六分区冻结校准合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping

from .tasks.kge import (
    DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
    SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
    load_fb15k237,
    partition_train_triples_by_graph_local_no_primary,
    partition_train_triples_by_semantic_domain_graph_local,
    partition_train_triples_by_semantic_domain_no_graph_local,
    relation_domains,
)
from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    write_json_report,
)


SUITE_NAME = "v5_graph_semantic_mechanism_ablation_fedadam_formal150"
SEEDS = (42, 2024, 2025)
CLIENT_COUNT = 37
LOAD_TOLERANCE = 0.05
SEARCH_RESTARTS = 8
CONFIG_DIR = (
    PACKAGE_DIR / "configs" / "graph_semantic_mechanism_ablation"
).resolve()
REFERENCE_DIR = (CONFIG_DIR / "frozen_full_v5_reference").resolve()
DEFAULT_OUTPUT = (CONFIG_DIR / "partition_calibration_contract.json").resolve()
REFERENCE_HASHES = {
    "partition_calibration_contract.json": (
        "5d8668ee2999ffc36a3303fdc5ca027b35588b3bce98465d6969c8481a9929b7"
    ),
    "batch_summary.json": (
        "4a710ffa1a74ec6d5370f2ee0bd5cf312591e5f6c34d21d2c8e241362e30a159"
    ),
    "official3_summary.json": (
        "e154cea42d65265092530e0106a4c7c8f55c836f0d671982ad820062bc2bc7b5"
    ),
    "graph_semantic_summary.json": (
        "c4caf8900168179b55080d337c047fa4cc649765e73d0935e0942272727d30a1"
    ),
    "full_v5_unit_metrics.json": (
        "ae4be0cbc40a22542b04980e0c14544ddbdbaf41070fe9205f03cb077bfc55f7"
    ),
}
ARM_BUILDERS = {
    "graph_only": partition_train_triples_by_graph_local_no_primary,
    "semantic_only": partition_train_triples_by_semantic_domain_no_graph_local,
}
ARM_STRATEGIES = {
    "graph_only": DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
    "semantic_only": SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
}


def _file_sha256(path: Path) -> str:
    """分块计算文件的SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, object]:
    """读取UTF-8 JSON对象并拒绝非对象顶层结构。"""

    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("JSON顶层必须是对象：{}".format(path))
    return payload


def _validate_frozen_references() -> Dict[str, object]:
    """校验四份完整V5冻结参考及其固定哈希。"""

    files = {}
    for name, expected_hash in REFERENCE_HASHES.items():
        path = (REFERENCE_DIR / name).resolve()
        if not path.is_file():
            raise FileNotFoundError("缺少完整V5冻结参考：{}".format(path))
        actual_hash = _file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError("完整V5冻结参考哈希漂移：{}".format(name))
        files[name] = {
            "relative_path": (
                "configs/graph_semantic_mechanism_ablation/"
                "frozen_full_v5_reference/{}"
            ).format(name),
            "sha256": actual_hash,
        }
    calibration = _load_json(
        REFERENCE_DIR / "partition_calibration_contract.json"
    )
    official = _load_json(REFERENCE_DIR / "official3_summary.json")
    analysis = _load_json(REFERENCE_DIR / "graph_semantic_summary.json")
    control_metrics = _load_json(REFERENCE_DIR / "full_v5_unit_metrics.json")
    if calibration.get("status") != "passed":
        raise ValueError("完整V5冻结校准合同状态必须为passed")
    if official.get("status") != "passed":
        raise ValueError("完整V5冻结官方评估状态必须为passed")
    if analysis.get("status") != "passed":
        raise ValueError("完整V5冻结分析摘要状态必须为passed")
    if control_metrics.get("status") != "passed":
        raise ValueError("完整V5冻结控制指标状态必须为passed")
    return {
        "files": files,
        "partition_contract": calibration,
        "official_summary": official,
        "analysis_summary": analysis,
    }


def _packet_ownership_valid(dataset, partition) -> bool:
    """检查同一语义域和头实体包没有跨客户端拆分。"""

    domain_by_relation = relation_domains(dataset)
    owners = {}
    for client in partition.partitions:
        for head_id, relation_id, _ in client.train_triples.tolist():
            key = (domain_by_relation[int(relation_id)], int(head_id))
            existing = owners.setdefault(key, int(client.client_id))
            if existing != int(client.client_id):
                return False
    return True


def _common_checks(
    summary: Mapping[str, object],
    packet_ownership_valid: bool,
) -> Dict[str, bool]:
    """构造两个消融臂共享的结构和负载门禁。"""

    return {
        "triple_completeness": int(
            summary["total_train_triple_count"]
        ) == 272115,
        "client_count": int(summary["client_count"]) == CLIENT_COUNT,
        "nonempty_clients": int(summary["min_client_triple_count"]) > 0,
        "load_tolerance": float(
            summary["max_relative_load_deviation"]
        ) <= LOAD_TOLERANCE + 1e-12,
        "domain_head_packet_exclusive": bool(packet_ownership_valid),
        "packet_count": int(summary["domain_head_packet_count"]) == 33155,
    }


def _arm_checks(
    arm: str,
    summary: Mapping[str, object],
    full_primary_domains: Mapping[str, object],
) -> Dict[str, bool]:
    """构造A无主域和B主域配额一致性的机制门禁。"""

    if arm == "graph_only":
        return {
            "has_no_primary_domain": (
                summary["has_primary_domain"] is False
                and summary["client_primary_domains"] is None
                and summary[
                    "triple_weighted_primary_domain_fraction"
                ] is None
            ),
            "uses_entity_locality": (
                summary["uses_entity_locality_objective"] is True
            ),
            "objective_order": summary["assignment_objective_order"] == [
                "new_entity_count",
                "current_load",
                "seeded_tie_rank",
            ],
        }
    expected = Counter(str(value) for value in full_primary_domains)
    actual = Counter(str(value) for value in summary["client_primary_domains"])
    return {
        "has_primary_domain": summary["has_primary_domain"] is True,
        "primary_domain_quota_matches_full": actual == expected,
        "does_not_use_entity_locality": (
            summary["uses_entity_locality_objective"] is False
        ),
        "objective_order": summary["assignment_objective_order"] == [
            "primary_domain_mismatch",
            "current_load",
            "seeded_tie_rank",
        ],
    }


def _full_reference_candidates(
    frozen: Mapping[str, object],
) -> Mapping[str, object]:
    """返回完整V5冻结校准合同中的三种子候选。"""

    contract = frozen["partition_contract"]
    candidates = contract.get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("完整V5冻结校准合同缺少candidates对象")
    return candidates


def build_calibration_contract(
    search_restarts: int = SEARCH_RESTARTS,
    reproduce: bool = True,
) -> Dict[str, object]:
    """计算A/B六个分区、机制门禁和完整V5哈希回归。"""

    normalized_restarts = int(search_restarts)
    if normalized_restarts <= 0:
        raise ValueError("search_restarts必须大于0")
    frozen = _validate_frozen_references()
    full_candidates = _full_reference_candidates(frozen)
    dataset = load_fb15k237(PACKAGE_DIR / "data" / "FB15k-237")
    candidates: Dict[str, Dict[str, object]] = {
        arm: {} for arm in ARM_BUILDERS
    }
    all_hashes = []
    all_checks = []
    for arm, builder in ARM_BUILDERS.items():
        for seed in SEEDS:
            partition = builder(
                dataset=dataset,
                client_count=CLIENT_COUNT,
                seed=seed,
                load_tolerance=LOAD_TOLERANCE,
                search_restarts=normalized_restarts,
            )
            summary = partition.summary()
            full_summary = full_candidates[str(seed)]["summary"]
            checks = _common_checks(
                summary,
                _packet_ownership_valid(dataset, partition),
            )
            checks.update(
                _arm_checks(
                    arm,
                    summary,
                    full_summary["client_primary_domains"],
                )
            )
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                raise RuntimeError(
                    "{} seed{}校准门禁失败：{}".format(arm, seed, failed)
                )
            candidates[arm][str(seed)] = {
                "summary": summary,
                "checks": checks,
            }
            all_hashes.append(partition.partition_hash)
            all_checks.extend(checks.values())
    unique_hashes = len(set(all_hashes)) == len(all_hashes)
    if not unique_hashes:
        raise RuntimeError("A/B六个正式分区哈希必须互不相同")

    reproduction: Dict[str, Dict[str, object]] = {
        arm: {} for arm in ARM_BUILDERS
    }
    if reproduce:
        for arm, builder in ARM_BUILDERS.items():
            for seed in SEEDS:
                reproduced = builder(
                    dataset=dataset,
                    client_count=CLIENT_COUNT,
                    seed=seed,
                    load_tolerance=LOAD_TOLERANCE,
                    search_restarts=normalized_restarts,
                )
                expected = str(
                    candidates[arm][str(seed)]["summary"]["partition_hash"]
                )
                reproduction[arm][str(seed)] = {
                    "partition_hash": reproduced.partition_hash,
                    "expected_partition_hash": expected,
                    "passed": reproduced.partition_hash == expected,
                }
        if not all(
            item["passed"]
            for arm_values in reproduction.values()
            for item in arm_values.values()
        ):
            raise RuntimeError("A/B分区独立哈希复算失败")

    full_reproduction = {}
    for seed in SEEDS:
        reproduced = partition_train_triples_by_semantic_domain_graph_local(
            dataset=dataset,
            client_count=CLIENT_COUNT,
            seed=seed,
            load_tolerance=LOAD_TOLERANCE,
            search_restarts=normalized_restarts,
        )
        expected = str(full_candidates[str(seed)]["summary"]["partition_hash"])
        full_reproduction[str(seed)] = {
            "partition_hash": reproduced.partition_hash,
            "expected_partition_hash": expected,
            "passed": reproduced.partition_hash == expected,
        }
    if not all(item["passed"] for item in full_reproduction.values()):
        raise RuntimeError("完整V5冻结分区哈希回归失败")

    data_files = {}
    for name in ("train.txt", "valid.txt", "test.txt"):
        path = PACKAGE_DIR / "data" / "FB15k-237" / name
        data_files[name] = {
            "relative_path": "data/FB15k-237/{}".format(name),
            "sha256": _file_sha256(path),
        }
    reproduced_ok = reproduce and all(
        item["passed"]
        for arm_values in reproduction.values()
        for item in arm_values.values()
    )
    status = (
        "passed"
        if reproduced_ok
        and all(all_checks)
        and unique_hashes
        and all(item["passed"] for item in full_reproduction.values())
        else "unverified"
    )
    return {
        "contract_schema_version": 1,
        "suite": SUITE_NAME,
        "status": status,
        "dataset": dataset.dataset_name,
        "dataset_summary": dataset.summary(),
        "data_files": data_files,
        "domain_extractor": "freebase_top_level",
        "packet_unit": "semantic_domain_head_entity",
        "client_count": CLIENT_COUNT,
        "seeds": list(SEEDS),
        "constraints": {
            "load_tolerance": LOAD_TOLERANCE,
            "search_restarts": normalized_restarts,
        },
        "strategies": dict(ARM_STRATEGIES),
        "frozen_full_v5_reference": frozen["files"],
        "candidates": candidates,
        "unique_partition_hashes": unique_hashes,
        "reproduction": reproduction,
        "full_v5_partition_reproduction": full_reproduction,
        "interpretation_boundary": (
            "仅比较完整V5分别删除语义主域或实体图局部目标后的边际变化；"
            "没有双删除实验臂，不分解二因素交互效应。"
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """创建双消融无训练校准命令行参数。"""

    parser = argparse.ArgumentParser(
        description="生成V5语义—图局部性双消融六分区校准合同"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="校准合同JSON输出路径",
    )
    parser.add_argument(
        "--search-restarts",
        type=int,
        default=SEARCH_RESTARTS,
        help="每个实验臂和种子的确定性搜索重启次数",
    )
    parser.add_argument(
        "--skip-reproduction",
        action="store_true",
        help="仅供开发探测；跳过后合同状态不会成为passed",
    )
    return parser


def main() -> None:
    """执行双消融无训练校准并写出UTF-8合同。"""

    args = build_argument_parser().parse_args()
    contract = build_calibration_contract(
        search_restarts=int(args.search_restarts),
        reproduce=not bool(args.skip_reproduction),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_report(contract, output)
    print("双消融分区校准状态：{}".format(contract["status"]))
    print("校准合同：{}".format(output))


if __name__ == "__main__":
    main()
