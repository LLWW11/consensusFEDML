"""生成V5图语义客户端划分的三种子冻结校准合同。"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .tasks.kge import (
    load_fb15k237,
    partition_train_triples_by_head,
    partition_train_triples_by_semantic_domain_graph_local,
    relation_domains,
    semantic_graph_statistics,
)
from .tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    write_json_report,
)


SEEDS = (42, 2024, 2025)
CLIENT_COUNT = 37
LOAD_TOLERANCE = 0.05
PURITY_IMPROVEMENT = 0.20
SEARCH_RESTARTS = 8
DEFAULT_OUTPUT = (
    PACKAGE_DIR
    / "configs"
    / "graph_semantic"
    / "partition_calibration_contract.json"
)


def _file_sha256(path: Path) -> str:
    """分块计算一个文件的SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dominant_domains(dataset, partitions) -> Sequence[str]:
    """返回各客户端训练三元组数量最多的顶级语义域。"""

    domain_by_relation = relation_domains(dataset)
    values = []
    for partition in partitions:
        counts = Counter(
            domain_by_relation[int(relation_id)]
            for relation_id in partition.train_triples[:, 1].tolist()
        )
        values.append(
            sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
        )
    return tuple(values)


def _packet_ownership_valid(dataset, partition) -> bool:
    """检查同一域和头实体数据包没有分配给多个客户端。"""

    domain_by_relation = relation_domains(dataset)
    owners = {}
    for client in partition.partitions:
        for head_id, relation_id, _ in client.train_triples.tolist():
            key = (
                domain_by_relation[int(relation_id)],
                int(head_id),
            )
            existing = owners.setdefault(key, int(client.client_id))
            if existing != int(client.client_id):
                return False
    return True


def _baseline_summary(dataset, seed: int) -> Dict[str, object]:
    """计算同种子原始头实体划分的图语义比较指标。"""

    baseline = partition_train_triples_by_head(
        dataset=dataset,
        client_count=CLIENT_COUNT,
        seed=int(seed),
    )
    primary_domains = _dominant_domains(dataset, baseline.partitions)
    metrics = semantic_graph_statistics(
        dataset,
        baseline.partitions,
        primary_domains,
    )
    summary = baseline.summary()
    summary.update(metrics)
    return summary


def _candidate_checks(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    packet_ownership_valid: bool,
) -> Dict[str, bool]:
    """按预注册阈值构造单个种子的校准门禁。"""

    return {
        "triple_completeness": int(
            candidate["total_train_triple_count"]
        ) == int(baseline["total_train_triple_count"]),
        "client_count": int(candidate["client_count"]) == CLIENT_COUNT,
        "nonempty_clients": int(candidate["min_client_triple_count"]) > 0,
        "load_tolerance": float(
            candidate["max_relative_load_deviation"]
        ) <= LOAD_TOLERANCE + 1e-12,
        "domain_head_packet_exclusive": bool(packet_ownership_valid),
        "semantic_purity_improvement": (
            float(candidate["triple_weighted_dominant_domain_purity"])
            - float(baseline["triple_weighted_dominant_domain_purity"])
            >= PURITY_IMPROVEMENT
        ),
        "relation_js_divergence_improvement": (
            float(candidate["mean_relation_js_divergence"])
            > float(baseline["mean_relation_js_divergence"])
        ),
        "local_entity_reuse_improvement": (
            float(candidate["local_entity_reuse_ratio"])
            > float(baseline["local_entity_reuse_ratio"])
        ),
    }


def build_calibration_contract(
    search_restarts: int = SEARCH_RESTARTS,
    reproduce: bool = True,
) -> Dict[str, object]:
    """计算三种子图语义分区、门禁和可选独立哈希复算。"""

    dataset = load_fb15k237(PACKAGE_DIR / "data" / "FB15k-237")
    baselines = {}
    candidates = {}
    all_checks = []
    hashes = []
    for seed in SEEDS:
        baseline = _baseline_summary(dataset, seed)
        partition = (
            partition_train_triples_by_semantic_domain_graph_local(
                dataset=dataset,
                client_count=CLIENT_COUNT,
                seed=seed,
                load_tolerance=LOAD_TOLERANCE,
                search_restarts=int(search_restarts),
            )
        )
        candidate = partition.summary()
        checks = _candidate_checks(
            baseline,
            candidate,
            _packet_ownership_valid(dataset, partition),
        )
        if not all(checks.values()):
            raise RuntimeError(
                "seed{}图语义划分校准门禁失败：{}".format(
                    seed,
                    [
                        name
                        for name, passed in checks.items()
                        if not passed
                    ],
                )
            )
        baselines[str(seed)] = baseline
        candidates[str(seed)] = {
            "summary": candidate,
            "checks": checks,
        }
        hashes.append(partition.partition_hash)
        all_checks.extend(checks.values())
    unique_hashes = len(set(hashes)) == len(SEEDS)
    if not unique_hashes:
        raise RuntimeError("三个正式图语义分区哈希必须互不相同")

    reproduction = {}
    if reproduce:
        for seed in SEEDS:
            reproduced = (
                partition_train_triples_by_semantic_domain_graph_local(
                    dataset=dataset,
                    client_count=CLIENT_COUNT,
                    seed=seed,
                    load_tolerance=LOAD_TOLERANCE,
                    search_restarts=int(search_restarts),
                )
            )
            expected = str(
                candidates[str(seed)]["summary"]["partition_hash"]
            )
            reproduction[str(seed)] = {
                "partition_hash": reproduced.partition_hash,
                "expected_partition_hash": expected,
                "passed": reproduced.partition_hash == expected,
            }
        if not all(item["passed"] for item in reproduction.values()):
            raise RuntimeError("图语义分区独立哈希复算失败")

    data_files = {}
    for name in ("train.txt", "valid.txt", "test.txt"):
        path = PACKAGE_DIR / "data" / "FB15k-237" / name
        data_files[name] = {
            "relative_path": "data/FB15k-237/{}".format(name),
            "sha256": _file_sha256(path),
        }
    status = (
        "passed"
        if reproduce
        and all(item["passed"] for item in reproduction.values())
        and unique_hashes
        and all(all_checks)
        else "unverified"
    )
    return {
        "contract_schema_version": 1,
        "suite": "v5_hflsnf_graph_semantic_fedadam_formal150",
        "status": status,
        "dataset": dataset.dataset_name,
        "dataset_summary": dataset.summary(),
        "data_files": data_files,
        "partition_strategy": "semantic_domain_graph_local_balanced",
        "domain_extractor": "freebase_top_level",
        "client_count": CLIENT_COUNT,
        "seeds": list(SEEDS),
        "constraints": {
            "load_tolerance": LOAD_TOLERANCE,
            "purity_improvement": PURITY_IMPROVEMENT,
            "search_restarts": int(search_restarts),
        },
        "baselines": baselines,
        "candidates": candidates,
        "unique_partition_hashes": unique_hashes,
        "reproduction": reproduction,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """创建图语义无训练校准命令行参数。"""

    parser = argparse.ArgumentParser(
        description="生成V5图语义三种子无训练校准合同"
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
        help="每个种子的确定性搜索重启次数",
    )
    parser.add_argument(
        "--skip-reproduction",
        action="store_true",
        help="仅供开发探测；跳过后合同状态不会成为passed",
    )
    return parser


def main() -> None:
    """执行无训练校准并写出UTF-8合同。"""

    args = build_argument_parser().parse_args()
    contract = build_calibration_contract(
        search_restarts=int(args.search_restarts),
        reproduce=not bool(args.skip_reproduction),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_report(contract, output)
    print("图语义分区校准状态：{}".format(contract["status"]))
    print("校准合同：{}".format(output))


if __name__ == "__main__":
    main()
