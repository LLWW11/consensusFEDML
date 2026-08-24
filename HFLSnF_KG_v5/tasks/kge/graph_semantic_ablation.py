"""V5图语义三种子正式实验的配置、基线和结果合同。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from . import (
    SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED,
    load_fb15k237,
    partition_train_triples_by_semantic_domain_graph_local,
)
from .final_dynamic_fedadam import ARM_CONTRACTS, schedule_statistics
from .fixed_count_four_scenarios import (
    PACKAGE_DIR,
    _load_json,
    _load_metrics,
    load_flat_config,
)
from .overlap_ablation import (
    BASELINE_INITIAL_MODEL_HASHES,
    BASELINE_RESULT_DIRS,
    ROUND_COUNT,
    _append_check,
    _file_sha256,
    _is_sha256,
    _topology_scenario,
    _validate_baselines,
    _validate_round_metrics,
    communication_statistics,
    convergence_statistics,
)


SUITE_NAME = "v5_hflsnf_graph_semantic_fedadam_formal150"
SEEDS = (42, 2024, 2025)
CALIBRATION_CONTRACT_PATH = (
    PACKAGE_DIR
    / "configs"
    / "graph_semantic"
    / "partition_calibration_contract.json"
).resolve()
V4_REFERENCE_DIR = (
    PACKAGE_DIR
    / "configs"
    / "graph_semantic"
    / "frozen_v4_reference"
).resolve()
V4_REFERENCE_FILES = {
    "analysis_summary": {
        "path": V4_REFERENCE_DIR / "overlap_ablation_summary.json",
        "sha256": (
            "ca3032bae7cadb67cbc10d4981394a21"
            "dac2b597b8a85a6f5602108e32e3d6a4"
        ),
    },
    "official_manifest": {
        "path": V4_REFERENCE_DIR / "official12_summary.json",
        "sha256": (
            "d419863ac8953087b072486ff37527932"
            "ca20ae827546add7cd040cec3b80028"
        ),
    },
    "batch_manifest": {
        "path": V4_REFERENCE_DIR / "batch_summary.json",
        "sha256": (
            "4029e345ea9889edb9c34a6493356662"
            "147dbf959094a43694ce189cd138bad3"
        ),
    },
}


@dataclass(frozen=True)
class GraphSemanticScenario:
    """描述一个正式图语义随机种子、配置和冻结分区。"""

    scenario_id: str
    seed: int
    partition_hash: str
    config_path: Path


def _baseline_config_path(seed: int) -> Path:
    """返回同种子冻结HFLSnF动态基线配置。"""

    return (
        PACKAGE_DIR
        / "configs"
        / "dynamic"
        / (
            "final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed{}_"
            "150round_cuda.yaml"
        ).format(seed)
    ).resolve()


def _scenario_config_path(seed: int) -> Path:
    """返回一个图语义正式场景的YAML路径。"""

    return (
        PACKAGE_DIR
        / "configs"
        / "graph_semantic"
        / "graph_semantic_hflsnf_seed{}_150round_cuda.yaml".format(
            seed
        )
    ).resolve()


def load_calibration_contract(
    path: Path = CALIBRATION_CONTRACT_PATH,
) -> Dict[str, object]:
    """读取并校验已通过的图语义正式校准合同。"""

    payload = _load_json(Path(path).expanduser().resolve())
    if payload.get("status") != "passed":
        raise ValueError("图语义校准合同状态必须是passed")
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("图语义校准合同套件身份不一致")
    if payload.get("partition_strategy") != (
        SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED
    ):
        raise ValueError("图语义校准合同划分策略不一致")
    if payload.get("seeds") != list(SEEDS):
        raise ValueError("图语义校准合同随机种子不一致")
    return payload


def scenarios_from_contract(
    contract: Optional[Mapping[str, object]] = None,
) -> Tuple[GraphSemanticScenario, ...]:
    """从冻结合同生成seed42、2024、2025正式场景。"""

    payload = (
        dict(contract)
        if contract is not None
        else load_calibration_contract()
    )
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("图语义合同candidates必须是对象")
    scenarios = []
    for seed in SEEDS:
        item = candidates.get(str(seed))
        if not isinstance(item, dict) or not isinstance(
            item.get("summary"), dict
        ):
            raise KeyError("图语义合同缺少seed{}摘要".format(seed))
        partition_hash = str(item["summary"].get("partition_hash", ""))
        if not _is_sha256(partition_hash):
            raise ValueError("seed{}分区哈希格式错误".format(seed))
        scenarios.append(
            GraphSemanticScenario(
                scenario_id="graph_semantic_seed{}".format(seed),
                seed=seed,
                partition_hash=partition_hash,
                config_path=_scenario_config_path(seed),
            )
        )
    return tuple(scenarios)


def scenario_by_id(scenario_id: str) -> GraphSemanticScenario:
    """按稳定场景身份查找正式图语义场景。"""

    for scenario in scenarios_from_contract():
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知图语义场景：{}".format(scenario_id))


def expected_flat_config(
    scenario: GraphSemanticScenario,
) -> Dict[str, object]:
    """从冻结动态基线派生图语义场景的唯一配置。"""

    expected = dict(load_flat_config(_baseline_config_path(scenario.seed)))
    expected.update(
        {
            "partition_strategy": (
                SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED
            ),
            "partition_domain_extractor": "freebase_top_level",
            "partition_load_tolerance": 0.05,
            "partition_search_seed": scenario.seed,
            "partition_search_restarts": 8,
            "expected_partition_hash": scenario.partition_hash,
            "ablation_suite": SUITE_NAME,
            "ablation_arm": scenario.scenario_id,
            "comparison_scenario": (
                "V5-HFLSnF-GraphSemantic-Seed{}-Formal150"
            ).format(scenario.seed),
            "run_name": (
                "hflsnf_kg_v5_graph_semantic_seed{}_150round_cuda"
            ).format(scenario.seed),
            "result_root": "results/graph_semantic",
        }
    )
    return expected


def validate_v4_references() -> Dict[str, object]:
    """校验V5内置的V4十二格冻结参考文件及固定哈希。"""

    checks: List[Dict[str, object]] = []
    for name, item in V4_REFERENCE_FILES.items():
        path = Path(item["path"]).resolve()
        _append_check(checks, "{}.存在".format(name), path.is_file(), True)
        if path.is_file():
            _append_check(
                checks,
                "{}.sha256".format(name),
                _file_sha256(path),
                str(item["sha256"]),
            )
    return {
        "status": (
            "passed"
            if checks and all(item["passed"] for item in checks)
            else "failed"
        ),
        "checks": checks,
    }


def _validate_data_files(
    contract: Mapping[str, object],
    checks: List[Dict[str, object]],
) -> None:
    """校验三份FB15k-237文本的冻结哈希。"""

    files = contract.get("data_files")
    if not isinstance(files, dict):
        raise TypeError("图语义合同data_files必须是对象")
    for name in ("train.txt", "valid.txt", "test.txt"):
        item = files.get(name)
        if not isinstance(item, dict):
            raise KeyError("图语义合同缺少{}哈希".format(name))
        path = PACKAGE_DIR / "data" / "FB15k-237" / name
        _append_check(
            checks,
            "数据文件{}.sha256".format(name),
            _file_sha256(path),
            str(item.get("sha256", "")),
        )


def validate_configs(
    recompute_partitions: bool = True,
) -> Dict[str, object]:
    """校验合同、三份YAML、基线、拓扑和可选分区复算。"""

    contract = load_calibration_contract()
    scenarios = scenarios_from_contract(contract)
    checks: List[Dict[str, object]] = []
    _validate_data_files(contract, checks)
    _validate_baselines(checks)
    reference_report = validate_v4_references()
    _append_check(
        checks,
        "V4十二格参考合同",
        reference_report["status"],
        "passed",
    )
    dataset = (
        load_fb15k237(PACKAGE_DIR / "data" / "FB15k-237")
        if recompute_partitions
        else None
    )
    for scenario in scenarios:
        actual = load_flat_config(scenario.config_path)
        _append_check(
            checks,
            "{}.完整配置".format(scenario.scenario_id),
            actual,
            expected_flat_config(scenario),
        )
        stats = schedule_statistics(_topology_scenario(scenario.seed))
        _append_check(
            checks,
            "{}.拓扑调度哈希".format(scenario.scenario_id),
            stats["schedule_hash"],
            ARM_CONTRACTS["hflsnf"].schedule_hash,
        )
        if dataset is not None:
            partition = (
                partition_train_triples_by_semantic_domain_graph_local(
                    dataset=dataset,
                    client_count=37,
                    seed=scenario.seed,
                    load_tolerance=0.05,
                    search_restarts=8,
                )
            )
            _append_check(
                checks,
                "{}.重算分区哈希".format(scenario.scenario_id),
                partition.partition_hash,
                scenario.partition_hash,
            )
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "scenario_order": [
            scenario.scenario_id for scenario in scenarios
        ],
        "recompute_partitions": bool(recompute_partitions),
        "checks": checks,
    }


def validate_result(
    result_dir: Path,
    scenario: GraphSemanticScenario,
) -> Dict[str, object]:
    """校验一次150轮图语义训练及其同种子可比性。"""

    resolved = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved / "summary.json")
    snapshot = _load_json(resolved / "config_snapshot.json")
    metadata = _load_json(resolved / "topology_metadata.json")
    participation = _load_json(
        resolved / "dynamic_participation_summary.json"
    )
    partition_summary = _load_json(
        resolved / "client_partition_summary.json"
    )
    rows = _load_metrics(resolved / "metrics.csv")
    stats = schedule_statistics(_topology_scenario(scenario.seed))
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "architecture": "hfl",
        "snf_enabled": True,
        "edge_mode": "fixed",
        "client_count": 37,
        "client_num_in_total": 37,
        "client_num_per_round": stats["participant_count_max"],
        "comm_round": ROUND_COUNT,
        "local_epochs": 3,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.05,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
        "server_bias_correction": False,
        "server_optimizer_step_count": ROUND_COUNT,
        "client_optimizer_state_mode": "reset",
        "topology_schedule_hash": (
            ARM_CONTRACTS["hflsnf"].schedule_hash
        ),
        "test_evaluation_performed": False,
        "final_test_metrics": None,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
    for field, expected in load_flat_config(
        scenario.config_path
    ).items():
        _append_check(
            checks,
            "config_snapshot.{}".format(field),
            snapshot.get(field),
            expected,
        )
    for field, expected in {
        "provider_type": "matlab_adapter",
        "architecture": "hfl",
        "snf_enabled": True,
        "edge_mode": "fixed",
        "topology_util": 0.6,
        "topology_schedule_policy": "strict",
    }.items():
        _append_check(
            checks,
            "topology_metadata.{}".format(field),
            metadata.get(field),
            expected,
        )
    _append_check(
        checks,
        "参与汇总哈希",
        participation.get("schedule_hash"),
        ARM_CONTRACTS["hflsnf"].schedule_hash,
    )
    _append_check(
        checks,
        "分区哈希",
        summary.get("partition_hash"),
        scenario.partition_hash,
    )
    _append_check(
        checks,
        "同seed原始初始模型哈希",
        summary.get("initial_model_hash"),
        BASELINE_INITIAL_MODEL_HASHES[scenario.seed],
    )
    for field, expected in {
        "partition_strategy": (
            SEMANTIC_DOMAIN_GRAPH_LOCAL_BALANCED
        ),
        "partition_hash": scenario.partition_hash,
        "domain_extractor": "freebase_top_level",
        "search_seed": scenario.seed,
        "search_restarts": 8,
    }.items():
        _append_check(
            checks,
            "分区摘要.{}".format(field),
            partition_summary.get(field),
            expected,
        )
    _append_check(
        checks,
        "分区负载上限",
        float(
            partition_summary.get(
                "max_relative_load_deviation",
                float("inf"),
            )
        ) <= 0.05 + 1e-12,
        True,
    )
    contract = load_calibration_contract()
    baseline = contract["baselines"][str(scenario.seed)]
    _append_check(
        checks,
        "语义纯度提升门禁",
        (
            float(
                partition_summary[
                    "triple_weighted_dominant_domain_purity"
                ]
            )
            - float(
                baseline[
                    "triple_weighted_dominant_domain_purity"
                ]
            )
        ) >= 0.20,
        True,
    )
    _append_check(
        checks,
        "关系JS散度提升门禁",
        float(partition_summary["mean_relation_js_divergence"])
        > float(baseline["mean_relation_js_divergence"]),
        True,
    )
    _append_check(
        checks,
        "局部实体复用提升门禁",
        float(partition_summary["local_entity_reuse_ratio"])
        > float(baseline["local_entity_reuse_ratio"]),
        True,
    )
    _validate_round_metrics(checks, rows, scenario)
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "result_dir": str(resolved),
        "partition_hash": str(summary.get("partition_hash", "")),
        "initial_model_hash": str(
            summary.get("initial_model_hash", "")
        ),
        "checks": checks,
    }


__all__ = [
    "BASELINE_INITIAL_MODEL_HASHES",
    "BASELINE_RESULT_DIRS",
    "CALIBRATION_CONTRACT_PATH",
    "GraphSemanticScenario",
    "ROUND_COUNT",
    "SEEDS",
    "SUITE_NAME",
    "V4_REFERENCE_DIR",
    "V4_REFERENCE_FILES",
    "communication_statistics",
    "convergence_statistics",
    "expected_flat_config",
    "load_calibration_contract",
    "scenario_by_id",
    "scenarios_from_contract",
    "validate_configs",
    "validate_result",
    "validate_v4_references",
]
