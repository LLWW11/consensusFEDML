"""V5语义—图局部性双消融的配置、冻结参考和结果合同。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from . import (
    DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
    SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
    load_fb15k237,
    partition_train_triples_by_graph_local_no_primary,
    partition_train_triples_by_semantic_domain_no_graph_local,
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
    ROUND_COUNT,
    _append_check,
    _file_sha256,
    _is_sha256,
    _topology_scenario,
    _validate_round_metrics,
    communication_statistics,
    convergence_statistics,
)


SUITE_NAME = "v5_graph_semantic_mechanism_ablation_fedadam_formal150"
SEEDS = (42, 2024, 2025)
ARM_ORDER = ("graph_only", "semantic_only")
CALIBRATION_CONTRACT_PATH = (
    PACKAGE_DIR
    / "configs"
    / "graph_semantic_mechanism_ablation"
    / "partition_calibration_contract.json"
).resolve()
REFERENCE_DIR = (
    CALIBRATION_CONTRACT_PATH.parent / "frozen_full_v5_reference"
).resolve()
REFERENCE_FILES = {
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
ARM_STRATEGIES = {
    "graph_only": DOMAIN_HEAD_GRAPH_LOCAL_NO_PRIMARY_BALANCED,
    "semantic_only": SEMANTIC_DOMAIN_NO_GRAPH_LOCAL_BALANCED,
}
ARM_BUILDERS = {
    "graph_only": partition_train_triples_by_graph_local_no_primary,
    "semantic_only": partition_train_triples_by_semantic_domain_no_graph_local,
}


@dataclass(frozen=True)
class MechanismAblationScenario:
    """描述一个正式机制消融实验臂、随机种子和冻结分区。"""

    scenario_id: str
    arm: str
    seed: int
    partition_strategy: str
    partition_hash: str
    config_path: Path


def _full_config_path(seed: int) -> Path:
    """返回同种子完整V5冻结正式配置路径。"""

    return (
        PACKAGE_DIR
        / "configs"
        / "graph_semantic"
        / "graph_semantic_hflsnf_seed{}_150round_cuda.yaml".format(seed)
    ).resolve()


def _scenario_config_path(arm: str, seed: int) -> Path:
    """返回一个A/B正式场景的YAML路径。"""

    return (
        CALIBRATION_CONTRACT_PATH.parent
        / "{}_hflsnf_seed{}_150round_cuda.yaml".format(arm, seed)
    ).resolve()


def load_calibration_contract(
    path: Path = CALIBRATION_CONTRACT_PATH,
) -> Dict[str, object]:
    """读取并校验已通过的双消融正式校准合同。"""

    payload = _load_json(Path(path).expanduser().resolve())
    if payload.get("status") != "passed":
        raise ValueError("双消融校准合同状态必须是passed")
    if payload.get("suite") != SUITE_NAME:
        raise ValueError("双消融校准合同套件身份不一致")
    if payload.get("seeds") != list(SEEDS):
        raise ValueError("双消融校准合同随机种子不一致")
    if payload.get("strategies") != ARM_STRATEGIES:
        raise ValueError("双消融校准合同策略身份不一致")
    return payload


def scenarios_from_contract(
    contract: Optional[Mapping[str, object]] = None,
) -> Tuple[MechanismAblationScenario, ...]:
    """按A42、B42、A2024、B2024、A2025、B2025生成场景。"""

    payload = (
        dict(contract)
        if contract is not None
        else load_calibration_contract()
    )
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("双消融合同candidates必须是对象")
    scenarios = []
    for seed in SEEDS:
        for arm in ARM_ORDER:
            arm_values = candidates.get(arm)
            if not isinstance(arm_values, dict):
                raise KeyError("双消融合同缺少{}实验臂".format(arm))
            item = arm_values.get(str(seed))
            if not isinstance(item, dict) or not isinstance(
                item.get("summary"), dict
            ):
                raise KeyError(
                    "双消融合同缺少{} seed{}摘要".format(arm, seed)
                )
            partition_hash = str(item["summary"].get("partition_hash", ""))
            if not _is_sha256(partition_hash):
                raise ValueError("{} seed{}分区哈希格式错误".format(arm, seed))
            scenarios.append(
                MechanismAblationScenario(
                    scenario_id="{}_seed{}".format(arm, seed),
                    arm=arm,
                    seed=seed,
                    partition_strategy=ARM_STRATEGIES[arm],
                    partition_hash=partition_hash,
                    config_path=_scenario_config_path(arm, seed),
                )
            )
    return tuple(scenarios)


def scenario_by_id(scenario_id: str) -> MechanismAblationScenario:
    """按稳定场景身份查找正式机制消融场景。"""

    for scenario in scenarios_from_contract():
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知双消融场景：{}".format(scenario_id))


def expected_flat_config(
    scenario: MechanismAblationScenario,
) -> Dict[str, object]:
    """从同种子完整V5配置派生一个A/B场景的唯一配置。"""

    expected = dict(load_flat_config(_full_config_path(scenario.seed)))
    label = "GraphOnly" if scenario.arm == "graph_only" else "SemanticOnly"
    expected.update(
        {
            "partition_strategy": scenario.partition_strategy,
            "partition_domain_extractor": "freebase_top_level",
            "partition_load_tolerance": 0.05,
            "partition_search_seed": scenario.seed,
            "partition_search_restarts": 8,
            "expected_partition_hash": scenario.partition_hash,
            "ablation_suite": SUITE_NAME,
            "ablation_arm": scenario.scenario_id,
            "comparison_scenario": (
                "V5-HFLSnF-{}-Seed{}-Formal150"
            ).format(label, scenario.seed),
            "run_name": (
                "hflsnf_kg_v5_{}_seed{}_150round_cuda"
            ).format(scenario.arm, scenario.seed),
            "result_root": "results/graph_semantic_mechanism_ablation",
        }
    )
    return expected


def validate_frozen_references() -> Dict[str, object]:
    """校验完整V5的五份按字节冻结JSON参考。"""

    checks: List[Dict[str, object]] = []
    for name, expected_hash in REFERENCE_FILES.items():
        path = (REFERENCE_DIR / name).resolve()
        _append_check(checks, "{}.存在".format(name), path.is_file(), True)
        if path.is_file():
            _append_check(
                checks,
                "{}.sha256".format(name),
                _file_sha256(path),
                expected_hash,
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
        raise TypeError("双消融合同data_files必须是对象")
    for name in ("train.txt", "valid.txt", "test.txt"):
        item = files.get(name)
        if not isinstance(item, dict):
            raise KeyError("双消融合同缺少{}哈希".format(name))
        _append_check(
            checks,
            "数据文件{}.sha256".format(name),
            _file_sha256(PACKAGE_DIR / "data" / "FB15k-237" / name),
            str(item.get("sha256", "")),
        )


def validate_configs(
    recompute_partitions: bool = True,
) -> Dict[str, object]:
    """校验合同、六份YAML、拓扑、冻结参考和可选分区复算。"""

    contract = load_calibration_contract()
    scenarios = scenarios_from_contract(contract)
    checks: List[Dict[str, object]] = []
    _validate_data_files(contract, checks)
    references = validate_frozen_references()
    _append_check(
        checks,
        "完整V5冻结参考合同",
        references["status"],
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
            partition = ARM_BUILDERS[scenario.arm](
                dataset=dataset,
                client_count=37,
                seed=scenario.seed,
                load_tolerance=0.05,
                search_restarts=8,
            )
            _append_check(
                checks,
                "{}.重算分区哈希".format(scenario.scenario_id),
                partition.partition_hash,
                scenario.partition_hash,
            )
    return {
        "status": (
            "passed" if all(item["passed"] for item in checks) else "failed"
        ),
        "suite": SUITE_NAME,
        "scenario_order": [item.scenario_id for item in scenarios],
        "recompute_partitions": bool(recompute_partitions),
        "checks": checks,
    }


def validate_result(
    result_dir: Path,
    scenario: MechanismAblationScenario,
) -> Dict[str, object]:
    """校验一次150轮机制消融训练及其与完整V5的可比性。"""

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
        "topology_schedule_hash": ARM_CONTRACTS["hflsnf"].schedule_hash,
        "test_evaluation_performed": False,
        "final_test_metrics": None,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
    for field, expected in load_flat_config(scenario.config_path).items():
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
        "同seed完整V5初始模型哈希",
        summary.get("initial_model_hash"),
        BASELINE_INITIAL_MODEL_HASHES[scenario.seed],
    )
    for field, expected in {
        "partition_strategy": scenario.partition_strategy,
        "partition_hash": scenario.partition_hash,
        "domain_extractor": "freebase_top_level",
        "search_seed": scenario.seed,
        "search_restarts": 8,
        "packet_unit": "semantic_domain_head_entity",
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
            partition_summary.get("max_relative_load_deviation", float("inf"))
        ) <= 0.05 + 1e-12,
        True,
    )
    if scenario.arm == "graph_only":
        mechanism_expected = {
            "has_primary_domain": False,
            "client_primary_domains": None,
            "uses_entity_locality_objective": True,
        }
    else:
        mechanism_expected = {
            "has_primary_domain": True,
            "uses_entity_locality_objective": False,
        }
        _append_check(
            checks,
            "消融B客户端主域数量",
            len(partition_summary.get("client_primary_domains", [])),
            37,
        )
    for field, expected in mechanism_expected.items():
        _append_check(
            checks,
            "分区摘要.{}".format(field),
            partition_summary.get(field),
            expected,
        )
    _validate_round_metrics(checks, rows, scenario)
    return {
        "status": (
            "passed" if all(item["passed"] for item in checks) else "failed"
        ),
        "suite": SUITE_NAME,
        "scenario_id": scenario.scenario_id,
        "arm": scenario.arm,
        "seed": scenario.seed,
        "result_dir": str(resolved),
        "partition_hash": str(summary.get("partition_hash", "")),
        "initial_model_hash": str(summary.get("initial_model_hash", "")),
        "checks": checks,
    }


__all__ = [
    "ARM_ORDER",
    "BASELINE_INITIAL_MODEL_HASHES",
    "CALIBRATION_CONTRACT_PATH",
    "MechanismAblationScenario",
    "REFERENCE_DIR",
    "REFERENCE_FILES",
    "ROUND_COUNT",
    "SEEDS",
    "SUITE_NAME",
    "communication_statistics",
    "convergence_statistics",
    "expected_flat_config",
    "load_calibration_contract",
    "scenario_by_id",
    "scenarios_from_contract",
    "validate_configs",
    "validate_frozen_references",
    "validate_result",
]
