"""FedAdam阶段一八组40轮实验的配置、调度与结果合同。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from ...core.topology import MatlabTopologyProvider
from .fixed_count_four_scenarios import (
    INITIAL_MODEL_HASH,
    PACKAGE_DIR,
    PARTITION_HASH,
    _append_check,
    _dynamic_formal_shared_contract,
    _dynamic_schedule_statistics,
    _load_json,
    _load_metrics,
    _validate_dynamic_round_metrics,
    load_flat_config,
)


SUITE_NAME = (
    "v3_fedadam_stage1_alpha0p1_u0p5_e3_eval1_seed42_40round"
)
ROUND_COUNT = 40
MAT_RELATIVE_PATH = (
    "matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat"
)
MAT_FILE = PACKAGE_DIR / MAT_RELATIVE_PATH


@dataclass(frozen=True)
class OptimizerProfile:
    """描述一套阶段一服务器FedAdam参数。"""

    key: str
    learning_rate: float
    tau: float


@dataclass(frozen=True)
class Stage1Scenario:
    """描述一个阶段一实验臂及其40轮调度合同。"""

    scenario_id: str
    profile: OptimizerProfile
    arm: str
    architecture: str
    snf_enabled: bool
    edge_mode: str
    formal_config: str
    formal_hash: str
    participant_min: int
    participant_max: int
    group_min: int
    group_max: int
    unique_participant_sets: int
    unique_topologies: int


PROFILES: Tuple[OptimizerProfile, ...] = (
    OptimizerProfile("lr0p1_tau0p001", 0.1, 0.001),
    OptimizerProfile("lr0p05_tau0p001", 0.05, 0.001),
    OptimizerProfile("lr0p03_tau0p001", 0.03, 0.001),
    OptimizerProfile("lr0p05_tau0p01", 0.05, 0.01),
)


_ARM_CONTRACTS = {
    "hflsnf": {
        "snf_enabled": True,
        "formal_hash": (
            "5a38d5f31622707cd01303ffd1af9d434"
            "a0b1f4cae6bc9a1284ad28f7ba22c78"
        ),
        "participant_min": 31,
        "participant_max": 36,
        "group_min": 4,
        "group_max": 6,
        "unique_participant_sets": 22,
        "unique_topologies": 40,
    },
    "hflnosnf": {
        "snf_enabled": False,
        "formal_hash": (
            "7e5803167f9a0689aea9cce54c63d273"
            "009bd5c57d99d36334ffa779caa78a97"
        ),
        "participant_min": 15,
        "participant_max": 25,
        "group_min": 1,
        "group_max": 6,
        "unique_participant_sets": 39,
        "unique_topologies": 39,
    },
}


def _build_scenarios() -> Tuple[Stage1Scenario, ...]:
    """按参数组和实验臂的固定顺序构造八组实验。"""

    scenarios: List[Stage1Scenario] = []
    for profile in PROFILES:
        for arm in ("hflsnf", "hflnosnf"):
            contract = _ARM_CONTRACTS[arm]
            scenario_id = "{}_{}".format(arm, profile.key)
            scenarios.append(
                Stage1Scenario(
                    scenario_id=scenario_id,
                    profile=profile,
                    arm=arm,
                    architecture="hfl",
                    snf_enabled=bool(contract["snf_enabled"]),
                    edge_mode="fixed",
                    formal_config=(
                        "fedadam_stage1_{}_{}_seed42_"
                        "40round_cuda.yaml".format(arm, profile.key)
                    ),
                    formal_hash=str(contract["formal_hash"]),
                    participant_min=int(contract["participant_min"]),
                    participant_max=int(contract["participant_max"]),
                    group_min=int(contract["group_min"]),
                    group_max=int(contract["group_max"]),
                    unique_participant_sets=int(
                        contract["unique_participant_sets"]
                    ),
                    unique_topologies=int(
                        contract["unique_topologies"]
                    ),
                )
            )
    return tuple(scenarios)


SCENARIOS: Tuple[Stage1Scenario, ...] = _build_scenarios()


def scenario_by_id(scenario_id: str) -> Stage1Scenario:
    """按稳定场景编号返回阶段一实验合同。"""

    for scenario in SCENARIOS:
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知FedAdam阶段一实验：{}".format(scenario_id))


def build_provider(scenario: Stage1Scenario) -> MatlabTopologyProvider:
    """为一个阶段一实验创建α=0.1、利用率0.5的MAT拓扑。"""

    return MatlabTopologyProvider(
        mat_path=MAT_FILE,
        architecture=scenario.architecture,
        snf_enabled=scenario.snf_enabled,
        edge_mode=scenario.edge_mode,
        util=0.5,
        client_count=37,
        schedule_policy="strict",
    )


def schedule_statistics(scenario: Stage1Scenario) -> Dict[str, object]:
    """计算一个实验臂前40轮的参与和拓扑统计。"""

    return _dynamic_schedule_statistics(scenario, ROUND_COUNT)


def expected_config(scenario: Stage1Scenario) -> Dict[str, object]:
    """返回一份阶段一YAML必须满足的完整关键字段合同。"""

    expected = _dynamic_formal_shared_contract()
    expected.update(
        {
            "ablation_suite": SUITE_NAME,
            "ablation_arm": scenario.scenario_id,
            "comparison_scenario": (
                "V3-FedAdamStage1-{}-{}-E3-Eval1-"
                "Seed42-Formal40".format(
                    scenario.arm.upper(),
                    scenario.profile.key,
                )
            ),
            "topology_architecture": scenario.architecture,
            "topology_snf": scenario.snf_enabled,
            "topology_edge_mode": scenario.edge_mode,
            "edge_num": 6,
            "dynamic_group_mat_file": MAT_RELATIVE_PATH,
            "server_learning_rate": scenario.profile.learning_rate,
            "server_tau": scenario.profile.tau,
            "server_bias_correction": True,
            "expected_topology_schedule_hash": scenario.formal_hash,
            "comm_round": ROUND_COUNT,
            "epochs": 3,
            "eval_every": 1,
            "run_name": (
                "hflsnf_kg_v3_fedadam_stage1_{}_{}_"
                "seed42_40round_cuda".format(
                    scenario.arm,
                    scenario.profile.key,
                )
            ),
        }
    )
    return expected


def _normalized_behavior_config(
    config: Mapping[str, object],
) -> Dict[str, object]:
    """移除阶段一允许变化的身份、实验臂和服务器参数字段。"""

    allowed_differences = {
        "ablation_arm",
        "comparison_scenario",
        "topology_snf",
        "server_learning_rate",
        "server_tau",
        "expected_topology_schedule_hash",
        "run_name",
    }
    return {
        key: value
        for key, value in config.items()
        if key not in allowed_differences
    }


def validate_configs() -> Dict[str, object]:
    """校验八份YAML、四套参数和两种40轮MAT调度合同。"""

    reports: List[Dict[str, object]] = []
    normalized_configs: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        config_path = PACKAGE_DIR / "configs" / scenario.formal_config
        config = load_flat_config(config_path)
        statistics = schedule_statistics(scenario)
        checks: List[Dict[str, object]] = []
        for field, value in expected_config(scenario).items():
            _append_check(checks, field, config.get(field), value)
        for field, value in (
            ("schedule_hash", scenario.formal_hash),
            ("participant_count_min", scenario.participant_min),
            ("participant_count_max", scenario.participant_max),
            ("group_count_min", scenario.group_min),
            ("group_count_max", scenario.group_max),
            (
                "unique_participant_set_count",
                scenario.unique_participant_sets,
            ),
            ("unique_topology_count", scenario.unique_topologies),
            ("source_round_count", 200),
        ):
            _append_check(
                checks,
                "MAT.{}".format(field),
                statistics[field],
                value,
            )
        normalized_configs.append(_normalized_behavior_config(config))
        reports.append(
            {
                "scenario_id": scenario.scenario_id,
                "profile": scenario.profile.key,
                "arm": scenario.arm,
                "config": str(config_path),
                "status": (
                    "passed"
                    if all(item["passed"] for item in checks)
                    else "failed"
                ),
                "schedule_statistics": statistics,
                "checks": checks,
            }
        )
    behavior_configs_equal = bool(normalized_configs) and all(
        item == normalized_configs[0] for item in normalized_configs[1:]
    )
    return {
        "status": (
            "passed"
            if behavior_configs_equal
            and all(item["status"] == "passed" for item in reports)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "mat_file": str(MAT_FILE),
        "behavior_configs_equal": behavior_configs_equal,
        "formal_configs": reports,
    }


def validate_result(
    result_dir: Path,
    scenario: Stage1Scenario,
) -> Dict[str, object]:
    """校验一份阶段一结果的配置快照、调度和40轮FedAdam指标。"""

    resolved_dir = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved_dir / "summary.json")
    config_snapshot = _load_json(
        resolved_dir / "config_snapshot.json"
    )
    topology_metadata = _load_json(
        resolved_dir / "topology_metadata.json"
    )
    rows = _load_metrics(resolved_dir / "metrics.csv")
    statistics = schedule_statistics(scenario)
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "architecture": scenario.architecture,
        "snf_enabled": scenario.snf_enabled,
        "edge_mode": scenario.edge_mode,
        "client_count": 37,
        "client_num_in_total": 37,
        "client_num_per_round": scenario.participant_max,
        "client_num_per_round_config": 37,
        "client_num_per_round_source": "matlab",
        "participant_count_min": scenario.participant_min,
        "participant_count_max": scenario.participant_max,
        "group_count_min": scenario.group_min,
        "group_count_max": scenario.group_max,
        "unique_participant_set_count": (
            scenario.unique_participant_sets
        ),
        "unique_topology_count": scenario.unique_topologies,
        "dynamic_client_selection": True,
        "dynamic_grouping": True,
        "comm_round": ROUND_COUNT,
        "local_epochs": 3,
        "topology_schedule_policy": "strict",
        "source_topology_round_count": 200,
        "cycled_topology_round_count": 0,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": scenario.profile.learning_rate,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": scenario.profile.tau,
        "server_bias_correction": True,
        "server_optimizer_step_count": ROUND_COUNT,
        "client_optimizer_state_mode": "reset",
        "partition_hash": PARTITION_HASH,
        "topology_schedule_hash": scenario.formal_hash,
        "initial_model_hash": INITIAL_MODEL_HASH,
    }
    for field, value in expected_summary.items():
        _append_check(checks, field, summary.get(field), value)
    # 快照必须保留正式YAML的全部关键行为字段，不能只核对优化器参数。
    for field, value in expected_config(scenario).items():
        _append_check(
            checks,
            "config_snapshot.{}".format(field),
            config_snapshot.get(field),
            value,
        )
    metadata_contract = {
        "provider_type": "matlab_adapter",
        "architecture": scenario.architecture,
        "snf_enabled": scenario.snf_enabled,
        "edge_mode": scenario.edge_mode,
        "topology_util": 0.5,
        "round_count": 200,
        "source_round_count": 200,
        "topology_schedule_policy": "strict",
    }
    for field, value in metadata_contract.items():
        _append_check(
            checks,
            "topology_metadata.{}".format(field),
            topology_metadata.get(field),
            value,
        )
    _append_check(
        checks,
        "调度统计哈希",
        statistics["schedule_hash"],
        scenario.formal_hash,
    )
    _validate_dynamic_round_metrics(
        checks,
        rows,
        scenario,
        ROUND_COUNT,
    )
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "scenario_id": scenario.scenario_id,
        "profile": scenario.profile.key,
        "arm": scenario.arm,
        "result_dir": str(resolved_dir),
        "expected_rounds": ROUND_COUNT,
        "schedule_statistics": statistics,
        "checks": checks,
    }
