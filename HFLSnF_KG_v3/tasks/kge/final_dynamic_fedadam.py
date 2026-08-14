"""最终动态拓扑FedAdam三实验臂、三随机种子的配置与结果合同。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from ...core.topology import MatlabTopologyProvider
from .fixed_count_four_scenarios import (
    PACKAGE_DIR,
    _append_check,
    _finite_number,
    _load_json,
    _load_metrics,
    _schedule_hash,
    load_flat_config,
)


SUITE_NAME = "v3_final_dynamic_fedadam_u0p6_bcfalse_e3_eval1_formal150"
ROUND_COUNT = 150
SEEDS: Tuple[int, ...] = (42, 2024, 2025)
ARMS: Tuple[str, ...] = ("hflsnf", "hflnosnf", "flnosnf")
MAT_RELATIVE_PATH = (
    "../Topo_opt/postprocess/"
    "result-U-6fixedge_epoch200_hard_varAlpha_0p1_trainable.mat"
)
MAT_FILE = (PACKAGE_DIR / MAT_RELATIVE_PATH).resolve()
PARTITION_HASHES = {
    42: "8bcac64b705ec2db8721de6a36130625a460c11e0da46e2c22bd852ff015fb19",
    2024: "4653f60364e43ad9991ee3393f1a222d665774489b88f0d242ce322107d1b430",
    2025: "8a20d7ddb6556a419fea5a35ff5f8a16eb534704de7d160159b8b3ce95ee2302",
}


@dataclass(frozen=True)
class ArmContract:
    """描述一个最终实验臂的拓扑语义和150轮调度合同。"""

    architecture: str
    snf_enabled: bool
    edge_mode: str
    edge_num: int
    schedule_hash: str
    zero_participation_clients: int
    participant_min: int
    participant_max: int
    participant_mean: float
    group_min: int
    group_max: int
    group_mean: float
    unique_participant_sets: int
    unique_topologies: int
    client_min: int
    client_median: float
    client_max: int


ARM_CONTRACTS = {
    "hflsnf": ArmContract(
        "hfl", True, "fixed", 6,
        "e383d1c5727c9541a8ea5df105c4a1ce93590b4ce6a6956ffe2bfaf93e2f84fe",
        0, 24, 35, 34.21333333333333, 1, 6, 5.793333333333333,
        111, 149, 90, 141, 150,
    ),
    "hflnosnf": ArmContract(
        "hfl", False, "fixed", 6,
        "0c7b70472476933389e1b8347a1583dbb0c847ae74a6469704a8cf09d025cabc",
        0, 9, 20, 12.493333333333334, 1, 6, 2.7666666666666666,
        143, 143, 2, 47, 102,
    ),
    "flnosnf": ArmContract(
        "fl", False, "none", 1,
        "def543cd55e67e72f9016ae8e81493730a663d3da7a1a9af27c014e7ce2a0151",
        0, 4, 8, 4.74, 1, 1, 1.0, 105, 105, 1, 2, 97,
    ),
}


@dataclass(frozen=True)
class FinalDynamicScenario:
    """描述最终九组运行中一份不可变配置的身份。"""

    scenario_id: str
    arm: str
    seed: int
    config_path: Path
    contract: ArmContract


def _config_path(arm: str, seed: int) -> Path:
    """返回一个最终配置在工程中的绝对路径。"""

    name = (
        "final_dynamic_fedadam_{}_u0p6_bcfalse_seed{}_"
        "150round_cuda.yaml"
    ).format(arm, seed)
    return (PACKAGE_DIR / "configs" / name).resolve()


SCENARIOS: Tuple[FinalDynamicScenario, ...] = tuple(
    FinalDynamicScenario(
        scenario_id="final_{}_u0p6_bcfalse_seed{}".format(arm, seed),
        arm=arm,
        seed=seed,
        config_path=_config_path(arm, seed),
        contract=ARM_CONTRACTS[arm],
    )
    for seed in SEEDS
    for arm in ARMS
)


def scenario_by_id(scenario_id: str) -> FinalDynamicScenario:
    """按稳定身份键返回最终实验场景。"""

    for scenario in SCENARIOS:
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知最终动态实验场景：{}".format(scenario_id))


def build_provider(scenario: FinalDynamicScenario) -> MatlabTopologyProvider:
    """按实验臂构造只读MAT动态拓扑提供器。"""

    contract = scenario.contract
    return MatlabTopologyProvider(
        mat_path=MAT_FILE,
        architecture=contract.architecture,
        snf_enabled=contract.snf_enabled,
        edge_mode=contract.edge_mode,
        util=0.6,
        client_count=37,
        schedule_policy="strict",
    )


def schedule_statistics(
    scenario: FinalDynamicScenario,
    rounds: int = ROUND_COUNT,
) -> Dict[str, object]:
    """计算逐轮参与规模、覆盖、拓扑多样性和规范化哈希。"""

    provider = build_provider(scenario)
    topologies = tuple(provider.get_round(index) for index in range(rounds))
    participant_counts = [item.participant_count for item in topologies]
    group_counts = [len(item.group_to_client_indexes) for item in topologies]
    selection_counts = {client_id: 0 for client_id in range(37)}
    for topology in topologies:
        for client_id in topology.active_client_indexes:
            selection_counts[int(client_id)] += 1
    ordered = sorted(selection_counts.values())
    participant_sets = {item.active_client_indexes for item in topologies}
    topology_sets = {
        tuple(
            (
                int(group_id),
                tuple(int(value) for value in clients),
                int(item.edge_node_ids.get(group_id, -1)),
            )
            for group_id, clients in item.group_to_client_indexes.items()
        )
        for item in topologies
    }
    return {
        "rounds": rounds,
        "schedule_hash": _schedule_hash(provider, rounds),
        "participant_count_min": min(participant_counts),
        "participant_count_max": max(participant_counts),
        "participant_count_mean": sum(participant_counts) / float(rounds),
        "group_count_min": min(group_counts),
        "group_count_max": max(group_counts),
        "group_count_mean": sum(group_counts) / float(rounds),
        "unique_participant_set_count": len(participant_sets),
        "unique_topology_count": len(topology_sets),
        "client_selection_counts": {
            str(key): int(value) for key, value in selection_counts.items()
        },
        "client_participation_min": ordered[0],
        "client_participation_median": statistics.median(ordered),
        "client_participation_max": ordered[-1],
        "zero_participation_clients": sum(value == 0 for value in ordered),
        "zero_participation_client_ids": [
            key for key, value in selection_counts.items() if value == 0
        ],
        "cumulative_client_rounds": sum(participant_counts),
    }


def expected_flat_config(scenario: FinalDynamicScenario) -> Dict[str, object]:
    """返回最终YAML必须满足的关键字段合同。"""

    contract = scenario.contract
    label = {
        "hflsnf": "HFLSNF",
        "hflnosnf": "HFLNOSNF",
        "flnosnf": "FLNOSNF",
    }[scenario.arm]
    return {
        "random_seed": scenario.seed,
        "dataset": "fb15k-237",
        "partition_strategy": "balanced_head_entity",
        "federated_optimizer": "DynamicTopologyTransE",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "comparison_scenario": (
            "V3-FinalDynamicFedAdam-{}-u0p6_bcfalse-Seed{}-Formal150"
        ).format(label, scenario.seed),
        "client_num_in_total": 37,
        "client_num_per_round": 37,
        "topology_type": "matlab_direct",
        "topology_architecture": contract.architecture,
        "topology_snf": contract.snf_enabled,
        "topology_edge_mode": contract.edge_mode,
        "edge_num": contract.edge_num,
        "dynamic_group_mat_file": MAT_RELATIVE_PATH,
        "topology_util": 0.6,
        "topology_schedule_policy": "strict",
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.05,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
        "server_bias_correction": False,
        "expected_partition_hash": PARTITION_HASHES[scenario.seed],
        "expected_topology_schedule_hash": contract.schedule_hash,
        "comm_round": ROUND_COUNT,
        "epochs": 3,
        "client_optimizer": "adam",
        "client_optimizer_state_mode": "reset",
        "eval_every": 1,
        "test_max_triples": 0,
        "evaluate_test_after_training": False,
        "using_gpu": True,
        "require_cuda": True,
        "gpu_id": 0,
        "run_name": (
            "hflsnf_kg_v3_final_dynamic_fedadam_{}_u0p6_bcfalse_"
            "seed{}_150round_cuda"
        ).format(scenario.arm, scenario.seed),
        "result_root": "results",
    }


def _normalized_config(config: Mapping[str, object]) -> Dict[str, object]:
    """移除九份配置按实验臂和种子允许变化的字段。"""

    allowed = {
        "random_seed", "ablation_arm", "comparison_scenario",
        "topology_architecture", "topology_snf", "topology_edge_mode",
        "edge_num", "expected_partition_hash",
        "expected_topology_schedule_hash", "run_name",
    }
    return {key: value for key, value in config.items() if key not in allowed}


def validate_configs() -> Dict[str, object]:
    """一次性校验九份YAML及三种MAT调度合同。"""

    reports: List[Dict[str, object]] = []
    normalized: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        config = load_flat_config(scenario.config_path)
        stats = schedule_statistics(scenario)
        checks: List[Dict[str, object]] = []
        for field, expected in expected_flat_config(scenario).items():
            _append_check(checks, field, config.get(field), expected)
        contract = scenario.contract
        for field, expected in {
            "schedule_hash": contract.schedule_hash,
            "participant_count_min": contract.participant_min,
            "participant_count_max": contract.participant_max,
            "participant_count_mean": contract.participant_mean,
            "group_count_min": contract.group_min,
            "group_count_max": contract.group_max,
            "group_count_mean": contract.group_mean,
            "unique_participant_set_count": contract.unique_participant_sets,
            "unique_topology_count": contract.unique_topologies,
            "client_participation_min": contract.client_min,
            "client_participation_median": contract.client_median,
            "client_participation_max": contract.client_max,
            "zero_participation_clients": contract.zero_participation_clients,
        }.items():
            _append_check(checks, "MAT.{}".format(field), stats[field], expected)
        normalized.append(_normalized_config(config))
        reports.append({
            "scenario_id": scenario.scenario_id,
            "arm": scenario.arm,
            "seed": scenario.seed,
            "config": str(scenario.config_path),
            "status": "passed" if all(item["passed"] for item in checks) else "failed",
            "schedule_statistics": stats,
            "checks": checks,
        })
    behavior_equal = bool(normalized) and all(
        item == normalized[0] for item in normalized[1:]
    )
    return {
        "status": "passed" if behavior_equal and all(
            item["status"] == "passed" for item in reports
        ) else "failed",
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "scenario_order": [item.scenario_id for item in SCENARIOS],
        "behavior_configs_equal": behavior_equal,
        "formal_configs": reports,
    }


def _is_sha256(value: object) -> bool:
    """判断一个值是否为64位十六进制SHA-256。"""

    text = str(value).lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    scenario: FinalDynamicScenario,
) -> None:
    """校验150轮拓扑规模、FedAdam步数、状态与指标完整性。"""

    topologies = tuple(build_provider(scenario).get_round(i) for i in range(ROUND_COUNT))
    expected_rounds = list(range(1, ROUND_COUNT + 1))
    actual_rounds = [int(_finite_number(row, "round")) for row in rows]
    _append_check(checks, "逐轮编号", actual_rounds, expected_rounds)
    if actual_rounds != expected_rounds:
        return
    participants = [item.participant_count for item in topologies]
    groups = [len(item.group_to_client_indexes) for item in topologies]
    for field in ("active_client_count", "contributing_client_count"):
        values = [int(_finite_number(row, field)) for row in rows]
        _append_check(checks, "逐轮{}".format(field), values, participants)
    values = [int(_finite_number(row, "active_group_count")) for row in rows]
    _append_check(checks, "逐轮active_group_count", values, groups)
    steps = [int(_finite_number(row, "server_optimizer_step")) for row in rows]
    _append_check(checks, "FedAdam步数连续", steps, expected_rounds)
    for field in (
        "mean_client_train_loss", "server_active_row_count",
        "server_model_delta_l2", "server_update_l2", "val_mrr",
        "val_hits_at_1", "val_hits_at_3", "val_hits_at_10",
    ):
        values = [_finite_number(row, field) for row in rows]
        _append_check(checks, "逐轮{}有限".format(field), len(values), ROUND_COUNT)
    for field in ("server_active_row_count", "server_model_delta_l2", "server_update_l2"):
        values = [_finite_number(row, field) for row in rows]
        _append_check(checks, "逐轮{}非零".format(field), all(v > 0 for v in values), True)
    hashes = [str(row.get("server_optimizer_state_hash", "")) for row in rows]
    _append_check(
        checks, "FedAdam状态指纹逐轮变化",
        bool(all(hashes)) and len(set(hashes)) == ROUND_COUNT, True,
    )


def validate_result(
    result_dir: Path,
    scenario: FinalDynamicScenario,
    expected_initial_model_hash: str = "",
) -> Dict[str, object]:
    """校验一次最终150轮结果及同seed可比性哈希。"""

    resolved = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved / "summary.json")
    snapshot = _load_json(resolved / "config_snapshot.json")
    metadata = _load_json(resolved / "topology_metadata.json")
    participation = _load_json(resolved / "dynamic_participation_summary.json")
    rows = _load_metrics(resolved / "metrics.csv")
    stats = schedule_statistics(scenario)
    contract = scenario.contract
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0", "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "architecture": contract.architecture,
        "snf_enabled": contract.snf_enabled,
        "edge_mode": contract.edge_mode,
        "client_count": 37, "client_num_in_total": 37,
        "client_num_per_round": stats["participant_count_max"],
        "participant_count_min": stats["participant_count_min"],
        "participant_count_max": stats["participant_count_max"],
        "comm_round": ROUND_COUNT, "local_epochs": 3,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam", "server_learning_rate": 0.05,
        "server_beta1": 0.9, "server_beta2": 0.99,
        "server_tau": 0.001, "server_bias_correction": False,
        "server_optimizer_step_count": ROUND_COUNT,
        "client_optimizer_state_mode": "reset",
        "topology_schedule_hash": contract.schedule_hash,
        "test_evaluation_performed": False, "final_test_metrics": None,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
    config = load_flat_config(scenario.config_path)
    for field, expected in config.items():
        _append_check(checks, "config_snapshot.{}".format(field), snapshot.get(field), expected)
    for field, expected in {
        "provider_type": "matlab_adapter",
        "architecture": contract.architecture,
        "snf_enabled": contract.snf_enabled,
        "edge_mode": contract.edge_mode,
        "topology_util": 0.6,
        "round_count": 200,
        "source_round_count": 200,
        "topology_schedule_policy": "strict",
    }.items():
        _append_check(checks, "topology_metadata.{}".format(field), metadata.get(field), expected)
    _append_check(checks, "参与汇总哈希", participation.get("schedule_hash"), contract.schedule_hash)
    _append_check(
        checks, "参与汇总客户端计数",
        participation.get("client_selection_counts"), stats["client_selection_counts"],
    )
    partition_hash = str(summary.get("partition_hash", ""))
    initial_hash = str(summary.get("initial_model_hash", ""))
    _append_check(checks, "分区哈希格式", _is_sha256(partition_hash), True)
    _append_check(checks, "分区哈希", partition_hash, PARTITION_HASHES[scenario.seed])
    _append_check(checks, "初始模型哈希格式", _is_sha256(initial_hash), True)
    if expected_initial_model_hash:
        _append_check(checks, "同seed初始模型哈希", initial_hash, expected_initial_model_hash)
    _append_check(
        checks, "永久缺席客户端合同",
        stats["zero_participation_clients"], contract.zero_participation_clients,
    )
    _validate_round_metrics(checks, rows, scenario)
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "suite": SUITE_NAME, "scenario_id": scenario.scenario_id,
        "arm": scenario.arm, "seed": scenario.seed,
        "result_dir": str(resolved), "partition_hash": partition_hash,
        "initial_model_hash": initial_hash, "expected_rounds": ROUND_COUNT,
        "schedule_statistics": stats, "checks": checks,
    }
