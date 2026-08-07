"""varAlpha=0.5动态MAT四场景实验的配置、调度与结果合同。"""

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
    _finite_number,
    _load_json,
    _load_metrics,
    _schedule_hash,
    load_flat_config,
)


SUITE_NAME = "v3_dynamic_mat_varalpha0p5_four_scenario_e3_eval1_seed42"
MAT_RELATIVE_PATH = (
    "matlab/result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat"
)
MAT_FILE = PACKAGE_DIR / MAT_RELATIVE_PATH


@dataclass(frozen=True)
class VarAlpha05Scenario:
    """描述一个varAlpha=0.5动态MAT正式实验臂及其调度合同。"""

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


SCENARIOS: Tuple[VarAlpha05Scenario, ...] = (
    VarAlpha05Scenario(
        arm="hflsnf",
        architecture="hfl",
        snf_enabled=True,
        edge_mode="fixed",
        formal_config=(
            "formal_dynamic_mat_varalpha0p5_hflsnf_"
            "seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "277c448c9a427c40149e6b203476de8c8"
            "ab507e8c4229a13a5971cdce3a66fdc"
        ),
        participant_min=10,
        participant_max=37,
        group_min=3,
        group_max=6,
        unique_participant_sets=48,
        unique_topologies=148,
    ),
    VarAlpha05Scenario(
        arm="flsnf",
        architecture="fl",
        snf_enabled=True,
        edge_mode="none",
        formal_config=(
            "formal_dynamic_mat_varalpha0p5_flsnf_"
            "seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "d593c975f2fa9021e02577c58ada3a8f"
            "ec404d432cadf522fb90178ae81ea010"
        ),
        participant_min=8,
        participant_max=35,
        group_min=1,
        group_max=1,
        unique_participant_sets=134,
        unique_topologies=148,
    ),
    VarAlpha05Scenario(
        arm="hflnosnf",
        architecture="hfl",
        snf_enabled=False,
        edge_mode="fixed",
        formal_config=(
            "formal_dynamic_mat_varalpha0p5_hflnosnf_"
            "seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "c7f9fb2ea8af22e9d8d585537f18190d"
            "0ec40d8053c6171fde12f487c9c0f0d3"
        ),
        participant_min=6,
        participant_max=32,
        group_min=1,
        group_max=6,
        unique_participant_sets=145,
        unique_topologies=145,
    ),
    VarAlpha05Scenario(
        arm="flnosnf",
        architecture="fl",
        snf_enabled=False,
        edge_mode="none",
        formal_config=(
            "formal_dynamic_mat_varalpha0p5_flnosnf_"
            "seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "a17f5e62c31e75b357c67864a756019a"
            "5e202a50f3b36247a3431d87c183f8b5"
        ),
        participant_min=2,
        participant_max=12,
        group_min=1,
        group_max=1,
        unique_participant_sets=124,
        unique_topologies=124,
    ),
)


def scenario_by_arm(arm: str) -> VarAlpha05Scenario:
    """按实验臂名称返回varAlpha=0.5动态场景。"""

    for scenario in SCENARIOS:
        if scenario.arm == str(arm):
            return scenario
    raise KeyError("未知varAlpha=0.5实验臂：{}".format(arm))


def selected_scenarios(arm: str) -> Sequence[VarAlpha05Scenario]:
    """返回全部实验臂或用户指定的单个实验臂。"""

    if str(arm) == "all":
        return SCENARIOS
    return (scenario_by_arm(str(arm)),)


def build_provider(scenario: VarAlpha05Scenario) -> MatlabTopologyProvider:
    """使用0p5正式MAT文件构造保留原始参与和分组的提供器。"""

    return MatlabTopologyProvider(
        mat_path=MAT_FILE,
        architecture=scenario.architecture,
        snf_enabled=scenario.snf_enabled,
        edge_mode=scenario.edge_mode,
        util=0.5,
        client_count=37,
        schedule_policy="strict",
    )


def schedule_statistics(
    scenario: VarAlpha05Scenario,
    rounds: int = 150,
) -> Dict[str, object]:
    """计算一个0p5场景的参与人数、组数、拓扑种类和调度哈希。"""

    provider = build_provider(scenario)
    topologies = tuple(
        provider.get_round(round_index)
        for round_index in range(int(rounds))
    )
    participant_counts = [item.participant_count for item in topologies]
    group_counts = [
        len(item.group_to_client_indexes) for item in topologies
    ]
    participant_sets = {
        item.active_client_indexes for item in topologies
    }
    topology_sets = {
        tuple(
            (
                int(group_id),
                tuple(int(value) for value in client_ids),
                int(item.edge_node_ids.get(group_id, -1)),
            )
            for group_id, client_ids in (
                item.group_to_client_indexes.items()
            )
        )
        for item in topologies
    }
    return {
        "rounds": int(rounds),
        "schedule_hash": _schedule_hash(provider, rounds),
        "participant_count_min": min(participant_counts),
        "participant_count_max": max(participant_counts),
        "group_count_min": min(group_counts),
        "group_count_max": max(group_counts),
        "unique_participant_set_count": len(participant_sets),
        "unique_topology_count": len(topology_sets),
        "source_round_count": provider.round_count,
    }


def validate_configs() -> Dict[str, object]:
    """校验四份0p5 YAML和前150轮MAT调度合同。"""

    shared_contract = _dynamic_formal_shared_contract()
    shared_contract["dynamic_group_mat_file"] = MAT_RELATIVE_PATH
    reports: List[Dict[str, object]] = []
    for scenario in SCENARIOS:
        config_path = PACKAGE_DIR / "configs" / scenario.formal_config
        config = load_flat_config(config_path)
        statistics = schedule_statistics(scenario, 150)
        checks: List[Dict[str, object]] = []
        expected = dict(shared_contract)
        expected.update(
            {
                "ablation_suite": SUITE_NAME,
                "ablation_arm": scenario.arm,
                "topology_architecture": scenario.architecture,
                "topology_snf": scenario.snf_enabled,
                "topology_edge_mode": scenario.edge_mode,
                "edge_num": scenario.group_max,
                "expected_topology_schedule_hash": (
                    scenario.formal_hash
                ),
            }
        )
        for field, value in expected.items():
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
            _append_check(checks, "MAT.{}".format(field), statistics[field], value)
        reports.append(
            {
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
    return {
        "status": (
            "passed"
            if all(item["status"] == "passed" for item in reports)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "mat_file": str(MAT_FILE),
        "formal_configs": reports,
    }


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    scenario: VarAlpha05Scenario,
    expected_rounds: int,
) -> None:
    """按0p5 MAT逐轮校验人数、组数和FedAdam更新。"""

    provider = build_provider(scenario)
    topologies = tuple(
        provider.get_round(round_index)
        for round_index in range(int(expected_rounds))
    )
    expected_numbers = list(range(1, int(expected_rounds) + 1))
    round_numbers = [int(_finite_number(row, "round")) for row in rows]
    _append_check(checks, "逐轮编号", round_numbers, expected_numbers)
    if round_numbers != expected_numbers:
        return
    expected_participants = [item.participant_count for item in topologies]
    expected_groups = [
        len(item.group_to_client_indexes) for item in topologies
    ]
    for field in ("active_client_count", "contributing_client_count"):
        values = [int(_finite_number(row, field)) for row in rows]
        _append_check(checks, "逐轮{}".format(field), values, expected_participants)
    group_values = [
        int(_finite_number(row, "active_group_count")) for row in rows
    ]
    _append_check(
        checks,
        "逐轮active_group_count",
        group_values,
        expected_groups,
    )
    steps = [
        int(_finite_number(row, "server_optimizer_step")) for row in rows
    ]
    _append_check(checks, "FedAdam步数连续", steps, expected_numbers)
    for field in (
        "server_active_row_count",
        "server_model_delta_l2",
        "server_update_l2",
    ):
        values = [_finite_number(row, field) for row in rows]
        _append_check(
            checks,
            "逐轮{}非零".format(field),
            all(value > 0.0 for value in values),
            True,
        )


def validate_result(
    result_dir: Path,
    scenario: VarAlpha05Scenario,
    expected_rounds: int = 150,
) -> Dict[str, object]:
    """校验0p5正式结果的元数据、调度指纹和完整轮次。"""

    resolved_dir = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved_dir / "summary.json")
    topology_metadata = _load_json(
        resolved_dir / "topology_metadata.json"
    )
    rows = _load_metrics(resolved_dir / "metrics.csv")
    statistics = schedule_statistics(scenario, expected_rounds)
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.arm,
        "architecture": scenario.architecture,
        "snf_enabled": scenario.snf_enabled,
        "edge_mode": scenario.edge_mode,
        "client_count": 37,
        "client_num_in_total": 37,
        "client_num_per_round_config": 37,
        "client_num_per_round_source": "matlab",
        "participant_count_min": scenario.participant_min,
        "participant_count_max": scenario.participant_max,
        "group_count_min": scenario.group_min,
        "group_count_max": scenario.group_max,
        "unique_participant_set_count": scenario.unique_participant_sets,
        "unique_topology_count": scenario.unique_topologies,
        "comm_round": int(expected_rounds),
        "local_epochs": 3,
        "topology_schedule_policy": "strict",
        "source_topology_round_count": 200,
        "cycled_topology_round_count": 0,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_bias_correction": True,
        "server_optimizer_step_count": int(expected_rounds),
        "client_optimizer_state_mode": "reset",
        "partition_hash": PARTITION_HASH,
        "topology_schedule_hash": scenario.formal_hash,
        "initial_model_hash": INITIAL_MODEL_HASH,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
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
    for field, expected in metadata_contract.items():
        _append_check(
            checks,
            "topology_metadata.{}".format(field),
            topology_metadata.get(field),
            expected,
        )
    _append_check(
        checks,
        "调度统计哈希",
        statistics["schedule_hash"],
        scenario.formal_hash,
    )
    _validate_round_metrics(checks, rows, scenario, expected_rounds)
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "arm": scenario.arm,
        "result_dir": str(resolved_dir),
        "expected_rounds": int(expected_rounds),
        "schedule_statistics": statistics,
        "checks": checks,
    }
