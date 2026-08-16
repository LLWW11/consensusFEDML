"""最终随机拓扑FedAdam三个预算档位、三随机种子的合同。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from ...core.topology import FixedCountTopologyProvider
from .fixed_count_four_scenarios import (
    PACKAGE_DIR,
    _append_check,
    _finite_number,
    _load_json,
    _load_metrics,
    _schedule_hash,
    load_flat_config,
)


SUITE_NAME = (
    "v3_final_stochastic_fedadam_matmean_profiles_e3_eval1_formal150"
)
ROUND_COUNT = 150
SEEDS: Tuple[int, ...] = (42, 2024, 2025)
ARMS: Tuple[str, ...] = (
    "hflsnf_profile",
    "hflnosnf_profile",
    "flnosnf_profile",
)
PARTITION_HASHES = {
    42: "8bcac64b705ec2db8721de6a36130625a460c11e0da46e2c22bd852ff015fb19",
    2024: "4653f60364e43ad9991ee3393f1a222d665774489b88f0d242ce322107d1b430",
    2025: "8a20d7ddb6556a419fea5a35ff5f8a16eb534704de7d160159b8b3ce95ee2302",
}
# 哈希由本模块的确定性提供器按训练器同一规范计算，用于阻止调度漂移。
SCHEDULE_HASHES = {
    ("hflsnf_profile", 42): "2c90b460a0f22912d06e0afcfb906c430173857004f7974f5bc024e781d8ba31",
    ("hflnosnf_profile", 42): "65f5620bdb2cb7022f2f4f065a8d182d5463eb55322187417329e4c07e992008",
    ("flnosnf_profile", 42): "6b90b87b5718e1973c1da9d96a9e0e2cfd726f71c5644d0cb8b937f6e6eaf287",
    ("hflsnf_profile", 2024): "fb586895895269f4affd23c334c57f1da07f98a9d0c7b823662505257627e45f",
    ("hflnosnf_profile", 2024): "6391745471f8cd92f94ff982e7f9faf5643ba6d556c06d939cad0ada312d647f",
    ("flnosnf_profile", 2024): "1c50bd4608d260b811241c2d4e0521eb384b056f76fe40633a6b6157ec74a4f2",
    ("hflsnf_profile", 2025): "c30b6b410bce214b7195a8b2a642059348f9447ad472c0589034292b863f2bbb",
    ("hflnosnf_profile", 2025): "e734638dd08079656cee47a4c6514ab3ff7c5218d3519713c4e4c35026ca8bed",
    ("flnosnf_profile", 2025): "ac18806d15c7d2014a88c743322ccd610e837f3100a9d8d9c333f5dfaefa3910",
}


@dataclass(frozen=True)
class ArmContract:
    """描述从MAT平均值派生的随机拓扑预算档位。"""

    label: str
    architecture: str
    edge_mode: str
    participant_count: int
    group_count: int
    mat_participant_mean: float
    mat_group_mean: float


ARM_CONTRACTS = {
    "hflsnf_profile": ArmContract(
        "HFLSnF-profile", "hfl", "fixed", 34, 6,
        34.21333333333333, 5.793333333333333,
    ),
    "hflnosnf_profile": ArmContract(
        "HFLnoSnF-profile", "hfl", "fixed", 12, 3,
        12.493333333333334, 2.7666666666666666,
    ),
    "flnosnf_profile": ArmContract(
        "FLnoSnF-profile", "fl", "none", 5, 1, 4.74, 1.0,
    ),
}


@dataclass(frozen=True)
class FinalStochasticScenario:
    """描述一份随机拓扑正式配置及其不可变合同。"""

    scenario_id: str
    arm: str
    seed: int
    config_path: Path
    contract: ArmContract
    schedule_hash: str


def _config_path(arm: str, seed: int) -> Path:
    """返回随机拓扑配置的绝对路径。"""

    name = (
        "final_stochastic_fedadam_{}_seed{}_150round_cuda.yaml"
    ).format(arm, seed)
    return (PACKAGE_DIR / "configs" / "stochastic" / name).resolve()


SCENARIOS: Tuple[FinalStochasticScenario, ...] = tuple(
    FinalStochasticScenario(
        scenario_id="final_stochastic_{}_seed{}".format(arm, seed),
        arm=arm,
        seed=seed,
        config_path=_config_path(arm, seed),
        contract=ARM_CONTRACTS[arm],
        schedule_hash=SCHEDULE_HASHES[(arm, seed)],
    )
    for seed in SEEDS
    for arm in ARMS
)


def scenario_by_id(scenario_id: str) -> FinalStochasticScenario:
    """按稳定身份键返回随机拓扑实验场景。"""

    for scenario in SCENARIOS:
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知最终随机实验场景：{}".format(scenario_id))


def build_provider(
    scenario: FinalStochasticScenario,
) -> FixedCountTopologyProvider:
    """构造完全不依赖MAT文件的随机抽样与随机均衡分组提供器。"""

    contract = scenario.contract
    # 显式传入None，保证三个随机预算档位都不会回退到MAT投影路径。
    return FixedCountTopologyProvider(
        client_ids=tuple(range(37)),
        participant_count=contract.participant_count,
        architecture=contract.architecture,
        group_count=contract.group_count,
        selection_mode="seeded_random",
        seed=scenario.seed,
        grouping_mode="seeded_random_balanced",
        source_provider=None,
    )


def schedule_statistics(
    scenario: FinalStochasticScenario,
    rounds: int = ROUND_COUNT,
) -> Dict[str, object]:
    """计算随机调度的参与、分组、覆盖和稳定哈希统计。"""

    provider = build_provider(scenario)
    topologies = tuple(provider.get_round(index) for index in range(rounds))
    participant_counts = [item.participant_count for item in topologies]
    group_counts = [len(item.group_to_client_indexes) for item in topologies]
    selection_counts = {client_id: 0 for client_id in range(37)}
    maximum_group_size_gap = 0
    for topology in topologies:
        for client_id in topology.active_client_indexes:
            selection_counts[int(client_id)] += 1
        sizes = [
            len(client_ids)
            for client_ids in topology.group_to_client_indexes.values()
        ]
        maximum_group_size_gap = max(
            maximum_group_size_gap,
            max(sizes) - min(sizes),
        )
    ordered_counts = sorted(selection_counts.values())
    participant_sets = {item.active_client_indexes for item in topologies}
    topology_sets = {
        tuple(
            (int(group_id), tuple(int(value) for value in clients))
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
        "maximum_group_size_gap": maximum_group_size_gap,
        "unique_participant_set_count": len(participant_sets),
        "unique_topology_count": len(topology_sets),
        "client_selection_counts": {
            str(key): int(value) for key, value in selection_counts.items()
        },
        "client_participation_min": ordered_counts[0],
        "client_participation_median": statistics.median(ordered_counts),
        "client_participation_max": ordered_counts[-1],
        "zero_participation_clients": sum(
            value == 0 for value in ordered_counts
        ),
    }


def expected_flat_config(
    scenario: FinalStochasticScenario,
) -> Dict[str, object]:
    """返回一份随机实验YAML必须满足的关键字段合同。"""

    contract = scenario.contract
    return {
        "random_seed": scenario.seed,
        "dataset": "fb15k-237",
        "partition_strategy": "balanced_head_entity",
        "federated_optimizer": "DynamicTopologyTransE",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "comparison_scenario": (
            "V3-FinalStochasticFedAdam-{}-Seed{}-Formal150"
        ).format(contract.label, scenario.seed),
        "client_num_in_total": 37,
        "client_num_per_round": contract.participant_count,
        "topology_type": "fixed_count",
        "fixed_count_selection_mode": "seeded_random",
        "fixed_count_grouping_mode": "seeded_random_balanced",
        "fixed_count_seed": scenario.seed,
        "topology_architecture": contract.architecture,
        "topology_snf": False,
        "topology_edge_mode": contract.edge_mode,
        "edge_num": contract.group_count,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.05,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
        "server_bias_correction": False,
        "expected_partition_hash": PARTITION_HASHES[scenario.seed],
        "expected_topology_schedule_hash": scenario.schedule_hash,
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
            "hflsnf_kg_v3_final_stochastic_fedadam_{}_seed{}_"
            "150round_cuda"
        ).format(scenario.arm, scenario.seed),
        "result_root": "results",
    }


def validate_configs() -> Dict[str, object]:
    """校验九份随机YAML和各自150轮确定性随机调度。"""

    reports: List[Dict[str, object]] = []
    forbidden_fields = (
        "dynamic_group_mat_file",
        "topology_util",
        "topology_schedule_policy",
    )
    for scenario in SCENARIOS:
        config = load_flat_config(scenario.config_path)
        statistics_payload = schedule_statistics(scenario)
        checks: List[Dict[str, object]] = []
        for field, expected in expected_flat_config(scenario).items():
            _append_check(checks, field, config.get(field), expected)
        for field in forbidden_fields:
            _append_check(checks, "禁止字段{}".format(field), field in config, False)
        contract = scenario.contract
        for field, expected in {
            "schedule_hash": scenario.schedule_hash,
            "participant_count_min": contract.participant_count,
            "participant_count_max": contract.participant_count,
            "group_count_min": contract.group_count,
            "group_count_max": contract.group_count,
            "maximum_group_size_gap": (
                0
                if contract.participant_count % contract.group_count == 0
                else 1
            ),
            "zero_participation_clients": 0,
        }.items():
            _append_check(
                checks,
                "随机调度.{}".format(field),
                statistics_payload[field],
                expected,
            )
        reports.append({
            "scenario_id": scenario.scenario_id,
            "arm": scenario.arm,
            "seed": scenario.seed,
            "config": str(scenario.config_path),
            "status": "passed" if all(item["passed"] for item in checks) else "failed",
            "schedule_statistics": statistics_payload,
            "checks": checks,
        })
    return {
        "status": "passed" if all(
            item["status"] == "passed" for item in reports
        ) else "failed",
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "scenario_order": [item.scenario_id for item in SCENARIOS],
        "formal_configs": reports,
    }


def _is_sha256(value: object) -> bool:
    """判断值是否为64位十六进制SHA-256。"""

    text = str(value).lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    scenario: FinalStochasticScenario,
) -> None:
    """校验150轮随机预算、组数、FedAdam步数和有限指标。"""

    expected_rounds = list(range(1, ROUND_COUNT + 1))
    actual_rounds = [int(_finite_number(row, "round")) for row in rows]
    _append_check(checks, "逐轮编号", actual_rounds, expected_rounds)
    if actual_rounds != expected_rounds:
        return
    for field, expected in (
        ("active_client_count", scenario.contract.participant_count),
        ("contributing_client_count", scenario.contract.participant_count),
        ("active_group_count", scenario.contract.group_count),
    ):
        values = [int(_finite_number(row, field)) for row in rows]
        _append_check(checks, "逐轮{}".format(field), values, [expected] * ROUND_COUNT)
    steps = [int(_finite_number(row, "server_optimizer_step")) for row in rows]
    _append_check(checks, "FedAdam步数连续", steps, expected_rounds)
    for field in (
        "mean_client_train_loss", "server_active_row_count",
        "server_model_delta_l2", "server_update_l2", "val_mrr",
        "val_hits_at_1", "val_hits_at_3", "val_hits_at_10",
    ):
        values = [_finite_number(row, field) for row in rows]
        _append_check(checks, "逐轮{}有限".format(field), len(values), ROUND_COUNT)


def validate_result(
    result_dir: Path,
    scenario: FinalStochasticScenario,
    expected_initial_model_hash: str = "",
) -> Dict[str, object]:
    """校验一次随机拓扑正式结果及同种子可比性哈希。"""

    resolved = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved / "summary.json")
    snapshot = _load_json(resolved / "config_snapshot.json")
    metadata = _load_json(resolved / "topology_metadata.json")
    participation = _load_json(resolved / "dynamic_participation_summary.json")
    rows = _load_metrics(resolved / "metrics.csv")
    statistics_payload = schedule_statistics(scenario)
    contract = scenario.contract
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0", "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "architecture": contract.architecture,
        "snf_enabled": False, "edge_mode": contract.edge_mode,
        "client_count": 37, "client_num_in_total": 37,
        "client_num_per_round": contract.participant_count,
        "client_num_per_round_config": contract.participant_count,
        "client_num_per_round_source": "yaml_fixed_count",
        "participant_count_min": contract.participant_count,
        "participant_count_max": contract.participant_count,
        "group_count_min": contract.group_count,
        "group_count_max": contract.group_count,
        "comm_round": ROUND_COUNT, "local_epochs": 3,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam", "server_learning_rate": 0.05,
        "server_beta1": 0.9, "server_beta2": 0.99,
        "server_tau": 0.001, "server_bias_correction": False,
        "server_optimizer_step_count": ROUND_COUNT,
        "client_optimizer_state_mode": "reset",
        "topology_schedule_hash": scenario.schedule_hash,
        "test_evaluation_performed": False, "final_test_metrics": None,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
    config = load_flat_config(scenario.config_path)
    for field, expected in config.items():
        _append_check(checks, "config_snapshot.{}".format(field), snapshot.get(field), expected)
    for field, expected in {
        "provider_type": "fixed_count",
        "architecture": contract.architecture,
        "snf_enabled": False,
        "edge_mode": contract.edge_mode,
        "fixed_participant_count": contract.participant_count,
        "fixed_group_count": contract.group_count,
        "fixed_count_selection_mode": "seeded_random",
        "fixed_count_grouping_mode": "seeded_random_balanced",
        "fixed_count_seed": scenario.seed,
        "source_topology": None,
        "mat_file": None,
        "topology_util": None,
        "round_count": None,
        "topology_schedule_policy": "unbounded",
    }.items():
        _append_check(checks, "topology_metadata.{}".format(field), metadata.get(field), expected)
    _append_check(checks, "参与汇总哈希", participation.get("schedule_hash"), scenario.schedule_hash)
    _append_check(
        checks, "参与汇总客户端计数",
        participation.get("client_selection_counts"),
        statistics_payload["client_selection_counts"],
    )
    partition_hash = str(summary.get("partition_hash", ""))
    initial_hash = str(summary.get("initial_model_hash", ""))
    _append_check(checks, "分区哈希格式", _is_sha256(partition_hash), True)
    _append_check(checks, "分区哈希", partition_hash, PARTITION_HASHES[scenario.seed])
    _append_check(checks, "初始模型哈希格式", _is_sha256(initial_hash), True)
    if expected_initial_model_hash:
        _append_check(checks, "同种子初始模型哈希", initial_hash, expected_initial_model_hash)
    _validate_round_metrics(checks, rows, scenario)
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "suite": SUITE_NAME, "scenario_id": scenario.scenario_id,
        "arm": scenario.arm, "seed": scenario.seed,
        "result_dir": str(resolved), "partition_hash": partition_hash,
        "initial_model_hash": initial_hash, "expected_rounds": ROUND_COUNT,
        "schedule_statistics": statistics_payload, "checks": checks,
    }
