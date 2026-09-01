"""V5图语义HFLnoSnF与FLnoSnF六实验配置和结果合同。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .final_dynamic_fedadam import (
    ARM_CONTRACTS,
    FinalDynamicScenario,
    build_provider,
    schedule_statistics,
)
from .fixed_count_four_scenarios import (
    PACKAGE_DIR,
    _append_check,
    _finite_number,
    _load_json,
    _load_metrics,
    load_flat_config,
)
from .graph_semantic_ablation import (
    ROUND_COUNT,
    SEEDS,
    load_calibration_contract,
    validate_configs as validate_hflsnf_configs,
)


SUITE_NAME = "v5_graph_semantic_topology_extension_fedadam_formal150"
ARM_ORDER = ("hflnosnf", "flnosnf")
RESULT_ROOT = "results/graph_semantic_topology_extension"


@dataclass(frozen=True)
class GraphSemanticTopologyScenario:
    """描述一个图语义分区与拓扑对照臂的正式运行。"""

    scenario_id: str
    arm: str
    seed: int
    partition_hash: str
    config_path: Path

    @property
    def topology_contract(self):
        """返回当前拓扑臂的冻结合同。"""

        return ARM_CONTRACTS[self.arm]


def _hflsnf_config_path(seed: int) -> Path:
    """返回同种子V5图语义HFLSnF冻结配置。"""

    name = "graph_semantic_hflsnf_seed{}_150round_cuda.yaml".format(seed)
    return (PACKAGE_DIR / "configs" / "graph_semantic" / name).resolve()


def _scenario_config_path(arm: str, seed: int) -> Path:
    """返回一个新增拓扑场景的正式YAML路径。"""

    name = "graph_semantic_{}_seed{}_150round_cuda.yaml".format(arm, seed)
    return (
        PACKAGE_DIR
        / "configs"
        / "graph_semantic_topology_extension"
        / name
    ).resolve()


def scenarios_from_contract(
    contract: Optional[Mapping[str, object]] = None,
) -> Tuple[GraphSemanticTopologyScenario, ...]:
    """按种子优先顺序生成HFLnoSnF与FLnoSnF六场景。"""

    payload = (
        dict(contract)
        if contract is not None
        else load_calibration_contract()
    )
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("图语义校准合同candidates必须是对象")
    scenarios = []
    for seed in SEEDS:
        item = candidates.get(str(seed))
        if not isinstance(item, dict) or not isinstance(
            item.get("summary"), dict
        ):
            raise KeyError("图语义校准合同缺少seed{}摘要".format(seed))
        partition_hash = str(item["summary"].get("partition_hash", ""))
        for arm in ARM_ORDER:
            scenarios.append(
                GraphSemanticTopologyScenario(
                    scenario_id="graph_semantic_{}_seed{}".format(
                        arm, seed
                    ),
                    arm=arm,
                    seed=seed,
                    partition_hash=partition_hash,
                    config_path=_scenario_config_path(arm, seed),
                )
            )
    return tuple(scenarios)


def scenario_by_id(scenario_id: str) -> GraphSemanticTopologyScenario:
    """按稳定身份返回一个新增拓扑场景。"""

    for scenario in scenarios_from_contract():
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知图语义拓扑场景：{}".format(scenario_id))


def _arm_label(arm: str) -> str:
    """返回用于实验身份字段的规范组别名称。"""

    return {"hflnosnf": "HFLnoSnF", "flnosnf": "FLnoSnF"}[arm]


def expected_flat_config(
    scenario: GraphSemanticTopologyScenario,
) -> Dict[str, object]:
    """从同种子HFLSnF配置派生唯一合法的拓扑对照配置。"""

    contract = scenario.topology_contract
    expected = dict(load_flat_config(_hflsnf_config_path(scenario.seed)))
    expected.update(
        {
            "ablation_suite": SUITE_NAME,
            "ablation_arm": scenario.scenario_id,
            "comparison_scenario": (
                "V5-{}-GraphSemantic-Seed{}-Formal150"
            ).format(_arm_label(scenario.arm), scenario.seed),
            "topology_architecture": contract.architecture,
            "topology_snf": contract.snf_enabled,
            "topology_edge_mode": contract.edge_mode,
            "edge_num": contract.edge_num,
            "expected_topology_schedule_hash": contract.schedule_hash,
            "run_name": (
                "hflsnf_kg_v5_graph_semantic_{}_seed{}_150round_cuda"
            ).format(scenario.arm, scenario.seed),
            "result_root": RESULT_ROOT,
        }
    )
    return expected


def _topology_scenario(
    scenario: GraphSemanticTopologyScenario,
) -> FinalDynamicScenario:
    """把新增场景适配为现有冻结拓扑统计接口。"""

    return FinalDynamicScenario(
        scenario_id=scenario.scenario_id,
        arm=scenario.arm,
        seed=scenario.seed,
        config_path=scenario.config_path,
        contract=scenario.topology_contract,
    )


def validate_configs(
    recompute_partitions: bool = True,
) -> Dict[str, object]:
    """校验HFLSnF基线合同、六份YAML和两个拓扑合同。"""

    baseline = validate_hflsnf_configs(
        recompute_partitions=recompute_partitions
    )
    checks: List[Dict[str, object]] = []
    _append_check(checks, "HFLSnF图语义配置合同", baseline["status"], "passed")
    scenarios = scenarios_from_contract()
    topology_hashes: Dict[str, object] = {}
    for scenario in scenarios:
        actual = load_flat_config(scenario.config_path)
        _append_check(
            checks,
            "{}.完整配置".format(scenario.scenario_id),
            actual,
            expected_flat_config(scenario),
        )
        if scenario.arm not in topology_hashes:
            stats = schedule_statistics(_topology_scenario(scenario))
            topology_hashes[scenario.arm] = stats["schedule_hash"]
        _append_check(
            checks,
            "{}.拓扑调度哈希".format(scenario.scenario_id),
            topology_hashes[scenario.arm],
            scenario.topology_contract.schedule_hash,
        )
        _append_check(
            checks,
            "{}.分区哈希".format(scenario.scenario_id),
            actual.get("expected_partition_hash"),
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


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    scenario: GraphSemanticTopologyScenario,
) -> None:
    """校验150轮编号、真实参与规模、分组数和FedAdam步数。"""

    expected_rounds = list(range(1, ROUND_COUNT + 1))
    actual_rounds = [int(_finite_number(row, "round")) for row in rows]
    _append_check(checks, "逐轮编号", actual_rounds, expected_rounds)
    if actual_rounds != expected_rounds:
        return
    provider = build_provider(_topology_scenario(scenario))
    topologies = tuple(provider.get_round(index) for index in range(ROUND_COUNT))
    participant_counts = [item.participant_count for item in topologies]
    group_counts = [len(item.group_to_client_indexes) for item in topologies]
    for field in ("active_client_count", "contributing_client_count"):
        actual = [int(_finite_number(row, field)) for row in rows]
        _append_check(checks, "逐轮{}".format(field), actual, participant_counts)
    actual_groups = [
        int(_finite_number(row, "active_group_count")) for row in rows
    ]
    _append_check(checks, "逐轮active_group_count", actual_groups, group_counts)
    steps = [
        int(_finite_number(row, "server_optimizer_step")) for row in rows
    ]
    _append_check(checks, "FedAdam步数连续", steps, expected_rounds)
    for field in (
        "mean_client_train_loss",
        "server_model_delta_l2",
        "server_update_l2",
        "val_mrr",
    ):
        values = [_finite_number(row, field) for row in rows]
        _append_check(checks, "逐轮{}有限".format(field), len(values), ROUND_COUNT)


def validate_result(
    result_dir: Path,
    scenario: GraphSemanticTopologyScenario,
) -> Dict[str, object]:
    """校验一次新增拓扑对照的150轮正式训练结果。"""

    resolved = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved / "summary.json")
    snapshot = _load_json(resolved / "config_snapshot.json")
    metadata = _load_json(resolved / "topology_metadata.json")
    participation = _load_json(
        resolved / "dynamic_participation_summary.json"
    )
    partition = _load_json(resolved / "client_partition_summary.json")
    rows = _load_metrics(resolved / "metrics.csv")
    contract = scenario.topology_contract
    checks: List[Dict[str, object]] = []
    for field, expected in {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "architecture": contract.architecture,
        "snf_enabled": contract.snf_enabled,
        "edge_mode": contract.edge_mode,
        "comm_round": ROUND_COUNT,
        "local_epochs": 3,
        "server_optimizer": "fedadam",
        "server_optimizer_step_count": ROUND_COUNT,
        "topology_schedule_hash": contract.schedule_hash,
        "test_evaluation_performed": False,
        "final_test_metrics": None,
    }.items():
        _append_check(checks, field, summary.get(field), expected)
    for field, expected in expected_flat_config(scenario).items():
        _append_check(
            checks,
            "config_snapshot.{}".format(field),
            snapshot.get(field),
            expected,
        )
    for field, expected in {
        "architecture": contract.architecture,
        "snf_enabled": contract.snf_enabled,
        "edge_mode": contract.edge_mode,
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
        contract.schedule_hash,
    )
    _append_check(
        checks,
        "训练汇总分区哈希",
        summary.get("partition_hash"),
        scenario.partition_hash,
    )
    _append_check(
        checks,
        "分区摘要哈希",
        partition.get("partition_hash"),
        scenario.partition_hash,
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
        "checks": checks,
    }


__all__ = [
    "ARM_ORDER",
    "GraphSemanticTopologyScenario",
    "RESULT_ROOT",
    "SUITE_NAME",
    "expected_flat_config",
    "scenario_by_id",
    "scenarios_from_contract",
    "validate_configs",
    "validate_result",
]
