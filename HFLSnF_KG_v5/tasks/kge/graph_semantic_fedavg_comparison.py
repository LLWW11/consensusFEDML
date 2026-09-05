"""V5图语义FedAvg三拓扑九实验的配置与结果合同。"""

from __future__ import annotations

import json
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
)
from .graph_semantic_topology_extension import (
    validate_configs as validate_source_configs,
)
from .overlap_ablation import BASELINE_INITIAL_MODEL_HASHES, _is_sha256


SUITE_NAME = "v5_graph_semantic_fedavg_three_topology_formal150"
ARM_ORDER = ("hflsnf", "hflnosnf", "flnosnf")
RESULT_ROOT = "results/graph_semantic_fedavg_comparison"
CONFIG_DIR = (
    PACKAGE_DIR / "configs" / "graph_semantic_fedavg_comparison"
).resolve()
FEDADAM_ONLY_FIELDS = (
    "server_learning_rate",
    "server_beta1",
    "server_beta2",
    "server_tau",
    "server_bias_correction",
)


@dataclass(frozen=True)
class GraphSemanticFedAvgScenario:
    """描述一个图语义分区、拓扑臂与FedAvg正式运行。"""

    scenario_id: str
    arm: str
    seed: int
    partition_hash: str
    config_path: Path
    source_config_path: Path

    @property
    def topology_contract(self):
        """返回当前拓扑臂的冻结调度合同。"""

        return ARM_CONTRACTS[self.arm]


def _source_config_path(arm: str, seed: int) -> Path:
    """返回同种子、同拓扑的现有FedAdam来源配置。"""

    if arm == "hflsnf":
        directory = PACKAGE_DIR / "configs" / "graph_semantic"
        name = "graph_semantic_hflsnf_seed{}_150round_cuda.yaml".format(seed)
    else:
        directory = (
            PACKAGE_DIR / "configs" / "graph_semantic_topology_extension"
        )
        name = "graph_semantic_{}_seed{}_150round_cuda.yaml".format(
            arm, seed
        )
    return (directory / name).resolve()


def _config_path(arm: str, seed: int) -> Path:
    """返回一个FedAvg正式YAML的固定路径。"""

    name = "graph_semantic_fedavg_{}_seed{}_150round_cuda.yaml".format(
        arm, seed
    )
    return (CONFIG_DIR / name).resolve()


def _arm_label(arm: str) -> str:
    """返回用于报告与运行名的规范拓扑名称。"""

    return {
        "hflsnf": "HFLSnF",
        "hflnosnf": "HFLnoSnF",
        "flnosnf": "FLnoSnF",
    }[arm]


def scenarios_from_contract(
    contract: Optional[Mapping[str, object]] = None,
) -> Tuple[GraphSemanticFedAvgScenario, ...]:
    """按种子优先、拓扑次序生成九个正式场景。"""

    payload = (
        dict(contract) if contract is not None else load_calibration_contract()
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
        if not _is_sha256(partition_hash):
            raise ValueError("seed{}分区哈希格式错误".format(seed))
        for arm in ARM_ORDER:
            scenarios.append(
                GraphSemanticFedAvgScenario(
                    scenario_id=(
                        "graph_semantic_fedavg_{}_seed{}"
                    ).format(arm, seed),
                    arm=arm,
                    seed=seed,
                    partition_hash=partition_hash,
                    config_path=_config_path(arm, seed),
                    source_config_path=_source_config_path(arm, seed),
                )
            )
    return tuple(scenarios)


def scenario_by_id(scenario_id: str) -> GraphSemanticFedAvgScenario:
    """按稳定场景身份返回一个FedAvg正式场景。"""

    for scenario in scenarios_from_contract():
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知图语义FedAvg场景：{}".format(scenario_id))


def expected_flat_config(
    scenario: GraphSemanticFedAvgScenario,
) -> Dict[str, object]:
    """从同种子同拓扑FedAdam配置派生唯一FedAvg配置。"""

    expected = dict(load_flat_config(scenario.source_config_path))
    for field in FEDADAM_ONLY_FIELDS:
        expected.pop(field, None)
    expected.update(
        {
            "ablation_suite": SUITE_NAME,
            "ablation_arm": scenario.scenario_id,
            "comparison_scenario": (
                "V5-{}-GraphSemantic-FedAvg-Seed{}-Formal150"
            ).format(_arm_label(scenario.arm), scenario.seed),
            "server_optimizer": "fedavg",
            "run_name": (
                "hflsnf_kg_v5_graph_semantic_fedavg_{}_seed{}_"
                "150round_cuda"
            ).format(scenario.arm, scenario.seed),
            "result_root": RESULT_ROOT,
        }
    )
    return expected


def _topology_scenario(
    scenario: GraphSemanticFedAvgScenario,
) -> FinalDynamicScenario:
    """把FedAvg场景适配为冻结拓扑提供器输入。"""

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
    """校验来源合同、九份YAML及分区和拓扑绑定。"""

    source_report = validate_source_configs(
        recompute_partitions=recompute_partitions
    )
    checks: List[Dict[str, object]] = []
    _append_check(
        checks, "现有三拓扑来源配置合同", source_report["status"], "passed"
    )
    topology_hashes: Dict[str, str] = {}
    for scenario in scenarios_from_contract():
        actual = load_flat_config(scenario.config_path)
        expected = expected_flat_config(scenario)
        _append_check(
            checks,
            "{}.完整配置".format(scenario.scenario_id),
            actual,
            expected,
        )
        for field in FEDADAM_ONLY_FIELDS:
            _append_check(
                checks,
                "{}.不含{}".format(scenario.scenario_id, field),
                field in actual,
                False,
            )
        _append_check(
            checks,
            "{}.强制CUDA".format(scenario.scenario_id),
            actual.get("require_cuda"),
            True,
        )
        _append_check(
            checks,
            "{}.服务器优化器".format(scenario.scenario_id),
            actual.get("server_optimizer"),
            "fedavg",
        )
        if scenario.arm not in topology_hashes:
            stats = schedule_statistics(_topology_scenario(scenario))
            topology_hashes[scenario.arm] = str(stats["schedule_hash"])
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
            "passed" if checks and all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "scenario_order": [
            scenario.scenario_id for scenario in scenarios_from_contract()
        ],
        "recompute_partitions": bool(recompute_partitions),
        "checks": checks,
    }


def _load_json_lines(path: Path) -> List[Dict[str, object]]:
    """读取非空JSONL记录并要求每行都是对象。"""

    records: List[Dict[str, object]] = []
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise TypeError("JSONL第{}行必须是对象".format(line_number))
            records.append(value)
    return records


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    audits: Sequence[Mapping[str, object]],
    scenario: GraphSemanticFedAvgScenario,
) -> None:
    """校验150轮拓扑、有限指标与直接FedAvg审计字段。"""

    expected_rounds = list(range(1, ROUND_COUNT + 1))
    actual_rounds = [int(_finite_number(row, "round")) for row in rows]
    audit_rounds = [int(_finite_number(row, "round")) for row in audits]
    _append_check(checks, "逐轮编号", actual_rounds, expected_rounds)
    _append_check(checks, "聚合审计逐轮编号", audit_rounds, expected_rounds)
    if actual_rounds != expected_rounds or audit_rounds != expected_rounds:
        return
    provider = build_provider(_topology_scenario(scenario))
    topologies = tuple(provider.get_round(index) for index in range(ROUND_COUNT))
    participants = [item.participant_count for item in topologies]
    group_counts = [len(item.group_to_client_indexes) for item in topologies]
    for field in ("active_client_count", "contributing_client_count"):
        actual = [int(_finite_number(row, field)) for row in rows]
        _append_check(checks, "逐轮{}".format(field), actual, participants)
    actual_groups = [
        int(_finite_number(row, "active_group_count")) for row in rows
    ]
    _append_check(checks, "逐轮active_group_count", actual_groups, group_counts)
    for field in (
        "mean_client_train_loss",
        "server_active_row_count",
        "server_model_delta_l2",
        "server_update_l2",
        "val_mrr",
        "round_seconds",
    ):
        values = [_finite_number(row, field) for row in rows]
        _append_check(checks, "逐轮{}有限".format(field), len(values), ROUND_COUNT)
    _append_check(
        checks,
        "逐轮服务器优化器",
        [str(row.get("server_optimizer", "")) for row in rows],
        ["fedavg"] * ROUND_COUNT,
    )
    _append_check(
        checks,
        "逐轮服务器步数为零",
        [int(_finite_number(row, "server_optimizer_step")) for row in rows],
        [0] * ROUND_COUNT,
    )
    _append_check(
        checks,
        "逐轮服务器状态为空",
        [str(row.get("server_optimizer_state_hash", "")) for row in rows],
        [""] * ROUND_COUNT,
    )
    candidate_hashes = [
        str(row.get("fedavg_candidate_state_hash", "")) for row in audits
    ]
    cloud_hashes = [
        str(row.get("cloud_parameter_state_hash", "")) for row in audits
    ]
    _append_check(
        checks,
        "FedAvg候选状态哈希有效",
        all(_is_sha256(value) for value in candidate_hashes),
        True,
    )
    _append_check(
        checks,
        "FedAvg候选状态由云端直接采用",
        candidate_hashes,
        cloud_hashes,
    )
    _append_check(
        checks,
        "聚合审计服务器状态为空",
        [str(row.get("server_optimizer_state_hash", "")) for row in audits],
        [""] * ROUND_COUNT,
    )


def validate_result(
    result_dir: Path,
    scenario: GraphSemanticFedAvgScenario,
) -> Dict[str, object]:
    """校验一次150轮图语义FedAvg正式训练结果。"""

    resolved = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved / "summary.json")
    snapshot = _load_json(resolved / "config_snapshot.json")
    metadata = _load_json(resolved / "topology_metadata.json")
    participation = _load_json(
        resolved / "dynamic_participation_summary.json"
    )
    partition = _load_json(resolved / "client_partition_summary.json")
    rows = _load_metrics(resolved / "metrics.csv")
    audits = _load_json_lines(resolved / "dynamic_topology_schedule.jsonl")
    contract = scenario.topology_contract
    stats = schedule_statistics(_topology_scenario(scenario))
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "architecture": contract.architecture,
        "snf_enabled": contract.snf_enabled,
        "edge_mode": contract.edge_mode,
        "client_count": 37,
        "client_num_in_total": 37,
        "client_num_per_round": stats["participant_count_max"],
        "comm_round": ROUND_COUNT,
        "local_epochs": 3,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedavg",
        "server_learning_rate": 1.0,
        "server_beta1": 0.0,
        "server_beta2": 0.0,
        "server_tau": 0.0,
        "server_bias_correction": False,
        "server_optimizer_step_count": 0,
        "server_optimizer_state_hash": "",
        "client_optimizer_state_mode": "reset",
        "topology_schedule_hash": contract.schedule_hash,
        "test_evaluation_performed": False,
        "final_test_metrics": None,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
    for field, expected in expected_flat_config(scenario).items():
        _append_check(
            checks,
            "config_snapshot.{}".format(field),
            snapshot.get(field),
            expected,
        )
    for field, expected in {
        "provider_type": "matlab_adapter",
        "architecture": contract.architecture,
        "snf_enabled": contract.snf_enabled,
        "edge_mode": contract.edge_mode,
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
        contract.schedule_hash,
    )
    _append_check(
        checks, "训练汇总分区哈希", summary.get("partition_hash"), scenario.partition_hash
    )
    _append_check(
        checks, "分区摘要哈希", partition.get("partition_hash"), scenario.partition_hash
    )
    _append_check(
        checks,
        "同seed初始模型哈希",
        summary.get("initial_model_hash"),
        BASELINE_INITIAL_MODEL_HASHES[scenario.seed],
    )
    _append_check(
        checks,
        "分区负载上限",
        float(partition.get("max_relative_load_deviation", float("inf")))
        <= 0.05 + 1e-12,
        True,
    )
    _validate_round_metrics(checks, rows, audits, scenario)
    return {
        "status": (
            "passed" if checks and all(item["passed"] for item in checks)
            else "failed"
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
    "CONFIG_DIR",
    "FEDADAM_ONLY_FIELDS",
    "GraphSemanticFedAvgScenario",
    "RESULT_ROOT",
    "SEEDS",
    "SUITE_NAME",
    "expected_flat_config",
    "scenario_by_id",
    "scenarios_from_contract",
    "validate_configs",
    "validate_result",
]
