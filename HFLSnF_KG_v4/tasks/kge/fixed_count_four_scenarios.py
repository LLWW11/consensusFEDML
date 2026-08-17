"""固定人数四组对照实验的配置、调度和结果合同工具。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import yaml

from ...core.topology import (
    FixedCountTopologyProvider,
    MatlabTopologyProvider,
)


PACKAGE_DIR = Path(__file__).resolve().parents[2]
SUITE_NAME = "v3_fixed_count_four_scenario_seed42"
DYNAMIC_SUITE_NAME = (
    "v3_dynamic_mat_four_scenario_e3_eval1_seed42"
)
SMOKE_CONFIG = "zOld/smoke/smoke_four_scenario_pipeline_cpu.yaml"
MAT_FILE = (
    PACKAGE_DIR
    / "matlab"
    / "result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat"
)
PARTITION_HASH = (
    "8bcac64b705ec2db8721de6a36130625"
    "a460c11e0da46e2c22bd852ff015fb19"
)
INITIAL_MODEL_HASH = (
    "37e5e4ee9af5e7a774027486be6b571f"
    "7ec9e18d18ec908e354379c8a761789e"
)


@dataclass(frozen=True)
class FixedCountScenario:
    """描述一个固定人数正式实验臂及其不可混淆的运行合同。"""

    arm: str
    participant_count: int
    architecture: str
    snf_enabled: bool
    group_count: int
    selection_mode: str
    formal_config: str
    formal_hash: str


@dataclass(frozen=True)
class DynamicMatScenario:
    """描述一个MAT原样回放正式实验臂及其调度合同。"""

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


SCENARIOS: Tuple[FixedCountScenario, ...] = (
    FixedCountScenario(
        "hflsnf_k35",
        35,
        "hfl",
        True,
        6,
        "snf_mat_projected",
        (
            "zOld/fixed_count_four_scenarios/"
            "formal_fixed_count_hflsnf_k35_seed42_150round_cuda.yaml"
        ),
        "c06abaa1b9f31ea79a64e6e171951e62331ec67df05f1ab0ce2f7af12f518553",
    ),
    FixedCountScenario(
        "flsnf_k25",
        25,
        "fl",
        True,
        1,
        "snf_mat_projected",
        (
            "zOld/fixed_count_four_scenarios/"
            "formal_fixed_count_flsnf_k25_seed42_150round_cuda.yaml"
        ),
        "4d7e193b45bec2493061dd1181e0704f3fbc31d185f1f4d7dae4e89c1bd99540",
    ),
    FixedCountScenario(
        "hflnosnf_k15",
        15,
        "hfl",
        False,
        6,
        "seeded_round_robin",
        (
            "zOld/fixed_count_four_scenarios/"
            "formal_fixed_count_hflnosnf_k15_seed42_150round_cuda.yaml"
        ),
        "dc8bada4757da1b5b063b72b4d11c16e6cdbeb119580ff28eb25025eb5dff73e",
    ),
    FixedCountScenario(
        "flnosnf_k5",
        5,
        "fl",
        False,
        1,
        "seeded_round_robin",
        (
            "zOld/fixed_count_four_scenarios/"
            "formal_fixed_count_flnosnf_k5_seed42_150round_cuda.yaml"
        ),
        "afca046001e45cef49af90bf22757962dae9c31ffed6f724c6dbd2d8db198329",
    ),
)


DYNAMIC_SCENARIOS: Tuple[DynamicMatScenario, ...] = (
    DynamicMatScenario(
        "hflsnf",
        "hfl",
        True,
        "fixed",
        (
            "zOld/dynamic_alpha0p1_legacy/"
            "formal_dynamic_mat_hflsnf_seed42_150round_cuda.yaml"
        ),
        "6b116d00ec61065f6719e221d98d6a10a14aecbd1ce077bf92a437b284053ebb",
        30,
        36,
        3,
        6,
        49,
        147,
    ),
    DynamicMatScenario(
        "flsnf",
        "fl",
        True,
        "none",
        (
            "zOld/dynamic_alpha0p1_legacy/"
            "formal_dynamic_mat_flsnf_seed42_150round_cuda.yaml"
        ),
        "acca8632ba2eb67559acf53e8fae1ef97083ff6b9e57cb5c7d56f5d304b281cf",
        21,
        31,
        1,
        1,
        141,
        147,
    ),
    DynamicMatScenario(
        "hflnosnf",
        "hfl",
        False,
        "fixed",
        (
            "zOld/dynamic_alpha0p1_legacy/"
            "formal_dynamic_mat_hflnosnf_seed42_150round_cuda.yaml"
        ),
        "c174e003d197eb0b1d4265f675835642341ab667d3e0ec5f470cd2cd7c2c0b9b",
        14,
        25,
        1,
        6,
        143,
        144,
    ),
    DynamicMatScenario(
        "flnosnf",
        "fl",
        False,
        "none",
        (
            "zOld/dynamic_alpha0p1_legacy/"
            "formal_dynamic_mat_flnosnf_seed42_150round_cuda.yaml"
        ),
        "ddd0012804ba13f1b79adf4b549f6181fb9e384a5a4cecc98727d1ea93db6c34",
        5,
        9,
        1,
        1,
        117,
        117,
    ),
)


def scenario_by_arm(arm: str) -> FixedCountScenario:
    """按实验臂名称返回固定人数场景，名称错误时明确报错。"""

    for scenario in SCENARIOS:
        if scenario.arm == str(arm):
            return scenario
    raise KeyError("未知固定人数实验臂：{}".format(arm))


def dynamic_scenario_by_arm(arm: str) -> DynamicMatScenario:
    """按实验臂名称返回MAT原样回放场景。"""

    for scenario in DYNAMIC_SCENARIOS:
        if scenario.arm == str(arm):
            return scenario
    raise KeyError("未知动态MAT实验臂：{}".format(arm))


def load_flat_config(config_path: Path) -> Dict[str, object]:
    """读取FedML分节YAML并合并为便于合同校验的扁平字典。"""

    with Path(config_path).resolve().open("r", encoding="utf-8") as handle:
        sections = yaml.safe_load(handle)
    if not isinstance(sections, dict):
        raise ValueError("配置顶层必须是对象：{}".format(config_path))
    flattened: Dict[str, object] = {}
    for section in sections.values():
        if not isinstance(section, dict):
            raise ValueError("配置分节必须是对象：{}".format(config_path))
        flattened.update(section)
    return flattened


def scenario_from_config(
    template: FixedCountScenario,
    config_name: str,
) -> FixedCountScenario:
    """从正式YAML读取可修改字段并构造用于校验的实验场景。"""

    config = load_flat_config(PACKAGE_DIR / "configs" / config_name)
    return replace(
        template,
        participant_count=int(config["client_num_per_round"]),
        architecture=str(config["topology_architecture"]).strip().lower(),
        snf_enabled=bool(config["topology_snf"]),
        group_count=int(config["edge_num"]),
        selection_mode=str(
            config["fixed_count_selection_mode"]
        ).strip().lower(),
        formal_hash=str(
            config.get("expected_topology_schedule_hash", "")
        ).strip(),
    )


def _schedule_hash(
    provider: FixedCountTopologyProvider,
    rounds: int,
) -> str:
    """按照训练器的规范化记录算法计算固定人数调度哈希。"""

    digest = hashlib.sha256()
    for round_index in range(int(rounds)):
        topology = provider.get_round(round_index)
        record = {
            "source_round_index": int(topology.source_round_index),
            "groups": topology.copy_groups(),
            "edge_node_ids": {
                str(group_id): int(edge_id)
                for group_id, edge_id in topology.edge_node_ids.items()
            },
        }
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def build_scenario_provider(
    scenario: FixedCountScenario,
) -> FixedCountTopologyProvider:
    """使用正式MAT文件和固定种子构造一个场景的调度提供器。"""

    source_provider = None
    if scenario.snf_enabled:
        source_provider = MatlabTopologyProvider(
            mat_path=MAT_FILE,
            architecture=scenario.architecture,
            snf_enabled=True,
            edge_mode=(
                "fixed" if scenario.architecture == "hfl" else "none"
            ),
            util=0.5,
            client_count=37,
            schedule_policy="strict",
        )
    return FixedCountTopologyProvider(
        client_ids=tuple(range(37)),
        participant_count=scenario.participant_count,
        architecture=scenario.architecture,
        group_count=scenario.group_count,
        selection_mode=scenario.selection_mode,
        seed=42,
        source_provider=source_provider,
    )


def build_dynamic_mat_provider(
    scenario: DynamicMatScenario,
) -> MatlabTopologyProvider:
    """构造保留MAT参与集合、分组和边缘编号的直接回放提供器。"""

    return MatlabTopologyProvider(
        mat_path=MAT_FILE,
        architecture=scenario.architecture,
        snf_enabled=scenario.snf_enabled,
        edge_mode=scenario.edge_mode,
        util=0.5,
        client_count=37,
        schedule_policy="strict",
    )


def _dynamic_schedule_statistics(
    scenario: DynamicMatScenario,
    rounds: int = 150,
) -> Dict[str, object]:
    """计算一个MAT直接回放场景的参与和分组统计。"""

    provider = build_dynamic_mat_provider(scenario)
    topologies = tuple(
        provider.get_round(round_index)
        for round_index in range(int(rounds))
    )
    participant_counts = [
        topology.participant_count for topology in topologies
    ]
    group_counts = [
        len(topology.group_to_client_indexes)
        for topology in topologies
    ]
    participant_sets = {
        topology.active_client_indexes for topology in topologies
    }
    topology_sets = {
        tuple(
            (
                int(group_id),
                tuple(int(value) for value in client_ids),
                int(topology.edge_node_ids.get(group_id, -1)),
            )
            for group_id, client_ids in (
                topology.group_to_client_indexes.items()
            )
        )
        for topology in topologies
    }
    return {
        "rounds": int(rounds),
        "schedule_hash": _schedule_hash(provider, rounds),
        "participant_count_min": min(participant_counts),
        "participant_count_max": max(participant_counts),
        "participant_count_mean": (
            sum(participant_counts) / float(rounds)
        ),
        "group_count_min": min(group_counts),
        "group_count_max": max(group_counts),
        "group_count_mean": sum(group_counts) / float(rounds),
        "unique_participant_set_count": len(participant_sets),
        "unique_topology_count": len(topology_sets),
        "source_round_count": provider.round_count,
    }


def _same_value(actual: object, expected: object) -> bool:
    """以严格字符串或稳定浮点容差比较合同字段。"""

    if isinstance(expected, float):
        try:
            return math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _append_check(
    checks: List[Dict[str, object]],
    name: str,
    actual: object,
    expected: object,
) -> None:
    """向报告追加一项包含实际值和期望值的结构化检查。"""

    checks.append(
        {
            "name": str(name),
            "passed": _same_value(actual, expected),
            "actual": actual,
            "expected": expected,
        }
    )


def _formal_shared_contract() -> Dict[str, object]:
    """返回四份正式YAML必须共享的训练配置合同。"""

    return {
        "random_seed": 42,
        "dataset": "fb15k-237",
        "partition_strategy": "balanced_head_entity",
        "embedding_dim": 256,
        "distance_norm": 1,
        "client_num_in_total": 37,
        "topology_type": "fixed_count",
        "fixed_count_seed": 42,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.1,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
        "server_bias_correction": True,
        "local_objective": "bidirectional_self_adversarial",
        "epochs": 2,
        "batch_size": 1024,
        "client_optimizer_state_mode": "reset",
        "learning_rate": 0.00005,
        "negative_sample_count": 256,
        "using_gpu": True,
        "require_cuda": True,
        "gpu_id": 0,
        "expected_partition_hash": PARTITION_HASH,
        "comm_round": 150,
    }


def _dynamic_formal_shared_contract() -> Dict[str, object]:
    """返回四份动态MAT正式YAML必须共享的训练合同。"""

    contract = _formal_shared_contract()
    contract.update(
        {
            "topology_type": "matlab_direct",
            "epochs": 3,
            "eval_every": 1,
            "client_num_per_round": 37,
            "dynamic_group_mat_file": (
                "matlab/result-U-6fixedge_epoch200_"
                "varAlpha_0p1_trainable.mat"
            ),
            "topology_util": 0.5,
            "topology_schedule_policy": "strict",
        }
    )
    contract.pop("fixed_count_seed", None)
    return contract


def _validate_smoke_config() -> Dict[str, object]:
    """校验唯一CPU烟雾配置覆盖正式链路的关键组件。"""

    config_path = PACKAGE_DIR / "configs" / SMOKE_CONFIG
    config = load_flat_config(config_path)
    expected = {
        "dataset": "synthetic-kg",
        "partition_strategy": "balanced_head_entity",
        "federated_optimizer": "DynamicTopologyTransE",
        "client_num_in_total": 4,
        "client_num_per_round": 3,
        "topology_type": "fixed_count",
        "fixed_count_selection_mode": "seeded_round_robin",
        "topology_architecture": "hfl",
        "topology_snf": False,
        "edge_num": 2,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "local_objective": "bidirectional_self_adversarial",
        "comm_round": 2,
        "using_gpu": False,
        "require_cuda": False,
    }
    checks: List[Dict[str, object]] = []
    for field, value in expected.items():
        _append_check(checks, field, config.get(field), value)
    return {
        "config": str(config_path),
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "checks": checks,
    }


def validate_four_scenario_configs() -> Dict[str, object]:
    """校验固定人数、动态MAT正式YAML和唯一CPU烟雾配置。"""

    reports: List[Dict[str, object]] = []
    shared_contract = _formal_shared_contract()
    for template in SCENARIOS:
        config_path = PACKAGE_DIR / "configs" / template.formal_config
        config = load_flat_config(config_path)
        scenario = scenario_from_config(template, template.formal_config)
        provider = build_scenario_provider(scenario)
        actual_hash = _schedule_hash(provider, 150)
        checks: List[Dict[str, object]] = []
        expected = dict(shared_contract)
        expected.update(
            {
                "ablation_suite": SUITE_NAME,
                "ablation_arm": scenario.arm,
                "client_num_per_round": scenario.participant_count,
                "fixed_count_selection_mode": scenario.selection_mode,
                "topology_architecture": scenario.architecture,
                "topology_snf": scenario.snf_enabled,
                "topology_edge_mode": (
                    "fixed"
                    if scenario.architecture == "hfl"
                    else "none"
                ),
                "edge_num": scenario.group_count,
                "expected_topology_schedule_hash": scenario.formal_hash,
            }
        )
        for field, value in expected.items():
            _append_check(checks, field, config.get(field), value)
        _append_check(
            checks,
            "实际调度哈希",
            actual_hash,
            scenario.formal_hash,
        )
        reports.append(
            {
                "arm": scenario.arm,
                "rounds": 150,
                "config": str(config_path),
                "status": (
                    "passed"
                    if all(item["passed"] for item in checks)
                    else "failed"
                ),
                "checks": checks,
            }
        )
    smoke_report = _validate_smoke_config()
    dynamic_reports = validate_dynamic_mat_configs()
    return {
        "status": (
            "passed"
            if (
                all(report["status"] == "passed" for report in reports)
                and all(
                    report["status"] == "passed"
                    for report in dynamic_reports
                )
                and smoke_report["status"] == "passed"
            )
            else "failed"
        ),
        "fixed_suite": SUITE_NAME,
        "dynamic_suite": DYNAMIC_SUITE_NAME,
        "fixed_formal_configs": reports,
        "dynamic_formal_configs": dynamic_reports,
        "smoke_config": smoke_report,
    }


def validate_dynamic_mat_configs() -> List[Dict[str, object]]:
    """全量读取并校验四份动态MAT YAML及前150轮原始拓扑。"""

    reports: List[Dict[str, object]] = []
    shared_contract = _dynamic_formal_shared_contract()
    for scenario in DYNAMIC_SCENARIOS:
        config_path = PACKAGE_DIR / "configs" / scenario.formal_config
        config = load_flat_config(config_path)
        statistics = _dynamic_schedule_statistics(scenario, 150)
        checks: List[Dict[str, object]] = []
        expected = dict(shared_contract)
        expected.update(
            {
                "ablation_suite": DYNAMIC_SUITE_NAME,
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
            _append_check(
                checks,
                "MAT.{}".format(field),
                statistics[field],
                value,
            )
        reports.append(
            {
                "arm": scenario.arm,
                "rounds": 150,
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
    return reports


def _load_json(path: Path) -> Dict[str, object]:
    """读取顶层为对象的UTF-8 JSON结果文件。"""

    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON顶层必须是对象：{}".format(path))
    return payload


def _load_metrics(path: Path) -> List[Dict[str, str]]:
    """读取非空逐轮CSV并返回按原顺序保存的行。"""

    with Path(path).resolve().open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("逐轮指标为空：{}".format(path))
    return rows


def _finite_number(
    row: Mapping[str, object],
    field: str,
) -> float:
    """读取一个必须存在且必须有限的数值字段。"""

    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError("字段{}不是有限数".format(field))
    return value


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    expected_rounds: int,
    participant_count: int,
    group_count: int,
) -> None:
    """校验逐轮编号、参与规模、FedAdam状态和有限损失。"""

    expected_numbers = list(range(1, int(expected_rounds) + 1))
    round_numbers = [
        int(_finite_number(row, "round")) for row in rows
    ]
    _append_check(checks, "逐轮编号", round_numbers, expected_numbers)
    if round_numbers != expected_numbers:
        return
    for field, expected in (
        ("active_client_count", participant_count),
        ("contributing_client_count", participant_count),
        ("active_group_count", group_count),
    ):
        values = [int(_finite_number(row, field)) for row in rows]
        _append_check(
            checks,
            "逐轮{}".format(field),
            all(value == expected for value in values),
            True,
        )
    steps = [
        int(_finite_number(row, "server_optimizer_step"))
        for row in rows
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
    state_hashes = [
        str(row.get("server_optimizer_state_hash", ""))
        for row in rows
    ]
    _append_check(
        checks,
        "FedAdam状态指纹逐轮变化",
        bool(all(state_hashes))
        and len(set(state_hashes)) == len(state_hashes),
        True,
    )
    losses = [
        _finite_number(row, "mean_client_train_loss")
        for row in rows
    ]
    _append_check(
        checks,
        "训练损失全部有限",
        len(losses),
        expected_rounds,
    )


def validate_fixed_count_result(
    result_dir: Path,
    scenario: FixedCountScenario,
    expected_rounds: int = 150,
) -> Dict[str, object]:
    """校验正式结果的结构、优化器、调度指纹和完整轮次。"""

    result_dir = Path(result_dir).expanduser().resolve()
    summary = _load_json(result_dir / "summary.json")
    topology_metadata = _load_json(
        result_dir / "topology_metadata.json"
    )
    rows = _load_metrics(result_dir / "metrics.csv")
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.arm,
        "architecture": scenario.architecture,
        "snf_enabled": scenario.snf_enabled,
        "client_count": 37,
        "client_num_per_round": scenario.participant_count,
        "participant_count_min": scenario.participant_count,
        "participant_count_max": scenario.participant_count,
        "group_count_min": scenario.group_count,
        "group_count_max": scenario.group_count,
        "comm_round": int(expected_rounds),
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.1,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
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
        "provider_type": "fixed_count",
        "architecture": scenario.architecture,
        "snf_enabled": scenario.snf_enabled,
        "fixed_participant_count": scenario.participant_count,
        "fixed_group_count": scenario.group_count,
        "fixed_count_selection_mode": scenario.selection_mode,
        "fixed_count_seed": 42,
    }
    for field, expected in metadata_contract.items():
        _append_check(
            checks,
            "topology_metadata.{}".format(field),
            topology_metadata.get(field),
            expected,
        )
    _validate_round_metrics(
        checks,
        rows,
        expected_rounds,
        scenario.participant_count,
        scenario.group_count,
    )
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "arm": scenario.arm,
        "result_dir": str(result_dir),
        "expected_rounds": int(expected_rounds),
        "checks": checks,
    }


def _validate_dynamic_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    scenario: DynamicMatScenario,
    expected_rounds: int,
) -> None:
    """按MAT原始拓扑逐轮校验参与人数、分组和FedAdam更新。"""

    provider = build_dynamic_mat_provider(scenario)
    topologies = tuple(
        provider.get_round(round_index)
        for round_index in range(int(expected_rounds))
    )
    expected_numbers = list(range(1, int(expected_rounds) + 1))
    round_numbers = [
        int(_finite_number(row, "round")) for row in rows
    ]
    _append_check(checks, "逐轮编号", round_numbers, expected_numbers)
    if round_numbers != expected_numbers:
        return
    expected_participants = [
        topology.participant_count for topology in topologies
    ]
    expected_groups = [
        len(topology.group_to_client_indexes)
        for topology in topologies
    ]
    for field in (
        "active_client_count",
        "contributing_client_count",
    ):
        values = [int(_finite_number(row, field)) for row in rows]
        _append_check(
            checks,
            "逐轮{}".format(field),
            values,
            expected_participants,
        )
    group_values = [
        int(_finite_number(row, "active_group_count"))
        for row in rows
    ]
    _append_check(
        checks,
        "逐轮active_group_count",
        group_values,
        expected_groups,
    )
    steps = [
        int(_finite_number(row, "server_optimizer_step"))
        for row in rows
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
    state_hashes = [
        str(row.get("server_optimizer_state_hash", ""))
        for row in rows
    ]
    _append_check(
        checks,
        "FedAdam状态指纹逐轮变化",
        bool(all(state_hashes))
        and len(set(state_hashes)) == len(state_hashes),
        True,
    )
    losses = [
        _finite_number(row, "mean_client_train_loss")
        for row in rows
    ]
    _append_check(
        checks,
        "训练损失全部有限",
        len(losses),
        expected_rounds,
    )


def validate_dynamic_mat_result(
    result_dir: Path,
    scenario: DynamicMatScenario,
    expected_rounds: int = 150,
) -> Dict[str, object]:
    """校验MAT原样回放正式结果、调度指纹和完整轮次。"""

    result_dir = Path(result_dir).expanduser().resolve()
    summary = _load_json(result_dir / "summary.json")
    topology_metadata = _load_json(
        result_dir / "topology_metadata.json"
    )
    rows = _load_metrics(result_dir / "metrics.csv")
    statistics = _dynamic_schedule_statistics(
        scenario,
        expected_rounds,
    )
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": DYNAMIC_SUITE_NAME,
        "ablation_arm": scenario.arm,
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
        "comm_round": int(expected_rounds),
        "topology_schedule_policy": "strict",
        "source_topology_round_count": 200,
        "cycled_topology_round_count": 0,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.1,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
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
    _validate_dynamic_round_metrics(
        checks,
        rows,
        scenario,
        expected_rounds,
    )
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": DYNAMIC_SUITE_NAME,
        "arm": scenario.arm,
        "result_dir": str(result_dir),
        "expected_rounds": int(expected_rounds),
        "schedule_statistics": statistics,
        "checks": checks,
    }


def validate_smoke_result(result_dir: Path) -> Dict[str, object]:
    """校验唯一CPU烟雾结果已经贯通固定人数和FedAdam链路。"""

    result_dir = Path(result_dir).expanduser().resolve()
    summary = _load_json(result_dir / "summary.json")
    topology_metadata = _load_json(
        result_dir / "topology_metadata.json"
    )
    rows = _load_metrics(result_dir / "metrics.csv")
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cpu",
        "client_count": 4,
        "client_num_per_round": 3,
        "participant_count_min": 3,
        "participant_count_max": 3,
        "group_count_min": 2,
        "group_count_max": 2,
        "comm_round": 2,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_optimizer_step_count": 2,
        "client_optimizer_state_mode": "reset",
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
    for field, expected in (
        ("provider_type", "fixed_count"),
        ("architecture", "hfl"),
        ("snf_enabled", False),
        ("fixed_participant_count", 3),
        ("fixed_group_count", 2),
        ("fixed_count_selection_mode", "seeded_round_robin"),
    ):
        _append_check(
            checks,
            "topology_metadata.{}".format(field),
            topology_metadata.get(field),
            expected,
        )
    _validate_round_metrics(checks, rows, 2, 3, 2)
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "result_dir": str(result_dir),
        "expected_rounds": 2,
        "checks": checks,
    }


def write_json_report(
    payload: Mapping[str, object],
    output_path: Path,
) -> Path:
    """把配置或结果合同报告写成UTF-8 JSON。"""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
    return output_path
