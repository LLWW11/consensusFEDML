"""FedAdam阶段二全因子、复验和参与人数对照的实验合同。"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from ...core.topology import (
    FixedCountTopologyProvider,
    MatlabTopologyProvider,
)
from .fixed_count_four_scenarios import (
    INITIAL_MODEL_HASH,
    MAT_FILE,
    PACKAGE_DIR,
    PARTITION_HASH,
    _append_check,
    _finite_number,
    _load_json,
    _load_metrics,
    _schedule_hash,
    load_flat_config,
)


SUITE_NAME = "v3_fedadam_stage2_alpha0p1_e3_eval1_formal150"
ROUND_COUNT = 150
SCREEN_SEED = 42
CONFIRM_SEEDS: Tuple[int, ...] = (2024, 2025)
ALL_SEEDS: Tuple[int, ...] = (42, 2024, 2025)
BASELINE_SETTING_KEY = "u0p5_bctrue"
MAT_RELATIVE_PATH = (
    "matlab/result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat"
)
PLATFORM_START_ROUND = 131
PLATFORM_END_ROUND = 150


@dataclass(frozen=True)
class Stage2Setting:
    """描述一个利用率与FedAdam偏差修正的全因子组合。"""

    key: str
    topology_util: float
    bias_correction: bool


@dataclass(frozen=True)
class Stage2Scenario:
    """描述阶段二一次动态MAT或固定人数训练的完整身份。"""

    scenario_id: str
    phase: str
    arm: str
    seed: int
    setting: Stage2Setting
    config_path: Path
    topology_type: str
    snf_enabled: bool
    schedule_hash: str
    participant_count: Optional[int] = None


SETTINGS: Tuple[Stage2Setting, ...] = (
    Stage2Setting("u0p5_bctrue", 0.5, True),
    Stage2Setting("u0p5_bcfalse", 0.5, False),
    Stage2Setting("u0p6_bctrue", 0.6, True),
    Stage2Setting("u0p6_bcfalse", 0.6, False),
)


_DYNAMIC_CONTRACTS: Dict[Tuple[float, str], Dict[str, object]] = {
    (0.5, "hflsnf"): {
        "schedule_hash": (
            "6b116d00ec61065f6719e221d98d6a10"
            "a14aecbd1ce077bf92a437b284053ebb"
        ),
        "participant_min": 30,
        "participant_max": 36,
        "participant_mean": 35.53333333333333,
        "group_min": 3,
        "group_max": 6,
        "group_mean": 5.846666666666667,
        "unique_participant_sets": 49,
        "unique_topologies": 147,
        "client_min": 124,
        "client_median": 146,
        "client_max": 150,
    },
    (0.5, "hflnosnf"): {
        "schedule_hash": (
            "c174e003d197eb0b1d4265f675835642"
            "341ab667d3e0ec5f470cd2cd7c2c0b9b"
        ),
        "participant_min": 14,
        "participant_max": 25,
        "participant_mean": 19.366666666666667,
        "group_min": 1,
        "group_max": 6,
        "group_mean": 3.7866666666666666,
        "unique_participant_sets": 143,
        "unique_topologies": 144,
        "client_min": 20,
        "client_median": 78,
        "client_max": 145,
    },
    (0.6, "hflsnf"): {
        "schedule_hash": (
            "e383d1c5727c9541a8ea5df105c4a1c"
            "e93590b4ce6a6956ffe2bfaf93e2f84fe"
        ),
        "participant_min": 24,
        "participant_max": 35,
        "participant_mean": 34.21333333333333,
        "group_min": 1,
        "group_max": 6,
        "group_mean": 5.793333333333333,
        "unique_participant_sets": 111,
        "unique_topologies": 149,
        "client_min": 90,
        "client_median": 141,
        "client_max": 150,
    },
    (0.6, "hflnosnf"): {
        "schedule_hash": (
            "0c7b70472476933389e1b8347a1583db"
            "b0c847ae74a6469704a8cf09d025cabc"
        ),
        "participant_min": 9,
        "participant_max": 20,
        "participant_mean": 12.493333333333334,
        "group_min": 1,
        "group_max": 6,
        "group_mean": 2.7666666666666666,
        "unique_participant_sets": 143,
        "unique_topologies": 143,
        "client_min": 2,
        "client_median": 47,
        "client_max": 102,
    },
}


def setting_by_key(key: str) -> Stage2Setting:
    """按稳定键返回阶段二全因子组合。"""

    for setting in SETTINGS:
        if setting.key == str(key):
            return setting
    raise KeyError("未知FedAdam阶段二参数组合：{}".format(key))


def dynamic_contract(
    setting: Stage2Setting,
    arm: str,
) -> Dict[str, object]:
    """返回指定利用率和实验臂的150轮MAT调度合同。"""

    key = (float(setting.topology_util), str(arm))
    if key not in _DYNAMIC_CONTRACTS:
        raise KeyError("没有阶段二动态调度合同：{}".format(key))
    return dict(_DYNAMIC_CONTRACTS[key])


def _screen_config_name(
    arm: str,
    setting: Stage2Setting,
) -> str:
    """生成一份seed=42筛选YAML的固定文件名。"""

    return (
        "fedadam_stage2_screen_{}_{}_seed42_"
        "150round_cuda.yaml".format(arm, setting.key)
    )


def _dynamic_scenario(
    phase: str,
    arm: str,
    setting: Stage2Setting,
    seed: int,
    config_path: Path,
) -> Stage2Scenario:
    """构造一个动态MAT实验场景并绑定预计算调度哈希。"""

    contract = dynamic_contract(setting, arm)
    return Stage2Scenario(
        scenario_id="{}_{}_seed{}".format(arm, setting.key, int(seed)),
        phase=str(phase),
        arm=str(arm),
        seed=int(seed),
        setting=setting,
        config_path=Path(config_path).resolve(),
        topology_type="matlab_direct",
        snf_enabled=(str(arm) == "hflsnf"),
        schedule_hash=str(contract["schedule_hash"]),
    )


def screen_scenarios() -> Tuple[Stage2Scenario, ...]:
    """按参数组合再按实验臂返回八组筛选场景。"""

    scenarios: List[Stage2Scenario] = []
    for setting in SETTINGS:
        for arm in ("hflsnf", "hflnosnf"):
            scenarios.append(
                _dynamic_scenario(
                    "screen",
                    arm,
                    setting,
                    SCREEN_SEED,
                    PACKAGE_DIR
                    / "configs"
                    / "zOld"
                    / "fedadam_stage2_screen"
                    / _screen_config_name(arm, setting),
                )
            )
    return tuple(scenarios)


SCREEN_SCENARIOS: Tuple[Stage2Scenario, ...] = screen_scenarios()


def build_dynamic_provider(
    scenario: Stage2Scenario,
) -> MatlabTopologyProvider:
    """为阶段二动态场景构造严格的MAT原样回放提供器。"""

    if scenario.topology_type != "matlab_direct":
        raise ValueError("只有动态场景可以构造MAT提供器")
    return MatlabTopologyProvider(
        mat_path=MAT_FILE,
        architecture="hfl",
        snf_enabled=scenario.snf_enabled,
        edge_mode="fixed",
        util=scenario.setting.topology_util,
        client_count=37,
        schedule_policy="strict",
    )


def build_control_provider(
    scenario: Stage2Scenario,
) -> FixedCountTopologyProvider:
    """为阶段二人数对照构造可复现的逐轮随机HFL拓扑。"""

    if (
        scenario.topology_type != "fixed_count"
        or scenario.participant_count is None
    ):
        raise ValueError("只有固定人数场景可以构造人数对照提供器")
    return FixedCountTopologyProvider(
        client_ids=tuple(range(37)),
        participant_count=scenario.participant_count,
        architecture="hfl",
        group_count=6,
        selection_mode="seeded_random",
        seed=scenario.seed,
        source_provider=None,
    )


def build_provider(
    scenario: Stage2Scenario,
) -> object:
    """根据场景类型构造动态MAT或固定人数拓扑提供器。"""

    if scenario.topology_type == "matlab_direct":
        return build_dynamic_provider(scenario)
    if scenario.topology_type == "fixed_count":
        return build_control_provider(scenario)
    raise ValueError("未知阶段二拓扑类型：{}".format(scenario.topology_type))


def schedule_statistics(
    scenario: Stage2Scenario,
    rounds: int = ROUND_COUNT,
) -> Dict[str, object]:
    """计算场景的参与规模、客户端覆盖与规范化调度哈希。"""

    provider = build_provider(scenario)
    topologies = tuple(
        provider.get_round(round_index)
        for round_index in range(int(rounds))
    )
    participant_counts = [item.participant_count for item in topologies]
    group_counts = [
        len(item.group_to_client_indexes) for item in topologies
    ]
    selection_counts = {client_id: 0 for client_id in range(37)}
    for topology in topologies:
        for client_id in topology.active_client_indexes:
            selection_counts[int(client_id)] += 1
    ordered_client_counts = sorted(selection_counts.values())
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
        "participant_count_mean": (
            sum(participant_counts) / float(rounds)
        ),
        "group_count_min": min(group_counts),
        "group_count_max": max(group_counts),
        "group_count_mean": sum(group_counts) / float(rounds),
        "unique_participant_set_count": len(participant_sets),
        "unique_topology_count": len(topology_sets),
        "client_selection_counts": {
            str(key): int(value)
            for key, value in selection_counts.items()
        },
        "client_participation_min": ordered_client_counts[0],
        "client_participation_median": statistics.median(
            ordered_client_counts
        ),
        "client_participation_max": ordered_client_counts[-1],
        "zero_participation_clients": sum(
            value == 0 for value in ordered_client_counts
        ),
        "cumulative_client_rounds": sum(participant_counts),
        "effective_client_rounds_per_client": (
            sum(participant_counts) / 37.0
        ),
    }


def expected_flat_config(
    scenario: Stage2Scenario,
) -> Dict[str, object]:
    """返回动态或人数对照YAML必须满足的扁平关键字段合同。"""

    expected: Dict[str, object] = {
        "random_seed": scenario.seed,
        "dataset": "fb15k-237",
        "data_dir": "data/FB15k-237",
        "partition_strategy": "balanced_head_entity",
        "embedding_dim": 256,
        "distance_norm": 1,
        "federated_optimizer": "DynamicTopologyTransE",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "client_num_in_total": 37,
        "topology_architecture": "hfl",
        "topology_edge_mode": "fixed",
        "edge_num": 6,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.05,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
        "server_bias_correction": (
            scenario.setting.bias_correction
        ),
        "local_objective": "bidirectional_self_adversarial",
        "fede_gamma": 9.0,
        "adversarial_temperature": 1.0,
        "expected_partition_hash": (
            PARTITION_HASH if scenario.seed == SCREEN_SEED else ""
        ),
        "expected_topology_schedule_hash": scenario.schedule_hash,
        "comm_round": ROUND_COUNT,
        "epochs": 3,
        "batch_size": 1024,
        "client_optimizer": "adam",
        "client_optimizer_state_mode": "reset",
        "learning_rate": 0.00005,
        "lr": 0.00005,
        "margin": 1.0,
        "negative_sample_count": 256,
        "eval_every": 1,
        "validation_max_triples": 4096,
        "validation_selection": "relation_stratified",
        "final_validation_max_triples": 4096,
        "test_max_triples": 0,
        "evaluate_test_after_training": False,
        "evaluation_query_batch_size": 64,
        "evaluation_candidate_batch_size": 8192,
        "using_gpu": True,
        "require_cuda": True,
        "gpu_id": 0,
        "enable_wandb": False,
        "using_mlops": False,
        "run_name": run_name(scenario),
        "result_root": "results",
    }
    if scenario.topology_type == "matlab_direct":
        expected.update(
            {
                "comparison_scenario": (
                    "V3-FedAdamStage2-{}-{}-Seed{}-Formal150".format(
                        scenario.arm.upper(),
                        scenario.setting.key,
                        scenario.seed,
                    )
                ),
                "client_num_per_round": 37,
                "topology_type": "matlab_direct",
                "topology_snf": scenario.snf_enabled,
                "dynamic_group_mat_file": MAT_RELATIVE_PATH,
                "topology_util": scenario.setting.topology_util,
                "topology_schedule_policy": "strict",
            }
        )
    else:
        expected.update(
            {
                "comparison_scenario": (
                    "V3-FedAdamStage2-Control-K{}-{}-Seed{}-"
                    "Formal150".format(
                        scenario.participant_count,
                        scenario.setting.key,
                        scenario.seed,
                    )
                ),
                "client_num_per_round": scenario.participant_count,
                "topology_type": "fixed_count",
                "fixed_count_selection_mode": "seeded_random",
                "fixed_count_seed": scenario.seed,
                "topology_snf": False,
                "stage2_source_topology_util": (
                    scenario.setting.topology_util
                ),
            }
        )
    return expected


def run_name(scenario: Stage2Scenario) -> str:
    """生成不覆盖既有结果的阶段二运行名前缀。"""

    if scenario.topology_type == "matlab_direct":
        identity = "{}_{}".format(
            scenario.arm, scenario.setting.key
        )
    else:
        identity = "control_k{}_{}".format(
            scenario.participant_count, scenario.setting.key
        )
    return (
        "hflsnf_kg_v3_fedadam_stage2_{}_seed{}_"
        "150round_cuda".format(identity, scenario.seed)
    )


def scenario_config_sections(
    scenario: Stage2Scenario,
) -> Dict[str, Dict[str, object]]:
    """构造可直接写入FedML YAML的有序分节配置。"""

    flat = expected_flat_config(scenario)
    common_args = {"training_type": "simulation", "random_seed": flat["random_seed"]}
    data_args = {
        key: flat[key]
        for key in ("dataset", "data_dir", "partition_strategy")
    }
    model_args = {
        key: flat[key] for key in ("model", "embedding_dim", "distance_norm")
    } if "model" in flat else {
        "model": "transe",
        "embedding_dim": flat["embedding_dim"],
        "distance_norm": flat["distance_norm"],
    }
    excluded = {
        "random_seed", "dataset", "data_dir", "partition_strategy",
        "model", "embedding_dim", "distance_norm", "using_gpu",
        "require_cuda", "gpu_id", "enable_wandb", "using_mlops",
        "run_name", "result_root",
    }
    train_args = {
        key: value for key, value in flat.items() if key not in excluded
    }
    return {
        "common_args": common_args,
        "data_args": data_args,
        "model_args": model_args,
        "train_args": train_args,
        "device_args": {
            "using_gpu": flat["using_gpu"],
            "require_cuda": flat["require_cuda"],
            "gpu_id": flat["gpu_id"],
        },
        "comm_args": {"backend": "sp"},
        "tracking_args": {
            "enable_wandb": flat["enable_wandb"],
            "using_mlops": flat["using_mlops"],
            "run_name": flat["run_name"],
            "result_root": flat["result_root"],
        },
    }


def write_generated_config(
    scenario: Stage2Scenario,
) -> Path:
    """以英文说明头写入批次绑定的可审计YAML配置。"""

    scenario.config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        scenario_config_sections(scenario),
        allow_unicode=True,
        sort_keys=False,
    )
    header = (
        "# Auto-generated FedAdam stage-two formal configuration.\n"
        "# This file is bound to its batch manifest and must not be edited.\n"
    )
    scenario.config_path.write_text(
        header + payload,
        encoding="utf-8",
    )
    return scenario.config_path


def _normalized_screen_config(
    config: Mapping[str, object],
) -> Dict[str, object]:
    """移除筛选矩阵允许变化的实验臂、因素和身份字段。"""

    allowed = {
        "ablation_arm",
        "comparison_scenario",
        "topology_snf",
        "topology_util",
        "server_bias_correction",
        "expected_topology_schedule_hash",
        "run_name",
    }
    return {key: value for key, value in config.items() if key not in allowed}


def validate_screen_configs() -> Dict[str, object]:
    """校验八份筛选YAML仅包含全因子合同允许的差异。"""

    reports: List[Dict[str, object]] = []
    normalized: List[Dict[str, object]] = []
    for scenario in SCREEN_SCENARIOS:
        config = load_flat_config(scenario.config_path)
        expected = expected_flat_config(scenario)
        stats = schedule_statistics(scenario)
        contract = dynamic_contract(scenario.setting, scenario.arm)
        checks: List[Dict[str, object]] = []
        for field, value in expected.items():
            _append_check(checks, field, config.get(field), value)
        for field, value in (
            ("schedule_hash", contract["schedule_hash"]),
            ("participant_count_min", contract["participant_min"]),
            ("participant_count_max", contract["participant_max"]),
            ("participant_count_mean", contract["participant_mean"]),
            ("group_count_min", contract["group_min"]),
            ("group_count_max", contract["group_max"]),
            ("group_count_mean", contract["group_mean"]),
            (
                "unique_participant_set_count",
                contract["unique_participant_sets"],
            ),
            ("unique_topology_count", contract["unique_topologies"]),
            ("client_participation_min", contract["client_min"]),
            ("client_participation_median", contract["client_median"]),
            ("client_participation_max", contract["client_max"]),
            ("zero_participation_clients", 0),
        ):
            _append_check(checks, "MAT.{}".format(field), stats[field], value)
        normalized.append(_normalized_screen_config(config))
        reports.append(
            {
                "scenario_id": scenario.scenario_id,
                "setting": scenario.setting.key,
                "arm": scenario.arm,
                "config": str(scenario.config_path),
                "status": (
                    "passed"
                    if all(item["passed"] for item in checks)
                    else "failed"
                ),
                "schedule_statistics": stats,
                "checks": checks,
            }
        )
    behavior_equal = bool(normalized) and all(
        item == normalized[0] for item in normalized[1:]
    )
    return {
        "status": (
            "passed"
            if behavior_equal
            and all(item["status"] == "passed" for item in reports)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "round_count": ROUND_COUNT,
        "behavior_configs_equal": behavior_equal,
        "formal_configs": reports,
    }


def _is_sha256(value: object) -> bool:
    """判断一个值是否为64位十六进制SHA-256文本。"""

    text = str(value)
    return len(text) == 64 and all(
        char in "0123456789abcdef" for char in text.lower()
    )


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    scenario: Stage2Scenario,
    statistics_payload: Mapping[str, object],
) -> None:
    """校验150轮拓扑、FedAdam状态与逐轮验证指标全部完整。"""

    provider = build_provider(scenario)
    topologies = tuple(provider.get_round(index) for index in range(ROUND_COUNT))
    expected_rounds = list(range(1, ROUND_COUNT + 1))
    actual_rounds = [int(_finite_number(row, "round")) for row in rows]
    _append_check(checks, "逐轮编号", actual_rounds, expected_rounds)
    if actual_rounds != expected_rounds:
        return
    expected_participants = [item.participant_count for item in topologies]
    expected_groups = [len(item.group_to_client_indexes) for item in topologies]
    for field in ("active_client_count", "contributing_client_count"):
        actual = [int(_finite_number(row, field)) for row in rows]
        _append_check(checks, "逐轮{}".format(field), actual, expected_participants)
    actual_groups = [int(_finite_number(row, "active_group_count")) for row in rows]
    _append_check(checks, "逐轮active_group_count", actual_groups, expected_groups)
    steps = [int(_finite_number(row, "server_optimizer_step")) for row in rows]
    _append_check(checks, "FedAdam步数连续", steps, expected_rounds)
    for field in (
        "mean_client_train_loss",
        "server_active_row_count",
        "server_model_delta_l2",
        "server_update_l2",
        "val_mrr",
        "val_hits_at_1",
        "val_hits_at_3",
        "val_hits_at_10",
    ):
        values = [_finite_number(row, field) for row in rows]
        _append_check(checks, "逐轮{}有限".format(field), len(values), ROUND_COUNT)
    for field in ("server_active_row_count", "server_model_delta_l2", "server_update_l2"):
        values = [_finite_number(row, field) for row in rows]
        _append_check(checks, "逐轮{}非零".format(field), all(value > 0 for value in values), True)
    hashes = [str(row.get("server_optimizer_state_hash", "")) for row in rows]
    _append_check(
        checks,
        "FedAdam状态指纹逐轮变化",
        bool(all(hashes)) and len(set(hashes)) == ROUND_COUNT,
        True,
    )
    _append_check(
        checks,
        "调度覆盖无永久缺席",
        statistics_payload["zero_participation_clients"],
        0,
    )


def validate_result(
    result_dir: Path,
    scenario: Stage2Scenario,
    expected_partition_hash: Optional[str] = None,
    expected_initial_model_hash: Optional[str] = None,
) -> Dict[str, object]:
    """校验阶段二一次150轮结果及同seed可比性哈希。"""

    resolved_dir = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved_dir / "summary.json")
    config_snapshot = _load_json(resolved_dir / "config_snapshot.json")
    topology_metadata = _load_json(resolved_dir / "topology_metadata.json")
    participation = _load_json(
        resolved_dir / "dynamic_participation_summary.json"
    )
    rows = _load_metrics(resolved_dir / "metrics.csv")
    stats = schedule_statistics(scenario)
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.scenario_id,
        "architecture": "hfl",
        "snf_enabled": scenario.snf_enabled,
        "edge_mode": "fixed",
        "client_count": 37,
        "client_num_in_total": 37,
        "client_num_per_round": stats["participant_count_max"],
        "participant_count_min": stats["participant_count_min"],
        "participant_count_max": stats["participant_count_max"],
        "comm_round": ROUND_COUNT,
        "local_epochs": 3,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.05,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
        "server_bias_correction": scenario.setting.bias_correction,
        "server_optimizer_step_count": ROUND_COUNT,
        "client_optimizer_state_mode": "reset",
        "topology_schedule_hash": scenario.schedule_hash,
        "test_evaluation_performed": False,
        "final_test_metrics": None,
    }
    for field, value in expected_summary.items():
        _append_check(checks, field, summary.get(field), value)
    for field, value in expected_flat_config(scenario).items():
        _append_check(
            checks,
            "config_snapshot.{}".format(field),
            config_snapshot.get(field),
            value,
        )
    if scenario.topology_type == "matlab_direct":
        metadata_contract = {
            "provider_type": "matlab_adapter",
            "architecture": "hfl",
            "snf_enabled": scenario.snf_enabled,
            "edge_mode": "fixed",
            "topology_util": scenario.setting.topology_util,
            "round_count": 200,
            "source_round_count": 200,
            "topology_schedule_policy": "strict",
        }
    else:
        metadata_contract = {
            "provider_type": "fixed_count",
            "architecture": "hfl",
            "snf_enabled": False,
            "fixed_participant_count": scenario.participant_count,
            "fixed_group_count": 6,
            "fixed_count_selection_mode": "seeded_random",
            "fixed_count_seed": scenario.seed,
        }
    for field, value in metadata_contract.items():
        _append_check(
            checks,
            "topology_metadata.{}".format(field),
            topology_metadata.get(field),
            value,
        )
    _append_check(checks, "调度统计哈希", stats["schedule_hash"], scenario.schedule_hash)
    _append_check(checks, "参与汇总哈希", participation.get("schedule_hash"), scenario.schedule_hash)
    _append_check(
        checks,
        "参与汇总客户端计数",
        participation.get("client_selection_counts"),
        stats["client_selection_counts"],
    )
    partition_hash = str(summary.get("partition_hash", ""))
    initial_hash = str(summary.get("initial_model_hash", ""))
    _append_check(checks, "分区哈希格式", _is_sha256(partition_hash), True)
    _append_check(checks, "初始模型哈希格式", _is_sha256(initial_hash), True)
    if expected_partition_hash:
        _append_check(checks, "同seed分区哈希", partition_hash, expected_partition_hash)
    elif scenario.seed == SCREEN_SEED:
        _append_check(checks, "seed42分区哈希", partition_hash, PARTITION_HASH)
    if expected_initial_model_hash:
        _append_check(checks, "同seed初始模型哈希", initial_hash, expected_initial_model_hash)
    elif scenario.seed == SCREEN_SEED:
        _append_check(checks, "seed42初始模型哈希", initial_hash, INITIAL_MODEL_HASH)
    _validate_round_metrics(checks, rows, scenario, stats)
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "suite": SUITE_NAME,
        "scenario_id": scenario.scenario_id,
        "phase": scenario.phase,
        "arm": scenario.arm,
        "setting": scenario.setting.key,
        "seed": scenario.seed,
        "result_dir": str(resolved_dir),
        "partition_hash": partition_hash,
        "initial_model_hash": initial_hash,
        "expected_rounds": ROUND_COUNT,
        "schedule_statistics": stats,
        "checks": checks,
    }


def _mean(values: Iterable[float]) -> float:
    """返回一组非空浮点数的算术平均值。"""

    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("不能计算空序列均值")
    return sum(materialized) / len(materialized)


def _linear_slope(values: Sequence[float]) -> float:
    """按连续轮次计算一组数值的普通最小二乘斜率。"""

    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = _mean(values)
    numerator = sum(
        (index - x_mean) * (float(value) - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum(
        (index - x_mean) ** 2 for index in range(len(values))
    )
    return numerator / denominator


def _cold_start_statistics(
    values: Sequence[float],
    window: int = 20,
) -> Dict[str, object]:
    """计算早期最大峰谷回撤及首次恢复到峰值的轮次。"""

    early = [float(value) for value in values[: int(window)]]
    best = (0.0, 0, 0)
    for peak_index in range(max(0, len(early) - 1)):
        for valley_index in range(peak_index + 1, len(early)):
            drawdown = early[peak_index] - early[valley_index]
            if drawdown > best[0]:
                best = (drawdown, peak_index, valley_index)
    drawdown, peak_index, valley_index = best
    recovery_round: Optional[int] = None
    peak_value = early[peak_index] if early else 0.0
    if drawdown > 0.0:
        for index in range(valley_index + 1, len(values)):
            if float(values[index]) >= peak_value:
                recovery_round = index + 1
                break
    return {
        "peak_round": peak_index + 1 if early else None,
        "peak_mrr": peak_value if early else None,
        "valley_round": valley_index + 1 if early else None,
        "valley_mrr": early[valley_index] if early else None,
        "drawdown": drawdown,
        "recovery_round": recovery_round,
    }


def summarize_result(
    result_dir: Path,
    scenario: Stage2Scenario,
) -> Dict[str, object]:
    """提取候选选择、复验和绘图需要的全部阶段二指标。"""

    resolved_dir = Path(result_dir).expanduser().resolve()
    rows = _load_metrics(resolved_dir / "metrics.csv")
    summary = _load_json(resolved_dir / "summary.json")
    participation = _load_json(
        resolved_dir / "dynamic_participation_summary.json"
    )
    mrr = [_finite_number(row, "val_mrr") for row in rows]
    late_rows = rows[PLATFORM_START_ROUND - 1 : PLATFORM_END_ROUND]
    late_mrr = [_finite_number(row, "val_mrr") for row in late_rows]
    client_counts = sorted(
        int(value)
        for value in participation["client_selection_counts"].values()
    )
    cold = _cold_start_statistics(mrr)
    platform = {
        "round_start": PLATFORM_START_ROUND,
        "round_end": PLATFORM_END_ROUND,
        "mrr_mean": _mean(late_mrr),
        "mrr_std": statistics.pstdev(late_mrr),
        "mrr_slope": _linear_slope(late_mrr),
        "label": (
            "platform"
            if abs(_linear_slope(late_mrr)) <= 0.0005
            else "late_window"
        ),
        "hits_at_1_mean": _mean(
            _finite_number(row, "val_hits_at_1") for row in late_rows
        ),
        "hits_at_3_mean": _mean(
            _finite_number(row, "val_hits_at_3") for row in late_rows
        ),
        "hits_at_10_mean": _mean(
            _finite_number(row, "val_hits_at_10") for row in late_rows
        ),
    }
    return {
        "scenario_id": scenario.scenario_id,
        "phase": scenario.phase,
        "arm": scenario.arm,
        "setting": scenario.setting.key,
        "seed": scenario.seed,
        "topology_util": scenario.setting.topology_util,
        "server_bias_correction": scenario.setting.bias_correction,
        "participant_count": scenario.participant_count,
        "result_dir": str(resolved_dir),
        "partition_hash": str(summary.get("partition_hash", "")),
        "initial_model_hash": str(summary.get("initial_model_hash", "")),
        "platform": platform,
        "cold_start": cold,
        "early_server_update_l2_mean": _mean(
            _finite_number(row, "server_update_l2") for row in rows[:10]
        ),
        "cumulative_client_rounds": sum(
            int(_finite_number(row, "active_client_count")) for row in rows
        ),
        "client_participation_min": min(client_counts),
        "client_participation_median": statistics.median(client_counts),
        "client_participation_max": max(client_counts),
        "zero_participation_clients": sum(value == 0 for value in client_counts),
        "best_round": int(summary.get("best_round", 0)),
        "best_validation_mrr": float(
            summary.get("best_validation_mrr_during_training", "nan")
        ),
        "final_test_metrics": summary.get("final_test_metrics"),
        "rounds": [
            {
                "round": int(_finite_number(row, "round")),
                "val_mrr": _finite_number(row, "val_mrr"),
                "val_hits_at_1": _finite_number(row, "val_hits_at_1"),
                "val_hits_at_3": _finite_number(row, "val_hits_at_3"),
                "val_hits_at_10": _finite_number(row, "val_hits_at_10"),
                "server_update_l2": _finite_number(row, "server_update_l2"),
            }
            for row in rows
        ],
    }


def _first_sustained_round(
    values: Sequence[float],
    threshold: float,
    width: int = 5,
) -> int:
    """返回首次连续指定轮数达到阈值的轮次，未达到时记为151。"""

    materialized = [float(value) for value in values]
    for start in range(0, len(materialized) - int(width) + 1):
        if all(
            value >= float(threshold)
            for value in materialized[start : start + int(width)]
        ):
            return start + 1
    return ROUND_COUNT + 1


def _pair_analysis(
    setting: Stage2Setting,
    arm_results: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """计算一个参数组合两实验臂的平台与共同阈值收敛差距。"""

    hflsnf = arm_results["hflsnf"]
    hflnosnf = arm_results["hflnosnf"]
    hflsnf_platform = float(hflsnf["platform"]["mrr_mean"])
    hflnosnf_platform = float(hflnosnf["platform"]["mrr_mean"])
    threshold = 0.95 * min(hflsnf_platform, hflnosnf_platform)
    hflsnf_round = _first_sustained_round(
        [item["val_mrr"] for item in hflsnf["rounds"]],
        threshold,
    )
    hflnosnf_round = _first_sustained_round(
        [item["val_mrr"] for item in hflnosnf["rounds"]],
        threshold,
    )
    return {
        "setting": setting.key,
        "topology_util": setting.topology_util,
        "server_bias_correction": setting.bias_correction,
        "hflsnf_platform_mrr": hflsnf_platform,
        "hflnosnf_platform_mrr": hflnosnf_platform,
        "platform_gap": hflsnf_platform - hflnosnf_platform,
        "convergence_threshold": threshold,
        "hflsnf_convergence_round": hflsnf_round,
        "hflnosnf_convergence_round": hflnosnf_round,
        "convergence_round_gap": hflnosnf_round - hflsnf_round,
        "mean_cold_start_drawdown": _mean(
            (
                hflsnf["cold_start"]["drawdown"],
                hflnosnf["cold_start"]["drawdown"],
            )
        ),
        "zero_participation_clients": {
            "hflsnf": hflsnf["zero_participation_clients"],
            "hflnosnf": hflnosnf["zero_participation_clients"],
        },
    }


def select_screen_candidate(
    summaries: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """按预注册门槛、标准化得分和稳定决胜规则选择候选。"""

    grouped: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for item in summaries:
        grouped.setdefault(str(item["setting"]), {})[str(item["arm"])] = item
    pairs: Dict[str, Dict[str, object]] = {}
    for setting in SETTINGS:
        if set(grouped.get(setting.key, {})) != {"hflsnf", "hflnosnf"}:
            raise ValueError("筛选结果缺少完整实验臂：{}".format(setting.key))
        pairs[setting.key] = _pair_analysis(setting, grouped[setting.key])
    baseline = pairs[BASELINE_SETTING_KEY]
    baseline_hflsnf = float(baseline["hflsnf_platform_mrr"])
    candidates: List[Dict[str, object]] = []
    for setting in SETTINGS:
        if setting.key == BASELINE_SETTING_KEY:
            continue
        pair = dict(pairs[setting.key])
        platform_improvement = float(pair["platform_gap"]) - float(
            baseline["platform_gap"]
        )
        convergence_improvement = int(pair["convergence_round_gap"]) - int(
            baseline["convergence_round_gap"]
        )
        noninferior = (
            float(pair["hflsnf_platform_mrr"])
            >= baseline_hflsnf - 0.015
        )
        stable = all(
            int(value) == 0
            for value in pair["zero_participation_clients"].values()
        )
        qualified = bool(
            noninferior
            and stable
            and (
                platform_improvement >= 0.01
                or convergence_improvement >= 10
            )
        )
        pair.update(
            {
                "platform_gap_improvement": platform_improvement,
                "convergence_gap_improvement": convergence_improvement,
                "hflsnf_noninferior": noninferior,
                "stable_and_covered": stable,
                "qualified": qualified,
                "selection_score": (
                    platform_improvement / 0.01
                    + convergence_improvement / 10.0
                ),
            }
        )
        candidates.append(pair)
    qualified_candidates = [item for item in candidates if item["qualified"]]
    pool = qualified_candidates if qualified_candidates else candidates
    # 最终布尔键让bias=True在此前指标完全相同时优先。
    chosen = max(
        pool,
        key=lambda item: (
            float(item["selection_score"]),
            float(item["platform_gap"]),
            -float(item["mean_cold_start_drawdown"]),
            bool(item["server_bias_correction"]),
        ),
    )
    chosen_key = str(chosen["setting"])
    util_effects = {
        "bias_true": (
            float(pairs["u0p6_bctrue"]["platform_gap"])
            - float(pairs["u0p5_bctrue"]["platform_gap"])
        ),
        "bias_false": (
            float(pairs["u0p6_bcfalse"]["platform_gap"])
            - float(pairs["u0p5_bcfalse"]["platform_gap"])
        ),
    }
    bias_effects = {
        "util_0p5": (
            float(pairs["u0p5_bcfalse"]["platform_gap"])
            - float(pairs["u0p5_bctrue"]["platform_gap"])
        ),
        "util_0p6": (
            float(pairs["u0p6_bcfalse"]["platform_gap"])
            - float(pairs["u0p6_bctrue"]["platform_gap"])
        ),
    }
    return {
        "baseline_setting": BASELINE_SETTING_KEY,
        "selected_setting": chosen_key,
        "selection_label": "winner" if chosen["qualified"] else "challenger",
        "selection_reason": (
            "qualified_improvement"
            if chosen["qualified"]
            else "no_candidate_met_gate"
        ),
        "pairs": pairs,
        "candidates": candidates,
        "util_effects_on_platform_gap": util_effects,
        "bias_effects_on_platform_gap": bias_effects,
        "difference_in_differences": (
            util_effects["bias_false"] - util_effects["bias_true"]
        ),
    }


def control_counts_for_setting(
    setting: Stage2Setting,
) -> Tuple[int, int]:
    """返回胜出利用率对应的两种平均参与人数四舍五入值。"""

    if math.isclose(setting.topology_util, 0.5):
        return (36, 19)
    if math.isclose(setting.topology_util, 0.6):
        return (34, 12)
    raise ValueError("阶段二人数对照只支持util=0.5或0.6")


def build_followup_scenarios(
    batch_dir: Path,
    selected_setting_key: str,
) -> Tuple[Tuple[Stage2Scenario, ...], Tuple[Stage2Scenario, ...]]:
    """根据筛选结果构造8组复验与6组参与人数对照场景。"""

    resolved_batch = Path(batch_dir).expanduser().resolve()
    generated_dir = resolved_batch / "generated_configs"
    selected = setting_by_key(selected_setting_key)
    baseline = setting_by_key(BASELINE_SETTING_KEY)
    confirm: List[Stage2Scenario] = []
    for seed in CONFIRM_SEEDS:
        for setting in (baseline, selected):
            for arm in ("hflsnf", "hflnosnf"):
                config_path = generated_dir / (
                    "fedadam_stage2_confirm_{}_{}_seed{}_"
                    "150round_cuda.yaml".format(arm, setting.key, seed)
                )
                scenario = _dynamic_scenario(
                    "confirm", arm, setting, seed, config_path
                )
                write_generated_config(scenario)
                confirm.append(scenario)
    controls: List[Stage2Scenario] = []
    for participant_count in control_counts_for_setting(selected):
        for seed in ALL_SEEDS:
            scenario_id = "hflkge_k{}_{}_seed{}".format(
                participant_count, selected.key, seed
            )
            config_path = generated_dir / (
                "fedadam_stage2_control_k{}_{}_seed{}_"
                "150round_cuda.yaml".format(
                    participant_count, selected.key, seed
                )
            )
            provisional = Stage2Scenario(
                scenario_id=scenario_id,
                phase="controls",
                arm="hflkge",
                seed=seed,
                setting=selected,
                config_path=config_path,
                topology_type="fixed_count",
                snf_enabled=False,
                schedule_hash="",
                participant_count=participant_count,
            )
            provider = build_control_provider(provisional)
            scenario = Stage2Scenario(
                **{
                    **provisional.__dict__,
                    "schedule_hash": _schedule_hash(provider, ROUND_COUNT),
                }
            )
            write_generated_config(scenario)
            controls.append(scenario)
    return tuple(confirm), tuple(controls)


def scenario_to_manifest_entry(
    scenario: Stage2Scenario,
) -> Dict[str, object]:
    """把不可变场景转换为可持久化的批次项目。"""

    return {
        "scenario_id": scenario.scenario_id,
        "phase": scenario.phase,
        "arm": scenario.arm,
        "seed": scenario.seed,
        "setting": scenario.setting.key,
        "topology_util": scenario.setting.topology_util,
        "server_bias_correction": scenario.setting.bias_correction,
        "topology_type": scenario.topology_type,
        "snf_enabled": scenario.snf_enabled,
        "participant_count": scenario.participant_count,
        "config": str(scenario.config_path),
        "schedule_hash": scenario.schedule_hash,
        "status": "pending",
        "result_dir": None,
        "contract_file": None,
        "analysis": None,
        "error": None,
        "attempts": [],
    }


def scenario_from_manifest_entry(
    entry: Mapping[str, object],
) -> Stage2Scenario:
    """从批次项目重建场景并拒绝被手工替换的调度哈希。"""

    setting = setting_by_key(str(entry["setting"]))
    topology_type = str(entry["topology_type"])
    participant = entry.get("participant_count")
    scenario = Stage2Scenario(
        scenario_id=str(entry["scenario_id"]),
        phase=str(entry["phase"]),
        arm=str(entry["arm"]),
        seed=int(entry["seed"]),
        setting=setting,
        config_path=Path(str(entry["config"])).resolve(),
        topology_type=topology_type,
        snf_enabled=bool(entry["snf_enabled"]),
        schedule_hash=str(entry["schedule_hash"]),
        participant_count=(
            int(participant) if participant is not None else None
        ),
    )
    actual_hash = schedule_statistics(scenario)["schedule_hash"]
    if actual_hash != scenario.schedule_hash:
        raise ValueError(
            "批次项目调度哈希与当前场景不一致：{}".format(
                scenario.scenario_id
            )
        )
    return scenario


def stable_payload_hash(payload: Mapping[str, object]) -> str:
    """返回批次绑定字段的规范化SHA-256，防止恢复时换参。"""

    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
