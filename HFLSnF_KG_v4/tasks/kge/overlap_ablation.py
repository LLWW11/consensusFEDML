"""V4实体重叠率正式消融的校准、配置、基线和结果合同。"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .data import load_fb15k237
from .federated_data import (
    BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
    partition_train_triples_by_overlap_target,
)
from .final_dynamic_fedadam import (
    ARM_CONTRACTS,
    MAT_RELATIVE_PATH,
    PARTITION_HASHES as BASELINE_PARTITION_HASHES,
    ROUND_COUNT,
    FinalDynamicScenario,
    _is_sha256,
    schedule_statistics,
    validate_result as validate_baseline_result,
)
from .fixed_count_four_scenarios import (
    PACKAGE_DIR,
    _append_check,
    _finite_number,
    _load_json,
    _load_metrics,
    load_flat_config,
)


SUITE_NAME = "v4_hflsnf_entity_overlap_fedadam_u0p6_e3_eval1_formal150"
LEVELS: Tuple[str, ...] = ("low", "medium", "high")
SEEDS: Tuple[int, ...] = (42, 2024, 2025)
CALIBRATION_CONTRACT_PATH = (
    PACKAGE_DIR / "configs" / "overlap" / "partition_calibration_contract.json"
).resolve()
BASELINE_INITIAL_MODEL_HASHES = {
    42: "37e5e4ee9af5e7a774027486be6b571f7ec9e18d18ec908e354379c8a761789e",
    2024: "6ceff922d3d8ca72b3132ba2d25ea6e47de48724d38a4bdb459b0eb4a7db90dd",
    2025: "d47cc3d7634f6fd031461ac8e4991d770e4d71f001da0930cf2a26e302eeb78a",
}
BASELINE_RESULT_NAMES = {
    42: (
        "hflsnf_kg_v3_final_dynamic_fedadam_hflsnf_u0p6_bcfalse_"
        "seed42_150round_cuda_20260814_103232_392496"
    ),
    2024: (
        "hflsnf_kg_v3_final_dynamic_fedadam_hflsnf_u0p6_bcfalse_"
        "seed2024_150round_cuda_20260814_135116_313847"
    ),
    2025: (
        "hflsnf_kg_v3_final_dynamic_fedadam_hflsnf_u0p6_bcfalse_"
        "seed2025_150round_cuda_20260814_170841_038096"
    ),
}
BASELINE_RESULT_DIRS = {
    seed: (
        PACKAGE_DIR / "results" / "三个随机数种子" / result_name
    ).resolve()
    for seed, result_name in BASELINE_RESULT_NAMES.items()
}


@dataclass(frozen=True)
class OverlapScenario:
    """描述一个重叠档位、随机种子及其冻结分区合同。"""

    scenario_id: str
    level: str
    seed: int
    target_entity_overlap: float
    partition_hash: str
    config_path: Path


def _file_sha256(path: Path) -> str:
    """分块计算文件SHA-256，用于绑定数据和配置输入。"""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration_contract(
    path: Path = CALIBRATION_CONTRACT_PATH,
) -> Dict[str, object]:
    """读取并校验正式8重启校准合同的固定结构和阈值。"""

    resolved = Path(path).expanduser().resolve()
    payload = _load_json(resolved)
    required_equal = {
        "status": "passed",
        "contract_schema_version": 1,
        "partition_strategy": BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
        "client_count": 37,
        "seeds": list(SEEDS),
        "search_seed_policy": "experiment_seed",
    }
    for field, expected in required_equal.items():
        if payload.get(field) != expected:
            raise ValueError("校准合同字段{}不符合正式要求".format(field))
    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        raise TypeError("校准合同constraints必须是对象")
    expected_constraints = {
        "overlap_tolerance": 0.005,
        "load_tolerance": 0.05,
        "relation_overlap_tolerance": 0.02,
        "search_restarts": 8,
        "minimum_overlap_span": 0.06,
    }
    for field, expected in expected_constraints.items():
        if float(constraints.get(field, float("nan"))) != float(expected):
            raise ValueError("校准约束{}不符合正式要求".format(field))
    verification = payload.get("reproduction_verification")
    if not isinstance(verification, dict) or verification.get("status") != "passed":
        raise ValueError("正式校准尚未通过第二次独立复现")
    interval = payload.get("common_reachable_interval")
    if not isinstance(interval, dict) or float(interval.get("span", 0.0)) < 0.06:
        raise ValueError("正式校准共同可达区间跨度不足0.06")
    levels = payload.get("levels")
    if not isinstance(levels, dict) or set(levels) != set(LEVELS):
        raise ValueError("正式校准必须保存low、medium、high三个档位")
    hashes = []
    previous_target = -1.0
    for level_name in LEVELS:
        level = levels[level_name]
        if not isinstance(level, dict):
            raise TypeError("校准档位{}必须是对象".format(level_name))
        target = float(level["target_entity_overlap"])
        if target <= previous_target:
            raise ValueError("低中高目标实体重叠率必须严格递增")
        previous_target = target
        per_seed = level.get("per_seed")
        if not isinstance(per_seed, dict) or set(per_seed) != {str(x) for x in SEEDS}:
            raise ValueError("档位{}缺少三个种子".format(level_name))
        for seed in SEEDS:
            summary = per_seed[str(seed)]
            if not isinstance(summary, dict):
                raise TypeError("档位种子摘要必须是对象")
            if float(summary["entity_overlap_absolute_error"]) > 0.005:
                raise ValueError("档位{} seed{}实体目标误差超限".format(level_name, seed))
            if float(summary["relation_overlap_absolute_error"]) > 0.02:
                raise ValueError("档位{} seed{}关系误差超限".format(level_name, seed))
            if float(summary["max_relative_load_deviation"]) > 0.05:
                raise ValueError("档位{} seed{}负载偏差超限".format(level_name, seed))
            partition_hash = str(summary["partition_hash"])
            if not _is_sha256(partition_hash):
                raise ValueError("档位{} seed{}分区哈希无效".format(level_name, seed))
            hashes.append(partition_hash)
    if len(set(hashes)) != 9:
        raise ValueError("九个正式重叠分区哈希必须互不相同")
    return payload


def scenarios_from_contract(
    contract: Optional[Mapping[str, object]] = None,
) -> Tuple[OverlapScenario, ...]:
    """按seed优先、档位次序从正式校准合同构造九个场景。"""

    payload = dict(contract) if contract is not None else load_calibration_contract()
    levels = payload["levels"]
    assert isinstance(levels, dict)
    scenarios = []
    for seed in SEEDS:
        for level_name in LEVELS:
            level = levels[level_name]
            assert isinstance(level, dict)
            per_seed = level["per_seed"]
            assert isinstance(per_seed, dict)
            seed_summary = per_seed[str(seed)]
            assert isinstance(seed_summary, dict)
            scenario_id = "overlap_{}_seed{}".format(level_name, seed)
            scenarios.append(
                OverlapScenario(
                    scenario_id=scenario_id,
                    level=level_name,
                    seed=seed,
                    target_entity_overlap=float(level["target_entity_overlap"]),
                    partition_hash=str(seed_summary["partition_hash"]),
                    config_path=(
                        PACKAGE_DIR
                        / "configs"
                        / "overlap"
                        / "{}_150round_cuda.yaml".format(scenario_id)
                    ).resolve(),
                )
            )
    return tuple(scenarios)


def scenario_by_id(
    scenario_id: str,
    contract: Optional[Mapping[str, object]] = None,
) -> OverlapScenario:
    """按稳定场景身份返回一个正式重叠率场景。"""

    for scenario in scenarios_from_contract(contract):
        if scenario.scenario_id == str(scenario_id):
            return scenario
    raise KeyError("未知重叠率实验场景：{}".format(scenario_id))


def _baseline_config_path(seed: int) -> Path:
    """返回同种子HFLSnF原始对照配置路径。"""

    name = (
        "final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed{}_"
        "150round_cuda.yaml"
    ).format(seed)
    return (PACKAGE_DIR / "configs" / "dynamic" / name).resolve()


def _topology_scenario(seed: int) -> FinalDynamicScenario:
    """构造只用于复用HFLSnF拓扑统计的动态场景适配器。"""

    return FinalDynamicScenario(
        scenario_id="overlap_topology_seed{}".format(seed),
        arm="hflsnf",
        seed=int(seed),
        config_path=_baseline_config_path(seed),
        contract=ARM_CONTRACTS["hflsnf"],
    )


def expected_flat_config(scenario: OverlapScenario) -> Dict[str, object]:
    """从同种子原始HFLSnF配置派生唯一合法的重叠率配置。"""

    expected = dict(load_flat_config(_baseline_config_path(scenario.seed)))
    expected.update(
        {
            "partition_strategy": BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
            "partition_target_entity_overlap": scenario.target_entity_overlap,
            "partition_overlap_tolerance": 0.005,
            "partition_load_tolerance": 0.05,
            "relation_overlap_tolerance": 0.02,
            "partition_search_seed": scenario.seed,
            "partition_search_restarts": 8,
            "expected_partition_hash": scenario.partition_hash,
            "ablation_suite": SUITE_NAME,
            "ablation_arm": scenario.scenario_id,
            "comparison_scenario": (
                "V4-HFLSnF-EntityOverlap-{}-Seed{}-Formal150"
            ).format(scenario.level.upper(), scenario.seed),
            "run_name": (
                "hflsnf_kg_v4_entity_overlap_{}_seed{}_150round_cuda"
            ).format(scenario.level, scenario.seed),
        }
    )
    return expected


def _validate_data_files(
    contract: Mapping[str, object],
    checks: List[Dict[str, object]],
) -> None:
    """校验当前FB15k-237文本与正式校准绑定的三个哈希。"""

    data_files = contract.get("data_files")
    if not isinstance(data_files, dict):
        raise TypeError("校准合同data_files必须是对象")
    for name in ("train.txt", "valid.txt", "test.txt"):
        item = data_files.get(name)
        if not isinstance(item, dict):
            raise KeyError("校准合同缺少{}哈希".format(name))
        path = PACKAGE_DIR / "data" / "FB15k-237" / name
        _append_check(
            checks,
            "数据文件{}.sha256".format(name),
            _file_sha256(path),
            str(item.get("sha256", "")),
        )


def _validate_baselines(checks: List[Dict[str, object]]) -> None:
    """复核三个原始HFLSnF结果及固定初始模型哈希。"""

    for seed in SEEDS:
        scenario = FinalDynamicScenario(
            scenario_id="final_hflsnf_u0p6_bcfalse_seed{}".format(seed),
            arm="hflsnf",
            seed=seed,
            config_path=_baseline_config_path(seed),
            contract=ARM_CONTRACTS["hflsnf"],
        )
        report = validate_baseline_result(
            BASELINE_RESULT_DIRS[seed],
            scenario,
            expected_initial_model_hash=BASELINE_INITIAL_MODEL_HASHES[seed],
        )
        _append_check(checks, "seed{}原始结果合同".format(seed), report["status"], "passed")
        _append_check(
            checks,
            "seed{}原始分区哈希".format(seed),
            report["partition_hash"],
            BASELINE_PARTITION_HASHES[seed],
        )


def validate_configs(
    recompute_partitions: bool = True,
    contract_path: Path = CALIBRATION_CONTRACT_PATH,
) -> Dict[str, object]:
    """校验正式校准、九份YAML、基线、拓扑及可选分区复算。"""

    contract = load_calibration_contract(contract_path)
    scenarios = scenarios_from_contract(contract)
    checks: List[Dict[str, object]] = []
    _validate_data_files(contract, checks)
    _validate_baselines(checks)
    schedule_hash = ARM_CONTRACTS["hflsnf"].schedule_hash
    dataset = (
        load_fb15k237(PACKAGE_DIR / "data" / "FB15k-237")
        if recompute_partitions
        else None
    )
    for scenario in scenarios:
        config = load_flat_config(scenario.config_path)
        expected = expected_flat_config(scenario)
        _append_check(checks, "{}.完整配置".format(scenario.scenario_id), config, expected)
        stats = schedule_statistics(_topology_scenario(scenario.seed))
        _append_check(
            checks,
            "{}.拓扑调度哈希".format(scenario.scenario_id),
            stats["schedule_hash"],
            schedule_hash,
        )
        if dataset is not None:
            partition = partition_train_triples_by_overlap_target(
                dataset=dataset,
                client_count=37,
                seed=scenario.seed,
                target_entity_overlap=scenario.target_entity_overlap,
                overlap_tolerance=0.005,
                load_tolerance=0.05,
                relation_overlap_tolerance=0.02,
                search_restarts=8,
                search_seed=scenario.seed,
                strict=True,
            )
            _append_check(
                checks,
                "{}.重算分区哈希".format(scenario.scenario_id),
                partition.partition_hash,
                scenario.partition_hash,
            )
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "suite": SUITE_NAME,
        "scenario_order": [item.scenario_id for item in scenarios],
        "recompute_partitions": bool(recompute_partitions),
        "checks": checks,
    }


def _validate_round_metrics(
    checks: List[Dict[str, object]],
    rows: Sequence[Mapping[str, object]],
    scenario: OverlapScenario,
) -> None:
    """校验150轮编号、动态参与者、FedAdam步数和有限指标。"""

    stats = schedule_statistics(_topology_scenario(scenario.seed))
    expected_rounds = list(range(1, ROUND_COUNT + 1))
    actual_rounds = [int(_finite_number(row, "round")) for row in rows]
    _append_check(checks, "逐轮编号", actual_rounds, expected_rounds)
    if actual_rounds != expected_rounds:
        return
    topology = _topology_scenario(scenario.seed)
    from ...core.topology import MatlabTopologyProvider

    provider = MatlabTopologyProvider(
        mat_path=(PACKAGE_DIR / MAT_RELATIVE_PATH).resolve(),
        architecture=topology.contract.architecture,
        snf_enabled=topology.contract.snf_enabled,
        edge_mode=topology.contract.edge_mode,
        util=0.6,
        client_count=37,
        schedule_policy="strict",
    )
    topologies = tuple(provider.get_round(index) for index in range(ROUND_COUNT))
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


def validate_result(
    result_dir: Path,
    scenario: OverlapScenario,
) -> Dict[str, object]:
    """校验一个150轮重叠率结果及其与原始对照的可比性。"""

    resolved = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved / "summary.json")
    snapshot = _load_json(resolved / "config_snapshot.json")
    metadata = _load_json(resolved / "topology_metadata.json")
    participation = _load_json(resolved / "dynamic_participation_summary.json")
    partition_summary = _load_json(resolved / "client_partition_summary.json")
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
    config = load_flat_config(scenario.config_path)
    for field, expected in config.items():
        _append_check(checks, "config_snapshot.{}".format(field), snapshot.get(field), expected)
    for field, expected in {
        "provider_type": "matlab_adapter",
        "architecture": "hfl",
        "snf_enabled": True,
        "edge_mode": "fixed",
        "topology_util": 0.6,
        "topology_schedule_policy": "strict",
    }.items():
        _append_check(checks, "topology_metadata.{}".format(field), metadata.get(field), expected)
    _append_check(
        checks,
        "参与汇总哈希",
        participation.get("schedule_hash"),
        ARM_CONTRACTS["hflsnf"].schedule_hash,
    )
    _append_check(checks, "分区哈希", summary.get("partition_hash"), scenario.partition_hash)
    _append_check(
        checks,
        "同seed原始初始模型哈希",
        summary.get("initial_model_hash"),
        BASELINE_INITIAL_MODEL_HASHES[scenario.seed],
    )
    partition_expected = {
        "partition_strategy": BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
        "partition_hash": scenario.partition_hash,
        "target_entity_overlap": scenario.target_entity_overlap,
        "search_seed": scenario.seed,
        "search_restarts": 8,
    }
    for field, expected in partition_expected.items():
        _append_check(checks, "分区摘要.{}".format(field), partition_summary.get(field), expected)
    for field, limit in {
        "entity_overlap_absolute_error": 0.005,
        "relation_overlap_absolute_error": 0.02,
        "max_relative_load_deviation": 0.05,
    }.items():
        actual = float(partition_summary.get(field, float("inf")))
        _append_check(checks, "分区摘要.{}上限".format(field), actual <= limit, True)
    _validate_round_metrics(checks, rows, scenario)
    return {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "suite": SUITE_NAME,
        "scenario_id": scenario.scenario_id,
        "level": scenario.level,
        "seed": scenario.seed,
        "result_dir": str(resolved),
        "partition_hash": str(summary.get("partition_hash", "")),
        "initial_model_hash": str(summary.get("initial_model_hash", "")),
        "checks": checks,
    }


def convergence_statistics(rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    """计算验证MRR曲线的面积、后20轮趋势和95%到达轮次。"""

    values = [float(_finite_number(row, "val_mrr")) for row in rows]
    tail = values[-20:]
    x_values = list(range(20))
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(tail)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, tail)
    ) / denominator
    threshold = 0.95 * y_mean
    first_round = next(
        (index + 1 for index, value in enumerate(values) if value >= threshold),
        len(values),
    )
    area = sum(
        (values[index - 1] + values[index]) / 2.0
        for index in range(1, len(values))
    )
    return {
        "best_validation_mrr": max(values),
        "last20_validation_mrr_mean": y_mean,
        "last20_validation_mrr_slope": slope,
        "validation_mrr_auc": area,
        "round_to_95pct_last20_mean": float(first_round),
    }


def communication_statistics(
    rows: Sequence[Mapping[str, object]],
    num_entities: int = 14541,
    num_relations: int = 237,
    embedding_dim: int = 256,
) -> Dict[str, int]:
    """按全局14541行参数表和本地活动行计算两类上传字节数。"""

    bytes_per_row = int(embedding_dim) * 4
    dense_rows = sum(
        int(_finite_number(row, "active_client_count"))
        * (int(num_entities) + int(num_relations))
        for row in rows
    )
    logical_rows = sum(
        int(_finite_number(row, "entity_total_row_occurrences"))
        + int(_finite_number(row, "relation_total_row_occurrences"))
        for row in rows
    )
    return {
        "actual_dense_upload_bytes": dense_rows * bytes_per_row,
        "logical_sparse_activity_bytes": logical_rows * bytes_per_row,
        "logical_sparse_activity_rows": logical_rows,
    }
