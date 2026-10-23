"""固定HFLSnF和种子42的FedAvg四学习率配置与结果合同。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .fixed_count_four_scenarios import (
    PACKAGE_DIR, _append_check, _finite_number, _load_json, _load_metrics,
    load_flat_config,
)
from .graph_semantic_ablation import (
    _validate_data_files, load_calibration_contract,
)
from .graph_semantic_fedavg_comparison import (
    GraphSemanticFedAvgScenario, _load_json_lines, _topology_scenario,
    _validate_round_metrics, expected_flat_config as source_expected_config,
    scenario_by_id as source_scenario_by_id,
)
from .final_dynamic_fedadam import schedule_statistics
from .overlap_ablation import BASELINE_INITIAL_MODEL_HASHES
from .data import load_fb15k237
from .graph_semantic_partition import (
    partition_train_triples_by_semantic_domain_graph_local,
)


SUITE_NAME = "v5_graph_semantic_fedavg_hflsnf_seed42_lr_sweep_formal150"
RESULT_ROOT = "results/graph_semantic_fedavg_lr_sweep"
CONFIG_DIR = PACKAGE_DIR / "configs" / "graph_semantic_fedavg_lr_sweep"
SOURCE_SCENARIO_ID = "graph_semantic_fedavg_hflsnf_seed42"
ROUND_COUNT = 150
LEARNING_RATES = (
    ("lr5e-5", 5e-5), ("lr1e-4", 1e-4),
    ("lr2e-4", 2e-4), ("lr5e-4", 5e-4),
)


@dataclass(frozen=True)
class LearningRateScenario:
    """保存学习率身份及其对应的冻结图语义场景。"""

    tag: str
    learning_rate: float
    scenario: GraphSemanticFedAvgScenario


def scenarios() -> Tuple[LearningRateScenario, ...]:
    """按学习率升序构造四个HFLSnF、种子42场景。"""

    source = source_scenario_by_id(SOURCE_SCENARIO_ID)
    items = []
    for tag, rate in LEARNING_RATES:
        identity = "graph_semantic_fedavg_hflsnf_seed42_{}".format(tag)
        config = CONFIG_DIR / (identity + "_150round_cuda.yaml")
        items.append(LearningRateScenario(tag, rate, replace(
            source, scenario_id=identity, config_path=config.resolve(),
            source_config_path=source.config_path,
        )))
    return tuple(items)


def expected_flat_config(item: LearningRateScenario) -> Dict[str, object]:
    """从原FedAvg配方派生仅改变学习率和审计身份的新配置。"""

    expected = dict(load_flat_config(item.scenario.source_config_path))
    expected.update({
        "learning_rate": item.learning_rate,
        "lr": item.learning_rate,
        "ablation_suite": SUITE_NAME,
        "ablation_arm": item.scenario.scenario_id,
        "comparison_scenario": (
            "V5-HFLSnF-GraphSemantic-FedAvg-Seed42-{}-Formal150"
        ).format(item.tag),
        "run_name": "hflsnf_kg_v5_" + item.scenario.scenario_id + "_150round_cuda",
        "result_root": RESULT_ROOT,
    })
    return expected


def validate_configs(recompute_partitions: bool = True) -> Dict[str, object]:
    """检查来源、四份完整配置、真实数据和150轮拓扑及可选分区复算。"""

    source = source_scenario_by_id(SOURCE_SCENARIO_ID)
    source_config = load_flat_config(source.config_path)
    checks: List[Dict[str, object]] = []
    _append_check(checks, "原FedAvg完整配置", source_config,
                  source_expected_config(source))
    # 显式锁住关键基线，避免来源配置漂移后派生出不同实验。
    for field, expected in {
        "random_seed": 42, "learning_rate": 5e-5, "lr": 5e-5,
        "server_optimizer": "fedavg", "client_optimizer": "adam",
        "client_optimizer_state_mode": "reset", "epochs": 3,
        "comm_round": ROUND_COUNT, "require_cuda": True,
        "expected_partition_hash": source.partition_hash,
        "expected_topology_schedule_hash": source.topology_contract.schedule_hash,
    }.items():
        _append_check(checks, "来源.{}".format(field), source_config.get(field), expected)
    calibration = load_calibration_contract()
    _validate_data_files(calibration, checks)
    stats = schedule_statistics(_topology_scenario(source))
    _append_check(checks, "真实150轮拓扑哈希", stats["schedule_hash"],
                  source.topology_contract.schedule_hash)
    if recompute_partitions:
        dataset = load_fb15k237(PACKAGE_DIR / "data" / "FB15k-237")
        partition = partition_train_triples_by_semantic_domain_graph_local(
            dataset=dataset, client_count=37, seed=42,
            load_tolerance=0.05, search_restarts=8,
        )
        _append_check(checks, "真实seed42分区哈希", partition.partition_hash,
                      source.partition_hash)
    for item in scenarios():
        _append_check(checks, item.scenario.scenario_id + ".完整配置",
                      load_flat_config(item.scenario.config_path), expected_flat_config(item))
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "suite": SUITE_NAME, "recompute_partitions": recompute_partitions,
        "partition_hash": source.partition_hash,
        "topology_hash": stats["schedule_hash"], "checks": checks,
    }


def validate_result(result_dir: Path, item: LearningRateScenario,
                    suite_name: str = SUITE_NAME,
                    configuration: Optional[Dict[str, object]] = None,
                    check_participation_budget: bool = False) -> Dict[str, object]:
    """校验指定实验身份、学习率和拓扑，默认维持四学习率结果合同。"""

    config = expected_flat_config(item) if configuration is None else configuration
    topology = item.scenario.topology_contract
    directory = Path(result_dir).resolve()
    summary = _load_json(directory / "summary.json")
    snapshot = _load_json(directory / "config_snapshot.json")
    partition = _load_json(directory / "client_partition_summary.json")
    participation = _load_json(directory / "dynamic_participation_summary.json")
    metadata = _load_json(directory / "topology_metadata.json")
    rows = _load_metrics(directory / "metrics.csv")
    audits = _load_json_lines(directory / "dynamic_topology_schedule.jsonl")
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0", "ablation_suite": suite_name,
        "ablation_arm": item.scenario.scenario_id,
        "scenario": config["comparison_scenario"],
        "architecture": topology.architecture, "snf_enabled": topology.snf_enabled,
        "edge_mode": topology.edge_mode,
        "client_count": 37, "client_num_in_total": 37,
        "comm_round": ROUND_COUNT, "local_epochs": 3,
        "learning_rate": item.learning_rate, "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedavg", "server_learning_rate": 1.0,
        "server_beta1": 0.0, "server_beta2": 0.0, "server_tau": 0.0,
        "server_bias_correction": False, "server_optimizer_step_count": 0,
        "server_optimizer_state_hash": "", "client_optimizer_state_mode": "reset",
        "initial_model_hash": BASELINE_INITIAL_MODEL_HASHES[item.scenario.seed],
        "partition_hash": item.scenario.partition_hash,
        "topology_schedule_hash": item.scenario.topology_contract.schedule_hash,
        "test_evaluation_performed": False, "final_test_metrics": None,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, "summary." + field, summary.get(field), expected)
    for field, expected in config.items():
        _append_check(checks, "snapshot." + field, snapshot.get(field), expected)
    _append_check(checks, "分区摘要哈希", partition.get("partition_hash"),
                  item.scenario.partition_hash)
    _append_check(checks, "参与摘要哈希", participation.get("schedule_hash"),
                  item.scenario.topology_contract.schedule_hash)
    _append_check(checks, "分区负载上限",
                  float(partition.get("max_relative_load_deviation", float("inf")))
                  <= 0.05 + 1e-12, True)
    for field, expected in {
        "provider_type": "matlab_adapter", "architecture": topology.architecture,
        "snf_enabled": topology.snf_enabled, "edge_mode": topology.edge_mode, "topology_util": 0.6,
        "topology_schedule_policy": "strict",
    }.items():
        _append_check(checks, "metadata." + field, metadata.get(field), expected)
    _validate_round_metrics(checks, rows, audits, item.scenario)
    if check_participation_budget:
        # 新三拓扑报告使用这些预算字段，必须与逐轮参与审计一致。
        for summary_field, row_field in (
            ("participant_count_mean", "active_client_count"),
            ("group_count_mean", "active_group_count"),
        ):
            mean = sum(_finite_number(row, row_field) for row in rows) / len(rows)
            _append_check(checks, "summary." + summary_field, summary.get(summary_field), mean)
        _append_check(checks, "有效数据遍数有限且为正",
                      _finite_number(summary, "effective_global_passes") > 0, True)
    # 使用训练曲线校核最佳模型，避免把恢复后的最佳验证值误当作第150轮值。
    best = max(rows, key=lambda row: _finite_number(row, "val_mrr"))
    _append_check(checks, "最佳轮次", summary.get("best_round"), int(best["round"]))
    _append_check(checks, "最佳验证MRR", summary.get("best_validation_mrr_during_training"),
                  _finite_number(best, "val_mrr"))
    _append_check(checks, "最佳检查点存在", (directory / "model_best.pt").is_file(), True)
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "suite": suite_name, "scenario_id": item.scenario.scenario_id,
        "learning_rate": item.learning_rate, "result_dir": str(directory),
        "checks": checks,
    }
