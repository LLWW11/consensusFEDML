"""冻结本地学习率5e-4的种子42三拓扑配置与结果合同。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from .fixed_count_four_scenarios import PACKAGE_DIR, _append_check, load_flat_config
from .graph_semantic_ablation import _validate_data_files, load_calibration_contract
from .graph_semantic_fedavg_comparison import (
    ARM_ORDER, _arm_label, _topology_scenario,
    expected_flat_config as source_expected_config, scenario_by_id,
)
from .graph_semantic_fedavg_lr_sweep import (
    LearningRateScenario, validate_result as validate_monitored_result,
)
from .final_dynamic_fedadam import schedule_statistics
from .data import load_fb15k237
from .graph_semantic_partition import partition_train_triples_by_semantic_domain_graph_local


SUITE_NAME = "v5_graph_semantic_fedavg_seed42_lr5e-4_three_topology_formal150"
RESULT_ROOT = "results/graph_semantic_fedavg_seed42_lr5e-4"
CONFIG_DIR = PACKAGE_DIR / "configs" / "graph_semantic_fedavg_seed42_lr5e-4"
LEARNING_RATE = 5e-4
ROUND_COUNT = 150
# 拓扑之外的所有有效参数必须相同；审计身份不参与训练语义。
TOPOLOGY_AND_IDENTITY_FIELDS = {
    "topology_architecture", "topology_snf", "topology_edge_mode", "edge_num",
    "expected_topology_schedule_hash", "ablation_suite", "ablation_arm",
    "comparison_scenario", "run_name", "result_root",
}


def scenarios() -> Tuple[LearningRateScenario, ...]:
    """只生成seed42的HFLSnF、HFLnoSnF和FLnoSnF三个独立新场景。"""

    items = []
    for arm in ARM_ORDER:
        source = scenario_by_id("graph_semantic_fedavg_{}_seed42".format(arm))
        identity = source.scenario_id + "_lr5e-4"
        items.append(LearningRateScenario(arm, LEARNING_RATE, replace(
            source, scenario_id=identity,
            config_path=(CONFIG_DIR / (identity + "_150round_cuda.yaml")).resolve(),
            source_config_path=source.config_path,
        )))
    return tuple(items)


def expected_flat_config(item: LearningRateScenario) -> Dict[str, object]:
    """从同拓扑原FedAvg配置派生，仅改变学习率和独立实验身份。"""

    expected = dict(load_flat_config(item.scenario.source_config_path))
    expected.update(
        learning_rate=LEARNING_RATE, lr=LEARNING_RATE, ablation_suite=SUITE_NAME,
        ablation_arm=item.scenario.scenario_id,
        comparison_scenario="V5-{}-GraphSemantic-FedAvg-Seed42-lr5e-4-Formal150".format(
            _arm_label(item.scenario.arm)),
        run_name="hflsnf_kg_v5_" + item.scenario.scenario_id + "_150round_cuda",
        result_root=RESULT_ROOT,
    )
    return expected


def validate_scenario_configs(
    items: Sequence[LearningRateScenario], suite_name: str,
    configuration_builder: Callable[[LearningRateScenario], Dict[str, object]],
    recompute_partitions: bool = True,
) -> Dict[str, object]:
    """复用固定学习率拓扑校验，每种子复算一次分区并验证同种子三臂。"""

    checks: List[Dict[str, object]] = []
    calibration = load_calibration_contract()
    _validate_data_files(calibration, checks)
    topology_hashes = {}
    partition_hashes = {}
    common = None
    for item in items:
        seed = item.scenario.seed
        source = scenario_by_id("graph_semantic_fedavg_{}_seed{}".format(item.scenario.arm, seed))
        _append_check(checks, item.tag + ".原FedAvg完整配置",
                      load_flat_config(source.config_path), source_expected_config(source))
        actual = load_flat_config(item.scenario.config_path)
        _append_check(checks, item.tag + ".新完整配置", actual, configuration_builder(item))
        for field, expected in {
            "random_seed": seed, "partition_search_seed": seed,
            "learning_rate": LEARNING_RATE, "lr": LEARNING_RATE,
            "client_optimizer": "adam", "client_optimizer_state_mode": "reset",
            "server_optimizer": "fedavg", "epochs": 3, "comm_round": ROUND_COUNT,
            "require_cuda": True, "evaluate_test_after_training": False,
        }.items():
            _append_check(checks, item.tag + "." + field, actual.get(field), expected)
        seed_fields = {"random_seed", "partition_search_seed", "expected_partition_hash"}
        effective = {key: value for key, value in actual.items()
                     if key not in TOPOLOGY_AND_IDENTITY_FIELDS | seed_fields}
        if common is None:
            common = effective
        _append_check(checks, item.tag + ".共同训练条件", effective, common)
        stats = schedule_statistics(_topology_scenario(item.scenario))
        topology_hashes[item.scenario.arm] = stats["schedule_hash"]
        _append_check(checks, item.tag + ".真实150轮调度哈希", stats["schedule_hash"],
                      item.scenario.topology_contract.schedule_hash)
        expected_partition = partition_hashes.setdefault(str(seed), source.partition_hash)
        _append_check(checks, item.tag + ".共同分区哈希", item.scenario.partition_hash,
                      expected_partition)
    if recompute_partitions:
        dataset = load_fb15k237(PACKAGE_DIR / "data" / "FB15k-237")
        for seed, expected in partition_hashes.items():
            partition = partition_train_triples_by_semantic_domain_graph_local(
                dataset=dataset, client_count=37, seed=int(seed),
                load_tolerance=0.05, search_restarts=8,
            )
            _append_check(checks, "真实seed{}分区哈希".format(seed), partition.partition_hash, expected)
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "suite": suite_name, "recompute_partitions": recompute_partitions,
        "partition_hashes": partition_hashes,
        "partition_hash": items[0].scenario.partition_hash if len(partition_hashes) == 1 else None,
        "topology_hashes": topology_hashes, "checks": checks,
    }


def validate_configs(recompute_partitions: bool = True) -> Dict[str, object]:
    """校验seed42三份配置，共同真实分区及各自真实150轮调度。"""

    return validate_scenario_configs(scenarios(), SUITE_NAME, expected_flat_config,
                                     recompute_partitions)


def validate_result(result_dir: Path, item: LearningRateScenario) -> Dict[str, object]:
    """复用逐轮审计，额外绑定当前三拓扑身份及实际本地学习率。"""

    return validate_monitored_result(result_dir, item, suite_name=SUITE_NAME,
                                     configuration=expected_flat_config(item),
                                     check_participation_budget=True)
