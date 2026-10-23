"""固定本地学习率5e-4的三种子乘三拓扑九实验合同。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

from .fixed_count_four_scenarios import PACKAGE_DIR, load_flat_config
from .graph_semantic_fedavg_comparison import ARM_ORDER, _arm_label, scenario_by_id
from .graph_semantic_fedavg_lr_sweep import (
    LearningRateScenario, validate_result as validate_monitored_result,
)
from .graph_semantic_fedavg_seed42 import LEARNING_RATE, validate_scenario_configs


SUITE_NAME = "v5_graph_semantic_fedavg_three_seed_lr5e-4_three_topology_formal150"
RESULT_ROOT = "results/graph_semantic_fedavg_multiseed_lr5e-4"
CONFIG_DIR = PACKAGE_DIR / "configs" / "graph_semantic_fedavg_multiseed_lr5e-4"
SEEDS = (42, 2024, 2025)


def scenarios() -> Tuple[LearningRateScenario, ...]:
    """按42、2024、2025种子优先及三拓扑次序生成九个独立新场景。"""

    items = []
    for seed in SEEDS:
        for arm in ARM_ORDER:
            source = scenario_by_id("graph_semantic_fedavg_{}_seed{}".format(arm, seed))
            identity = source.scenario_id + "_lr5e-4"
            items.append(LearningRateScenario(
                "{}_seed{}".format(arm, seed), LEARNING_RATE,
                replace(source, scenario_id=identity, source_config_path=source.config_path,
                        config_path=(CONFIG_DIR / (identity + "_150round_cuda.yaml")).resolve()),
            ))
    return tuple(items)


def expected_flat_config(item: LearningRateScenario) -> Dict[str, object]:
    """从同种子同拓扑旧FedAvg配置派生，不混用不同种子的分区合同。"""

    expected = dict(load_flat_config(item.scenario.source_config_path))
    expected.update(
        learning_rate=LEARNING_RATE, lr=LEARNING_RATE, ablation_suite=SUITE_NAME,
        ablation_arm=item.scenario.scenario_id,
        comparison_scenario="V5-{}-GraphSemantic-FedAvg-Seed{}-lr5e-4-Formal150".format(
            _arm_label(item.scenario.arm), item.scenario.seed),
        run_name="hflsnf_kg_v5_" + item.scenario.scenario_id + "_150round_cuda",
        result_root=RESULT_ROOT,
    )
    return expected


def validate_configs(recompute_partitions: bool = True) -> Dict[str, object]:
    """复算三种子真实分区，校验九份配方及三拓扑各自调度合同。"""

    return validate_scenario_configs(scenarios(), SUITE_NAME, expected_flat_config,
                                     recompute_partitions)


def validate_result(result_dir: Path, item: LearningRateScenario) -> Dict[str, object]:
    """校验150轮、实际学习率、当前种子初始化及当前拓扑参与预算。"""

    return validate_monitored_result(result_dir, item, suite_name=SUITE_NAME,
                                     configuration=expected_flat_config(item),
                                     check_participation_budget=True)
