"""HFLKGE固定架构下的客户端人数单因素消融工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .fixed_count_four_scenarios import (
    INITIAL_MODEL_HASH,
    PACKAGE_DIR,
    PARTITION_HASH,
    _append_check,
    _load_json,
    _load_metrics,
    _schedule_hash,
    _validate_round_metrics,
    build_scenario_provider,
    load_flat_config,
)


SUITE_NAME = "v3_hflkge_random_client_count_e3_eval1_seed42"
IDENTITY_FIELDS = {
    "ablation_arm",
    "client_num_per_round",
    "comparison_scenario",
    "expected_topology_schedule_hash",
    "run_name",
}


@dataclass(frozen=True)
class HFLKGECountScenario:
    """描述一个只改变每轮客户端人数的HFLKGE实验臂。"""

    arm: str
    participant_count: int
    formal_config: str
    formal_hash: str
    architecture: str = "hfl"
    snf_enabled: bool = False
    group_count: int = 6
    selection_mode: str = "seeded_random"


SCENARIOS: Tuple[HFLKGECountScenario, ...] = (
    HFLKGECountScenario(
        arm="hflkge_k36",
        participant_count=36,
        formal_config=(
            "formal_hflkge_client_count_k36_seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "563d57f34917edfd9f9bfa214d3a9e2c"
            "5ec3e0a06afd06933507ec1315b7c226"
        ),
    ),
    HFLKGECountScenario(
        arm="hflkge_k30",
        participant_count=30,
        formal_config=(
            "formal_hflkge_client_count_k30_seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "1b387cd31062488bb4470244dbb8c6fa9"
            "c360a484a814b4c4a0308dbad5919bc"
        ),
    ),
    HFLKGECountScenario(
        arm="hflkge_k24",
        participant_count=24,
        formal_config=(
            "formal_hflkge_client_count_k24_seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "86e45ec99a70422d3acc22e79b0afcda"
            "e762d1f8bf478c4a54430f6356bc9ce7"
        ),
    ),
    HFLKGECountScenario(
        arm="hflkge_k18",
        participant_count=18,
        formal_config=(
            "formal_hflkge_client_count_k18_seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "aaff3219861d4c2c68cd661876eece27d"
            "ceab08a711a003584c7e64f29231645"
        ),
    ),
    HFLKGECountScenario(
        arm="hflkge_k12",
        participant_count=12,
        formal_config=(
            "formal_hflkge_client_count_k12_seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "713ec8c55241b051acd35a72dccd6f69"
            "6753f53d2c029cfecdc6f8f35001b565"
        ),
    ),
    HFLKGECountScenario(
        arm="hflkge_k6",
        participant_count=6,
        formal_config=(
            "formal_hflkge_client_count_k6_seed42_150round_cuda.yaml"
        ),
        formal_hash=(
            "c5cddba3bdb8b5b0a9362bbc4f54a8eb"
            "498f8fd1b88f0260012005be725e081d"
        ),
    ),
)


def scenario_by_arm(arm: str) -> HFLKGECountScenario:
    """按名称返回一个HFLKGE人数实验臂。"""

    for scenario in SCENARIOS:
        if scenario.arm == str(arm):
            return scenario
    raise KeyError("未知HFLKGE人数实验臂：{}".format(arm))


def build_provider(scenario: HFLKGECountScenario):
    """构造固定HFL架构、每轮独立随机选人的调度提供器。"""

    return build_scenario_provider(scenario)


def _shared_contract() -> Dict[str, object]:
    """返回六份正式配置必须完全共享的训练合同。"""

    return {
        "random_seed": 42,
        "dataset": "fb15k-237",
        "partition_strategy": "balanced_head_entity",
        "embedding_dim": 256,
        "distance_norm": 1,
        "federated_optimizer": "DynamicTopologyTransE",
        "ablation_suite": SUITE_NAME,
        "client_num_in_total": 37,
        "topology_type": "fixed_count",
        "fixed_count_selection_mode": "seeded_random",
        "fixed_count_seed": 42,
        "topology_architecture": "hfl",
        "topology_snf": False,
        "topology_edge_mode": "fixed",
        "edge_num": 6,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_learning_rate": 0.1,
        "server_beta1": 0.9,
        "server_beta2": 0.99,
        "server_tau": 0.001,
        "server_bias_correction": True,
        "local_objective": "bidirectional_self_adversarial",
        "expected_partition_hash": PARTITION_HASH,
        "comm_round": 150,
        "epochs": 3,
        "batch_size": 1024,
        "client_optimizer_state_mode": "reset",
        "learning_rate": 0.00005,
        "negative_sample_count": 256,
        "eval_every": 1,
        "validation_max_triples": 4096,
        "test_max_triples": 0,
        "using_gpu": True,
        "require_cuda": True,
        "gpu_id": 0,
    }


def _behavior_config(config: Dict[str, object]) -> Dict[str, object]:
    """移除人数及派生身份字段，得到可直接比较的行为配置。"""

    return {
        key: value
        for key, value in config.items()
        if key not in IDENTITY_FIELDS
    }


def validate_configs() -> Dict[str, object]:
    """校验六份YAML仅在人数和必要派生标识上存在差异。"""

    reports: List[Dict[str, object]] = []
    behavior_configs: List[Dict[str, object]] = []
    shared_contract = _shared_contract()
    for scenario in SCENARIOS:
        config_path = PACKAGE_DIR / "configs" / scenario.formal_config
        config = load_flat_config(config_path)
        provider = build_provider(scenario)
        actual_hash = _schedule_hash(provider, 150)
        checks: List[Dict[str, object]] = []
        for field, expected in shared_contract.items():
            _append_check(checks, field, config.get(field), expected)
        for field, expected in (
            ("ablation_arm", scenario.arm),
            ("client_num_per_round", scenario.participant_count),
            ("expected_topology_schedule_hash", scenario.formal_hash),
            ("实际调度哈希", scenario.formal_hash),
        ):
            actual = actual_hash if field == "实际调度哈希" else config.get(field)
            _append_check(checks, field, actual, expected)
        behavior_configs.append(_behavior_config(config))
        reports.append(
            {
                "arm": scenario.arm,
                "config": str(config_path),
                "status": (
                    "passed"
                    if all(item["passed"] for item in checks)
                    else "failed"
                ),
                "checks": checks,
            }
        )
    # 身份字段以外必须逐项相同，防止人数消融混入其他训练变量。
    behavior_equal = all(
        config == behavior_configs[0]
        for config in behavior_configs[1:]
    )
    return {
        "status": (
            "passed"
            if behavior_equal
            and all(report["status"] == "passed" for report in reports)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "behavior_configs_equal": behavior_equal,
        "allowed_different_fields": sorted(IDENTITY_FIELDS),
        "formal_configs": reports,
    }


def validate_result(
    result_dir: Path,
    scenario: HFLKGECountScenario,
    expected_rounds: int = 150,
) -> Dict[str, object]:
    """校验人数消融正式结果的轮次、人数、分组和优化器合同。"""

    resolved_dir = Path(result_dir).expanduser().resolve()
    summary = _load_json(resolved_dir / "summary.json")
    rows = _load_metrics(resolved_dir / "metrics.csv")
    checks: List[Dict[str, object]] = []
    expected_summary = {
        "device": "cuda:0",
        "ablation_suite": SUITE_NAME,
        "ablation_arm": scenario.arm,
        "architecture": "hfl",
        "snf_enabled": False,
        "client_count": 37,
        "client_num_per_round": scenario.participant_count,
        "participant_count_min": scenario.participant_count,
        "participant_count_max": scenario.participant_count,
        "group_count_min": 6,
        "group_count_max": 6,
        "comm_round": int(expected_rounds),
        "local_epochs": 3,
        "aggregation_mode": "row_count_weighted",
        "server_optimizer": "fedadam",
        "server_bias_correction": True,
        "server_optimizer_step_count": int(expected_rounds),
        "partition_hash": PARTITION_HASH,
        "topology_schedule_hash": scenario.formal_hash,
        "initial_model_hash": INITIAL_MODEL_HASH,
    }
    for field, expected in expected_summary.items():
        _append_check(checks, field, summary.get(field), expected)
    _validate_round_metrics(
        checks,
        rows,
        expected_rounds,
        scenario.participant_count,
        6,
    )
    return {
        "status": (
            "passed"
            if all(item["passed"] for item in checks)
            else "failed"
        ),
        "suite": SUITE_NAME,
        "arm": scenario.arm,
        "result_dir": str(resolved_dir),
        "checks": checks,
    }


def selected_scenarios(arm: str) -> Sequence[HFLKGECountScenario]:
    """返回全部实验臂或用户指定的单个实验臂。"""

    if str(arm) == "all":
        return SCENARIOS
    return (scenario_by_arm(str(arm)),)
