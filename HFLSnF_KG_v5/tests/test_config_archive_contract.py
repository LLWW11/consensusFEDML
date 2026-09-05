"""校验当前最终配置与历史配置归档结构。"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Dict, Mapping, Tuple

import yaml

from HFLSnF_KG_v5.tasks.kge.dynamic_mat_varalpha0p5 import (
    SCENARIOS as VARALPHA_SCENARIOS,
)
from HFLSnF_KG_v5.tasks.kge.fedadam_stage1 import (
    SCENARIOS as STAGE1_SCENARIOS,
)
from HFLSnF_KG_v5.tasks.kge.fedadam_stage2 import SCREEN_SCENARIOS
from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import (
    DYNAMIC_SCENARIOS,
    SCENARIOS as FIXED_COUNT_SCENARIOS,
    SMOKE_CONFIG,
)
from HFLSnF_KG_v5.tasks.kge.hflkge_client_count_ablation import (
    SCENARIOS as CLIENT_COUNT_SCENARIOS,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / "configs"
FINAL_DYNAMIC_PATTERN = re.compile(
    r"final_dynamic_fedadam_(hflsnf|hflnosnf|flnosnf)_u0p6_bcfalse_"
    r"seed(42|2024|2025)_150round_cuda\.yaml"
)

EXPECTED_DYNAMIC_FILES = {
    "final_dynamic_fedadam_{}_u0p6_bcfalse_seed{}_150round_cuda.yaml".format(
        arm, seed
    )
    for arm in ("hflsnf", "hflnosnf", "flnosnf")
    for seed in (42, 2024, 2025)
}

FINAL_STOCHASTIC_PATTERN = re.compile(
    r"final_stochastic_fedadam_"
    r"(hflsnf_profile|hflnosnf_profile|flnosnf_profile)_"
    r"seed(42|2024|2025)_150round_cuda\.yaml"
)

EXPECTED_STOCHASTIC_FILES = {
    "final_stochastic_fedadam_{}_seed{}_150round_cuda.yaml".format(
        arm, seed
    )
    for arm in ("hflsnf_profile", "hflnosnf_profile", "flnosnf_profile")
    for seed in (42, 2024, 2025)
}

EXPECTED_GRAPH_SEMANTIC_MECHANISM_ABLATION_FILES = {
    "{}_hflsnf_seed{}_150round_cuda.yaml".format(arm, seed)
    for arm in ("graph_only", "semantic_only")
    for seed in (42, 2024, 2025)
}
EXPECTED_GRAPH_SEMANTIC_TOPOLOGY_EXTENSION_FILES = {
    "graph_semantic_{}_seed{}_150round_cuda.yaml".format(arm, seed)
    for arm in ("hflnosnf", "flnosnf")
    for seed in (42, 2024, 2025)
}

EXPECTED_GRAPH_SEMANTIC_FEDAVG_FILES = {
    "graph_semantic_fedavg_{}_seed{}_150round_cuda.yaml".format(arm, seed)
    for arm in ("hflsnf", "hflnosnf", "flnosnf")
    for seed in (42, 2024, 2025)
}



EXPECTED_ARCHIVE_COUNTS = {
    "fedadam_stage1": 8,
    "fedadam_stage2_screen": 8,
    "dynamic_alpha0p1_legacy": 4,
    "dynamic_alpha0p5_legacy": 4,
    "fixed_count_four_scenarios": 4,
    "client_count_ablation": 6,
    "smoke": 1,
}

PARTITION_HASHES = {
    42: "8bcac64b705ec2db8721de6a36130625a460c11e0da46e2c22bd852ff015fb19",
    2024: "4653f60364e43ad9991ee3393f1a222d665774489b88f0d242ce322107d1b430",
    2025: "8a20d7ddb6556a419fea5a35ff5f8a16eb534704de7d160159b8b3ce95ee2302",
}

ARM_CONTRACTS = {
    "hflsnf": (
        "hfl", True, "fixed", 6,
        "e383d1c5727c9541a8ea5df105c4a1ce93590b4ce6a6956ffe2bfaf93e2f84fe",
    ),
    "hflnosnf": (
        "hfl", False, "fixed", 6,
        "0c7b70472476933389e1b8347a1583dbb0c847ae74a6469704a8cf09d025cabc",
    ),
    "flnosnf": (
        "fl", False, "none", 1,
        "def543cd55e67e72f9016ae8e81493730a663d3da7a1a9af27c014e7ce2a0151",
    ),
}

ALLOWED_FINAL_DIFFERENCES = {
    "random_seed",
    "ablation_arm",
    "comparison_scenario",
    "topology_snf",
    "topology_architecture",
    "topology_edge_mode",
    "edge_num",
    "expected_partition_hash",
    "expected_topology_schedule_hash",
    "run_name",
}


def _load_flat_yaml(path: Path) -> Dict[str, object]:
    """读取分节YAML并合并为便于合同比较的扁平字典。"""

    with path.open("r", encoding="utf-8") as handle:
        sections = yaml.safe_load(handle)
    if not isinstance(sections, dict):
        raise TypeError("YAML顶层必须为对象：{}".format(path))
    flattened: Dict[str, object] = {}
    for section_name, section in sections.items():
        if not isinstance(section, dict):
            raise TypeError(
                "YAML分节{}必须为对象：{}".format(section_name, path)
            )
        flattened.update(section)
    return flattened


def _normalized_final_config(config: Mapping[str, object]) -> Dict[str, object]:
    """删除最终配置允许随实验臂或种子变化的字段。"""

    return {
        key: value
        for key, value in config.items()
        if key not in ALLOWED_FINAL_DIFFERENCES
    }


def _archive_path(relative_path: str) -> Path:
    """把任务模块中的配置相对路径解析到配置目录。"""

    return (CONFIG_DIR / relative_path).resolve()


class ConfigArchiveContractTest(unittest.TestCase):
    """检查最终配置合同、归档计数以及历史入口兼容性。"""

    def test_file_layout_and_all_yaml_are_readable(self) -> None:
        """确认动态和随机配置分目录保存且历史归档计数不变。"""

        root_yaml = {path.name for path in CONFIG_DIR.glob("*.yaml")}
        self.assertEqual(root_yaml, set())
        dynamic_yaml = {
            path.name for path in (CONFIG_DIR / "dynamic").glob("*.yaml")
        }
        stochastic_yaml = {
            path.name for path in (CONFIG_DIR / "stochastic").glob("*.yaml")
        }
        overlap_yaml = {
            path.name for path in (CONFIG_DIR / "overlap").glob("*.yaml")
        }
        graph_semantic_yaml = {
            path.name
            for path in (CONFIG_DIR / "graph_semantic").glob("*.yaml")
        }
        mechanism_ablation_yaml = {
            path.name
            for path in (
                CONFIG_DIR / "graph_semantic_mechanism_ablation"
            ).glob("*.yaml")
        }
        topology_extension_yaml = {
            path.name
            for path in (
                CONFIG_DIR / "graph_semantic_topology_extension"
            ).glob("*.yaml")
        }
        fedavg_yaml = {
            path.name
            for path in (
                CONFIG_DIR / "graph_semantic_fedavg_comparison"
            ).glob("*.yaml")
        }
        self.assertEqual(dynamic_yaml, EXPECTED_DYNAMIC_FILES)
        self.assertEqual(stochastic_yaml, EXPECTED_STOCHASTIC_FILES)
        self.assertEqual(len(overlap_yaml), 9)
        self.assertEqual(
            graph_semantic_yaml,
            {
                "graph_semantic_hflsnf_seed{}_150round_cuda.yaml".format(seed)
                for seed in (42, 2024, 2025)
            },
        )
        self.assertEqual(
            mechanism_ablation_yaml,
            EXPECTED_GRAPH_SEMANTIC_MECHANISM_ABLATION_FILES,
        )
        self.assertEqual(
            topology_extension_yaml,
            EXPECTED_GRAPH_SEMANTIC_TOPOLOGY_EXTENSION_FILES,
        )
        self.assertEqual(
            fedavg_yaml,
            EXPECTED_GRAPH_SEMANTIC_FEDAVG_FILES,
        )

        archive_root = CONFIG_DIR / "zOld"
        archive_yaml = tuple(sorted(archive_root.rglob("*.yaml")))
        self.assertEqual(len(archive_yaml), 35)
        for directory, expected_count in EXPECTED_ARCHIVE_COUNTS.items():
            actual_count = len(tuple((archive_root / directory).glob("*.yaml")))
            self.assertEqual(actual_count, expected_count, directory)

        # 文件名在整个配置树中保持唯一，避免按名称查找时出现歧义。
        all_yaml = tuple(sorted(CONFIG_DIR.rglob("*.yaml")))
        self.assertEqual(len(all_yaml), 86)
        self.assertEqual(len({path.name for path in all_yaml}), 86)
        self.assertTrue((CONFIG_DIR / "README.md").is_file())
        self.assertTrue((archive_root / "README.md").is_file())
        for path in all_yaml:
            _load_flat_yaml(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    self.assertTrue(line.isascii(), str(path))

    def test_final_configs_match_fixed_contract(self) -> None:
        """确认九份主配置仅按实验臂、种子及其派生身份发生变化。"""

        parsed: Dict[Tuple[str, int], Dict[str, object]] = {}
        for file_name in sorted(EXPECTED_DYNAMIC_FILES):
            match = FINAL_DYNAMIC_PATTERN.fullmatch(file_name)
            self.assertIsNotNone(match, file_name)
            assert match is not None
            arm = match.group(1)
            seed = int(match.group(2))
            parsed[(arm, seed)] = _load_flat_yaml(
                CONFIG_DIR / "dynamic" / file_name
            )

        key_sets = {frozenset(config) for config in parsed.values()}
        self.assertEqual(len(key_sets), 1)
        normalized = [_normalized_final_config(item) for item in parsed.values()]
        self.assertTrue(all(item == normalized[0] for item in normalized[1:]))

        shared_expected = {
            "dataset": "fb15k-237",
            "client_num_in_total": 37,
            "client_num_per_round": 37,
            "topology_type": "matlab_direct",
            "dynamic_group_mat_file": (
                "../Topo_opt/postprocess/"
                "result-U-6fixedge_epoch200_hard_varAlpha_0p1_trainable.mat"
            ),
            "topology_util": 0.6,
            "topology_schedule_policy": "strict",
            "aggregation_mode": "row_count_weighted",
            "server_optimizer": "fedadam",
            "server_learning_rate": 0.05,
            "server_beta1": 0.9,
            "server_beta2": 0.99,
            "server_tau": 0.001,
            "server_bias_correction": False,
            "comm_round": 150,
            "epochs": 3,
            "eval_every": 1,
            "test_max_triples": 0,
            "evaluate_test_after_training": False,
            "using_gpu": True,
            "require_cuda": True,
        }
        for (arm, seed), config in parsed.items():
            for field, expected in shared_expected.items():
                self.assertEqual(config.get(field), expected, field)
            self.assertNotIn("centralized_reference_mrr", config)
            self.assertEqual(config["random_seed"], seed)
            self.assertEqual(config["expected_partition_hash"], PARTITION_HASHES[seed])
            architecture, expected_snf, edge_mode, edge_num, schedule_hash = (
                ARM_CONTRACTS[arm]
            )
            self.assertEqual(config["topology_architecture"], architecture)
            self.assertEqual(config["topology_snf"], expected_snf)
            self.assertEqual(config["topology_edge_mode"], edge_mode)
            self.assertEqual(config["edge_num"], edge_num)
            self.assertEqual(config["expected_topology_schedule_hash"], schedule_hash)
            self.assertEqual(
                config["ablation_suite"],
                "v3_final_dynamic_fedadam_u0p6_bcfalse_e3_eval1_formal150",
            )
            self.assertEqual(
                config["ablation_arm"],
                "final_{}_u0p6_bcfalse_seed{}".format(arm, seed),
            )
            self.assertEqual(
                config["comparison_scenario"],
                "V3-FinalDynamicFedAdam-{}-u0p6_bcfalse-Seed{}-Formal150".format(
                    {
                        "hflsnf": "HFLSNF",
                        "hflnosnf": "HFLNOSNF",
                        "flnosnf": "FLNOSNF",
                    }[arm],
                    seed,
                ),
            )
            self.assertEqual(
                config["run_name"],
                "hflsnf_kg_v3_final_dynamic_fedadam_{}_u0p6_bcfalse_"
                "seed{}_150round_cuda".format(arm, seed),
            )

    def test_historical_entry_points_reference_archived_configs(self) -> None:
        """确认全部旧任务入口都已切换到正确的zOld分类。"""

        grouped_paths = {
            "fedadam_stage1": [item.formal_config for item in STAGE1_SCENARIOS],
            "fedadam_stage2_screen": [
                str(item.config_path.relative_to(CONFIG_DIR))
                for item in SCREEN_SCENARIOS
            ],
            "dynamic_alpha0p1_legacy": [
                item.formal_config for item in DYNAMIC_SCENARIOS
            ],
            "dynamic_alpha0p5_legacy": [
                item.formal_config for item in VARALPHA_SCENARIOS
            ],
            "fixed_count_four_scenarios": [
                item.formal_config for item in FIXED_COUNT_SCENARIOS
            ],
            "client_count_ablation": [
                item.formal_config for item in CLIENT_COUNT_SCENARIOS
            ],
            "smoke": [SMOKE_CONFIG],
        }
        for directory, relative_paths in grouped_paths.items():
            self.assertEqual(len(relative_paths), EXPECTED_ARCHIVE_COUNTS[directory])
            for relative_path in relative_paths:
                normalized = Path(relative_path).as_posix()
                self.assertTrue(
                    normalized.startswith("zOld/{}/".format(directory)),
                    normalized,
                )
                self.assertTrue(_archive_path(relative_path).is_file(), normalized)


if __name__ == "__main__":
    unittest.main()
