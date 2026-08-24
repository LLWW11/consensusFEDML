"""验证varAlpha=0.5动态MAT的四场景配置与调度。"""

from __future__ import annotations

import unittest

from HFLSnF_KG_v5.tasks.kge.dynamic_mat_varalpha0p5 import (
    MAT_FILE,
    MAT_RELATIVE_PATH,
    SCENARIOS,
    build_provider,
    schedule_statistics,
    validate_configs,
)
from HFLSnF_KG_v5.tasks.kge.fixed_count_four_scenarios import (
    PACKAGE_DIR,
    load_flat_config,
)


class DynamicMatVarAlpha05Test(unittest.TestCase):
    """检查0p5 MAT路径、非空轮次、调度哈希和YAML合同。"""

    def test_mat_file_exists(self) -> None:
        """确认同步服务器前0p5 MAT文件位于约定包内路径。"""

        self.assertTrue(MAT_FILE.is_file())
        self.assertEqual(
            MAT_FILE.resolve(),
            (PACKAGE_DIR / MAT_RELATIVE_PATH).resolve(),
        )

    def test_four_configs_pass_contract(self) -> None:
        """确认四份0p5正式YAML及其150轮调度合同全部通过。"""

        report = validate_configs()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["formal_configs"]), 4)

    def test_first_150_rounds_are_nonempty_and_match_hashes(self) -> None:
        """确认四个场景前150轮无空轮且统计与固化哈希一致。"""

        for scenario in SCENARIOS:
            provider = build_provider(scenario)
            topologies = [
                provider.get_round(round_index)
                for round_index in range(150)
            ]
            self.assertTrue(
                all(item.participant_count > 0 for item in topologies)
            )
            statistics = schedule_statistics(scenario, 150)
            self.assertEqual(
                statistics["schedule_hash"],
                scenario.formal_hash,
            )
            self.assertEqual(
                statistics["participant_count_min"],
                scenario.participant_min,
            )
            self.assertEqual(
                statistics["participant_count_max"],
                scenario.participant_max,
            )

    def test_configs_use_only_the_0p5_mat_path(self) -> None:
        """确认新配置独立引用0p5文件且不覆盖原0p1配置。"""

        for scenario in SCENARIOS:
            config = load_flat_config(
                PACKAGE_DIR / "configs" / scenario.formal_config
            )
            self.assertEqual(
                config["dynamic_group_mat_file"],
                MAT_RELATIVE_PATH,
            )
            self.assertEqual(
                config["expected_topology_schedule_hash"],
                scenario.formal_hash,
            )


if __name__ == "__main__":
    unittest.main()
