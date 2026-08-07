import os
import unittest

from topology_schedule import (
    MatlabTopologySchedule,
    build_balanced_candidate_groups,
)


class MatlabTopologyScheduleTest(unittest.TestCase):
    """使用正式 200 轮 MAT 验证 Python 拓扑调度。"""

    @classmethod
    def setUpClass(cls):
        """定位测试所需的正式 MATLAB 结果文件。"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.mat_path = os.path.join(
            script_dir, "matlab", "result-U-6fixedge_epoch200.mat"
        )
        if not os.path.isfile(cls.mat_path):
            raise unittest.SkipTest("缺少旧版正式 MAT，跳过精确映射兼容测试。")

    def _create_schedule(self, architecture, snf_enabled, edge_mode="fixed"):
        """创建 util=0.5、200 个真实客户端和 37 个候选槽位的调度器。"""
        return MatlabTopologySchedule(
            self.mat_path,
            architecture=architecture,
            snf_enabled=snf_enabled,
            edge_mode=edge_mode,
            util=0.5,
            client_num_in_total=200,
            candidate_client_count=37,
        )

    def _assert_all_rounds_valid(self, schedule):
        """校验全部轮次的客户端编号范围、唯一性和人数。"""
        self.assertEqual(schedule.round_count, 200)
        for round_index in range(schedule.round_count):
            topology = schedule.get_round(round_index)
            active_clients = list(topology.active_client_indexes)
            self.assertEqual(len(active_clients), topology.participant_count)
            self.assertEqual(len(active_clients), len(set(active_clients)))
            self.assertTrue(all(0 <= client_index < 37 for client_index in active_clients))
            self.assertLessEqual(topology.participant_count, 37)

            grouped_clients = []
            for group_index, client_indexes in topology.group_to_client_indexes.items():
                grouped_clients.extend(client_indexes)
                self.assertEqual(
                    topology.group_client_counts[group_index], len(client_indexes)
                )
            self.assertEqual(sorted(grouped_clients), active_clients)
            self.assertEqual(
                sum(topology.group_client_counts.values()), topology.participant_count
            )

    def test_matlab_id_conversion_skips_cloud_node(self):
        """验证物理节点 18 被跳过且两侧连续映射为候选槽位。"""
        schedule = self._create_schedule("hfl", True)
        self.assertEqual(schedule.matlab_id_to_candidate_slot(1), 0)
        self.assertEqual(schedule.matlab_id_to_python(1), 0)
        self.assertEqual(schedule.matlab_id_to_python(17), 16)
        self.assertEqual(schedule.matlab_id_to_python(19), 17)
        self.assertEqual(schedule.matlab_id_to_python(38), 36)
        with self.assertRaises(ValueError):
            schedule.matlab_id_to_python(18)

    def test_real_client_pool_is_decoupled_from_mat_slots(self):
        """验证 200 个真实客户端与 MAT 的 37 个候选槽位相互独立。"""
        schedule = self._create_schedule("hfl", True)
        metadata = schedule.to_metadata()
        self.assertEqual(metadata["client_num_in_total"], 200)
        self.assertEqual(metadata["candidate_client_count"], 37)
        self.assertEqual(metadata["mat_physical_client_count"], 37)
        self.assertEqual(metadata["physical_to_candidate_slot"]["38"], 36)

    def test_candidate_count_must_match_mat_slots(self):
        """验证候选人数与 MAT 物理槽位数不一致时给出明确错误。"""
        for candidate_count in (36, 38):
            with self.subTest(candidate_count=candidate_count):
                with self.assertRaises(ValueError):
                    MatlabTopologySchedule(
                        self.mat_path,
                        architecture="hfl",
                        snf_enabled=True,
                        edge_mode="fixed",
                        util=0.5,
                        client_num_in_total=200,
                        candidate_client_count=candidate_count,
                    )

    def test_real_pool_cannot_be_smaller_than_candidates(self):
        """验证真实客户端池不能小于首次采样候选人数。"""
        with self.assertRaises(ValueError):
            MatlabTopologySchedule(
                self.mat_path,
                architecture="hfl",
                snf_enabled=True,
                edge_mode="fixed",
                util=0.5,
                client_num_in_total=36,
                candidate_client_count=37,
            )

    def test_legacy_thirty_seven_client_call_remains_supported(self):
        """验证旧代码只传 37 时仍可自动推导候选槽位数量。"""
        schedule = MatlabTopologySchedule(
            self.mat_path,
            architecture="hfl",
            snf_enabled=True,
            edge_mode="fixed",
            util=0.5,
            client_num_in_total=37,
        )
        self.assertEqual(schedule.candidate_client_count, 37)

    def test_four_fixed_scenarios(self):
        """验证四个 fixed 对照场景的轮数、容量和平均参与人数。"""
        cases = [
            ("hfl", True, "hfl_snf_fixed", 6, 35.075),
            ("hfl", False, "hfl_no_snf_fixed", 6, 18.85),
            ("fl", True, "fl_snf", 1, 28.415),
            ("fl", False, "fl_no_snf", 1, 6.495),
        ]
        for architecture, snf_enabled, name, capacity, participant_mean in cases:
            with self.subTest(name=name):
                schedule = self._create_schedule(architecture, snf_enabled)
                metadata = schedule.to_metadata()
                self.assertEqual(schedule.scenario_name, name)
                self.assertEqual(schedule.group_capacity, capacity)
                self.assertAlmostEqual(
                    metadata["participant_count_mean"], participant_mean, places=9
                )
                self._assert_all_rounds_valid(schedule)

    def test_dynamic_hfl_capacity_is_discovered_from_mat(self):
        """验证 dynamic 模式可从 MAT 自动扩展边缘槽位而不受固定值 6 限制。"""
        snf_schedule = self._create_schedule("hfl", True, edge_mode="dynamic")
        no_snf_schedule = self._create_schedule("hfl", False, edge_mode="dynamic")
        self.assertEqual(snf_schedule.group_capacity, 12)
        self.assertEqual(no_snf_schedule.group_capacity, 10)
        self._assert_all_rounds_valid(snf_schedule)
        self._assert_all_rounds_valid(no_snf_schedule)

    def test_unknown_util_is_rejected(self):
        """验证不在 total_util 中的利用率会被明确拒绝。"""
        with self.assertRaises(ValueError):
            MatlabTopologySchedule(
                self.mat_path,
                architecture="hfl",
                snf_enabled=True,
                edge_mode="fixed",
                util=0.55,
                client_num_in_total=200,
                candidate_client_count=37,
            )


class BalancedCandidateGroupTest(unittest.TestCase):
    """验证 FEMNIST 式 k/n 连续候选段分组及零参与兼容。"""

    def test_balanced_groups_allocate_quotient_and_remainder(self):
        """验证参与人数平均分配且余数依次补给前面的组。"""
        groups = build_balanced_candidate_groups(37, 4, 21)
        self.assertEqual(
            {group_idx: len(slots) for group_idx, slots in groups.items()},
            {0: 6, 1: 5, 2: 5, 3: 5},
        )
        self.assertEqual(groups[0], (0, 1, 2, 3, 4, 5))
        self.assertEqual(groups[1], (9, 10, 11, 12, 13))

    def test_zero_participant_round_returns_no_groups(self):
        """验证 HFL 和 FL 的零参与轮都返回空分组。"""
        self.assertEqual(build_balanced_candidate_groups(37, 0, 0), {})
        self.assertEqual(build_balanced_candidate_groups(37, 6, 0), {})
        self.assertEqual(build_balanced_candidate_groups(37, 1, 0), {})

    def test_nonzero_participants_require_a_valid_group_count(self):
        """验证非零参与轮不能使用零个边缘组。"""
        with self.assertRaises(ValueError):
            build_balanced_candidate_groups(37, 0, 1)


if __name__ == "__main__":
    unittest.main()
