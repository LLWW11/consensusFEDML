"""验证37客户端CUDA加速基准没有偏离正式实验核心口径。"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import yaml

from HFLSnF_KG_v3.core.topology import (
    MatlabTopologyProvider,
    StaticTopologyProvider,
)
from HFLSnF_KG_v3.run_from_ide import resolve_profile


class HFLSnF37AccelerationBenchmarkTest(unittest.TestCase):
    """检查吞吐与剖析配置的客户端、拓扑和强TransE合同。"""

    @classmethod
    def setUpClass(cls) -> None:
        """只加载一次正式MAT调度，供两个短基准哈希测试复用。"""

        cls.topology_provider = MatlabTopologyProvider(
            mat_path=(
                cls._package_dir()
                / "matlab"
                / "result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat"
            ),
            architecture="hfl",
            snf_enabled=True,
            edge_mode="dynamic",
            util=0.5,
            client_count=37,
        )
        cls.varalpha0p1_cycle_provider = MatlabTopologyProvider(
            mat_path=(
                cls._package_dir()
                / "matlab"
                / "result-U-6fixedge_epoch200_varAlpha_0p1_trainable.mat"
            ),
            architecture="hfl",
            snf_enabled=True,
            edge_mode="dynamic",
            util=0.5,
            client_count=37,
            schedule_policy="cycle",
        )

    @staticmethod
    def _package_dir() -> Path:
        """返回HFLSnF_KG_v3包目录。"""

        return Path(__file__).resolve().parents[1]

    @classmethod
    def _load_config(cls, filename: str):
        """读取一个基准YAML并合并为便于断言的配置字典。"""

        path = cls._package_dir() / "configs" / filename
        with path.open("r", encoding="utf-8") as handle:
            sections = yaml.safe_load(handle)
        merged = {}
        for section in sections.values():
            merged.update(section)
        return merged

    def _assert_strong_federated_contract(
        self,
        config,
        aggregation_mode: str = "row_mask_presence",
    ) -> None:
        """断言短基准保持37客户端正式强配方的核心实验口径。"""

        self.assertEqual(config["dataset"], "fb15k-237")
        self.assertEqual(config["partition_strategy"], "balanced_head_entity")
        self.assertEqual(config["client_num_in_total"], 37)
        self.assertEqual(config["client_num_per_round"], 37)
        self.assertEqual(config["topology_type"], "matlab")
        self.assertEqual(config["topology_architecture"], "hfl")
        self.assertTrue(config["topology_snf"])
        self.assertEqual(config["topology_edge_mode"], "dynamic")
        self.assertEqual(
            config["aggregation_mode"], aggregation_mode
        )
        self.assertEqual(
            config["local_objective"],
            "bidirectional_self_adversarial",
        )
        self.assertEqual(config["embedding_dim"], 256)
        self.assertEqual(config["distance_norm"], 1)
        self.assertEqual(config["epochs"], 2)
        self.assertEqual(config["batch_size"], 1024)
        self.assertEqual(config["negative_sample_count"], 256)
        self.assertEqual(config["evaluation_query_batch_size"], 64)
        self.assertEqual(config["evaluation_candidate_batch_size"], 8192)
        self.assertTrue(config["using_gpu"])
        self.assertTrue(config["require_cuda"])

    def _schedule_hash(
        self,
        round_count: int,
        topology_provider=None,
    ) -> str:
        """按训练器规则计算指定提供器前若干通信轮的调度哈希。"""

        provider = topology_provider or self.topology_provider
        digest = hashlib.sha256()
        for round_index in range(int(round_count)):
            topology = provider.get_round(round_index)
            record = {
                "source_round_index": int(
                    topology.source_round_index
                ),
                "groups": topology.copy_groups(),
                "edge_node_ids": {
                    str(group_id): int(edge_id)
                    for group_id, edge_id in (
                        topology.edge_node_ids.items()
                    )
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

    def test_five_round_throughput_config(self) -> None:
        """确认五轮配置关闭同步剖析并使用前五轮MAT哈希。"""

        config = self._load_config(
            "benchmark_hflsnf37_accelerated_cuda.yaml"
        )
        self._assert_strong_federated_contract(config)
        self.assertEqual(config["comm_round"], 5)
        self.assertEqual(config["eval_every"], 5)
        self.assertFalse(config["profile_training_timing"])
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            "4df560f2d0ee2351cf93722378db18bf"
            "6f236546e91312cb2b44f3359670e2d4",
        )
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            self._schedule_hash(5),
        )

    def test_two_round_profile_config(self) -> None:
        """确认两轮配置开启同步剖析并使用前两轮MAT哈希。"""

        config = self._load_config(
            "benchmark_hflsnf37_profile_cuda.yaml"
        )
        self._assert_strong_federated_contract(config)
        self.assertEqual(config["comm_round"], 2)
        self.assertEqual(config["eval_every"], 2)
        self.assertTrue(config["profile_training_timing"])
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            "c8f9f69d13c0d2337c66572cbfe1187"
            "b232a59383f534b627d8849f25a7e309a",
        )
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            self._schedule_hash(2),
        )

    def test_ide_profiles_route_to_hflsnf_entry(self) -> None:
        """确认IDE短基准方案不会误调用集中式入口。"""

        for profile, expected_filename in (
            (
                "hflsnf37_benchmark_cuda",
                "configs/benchmark_hflsnf37_accelerated_cuda.yaml",
            ),
            (
                "hflsnf37_profile_cuda",
                "configs/benchmark_hflsnf37_profile_cuda.yaml",
            ),
            (
                "hflsnf37_row_count_benchmark_cuda",
                "configs/benchmark_hflsnf37_row_count_cuda.yaml",
            ),
            (
                "hflsnf37_row_count_screen40_cuda",
                "configs/screen_hflsnf37_row_count_seed42_40round_cuda.yaml",
            ),
            (
                "hflsnf37_varalpha0p1_formal300_cuda",
                "configs/hflsnf37_row_count_varalpha0p1_seed42_300round_cuda.yaml",
            ),
            (
                "fixed37_fixed6_reset_adam_screen80_cuda",
                "configs/screen_fixed37_fixed6_reset_adam_seed42_80round_cuda.yaml",
            ),
            (
                "fixed37_fixed6_persistent_adam_screen80_cuda",
                "configs/screen_fixed37_fixed6_persistent_adam_seed42_80round_cuda.yaml",
            ),
        ):
            config_path, entry_type = resolve_profile(profile)
            self.assertEqual(config_path, expected_filename)
            self.assertEqual(entry_type, "hflsnf")

    def test_stage3_row_count_config(self) -> None:
        """确认阶段3短配置只把聚合口径切换为逐行计数加权。"""

        config = self._load_config(
            "benchmark_hflsnf37_row_count_cuda.yaml"
        )
        self._assert_strong_federated_contract(
            config,
            aggregation_mode="row_count_weighted",
        )
        self.assertEqual(config["comm_round"], 5)
        self.assertEqual(config["eval_every"], 5)
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            self._schedule_hash(5),
        )
        formal = self._load_config(
            "hflsnf37_strong_transe_cuda.yaml"
        )
        self.assertEqual(
            formal["aggregation_mode"], "row_count_weighted"
        )

    def test_stage3_seed42_screen40_config(self) -> None:
        """确认40轮筛选使用动态MAT前40轮和固定验证口径。"""

        config = self._load_config(
            "screen_hflsnf37_row_count_seed42_40round_cuda.yaml"
        )
        self._assert_strong_federated_contract(
            config,
            aggregation_mode="row_count_weighted",
        )
        self.assertEqual(config["random_seed"], 42)
        self.assertEqual(config["comm_round"], 40)
        self.assertEqual(config["eval_every"], 10)
        self.assertEqual(config["validation_max_triples"], 4096)
        self.assertEqual(
            config["validation_selection"], "relation_stratified"
        )
        self.assertEqual(config["final_validation_max_triples"], 4096)
        self.assertEqual(config["test_max_triples"], 512)
        self.assertFalse(config["profile_training_timing"])
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            "c1dee647ab91c44189b9f7dd5864b2f8"
            "6bc84113467be3a02dc3f53bda3753ca",
        )
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            self._schedule_hash(40),
        )

    def test_varalpha0p1_formal300_cycle_config(self) -> None:
        """确认300轮正式配置循环复用200轮0.1 MAT并执行完整终评。"""

        config = self._load_config(
            "hflsnf37_row_count_varalpha0p1_seed42_300round_cuda.yaml"
        )
        self._assert_strong_federated_contract(
            config,
            aggregation_mode="row_count_weighted",
        )
        self.assertEqual(config["random_seed"], 42)
        self.assertEqual(config["comm_round"], 300)
        self.assertEqual(config["topology_schedule_policy"], "cycle")
        self.assertTrue(
            config["dynamic_group_mat_file"].endswith(
                "varAlpha_0p1_trainable.mat"
            )
        )
        self.assertEqual(config["eval_every"], 10)
        self.assertEqual(config["validation_max_triples"], 4096)
        self.assertEqual(
            config["validation_selection"], "relation_stratified"
        )
        self.assertEqual(config["final_validation_max_triples"], 0)
        self.assertEqual(config["test_max_triples"], 0)
        self.assertFalse(config["profile_training_timing"])
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            "8bc1f635b4126dde5454b6bc776381fb"
            "779c8c459d36a4171cf2ac3a523ccda3",
        )
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            self._schedule_hash(
                300, self.varalpha0p1_cycle_provider
            ),
        )

        topologies = [
            self.varalpha0p1_cycle_provider.get_round(round_index)
            for round_index in range(300)
        ]
        participant_counts = [
            topology.participant_count for topology in topologies
        ]
        group_counts = [
            len(topology.group_to_client_indexes)
            for topology in topologies
        ]
        self.assertEqual(topologies[199].source_round_index, 199)
        self.assertEqual(topologies[200].source_round_index, 0)
        self.assertEqual(topologies[299].source_round_index, 99)
        self.assertEqual(min(participant_counts), 29)
        self.assertEqual(max(participant_counts), 37)
        self.assertAlmostEqual(
            sum(participant_counts) / 300.0,
            36.373333333333335,
        )
        self.assertEqual(min(group_counts), 2)
        self.assertEqual(max(group_counts), 12)
        with self.assertRaises(IndexError):
            self.topology_provider.get_round(200)

    def test_fixed37_fixed6_reset_adam_screen80_config(self) -> None:
        """确认步骤一80轮基线只固定全参与客户端和六个边缘组。"""

        config = self._load_config(
            "screen_fixed37_fixed6_reset_adam_seed42_80round_cuda.yaml"
        )
        self.assertEqual(config["random_seed"], 42)
        self.assertEqual(config["dataset"], "fb15k-237")
        self.assertEqual(
            config["partition_strategy"], "balanced_head_entity"
        )
        self.assertEqual(config["client_num_in_total"], 37)
        self.assertEqual(config["client_num_per_round"], 37)
        self.assertEqual(config["topology_type"], "static")
        self.assertEqual(config["topology_architecture"], "hfl")
        self.assertFalse(config["topology_snf"])
        self.assertEqual(config["topology_edge_mode"], "fixed")
        self.assertEqual(config["edge_num"], 6)
        self.assertNotIn("dynamic_group_mat_file", config)
        self.assertEqual(
            config["aggregation_mode"], "row_count_weighted"
        )
        self.assertEqual(
            config["local_objective"],
            "bidirectional_self_adversarial",
        )
        self.assertEqual(config["embedding_dim"], 256)
        self.assertEqual(config["distance_norm"], 1)
        self.assertEqual(config["comm_round"], 80)
        self.assertEqual(config["epochs"], 2)
        self.assertEqual(config["batch_size"], 1024)
        self.assertEqual(config["negative_sample_count"], 256)
        self.assertEqual(
            config["client_optimizer_state_mode"], "reset"
        )
        self.assertEqual(config["learning_rate"], 0.00005)
        self.assertEqual(config["eval_every"], 10)
        self.assertEqual(config["validation_max_triples"], 4096)
        self.assertEqual(
            config["validation_selection"], "relation_stratified"
        )
        self.assertEqual(config["final_validation_max_triples"], 4096)
        self.assertEqual(config["test_max_triples"], 512)
        self.assertFalse(config["profile_training_timing"])
        self.assertTrue(config["using_gpu"])
        self.assertTrue(config["require_cuda"])

        provider = StaticTopologyProvider.round_robin(
            client_ids=tuple(range(37)),
            group_count=6,
        )
        first_topology = provider.get_round(0)
        self.assertEqual(first_topology.participant_count, 37)
        self.assertEqual(
            [
                len(client_ids)
                for client_ids in (
                    first_topology.group_to_client_indexes.values()
                )
            ],
            [7, 6, 6, 6, 6, 6],
        )
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            "69b9d605d4b8fa7da0d1f1900a081966"
            "f35ec1ffa84ca3d96baf1047ad25fc9c",
        )
        self.assertEqual(
            config["expected_topology_schedule_hash"],
            self._schedule_hash(80, provider),
        )

    def test_fixed37_persistent_adam_is_single_variable_ablation(
        self,
    ) -> None:
        """确认步骤二配置相对步骤一只改变Adam状态和实验标识。"""

        baseline = self._load_config(
            "screen_fixed37_fixed6_reset_adam_seed42_80round_cuda.yaml"
        )
        persistent = self._load_config(
            "screen_fixed37_fixed6_persistent_adam_seed42_80round_cuda.yaml"
        )
        self.assertEqual(
            baseline["client_optimizer_state_mode"], "reset"
        )
        self.assertEqual(
            persistent["client_optimizer_state_mode"],
            "persistent_per_client",
        )
        for field in (
            "random_seed",
            "dataset",
            "partition_strategy",
            "embedding_dim",
            "distance_norm",
            "client_num_in_total",
            "client_num_per_round",
            "topology_type",
            "topology_architecture",
            "topology_snf",
            "topology_edge_mode",
            "edge_num",
            "aggregation_mode",
            "local_objective",
            "fede_gamma",
            "adversarial_temperature",
            "expected_partition_hash",
            "expected_topology_schedule_hash",
            "comm_round",
            "epochs",
            "batch_size",
            "client_optimizer",
            "learning_rate",
            "negative_sample_count",
            "eval_every",
            "validation_max_triples",
            "validation_selection",
            "final_validation_max_triples",
            "test_max_triples",
            "evaluation_query_batch_size",
            "evaluation_candidate_batch_size",
            "profile_training_timing",
        ):
            self.assertEqual(
                persistent[field],
                baseline[field],
                msg=field,
            )
