"""验证V3工程边界、37客户端划分和V2回归采样语义。"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from HFLSnF_KG_v2.fedml_kge.model_trainer import (
    FedMLTransEModelTrainer as V2ModelTrainer,
)
from HFLSnF_KG_v2.tasks.kge import (
    TransE as V2TransE,
    build_synthetic_knowledge_graph as build_v2_synthetic_graph,
    partition_train_triples_by_head as partition_v2_triples,
)
from HFLSnF_KG_v2.tasks.kge.negative_sampling import (
    FilteredNegativeSampler as V2NegativeSampler,
)
from HFLSnF_KG_v3.fedml_kge.model_trainer import (
    FedMLTransEModelTrainer as V3ModelTrainer,
)
from HFLSnF_KG_v3.tasks.kge import (
    FilteredNegativeSampler,
    TransE as V3TransE,
    build_synthetic_knowledge_graph as build_v3_synthetic_graph,
    load_fb15k237,
    partition_train_triples_by_head,
)


class Stage0CleanProjectTest(unittest.TestCase):
    """检查阶段0复制边界和关键回归指纹。"""

    @staticmethod
    def _package_dir() -> Path:
        """返回HFLSnF_KG_v3包目录。"""

        return Path(__file__).resolve().parents[1]

    def test_project_does_not_copy_v2_results(self) -> None:
        """确认V3没有复制V2已有实验结果和旧四臂入口。"""

        package_dir = self._package_dir()
        result_root = package_dir / "results"
        if result_root.exists():
            copied_v2_names = [
                path.name
                for path in result_root.iterdir()
                if "v2" in path.name.lower()
            ]
            self.assertEqual(copied_v2_names, [])
        self.assertFalse(
            (package_dir / "run_four_arm_ablation.py").exists()
        )
        self.assertFalse(
            (package_dir / "run_three_arm_ablation.py").exists()
        )

    def test_v2_legacy_sampler_sequence_is_preserved(self) -> None:
        """确认V3旧名称负采样器与V2固定种子输出完全一致。"""

        positives = torch.tensor(
            [[0, 0, 1], [1, 1, 2]],
            dtype=torch.long,
        )
        true_triples = {(0, 0, 1), (1, 1, 2)}
        v2_sampler = V2NegativeSampler(5, true_triples, seed=42)
        v3_sampler = FilteredNegativeSampler(
            5,
            true_triples,
            seed=42,
        )
        expected = v2_sampler.sample(
            positives,
            negative_sample_count=4,
            corruption_mode="tail",
        )
        actual = v3_sampler.sample(
            positives,
            negative_sample_count=4,
            corruption_mode="tail",
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_v2_c_arm_local_update_is_exactly_preserved(self) -> None:
        """确认C臂小规模本地更新与V2固定种子参数完全一致。"""

        v2_dataset = build_v2_synthetic_graph()
        v3_dataset = build_v3_synthetic_graph()
        v2_partition = partition_v2_triples(
            v2_dataset,
            client_count=3,
            seed=42,
        ).partitions[0]
        v3_partition = partition_train_triples_by_head(
            v3_dataset,
            client_count=3,
            seed=42,
        ).partitions[0]

        torch.manual_seed(2026)
        v2_model = V2TransE(
            v2_dataset.num_entities,
            v2_dataset.num_relations,
            embedding_dim=8,
            distance_norm=1,
        )
        v3_model = V3TransE(
            v3_dataset.num_entities,
            v3_dataset.num_relations,
            embedding_dim=8,
            distance_norm=1,
        )
        v3_model.load_state_dict(v2_model.state_dict())
        args = SimpleNamespace(
            client_optimizer="adam",
            learning_rate=0.01,
            lr=0.01,
            epochs=1,
            batch_size=2,
            margin=1.0,
            negative_sample_count=3,
            local_objective="fede_self_adversarial",
            random_seed=42,
            round_idx=0,
            fede_gamma=6.0,
            adversarial_temperature=1.0,
        )
        v2_trainer = V2ModelTrainer(v2_model, args, v2_dataset)
        v3_trainer = V3ModelTrainer(v3_model, args, v3_dataset)
        v2_trainer.set_id(0)
        v3_trainer.set_id(0)
        v2_metrics = v2_trainer.train(
            v2_partition,
            torch.device("cpu"),
            args,
        )
        v3_metrics = v3_trainer.train(
            v3_partition,
            torch.device("cpu"),
            args,
        )

        self.assertEqual(
            v2_metrics["optimizer_step_positive_count"],
            v3_metrics["optimizer_step_positive_count"],
        )
        self.assertEqual(
            v2_metrics["train_loss"],
            v3_metrics["train_loss"],
        )
        for name, v2_value in v2_model.state_dict().items():
            torch.testing.assert_close(
                v2_value,
                v3_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_fb15k237_partition_keeps_expected_hash(self) -> None:
        """确认37客户端数据无遗漏且划分哈希与V2正式结果一致。"""

        dataset = load_fb15k237(
            self._package_dir() / "data" / "FB15k-237"
        )
        federated = partition_train_triples_by_head(
            dataset,
            client_count=37,
            seed=42,
        )
        self.assertEqual(federated.client_count, 37)
        self.assertEqual(
            federated.partition_hash,
            "8bcac64b705ec2db8721de6a36130625"
            "a460c11e0da46e2c22bd852ff015fb19",
        )
        counts = [
            partition.triple_count
            for partition in federated.partitions
        ]
        self.assertEqual(sum(counts), 272115)
        self.assertEqual(min(counts), 7354)
        self.assertEqual(max(counts), 7355)


if __name__ == "__main__":
    unittest.main()
