"""验证集中式正式配置和无需FedML的CPU冒烟链路。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from KGE_central.configuration import PACKAGE_DIR, load_flat_config
from KGE_central.experiment import run_experiment


class CentralizedProjectTest(unittest.TestCase):
    """检查历史强配方合同和集中式输出产物。"""

    def test_strong_cuda_config_matches_runtime_recipe(self) -> None:
        """确认正式YAML保留核心配方并使用当前评估间隔。"""

        config = load_flat_config(
            PACKAGE_DIR
            / "configs"
            / "centralized_fb15k237_strong_transe_cuda.yaml"
        )
        expected = {
            "random_seed": 42,
            "dataset": "fb15k-237",
            "embedding_dim": 256,
            "distance_norm": 1,
            "local_objective": "bidirectional_self_adversarial",
            "fede_gamma": 9.0,
            "adversarial_temperature": 1.0,
            "epochs": 380,
            "batch_size": 1024,
            "learning_rate": 0.00005,
            "negative_sample_count": 256,
            "eval_every": 10,
            "validation_max_triples": 4096,
            "validation_selection": "relation_stratified",
            "final_validation_max_triples": 0,
            "test_max_triples": 0,
            "evaluation_query_batch_size": 64,
            "evaluation_candidate_batch_size": 8192,
            "using_gpu": True,
            "require_cuda": True,
            "gpu_id": 0,
        }
        for field, value in expected.items():
            self.assertEqual(config[field], value)

    def test_cpu_smoke_writes_complete_centralized_artifacts(self) -> None:
        """确认合成数据训练能够保存最佳模型和完整结果合同。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir, summary = run_experiment(
                PACKAGE_DIR
                / "configs"
                / "smoke_synthetic_cpu.yaml",
                result_root_override=Path(temp_dir),
            )
            self.assertEqual(summary["device"], "cpu")
            self.assertEqual(
                summary["runtime"],
                "standalone_centralized_transe",
            )
            self.assertEqual(summary["epochs_ran"], 2)
            self.assertEqual(
                summary["final_test_metrics"][
                    "evaluated_triple_count"
                ],
                2.0,
            )
            for filename in (
                "config_snapshot.json",
                "dataset_summary.json",
                "entity2id.json",
                "relation2id.json",
                "metrics.csv",
                "summary.json",
                "model_best.pt",
            ):
                self.assertTrue((result_dir / filename).is_file())


if __name__ == "__main__":
    unittest.main()
