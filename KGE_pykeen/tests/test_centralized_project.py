"""验证matched与native两份正式CUDA配置合同。"""

from __future__ import annotations

import unittest

from KGE_pykeen.configuration import PACKAGE_DIR, load_flat_config
from KGE_pykeen.runtime import should_run_selection_evaluation


class FormalConfigurationTest(unittest.TestCase):
    """检查两种正式模式的共同配方和语义标识。"""

    COMMON_EXPECTED = {
        "random_seed": 42,
        "dataset": "fb15k-237",
        "embedding_dim": 256,
        "distance_norm": 1,
        "local_objective": "bidirectional_self_adversarial",
        "fede_gamma": 9.0,
        "adversarial_temperature": 1.0,
        "epochs": 450,
        "batch_size": 1024,
        "learning_rate": 0.00005,
        "negative_sample_count": 256,
        "eval_every": 3,
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

    def _load_formal_config(self, filename: str):
        """读取一份正式配置并核对共同超参数。"""

        config = load_flat_config(PACKAGE_DIR / "configs" / filename)
        for field, value in self.COMMON_EXPECTED.items():
            self.assertEqual(config[field], value, msg=field)
        return config

    def test_matched_formal_config(self) -> None:
        """确认matched正式配置使用严格配方模式。"""

        config = self._load_formal_config(
            "matched_fb15k237_seed42_cuda.yaml"
        )
        self.assertEqual(config["comparison_mode"], "matched_recipe")

    def test_native_formal_config(self) -> None:
        """确认native正式配置使用PyKEEN原生模式。"""

        config = self._load_formal_config(
            "native_fb15k237_seed42_cuda.yaml"
        )
        self.assertEqual(config["comparison_mode"], "pykeen_native")

    def test_selection_schedule_uses_150_aligned_points(self) -> None:
        """确认matched与native共享3轮间隔和150个选模时点。"""

        evaluation_epochs = [
            epoch
            for epoch in range(1, 451)
            if should_run_selection_evaluation(epoch, 3)
        ]
        self.assertEqual(evaluation_epochs, list(range(3, 451, 3)))
        self.assertNotIn(1, evaluation_epochs)
        self.assertEqual(evaluation_epochs[-1], 450)
        self.assertEqual(len(evaluation_epochs), 150)


if __name__ == "__main__":
    unittest.main()
