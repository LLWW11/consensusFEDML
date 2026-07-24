"""集中式TransE训练与结果输出轻量测试。"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from HFLSnF_KG.core.randomness import seed_everything
from HFLSnF_KG.tasks.kge import (
    CentralizedTransETrainer,
    TransE,
    build_synthetic_knowledge_graph,
)


def build_trainer_args() -> SimpleNamespace:
    """构造两轮集中式TransE轻量训练参数。"""

    return SimpleNamespace(
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        margin=1.0,
        negative_sample_count=1,
        eval_every=1,
        validation_max_triples=0,
        final_validation_max_triples=0,
        test_max_triples=0,
        evaluation_candidate_batch_size=8,
        early_stopping_patience=0,
        random_seed=13,
    )


class CentralizedTransETrainerTest(unittest.TestCase):
    """验证集中式基线可以在CPU上完成训练、选模和测试。"""

    def test_training_writes_metrics_and_returns_finite_results(self) -> None:
        """验证两轮训练输出CSV且最终filtered指标均为有限值。"""

        seed_everything(13)
        dataset = build_synthetic_knowledge_graph()
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
        )
        trainer = CentralizedTransETrainer(
            build_trainer_args(),
            dataset,
            model,
            torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = trainer.train(Path(temp_dir))
            self.assertTrue((Path(temp_dir) / "metrics.csv").is_file())
        self.assertEqual(summary["runtime"], "fedml_configured_centralized_transe")
        self.assertGreaterEqual(summary["best_epoch"], 1)
        for value in summary["final_test_metrics"].values():
            self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
