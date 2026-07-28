"""集中式TransE训练与结果输出轻量测试。"""

from __future__ import annotations

import csv
import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from HFLSnF_KG_v2.core.randomness import seed_everything
from HFLSnF_KG_v2.tasks.kge import (
    CentralizedTransETrainer,
    TransE,
    build_synthetic_knowledge_graph,
)


def build_trainer_args() -> SimpleNamespace:
    """构造三轮集中式TransE轻量训练参数。"""

    return SimpleNamespace(
        epochs=3,
        batch_size=4,
        learning_rate=0.01,
        margin=1.0,
        negative_sample_count=1,
        monitor_every_epoch=True,
        monitor_validation_max_triples=0,
        eval_every=3,
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
        """验证逐轮监控输出、周期选模和最终指标均正确。"""

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
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.csv"
            with redirect_stdout(stdout):
                summary = trainer.train(Path(temp_dir))
            self.assertTrue(metrics_path.is_file())
            with metrics_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as metrics_file:
                rows = list(csv.DictReader(metrics_file))

        output = stdout.getvalue()
        self.assertEqual(len(rows), 3)
        self.assertEqual(output.count("[Epoch "), 3)
        self.assertEqual(output.count("监控MRR="), 3)
        self.assertEqual(output.count("监控Hits@3="), 3)
        self.assertEqual(output.count("选模MRR="), 2)
        self.assertTrue(math.isfinite(float(rows[1]["monitor_mrr"])))
        self.assertTrue(
            math.isfinite(float(rows[1]["monitor_hits_at_3"]))
        )
        self.assertTrue(math.isnan(float(rows[1]["val_mrr"])))
        self.assertGreater(float(rows[1]["epoch_seconds"]), 0.0)
        self.assertEqual(summary["runtime"], "fedml_configured_centralized_transe")
        self.assertTrue(summary["monitor_every_epoch"])
        self.assertEqual(summary["selection_eval_every"], 3)
        self.assertGreaterEqual(summary["best_epoch"], 1)
        for value in summary["final_test_metrics"].values():
            self.assertTrue(math.isfinite(float(value)))


if __name__ == "__main__":
    unittest.main()
