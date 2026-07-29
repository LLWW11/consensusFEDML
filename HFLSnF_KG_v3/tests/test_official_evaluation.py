"""验证最佳检查点完整官方测试合同。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from HFLSnF_KG_v3.tasks.kge import (
    TransE,
    build_synthetic_knowledge_graph,
    run_best_checkpoint_official_evaluation,
)


class OfficialEvaluationTest(unittest.TestCase):
    """使用合成图验证完整测试不可静默退化为子集。"""

    def test_best_checkpoint_full_test_writes_contract(self) -> None:
        """确认全部测试事实、头尾指标和中文报告均被写出。"""

        dataset = build_synthetic_knowledge_graph()
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
            distance_norm=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "training"
            output_dir = Path(temporary) / "official"
            result_dir.mkdir(parents=True)
            summary = {
                "dataset": dataset.dataset_name,
                "best_round": 3,
                "best_validation_mrr_during_training": 0.25,
                "centralized_reference_test_mrr": 0.50,
                "final_test_metrics": {
                    "mrr": 0.20,
                    "evaluated_triple_count": 1,
                },
            }
            with (result_dir / "summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(summary, handle)
            with (result_dir / "config_snapshot.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump({"distance_norm": 1}, handle)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "entity_to_id": dict(dataset.entity_to_id),
                    "relation_to_id": dict(dataset.relation_to_id),
                },
                result_dir / "model_best.pt",
            )
            contract = run_best_checkpoint_official_evaluation(
                dataset=dataset,
                result_dir=result_dir,
                output_dir=output_dir,
                device=torch.device("cpu"),
                query_batch_size=2,
                candidate_batch_size=4,
                progress_every=0,
            )
            self.assertEqual(contract["status"], "passed")
            self.assertTrue(contract["full_official_test"])
            self.assertEqual(
                contract["official_test_triple_count"],
                int(dataset.test_triples.shape[0]),
            )
            self.assertEqual(
                contract["official_test_query_count"],
                int(dataset.test_triples.shape[0]) * 2,
            )
            for filename in (
                "official_evaluation_summary.json",
                "完整官方测试报告.md",
                "directional_summary.json",
                "query_ranks.csv",
                "relation_metrics.csv",
            ):
                self.assertTrue((output_dir / filename).is_file())


if __name__ == "__main__":
    unittest.main()
