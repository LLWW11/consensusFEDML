"""验证完整官方测试使用的头尾方向检查点读取和输出。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from HFLSnF_KG_v5.tasks.kge import (
    TransE,
    build_synthetic_knowledge_graph,
    run_directional_diagnostic,
)


class Stage1DirectionalTest(unittest.TestCase):
    """使用合成图检查单检查点头尾诊断链路。"""

    def test_directional_diagnostic_writes_complete_artifacts(self) -> None:
        """确认诊断同时输出头、尾、综合指标和逐查询文件。"""

        dataset = build_synthetic_knowledge_graph()
        model = TransE(
            dataset.num_entities,
            dataset.num_relations,
            embedding_dim=8,
            distance_norm=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary) / "checkpoint"
            result_dir.mkdir(parents=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "entity_to_id": dict(dataset.entity_to_id),
                    "relation_to_id": dict(dataset.relation_to_id),
                },
                result_dir / "model_best.pt",
            )
            with (result_dir / "config_snapshot.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "model": "transe",
                        "embedding_dim": 8,
                        "distance_norm": 1,
                    },
                    handle,
                )
            output_dir = Path(temporary) / "diagnostic"
            summary = run_directional_diagnostic(
                dataset=dataset,
                checkpoint=result_dir,
                output_dir=output_dir,
                device=torch.device("cpu"),
                max_triples=4,
                selection_seed=42,
                query_batch_size=2,
                candidate_batch_size=4,
                progress_every=0,
            )

            expected_triples = min(4, len(dataset.test_triples))
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["training_performed"])
            self.assertEqual(
                summary["selected_triple_count"], expected_triples
            )
            self.assertEqual(
                summary["head_metrics"]["evaluated_query_count"],
                expected_triples,
            )
            self.assertEqual(
                summary["tail_metrics"]["evaluated_query_count"],
                expected_triples,
            )
            self.assertEqual(
                summary["combined_metrics"]["evaluated_query_count"],
                expected_triples * 2,
            )
            for filename in (
                "directional_summary.json",
                "query_ranks.csv",
                "relation_metrics.csv",
                "方向诊断报告.md",
            ):
                self.assertTrue((output_dir / filename).is_file())


if __name__ == "__main__":
    unittest.main()
