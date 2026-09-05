"""验证图语义FedAvg九实验的断点恢复行为。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from HFLSnF_KG_v5 import run_graph_semantic_fedavg_comparison as runner


class GraphSemanticFedAvgResumeTest(unittest.TestCase):
    """检查已通过场景跳过与剩余场景顺序恢复。"""

    def test_resume_skips_passed_and_finishes_remaining_entries(self) -> None:
        """恢复批次不得重跑已通过项，并应完成剩余八项。"""

        payload = runner._empty_batch_payload()
        first = payload["entries"][0]
        first.update(
            {
                "status": "passed",
                "result_dir": "existing-result",
                "contract_file": "existing-contract",
            }
        )
        created = iter(
            Path("result-{}".format(index)) for index in range(1, 9)
        )
        with mock.patch.object(
            runner, "_load_batch_manifest", return_value=payload
        ), mock.patch.object(runner, "_save_batch_manifest"), mock.patch.object(
            runner, "_run_scenario", side_effect=lambda scenario: next(created)
        ) as run_scenario, mock.patch.object(
            runner,
            "validate_result",
            side_effect=lambda result_dir, scenario: {"status": "passed"},
        ), mock.patch.object(runner, "write_json_report"):
            status = runner.run_training_batch(Path("unused.json"))
        self.assertEqual(status, 0)
        self.assertEqual(run_scenario.call_count, 8)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["pilot_seed42_gate"], "passed")
        self.assertTrue(
            all(entry["status"] == "passed" for entry in payload["entries"])
        )
        self.assertEqual(first["result_dir"], "existing-result")


if __name__ == "__main__":
    unittest.main()
