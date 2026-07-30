"""验证四组正式实验使用的逐行服务器FedAdam。"""

from __future__ import annotations

import unittest

import torch

from HFLSnF_KG_v3.core.server_optimization import (
    RowWiseFedAdamOptimizer,
)


class RowWiseFedAdamOptimizerTest(unittest.TestCase):
    """检查FedAdam数值更新和未参与行保护。"""

    def test_one_step_updates_only_active_rows(self) -> None:
        """确认首步只更新掩码内行且动量状态可审计。"""

        optimizer = RowWiseFedAdamOptimizer(0.1, 0.9, 0.99, 0.001)
        global_state = {"weight": torch.tensor([[1.0], [5.0]])}
        candidate = {"weight": torch.tensor([[3.0], [9.0]])}
        updated, audit = optimizer.step(
            global_state,
            candidate,
            {"weight": torch.tensor([True, False])},
        )
        expected = 1.0 + 0.1 * 0.2 / (0.2 + 0.001)
        self.assertAlmostEqual(
            float(updated["weight"][0, 0]),
            expected,
            places=6,
        )
        self.assertEqual(float(updated["weight"][1, 0]), 5.0)
        self.assertEqual(audit["server_optimizer_step"], 1)
        self.assertEqual(audit["server_active_row_count"], 1)

    def test_inactive_row_keeps_old_moments_across_steps(self) -> None:
        """确认下一轮未参与的行不会被历史动量继续推动。"""

        optimizer = RowWiseFedAdamOptimizer(0.1, 0.9, 0.99, 0.001)
        state = {"weight": torch.tensor([[0.0], [0.0]])}
        first, _ = optimizer.step(
            state,
            {"weight": torch.tensor([[1.0], [1.0]])},
            {"weight": torch.tensor([True, True])},
        )
        second, _ = optimizer.step(
            first,
            {"weight": first["weight"] + 1.0},
            {"weight": torch.tensor([False, True])},
        )
        self.assertEqual(
            float(second["weight"][0, 0]),
            float(first["weight"][0, 0]),
        )
        self.assertGreater(
            float(second["weight"][1, 0]),
            float(first["weight"][1, 0]),
        )


if __name__ == "__main__":
    unittest.main()
