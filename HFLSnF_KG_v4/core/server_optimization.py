"""服务器端逐行FedAdam优化器。"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch


class RowWiseFedAdamOptimizer:
    """在FedAvg候选模型上执行带有效行保护的服务器端FedAdam。"""

    def __init__(
        self,
        learning_rate: float,
        beta1: float,
        beta2: float,
        tau: float,
        bias_correction: bool = False,
    ) -> None:
        """保存服务器超参数并初始化空的一阶、二阶动量。"""

        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.bias_correction = bool(bias_correction)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("server_learning_rate必须是有限正数")
        for name, value in (
            ("server_beta1", self.beta1),
            ("server_beta2", self.beta2),
        ):
            if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError("{}必须位于[0, 1)".format(name))
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("server_tau必须是有限正数")

        self._step_count = 0
        self._first_moments: Dict[str, torch.Tensor] = {}
        self._second_moments: Dict[str, torch.Tensor] = {}

    @property
    def step_count(self) -> int:
        """返回已经完成的服务器端优化步数。"""

        return int(self._step_count)

    @staticmethod
    def _validate_state_keys(
        global_state: Mapping[str, torch.Tensor],
        candidate_state: Mapping[str, torch.Tensor],
    ) -> Tuple[str, ...]:
        """校验轮初全局状态和FedAvg候选状态的参数键一致。"""

        global_keys = tuple(global_state.keys())
        candidate_keys = tuple(candidate_state.keys())
        if set(global_keys) != set(candidate_keys):
            raise ValueError(
                "FedAdam全局状态与候选状态参数键不一致：{}与{}".format(
                    sorted(global_keys),
                    sorted(candidate_keys),
                )
            )
        return global_keys

    @staticmethod
    def _validate_active_mask(
        name: str,
        value: torch.Tensor,
        active_row_masks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """读取并校验一个浮点参数的一维有效行掩码。"""

        if name not in active_row_masks:
            raise ValueError(
                "FedAdam缺少参数{}的有效行掩码".format(name)
            )
        mask = active_row_masks[name]
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise TypeError(
                "参数{}的FedAdam有效行掩码必须是一维bool张量".format(
                    name
                )
            )
        if value.ndim <= 0 or int(mask.shape[0]) != int(value.shape[0]):
            raise ValueError(
                "参数{}的FedAdam有效行掩码长度与参数首维不一致".format(
                    name
                )
            )
        return mask.to(device=value.device)

    def _previous_moments(
        self,
        name: str,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回与当前参数设备和类型一致的历史一阶、二阶动量。"""

        first = self._first_moments.get(name)
        second = self._second_moments.get(name)
        if first is None or second is None:
            return torch.zeros_like(reference), torch.zeros_like(reference)
        if first.shape != reference.shape or second.shape != reference.shape:
            raise ValueError(
                "参数{}的FedAdam历史动量形状发生变化".format(name)
            )
        return (
            first.to(device=reference.device, dtype=reference.dtype),
            second.to(device=reference.device, dtype=reference.dtype),
        )

    def step(
        self,
        global_state: Mapping[str, torch.Tensor],
        candidate_state: Mapping[str, torch.Tensor],
        active_row_masks: Mapping[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """根据候选模型增量更新有效行，并返回新状态和完整审计统计。"""

        parameter_names = self._validate_state_keys(
            global_state, candidate_state
        )
        next_step = self._step_count + 1
        first_bias = 1.0 - self.beta1 ** next_step
        second_bias = 1.0 - self.beta2 ** next_step
        optimized_state: Dict[str, torch.Tensor] = {}
        parameter_statistics: Dict[str, Dict[str, float]] = {}
        total_delta_square = 0.0
        total_update_square = 0.0
        maximum_update = 0.0
        total_active_rows = 0

        # 先完成全部校验和候选计算，成功后才提交步数与动量。
        next_first_moments: Dict[str, torch.Tensor] = {}
        next_second_moments: Dict[str, torch.Tensor] = {}
        for name in parameter_names:
            global_value = global_state[name].detach()
            candidate_value = candidate_state[name].detach().to(
                device=global_value.device,
                dtype=global_value.dtype,
            )
            if global_value.shape != candidate_value.shape:
                raise ValueError(
                    "参数{}的全局状态与FedAvg候选形状不一致".format(
                        name
                    )
                )
            if not (
                global_value.is_floating_point()
                or torch.is_complex(global_value)
            ):
                if not torch.equal(global_value, candidate_value):
                    raise ValueError(
                        "非浮点缓冲区{}不能由FedAdam修改".format(name)
                    )
                optimized_state[name] = global_value.clone()
                continue
            if torch.is_complex(global_value):
                raise TypeError("当前逐行FedAdam不支持复数参数{}".format(name))

            mask = self._validate_active_mask(
                name, global_value, active_row_masks
            )
            broadcast_shape = [int(mask.shape[0])] + [
                1 for _ in range(global_value.ndim - 1)
            ]
            broadcast_mask = mask.reshape(broadcast_shape)
            previous_first, previous_second = self._previous_moments(
                name, global_value
            )
            model_delta = candidate_value - global_value
            proposed_first = (
                self.beta1 * previous_first
                + (1.0 - self.beta1) * model_delta
            )
            proposed_second = (
                self.beta2 * previous_second
                + (1.0 - self.beta2) * model_delta.square()
            )
            next_first = torch.where(
                broadcast_mask, proposed_first, previous_first
            )
            next_second = torch.where(
                broadcast_mask, proposed_second, previous_second
            )
            first_for_update = next_first
            second_for_update = next_second
            if self.bias_correction:
                first_for_update = first_for_update / first_bias
                second_for_update = second_for_update / second_bias
            server_update = (
                self.learning_rate
                * first_for_update
                / (torch.sqrt(second_for_update) + self.tau)
            )
            masked_update = torch.where(
                broadcast_mask,
                server_update,
                torch.zeros_like(server_update),
            )
            optimized_state[name] = global_value + masked_update
            next_first_moments[name] = next_first.detach().clone()
            next_second_moments[name] = next_second.detach().clone()

            active_delta = model_delta[mask]
            active_update = masked_update[mask]
            active_first = next_first[mask]
            active_second = next_second[mask]
            active_row_count = int(mask.sum().item())
            total_active_rows += active_row_count
            delta_square = float(active_delta.square().sum().item())
            update_square = float(active_update.square().sum().item())
            update_max = (
                float(active_update.abs().max().item())
                if int(active_update.numel()) > 0
                else 0.0
            )
            total_delta_square += delta_square
            total_update_square += update_square
            maximum_update = max(maximum_update, update_max)
            parameter_statistics[name] = {
                "row_count": int(mask.numel()),
                "active_row_count": active_row_count,
                "inactive_row_count": int((~mask).sum().item()),
                "model_delta_l2": math.sqrt(delta_square),
                "server_update_l2": math.sqrt(update_square),
                "server_update_mean_abs": (
                    float(active_update.abs().mean().item())
                    if int(active_update.numel()) > 0
                    else 0.0
                ),
                "server_update_max_abs": update_max,
                "first_moment_l2": math.sqrt(
                    float(active_first.square().sum().item())
                )
                if int(active_first.numel()) > 0
                else 0.0,
                "second_moment_l2": math.sqrt(
                    float(active_second.square().sum().item())
                )
                if int(active_second.numel()) > 0
                else 0.0,
            }

        self._step_count = next_step
        self._first_moments = next_first_moments
        self._second_moments = next_second_moments
        return optimized_state, {
            "server_optimizer": "fedadam",
            "server_optimizer_step": int(self._step_count),
            "server_learning_rate": self.learning_rate,
            "server_beta1": self.beta1,
            "server_beta2": self.beta2,
            "server_tau": self.tau,
            "server_bias_correction": self.bias_correction,
            "server_active_row_count": int(total_active_rows),
            "server_model_delta_l2": math.sqrt(total_delta_square),
            "server_update_l2": math.sqrt(total_update_square),
            "server_update_max_abs": float(maximum_update),
            "server_parameter_statistics": parameter_statistics,
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """返回可用于哈希审计的服务器动量与步数张量副本。"""

        state: Dict[str, torch.Tensor] = {
            "server_optimizer_step": torch.tensor(
                [self._step_count], dtype=torch.int64
            )
        }
        for name, value in self._first_moments.items():
            state["first_moment.{}".format(name)] = (
                value.detach().clone()
            )
        for name, value in self._second_moments.items():
            state["second_moment.{}".format(name)] = (
                value.detach().clone()
            )
        return state
