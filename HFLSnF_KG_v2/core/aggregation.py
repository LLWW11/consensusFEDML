"""可在边缘层和云端继续合并的稠密FedAvg实现。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch

from .types import (
    AggregateStats,
    ClientUpdate,
    RowMaskedAggregateStats,
    clone_state_dict,
)


def _is_weighted_tensor(tensor: torch.Tensor) -> bool:
    """判断张量是否可安全参与浮点加权平均。"""

    return bool(tensor.is_floating_point() or torch.is_complex(tensor))


def _assert_same_keys(
    expected_keys: Iterable[str],
    actual_state: Mapping[str, torch.Tensor],
    context: str,
) -> None:
    """校验模型状态或聚合统计的参数键集合完全一致。"""

    expected = tuple(expected_keys)
    actual = tuple(actual_state.keys())
    if set(expected) != set(actual):
        raise ValueError(
            "{}参数键不一致，期望{}，实际{}".format(
                context, sorted(expected), sorted(actual)
            )
        )


class DenseFedAvgAggregator:
    """对同形状模型状态执行按标量权重加权的可合并FedAvg。"""

    def accumulate(self, updates: Iterable[ClientUpdate]) -> AggregateStats:
        """把客户端更新转换为可继续向云端合并的边缘统计量。"""

        valid_updates = [update for update in updates if update.weight > 0.0]
        if not valid_updates:
            raise ValueError("至少需要一个正权重客户端更新")

        reference_state = valid_updates[0].state_dict
        reference_keys = tuple(reference_state.keys())
        weighted_sums: Dict[str, torch.Tensor] = {}
        constant_tensors: Dict[str, torch.Tensor] = {}
        total_weight = 0.0
        contributor_ids: List[int] = []

        for update in valid_updates:
            _assert_same_keys(reference_keys, update.state_dict, "客户端更新")
            total_weight += float(update.weight)
            contributor_ids.append(int(update.client_id))
            for name in reference_keys:
                value = update.state_dict[name].detach()
                reference_value = reference_state[name]
                if value.shape != reference_value.shape:
                    raise ValueError(
                        "参数{}形状不一致：{}与{}".format(
                            name, tuple(reference_value.shape), tuple(value.shape)
                        )
                    )
                if value.dtype != reference_value.dtype:
                    raise ValueError(
                        "参数{}数据类型不一致：{}与{}".format(
                            name, reference_value.dtype, value.dtype
                        )
                    )

                if _is_weighted_tensor(value):
                    weighted_value = value.clone() * float(update.weight)
                    if name not in weighted_sums:
                        weighted_sums[name] = weighted_value
                    else:
                        weighted_sums[name] = weighted_sums[name] + weighted_value
                else:
                    if name not in constant_tensors:
                        constant_tensors[name] = value.clone()
                    elif not torch.equal(constant_tensors[name], value):
                        raise ValueError(
                            "非浮点缓冲区{}在客户端之间不一致，无法安全平均".format(
                                name
                            )
                        )

        return AggregateStats(
            weighted_sums=weighted_sums,
            constant_tensors=constant_tensors,
            total_weight=total_weight,
            contributor_ids=tuple(contributor_ids),
        )

    def merge(self, statistics: Iterable[AggregateStats]) -> AggregateStats:
        """合并多个边缘统计量，保留逐参数加权分子和全局分母。"""

        stats_list = list(statistics)
        if not stats_list:
            raise ValueError("至少需要一个边缘聚合统计")

        reference = stats_list[0]
        weighted_keys = tuple(reference.weighted_sums.keys())
        constant_keys = tuple(reference.constant_tensors.keys())
        weighted_sums = clone_state_dict(reference.weighted_sums, to_cpu=False)
        constant_tensors = clone_state_dict(
            reference.constant_tensors, to_cpu=False
        )
        total_weight = float(reference.total_weight)
        contributor_ids = list(reference.contributor_ids)

        for stats in stats_list[1:]:
            _assert_same_keys(weighted_keys, stats.weighted_sums, "边缘浮点统计")
            _assert_same_keys(constant_keys, stats.constant_tensors, "边缘常量统计")
            duplicate_ids = set(contributor_ids).intersection(stats.contributor_ids)
            if duplicate_ids:
                raise ValueError(
                    "客户端不能跨边缘组重复贡献：{}".format(sorted(duplicate_ids))
                )
            for name in weighted_keys:
                if weighted_sums[name].shape != stats.weighted_sums[name].shape:
                    raise ValueError("边缘统计中的参数{}形状不一致".format(name))
                weighted_sums[name] = (
                    weighted_sums[name] + stats.weighted_sums[name]
                )
            for name in constant_keys:
                if not torch.equal(
                    constant_tensors[name], stats.constant_tensors[name]
                ):
                    raise ValueError(
                        "边缘统计中的非浮点缓冲区{}不一致".format(name)
                    )
            total_weight += float(stats.total_weight)
            contributor_ids.extend(stats.contributor_ids)

        return AggregateStats(
            weighted_sums=weighted_sums,
            constant_tensors=constant_tensors,
            total_weight=total_weight,
            contributor_ids=tuple(contributor_ids),
        )

    def finalize(self, statistics: AggregateStats) -> Dict[str, torch.Tensor]:
        """用总权重归一化聚合分子并还原完整模型状态。"""

        averaged_state = {}
        for name, weighted_sum in statistics.weighted_sums.items():
            averaged_state[name] = weighted_sum / float(statistics.total_weight)
        for name, value in statistics.constant_tensors.items():
            averaged_state[name] = value.detach().clone()
        return averaged_state

    def aggregate(self, updates: Iterable[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """直接完成客户端更新的单层FedAvg，供FL路径和等价性测试使用。"""

        return self.finalize(self.accumulate(updates))


class RowMaskedFedAvgAggregator:
    """按FedE实体存在向量思想逐参数行等权聚合客户端状态。"""

    @staticmethod
    def _validate_mask(
        name: str,
        value: torch.Tensor,
        update: ClientUpdate,
    ) -> torch.Tensor:
        """读取并校验一个客户端针对指定参数提供的行所有权掩码。"""

        if name not in update.parameter_masks:
            raise ValueError(
                "客户端{}没有为参数{}提供逐行掩码".format(
                    update.client_id, name
                )
            )
        mask = update.parameter_masks[name]
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise TypeError("参数{}的行掩码必须是一维bool张量".format(name))
        if int(mask.shape[0]) != int(value.shape[0]):
            raise ValueError("参数{}的行掩码长度与参数首维不一致".format(name))
        return mask.to(device=value.device)

    def accumulate(
        self, updates: Iterable[ClientUpdate]
    ) -> RowMaskedAggregateStats:
        """把一组客户端状态转换为可继续向云端合并的逐行统计。"""

        update_list = list(updates)
        if not update_list:
            raise ValueError("逐行聚合至少需要一个客户端更新")
        reference_state = update_list[0].state_dict
        reference_keys = tuple(reference_state.keys())
        row_sums: Dict[str, torch.Tensor] = {}
        row_denominators: Dict[str, torch.Tensor] = {}
        constant_tensors: Dict[str, torch.Tensor] = {}
        contributor_ids: List[int] = []

        for update in update_list:
            _assert_same_keys(reference_keys, update.state_dict, "客户端更新")
            contributor_ids.append(int(update.client_id))
            for name in reference_keys:
                value = update.state_dict[name].detach()
                reference_value = reference_state[name]
                if value.shape != reference_value.shape:
                    raise ValueError("参数{}形状不一致".format(name))
                if value.dtype != reference_value.dtype:
                    raise ValueError("参数{}数据类型不一致".format(name))

                if _is_weighted_tensor(value):
                    if value.ndim <= 0:
                        raise ValueError(
                            "逐行聚合不支持零维浮点参数{}".format(name)
                        )
                    mask = self._validate_mask(name, value, update)
                    row_weight = mask.to(dtype=value.dtype)
                    broadcast_shape = [int(value.shape[0])] + [
                        1 for _ in range(value.ndim - 1)
                    ]
                    contribution = value * row_weight.reshape(
                        broadcast_shape
                    )
                    if name not in row_sums:
                        row_sums[name] = contribution.clone()
                        row_denominators[name] = row_weight.clone()
                    else:
                        row_sums[name] = row_sums[name] + contribution
                        row_denominators[name] = (
                            row_denominators[name] + row_weight
                        )
                else:
                    if name not in constant_tensors:
                        constant_tensors[name] = value.clone()
                    elif not torch.equal(constant_tensors[name], value):
                        raise ValueError(
                            "非浮点缓冲区{}在客户端之间不一致".format(name)
                        )

        return RowMaskedAggregateStats(
            row_sums=row_sums,
            row_denominators=row_denominators,
            constant_tensors=constant_tensors,
            contributor_ids=tuple(contributor_ids),
        )

    def merge(
        self, statistics: Iterable[RowMaskedAggregateStats]
    ) -> RowMaskedAggregateStats:
        """合并多个边缘端逐行分子和分母而不提前丢失行权重。"""

        stats_list = list(statistics)
        if not stats_list:
            raise ValueError("至少需要一个边缘逐行聚合统计")
        reference = stats_list[0]
        row_keys = tuple(reference.row_sums.keys())
        constant_keys = tuple(reference.constant_tensors.keys())
        row_sums = clone_state_dict(reference.row_sums, to_cpu=False)
        row_denominators = clone_state_dict(
            reference.row_denominators, to_cpu=False
        )
        constant_tensors = clone_state_dict(
            reference.constant_tensors, to_cpu=False
        )
        contributor_ids = list(reference.contributor_ids)

        for stats in stats_list[1:]:
            _assert_same_keys(row_keys, stats.row_sums, "边缘逐行分子")
            _assert_same_keys(
                row_keys, stats.row_denominators, "边缘逐行分母"
            )
            _assert_same_keys(
                constant_keys, stats.constant_tensors, "边缘常量"
            )
            duplicate_ids = set(contributor_ids).intersection(
                stats.contributor_ids
            )
            if duplicate_ids:
                raise ValueError(
                    "客户端不能跨边缘组重复贡献：{}".format(
                        sorted(duplicate_ids)
                    )
                )
            for name in row_keys:
                if row_sums[name].shape != stats.row_sums[name].shape:
                    raise ValueError("边缘参数{}的分子形状不一致".format(name))
                if (
                    row_denominators[name].shape
                    != stats.row_denominators[name].shape
                ):
                    raise ValueError("边缘参数{}的分母形状不一致".format(name))
                row_sums[name] = row_sums[name] + stats.row_sums[name]
                row_denominators[name] = (
                    row_denominators[name]
                    + stats.row_denominators[name]
                )
            for name in constant_keys:
                if not torch.equal(
                    constant_tensors[name], stats.constant_tensors[name]
                ):
                    raise ValueError(
                        "边缘非浮点缓冲区{}不一致".format(name)
                    )
            contributor_ids.extend(stats.contributor_ids)

        return RowMaskedAggregateStats(
            row_sums=row_sums,
            row_denominators=row_denominators,
            constant_tensors=constant_tensors,
            contributor_ids=tuple(contributor_ids),
        )

    def finalize(
        self,
        statistics: RowMaskedAggregateStats,
        global_state: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """逐行归一化，并用轮初全局参数填充本轮无人拥有的行。"""

        expected_keys = tuple(statistics.row_sums.keys()) + tuple(
            statistics.constant_tensors.keys()
        )
        _assert_same_keys(expected_keys, global_state, "轮初全局状态")
        averaged_state: Dict[str, torch.Tensor] = {}
        for name, numerator in statistics.row_sums.items():
            denominator = statistics.row_denominators[name]
            global_value = global_state[name].detach().to(
                device=numerator.device, dtype=numerator.dtype
            )
            if global_value.shape != numerator.shape:
                raise ValueError("全局参数{}形状与逐行统计不一致".format(name))
            broadcast_shape = [int(denominator.shape[0])] + [
                1 for _ in range(numerator.ndim - 1)
            ]
            safe_denominator = torch.where(
                denominator > 0,
                denominator,
                torch.ones_like(denominator),
            )
            averaged = numerator / safe_denominator.reshape(
                broadcast_shape
            )
            active_rows = (denominator > 0).reshape(broadcast_shape)
            averaged_state[name] = torch.where(
                active_rows, averaged, global_value
            )
        for name, value in statistics.constant_tensors.items():
            averaged_state[name] = value.detach().clone()
        return averaged_state

    @staticmethod
    def summarize(
        statistics: RowMaskedAggregateStats,
    ) -> Dict[str, Dict[str, float]]:
        """返回每个参数的有效行、回退行和贡献客户端统计。"""

        summary: Dict[str, Dict[str, float]] = {}
        for name, denominator in statistics.row_denominators.items():
            active = denominator > 0
            active_values = denominator[active]
            summary[name] = {
                "row_count": int(denominator.numel()),
                "updated_row_count": int(active.sum().item()),
                "fallback_row_count": int((~active).sum().item()),
                "max_row_contributors": (
                    float(active_values.max().item())
                    if int(active_values.numel()) > 0
                    else 0.0
                ),
                "mean_row_contributors": (
                    float(active_values.mean().item())
                    if int(active_values.numel()) > 0
                    else 0.0
                ),
            }
        return summary

    def aggregate(
        self,
        updates: Iterable[ClientUpdate],
        global_state: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """直接完成单层逐行聚合，供FL路径和等价性测试使用。"""

        return self.finalize(self.accumulate(updates), global_state)
