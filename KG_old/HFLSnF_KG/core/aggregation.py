"""可在边缘层和云端继续合并的稠密FedAvg实现。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch

from .types import AggregateStats, ClientUpdate, clone_state_dict


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
