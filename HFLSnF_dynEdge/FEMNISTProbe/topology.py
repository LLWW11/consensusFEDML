"""为有限长度 MATLAB 拓扑增加显式、可审计的循环调度。"""

from __future__ import absolute_import

from dataclasses import dataclass

from topology_schedule import MatlabTopologySchedule


@dataclass(frozen=True)
class CyclicTopologyRound:
    """保存全局轮次、MAT源行和已解析轮次拓扑。"""

    global_epoch: int
    topology_cycle_index: int
    mat_topology_index: int
    topology: object


class CyclicMatlabTopology:
    """按显式模式读取 MAT；cycle 模式允许有限拓扑重复使用。"""

    def __init__(self, schedule, repeat_mode="error"):
        """初始化循环包装器并校验重复模式。"""
        if not isinstance(schedule, MatlabTopologySchedule):
            raise TypeError("schedule 必须是 MatlabTopologySchedule。")
        repeat_mode = str(repeat_mode).strip().lower()
        if repeat_mode not in {"error", "cycle"}:
            raise ValueError("topology_repeat_mode 只能是 error 或 cycle。")
        self.schedule = schedule
        self.repeat_mode = repeat_mode

    @property
    def round_count(self):
        """返回源 MAT 中的真实拓扑行数。"""
        return int(self.schedule.round_count)

    def get_round(self, global_epoch):
        """返回全局 epoch 对应的源 MAT 行及循环编号。"""
        global_epoch = int(global_epoch)
        if global_epoch < 0:
            raise IndexError("global_epoch 不能为负数。")
        if self.repeat_mode == "error" and global_epoch >= self.round_count:
            raise IndexError(
                "global_epoch={} 超过 MAT 的 {} 行，且未启用 cycle。".format(
                    global_epoch, self.round_count
                )
            )
        mat_index = (
            global_epoch % self.round_count
            if self.repeat_mode == "cycle"
            else global_epoch
        )
        cycle_index = global_epoch // self.round_count
        return CyclicTopologyRound(
            global_epoch=global_epoch,
            topology_cycle_index=cycle_index,
            mat_topology_index=mat_index,
            topology=self.schedule.get_round(mat_index),
        )

    def to_metadata(self):
        """返回包含循环策略的可序列化元数据。"""
        metadata = self.schedule.to_metadata()
        metadata.update({
            "topology_repeat_mode": self.repeat_mode,
            "source_mat_round_count": self.round_count,
        })
        return metadata
