"""分层联邦模拟器与具体学习任务之间的抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import torch

from ..core.types import ClientUpdate


class FederatedTask(ABC):
    """定义模型、客户端本地训练和全局评估必须实现的任务接口。"""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """返回用于日志和结果摘要的任务名称。"""

    @property
    @abstractmethod
    def client_ids(self) -> Sequence[int]:
        """返回当前任务中全部合法客户端编号。"""

    @abstractmethod
    def get_global_state(self) -> Dict[str, torch.Tensor]:
        """返回当前全局模型状态的CPU深拷贝。"""

    @abstractmethod
    def set_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """把聚合后的模型状态加载为新的全局模型。"""

    @abstractmethod
    def train_client(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor],
        local_epochs: int,
        round_index: int,
    ) -> Optional[ClientUpdate]:
        """从全局状态开始训练一个客户端并返回可聚合更新。"""

    @abstractmethod
    def evaluate_global(self) -> Dict[str, float]:
        """在任务定义的全局训练、验证和测试划分上评估模型。"""

    @abstractmethod
    def partition_summary(self) -> Dict[str, object]:
        """返回便于复核客户端数据划分的摘要。"""
