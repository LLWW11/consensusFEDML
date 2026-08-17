"""知识图谱补全任务的严格过滤负采样器。"""

from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

import numpy as np
import torch


IdTriple = Tuple[int, int, int]


class VectorizedFilteredNegativeSampler:
    """在正样本所在设备上批量生成严格过滤的头或尾负样本。"""

    VALID_MODES = {"head", "tail", "head_tail"}
    backend = "torch_device_searchsorted"

    def __init__(
        self,
        num_entities: int,
        true_triples: Iterable[IdTriple],
        seed: int,
        num_relations: int = None,
    ):
        """把全部真三元组编码成排序张量，并初始化设备随机数状态。"""

        if int(num_entities) <= 1:
            raise ValueError("负采样至少需要两个候选实体")
        triples = [
            tuple(int(value) for value in triple)
            for triple in true_triples
        ]
        if not triples:
            raise ValueError("严格过滤负采样需要非空真三元组集合")
        inferred_relations = max(triple[1] for triple in triples) + 1
        self.num_entities = int(num_entities)
        self.num_relations = int(
            inferred_relations
            if num_relations is None
            else num_relations
        )
        if self.num_relations < inferred_relations:
            raise ValueError("num_relations小于真三元组中的关系编号范围")
        encoded = sorted(
            {
                self._encode_scalar(head, relation, tail)
                for head, relation, tail in triples
            }
        )
        self._true_keys_cpu = torch.tensor(encoded, dtype=torch.long)
        self._true_keys_by_device: Dict[str, torch.Tensor] = {
            "cpu": self._true_keys_cpu
        }
        self._generators: Dict[str, torch.Generator] = {}
        self.reset_seed(seed)

    def _encode_scalar(
        self, head: int, relation: int, tail: int
    ) -> int:
        """把一个三元组无冲突地编码为整数键。"""

        return (
            (int(head) * self.num_relations + int(relation))
            * self.num_entities
            + int(tail)
        )

    def _encode_tensor(self, triples: torch.Tensor) -> torch.Tensor:
        """把任意前缀形状的三元组张量编码为整数键张量。"""

        return (
            (
                triples[..., 0] * self.num_relations
                + triples[..., 1]
            )
            * self.num_entities
            + triples[..., 2]
        )

    def reset_seed(self, seed: int) -> None:
        """重置采样种子并清空各设备生成器，保证客户端调用可复现。"""

        self.seed = int(seed)
        self._generators = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        """返回与指定设备绑定且按需创建的独立随机数生成器。"""

        device = torch.device(device)
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return self._generators[key]

    def _true_keys(self, device: torch.device) -> torch.Tensor:
        """返回缓存到指定设备上的排序真三元组编码。"""

        device = torch.device(device)
        key = str(device)
        if key not in self._true_keys_by_device:
            self._true_keys_by_device[key] = self._true_keys_cpu.to(
                device=device,
                non_blocking=True,
            )
        return self._true_keys_by_device[key]

    @staticmethod
    def _normalized_mode(corruption_mode: str) -> str:
        """校验并返回统一的小写负采样方向名称。"""

        mode = str(corruption_mode).strip().lower()
        if mode not in VectorizedFilteredNegativeSampler.VALID_MODES:
            raise ValueError(
                "corruption_mode必须是head、tail或head_tail，实际为{}".format(
                    corruption_mode
                )
            )
        return mode

    def _known_true_mask(self, triples: torch.Tensor) -> torch.Tensor:
        """使用排序键和二分查找判断候选是否属于任一已知真三元组。"""

        true_keys = self._true_keys(triples.device)
        candidate_keys = self._encode_tensor(triples)
        positions = torch.searchsorted(true_keys, candidate_keys)
        safe_positions = positions.clamp(max=int(true_keys.numel()) - 1)
        return (positions < int(true_keys.numel())) & (
            true_keys[safe_positions] == candidate_keys
        )

    def sample(
        self,
        positive_triples: torch.Tensor,
        negative_sample_count: int = 1,
        corruption_mode: str = "head_tail",
    ) -> torch.Tensor:
        """为一个正样本批次并行生成指定数量的严格过滤负样本。"""

        if positive_triples.dtype != torch.long:
            raise TypeError("正三元组必须使用torch.long")
        if positive_triples.ndim != 2 or positive_triples.shape[1] != 3:
            raise ValueError("正三元组形状必须为[N, 3]")
        if int(positive_triples.shape[0]) <= 0:
            raise ValueError("正三元组不能为空")
        if int(negative_sample_count) <= 0:
            raise ValueError("negative_sample_count必须大于0")

        mode = self._normalized_mode(corruption_mode)
        batch_size = int(positive_triples.shape[0])
        sample_count = int(negative_sample_count)
        device = positive_triples.device
        generator = self._generator(device)
        negatives = positive_triples[:, None, :].expand(
            batch_size, sample_count, 3
        ).clone()
        if mode == "head":
            replace_heads = torch.ones(
                batch_size, dtype=torch.bool, device=device
            )
        elif mode == "tail":
            replace_heads = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
        else:
            # 与原强配方一致：每个正样本随机选择一个固定破坏方向。
            replace_heads = torch.randint(
                0,
                2,
                (batch_size,),
                dtype=torch.long,
                device=device,
                generator=generator,
            ).bool()

        candidates = torch.randint(
            0,
            self.num_entities,
            (batch_size, sample_count),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        for _ in range(128):
            negatives[..., 0] = torch.where(
                replace_heads[:, None],
                candidates,
                negatives[..., 0],
            )
            negatives[..., 2] = torch.where(
                replace_heads[:, None],
                negatives[..., 2],
                candidates,
            )
            invalid = self._known_true_mask(negatives)
            if not bool(invalid.any().item()):
                return negatives.reshape(-1, 3)
            # 只重采仍然命中真三元组的位置，不在正样本维度执行Python循环。
            candidates[invalid] = torch.randint(
                0,
                self.num_entities,
                (int(invalid.sum().item()),),
                dtype=torch.long,
                device=device,
                generator=generator,
            )
        raise RuntimeError(
            "128次批量重采样后仍存在真三元组；请检查是否有全实体稠密查询"
        )


class LegacyFilteredNegativeSampler:
    """保留V2逐样本随机序列，用于阶段0严格回归旧训练链路。"""

    def __init__(
        self,
        num_entities: int,
        true_triples: Iterable[IdTriple],
        seed: int,
    ):
        """保存真三元组集合并初始化V2兼容随机数生成器。"""

        if int(num_entities) <= 1:
            raise ValueError("负采样至少需要两个候选实体")
        self.num_entities = int(num_entities)
        self.true_triples: Set[IdTriple] = {
            tuple(int(value) for value in triple)
            for triple in true_triples
        }
        self.rng = np.random.RandomState(int(seed))

    def _sample_one(
        self,
        triple: IdTriple,
        corruption_mode: str,
    ) -> IdTriple:
        """按V2顺序为一条正事实寻找一个严格过滤负样本。"""

        head, relation, tail = triple
        mode = str(corruption_mode).strip().lower()
        if mode == "head_tail":
            replace_head = bool(self.rng.randint(0, 2))
        elif mode == "tail":
            replace_head = False
        else:
            raise ValueError("V2兼容模式只支持head_tail或tail")
        for _ in range(64):
            candidate = int(self.rng.randint(0, self.num_entities))
            negative = (
                (candidate, relation, tail)
                if replace_head
                else (head, relation, candidate)
            )
            if negative != triple and negative not in self.true_triples:
                return negative
        start = int(self.rng.randint(0, self.num_entities))
        for offset in range(self.num_entities):
            candidate = (start + offset) % self.num_entities
            negative = (
                (candidate, relation, tail)
                if replace_head
                else (head, relation, candidate)
            )
            if negative != triple and negative not in self.true_triples:
                return negative
        raise RuntimeError(
            "无法为三元组{}生成filtered负样本".format(triple)
        )

    def sample(
        self,
        positive_triples: torch.Tensor,
        negative_sample_count: int = 1,
        corruption_mode: str = "head_tail",
    ) -> torch.Tensor:
        """按V2随机调用顺序生成扁平的严格过滤负三元组。"""

        if positive_triples.dtype != torch.long:
            raise TypeError("正三元组必须使用torch.long")
        if positive_triples.ndim != 2 or positive_triples.shape[1] != 3:
            raise ValueError("正三元组形状必须为[N, 3]")
        if int(negative_sample_count) <= 0:
            raise ValueError("negative_sample_count必须大于0")
        negatives = [
            self._sample_one(
                tuple(int(value) for value in triple),
                corruption_mode,
            )
            for triple in positive_triples.detach().cpu().tolist()
            for _ in range(int(negative_sample_count))
        ]
        return torch.tensor(
            negatives,
            dtype=torch.long,
            device=positive_triples.device,
        )


# V2复制代码继续使用旧名称；V3强配方显式使用设备端向量化类。
FilteredNegativeSampler = LegacyFilteredNegativeSampler
