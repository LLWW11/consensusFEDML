"""知识图谱补全的filtered MRR、Mean Rank和Hits@K评估。"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch

from .data import IdTriple, KnowledgeGraphDataset


class FilteredRankingEvaluator:
    """对头实体和尾实体替换查询执行filtered排名评估。"""

    def __init__(self, dataset: KnowledgeGraphDataset):
        """根据全部已知真三元组建立头尾过滤索引。"""

        self.dataset = dataset
        self.true_tails: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(
            set
        )
        self.true_heads: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(
            set
        )
        for head, relation, tail in dataset.all_true_triples:
            self.true_tails[(head, relation)].add(tail)
            self.true_heads[(relation, tail)].add(head)

    @staticmethod
    def _select_triples(
        triples: torch.Tensor, max_triples: int, seed: int
    ) -> torch.Tensor:
        """按固定种子选择有限评估子集，0表示使用全部三元组。"""

        count = int(triples.shape[0])
        maximum = int(max_triples)
        if maximum <= 0 or maximum >= count:
            return triples
        rng = np.random.RandomState(int(seed))
        indices = np.sort(rng.choice(count, size=maximum, replace=False))
        return triples.index_select(
            0, torch.tensor(indices, dtype=torch.long)
        )

    def _candidate_scores(
        self,
        model: torch.nn.Module,
        triple: IdTriple,
        predict_head: bool,
        device: torch.device,
        candidate_batch_size: int,
    ) -> torch.Tensor:
        """分块计算一个查询替换全部实体后的CPU距离向量。"""

        head, relation, tail = triple
        score_chunks: List[torch.Tensor] = []
        for start in range(
            0, self.dataset.num_entities, int(candidate_batch_size)
        ):
            stop = min(
                start + int(candidate_batch_size),
                self.dataset.num_entities,
            )
            candidates = torch.arange(
                start, stop, dtype=torch.long, device=device
            )
            query = torch.empty(
                (stop - start, 3), dtype=torch.long, device=device
            )
            query[:, 0] = candidates if predict_head else int(head)
            query[:, 1] = int(relation)
            query[:, 2] = int(tail) if predict_head else candidates
            score_chunks.append(model.score_triples(query).detach().cpu())
        return torch.cat(score_chunks, dim=0)

    def rank_query(
        self,
        model: torch.nn.Module,
        triple: IdTriple,
        predict_head: bool,
        device: torch.device,
        candidate_batch_size: int = 4096,
    ) -> int:
        """计算单个头或尾预测查询的filtered乐观排名。"""

        if int(candidate_batch_size) <= 0:
            raise ValueError("candidate_batch_size必须大于0")
        head, relation, tail = (
            int(triple[0]),
            int(triple[1]),
            int(triple[2]),
        )
        scores = self._candidate_scores(
            model,
            (head, relation, tail),
            bool(predict_head),
            torch.device(device),
            int(candidate_batch_size),
        )
        target_id = head if predict_head else tail
        target_score = float(scores[target_id].item())
        filtered_ids = (
            self.true_heads[(relation, tail)]
            if predict_head
            else self.true_tails[(head, relation)]
        )
        for entity_id in filtered_ids:
            if int(entity_id) != target_id:
                scores[int(entity_id)] = float("inf")
        # 距离越小越优；相同距离采用乐观排名，避免浮点相等顺序影响结果。
        return 1 + int((scores < target_score).sum().item())

    def evaluate(
        self,
        model: torch.nn.Module,
        triples: torch.Tensor,
        device: torch.device,
        max_triples: int = 0,
        seed: int = 0,
        candidate_batch_size: int = 4096,
    ) -> Dict[str, float]:
        """在头尾两个方向计算filtered排名指标。"""

        selected = self._select_triples(triples, max_triples, seed)
        if int(selected.shape[0]) <= 0:
            raise ValueError("评估三元组不能为空")
        device = torch.device(device)
        model.to(device)
        model.eval()
        ranks: List[int] = []
        with torch.no_grad():
            for row in selected.tolist():
                triple = tuple(int(value) for value in row)
                ranks.append(
                    self.rank_query(
                        model,
                        triple,
                        True,
                        device,
                        candidate_batch_size,
                    )
                )
                ranks.append(
                    self.rank_query(
                        model,
                        triple,
                        False,
                        device,
                        candidate_batch_size,
                    )
                )
        rank_array = np.asarray(ranks, dtype=np.float64)
        return {
            "mrr": float(np.mean(1.0 / rank_array)),
            "mean_rank": float(np.mean(rank_array)),
            "hits_at_1": float(np.mean(rank_array <= 1.0)),
            "hits_at_3": float(np.mean(rank_array <= 3.0)),
            "hits_at_10": float(np.mean(rank_array <= 10.0)),
            "evaluated_triple_count": float(selected.shape[0]),
            "evaluated_query_count": float(rank_array.size),
        }
