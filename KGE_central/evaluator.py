"""知识图谱补全的filtered MRR、Mean Rank和Hits@K评估。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
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
        triples: torch.Tensor,
        max_triples: int,
        seed: int,
        relation_stratified: bool = False,
    ) -> torch.Tensor:
        """按固定种子选择普通或关系分层子集，0表示使用全部。"""

        count = int(triples.shape[0])
        maximum = int(max_triples)
        if maximum <= 0 or maximum >= count:
            return triples
        rng = np.random.RandomState(int(seed))
        if not bool(relation_stratified):
            indices = np.sort(
                rng.choice(count, size=maximum, replace=False)
            )
        else:
            relation_to_indices: DefaultDict[int, List[int]] = defaultdict(
                list
            )
            for row_index, relation in enumerate(
                triples[:, 1].detach().cpu().tolist()
            ):
                relation_to_indices[int(relation)].append(
                    int(row_index)
                )
            if maximum < len(relation_to_indices):
                raise ValueError(
                    "关系分层子集上限{}小于关系数{}".format(
                        maximum,
                        len(relation_to_indices),
                    )
                )
            selected = [
                int(rng.choice(indices_for_relation))
                for _, indices_for_relation in sorted(
                    relation_to_indices.items()
                )
            ]
            selected_set = set(selected)
            remaining_candidates = np.asarray(
                [
                    index
                    for index in range(count)
                    if index not in selected_set
                ],
                dtype=np.int64,
            )
            remaining_count = maximum - len(selected)
            if remaining_count > 0:
                selected.extend(
                    int(value)
                    for value in rng.choice(
                        remaining_candidates,
                        size=remaining_count,
                        replace=False,
                    )
                )
            indices = np.asarray(sorted(selected), dtype=np.int64)
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
        relation_stratified: bool = False,
        query_batch_size: int = 1,
    ) -> Dict[str, float]:
        """按串行或批量精确路径计算头尾filtered排名指标。"""

        selected = self._select_triples(
            triples,
            max_triples,
            seed,
            relation_stratified=relation_stratified,
        )
        if int(selected.shape[0]) <= 0:
            raise ValueError("评估三元组不能为空")
        if int(query_batch_size) <= 0:
            raise ValueError("query_batch_size必须大于0")
        device = torch.device(device)
        model.to(device)
        model.eval()
        if int(query_batch_size) > 1:
            return self._evaluate_batched(
                model,
                selected,
                device,
                int(query_batch_size),
                int(candidate_batch_size),
            )
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

    def _evaluate_batched(
        self,
        model: torch.nn.Module,
        selected: torch.Tensor,
        device: torch.device,
        query_batch_size: int,
        candidate_batch_size: int,
    ) -> Dict[str, float]:
        """复用已验证的方向诊断内核，批量计算与串行定义相同的精确排名。"""

        # 延迟导入避免方向诊断模块和常规评估模块在加载时形成循环依赖。
        from .directional import (
            BatchedDirectionalEvaluator,
            TransEEmbeddingBundle,
        )

        bundle = TransEEmbeddingBundle(
            entity_embeddings=model.entity_embeddings.weight,
            relation_embeddings=model.relation_embeddings.weight,
            distance_norm=int(model.distance_norm),
            checkpoint_path=Path("<训练中模型>"),
            checkpoint_sha256="",
        )
        triples = tuple(
            tuple(int(value) for value in row)
            for row in selected.detach().cpu().tolist()
        )
        evaluator = BatchedDirectionalEvaluator(
            self.dataset.num_entities,
            self.dataset.all_true_triples,
        )
        head_ranks = evaluator.evaluate_direction(
            bundle,
            triples,
            predict_head=True,
            query_batch_size=query_batch_size,
            candidate_batch_size=candidate_batch_size,
        )
        tail_ranks = evaluator.evaluate_direction(
            bundle,
            triples,
            predict_head=False,
            query_batch_size=query_batch_size,
            candidate_batch_size=candidate_batch_size,
        )
        rank_array = np.concatenate(
            [head_ranks, tail_ranks]
        ).astype(np.float64)
        return {
            "mrr": float(np.mean(1.0 / rank_array)),
            "mean_rank": float(np.mean(rank_array)),
            "hits_at_1": float(np.mean(rank_array <= 1.0)),
            "hits_at_3": float(np.mean(rank_array <= 3.0)),
            "hits_at_10": float(np.mean(rank_array <= 10.0)),
            "evaluated_triple_count": float(selected.shape[0]),
            "evaluated_query_count": float(rank_array.size),
        }


