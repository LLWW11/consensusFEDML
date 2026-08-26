"""从标准TransE检查点执行头尾、逐关系和逐查询诊断。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

from .data import IdTriple, KnowledgeGraphDataset


def _safe_torch_load(path: Path):
    """兼容不同PyTorch版本，以CPU方式读取可信本地检查点。"""

    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _sha256_file(path: Path) -> str:
    """流式计算文件SHA-256，避免一次读入大型检查点。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_ranks(ranks: Sequence[int]) -> Dict[str, float]:
    """从正整数排名计算MRR、Mean Rank和Hits指标。"""

    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("排名必须是一维非空序列")
    if not np.isfinite(values).all() or bool(np.any(values < 1)):
        raise ValueError("排名必须是有限正数")
    return {
        "mrr": float(np.mean(1.0 / values)),
        "mean_rank": float(np.mean(values)),
        "hits_at_1": float(np.mean(values <= 1.0)),
        "hits_at_3": float(np.mean(values <= 3.0)),
        "hits_at_10": float(np.mean(values <= 10.0)),
        "evaluated_query_count": int(values.size),
    }


@dataclass
class TransEEmbeddingBundle:
    """保存无需重新训练即可执行排名的TransE嵌入表。"""

    entity_embeddings: torch.Tensor
    relation_embeddings: torch.Tensor
    distance_norm: int
    checkpoint_path: Path
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        """校验实体、关系嵌入维数以及距离范数。"""

        if self.entity_embeddings.ndim != 2:
            raise ValueError("实体嵌入必须是二维张量")
        if self.relation_embeddings.ndim != 2:
            raise ValueError("关系嵌入必须是二维张量")
        if int(self.entity_embeddings.shape[1]) != int(
            self.relation_embeddings.shape[1]
        ):
            raise ValueError("实体和关系嵌入维数不一致")
        if int(self.distance_norm) not in {1, 2}:
            raise ValueError("TransE距离范数必须是1或2")

    def to(self, device: torch.device) -> "TransEEmbeddingBundle":
        """返回把两张嵌入表移动到指定设备后的独立对象。"""

        device = torch.device(device)
        return TransEEmbeddingBundle(
            entity_embeddings=self.entity_embeddings.detach().to(device),
            relation_embeddings=self.relation_embeddings.detach().to(device),
            distance_norm=int(self.distance_norm),
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256,
        )


def load_project_checkpoint(
    path_or_directory: Path,
    dataset: KnowledgeGraphDataset,
    distance_norm_override: int = 0,
) -> TransEEmbeddingBundle:
    """加载V2或V3标准检查点并严格核对实体和关系编号。"""

    checkpoint_path = Path(path_or_directory).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "找不到TransE检查点：{}".format(checkpoint_path)
        )
    payload = _safe_torch_load(checkpoint_path)
    if not isinstance(payload, dict):
        raise TypeError("TransE检查点顶层必须是字典")
    required = {
        "model_state_dict",
        "entity_to_id",
        "relation_to_id",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError("TransE检查点缺少字段{}".format(sorted(missing)))
    if dict(payload["entity_to_id"]) != dict(dataset.entity_to_id):
        raise ValueError("检查点实体编号与数据集不一致")
    if dict(payload["relation_to_id"]) != dict(dataset.relation_to_id):
        raise ValueError("检查点关系编号与数据集不一致")

    state = payload["model_state_dict"]
    expected = {
        "entity_embeddings.weight",
        "relation_embeddings.weight",
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("检查点不是标准TransE模型状态")
    entity_embeddings = state[
        "entity_embeddings.weight"
    ].detach().cpu().clone()
    relation_embeddings = state[
        "relation_embeddings.weight"
    ].detach().cpu().clone()
    if int(entity_embeddings.shape[0]) != dataset.num_entities:
        raise ValueError("检查点实体行数与数据集不一致")
    if int(relation_embeddings.shape[0]) != dataset.num_relations:
        raise ValueError("检查点关系行数与数据集不一致")

    distance_norm = int(distance_norm_override)
    config_path = checkpoint_path.parent / "config_snapshot.json"
    if distance_norm <= 0 and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        distance_norm = int(config.get("distance_norm", 0))
    if distance_norm not in {1, 2}:
        raise ValueError(
            "无法确定检查点distance_norm，请使用命令行显式提供"
        )
    return TransEEmbeddingBundle(
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        distance_norm=distance_norm,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256_file(checkpoint_path),
    )


class BatchedDirectionalEvaluator:
    """按查询批次和候选分块计算全局filtered头尾排名。"""

    def __init__(
        self,
        num_entities: int,
        all_true_triples: Iterable[IdTriple],
    ):
        """建立头尾filtered索引并保存候选实体总数。"""

        if int(num_entities) <= 1:
            raise ValueError("评估至少需要两个候选实体")
        self.num_entities = int(num_entities)
        self.true_heads: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(
            set
        )
        self.true_tails: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(
            set
        )
        for head, relation, tail in all_true_triples:
            self.true_heads[(int(relation), int(tail))].add(int(head))
            self.true_tails[(int(head), int(relation))].add(int(tail))

    @staticmethod
    def _target_distances(
        bundle: TransEEmbeddingBundle,
        triples: torch.Tensor,
    ) -> torch.Tensor:
        """计算一批目标三元组自身的TransE距离。"""

        heads = bundle.entity_embeddings.index_select(0, triples[:, 0])
        relations = bundle.relation_embeddings.index_select(
            0, triples[:, 1]
        )
        tails = bundle.entity_embeddings.index_select(0, triples[:, 2])
        return torch.linalg.vector_norm(
            heads + relations - tails,
            ord=int(bundle.distance_norm),
            dim=1,
        )

    @staticmethod
    def _candidate_distances(
        bundle: TransEEmbeddingBundle,
        triples: torch.Tensor,
        candidate_ids: torch.Tensor,
        predict_head: bool,
    ) -> torch.Tensor:
        """计算查询批次相对一块候选实体的距离矩阵。"""

        candidates = bundle.entity_embeddings.index_select(
            0, candidate_ids
        )
        relations = bundle.relation_embeddings.index_select(
            0, triples[:, 1]
        )
        if bool(predict_head):
            tails = bundle.entity_embeddings.index_select(
                0, triples[:, 2]
            )
            residual = (
                candidates.unsqueeze(0)
                + relations.unsqueeze(1)
                - tails.unsqueeze(1)
            )
        else:
            heads = bundle.entity_embeddings.index_select(
                0, triples[:, 0]
            )
            residual = (
                heads.unsqueeze(1)
                + relations.unsqueeze(1)
                - candidates.unsqueeze(0)
            )
        return torch.linalg.vector_norm(
            residual,
            ord=int(bundle.distance_norm),
            dim=2,
        )

    def _invalid_mask(
        self,
        triples: torch.Tensor,
        predict_head: bool,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """生成一个候选分块的已知真实答案屏蔽矩阵。"""

        invalid = torch.zeros(
            (int(triples.shape[0]), int(stop - start)),
            dtype=torch.bool,
            device=triples.device,
        )
        for row_index, row in enumerate(
            triples.detach().cpu().tolist()
        ):
            head, relation, tail = (int(value) for value in row)
            target = head if predict_head else tail
            filtered = (
                self.true_heads[(relation, tail)]
                if predict_head
                else self.true_tails[(head, relation)]
            )
            local_ids = [
                int(entity_id) - int(start)
                for entity_id in filtered
                if int(start) <= int(entity_id) < int(stop)
                and int(entity_id) != int(target)
            ]
            if local_ids:
                invalid[row_index, local_ids] = True
        return invalid

    def evaluate_direction(
        self,
        bundle: TransEEmbeddingBundle,
        triples: Sequence[IdTriple],
        predict_head: bool,
        query_batch_size: int,
        candidate_batch_size: int,
        progress_every: int = 0,
    ) -> np.ndarray:
        """使用乐观并列规则计算一个方向的全局filtered排名。"""

        if not triples:
            raise ValueError("评估三元组不能为空")
        if int(query_batch_size) <= 0:
            raise ValueError("query_batch_size必须大于0")
        if int(candidate_batch_size) <= 0:
            raise ValueError("candidate_batch_size必须大于0")
        device = bundle.entity_embeddings.device
        triple_tensor = torch.tensor(
            list(triples),
            dtype=torch.long,
            device=device,
        )
        rank_chunks: List[torch.Tensor] = []
        with torch.inference_mode():
            for query_start in range(
                0,
                int(triple_tensor.shape[0]),
                int(query_batch_size),
            ):
                query_stop = min(
                    query_start + int(query_batch_size),
                    int(triple_tensor.shape[0]),
                )
                query = triple_tensor[query_start:query_stop]
                targets = self._target_distances(bundle, query)
                ranks = torch.ones(
                    int(query.shape[0]),
                    dtype=torch.long,
                    device=device,
                )
                for candidate_start in range(
                    0,
                    self.num_entities,
                    int(candidate_batch_size),
                ):
                    candidate_stop = min(
                        candidate_start + int(candidate_batch_size),
                        self.num_entities,
                    )
                    candidate_ids = torch.arange(
                        candidate_start,
                        candidate_stop,
                        dtype=torch.long,
                        device=device,
                    )
                    distances = self._candidate_distances(
                        bundle,
                        query,
                        candidate_ids,
                        bool(predict_head),
                    )
                    invalid = self._invalid_mask(
                        query,
                        bool(predict_head),
                        candidate_start,
                        candidate_stop,
                    )
                    ranks += (
                        (distances < targets.view(-1, 1)) & ~invalid
                    ).sum(dim=1)
                rank_chunks.append(ranks.detach().cpu())
                if (
                    int(progress_every) > 0
                    and (
                        query_stop % int(progress_every) == 0
                        or query_stop == int(triple_tensor.shape[0])
                    )
                ):
                    print(
                        "{}预测：已完成 {}/{}".format(
                            "头" if predict_head else "尾",
                            query_stop,
                            int(triple_tensor.shape[0]),
                        ),
                        flush=True,
                    )
        return torch.cat(rank_chunks).numpy().astype(np.int64)


def select_test_triples(
    dataset: KnowledgeGraphDataset,
    maximum: int,
    seed: int,
) -> Tuple[IdTriple, ...]:
    """按固定种子选择测试事实，0表示使用完整官方测试集。"""

    triples = tuple(
        tuple(int(value) for value in row)
        for row in dataset.test_triples.tolist()
    )
    if int(maximum) <= 0 or int(maximum) >= len(triples):
        return triples
    rng = np.random.RandomState(int(seed))
    indices = np.sort(
        rng.choice(len(triples), size=int(maximum), replace=False)
    )
    return tuple(triples[int(index)] for index in indices)


def _relation_rows(
    records: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """把逐查询排名按关系和方向汇总为审计表。"""

    buckets: DefaultDict[Tuple[int, str], List[int]] = defaultdict(list)
    for record in records:
        relation_id = int(record["relation_id"])
        direction = str(record["direction"])
        rank = int(record["rank"])
        buckets[(relation_id, direction)].append(rank)
        buckets[(relation_id, "combined")].append(rank)
    rows: List[Dict[str, object]] = []
    for relation_id, direction in sorted(buckets):
        rows.append(
            {
                "relation_id": relation_id,
                "direction": direction,
                **metrics_from_ranks(
                    buckets[(relation_id, direction)]
                ),
            }
        )
    return rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """把同构字典列表写为带BOM的UTF-8 CSV。"""

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with Path(path).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_directional_diagnostic(
    dataset: KnowledgeGraphDataset,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    max_triples: int,
    selection_seed: int,
    query_batch_size: int,
    candidate_batch_size: int,
    progress_every: int,
    distance_norm_override: int = 0,
) -> Dict[str, object]:
    """执行一次只读头尾诊断并写出摘要、明细和简体中文报告。"""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    triples = select_test_triples(
        dataset,
        int(max_triples),
        int(selection_seed),
    )
    bundle = load_project_checkpoint(
        checkpoint,
        dataset,
        distance_norm_override=distance_norm_override,
    ).to(device)
    evaluator = BatchedDirectionalEvaluator(
        dataset.num_entities,
        dataset.all_true_triples,
    )
    head_ranks = evaluator.evaluate_direction(
        bundle,
        triples,
        predict_head=True,
        query_batch_size=query_batch_size,
        candidate_batch_size=candidate_batch_size,
        progress_every=progress_every,
    )
    tail_ranks = evaluator.evaluate_direction(
        bundle,
        triples,
        predict_head=False,
        query_batch_size=query_batch_size,
        candidate_batch_size=candidate_batch_size,
        progress_every=progress_every,
    )
    combined_ranks = np.concatenate([head_ranks, tail_ranks])
    records: List[Dict[str, object]] = []
    for triple_index, (head, relation, tail) in enumerate(triples):
        for direction, ranks in (
            ("head", head_ranks),
            ("tail", tail_ranks),
        ):
            rank = int(ranks[triple_index])
            records.append(
                {
                    "triple_index": int(triple_index),
                    "head_id": int(head),
                    "relation_id": int(relation),
                    "tail_id": int(tail),
                    "direction": direction,
                    "rank": rank,
                    "reciprocal_rank": 1.0 / float(rank),
                    "hit_at_1": int(rank <= 1),
                    "hit_at_3": int(rank <= 3),
                    "hit_at_10": int(rank <= 10),
                }
            )
    summary = {
        "status": "completed",
        "training_performed": False,
        "checkpoint_path": str(bundle.checkpoint_path),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "device": str(device),
        "selected_triple_count": len(triples),
        "official_test_triple_count": int(
            dataset.test_triples.shape[0]
        ),
        "full_official_test": len(triples)
        == int(dataset.test_triples.shape[0]),
        "head_metrics": metrics_from_ranks(head_ranks),
        "tail_metrics": metrics_from_ranks(tail_ranks),
        "combined_metrics": metrics_from_ranks(combined_ranks),
    }
    with (output_dir / "directional_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_csv(output_dir / "query_ranks.csv", records)
    _write_csv(
        output_dir / "relation_metrics.csv",
        _relation_rows(records),
    )
    report = "\n".join(
        [
            "# TransE头尾方向诊断",
            "",
            "本次诊断没有重新训练模型。",
            "",
            "- 是否完整官方测试：`{}`".format(
                (
                    "是"
                    if summary["full_official_test"]
                    else "否"
                )
            ),
            "- 评估三元组数：`{}`".format(len(triples)),
            "- 头预测MRR：`{:.6f}`".format(
                float(summary["head_metrics"]["mrr"])
            ),
            "- 尾预测MRR：`{:.6f}`".format(
                float(summary["tail_metrics"]["mrr"])
            ),
            "- 综合MRR：`{:.6f}`".format(
                float(summary["combined_metrics"]["mrr"])
            ),
            "",
        ]
    )
    (output_dir / "方向诊断报告.md").write_text(
        report,
        encoding="utf-8",
    )
    return summary


