"""固定人数四组对照使用的知识图谱嵌入组件。"""

from .data import (
    KnowledgeGraphDataset,
    build_knowledge_graph_dataset,
    build_synthetic_knowledge_graph,
    load_fb15k237,
)
from .directional import (
    BatchedDirectionalEvaluator,
    TransEEmbeddingBundle,
    load_project_checkpoint,
    metrics_from_ranks,
    run_directional_diagnostic,
    select_test_triples,
)
from .evaluator import FilteredRankingEvaluator
from .federated_data import (
    BALANCED_HEAD_ENTITY,
    BALANCED_HEAD_ENTITY_OVERLAP_TARGET,
    FederatedKnowledgeGraphData,
    KnowledgeGraphClientPartition,
    calibrate_entity_overlap_levels,
    partition_train_triples_by_head,
    partition_train_triples_by_overlap_target,
)
from .model import TransE
from .negative_sampling import (
    FilteredNegativeSampler,
    VectorizedFilteredNegativeSampler,
)
from .official_evaluation import (
    build_official_evaluation_contract,
    run_best_checkpoint_official_evaluation,
    write_official_evaluation_report,
)
from .objectives import self_adversarial_loss
from .subsampling import TripleFrequencySubsampler

__all__ = [
    "BatchedDirectionalEvaluator",
    "BALANCED_HEAD_ENTITY",
    "BALANCED_HEAD_ENTITY_OVERLAP_TARGET",
    "FederatedKnowledgeGraphData",
    "FilteredNegativeSampler",
    "FilteredRankingEvaluator",
    "KnowledgeGraphClientPartition",
    "KnowledgeGraphDataset",
    "TransE",
    "TransEEmbeddingBundle",
    "TripleFrequencySubsampler",
    "VectorizedFilteredNegativeSampler",
    "build_knowledge_graph_dataset",
    "build_official_evaluation_contract",
    "build_synthetic_knowledge_graph",
    "calibrate_entity_overlap_levels",
    "load_fb15k237",
    "load_project_checkpoint",
    "metrics_from_ranks",
    "partition_train_triples_by_head",
    "partition_train_triples_by_overlap_target",
    "run_best_checkpoint_official_evaluation",
    "run_directional_diagnostic",
    "select_test_triples",
    "self_adversarial_loss",
    "write_official_evaluation_report",
]
