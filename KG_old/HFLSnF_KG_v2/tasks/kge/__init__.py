"""集中式与联邦知识图谱嵌入任务的公共组件。"""

from .data import (
    KnowledgeGraphDataset,
    build_knowledge_graph_dataset,
    build_synthetic_knowledge_graph,
    load_fb15k237,
)
from .ablation import (
    ABLATION_SUITE_NAME,
    THREE_ARM_SPECS,
    compare_three_arm_results,
    validate_three_arm_configs,
    write_comparison_outputs,
)
from .evaluator import FilteredRankingEvaluator
from .directional_diagnostics import (
    build_pairwise_query_outcomes,
    build_relation_metrics,
    run_directional_diagnostics,
    select_test_triples,
    validate_checkpoint_fingerprints,
    write_directional_outputs,
)
from .evaluation_bridge import (
    BatchedFilteredTransEEvaluator,
    CommonHoldout,
    FedEClientPartition,
    FedEDataBundle,
    TransEEmbeddingBundle,
    bootstrap_mrr_interval,
    bootstrap_paired_delta_interval,
    build_common_holdout,
    build_fede_data_bundle,
    evaluate_fede_original_protocol,
    evaluate_global_protocol,
    hash_assignments,
    hash_json,
    hash_triples,
    load_fede_data_bundle,
    load_fede_embedding_bundle,
    load_project_embedding_bundle,
    metrics_from_ranks,
)
from .federated_data import (
    FederatedKnowledgeGraphData,
    KnowledgeGraphClientPartition,
    partition_train_triples_by_head,
)
from .fixed_topology import (
    COMPARISON_SCENARIOS,
    FixedParticipantTopology,
    build_fixed_participant_topology,
)
from .factorial_ablation import (
    D_ARM_SPEC,
    FOUR_ARM_SPECS,
    compare_four_arm_results,
    validate_four_arm_configs,
    write_factorial_outputs,
)
from .model import TransE
from .negative_sampling import FilteredNegativeSampler
from .trainer import CentralizedTransETrainer

__all__ = [
    "ABLATION_SUITE_NAME",
    "CentralizedTransETrainer",
    "BatchedFilteredTransEEvaluator",
    "CommonHoldout",
    "FedEClientPartition",
    "FedEDataBundle",
    "FilteredNegativeSampler",
    "FilteredRankingEvaluator",
    "FixedParticipantTopology",
    "FederatedKnowledgeGraphData",
    "KnowledgeGraphDataset",
    "KnowledgeGraphClientPartition",
    "TransE",
    "TransEEmbeddingBundle",
    "THREE_ARM_SPECS",
    "bootstrap_mrr_interval",
    "bootstrap_paired_delta_interval",
    "build_common_holdout",
    "build_fede_data_bundle",
    "COMPARISON_SCENARIOS",
    "D_ARM_SPEC",
    "FOUR_ARM_SPECS",
    "build_fixed_participant_topology",
    "build_pairwise_query_outcomes",
    "build_relation_metrics",
    "build_knowledge_graph_dataset",
    "build_synthetic_knowledge_graph",
    "compare_three_arm_results",
    "compare_four_arm_results",
    "evaluate_fede_original_protocol",
    "evaluate_global_protocol",
    "hash_assignments",
    "hash_json",
    "hash_triples",
    "load_fb15k237",
    "load_fede_data_bundle",
    "load_fede_embedding_bundle",
    "load_project_embedding_bundle",
    "metrics_from_ranks",
    "partition_train_triples_by_head",
    "run_directional_diagnostics",
    "select_test_triples",
    "validate_checkpoint_fingerprints",
    "validate_four_arm_configs",
    "validate_three_arm_configs",
    "write_comparison_outputs",
    "write_directional_outputs",
    "write_factorial_outputs",
]
