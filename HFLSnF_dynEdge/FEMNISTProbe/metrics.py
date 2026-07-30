"""计算 FEMNIST 三层固定探针的训练期摘要指标。"""

from __future__ import absolute_import

import numpy as np

from probe_metrics import calculate_population_probe_metrics


SUMMARY_COLUMNS = [
    "global_epoch",
    "topology_cycle_index",
    "mat_topology_index",
    "active_client_count",
    "active_coverage",
    "candidate_agreement",
    "candidate_certainty",
    "candidate_effective",
    "candidate_correct_effective",
    "candidate_wrong_effective",
    "active_agreement",
    "active_certainty",
    "active_effective",
    "active_correct_effective",
    "active_wrong_effective",
    "coverage_weighted_active_correct_effective",
    "within_edge_effective",
    "edge_effective",
    "edge_correct_effective",
    "edge_wrong_effective",
    "edge_cloud_effective",
    "cloud_probe_accuracy",
]


def _safe_weighted_mean(values, weights):
    """忽略非有限值并计算非负权重的加权均值。"""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def summarize_probe_observation(
        global_epoch,
        topology_cycle_index,
        mat_topology_index,
        client_probabilities,
        edge_probabilities,
        cloud_probabilities,
        active_client_mask,
        edge_active_mask,
        true_labels,
        groups,
):
    """计算候选、活跃、组内、边缘和云端的统一摘要。"""
    clients = np.asarray(client_probabilities, dtype=np.float32)
    edges = np.asarray(edge_probabilities, dtype=np.float32)
    cloud = np.asarray(cloud_probabilities, dtype=np.float32)
    active_mask = np.asarray(active_client_mask, dtype=np.bool_)
    edge_mask = np.asarray(edge_active_mask, dtype=np.bool_)
    labels = np.asarray(true_labels, dtype=np.int64)
    candidate = calculate_population_probe_metrics(clients, labels)
    active = calculate_population_probe_metrics(clients[active_mask], labels)
    valid_edges = edges[edge_mask]
    edge = calculate_population_probe_metrics(valid_edges, labels)

    within_values = []
    within_weights = []
    for client_slots in groups.values():
        slots = np.asarray(client_slots, dtype=np.int64)
        group_summary = calculate_population_probe_metrics(
            clients[slots], labels
        )
        within_values.append(group_summary["effective_mean"])
        within_weights.append(int(slots.size))
    within_edge = _safe_weighted_mean(within_values, within_weights)

    if valid_edges.shape[0] >= 1:
        edge_cloud_values = np.concatenate(
            [valid_edges, cloud[None, :, :]], axis=0
        )
        edge_cloud = calculate_population_probe_metrics(
            edge_cloud_values, labels
        )["effective_mean"]
    else:
        edge_cloud = np.nan

    active_count = int(np.sum(active_mask))
    coverage = float(active_count) / float(clients.shape[0])
    active_correct = active["correct_effective_mean"]
    coverage_weighted = (
        coverage * float(active_correct)
        if np.isfinite(active_correct)
        else 0.0
    )
    cloud_accuracy = float(
        np.mean(np.argmax(cloud, axis=1) == labels)
    )
    return {
        "global_epoch": int(global_epoch),
        "topology_cycle_index": int(topology_cycle_index),
        "mat_topology_index": int(mat_topology_index),
        "active_client_count": active_count,
        "active_coverage": coverage,
        "candidate_agreement": candidate["agreement_mean"],
        "candidate_certainty": candidate["certainty_mean"],
        "candidate_effective": candidate["effective_mean"],
        "candidate_correct_effective": candidate["correct_effective_mean"],
        "candidate_wrong_effective": candidate["wrong_effective_mean"],
        "active_agreement": active["agreement_mean"],
        "active_certainty": active["certainty_mean"],
        "active_effective": active["effective_mean"],
        "active_correct_effective": active_correct,
        "active_wrong_effective": active["wrong_effective_mean"],
        "coverage_weighted_active_correct_effective": coverage_weighted,
        "within_edge_effective": within_edge,
        "edge_effective": edge["effective_mean"],
        "edge_correct_effective": edge["correct_effective_mean"],
        "edge_wrong_effective": edge["wrong_effective_mean"],
        "edge_cloud_effective": edge_cloud,
        "cloud_probe_accuracy": cloud_accuracy,
    }
