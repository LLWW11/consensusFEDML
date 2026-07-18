"""固定多图探针共识指标的纯 NumPy 实现。"""

import numpy as np


def validate_probability_tensor(probabilities, name="概率张量"):
    """校验 [模型, 探针, 类别] 概率的维度、有限性、范围与归一化。"""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("{} 必须是 [模型, 探针, 类别] 三维数组。".format(name))
    if values.shape[2] < 2:
        raise ValueError("{} 的类别维度必须至少为 2。".format(name))
    if not np.all(np.isfinite(values)):
        raise ValueError("{} 包含 NaN 或无穷值。".format(name))
    tolerance = 1e-5
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("{} 包含超出 [0, 1] 范围的概率。".format(name))
    probability_sums = np.sum(values, axis=2)
    if not np.allclose(probability_sums, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("{} 中至少一个概率向量的元素和不为 1。".format(name))
    return np.clip(values, 0.0, 1.0)


def _empty_population_summary(sample_count):
    """为不足两个模型的口径创建全部为空值的共识结果。"""
    empty = np.full(int(sample_count), np.nan, dtype=np.float64)
    return {
        "agreement_by_sample": empty.copy(),
        "certainty_by_sample": empty.copy(),
        "effective_by_sample": empty.copy(),
        "correct_effective_by_sample": empty.copy(),
        "wrong_effective_by_sample": empty.copy(),
        "majority_labels": np.full(int(sample_count), -1, dtype=np.int64),
        "agreement_mean": np.nan,
        "certainty_mean": np.nan,
        "effective_mean": np.nan,
        "correct_effective_mean": np.nan,
        "wrong_effective_mean": np.nan,
        "effective_q25": np.nan,
        "effective_q50": np.nan,
        "effective_q75": np.nan,
    }


def calculate_population_probe_metrics(probabilities, true_labels=None):
    """逐图计算 A、C、S，再对探针取均值并按多数投票拆分正确与错误共识。

    ``probabilities`` 形状为 [模型, 探针, 类别]。确定性定义为各模型
    自身归一化熵的平均补数；模型间分歧只由广义 JSD 对应的一致性惩罚。
    没有真实标签时仍计算纯共识，正确和错误共识返回空值。
    """
    raw_values = np.asarray(probabilities)
    if raw_values.ndim != 3:
        raise ValueError("probabilities 必须是 [模型, 探针, 类别] 三维数组。")
    sample_count = int(raw_values.shape[1])
    if true_labels is None:
        labels = np.full(sample_count, -1, dtype=np.int64)
    else:
        labels = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    if sample_count != labels.shape[0]:
        raise ValueError("概率探针数与真实标签数不一致。")
    if raw_values.shape[0] < 2:
        return _empty_population_summary(sample_count)

    values = validate_probability_tensor(raw_values, "群体探针概率")
    class_count = int(values.shape[2])
    labels_available = bool(np.all((labels >= 0) & (labels < class_count)))
    if np.any(labels >= class_count):
        raise ValueError("真实标签超出概率类别范围。")

    epsilon = np.finfo(np.float64).tiny
    mean_probabilities = np.mean(values, axis=0)
    safe_values = np.clip(values, epsilon, 1.0)
    safe_mean = np.clip(mean_probabilities, epsilon, 1.0)
    per_model_entropy = -np.sum(safe_values * np.log(safe_values), axis=2)
    mean_model_entropy = np.mean(per_model_entropy, axis=0)
    mean_distribution_entropy = -np.sum(safe_mean * np.log(safe_mean), axis=1)
    log_class_count = np.log(float(class_count))
    normalized_jsd = (
        mean_distribution_entropy - mean_model_entropy
    ) / log_class_count
    agreement = np.clip(1.0 - normalized_jsd, 0.0, 1.0)
    certainty = np.clip(1.0 - mean_model_entropy / log_class_count, 0.0, 1.0)
    effective = agreement * certainty

    predicted_labels = np.argmax(values, axis=2)
    majority_labels = np.empty(sample_count, dtype=np.int64)
    for sample_index in range(sample_count):
        vote_counts = np.bincount(
            predicted_labels[:, sample_index], minlength=class_count
        )
        # 平票固定选择最小类别编号，保证跨平台和跨方案结果稳定。
        majority_labels[sample_index] = int(np.argmax(vote_counts))
    if labels_available:
        correct_mask = majority_labels == labels
        correct_effective = effective * correct_mask.astype(np.float64)
        wrong_effective = effective * (~correct_mask).astype(np.float64)
        correct_mean = float(np.mean(correct_effective))
        wrong_mean = float(np.mean(wrong_effective))
    else:
        correct_effective = np.full(sample_count, np.nan, dtype=np.float64)
        wrong_effective = np.full(sample_count, np.nan, dtype=np.float64)
        correct_mean = np.nan
        wrong_mean = np.nan

    return {
        "agreement_by_sample": agreement,
        "certainty_by_sample": certainty,
        "effective_by_sample": effective,
        "correct_effective_by_sample": correct_effective,
        "wrong_effective_by_sample": wrong_effective,
        "majority_labels": majority_labels,
        "agreement_mean": float(np.mean(agreement)),
        "certainty_mean": float(np.mean(certainty)),
        "effective_mean": float(np.mean(effective)),
        "correct_effective_mean": correct_mean,
        "wrong_effective_mean": wrong_mean,
        "effective_q25": float(np.quantile(effective, 0.25)),
        "effective_q50": float(np.quantile(effective, 0.50)),
        "effective_q75": float(np.quantile(effective, 0.75)),
    }
