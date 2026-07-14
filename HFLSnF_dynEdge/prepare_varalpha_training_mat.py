"""把 Topo_opt 的 varAlpha 后处理结果转换为可直接训练的 MATLAB 拓扑文件。"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


CLIENT_FIELDS = (
    "client_num_HFLSnF_fix",
    "client_num_HFLSnF_los",
    "client_num_HFLnoSnF_fix",
    "client_num_HFLnoSnF_los",
    "client_num_FLSnF",
    "client_num_FLnoSnF",
)


def _flatten_integer_values(value):
    """递归展开 MATLAB cell 或数值数组，并返回 Python 整数列表。"""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        values = []
        for item in value.reshape(-1):
            values.extend(_flatten_integer_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten_integer_values(item))
        return values
    numeric_value = float(value)
    integer_value = int(numeric_value)
    if not np.isfinite(numeric_value) or numeric_value != integer_value:
        raise ValueError("MATLAB 映射包含非有限或非整数值：{!r}".format(value))
    return [integer_value]


def _extract_policy(container, policy_index):
    """从一个 MATLAB 三策略 cell 中提取指定的零基策略项。"""
    items = list(np.asarray(container, dtype=object).reshape(-1))
    if policy_index >= len(items):
        raise ValueError("策略 cell 缺少索引 {}".format(policy_index))
    return items[policy_index]


def _extract_edges(edge_container, policy_index):
    """从边缘策略 cell 中提取边缘节点 ID。"""
    return _flatten_integer_values(_extract_policy(edge_container, policy_index))


def _extract_groups(mapping_container, policy_index, edge_count):
    """按边缘槽位边界提取 HFL 每组的物理客户端 ID。"""
    policy_mapping = _extract_policy(mapping_container, policy_index)
    if edge_count == 0:
        return []
    array = np.asarray(policy_mapping)
    if array.dtype == object:
        items = list(array.reshape(-1))
        if len(items) != edge_count:
            raise ValueError(
                "HFL 映射有 {} 个组槽位，但边缘集合有 {} 个".format(
                    len(items), edge_count
                )
            )
        return [_flatten_integer_values(item) for item in items]
    if edge_count == 1:
        return [_flatten_integer_values(policy_mapping)]
    raise ValueError("非 cell HFL 映射无法对应 {} 个边缘槽位".format(edge_count))


def _target_count(data, field_name, round_index, util_index):
    """读取一个 varctrl 客户端数量，并校验其为合法整数。"""
    controlled_field = field_name + "_varctrl"
    if controlled_field not in data:
        raise KeyError("输入 MAT 缺少字段 {}".format(controlled_field))
    value = float(np.asarray(data[controlled_field])[round_index, util_index])
    target = int(value)
    if not np.isfinite(value) or value != target or target < 0:
        raise ValueError("{} 包含非法目标人数 {}".format(controlled_field, value))
    return target


def _allocate_group_counts(original_sizes, target_total):
    """在尽量保留原有效组的前提下分配目标组内人数。"""
    original_sizes = np.asarray(original_sizes, dtype=int)
    slot_count = original_sizes.size
    if target_total == 0:
        return np.zeros(slot_count, dtype=int)
    if slot_count == 0:
        raise ValueError("目标人数大于 0，但没有可用边缘槽位")

    eligible = np.flatnonzero(original_sizes > 0)
    if eligible.size == 0:
        eligible = np.arange(slot_count)
    counts = np.zeros(slot_count, dtype=int)

    # 优先让原有大组继续有效；目标人数不足时只保留最大的若干组。
    ranked_eligible = sorted(
        eligible.tolist(), key=lambda index: (-original_sizes[index], index)
    )
    initial_group_count = min(target_total, len(ranked_eligible))
    for index in ranked_eligible[:initial_group_count]:
        counts[index] = 1

    remaining = target_total - int(np.sum(counts))
    if remaining == 0:
        return counts
    weights = original_sizes[eligible].astype(float)
    if float(np.sum(weights)) == 0:
        weights = np.ones(eligible.size, dtype=float)
    quotas = remaining * weights / float(np.sum(weights))
    additions = np.floor(quotas).astype(int)
    counts[eligible] += additions
    remainder = remaining - int(np.sum(additions))
    if remainder > 0:
        fractions = quotas - additions
        order = sorted(
            range(eligible.size),
            key=lambda local_index: (
                -fractions[local_index],
                -weights[local_index],
                int(eligible[local_index]),
            ),
        )
        for local_index in order[:remainder]:
            counts[int(eligible[local_index])] += 1
    if int(np.sum(counts)) != target_total:
        raise AssertionError("组内人数分配总和与目标人数不一致")
    return counts


def _unique_valid_ids(values, valid_ids):
    """按原顺序保留合法且不重复的物理客户端 ID。"""
    valid_set = set(valid_ids)
    seen = set()
    result = []
    for value in values:
        if value not in valid_set:
            raise ValueError("映射包含非法物理客户端 ID {}".format(value))
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _resize_flat_mapping(original_ids, target_total, valid_ids):
    """保留原参与者优先级，并将 FL 映射调整为目标人数。"""
    original_ids = _unique_valid_ids(original_ids, valid_ids)
    selected = original_ids[:target_total]
    selected_set = set(selected)
    for client_id in valid_ids:
        if len(selected) >= target_total:
            break
        if client_id not in selected_set:
            selected.append(client_id)
            selected_set.add(client_id)
    if len(selected) != target_total:
        raise ValueError("无法构造目标人数为 {} 的 FL 映射".format(target_total))
    return selected


def _resize_hfl_groups(original_groups, target_total, valid_ids):
    """按目标总人数调整 HFL 组内映射，同时保持全局客户端唯一。"""
    normalized_groups = [
        _unique_valid_ids(group, valid_ids) for group in original_groups
    ]
    original_sizes = [len(group) for group in normalized_groups]
    target_sizes = _allocate_group_counts(original_sizes, target_total)
    resized_groups = [[] for _ in normalized_groups]
    used = set()

    # 先保留每个客户端原来的组归属。
    for group_index, group in enumerate(normalized_groups):
        for client_id in group:
            if len(resized_groups[group_index]) >= target_sizes[group_index]:
                break
            if client_id not in used:
                resized_groups[group_index].append(client_id)
                used.add(client_id)

    original_priority = []
    for group in normalized_groups:
        original_priority.extend(group)
    fill_pool = _unique_valid_ids(original_priority + list(valid_ids), valid_ids)
    for group_index, target_size in enumerate(target_sizes):
        for client_id in fill_pool:
            if len(resized_groups[group_index]) >= target_size:
                break
            if client_id not in used:
                resized_groups[group_index].append(client_id)
                used.add(client_id)

    if sum(len(group) for group in resized_groups) != target_total:
        raise ValueError("无法构造目标人数为 {} 的 HFL 映射".format(target_total))
    return resized_groups


def _nearest_nonempty_edges(edge_matrix, round_index, util_index, policy_index):
    """为原本无动态边缘但目标人数大于零的轮次寻找最近边缘集合。"""
    round_count = edge_matrix.shape[0]
    for distance in range(round_count):
        candidates = [round_index - distance, round_index + distance]
        for candidate in candidates:
            if candidate < 0 or candidate >= round_count:
                continue
            edges = _extract_edges(edge_matrix[candidate, util_index], policy_index)
            if edges:
                return edges
    return []


def _make_numeric_row(values):
    """创建适合保存为 MATLAB 行向量的物理 ID 数组。"""
    return np.asarray(values, dtype=np.uint8).reshape(1, -1)


def _make_group_cell(groups):
    """把 Python 分组列表转换为保留组边界的 MATLAB cell 行。"""
    group_cell = np.empty((1, len(groups)), dtype=object)
    for group_index, group in enumerate(groups):
        group_cell[0, group_index] = _make_numeric_row(group)
    return group_cell


def _make_policy_cell(dynamic_value, fixed_value):
    """创建与原实验一致的三策略 MATLAB cell，保留空的策略2。"""
    policy_cell = np.empty((1, 3), dtype=object)
    policy_cell[0, 0] = dynamic_value
    policy_cell[0, 1] = np.empty((1, 0), dtype=float)
    policy_cell[0, 2] = fixed_value
    return policy_cell


def _rebuild_hfl_scenario(data, method_name, valid_ids):
    """重建一个 HFL 方法的动态与固定策略映射及标准训练字段。"""
    mapping_field = "actual_c2e_map_{}".format(method_name)
    edge_field = "DynEdgeSet_{}".format(method_name)
    source_mappings = data[mapping_field]
    source_edges = data[edge_field]
    round_count, util_count = source_mappings.shape
    output_mappings = np.empty((round_count, util_count), dtype=object)
    output_edges = np.empty((round_count, util_count), dtype=object)
    client_dynamic = np.zeros((round_count, util_count), dtype=np.uint8)
    client_fixed = np.zeros((round_count, util_count), dtype=np.uint8)
    group_dynamic = np.zeros((round_count, util_count), dtype=np.uint8)
    group_fixed = np.zeros((round_count, util_count), dtype=np.uint8)

    for round_index in range(round_count):
        for util_index in range(util_count):
            policy_groups = {}
            policy_edges = {}
            for policy_index, suffix in ((0, "los"), (2, "fix")):
                target = _target_count(
                    data,
                    "client_num_{}_{}".format(method_name, suffix),
                    round_index,
                    util_index,
                )
                edges = _extract_edges(
                    source_edges[round_index, util_index], policy_index
                )
                used_fallback_edges = False
                if target > 0 and not edges:
                    edges = _nearest_nonempty_edges(
                        source_edges, round_index, util_index, policy_index
                    )
                    used_fallback_edges = bool(edges)
                if target > 0 and not edges:
                    raise ValueError(
                        "{} round={} util={} 目标人数为 {}，但找不到边缘槽位".format(
                            method_name, round_index, util_index, target
                        )
                    )
                if used_fallback_edges:
                    # 当前轮原本没有边缘槽位，回退槽位没有对应的原映射，因此从空组开始重建。
                    original_groups = [[] for _ in edges]
                else:
                    original_groups = _extract_groups(
                        source_mappings[round_index, util_index],
                        policy_index,
                        len(edges),
                    )
                resized_groups = _resize_hfl_groups(
                    original_groups, target, valid_ids
                )
                policy_groups[policy_index] = resized_groups
                policy_edges[policy_index] = edges
                effective_group_count = sum(bool(group) for group in resized_groups)
                if suffix == "los":
                    client_dynamic[round_index, util_index] = target
                    group_dynamic[round_index, util_index] = effective_group_count
                else:
                    client_fixed[round_index, util_index] = target
                    group_fixed[round_index, util_index] = effective_group_count

            output_mappings[round_index, util_index] = _make_policy_cell(
                _make_group_cell(policy_groups[0]),
                _make_group_cell(policy_groups[2]),
            )
            output_edges[round_index, util_index] = _make_policy_cell(
                _make_numeric_row(policy_edges[0]),
                _make_numeric_row(policy_edges[2]),
            )

    return {
        mapping_field: output_mappings,
        edge_field: output_edges,
        "client_num_{}_los".format(method_name): client_dynamic,
        "client_num_{}_fix".format(method_name): client_fixed,
        "group_num_{}_los".format(method_name): group_dynamic,
        "group_num_{}_fix".format(method_name): group_fixed,
    }


def _rebuild_fl_scenario(data, method_name, valid_ids):
    """重建一个 FL 方法的客户端映射和标准训练人数矩阵。"""
    mapping_field = "c2cmap_{}_all".format(method_name)
    client_field = "client_num_{}".format(method_name)
    source_mappings = data[mapping_field]
    round_count, util_count = source_mappings.shape
    output_mappings = np.empty((round_count, util_count), dtype=object)
    output_counts = np.zeros((round_count, util_count), dtype=np.uint8)

    for round_index in range(round_count):
        for util_index in range(util_count):
            target = _target_count(
                data, client_field, round_index, util_index
            )
            original_ids = _flatten_integer_values(
                source_mappings[round_index, util_index]
            )
            resized_ids = _resize_flat_mapping(original_ids, target, valid_ids)
            output_mappings[round_index, util_index] = _make_numeric_row(resized_ids)
            output_counts[round_index, util_index] = target
    return {mapping_field: output_mappings, client_field: output_counts}


def create_trainable_mat(input_path, output_path):
    """生成标准训练字段与 varctrl 人数、映射完全一致的新 MAT 文件。"""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("输出路径不能覆盖输入 MAT")
    if not input_path.is_file():
        raise FileNotFoundError("找不到输入 MAT：{}".format(input_path))

    data = loadmat(str(input_path), squeeze_me=False, struct_as_record=True)
    required_common = {
        "epoch_num", "total_util", "num_of_nodes", "Cloud",
        "actual_c2e_map_HFLSnF", "actual_c2e_map_HFLnoSnF",
        "DynEdgeSet_HFLSnF", "DynEdgeSet_HFLnoSnF",
        "c2cmap_FLSnF_all", "c2cmap_FLnoSnF_all",
    }
    missing = sorted(required_common.difference(data))
    missing.extend(
        field + "_varctrl" for field in CLIENT_FIELDS
        if field + "_varctrl" not in data
    )
    if missing:
        raise KeyError("输入 MAT 缺少字段：{}".format(", ".join(missing)))

    num_of_nodes = int(np.asarray(data["num_of_nodes"]).reshape(-1)[0])
    cloud_node = int(np.asarray(data["Cloud"]).reshape(-1)[0])
    valid_ids = tuple(
        node_id for node_id in range(1, num_of_nodes + 1)
        if node_id != cloud_node
    )

    output = {
        "schema_version": "training-varalpha-1.0",
        "training_source_file": str(input_path),
        "training_var_alpha": float(
            np.asarray(data.get("variance_control_varAlpha", [[np.nan]])).reshape(-1)[0]
        ),
        "training_note": (
            "标准训练字段已使用 varctrl 客户端数量，并同步重建全部 HFL/FL 映射；"
            "源 MAT 未被覆盖。"
        ),
        "epoch_num": data["epoch_num"],
        "total_util": data["total_util"],
        "num_of_nodes": data["num_of_nodes"],
        "Cloud": data["Cloud"],
    }
    for optional_field in ("EdgeSet", "TopoOption", "base_seed", "trace_id"):
        if optional_field in data:
            output[optional_field] = data[optional_field]

    output.update(_rebuild_hfl_scenario(data, "HFLSnF", valid_ids))
    output.update(_rebuild_hfl_scenario(data, "HFLnoSnF", valid_ids))
    output.update(_rebuild_fl_scenario(data, "FLSnF", valid_ids))
    output.update(_rebuild_fl_scenario(data, "FLnoSnF", valid_ids))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(output_path), output, do_compression=True, long_field_names=True)
    return output_path


def _default_paths():
    """返回当前工作区中默认的输入和训练输出路径。"""
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    input_path = workspace / "Topo_opt" / "result-U-6fixedge_epoch200_varAlpha_0p5.mat"
    output_path = (
        script_dir / "matlab" /
        "result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat"
    )
    return input_path, output_path


def main():
    """解析命令行参数并生成可直接用于 HFLSnF_dynEdge 训练的 MAT。"""
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    output_path = create_trainable_mat(args.input, args.output)
    print("训练兼容 MAT 已生成：{}".format(output_path))


if __name__ == "__main__":
    main()
