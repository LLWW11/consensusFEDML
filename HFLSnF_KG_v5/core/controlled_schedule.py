"""将OPEN统计受控人数显式转换为可回放的合成参与调度。"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def numeric_ids(value):
    """递归读取MAT单元数组中的整数物理节点编号。"""

    array = np.asarray(value)
    if array.dtype == object:
        return [item for child in array.ravel() for item in numeric_ids(child)]
    if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
        raise ValueError("映射包含非法编号")
    return [int(item) for item in array.ravel()]


def policy_cell(value):
    """构造兼容旧加载器的三策略单元数组，仅填充固定边缘策略。"""

    result = np.empty((1, 3), dtype=object)
    result[0, 0] = np.empty((1, 0))
    result[0, 1] = np.empty((1, 0))
    result[0, 2] = value
    return result


def rebuild_groups(source_groups, target, clients, usage, priority):
    """尽量保留原客户端及分组，以累计参与较少者确定增减和补位。"""

    groups = [list(numeric_ids(group)) for group in source_groups]
    flat = [client for group in groups for client in group]
    if len(set(flat)) != len(flat) or any(client not in usage for client in flat):
        raise ValueError("原映射存在重复、云节点或越界客户端")
    target = int(target)
    if not 0 <= target <= len(clients):
        raise ValueError("受控人数超出客户端范围")
    keep = set(sorted(flat, key=lambda client: (usage[client], priority[client]))[:target])
    groups = [[client for client in group if client in keep] for group in groups]
    missing = sorted((client for client in clients if client not in keep), key=lambda client: (usage[client], priority[client]))
    for client in missing[:target - len(keep)]:
        destination = min(range(len(groups)), key=lambda index: (len(groups[index]), index))
        groups[destination].append(client)
    for group in groups:
        for client in group:
            usage[client] += 1
    assert sum(map(len, groups)) == target
    return groups


def prepare_open_schedule(source_path, output_path, seed=42):
    """根据全部受控人数重建合成映射，保留来源哈希并验证标准解析器。"""

    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("不能覆盖原始统计文件")
    data = loadmat(str(source_path))
    if str(data["TopoOption"].item()) != "OPEN" or not np.isclose(float(data["variance_control_varAlpha"].item()), 0.1):
        raise ValueError("需要OPEN的0.1倍受控统计文件")
    nodes = int(data["num_of_nodes"].item())
    cloud = int(data["Cloud"].item())
    clients = [node for node in range(1, nodes + 1) if node != cloud]
    edges = numeric_ids(data["EdgeSet"])
    rounds = int(data["epoch_num"].item())
    utils = data["total_util"].ravel()
    shape = (rounds, len(utils))
    priority = {int(client): index for index, client in enumerate(np.random.RandomState(seed).permutation(clients))}
    output = {key: data[key] for key in ("TopoOption", "Cloud", "EdgeSet", "epoch_num", "total_util", "num_of_nodes")}
    output.update(schema_version="count-reconstructed-1.0", schedule_provenance="synthetic_reconstruction_from_varctrl_counts", physical_feasibility_validated=np.uint8(0), mapping_reconstruction_seed=int(seed), training_varAlpha=0.1, variance_control_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), mapping_reconstruction_note="按受控人数尽量保留原HFL身份与分组；FL身份缺失时按累计参与次数补位。不代表重新优化或验证物理链路可行性。")
    for method, count_field in (("HFLSnF", "client_num_HFLSnF_fix"), ("HFLnoSnF", "client_num_HFLnoSnF"), ("FLnoSnF", "client_num_FLnoSnF")):
        counts = data[count_field + "_varctrl"]
        if counts.shape != shape or not np.all(counts == np.round(counts)):
            raise ValueError("受控人数矩阵不符合轮数与利用率")
        mappings = np.empty(shape, dtype=object)
        edge_maps = np.empty(shape, dtype=object)
        group_counts = np.zeros(shape, dtype=np.uint8)
        for util_index in range(len(utils)):
            usage = {client: 0 for client in clients}
            for round_index in range(rounds):
                if method == "HFLSnF":
                    source_groups = data["actual_c2e_map_HFLSnF"][round_index, util_index].ravel()[2].ravel()
                elif method == "HFLnoSnF":
                    source_groups = data["actual_c2e_map_HFLnoSnF"][round_index, util_index].ravel()
                else:
                    source_groups = [np.empty((1, 0))]
                if len(source_groups) != (1 if method == "FLnoSnF" else len(edges)):
                    raise ValueError("原固定边缘映射槽位数不一致")
                groups = rebuild_groups(source_groups, counts[round_index, util_index], clients, usage, priority)
                if method == "FLnoSnF":
                    mappings[round_index, util_index] = np.array(groups[0], dtype=np.uint16).reshape(1, -1)
                else:
                    group_cell = np.empty((1, len(edges)), dtype=object)
                    for index, group in enumerate(groups):
                        group_cell[0, index] = np.array(group, dtype=np.uint16).reshape(1, -1)
                    mappings[round_index, util_index] = policy_cell(group_cell)
                    edge_maps[round_index, util_index] = policy_cell(np.array(edges).reshape(1, -1))
                    group_counts[round_index, util_index] = sum(bool(group) for group in groups)
        if method == "FLnoSnF":
            output["c2cmap_FLnoSnF_all"] = mappings
            output["client_num_FLnoSnF"] = counts
        else:
            output["actual_c2e_map_" + method] = mappings
            output["DynEdgeSet_" + method] = edge_maps
            output["client_num_" + method + "_fix"] = counts
            output["group_num_" + method + "_fix"] = group_counts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(output_path), output, do_compression=True)
    from .matlab_topology_schedule import MatlabTopologySchedule

    checks = []
    for util in utils:
        for architecture, snf in (("hfl", True), ("hfl", False), ("fl", False)):
            schedule = MatlabTopologySchedule(str(output_path), architecture, snf, "fixed", float(util), client_num_in_total=len(clients))
            coverage = set(client for index in range(min(150, rounds)) for client in schedule.get_round(index).active_client_indexes)
            if len(coverage) != len(clients):
                raise ValueError("前150轮未覆盖全部客户端：{} {}".format(architecture, util))
            checks.append({"util": float(util), "architecture": architecture, "snf": snf, "covered_first150": len(coverage), "rounds": schedule.round_count})
    return {"source_sha256": output["variance_control_source_sha256"], "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "checks": checks, "schedule_provenance": output["schedule_provenance"], "physical_feasibility_validated": False}
