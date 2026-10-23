"""多数据集、多MAT拓扑实验的配置、分区及调度预检。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .core.randomness import seed_everything
from .core.topology import MatlabTopologyProvider
from .experiment import build_federated_data, build_transe, load_configured_dataset
from .selection import PACKAGE_DIR, apply_experiment_selection, package_path


def file_sha256(path):
    """按内容绑定配置、数据和调度，避免续跑混入其他实验输入。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_configuration(path):
    """读取分节YAML并应用多数据集拓扑规则，不加载FedML。"""

    sections = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(sections, dict) or not all(isinstance(section, dict) for section in sections.values()):
        raise ValueError("配置必须是分节YAML")
    flat = {}
    for section in sections.values():
        if set(flat).intersection(section):
            raise ValueError("配置不同分节包含重复字段")
        flat.update(section)
    args = apply_experiment_selection(SimpleNamespace(**flat))
    if args.topology_type != "matlab_direct":
        raise ValueError("多拓扑入口使用matlab_direct，固定人数实验请使用旧入口")
    if str(args.topology_schedule_policy).lower() != "strict":
        raise ValueError("首轮多拓扑实验要求strict，不循环复用源轮次")
    if int(args.partition_search_seed) != int(args.random_seed):
        raise ValueError("partition_search_seed必须与random_seed一致")
    return args


def build_provider(args):
    """构造与正式训练相同的MAT提供器，并适配任意合法节点数量。"""

    return MatlabTopologyProvider(
        package_path(args.dynamic_group_mat_file), args.topology_architecture,
        args.topology_snf, args.topology_edge_mode, args.topology_util,
        args.client_num_in_total, args.topology_schedule_policy,
    )


def schedule_statistics(provider, rounds, partitions):
    """按实际训练窗口计算调度哈希、参与方差、覆盖和有效数据遍数。"""

    digest = hashlib.sha256()
    counts, group_counts = [], []
    selections = {int(partition.client_id): 0 for partition in partitions}
    sizes = {int(partition.client_id): partition.triple_count for partition in partitions}
    scheduled = 0
    for index in range(int(rounds)):
        topology = provider.get_round(index)
        if not topology.participant_count:
            raise ValueError("训练窗口含零参与轮次：{}".format(index))
        record = {
            "source_round_index": int(topology.source_round_index),
            "groups": topology.copy_groups(),
            "edge_node_ids": {str(key): int(value) for key, value in topology.edge_node_ids.items()},
        }
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        counts.append(topology.participant_count)
        group_counts.append(len(topology.group_to_client_indexes))
        for client in topology.active_client_indexes:
            selections[client] += 1
            scheduled += sizes[client]
    if any(value == 0 for value in selections.values()):
        raise ValueError("训练窗口存在从未参与的客户端")
    return {
        "schedule_hash": digest.hexdigest(), "rounds": int(rounds),
        "participant_mean": float(np.mean(counts)), "participant_variance": float(np.var(counts, ddof=1)) if rounds > 1 else 0.0,
        "participant_min": min(counts), "participant_max": max(counts),
        "group_mean": float(np.mean(group_counts)), "selection_counts": selections,
        "scheduled_positive_triples_per_local_epoch": scheduled,
    }


def model_hash(model):
    """使用与动态训练器相同的状态哈希格式核对初始化。"""

    digest = hashlib.sha256()
    for name, raw in sorted(model.state_dict().items()):
        value = raw.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_configurations(paths):
    """每种数据分区仅复算一次，核对同场景共同配方及各方法真实调度。"""

    if not paths:
        raise ValueError("至少需要一份配置")
    partition_cache, common_conditions, entries = {}, {}, []
    identities = set()
    excluded = {"topology_architecture", "topology_snf", "topology_edge_mode", "edge_num", "run_name", "ablation_arm", "comparison_scenario", "expected_topology_schedule_hash", "expected_partition_hash"}
    for path in paths:
        path = Path(path).resolve()
        args = load_configuration(path)
        if args.run_name in identities:
            raise ValueError("同一批次的run_name必须唯一")
        identities.add(args.run_name)
        data_hashes = {split: file_sha256(package_path(args.data_dir) / (split + ".txt")) for split in ("train", "valid", "test")}
        cache_key = (args.dataset, tuple(data_hashes.values()), int(args.client_num_in_total), int(args.random_seed), args.partition_strategy, args.partition_domain_extractor, float(args.partition_load_tolerance), int(args.partition_search_restarts))
        if cache_key not in partition_cache:
            print("复算{}：{}客户端，种子{}，图语义{}次重启".format(args.dataset, args.client_num_in_total, args.random_seed, args.partition_search_restarts), flush=True)
            dataset = load_configured_dataset(args)
            partition_cache[cache_key] = build_federated_data(args, dataset)
        partition = partition_cache[cache_key]
        provider = build_provider(args)
        statistics = schedule_statistics(provider, int(args.comm_round), partition.partitions)
        seed_everything(int(args.random_seed))
        initial_hash = model_hash(build_transe(args, partition.dataset.num_entities, partition.dataset.num_relations))
        for field, actual in (("expected_partition_hash", partition.partition_hash), ("expected_topology_schedule_hash", statistics["schedule_hash"])):
            expected = getattr(args, field, None)
            if expected and expected != actual:
                raise ValueError("配置{}与真实计算不一致".format(field))
        group = (args.network_topology, args.dataset, int(args.random_seed))
        conditions = {key: value for key, value in vars(args).items() if key not in excluded}
        if group in common_conditions and common_conditions[group] != conditions:
            raise ValueError("同一网络、数据集和种子的共同训练参数不一致")
        common_conditions[group] = conditions
        metadata = provider.describe()
        statistics["effective_full_data_passes"] = statistics["scheduled_positive_triples_per_local_epoch"] * int(args.epochs) / float(partition.dataset.train_triples.shape[0])
        entries.append({
            "config": str(path), "config_sha256": file_sha256(path), "configuration": vars(args),
            "data_sha256": data_hashes, "mat_sha256": file_sha256(args.dynamic_group_mat_file),
            "partition_hash": partition.partition_hash, "initial_model_hash": initial_hash,
            "schedule": statistics, "topology": metadata, "partition": partition.summary(),
        })
    return {"status": "passed", "entries": entries, "partition_recomputations": len(partition_cache)}
