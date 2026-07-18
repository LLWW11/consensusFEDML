import copy
import csv
from datetime import datetime
import json
import logging
import os

import numpy as np
from scipy.io import loadmat
import torch

from client_test import HFLClient
from fedavg_test import FedAvgAPI
from group_test import Group
from probe_batch import ProbeBatchRecorder, select_fixed_balanced_probe
from topology_schedule import MatlabTopologySchedule


class HierarchicalTrainer(FedAvgAPI):
    """
    层次联邦训练器。

    该训练器支持按 .mat 文件动态分组，并为每个客户端维护跨轮持久本地模型状态。
    """

    def __init__(self, args, device, dataset, model):
        """
        初始化层次训练器，并在使用正式 MATLAB 映射时预加载全部拓扑。

        拓扑必须在父类创建客户端和边缘组之前完成校验，以便动态调整所需的最大边缘槽位数。
        """
        self.topology_schedule = None
        self.current_round_topology = None
        self.fixed_candidate_client_indexes = None
        self.model_distribution_scope = str(
            getattr(args, "model_distribution_scope", "all")
        ).strip().lower()
        if self.model_distribution_scope not in {"active", "all"}:
            raise ValueError(
                "model_distribution_scope 必须是 active 或 all，实际为 {}".format(
                    self.model_distribution_scope
                )
            )
        if args.client_num_per_round > args.client_num_in_total:
            raise ValueError(
                "client_num_per_round={} 不能超过 client_num_in_total={}".format(
                    args.client_num_per_round, args.client_num_in_total
                )
            )
        if getattr(args, "group_method", None) == "matlab":
            mat_path = self._resolve_matlab_topology_path(args)
            self.topology_schedule = MatlabTopologySchedule(
                mat_path=mat_path,
                architecture=getattr(args, "topology_architecture", "hfl"),
                snf_enabled=getattr(args, "topology_snf", True),
                edge_mode=getattr(args, "topology_edge_mode", "fixed"),
                util=getattr(args, "topology_util", 0.5),
                client_num_in_total=args.client_num_in_total,
                candidate_client_count=args.client_num_per_round,
            )
            total_local_epochs = (
                int(args.comm_round)
                * int(args.group_comm_round)
                * int(args.epochs)
            )
            if total_local_epochs > self.topology_schedule.round_count:
                raise ValueError(
                    "展平后的本地 epoch 数 {} 超过 MATLAB 拓扑行数 {}".format(
                        total_local_epochs, self.topology_schedule.round_count
                    )
                )
            args.group_num = self.topology_schedule.group_capacity
            logging.info("MATLAB topology metadata = %s", self.topology_schedule.to_metadata())
        super().__init__(args, device, dataset, model)
        if self.topology_schedule is not None:
            self._initialize_fixed_candidate_clients()

    @staticmethod
    def _resolve_matlab_topology_path(args):
        """解析正式 MATLAB 拓扑文件的绝对路径。"""
        filename = getattr(
            args, "dynamic_group_mat_file", "matlab/result-U-6fixedge_epoch200.mat"
        )
        if os.path.isabs(filename):
            return filename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.dirname(filename):
            return os.path.normpath(os.path.join(script_dir, filename))
        return os.path.join(script_dir, "matlab", filename)

    def _setup_clients(
            self,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
    ):
        """
        初始化训练阶段需要用到的客户端注册表、边缘组对象和测试用客户端。

        dynamic 模式下不会提前固定客户端归属，而是在每轮训练时按 .mat 当前行重新生成分组。
        """
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == "random":
            self.group_indexes = np.random.randint(
                0, self.args.group_num, self.args.client_num_in_total
            )

            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if group_idx not in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)

        elif self.args.group_method == "average":
            # 这段产生 0 到 group_num-1 范围内的整数，共计 client_num_in_total 个。
            random_ints = np.random.choice(self.args.group_num, self.args.group_num, replace=False)
            result1 = np.repeat(random_ints, self.args.client_num_in_total // self.args.group_num)
            result2 = np.random.choice(
                self.args.group_num,
                self.args.client_num_in_total % self.args.group_num,
                replace=False,
            )
            self.group_indexes = np.concatenate([result1, result2], axis=0)
            np.random.shuffle(self.group_indexes)

            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if group_idx not in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)

        elif self.args.group_method in {"dynamic", "matlab"}:
            self.group_indexes = None
            # 动态分组每轮会重新选择组内客户端，所以每个组对象先保留完整客户端池。
            group_to_client_indexes = {
                group_idx: list(range(self.args.client_num_in_total))
                for group_idx in range(self.args.group_num)
            }

        else:
            raise Exception(self.args.group_method)

        self.client_registry = self._build_client_registry(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )

        self.group_dict = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            self.group_dict[group_idx] = Group(
                group_idx,
                client_indexes,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                self.args,
                self.device,
                self.model,
                self.model_trainer,
                client_registry=self.client_registry,
            )

        # 保留一个独立 dummy client，供 FedAvgAPI._local_test_on_all_clients() 复用。
        self.client_list = [
            HFLClient(
                -1,
                train_data_local_dict[0],
                test_data_local_dict[0],
                train_data_local_num_dict[0],
                self.args,
                self.device,
                self.model,
                self.model_trainer,
            )
        ]
        logging.info("############setup_clients (END)#############")

    def _build_client_registry(
            self,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
    ):
        """
        创建全局唯一的客户端对象注册表。

        每个 client_idx 只对应一个 HFLClient，避免同一客户端在多个组里拥有多份本地模型状态。
        """
        client_registry = {}
        for client_idx in range(self.args.client_num_in_total):
            client_registry[client_idx] = HFLClient(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model,
                model_trainer,
            )
        return client_registry

    def _client_sampling(
            self, global_round_idx, client_num_in_total, client_num_per_round
    ):
        """
        按普通随机分组方式采样客户端，并返回 group -> client_indexes 映射。
        """
        sampled_client_indexes = super()._client_sampling(
            global_round_idx, client_num_in_total, client_num_per_round
        )
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if group_idx not in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        logging.info("client_indexes of each group = {}".format(group_to_client_indexes))
        return group_to_client_indexes

    def _client_sampling_average(
            self, global_round_idx, client_num_in_total, client_num_per_round
    ):
        """
        按平均分组方式采样客户端，并返回 group -> client_indexes 映射。
        """
        group_to_client_indexes = {}
        for client_idx, group_idx in enumerate(self.group_indexes):
            if group_idx not in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)

        sampled_client_indexes = super()._client_sampling_average(
            global_round_idx, client_num_in_total, client_num_per_round, group_to_client_indexes
        )
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]
            if group_idx not in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        logging.info("client_indexes of each group = {}".format(group_to_client_indexes))
        return group_to_client_indexes

    def _client_sampling_dynamic(
            self, global_round_idx, client_num_in_total, each_client_num, each_group_num, each_group_num_index
    ):
        """
        根据 .mat 当前轮的每组客户端数量，直接构造本轮 group -> client_indexes 映射。

        .mat 中的 client_num 只记录每组需要多少客户端，所以这里按客户端编号顺序切分：
        例如 [8, 6, 5, 13] 会得到 0-7、8-13、14-18、19-31 四个组。
        """
        if each_client_num > client_num_in_total:
            raise ValueError(
                "Round {} needs {} clients from .mat, but client_num_in_total is {}.".format(
                    global_round_idx, each_client_num, client_num_in_total
                )
            )

        group_to_client_indexes = {}
        next_client_idx = 0
        for group_idx in range(each_group_num):
            client_count = int(each_group_num_index[group_idx])
            group_clients = list(range(next_client_idx, next_client_idx + client_count))
            next_client_idx += client_count
            # 客户端数量为 0 的组不参与本轮训练和聚合。
            if group_clients:
                group_to_client_indexes[group_idx] = group_clients

        logging.info("client_indexes of each group = {}".format(group_to_client_indexes))
        return group_to_client_indexes

    def _get_dynamic_group_mat_file(self):
        """
        返回动态分组 .mat 文件的绝对路径。

        相对文件名会优先从当前代码目录下的 matlab/ 文件夹读取，兼容所有 .mat 数据集中存放的目录结构。
        """
        filename = getattr(self.args, "dynamic_group_mat_file", "my_data_150.mat")
        if os.path.isabs(filename):
            return filename

        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.dirname(filename):
            candidate_paths = [os.path.join(script_dir, filename)]
        else:
            candidate_paths = [
                os.path.join(script_dir, "matlab", filename),
                os.path.join(script_dir, filename),
            ]

        # 优先返回真实存在的路径；如果都不存在，返回首选路径，让后续 loadmat 报出明确文件名。
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return candidate_path
        return candidate_paths[0]

    def _get_client(self, global_round_idx):
        """
        从 .mat 文件读取指定轮次的分组数量和每组客户端数量。

        返回值分别是本轮启用组数、每组客户端数量数组、本轮参与客户端总数。
        """
        filename = self._get_dynamic_group_mat_file()
        data = loadmat(filename)
        total_group_num = data["group_num"]
        group_num_index = data["client_num"]

        if global_round_idx >= group_num_index.shape[0]:
            raise ValueError(
                "Round {} is out of range for dynamic group file {} with {} rows.".format(
                    global_round_idx, filename, group_num_index.shape[0]
                )
            )

        each_group_num = int(np.asarray(total_group_num[global_round_idx]).reshape(-1)[0])
        if each_group_num > self.args.group_num:
            raise ValueError(
                "Round {} needs {} groups from .mat, but group_num is {}.".format(
                    global_round_idx, each_group_num, self.args.group_num
                )
            )

        round_group_counts = np.asarray(group_num_index[global_round_idx]).astype(int).reshape(-1)
        each_group_num_index = round_group_counts[:each_group_num]
        if np.any(each_group_num_index < 0):
            raise ValueError("Round {} has negative client counts in .mat.".format(global_round_idx))

        # 只统计本轮启用组的客户端数量，后面的 0 表示未启用组。
        total_client_num_per_round = int(np.sum(each_group_num_index))
        if total_client_num_per_round > self.args.client_num_in_total:
            raise ValueError(
                "Round {} needs {} clients from .mat, but client_num_in_total is {}.".format(
                    global_round_idx, total_client_num_per_round, self.args.client_num_in_total
                )
            )

        return each_group_num, each_group_num_index, total_client_num_per_round

    def _get_active_client_indexes(self, group_to_client_indexes):
        """
        从本轮 group -> client_indexes 映射中提取参与聚合的客户端编号。
        """
        active_client_indexes = []
        for client_indexes in group_to_client_indexes.values():
            active_client_indexes.extend(client_indexes)
        return sorted(set(active_client_indexes))

    def _initialize_fixed_candidate_clients(self):
        """
        按 random_seed 从真实客户端池中无放回抽取一次固定候选客户端。

        返回顺序同时作为 MAT 候选槽位 0..N-1 与客户端探针 CSV 的列顺序，
        整次实验不再随 global_epoch 改变。
        """
        client_num_in_total = int(self.args.client_num_in_total)
        candidate_client_count = int(self.args.client_num_per_round)
        if candidate_client_count <= 0:
            raise ValueError("client_num_per_round 必须大于 0")
        if candidate_client_count > client_num_in_total:
            raise ValueError(
                "client_num_per_round={} 不能超过 client_num_in_total={}".format(
                    candidate_client_count, client_num_in_total
                )
            )

        if candidate_client_count == client_num_in_total:
            candidate_indexes = list(range(client_num_in_total))
        else:
            rng = np.random.RandomState(int(getattr(self.args, "random_seed", 0)))
            candidate_indexes = rng.choice(
                client_num_in_total,
                size=candidate_client_count,
                replace=False,
            ).astype(int).tolist()

        if len(set(candidate_indexes)) != candidate_client_count:
            raise ValueError("固定候选客户端中存在重复编号")
        self.fixed_candidate_client_indexes = candidate_indexes
        logging.info("fixed candidate clients = %s", candidate_indexes)
        return list(candidate_indexes)

    def _get_mat_group_client_counts(self, round_topology):
        """读取 MAT 当前 epoch 的各边缘组人数，并补齐人数为零的边缘槽位。"""
        group_client_counts = {
            int(group_idx): int(client_count)
            for group_idx, client_count in round_topology.group_client_counts.items()
        }
        participant_count = sum(group_client_counts.values())
        if participant_count != int(round_topology.participant_count):
            raise ValueError(
                "MAT 分组人数之和 {} 与 participant_count={} 不一致".format(
                    participant_count, round_topology.participant_count
                )
            )
        if participant_count > self.args.client_num_per_round:
            raise ValueError(
                "MAT 需要 {} 个客户端，但候选池只有 {} 个".format(
                    participant_count, self.args.client_num_per_round
                )
            )
        return group_client_counts

    def _map_mat_slots_to_clients(self, candidate_client_indexes, round_topology):
        """
        将 MAT 当前 epoch 的候选槽位分组映射为固定候选池中的真实客户端。

        candidate_client_indexes 的位置就是 MAT 槽位编号，因此该映射不会再随机
        打乱候选客户端，也不会仅按各组人数重新切片。
        """
        if len(candidate_client_indexes) != self.args.client_num_per_round:
            raise ValueError(
                "候选客户端数量 {} 与 client_num_per_round={} 不一致".format(
                    len(candidate_client_indexes), self.args.client_num_per_round
                )
            )
        if len(set(candidate_client_indexes)) != len(candidate_client_indexes):
            raise ValueError("候选客户端中存在重复编号")
        if any(
                client_idx < 0 or client_idx >= self.args.client_num_in_total
                for client_idx in candidate_client_indexes
        ):
            raise ValueError("固定候选客户端编号超出真实客户端池范围")

        group_to_client_indexes = {}
        for group_idx, candidate_slots in round_topology.copy_groups().items():
            group_clients = []
            for candidate_slot in candidate_slots:
                if candidate_slot < 0 or candidate_slot >= len(candidate_client_indexes):
                    raise ValueError(
                        "MAT 候选槽位 {} 超出 0..{} 范围".format(
                            candidate_slot, len(candidate_client_indexes) - 1
                        )
                    )
                group_clients.append(candidate_client_indexes[candidate_slot])
            if group_clients:
                group_to_client_indexes[int(group_idx)] = group_clients

        active_client_indexes = self._get_active_client_indexes(group_to_client_indexes)
        if len(active_client_indexes) != int(round_topology.participant_count):
            raise ValueError(
                "MAT 映射得到 {} 个真实客户端，但 participant_count={}".format(
                    len(active_client_indexes), round_topology.participant_count
                )
            )
        return group_to_client_indexes

    def _build_round_groups(self, global_epoch):
        """
        根据当前分组策略生成本地 epoch 的 group -> client_indexes 映射。

        matlab 模式严格使用正式 MAT 的候选槽位身份与分组；旧 dynamic、random 和
        average 模式保持原有采样行为。
        """
        if self.args.group_method == "random":
            return self._client_sampling(
                global_epoch,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )
        if self.args.group_method == "average":
            return self._client_sampling_average(
                global_epoch,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )
        if self.args.group_method == "dynamic":
            each_group_num, each_group_num_index, each_client_num = self._get_client(global_epoch)
            return self._client_sampling_dynamic(
                global_epoch,
                self.args.client_num_in_total,
                each_client_num,
                each_group_num,
                each_group_num_index,
            )
        if self.args.group_method == "matlab":
            if self.fixed_candidate_client_indexes is None:
                self._initialize_fixed_candidate_clients()
            self.current_round_topology = self.topology_schedule.get_round(global_epoch)
            group_to_client_indexes = self._map_mat_slots_to_clients(
                self.fixed_candidate_client_indexes,
                self.current_round_topology,
            )
            logging.info(
                "MATLAB topology scenario=%s util=%s global_epoch=%d groups=%s edge_nodes=%s active_clients=%s",
                self.topology_schedule.scenario_name,
                self.topology_schedule.requested_util,
                global_epoch,
                group_to_client_indexes,
                self.current_round_topology.edge_node_ids,
                list(self._get_active_client_indexes(group_to_client_indexes)),
            )
            return group_to_client_indexes
        raise Exception(self.args.group_method)

    def _calculate_global_epoch(self, global_round_idx, group_round_idx, epoch_idx):
        """
        将通信轮、组内通信轮和本地 epoch 展平为 MAT 使用的全局 epoch 编号。
        """
        return (
            global_round_idx * self.args.group_comm_round * self.args.epochs
            + group_round_idx * self.args.epochs
            + epoch_idx
        )

    def _train_active_clients_one_epoch(
            self,
            global_round_idx,
            group_round_idx,
            epoch_idx,
            active_client_indexes,
    ):
        """
        仅让指定活跃客户端各训练一个本地 epoch，并返回全局 epoch 编号。

        空参与 epoch 不调用任何客户端训练方法，但仍返回可用于读取 MAT 和写指标的
        展平编号。
        """
        global_epoch = self._calculate_global_epoch(
            global_round_idx, group_round_idx, epoch_idx
        )
        if len(set(active_client_indexes)) != len(active_client_indexes):
            raise ValueError("活跃客户端列表中存在重复编号")
        for client_idx in active_client_indexes:
            if client_idx < 0 or client_idx >= self.args.client_num_in_total:
                raise ValueError("活跃客户端编号 {} 超出范围".format(client_idx))
            client = self.client_registry[client_idx]
            trained_global_epoch, _ = client.train_one_epoch(
                global_round_idx, group_round_idx, epoch_idx
            )
            if trained_global_epoch != global_epoch:
                raise ValueError(
                    "客户端 {} 返回 global_epoch={}，期望 {}".format(
                        client_idx, trained_global_epoch, global_epoch
                    )
                )
        logging.info(
            "active clients have finished global_epoch=%d clients=%s",
            global_epoch,
            list(active_client_indexes),
        )
        return global_epoch

    def _train_one_epoch_all_clients(self, global_round_idx, group_round_idx, epoch_idx):
        """保留旧分组模式兼容性，让全部真实客户端训练一个本地 epoch。"""
        return self._train_active_clients_one_epoch(
            global_round_idx,
            group_round_idx,
            epoch_idx,
            list(range(self.args.client_num_in_total)),
        )

    def _is_consensus_probe_enabled(self):
        """
        判断是否启用层级概率探针。

        默认启用，便于当前实验直接生成三份 CSV；如需关闭可在 YAML 中设为 false。
        """
        return bool(getattr(self.args, "enable_consensus_probe", True))

    def _get_probe_output_format(self):
        """返回探针输出格式，并保留未配置旧实验继续写 CSV 的兼容行为。"""
        output_format = str(
            getattr(self.args, "probe_output_format", "legacy_csv")
        ).strip().lower()
        aliases = {"csv": "legacy_csv", "legacy": "legacy_csv"}
        output_format = aliases.get(output_format, output_format)
        if output_format not in {"legacy_csv", "npz"}:
            raise ValueError(
                "probe_output_format 只能是 legacy_csv（或 csv）与 npz，实际为 {}。".format(
                    output_format
                )
            )
        return output_format

    def _get_fixed_probe_candidate_ids(self):
        """返回 NPZ 每个 epoch 始终使用的固定候选客户端编号。"""
        if self.fixed_candidate_client_indexes is None:
            raise ValueError(
                "NPZ 固定探针模式要求实验开始时存在固定候选客户端集合；"
                "非 MATLAB 旧分组模式请继续使用 legacy_csv。"
            )
        return [int(value) for value in self.fixed_candidate_client_indexes]

    def _prepare_fixed_probe_set(self):
        """从配置指定的数据源选择固定、类别均衡且不改变训练随机状态的探针。"""
        probe_source = str(getattr(self.args, "probe_source", "test")).strip().lower()
        if probe_source == "test":
            probe_data = self.test_global
        elif probe_source == "train":
            probe_data = self.train_global
        else:
            raise ValueError("probe_source 只能是 test 或 train。")
        return select_fixed_balanced_probe(
            probe_data=probe_data,
            samples_per_class=int(getattr(self.args, "probe_samples_per_class", 10)),
            seed=int(getattr(self.args, "probe_seed", 0)),
            source=probe_source,
        )

    def _create_batch_probe_recorder(self, probe_set):
        """根据正式配置创建固定探针 NPZ 与逐 epoch 摘要记录器。"""
        unique_labels = np.unique(probe_set.true_labels)
        dataset_name = str(getattr(self.args, "dataset", "")).strip().lower()
        expected_class_count = 10 if dataset_name == "mnist" else int(unique_labels.shape[0])
        expected_labels = np.arange(expected_class_count, dtype=np.int64)
        if not np.array_equal(unique_labels, expected_labels):
            raise ValueError(
                "固定探针必须从 0 开始覆盖全部 {} 个类别，实际为 {}。".format(
                    expected_class_count, unique_labels.tolist()
                )
            )
        expected_per_class = int(getattr(self.args, "probe_samples_per_class", 10))
        for class_id in expected_labels:
            actual_count = int(np.sum(probe_set.true_labels == class_id))
            if actual_count != expected_per_class:
                raise ValueError(
                    "固定探针类别 {} 应有 {} 张，实际为 {} 张。".format(
                        int(class_id), expected_per_class, actual_count
                    )
                )
        recorder = ProbeBatchRecorder(
            result_dir=self.args.result_dir,
            total_epochs=(
                int(self.args.comm_round)
                * int(self.args.group_comm_round)
                * int(self.args.epochs)
            ),
            candidate_client_ids=self._get_fixed_probe_candidate_ids(),
            edge_slot_count=int(self.args.group_num),
            probe_set=probe_set,
            class_count=expected_class_count,
            npz_filename=getattr(
                self.args, "probe_npz_file", "probe_probabilities.npz"
            ),
            summary_filename=getattr(
                self.args, "probe_summary_file", "probe_epoch_summary.csv"
            ),
            checkpoint_interval=int(
                getattr(self.args, "probe_checkpoint_interval", 10)
            ),
        )
        try:
            self._update_batch_probe_metadata(recorder)
        except Exception:
            # 元数据写入失败时也关闭摘要句柄，避免Windows下遗留被占用文件。
            recorder.close()
            raise
        return recorder

    def _update_batch_probe_metadata(self, recorder):
        """在已生成的拓扑元数据中补充固定探针哈希、配置和预期形状。"""
        metadata_path = os.path.join(self.args.result_dir, "topology_metadata.json")
        metadata = {}
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)
        metadata.update(recorder.metadata())
        metadata.update({
            "probe_seed": int(getattr(self.args, "probe_seed", 0)),
            "probe_inference_batch_size": int(
                getattr(self.args, "probe_inference_batch_size", 100)
            ),
            "probe_checkpoint_interval": int(
                getattr(self.args, "probe_checkpoint_interval", 10)
            ),
            "probe_indices": [
                int(value) for value in recorder.probe_set.indices
            ],
            "probe_true_labels": [
                int(value) for value in recorder.probe_set.true_labels
            ],
        })
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    def _format_result_name_part(self, value):
        """
        将配置值转换成适合放进结果文件夹名称的字符串。

        这里主要去掉路径分隔符和空格，避免不同平台下生成非法目录名。
        """
        text = str(value)
        for old, new in [("\\", "_"), ("/", "_"), (":", "_"), (" ", "")]:
            text = text.replace(old, new)
        return text

    def _setup_result_dir(self):
        """
        为本次训练创建统一结果目录。

        目录格式为 result/{dataset}_{partition_alpha}_{group_method}_{yyyymmdd}。

        MATLAB 模式会把 group_method 展开为场景和 util，避免四组对照实验在同一天互相覆盖。
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result_root = os.path.join(script_dir, "result")
        dataset = self._format_result_name_part(getattr(self.args, "dataset", "dataset"))
        partition_alpha = self._format_result_name_part(getattr(self.args, "partition_alpha", "alpha"))
        if self.topology_schedule is not None:
            group_method = "{}_u{}".format(
                self.topology_schedule.scenario_name,
                self.topology_schedule.util_label,
            )
        else:
            group_method = self._format_result_name_part(getattr(self.args, "group_method", "group"))
        experiment_tag = self._format_result_name_part(
            getattr(self.args, "experiment_tag", "")
        )
        if experiment_tag:
            group_method = "{}_{}".format(group_method, experiment_tag)
        date_text = datetime.now().strftime("%Y%m%d")
        result_dir = os.path.join(result_root, "{}_{}_{}_{}".format(
            dataset, partition_alpha, group_method, date_text
        ))
        os.makedirs(result_dir, exist_ok=True)
        self.args.result_dir = result_dir
        self._write_topology_metadata(result_dir)
        logging.info("training result dir = {}".format(result_dir))
        return result_dir

    def _write_topology_metadata(self, result_dir):
        """写入静态实验元数据，并初始化逐 epoch 运行时调度文件。"""
        if self.topology_schedule is None:
            return
        metadata = self.topology_schedule.to_metadata()
        metadata.update({
            "configured_comm_round": int(self.args.comm_round),
            "epochs": int(self.args.epochs),
            "group_comm_round": int(self.args.group_comm_round),
            "partition_alpha": float(self.args.partition_alpha),
            "random_seed": int(getattr(self.args, "random_seed", 0)),
            "probe_output_format": self._get_probe_output_format(),
            "probe_source": getattr(self.args, "probe_source", "test"),
            "experiment_tag": getattr(self.args, "experiment_tag", ""),
            "client_num_in_total": int(self.args.client_num_in_total),
            "client_num_per_round": int(self.args.client_num_per_round),
            "model_distribution_scope": self.model_distribution_scope,
            "candidate_sampling_mode": "fixed_once_by_random_seed",
            "fixed_candidate_client_indexes": [
                int(value) for value in self.fixed_candidate_client_indexes
            ],
            "mat_candidate_slot_to_client_index": {
                str(candidate_slot): int(client_idx)
                for candidate_slot, client_idx
                in enumerate(self.fixed_candidate_client_indexes)
            },
            "total_local_epochs": (
                int(self.args.comm_round)
                * int(self.args.group_comm_round)
                * int(self.args.epochs)
            ),
        })
        if self._get_probe_output_format() == "legacy_csv":
            # 历史格式仍记录逐epoch单图标签文件的列定义。
            metadata.update({
                "probe_meta_file": getattr(
                    self.args, "probe_meta_file", "probe_meta.csv"
                ),
                "probe_meta_columns": [
                    "global_epoch",
                    "global_round_idx",
                    "group_round_idx",
                    "local_epoch_idx",
                    "probe_source",
                    "probe_index",
                    "true_label",
                ],
            })
        output_path = os.path.join(result_dir, "topology_metadata.json")
        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

        self.topology_schedule_output_path = os.path.join(
            result_dir, "topology_schedule.jsonl"
        )
        # 每个 epoch 的 MAT 活跃槽位仍不同，因此运行时映射继续逐行写入 JSONL。
        with open(self.topology_schedule_output_path, "w", encoding="utf-8"):
            pass

    def _append_runtime_topology_record(
            self,
            global_round_idx,
            group_round_idx,
            epoch_idx,
            global_epoch,
            candidate_client_indexes,
            group_to_client_indexes,
            active_client_indexes,
            distributed_client_indexes,
            aggregated,
    ):
        """追加一条可审计的固定候选映射、聚合和参数下发记录。"""
        if self.topology_schedule is None:
            return
        round_topology = self.current_round_topology
        group_client_counts = self._get_mat_group_client_counts(round_topology)
        record = {
            "global_round": int(global_round_idx),
            "group_round": int(group_round_idx),
            "local_epoch": int(epoch_idx),
            "global_epoch": int(global_epoch),
            "mat_topology_index": int(global_epoch),
            "candidate_client_indexes": [int(value) for value in candidate_client_indexes],
            "candidate_client_count": len(candidate_client_indexes),
            "mat_group_client_counts": {
                str(group_index): int(client_count)
                for group_index, client_count in group_client_counts.items()
            },
            "mat_group_to_candidate_slots": {
                str(group_index): [int(value) for value in candidate_slots]
                for group_index, candidate_slots
                in round_topology.copy_groups().items()
            },
            "mat_active_candidate_slots": [
                int(value) for value in round_topology.active_candidate_slots
            ],
            "group_to_client_indexes": {
                str(group_index): [int(value) for value in client_indexes]
                for group_index, client_indexes in group_to_client_indexes.items()
            },
            "active_client_indexes": [int(value) for value in active_client_indexes],
            "active_client_count": len(active_client_indexes),
            "edge_node_ids": {
                str(group_index): int(edge_node_id)
                for group_index, edge_node_id in round_topology.edge_node_ids.items()
            },
            "aggregated": bool(aggregated),
            "model_distribution_scope": self.model_distribution_scope,
            "distributed_client_indexes": [
                int(value) for value in distributed_client_indexes
            ],
            "distributed_client_count": len(distributed_client_indexes),
        }
        with open(self.topology_schedule_output_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
            file_obj.flush()

    def _initialize_metric_output_files(self):
        """
        初始化训练和测试指标文件。

        每次运行开始时清空本次结果目录中的旧指标，避免同一天多次运行时数值混在一起。
        """
        result_dir = getattr(self.args, "result_dir", None)
        if result_dir is None:
            return
        for filename in ["train_acc.txt", "train_loss.txt", "test_acc.txt", "test_loss.txt"]:
            output_path = os.path.join(result_dir, filename)
            with open(output_path, "w", encoding="utf-8"):
                pass

    def _get_probe_output_path(self, arg_name, default_filename):
        """
        生成探针 CSV 输出文件路径。

        相对路径默认落在本次 result 目录，避免从不同终端目录启动时文件散落到其他位置。
        """
        filename = getattr(self.args, arg_name, default_filename)
        if os.path.isabs(filename):
            return filename
        result_dir = getattr(self.args, "result_dir", None)
        if result_dir is not None:
            return os.path.join(result_dir, filename)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    def _open_probe_outputs(self):
        """
        打开三份概率探针和一份标签元数据 CSV，并返回文件对象和 writer 对象。

        文件以覆盖方式写入，保证每次训练得到的是当前实验的完整矩阵。
        """
        output_specs = {
            "client": self._get_probe_output_path("probe_client_pre_file", "probe_client_pre.csv"),
            "edge": self._get_probe_output_path("probe_edge_post_file", "probe_edge_post.csv"),
            "cloud": self._get_probe_output_path("probe_cloud_post_file", "probe_cloud_post.csv"),
            "meta": self._get_probe_output_path("probe_meta_file", "probe_meta.csv"),
        }
        files = {}
        writers = {}
        for key, output_path in output_specs.items():
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            files[key] = open(output_path, "w", newline="", encoding="utf-8")
            writers[key] = csv.writer(files[key])
            logging.info("consensus probe {} csv = {}".format(key, output_path))
        # 标签元数据使用固定表头，便于按 global_epoch 与三份无表头概率 CSV 逐行对齐。
        writers["meta"].writerow([
            "global_epoch",
            "global_round_idx",
            "group_round_idx",
            "local_epoch_idx",
            "probe_source",
            "probe_index",
            "true_label",
        ])
        return files, writers

    def _close_probe_outputs(self, files):
        """
        关闭所有已打开的探针 CSV 文件。
        """
        for file_obj in files.values():
            file_obj.close()

    def _flush_probe_outputs(self, files):
        """
        将当前 epoch 的探针结果立即刷新到磁盘。

        长时间训练时即使中途停止，也能保留已经完成的 epoch 数据。
        """
        for file_obj in files.values():
            file_obj.flush()

    def _get_probe_sample(self, global_epoch):
        """
        从 MNIST 数据集中按全局 epoch 确定性抽取探针图片、标签和样本索引。

        默认使用测试集；如果 probe_source 配置为 train，则改用训练集。
        """
        probe_source = getattr(self.args, "probe_source", "test")
        if probe_source == "train":
            probe_data = self.train_global
            total_num = self.train_data_num_in_total
        else:
            probe_data = self.test_global
            total_num = self.test_data_num_in_total

        if total_num <= 0:
            raise ValueError("Probe source {} has no samples.".format(probe_source))

        target_index = global_epoch % total_num
        seen_num = 0
        for x, labels in probe_data:
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels)

            batch_size = x.shape[0]
            if seen_num + batch_size <= target_index:
                seen_num += batch_size
                continue

            local_index = target_index - seen_num
            # 保留 batch 维度，确保后续模型推理形状与训练阶段一致。
            sample_x = x[local_index: local_index + 1]
            sample_label = int(labels[local_index].item())
            return sample_x, sample_label, target_index

        raise ValueError("Failed to fetch probe sample {} from {} data.".format(target_index, probe_source))

    def _build_probe_metadata_row(
            self, global_epoch, global_round_idx, group_round_idx, local_epoch_idx,
            probe_index, probe_label
    ):
        """构造一行可与概率探针逐轮对齐的结构化真实标签元数据。"""

        return [
            int(global_epoch),
            int(global_round_idx),
            int(group_round_idx),
            int(local_epoch_idx),
            str(getattr(self.args, "probe_source", "test")),
            int(probe_index),
            int(probe_label),
        ]

    def _format_probability_vector(self, probabilities):
        """
        将 10 维概率向量格式化为可放入单个 CSV 单元格的 JSON 字符串。
        """
        vector = [float(value) for value in probabilities]
        return json.dumps(vector, ensure_ascii=False, separators=(",", ":"))

    def _predict_model_proba(self, probe_x, model_state):
        """
        使用任意模型参数对探针图片输出概率向量。

        这里复用 0 号客户端的推理方法，只借用模型结构，不改变任何客户端的持久模型状态。
        """
        return self.client_registry[0].predict_proba(probe_x, model_state=model_state)

    def _build_client_probe_row(self, probe_x, candidate_client_indexes):
        """
        构造聚合前客户端探针 CSV 的一行数据。

        MATLAB 模式行内固定包含实验开始时抽取的 37 个候选客户端，
        每列在全部 epoch 中始终对应同一个真实客户端。
        """
        if (
                self.args.group_method == "matlab"
                and len(candidate_client_indexes) != self.args.client_num_per_round
        ):
            raise ValueError(
                "客户端探针列数 {} 与 client_num_per_round={} 不一致".format(
                    len(candidate_client_indexes), self.args.client_num_per_round
                )
            )
        if (
                self.args.group_method == "matlab"
                and list(candidate_client_indexes) != self.fixed_candidate_client_indexes
        ):
            raise ValueError("MATLAB 客户端探针必须使用固定候选顺序")
        row = []
        for client_idx in candidate_client_indexes:
            probabilities = self.client_registry[client_idx].predict_proba(probe_x)
            row.append(self._format_probability_vector(probabilities))
        return row

    def _build_edge_probe_row(self, probe_x, edge_model_states):
        """
        构造边缘模型聚合后探针 CSV 的一行数据。

        行内共 group_num 列，未在当前 .mat 轮次启用的边缘组留空。
        """
        row = ["" for _ in range(self.args.group_num)]
        for group_idx, model_state in edge_model_states.items():
            probabilities = self._predict_model_proba(probe_x, model_state)
            row[group_idx] = self._format_probability_vector(probabilities)
        return row

    def _build_cloud_probe_row(self, probe_x, w_global):
        """
        构造云模型聚合后探针 CSV 的一行数据。

        行内只有一列，保存云模型对当前探针图片的 10 维概率向量。
        """
        probabilities = self._predict_model_proba(probe_x, w_global)
        return [self._format_probability_vector(probabilities)]

    def _predict_model_proba_batch(self, probe_x, model_state):
        """借用客户端模型结构，对固定探针批次执行一次或少量分块前向推理。"""
        inference_batch_size = int(
            getattr(self.args, "probe_inference_batch_size", probe_x.shape[0])
        )
        return np.asarray(
            self.client_registry[0].predict_proba_batch(
                probe_x,
                model_state=model_state,
                inference_batch_size=inference_batch_size,
            ),
            dtype=np.float32,
        )

    def _build_client_probe_tensor(self, probe_x, candidate_client_indexes):
        """按固定候选顺序批量推理全部探针，返回 [候选, 探针, 类别]。"""
        expected_candidates = self._get_fixed_probe_candidate_ids()
        if [int(value) for value in candidate_client_indexes] != expected_candidates:
            raise ValueError("NPZ 客户端探针必须始终使用实验开始时固定的候选顺序。")
        inference_batch_size = int(
            getattr(self.args, "probe_inference_batch_size", probe_x.shape[0])
        )
        probability_batches = []
        for client_idx in candidate_client_indexes:
            # 每个客户端只加载一次本地模型；正式配置100张图片恰好一次前向推理。
            probability_batches.append(np.asarray(
                self.client_registry[client_idx].predict_proba_batch(
                    probe_x,
                    inference_batch_size=inference_batch_size,
                ),
                dtype=np.float32,
            ))
        return np.stack(probability_batches, axis=0)

    def _build_edge_probe_tensor(self, probe_x, edge_model_states):
        """批量推理当轮边缘模型，未启用槽位保留为整块 NaN。"""
        probe_count = int(probe_x.shape[0])
        class_count = int(np.max(self.fixed_probe_set.true_labels)) + 1
        probabilities = np.full(
            (int(self.args.group_num), probe_count, class_count),
            np.nan,
            dtype=np.float32,
        )
        for group_idx, model_state in edge_model_states.items():
            probabilities[int(group_idx)] = self._predict_model_proba_batch(
                probe_x, model_state
            )
        return probabilities

    def _build_cloud_probe_tensor(self, probe_x, w_global):
        """批量推理当前云模型，返回 [探针, 类别] 概率矩阵。"""
        return self._predict_model_proba_batch(probe_x, w_global)

    def _get_distribution_client_indexes(self, active_client_indexes):
        """根据 YAML 下发范围返回需要同步云模型的真实客户端编号。"""
        if self.model_distribution_scope == "all":
            return list(range(self.args.client_num_in_total))
        if not active_client_indexes:
            return []
        return list(active_client_indexes)

    def _sync_clients_to_global(self, client_indexes, w_global):
        """将指定客户端的持久本地模型替换为最新云模型。"""
        for client_idx in client_indexes:
            self.client_registry[client_idx].set_local_model_state(w_global)

    def _evaluate_all_client_local_models(self, global_epoch):
        """
        使用全部真实客户端各自的持久模型和数据分区计算总体指标。

        主训练流程会在全量下发后调用本方法。每个客户端评估前都会显式加载自己的
        local_model_state，最终准确率由父类统一按总正确数除以总样本数计算。
        """
        logging.info(
            "################evaluate_all_client_local_models : %s",
            global_epoch,
        )
        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}
        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        for client_idx in range(self.args.client_num_in_total):
            client = self.client_registry[client_idx]
            if client.local_test_data is None:
                continue

            # 分别用该真实客户端自己的模型评估自己的训练集和测试集。
            train_local_metrics = client.evaluate_local_model(False)
            test_local_metrics = client.evaluate_local_model(True)
            train_metrics["num_samples"].append(train_local_metrics["test_total"])
            train_metrics["num_correct"].append(train_local_metrics["test_correct"])
            train_metrics["losses"].append(train_local_metrics["test_loss"])
            test_metrics["num_samples"].append(test_local_metrics["test_total"])
            test_metrics["num_correct"].append(test_local_metrics["test_correct"])
            test_metrics["losses"].append(test_local_metrics["test_loss"])

        return self._summarize_local_test_metrics(
            global_epoch, train_metrics, test_metrics
        )

    def _uses_direct_cloud_aggregation(self):
        """判断当前 MATLAB 场景是否为无边缘层的普通 FL。"""
        return (
            self.topology_schedule is not None
            and self.topology_schedule.architecture == "fl"
        )

    def _collect_direct_cloud_inputs(self, active_client_indexes):
        """收集普通 FL 本轮参与客户端的样本数和本地模型。"""
        cloud_inputs = []
        for client_idx in active_client_indexes:
            client = self.client_registry[client_idx]
            cloud_inputs.append((client.get_sample_number(), client.get_local_model_state()))
        return cloud_inputs

    def _collect_hierarchical_cloud_inputs(self, group_to_client_indexes):
        """执行边缘聚合并返回云聚合输入和边缘模型。"""
        cloud_inputs = []
        edge_model_states = {}
        for group_idx in sorted(group_to_client_indexes.keys()):
            sampled_client_indexes = group_to_client_indexes[group_idx]
            group = self.group_dict[group_idx]
            w_group = group.aggregate_client_states(sampled_client_indexes)
            group_sample_number = group.get_sample_number(sampled_client_indexes)
            edge_model_states[group_idx] = w_group
            cloud_inputs.append((group_sample_number, w_group))
        return cloud_inputs, edge_model_states

    def train(self):
        """
        执行“固定候选、MAT 活跃训练、层级聚合、全量下发与客户端评估”的流程。

        每个展平后的本地 epoch 都单独读取一行 MAT 拓扑并完成一次评估。
        """
        w_global = copy.deepcopy(self.model.state_dict())
        self._setup_result_dir()
        self._initialize_metric_output_files()
        probe_enabled = self._is_consensus_probe_enabled()
        probe_output_format = self._get_probe_output_format()
        probe_files = {}
        probe_writers = {}
        batch_probe_recorder = None
        self.fixed_probe_set = None
        if probe_enabled:
            if probe_output_format == "npz":
                # 固定探针仅准备一次，四种方案在相同数据与种子下会得到相同内容哈希。
                self.fixed_probe_set = self._prepare_fixed_probe_set()
                batch_probe_recorder = self._create_batch_probe_recorder(
                    self.fixed_probe_set
                )
                logging.info(
                    "fixed probe count = %s, hash = %s",
                    self.fixed_probe_set.sample_count,
                    self.fixed_probe_set.content_hash,
                )
            else:
                probe_files, probe_writers = self._open_probe_outputs()

        try:
            for global_round_idx in range(self.args.comm_round):
                logging.info("################Global Communication Round : {}".format(global_round_idx))
                for group_round_idx in range(self.args.group_comm_round):
                    for epoch_idx in range(self.args.epochs):
                        global_epoch = self._calculate_global_epoch(
                            global_round_idx, group_round_idx, epoch_idx
                        )
                        if self.args.group_method == "matlab":
                            candidate_client_indexes = list(
                                self.fixed_candidate_client_indexes
                            )
                            # MAT 槽位直接映射固定候选，不再进行任何 epoch 级随机采样。
                            group_to_client_indexes = self._build_round_groups(global_epoch)
                        else:
                            group_to_client_indexes = self._build_round_groups(global_epoch)
                            candidate_client_indexes = self._get_active_client_indexes(
                                group_to_client_indexes
                            )
                        active_client_indexes = self._get_active_client_indexes(
                            group_to_client_indexes
                        )
                        if self.args.group_method == "matlab":
                            # 正式 MATLAB 模式只训练当前行实际启用的固定候选客户端。
                            self._train_active_clients_one_epoch(
                                global_round_idx,
                                group_round_idx,
                                epoch_idx,
                                active_client_indexes,
                            )
                        else:
                            # 旧分组模式继续保持全部真实客户端先训练的原有行为。
                            self._train_one_epoch_all_clients(
                                global_round_idx, group_round_idx, epoch_idx
                            )

                        client_probe_probabilities = None
                        if probe_enabled and probe_output_format == "npz":
                            probe_x = self.fixed_probe_set.inputs
                            client_probe_probabilities = self._build_client_probe_tensor(
                                probe_x, candidate_client_indexes
                            )
                        elif probe_enabled:
                            probe_x, probe_label, probe_index = self._get_probe_sample(global_epoch)
                            logging.info(
                                "consensus probe global_epoch = {}, label = {}".format(global_epoch, probe_label)
                            )
                            probe_writers["client"].writerow(
                                self._build_client_probe_row(
                                    probe_x, candidate_client_indexes
                                )
                            )
                            # 旧 CSV 仍逐轮保存真实标签，保证历史分析入口不受影响。
                            probe_writers["meta"].writerow(
                                self._build_probe_metadata_row(
                                    global_epoch=global_epoch,
                                    global_round_idx=global_round_idx,
                                    group_round_idx=group_round_idx,
                                    local_epoch_idx=epoch_idx,
                                    probe_index=probe_index,
                                    probe_label=probe_label,
                                )
                            )
                        else:
                            probe_x = None

                        # HFL 先形成边缘模型，普通 FL 则把真实参与客户端模型直接交给云端。
                        if self._uses_direct_cloud_aggregation():
                            cloud_inputs = self._collect_direct_cloud_inputs(active_client_indexes)
                            edge_model_states = {}
                        else:
                            cloud_inputs, edge_model_states = self._collect_hierarchical_cloud_inputs(
                                group_to_client_indexes
                            )

                        if not cloud_inputs:
                            logging.warning(
                                "Global epoch {} has no active model weights; keep previous global model.".format(
                                    global_epoch
                                )
                            )
                            if probe_enabled and probe_output_format == "npz":
                                # 空聚合轮仍提交完整记录：边缘为空，云模型沿用上一轮。
                                edge_probe_probabilities = self._build_edge_probe_tensor(
                                    probe_x, {}
                                )
                                cloud_probe_probabilities = self._build_cloud_probe_tensor(
                                    probe_x, w_global
                                )
                                batch_probe_recorder.record_epoch(
                                    global_epoch=global_epoch,
                                    client_probabilities=client_probe_probabilities,
                                    edge_probabilities=edge_probe_probabilities,
                                    cloud_probabilities=cloud_probe_probabilities,
                                    active_client_ids=active_client_indexes,
                                )
                            elif probe_enabled:
                                # 空聚合轮没有边缘模型，边缘 CSV 保留一行空单元格以维持矩阵行数。
                                probe_writers["edge"].writerow(["" for _ in range(self.args.group_num)])
                                # 云模型沿用上一轮 w_global，使云 CSV 在每个全局 epoch 都有一行。
                                probe_writers["cloud"].writerow(self._build_cloud_probe_row(probe_x, w_global))
                                self._flush_probe_outputs(probe_files)
                            # 空参与 epoch 沿用上一云模型，仍按配置向全部客户端下发并逐客户端评估。
                            distributed_client_indexes = self._get_distribution_client_indexes(
                                active_client_indexes
                            )
                            self._sync_clients_to_global(
                                distributed_client_indexes, w_global
                            )
                            self._evaluate_all_client_local_models(global_epoch)
                            self._append_runtime_topology_record(
                                global_round_idx,
                                group_round_idx,
                                epoch_idx,
                                global_epoch,
                                candidate_client_indexes,
                                group_to_client_indexes,
                                active_client_indexes,
                                distributed_client_indexes=distributed_client_indexes,
                                aggregated=False,
                            )
                            continue

                        edge_probe_probabilities = None
                        if probe_enabled and probe_output_format == "npz":
                            edge_probe_probabilities = self._build_edge_probe_tensor(
                                probe_x, edge_model_states
                            )
                        elif probe_enabled:
                            if self._uses_direct_cloud_aggregation():
                                # 普通 FL 没有边缘模型，保留一个空单元格维持按 epoch 对齐。
                                probe_writers["edge"].writerow(
                                    ["" for _ in range(self.args.group_num)]
                                )
                            else:
                                probe_writers["edge"].writerow(
                                    self._build_edge_probe_row(probe_x, edge_model_states)
                                )

                        # 云端只聚合 MAT 当前行实际启用客户端贡献的模型。
                        w_global = self._aggregate(cloud_inputs)

                        if probe_enabled and probe_output_format == "npz":
                            cloud_probe_probabilities = self._build_cloud_probe_tensor(
                                probe_x, w_global
                            )
                            # 三层数据全部成功后才提交该 epoch，异常时不会产生半条记录。
                            batch_probe_recorder.record_epoch(
                                global_epoch=global_epoch,
                                client_probabilities=client_probe_probabilities,
                                edge_probabilities=edge_probe_probabilities,
                                cloud_probabilities=cloud_probe_probabilities,
                                active_client_ids=active_client_indexes,
                            )
                        elif probe_enabled:
                            probe_writers["cloud"].writerow(self._build_cloud_probe_row(probe_x, w_global))
                            self._flush_probe_outputs(probe_files)

                        distributed_client_indexes = self._get_distribution_client_indexes(
                            active_client_indexes
                        )
                        self._sync_clients_to_global(distributed_client_indexes, w_global)

                        self._evaluate_all_client_local_models(global_epoch)
                        self._append_runtime_topology_record(
                            global_round_idx,
                            group_round_idx,
                            epoch_idx,
                            global_epoch,
                            candidate_client_indexes,
                            group_to_client_indexes,
                            active_client_indexes,
                            distributed_client_indexes,
                            aggregated=True,
                        )
        finally:
            if batch_probe_recorder is not None:
                # 正常结束和异常退出都保存最后一个完整 epoch 前缀。
                batch_probe_recorder.close()
            elif probe_enabled:
                self._close_probe_outputs(probe_files)
