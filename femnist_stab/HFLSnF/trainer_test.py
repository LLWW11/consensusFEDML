import logging

import numpy as np

from client_test import HFLClient
from group_test import Group
from fedavg_test import FedAvgAPI


class HierarchicalTrainer(FedAvgAPI):
    def _setup_clients(
            self,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == "random":
            self.group_indexes = np.random.randint(
                0, self.args.group_num, self.args.client_num_in_total
            )

            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)

        elif self.args.group_method == 'average':
            # 这段产生0-group_num范围内的整数，共计client_num_in_total个
            # make sure for each comparison, we are selecting the same clients each round
            # np.random.seed(10)
            # 不重复的产生args.group_num个(0, args.group_num)
            random_ints = np.random.choice(self.args.group_num, self.args.group_num, replace=False)
            # 重复client_num_per_round // group_num次
            result1 = np.repeat(random_ints, self.args.client_num_in_total // self.args.group_num)
            # 最后补充剩下的几个分组
            result2 = np.random.choice(self.args.group_num, self.args.client_num_in_total % self.args.group_num,
                                       replace=False)
            self.group_indexes = np.concatenate([result1, result2], axis=0)
            # np.random.shuffle(self.group_indexes)
            # my_group_indexes.sort()

            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)

        else:
            raise Exception(self.args.group_method)

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
                self.model_trainer
            )

        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        client_idx = -1
        self.client_list = [
            HFLClient(
                client_idx,
                train_data_local_dict[0],
                test_data_local_dict[0],
                train_data_local_num_dict[0],
                self.args,
                self.device,
                self.model,
                self.model_trainer
            )
        ]
        logging.info("############setup_clients (END)#############")

    # 系统库里原来的函数
    def _client_sampling(
            self, global_round_idx, client_num_in_total, client_num_per_round
    ):
        sampled_client_indexes = super()._client_sampling(  # 从client_num_in_total个数据中，选择client_num_per_round个客户端
            global_round_idx, client_num_in_total, client_num_per_round
        )
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]  # client_num_in_total个[0-group_num-1]
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        logging.info(
            "client_indexes of each group = {}".format(group_to_client_indexes)
        )
        return group_to_client_indexes

    # 自己写的客户端采样, 可以保证每个group的客户端数量基本相等
    def _client_sampling_average(
            self, global_round_idx, client_num_in_total, client_num_per_round
    ):
        # 新添，现在的group_to_client_indexes为每个group中客户端的索引
        group_to_client_indexes = {}
        for client_idx, group_idx in enumerate(self.group_indexes):
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)

        sampled_client_indexes = super()._client_sampling_average(  # 从client_num_in_total个数据中，选择client_num_per_round个客户端
            global_round_idx, client_num_in_total, client_num_per_round, group_to_client_indexes
        )
        group_to_client_indexes = {}
        for client_idx in sampled_client_indexes:
            group_idx = self.group_indexes[client_idx]  # client_num_in_total个[0-group_num-1]
            if not group_idx in group_to_client_indexes:
                group_to_client_indexes[group_idx] = []
            group_to_client_indexes[group_idx].append(client_idx)
        logging.info(
            "client_indexes of each group = {}".format(group_to_client_indexes)
        )
        return group_to_client_indexes

    # # 自己写的客户端采样，无用
    # def _client_sampling(
    #         self, global_round_idx, client_num_in_total, client_num_per_round
    # ):
    #     sampled_client_indexes = super()._client_sampling(  # 从client_num_in_total个数据中，选择client_num_per_round个客户端
    #         global_round_idx, client_num_in_total, client_num_per_round
    #     )
    #
    #     group_to_client_indexes = {}
    #     for idx in range(len(sampled_client_indexes)):
    #         group_idx = self.group_indexes[idx]  # client_num_in_total个[0-group_num-1]
    #         if not group_idx in group_to_client_indexes:
    #             group_to_client_indexes[group_idx] = []
    #         group_to_client_indexes[group_idx].append(sampled_client_indexes[idx])
    #     logging.info(
    #         "client_indexes of each group = {}".format(group_to_client_indexes)
    #     )
    #     return group_to_client_indexes

    def train(self):
        w_global = self.model.state_dict()  # w_global={OrderedDict:2}=weights(10,784),bias(10,)
        for global_round_idx in range(self.args.comm_round):  # 总轮次
            logging.info(
                "################Global Communication Round : {}".format(
                    global_round_idx
                )
            )
            if self.args.group_method == 'random':
                group_to_client_indexes = self._client_sampling(
                    global_round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round,
                )  # {dict:2}:{0: [993, 859, 553, 672], 1: [298]}
            elif self.args.group_method == 'average':
                group_to_client_indexes = self._client_sampling_average(
                    global_round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round,
                )  # {dict:2}:{0: [993, 859, 553, 672], 1: [298]}

            # train each group
            w_groups_dict = {}
            for group_idx in sorted(group_to_client_indexes.keys()):
                sampled_client_indexes = group_to_client_indexes[group_idx]  # {list:4}[993, 859, 553, 672]
                group = self.group_dict[group_idx]
                w_group_list = group.train(
                    global_round_idx, w_global, sampled_client_indexes
                )
                for global_epoch, w in w_group_list:  # w_group_list大小取决于group_comm_round
                    if not global_epoch in w_groups_dict:
                        w_groups_dict[global_epoch] = []
                    w_groups_dict[global_epoch].append(
                        (group.get_sample_number(sampled_client_indexes), w)
                    )

            # aggregate group weights into the global weight
            for global_epoch in sorted(w_groups_dict.keys()):  # w_group_dict大小取决于group_comm_round
                w_groups = w_groups_dict[global_epoch]
                w_global = self._aggregate(w_groups)

                # # evaluate performance
                # if (
                #         global_epoch % self.args.frequency_of_the_test == 0
                #         or global_epoch
                #         == self.args.comm_round
                #         * self.args.group_comm_round
                #         * self.args.epochs
                #         - 1
                # ):
                #     self.model.load_state_dict(w_global)
                #     self._local_test_on_all_clients(global_epoch)

            self.model.load_state_dict(w_global)
            self._local_test_on_all_clients(global_epoch)
