import logging

from client_test import HFLClient
from fedavg_test import FedAvgAPI


class Group(FedAvgAPI):
    """
    层次联邦中的边缘组。

    组对象负责调度组内客户端训练，并把组内客户端模型聚合为边缘模型。
    """

    def __init__(
        self,
        idx,
        total_client_indexes,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        args,
        device,
        model,
        model_trainer,
        client_registry=None,
    ):
        """
        初始化边缘组并绑定组内客户端。

        client_registry 不为空时复用全局唯一客户端对象；为空时保留旧逻辑，按组创建客户端。
        """
        self.idx = idx
        self.args = args
        self.device = device
        self.client_dict = {}
        self.train_data_local_num_dict = train_data_local_num_dict
        for client_idx in total_client_indexes:
            if client_registry is not None:
                self.client_dict[client_idx] = client_registry[client_idx]
            else:
                self.client_dict[client_idx] = HFLClient(
                    client_idx,
                    train_data_local_dict[client_idx],
                    test_data_local_dict[client_idx],
                    train_data_local_num_dict[client_idx],
                    args,
                    device,
                    model,
                    model_trainer,
                )

    def get_sample_number(self, sampled_client_indexes):
        """
        计算指定客户端集合对应的训练样本总数。
        """
        self.group_sample_number = 0
        for client_idx in sampled_client_indexes:
            self.group_sample_number += self.train_data_local_num_dict[client_idx]
        return self.group_sample_number

    def train(self, global_round_idx, w, sampled_client_indexes):
        """
        训练本轮被选中的组内客户端并返回边缘聚合模型。

        参数 w 仅保留旧接口兼容；当前持久本地模型流程下，客户端会从自己的 local_model_state 继续训练。
        """
        sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes]
        w_group_list = []
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # 参与本轮聚合的客户端从各自持久本地模型继续训练。
            for client in sampled_client_list:
                w_local_list = client.train(global_round_idx, group_round_idx)
                for global_epoch, w_local in w_local_list:
                    if global_epoch not in w_locals_dict:
                        w_locals_dict[global_epoch] = []
                    w_locals_dict[global_epoch].append((client.get_sample_number(), w_local))

            # 组内模型聚合，得到边缘模型。
            for global_epoch in sorted(w_locals_dict.keys()):
                w_locals = w_locals_dict[global_epoch]
                w_group_list.append((global_epoch, self._aggregate(w_locals)))
        return w_group_list

    def aggregate_client_states(self, sampled_client_indexes):
        """
        从已经完成本地训练的客户端状态中执行边缘聚合。

        该方法不会再次训练客户端，只读取每个客户端当前的 local_model_state。
        """
        w_locals = []
        for client_idx in sampled_client_indexes:
            client = self.client_dict[client_idx]
            # 只收集客户端已经训练完成的本地模型，避免在采样后重复训练。
            w_locals.append((client.get_sample_number(), client.get_local_model_state()))
        return self._aggregate(w_locals)
