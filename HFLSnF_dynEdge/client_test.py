import copy

import torch
import torch.nn as nn

from fedml.simulation.sp.fedavg.client import Client


class HFLClient(Client):
    """
    层次联邦客户端。

    该类在 FedML 客户端基础上增加跨轮保留的本地模型状态，用于支持未参与聚合客户端继续本地更新。
    """

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model,
                 model_trainer):
        """
        初始化客户端数据、训练配置和本地模型状态。

        local_model_state 从初始模型参数深拷贝得到，后续每轮本地训练都会在该状态上继续更新。
        """
        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                         model_trainer)
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model = model
        self.model_trainer = model_trainer
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.local_model_state = self._clone_model_state(model.state_dict())

    def _clone_model_state(self, model_state):
        """
        深拷贝模型参数，避免不同客户端共享同一个 state_dict 引用。
        """
        return copy.deepcopy(model_state)

    def get_local_model_state(self):
        """
        读取当前客户端持久保存的本地模型参数。

        返回深拷贝结果，避免调用方修改客户端内部状态。
        """
        return self._clone_model_state(self.local_model_state)

    def set_local_model_state(self, model_state):
        """
        将客户端本地模型状态替换为指定聚合模型参数。

        该方法用于云聚合完成后，把参与本轮聚合的客户端同步到最新全局模型。
        """
        self.local_model_state = self._clone_model_state(model_state)

    def evaluate_local_model(self, use_test_dataset):
        """
        使用当前客户端持久本地模型评估自己的训练分区或测试分区。

        use_test_dataset 为 True 时评估本地测试集，否则评估本地训练集。
        评估前会显式把该客户端的持久状态加载到共享模型和模型训练器，
        避免误用上一个客户端或云端评估遗留的参数。
        """
        model_state = self.get_local_model_state()
        # HFLClient 与 FedML 模型训练器共享同一模型，但两处都显式同步可避免状态歧义。
        self.model.load_state_dict(model_state)
        self.model_trainer.set_model_params(model_state)
        return self.local_test(use_test_dataset)

    def _create_optimizer(self):
        """
        根据配置创建客户端本地训练优化器。

        该方法集中保留原来的 SGD/Adam 选择逻辑，供单 epoch 训练和旧接口复用。
        """
        if self.args.client_optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.wd,
            amsgrad=True,
        )

    def _get_global_epoch(self, global_round_idx, group_round_idx, epoch_idx):
        """
        计算当前本地 epoch 对应的全局 epoch 编号。

        全局编号用于把多轮通信、多次组内通信和本地 epoch 展平成一条时间轴。
        """
        return (
            global_round_idx * self.args.group_comm_round * self.args.epochs
            + group_round_idx * self.args.epochs
            + epoch_idx
        )

    def train_one_epoch(self, global_round_idx, group_round_idx, epoch_idx, w=None):
        """
        从客户端当前本地模型状态继续训练一个 epoch。

        如果传入 w，则先用 w 重置本地模型状态；训练完成后刷新持久本地模型并返回全局 epoch 与模型参数。
        """
        if w is not None:
            # 保留旧调用兼容性：显式传入模型时，先把本地缓存同步到该模型。
            self.set_local_model_state(w)

        self.model.load_state_dict(self.local_model_state)
        self.model.to(self.device)
        self.model.train()

        optimizer = self._create_optimizer()
        for x, labels in self.local_training_data:
            x, labels = x.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            loss.backward()  # 计算梯度。
            optimizer.step()  # 更新本地模型参数。

        global_epoch = self._get_global_epoch(global_round_idx, group_round_idx, epoch_idx)
        # 每个 epoch 后刷新客户端持久本地模型状态。
        self.local_model_state = self._clone_model_state(self.model.state_dict())
        return global_epoch, self.get_local_model_state()

    def predict_proba(self, sample_x, model_state=None):
        """
        使用客户端模型对单张 MNIST 图片输出 0 到 9 的概率分布。

        model_state 为空时使用客户端当前持久本地模型；否则使用传入的边缘模型或云模型参数。
        """
        if model_state is None:
            model_state = self.local_model_state

        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            probe_x = sample_x.to(self.device)
            # 探针样本必须带 batch 维度，便于兼容当前训练入口的模型形状。
            if probe_x.dim() == 1:
                probe_x = probe_x.unsqueeze(0)
            logits = self.model(probe_x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.squeeze(0).detach().cpu().tolist()

    def train(self, global_round_idx, group_round_idx, w=None):
        """
        从客户端当前本地模型状态继续训练配置中的多个 epoch。

        该方法保留旧接口兼容性，内部逐次调用 train_one_epoch。
        """
        if w is not None:
            # 保留旧调用兼容性：显式传入模型时，只在第一个 epoch 前同步一次。
            self.set_local_model_state(w)

        w_list = []
        for epoch in range(self.args.epochs):
            global_epoch, model_state = self.train_one_epoch(global_round_idx, group_round_idx, epoch)
            w_list.append((global_epoch, model_state))
        return w_list
