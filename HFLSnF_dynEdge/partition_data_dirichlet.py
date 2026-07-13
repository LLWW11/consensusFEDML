import numpy as np
import pandas as pd
import params
from Client import Client
import matplotlib.pyplot as plt
from MNISTReader import MNISTImageReader as ds_r
from MNISTReader import MNISTLabelReader as lb_r


# 具体的划分函数
def split_index(labels, n_clients, split_policy, alpha):
    '''
    具体的划分函数
    :param target: 数据集对应的标签下标
    :param n_clients:
    :param split_policy:
    :param alpha:
    :return:
        client_idcs: {list：3}每个客户端被分配到的数据的索引，是按0,1,...,9顺序排的
        df_client: {ndarray:60000,}表示数据集中该行数据分给哪个客户端
    '''
    # 设置随机数种子，保证相同的数据拆分,可获得结果的复现
    np.random.seed(5)

    # 记录总类别数
    n_classes = labels.max() + 1

    # 记录每个类别对应的样本下标, np.argwhere(labels == y)：返回数组中等于的下标。
    class_idcs = [np.argwhere(labels == y).flatten()
                  for y in range(n_classes)]

    if split_policy == "iid":
        label_distribution = [[1.0 / n_clients for _ in range(n_clients)]
                              for _ in range(n_classes)]
    elif split_policy == "non-iid":
        # alpha: 对应分布函数中的参数向量 α ，长度为n_clients
        # size:
        # 返回：
        # out: 采出的样本，大小为(size, k) 。
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    # 定义变量作最后的返回值
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    # zip：将对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    for c, fracs in zip(class_idcs, label_distribution):
        # split_indexs 记录划分时的间断点
        # np.cumsum: 累加
        split_indexs = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
        # for i, idcs 为遍历第i个client对应样本集合的索引
        # np.split按照比例将类别为k的样本划分为了N个子集
        # enumerate：将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        # 同时列出数据和数据下标，一般用在 for 循环当中。
        for i, idcs in enumerate(np.split(c, split_indexs)):
            client_idcs[i] += [idcs]

    # 原来client_idcs{list:3}, 每个list里又是{list:10}
    # 结束后client_idcs{list:3}
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    df_client = np.full((params.DATASET_SIZE_USED_TO_TRAIN), -1)
    for i, indexs in enumerate(client_idcs):
        df_client[indexs] = i

    return client_idcs, df_client


def show_split_img(labels, n_clients, client_idcs, num_cls, label_name):
    '''
    绘图函数
    :param labels: 数据集对应的标签下标
    :param n_clients: 客户端数量
    :param client_idcs: 每个客户端被分配到的数据的索引，是按0,1,...,9顺序排的
    :param num_cls: 标签个数
    :param target_name: 画图时的横坐标名字
    :return:
    '''
    # 展示不同client的不同label的数据分布
    plt.figure(figsize=(20, 3))
    # x：指定要绘制直方图的数据
    # bins：指定直方图条形的个数
    # 先把第0个用户分到的都画最下面，再往上画
    plt.hist([labels[idc] for idc in client_idcs], stacked=True,
             bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
             label=["Client {}".format(i) for i in range(n_clients)], rwidth=0.5)
    plt.xticks(np.arange(num_cls), label_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dr = ds_r(params.MNIST_TRAIN_IMAGES_PATH)
    lr = lb_r(params.MNIST_TRAIN_LABELS_PATH)
    dr.open()
    lr.open()
    index, data = dr.read(params.DATASET_SIZE_USED_TO_TRAIN)  # data:ndarray:(60000,28,28),index:range(1,60001)
    index, label = lr.read(params.DATASET_SIZE_USED_TO_TRAIN)  # label:ndarray(60000,),index:range(1,60001)
    data = data[..., np.newaxis] / 255.0  # 添加一个维度，data:(6000,28,28,1),还归一化了一下
    dr.close()
    lr.close()

    client_idcs, df_client = split_index(label, params.client_num_in_total, 'iid', 0.5)
    label_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    show_split_img(label, params.client_num_in_total, client_idcs, label.max() + 1, label_name)

    clients = []
    # 创建20个Client
    for _ in range(params.client_num_in_total):  # _就是平时i的作用，它不在意变量的值，只用于循环遍历n次
        clients.append(Client())

    # 每个类别的前面几个，都被分给了客户端0; 然后再依次往下
    for i in range(params.client_num_in_total):
        # 客户端i被分配到的数据的索引，是按数据集原来的顺序排的，不是按0,1,...,9顺序排的
        index_client_i = np.argwhere(df_client == i).reshape(-1)
        exec('data{} = data[index_client_i, :, :, :]'.format(i))
        exec('label{} = label[index_client_i]'.format(i))

