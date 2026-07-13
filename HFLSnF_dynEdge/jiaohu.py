import os
from scipy.io import loadmat
import numpy as np


def get_mat_file_path(filename):
    """
    获取 matlab/ 文件夹下 .mat 文件的绝对路径。

    该函数用于适配当前所有 .mat 数据集中移动到 matlab/ 目录后的文件结构。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "matlab", filename)


if __name__ == "__main__":
    filename = get_mat_file_path('matoutput.mat')
    data = loadmat(filename)
    total_group_num = data['group_num']
    group_num_index = data['client_num']
    total_client_num_per_round = np.sum(group_num_index, axis=1)
    for global_round_idx in range(2):
        each_group_num = total_group_num[global_round_idx]
        each_client_num = total_client_num_per_round[global_round_idx]
        each_group_num_index = group_num_index[global_round_idx]

