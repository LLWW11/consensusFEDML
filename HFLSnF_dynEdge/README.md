# 动态拓扑联邦学习实验

本目录使用 `matlab/result-U-6fixedge_epoch200.mat` 中保存的拓扑组人数，在 MNIST 上运行四组固定边缘对照实验：

- HFL-SnF-fixed
- HFL-noSnF-fixed
- FL-SnF
- FL-noSnF

MATLAB 中的 37 个物理客户端只表示候选槽位，不再直接对应真实数据客户端。Python 训练池包含 200 个真实客户端，每个本地 epoch 都会重新进行两级随机采样。

## 每个 epoch 的执行流程

1. 全部 200 个客户端基于各自持久本地模型完成一个本地 epoch。
2. 使用独立且可复现的随机数生成器，从 200 人中均匀、无放回抽取 37 人。
3. 读取 MAT 当前 epoch 的各边缘组人数，随机打乱候选 37 人并按组人数依次切分。
4. 候选 37 人执行聚合前客户端探针，最终二次采样客户端进入边缘或云端聚合。
5. HFL 先执行组内样本数加权聚合，再执行云端聚合；FL 直接执行云端聚合。
6. 根据 `model_distribution_scope` 下发云模型，并使用最终云模型在完整训练集和测试集上计算指标。

`model_distribution_scope: "active"` 只向最终二次采样客户端下发云模型；设置为 `"all"` 时，在存在有效聚合结果的 epoch 向全部 200 个客户端下发。MAT 参与人数为 0 时沿用上一云模型，不聚合也不下发。

## 运行环境

本机训练环境为 Conda 的 `py37`。应通过 `conda run` 启动，以便正确加载 OpenSSL 和 FedML 依赖：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf fedml_config.yaml
```

## 单独运行四组实验

```powershell
# HFL-SnF-fixed
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_hfl_snf_fixed_u05.yaml

# HFL-noSnF-fixed
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_hfl_no_snf_fixed_u05.yaml

# FL-SnF
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_fl_snf_u05.yaml

# FL-noSnF
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_fl_no_snf_u05.yaml
```

也可以依次运行全部四组 200 轮实验：

```powershell
.\run_fixed_u05_experiments.ps1
```

## 冒烟测试

HFL-SnF-fixed 的 3 轮冒烟配置：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_smoke_hfl_snf_fixed_u05.yaml
```

FL-noSnF 的 1 轮直接云聚合验证配置：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf configs/fedml_config_smoke_fl_no_snf_u05.yaml
```

## 单元测试

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest -v test_topology_schedule.py test_hierarchical_sampling.py
```

## 输出文件

每个实验写入独立的 `result/` 子目录，包含：

- `train_acc.txt`、`train_loss.txt`
- `test_acc.txt`、`test_loss.txt`
- `probe_client_pre.csv`
- `probe_edge_post.csv`
- `probe_cloud_post.csv`
- `topology_metadata.json`
- `topology_schedule.jsonl`

HFL fixed 的三份探针矩阵为 `200×37`、`200×6`、`200×1`。客户端探针的 37 列按 `topology_schedule.jsonl` 中的 `candidate_client_indexes` 顺序对应真实客户端。普通 FL 没有边缘模型，因此 edge CSV 保留空列，用于保持 epoch 对齐。

`topology_schedule.jsonl` 每行对应一个展平后的本地 epoch，主要记录：

- 通信轮、组内通信轮、本地 epoch 和 `global_epoch`；
- 首次采样的 37 个真实客户端；
- MAT 各组人数、二次采样后的真实客户端分组和最终参与者；
- 是否完成聚合、参数下发范围和实际接收客户端。

## 文件说明

文件名带 `_test` 的模块主要从 FedML 对应实现调整而来；其余模块为本项目新增或重写的实验代码。
