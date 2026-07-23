# FedGCN 的 FedML 核心复现

## 模块定位

本目录把 `1paperAbout/Collaborative_Reasoning` 中基于 FedGCN 的图节点分类实验整理为独立、可配置的 FedML 单进程模拟程序。它不会替换 `HFLSnF_dynEdge` 的 MNIST CNN，也不会读取 `Topo_opt` 生成的动态通信拓扑。

两个项目中的“图”含义不同：

- 本模块中的图是 Cora 或 Citeseer 知识图，节点特征和节点类别用于训练 GCN。
- 原 HFL 项目中的图是通信拓扑，用于决定客户端参与、边缘分组和模型传输路径。

本模块复用的是“多设备本地训练 GCN，再聚合模型参数”的 FedGCN 思路，而不是把知识图直接解释成通信拓扑。

## 训练流程

1. 读取 Cora 或 Citeseer 的 Planetoid 格式数据。
2. 给邻接矩阵添加自环并执行按行归一化。
3. 按 `iid_fraction` 把图节点划分为类别偏置部分和随机 IID 部分，再无重叠地分给各设备。
4. 每台设备只在自己的节点诱导子图上训练两层 GCN，跨设备边在本地训练时不参与消息传播。
5. 每轮所有设备都参与；没有本地标注训练节点的设备会被跳过。
6. 按每台设备的本地标注训练节点数进行 FedAvg，并把聚合参数广播给全部设备。
7. 每轮使用完整图记录训练集和验证集指标，最后一次聚合后计算测试集指标。

当前默认的 `aggregation_weight_basis: labeled_train_nodes` 与仓库中 notebook 的实际代码一致。论文公式使用知识库中的全部实体数作为权重，因此二者不是完全相同的统计口径；本核心复现优先保持现有代码行为。

## 目录说明

- `run_fedgcn.py`：FedML初始化和训练入口。
- `data.py`：Planetoid数据读取、邻接矩阵归一化和联邦划分。
- `model.py`：两层GCN和基础图卷积层。
- `simulator.py`：局部训练、FedAvg、完整图评估和结果保存。
- `device.py`：FedML设备解析与服务器CUDA强制检查。
- `configs/`：本地CPU冒烟配置和服务器CUDA正式配置。
- `tests/`：不依赖GPU的轻量级单元测试。

## 本地轻量验证

本机的 `py37` 环境是 Python 3.7、FedML 0.7.600 和 PyTorch 1.13.1 CPU版，仅用于轻量验证，不代表服务器正式环境。

本入口使用FedML官方 `load_arguments` 读取YAML，并使用 `fedml.device.get_device` 选择设备。没有直接调用 `fedml.init()`，因为FedML 0.7.600会在单进程离线模拟前强制测试其内置S3凭据；无有效云凭据时，这一步会阻断本地与服务器训练。本模块不使用MLOps或远程存储，因此跳过该云诊断不会改变FedGCN训练流程。

在项目根目录执行单元测试：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest discover -s FedGCN_fedml\tests -v
```

执行两轮 Cora CPU 冒烟训练：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m FedGCN_fedml.run_fedgcn `
  --cf FedGCN_fedml\configs\fedml_config_smoke_cpu.yaml
```

本地冒烟只验证数据加载、训练、聚合、评估和结果文件是否完整，不用于判断论文精度，也不运行100轮正式实验。

## 在PyCharm或VS Code中一键运行

通用IDE入口是 `FedGCN_fedml/run_from_ide.py`。该文件使用配置绝对路径，不依赖IDE当前工作目录，内部仍调用与终端相同的 `run_fedgcn.py`，因此两种启动方式不会形成两套训练逻辑。

在PyCharm中：

1. 为项目选择包含FedML依赖的Python解释器；本机轻量验证可选择 `py37`。
2. 打开 `FedGCN_fedml/run_from_ide.py`。
3. 点击编辑器右上角运行按钮，或右键选择“运行”。
4. 默认执行 `smoke_cpu`。在服务器IDE中需要正式训练时，将文件顶部 `DEFAULT_PROFILE` 改为 `server_cuda`，也可以在PyCharm运行配置中设置环境变量 `FEDGCN_IDE_PROFILE=server_cuda`。

在VS Code中：

1. 选择包含FedML依赖的Python解释器。
2. 打开“运行和调试”面板。
3. 选择“FedGCN：本地CPU冒烟”或“FedGCN：服务器CUDA正式训练”。
4. 点击绿色运行按钮或按F5。对应配置已写入项目的 `.vscode/launch.json`。

也可以打开 `run_from_ide.py` 后直接点击编辑器右上角的“运行Python文件”按钮；此时使用文件顶部的 `DEFAULT_PROFILE`。

## 服务器CUDA运行

先根据服务器显卡驱动和CUDA版本，从PyTorch官方渠道安装匹配的CUDA版PyTorch，再安装其余依赖：

```bash
python -m pip install -r FedGCN_fedml/requirements-server.txt
```

从项目根目录启动正式配置：

```bash
python -m FedGCN_fedml.run_fedgcn \
  --cf FedGCN_fedml/configs/fedml_config_server_cuda.yaml
```

正式配置同时设置 `using_gpu: true` 和 `require_cuda: true`。如果服务器上的PyTorch没有检测到CUDA，程序会在读取和训练图数据之前明确报错，不会静默退回CPU。

如需使用其他GPU，只修改服务器配置中的 `gpu_id`。代码中的所有模型和张量迁移都使用FedML返回的 `torch.device`，没有写死 `cuda:0`。

## 输出结果

每次运行都会在 `FedGCN_fedml/results/` 下创建独立的时间戳目录，包含：

- `metrics.csv`：逐轮训练集和验证集指标，以及参与聚合的设备数和权重总数。
- `config_snapshot.json`：本次FedML参数快照。
- `partition_summary.json`：每台设备的节点编号、节点数和本地标注训练节点数。
- `summary.json`：最终训练、验证和测试指标。
- `final_model.pt`：最终全局GCN参数及指标摘要。

核心复现只保证算法流程、数据不变量和输出可复核，不承诺单次运行严格复现论文图8至图10的曲线。
