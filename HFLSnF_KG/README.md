# HFLSnF知识图谱迁移框架

本目录实现迁移计划的阶段一和阶段二：先建立任务无关的单进程分层联邦核心，再将已经验证的Cora/Citeseer两层GCN作为第一个任务适配器接入。当前终端入口和IDE入口默认运行FedML框架版，不再以自定义简易模拟器作为正式入口。

默认运行链如下：

```text
FedMLRunner（由HFLSnF运行器迁移）
          ↓
SimulatorSingleProcess（FedML simulation + sp）
          ↓
FedML Client → FedML ClientTrainer
          ↓
客户端局部诱导子图训练
          ↓
边缘端保留加权分子与分母
          ↓
云端合并边缘统计
          ↓
完整图训练/验证/测试评估
```

当前版本尚未实现FB15k-237、TransE、知识图谱负采样、filtered MRR、实体行级掩码聚合和动态知识源失效实验。

## 工程边界

- `HFLSnF_dynEdge/` 保持原样，继续作为MNIST和MAT拓扑参考。
- `FedGCN_fedml/` 保持原样，新任务适配器复用其中已验证的Planetoid数据结构和两层GCN。
- 新代码全部位于 `HFLSnF_KG/`。
- `fedml_framework/` 是默认正式运行链，结构来自 `HFLSnF_dynEdge/` 的Runner、Simulator、Client、Group和Trainer。
- `core/simulator.py` 只保留为任务无关参考实现和等价性测试基准，不再由默认入口调用。
- 本地只执行CPU轻量验证；服务器配置明确要求CUDA，不允许静默回退CPU。

## 核心设计

### FedML框架调用链

`fedml_framework/` 中的主要对象如下：

- `runner.py`：校验FedML `simulation` 训练平台和 `sp` 单进程后端；
- `simulator.py`：根据 `federated_optimizer: HierarchicalGCN` 创建分层GCN训练器；
- `model_trainer.py`：继承FedML `ClientTrainer`，适配 `model(features, adjacency)` 双输入接口；
- `client.py`：继承FedML FedAvg `Client`，复用其参数下发、本地训练和参数回收生命周期；
- `group.py`：执行客户端到边缘端的第一层聚合；
- `hierarchical_trainer.py`：执行通信轮、边缘到云聚合、广播和完整图评估。

具体的原文件与新文件对应关系见 `SOURCE_MAP.md`。

### 拓扑和两级聚合

`core/topology.py` 提供三种任务无关拓扑：

- `StaticTopologyProvider`：全部通信轮使用相同边缘分组；
- `SequenceTopologyProvider`：用于动态参与、零参与轮和单元测试；
- `MatlabTopologyProvider`：只读包装原项目的MATLAB拓扑调度，知识客户端编号直接对应候选槽位。

`core/aggregation.py` 不会在边缘端只留下一个无法继续正确合并的普通均值，而是保留：

```text
weighted_sums：逐参数加权分子
total_weight：聚合分母
contributor_ids：实际贡献客户端
```

云端合并这些统计后再统一归一化，因此在参与客户端和权重相同时，两级FedAvg与直接FedAvg数学等价。该接口也为后续实体及关系行级掩码聚合保留扩展位置。

FedML框架版的 `fedml_framework/hierarchical_trainer.py` 将一行拓扑严格解释为一个通信轮。本地 `epochs` 只控制 `ClientTrainer` 在同一拓扑下训练多少次，不会推进MAT行号，也不会在每个本地epoch后重复聚合。

## 在IDE中直接运行

打开 `HFLSnF_KG/run_from_ide.py`，点击PyCharm或VS Code的“运行Python文件”即可。

文件顶部默认配置为：

```python
DEFAULT_PROFILE = "smoke_cpu"
```

可选值：

- `smoke_cpu`：Cora、2个客户端、2个边缘组、2轮通信、每轮1个本地epoch；
- `server_cuda`：Cora、6个客户端、3个边缘组、100轮通信、每轮3个本地epoch，强制CUDA。

也可以通过环境变量选择：

```powershell
$env:HFLSNF_KG_IDE_PROFILE = "server_cuda"
python HFLSnF_KG\run_from_ide.py
```

## 在终端运行

从项目根目录运行本地CPU冒烟：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m HFLSnF_KG.run_hfl_kg `
  --cf HFLSnF_KG/configs/smoke_cora_cpu.yaml
```

服务器CUDA验证：

```bash
python -m HFLSnF_KG.run_hfl_kg \
  --cf HFLSnF_KG/configs/server_cora_cuda.yaml
```

服务器需要安装与CUDA驱动匹配的PyTorch，以及FedML、SciPy、NetworkX和NumPy。服务器配置中的 `require_cuda: true` 会在CUDA不可用时于训练前报错。

## 运行测试

阶段一、阶段二及FedML框架等价性测试：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest discover -s HFLSnF_KG\tests -v
```

原项目拓扑和层次采样回归测试：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest -v `
  HFLSnF_dynEdge\test_topology_schedule.py `
  HFLSnF_dynEdge\test_hierarchical_sampling.py
```

## 输出文件

每次运行会在 `HFLSnF_KG/results/` 下创建带微秒时间戳的独立目录，并写出：

- `config_snapshot.json`：FedML配置快照；
- `partition_summary.json`：完整图和客户端局部子图摘要；
- `topology_metadata.json`：拓扑提供器元数据；
- `topology_schedule.jsonl`：逐轮分组、活跃客户端和实际贡献记录；
- `metrics.csv`：逐轮本地损失和完整图指标；
- `summary.json`：最终运行汇总；
- `model_final.pt`：最终全局GCN参数。

当前Cora结果只用于验证迁移框架，不用于判断论文精度。
