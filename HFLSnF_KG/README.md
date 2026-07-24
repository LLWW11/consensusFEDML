# HFLSnF知识图谱迁移框架

本目录已经实现迁移计划的阶段一至阶段四：任务无关分层联邦核心、FedML版Cora/Citeseer GCN、FB15k-237集中式TransE基线，以及37客户端普通联邦TransE。阶段二GCN和阶段四TransE都使用FedML Client/ClientTrainer接口，不以自定义简易模拟器作为正式入口。

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

阶段四已经实现37客户端头实体均衡划分、全客户端参与、按本地正三元组数加权的单层稠密FedAvg，以及最佳全局模型的filtered评估。MAT分层拓扑、实体关系行级掩码聚合和动态知识源失效实验仍属于后续阶段。

## 工程边界

- `HFLSnF_dynEdge/` 保持原样，继续作为MNIST和MAT拓扑参考。
- `FedGCN_fedml/` 保持原样，新任务适配器复用其中已验证的Planetoid数据结构和两层GCN。
- 新代码全部位于 `HFLSnF_KG/`。
- `fedml_framework/` 是默认正式运行链，结构来自 `HFLSnF_dynEdge/` 的Runner、Simulator、Client、Group和Trainer。
- `core/simulator.py` 只保留为任务无关参考实现和等价性测试基准，不再由默认入口调用。
- `tasks/kge/` 是阶段三集中式TransE组件；它使用FedML解析YAML和选择设备，但集中式基线本身不伪装成联邦训练。
- `fedml_kge/` 是阶段四普通联邦TransE链路，继承FedML `Client` 和 `ClientTrainer`，只执行客户端到云端的单层FedAvg。
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

## 阶段三：集中式TransE基线

阶段三调用链如下：

```text
FedML YAML解析与torch.device选择
          ↓
FB15k-237统一实体/关系编号
          ↓
TransE间隔排序损失与filtered负采样
          ↓
验证集filtered MRR选择最佳模型
          ↓
测试集头尾双向filtered排名
```

`tasks/kge/` 包含：

- `data.py`：读取 `train.txt`、`valid.txt` 和 `test.txt`，建立统一全局编号并拦截重复或跨划分重叠三元组；
- `model.py`：实体嵌入、关系嵌入和L1/L2 TransE距离；
- `negative_sampling.py`：替换头实体或尾实体，并排除全部已知真三元组；
- `evaluator.py`：头预测和尾预测的filtered MRR、Mean Rank、Hits@1/3/10；
- `trainer.py`：集中式小批次训练、验证选模、早停和最终测试。

FB15k-237文件放置方式见 `data/README.md`。内置 `synthetic-kg` 只用于无网络CPU流程验证，不能用其指标判断论文效果。

## 阶段四：普通联邦TransE

阶段四将全部训练三元组按头实体归属分给37个知识客户端。同一头实体的事实不会跨客户端，头实体按照三元组数量贪心分配到当前负载最低的客户端。种子42下，当前FB15k-237每个客户端持有7,354–7,355条训练三元组，全部272,115条训练三元组无重复、无遗漏。

每个客户端维护统一编号下的完整实体和关系嵌入表，但本地损失只使用自己的三元组。每轮37个客户端都从同一个全局模型出发训练1个本地epoch，再按本地正三元组数直接执行云端FedAvg。未被客户端使用的嵌入行仍参与完整模型平均，这是普通稠密FedAvg对照组的预期局限；行级掩码聚合不属于本阶段。

阶段四详细方法和实验口径见 `STAGE4_FEDERATED_TRANSE.md`。

## 在IDE中直接运行

打开 `HFLSnF_KG/run_from_ide.py`，点击PyCharm或VS Code的“运行Python文件”即可。

文件顶部运行方案由 `DEFAULT_PROFILE` 决定。当前工作树保留用户设置：

```python
DEFAULT_PROFILE = "transe_server_cuda"
```

可选值：

- `smoke_cpu`：Cora、2个客户端、2个边缘组、2轮通信、每轮1个本地epoch；
- `server_cuda`：Cora、6个客户端、3个边缘组、100轮通信、每轮3个本地epoch，强制CUDA。
- `transe_smoke_cpu`：内置微型知识图谱、3个集中式epoch、CPU轻量验证；
- `transe_server_cuda`：FB15k-237、100个集中式epoch、完整测试集filtered评估，强制CUDA。
- `fedtranse_smoke_cpu`：3个知识客户端、2轮普通FedAvg、CPU轻量验证；
- `fedtranse_server_cuda`：37个知识客户端、100轮普通FedAvg、每轮1个本地epoch，强制CUDA。

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

阶段三TransE本地CPU冒烟：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m HFLSnF_KG.run_transe `
  --cf HFLSnF_KG/configs/smoke_transe_synthetic_cpu.yaml
```

阶段三FB15k-237服务器训练：

```bash
python -m HFLSnF_KG.run_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_transe_cuda.yaml
```

阶段四普通联邦TransE本地CPU冒烟：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m HFLSnF_KG.run_federated_transe `
  --cf HFLSnF_KG/configs/smoke_fedtranse_synthetic_cpu.yaml
```

阶段四FB15k-237服务器训练：

```bash
python -m HFLSnF_KG.run_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_fedtranse_cuda.yaml
```

服务器需要安装与CUDA驱动匹配的PyTorch，以及FedML、SciPy、NetworkX和NumPy。服务器配置中的 `require_cuda: true` 会在CUDA不可用时于训练前报错。

## 运行测试

阶段一至阶段四及FedML框架等价性测试：

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

阶段三结果目录另外包含：

- `dataset_summary.json`：实体、关系及三个数据划分的规模；
- `entity2id.json` 和 `relation2id.json`：全局编号映射；
- `metrics.csv`：逐epoch训练损失和按配置周期计算的验证集filtered指标；
- `summary.json`：最佳epoch及最终验证、测试filtered指标；
- `model_best.pt`：最佳TransE参数、编号映射和训练汇总。

阶段四结果目录另外包含：

- `client_partition_summary.json`：划分指纹、客户端负载、知识范围及实体重叠统计；
- `participation_schedule.jsonl`：逐轮37客户端参与、贡献及聚合权重；
- `metrics.csv`：逐轮本地加权损失和按周期计算的全局验证指标；
- `summary.json`：最佳轮、完整filtered指标、集中式测试MRR参考值及差值；
- `model_best.pt`：最佳全局TransE参数、统一编号映射和分区摘要。
