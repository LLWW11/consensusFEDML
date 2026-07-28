# HFLSnF知识图谱迁移框架

本目录已经实现迁移计划的阶段一至阶段六：任务无关分层联邦核心、FedML版Cora/Citeseer GCN、FB15k-237集中式TransE基线、37客户端普通联邦TransE、FLnoSnF/FLSnF/HFLnoSnF/HFLSnF四种固定参与拓扑对照，以及MAT逐轮动态采样和动态分组的HFLSnF TransE。阶段二GCN和联邦TransE都使用FedML Client/ClientTrainer接口，不以自定义简易模拟器作为正式入口。

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

阶段四已经实现37客户端头实体均衡划分、全客户端参与、按本地正三元组数加权的单层稠密FedAvg，以及最佳全局模型的filtered评估。阶段五进一步实现固定参与身份的四种FL/HFL拓扑对照。阶段六从项目MAT文件逐轮读取客户端和边缘组；实体关系行级掩码聚合和动态知识源失效实验仍属于后续阶段。

## 工程边界

- `HFLSnF_dynEdge/` 保持原样，继续作为MNIST和MAT拓扑参考。
- `FedGCN_fedml/` 保持原样，新任务适配器复用其中已验证的Planetoid数据结构和两层GCN。
- 新代码全部位于 `HFLSnF_KG/`。
- `fedml_framework/` 是默认正式运行链，结构来自 `HFLSnF_dynEdge/` 的Runner、Simulator、Client、Group和Trainer。
- `core/simulator.py` 只保留为任务无关参考实现和等价性测试基准，不再由默认入口调用。
- `tasks/kge/` 是阶段三集中式TransE组件；它使用FedML解析YAML和选择设备，但集中式基线本身不伪装成联邦训练。
- `fedml_kge/` 是阶段四普通联邦TransE链路，继承FedML `Client` 和 `ClientTrainer`，只执行客户端到云端的单层FedAvg。
- `tasks/kge/fixed_topology.py` 和 `run_fixed_federated_transe.py` 实现阶段五固定客户端身份、六组分层聚合和四方案独立输出。
- `fedml_kge/dynamic_trainer.py` 和 `run_dynamic_federated_transe.py` 实现阶段六MAT逐轮动态客户端采样、动态边缘分组和两级聚合。
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

服务器配置将逐轮监控与正式选模分开：

- `monitor_every_epoch: true`：每个epoch在终端打印训练损失、耗时、监控MRR和监控Hits@3；
- `monitor_validation_max_triples: 10`：使用固定10条验证三元组观察短期趋势，不参与最佳模型选择；
- `eval_every: 5`和`validation_max_triples: 1000`：每5个epoch使用固定1000条验证三元组选择最佳模型和推进早停计数；
- `metrics.csv`同时保存逐轮`monitor_*`字段和按周期产生的`val_*`正式选模字段。

终端输出中的“监控MRR”只用于观察训练趋势，“选模MRR”才决定`model_best.pt`。这样既能逐epoch看到指标，又不会再让10条验证三元组决定最佳模型。

RTX 4060等消费级GPU运行基础TransE时，利用率低于图像联邦学习属于常见现象。当前每个训练epoch要在CPU和Python循环中生成`272115 × 30 = 8163450`个filtered负样本，而GPU侧主要是嵌入查表和向量距离，计算量与卷积网络相比很小。filtered排名评估还会逐查询分块启动GPU计算，因此利用率通常呈短时脉冲。正式配置使用`require_cuda: true`，若CUDA不可用会直接报错，不会静默改用CPU；入口还会在启动时打印实际设备编号和GPU名称。

FB15k-237文件放置方式见 `data/README.md`。内置 `synthetic-kg` 只用于无网络CPU流程验证，不能用其指标判断论文效果。

## 阶段四：普通联邦TransE

阶段四将全部训练三元组按头实体归属分给37个知识客户端。同一头实体的事实不会跨客户端，头实体按照三元组数量贪心分配到当前负载最低的客户端。种子42下，当前FB15k-237每个客户端持有7,354–7,355条训练三元组，全部272,115条训练三元组无重复、无遗漏。

每个客户端维护统一编号下的完整实体和关系嵌入表，但本地损失只使用自己的三元组。每轮37个客户端都从同一个全局模型出发训练1个本地epoch，再按本地正三元组数直接执行云端FedAvg。未被客户端使用的嵌入行仍参与完整模型平均，这是普通稠密FedAvg对照组的预期局限；行级掩码聚合不属于本阶段。

阶段四详细方法和实验口径见 `STAGE4_FEDERATED_TRANSE.md`。

## 阶段五：固定FL/HFL TransE四方案

四个方案都从相同的37客户端固定排列中截取一个前缀，并在所有通信轮复用同一批编号：

| 方案 | `client_num_per_round` | 聚合结构 | 组数 |
| --- | ---: | --- | ---: |
| FLnoSnF | 5 | 直接云端FedAvg | 1 |
| FLSnF | 25 | 直接云端FedAvg | 1 |
| HFLnoSnF | 15 | 组内再到云端 | 6 |
| HFLSnF | 35 | 组内再到云端 | 6 |

本阶段不进行动态SnF选择；`SnF/noSnF`表示固定实验场景和对应参与预算。HFL的6组在实验开始时固定，组内和云端都按本地正三元组数合并稠密FedAvg统计。详细定义、输出文件和运行方式见 `STAGE5_FIXED_FEDERATED_TRANSE.md`。

## 阶段六：MAT动态联邦TransE

阶段六读取`matlab/result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat`，使用`HFLSnF + dynamic edge + topology_util=0.5`场景。37个知识客户端与MAT候选槽位一一对应，每个通信epoch使用一行MAT数据决定实际参与客户端和动态边缘组。

完整200轮中，每轮参与11至37个客户端、均值35.730；动态组数为2至12、均值8.135。训练仍使用完整实体和关系表的稠密FedAvg，组内与云端都按本地正三元组数聚合。方法口径、参数解释和输出文件见 `STAGE6_DYNAMIC_MAT_FEDERATED_TRANSE.md`。

## 在IDE中直接运行

打开 `HFLSnF_KG/run_from_ide.py`，点击PyCharm或VS Code的“运行Python文件”即可。

文件顶部运行方案由 `DEFAULT_PROFILE` 决定。当前默认直接运行动态MAT方案：

```python
DEFAULT_PROFILE = "dynamic_fedtranse_hflsnf_mat_cuda"
```

可选值：

- `smoke_cpu`：Cora、2个客户端、2个边缘组、2轮通信、每轮1个本地epoch；
- `server_cuda`：Cora、6个客户端、3个边缘组、100轮通信、每轮3个本地epoch，强制CUDA。
- `transe_smoke_cpu`：内置微型知识图谱、3个集中式epoch、CPU轻量验证；
- `transe_server_cuda`：FB15k-237、最多1000个集中式epoch、256维和30个负样本的高开销配置，强制CUDA；
- `transe_server_cuda_fast`：FB15k-237、最多400个集中式epoch、128维、5个负样本和4096批次的4060 Laptop平衡加速配置，强制CUDA；
- `fedtranse_smoke_cpu`：3个知识客户端、2轮普通FedAvg、CPU轻量验证；
- `fedtranse_server_cuda`：37个知识客户端、100轮普通FedAvg、每轮1个本地epoch，强制CUDA。
- `fixed_fedtranse_flnosnf_cuda`：固定5客户端直接云聚合；
- `fixed_fedtranse_flsnf_cuda`：固定25客户端直接云聚合；
- `fixed_fedtranse_hflnosnf_cuda`：固定15客户端、6组分层聚合；
- `fixed_fedtranse_hflsnf_cuda`：固定35客户端、6组分层聚合；
- `dynamic_fedtranse_hflsnf_mat_cuda`：MAT逐轮动态选择11至37个客户端并动态分组，完整运行200轮，强制CUDA。

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

4060 Laptop平衡加速训练：

```bash
python -m HFLSnF_KG.run_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_transe_cuda_fast.yaml
```

两份服务器配置的主要差异如下：

| 配置项 | 高开销正式配置 | 4060平衡加速配置 |
| --- | ---: | ---: |
| 嵌入维度 | 256 | 128 |
| 每个正样本的负样本数 | 30 | 5 |
| 批次大小 | 1024 | 4096 |
| 最大epoch | 1000 | 400 |
| 正式选模周期 | 每5个epoch | 每10个epoch |
| 正式选模三元组数 | 1000 | 1000 |
| 候选实体批次 | 4096 | 16384 |
| 每个epoch监控 | 10条 | 10条 |

加速版每个epoch的Python负采样次数约为正式配置的六分之一，负样本数与嵌入维度共同决定的主要向量计算量约为十二分之一。它适合调试训练趋势、筛选参数和生成阶段四前的快速集中式参考，但其训练预算与高开销配置不同，不能把两者指标直接写成严格消融结论。若加速版仍然过慢，可把`negative_sample_count`进一步改为1，或把`validation_max_triples`临时改为500；这两项都会降低结果精度或选模稳定性，只适合快速试跑。

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

阶段五四方案可以分别使用对应YAML运行，也可以在Windows服务器一次顺序执行：

```powershell
.\HFLSnF_KG\run_fixed_fedtranse_comparisons.ps1
```

单个方案的终端命令格式为：

```bash
python -m HFLSnF_KG.run_fixed_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_hflsnf_fixed_cuda.yaml
```

阶段六MAT动态采样和分组的服务器命令：

```bash
python -m HFLSnF_KG.run_dynamic_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_hflsnf_dynamic_mat_cuda.yaml
```

服务器需要安装与CUDA驱动匹配的PyTorch，以及FedML、SciPy、NetworkX和NumPy。服务器配置中的 `require_cuda: true` 会在CUDA不可用时于训练前报错。

## 运行测试

阶段一至阶段六及FedML框架等价性测试：

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
- `metrics.csv`：逐epoch训练损失、耗时和小样本监控指标，以及按配置周期计算的正式验证集filtered指标；
- `summary.json`：最佳epoch及最终验证、测试filtered指标；
- `model_best.pt`：最佳TransE参数、编号映射和训练汇总。

阶段四结果目录另外包含：

- `client_partition_summary.json`：划分指纹、客户端负载、知识范围及实体重叠统计；
- `participation_schedule.jsonl`：逐轮37客户端参与、贡献及聚合权重；
- `metrics.csv`：逐轮本地加权损失和按周期计算的全局验证指标；
- `summary.json`：最佳轮、完整filtered指标、集中式测试MRR参考值及差值；
- `model_best.pt`：最佳全局TransE参数、统一编号映射和分区摘要。

阶段五四方案结果目录另外包含：

- `fixed_participation.json`：整次实验不变的客户端编号、六组映射、知识规模和拓扑指纹；
- `participation_schedule.jsonl`：逐轮固定身份、客户端权重、组权重和实际聚合路径；
- `metrics.csv`：逐轮方案、FL/HFL结构、固定参与人数、组数、损失、轮次耗时和验证指标；
- `summary.json`：固定拓扑、最佳轮、完整filtered指标和集中式差值；
- `model_best.pt`：最佳全局TransE参数、固定拓扑、统一编号映射和客户端分区。

四种固定联邦TransE方案会在每个通信epoch结束后打印方案、轮次、参与人数、分组数、加权损失、聚合权重和耗时。到达`eval_every`指定的选模轮时，还会打印验证MRR与Hits@3；非选模轮明确显示“未评估”。
