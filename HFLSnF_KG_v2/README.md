# HFLSnF知识图谱迁移框架V2

本目录以`HFLSnF_KG/`当前源码为基线独立建立，保留阶段一至阶段六，并新增FedE式实体及关系行级聚合和本地训练目标消融。V2不会把结果写入旧项目，也不会直接导入`1paperAbout/FedE-master/`执行训练。

V2仍使用37个头实体均衡客户端、MAT逐轮动态采样和动态边缘分组，以及当前全局头尾filtered评估。因此，新结果可以和V2内保留的动态稠密方案使用同一数据划分哈希、调度哈希和评估口径比较。

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

阶段四已经实现37客户端头实体均衡划分、全客户端参与、按本地正三元组数加权的单层稠密FedAvg，以及最佳全局模型的filtered评估。阶段五进一步实现固定参与身份的四种FL/HFL拓扑对照。阶段六从项目MAT文件逐轮读取客户端和边缘组。V2在阶段六相同调度上增加行级聚合、FedE式尾负采样和自对抗逻辑损失。

## 工程边界

- `HFLSnF_dynEdge/` 保持原样，继续作为MNIST和MAT拓扑参考。
- `HFLSnF_KG/` 保持原样；V2不读取其数据、配置和结果目录。
- `FedGCN_fedml/` 保持原样，新任务适配器复用其中已验证的Planetoid数据结构和两层GCN。
- 新代码全部位于 `HFLSnF_KG_v2/`。
- `fedml_framework/` 是默认正式运行链，结构来自 `HFLSnF_dynEdge/` 的Runner、Simulator、Client、Group和Trainer。
- `core/simulator.py` 只保留为任务无关参考实现和等价性测试基准，不再由默认入口调用。
- `tasks/kge/` 是阶段三集中式TransE组件；它使用FedML解析YAML和选择设备，但集中式基线本身不伪装成联邦训练。
- `fedml_kge/` 是阶段四普通联邦TransE链路，继承FedML `Client` 和 `ClientTrainer`，只执行客户端到云端的单层FedAvg。
- `tasks/kge/fixed_topology.py` 和 `run_fixed_federated_transe.py` 实现阶段五固定客户端身份、六组分层聚合和四方案独立输出。
- `fedml_kge/dynamic_trainer.py` 和 `run_dynamic_federated_transe.py` 实现MAT逐轮动态客户端采样、动态边缘分组，以及稠密或行级两级聚合。
- `run_three_arm_ablation.py`负责A、B、C三臂公平合同校验、一键服务器训练和统一结果审计。
- `run_four_arm_ablation.py`只补跑D：`dense+FedE-fair`，并复用已有A、B、C完成二乘二汇总。
- `run_directional_diagnostics.py`只读最佳检查点，输出头尾、逐关系和逐查询胜负，不重新训练。
- `core/aggregation.py` 中的`RowMaskedFedAvgAggregator`保存逐行分子和分母，使边缘统计可继续在云端无损合并。
- V2只携带本地`data/FB15k-237/train.txt`、`valid.txt`和`test.txt`及`matlab/`内当前MAT文件；不依赖旧项目数据路径。
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

云端合并这些统计后再统一归一化，因此在参与客户端和权重相同时，两级稠密FedAvg与直接稠密FedAvg数学等价。

V2的行级聚合另外保存：

```text
row_sums：逐参数行聚合分子
row_denominators：每行拥有该知识的活跃客户端数
contributor_ids：当前边缘组或云端统计覆盖的客户端
```

实体掩码只由本地正三元组中的头、尾实体确定，关系掩码只由正三元组中的关系确定。负采样偶然访问的行不获得所有权。每行在拥有该行的活跃客户端之间等权平均；本轮无人拥有的行从轮初全局模型回退。边缘端先保存分子和分母，云端求和后再归一化，所以两级行聚合与同一参与集合的直接行聚合等价。

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

完整200轮中，每轮参与11至37个客户端、均值35.730；动态组数为2至12、均值8.135。V2保留原动态稠密配置作为对照，并增加行级聚合配置。方法口径、参数解释和输出文件见 `STAGE6_DYNAMIC_MAT_FEDERATED_TRANSE.md`。

## V2：FedE式行级聚合与本地目标

三套新增配置使用相同的37客户端划分、200行MAT调度和全局头尾filtered评估：

| 方案 | 嵌入与距离 | 负样本 | 批次 | 本地epoch | 本地目标 |
| --- | --- | ---: | ---: | ---: | --- |
| `dynamic_mat_masked` | 256维、L2 | 5个头尾负样本 | 4096 | 1 | 间隔排序损失 |
| `dynamic_mat_masked_fede_fair` | 256维、L1 | 5个尾负样本 | 4096 | 1 | FedE式自对抗逻辑损失 |
| `dynamic_mat_masked_fede_paper` | 128维、L1 | 256个尾负样本 | 512 | 3 | FedE式自对抗逻辑损失 |

公平预算方案只在当前动态稠密预算内切换距离、负采样方向、损失和行级聚合；论文参数参考方案的本地开销明显更高。三套配置继续维护统一实体和关系列表，并对实体、关系都执行行级聚合，这一点不同于FedE开源实现中“只聚合实体、关系留在客户端”的做法。

详细来源对应、适配差异和输出定义见`V2_FEDE_MASKED_TRANSE.md`。

### 同MAT三臂公平消融

当前下一步已经封装为A：`dense+margin`、B：`masked+margin`、C：`masked+FedE-fair`三组实验。程序在训练前固定检查数据文件、客户端划分、200行MAT调度和训练预算；训练结果还会记录初始模型哈希，汇总时再次拒绝不公平的三次运行。

先做不训练的安全检查：

```powershell
python -m HFLSnF_KG_v2.run_three_arm_ablation --action validate
```

CUDA服务器一键运行：

```bash
python -m HFLSnF_KG_v2.run_three_arm_ablation \
  --action run \
  --arm all
```

完整的大白话运行和判定说明见`THREE_ARM_ABLATION.md`。

种子42的三臂正式结果已经完成。C：`masked+FedE-fair`取得测试MRR `0.197400`，B相对A和C相对B均通过预设初筛。结果含义、Hits@1例外、收敛状态和补D臂建议见`THREE_ARM_RESULT_ANALYSIS.md`。

### 四臂二乘二与方向诊断

补D臂和无需重训方向诊断已经实现。D与C使用相同的L1、5个尾负样本、FedE自对抗目标和训练预算，只把聚合改回稠密FedAvg。

先校验四臂合同：

```powershell
python -m HFLSnF_KG_v2.run_four_arm_ablation --action validate
```

CUDA服务器只补跑D，并复用`results/三种`中的A、B、C：

```bash
python -m HFLSnF_KG_v2.run_four_arm_ablation \
  --action run-d \
  --existing-root HFLSnF_KG_v2/results/三种
```

D完成后执行完整头尾方向诊断：

```bash
python -m HFLSnF_KG_v2.run_directional_diagnostics \
  --result-root HFLSnF_KG_v2/results/三种 \
  --result dense_fede_fair=/path/to/d_result \
  --using-gpu \
  --require-cuda \
  --max-triples 0
```

运行方法、输出文件和交互项的大白话解释见`FOUR_ARM_FACTORIAL_AND_DIAGNOSTICS.md`。

## 无需重训的评估桥接

V2现在可以不训练模型，直接把FedE、集中式、FLSnF和HFLnoSnF检查点放到统一口径下复评。

它先复现FedE原来的“本地候选、只预测尾实体”口径，再逐步切换到全局候选头尾预测；最后使用2,048条双方训练和验证阶段都没有见过的严格公共测试事实，让所有检查点完成相同的4,096个头尾查询。

这个入口不会创建优化器或修改检查点。完整的大白话说明见`EVALUATION_BRIDGE.md`，下一步实验顺序见`NEXT_STEP_EXPERIMENT_PLAN.md`。

## 在IDE中直接运行

打开 `HFLSnF_KG_v2/run_from_ide.py`，点击PyCharm或VS Code的“运行Python文件”即可。

文件顶部运行方案由 `DEFAULT_PROFILE` 决定。当前默认运行公平预算FedE消融：

```python
DEFAULT_PROFILE = "dynamic_fedtranse_masked_fede_fair_cuda"
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
- `dynamic_fedtranse_hflsnf_mat_cuda`：原动态稠密方案，MAT逐轮选择客户端并动态分组；
- `dynamic_fedtranse_masked_cuda`：当前本地目标和预算，只切换为实体及关系行级聚合；
- `dynamic_fedtranse_masked_fede_fair_cuda`：行级聚合与FedE式本地目标的公平预算方案；
- `dynamic_fedtranse_masked_fede_paper_cuda`：128维、256个尾负样本和3个本地epoch的高开销参考方案；
- `evaluation_bridge_smoke_cpu`：不训练模型，只抽少量真实查询完成CPU全链路冒烟；
- `evaluation_bridge_full_cuda`：不训练模型，完成E0、E1和E2正式复评，强制CUDA。

如果只想运行评估桥接，也可以直接打开`run_evaluation_bridge_from_ide.py`点击运行。它默认选择安全的CPU冒烟方案；放到CUDA服务器后把文件顶部方案改为`evaluation_bridge_full_cuda`。

如果要运行同MAT三臂消融，直接打开`run_three_arm_ablation_from_ide.py`。它默认只执行`validate`，不会训练；放到CUDA服务器后把`DEFAULT_ACTION`改为`"run"`即可按A、B、C顺序一键训练和汇总。

如果只想补跑D，直接打开`run_four_arm_ablation_from_ide.py`。本机默认只校验；服务器上把`DEFAULT_ACTION`改为`"run-d"`。

如果要做头尾、逐关系和逐查询诊断，直接打开`run_directional_diagnostics_from_ide.py`。本机默认只评估8条事实；服务器上填写`D_RESULT_DIR`并把`FULL_CUDA`改为`True`。

也可以通过环境变量选择：

```powershell
$env:HFLSNF_KG_V2_IDE_PROFILE = "dynamic_fedtranse_masked_fede_fair_cuda"
python HFLSnF_KG_v2\run_from_ide.py
```

## 在终端运行

从项目根目录运行本地CPU冒烟：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m HFLSnF_KG_v2.run_hfl_kg `
  --cf HFLSnF_KG_v2/configs/smoke_cora_cpu.yaml
```

服务器CUDA验证：

```bash
python -m HFLSnF_KG_v2.run_hfl_kg \
  --cf HFLSnF_KG_v2/configs/server_cora_cuda.yaml
```

阶段三TransE本地CPU冒烟：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m HFLSnF_KG_v2.run_transe `
  --cf HFLSnF_KG_v2/configs/smoke_transe_synthetic_cpu.yaml
```

阶段三FB15k-237服务器训练：

```bash
python -m HFLSnF_KG_v2.run_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_transe_cuda.yaml
```

4060 Laptop平衡加速训练：

```bash
python -m HFLSnF_KG_v2.run_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_transe_cuda_fast.yaml
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
  -m HFLSnF_KG_v2.run_federated_transe `
  --cf HFLSnF_KG_v2/configs/smoke_fedtranse_synthetic_cpu.yaml
```

阶段四FB15k-237服务器训练：

```bash
python -m HFLSnF_KG_v2.run_federated_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_fedtranse_cuda.yaml
```

阶段五四方案可以分别使用对应YAML运行，也可以在Windows服务器一次顺序执行：

```powershell
.\HFLSnF_KG_v2\run_fixed_fedtranse_comparisons.ps1
```

单个方案的终端命令格式为：

```bash
python -m HFLSnF_KG_v2.run_fixed_federated_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_hflsnf_fixed_cuda.yaml
```

阶段六MAT动态采样和分组的服务器命令：

```bash
python -m HFLSnF_KG_v2.run_dynamic_federated_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_hflsnf_dynamic_mat_cuda.yaml
```

V2三种行级聚合方案的服务器命令：

```bash
python -m HFLSnF_KG_v2.run_dynamic_federated_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_hflsnf_dynamic_mat_masked_cuda.yaml

python -m HFLSnF_KG_v2.run_dynamic_federated_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_hflsnf_dynamic_mat_masked_fede_fair_cuda.yaml

python -m HFLSnF_KG_v2.run_dynamic_federated_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_hflsnf_dynamic_mat_masked_fede_paper_cuda.yaml
```

三条命令的结果都只写到`HFLSnF_KG_v2/results/`。正式配置在读取FB15k-237前检查CUDA；本机不应运行200轮正式实验。

无需重训评估桥接的本地CPU冒烟：

```powershell
& 'D:\Anaconda3\envs\py37\python.exe' `
  -m HFLSnF_KG_v2.run_evaluation_bridge `
  --cf HFLSnF_KG_v2/configs/evaluation_bridge_smoke_cpu.yaml
```

CUDA服务器完整复评：

```bash
python -m HFLSnF_KG_v2.run_evaluation_bridge \
  --cf HFLSnF_KG_v2/configs/evaluation_bridge_full_cuda.yaml
```

冒烟只用于验证代码和输出，不用于比较精度。完整配置不会训练模型，但全局排名仍需让每条查询和14,541个候选实体比较，因此建议在GPU上运行。

服务器需要安装与CUDA驱动匹配的PyTorch，以及FedML、SciPy、NetworkX和NumPy。服务器配置中的 `require_cuda: true` 会在CUDA不可用时于训练前报错。

## 运行测试

V2全部测试：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest discover -s HFLSnF_KG_v2\tests -v
```

原项目拓扑和层次采样回归测试：

```powershell
& 'D:\Anaconda3\Scripts\conda.exe' run --no-capture-output -n py37 python `
  -m unittest -v `
  HFLSnF_dynEdge\test_topology_schedule.py `
  HFLSnF_dynEdge\test_hierarchical_sampling.py
```

## 输出文件

每次运行会在 `HFLSnF_KG_v2/results/` 下创建带微秒时间戳的独立目录，并写出：

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

V2动态行级方案每个通信epoch同样打印MAT行号、参与人数、动态组数、本地加权损失、耗时，以及按配置周期得到的MRR和Hits@3。其结果额外包含：

- `metrics.csv`中的实体及关系有效聚合行数和回退行数；
- `dynamic_topology_schedule.jsonl`中的聚合模式、本地目标、每个动态组的逐参数行贡献统计、云端逐参数行统计和实际贡献客户端；
- `dynamic_participation_summary.json`及`summary.json`中的MAT调度哈希；
- `summary.json`中的聚合模式、本地目标、划分哈希和最终完整filtered指标；
- `model_best.pt`中的最佳全局实体及关系表、统一编号映射、划分摘要和调度摘要。

评估桥接结果目录包含：

- `data_audit.json`：数据全集、文件哈希、交叉测试泄漏和公共集覆盖；
- `common_valid.tsv`与`common_test.tsv`：严格公共事实、名称和FedE客户端归属；
- `model_manifest.json`：检查点路径、哈希、嵌入维数和L1/L2距离；
- `protocol_metrics.json`：E0、E1a、E1b、E1c和E2指标；
- `query_ranks.csv`：每个模型对每条事实的头尾排名；
- `summary.json`：明确记录`training_performed: false`的总汇总。
