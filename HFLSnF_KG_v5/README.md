# HFLSnF KGE V5：图语义感知联邦划分实验

## 项目定位

V5是在V4冻结实现基础上建立的独立实验工程。V4研究固定HFLSnF条件下实体重叠率的可控变化；V5采用“Freebase语义域＋实体图局部化”划分，验证更接近机构知识子图的数据分布。

FB15k-237没有真实机构所有权标签，因此V5是现实代理实验，不是实际机构数据实验。V4代码和结果保持不变；V5在自身配置目录内保存三份按字节冻结的十二格参考JSON并校验固定哈希，运行时不访问V4目录。

## 当前正式状态

- 数据集：FB15k-237，272,115条训练三元组；
- 客户端：37个；
- 划分策略：`semantic_domain_graph_local_balanced`；
- 语义域：30个；
- 不可拆分域－头实体包：33,155个；
- 正式种子：42、2024、2025；
- 每个种子：8次确定性搜索重启并独立复算；
- 正式分区合同：已通过；
- 完整V5三种子150轮CUDA训练与官方全量评估：已通过并冻结；
- 语义—图局部性双消融：A/B两个实验臂、每臂三个种子的正式实验已完成；
- 图语义拓扑对照扩展：已加入HFLnoSnF与FLnoSnF两个组别，每组使用42、2024、2025三个种子，共新增6次正式训练。

本机 `D:\Anaconda3\envs\py37` 为PyTorch CPU环境，只用于校准、测试和冒烟。正式配置保留 `require_cuda: true`。

## 执行命令

在仓库根目录执行无训练正式校准：

```powershell
D:\Anaconda3\envs\py37\python.exe -m HFLSnF_KG_v5.run_graph_semantic_partition_calibration
```

校验正式合同和三种子分区：

```powershell
D:\Anaconda3\envs\py37\python.exe -m HFLSnF_KG_v5.run_graph_semantic_ablation validate
```

在CUDA环境顺序执行三组训练：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_ablation formal150
python -m HFLSnF_KG_v5.run_graph_semantic_ablation formal150 --resume "<batch_summary.json>"
```

训练通过后执行三个完整官方测试并生成报告：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_ablation official3 --batch "<batch_summary.json>"
python -m HFLSnF_KG_v5.run_graph_semantic_ablation report --batch "<batch_summary.json>"
```

双消融实现完成后，先校验六个新分区，再在CUDA环境运行六组正式训练：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_mechanism_ablation validate
python -m HFLSnF_KG_v5.run_graph_semantic_mechanism_ablation formal150
python -m HFLSnF_KG_v5.run_graph_semantic_mechanism_ablation formal150 --resume "<batch_summary.json>"
python -m HFLSnF_KG_v5.run_graph_semantic_mechanism_ablation official6 --batch "<batch_summary.json>"
python -m HFLSnF_KG_v5.run_graph_semantic_mechanism_ablation report --batch "<batch_summary.json>"
```

新增图语义拓扑对照先在本地校验，再在CUDA环境顺序执行6次正式训练：

```powershell
D:\Anaconda3\envs\py37\python.exe -m HFLSnF_KG_v5.run_graph_semantic_topology_extension validate
python -m HFLSnF_KG_v5.run_graph_semantic_topology_extension formal6
python -m HFLSnF_KG_v5.run_graph_semantic_topology_extension formal6 --resume "<batch_summary.json>"
```

## 关键目录

- `configs/graph_semantic/`：三份正式YAML、冻结校准合同和内置历史参考；
- `tasks/kge/graph_semantic_partition.py`：语义域解析、图局部化分配和统计；
- `tasks/kge/graph_semantic_ablation.py`：配置、基线、V5内置历史参考和结果合同；
- `run_graph_semantic_ablation.py`：训练恢复、官方测试和报告；
- `results/graph_semantic/`：新实验结果，默认不进入Git。
- `configs/graph_semantic_mechanism_ablation/`：A/B六份正式配置、校准合同和完整V5冻结参考；
- `results/graph_semantic_mechanism_ablation/`：A/B正式结果，默认不进入Git。
- `configs/graph_semantic_topology_extension/`：HFLnoSnF与FLnoSnF六份图语义正式配置；
- `run_graph_semantic_topology_extension.py`：六实验校验、失败停止和恢复入口；
- `results/graph_semantic_topology_extension/`：六次新增训练的隔离结果目录。

详细预注册内容见[图语义感知划分实验计划](图语义感知划分实验计划.md)，完整训练和聚合数据流见[实验流程说明](实验流程说明.md)。

语义主域与实体图局部性的删除式消融见[图语义机制双消融实验计划](图语义机制双消融实验计划.md)。

HFLnoSnF与FLnoSnF的三种子扩展见[图语义拓扑对照六实验计划](图语义拓扑对照六实验计划.md)。

## 冻结边界

V5中复制的三种子原始结果与V4原件哈希一致，只作为基线。不得修改历史结果、配置快照、模型检查点或V5内置的十二格冻结参考。历史配置中的V3/V4身份字段属于来源凭据，不批量改名。

当前通信仍上传完整实体和关系嵌入表。图局部化提高逻辑活动行复用并不等同于实际网络字节已经下降。
