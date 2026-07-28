# V2 FedE式行级聚合与本地训练消融

## 实验目的

V2在现有37客户端动态MAT联邦TransE上引入两个可独立观察的变化：

1. 把完整参数表稠密FedAvg改为实体及关系行级所有权聚合；
2. 把当前L2、头尾负采样和间隔排序损失改为FedE式L1、尾负采样和自对抗逻辑损失。

本阶段的验收标准是语义正确、口径一致和输出完整，不预设MRR一定提高。

## 行所有权

客户端只根据本地正训练三元组声明知识所有权：

- 头实体和尾实体对应的实体行标记为真；
- 正三元组关系对应的关系行标记为真；
- 负采样偶然访问的实体行不取得所有权；
- 行出现次数只用于结果诊断，不作为聚合权重。

设当前轮活跃客户端集合为`S`，客户端`k`对参数行`i`的布尔所有权为`m(k,i)`，本地行参数为`w(k,i)`。云端行参数为：

```text
若存在活跃拥有者：
    w(i) = sum[m(k,i) * w(k,i)] / sum[m(k,i)]
否则：
    w(i) = 本轮开始时的全局参数行
```

MAT没有选中的客户端不会训练，也不会进入分子或分母。

## 两级聚合

每个动态边缘组不提前计算普通均值，而是保存每行分子和分母：

```text
边缘分子 = 组内拥有者参数行之和
边缘分母 = 组内拥有者数量
```

云端对所有边缘分子和分母分别求和，再统一归一化。这样可以保证两级行聚合与同一参与集合的直接行聚合等价，也不会让组大小改变客户端行权重。

## FedE式本地目标

公平预算和论文参数参考方案都只替换尾实体。采样器继续使用全局已知真三元组集合过滤，避免37客户端共享关系时把其他客户端或验证、测试中的真事实当作负例。

TransE距离为：

```text
d(h,r,t) = ||h + r - t||_1
```

逻辑分数为：

```text
s(h,r,t) = gamma - d(h,r,t)
```

负样本先按`softmax(temperature * s)`得到自对抗权重，计算权重时停止梯度，再计算正样本与负样本逻辑损失的平均值。默认`gamma=10`、`temperature=1`。

## 与FedE开源实现的差异

本实现迁移`1paperAbout/FedE-master/fede.py`、`dataloader.py`和`kge_model.py`中的核心语义，但有以下有意差异：

- FedE服务端只聚合全局实体，关系嵌入由客户端私有保存；V2为了统一全局评估，同时行级聚合实体和关系。
- FedE数据按互斥关系集合划分少量客户端；V2保留当前37客户端头实体均衡划分。
- FedE原负采样只按客户端本地`(头实体, 关系)`真尾实体过滤；V2使用全局训练、验证和测试真三元组过滤。
- V2只接收MAT当前轮选中客户端的更新，避免未参与客户端的旧全局行稀释活跃更新。
- V2继续使用当前完整验证集和测试集的头尾双向filtered评估。

因此，配置名中的“FedE”表示损失与行所有权聚合语义的兼容移植，不表示复现FedE论文的关系划分、私有关系和客户端数量。

## 三套配置

### 当前目标加行掩码

配置文件：

`configs/server_fb15k237_hflsnf_dynamic_mat_masked_cuda.yaml`

使用256维、L2、5个头尾负样本、批次4096和1个本地epoch，只把聚合切换为实体及关系行级所有权平均。

### 公平预算FedE

配置文件：

`configs/server_fb15k237_hflsnf_dynamic_mat_masked_fede_fair_cuda.yaml`

使用256维、L1、5个尾负样本、批次4096和1个本地epoch。该方案和当前动态方案匹配本地遍历次数及负样本数量，适合优先比较。

### FedE论文参数形态参考

配置文件：

`configs/server_fb15k237_hflsnf_dynamic_mat_masked_fede_paper_cuda.yaml`

使用128维、L1、256个尾负样本、批次512和3个本地epoch。其每轮本地开销远高于公平预算方案，不属于等计算预算比较。

## 结果字段

每个通信epoch都会打印MAT行号、参与客户端数、动态组数、本地加权损失、轮次耗时、MRR和Hits@3状态。周期性评估仍由`eval_every`控制，非评估轮打印“未评估”。

`metrics.csv`新增：

- `aggregation_mode`和`local_objective`；
- `entity_updated_row_count`和`entity_fallback_row_count`；
- `relation_updated_row_count`和`relation_fallback_row_count`。

`dynamic_topology_schedule.jsonl`逐轮新增：

- 云端`parameter_row_statistics`；
- 每个动态组的`group_parameter_row_statistics`；
- 每组和整轮实际贡献客户端；
- 聚合模式和本地目标。

`summary.json`保存划分哈希、MAT调度哈希、聚合模式、本地目标、最佳轮和最终完整filtered指标。

## 运行方式

如果目标是正式比较稠密聚合、行级聚合和FedE公平目标，优先使用`THREE_ARM_ABLATION.md`中的三臂入口。该入口会自动检查公平性并在三组完成后生成统一报告。

IDE直接运行`run_from_ide.py`，默认方案是：

```python
DEFAULT_PROFILE = "dynamic_fedtranse_masked_fede_fair_cuda"
```

终端公平预算方案：

```bash
python -m HFLSnF_KG_v2.run_dynamic_federated_transe \
  --cf HFLSnF_KG_v2/configs/server_fb15k237_hflsnf_dynamic_mat_masked_fede_fair_cuda.yaml
```

三套正式配置都设置`require_cuda: true`。无CUDA时会在读取FB15k-237前报错；本机只运行合成小图CPU测试，不运行200轮正式训练。
