# 阶段六：MAT动态采样与分组联邦TransE

## 目标

本阶段在37个头实体均衡知识客户端上读取MATLAB逐轮拓扑，实现真正变化的客户端参与集合和边缘分组。它不覆盖阶段五的四个固定方案，使用独立入口、配置和结果目录。

正式场景为：

- 结构：HFL；
- SnF：开启；
- 边缘模式：动态；
- 网络利用率：0.5；
- 候选知识客户端：37个；
- 拓扑来源：`matlab/result-U-6fixedge_epoch200_varAlpha_0p5_trainable.mat`。

## MAT文件的实际含义

MAT文件包含200轮和0.1至0.8共8个利用率位置。当前配置读取利用率0.5对应列，并选择`HFLSnF`的动态边缘策略。对全部200轮预检得到：

| 统计项 | 数值 |
| --- | ---: |
| 每轮参与客户端最少值 | 11 |
| 每轮参与客户端最多值 | 37 |
| 每轮参与客户端均值 | 35.730 |
| 不同参与集合数量 | 43 |
| 每轮动态组最少值 | 2 |
| 每轮动态组最多值 | 12 |
| 每轮动态组均值 | 8.135 |
| 不同边缘集合数量 | 189 |

MAT中的云节点18不会被映射为客户端。其余37个物理节点按升序映射到Python的0至36号候选槽位，再与37个知识客户端一一对应。加载阶段会检查非法编号、组内重复、跨组重复、参与人数和分组字段；任一行不一致都会在训练前报错。

配置里的`client_num_per_round: 37`表示FedML候选池容量，不表示每轮强制37人参加。实际人数、客户端编号、组数和组成员完全取自MAT当前行。

## 每轮训练流程

每个通信epoch严格消费一行MAT拓扑：

```text
读取MAT第r行
      ↓
确定本轮参与客户端和动态边缘组
      ↓
所有参与客户端加载同一份轮初全局TransE参数
      ↓
各客户端使用本地三元组训练1个本地epoch
      ↓
每个动态组按本地正三元组数聚合完整参数表
      ↓
云端合并各组的加权分子和分母
      ↓
广播新的全局实体及关系嵌入表
```

组内和云端都保留加权分子与分母，所以两级稠密FedAvg与把同一批参与客户端直接送到云端聚合数学等价。动态分组改变通信路径和边缘统计，但在当前稠密FedAvg下，只有参与客户端集合变化会改变全局参数；若希望组结构本身改变模型结果，需要后续加入组级非线性权重、异步时延或实体关系行级掩码。

## 正式参数

服务器配置使用以下折中参数：

| 参数 | 数值 | 说明 |
| --- | ---: | --- |
| 通信轮数 | 200 | 完整消费MAT的200行拓扑 |
| 本地epoch | 1 | 避免动态客户端单轮过拟合 |
| 嵌入维度 | 256 | 与当前固定四方案一致 |
| 距离 | L2 | 与当前固定四方案一致 |
| 每个正样本负样本数 | 5 | 控制4060 Laptop的Python负采样开销 |
| 批次大小 | 4096 | 减少小批次调度次数 |
| 学习率 | 0.001 | 保持当前联邦TransE稳定设置 |
| 选模周期 | 10轮 | 每轮打印训练信息，每10轮计算排名指标 |
| 选模验证三元组 | 1000 | 在速度和选模稳定性之间折中 |
| 候选实体批次 | 16384 | 降低filtered评估的内核启动次数 |

200轮下的真实等效完整数据遍历次数不是固定200。程序会根据每轮被选客户端的本地三元组数计算`effective_full_data_passes`并写入汇总，不能再用`comm_round × epochs`代替。

## 运行方式

服务器终端运行：

```bash
python -m HFLSnF_KG.run_dynamic_federated_transe \
  --cf HFLSnF_KG/configs/server_fb15k237_hflsnf_dynamic_mat_cuda.yaml
```

在PyCharm或VS Code中直接运行`HFLSnF_KG/run_from_ide.py`。当前默认方案是：

```python
DEFAULT_PROFILE = "dynamic_fedtranse_hflsnf_mat_cuda"
```

也可以通过环境变量选择：

```powershell
$env:HFLSNF_KG_IDE_PROFILE = "dynamic_fedtranse_hflsnf_mat_cuda"
python HFLSnF_KG\run_from_ide.py
```

正式配置设置`require_cuda: true`。没有CUDA时会在读取FB15k-237前立即报错，不会静默回退到CPU。

## 输出说明

每次运行使用独立时间戳目录，主要文件包括：

- `config_snapshot.json`：实际FedML参数；
- `dataset_summary.json`：FB15k-237规模；
- `client_partition_summary.json`：37个客户端的知识规模和划分哈希；
- `topology_metadata.json`：MAT路径、场景、物理节点映射和完整200轮统计；
- `dynamic_participation_summary.json`：本次使用轮数、调度哈希、参与频次、动态规模和真实数据暴露预算；
- `dynamic_topology_schedule.jsonl`：每轮MAT行号、客户端编号、动态组、边缘节点、客户端权重和组级聚合权重；
- `metrics.csv`：每轮损失、耗时、参与比例以及周期性验证MRR和Hits@3；
- `summary.json`：最佳轮次、最终完整验证及测试指标和集中式MRR差值；
- `model_best.pt`：恢复最佳验证轮次后的完整检查点。

终端每个epoch都会打印一行信息。为避免严重拖慢训练，MRR和Hits@3只在`eval_every`指定的选模轮计算；其他轮明确显示“未评估”。
