# 图语义FedAvg种子42三拓扑实验说明

## 实验范围

根据种子42的HFLSnF四学习率验证结果，预先固定本地学习率为`5e-4`，仅执行以下三个独立实验，不启动2024、2025种子，也不继续搜索学习率：

| 次序 | 拓扑 | 架构 | 存储转发 | 边缘层 | 本地学习率 | 通信轮数 |
|---|---|---|---|---|---:|---:|
| 1 | HFLSnF | 分层 | 开启 | 固定6组 | 5e-4 | 150 |
| 2 | HFLnoSnF | 分层 | 关闭 | 固定6组 | 5e-4 | 150 |
| 3 | FLnoSnF | 扁平 | 关闭 | 无边缘层 | 5e-4 | 150 |

三组均从相同初始化重新训练，包括已在学习率扫描中完成的HFLSnF；不复用或覆盖扫描结果。共同使用FB15k-237、种子42图语义分区、256维TransE、L1范数、双向自对抗负采样、本地Adam且每次参与重置状态、本地3周期、批量1024、256个负样本及逐行出现次数加权FedAvg。同步设置`learning_rate=lr=0.0005`。

配置从原三拓扑FedAvg的seed42配置分别派生；除学习率及实验身份外，保留各自来源配方。三组之间只允许拓扑字段和审计身份不同。初始模型哈希、分区哈希相同；调度哈希按每种拓扑的冻结合同校验，不要求跨拓扑相同。

## 运行命令

在仓库根目录，即`HFLSnF_KG_v5`的上一级执行。正式环境需要CUDA版PyTorch、可正常导入的FedML、完整数据以及原MAT调度文件`Topo_opt/postprocess/result-U-6fixedge_epoch200_hard_varAlpha_0p1_trainable.mat`。

校验配置、数据、真实种子42分区和三种150轮调度，不执行训练：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_seed42 validate
```

在GPU机器依次执行三组150轮并自动生成报告：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_seed42 formal3
```

正式入口首先检查当前解释器的CUDA和FedML，不回退CPU；环境不满足时不创建训练批次。真实分区复算不能跳过。`validate --skip-partition-recompute`仅用于快速校验配置和真实调度。

恢复使用程序打印的批次清单：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_seed42 formal3 --resume "<batch_summary.json>"
```

重新生成报告，不需要CUDA：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_seed42 report --batch "<batch_summary.json>"
```

## 监测与恢复

新入口复用四学习率实验的监测引擎，并通过原统一训练入口启动每个拓扑的独立子进程，不改变训练算法。

- 每秒检查已完整写出的轮级CSV记录；客户端平均损失或任一验证指标非有限时，停止当前组，记录`unstable`并继续下一组。
- 有限的MRR正常波动不淘汰候选；剖析关闭时的计时占位非数值不视为数值发散。
- 子进程异常退出、环境故障、结果合同失败或新增目录不唯一时停止批次，退出码1。
- 用户中断停止当前子进程，保存恢复信息并返回130；终止10秒超时后强制结束，仅清理当前入口创建的子进程。
- 三组处理结束返回0，批次为`passed`或`completed_with_unstable`；返回0不意味着每组均通过，也不意味着性能符合预期排序。

恢复重新验证配置、来源配置、校准合同、通过结果和最佳检查点哈希；已通过及已判不稳定组跳过，其余组从第1轮重新训练，保留旧尝试并写入新日志。不支持轮内续训。不能使用旧四学习率或九实验的批次清单；迁移工程路径后应新建批次。

数值检测发生在完整轮记录写出后。若训练在写出指标前崩溃，按执行异常处理，不自动断言数值发散。

## 输出与解读

新配置位于`configs/graph_semantic_fedavg_seed42_lr5e-4/`，新结果位于`results/graph_semantic_fedavg_seed42_lr5e-4/`，与所有旧结果隔离。

每组训练目录包含配置快照、轮级CSV、聚合审计、汇总、最佳模型及额外的`fedavg_seed42_lr5e-4_formal150_contract.json`。批次目录以`seed42_topology_batch_`开头，包含`batch_summary.json`、`logs/`和`analysis/`。

报告文件为：

- `seed42_topology_summary.json`；
- `seed42_topology_units.csv`；
- `FedAvg种子42三拓扑实验报告.md`。

汇总每组最佳验证MRR及轮次、第150轮MRR、末30轮均值及总体标准差、全程最大绝对回撤和轮级训练时间之和，并记录平均参与客户端数、平均分组数及有效数据遍数。第150轮指标来自CSV，不把训练结束后恢复的最佳模型验证值当作末轮值。

同时计算HFLSnF减HFLnoSnF、HFLSnF减FLnoSnF和HFLnoSnF减FLnoSnF的验证指标配对差。只对均通过的完整150轮结果计算配对差；未完成或不稳定组的有限前缀不能替代完整候选。结果有效性不依赖三组MRR排序。

三种调度的参与人数和数据暴露量不同，差值表示固定150轮预算下的系统级复合收益，不能直接归因于纯分层聚合机制。单种子结果与逐轮标准差不构成跨种子显著性证据，也不能证明优化器普遍优劣。

本轮只按验证集选模，不执行测试集评估，不根据测试结果改学习率。正式测试、多种子扩展与等参与预算对照属于后续实验。
