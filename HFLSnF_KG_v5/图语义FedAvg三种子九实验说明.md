# 图语义FedAvg三种子九实验说明

## 固定矩阵

将本地学习率预先固定为`5e-4`，直接执行三个随机种子乘三个拓扑，共九次独立150轮训练：

| 次序 | 随机种子 | 拓扑 |
|---|---:|---|
| 1 | 42 | HFLSnF |
| 2 | 42 | HFLnoSnF |
| 3 | 42 | FLnoSnF |
| 4 | 2024 | HFLSnF |
| 5 | 2024 | HFLnoSnF |
| 6 | 2024 | FLnoSnF |
| 7 | 2025 | HFLSnF |
| 8 | 2025 | HFLnoSnF |
| 9 | 2025 | FLnoSnF |

所有实验统一FedAvg、`learning_rate=lr=0.0005`、客户端Adam且每次参与重置状态、本地3周期、150通信轮、批量1024、256负样本和原图语义训练配方。不增加学习率搜索，不执行测试集选参或正式测试。

每个种子使用其冻结图语义分区和初始模型哈希；同种子的三拓扑初始化与分区相同，不要求三个不同种子的分区和初始化哈希相同。每种拓扑使用自身冻结调度合同。

新批次九组都从第1轮重新训练，包括seed42，不导入旧三组或学习率扫描结果。旧三组入口和原本地学习率为`5e-5`的九实验入口保留，不修改历史配置或结果。

## 一条批量命令

在仓库根目录，即`HFLSnF_KG_v5`的上一级，使用安装了CUDA版PyTorch及FedML的环境执行：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_multiseed formal9
```

正式入口先检查当前解释器的CUDA和FedML，再复算三个真实分区与调度合同，然后依上述次序逐组运行。禁止回退CPU；需要原数据及`Topo_opt/postprocess/result-U-6fixedge_epoch200_hard_varAlpha_0p1_trainable.mat`。

仅校验配置、真实三分区及调度，不训练：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_multiseed validate
```

恢复和重新生成报告：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_multiseed formal9 --resume "<batch_summary.json>"
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_multiseed report --batch "<batch_summary.json>"
```

`validate --skip-partition-recompute`仅供快速核对，不允许正式训练跳过真实三分区复算。

## 监测与恢复

复用既有监测引擎，每组独立子进程调用原统一训练器。检查完整写出的轮级CSV，非有限平均损失或任一验证指标停止当前组并标记`unstable`，继续下一组；正常MRR波动不淘汰。

环境错误、子进程异常或结果合同失败停止批次，退出1；用户中断停止当前子进程，保存状态并退出130。九组处理结束退出0，状态为`passed`或`completed_with_unstable`，不代表所有候选都有效或性能达到预期。

恢复先重验配置、来源、校准合同、已通过结果及检查点。已经通过或判为不稳定的组跳过，其余组从第1轮重新训练，旧尝试和日志保留。不支持轮内续训；不能用旧三组、旧九实验或扫描批次清单恢复新九实验。迁移工程路径后应新建批次。

## 报告

九份新配置位于`configs/graph_semantic_fedavg_multiseed_lr5e-4/`。结果隔离至`results/graph_semantic_fedavg_multiseed_lr5e-4/`。批次目录以`multiseed_topology_batch_`开头，保存清单、独立尝试日志和分析产物。

每组包含原训练结果及`fedavg_multiseed_lr5e-4_formal150_contract.json`。批次分析目录生成：

- `multiseed_topology_summary.json`：九组指标、同种子配对差及跨种子统计；
- `multiseed_topology_units.csv`：九组曲线统计、预算和哈希；
- `multiseed_topology_paired_differences.csv`：每个种子的三种配对差，共九行；
- `multiseed_topology_cross_seed_summary.csv`：每种拓扑的跨种子均值及样本标准差；
- `multiseed_topology_cross_seed_paired_summary.csv`：同种子配对差的跨种子均值及样本标准差；
- `FedAvg三种子三拓扑九实验报告.md`：中文结果报告。

每组统计最佳验证MRR及轮次、第150轮MRR、末30轮均值与总体标准差、全程最大回撤、训练时间、平均参与人数、分组数及数据暴露量。末30轮标准差是单条曲线的波动，不是跨种子标准差。

跨种子统计采用样本标准差，分别汇总最佳MRR、第150轮MRR和每次运行末30轮均值。配对差先在同种子内做HFLSnF减HFLnoSnF、HFLSnF减FLnoSnF、HFLnoSnF减FLnoSnF，再对三个种子的差值求均值和样本标准差。

某拓扑或某配对缺少任一有效种子时，显示有效种子数且完整三种子统计留空，不以两种子平均值冒充三种子结果。结果合同不依赖预期MRR排序。三种子统计不自动等同于统计显著性结论；三拓扑的参与预算不同，差异仍是系统级复合收益。
