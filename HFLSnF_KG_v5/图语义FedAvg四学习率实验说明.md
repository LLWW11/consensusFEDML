# 图语义FedAvg四学习率实验说明

## 实验矩阵

固定HFLSnF、随机种子42，按以下顺序运行四次独立正式训练：

| 次序 | 配置标识 | 客户端学习率 | 通信轮数 |
|---|---|---:|---:|
| 1 | `lr5e-5` | `5e-5` | 150 |
| 2 | `lr1e-4` | `1e-4` | 150 |
| 3 | `lr2e-4` | `2e-4` | 150 |
| 4 | `lr5e-4` | `5e-4` | 150 |

四组均从相同初始化重新训练，第一组不会复用原有结果。每份配置同步设置`learning_rate`与`lr`，客户端仍使用Adam且每次参与重置状态；本地训练3周期，批量大小1024，负样本256，服务器直接采用逐行加权FedAvg候选模型。

配置来源为`configs/graph_semantic_fedavg_comparison/graph_semantic_fedavg_hflsnf_seed42_150round_cuda.yaml`。除学习率、实验身份和隔离输出路径外，其他设置与来源一致。原三拓扑九实验入口及其配置、结果保持独立。

## 环境与运行

在仓库根目录，即`HFLSnF_KG_v5`的上一级，使用安装了CUDA版PyTorch和FedML的Python环境。需要完整FB15k-237数据和原MATLAB调度文件：`Topo_opt/postprocess/result-U-6fixedge_epoch200_hard_varAlpha_0p1_trainable.mat`。

先校验四份配置、真实数据哈希、种子42分区与150轮调度，不执行训练：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_lr_sweep validate
```

一条命令依次执行四组，并自动生成报告：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_lr_sweep formal4
```

正式入口首先检查CUDA和FedML，再进行真实分区复算。无CUDA时明确拒绝启动，不回退CPU，也不创建正式训练批次。`validate --skip-partition-recompute`仅用于快速静态核对，不能作为正式启动参数。

## 数值监测与恢复

每组通过独立子进程调用原`run_federated_transe`训练链路。入口每秒检查已写完整的CSV记录，并打印最新完成轮的MRR和损失；未写完的末行暂不解析。

- 每轮客户端平均训练损失或任一验证数值出现NaN、正无穷或负无穷时，停止当前组并标记`unstable`，继续下一组。
- 未开启性能剖析时的计时占位NaN不属于不稳定。正常MRR回落也不自动淘汰，有限候选继续完成150轮。
- 环境错误、子进程异常退出、结果合同失败或非唯一结果目录会停止批次，并记录`failed`。
- 用户中断时停止当前子进程，保存`interrupted`状态，退出码130；其他执行失败退出码1。
- 四组均结束后，状态为`passed`或`completed_with_unstable`，退出码0表示批次处理完成，不表示每组性能达标。

数值检查发生在每轮指标完成写出之后，不是每个本地批次检查。若训练在写出轮指标前异常退出，则按执行失败处理，保留日志并停止批次。

终止请求只作用于入口创建的训练子进程；若10秒内未退出，再强制结束并等待退出。

恢复命令使用入口打印的批次清单：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_lr_sweep formal4 --resume "<batch_summary.json>"
```

恢复先核对配置、来源配置及校准合同哈希；重新验证已通过组的结果合同和最佳检查点内容。已经`passed`或`unstable`的组跳过，其他组从第1轮重新训练，追加新尝试日志。当前不支持从某一通信轮续训，也不覆盖旧尝试目录。清单绑定运行机器上的配置路径，迁移工程后应新建批次。

## 输出与判读

全部产物位于`results/graph_semantic_fedavg_lr_sweep/`：

- 每次训练使用原训练器的独立时间戳目录，包含配置快照、逐轮指标、聚合审计、训练汇总和最佳验证模型。
- 每个批次拥有`batch_summary.json`，`logs/`保存每组每次尝试的独立训练日志。
- 每个通过的训练目录额外保存`fedavg_lr_sweep_formal150_contract.json`，验证学习率、150轮编号、初始化及分区和调度哈希，并确认没有服务器动量。
- 批次`analysis/`下生成`lr_sweep_summary.json`、`lr_sweep_units.csv`和`FedAvg四学习率实验报告.md`。

报告包含最佳验证MRR及轮次、第150轮MRR、末30轮MRR均值与总体标准差、全程最大绝对MRR回撤、轮级训练时间之和及完成状态。最佳MRR使用最早达到该值的轮次，与现有选模规则一致。

不稳定组只汇总非有限记录之前的有限曲线前缀。第150轮和末30轮指标仅对完整有限的150轮曲线计算；有限前缀最佳值不能当作150轮模型比较。

报告可在无CUDA环境重新生成，仍会重新校验通过组的结果和最佳检查点：

```powershell
python -m HFLSnF_KG_v5.run_graph_semantic_fedavg_lr_sweep report --batch "<batch_summary.json>"
```

本实验不执行测试集评估，也不按测试结果选择学习率。固定种子42的结果只能支持本轮筛选，不能替代后续多种子复核。
