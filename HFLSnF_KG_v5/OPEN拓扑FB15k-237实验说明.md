# OPEN拓扑＋FB15k-237联邦实验说明

本轮保持此前 OPEN＋WN18RR 的训练配方，只切换为 FB15k-237，进行种子42的 HFLSnF、HFLnoSnF、FLnoSnF 三方法对照。真实数据分区和调度预检已通过；本机没有可用的 CUDA 张量计算环境，正式训练尚未执行。

## 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | FB15k-237，默认 `data/FB15k-237` |
| 拓扑 | OPEN，18个客户端；分层方法4个边缘槽位 |
| 参与调度 | 沿用0.1倍人数方差版本重建的独立合成调度 |
| 数据划分 | Freebase顶级语义域×头实体包、图局部性、±5%负载、8次重启 |
| 方法顺序 | HFLSnF、HFLnoSnF、FLnoSnF |
| 种子 | 42 |
| 模型 | TransE，256维，L1距离 |
| 聚合与优化 | 逐行计数加权FedAvg，本地Adam，每次本地训练重置优化器状态 |
| 学习率 | `learning_rate = lr = 5e-4` |
| 预算 | 150个通信轮次，每次本地训练3个周期 |
| 批量与负样本 | 1,024，每正样本256个负样本 |
| 目标函数 | 双向自对抗，gamma＝9，温度＝1 |
| 网络利用率 | 0.6，严格使用前150轮，不循环 |
| 验证选模 | 每轮固定关系分层抽取的4,096条验证事实，按验证MRR选最佳模型 |
| 完整测试 | 训练结束后用独立入口评估全部20,466条测试事实、40,932次头尾查询 |
| 正式设备 | CUDA，禁止回退到处理器训练 |

没有应用此前讨论的本地周期减少或学习率下降，便于先建立与词汇图谱实验配方一致的对照。数据规模不同，实际小批次数和每轮优化步数也随之改变，不能把相同通信轮数当作相同计算量。

## 真实预检结果

三方法采用同一分区和初始模型。训练272,115条、验证17,535条、测试20,466条；图语义划分包含30个语义域、33,155个不可拆分包。

- 客户端训练样本数：14,362～15,873。
- 最大相对负载偏差：4.9975%，满足±5%约束。
- 按三元组加权的主域占比：84.54%。
- 三方法前150轮均覆盖全部18个客户端。
- 分区摘要：`da1762a9523dc22df69a329acb90fde55ca94b29a9a7d7a1a0ec94a0b92ca42e`。

| 方法 | 平均参与人数 | 人数样本方差 | 有效全数据暴露遍数 |
|---|---:|---:|---:|
| HFLSnF | 17.747 | 0.7005 | 443.644 |
| HFLnoSnF | 10.567 | 2.4754 | 264.114 |
| FLnoSnF | 4.913 | 0.9522 | 122.825 |

这些是输入与预算统计，不是训练性能结果。预检保存在 `artifacts/multi_topology_dataset/validation_open_fb15k-237_seed42.json`，不会覆盖原 WN18RR 预检文件。

## 运行命令

在 `D:\1\1myworkcode` 仓库根目录执行。先激活显卡训练环境，确认 CUDA 可用。

```powershell
python -m HFLSnF_KG_v5.run_multi_topology_dataset validate --dataset fb15k-237
python -m HFLSnF_KG_v5.run_multi_topology_dataset formal3 --dataset fb15k-237
```

正式入口会自行复算并校验输入，顺序启动三方法独立训练。结果写入：

```text
HFLSnF_KG_v5/results/multi_topology_dataset/open_fb15k-237_seed42/
```

中断或失败后恢复时仍需选择同一数据集：

```powershell
python -m HFLSnF_KG_v5.run_multi_topology_dataset formal3 --dataset fb15k-237 --resume "<batch_summary.json的路径>"
python -m HFLSnF_KG_v5.run_multi_topology_dataset report --batch "<batch_summary.json的路径>"
```

已通过的组会校验检查点摘要后跳过；失败组从头重跑。报告展示验证指标，不把验证结果当作完整测试结果。

对三个训练目录分别执行完整测试：

```powershell
python -m HFLSnF_KG_v5.run_best_checkpoint_official_evaluation --dataset fb15k-237 --result-dir "<单个训练结果目录>" --using-gpu --require-cuda
```

新增三份配置位于 `configs/multi_topology_dataset/open_fb15k237_*_seed42.yaml`。命令行 `--dataset` 选择内置三方法配置，兼容 `fb15k237` 别名和大小写；`--configs` 可选择自定义配置，两者互斥。未填写两项时仍默认使用原 WN18RR 三方法配置。

## 调度与复现边界

调度文件为 `artifacts/topologies/open_varalpha0p1_reconstructed.mat`，与前一次词汇图谱实验相同。其客户端身份按受控人数重建，未重新执行网络优化，也未验证链路物理可行性；比较结论应限定为该合成参与调度下的学习行为。

迁移到显卡机器时需携带真实数据和调度文件；它们受仓库忽略规则影响。若未携带重建调度，需要先携带原受控统计文件，再执行通用入口的 `prepare`。源文件位置和重建规则详见[多拓扑多数据集实验说明](多拓扑多数据集实验说明.md)。
