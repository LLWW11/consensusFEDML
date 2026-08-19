# HFLSnF KGE V4：重叠率消融实验工程

## 工程定位

本目录由 `HFLSnF_KG_v3` 复制并精简而来，用于在 FB15k-237 上继续开展联邦知识图谱嵌入实验。V4 保留 V3 的 TransE、本地训练、逐行聚合、服务器 FedAdam、动态拓扑和正式评估实现，但当前新增研究重点是：在固定 HFLSnF 动态正式配置下，把客户端知识子图实体重叠率作为可控变量。

完整的重叠率定义、校准规则、12格实验矩阵和停止条件见[重叠率消融实验计划](重叠率消融实验计划.md)。目标重叠划分器、8重启正式校准合同、9份YAML、批量训练、恢复、12格官方测试和报告入口均已实现。当前尚未产生9组新训练结果。

## 当前保留内容

实际结果目录名为 `results/`，不是 `result/`。V4 当前只保留两类已有结果：

1. `results/三个随机数种子/`：HFLSnF、HFLnoSnF、FLnoSnF 三个动态实验臂在 seed42、seed2024、seed2025 下的 9 个正式结果；
2. `results/固定人数随机抽取/`：每轮参与人数 $K=6,12,18,24,30,36$ 的 6 个 seed42 消融结果。

动态三种子结果中的三个 HFLSnF 结果将作为重叠率消融的“原始划分对照”。另外两个动态实验臂和固定人数结果只作为既有背景，不加入本次单模型重叠率正式矩阵。

`configs/dynamic/` 和 `configs/stochastic/` 各保留9份正式配置，`configs/overlap/` 新增9份重叠率配置和1份校准合同。当前V4只保留了动态三种子结果，没有保留随机拓扑或重叠率新训练结果。历史筛选、早期动态实验、固定四场景和人数消融配置仍在 `configs/zOld/`，用于审计和必要时复现。

## 已冻结的主实验条件

重叠率消融固定使用 HFLSnF 动态正式配置：

- 数据集：FB15k-237；
- 客户端总数：37；
- 模型：256维 TransE，L1距离；
- 拓扑：MATLAB alpha=0.1 硬覆盖动态调度，`topology_util=0.6`；
- 通信轮次：150；
- 本地训练：每轮3个epoch，Adam学习率 $5\times10^{-5}$；
- 聚合：逐行出现次数加权；
- 服务器优化器：FedAdam，学习率0.05，$\tau=0.001$，$\beta_1=0.9$，$\beta_2=0.99$；
- 服务器偏差修正：关闭；
- 评估：每轮验证，训练阶段不自动读取测试集；
- 随机种子：42、2024、2025。

后续低、中、高重叠实验只允许改变客户端分区及其哈希，不允许改变上述训练、拓扑和评估条件。

## 重叠率实验概览

正式矩阵为“原始划分对照＋低/中/高重叠”四种数据条件乘以三个随机种子，共12格。

| 数据条件 | seed42 | seed2024 | seed2025 |
| --- | --- | --- | --- |
| 原始划分对照 | 复用已有HFLSnF结果 | 复用已有HFLSnF结果 | 复用已有HFLSnF结果 |
| 低重叠 | 配置就绪，待训练 | 配置就绪，待训练 | 配置就绪，待训练 |
| 中重叠 | 配置就绪，待训练 | 配置就绪，待训练 | 配置就绪，待训练 |
| 高重叠 | 配置就绪，待训练 | 配置就绪，待训练 | 配置就绪，待训练 |

因此仍需新增9次训练。8重启正式校准及独立九分区复算已经通过，共同可达区间为0.172249～0.285359，跨度为0.113110；低、中、高正式目标分别为0.172249、0.228804和0.285359。当前 `balanced_head_entity` 原始划分的归一化实体重叠率约为0.243，继续作为独立对照，不并入低、中、高命名。

## 当前可用命令

以下命令包含现有配置入口和只执行数据划分的重叠率校准入口。

运行三个种子的无训练重叠率校准：

```powershell
python -m HFLSnF_KG_v4.run_overlap_partition_calibration --output HFLSnF_KG_v4/configs/overlap/partition_calibration_contract.json
```

该命令不创建模型、不读取CUDA，也不启动训练。默认使用37个客户端、种子42/2024/2025、目标容差0.005、负载容差0.05、关系重叠容差0.02和8次确定性重启，并自动独立复算九个正式分区确认哈希可复现。当前正式合同已经生成，无需重复校准。

校验、训练、恢复、官方测试和报告命令：

```powershell
python -m HFLSnF_KG_v4.run_overlap_ablation validate
python -m HFLSnF_KG_v4.run_overlap_ablation formal150
python -m HFLSnF_KG_v4.run_overlap_ablation formal150 --resume "<batch_summary.json>"
python -m HFLSnF_KG_v4.run_overlap_ablation official12 --batch "<batch_summary.json>"
python -m HFLSnF_KG_v4.run_overlap_ablation report --batch "<batch_summary.json>"
```

正式训练和完整官方测试要求CUDA，不允许回退CPU。运行顺序固定为seed42低中高、seed2024低中高、seed2025低中高；seed42三组合同未全部通过时不会进入后续种子。

在仓库根目录 `D:\1\1myworkcode` 下运行动态配置只读校验：

```powershell
python -m HFLSnF_KG_v4.run_final_dynamic_fedadam validate
```

运行现有9份动态正式配置：

```powershell
python -m HFLSnF_KG_v4.run_final_dynamic_fedadam formal150
```

运行一份现有 HFLSnF 配置：

```powershell
python -m HFLSnF_KG_v4.run_federated_transe --cf HFLSnF_KG_v4/configs/dynamic/final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed42_150round_cuda.yaml
```

随机拓扑配置只读校验：

```powershell
python -m HFLSnF_KG_v4.run_final_stochastic_fedadam validate
```

历史固定人数配置和运行方式见[配置说明](configs/README.md)与[历史配置归档说明](configs/zOld/README.md)。当前保留的固定人数结果无需重新训练。

## 完整官方测试

最终正式配置在训练阶段设置 `evaluate_test_after_training=false`，避免用测试集参与模型选择。训练完成并通过合同后，将结果目录显式传给官方测试入口：

```powershell
python -m HFLSnF_KG_v4.run_best_checkpoint_official_evaluation `
  --result-dir "D:\1\1myworkcode\HFLSnF_KG_v4\results\正式结果目录" `
  --using-gpu `
  --require-cuda `
  --query-batch-size 64 `
  --candidate-batch-size 8192
```

默认读取 `model_best.pt`，并在目标结果目录的 `full_official_evaluation/` 下写入完整 filtered 头预测、尾预测和综合指标。

## 工程结构

```text
HFLSnF_KG_v4/
├─ README.md
├─ 实验流程说明.md
├─ 重叠率消融实验计划.md
├─ configs/
│  ├─ dynamic/                 # 9份动态正式配置
│  ├─ overlap/                 # 校准合同和9份重叠率正式配置
│  ├─ stochastic/              # 9份随机拓扑正式配置
│  └─ zOld/                    # 35份历史配置
├─ data/FB15k-237/             # 官方训练、验证、测试文本
├─ core/                       # 拓扑、聚合、FedAdam和结果写入
├─ fedml_kge/                  # 客户端与动态训练流程
├─ tasks/kge/                  # 数据、分区、模型、评估和实验合同
├─ tests/                      # 单元测试与配置合同测试
├─ results/
│  ├─ 三个随机数种子/          # 9个动态正式结果
│  └─ 固定人数随机抽取/        # 6个人数消融结果
├─ run_final_dynamic_fedadam.py
├─ run_final_stochastic_fedadam.py
├─ run_overlap_partition_calibration.py
├─ run_overlap_ablation.py
├─ run_federated_transe.py
└─ run_best_checkpoint_official_evaluation.py
```

## 文档导航

- [实验流程说明](实验流程说明.md)：当前数据划分、本地训练、分层聚合和服务器更新流程；
- [重叠率消融实验计划](重叠率消融实验计划.md)：主变量、校准合同、实验矩阵和分析口径；
- [配置说明](configs/README.md)：现有动态、随机和归档配置；
- [数据说明](data/README.md)：FB15k-237本地副本及加载规则；
- [历史参数规划分析](reports/实验参数规划分析.md)：V3时期参数筛选记录，仅作历史参考。

## 历史结果与V3名称

V4 中保留的结果由 V3 运行产生，因此部分结果目录、JSON、HTML、配置快照和绝对路径仍含 `HFLSnF_KG_v3` 或 `hflsnf_kg_v3`。这些字段属于历史来源凭据，不代表 V4 的当前运行入口。

不得为了统一名称而修改已有结果产物、批次清单、配置快照、合同哈希或带V3名称的结果目录。V4新命令统一使用 `HFLSnF_KG_v4`，未来新实验使用新的V4身份字段和结果目录。

## 结果解释边界

现有动态三臂使用不同的MAT参与和分组过程，因此它们只能解释为动态编排造成的系统级差异，不能单独归因于SnF或分层结构。

正式重叠率消融只使用 HFLSnF 单模型，可以解释固定 HFLSnF 配置下实体重叠率变化的影响，但不能单独证明 HFLSnF 相对其他算法的拓扑收益。

当前客户端上传完整实体和关系嵌入表。降低局部实体重叠率不会自动降低实际密集通信字节数；在实现稀疏行上传前，只能报告逻辑活动行和潜在稀疏通信量。
