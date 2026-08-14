# 配置文件说明

本目录只保留阶段二实验完成后确定的正式动态拓扑配置。历史筛选、消融、对照与烟雾测试配置已移动到 [`zOld/`](zOld/README.md)，不再作为默认实验入口。

## 当前正式配置

正式配置采用以下统一参数：

- 动态拓扑：读取`Topo_opt/postprocess`中的alpha=0.1硬覆盖后处理MAT，`topology_util=0.6`；
- 服务器优化器：FedAdam，学习率为`0.05`，`tau=0.001`；
- 偏差修正：关闭；
- 本地训练：每轮3个epoch；
- 训练与评估：150轮，每轮评估；
- 随机种子：42、2024和2025；
- 测试集：训练结束后不自动评估，保留验证选型与最终测试隔离。

三个实验臂共享同一训练参数和同一seed的分区哈希，只在HFL/FL架构、SnF开关、边缘组设置、MAT调度哈希及派生身份字段上不同。三个seed对应的分区哈希已经根据阶段二正式结果固化；同一实验臂在三个seed下复用同一150轮MAT调度哈希。

| 实验臂 | 随机种子 | 配置文件 |
| --- | ---: | --- |
| HFLSnF | 42 | `final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed42_150round_cuda.yaml` |
| HFLnoSnF | 42 | `final_dynamic_fedadam_hflnosnf_u0p6_bcfalse_seed42_150round_cuda.yaml` |
| FLnoSnF | 42 | `final_dynamic_fedadam_flnosnf_u0p6_bcfalse_seed42_150round_cuda.yaml` |
| HFLSnF | 2024 | `final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed2024_150round_cuda.yaml` |
| HFLnoSnF | 2024 | `final_dynamic_fedadam_hflnosnf_u0p6_bcfalse_seed2024_150round_cuda.yaml` |
| FLnoSnF | 2024 | `final_dynamic_fedadam_flnosnf_u0p6_bcfalse_seed2024_150round_cuda.yaml` |
| HFLSnF | 2025 | `final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed2025_150round_cuda.yaml` |
| HFLnoSnF | 2025 | `final_dynamic_fedadam_hflnosnf_u0p6_bcfalse_seed2025_150round_cuda.yaml` |
| FLnoSnF | 2025 | `final_dynamic_fedadam_flnosnf_u0p6_bcfalse_seed2025_150round_cuda.yaml` |

## 使用方式

在项目根目录的上一级目录运行，例如：

```powershell
python -m HFLSnF_KG_v3.run_federated_transe --cf HFLSnF_KG_v3/configs/final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed42_150round_cuda.yaml
```

推荐先执行不启动训练的九组合同校验：

```powershell
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam validate
```

一次性按`seed42 → seed2024 → seed2025`运行；每个seed内部固定执行`HFLSnF → HFLnoSnF → FLnoSnF`：

```powershell
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam formal150
```

任一训练或结果合同失败时，批次会立即停止并给出恢复命令。使用生成的批次清单恢复：

```powershell
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam formal150 --resume "HFLSnF_KG_v3/results/final_dynamic_fedadam_batch_<时间戳>/batch_summary.json"
```

切换单个实验臂或随机种子时，只需替换为表中对应的配置文件。九份配置已经分别固化分区哈希与三种实验臂的调度哈希，不应在运行前手工交换哈希字段。

这些配置复现的是MAT动态参与者组成、覆盖、分组和参与预算共同形成的系统级结果。HFLSnF、HFLnoSnF与FLnoSnF之间的差异不能在缺少严格配对拓扑控制时单独归因于SnF或分层机制。

硬覆盖后处理保持三组实验臂原有的逐轮参与人数不变，并确保前150轮覆盖全部37个客户端。`topology_util=0.6`下，FLnoSnF每轮仍只实际参与4至8个客户端，因此三组实验臂的参与预算和分组机制仍不相同，不能解释为严格单因素对照。

## 维护原则

- 新的正式配置保留在本目录根级；阶段性配置放入`zOld/`相应分类。
- YAML中的注释统一使用英文。
- 不要直接修改已归档配置；需要复现实验时应复制到新文件并使用清晰的新身份字段。
- 配置归档只改变文件位置，不改变历史YAML内容。
