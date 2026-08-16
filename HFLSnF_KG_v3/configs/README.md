# 配置文件说明

正式配置按拓扑来源分为动态拓扑和随机拓扑两个目录。历史筛选、消融、对照与烟雾测试配置保存在 [`zOld/`](zOld/README.md)，不作为当前正式实验入口。

## 目录结构

- `dynamic/`：9份MAT动态拓扑正式配置，覆盖HFLSnF、HFLnoSnF、FLnoSnF和随机种子42、2024、2025。
- `stochastic/`：9份随机拓扑正式配置，覆盖三个MAT平均预算参考档位和相同的三个随机种子。
- `zOld/`：35份历史配置，保持原有内容和分类不变。

## 动态拓扑实验

动态配置读取 `Topo_opt/postprocess` 中 `util=0.6` 的硬覆盖MAT调度，逐轮参与者和分组完全由MAT决定。服务器优化器、本地训练轮数、评估频率和数据划分合同保持为最终FedAdam实验口径。

先校验9份动态配置：

```powershell
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam validate
```

运行或恢复9组正式实验：

```powershell
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam formal150
python -m HFLSnF_KG_v3.run_final_dynamic_fedadam formal150 --resume "HFLSnF_KG_v3/results/final_dynamic_fedadam_batch_<时间戳>/batch_summary.json"
```

单独运行配置示例：

```powershell
python -m HFLSnF_KG_v3.run_federated_transe --cf HFLSnF_KG_v3/configs/dynamic/final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed42_150round_cuda.yaml
```

## 随机拓扑实验

随机实验不读取MAT的逐轮参与者和分组，只使用 `util=0.6` 前150轮统计均值四舍五入后的固定预算。

| 参考档位 | 每轮参与人数 | 每轮分组数 | 实际SnF |
| --- | ---: | ---: | --- |
| HFLSnF-profile | 34 | 6 | 关闭 |
| HFLnoSnF-profile | 12 | 3 | 关闭 |
| FLnoSnF-profile | 5 | 1 | 关闭 |

每轮从37个客户端中按种子无放回随机抽取固定人数，再使用独立随机流打乱并均衡分组。相同种子能够完整复现参与者、分组和调度哈希；不同种子的随机调度不同。

三个名称表示继承原动态方法的平均人数和平均组数档位，不表示随机实验仍执行SnF。随机配置及结果中的 `topology_snf` 和 `snf_enabled` 均为 `false`。

先校验9份随机配置和150轮调度合同：

```powershell
python -m HFLSnF_KG_v3.run_final_stochastic_fedadam validate
```

运行或恢复9组正式实验：

```powershell
python -m HFLSnF_KG_v3.run_final_stochastic_fedadam formal150
python -m HFLSnF_KG_v3.run_final_stochastic_fedadam formal150 --resume "HFLSnF_KG_v3/results/final_stochastic_fedadam_batch_<时间戳>/batch_summary.json"
```

单独运行配置示例：

```powershell
python -m HFLSnF_KG_v3.run_federated_transe --cf HFLSnF_KG_v3/configs/stochastic/final_stochastic_fedadam_hflsnf_profile_seed42_150round_cuda.yaml
```

两个批量入口都按 `seed42 → seed2024 → seed2025` 执行，每个种子内部依次运行三个实验臂或参考档位。任一配置、训练或结果合同失败时立即停止，并将恢复信息写入批次清单。

## 维护原则

- 正式配置只能放入 `dynamic/` 或 `stochastic/`，`configs/` 根级不放置YAML。
- YAML注释统一使用英文，Markdown说明统一使用简体中文。
- 不直接修改已归档配置；需要复现实验时从归档复制并使用新的身份字段。
- 动态配置移动目录只改变路径，不改变文件内容或原MAT调度哈希。
