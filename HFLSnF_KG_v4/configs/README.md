# V4配置文件说明

## 当前目录结构

V4现有配置按用途分为三类：

- `dynamic/`：9份MAT动态拓扑正式配置，覆盖HFLSnF、HFLnoSnF、FLnoSnF和随机种子42、2024、2025；
- `stochastic/`：9份随机拓扑正式配置，覆盖三个MAT平均预算参考档位和相同的三个随机种子；
- `zOld/`：35份历史筛选、消融、对照和烟雾测试配置，只用于审计或必要时复现。

V4的 `results/` 当前保留了9个动态三种子结果和6个固定人数随机抽取结果，没有保留随机拓扑正式结果。配置文件仍然存在不代表对应结果仍在V4中。

## 动态拓扑配置

动态配置读取 `Topo_opt/postprocess` 中 `util=0.6` 的硬覆盖MAT调度。每轮参与者和HFL分组由MAT决定，训练参数固定为最终FedAdam口径。

只读校验9份动态配置：

```powershell
python -m HFLSnF_KG_v4.run_final_dynamic_fedadam validate
```

运行或恢复9组动态正式实验：

```powershell
python -m HFLSnF_KG_v4.run_final_dynamic_fedadam formal150
python -m HFLSnF_KG_v4.run_final_dynamic_fedadam formal150 --resume "HFLSnF_KG_v4/results/final_dynamic_fedadam_batch_<时间戳>/batch_summary.json"
```

单独运行一份现有HFLSnF配置：

```powershell
python -m HFLSnF_KG_v4.run_federated_transe --cf HFLSnF_KG_v4/configs/dynamic/final_dynamic_fedadam_hflsnf_u0p6_bcfalse_seed42_150round_cuda.yaml
```

V4重叠率消融固定使用HFLSnF动态正式配置。原始划分对照复用现有三个HFLSnF结果；`overlap/` 已保存8重启正式校准合同和低、中、高乘三个种子的9份配置。

## 随机拓扑配置

随机配置不读取MAT逐轮参与者和分组，只使用 `util=0.6` 前150轮统计均值四舍五入后的固定预算。

| 参考档位 | 每轮参与人数 | 每轮分组数 | 实际SnF |
| --- | ---: | ---: | --- |
| HFLSnF-profile | 34 | 6 | 关闭 |
| HFLnoSnF-profile | 12 | 3 | 关闭 |
| FLnoSnF-profile | 5 | 1 | 关闭 |

每轮从37个客户端中按种子无放回随机抽取固定人数，再用独立随机流打乱并均衡分组。档位名称只表示继承对应动态方法的平均资源预算，不表示随机实验执行SnF。

只读校验9份随机配置和确定性调度合同：

```powershell
python -m HFLSnF_KG_v4.run_final_stochastic_fedadam validate
```

运行或恢复随机拓扑正式实验：

```powershell
python -m HFLSnF_KG_v4.run_final_stochastic_fedadam formal150
python -m HFLSnF_KG_v4.run_final_stochastic_fedadam formal150 --resume "HFLSnF_KG_v4/results/final_stochastic_fedadam_batch_<时间戳>/batch_summary.json"
```

上述命令是现有配置入口，但V4当前没有保留这9组随机拓扑正式结果。

## 重叠率配置准备状态

完整设计见[重叠率消融实验计划](../重叠率消融实验计划.md)。头实体互斥目标重叠划分策略及目标重叠率、重叠容差、负载容差、搜索种子和搜索重启次数字段已经实现，正式配置见[overlap说明](overlap/README.md)。

正式低、中、高目标分别为0.172249、0.228804和0.285359，九个分区哈希已经第二次独立复算确认。现有 `dynamic/` 原始配置不作修改；训练必须通过 `run_overlap_ablation` 的配置、分区、基线和拓扑合同后启动。

## 历史配置

历史配置清单和兼容性边界见[zOld归档说明](zOld/README.md)。历史YAML保留原有身份字段、参数和文件名，不作为V4当前主实验的默认配置。

## 维护原则

- 新增Markdown说明统一使用简体中文；
- YAML注释统一使用英文；
- 不直接修改已归档配置；
- 不改写已有结果目录中的配置快照、批次清单和合同哈希；
- V4当前操作命令统一使用 `HFLSnF_KG_v4`；
- 结果产物中遗留的V3名称作为历史来源凭据保留。
