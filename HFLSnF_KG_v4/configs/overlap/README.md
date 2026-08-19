# V4实体重叠率正式配置

本目录保存HFLSnF实体重叠率正式消融使用的9份YAML和唯一校准合同 `partition_calibration_contract.json`。三种数据条件只改变客户端训练三元组分区，模型、动态拓扑、参与者序列、训练参数、FedAdam和评估设置均与同种子HFLSnF原始对照保持一致。

## 冻结目标

| 档位 | 正式目标实体重叠率 |
| --- | ---: |
| 低 | 0.1722490329005324 |
| 中 | 0.22880424374736683 |
| 高 | 0.28535945459420126 |

三个种子的共同可达区间为0.1722490329005324至0.28535945459420126，跨度为0.11311042169366886。正式校准使用8次确定性重启，第二次独立复算得到相同的九个分区哈希。

## 配置与分区哈希

| 档位 | seed42 | seed2024 | seed2025 |
| --- | --- | --- | --- |
| 低 | `9938b8eb7424e8de4275db0b27af974f719ec896bd9bb4e97928743858b47aac` | `f30fbc1689ccb9446f68bc76e7f84f23372e54c18eea8017174f11034e483333` | `13a037951960864afa663ce4e82a32749c1ac67fa64a8ef7296f39d29a60d0eb` |
| 中 | `666a6f14648d05d6aeb465619fd7aa789f520b061a866959609e03a261e590ef` | `ce30eab3506c31de0eb40a5c70b75267d0cb227d12f7546416a884e02de3fe13` | `61f7ee4a4f264cd1af8f6a127933ff3eb3f59e973cc04a866d4933674bdd8b91` |
| 高 | `120539709b0679f969fbd3e680753a250de4d30ab04d9ca8df4d171231046226` | `7995c494945793505e34bba4478b1c2e9ecd65b3fe6b689c5aa6a0cd0d12b642` | `9a30d5f94e55d10c60c25f83dcd6069ce7a8039538543db6357eeed9fc29f256` |

## 执行入口

在仓库根目录执行完整只读合同校验：

```powershell
python -m HFLSnF_KG_v4.run_overlap_ablation validate
```

顺序运行9组正式训练并支持失败恢复：

```powershell
python -m HFLSnF_KG_v4.run_overlap_ablation formal150
python -m HFLSnF_KG_v4.run_overlap_ablation formal150 --resume "<batch_summary.json>"
```

训练全部通过后执行12格完整官方测试和报告：

```powershell
python -m HFLSnF_KG_v4.run_overlap_ablation official12 --batch "<batch_summary.json>"
python -m HFLSnF_KG_v4.run_overlap_ablation report --batch "<batch_summary.json>"
```

正式训练要求CUDA且禁止回退CPU。不得手工修改目标值、搜索参数、分区哈希或校准合同；如需重新校准，应生成新合同并重新执行全部配置与分区验证。
