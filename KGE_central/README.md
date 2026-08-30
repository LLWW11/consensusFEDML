# KGE_central集中式TransE工程

## 工程定位

本目录恢复此前在FB15k-237上完成的集中式TransE强基线。它不创建客户端、不做联邦划分、不读取MAT拓扑，也不依赖FedML参数加载器。

为避免重新设计造成数值漂移，本工程逐文件恢复`HFLSnF_KG_v3.tasks.kge`中已经得到完整测试MRR `0.386967`的算法内核，包括：

- TransE模型；
- FB15k-237统一编号和数据完整性校验；
- CUDA向量化filtered负采样；
- 频率子采样权重；
- 头尾交替的双向自对抗目标；
- 精确filtered头尾排名评估；
- 以验证MRR选择并恢复最佳模型。

这些模块现在实际保存在`KGE_central`目录中，不再依赖V3包。`KGE_central`还自己负责YAML读取、CUDA强制检查、随机种子、结果目录、配置快照和检查点保存，因此运行集中式实验时不会进入任何联邦训练代码。

## 历史正式基线

历史结果保存在：

```text
HFLSnF_KG_v3/results/hflsnf_kg_v3_centralized_strong_transe_cuda_20260727_231830_874259
```

主要指标：

| 指标 | 数值 |
|---|---:|
| 最佳epoch | 380 |
| 完整验证MRR | 0.393167 |
| 完整测试MRR | 0.386967 |
| 测试Hits@1 | 0.294782 |
| 测试Hits@3 | 0.426561 |
| 测试Hits@10 | 0.565035 |
| 测试平均排名 | 172.847 |
| 测试三元组数 | 20,466 |
| 头尾查询总数 | 40,932 |

机器可读版本见`baseline_reference.json`。

## 正式CUDA训练

在工作区根目录`D:\1\1myworkcode`执行：

```powershell
python -m KGE_central.run
```

默认加载：

```text
KGE_central/configs/centralized_fb15k237_strong_transe_cuda.yaml
```

该配置强制要求CUDA，不允许静默降级到CPU。当前对齐配方为种子42、256维、L1距离、450个epoch、每3个epoch使用关系分层验证子集选模、批量1024、256个负样本、学习率`5e-5`、双向自对抗目标，以及训练结束后的完整验证和完整测试。

FB15k-237默认复用：

```text
HFLSnF_KG_v3/data/FB15k-237
```

如果以后把数据复制到其他位置，只需要修改正式YAML中的`data_dir`。

## CPU冒烟

下面的命令使用内存中的微型知识图谱，只验证代码链路，不代表正式精度：

```powershell
python -m KGE_central.run `
  --config KGE_central/configs/smoke_synthetic_cpu.yaml
```

也可以打开`run_from_ide.py`直接运行。该文件默认使用CPU冒烟配置；在GPU服务器上可把`DEFAULT_CONFIG_NAME`改为正式配置文件名。

## 输出文件

每次运行都会创建唯一的时间戳目录，并保存：

- `config_snapshot.json`：实际运行配置；
- `dataset_summary.json`：实体、关系和三元组数量；
- `entity2id.json`与`relation2id.json`：全局编号映射；
- `metrics.csv`：逐epoch损失、耗时和验证指标；
- `summary.json`：最佳epoch、完整验证和完整测试指标；
- `model_best.pt`：已恢复最佳验证模型的检查点。

## 修改实验参数

所有模型、训练、评估和设备参数都在YAML中。建议复制正式配置后只改变一个变量，并修改`run_name`，这样不会把新实验误认为历史强基线。

正式复现实验不要修改以下关键字段：

- `random_seed: 42`
- `embedding_dim: 256`
- `distance_norm: 1`
- `epochs: 450`
- `eval_every: 3`
- `batch_size: 1024`
- `learning_rate: 0.00005`
- `negative_sample_count: 256`
- `local_objective: bidirectional_self_adversarial`
- `test_max_triples: 0`

其中`test_max_triples: 0`表示使用全部官方测试三元组。
